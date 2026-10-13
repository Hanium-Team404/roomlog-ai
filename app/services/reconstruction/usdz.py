"""[5단계] textured.obj -> usdz.

texrecon 출력(textured.obj + mtl + 아틀라스 png)을 USD 스테이지로 구성해 usdz로 패키징한다.
MaterialBindingAPI를 정식 적용한다. 좌표계는 ARKit world 그대로 (upAxis Y, 단위 미터).

미관측 면 처리: texrecon이 어느 사진에도 못 본 면(obj에서 빠진 면)을 mesh_tex.ply에서 찾아
TSDF가 전체 프레임으로 융합해 둔 정점 색을 작은 텍스처에 구워 입힌다. 정점 색(displayColor)을
그대로 넣지 않는 이유: AR QuickLook이 displayColor를 쓰지 않아 핑크로 나온다
(Apple 개발자 포럼 thread/691674). 텍스처로 구워야 보인다.
"""

import logging
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from PIL import Image
from pxr import Sdf, Usd, UsdGeom, UsdShade, UsdUtils
from scipy.spatial import cKDTree

logger = logging.getLogger("app")

CELL = 8           # 미관측 면 1개가 차지하는 텍셀 칸 (CELL x CELL)
MATCH_TOL = 1e-4   # obj와 ply의 면 중심 매칭 허용 오차 (m). obj는 텍스트 float라 ply(double)와 미세하게 다름


def _unseen_faces(mesh_tex: o3d.geometry.TriangleMesh, geoms: list) -> np.ndarray:
    """mesh_tex.ply의 면 중 textured.obj에 없는 면의 인덱스. texrecon이 정점을 패치별로 복제해
    인덱스가 바뀌므로 면 중심 좌표로 맞춘다."""
    V = np.asarray(mesh_tex.vertices)
    F = np.asarray(mesh_tex.triangles)
    ply_c = V[F].mean(1)
    obj_c = np.concatenate([g.triangles_center for g in geoms if len(g.faces)])
    dist, _ = cKDTree(obj_c).query(ply_c, distance_upper_bound=MATCH_TOL)
    return np.where(~np.isfinite(dist))[0]


def _bake_vertex_colors(V: np.ndarray, C: np.ndarray, F: np.ndarray, cell: int = CELL):
    """면마다 cell x cell 텍셀 칸을 주고 세 정점 색을 무게중심 보간으로 채운다.
    삼각형은 칸의 왼쪽 아래 직각삼각형에 놓이고, 빗변 밖 텍셀은 빗변 위로 사영한 색으로 채워
    bilinear 샘플이 칸 밖으로 새지 않게 한다. UV는 꼭짓점을 반 텍셀 안쪽으로 넣는다.
    반환: (아틀라스 uint8, faceVarying UV (3n, 2) 좌하단 원점)"""
    n = len(F)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    g = (np.arange(cell) + 0.5) / cell
    u, v = np.meshgrid(g, g)
    wB, wC = u.copy(), v.copy()
    s = wB + wC
    over = s > 1
    wB[over] /= s[over]
    wC[over] /= s[over]
    W = np.stack([1 - wB - wC, wB, wC], -1)
    block = np.einsum("hwk,nkc->nhwc", W, C[F])
    img = np.zeros((rows * cell, cols * cell, 3), np.float32)
    idx = np.arange(n)
    r, c = idx // cols, idx % cols
    img.reshape(rows, cell, cols, cell, 3)[r, :, c, :, :] = block
    img = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)

    Hpx, Wpx = img.shape[:2]
    x0, y0 = c * cell + 0.5, r * cell + 0.5
    x1, y1 = (c + 1) * cell - 0.5, (r + 1) * cell - 0.5
    corners = np.stack([np.stack([x0, y0], 1), np.stack([x1, y0], 1), np.stack([x0, y1], 1)], 1)
    uv = corners.reshape(-1, 2).astype(np.float32)
    uv[:, 0] /= Wpx
    uv[:, 1] = 1.0 - uv[:, 1] / Hpx
    return img, uv


def _material(stage, path: str, tex_file: str | None, varname: str = "st"):
    material = UsdShade.Material.Define(stage, path)
    surface = UsdShade.Shader.Define(stage, f"{path}/surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
    surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(surface.CreateOutput("surface", Sdf.ValueTypeNames.Token))
    if tex_file is None:
        surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.6, 0.6, 0.6))
        return material
    reader = UsdShade.Shader.Define(stage, f"{path}/stReader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set(varname)
    tex = UsdShade.Shader.Define(stage, f"{path}/diffuseTex")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(tex_file)
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float2))
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
    return material


def _define_mesh(stage, path: str, vertices, faces, normals):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(np.asarray(vertices, dtype=np.float32).tolist())
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr(np.asarray(faces).flatten().tolist())
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    # 법선을 명시한다. 안 적으면 렌더러에 따라 면 단위로 잡아 삼각형 경계가 드러난다
    mesh.CreateNormalsAttr(np.asarray(normals, dtype=np.float32).tolist())
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    return mesh


def _add_textured(stage, idx: int, geom: trimesh.Trimesh, tex_dir: Path) -> bool:
    mesh = _define_mesh(stage, f"/Room/mesh_{idx}", geom.vertices, geom.faces, geom.vertex_normals)
    # trimesh는 obj 재질을 SimpleMaterial(.image) 또는 PBRMaterial(.baseColorTexture)로 싣는다
    tm_material = getattr(geom.visual, "material", None)
    image = getattr(tm_material, "image", None) or getattr(tm_material, "baseColorTexture", None)
    uv = getattr(geom.visual, "uv", None)
    textured = image is not None and uv is not None and len(uv) == len(geom.vertices)
    if textured:
        pv = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        pv.Set(np.asarray(uv, dtype=np.float32).tolist())
        image.save(tex_dir / f"tex_{idx}.png")
    material = _material(stage, f"/Room/materials/mat_{idx}", f"textures/tex_{idx}.png" if textured else None)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
    UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(material)
    return textured


def _add_unseen(stage, mesh_tex: o3d.geometry.TriangleMesh, face_idx: np.ndarray, tex_dir: Path) -> tuple:
    V = np.asarray(mesh_tex.vertices)
    C = np.asarray(mesh_tex.vertex_colors)
    F = np.asarray(mesh_tex.triangles)[face_idx]
    used, F_local = np.unique(F, return_inverse=True)
    F_local = F_local.reshape(-1, 3)
    img, uv = _bake_vertex_colors(V, C, F, CELL)
    Image.fromarray(img).save(tex_dir / "tex_unseen.png")
    sub = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V[used]), o3d.utility.Vector3iVector(F_local))
    sub.compute_vertex_normals()
    mesh = _define_mesh(stage, "/Room/mesh_unseen", V[used], F_local, np.asarray(sub.vertex_normals))
    pv = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    pv.Set(uv.tolist())
    material = _material(stage, "/Room/materials/mat_unseen", "textures/tex_unseen.png")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
    UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(material)
    return img.shape


def build_usdz(obj_path: Path, mesh_tex_path: Path, usdz_path: Path) -> dict:
    loaded = trimesh.load(obj_path, process=False)
    geoms = [g for g in (loaded.geometry.values() if isinstance(loaded, trimesh.Scene) else [loaded]) if len(g.faces)]

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "textures").mkdir()
        stage = Usd.Stage.CreateNew(str(tmp / "scene.usda"))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        root = UsdGeom.Xform.Define(stage, "/Room")
        stage.SetDefaultPrim(root.GetPrim())

        n_textured = sum(_add_textured(stage, i, g, tmp / "textures") for i, g in enumerate(geoms))
        if n_textured == 0:
            raise RuntimeError("텍스처가 하나도 연결되지 않음 (obj/mtl 로드 문제)")

        mesh_tex = o3d.io.read_triangle_mesh(str(mesh_tex_path))
        assert mesh_tex.has_vertex_colors(), "mesh_tex.ply에 정점 색이 없음"
        missing = _unseen_faces(mesh_tex, geoms)
        n_faces = len(mesh_tex.triangles)
        if len(missing):
            _add_unseen(stage, mesh_tex, missing, tmp / "textures")
            n_textured += 1
        del mesh_tex, geoms, loaded
        stage.GetRootLayer().Save()

        usdz_path.parent.mkdir(parents=True, exist_ok=True)
        if not UsdUtils.CreateNewUsdzPackage(str(tmp / "scene.usda"), str(usdz_path)):
            raise RuntimeError("usdz 패키징 실패")

    with zipfile.ZipFile(usdz_path) as z:
        n_png = sum(1 for n in z.namelist() if n.endswith(".png"))
    assert n_png >= n_textured, f"usdz에 텍스처 누락: {n_png}/{n_textured}"
    assert Usd.Stage.Open(str(usdz_path)), "생성된 usdz를 다시 열 수 없음"
    stats = {"n_submeshes": n_textured, "n_faces_tex": n_faces, "n_unseen_faces": int(len(missing)),
             "usdz_mb": round(usdz_path.stat().st_size / 1e6, 1)}
    logger.info("[R01 usdz] 완료 %s: 서브메시 %d, 미관측 면 %d/%d (%.1f%%) 정점 색으로 채움, %.1fMB",
                usdz_path.name, n_textured, len(missing), n_faces, len(missing) / max(n_faces, 1) * 100,
                stats["usdz_mb"])
    return stats
