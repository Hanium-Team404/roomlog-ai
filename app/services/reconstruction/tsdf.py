"""[2단계] TSDF 재구성 + cleanup + 텍스처용 단순화.

LiDAR 실측 depth를 보정 pose로 융합한다. 텍스처링용 mesh_tex는 quadric decimation으로
단순화한다 (texrecon 속도와 아틀라스 조각 수를 줄이고, 색 디테일은 텍스처가 담당).
"""

import gc
import logging
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from PIL import Image

from app.services.reconstruction.preprocess import Preprocessed, load_depth

logger = logging.getLogger("app")

MAX_DEPTH = 20.0
TSDF = {"voxel_size": 0.015, "sdf_trunc": 0.05}
CLEAN = {"min_component_tri": 100}


def _cleanup(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    tri_clusters, cluster_n, _ = mesh.cluster_connected_triangles()
    cluster_n = np.asarray(cluster_n)
    small = cluster_n[np.asarray(tri_clusters)] < CLEAN["min_component_tri"]
    mesh.remove_triangles_by_mask(small.tolist())
    mesh.remove_unreferenced_vertices()
    logger.info("[R01 tsdf] cleanup: 파편 삼각형 %d개 제거", int(small.sum()))
    return mesh


def build_mesh(prep: Preprocessed, out_dir: Path, target_faces: int) -> tuple[Path, Path]:
    """반환: (mesh.ply 정리본 (썸네일 렌더용), mesh_tex.ply 단순화본 (텍스처링 입력))"""
    out_dir.mkdir(parents=True, exist_ok=True)
    scan = prep.scan
    rgb_w, rgb_h = prep.rgb_size
    depth_w, depth_h = prep.depth_size
    K = np.loadtxt(scan / "camera_matrix.csv", delimiter=",")
    sx, sy = depth_w / rgb_w, depth_h / rgb_h
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        depth_w, depth_h, K[0, 0] * sx, K[1, 1] * sy, K[0, 2] * sx, K[1, 2] * sy)

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=TSDF["voxel_size"],
        sdf_trunc=TSDF["sdf_trunc"],
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    common = sorted(prep.poses)
    cap = cv2.VideoCapture(str(scan / "rgb.mp4"))
    video_idx, bgr, integrated = -1, None, 0
    try:
        for c in common:
            k = c - prep.rgb_offset
            if k < 0:
                continue
            while video_idx < k:
                ret, frame = cap.read()
                if not ret:
                    break
                video_idx += 1
                bgr = frame
            if video_idx != k or bgr is None:
                break
            rgb_small = np.asarray(Image.fromarray(
                cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).resize((depth_w, depth_h), resample=Image.BILINEAR))
            d = load_depth(scan, c)
            if d is None:
                continue
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(rgb_small), o3d.geometry.Image(d),
                depth_scale=1.0, depth_trunc=MAX_DEPTH, convert_rgb_to_intensity=False)
            volume.integrate(rgbd, intrinsic, np.linalg.inv(prep.poses[c]))
            integrated += 1
            del rgbd
    finally:
        cap.release()
    logger.info("[R01 tsdf] 통합 %d/%d프레임", integrated, len(common))

    mesh = volume.extract_triangle_mesh()
    del volume
    gc.collect()
    mesh = _cleanup(mesh)
    mesh.compute_vertex_normals()
    assert len(mesh.triangles) > 0, "mesh가 비어 있음 (pose 또는 depth 확인)"
    mesh_path = out_dir / "mesh.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)
    logger.info("[R01 tsdf] mesh: 정점 %d, 삼각형 %d", len(mesh.vertices), len(mesh.triangles))

    mesh_tex = mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)
    del mesh
    mesh_tex.remove_degenerate_triangles()
    mesh_tex.remove_unreferenced_vertices()
    mesh_tex.compute_vertex_normals()
    assert len(mesh_tex.triangles) > 0, "단순화 결과가 비어 있음"
    tex_path = out_dir / "mesh_tex.ply"
    o3d.io.write_triangle_mesh(str(tex_path), mesh_tex)
    logger.info("[R01 tsdf] mesh_tex: 정점 %d, 삼각형 %d", len(mesh_tex.vertices), len(mesh_tex.triangles))
    del mesh_tex
    gc.collect()
    return mesh_path, tex_path
