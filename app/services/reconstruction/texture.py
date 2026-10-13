"""[3, 4단계] texrecon 입력 씬 구성 + 좌표계 정합 검증 + 텍스처링.

texrecon(mvs-texturing)은 "이미지 + 같은 이름의 .cam 파일" 폴더를 씬으로 받는다 (MVE 카메라 형식).
  줄1: tx ty tz R00 ... R22   (world-to-camera)
  줄2: f d0 d1 paspect ppx ppy  (f = fx/max(W,H), paspect = fy/fx, ppx = cx/W, ppy = cy/H)

검증: mesh_tex를 표본 프레임 pose로 렌더한 depth와 LiDAR depth를 비교한다. 좌표계가 틀리면
오차가 수십 %로 커지므로 확실히 걸린다. 실패하면 텍스처가 통째로 어긋나므로 중단한다.

texrecon 옵션: --outlier_removal=gauss_damping (색 아웃라이어 가중치 감쇠).
--keep_unseen_faces는 쓰지 않는다. 켜면 못 본 면이 색 없는 공용 텍스처에 묶여 검게 나온다
(mvs-texturing issue #152). 대신 그 면들을 obj에서 빠지게 두고 usdz 단계에서 정점 색으로 채운다.
"""

import logging
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import open3d as o3d

from app.services.reconstruction.preprocess import Preprocessed

logger = logging.getLogger("app")

VALIDATE = {"n_frames": 6, "max_median_rel_err": 0.08, "min_inlier_ratio": 0.6, "inlier_rel_err": 0.10}


def _write_cam(path: Path, w2c: np.ndarray, fx, fy, cx, cy, w, h):
    t = w2c[:3, 3]
    R = w2c[:3, :3]
    line1 = " ".join(f"{v:.9f}" for v in [t[0], t[1], t[2], *R.flatten()])
    line2 = f"{fx / max(w, h):.9f} 0 0 {fy / fx:.9f} {cx / w:.9f} {cy / h:.9f}"
    path.write_text(line1 + "\n" + line2 + "\n")


def _load_lidar_depth(scan: Path, frame: int):
    import cv2
    depth = cv2.imread(str(scan / "depth" / f"{frame:06d}.png"), cv2.IMREAD_UNCHANGED)
    if depth is None:
        return None
    d = depth.astype(np.float32) / 1000.0
    cp = scan / "confidence" / f"{frame:06d}.png"
    if cp.exists():
        conf = cv2.imread(str(cp), cv2.IMREAD_UNCHANGED)
        d[conf < 2] = 0.0
    return d


def validate_alignment(mesh_path: Path, prep: Preprocessed) -> dict:
    (rgb_w, rgb_h), (depth_w, depth_h) = prep.rgb_size, prep.depth_size
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    del mesh

    frames = prep.frames
    picks = [frames[i] for i in np.linspace(0, len(frames) - 1, VALIDATE["n_frames"]).astype(int)]
    results = []
    for c in picks:
        fx, fy, cx, cy = prep.intrinsics[c]
        sx, sy = depth_w / rgb_w, depth_h / rgb_h
        K_d = o3d.core.Tensor([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]],
                              dtype=o3d.core.Dtype.Float64)
        w2c = o3d.core.Tensor(np.linalg.inv(prep.poses[c]), dtype=o3d.core.Dtype.Float64)
        rays = scene.create_rays_pinhole(K_d, w2c, depth_w, depth_h)
        hit = scene.cast_rays(rays)
        t_hit = hit["t_hit"].numpy().reshape(depth_h, depth_w)
        # 교차점 = origin + t_hit * dir(원시 벡터) 이므로 z = t_hit * (dir · forward). 방향벡터를 정규화하면 안 됨
        dirs = rays.numpy()[..., 3:].reshape(depth_h, depth_w, 3)
        forward = prep.poses[c][:3, 2]
        z_render = t_hit * (dirs @ forward)

        d_lidar = _load_lidar_depth(prep.scan, c)
        valid = (d_lidar > 0.3) & np.isfinite(z_render) & (z_render > 0.3)
        if valid.sum() < 500:
            results.append({"frame": c, "usable": False})
            continue
        rel = np.abs(z_render[valid] - d_lidar[valid]) / d_lidar[valid]
        results.append({"frame": c, "usable": True, "median_rel_err": float(np.median(rel)),
                        "inlier_ratio": float(np.mean(rel < VALIDATE["inlier_rel_err"]))})

    usable = [r for r in results if r["usable"]]
    assert usable, "검증 가능한 프레임 없음"
    med = float(np.median([r["median_rel_err"] for r in usable]))
    inl = float(np.median([r["inlier_ratio"] for r in usable]))
    passed = med <= VALIDATE["max_median_rel_err"] and inl >= VALIDATE["min_inlier_ratio"]
    return {"passed": passed, "median_rel_err": med, "inlier_ratio": inl}


def make_scene(prep: Preprocessed, scene_dir: Path) -> int:
    scene_dir.mkdir(parents=True, exist_ok=True)
    rgb_w, rgb_h = prep.rgb_size
    n = 0
    for c in prep.frames:
        img_src = prep.images_dir / f"frame_{c:06d}.jpg"
        if not img_src.exists() or c not in prep.poses:
            continue
        dst = scene_dir / f"frame_{c:06d}.jpg"
        if not dst.exists():
            try:
                dst.symlink_to(img_src.resolve())
            except OSError:
                shutil.copy(img_src, dst)
        fx, fy, cx, cy = prep.intrinsics[c]
        _write_cam(scene_dir / f"frame_{c:06d}.cam", np.linalg.inv(prep.poses[c]), fx, fy, cx, cy, rgb_w, rgb_h)
        n += 1
    return n


def run_texrecon(scene_dir: Path, mesh_tex: Path, out_dir: Path, texrecon_bin: str, num_threads: int | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [texrecon_bin, str(scene_dir), str(mesh_tex), str(out_dir / "textured"),
           "--outlier_removal=gauss_damping", "--no_intermediate_results"]
    if num_threads:
        cmd.append(f"--num_threads={num_threads}")
    log_path = out_dir / "texrecon.log"
    with open(log_path, "w") as fp:
        proc = subprocess.run(cmd, stdout=fp, stderr=subprocess.STDOUT, env=dict(os.environ))
    obj = out_dir / "textured.obj"
    if proc.returncode != 0 or not obj.exists():
        tail = log_path.read_text(errors="ignore")[-2000:]
        raise RuntimeError(f"texrecon 실패 (exit {proc.returncode}): {tail}")
    unseen = next((line.strip() for line in log_path.read_text(errors="ignore").splitlines()
                   if "have not been seen" in line), "")
    logger.info("[R01 texture] texrecon 완료 (%s)", unseen or "미관측 면 정보 없음")
    return obj


def texture_mesh(prep: Preprocessed, mesh_tex: Path, work_dir: Path, texrecon_bin: str,
                 num_threads: int | None = None) -> Path:
    scene_dir = work_dir / "scene"
    n = make_scene(prep, scene_dir)
    logger.info("[R01 texture] 씬 구성: 이미지+cam %d쌍", n)
    report = validate_alignment(mesh_tex, prep)
    if not report["passed"]:
        raise RuntimeError(
            f"좌표계 정합 검증 실패: 중앙 상대오차 {report['median_rel_err'] * 100:.1f}% "
            f"(기준 {VALIDATE['max_median_rel_err'] * 100:.0f}%), inlier {report['inlier_ratio'] * 100:.0f}%")
    logger.info("[R01 texture] 정합 검증 통과: 중앙 상대오차 %.1f%%, inlier %.0f%%",
                report["median_rel_err"] * 100, report["inlier_ratio"] * 100)
    return run_texrecon(scene_dir, mesh_tex, work_dir / "textured", texrecon_bin, num_threads)
