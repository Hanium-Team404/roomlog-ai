"""[1단계] 동기화, depth 필터, pose 보정, 텍스처용 키프레임 선택.

1. RGB 동기화: 비디오 모션 vs odometry 이동량 상관으로 전역 프레임 offset 추정
   (iOS 영상이 시작 부근에서 1프레임 드랍되어 RGB가 밀리는 현상 보정)
2. depth 필터: 불연속 경계(flying pixel) + 작은 고립 영역 제거
3. pose 보정: 인접은 ARKit VIO 신뢰, 시공간 재방문 쌍만 ICP 검증 loop closure,
   pose graph 최적화 후 보정 회전은 yaw(중력축)로 사영
4. 프레임 선택: keyframe 구간은 gyro_mag + 노출회전량, video 구간은 Laplacian 선명도 + 노출회전량
"""

import csv
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation, Slerp

logger = logging.getLogger("app")

FILTER = {"abs_thresh": 0.03, "rel_thresh": 0.04, "edge_dilate": 1, "min_region": 20, "confidence": 2}
POSE = {
    "kf_interval_s": 0.5, "icp_voxel": 0.03, "icp_dist": 0.07, "icp_depth_trunc": 5.0,
    "loop_min_gap_s": 10.0, "loop_max_dist": 1.5, "loop_fitness": 0.6, "loop_rmse": 0.03,
    "loop_max_corr": 0.2, "loop_sym_tol": 0.03, "yaw_only": True,
}


@dataclass
class Preprocessed:
    scan: Path
    images_dir: Path                      # frame_XXXXXX.jpg (원본 해상도 텍스처 소스)
    poses: dict                           # frame -> 4x4 c2w (보정 적용)
    stamps: dict                          # frame -> timestamp
    frames: list                          # 선택된 프레임 번호 (오름차순)
    intrinsics: dict                      # frame -> (fx, fy, cx, cy) 원본 해상도 기준
    rgb_size: tuple
    depth_size: tuple
    rgb_offset: int
    stats: dict = field(default_factory=dict)


def probe_sizes(scan: Path) -> dict:
    cap = cv2.VideoCapture(str(scan / "rgb.mp4"))
    rgb_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    rgb_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    depth_files = sorted((scan / "depth").glob("*.png"))
    assert depth_files, "depth 폴더가 비어 있음"
    d = cv2.imread(str(depth_files[0]), cv2.IMREAD_UNCHANGED)
    assert rgb_w > 0 and rgb_h > 0 and d is not None, "rgb.mp4 또는 depth를 읽지 못함"
    return {"rgb": (rgb_w, rgb_h), "depth": (int(d.shape[1]), int(d.shape[0]))}


def load_poses(odometry_path: Path):
    data = np.loadtxt(odometry_path, delimiter=",", skiprows=1)
    poses, stamps = {}, {}
    for row in data:
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(row[5:9]).as_matrix()
        T[:3, 3] = row[2:5]
        f = int(row[1])
        poses[f] = T
        stamps[f] = float(row[0])
    return poses, stamps


def filter_depth(d: np.ndarray) -> np.ndarray:
    H, W = d.shape
    grad = np.zeros_like(d)

    def acc(a_sl, b_sl):
        da, db = d[a_sl], d[b_sl]
        diff = np.where((da > 0) & (db > 0), np.abs(da - db), 0.0)
        grad[a_sl] = np.maximum(grad[a_sl], diff)
        grad[b_sl] = np.maximum(grad[b_sl], diff)

    acc((slice(0, H - 1), slice(None)), (slice(1, H), slice(None)))
    acc((slice(None), slice(0, W - 1)), (slice(None), slice(1, W)))
    edge = (grad > np.maximum(FILTER["abs_thresh"], FILTER["rel_thresh"] * d)) & (d > 0)
    k = 2 * FILTER["edge_dilate"] + 1
    edge = cv2.dilate(edge.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    d = np.where(edge, 0.0, d)
    n, labels, stats, _ = cv2.connectedComponentsWithStats((d > 0).astype(np.uint8), connectivity=8)
    small = np.where(stats[:, cv2.CC_STAT_AREA] < FILTER["min_region"])[0]
    if len(small):
        d = np.where(np.isin(labels, small), 0.0, d)
    return d


def load_depth(scan: Path, frame: int) -> np.ndarray | None:
    depth = cv2.imread(str(scan / "depth" / f"{frame:06d}.png"), cv2.IMREAD_UNCHANGED)
    if depth is None:
        return None
    d = depth.astype(np.float32) / 1000.0
    cp = scan / "confidence" / f"{frame:06d}.png"
    if cp.exists():
        conf = cv2.imread(str(cp), cv2.IMREAD_UNCHANGED)
        d[conf < FILTER["confidence"]] = 0.0
    return filter_depth(d)


def load_keyframes(scan: Path) -> dict:
    kf_path = scan / "keyframes.csv"
    if not kf_path.exists():
        return {}
    keyframes = {}
    with open(kf_path, newline="") as fp:
        for row in csv.DictReader(fp, skipinitialspace=True):
            frame = int(row["frame"])
            jpg = scan / "keyframes" / f"frame_{frame:06d}.jpg"
            if not jpg.exists():
                continue
            keyframes[frame] = {
                "path": jpg,
                "fx": float(row["fx"]), "fy": float(row["fy"]),
                "cx": float(row["cx"]), "cy": float(row["cy"]),
                "gyro_mag": float(row["gyro_mag"]),
                "exposure_ms": float(row["exposure_ms"]),
            }
    return keyframes


class OdometryRotator:
    """보정 pose를 SLERP 보간해 노출구간 [t - exp/2, t + exp/2]의 실제 회전량(rad)을 구한다."""

    def __init__(self, stamps: dict, poses: dict):
        frames_sorted = sorted(stamps)
        self.ts = np.array([stamps[f] for f in frames_sorted])
        quats = np.array([Rotation.from_matrix(poses[f][:3, :3]).as_quat() for f in frames_sorted])
        assert np.all(np.diff(self.ts) > 0), "odometry timestamps must be strictly increasing"
        self._slerp = Slerp(self.ts, Rotation.from_quat(quats))
        self.t_min, self.t_max = float(self.ts[0]), float(self.ts[-1])

    def rot_at(self, t: float) -> Rotation:
        t = float(np.clip(t, self.t_min, self.t_max))
        return self._slerp([t])[0]

    def exposure_rotation(self, t_center: float, exp_s: float) -> float:
        r0 = self.rot_at(t_center - exp_s / 2.0)
        r1 = self.rot_at(t_center + exp_s / 2.0)
        return float((r0.inv() * r1).magnitude())


def rank_norm(x):
    r = np.argsort(np.argsort(x)).astype(np.float64)
    return (r - r.mean()) / (r.std() + 1e-9)


def estimate_rgb_offset(video_path: Path, odo: np.ndarray, max_offset: int = 2):
    cap = cv2.VideoCapture(str(video_path))
    motions, sharpness, prev = [], [], None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        small = cv2.cvtColor(cv2.resize(frame, (320, 240)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        sharpness.append(float(cv2.Laplacian(small, cv2.CV_32F).var()))
        small_m = cv2.resize(small, (160, 120))
        if prev is not None:
            motions.append(float(np.mean(np.abs(small_m - prev))))
        prev = small_m
    cap.release()
    trans = np.linalg.norm(np.diff(odo[:, 2:5], axis=0), axis=1)
    rots = Rotation.from_quat(odo[:, 5:9])
    ang = np.array([(rots[i].inv() * rots[i + 1]).magnitude() for i in range(len(rots) - 1)])
    v, o = rank_norm(np.array(motions)), rank_norm(trans + 0.5 * ang)
    best_off, best_corr = 0, -np.inf
    for off in range(-max_offset, max_offset + 1):
        if off >= 0:
            m = min(len(v), len(o) - off)
            corr = float(np.mean(v[:m] * o[off:off + m])) if m > 100 else -np.inf
        else:
            m = min(len(v) + off, len(o))
            corr = float(np.mean(v[-off:-off + m] * o[:m])) if m > 100 else -np.inf
        if corr > best_corr:
            best_off, best_corr = off, corr
    logger.info("[R01 preprocess] RGB offset %+d (상관 %.4f)", best_off, best_corr)
    return best_off, np.array(sharpness)


def _kf_pointcloud(scan: Path, frame: int, intrinsic):
    d = load_depth(scan, frame)
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        o3d.geometry.Image(d), intrinsic, extrinsic=np.eye(4),
        depth_scale=1.0, depth_trunc=POSE["icp_depth_trunc"])
    pcd = pcd.voxel_down_sample(POSE["icp_voxel"])
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=POSE["icp_voxel"] * 3, max_nn=30))
    return pcd


def _icp(source, target, init):
    return o3d.pipelines.registration.registration_icp(
        source, target, POSE["icp_dist"], init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())


def refine_poses(poses: dict, stamps: dict, scan: Path, K: np.ndarray, sizes: dict) -> tuple[dict, dict]:
    (rgb_w, rgb_h), (depth_w, depth_h) = sizes["rgb"], sizes["depth"]
    sx, sy = depth_w / rgb_w, depth_h / rgb_h
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        depth_w, depth_h, K[0, 0] * sx, K[1, 1] * sy, K[0, 2] * sx, K[1, 2] * sy)
    frames = sorted(poses)
    kfs = [frames[0]]
    for f in frames:
        if stamps[f] - stamps[kfs[-1]] >= POSE["kf_interval_s"]:
            kfs.append(f)
    if kfs[-1] != frames[-1]:
        kfs.append(frames[-1])
    pcds = {f: _kf_pointcloud(scan, f, intrinsic) for f in kfs}

    pg = o3d.pipelines.registration.PoseGraph()
    for f in kfs:
        pg.nodes.append(o3d.pipelines.registration.PoseGraphNode(poses[f]))
    for i in range(len(kfs) - 1):
        a, b = kfs[i], kfs[i + 1]
        T = np.linalg.inv(poses[b]) @ poses[a]
        info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            pcds[a], pcds[b], POSE["icp_dist"], T)
        pg.edges.append(o3d.pipelines.registration.PoseGraphEdge(i, i + 1, T, info, uncertain=False))

    positions = np.array([poses[f][:3, 3] for f in kfs])
    accepted = 0
    for i in range(len(kfs)):
        best_j, best_d = None, POSE["loop_max_dist"]
        for j in range(i + 1, len(kfs)):
            if stamps[kfs[j]] - stamps[kfs[i]] < POSE["loop_min_gap_s"]:
                continue
            dd = float(np.linalg.norm(positions[j] - positions[i]))
            if dd < best_d:
                best_j, best_d = j, dd
        if best_j is None:
            continue
        a, b = kfs[i], kfs[best_j]
        init = np.linalg.inv(poses[b]) @ poses[a]
        result = _icp(pcds[a], pcds[b], init)
        corr = float(np.linalg.norm(result.transformation[:3, 3] - init[:3, 3]))
        result_rev = _icp(pcds[b], pcds[a], np.linalg.inv(init))
        sym_err = float(np.linalg.norm((result.transformation @ result_rev.transformation)[:3, 3]))
        if (result.fitness >= POSE["loop_fitness"] and result.inlier_rmse <= POSE["loop_rmse"]
                and corr <= POSE["loop_max_corr"] and sym_err <= POSE["loop_sym_tol"]):
            info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                pcds[a], pcds[b], POSE["icp_dist"], result.transformation)
            pg.edges.append(o3d.pipelines.registration.PoseGraphEdge(
                i, best_j, result.transformation, info, uncertain=True))
            accepted += 1
    del pcds

    o3d.pipelines.registration.global_optimization(
        pg,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=POSE["icp_dist"], edge_prune_threshold=0.25, reference_node=0))

    corrections = {}
    for i, f in enumerate(kfs):
        C = np.asarray(pg.nodes[i].pose) @ np.linalg.inv(poses[f])
        if POSE["yaw_only"]:
            v = C[:3, :3] @ np.array([1.0, 0.0, 0.0])
            yaw = np.arctan2(-v[2], v[0])
            cy, sy_ = np.cos(yaw), np.sin(yaw)
            C[:3, :3] = np.array([[cy, 0, sy_], [0, 1, 0], [-sy_, 0, cy]])
        corrections[f] = C
    deltas = [np.linalg.norm(corrections[f][:3, 3]) for f in kfs]
    stats = {"n_keyframes": len(kfs), "n_loop_edges": accepted,
             "correction_mean_cm": float(np.mean(deltas) * 100), "correction_max_cm": float(np.max(deltas) * 100)}
    logger.info("[R01 preprocess] pose 보정: 키프레임 %d, loop edge %d, 보정량 평균 %.1fcm 최대 %.1fcm",
                len(kfs), accepted, stats["correction_mean_cm"], stats["correction_max_cm"])

    refined = {}
    kf_arr = np.array(kfs)
    for f in frames:
        idx = int(np.searchsorted(kf_arr, f, side="right")) - 1
        if idx < 0:
            C = corrections[kfs[0]]
        elif idx >= len(kfs) - 1:
            C = corrections[kfs[-1]]
        else:
            a, b = kfs[idx], kfs[idx + 1]
            t = (stamps[f] - stamps[a]) / max(stamps[b] - stamps[a], 1e-9)
            Ca, Cb = corrections[a], corrections[b]
            slerp = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([Ca[:3, :3], Cb[:3, :3]])))
            C = np.eye(4)
            C[:3, :3] = slerp([t]).as_matrix()[0]
            C[:3, 3] = (1 - t) * Ca[:3, 3] + t * Cb[:3, 3]
        refined[f] = C @ poses[f]
    return refined, stats


def preprocess(scan: Path, out_dir: Path, frame_stride: int, refine: bool = True) -> Preprocessed:
    images_dir = out_dir / "images"
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    sizes = probe_sizes(scan)
    rgb_w, rgb_h = sizes["rgb"]
    K = np.loadtxt(scan / "camera_matrix.csv", delimiter=",")
    odo_raw = np.loadtxt(scan / "odometry.csv", delimiter=",", skiprows=1)
    poses, stamps = load_poses(scan / "odometry.csv")
    depth_frames = {int(f.stem) for f in (scan / "depth").glob("*.png")}
    common = sorted(set(poses) & depth_frames)
    assert common, "odometry와 depth 프레임 번호가 겹치지 않음"

    keyframes = load_keyframes(scan)
    common_set = set(common)
    keyframes = {f: v for f, v in keyframes.items() if f in common_set}

    rgb_offset, sharpness = estimate_rgb_offset(scan / "rgb.mp4", odo_raw)
    common_poses = {f: poses[f] for f in common}
    if refine:
        refined, pose_stats = refine_poses(common_poses, stamps, scan, K, sizes)
    else:
        refined, pose_stats = common_poses, {"refine": False}

    rotator = OdometryRotator({f: stamps[f] for f in common}, refined)
    if keyframes:
        default_exp_s = float(np.median([kf["exposure_ms"] for kf in keyframes.values()])) / 1000.0
    else:
        default_exp_s = 16.67 / 1000.0

    # 프레임 선택: stride 프레임마다 구간 하나에서 가장 덜 흔들린 1장
    stride = max(1, frame_stride)
    n_video = len(sharpness)
    selected = {}
    for w0 in range(0, len(common), stride):
        window = common[w0:w0 + stride]
        kf_in_win = [c for c in window if c in keyframes]
        if kf_in_win:
            if len(kf_in_win) == 1:
                selected[kf_in_win[0]] = "keyframe"
                continue
            g = np.array([keyframes[c]["gyro_mag"] for c in kf_in_win])
            r = np.array([rotator.exposure_rotation(stamps[c], keyframes[c]["exposure_ms"] / 1000.0)
                          for c in kf_in_win])
            score = -(rank_norm(g) + rank_norm(r))
            selected[kf_in_win[int(np.argmax(score))]] = "keyframe"
            continue
        cand_frames = [c for c in window if 0 <= c - rgb_offset < n_video]
        if not cand_frames:
            continue
        if len(cand_frames) == 1:
            selected[cand_frames[0]] = "video"
            continue
        sh = np.array([sharpness[c - rgb_offset] for c in cand_frames])
        r = np.array([rotator.exposure_rotation(stamps[c], default_exp_s) for c in cand_frames])
        score = rank_norm(sh) - rank_norm(r)
        selected[cand_frames[int(np.argmax(score))]] = "video"

    # 텍스처 소스 저장 (원본 해상도). 해상도가 intrinsics 기준과 어긋나면 텍스처가 조용히 밀리므로 검증
    intrinsics = {}
    size_checked = False
    for c in sorted(f for f, src in selected.items() if src == "keyframe"):
        if not size_checked:
            img = cv2.imread(str(keyframes[c]["path"]))
            assert img is not None and img.shape[1] == rgb_w and img.shape[0] == rgb_h, \
                f"keyframe 해상도가 rgb.mp4({rgb_w}x{rgb_h})와 다름"
            size_checked = True
        shutil.copy(keyframes[c]["path"], images_dir / f"frame_{c:06d}.jpg")
        kf = keyframes[c]
        intrinsics[c] = (kf["fx"], kf["fy"], kf["cx"], kf["cy"])

    video_selected = {c for c, src in selected.items() if src == "video"}
    want = {c - rgb_offset: c for c in video_selected if 0 <= c - rgb_offset < n_video}
    cap = cv2.VideoCapture(str(scan / "rgb.mp4"))
    video_idx = -1
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        video_idx += 1
        c = want.get(video_idx)
        if c is None:
            continue
        assert bgr.shape[1] == rgb_w and bgr.shape[0] == rgb_h, "rgb.mp4 프레임 해상도가 일정하지 않음"
        cv2.imwrite(str(images_dir / f"frame_{c:06d}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        intrinsics[c] = (float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))
    cap.release()

    frames = sorted(intrinsics)
    assert frames, "선택된 프레임이 없음"
    n_kf = sum(1 for c in frames if selected.get(c) == "keyframe")
    logger.info("[R01 preprocess] 유효 %d프레임, 텍스처 소스 %d장 (keyframe %d, video %d, stride %d)",
                len(common), len(frames), n_kf, len(frames) - n_kf, stride)
    return Preprocessed(
        scan=scan, images_dir=images_dir, poses=refined, stamps={f: stamps[f] for f in common},
        frames=frames, intrinsics=intrinsics, rgb_size=(rgb_w, rgb_h), depth_size=sizes["depth"],
        rgb_offset=rgb_offset,
        stats={"n_common": len(common), "n_selected": len(frames), "n_keyframe_sourced": n_kf,
               "rgb_offset": rgb_offset, "stride": stride, **pose_stats},
    )
