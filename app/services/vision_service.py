import asyncio
import base64
import gc
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("app")

import cv2
import fal_client
import numpy as np
from openai import AsyncOpenAI
from scipy.spatial.transform import Rotation

from app.core.config import settings
from app.core.storage import upload_to_s3
from app.models.response import ComparisonDefectItem, DefectItem, DefectType, Point3D, Severity
from app.services.frame_service import extract_frames
from app.services.pose_service import polygon_to_3d

_openai = AsyncOpenAI(api_key=settings.openai_api_key)
os.environ.setdefault("FAL_KEY", settings.fal_api_key)

_PROMPT_DIR = Path(__file__).parent.parent / "prompts"
_SYSTEM_PROMPT = (_PROMPT_DIR / "defect_detection.txt").read_text(encoding="utf-8")
_VERIFY_PROMPT = (_PROMPT_DIR / "defect_verification.txt").read_text(encoding="utf-8")

_BATCH_SIZE = 6
_MIN_CONFIDENCE = 0.5
_MATCH_THRESHOLD_M = 0.3
_DUP_MAX_TRANS_M = 0.15  # 중복 시점 스킵: 이동 15cm 미만
_DUP_MAX_ROT_DEG = 15.0  # 중복 시점 스킵: 회전 15도 미만
_MIN_SIDE_M = 0.005  # 짧은 변 5mm 미만 하자 제외
_SAM3_RETRIES = 2
_LOG_DIR = Path(__file__).parent.parent.parent / "logs"

_DETECTION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "defect_detection",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "defects": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": ["SCRATCH", "CRACK", "PEELING", "STAIN", "BREAKAGE"]},
                            "severity": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]},
                            "location": {"type": "string"},
                            "description": {"type": "string"},
                            "bbox": {"type": "array", "items": {"type": "number"}},
                            "frame_index": {"type": "integer"},
                            "confidence": {"type": "number"},
                        },
                        "required": ["type", "severity", "location", "description", "bbox", "frame_index", "confidence"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["defects"],
            "additionalProperties": False,
        },
    },
}

_VERIFY_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "defect_verification",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "confirmed": {"type": "boolean"},
                "reason": {"type": "string"},
                "bbox": {"type": "array", "items": {"type": "number"}},
            },
            "required": ["confirmed", "reason", "bbox"],
            "additionalProperties": False,
        },
    },
}


def _load_fewshot() -> list[dict]:
    manifest_path = _PROMPT_DIR / "fewshot" / "fewshot.json"
    if not manifest_path.exists():
        return []
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    content: list[dict] = []
    for e in entries:
        img_path = manifest_path.parent / e["file"]
        mime = "image/png" if img_path.suffix.lower() == ".png" else "image/jpeg"
        b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")
        content.append({"type": "text", "text": f"[example] {e['defect']} ({e['severity']}): {e['description']}"})
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"}})
    return content


_FEWSHOT_CONTENT = _load_fewshot()


def _build_poses(odometry_path: Path) -> dict[int, np.ndarray]:
    data = np.loadtxt(odometry_path, delimiter=",", skiprows=1)
    poses: dict[int, np.ndarray] = {}
    for row in data:
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(row[5:]).as_matrix()
        T[:3, 3] = row[2:5]
        poses[int(row[1])] = T
    return poses


def _load_depth(depth_dir: Path, conf_dir: Path, frame_idx: int) -> np.ndarray | None:
    depth = cv2.imread(str(depth_dir / f"{frame_idx:06d}.png"), cv2.IMREAD_UNCHANGED)
    if depth is None:
        return None
    depth_m = depth.astype(np.float32) / 1000.0
    conf_path = conf_dir / f"{frame_idx:06d}.png"
    if conf_path.exists():
        conf = cv2.imread(str(conf_path), cv2.IMREAD_UNCHANGED)
        depth_m[conf < 1] = 0.0
    return depth_m


def _centroid(region_3d: list[Point3D]) -> np.ndarray:
    pts = np.array([[p.x, p.y, p.z] for p in region_3d])
    return pts.mean(axis=0)


def _scale_camera_matrix(K: np.ndarray, sx: float, sy: float) -> np.ndarray:
    K_scaled = K.copy()
    K_scaled[0, 0] *= sx
    K_scaled[1, 1] *= sy
    K_scaled[0, 2] *= sx
    K_scaled[1, 2] *= sy
    return K_scaled


def _bbox_center_3d(
    bbox: list[float],
    depth_map: np.ndarray,
    K: np.ndarray,
    T_WC: np.ndarray,
    img_w: int,
    img_h: int,
) -> np.ndarray | None:
    depth_h, depth_w = depth_map.shape
    sx = depth_w / img_w
    sy = depth_h / img_h

    x1 = max(0, int(bbox[0] * sx))
    y1 = max(0, int(bbox[1] * sy))
    x2 = min(depth_w, int(bbox[2] * sx))
    y2 = min(depth_h, int(bbox[3] * sy))
    if x2 <= x1 or y2 <= y1:
        return None

    region = depth_map[y1:y2, x1:x2]
    valid = region[region > 0]
    if valid.size == 0:
        return None
    depth = float(np.median(valid))

    K_depth = _scale_camera_matrix(K, sx, sy)
    u = (x1 + x2) / 2
    v = (y1 + y2) / 2
    x_c = (u - K_depth[0, 2]) * depth / K_depth[0, 0]
    y_c = (v - K_depth[1, 2]) * depth / K_depth[1, 1]
    p_w = T_WC @ np.array([x_c, y_c, depth, 1.0])
    return p_w[:3]


def _normalized_to_pixel_bbox(bbox_n: list[float], img_w: int, img_h: int) -> list[float] | None:
    if max(bbox_n) > 1200:  # 정규화 지시를 무시하고 픽셀 좌표를 반환한 경우
        return None
    x1 = max(0.0, min(bbox_n[0], bbox_n[2]) / 1000 * img_w)
    y1 = max(0.0, min(bbox_n[1], bbox_n[3]) / 1000 * img_h)
    x2 = min(float(img_w), max(bbox_n[0], bbox_n[2]) / 1000 * img_w)
    y2 = min(float(img_h), max(bbox_n[1], bbox_n[3]) / 1000 * img_h)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return [x1, y1, x2, y2]


def _expand_bbox(
    bbox: list[float], img_w: int, img_h: int, factor: float = 2.5, min_size: int = 240
) -> tuple[int, int, int, int]:
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    w = max((bbox[2] - bbox[0]) * factor, min_size)
    h = max((bbox[3] - bbox[1]) * factor, min_size)
    x1 = max(0, int(cx - w / 2))
    y1 = max(0, int(cy - h / 2))
    x2 = min(img_w, int(cx + w / 2))
    y2 = min(img_h, int(cy + h / 2))
    return x1, y1, x2, y2


def _crop_around_bbox(img: np.ndarray, bbox: list[float]) -> np.ndarray | None:
    img_h, img_w = img.shape[:2]
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    bw = bbox[2] - bbox[0]
    bh = bbox[3] - bbox[1]
    w = max(bw, bh / 1.6)
    h = w * 1.6
    x1 = max(0, int(cx - w / 2))
    y1 = max(0, int(cy - h / 2))
    x2 = min(img_w, int(cx + w / 2))
    y2 = min(img_h, int(cy + h / 2))
    crop = img[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


async def _gpt_detect(frames: list[tuple[int, str]]) -> list[dict]:
    content: list = []
    for frame_idx, data_uri in frames:
        content.append({"type": "text", "text": f"[frame_index: {frame_idx}]"})
        content.append({"type": "image_url", "image_url": {"url": data_uri, "detail": "high"}})

    messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
    if _FEWSHOT_CONTENT:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Labeled reference examples. Do NOT report defects from these images."},
                *_FEWSHOT_CONTENT,
            ],
        })
        messages.append({
            "role": "assistant",
            "content": "Understood. I will use these examples as reference and only report defects from the frames that follow.",
        })
    messages.append({"role": "user", "content": content})

    response = await _openai.chat.completions.create(
        model="gpt-5.5",
        messages=messages,
        response_format=_DETECTION_SCHEMA,
        reasoning_effort="low",
    )
    return json.loads(response.choices[0].message.content)["defects"]


async def _gpt_detect_batched(frames: list[tuple[int, str]]) -> list[dict]:
    batches = [frames[i:i + _BATCH_SIZE] for i in range(0, len(frames), _BATCH_SIZE)]
    first = await asyncio.gather(_gpt_detect(batches[0]), return_exceptions=True)
    rest = await asyncio.gather(*[_gpt_detect(b) for b in batches[1:]], return_exceptions=True)
    results = list(first) + list(rest)

    defects: list[dict] = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            logger.warning("[D01] 배치 %d/%d GPT 호출 실패: %s", i + 1, len(batches), r)
            continue
        defects.extend(r)
    return defects


async def _gpt_verify(crop_bytes: bytes, defect: dict) -> dict:
    b64 = base64.b64encode(crop_bytes).decode("utf-8")
    claimed = f"type={defect['type']}, severity={defect['severity']}, description={defect['description']}"
    response = await _openai.chat.completions.create(
        model="gpt-5.5",
        messages=[
            {"role": "system", "content": _VERIFY_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Claimed defect: {claimed}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}},
                ],
            },
        ],
        response_format=_VERIFY_SCHEMA,
        reasoning_effort="low",
    )
    return json.loads(response.choices[0].message.content)


def _sam3_segment(image_bytes: bytes, bbox: list[float]) -> np.ndarray | None:
    import httpx

    image_url = fal_client.upload(image_bytes, "image/jpeg")
    result = fal_client.subscribe(
        "fal-ai/sam-3/image",
        arguments={
            "image_url": image_url,
            "box_prompts": [{"x_min": int(bbox[0]), "y_min": int(bbox[1]), "x_max": int(bbox[2]), "y_max": int(bbox[3])}],
            "apply_mask": False,
            "output_format": "png",
        },
    )

    masks = result.get("masks", [])
    if not masks:
        return None

    mask_bytes = httpx.get(masks[0]["url"]).content
    mask_arr = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    if mask_arr is None:
        return None

    if mask_arr.ndim == 3:
        mask_arr = mask_arr[:, :, 3] if mask_arr.shape[2] == 4 else cv2.cvtColor(mask_arr, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(mask_arr, 127, 255, cv2.THRESH_BINARY)
    return binary if np.any(binary) else None


def _mask_to_polygon(mask: np.ndarray) -> list[tuple[float, float]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    epsilon = 0.005 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    return [(float(p[0][0]), float(p[0][1])) for p in approx]


def _mask_depth_median(mask: np.ndarray, depth_map: np.ndarray) -> tuple[float | None, bool]:
    """마스크 영역의 유효 depth 중앙값. 영역에 depth가 없으면 프레임 전체로 어림(estimated=True)."""
    depth_h, depth_w = depth_map.shape
    # INTER_AREA: 가는 마스크도 축소 후 0이 되지 않도록 부분 커버 픽셀을 남긴다
    mask_d = cv2.resize(mask, (depth_w, depth_h), interpolation=cv2.INTER_AREA)
    valid = depth_map[(mask_d > 0) & (depth_map > 0)]
    if valid.size > 0:
        return float(np.median(valid)), False
    valid = depth_map[depth_map > 0]
    if valid.size > 0:
        return float(np.median(valid)), True
    return None, True


def _mask_area_cm2(mask: np.ndarray, depth: float, K: np.ndarray) -> float:
    """RGB 해상도 마스크 픽셀 수 × 픽셀당 실면적(depth²/(fx·fy))"""
    n = int(np.count_nonzero(mask))
    return n * depth ** 2 / (K[0, 0] * K[1, 1]) * 1e4


def _mask_short_side_m(mask: np.ndarray, depth: float, K: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    side_px = min(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
    return float(side_px * depth / min(K[0, 0], K[1, 1]))


# iOS 3D 뷰어의 하자 영역 표시와 동일한 systemBlue #007AFF (BGR), 채움/테두리 알파 0.15
_OVERLAY_COLOR = (255, 122, 0)
_OVERLAY_ALPHA = 0.15


def _draw_seg_overlay(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vis = img.copy()
    colored = img.copy()
    colored[mask > 0] = _OVERLAY_COLOR
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(colored, contours, -1, _OVERLAY_COLOR, 2)
    cv2.addWeighted(colored, _OVERLAY_ALPHA, vis, 1 - _OVERLAY_ALPHA, 0, vis)
    return vis


async def _segment_with_retry(jpeg: bytes, bbox: list[float]) -> np.ndarray | None:
    for attempt in range(1 + _SAM3_RETRIES):
        try:
            mask = await asyncio.to_thread(_sam3_segment, jpeg, bbox)
        except Exception as e:
            mask = None
            logger.warning("[D01] SAM3 호출 예외 (시도 %d/%d): %s", attempt + 1, 1 + _SAM3_RETRIES, e)
        if mask is not None:
            return mask
    return None


def _log_dropped(scan_id: int | str, reason: str, d: dict, frame_idx: int, bbox: list[float]) -> None:
    """콜백에서 제외한 하자를 logs/{scan_id}.log에 기록. 개인정보(위치 설명, URL)는 남기지 않는다."""
    _LOG_DIR.mkdir(exist_ok=True)
    line = (
        f"{datetime.now():%Y-%m-%d %H:%M:%S} | {reason} | "
        f"type={d['type']} frame={frame_idx} bbox={[round(v) for v in bbox]}\n"
    )
    with open(_LOG_DIR / f"{scan_id}.log", "a", encoding="utf-8") as f:
        f.write(line)


def _filter_duplicate_frames(
    frames: list[tuple[int, str]],
    poses: dict[int, np.ndarray],
    max_trans_m: float,
    max_rot_deg: float,
) -> list[tuple[int, str]]:
    kept: list[tuple[int, str]] = []
    kept_poses: list[np.ndarray] = []
    for frame_idx, data_uri in frames:
        T = poses.get(frame_idx)
        if T is None:
            kept.append((frame_idx, data_uri))
            continue

        duplicate = False
        for T_k in kept_poses:
            if float(np.linalg.norm(T[:3, 3] - T_k[:3, 3])) >= max_trans_m:
                continue
            R_rel = T_k[:3, :3].T @ T[:3, :3]
            angle = float(np.degrees(np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1.0, 1.0))))
            if angle < max_rot_deg:
                duplicate = True
                break

        if not duplicate:
            kept.append((frame_idx, data_uri))
            kept_poses.append(T)
    return kept


async def _upload_with_retry(data: bytes, key: str) -> str:
    for attempt in range(3):
        try:
            return await asyncio.to_thread(upload_to_s3, data, key)
        except Exception as e:
            if attempt == 2:
                raise
            logger.warning("[D01] S3 업로드 실패, 재시도 %d/2 key=%s error=%s", attempt + 1, key, e)


def _dedup_candidates(candidates: list[dict]) -> list[dict]:
    kept: list[dict] = []
    for c in sorted(candidates, key=lambda c: c["defect"]["confidence"], reverse=True):
        if c["center"] is not None and any(
            k["center"] is not None
            and float(np.linalg.norm(c["center"] - k["center"])) < _MATCH_THRESHOLD_M
            for k in kept
        ):
            continue
        kept.append(c)
    return kept


async def _run_detection(
    scan_id: int | str,
    video_path: Path,
    depth_dir: Path,
    conf_dir: Path,
    camera_matrix_path: Path,
    odometry_path: Path,
    frames_per_sec: float = 1.0,
) -> list[DefectItem]:
    poses = _build_poses(odometry_path)
    K = np.loadtxt(camera_matrix_path, delimiter=",")
    frames = extract_frames(video_path, frames_per_sec)
    n_extracted = len(frames)
    frames = _filter_duplicate_frames(frames, poses, _DUP_MAX_TRANS_M, _DUP_MAX_ROT_DEG)
    if len(frames) < n_extracted:
        logger.info("[D01] 중복 시점 프레임 스킵 %d장 → %d장", n_extracted, len(frames))
    frame_dict: dict[int, str] = dict(frames)
    raw_defects = await _gpt_detect_batched(frames)
    logger.info("[D01] GPT 탐지 결과 %d개 (프레임 %d개, 배치 %d개 분석)",
                len(raw_defects), len(frames), (len(frames) + _BATCH_SIZE - 1) // _BATCH_SIZE)
    del frames

    # 1) confidence 필터 + 필수 값 검증
    depth_cache: dict[int, np.ndarray | None] = {}
    img_cache: dict[int, np.ndarray | None] = {}

    def _get_depth(frame_idx: int) -> np.ndarray | None:
        if frame_idx not in depth_cache:
            depth_cache[frame_idx] = _load_depth(depth_dir, conf_dir, frame_idx)
        return depth_cache[frame_idx]

    def _get_jpeg(frame_idx: int) -> bytes | None:
        data_uri = frame_dict.get(frame_idx)
        if data_uri is None:
            return None
        _, encoded = data_uri.split(",", 1)
        return base64.b64decode(encoded)

    def _get_img(frame_idx: int) -> np.ndarray | None:
        if frame_idx not in img_cache:
            jpeg = _get_jpeg(frame_idx)
            img_cache[frame_idx] = (
                cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR) if jpeg is not None else None
            )
        return img_cache[frame_idx]

    candidates: list[dict] = []
    for d in raw_defects:
        if d["confidence"] < _MIN_CONFIDENCE:
            logger.info("[D01] confidence 미달로 제외 type=%s confidence=%.2f", d["type"], d["confidence"])
            continue
        if len(d["bbox"]) != 4:
            logger.warning("[D01] bbox 형식 오류로 제외 bbox=%s", d["bbox"])
            continue

        frame_idx: int = d["frame_index"]
        if frame_dict.get(frame_idx) is None:
            logger.warning("[D01] 프레임 없음 frame_idx=%s (GPT가 잘못된 frame_index 반환)", frame_idx)
            continue
        if poses.get(frame_idx) is None:
            logger.warning("[D01] pose 없음 frame_idx=%s — odometry에 해당 프레임 없어 skip", frame_idx)
            continue

        img_arr = _get_img(frame_idx)
        if img_arr is None:
            logger.warning("[D01] 이미지 디코딩 실패 frame_idx=%s", frame_idx)
            continue
        depth_map = _get_depth(frame_idx)
        if depth_map is None:
            logger.warning("[D01] depth 파일 없음 frame_idx=%s", frame_idx)
            continue

        img_h, img_w = img_arr.shape[:2]
        bbox = _normalized_to_pixel_bbox(d["bbox"], img_w, img_h)
        if bbox is None:
            logger.warning("[D01] bbox 좌표 이상으로 제외 bbox=%s", d["bbox"])
            continue
        center = _bbox_center_3d(bbox, depth_map, K, poses[frame_idx], img_w, img_h)
        candidates.append({"defect": d, "frame_idx": frame_idx, "bbox": bbox, "center": center})

    # 2) 배치 간 3D 중복 제거 (검증 전에 수행해 검증 호출 수 최소화)
    deduped = _dedup_candidates(candidates)
    logger.info("[D01] 중복 제거 %d개 → %d개", len(candidates), len(deduped))

    # 탈락한 프레임의 캐시 해제
    keep_frames = {c["frame_idx"] for c in deduped}
    for cache in (img_cache, depth_cache):
        for fi in [fi for fi in cache if fi not in keep_frames]:
            del cache[fi]

    # 3) 위치 재보정 — bbox 주변을 여유 있게 잘라 정확한 bbox를 다시 받는다 (병렬).
    # 하자를 기각하지 않는다: 재보정에 실패해도 원본 bbox로 유지.
    async def _refine(c: dict) -> None:
        img = _get_img(c["frame_idx"])
        img_h, img_w = img.shape[:2]
        vx1, vy1, vx2, vy2 = _expand_bbox(c["bbox"], img_w, img_h)
        crop = img[vy1:vy2, vx1:vx2]
        if crop.size == 0:
            return

        # 작은 크롭은 확대해서 위치를 찍기 쉽게
        ch, cw = crop.shape[:2]
        if max(ch, cw) < 512:
            s = 512 / max(ch, cw)
            crop = cv2.resize(crop, (int(cw * s), int(ch * s)), interpolation=cv2.INTER_CUBIC)

        _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        try:
            result = await _gpt_verify(buf.tobytes(), c["defect"])
        except Exception as e:
            logger.warning("[D01] 위치 재보정 호출 실패 — 원본 bbox 유지: %s", e)
            return

        if not result["confirmed"]:
            logger.info("[D01] 재보정 단계 하자 미확인(기각 없이 유지) type=%s reason=%s",
                        c["defect"]["type"], result["reason"])

        # 크롭 안에서 다시 찍은 정규화 좌표(0~1000)를 원본 픽셀로 역변환해 bbox 보정
        rb = result["bbox"]
        if len(rb) == 4 and 0 <= min(rb) and max(rb) <= 1000:
            x1 = vx1 + min(rb[0], rb[2]) / 1000 * (vx2 - vx1)
            y1 = vy1 + min(rb[1], rb[3]) / 1000 * (vy2 - vy1)
            x2 = vx1 + max(rb[0], rb[2]) / 1000 * (vx2 - vx1)
            y2 = vy1 + max(rb[1], rb[3]) / 1000 * (vy2 - vy1)
            if x2 - x1 >= 2 and y2 - y1 >= 2:
                c["bbox"] = [x1, y1, x2, y2]

    await asyncio.gather(*[_refine(c) for c in deduped])
    logger.info("[D01] 위치 재보정 완료 %d개 (기각 없음)", len(deduped))

    # 4) SAM3 세그멘테이션 + 3D 역투영 + area 실측 + 시각화 업로드
    items: list[DefectItem] = []
    for c in deduped:
        d = c["defect"]
        frame_idx = c["frame_idx"]
        bbox: list[float] = c["bbox"]
        img_arr = _get_img(frame_idx)
        depth_map = _get_depth(frame_idx)
        img_h, img_w = img_arr.shape[:2]
        depth_h, depth_w = depth_map.shape
        sx = depth_w / img_w
        sy = depth_h / img_h
        K_depth = _scale_camera_matrix(K, sx, sy)

        mask = await _segment_with_retry(_get_jpeg(frame_idx), bbox)
        if mask is None:
            logger.warning("[D01] SAM3 세그멘테이션 실패로 제외 frame_idx=%s", frame_idx)
            _log_dropped(scan_id, "SAM3 실패", d, frame_idx, bbox)
            continue

        # 면적: RGB 해상도 마스크 픽셀 수 × 마스크 영역 depth 중앙값 기준 픽셀 실면적
        depth, estimated = _mask_depth_median(mask, depth_map)
        if depth is None:
            logger.warning("[D01] 프레임 depth 없음으로 제외 frame_idx=%s", frame_idx)
            _log_dropped(scan_id, "depth 없음", d, frame_idx, bbox)
            continue
        if estimated:
            logger.info("[D01] 마스크 영역 depth 없음, 프레임 전체 depth로 면적 어림 frame_idx=%s", frame_idx)
        short_side = _mask_short_side_m(mask, depth, K)
        if short_side < _MIN_SIDE_M:
            logger.info("[D01] 짧은 변 %.1fmm로 제외 type=%s frame_idx=%s", short_side * 1000, d["type"], frame_idx)
            _log_dropped(scan_id, f"짧은 변 {short_side * 1000:.1f}mm", d, frame_idx, bbox)
            continue
        area = round(_mask_area_cm2(mask, depth, K), 2)

        # region_3d: 마스크에서 단순화 폴리곤을 뽑아 depth 해상도로 스케일 후 역투영
        region_3d: list[Point3D] = []
        polygon_2d = _mask_to_polygon(mask)
        if polygon_2d:
            polygon_depth = [(u * sx, v * sy) for u, v in polygon_2d]
            region_3d = polygon_to_3d(polygon_depth, depth_map, K_depth, poses[frame_idx])
            if not region_3d:
                logger.warning("[D01] region_3d 비어있음 frame_idx=%s polygon=%d개 depth_nonzero=%d",
                               frame_idx, len(polygon_2d), int(np.count_nonzero(depth_map)))

        # 시각화 2종: 크롭+세그멘테이션(콜백 포함) / 크롭+bbox(S3 백업)
        image_url = None
        try:
            key_base = uuid.uuid4().hex

            seg_crop = _crop_around_bbox(_draw_seg_overlay(img_arr, mask), bbox)
            if seg_crop is not None:
                _, buf = cv2.imencode(".jpg", seg_crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
                image_url = await _upload_with_retry(buf.tobytes(), f"defects/{key_base}.jpg")

            bbox_vis = img_arr.copy()
            cv2.rectangle(bbox_vis, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 3)
            bbox_crop = _crop_around_bbox(bbox_vis, bbox)
            if bbox_crop is not None:
                _, buf = cv2.imencode(".jpg", bbox_crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
                await _upload_with_retry(buf.tobytes(), f"defects/{key_base}_bbox.jpg")
        except Exception as e:
            logger.warning("[D01] 하자 이미지 업로드 실패 frame_idx=%s error=%s", frame_idx, e)

        items.append(
            DefectItem(
                type=DefectType(d["type"]),
                severity=Severity(d["severity"]),
                location=d["location"],
                area=area,
                description=d["description"],
                region_3d=region_3d,
                image_url=image_url,
            )
        )

    depth_cache.clear()
    img_cache.clear()
    gc.collect()
    return items


async def detect_defects(
    scan_id: int | str,
    video_path: Path,
    depth_dir: Path,
    conf_dir: Path,
    camera_matrix_path: Path,
    odometry_path: Path,
    frames_per_sec: float = 1.0,
) -> list[DefectItem]:
    return await _run_detection(
        scan_id, video_path, depth_dir, conf_dir, camera_matrix_path, odometry_path, frames_per_sec
    )


async def compare_defects(
    in_defects: list[DefectItem],
    out_defects: list[DefectItem],
) -> list[ComparisonDefectItem]:
    in_centroids = [_centroid(d.region_3d) for d in in_defects if d.region_3d]

    results: list[ComparisonDefectItem] = []
    for d in out_defects:
        if not d.region_3d:
            continue
        out_centroid = _centroid(d.region_3d)

        is_existing = any(
            float(np.linalg.norm(out_centroid - c)) < _MATCH_THRESHOLD_M
            for c in in_centroids
        )
        if is_existing:
            continue

        results.append(
            ComparisonDefectItem(
                type=d.type,
                severity=d.severity,
                location=d.location,
                area=d.area,
                description=d.description,
                region_3d=d.region_3d,
                image_url=d.image_url,
            )
        )

    return results
