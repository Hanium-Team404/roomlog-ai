import ctypes
import ctypes.util
import gc
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.core.config import settings
from app.services.reconstruction.preprocess import preprocess
from app.services.reconstruction.texture import texture_mesh
from app.services.reconstruction.tsdf import build_mesh
from app.services.reconstruction.usdz import build_usdz

logger = logging.getLogger("app")


def _release_memory() -> None:
    """texrecon(별도 프로세스, 2GB 이상)과 겹치는 동안 컨테이너 한도(3GB)를 넘지 않도록
    파이썬이 쥔 메시 메모리를 OS에 돌려준다. gc만으로는 glibc 힙이 반환되지 않는다."""
    gc.collect()
    try:
        ctypes.CDLL(ctypes.util.find_library("c")).malloc_trim(0)
    except Exception as e:  # malloc_trim은 glibc 전용
        logger.warning("[R01] malloc_trim 실패 (무시): %s", e)


@dataclass
class ReconstructionResult:
    usdz_path: Path          # S3에 올리는 유일한 산출물
    mesh_path: Path          # 정리된 TSDF 메시 (썸네일 렌더용, 업로드 안 함)
    stats: dict = field(default_factory=dict)


def run_reconstruction(scan_root: Path, work_dir: Path) -> ReconstructionResult:
    """raw 스캔 폴더 -> 텍스처 usdz. 전 과정 CPU. work_dir 아래 중간 산출물을 쓴다."""
    t0 = time.time()
    stats: dict = {}

    prep = preprocess(scan_root, work_dir / "01_preprocess", settings.recon_frame_stride,
                      refine=settings.recon_refine_poses)
    stats["preprocess"] = prep.stats
    stats["t_preprocess_s"] = round(time.time() - t0)
    gc.collect()

    t1 = time.time()
    mesh_path, mesh_tex = build_mesh(prep, work_dir / "02_tsdf_mesh", settings.recon_target_faces)
    stats["t_tsdf_s"] = round(time.time() - t1)
    _release_memory()

    t2 = time.time()
    obj_path = texture_mesh(prep, mesh_tex, work_dir / "04_texture", settings.texrecon_bin,
                            settings.recon_num_threads)
    stats["t_texture_s"] = round(time.time() - t2)
    gc.collect()

    t3 = time.time()
    usdz_path = work_dir / "05_usdz" / "room.usdz"
    stats["usdz"] = build_usdz(obj_path, mesh_tex, usdz_path)
    stats["t_usdz_s"] = round(time.time() - t3)
    stats["t_total_s"] = round(time.time() - t0)
    gc.collect()
    logger.info("[R01] 재구성 완료 %ds (전처리 %ds, TSDF %ds, 텍스처 %ds, usdz %ds)",
                stats["t_total_s"], stats["t_preprocess_s"], stats["t_tsdf_s"], stats["t_texture_s"], stats["t_usdz_s"])
    return ReconstructionResult(usdz_path=usdz_path, mesh_path=mesh_path, stats=stats)
