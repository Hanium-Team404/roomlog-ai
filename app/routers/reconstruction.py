import asyncio
import ctypes
import ctypes.util
import gc
import logging
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks

from app.core.callback import post_reconstruction_result, post_thumbnail_result
from app.core.storage import download_scan_zip, upload_file_to_s3, upload_to_s3
from app.models.request import ReconstructionRequest
from app.models.response import APIResponse, ErrorResponse
from app.services.reconstruction import run_reconstruction
from app.services.thumbnail_service import generate_thumbnail, get_render_frames

logger = logging.getLogger(__name__)
router = APIRouter(tags=["AI-R01. 3D 재구성"])


def _release_memory() -> None:
    gc.collect()
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c"))
        libc.malloc_trim(0)
        logger.info("[R01] malloc_trim 완료 — OS 메모리 반환")
    except Exception as e:
        logger.warning("[R01] malloc_trim 실패 (무시): %s", e)


async def _run(body: ReconstructionRequest) -> None:
    logger.info("[R01] 시작 scan_id=%s", body.scan_id)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            logger.info("[R01] ZIP 다운로드 중")
            scan = await asyncio.to_thread(download_scan_zip, body.scan_url, str(tmp_path / "scan"))

            logger.info("[R01] 재구성 처리 중 (전처리, TSDF, 텍스처링, usdz)")
            result = await asyncio.to_thread(run_reconstruction, scan.root, tmp_path / "work")

            # S3에는 최종 usdz 하나만 올린다. 메시(ply)는 썸네일 렌더에만 쓰고 업로드하지 않는다
            logger.info("[R01] S3 업로드 중")
            prefix = f"scans/{body.scan_id}/{uuid.uuid4().hex}"
            usdz_url = await asyncio.to_thread(upload_file_to_s3, result.usdz_path, f"{prefix}/room.usdz")

            # 1차 콜백: usdz 완료 즉시 전송 — 썸네일 생성을 기다리지 않는다
            logger.info("[R01] 1차 콜백 전송 중 usdz_url=%s", usdz_url)
            await post_reconstruction_result(body.callback_url, body.scan_id, usdz_url)
            logger.info("[R01] usdz 완료 scan_id=%s stats=%s", body.scan_id, result.stats)

            # 썸네일은 1차 콜백 이후 진행 — 실패해도 usdz 결과에는 영향 없음
            try:
                logger.info("[R01] 썸네일 생성 중")
                frames = await asyncio.to_thread(get_render_frames, result.mesh_path)
                await asyncio.gather(*[
                    asyncio.to_thread(upload_to_s3, frame, f"{prefix}/view_{i}.png")
                    for i, frame in enumerate(frames)
                ])
                logger.info("[R01] 렌더링 프레임 S3 업로드 완료")

                thumbnail_bytes = await generate_thumbnail(frames)
                del frames
                thumbnail_url = await asyncio.to_thread(
                    upload_to_s3, thumbnail_bytes, f"{prefix}/thumbnail.jpg"
                )
                del thumbnail_bytes
                logger.info("[R01] 썸네일 S3 업로드 완료")

                await post_thumbnail_result(body.callback_url, body.scan_id, thumbnail_url)
            except Exception as e:
                logger.error("[R01] 썸네일 처리 실패 (mesh는 완료) scan_id=%s error=%s", body.scan_id, e, exc_info=True)

        logger.info("[R01] 완료 scan_id=%s", body.scan_id)
    except Exception as e:
        logger.error("[R01] 실패 scan_id=%s error=%s", body.scan_id, e, exc_info=True)
    finally:
        await asyncio.to_thread(_release_memory)


@router.post(
    "/reconstruction",
    summary="AI-R01. 3D 재구성",
    status_code=202,
    response_model=APIResponse[None],
    responses={
        500: {"model": ErrorResponse, "description": "재구성 처리 중 오류 (AI_REC_001)"},
    },
)
async def reconstruct(body: ReconstructionRequest, background_tasks: BackgroundTasks) -> APIResponse[None]:
    background_tasks.add_task(_run, body)
    return APIResponse(code=202, message="처리 중")
