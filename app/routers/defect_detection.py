import asyncio
import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks

from app.core.callback import post_result
from app.core.storage import download_scan_zip
from app.models.request import DefectDetectionRequest
from app.models.response import APIResponse, DefectDetectionData, ErrorResponse
from app.services.vision_service import detect_defects as _detect_defects

logger = logging.getLogger(__name__)
router = APIRouter(tags=["AI-D01. 하자 탐지"])


async def _run(body: DefectDetectionRequest) -> None:
    logger.info("[D01] 시작 analysis_id=%s scan_id=%s", body.analysis_id, body.scan_id)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            logger.info("[D01] ZIP 다운로드 중")
            scan = await asyncio.to_thread(download_scan_zip, body.scan_url, str(tmp_path / "scan"))

            logger.info("[D01] 하자 탐지 중")
            defects = await _detect_defects(
                scan_id=body.scan_id,
                video_path=scan.video_path,
                depth_dir=scan.depth_dir,
                conf_dir=scan.conf_dir,
                camera_matrix_path=scan.camera_matrix_path,
                odometry_path=scan.odometry_path,
            )

        payload = {
            "success": True,
            "code": 200,
            "message": "요청 성공",
            "data": DefectDetectionData(defects=defects).model_dump(),
        }
        logger.info("[D01] 콜백 전송 중 analysis_id=%s scan_id=%s defects=%d개", body.analysis_id, body.scan_id, len(defects))
        await post_result(body.callback_url, payload)
        logger.info("[D01] 완료 analysis_id=%s", body.analysis_id)
    except Exception as e:
        logger.error("[D01] 실패 analysis_id=%s error=%s", body.analysis_id, e, exc_info=True)
        await post_result(
            body.callback_url,
            {
                "success": False,
                "code": 500,
                "message": str(e),
                "error": {"code": "AI_DET_001"},
                "data": None,
            },
        )


@router.post(
    "/defect-detection",
    summary="AI-D01. 하자 탐지",
    status_code=202,
    response_model=APIResponse[None],
    responses={
        500: {"model": ErrorResponse, "description": "하자 탐지 처리 중 오류 (AI_DET_001)"},
    },
)
async def detect_defects(body: DefectDetectionRequest, background_tasks: BackgroundTasks) -> APIResponse[None]:
    background_tasks.add_task(_run, body)
    return APIResponse(code=202, message="처리 중")
