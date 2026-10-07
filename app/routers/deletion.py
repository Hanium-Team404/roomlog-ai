import asyncio
import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.core.storage import delete_from_s3, delete_prefix_from_s3, s3_url_prefix
from app.models.request import DefectDeletionRequest
from app.models.response import APIResponse, DeletionData, ErrorResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["AI-X01. 산출물 삭제"])


@router.delete(
    "/scans/{scan_id}",
    summary="AI-X01. 스캔 3D 산출물 삭제",
    response_model=APIResponse[DeletionData],
)
async def delete_scan(scan_id: int) -> APIResponse[DeletionData]:
    deleted = await asyncio.to_thread(delete_prefix_from_s3, f"scans/{scan_id}/")
    logger.info("[X01] 스캔 산출물 삭제 scan_id=%s deleted=%d", scan_id, deleted)
    return APIResponse(data=DeletionData(deleted=deleted))


@router.delete(
    "/defects",
    summary="AI-X01. 하자 이미지 삭제",
    response_model=APIResponse[DeletionData],
    responses={
        400: {"model": ErrorResponse, "description": "이 서버가 발급한 하자 이미지 URL이 아님 (AI_DEL_001)"},
    },
)
async def delete_defects(body: DefectDeletionRequest):
    prefix = s3_url_prefix() + "defects/"
    invalid = [u for u in body.image_urls if not u.startswith(prefix)]
    if invalid:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "code": 400,
                "message": f"하자 이미지 URL 형식이 아님: {invalid}",
                "error": {"code": "AI_DEL_001"},
                "data": None,
            },
        )

    # 콜백으로 전달한 세그 오버레이와 함께 올라간 bbox 백업(_bbox.jpg)도 같이 삭제
    keys: list[str] = []
    for u in body.image_urls:
        key = u[len(s3_url_prefix()):]
        keys.append(key)
        keys.append(key[: -len(".jpg")] + "_bbox.jpg")
    deleted = await asyncio.to_thread(delete_from_s3, keys) if keys else 0
    logger.info("[X01] 하자 이미지 삭제 urls=%d deleted=%d", len(body.image_urls), deleted)
    return APIResponse(data=DeletionData(deleted=deleted))
