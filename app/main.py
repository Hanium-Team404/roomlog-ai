import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.routers import defect_comparison, defect_detection, deletion, reconstruction

_app_logger = logging.getLogger("app")
_app_logger.setLevel(logging.INFO)
if not _app_logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s: %(message)s"))
    _app_logger.addHandler(_handler)

    # R01/D01/D02 API 호출 로그를 날짜별 파일로도 기록 (append, 자정에 파일 교체, 30일 보관)
    _log_dir = Path(__file__).parent.parent / "logs"
    _log_dir.mkdir(exist_ok=True)
    _file_handler = TimedRotatingFileHandler(_log_dir / "api.log", when="midnight", backupCount=30, encoding="utf-8")
    _file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s:%(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    _app_logger.addHandler(_file_handler)

tags_metadata = [
    {
        "name": "AI-R01. 3D 재구성",
        "description": "스캔 원시 데이터를 받아 TSDF 기반 3D 재구성 후 결과 반환",
    },
    {
        "name": "AI-D01. 하자 탐지",
        "description": "스캔 ZIP 파일을 받아 GPT Vision으로 하자 탐지 후 결과 반환",
    },
    {
        "name": "AI-D02. 입주/퇴거 하자 비교",
        "description": "입주/퇴거 ZIP 또는 기존 탐지 결과 JSON을 받아 새로 생긴 하자 반환",
    },
    {
        "name": "AI-X01. 산출물 삭제",
        "description": "S3에 저장된 3D 재구성 산출물과 하자 이미지 삭제 (회원 탈퇴, 스캔/하자 삭제 시)",
    },
]

app = FastAPI(
    title="RoomLog AI API",
    version="0.1.0",
    openapi_tags=tags_metadata,
)


@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    if request.url.path in ("/docs", "/openapi.json", "/redoc"):
        return await call_next(request)
    if request.headers.get("X-Api-Key") != settings.api_key:
        return JSONResponse(
            status_code=401,
            content={"success": False, "code": 401, "message": "Invalid API key", "data": None},
        )
    return await call_next(request)


app.include_router(reconstruction.router)
app.include_router(defect_detection.router)
app.include_router(defect_comparison.router)
app.include_router(deletion.router)
