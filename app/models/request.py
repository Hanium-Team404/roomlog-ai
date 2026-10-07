from typing import Optional

from pydantic import BaseModel, Field, model_validator

from app.models.response import DefectItem


class ReconstructionRequest(BaseModel):
    scan_id: int = Field(description="백엔드 스캔 ID")
    scan_url: str = Field(description="스캔 원시 데이터 ZIP URL")
    callback_url: str = Field(description="처리 완료 후 결과를 전달할 콜백 URL")


class DefectDetectionRequest(BaseModel):
    analysis_id: int = Field(description="백엔드 분석 ID (콜백 시 사용)")
    scan_id: int = Field(description="백엔드 스캔 ID")
    scan_url: str = Field(description="스캔 원시 데이터 ZIP URL")
    callback_url: str = Field(description="처리 완료 후 결과를 전달할 콜백 URL")


class DefectDeletionRequest(BaseModel):
    image_urls: list[str] = Field(description="삭제할 하자 이미지 URL 목록 (D01/D02 콜백의 image_url 값 그대로)")


class DefectComparisonRequest(BaseModel):
    analysis_id: int = Field(description="백엔드 분석 ID (콜백 시 사용)")
    in_scan_id: int = Field(description="입주 시 스캔 ID")
    in_scan_url: Optional[str] = Field(default=None, description="입주 시 스캔 원시 데이터 ZIP URL")
    in_defects_json: Optional[list[DefectItem]] = Field(default=None, description="입주 시 기존 하자 탐지 결과 (있으면 ZIP 처리 스킵)")
    out_scan_id: int = Field(description="퇴거 시 스캔 ID")
    out_scan_url: Optional[str] = Field(default=None, description="퇴거 시 스캔 원시 데이터 ZIP URL")
    out_defects_json: Optional[list[DefectItem]] = Field(default=None, description="퇴거 시 기존 하자 탐지 결과 (있으면 ZIP 처리 스킵)")
    callback_url: str = Field(description="처리 완료 후 결과를 전달할 콜백 URL")

    @model_validator(mode="after")
    def check_scan_sources(self):
        if self.in_scan_url is None and self.in_defects_json is None:
            raise ValueError("in_scan_url 또는 in_defects_json 중 하나는 필수입니다")
        if self.out_scan_url is None and self.out_defects_json is None:
            raise ValueError("out_scan_url 또는 out_defects_json 중 하나는 필수입니다")
        return self
