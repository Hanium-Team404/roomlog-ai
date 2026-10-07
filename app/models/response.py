from enum import Enum
from typing import Generic, Optional, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class DefectType(str, Enum):
    SCRATCH = "SCRATCH"
    CRACK = "CRACK"
    PEELING = "PEELING"
    STAIN = "STAIN"
    BREAKAGE = "BREAKAGE"


class Severity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class APIResponse(BaseModel, Generic[T]):
    success: bool = Field(default=True, description="요청 성공 여부")
    code: int = Field(default=200, description="HTTP 상태 코드")
    message: str = Field(default="요청 성공", description="응답 메시지")
    data: T | None = Field(default=None, description="응답 데이터")


class ErrorDetail(BaseModel):
    code: str = Field(description="에러 코드 (예: AI_REC_001)")


class ErrorResponse(BaseModel):
    success: bool = Field(default=False)
    code: int = Field(description="HTTP 상태 코드")
    message: str = Field(description="에러 메시지")
    error: ErrorDetail
    data: None = None


class Point3D(BaseModel):
    x: float = Field(description="X 좌표 (m)")
    y: float = Field(description="Y 좌표 (m)")
    z: float = Field(description="Z 좌표 (m)")


class DefectItem(BaseModel):
    type: DefectType = Field(description="하자 유형")
    severity: Severity = Field(description="심각도")
    location: str = Field(description='위치 (예: "벽 하단")')
    area: float = Field(description="하자 면적 (㎠, 소수점 둘째 자리)")
    description: str = Field(description="하자 상세 설명")
    region_3d: list[Point3D] = Field(description="3D 공간 상 하자 영역 꼭짓점 좌표 (SAM 3 segmentation polygon → depth + camera matrix + odometry 역투영)")
    image_url: Optional[str] = Field(default=None, description="하자 영역 크롭 이미지 URL")


class DefectDetectionData(BaseModel):
    defects: list[DefectItem] = Field(description="탐지된 하자 목록")


class ComparisonDefectItem(BaseModel):
    type: DefectType = Field(description="하자 유형")
    severity: Severity = Field(description="심각도")
    location: str = Field(description='위치 (예: "벽 하단")')
    area: float = Field(description="하자 면적 (㎠, 소수점 둘째 자리)")
    description: str = Field(description="하자 상세 설명")
    region_3d: list[Point3D] = Field(description="3D 공간 상 하자 영역 꼭짓점 좌표 (SAM 3 segmentation polygon → depth + camera matrix + odometry 역투영)")
    image_url: Optional[str] = Field(default=None, description="하자 영역 크롭 이미지 URL")


class DefectComparisonData(BaseModel):
    defects: list[ComparisonDefectItem] = Field(description="신규 하자 목록 (퇴거 - 입주)")


class DeletionData(BaseModel):
    deleted: int = Field(description="삭제한 S3 객체 수")
