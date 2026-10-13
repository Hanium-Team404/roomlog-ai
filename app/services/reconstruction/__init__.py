"""AI-R01 3D 재구성: raw 스캔 -> 텍스처 usdz.

단계 (3D_reconstruction_todo/improve_usdz 의 01~05 기본 경로를 그대로 옮김):
  preprocess  동기화, depth 필터, loop closure pose 보정, 텍스처용 키프레임 선택
  tsdf        LiDAR depth TSDF 융합, cleanup, 텍스처용 단순화
  texture     texrecon(mvs-texturing) 입력 구성, 좌표계 검증, 텍스처링
  usdz        textured.obj -> usdz (미관측 면은 TSDF 정점 색으로 굽기)
"""

from app.services.reconstruction.pipeline import ReconstructionResult, run_reconstruction

__all__ = ["ReconstructionResult", "run_reconstruction"]
