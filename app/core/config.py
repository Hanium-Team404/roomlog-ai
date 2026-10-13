from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str
    fal_api_key: str
    s3_bucket_name: str
    s3_region: str = "ap-northeast-2"
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    api_key: str = "roomlog-ai-secret-key"
    redis_url: str = "redis://localhost:6379"

    # R01 3D 재구성 (usdz)
    recon_frame_stride: int = 9          # 몇 프레임마다 1장을 텍스처 소스로 쓸지. 3과 9의 미관측 면 차이는 0.5%p (측정)
    recon_target_faces: int = 150_000    # 텍스처링용 메시 단순화 목표 삼각형 수. 30만과 육안 차이 없음, usdz 92→72MB (측정)
    recon_refine_poses: bool = True      # loop closure pose 보정. D01 하자 좌표도 같은 보정을 쓴다
    recon_num_threads: Optional[int] = None  # texrecon 스레드 수 (None이면 전부)
    texrecon_bin: str = "texrecon"       # mvs-texturing 바이너리 경로 (Dockerfile에서 /usr/local/bin에 설치)

    model_config = {"env_file": ".env"}


settings = Settings()
