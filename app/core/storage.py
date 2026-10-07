import io
import zipfile
from pathlib import Path
from typing import NamedTuple

import boto3
import httpx


class ScanFiles(NamedTuple):
    video_path: Path
    camera_matrix_path: Path
    odometry_path: Path
    depth_dir: Path
    conf_dir: Path


def download_scan_zip(url: str, extract_dir: str) -> ScanFiles:
    base = Path(extract_dir)
    base.mkdir(parents=True, exist_ok=True)
    tmp_zip = base / "_scan.zip"
    with httpx.stream("GET", url) as r:
        r.raise_for_status()
        with open(tmp_zip, "wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)
    with zipfile.ZipFile(tmp_zip) as zf:
        zf.extractall(base)
    tmp_zip.unlink()
    return ScanFiles(
        video_path=base / "rgb.mp4",
        camera_matrix_path=base / "camera_matrix.csv",
        odometry_path=base / "odometry.csv",
        depth_dir=base / "depth",
        conf_dir=base / "confidence",
    )


_CONTENT_TYPES = {
    ".ply": "application/octet-stream",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}

_s3_client = None


def _get_s3():
    global _s3_client
    if _s3_client is None:
        from app.core.config import settings
        boto_kwargs = {}
        if settings.aws_access_key_id:
            boto_kwargs["aws_access_key_id"] = settings.aws_access_key_id
            boto_kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
        _s3_client = boto3.client(
            "s3",
            region_name=settings.s3_region,
            endpoint_url=f"https://s3.{settings.s3_region}.amazonaws.com",
            **boto_kwargs,
        )
    return _s3_client


def upload_to_s3(data: bytes, key: str) -> str:
    from app.core.config import settings

    s3 = _get_s3()
    s3.put_object(
        Bucket=settings.s3_bucket_name,
        Key=key,
        Body=io.BytesIO(data),
        ContentType=_CONTENT_TYPES.get(Path(key).suffix.lower(), "application/octet-stream"),
    )
    return f"https://{settings.s3_bucket_name}.s3.{settings.s3_region}.amazonaws.com/{key}"


def s3_url_prefix() -> str:
    from app.core.config import settings

    return f"https://{settings.s3_bucket_name}.s3.{settings.s3_region}.amazonaws.com/"


def delete_from_s3(keys: list[str]) -> int:
    """없는 키는 S3가 에러 없이 무시한다. 삭제 요청한 키 수를 반환."""
    from app.core.config import settings

    s3 = _get_s3()
    for i in range(0, len(keys), 1000):
        chunk = keys[i:i + 1000]
        s3.delete_objects(
            Bucket=settings.s3_bucket_name,
            Delete={"Objects": [{"Key": k} for k in chunk], "Quiet": True},
        )
    return len(keys)


def delete_prefix_from_s3(prefix: str) -> int:
    from app.core.config import settings

    s3 = _get_s3()
    keys: list[str] = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=settings.s3_bucket_name, Prefix=prefix):
        keys.extend(obj["Key"] for obj in page.get("Contents", []))
    return delete_from_s3(keys) if keys else 0


def upload_file_to_s3(path: Path, key: str) -> str:
    """파일을 RAM에 올리지 않고 디스크에서 직접 스트리밍 업로드."""
    from app.core.config import settings

    s3 = _get_s3()
    content_type = _CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")
    s3.upload_file(
        str(path),
        settings.s3_bucket_name,
        key,
        ExtraArgs={"ContentType": content_type},
    )
    return f"https://{settings.s3_bucket_name}.s3.{settings.s3_region}.amazonaws.com/{key}"
