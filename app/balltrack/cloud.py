"""S3 uploads for ball-track artifacts. Pose overlay path is unchanged."""

from __future__ import annotations

from pathlib import Path

from app.services import s3_service


def upload_video(path: Path, key: str) -> str | None:
    try:
        return s3_service.encode_and_upload_video(path, key)
    except Exception:
        return None


def upload_image(path: Path, key: str, content_type: str = "image/png") -> str | None:
    try:
        return s3_service.upload_file(path, key, content_type)
    except Exception:
        return None
