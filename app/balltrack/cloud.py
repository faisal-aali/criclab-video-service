"""Cloudinary uploads for ball-track artifacts. Uses existing helpers; pose paths unchanged."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.services import cloudinary_service

FOLDER = "criclab/balltrack"


def upload_video(path: Path, public_id: str) -> dict[str, Any] | None:
    return cloudinary_service.upload_video(path, public_id, folder=FOLDER)


def upload_image(path: Path, public_id: str) -> dict[str, Any] | None:
    if not cloudinary_service.is_configured():
        return None
    import cloudinary.uploader

    res = cloudinary.uploader.upload(
        str(path),
        resource_type="image",
        public_id=public_id,
        folder=FOLDER,
        overwrite=True,
    )
    return {"secure_url": res.get("secure_url"), "public_id": res.get("public_id")}
