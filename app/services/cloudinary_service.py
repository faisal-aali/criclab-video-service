"""Cloudinary upload for processed videos and PDF reports.

If Cloudinary is not configured (no creds in .env) every call returns None and
the app falls back to serving artifacts from local disk.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from app.config import get_settings

try:
    import cloudinary
    import cloudinary.uploader
    _CLOUDINARY_IMPORTED = True
except Exception:  # pragma: no cover
    cloudinary = None
    _CLOUDINARY_IMPORTED = False

_configured = False


def _ensure_configured() -> bool:
    global _configured
    if not _CLOUDINARY_IMPORTED:
        return False
    settings = get_settings()
    if not settings.cloudinary_configured:
        return False
    if _configured:
        return True
    cloud_name = settings.cloudinary_cloud_name
    api_key = settings.cloudinary_api_key
    api_secret = settings.cloudinary_api_secret

    # Fill any missing explicit field by parsing CLOUDINARY_URL
    # (cloudinary://<api_key>:<api_secret>@<cloud_name>).
    if settings.cloudinary_url and not (cloud_name and api_key and api_secret):
        parsed = urlparse(settings.cloudinary_url)
        api_key = api_key or parsed.username
        api_secret = api_secret or parsed.password
        cloud_name = cloud_name or parsed.hostname

    if not (cloud_name and api_key and api_secret):
        return False

    cloudinary.config(
        cloud_name=cloud_name,
        api_key=api_key,
        api_secret=api_secret,
        secure=True,
    )
    _configured = True
    return True


def is_configured() -> bool:
    return _ensure_configured()


def signed_video_upload_params(*, folder: str = "criclab/incoming") -> dict[str, Any] | None:
    """Browser uploads the clip to Cloudinary so Vercel never sees the bytes.

    Vercel Functions reject bodies over ~4.5 MB (`FUNCTION_PAYLOAD_TOO_LARGE`).
    A cricket clip is almost always larger, so the file must go around the API.
    """
    if not _ensure_configured():
        return None
    import time

    from cloudinary.utils import api_sign_request

    timestamp = int(time.time())
    params = {"timestamp": timestamp, "folder": folder}
    signature = api_sign_request(params, cloudinary.config().api_secret)
    return {
        "cloud_name": cloudinary.config().cloud_name,
        "api_key": cloudinary.config().api_key,
        "timestamp": timestamp,
        "signature": signature,
        "folder": folder,
        "resource_type": "video",
    }


def is_cloudinary_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host == "res.cloudinary.com" or host.endswith(".cloudinary.com")


async def download_to_path(
    url: str,
    dest: Path,
    *,
    max_bytes: int = 180_000_000,
    on_progress: Any | None = None,
) -> None:
    """Pull a remote video onto local disk (Vercel `/tmp`) for the pipeline."""
    import httpx

    dest.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    timeout = httpx.Timeout(180.0, connect=20.0)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length") or 0) or None
            with dest.open("wb") as out:
                async for chunk in response.aiter_bytes(1024 * 256):
                    written += len(chunk)
                    if written > max_bytes:
                        dest.unlink(missing_ok=True)
                        raise ValueError("Video is too large to analyse")
                    out.write(chunk)
                    if on_progress:
                        maybe = on_progress(written, total)
                        if inspect.isawaitable(maybe):
                            await maybe
    if written < 1024:
        dest.unlink(missing_ok=True)
        raise ValueError("Downloaded video was empty")
    if on_progress:
        maybe = on_progress(written, total or written)
        if inspect.isawaitable(maybe):
            await maybe


def upload_video(path: Path, public_id: str, folder: str = "criclab/videos") -> dict[str, Any] | None:
    if not _ensure_configured():
        return None
    res = cloudinary.uploader.upload_large(
        str(path),
        resource_type="video",
        public_id=public_id,
        folder=folder,
        overwrite=True,
        # Deliver a browser-friendly H.264 MP4 regardless of the source codec.
        eager=[{"format": "mp4", "video_codec": "h264", "quality": "auto"}],
        eager_async=False,
    )
    return _video_result(res)


def upload_pdf(path: Path, public_id: str, folder: str = "criclab/reports") -> dict[str, Any] | None:
    if not _ensure_configured():
        return None
    res = cloudinary.uploader.upload(
        str(path),
        resource_type="raw",
        public_id=f"{public_id}.pdf",
        folder=folder,
        overwrite=True,
    )
    return {"secure_url": res.get("secure_url"), "public_id": res.get("public_id"), "bytes": res.get("bytes")}


def _video_result(res: dict[str, Any]) -> dict[str, Any]:
    secure = res.get("secure_url")
    playback = secure
    eager = res.get("eager") or []
    if eager and eager[0].get("secure_url"):
        playback = eager[0]["secure_url"]
    return {
        "secure_url": secure,
        "playback_url": playback,
        "public_id": res.get("public_id"),
        "duration": res.get("duration"),
        "bytes": res.get("bytes"),
        "width": res.get("width"),
        "height": res.get("height"),
    }
