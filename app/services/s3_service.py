"""S3 download / upload and ffmpeg playback encode for the video worker.

This process never mints CloudFront URLs. Mongo stores object keys; the website
API signs GET URLs at read time.
"""

from __future__ import annotations

import inspect
import subprocess
from pathlib import Path
from typing import Any

from app.config import get_settings

PREFIXES = ("original/", "compressed/", "overlays/", "files/")
MAX_BYTES = 180_000_000
SCALE_FILTER = (
    "scale=1280:720:force_original_aspect_ratio=decrease,"
    "pad=1280:720:(ow-iw)/2:(oh-ih)/2"
)


def s3_endpoint_url(region: str | None) -> str | None:
    """Regional S3 API host. Global s3.amazonaws.com 301s ap-south-1 buckets."""
    r = (region or "").strip()
    if not r:
        return None
    return f"https://s3.{r}.amazonaws.com"


def _s3_client():
    import boto3
    from botocore.config import Config

    settings = get_settings()
    region = (settings.s3_region or settings.aws_region or "ap-south-1").strip()
    kwargs: dict[str, Any] = {
        "region_name": region,
        "config": Config(
            signature_version="s3v4",
            s3={"addressing_style": "virtual"},
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
    }
    endpoint = s3_endpoint_url(region)
    if endpoint:
        kwargs["endpoint_url"] = endpoint
    if settings.aws_access_key_id and settings.aws_secret_access_key:
        kwargs["aws_access_key_id"] = settings.aws_access_key_id
        kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
    return boto3.client("s3", **kwargs)


def s3_configured() -> bool:
    settings = get_settings()
    return bool(settings.s3_bucket and settings.s3_region)


def is_configured() -> bool:
    return s3_configured()


def is_our_object_key(key: str | None) -> bool:
    k = (key or "").strip().lstrip("/")
    if not k or ".." in k or "\\" in k or "\n" in k:
        return False
    return any(k.startswith(p) for p in PREFIXES)


def _ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def ffmpeg_argv(src: Path, dest: Path) -> list[str]:
    # CBR: ABR + maxrate alone undershoots on simple 720p clips (~750 kbps).
    return [
        _ffmpeg_exe(),
        "-y",
        "-i",
        str(src),
        "-an",
        "-r",
        "30",
        "-c:v",
        "libx264",
        "-b:v",
        "1000k",
        "-minrate",
        "1000k",
        "-maxrate",
        "1000k",
        "-bufsize",
        "2000k",
        "-x264-params",
        "nal-hrd=cbr",
        "-vf",
        SCALE_FILTER,
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(dest),
    ]


def transcode_playback(src: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(ffmpeg_argv(src, dest), capture_output=True, text=True)
    if result.returncode != 0 or not dest.is_file() or dest.stat().st_size < 1024:
        err = (result.stderr or result.stdout or "ffmpeg failed").strip()[-800:]
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"Could not encode playback video: {err}")
    return dest


def upload_file(
    path: Path,
    key: str,
    content_type: str,
    *,
    content_disposition: str | None = None,
) -> str | None:
    if not s3_configured() or not is_our_object_key(key) or not path.is_file():
        return None
    extra: dict[str, str] = {"ContentType": content_type}
    if content_disposition:
        extra["ContentDisposition"] = content_disposition
    _s3_client().upload_file(str(path), get_settings().s3_bucket, key, ExtraArgs=extra)
    return key


def encode_and_upload_video(src: Path, key: str) -> str | None:
    """H.264 1280×720 30 fps 1 Mbps, then PutObject. Replaces src on success."""
    encoded = src.with_name(f"{src.stem}_h264.mp4")
    transcode_playback(src, encoded)
    uploaded = upload_file(encoded, key, "video/mp4")
    try:
        encoded.replace(src)
    except OSError:
        pass
    return uploaded


async def download_object(
    key: str,
    dest: Path,
    *,
    max_bytes: int = MAX_BYTES,
    on_progress: Any | None = None,
) -> None:
    """Pull an S3 object onto this worker's disk for the pipeline."""
    k = key.strip().lstrip("/")
    if not s3_configured() or not is_our_object_key(k):
        raise ValueError("Clip has no S3 object key")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.unlink(missing_ok=True)
    client = _s3_client()
    obj = await _to_thread(client.get_object, Bucket=get_settings().s3_bucket, Key=k)
    body = obj["Body"]
    total = int(obj.get("ContentLength") or 0) or None
    written = 0
    with dest.open("wb") as out:
        while True:
            chunk = await _to_thread(body.read, 256 * 1024)
            if not chunk:
                break
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


async def _to_thread(fn, *args, **kwargs):
    import asyncio

    return await asyncio.to_thread(fn, *args, **kwargs)
