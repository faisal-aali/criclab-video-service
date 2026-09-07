"""Local disk cleanup for finished video-worker jobs.

S3 is the durable store. After a job is terminal (completed, failed, or
cancelled) we delete the worker's local copies of the source video, compressed
playback, overlay, PDF, stills, charts, pitch maps, per-delivery clips and temp
frames to avoid disk overflow.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from app.config import get_settings
from app.services import s3_service

log = logging.getLogger("criclab.cleanup")


def _safe_delete(path: Path) -> None:
    """Delete a file or directory, never raising on missing paths."""
    try:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.is_file():
            path.unlink(missing_ok=True)
        log.debug("cleanup deleted %s", path)
    except FileNotFoundError:
        pass
    except Exception:
        log.exception("cleanup failed for %s", path)


def _delete_glob(root: Path, pattern: str) -> None:
    """Delete all files/dirs matching ``root / pattern`` if the root exists."""
    try:
        if not root.is_dir():
            return
        for p in root.glob(pattern):
            _safe_delete(p)
    except Exception:
        log.exception("cleanup glob failed %s/%s", root, pattern)


def cleanup_action_files(
    job_id: str,
    video_id: str | None,
    source_key: str | None,
    overlay_key: str | None,
    pdf_key: str | None,
    status: str,
) -> None:
    """Delete local Action files after the job is terminal."""
    if not s3_service.s3_configured():
        return

    settings = get_settings()

    # Source video + any sibling files (e.g. the compressed playback file).
    if video_id and source_key:
        _delete_glob(settings.storage_path / "videos", f"{video_id}*")

    # Temp frames generated during processing.
    _safe_delete(settings.storage_path / "frames" / job_id)

    # Generated artifacts (overlay, PDF, stills, charts).
    art_dir = settings.storage_path / "artifacts" / job_id
    if status in ("failed", "cancelled"):
        _safe_delete(art_dir)
    elif status == "completed" and overlay_key and pdf_key:
        _safe_delete(art_dir)
    else:
        log.warning(
            "cleanup skip action artifacts job_id=%s status=%s overlay=%s pdf=%s",
            job_id,
            status,
            overlay_key,
            pdf_key,
        )


def cleanup_balltrack_files(
    job_id: str,
    session_id: str | None,
    source_key: str | None,
    overlay_key: str | None,
    pitch_map_key: str | None,
    status: str,
    clip_keys: list[str | None] | None = None,
) -> None:
    """Delete local Ball-flight files after the job is terminal."""
    if not s3_service.s3_configured():
        return

    settings = get_settings()

    # Source video + any sibling compressed playback file.
    if session_id and source_key:
        _delete_glob(
            settings.storage_path / "balltrack" / "videos",
            f"{session_id}*",
        )

    # Temp frames generated during processing.
    _safe_delete(settings.storage_path / "frames" / job_id)

    # Generated artifacts (overlay, pitch map, per-delivery clips).
    art_dir = settings.storage_path / "balltrack" / job_id
    if status in ("failed", "cancelled"):
        _safe_delete(art_dir)
    elif status == "completed":
        clips_ok = all(clip_keys or [])
        if overlay_key and pitch_map_key and clips_ok:
            _safe_delete(art_dir)
        else:
            log.warning(
                "cleanup skip balltrack artifacts job_id=%s overlay=%s map=%s clips_ok=%s",
                job_id,
                overlay_key,
                pitch_map_key,
                clips_ok,
            )
