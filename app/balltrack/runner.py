from __future__ import annotations

import asyncio
import logging
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from app.balltrack import cloud, repo
from app.balltrack.calibrate import homography_from_boxes
from app.balltrack.detect import collect_candidates
from app.balltrack.metrics import analyze_delivery
from app.balltrack.render import write_clip, write_overlay, write_pitch_map
from app.balltrack.split import split_deliveries
from app.balltrack.track import build_tracks
from app.balltrack.validate import reject_reason
from app.config import get_settings
from app.pipeline.cancel import JobCancelled, raise_if_cancelled
from app.pipeline.cv_vision import validate_ball_path_on_video
from app.pipeline.job_progress import BALLTRACK_BANDS, JobReporter, clamp_counts

log = logging.getLogger("criclab.balltrack")


async def run_balltrack_job(*, job_id: str, session_id: str, video_path: Path, calibration: dict[str, Any]) -> None:
    settings = get_settings()
    progress = JobReporter(job_id, bands=BALLTRACK_BANDS, update=repo.update_job)
    try:
        await raise_if_cancelled(job_id, get_job=repo.get_job)
        await progress.aset("calibrate", 0, "Measuring the pitch", force=True)
        art = settings.storage_path / "balltrack" / job_id
        art.mkdir(parents=True, exist_ok=True)
        log.debug("calibrate job_id=%s", job_id)

        await raise_if_cancelled(job_id, get_job=repo.get_job)
        await progress.aset("detect", 0, "Finding the ball", force=True)

        def on_detect(cur: int, tot: int) -> None:
            cur, tot = clamp_counts(cur, tot)
            progress.emit(
                "detect",
                cur / tot,
                f"Finding the ball — frame {cur} of {tot}",
                detail={"current": cur, "total": tot, "unit": "frames"},
            )

        frames, meta = await asyncio.to_thread(collect_candidates, video_path, on_progress=on_detect)
        fps = float(meta["fps"] or 30.0)
        w, h = int(meta["width"]), int(meta["height"])
        log.debug(
            "detect done job_id=%s frames=%s fps=%s %sx%s",
            job_id,
            len(frames) if frames is not None else 0,
            fps,
            w,
            h,
        )
        if w < 16 or h < 16:
            raise ValueError("Could not read video dimensions")

        cal = homography_from_boxes(
            calibration.get("bowler") or {},
            calibration.get("batter") or {},
            w,
            h,
            pitch_length_m=float(calibration.get("pitch_length_m") or 20.12),
        )
        H = np.array(cal["H"], dtype=np.float64)
        log.debug(
            "homography job_id=%s pitch_m=%s width_m=%s",
            job_id,
            cal.get("pitch_length_m"),
            cal.get("pitch_width_m"),
        )

        await raise_if_cancelled(job_id, get_job=repo.get_job)
        await progress.aset("track", 0, "Following each delivery", force=True)
        tracks = await asyncio.to_thread(build_tracks, frames, w, h, fps)
        deliveries_pts = split_deliveries(tracks, fps)
        log.debug("track/split job_id=%s n_paths=%s", job_id, len(deliveries_pts))
        if not deliveries_pts:
            raise ValueError(
                "No cricket ball detected. Film a real delivery down the pitch — empty or walking clips will not produce a speed."
            )

        n_paths = len(deliveries_pts)
        await raise_if_cancelled(job_id, get_job=repo.get_job)
        await progress.aset(
            "metrics",
            0,
            f"Measuring {n_paths} candidate path{'s' if n_paths != 1 else ''}",
            force=True,
        )
        analyzed: list[dict[str, Any]] = []
        delivery_ids: list[str] = []
        for i_path, pts in enumerate(deliveries_pts):
            await raise_if_cancelled(job_id, get_job=repo.get_job)
            await progress.aset(
                "metrics",
                (i_path + 1) / max(n_paths, 1),
                f"Measuring path {i_path + 1} of {n_paths}",
                detail={"current": i_path + 1, "total": n_paths, "unit": "paths"},
            )
            metrics = analyze_delivery(
                pts,
                H,
                fps,
                cal["pitch_length_m"],
                cal["pitch_width_m"],
            )
            why = reject_reason(metrics, fps, h, cal["pitch_length_m"], cal["pitch_width_m"])
            if why:
                log.debug("path reject job_id=%s i=%s reason=%s", job_id, i_path, why)
                continue
            flow_ok, _flow_why, _flow = validate_ball_path_on_video(video_path, pts, w, h)
            if not flow_ok:
                log.debug("path flow-reject job_id=%s i=%s reason=%s", job_id, i_path, _flow_why)
                continue
            i = len(analyzed)
            did = repo.new_id("btd")
            clip_path = art / f"ball_{i + 1}.mp4"
            write_clip(
                video_path,
                clip_path,
                metrics["start_frame"],
                metrics["end_frame"],
                fps,
                delivery=metrics,
                H_inv=np.array(cal["H_inv"], dtype=np.float64),
                pitch_length_m=cal["pitch_length_m"],
                stump_width_m=float(cal.get("stump_width_m") or 0.2286),
                bowler_box=calibration.get("bowler"),
                batter_box=calibration.get("batter"),
            )
            clip_key = f"overlays/{job_id}_ball_{i + 1}.mp4"
            uploaded_clip = cloud.upload_video(clip_path, clip_key)
            clip_url = f"/balltrack/media/{job_id}/ball_{i + 1}.mp4"
            bounce = metrics.get("bounce") or {}
            doc = {
                "_id": did,
                "session_id": session_id,
                "job_id": job_id,
                "index": i + 1,
                "created_at": repo.utcnow(),
                "metrics": {
                    "speed_kmh": metrics["speed_kmh"],
                    "line_m": metrics["line_m"],
                    "length_m": metrics["length_m"],
                },
                "bounce": {
                    "length_m": bounce.get("length_m"),
                    "width_m": bounce.get("width_m"),
                    "frame": bounce.get("frame"),
                },
                "n_points": metrics["n_points"],
                "start_frame": metrics["start_frame"],
                "end_frame": metrics["end_frame"],
                "artifacts": {
                    "clip_url": clip_url,
                    "clip_path": str(clip_path),
                    "clip_key": uploaded_clip,
                },
            }
            await raise_if_cancelled(job_id, get_job=repo.get_job)
            await repo.insert_delivery(doc)
            delivery_ids.append(did)
            analyzed.append({**metrics, "index": i + 1, "bounce": bounce})

        if not analyzed:
            raise ValueError(
                "No cricket ball detected. Nothing in this clip looked like a delivery (speed, bounce, and path toward the batter must all check out)."
            )
        log.debug("metrics job_id=%s deliveries=%s", job_id, len(analyzed))

        await raise_if_cancelled(job_id, get_job=repo.get_job)
        await progress.aset("render", 0, "Drawing the path onto your clip", force=True)
        overlay_path = art / "overlay.mp4"
        H_inv = np.array(cal["H_inv"], dtype=np.float64)
        write_overlay(
            video_path,
            overlay_path,
            analyzed,
            fps,
            H_inv=H_inv,
            pitch_length_m=cal["pitch_length_m"],
            pitch_width_m=cal["pitch_width_m"],
            stump_width_m=float(cal.get("stump_width_m") or 0.2286),
            bowler_box=calibration.get("bowler"),
            batter_box=calibration.get("batter"),
        )
        map_path = art / "pitch_map.png"
        write_pitch_map(map_path, analyzed, cal["pitch_length_m"], cal["pitch_width_m"])
        overlay_key = f"overlays/{job_id}_overlay.mp4"
        map_key = f"files/{job_id}_pitchmap.png"
        uploaded_overlay = cloud.upload_video(overlay_path, overlay_key)
        uploaded_map = cloud.upload_image(map_path, map_key)
        log.debug("render job_id=%s overlay_key=%s", job_id, uploaded_overlay)

        artifacts = {
            "job_id": job_id,
            "overlay_url": f"/balltrack/media/{job_id}/overlay.mp4",
            "pitch_map_url": f"/balltrack/media/{job_id}/pitch_map.png",
            "overlay_key": uploaded_overlay,
            "pitch_map_key": uploaded_map,
        }

        await raise_if_cancelled(job_id, get_job=repo.get_job)
        await progress.aset("agent", 0, "Matching drills to what we saw", force=True)
        from app.agent import ollama_agent
        from app.coaching.recommend import balltrack_tags

        first = analyzed[0]
        flight_metrics = {
            "ball_speed_kmh": first.get("speed_kmh"),
            "line_m": first.get("line_m"),
            "length_m": first.get("length_m"),
            "scale": {"calibrated": True, "method": "stump_homography", "note": "Pitch-plane from both stump sets"},
            "quality": {
                "camera_view": "behind_bowler",
                "camera_view_note": "Stump homography — pitch-plane speed, not a radar gun",
                "speed_view_ok": True,
                "calibrated": True,
            },
        }
        tags = balltrack_tags(analyzed)
        analysis = await ollama_agent.generate_report(
            metrics=flight_metrics,
            candidate_tags=tags,
            player_name="Bowler",
        )
        log.debug("drills job_id=%s", job_id)

        await raise_if_cancelled(job_id, get_job=repo.get_job)
        await repo.update_session(
            session_id,
            status="completed",
            delivery_ids=delivery_ids,
            delivery_count=len(delivery_ids),
            calibration_used=cal,
            artifacts=artifacts,
            analysis=analysis,
            meta=meta,
        )
        await repo.update_job(
            job_id,
            status="completed",
            progress=100,
            stage="done",
            message=f"Tracked {len(delivery_ids)} deliveries",
            session_id=session_id,
        )
        log.debug("ballflight complete job_id=%s deliveries=%s", job_id, len(delivery_ids))
    except JobCancelled:
        log.debug("ballflight cancelled job_id=%s", job_id)
        await repo.delete_deliveries_for_job(job_id)
        raise
    except Exception as exc:
        log.debug("ballflight fail job_id=%s err=%s", job_id, exc)
        await repo.update_job(
            job_id,
            status="failed",
            progress=100,
            stage="failed",
            message=str(exc),
            error=traceback.format_exc()[-1500:],
        )
        await repo.update_session(session_id, status="failed", error=str(exc))
        # Re-raise so the worker logs `job failed`, not `finished`, for a clip
        # that did not finish — the same contract as the Action runner. The
        # worker's own failure write is a no-op here because the status is no
        # longer in-flight.
        raise
