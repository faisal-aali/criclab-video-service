"""End-to-end bowling analysis pipeline runner.

Order (see memory-bank/systemPatterns.md):
  extract meta → pose estimation → action/release detection → scale
    → (best-effort) ball tracking → biomechanics metrics
      → slow-motion overlay video → Cloudinary upload
        → Gemma coaching narrative → Cric-Lab PDF → Cloudinary upload
          → persist MongoDB delivery.
"""

from __future__ import annotations

import asyncio
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from app.agent import ollama_agent
from app.config import get_settings
from app.db import repository as repo
from app.pdf.report import build_pdf
from app.pipeline import action as action_mod
from app.pipeline import calibrate, extract, pose as pose_mod
from app.pipeline import metrics as metrics_mod
from app.pipeline import render as render_mod
from app.pipeline import timebase, track
from app.pipeline.job_progress import JobReporter, clamp_counts
from app.pipeline.view import flight_is_trackable
from app.services import cloudinary_service


async def run_analysis_job(
    *,
    job_id: str,
    video_id: str,
    video_path: Path,
    player_name: str = "Bowler",
    meters_per_pixel: float | None = None,
    reference_height_m: float | None = None,
    player_profile: dict[str, Any] | None = None,
    user_id: str | None = None,
) -> None:
    settings = get_settings()
    progress = JobReporter(job_id)
    try:
        await progress.aset("extract", 0, "Reading video metadata", force=True)
        meta = await asyncio.to_thread(extract.extract_video_meta, video_path)
        fps = float(meta["fps"] or 30.0)

        artifact_dir = settings.storage_path / "artifacts" / job_id
        artifact_dir.mkdir(parents=True, exist_ok=True)

        n_frames = int(meta.get("frame_count") or 0)
        await progress.aset(
            "pose",
            0,
            f"Mapping the bowler — 0 of {n_frames} frames" if n_frames else "Mapping the bowler",
            force=True,
            detail={"current": 0, "total": n_frames, "unit": "frames"} if n_frames else None,
        )

        def on_pose(cur: int, tot: int) -> None:
            cur, tot = clamp_counts(cur, tot)
            progress.emit(
                "pose",
                cur / tot,
                f"Mapping the bowler — frame {cur} of {tot}",
                detail={"current": cur, "total": tot, "unit": "frames"},
            )

        pose_track = await asyncio.to_thread(
            pose_mod.extract_pose_track, video_path, on_progress=on_pose
        )
        if not pose_track.get("frames"):
            raise ValueError("No bowler pose detected — use a clearer, side-on video of the delivery.")

        await progress.aset("action", 0, "Finding the release and action phases", force=True)
        bowling_arm = (player_profile or {}).get("bowling_arm")
        action = await asyncio.to_thread(
            action_mod.analyze_action, pose_track, bowling_arm=bowling_arm
        )

        # --- Scale from upright (tall) body frames — never crouch-biased median ---
        body_heights = [h for h in (pose_mod.body_pixel_height(f) for f in pose_track["frames"]) if h]
        body_px = calibrate.upright_body_px_height(body_heights)
        scale = calibrate.resolve_scale(meters_per_pixel, reference_height_m, body_px)

        frame_w = int(meta.get("width") or pose_track.get("width") or 1280)
        frame_h = int(meta.get("height") or pose_track.get("height") or 720)

        await progress.aset("ball", 0, "Following the ball after release", force=True)

        def on_ball(cur: int, tot: int) -> None:
            cur, tot = clamp_counts(cur, tot)
            fitting = cur >= tot
            progress.emit(
                "ball",
                1.0 if fitting else 0.90 * cur / tot,
                "Locking onto the flight path"
                if fitting
                else f"Following the ball — frame {cur} of {tot}",
                detail={"current": cur, "total": tot, "unit": "frames"},
            )

        ball_track = await asyncio.to_thread(
            _track_ball_seeded, video_path, meta, pose_track, action, scale, on_ball
        )
        await progress.aset("ball", 1.0, "Ball path locked", force=True)

        # --- Timebase: is the clip slow motion? ---
        # Every phase window ("the front foot plants 60-600 ms before release")
        # is cut in frames from fps, so a wrong fps mis-detects the events
        # themselves. The ball's own fall says what the capture rate really was.
        tb = timebase.measure_capture_fps(ball_track, scale.get("meters_per_pixel"), fps)

        # The ball path is indexed by frame, so it does not change with the
        # timebase and is never re-tracked here. What it does give us is the frame
        # the ball left the hand — a measurement, where release detection from
        # wrist speed alone is a heuristic. Re-run the action pass keyed to that
        # frame (and to the corrected rate) so FFC, BFC and MER are searched
        # relative to a real release rather than an estimated one.
        leave = action_mod.ball_leave_frame(
            pose_track, action.get("throwing_side"), ball_track, action.get("release_frame")
        )
        if tb.get("slow_motion"):
            fps = float(tb["fps"])
            pose_track["fps"] = fps
            # Per-point km/h was baked in at the container rate while tracking.
            # Leaving it would make the two ball-speed estimators disagree by
            # exactly the slow-motion factor, and the disagreement guard would
            # then refuse a speed we can measure perfectly well.
            ball_track = track.annotate_frame_motion(
                ball_track, fps, scale.get("meters_per_pixel")
            )
        if tb.get("slow_motion") or leave is not None:
            action = action_mod.analyze_action(
                pose_track, bowling_arm=bowling_arm, release_override=leave
            )

        # A track is kept if it is a real flight — an object that left the hand and
        # kept moving. Whether its direction also supports a km/h is a *separate*
        # question, answered by flight_geometry_ok inside metrics, which refuses
        # the speed with its own reason. Deleting the path here because the speed
        # is unmeasurable would blank the ball in the overlay, unpin release from
        # the one frame we actually measured, and starve the gravity timebase —
        # on every clip filmed from behind or down the pitch.
        real_ok, _real_reason = flight_is_trackable(
            ball_track, frame_w, frame_h, fps, scale.get("meters_per_pixel")
        )
        if ball_track and not real_ok:
            ball_track = []
            if leave is not None:  # release was pinned to a flight we just rejected
                action = action_mod.analyze_action(pose_track, bowling_arm=bowling_arm)
        elif ball_track:
            action_mod.snap_release_to_ball_leave(pose_track, action, ball_track)

        # --- Metrics ---
        await progress.aset("metrics", 0, "Measuring the delivery", force=True)
        metrics = await asyncio.to_thread(
            metrics_mod.compute_metrics,
            fps=fps,
            pose_track=pose_track,
            action=action,
            scale=scale,
            ball_track=ball_track,
            player_profile=player_profile,
            timebase_info=tb,
        )

        # --- Slow-motion overlay video ---
        await progress.aset("render", 0, "Marking up the slow-motion clip", force=True)
        overlay_video_path = artifact_dir / "overlay.mp4"
        release_still_path = artifact_dir / "release.jpg"
        stills_dir = artifact_dir / "stills"

        def on_render(cur: int, tot: int) -> None:
            cur, tot = clamp_counts(cur, tot)
            progress.emit(
                "render",
                cur / tot,
                f"Marking up the clip — frame {cur} of {tot}",
                detail={"current": cur, "total": tot, "unit": "frames"},
            )

        render_info = await asyncio.to_thread(
            render_mod.render_overlay_video,
            video_path=video_path,
            out_path=overlay_video_path,
            pose_track=pose_track,
            action=action,
            metrics=metrics,
            ball_track=ball_track,
            player_name=player_name,
            release_still_path=release_still_path,
            stills_dir=stills_dir,
            capture_fps=fps,
            on_progress=on_render,
        )

        # --- Upload processed video to Cloudinary ---
        await progress.aset("upload", 0, "Saving your processed clip", force=True)
        cloud: dict[str, Any] = {"configured": cloudinary_service.is_configured()}
        try:
            vres = await asyncio.to_thread(
                cloudinary_service.upload_video, overlay_video_path, f"{job_id}_overlay"
            )
            if vres:
                cloud["video"] = vres
        except Exception as e:  # never fail the whole job on upload error
            cloud["video_error"] = str(e)

        # --- Agent narrative ---
        await progress.aset("agent", 0, "Writing your coaching notes", status="analyzing", force=True)
        previous = await repo.list_deliveries(limit=8, player_name=player_name)
        prev_metrics = [d.get("metrics") for d in previous if d.get("metrics")]
        comparison = ollama_agent.compare_deliveries(metrics, prev_metrics[:5])
        analysis = await ollama_agent.generate_report(
            metrics=metrics,
            comparison=comparison,
            player_name=player_name,
            player_profile=player_profile,
        )

        # --- PDF ---
        await progress.aset("pdf", 0, "Building your report", status="analyzing", force=True)
        pdf_path = artifact_dir / "bowling_report.pdf"
        created = repo.utcnow()
        await asyncio.to_thread(
            build_pdf,
            out_path=pdf_path,
            player_name=player_name,
            delivery_id=job_id,
            metrics=metrics,
            analysis=analysis,
            release_still=release_still_path if release_still_path.exists() else None,
            date_str=created.strftime("%b-%d-%Y"),
            chart_dir=artifact_dir,
            stills_dir=stills_dir if stills_dir.exists() else None,
        )
        try:
            pres = cloudinary_service.upload_pdf(pdf_path, public_id=f"{job_id}_report")
            if pres:
                cloud["pdf"] = pres
        except Exception as e:
            cloud["pdf_error"] = str(e)

        # --- Persist ---
        delivery_id = repo.new_id("del")
        delivery = {
            "_id": delivery_id,
            "job_id": job_id,
            "video_id": video_id,
            "user_id": user_id,
            "player_name": player_name,
            "player_profile": player_profile,
            "created_at": created,
            "meta": meta,
            "action": _strip_series(action),
            "timebase": tb,
            "release": {"frame": action.get("release_frame"), **(metrics.get("release_point") or {})},
            "metrics": _strip_metric_series(metrics),
            "analysis": analysis,
            "render": render_info,
            "cloudinary": cloud,
            "artifacts": {
                "release_still": str(release_still_path),
                "overlay_video": str(overlay_video_path),
                "stills_dir": str(stills_dir),
                "pdf": str(pdf_path),
                "cloudinary_video_url": (cloud.get("video") or {}).get("playback_url"),
                "cloudinary_pdf_url": (cloud.get("pdf") or {}).get("secure_url"),
            },
        }
        await repo.insert_delivery(delivery)

        await repo.update_job(
            job_id, status="completed", progress=100, stage="done", message="Analysis complete",
            stage_detail=None,
            delivery_id=delivery_id,
            result={
                "delivery_id": delivery_id,
                "overlay_video_url": f"/artifacts/{job_id}/overlay.mp4",
                "pdf_url": f"/artifacts/{job_id}/bowling_report.pdf",
                "release_still_url": f"/artifacts/{job_id}/release.jpg",
                "cloudinary_video_url": (cloud.get("video") or {}).get("playback_url"),
                "cloudinary_pdf_url": (cloud.get("pdf") or {}).get("secure_url"),
            },
        )
    except Exception as e:
        await repo.update_job(
            job_id, status="failed",
            message=str(e), error=traceback.format_exc(),
        )


def _strip_series(action: dict[str, Any]) -> dict[str, Any]:
    """Keep the action summary small in Mongo (drop the per-frame speed series)."""
    a = dict(action)
    a.pop("wrist_speed_series", None)
    return a


def _strip_metric_series(metrics: dict[str, Any]) -> dict[str, Any]:
    """rotation_series only feeds the PDF sequencing chart, which is already
    built by now — hundreds of rows per 200 fps clip don't belong in every
    history-listing response."""
    m = dict(metrics)
    m.pop("rotation_series", None)
    return m


def _throw_peak_frame(action: dict[str, Any], fps: float) -> int | None:
    """Wrist-speed peak in the delivery, not a run-up identity flicker."""
    series = action.get("wrist_speed_series") or []
    phases = action.get("phases") or {}
    stored = action.get("peak_wrist_frame")
    ffc = phases.get("front_foot_contact")
    mer = phases.get("max_external_rotation")
    rel = action.get("release_frame")
    lo = None
    if ffc is not None:
        lo = int(ffc) - 4
    if mer is not None:
        mer_lo = int(mer) - 4
        lo = mer_lo if lo is None else max(int(lo), mer_lo)
    if lo is None and rel is not None:
        lo = int(rel) - max(20, int(round(float(fps) * 0.25)))
    if series:
        cands = [p for p in series if lo is None or int(p.get("frame") or 0) >= int(lo)]
        if not cands:
            cands = list(series)
        best = max(cands, key=lambda p: float(p.get("speed_px") or 0))
        return int(best["frame"])
    if stored is not None:
        return int(stored)
    return None


def _body_segments(
    pose_track: dict[str, Any],
    side: str | None,
) -> tuple[dict[int, list[tuple[float, float, float, float]]], float]:
    """The bowler's limbs and torso per frame, for excluding them as ball candidates.

    Pose runs before ball tracking so this map exists by the time it is needed.
    The bowler is the largest source of false ball candidates on any clip — a
    forearm, shoulder or thigh is a fast-moving, ball-sized, ball-coloured blob,
    and one of them capturing the chain is how a track ends up measuring the arm
    instead of the delivery.

    The bowling forearm (elbow to wrist) is deliberately left out: at release the
    real ball is right beside it, and excluding it would delete the very frames
    release detection and release speed depend on.

    Returns the map and a margin scaled to the bowler's pixel size, so it means
    the same thing at 720p and 4K.
    """
    frames = pose_track.get("frames") or []
    other = "left" if side == "right" else "right"
    pairs = [
        ("left_shoulder", "right_shoulder"),
        ("left_hip", "right_hip"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        (f"{side}_shoulder", f"{side}_elbow") if side else ("left_shoulder", "left_elbow"),
        (f"{other}_shoulder", f"{other}_elbow"),
        (f"{other}_elbow", f"{other}_wrist"),
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
    ]
    segs: dict[int, list[tuple[float, float, float, float]]] = {}
    heights: list[float] = []
    for f in frames:
        out: list[tuple[float, float, float, float]] = []
        for a, b in pairs:
            pa = pose_mod.point(f, a)
            pb = pose_mod.point(f, b)
            if pa is None or pb is None:
                continue
            out.append((float(pa[0]), float(pa[1]), float(pb[0]), float(pb[1])))
        if out:
            segs[int(f["frame"])] = out
        h = pose_mod.body_pixel_height(f)
        if h:
            heights.append(float(h))
    body_px = float(np.median(heights)) if heights else 0.0
    return segs, max(6.0, 0.035 * body_px)


def _track_ball_seeded(
    video_path: Path,
    meta: dict[str, Any],
    pose_track: dict[str, Any],
    action: dict[str, Any],
    scale: dict[str, Any],
    on_progress: Any | None = None,
) -> list[dict[str, Any]]:
    """Track the ball leaving the bowling wrist at the detected release frame."""
    try:
        release = action.get("release_frame")
        side = action.get("throwing_side")
        if release is None or not side:
            return []
        fps = float(meta.get("fps") or pose_track.get("fps") or 30.0)
        frame_w = int(meta.get("width") or pose_track.get("width") or 1280)
        frame_h = int(meta.get("height") or pose_track.get("height") or 720)
        frames = pose_track.get("frames") or []
        peak_fr = _throw_peak_frame(action, fps)
        track_rel = int(release)
        max_gap = max(24, min(48, int(round(fps * 0.35))))
        if peak_fr is not None and abs(int(peak_fr) - int(release)) <= max_gap:
            # Refined REL often sits a few frames into follow-through, after the
            # ball has left and (on 4K) after the wrist landmark has jumped to a
            # wall sticker. The throw-peak is where the hand still holds it.
            track_rel = int(peak_fr)

        # MediaPipe often parks the bowling wrist on the chest at leave-hand.
        # Walk back to the last frame the landmark is still on the arm.
        track_rel, wr = action_mod.last_on_arm_wrist(frames, side, track_rel, frame_h)
        if wr is None and peak_fr is not None:
            track_rel, wr = action_mod.last_on_arm_wrist(frames, side, int(peak_fr), frame_h)
        if wr is None:
            track_rel, wr = action_mod.last_on_arm_wrist(frames, side, int(release), frame_h)
        if wr is None:
            return []

        rel = action_mod.frame_by_index(frames, track_rel)
        throw_dir = None
        el = pose_mod.point(rel, f"{side}_elbow") if rel is not None else None
        elbow_dir = None
        if el is not None:
            elbow_dir = (float(wr[0] - el[0]), float(wr[1] - el[1]))
            throw_dir = elbow_dir
        lead = "left" if side == "right" else "right"
        # Ankles at FFC show pitch direction more cleanly than at a drifted REL.
        ffc = (action.get("phases") or {}).get("front_foot_contact")
        plant_fr = int(ffc) if ffc is not None else int(track_rel)
        plant = action_mod.frame_by_index(frames, plant_fr)
        la = pose_mod.point(plant, f"{lead}_ankle") if plant is not None else None
        ta = pose_mod.point(plant, f"{side}_ankle") if plant is not None else None
        if la is not None and ta is not None:
            pitch = (float(la[0] - ta[0]), float(la[1] - ta[1]))
            if float(np.hypot(pitch[0], pitch[1])) > 12:
                throw_dir = (pitch[0], min(0.0, pitch[1]) * 0.35)
        prev = action_mod.frame_by_index(frames, int(track_rel) - 4)
        if prev is not None:
            wr0 = pose_mod.point(prev, f"{side}_wrist")
            if wr0 is not None and not action_mod._wrist_teleport(
                frames, side, int(track_rel) - 4, frame_h
            ):
                vel = (float(wr[0] - wr0[0]), float(wr[1] - wr0[1]))
                if float(np.hypot(vel[0], vel[1])) > 8 and vel[1] < 4 and abs(vel[0]) > abs(vel[1]) * 0.35:
                    throw_dir = vel
                elif throw_dir is None and elbow_dir is not None:
                    throw_dir = elbow_dir
        # Wrist travel through the throw peak — left/right of the pitch, not
        # the cocking forearm. Skip a 1-frame landmark teleport.
        if peak_fr is not None:
            a_i, b_i = int(peak_fr) - 3, int(peak_fr) + 2
            a = action_mod.frame_by_index(frames, a_i)
            b = action_mod.frame_by_index(frames, b_i)
            if a is not None and b is not None:
                wa = pose_mod.point(a, f"{side}_wrist")
                wb = pose_mod.point(b, f"{side}_wrist")
                if (
                    wa is not None
                    and wb is not None
                    and not action_mod._wrist_teleport(frames, side, a_i, frame_h)
                    and not action_mod._wrist_teleport(frames, side, b_i, frame_h)
                ):
                    velp = (float(wb[0] - wa[0]), float(wb[1] - wa[1]))
                    if float(np.hypot(velp[0], velp[1])) > 16 and abs(velp[0]) > abs(velp[1]) * 0.25:
                        throw_dir = velp
        segs, margin = _body_segments(pose_track, side)
        return track.track_ball_from_release(
            video_path,
            release_frame=int(track_rel),
            wrist_xy=(float(wr[0]), float(wr[1])),
            fps=fps,
            meters_per_pixel=scale.get("meters_per_pixel"),
            frame_w=frame_w,
            frame_h=frame_h,
            throw_dir=throw_dir,
            body_segments=segs,
            body_margin=margin,
            on_progress=on_progress,
        )
    except Exception:
        traceback.print_exc()
        return []

