"""Render the Cric-Lab processed video: original colour + burned-in HUD.

Detection may convert *copies* of frames to gray (optical flow, contours).
The overlay MP4 is written from the original BGR frames — grayscale is never
the output. Same metrics JSON as the results page and PDF.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.pipeline import pose as posemod

# Brand / phase colours in BGR.
GREEN = (46, 61, 11)
ORANGE = (38, 92, 196)
RED = (28, 28, 185)
WHITE = (245, 245, 245)
INK = (28, 24, 20)
# Overlays sit *over* footage a coach is trying to read. Blended, not opaque,
# and thin enough that the body stays visible underneath.
TRAIL_ALPHA = 0.55
MARKER_ALPHA = 0.75
TRAIL_W_HAND = 5
TRAIL_W_BALL = 3
SKELETON_W = 2
SKELETON_ALPHA = 0.6
BONE = (210, 190, 90)       # skeleton (light blue)
BLUE = (220, 150, 50)       # FFC
GREEN_PH = (90, 200, 90)    # MER
YELLOW = (40, 210, 230)     # REL
PURPLE = (190, 90, 190)     # FT

MAX_W = 1280

PHASE_COLOR = {
    "back_foot_contact": BLUE,
    "front_foot_contact": BLUE,
    "hip_rotation": GREEN_PH,
    "max_external_rotation": GREEN_PH,
    "arm_horizontal": GREEN_PH,
    "release": YELLOW,
    "follow_through": PURPLE,
}

TIMELINE_TAGS = [
    ("back_foot_contact", "BFC"),
    ("front_foot_contact", "FFC"),
    ("max_external_rotation", "MER"),
    ("release", "REL"),
    ("follow_through", "FT"),
]

PAUSE_SEC = 2.5
PAUSE_LABELS = {
    "back_foot_contact": "BACK-FOOT CONTACT",
    "front_foot_contact": "FRONT-FOOT CONTACT",
    "max_external_rotation": "MAX ARM COCKING",
    "arm_horizontal": "ARM HORIZONTAL",
    "release": "RELEASE",
    "follow_through": "FOLLOW-THROUGH",
}


def _skeleton_pairs() -> list[tuple[int, int]]:
    if posemod.POSE_CONNECTIONS:
        return [(int(a), int(b)) for a, b in posemod.POSE_CONNECTIONS]
    L = posemod.LANDMARKS
    names = [
        ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"), ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"), ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"), ("left_hip", "right_hip"),
        ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ]
    return [(L[a], L[b]) for a, b in names]


def _panel(img, x, y, w, h, alpha=0.62, color=INK):
    x, y, w, h = int(x), int(y), int(w), int(h)
    if w <= 0 or h <= 0:
        return
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(img.shape[1], x + w), min(img.shape[0], y + h)
    sub = img[y0:y1, x0:x1]
    if sub.size == 0:
        return
    overlay = np.full_like(sub, color, dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, sub, 1 - alpha, 0, sub)


def _text(img, s, org, scale=0.6, color=WHITE, thick=1, shadow=True):
    if shadow:
        cv2.putText(img, s, (org[0] + 1, org[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, INK, thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _metric_ok(metrics, key):
    m = metrics.get(key) or {}
    if m.get("status") != "ok" or m.get("value") is None:
        return None
    return m


def _height_label(metrics: dict[str, Any]) -> str:
    prof = metrics.get("player_profile") or {}
    ft, inches = prof.get("height_ft"), prof.get("height_in")
    if ft is not None:
        inn = int(round(float(inches or 0)))
        return f"{int(ft)}' {inn}\""
    hm = prof.get("height_m")
    if not hm:
        return ""
    total_in = float(hm) / 0.0254
    fti = int(total_in // 12)
    inn = int(round(total_in % 12))
    if inn == 12:
        fti += 1
        inn = 0
    return f"{fti}' {inn}\""


def _timeline_events(phases: dict[str, Any]) -> list[tuple[str, str, int]]:
    events = []
    used = set()
    for key, tag in TIMELINE_TAGS:
        fr = phases.get(key)
        if key == "max_external_rotation" and fr is None:
            fr = phases.get("arm_horizontal")
        if fr is None:
            continue
        fr = int(fr)
        if fr in used:
            continue
        used.add(fr)
        events.append((key, tag, fr))
    events.sort(key=lambda t: t[2])
    return events


def _color_for_frame(idx: int, phases: dict[str, Any]) -> tuple[int, int, int]:
    """Phase paint: blue wind-up → green FFC → yellow MER → purple REL."""
    rel = phases.get("release")
    mer = phases.get("max_external_rotation") or phases.get("arm_horizontal")
    ffc = phases.get("front_foot_contact") or phases.get("back_foot_contact")
    if rel is not None and idx >= int(rel):
        return PURPLE
    if mer is not None and idx >= int(mer):
        return YELLOW
    if ffc is not None and idx >= int(ffc):
        return GREEN_PH
    return BLUE


def render_overlay_video(
    *,
    video_path: Path,
    out_path: Path,
    pose_track: dict[str, Any],
    action: dict[str, Any],
    metrics: dict[str, Any],
    ball_track: list[dict[str, Any]] | None = None,
    player_name: str = "Bowler",
    release_still_path: Path | None = None,
    stills_dir: Path | None = None,
    capture_fps: float | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    # `capture_fps` is the rate the clip was really shot at (see
    # app.pipeline.timebase). A slow-motion export carries the *playback* rate in
    # its header, so trusting the header would make the overlay re-slow footage
    # that is already slowed.
    in_fps = float(capture_fps or cap.get(cv2.CAP_PROP_FPS) or 30.0)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    scale = min(1.0, MAX_W / src_w) if src_w else 1.0
    out_w = int(round(src_w * scale / 2) * 2)
    out_h = int(round(src_h * scale / 2) * 2)
    out_fps = 30.0

    lms_by_frame = {int(f["frame"]): f for f in (pose_track.get("frames") or [])}
    pairs = _skeleton_pairs()
    side = action.get("throwing_side") or (metrics.get("throwing_side"))
    phases = dict(metrics.get("phases") or action.get("phases") or {})
    release_frame = action.get("release_frame") or metrics.get("release_frame")
    win = action.get("delivery_window") or {}
    w_start = win.get("start", release_frame)
    w_end = win.get("end", release_frame)
    events = _timeline_events(phases)
    seq = metrics.get("kinematic_sequence") or []
    pause_at = _pause_map(events, in_fps, out_fps)

    ball_pts_by_frame: dict[int, dict[str, Any]] = {}
    for p in (ball_track or []):
        ball_pts_by_frame[int(p["frame"])] = {
            "pt": (int(p["x"] * scale), int(p["y"] * scale)),
            "speed_kmh": p.get("speed_kmh"),
            "r": max(6.0, float(p.get("r") or 8) * scale),
        }

    fps_src = float(pose_track.get("fps") or in_fps)
    pad = max(8, int(round(fps_src * 0.50)))
    trail_start = int(w_start) if w_start is not None else 0
    for k in ("back_foot_contact", "front_foot_contact"):
        if phases.get(k) is not None:
            trail_start = min(trail_start, int(phases[k]))
    trail_start = max(0, trail_start - pad)
    trail_end = int(w_end) if w_end is not None else int(release_frame or 0)
    if phases.get("follow_through") is not None:
        trail_end = max(trail_end, int(phases["follow_through"]))

    wrist_trail = _smooth_wrist_trail(lms_by_frame, side, trail_start, trail_end, scale, fps_src)

    fourcc, writer = _open_writer(out_path, out_fps, (out_w, out_h))

    rel_scaled = None
    if wrist_trail and release_frame is not None:
        nearest = min(wrist_trail, key=lambda t: abs(t[0] - int(release_frame)))
        rel_scaled = nearest[1]
    elif metrics.get("release_point"):
        rp = metrics["release_point"]
        rel_scaled = (int(rp["x"] * scale), int(rp["y"] * scale))

    saved_still = False
    still_targets = _still_frame_map(phases)
    saved_phase_stills: set[str] = set()
    if stills_dir is not None:
        stills_dir.mkdir(parents=True, exist_ok=True)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Playback normalisation. The source may be 30 fps or a 200+ fps slow-mo
    # capture; the output is always 30 fps. Emit output frames so playback is
    # ~real-time outside the delivery window and a gentle 2× slow-mo inside it
    # (SpinLab cadence) — never "everything ×7 slower" on high-fps clips.
    rate_out = out_fps / max(in_fps, 1e-6)

    idx = 0
    frames_written = 0
    emit_acc = 0.0
    while True:
        # Decide from idx alone before decoding — skipped frames are only
        # grab()bed (packet advance), never colour-converted, which matters
        # when a 200 fps source maps to 30 fps output.
        if not cap.grab():
            break

        in_window = (
            w_start is not None and w_end is not None and w_start - 3 <= idx <= w_end + 3
        )
        freeze = pause_at.get(idx)
        if freeze:
            reps = freeze["reps"]
            emit_acc = 0.0
        else:
            emit_acc += rate_out * (2.0 if in_window else 1.0)
            reps = int(emit_acc)
            emit_acc -= reps

        need_still = (stills_dir is not None and idx in still_targets) or (
            not saved_still
            and release_still_path is not None
            and release_frame is not None
            and idx >= release_frame
        )
        if reps <= 0 and not need_still:
            idx += 1
            continue

        ok, frame = cap.retrieve()
        if not ok:
            idx += 1
            continue

        if scale != 1.0:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

        pf = lms_by_frame.get(idx)
        _draw_skeleton(frame, pf, pairs, side, scale)
        _draw_trail(frame, idx, wrist_trail, ball_pts_by_frame, phases, events, release_frame,
                    release_speed_kmh=(_metric_ok(metrics, "ball_speed_kmh") or {}).get("value"))

        if rel_scaled is not None and release_frame is not None and idx >= release_frame:
            cv2.circle(frame, rel_scaled, 18, (18, 18, 18), 5, cv2.LINE_AA)
            cv2.circle(frame, rel_scaled, 16, YELLOW, 3, cv2.LINE_AA)
            cv2.circle(frame, rel_scaled, 6, WHITE, -1, cv2.LINE_AA)
            _text(frame, "REL", (rel_scaled[0] + 20, rel_scaled[1] - 12), 0.68, YELLOW, 2)

        # Clean phase stills for the PDF (skeleton only — no HUD), like SpinLab page 1.
        if stills_dir is not None and idx in still_targets:
            for key in still_targets[idx]:
                if key in saved_phase_stills:
                    continue
                cv2.imwrite(str(stills_dir / f"{key}.jpg"), frame)
                saved_phase_stills.add(key)

        _draw_top_bar(frame, out_w, player_name, metrics, in_window)
        _draw_metric_tiles(frame, out_w, metrics)
        _draw_sequence(frame, out_w, idx, seq)
        _draw_timeline(frame, out_w, out_h, idx, events, seq, total_frames)
        if freeze:
            _draw_pause_banner(frame, out_w, freeze["label"])

        if not saved_still and release_still_path is not None and release_frame is not None and idx >= release_frame:
            cv2.imwrite(str(release_still_path), frame)
            saved_still = True

        for _ in range(reps):
            writer.write(frame)
            frames_written += 1
        idx += 1
        if on_progress and total_frames:
            on_progress(min(idx, total_frames), total_frames)

    writer.release()
    cap.release()

    return {
        "path": str(out_path),
        "fourcc": fourcc,
        "width": out_w,
        "height": out_h,
        "fps": out_fps,
        "frames_written": frames_written,
        "duration_s": round(frames_written / out_fps, 2),
        "source_fps": in_fps,
        "stills": sorted(saved_phase_stills),
    }


def _still_frame_map(phases: dict[str, Any]) -> dict[int, list[str]]:
    """Video frame index → phase keys to snapshot for the PDF."""
    keys = [
        "back_foot_contact",
        "front_foot_contact",
        "hip_rotation",
        "max_external_rotation",
        "arm_horizontal",
        "release",
        "follow_through",
    ]
    out: dict[int, list[str]] = {}
    for key in keys:
        fr = phases.get(key)
        if fr is None:
            continue
        out.setdefault(int(fr), []).append(key)
    return out


def _pause_map(events: list[tuple[str, str, int]], src_fps: float, out_fps: float) -> dict[int, dict[str, Any]]:
    """Hold each timeline event for PAUSE_SEC, skipping a duplicate tag on the same frame."""
    reps = max(45, int(round(out_fps * PAUSE_SEC)))
    out: dict[int, dict[str, Any]] = {}
    last_fr = None
    for key, tag, fr in events:
        if last_fr is not None and int(fr) == int(last_fr):
            continue
        out[int(fr)] = {
            "reps": reps,
            "label": PAUSE_LABELS.get(key, tag),
            "tag": tag,
        }
        last_fr = fr
    return out


def _draw_pause_banner(frame, w, label: str):
    if not label:
        return
    tw = 12 * len(label) + 28
    x = max(12, w // 2 - tw // 2)
    _panel(frame, x, 44, tw, 32, alpha=0.78, color=ORANGE)
    _text(frame, label, (x + 14, 66), 0.55, WHITE, 2)


def _open_writer(out_path: Path, fps: float, size: tuple[int, int]):
    for cc in ("avc1", "H264", "mp4v"):
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*cc), fps, size)
        if writer.isOpened():
            return cc, writer
        writer.release()
    raise RuntimeError("Could not open any VideoWriter codec (avc1/H264/mp4v)")


def _draw_skeleton(frame, pf, pairs, side, scale):
    if not pf:
        return
    lms = pf["landmarks"]

    def sp(i):
        x, y, v = lms[i]
        if v < 0.25:
            return None
        return (int(x * scale), int(y * scale))

    bowling_ids = set()
    if side:
        L = posemod.LANDMARKS
        bowling_ids = {L[f"{side}_shoulder"], L[f"{side}_elbow"], L[f"{side}_wrist"]}

    # Blended as one layer rather than per-limb: overlapping strokes drawn with
    # per-shape alpha would compound at the joints and read as opaque there.
    layer = frame.copy()
    for a, b in pairs:
        pa, pb = sp(a), sp(b)
        if pa is None or pb is None:
            continue
        bowling = a in bowling_ids and b in bowling_ids
        cv2.line(layer, pa, pb, ORANGE if bowling else BONE,
                 SKELETON_W + 1 if bowling else SKELETON_W, cv2.LINE_AA)
    for i in range(len(lms)):
        p = sp(i)
        if p is None:
            continue
        cv2.circle(layer, p, 3, ORANGE if i in bowling_ids else BONE, -1, cv2.LINE_AA)
    cv2.addWeighted(layer, SKELETON_ALPHA, frame, 1.0 - SKELETON_ALPHA, 0, frame)


def _smooth_wrist_trail(
    lms_by_frame: dict[int, dict[str, Any]],
    side: str | None,
    start: int,
    end: int,
    scale: float,
    fps: float,
) -> list[tuple[int, tuple[int, int]]]:
    """Dense, smoothed bowling-wrist path."""
    if not side:
        return []
    wi = posemod.LANDMARKS[f"{side}_wrist"]
    raw_f, raw_x, raw_y = [], [], []
    for fr in range(int(start), int(end) + 1):
        pf = lms_by_frame.get(fr)
        if not pf:
            continue
        x, y, v = pf["landmarks"][wi]
        if v < 0.25:
            continue
        raw_f.append(int(fr))
        raw_x.append(float(x) * scale)
        raw_y.append(float(y) * scale)
    if len(raw_f) < 3:
        return [(f, (int(x), int(y))) for f, x, y in zip(raw_f, raw_x, raw_y)]
    win = max(3, min(9, int(round(fps * 0.028))))
    if win % 2 == 0:
        win += 1
    kernel = np.ones(win) / win
    pad = win // 2
    xs = np.convolve(np.pad(raw_x, (pad, pad), mode="edge"), kernel, mode="valid")[: len(raw_x)]
    ys = np.convolve(np.pad(raw_y, (pad, pad), mode="edge"), kernel, mode="valid")[: len(raw_y)]
    return [(int(f), (int(x), int(y))) for f, x, y in zip(raw_f, xs, ys)]


def _draw_trail(frame, idx, wrist_trail, ball_pts_by_frame, phases, events, release_frame,
                release_speed_kmh=None):
    past_wrist = [(fr, pt) for fr, pt in wrist_trail if fr <= idx]
    past_ball = sorted(
        (fr, v["pt"])
        for fr, v in ball_pts_by_frame.items()
        if fr <= idx and (release_frame is None or fr >= int(release_frame))
    )
    if release_frame is not None and past_wrist and past_ball:
        rel_pt = min(past_wrist, key=lambda t: abs(t[0] - int(release_frame)))[1]
        if past_ball[0][1] != rel_pt:
            past_ball = [(int(release_frame), rel_pt)] + past_ball

    def stroke(points: list[tuple[int, tuple[int, int]]], thickness: int):
        if len(points) < 2:
            return
        segs: list[tuple[tuple[int, int, int], list[tuple[int, int]]]] = []
        color = _color_for_frame(points[0][0], phases)
        buf = [points[0][1]]
        for fr, pt in points[1:]:
            c = _color_for_frame(fr, phases)
            if c != color:
                if len(buf) >= 2:
                    segs.append((color, buf))
                buf = [buf[-1], pt]
                color = c
            else:
                buf.append(pt)
        if len(buf) >= 2:
            segs.append((color, buf))
        # Drawn onto a scratch copy and blended back, so the footage stays
        # readable through the trail. Painting straight onto `frame` at these
        # widths hid the very mechanics the trail is meant to point at.
        layer = frame.copy()
        for col, pts in segs:
            arr = np.array(pts, np.int32)
            cv2.polylines(layer, [arr], False, (22, 22, 22), thickness + 3, cv2.LINE_AA)
            cv2.polylines(layer, [arr], False, col, thickness, cv2.LINE_AA)
        cv2.addWeighted(layer, TRAIL_ALPHA, frame, 1.0 - TRAIL_ALPHA, 0, frame)

    # Hand path is the hero. Ball path continues after REL.
    stroke(past_wrist, TRAIL_W_HAND)
    stroke(past_ball, TRAIL_W_BALL)

    # Phase beads on the hand path.
    if past_wrist:
        by_fr = {fr: pt for fr, pt in past_wrist}
        for key, tag, fr in events:
            if fr > idx:
                continue
            nearest = min(past_wrist, key=lambda t: abs(t[0] - fr))
            pt = by_fr.get(fr, nearest[1])
            col = PHASE_COLOR.get(key, WHITE)
            bead = frame.copy()
            cv2.circle(bead, pt, 8, (18, 18, 18), -1, cv2.LINE_AA)
            cv2.circle(bead, pt, 6, col, -1, cv2.LINE_AA)
            cv2.circle(bead, pt, 6, WHITE, 1, cv2.LINE_AA)
            cv2.addWeighted(bead, MARKER_ALPHA, frame, 1.0 - MARKER_ALPHA, 0, frame)

    live = ball_pts_by_frame.get(idx)
    if live:
        cx, cy = live["pt"]
        br = int(max(10, live.get("r") or 10))
        color = _color_for_frame(idx, phases)
        ring = frame.copy()
        cv2.ellipse(ring, (cx, cy), (br, max(8, int(br * 0.72))), 0, 0, 360, color, 2, cv2.LINE_AA)
        cv2.circle(ring, (cx, cy), 3, WHITE, -1, cv2.LINE_AA)
        cv2.addWeighted(ring, MARKER_ALPHA, frame, 1.0 - MARKER_ALPHA, 0, frame)
        _text(frame, "BALL", (cx + br + 8, cy - br), 0.5, color, 1)
        # Deliberately the *measured release speed*, not this frame's raw
        # displacement. The per-frame value swung 81-133 km/h on noise alone,
        # so the ball was being labelled 133 on the same frame the panel read
        # 100 — one render contradicting itself. There is one ball speed for
        # the delivery; it is shown here and in the panel, or nowhere.
        if release_speed_kmh is not None:
            _text(frame, f"{release_speed_kmh:.0f} km/h at release",
                  (cx + br + 8, cy - br + 22), 0.5, WHITE, 1)


def _draw_top_bar(frame, w, player_name, metrics, in_window):
    _panel(frame, 0, 0, w, 36, alpha=0.7)
    height = _height_label(metrics)
    left = "CricLab  |  Bowling Analysis"
    _text(frame, left, (12, 24), 0.62, WHITE, 2)
    right = "  |  ".join(p for p in (player_name, height) if p)
    _text(frame, right, (w - 12 - 9 * len(right), 24), 0.55, (160, 210, 120), 2)
    if in_window:
        _panel(frame, 12, 44, 132, 26, alpha=0.7, color=RED)
        _text(frame, "SLOW MOTION", (20, 63), 0.5, WHITE, 1)


def _fmt_tile(m, kind: str) -> tuple[str, str]:
    """(big value, small unit) — SpinLab tiles set the number much larger."""
    if not m:
        return "—", ""
    v = m["value"]
    if kind == "kmh":
        return f"{v:.0f}", "KM/H"
    if kind == "m":
        return f"{v:.2f}", "M"
    if kind == "ms":
        return f"{v:.0f}", "MS"
    return f"{v:.0f}", ""


TILE_W, TILE_H, TILE_GAP = 168, 64, 8


def _draw_metric_tiles(frame, w, metrics):
    tiles = [
        ("BALL SPEED", _fmt_tile(_metric_ok(metrics, "ball_speed_kmh"), "kmh")),
        ("ARM SPEED", _fmt_tile(_metric_ok(metrics, "arm_speed_kmh"), "kmh")),
        ("RELEASE HEIGHT", _fmt_tile(_metric_ok(metrics, "release_height_m"), "m")),
        ("RELEASE TIME", _fmt_tile(_metric_ok(metrics, "release_time_ms"), "ms")),
    ]
    tw, th, gap = TILE_W, TILE_H, TILE_GAP
    x0 = w - 12 - tw * 2 - gap
    y0 = 44
    for i, (lab, (val, unit)) in enumerate(tiles):
        col, row = i % 2, i // 2
        x = x0 + col * (tw + gap)
        y = y0 + row * (th + gap)
        _panel(frame, x, y, tw, th, alpha=0.72)
        _text(frame, lab, (x + 10, y + 17), 0.38, (180, 180, 180), 1)
        _text(frame, val, (x + 10, y + 50), 0.95, WHITE, 2)
        if unit:
            vw = cv2.getTextSize(val, cv2.FONT_HERSHEY_SIMPLEX, 0.95, 2)[0][0]
            _text(frame, unit, (x + 16 + vw, y + 50), 0.42, (185, 185, 185), 1)


def _draw_sequence(frame, w, idx, seq: list[dict[str, Any]]):
    if not seq:
        return
    tw = TILE_W * 2 + TILE_GAP
    x0 = w - 12 - tw
    y0 = 44 + TILE_H * 2 + TILE_GAP + 10
    _panel(frame, x0, y0, tw, 18 + 22 * len(seq), alpha=0.68)
    _text(frame, "KINEMATIC SEQUENCE", (x0 + 10, y0 + 16), 0.4, (180, 180, 180), 1)
    for i, item in enumerate(seq):
        y = y0 + 22 + i * 22
        n = int(item.get("n") or (i + 1))
        fr = item.get("frame")
        reached = fr is not None and idx >= int(fr)
        cx, cy = x0 + 18, y + 4
        color = BLUE if reached else (90, 90, 90)
        cv2.circle(frame, (cx, cy), 8, color, -1 if reached else 1, cv2.LINE_AA)
        _text(frame, str(n), (cx - 4, cy + 4), 0.35, WHITE if reached else (160, 160, 160), 1, shadow=False)
        label = str(item.get("label") or "").upper()
        if item.get("estimated"):
            label += "  EST."
        _text(frame, label, (x0 + 34, y + 8), 0.4, WHITE if reached else (150, 150, 150), 1)


def _draw_timeline(frame, w, h, idx, events, seq, total_frames):
    if not events:
        return
    pad_l, pad_r = 96, 48
    bar_y = h - 28
    bar_h = 10
    bar_x0, bar_x1 = pad_l, w - pad_r
    bar_w = bar_x1 - bar_x0
    # Full-video span (SpinLab): grey track everywhere, colour only between events.
    t0 = 0
    t1 = max(int(total_frames) - 1, max(fr for _, _, fr in events), 1)

    def x_at(fr: int) -> int:
        t = (fr - t0) / (t1 - t0)
        return int(bar_x0 + max(0.0, min(1.0, t)) * bar_w)

    _panel(frame, 8, bar_y - 36, bar_x1 - 8 + 16, 62, alpha=0.55)

    # Progress figure at the left of the track, SpinLab-style.
    pct = int(round(100.0 * min(1.0, max(0.0, idx / max(1, t1)))))
    _text(frame, f"{pct} %", (16, bar_y + bar_h - 1), 0.72, WHITE, 2)

    # Grey base track for the whole clip.
    cv2.rectangle(frame, (bar_x0, bar_y), (bar_x1, bar_y + bar_h), (74, 74, 74), -1)

    # Colored segments between consecutive events (delivery window only).
    for (k0, _tag0, fr0), (_k1, _tag1, fr1) in zip(events, events[1:]):
        x0, x1 = x_at(fr0), x_at(fr1)
        color = PHASE_COLOR.get(k0, BLUE)
        cv2.rectangle(frame, (x0, bar_y), (x1, bar_y + bar_h), color, -1)

    # Tags above the bar — nudged right when close events would overlap.
    last_right = -1e9
    for key, tag, fr in events:
        x = x_at(fr)
        active = abs(idx - fr) <= 4
        color = PHASE_COLOR.get(key, WHITE)
        tw = 8 * len(tag) + 16
        tx = max(int(x - tw // 2), int(last_right + 4))
        tx = min(tx, bar_x1 - tw)
        last_right = tx + tw
        _panel(frame, tx, bar_y - 28, tw, 20, alpha=0.8, color=color if active else INK)
        _text(frame, tag, (tx + 6, bar_y - 13), 0.4, WHITE, 1)

    # Sequence numbers under matching frames.
    key_to_n = {item.get("key"): item.get("n") for item in seq if item.get("frame") is not None}
    # MER row may have fallen back to arm_horizontal.
    if "max_external_rotation" in key_to_n:
        key_to_n.setdefault("arm_horizontal", key_to_n["max_external_rotation"])
    for key, _tag, fr in events:
        n = key_to_n.get(key)
        if n is None and key == "release":
            n = 4
        if n is None and key == "front_foot_contact":
            n = 1
        if n is None:
            continue
        x = x_at(fr)
        cv2.circle(frame, (x, bar_y + bar_h + 12), 8, BLUE, -1, cv2.LINE_AA)
        _text(frame, str(int(n)), (x - 4, bar_y + bar_h + 16), 0.35, WHITE, 1, shadow=False)

    play_x = x_at(idx)
    cv2.line(frame, (play_x, bar_y - 4), (play_x, bar_y + bar_h + 4), WHITE, 2, cv2.LINE_AA)
