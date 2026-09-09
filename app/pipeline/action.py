"""Bowling-action event detection from a pose track.

Release = last near-peak bowling-wrist speed after cocking, along the throw
axis; snapped to leave-hand when an in-air ball path is available.
Front-foot contact = lead ankle plant (lowest point after the downward strike).
Back-foot contact = trail ankle plant before FFC.
MER = max bowling-arm cocking (bent elbow, wrist high) between FFC and release.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.pipeline import pose as posemod


def _odd_win(n: int, lo: int = 3) -> int:
    n = max(lo, int(n))
    return n if n % 2 else n + 1


def _moving_avg(values: list[float], win: int) -> list[float]:
    if win < 3 or len(values) < 3:
        return values
    win = min(win, len(values) if len(values) % 2 else len(values) - 1)
    if win < 3:
        return values
    kernel = np.ones(win) / win
    padded = np.pad(values, (win // 2, win // 2), mode="edge")
    return list(np.convolve(padded, kernel, mode="valid")[: len(values)])


def _smooth_positions(pts: np.ndarray, win: int) -> np.ndarray:
    if len(pts) < 3 or win < 3:
        return pts
    xs = _moving_avg([float(p[0]) for p in pts], win)
    ys = _moving_avg([float(p[1]) for p in pts], win)
    return np.column_stack([xs, ys])


def _series(frames: list[dict[str, Any]], name: str) -> tuple[list[int], np.ndarray]:
    idxs: list[int] = []
    pts: list[list[float]] = []
    for f in frames:
        p = posemod.point(f, name)
        if p is not None:
            idxs.append(int(f["frame"]))
            pts.append([float(p[0]), float(p[1])])
    return idxs, (np.array(pts, dtype=float) if pts else np.empty((0, 2)))


def _speeds_from_positions(
    idxs: list[int],
    pts: np.ndarray,
    fps: float,
) -> list[float]:
    """Light ~25 ms position smooth, then central-difference speed."""
    if len(pts) < 3:
        return [0.0] * len(idxs)
    pos_win = _odd_win(max(3, int(round(fps * 0.025))))
    pts = _smooth_positions(pts, pos_win)
    speeds = [0.0]
    for i in range(1, len(pts)):
        df = max(1, idxs[i] - idxs[i - 1])
        speeds.append(float(np.linalg.norm(pts[i] - pts[i - 1]) / df))
    spd_win = _odd_win(max(3, int(round(fps * 0.018))))
    return _moving_avg(speeds, spd_win)


def _wrist_speed(
    frames: list[dict[str, Any]],
    name: str,
    *,
    fps: float = 30.0,
) -> tuple[list[int], list[float]]:
    """Light ~25 ms position smooth, then central-difference speed.

    Heavy averaging (~50 ms+) flattens the bowling-arm whip and under-reads km/h.
    Raw 1-frame deltas at 200 fps inflate km/h from landmark jitter.
    """
    idxs, pts = _series(frames, name)
    return idxs, _speeds_from_positions(idxs, pts, fps)


def _robust_peak_speed(idxs: list[int], speed: list[float]) -> tuple[int | None, float | None]:
    """Peak frame + median of the 3 samples around argmax (kills a single spike)."""
    if not speed or max(speed) <= 0:
        return None, None
    peak_pos = int(np.argmax(speed))
    lo = max(0, peak_pos - 1)
    hi = min(len(speed), peak_pos + 2)
    robust = float(np.median(speed[lo:hi]))
    return idxs[peak_pos], robust


def _wrist_at(frames: list[dict[str, Any]], side: str, frame_i: int):
    fr = frame_by_index(frames, frame_i)
    return posemod.point(fr, f"{side}_wrist") if fr is not None else None


def _wrist_on_arm(
    frames: list[dict[str, Any]],
    side: str,
    frame_i: int,
    wr: np.ndarray | None,
    frame_h: float,
) -> bool:
    """False when MediaPipe has parked the wrist on a wall sticker / poster."""
    if wr is None:
        return False
    fr = frame_by_index(frames, frame_i)
    el = posemod.point(fr, f"{side}_elbow") if fr is not None else None
    if el is None:
        return True
    reach = float(np.hypot(float(wr[0] - el[0]), float(wr[1] - el[1])))
    return reach <= max(160.0, 0.22 * float(frame_h or 1080))


# A bowling wrist tops out around 25 m/s. Anything past that in one frame is the
# landmark jumping, not the arm moving.
MAX_WRIST_MPS = 25.0


def _wrist_teleport(
    frames: list[dict[str, Any]],
    side: str,
    frame_i: int,
    frame_h: float,
    fps: float = 30.0,
) -> bool:
    """True when the wrist landmark jumped farther than an arm can move in one frame.

    "Farther than an arm can move" is a speed, so the bound has to carry the
    frame rate. As a fixed pixel count it inverts across the input range: at
    30 fps a real wrist covers ~90-190 px between frames at 1080p, so the true
    release frames get thrown out as teleports, while at 240 fps it covers ~12 px
    and genuine landmark jumps sail through.
    """
    wr = _wrist_at(frames, side, int(frame_i))
    prev = _wrist_at(frames, side, int(frame_i) - 1)
    if wr is None or prev is None:
        return False
    fr = frame_by_index(frames, int(frame_i))
    body = posemod.body_pixel_height(fr) if fr is not None else None
    if body:
        # Body height is ~1.8 m of the same pixels, so px-per-metre = body / 1.8.
        max_jump = MAX_WRIST_MPS * (float(body) / 1.8) / max(float(fps), 1.0)
    else:
        max_jump = 0.5 * float(frame_h or 1080) / max(float(fps), 1.0) * 3.0
    max_jump = max(max_jump, 0.03 * float(frame_h or 1080))
    return float(np.hypot(float(wr[0] - prev[0]), float(wr[1] - prev[1]))) > max_jump


def last_on_arm_wrist(
    frames: list[dict[str, Any]],
    side: str,
    frame_i: int,
    frame_h: float,
    lookback: int = 16,
    fps: float = 30.0,
) -> tuple[int, tuple[float, float] | None]:
    """Walk back across a 1-frame wrist teleport. Does not rewrite the pose series."""
    for back in range(0, max(0, int(lookback)) + 1):
        fi = int(frame_i) - back
        wr = _wrist_at(frames, side, fi)
        if wr is None or not _wrist_on_arm(frames, side, fi, wr, frame_h):
            continue
        if _wrist_teleport(frames, side, fi, frame_h, fps):
            continue
        return fi, (float(wr[0]), float(wr[1]))
    wr = _wrist_at(frames, side, int(frame_i))
    if wr is None:
        return int(frame_i), None
    return int(frame_i), (float(wr[0]), float(wr[1]))


def _refine_release_frame(
    frames: list[dict[str, Any]],
    side: str,
    idxs: list[int],
    speed: list[float],
    peak_pos: int,
    fps: float,
    frame_h: int = 1080,
) -> int:
    """Release = last near-peak wrist speed after cocking, along the throw.

    Peak wrist speed often lands in the cocking loop (highest hand). The ball
    leaves when the hand has come through — so we search *after* the highest
    wrist, along the hand's own downrange displacement, not along elbow→wrist
    (which points at the sky at MER).
    """
    peak = speed[peak_pos]
    pre = max(1, int(round(fps * 0.04)))
    # Leave is ~50-120 ms after the highest wrist, so the window is 0.16 s of
    # *real* time. Clamping it to 8-20 absolute frames only gave that near
    # 120 fps: a 30 fps container holding a 240 fps capture got 8 frames = 33 ms
    # of real time and stopped short of leave-hand, while a native 240 fps clip
    # got 20 frames = 83 ms and also stopped short. A small floor keeps enough
    # samples to choose between on genuinely low-rate clips.
    post = max(4, int(round(fps * 0.16)))
    lo, hi = peak_pos, peak_pos
    while lo > 0 and idxs[peak_pos] - idxs[lo] <= pre:
        lo -= 1
    while hi < len(idxs) - 1 and idxs[hi] - idxs[peak_pos] <= post:
        hi += 1

    mer_i = peak_pos
    mer_y = 1e18
    for i in range(lo, hi + 1):
        wr = _wrist_at(frames, side, idxs[i])
        if wr is None or not _wrist_on_arm(frames, side, idxs[i], wr, frame_h):
            continue
        if float(wr[1]) < mer_y:
            mer_y = float(wr[1])
            mer_i = i

    wr_mer = _wrist_at(frames, side, idxs[mer_i])
    later = min(hi, mer_i + max(4, int(round(fps * 0.12))))
    wr_later = _wrist_at(frames, side, idxs[later])
    if wr_later is None:
        wr_later = wr_mer
    if wr_mer is None:
        return int(idxs[peak_pos])
    dx = float(wr_later[0]) - float(wr_mer[0])
    dy = float(wr_later[1]) - float(wr_mer[1])
    n = float(np.hypot(dx, dy)) or 1.0
    dx, dy = dx / n, dy / n

    best_i = mer_i
    best_score = -1e18
    px = max(1.0, float(frame_h or 1080) / 1080.0)
    min_travel = 18.0 * px
    max_drop = 45.0 * px
    for i in range(mer_i, hi + 1):
        if speed[i] < 0.50 * peak and i > mer_i + 1:
            if speed[i] < 0.35 * peak:
                break
        wr = _wrist_at(frames, side, idxs[i])
        if wr is None or not _wrist_on_arm(frames, side, idxs[i], wr, frame_h):
            continue
        # Past leave-hand the bowling wrist falls toward the hip. Keep searching
        # along the throw, but stop once the hand has clearly dropped off MER.
        if float(wr[1]) > mer_y + max_drop:
            break
        travel = float(np.hypot(float(wr[0]) - float(wr_mer[0]), float(wr[1]) - float(wr_mer[1])))
        if travel < min_travel:
            continue  # still at cocking — the ball has not left
        proj = (float(wr[0]) - float(wr_mer[0])) * dx + (float(wr[1]) - float(wr_mer[1])) * dy
        # Prefer the last still-fast sample along the throw (leave-hand), not MER.
        score = proj + 0.25 * (speed[i] / max(peak, 1e-6)) * n
        if score >= best_score:
            best_score = score
            best_i = i
    if best_i == mer_i:
        best_i = min(hi, mer_i + 1)
    return int(idxs[best_i])


def _body_px_near(frames: list[dict[str, Any]], frame_i: int | None) -> float | None:
    """Bowler's pixel stature near a frame — the scale anatomy gates belong on."""
    if frame_i is None:
        return None
    best = None
    for f in frames:
        if abs(int(f["frame"]) - int(frame_i)) > 8:
            continue
        h = posemod.body_pixel_height(f)
        if h and (best is None or h > best):
            best = float(h)
    return best


def ball_leave_frame(
    pose_track: dict[str, Any],
    side: str | None,
    ball_track: list[dict[str, Any]] | None,
    pose_release: int | None,
) -> int | None:
    """The last pose frame where the bowling wrist is still on the ball.

    Fits the early in-air path and walks it backward to the wrist. This is the
    ball's own account of when it left the hand, so it outranks any wrist-speed
    heuristic — the hand keeps accelerating for a frame or two after the ball is
    gone, and decelerates before it on a slower delivery.
    """
    if not ball_track or not side:
        return None
    frames = pose_track.get("frames") or []
    fps = float(pose_track.get("fps") or 30.0)
    pose_rel = pose_release
    if pose_rel is None or not frames:
        return None

    pts = sorted(
        [p for p in ball_track if p.get("source") != "interpolated"],
        key=lambda p: int(p["frame"]),
    )
    if len(pts) < 3:
        pts = sorted(ball_track, key=lambda p: int(p["frame"]))
    pts = pts[:16]
    if len(pts) < 3:
        return None
    ts = np.array([float(p["frame"]) for p in pts])
    xs = np.array([float(p["x"]) for p in pts])
    ys = np.array([float(p["y"]) for p in pts])
    try:
        cx = np.polyfit(ts, xs, 1)
        cy = np.polyfit(ts, ys, 2 if len(pts) >= 5 else 1)
    except Exception:
        return None

    first_f = int(pts[0]["frame"])
    # Back-projecting the in-air parabola only stays valid near the first
    # detected sample. A second-based window at a recovered 120 fps walks
    # into the cocking loop, where the wrist is near the *imaginary*
    # ballistic path and REL freezes while the ball is still held.
    back = 10
    search_lo = max(0, first_f - back, int(pose_rel) - back)
    search_hi = first_f + 6
    r0 = float(pts[0].get("r") or 20.0)

    best_fr, best_d = None, 1e18
    dist_at: dict[int, float] = {}
    for f in frames:
        fr = int(f["frame"])
        if fr < search_lo or fr > search_hi:
            continue
        wr = posemod.point(f, f"{side}_wrist")
        if wr is None:
            continue
        bx, by = float(np.polyval(cx, fr)), float(np.polyval(cy, fr))
        d = float(np.hypot(float(wr[0]) - bx, float(wr[1]) - by))
        dist_at[fr] = d
        if d < best_d:
            best_d = d
            best_fr = fr

    # Scale purely off the ball and the bowler. With floors of 96 px and 14 px
    # the max() always won (r0 is typically 6-20 px), so both gates were absolute
    # pixels wearing a ball-relative disguise: 96 px is 13% of frame height at
    # 720p — accepting a wrist nowhere near the ball — and 4% at 4K, rejecting a
    # valid snap. These decide whether release is pinned to a measurement at all.
    body_px = _body_px_near(frames, best_fr if best_fr is not None else pose_rel)
    near_tol = max(3.2 * r0, 0.06 * body_px) if body_px else 3.2 * r0
    if best_fr is None or best_d > near_tol:
        return None

    thresh = best_d + (max(0.45 * r0, 0.012 * body_px) if body_px else 0.45 * r0)
    leave = best_fr
    for fr in sorted(k for k in dist_at if k >= best_fr):
        if dist_at[fr] <= thresh:
            leave = fr
        else:
            break
    return int(leave)


def snap_release_to_ball_leave(
    pose_track: dict[str, Any],
    action: dict[str, Any],
    ball_track: list[dict[str, Any]] | None,
) -> None:
    """Move release onto the ball's leave-hand frame, in place.

    REL stays on the hand — never on a ball already in the sky.
    """
    leave = ball_leave_frame(
        pose_track, action.get("throwing_side"), ball_track, action.get("release_frame")
    )
    if leave is None:
        return
    mer = (action.get("phases") or {}).get("max_external_rotation")
    if mer is not None:
        leave = max(int(leave), int(mer) + 1)
    ft = (action.get("phases") or {}).get("follow_through")
    if ft is not None:
        leave = min(int(leave), int(ft))

    action["release_frame"] = int(leave)
    action.setdefault("phases", {})["release"] = int(leave)
    action.setdefault("phase_sources", {})["release"] = "wrist_closest_to_ball_path"


def _plant_frame(
    candidates: list[tuple[int, Any]],
    *,
    fps: float,
    body_px: float | None = None,
    min_drop_px: float | None = None,
) -> int | None:
    """First frame of the final plateau: when the ankle reached the height it holds.

    A foot plants by descending and then staying put. Rather than hunting the
    hardest downward strike — which on a high-frame-rate clip is a run-up stride,
    not the delivery — we take the height the ankle sits at nearest the end of
    the window and walk backward to the first frame it reached that height. That
    is the contact frame, at any frame rate.

    Returns None when the ankle is already planted across the whole window (the
    contact happened before it) or never descends — both are honest "not seen".
    """
    if len(candidates) < 4:
        return None
    ys = [float(p[1]) for _, p in candidates]
    y_smooth = _moving_avg(ys, _odd_win(max(3, int(round(fps * 0.03)))))
    hi, lo = max(y_smooth), min(y_smooth)
    # "Did the foot descend at all" scales with the bowler, not the sensor: a
    # fixed 6 px is landmark noise at 4K and more than a genuine small plant on a
    # wide 480p shot.
    if min_drop_px is None:
        min_drop_px = max(2.0, 0.015 * float(body_px)) if body_px else 6.0
    if hi - lo < min_drop_px:
        return None  # no descent in this window — nothing planted here
    settled = float(y_smooth[-1])
    if settled < hi - 0.35 * (hi - lo):
        return None  # still travelling at the end of the window, not planted
    # A planted foot sits within a couple of pixels of its settled height; the
    # window's full range spans the run-up, so scale the tolerance tightly off it.
    level = settled - max(2.0, 0.02 * (hi - lo))
    j = len(y_smooth) - 1
    while j > 0 and y_smooth[j - 1] >= level:
        j -= 1
    if j == 0:
        return None  # planted for the whole window; contact is outside it
    return int(candidates[j][0])


def _detect_front_foot_contact(
    frames: list[dict[str, Any]],
    side: str,
    release_frame: int,
    *,
    fps: float = 30.0,
) -> int | None:
    """Lead-ankle plant before release.

    The gap to release spans roughly 60-600 ms: pace bowlers land the front foot
    ~100-160 ms before the ball leaves, and a slower action stretches further.
    The old 120 ms floor cut off legitimate quick actions entirely.
    """
    lead = "left" if side == "right" else "right"
    idxs, pts = _series(frames, f"{lead}_ankle")
    if len(pts) < 5:
        return None

    min_gap = max(2, int(round(fps * 0.06)))
    max_gap = max(min_gap + 6, int(round(fps * 0.60)))
    candidates = [
        (i, p) for i, p in zip(idxs, pts)
        if release_frame - max_gap <= i <= release_frame - min_gap
    ]
    return _plant_frame(candidates, fps=fps, body_px=_body_px_near(frames, release_frame))


def _detect_back_foot_contact(
    frames: list[dict[str, Any]],
    side: str,
    release_frame: int,
    ffc: int | None,
    *,
    fps: float = 30.0,
) -> int | None:
    """Trail-ankle plant before front-foot contact (or before release if FFC is missing)."""
    idxs, pts = _series(frames, f"{side}_ankle")
    if len(pts) < 5:
        return None
    end = int(ffc) if ffc is not None else int(release_frame)
    min_before = max(2, int(round(fps * 0.03)))
    max_before = max(min_before + 6, int(round(fps * 0.70)))
    candidates = [
        (i, p) for i, p in zip(idxs, pts)
        if end - max_before <= i <= end - min_before
    ]
    return _plant_frame(candidates, fps=fps, body_px=_body_px_near(frames, end))


def _detect_mer(
    frames: list[dict[str, Any]],
    side: str,
    start_frame: int,
    release_frame: int,
) -> int | None:
    """Max bowling-arm cocking: most flexed elbow with the wrist still high, before release."""
    best_fr = None
    best_score = -1e9
    for f in frames:
        fr = int(f["frame"])
        if fr < start_frame or fr >= release_frame:
            continue
        sh = posemod.point(f, f"{side}_shoulder")
        el = posemod.point(f, f"{side}_elbow")
        wr = posemod.point(f, f"{side}_wrist")
        if sh is None or el is None or wr is None:
            continue
        elbow = posemod.angle_3pt(sh, el, wr)
        if elbow is None or elbow > 165:
            continue
        score = (180.0 - float(elbow))
        if float(wr[1]) < float(el[1]):
            score += 18.0
        # Normalise the reach term against the bowler's own pixel size, or it
        # roughly doubles from 1080p to 4K and MER lands on a different frame at
        # different resolutions while every other term stays in degrees. With no
        # body height to normalise against, leave the term out rather than let a
        # raw pixel length outvote the angles.
        body_px = posemod.body_pixel_height(f)
        if body_px:
            score += 18.0 * float(np.linalg.norm(wr - sh)) / float(body_px)
        if score > best_score:
            best_score = score
            best_fr = fr
    return best_fr


# Slowest hip-shoulder separation change we would call "the hips turning".
# In degrees per second, so it means the same thing at any capture rate.
MIN_HIP_ROT_DEG_S = 4.5


def _detect_hip_rotation(
    frames: list[dict[str, Any]],
    start_frame: int,
    release_frame: int,
    fps: float = 30.0,
) -> int | None:
    """Frame of peak 2D hip-shoulder separation change (estimated, not 3D rotation).

    The threshold is a rate in degrees per second. Held as degrees per *frame* it
    fell by the frame-rate ratio for the same physical turn, so hip_rotation was
    simply absent from the phase set on every high-frame-rate or slow-motion clip
    and present on 30 fps ones.
    """
    prev = None
    best_fr = None
    best_d = 0.0
    for f in frames:
        fr = int(f["frame"])
        if fr < start_frame or fr >= release_frame:
            continue
        lsh = posemod.point(f, "left_shoulder")
        rsh = posemod.point(f, "right_shoulder")
        lhip = posemod.point(f, "left_hip")
        rhip = posemod.point(f, "right_hip")
        sh_line = posemod.segment_angle_deg(rsh, lsh)
        hip_line = posemod.segment_angle_deg(rhip, lhip)
        if sh_line is None or hip_line is None:
            continue
        sep = ((sh_line - hip_line + 180) % 360) - 180
        if prev is not None:
            d = abs(sep - prev[1]) / max(1, fr - prev[0])
            if d > best_d:
                best_d = d
                best_fr = fr
        prev = (fr, sep)
    if best_d * float(fps) < MIN_HIP_ROT_DEG_S:
        return None
    return best_fr


def analyze_action(
    pose_track: dict[str, Any],
    *,
    bowling_arm: str | None = None,
    release_override: int | None = None,
) -> dict[str, Any]:
    """Return throwing side, release, FFC, and wrist peak on the bowling arm.

    `release_override` pins release to a frame measured elsewhere — in practice
    the frame the tracked ball left the hand. Every other event is searched
    relative to release, so handing that in makes the whole phase set key off a
    measurement instead of a wrist-speed heuristic.
    """
    frames = pose_track.get("frames") or []
    result: dict[str, Any] = {
        "throwing_side": None,
        "release_frame": None,
        "peak_wrist_speed_px_per_frame": None,
        "phases": {},
        "phase_sources": {},
        "wrist_speed_series": [],
        "confidence": 0.1,
    }
    if len(frames) < 6:
        return result

    fps = float(pose_track.get("fps") or 30.0)
    frame_h = int(pose_track.get("height") or 1080)
    declared = (bowling_arm or "").strip().lower()
    if declared not in {"left", "right"}:
        declared = None

    r_idx, r_speed = _wrist_speed(frames, "right_wrist", fps=fps)
    l_idx, l_speed = _wrist_speed(frames, "left_wrist", fps=fps)
    r_peak = max(r_speed) if r_speed else 0.0
    l_peak = max(l_speed) if l_speed else 0.0

    if declared == "right":
        side, idxs, speed = "right", r_idx, r_speed
    elif declared == "left":
        side, idxs, speed = "left", l_idx, l_speed
    elif r_peak >= l_peak:
        side, idxs, speed = "right", r_idx, r_speed
    else:
        side, idxs, speed = "left", l_idx, l_speed

    peak_frame, peak = _robust_peak_speed(idxs, speed)
    if peak_frame is None or peak is None or peak <= 0:
        return result

    peak_wrist_frame = int(peak_frame)
    peak_pos = idxs.index(peak_frame) if peak_frame in idxs else int(np.argmax(speed))
    if release_override is not None:
        release_frame = int(release_override)
    else:
        release_frame = _refine_release_frame(
            frames, side, idxs, speed, peak_pos, fps, frame_h
        )
    if release_frame in idxs:
        peak_pos = idxs.index(release_frame)

    thresh = peak * 0.2
    start = peak_pos
    while start > 0 and speed[start - 1] > thresh:
        start -= 1
    end = peak_pos
    while end < len(speed) - 1 and speed[end + 1] > thresh:
        end += 1
    windup = idxs[start]
    follow = idxs[end]

    ffc = _detect_front_foot_contact(frames, side, release_frame, fps=fps)

    phases: dict[str, Any] = {
        "release": int(release_frame),
        "follow_through": int(follow),
    }
    sources: dict[str, str] = {
        "release": "ball_leave_hand" if release_override is not None else "peak_wrist_speed_then_throw_axis",
        "follow_through": "wrist_speed_decay",
    }
    if ffc is not None:
        phases["front_foot_contact"] = int(ffc)
        sources["front_foot_contact"] = "lead_ankle_plant"
    bfc = _detect_back_foot_contact(frames, side, release_frame, ffc, fps=fps)
    if bfc is not None:
        phases["back_foot_contact"] = int(bfc)
        sources["back_foot_contact"] = "trail_ankle_plant"
    mer_start = int(ffc) if ffc is not None else int(windup)
    mer = _detect_mer(frames, side, mer_start, release_frame)
    if mer is not None:
        phases["max_external_rotation"] = int(mer)
        sources["max_external_rotation"] = "max_bowling_arm_cocking"
    hip_start = int(ffc) if ffc is not None else int(windup)
    hip_rot_fr = _detect_hip_rotation(frames, hip_start, release_frame, fps)
    if hip_rot_fr is not None:
        phases["hip_rotation"] = int(hip_rot_fr)
        sources["hip_rotation"] = "peak_2d_hip_shoulder_change"
    arm_h = _detect_arm_horizontal(frames, side, windup, release_frame)
    if arm_h is not None:
        phases["arm_horizontal"] = int(arm_h)
        sources["arm_horizontal"] = "bowling_arm_near_horizontal"

    separation = (peak - thresh) / (peak + 1e-6)
    conf = min(0.85, 0.35 + 0.4 * separation + min(0.15, 0.01 * len(frames)))

    leave_px = None
    mer_fr = phases.get("max_external_rotation")
    pre = max(2, int(round(fps * 0.15)))
    start_fr = int(release_frame) - pre
    if mer_fr is not None:
        start_fr = max(start_fr, int(mer_fr))
    throw_spd = [
        float(s)
        for fr, s in zip(idxs, speed)
        if start_fr <= int(fr) <= int(release_frame)
    ]
    if len(throw_spd) >= 2:
        leave_px = float(np.percentile(throw_spd, 90))
    elif release_frame in idxs:
        leave_px = float(speed[idxs.index(release_frame)])

    result.update(
        {
            "throwing_side": side,
            "bowling_arm_source": "player_profile" if declared else "auto_detected",
            "release_frame": int(release_frame),
            "peak_wrist_frame": int(peak_wrist_frame),
            "peak_wrist_speed_px_per_frame": float(peak),
            "leave_hand_wrist_speed_px_per_frame": leave_px,
            "phases": phases,
            "phase_sources": sources,
            "wrist_speed_series": [
                {"frame": int(fr), "speed_px": float(s)} for fr, s in zip(idxs, speed)
            ],
            "delivery_window": {"start": int(windup), "end": int(follow)},
            "confidence": float(conf),
        }
    )
    return result


def _detect_arm_horizontal(
    frames: list[dict[str, Any]],
    side: str,
    start_frame: int,
    release_frame: int,
) -> int | None:
    best = None
    best_err = 25.0
    for f in frames:
        fr = int(f["frame"])
        if fr < start_frame or fr >= release_frame:
            continue
        sh = posemod.point(f, f"{side}_shoulder")
        wr = posemod.point(f, f"{side}_wrist")
        ang = posemod.segment_angle_deg(sh, wr)
        if ang is None:
            continue
        err = min(abs(ang), abs(abs(ang) - 180))
        if err < best_err:
            best_err = err
            best = fr
    return best


def frame_by_index(frames: list[dict[str, Any]], target: int | None) -> dict[str, Any] | None:
    if target is None or not frames:
        return None
    return min(frames, key=lambda f: abs(int(f["frame"]) - int(target)))
