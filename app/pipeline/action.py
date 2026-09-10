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


def _series(frames: list[dict[str, Any]], name: str) -> tuple[list[int], np.ndarray]:
    idxs: list[int] = []
    pts: list[list[float]] = []
    for f in frames:
        p = posemod.point(f, name)
        if p is not None:
            idxs.append(int(f["frame"]))
            pts.append([float(p[0]), float(p[1])])
    return idxs, (np.array(pts, dtype=float) if pts else np.empty((0, 2)))


def theil_sen_slope(t: np.ndarray, v: np.ndarray) -> float:
    """Median of every pairwise slope — the robust derivative the ball-speed fit uses.

    Least squares lets one bad sample bend the whole line; the median slope
    moves only if more than half the pairs are wrong. Cheap for the ≤ ~11
    samples a local window holds.
    """
    n = len(t)
    if n < 2:
        return 0.0
    sl = [
        (float(v[j]) - float(v[i])) / (float(t[j]) - float(t[i]))
        for i in range(n)
        for j in range(i + 1, n)
        if float(t[j]) > float(t[i])
    ]
    return float(np.median(sl)) if sl else 0.0


# Local window for every robust derivative on the pose track. Wide enough to
# hold 5 samples at 120 fps and 11 at 240, short enough that the bowling-arm
# whip (which peaks over ~30-40 ms) is not flattened.
DERIV_WINDOW_S = 0.04


def _window_bounds(idxs: list[int], i: int, half: int) -> tuple[int, int]:
    """Index range of samples within `half` frames of sample i (inclusive)."""
    lo, hi, n = i, i, len(idxs)
    while lo > 0 and idxs[i] - idxs[lo - 1] <= half:
        lo -= 1
    while hi < n - 1 and idxs[hi + 1] - idxs[i] <= half:
        hi += 1
    return lo, hi


def _robust_speed(
    idxs: list[int],
    pts: np.ndarray,
    fps: float,
    window_s: float = DERIV_WINDOW_S,
) -> tuple[list[float], list[float]]:
    """Speed in px/frame at every sample from a Theil–Sen velocity fit.

    For each sample the x(t) and y(t) slopes are the median pairwise slopes over
    a ~40 ms window centred on it; speed is their magnitude. A one-frame
    landmark jump corrupts only the pairs it appears in, so it no longer prints
    as a spike (and, through the moving average that used to follow, as two).
    Returns the speed series and, per sample, the median distance of the
    window's points from the fitted local line — the evidence for how well a
    straight-line velocity described those frames.
    """
    n = len(idxs)
    if n < 2:
        return [0.0] * n, [0.0] * n
    half = max(2, int(round(float(fps) * window_s / 2.0)))
    t_all = np.asarray(idxs, dtype=float)
    speeds: list[float] = []
    resids: list[float] = []
    for i in range(n):
        lo, hi = _window_bounds(idxs, i, half)
        if hi - lo + 1 < 3:
            lo, hi = max(0, i - 1), min(n - 1, i + 1)
        if hi - lo + 1 < 2:
            speeds.append(0.0)
            resids.append(0.0)
            continue
        t = t_all[lo: hi + 1]
        x = pts[lo: hi + 1, 0]
        y = pts[lo: hi + 1, 1]
        vx = theil_sen_slope(t, x)
        vy = theil_sen_slope(t, y)
        x0 = float(np.median(x - vx * t))
        y0 = float(np.median(y - vy * t))
        resid = np.hypot(x - (x0 + vx * t), y - (y0 + vy * t))
        speeds.append(float(np.hypot(vx, vy)))
        resids.append(float(np.median(resid)))
    return speeds, resids


def _wrist_speed(
    frames: list[dict[str, Any]],
    name: str,
    *,
    fps: float = 30.0,
) -> tuple[list[int], list[float], list[float]]:
    """Wrist speed series from the robust local velocity (see `_robust_speed`).

    Replaces a 25 ms position average followed by first differences and an
    18 ms speed average: at high frame rates that turned landmark jitter into
    speed noise, and one wrist teleport into a two-frame spike that could win
    the peak search. Returns (frame indices, px/frame, local fit residual px).
    """
    idxs, pts = _series(frames, name)
    speeds, resids = _robust_speed(idxs, pts, fps)
    return idxs, speeds, resids


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
    max_jump = _jump_limit_px(frames, int(frame_i), frame_h, fps)
    return float(np.hypot(float(wr[0] - prev[0]), float(wr[1] - prev[1]))) > max_jump


def _jump_limit_px(frames: list[dict[str, Any]], frame_i: int, frame_h: float, fps: float) -> float:
    """Farthest a real wrist can move in one frame, in this clip's pixels."""
    fr = frame_by_index(frames, int(frame_i))
    body = posemod.body_pixel_height(fr) if fr is not None else None
    if body:
        # Body height is ~1.8 m of the same pixels, so px-per-metre = body / 1.8.
        max_jump = MAX_WRIST_MPS * (float(body) / 1.8) / max(float(fps), 1.0)
    else:
        max_jump = 0.5 * float(frame_h or 1080) / max(float(fps), 1.0) * 3.0
    return max(max_jump, 0.03 * float(frame_h or 1080))


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


LEAVE_MIN_CONFIDENCE = 0.5
# Ball centre to wrist landmark when the ball is held, in ball radii (~9-11 cm).
HAND_OFFSET_R = 2.5
_G_MPS2 = 9.80665


def _no_leave(note: str, **extra: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"frame": None, "subframe": None, "confidence": 0.0, "note": note}
    out.update(extra)
    return out


def ball_leave_frame(
    pose_track: dict[str, Any],
    side: str | None,
    ball_track: list[dict[str, Any]] | None,
    pose_release: int | None,
    *,
    fps: float | None = None,
    meters_per_pixel: float | None = None,
) -> dict[str, Any]:
    """The last pose frame where the bowling wrist is still on the ball.

    Fits the early in-air path and walks it backward to the wrist. This is the
    ball's own account of when it left the hand, so it outranks any wrist-speed
    heuristic — the hand keeps accelerating for a frame or two after the ball is
    gone, and decelerates before it on a slower delivery.

    The fit is the same physics-constrained robust one the speed metric uses:
    Theil–Sen slopes, gravity pinned to ``g/(mpp·fps²)`` when the scale and
    capture rate are known, outliers trimmed against the fit's own residual
    spread — over the six detections nearest the hand only. A least-squares
    parabola over sixteen points was bent by one streak-centroid jump and
    back-projected to nowhere near the wrist, so the snap silently failed on
    every 4K clip and the pose heuristic's early release stood.

    Wrist samples whose elbow→wrist reach has collapsed (MediaPipe parks the
    wrist on the forearm or chest exactly at leave-hand) and one-frame
    teleports are excluded from the search; they are the reason the stored
    release point could sit hundreds of pixels below the first ball detection.

    Returns ``{"frame", "subframe", "confidence", "note", ...evidence}``.
    ``frame`` is None when nothing usable was found. ``confidence`` (0-1) is
    the geometric mean of how close the wrist got relative to the tolerance,
    how sharply the wrist-to-flight distance grows after leave, and how well
    the flight points fit — scaled down when fewer than 6 points took part.
    """
    if not ball_track or not side:
        return _no_leave("no tracked ball")
    frames = pose_track.get("frames") or []
    if pose_release is None or not frames:
        return _no_leave("no pose release to search around")
    if not fps:
        fps = float(pose_track.get("fps") or 30.0)
    frame_h = int(pose_track.get("height") or 1080)

    pts = sorted(
        [p for p in ball_track if p.get("source") != "interpolated"],
        key=lambda p: int(p["frame"]),
    )
    if len(pts) < 3:
        pts = sorted(ball_track, key=lambda p: int(p["frame"]))
    if len(pts) < 3:
        return _no_leave("fewer than 3 ball points")
    first_f = int(pts[0]["frame"])
    # Only the samples nearest the hand. Back-projecting a few frames is a
    # local extrapolation: a level shift in the centroid a dozen frames later
    # (the smear changing shape as the ball recedes) tilts a long fit's
    # vertical slope and lands the projection hundreds of pixels from the
    # wrist — which is what kept the snap from ever firing on 4K clips. Six
    # points still leave the Theil–Sen median a majority against one outlier.
    early = pts[:6]
    f = np.array([float(p["frame"]) for p in early])
    x = np.array([float(p["x"]) for p in early])
    y = np.array([float(p["y"]) for p in early])
    t = f - f[0]
    g_px = None
    if fps and meters_per_pixel and float(meters_per_pixel) > 0:
        g_px = _G_MPS2 / (float(meters_per_pixel) * float(fps) ** 2)

    def _fit(mask: np.ndarray) -> tuple[float, float, float, float]:
        tt, xx, yy = t[mask], x[mask], y[mask]
        vx = theil_sen_slope(tt, xx)
        x0 = float(np.median(xx - vx * tt))
        yl = yy - (0.5 * g_px * tt ** 2 if g_px else 0.0)
        vy = theil_sen_slope(tt, yl)
        y0 = float(np.median(yl - vy * tt))
        return vx, x0, vy, y0

    def _predict(vx: float, x0: float, vy: float, y0: float, tq: np.ndarray | float):
        tq = np.asarray(tq, dtype=float)
        px = x0 + vx * tq
        py = y0 + vy * tq + (0.5 * g_px * tq ** 2 if g_px else 0.0)
        return px, py

    r0_vals = [float(p.get("r") or 0) for p in early[:3] if float(p.get("r") or 0) > 0]
    r0 = float(np.median(r0_vals)) if r0_vals else 20.0
    keep = np.ones(len(t), dtype=bool)
    vx, x0, vy, y0 = _fit(keep)
    px, py = _predict(vx, x0, vy, y0, t)
    resid = np.hypot(x - px, y - py)
    med = float(np.median(resid))
    mad = float(np.median(np.abs(resid - med)))
    tol = max(med + 3.0 * 1.4826 * mad, 8.0, 0.35 * r0)
    keep = resid <= tol
    if keep.sum() >= 3 and not keep.all():
        vx, x0, vy, y0 = _fit(keep)
        px, py = _predict(vx, x0, vy, y0, t)
        resid = np.hypot(x - px, y - py)
    if keep.sum() < 3:
        keep = np.ones(len(t), dtype=bool)
    fit_resid = float(np.median(resid[keep]))
    step = float(np.hypot(vx, vy))

    # Back-projecting the in-air parabola only stays valid near the first
    # detected sample. A second-based window at a recovered 120 fps walks
    # into the cocking loop, where the wrist is near the *imaginary*
    # ballistic path and REL freezes while the ball is still held.
    back = 10
    search_lo = max(0, first_f - back, int(pose_release) - back)
    search_hi = first_f + 6
    wr_by: dict[int, np.ndarray] = {}
    reach_by: dict[int, float] = {}
    for fr_d in frames:
        fr = int(fr_d["frame"])
        if fr < search_lo or fr > search_hi:
            continue
        wr = posemod.point(fr_d, f"{side}_wrist")
        if wr is None:
            continue
        wr_by[fr] = wr
        el = posemod.point(fr_d, f"{side}_elbow")
        if el is not None:
            reach_by[fr] = float(np.hypot(float(wr[0] - el[0]), float(wr[1] - el[1])))
    if not wr_by:
        return _no_leave("bowling wrist not visible around the ball's first frames")
    reach_med = float(np.median(list(reach_by.values()))) if len(reach_by) >= 4 else None

    dist_at: dict[int, float] = {}
    skipped_collapsed: list[int] = []
    last_ok: tuple[int, np.ndarray] | None = None
    for fr in sorted(wr_by):
        reach = reach_by.get(fr)
        if reach_med and reach is not None and reach < 0.55 * reach_med:
            skipped_collapsed.append(fr)
            continue
        wr = wr_by[fr]
        # A teleport is judged against the last *usable* wrist, not simply the
        # previous frame — the frame after a collapsed landmark always looks
        # like a jump from the collapse, and that is the real wrist coming back.
        if last_ok is not None:
            df = max(1, fr - last_ok[0])
            jump = float(np.hypot(float(wr[0] - last_ok[1][0]), float(wr[1] - last_ok[1][1]))) / df
            if jump > _jump_limit_px(frames, fr, frame_h, fps):
                skipped_collapsed.append(fr)
                continue
        last_ok = (fr, wr)
        bx, by = _predict(vx, x0, vy, y0, float(fr - f[0]))
        dist_at[fr] = float(np.hypot(float(wr[0]) - float(bx), float(wr[1]) - float(by)))
    if not dist_at:
        return _no_leave(
            "wrist landmark collapsed or teleporting on every frame around leave-hand",
            skipped=skipped_collapsed,
        )
    best_fr = min(dist_at, key=lambda k: dist_at[k])
    best_d = dist_at[best_fr]

    # Scale purely off the ball and the bowler. With floors of 96 px and 14 px
    # the max() always won (r0 is typically 6-20 px), so both gates were absolute
    # pixels wearing a ball-relative disguise: 96 px is 13% of frame height at
    # 720p — accepting a wrist nowhere near the ball — and 4% at 4K, rejecting a
    # valid snap. These decide whether release is pinned to a measurement at all.
    body_px = _body_px_near(frames, best_fr)
    # A held ball is not centred on the wrist landmark: it sits in the fingers,
    # roughly 9-11 cm — about 2.5 radii of a 7.2 cm ball — beyond the wrist
    # joint. That offset is what a real release looks like, so the tolerance
    # is measured from it, not from zero: up to HAND_OFFSET_R radii is a
    # perfect match, and only the excess beyond that counts as evidence
    # against the back-projection.
    hand_px = HAND_OFFSET_R * r0
    near_tol = max(hand_px + 3.0 * r0, 0.1 * body_px) if body_px else hand_px + 3.0 * r0
    evidence = {
        "best_frame": int(best_fr),
        "best_d_px": round(best_d, 1),
        "near_tol_px": round(near_tol, 1),
        "hand_offset_px": round(hand_px, 1),
        "n_points": int(keep.sum()),
        "points_dropped": int(len(t) - keep.sum()),
        "fit_residual_px": round(fit_resid, 1),
        "step_px": round(step, 1),
        "first_ball_frame": first_f,
        "search": [int(search_lo), int(search_hi)],
        "skipped_frames": skipped_collapsed,
        "gravity_pinned": bool(g_px),
    }
    if best_d > near_tol:
        return _no_leave(
            f"wrist never within {near_tol:.0f} px of the back-projected flight "
            f"(closest {best_d:.0f} px at frame {best_fr})",
            **evidence,
        )

    thresh = best_d + (max(0.45 * r0, 0.012 * body_px) if body_px else 0.45 * r0)
    leave = best_fr
    ordered = sorted(k for k in dist_at if k >= best_fr)
    for fr in ordered:
        if dist_at[fr] <= thresh:
            leave = fr
        else:
            break
    # Sub-frame leave: the instant the wrist-to-flight distance crosses the
    # hand-size threshold, interpolated between the last held sample and the
    # next usable one. A constant fraction of a frame later than the true
    # separation, on every clip alike — precise, not merely quantised.
    subframe = float(leave)
    nxt = next((fr for fr in ordered if fr > leave), None)
    if nxt is not None and nxt - leave <= 3:
        d0, d1 = dist_at[leave], dist_at[nxt]
        if d1 > d0 + 1e-6:
            frac = float(np.clip((thresh - d0) / (d1 - d0), 0.0, 1.0))
            subframe = float(leave) + frac * float(nxt - leave)

    # Inside the hand offset the wrist is exactly where a held ball puts it;
    # beyond it the term decays linearly to zero at the tolerance.
    excess = max(0.0, best_d - hand_px)
    near_term = float(np.clip(1.0 - excess / max(near_tol - hand_px, 1e-6), 0.0, 1.0))
    # Departure: the next usable wrist samples after leave must sit well off
    # the flight. If there are none the wrist never separated from the tracked
    # object inside the window — that is a lock on the hand, not a ball leaving it.
    after = [dist_at[fr] for fr in ordered if fr > leave][:3]
    if after:
        sep = (float(np.median(after)) - best_d) / max(r0, 1.0)
        dep_term = float(np.clip(sep / 3.0, 0.0, 1.0))
    else:
        dep_term = 0.0
    fit_term = float(np.clip(1.0 - fit_resid / max(0.35 * step, 1e-6), 0.2, 1.0)) if step > 0 else 0.2
    n_term = min(1.0, float(keep.sum()) / 6.0)
    confidence = float((max(near_term, 1e-6) * max(dep_term, 1e-6) * fit_term) ** (1.0 / 3.0) * n_term)
    if dep_term <= 0.0:
        confidence = 0.0
    note = (
        f"wrist {best_d:.0f} px from the back-projected flight at frame {best_fr}; "
        f"ball leaves at {subframe:.2f}"
    )
    if confidence < LEAVE_MIN_CONFIDENCE:
        note = "ball leave back-projection low confidence — " + note
    out = {
        "frame": int(leave),
        "subframe": round(subframe, 3),
        "confidence": round(confidence, 3),
        "note": note,
        "terms": {
            "near": round(near_term, 3),
            "departure": round(dep_term, 3),
            "fit": round(fit_term, 3),
            "points": round(n_term, 3),
        },
    }
    out.update(evidence)
    return out


def snap_release_to_ball_leave(
    pose_track: dict[str, Any],
    action: dict[str, Any],
    ball_track: list[dict[str, Any]] | None,
    *,
    fps: float | None = None,
    meters_per_pixel: float | None = None,
    min_confidence: float = LEAVE_MIN_CONFIDENCE,
) -> dict[str, Any]:
    """Move release onto the ball's leave-hand frame, in place.

    REL stays on the hand — never on a ball already in the sky. The snap only
    happens when the back-projection is confident; otherwise the pose release
    stands and the reason is recorded on the action for the metrics to show.
    Returns the leave report either way.
    """
    leave = ball_leave_frame(
        pose_track, action.get("throwing_side"), ball_track, action.get("release_frame"),
        fps=fps, meters_per_pixel=meters_per_pixel,
    )
    action["release_confidence"] = float(leave.get("confidence") or 0.0)
    action["release_note"] = leave.get("note")
    if leave.get("frame") is None or float(leave.get("confidence") or 0.0) < min_confidence:
        return leave
    fr = int(leave["frame"])
    sub = float(leave.get("subframe") if leave.get("subframe") is not None else fr)
    mer = (action.get("phases") or {}).get("max_external_rotation")
    if mer is not None and fr < int(mer) + 1:
        fr = int(mer) + 1
        sub = float(fr)
    ft = (action.get("phases") or {}).get("follow_through")
    if ft is not None and fr > int(ft):
        fr = int(ft)
        sub = float(fr)

    action["release_frame"] = int(fr)
    action.setdefault("phases", {})["release"] = int(fr)
    action.setdefault("phase_sources", {})["release"] = "wrist_closest_to_ball_path"
    action.setdefault("phase_subframes", {})["release"] = round(sub, 3)
    return leave


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


def _plant_subframe(
    candidates: list[tuple[int, Any]],
    plant_frame: int,
    *,
    fps: float,
    body_px: float | None = None,
    min_drop_px: float | None = None,
) -> tuple[float, dict[str, Any]]:
    """Fractional frame at which the ankle first reached its settled height.

    `_plant_frame` answers in whole frames — ±8 ms at 120 fps, ±33 ms at 30.
    Here a Theil–Sen line through the last 2-4 descending samples before the
    plateau is solved for the instant it crosses the settled level; when the
    descent is too short or not monotonic the crossing is interpolated on the
    smoothed series instead. The residual of those samples around the line is
    returned as the evidence for how sharply the plant happened.
    """
    idx_of = {int(f): i for i, (f, _) in enumerate(candidates)}
    j = idx_of.get(int(plant_frame))
    fit: dict[str, Any] = {"residual_px": None, "drop_px": None, "samples": 0, "method": "none"}
    if j is None or j == 0 or len(candidates) < 4:
        return float(plant_frame), fit
    fs = [float(f) for f, _ in candidates]
    ys = [float(p[1]) for _, p in candidates]
    y_smooth = _moving_avg(ys, _odd_win(max(3, int(round(fps * 0.03)))))
    hi, lo = max(y_smooth), min(y_smooth)
    settled = float(y_smooth[-1])
    level = settled - max(2.0, 0.02 * (hi - lo))
    fit["drop_px"] = round(float(hi - lo), 1)
    # Descent samples: raw ankle heights still above the settled level, ending
    # at the sample before the plateau. Image y grows downward, so "above" is
    # a smaller y.
    lo_i = max(0, j - 4)
    seg = [(fs[i], ys[i]) for i in range(lo_i, j) if ys[i] < level]
    t_star: float | None = None
    if len(seg) >= 2:
        tt = np.array([a for a, _ in seg])
        yy = np.array([b for _, b in seg])
        slope = theil_sen_slope(tt, yy)
        if slope > 1e-6:
            icpt = float(np.median(yy - slope * tt))
            resid = float(np.median(np.abs(yy - (icpt + slope * tt))))
            cand = (level - icpt) / slope
            # The moving average that found the plateau lags the raw plant by up
            # to half its window, so the raw crossing may sit one frame earlier
            # than the smoothed interval.
            lo_ok = fs[j - 2] if j >= 2 else fs[j - 1]
            if lo_ok - 1e-9 <= cand <= fs[j] + 1e-9:
                t_star = float(cand)
                fit.update(residual_px=round(resid, 2), samples=len(seg), method="line")
    if t_star is None:
        y_prev, y_now = float(y_smooth[j - 1]), float(y_smooth[j])
        frac = float(np.clip((level - y_prev) / (y_now - y_prev), 0.0, 1.0)) if y_now > y_prev else 1.0
        t_star = fs[j - 1] + frac * (fs[j] - fs[j - 1])
        fit.update(samples=2, method="interp")
        if fit["residual_px"] is None:
            fit["residual_px"] = round(abs(y_now - y_prev) * 0.25, 2)
    return float(t_star), fit


def _detect_front_foot_contact(
    frames: list[dict[str, Any]],
    side: str,
    release_frame: int,
    *,
    fps: float = 30.0,
) -> tuple[int, float, dict[str, Any]] | None:
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
    body_px = _body_px_near(frames, release_frame)
    plant = _plant_frame(candidates, fps=fps, body_px=body_px)
    if plant is None:
        return None
    sub, fit = _plant_subframe(candidates, plant, fps=fps, body_px=body_px)
    return plant, sub, fit


def _detect_back_foot_contact(
    frames: list[dict[str, Any]],
    side: str,
    release_frame: int,
    ffc: int | None,
    *,
    fps: float = 30.0,
) -> tuple[int, float, dict[str, Any]] | None:
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
    body_px = _body_px_near(frames, end)
    plant = _plant_frame(candidates, fps=fps, body_px=body_px)
    if plant is None:
        return None
    sub, fit = _plant_subframe(candidates, plant, fps=fps, body_px=body_px)
    return plant, sub, fit


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
    release_subframe: float | None = None,
) -> dict[str, Any]:
    """Return throwing side, release, FFC, and wrist peak on the bowling arm.

    `release_override` pins release to a frame measured elsewhere — in practice
    the frame the tracked ball left the hand. Every other event is searched
    relative to release, so handing that in makes the whole phase set key off a
    measurement instead of a wrist-speed heuristic. `release_subframe` is the
    fractional leave instant that came with it.

    Besides the integer `phases` (overlay, timeline, PDF, stills), the result
    carries `phase_subframes` — fractional frames for the foot plants and, when
    measured, release — for the time-based metrics only.
    """
    frames = pose_track.get("frames") or []
    result: dict[str, Any] = {
        "throwing_side": None,
        "release_frame": None,
        "peak_wrist_speed_px_per_frame": None,
        "phases": {},
        "phase_sources": {},
        "phase_subframes": {},
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

    r_idx, r_speed, r_resid = _wrist_speed(frames, "right_wrist", fps=fps)
    l_idx, l_speed, l_resid = _wrist_speed(frames, "left_wrist", fps=fps)
    r_peak = max(r_speed) if r_speed else 0.0
    l_peak = max(l_speed) if l_speed else 0.0

    if declared == "right":
        side, idxs, speed, resid = "right", r_idx, r_speed, r_resid
    elif declared == "left":
        side, idxs, speed, resid = "left", l_idx, l_speed, l_resid
    elif r_peak >= l_peak:
        side, idxs, speed, resid = "right", r_idx, r_speed, r_resid
    else:
        side, idxs, speed, resid = "left", l_idx, l_speed, l_resid

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

    ffc_hit = _detect_front_foot_contact(frames, side, release_frame, fps=fps)
    ffc = ffc_hit[0] if ffc_hit else None

    phases: dict[str, Any] = {
        "release": int(release_frame),
        "follow_through": int(follow),
    }
    sources: dict[str, str] = {
        "release": "ball_leave_hand" if release_override is not None else "peak_wrist_speed_then_throw_axis",
        "follow_through": "wrist_speed_decay",
    }
    subframes: dict[str, float] = {}
    subframe_fit: dict[str, Any] = {}
    if release_override is not None and release_subframe is not None:
        subframes["release"] = round(float(release_subframe), 3)
    if ffc_hit is not None:
        phases["front_foot_contact"] = int(ffc)
        sources["front_foot_contact"] = "lead_ankle_plant"
        subframes["front_foot_contact"] = round(float(ffc_hit[1]), 3)
        subframe_fit["front_foot_contact"] = ffc_hit[2]
    bfc_hit = _detect_back_foot_contact(frames, side, release_frame, ffc, fps=fps)
    if bfc_hit is not None:
        phases["back_foot_contact"] = int(bfc_hit[0])
        sources["back_foot_contact"] = "trail_ankle_plant"
        subframes["back_foot_contact"] = round(float(bfc_hit[1]), 3)
        subframe_fit["back_foot_contact"] = bfc_hit[2]
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
    throw_res = [
        float(r)
        for fr, r in zip(idxs, resid)
        if start_fr <= int(fr) <= int(release_frame)
    ]
    wrist_fit = {
        "window_ms": round(DERIV_WINDOW_S * 1000.0),
        "median_residual_px": round(float(np.median(throw_res)), 2) if throw_res else None,
        "samples": len(throw_res),
    }

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
            "phase_subframes": subframes,
            "phase_subframe_fit": subframe_fit,
            "wrist_speed_fit": wrist_fit,
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
