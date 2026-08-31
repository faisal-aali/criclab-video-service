"""Trajectory extraction from per-frame ball candidates.

Robust method: RANSAC motion-model fitting. A ball in flight follows
    x(t) ~ x0 + vx*t          (near-constant horizontal pixel velocity)
    y(t) ~ y0 + vy*t + a*t^2   (parabola under gravity, in image pixels)
where t is the frame index. Random noise from grass, clouds, and camera shake
does NOT fit one consistent model across many consecutive frames, so it is
rejected as outliers. This is far more reliable than greedy nearest-neighbour
chaining for messy handheld footage.
"""

from __future__ import annotations

import os
import random
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

from app.pipeline.cv_vision import validate_ball_path_on_video
from app.pipeline.view import SLOWMO_HEADROOM, flight_geometry_ok, min_ball_step_px


def _px_scale(frame_w: int, frame_h: int) -> float:
    """Convert a 1080p-referenced pixel constant to this frame's pixels.

    Scales both ways. Clamping at 1.0 (as this once did) left every constant at
    its raw 1080p value on 720p and smaller footage, where it is 1.5-2.25x too
    large — the same over-fitting as not scaling at all, just in one direction.
    """
    return max(0.35, min(float(frame_w or 1), float(frame_h or 1)) / 1080.0)


def _dist(a: dict[str, Any], b: dict[str, Any]) -> float:
    return float(np.hypot(a["x"] - b["x"], a["y"] - b["y"]))


def build_trajectory(
    frames: list[dict[str, Any]],
    frame_w: int,
    frame_h: int,
    min_len: int = 5,
    iterations: int = 600,
    seed: int = 12345,
) -> list[dict[str, Any]]:
    """Return the inlier ball-flight points (time-ordered) of the best model."""
    rng = random.Random(seed)
    diag = float(np.hypot(frame_w, frame_h)) or 1.0
    tol = diag * 0.020  # inlier distance tolerance in pixels
    min_net = diag * 0.05  # require real displacement

    frames_with = [f for f in frames if f["candidates"]]
    if len(frames_with) < min_len:
        return []

    # Flatten candidates keyed by frame for fast inlier search.
    frame_index = [f["frame"] for f in frames_with]
    cand_by_frame = {f["frame"]: f["candidates"] for f in frames_with}

    best_inliers: list[dict[str, Any]] = []

    n = len(frames_with)
    for _ in range(iterations):
        # Sample 3 distinct frames within a short temporal window (a throw is brief).
        i0 = rng.randint(0, n - 3)
        window = frames_with[i0 : min(n, i0 + 30)]
        if len(window) < 3:
            continue
        picks = rng.sample(window, 3)
        picks.sort(key=lambda f: f["frame"])
        ts = [float(f["frame"]) for f in picks]
        if ts[0] == ts[2]:
            continue
        pxs = [rng.choice(f["candidates"]) for f in picks]
        xs = [p["x"] for p in pxs]
        ys = [p["y"] for p in pxs]

        try:
            cx = np.polyfit(ts, xs, 1)  # linear x(t)
            cy = np.polyfit(ts, ys, 2)  # quadratic y(t)
        except Exception:
            continue

        # Count inliers: best candidate per frame within tolerance.
        inliers: list[dict[str, Any]] = []
        for fr in frame_index:
            px = np.polyval(cx, fr)
            py = np.polyval(cy, fr)
            best_c = None
            best_d = tol
            for c in cand_by_frame[fr]:
                d = float(np.hypot(c["x"] - px, c["y"] - py))
                if d < best_d:
                    best_d = d
                    best_c = c
            if best_c is not None:
                inliers.append({"frame": int(fr), "x": float(best_c["x"]), "y": float(best_c["y"]), "r": float(best_c["r"])})

        if len(inliers) < min_len:
            continue
        inliers.sort(key=lambda p: p["frame"])
        net = _dist(inliers[0], inliers[-1])
        if net < min_net:
            continue
        if len(inliers) > len(best_inliers):
            best_inliers = inliers

    if not best_inliers:
        return []

    return _largest_contiguous(best_inliers)


def _fit_xy(ts: np.ndarray, xs: np.ndarray, ys: np.ndarray):
    """Fit the flight's image path: x AND y quadratic in frame index.

    x(t) is often modelled as a straight line, on the reasoning that horizontal
    velocity is constant. That holds in the world, not in the image: a delivery
    travels downrange, away from the lens, so its pixels-per-frame decay through
    the flight. Fitting x linearly mis-predicts the frames nearest release by
    tens of pixels — which is exactly where release speed is read — and an
    outlier filter built on it throws that early flight away.
    """
    deg = 2 if len(ts) >= 6 else 1
    t = ts - float(ts[0])
    return np.polyfit(t, xs, deg), np.polyfit(t, ys, deg), float(ts[0])


def _largest_contiguous(points: list[dict[str, Any]], max_gap: int = 8) -> list[dict[str, Any]]:
    """Keep the longest run of points without large frame gaps (one flight)."""
    if not points:
        return points
    runs: list[list[dict[str, Any]]] = [[points[0]]]
    for prev, cur in zip(points, points[1:]):
        if cur["frame"] - prev["frame"] <= max_gap:
            runs[-1].append(cur)
        else:
            runs.append([cur])
    return max(runs, key=len)


def estimate_release(
    ball_track: list[dict[str, Any]],
    player_track: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Release ~ first point of the clean flight path (ball leaving the hand)."""
    if len(ball_track) < 2:
        return None
    start = ball_track[0]
    conf = min(0.8, 0.4 + 0.04 * len(ball_track))
    return {
        "frame": int(start["frame"]),
        "x": float(start["x"]),
        "y": float(start["y"]),
        "confidence": float(conf),
    }


def track_ball_from_release(
    video_path,
    *,
    release_frame: int,
    wrist_xy: tuple[float, float],
    fps: float,
    meters_per_pixel: float | None = None,
    frame_w: int = 1280,
    frame_h: int = 720,
    throw_dir: tuple[float, float] | None = None,
    body_segments: dict[int, list[tuple[float, float, float, float]]] | None = None,
    body_margin: float = 0.0,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[dict[str, Any]]:
    """Track the ball in the air after release, one breakpoint per frame.

    Lock onto the object that *leaves* the hand and keeps moving downrange.
    A cricket ball is a small red/white disc; a training/football is a dark oval
    against the sky. We never treat the bowling hand itself as the ball.

    `body_segments` maps a frame index to the bowler's limb and torso segments
    for that frame, taken from the pose track. Pose runs first precisely so this
    is available: the single largest source of false ball candidates is the
    bowler themself — a forearm, a shoulder or a thigh is a fast-moving,
    ball-sized, ball-coloured blob, and it is what captured the greedy chain
    before this filter existed. The bowling forearm is deliberately NOT included
    by the caller, because at release the real ball is still beside it.
    """
    from app.pipeline import detect, extract

    if release_frame is None or wrist_xy is None:
        return []

    start = max(0, int(release_frame) - 8)
    # Scan a generous span of frames, not a span of seconds: `fps` here is the
    # container rate, which a slow-motion export understates several-fold, and a
    # short scan yields too few in-air samples for either the speed fit or the
    # gravity check in app.pipeline.timebase. Extra frames only cost decode time.
    end = int(release_frame) + min(150, max(40, int(round(fps * 0.50))))
    hand_x, hand_y = float(wrist_xy[0]), float(wrist_xy[1])
    # `directional` records whether we actually know which way the ball went.
    # Assuming rightward when we do not is a camera-setup assumption: on a clip
    # shot from the other side, every real candidate is then discarded as
    # travelling backwards. Without a direction we simply do not apply the
    # downrange test — the ballistic fit and optical flow still gate the path.
    directional = throw_dir is not None
    dx, dy = (1.0, -0.35)
    if throw_dir is not None:
        n = float(np.hypot(throw_dir[0], throw_dir[1])) or 1.0
        dx, dy = float(throw_dir[0]) / n, float(throw_dir[1]) / n
    # Elbow→wrist at cocking points at the sky. A cricket ball leaves along the
    # pitch, so a near-vertical "throw" would reject the real in-air path.
    if abs(dy) > abs(dx) * 1.25:
        # Flatten a sky-pointing forearm to pitch-ish, but never invent +x
        # when |dx| is noise — that drops a left-arm / left-stumps flight.
        dy = -0.18
        if abs(dx) >= 0.12:
            dx = 1.0 if dx >= 0 else -1.0
        else:
            # |dx| is noise — we genuinely do not know the downrange sign here.
            directional = False
        n = float(np.hypot(dx, dy)) or 1.0
        dx, dy = dx / n, dy / n
    # A throw_dir pointing at the turf is a follow-through / landmark jump onto
    # the chest, not leave-hand. The ball is still going up-and-out; requiring
    # it to sit *below* a collapsed wrist deletes the real flight.
    if dy > 0.10:
        dy = -0.18
        n = float(np.hypot(dx, dy)) or 1.0
        dx, dy = dx / n, dy / n

    per_frame: list[dict[str, Any]] = []
    prev_work_gray = None
    px = _px_scale(frame_w, frame_h)
    min_hand_early = 12.0 * px
    min_hand_late = 48.0 * px
    edge_pad_near, edge_pad_far = 24.0 * px, 48.0 * px
    min_down = 8.0 * px
    downrange_r = 28.0 * px
    turf_below_hand = 100.0 * px
    n_scan = max(1, int(end) - int(start) + 1)

    def _emit_progress(current: int, *, fitting: bool = False) -> None:
        if not on_progress:
            return
        if fitting:
            on_progress(n_scan, n_scan)
            return
        # Hold the last tick until path fitting finishes so the step bar is not
        # full with a spinner while greedy/RANSAC still run.
        shown = current if current < n_scan else max(1, n_scan - 1)
        on_progress(shown, n_scan)

    for idx, bgr in extract.iter_frame_range(video_path, start, end):
        elapsed = max(0, idx - int(release_frame))
        segs = (body_segments or {}).get(int(idx)) or []
        cands, prev_work_gray = detect.collect_flight_candidates(bgr, prev_work_gray)
        min_hand = min_hand_late if elapsed >= 4 else min_hand_early
        max_r = min(frame_w, frame_h) * 0.065
        # Downrange side follows the throw direction — never assume rightward.
        rightward = dx >= 0
        edge_near, edge_far = (edge_pad_near, edge_pad_far) if rightward else (edge_pad_far, edge_pad_near)
        filtered = []
        for c in cands:
            if float(c.get("r") or 0) > max_r:
                continue
            if c["x"] < edge_near or c["x"] > frame_w - edge_far:
                continue
            # Large blobs clipped by the downrange edge are trees/posts, not the ball.
            at_downrange_edge = c["x"] > frame_w * 0.88 if rightward else c["x"] < frame_w * 0.12
            if at_downrange_edge and float(c.get("r") or 0) > downrange_r and not c.get("streak"):
                continue
            dh = float(np.hypot(c["x"] - hand_x, c["y"] - hand_y))
            if dh < min_hand:
                continue
            # Turf is the bottom strip — not "72% of the frame". A 1080p clip
            # with the bowler in the lower half releases around y=0.7; a 0.72
            # cut deletes the white ball the moment it leaves the hand.
            turf_y = min(frame_h * 0.92, max(hand_y + turf_below_hand, frame_h * 0.82))
            if c["y"] > turf_y:
                continue
            if directional:
                down = (c["x"] - hand_x) * dx + (c["y"] - hand_y) * dy
                if elapsed >= 4 and down < min_down:
                    continue
            # On the bowler's own body. Allowed for the first couple of frames,
            # where the ball genuinely still overlaps the hand and arm.
            if elapsed >= 2 and body_margin > 0 and segs:
                if _near_any_segment(c["x"], c["y"], segs, body_margin + float(c.get("r") or 0)):
                    continue
            filtered.append(c)
        per_frame.append({"frame": int(idx), "candidates": filtered})
        _emit_progress(int(idx) - start + 1)

    _emit_progress(n_scan, fitting=True)
    # Two independent ways to find the flight, because each fails differently.
    # Greedy chaining follows the nearest candidate and can be captured by a big
    # arm or torso blob near release; ballistic RANSAC is order-independent and
    # scores whole hypotheses by how many candidates fit one parabola, so it
    # survives the ball being missed on a few frames. Both are carried through to
    # optical-flow validation and the better *surviving* one wins — picking a
    # single favourite up front loses the real flight when that favourite fails.
    seeds: list[list[dict[str, Any]]] = []
    greedy_paths = _best_flight_path(
        per_frame, hand_x, hand_y, dx, dy, fps, meters_per_pixel, frame_w, frame_h,
        directional,
    )
    for greedy in greedy_paths:
        if len(greedy) >= 5:
            seeds.append(greedy)
            # A 50-frame hybrid is mostly poster; ballistic_clean then keeps the
            # crawl and the in-air burst (first ~12 frames) is thrown away as
            # outliers — and the remnant starts after max_start. Seed the head
            # on its own so the leave-hand flight can survive.
            if len(greedy) >= 16:
                head = greedy[:12]
                if int(head[-1]["frame"]) - int(head[0]["frame"]) >= 5:
                    seeds.append(head)
    ransac = build_trajectory(per_frame, frame_w, frame_h, min_len=5)
    if len(ransac) >= 5:
        seeds.append([_point(p["frame"], p) for p in ransac])
    if os.environ.get("CRICLAB_TRACK_DEBUG"):
        print(f"greedy_n={len(greedy_paths)} ransac_n={len(ransac)} seeds={len(seeds)}", flush=True)
    if not seeds:
        return []

    # The ball leaves the hand at release. A path that only starts well after it
    # is something else in the scene that happens to move consistently — on a
    # 200 fps clip, RANSAC will happily find a long, clean, entirely wrong one
    # in the background once the real flight has left the frame.
    # Gate in frames (capped) *and* by distance to the hand: 0.15 s at 120 fps
    # is 18 frames, which drops a real lock a few frames late, while a poster
    # blob can start "soon" after REL hundreds of pixels from the wrist.
    max_start = int(release_frame) + max(16, min(40, int(round(fps * 0.28))))
    max_hand_dist = max(280.0, 0.28 * float(frame_w))
    max_start_above = max(220.0, 0.20 * float(frame_h))

    best: list[dict[str, Any]] = []
    best_key: tuple[float, int, int] | None = None
    min_geo_step = min_ball_step_px(frame_w, frame_h, fps, meters_per_pixel)
    dbg = os.environ.get("CRICLAB_TRACK_DEBUG")
    if dbg:
        print(f"track seeds={len(seeds)} max_start={max_start} min_step={min_geo_step:.1f}", flush=True)
    for seed in seeds:
        def _why(msg: str) -> None:
            if dbg:
                p0 = seed[0] if seed else {}
                p1 = seed[-1] if seed else {}
                print(
                    f"  skip n={len(seed)} f{p0.get('frame')}({p0.get('x',0):.0f},{p0.get('y',0):.0f})->"
                    f"f{p1.get('frame')} {msg}",
                    flush=True,
                )
        cand = _longest_continuous(seed)
        cand = _trim_lost_lock(cand)
        if len(cand) < 5:
            _why("continuous<5")
            continue
        cand = _ballistic_clean(cand, frame_w, frame_h)
        cand = _trim_lost_lock(cand)
        if len(cand) < 5:
            _why("clean<5")
            continue
        cand = _reassociate(per_frame, cand, frame_w, frame_h)
        cand = _ballistic_clean(cand, frame_w, frame_h)
        cand = _longest_continuous(cand)
        cand = _trim_lost_lock(cand)
        if len(cand) < 5:
            _why("reassoc<5")
            continue
        cand = _extend_backward(per_frame, cand, hand_x, hand_y, frame_w, frame_h, fps)
        cand = _trim_lost_lock(cand)
        if int(cand[0]["frame"]) > max_start:
            _why(f"late_start f{cand[0]['frame']}")
            continue
        dh0 = float(np.hypot(float(cand[0]["x"]) - hand_x, float(cand[0]["y"]) - hand_y))
        if dh0 > max_hand_dist:
            _why(f"far_hand {dh0:.0f}")
            continue
        if float(cand[0]["y"]) < hand_y - max_start_above:
            _why("too_high")
            continue
        span = max(1, int(cand[-1]["frame"]) - int(cand[0]["frame"]))
        net = _dist(cand[0], cand[-1])
        speed = net / span
        if speed < min_geo_step:
            _why(f"slow {speed:.1f}")
            continue
        ok_geo, geo_note = flight_geometry_ok(
            cand, frame_w, frame_h, fps, meters_per_pixel
        )
        if not ok_geo:
            _why(f"geo {geo_note}")
            continue
        cand = _refine_centroids(video_path, cand, frame_w, frame_h)
        cand = _trim_lost_lock(cand)
        detected = [p for p in cand if p.get("source") != "interpolated"]
        cand = fill_every_frame(cand)
        flow_pts = detected if len(detected) >= 5 else cand
        ok_flow, flow_note, flow_stats = validate_ball_path_on_video(video_path, flow_pts, frame_w, frame_h)
        if not ok_flow:
            _why(f"flow {flow_note} {flow_stats}")
            continue
        if dbg:
            print(f"  KEEP n={len(cand)} f{cand[0]['frame']} speed={speed:.1f} flow={flow_stats}", flush=True)
        key = (speed, len(cand), -int(cand[0]["frame"]))
        if best_key is None or key > best_key:
            best_key, best = key, cand
            best[0]["flow_note"] = flow_note
            best[0]["flow_stats"] = flow_stats
    if len(best) < 5:
        return []
    best = _trim_lost_lock(best)
    if len(best) < 5:
        return []
    # No km/h floor here: `fps` at this point is the *container* rate, which a
    # slow-motion export understates by 4-8x. Gating on km/h would throw away
    # exactly the flights that app.pipeline.timebase needs to detect that, and a
    # genuine ball would be discarded as "too slow". Speed sanity belongs in
    # metrics, after the timebase is settled. What remains here is
    # fps-independent: the path must cross the image (view.flight_geometry_ok)
    # and move with the optical flow (checked above).
    return annotate_frame_motion(best, fps, meters_per_pixel)


def _near_any_segment(
    x: float,
    y: float,
    segments: list[tuple[float, float, float, float]],
    margin: float,
) -> bool:
    """Is (x, y) within `margin` of any of these line segments?"""
    for x1, y1, x2, y2 in segments:
        vx, vy = x2 - x1, y2 - y1
        L2 = vx * vx + vy * vy
        if L2 <= 1e-6:
            if float(np.hypot(x - x1, y - y1)) <= margin:
                return True
            continue
        t = max(0.0, min(1.0, ((x - x1) * vx + (y - y1) * vy) / L2))
        if float(np.hypot(x - (x1 + t * vx), y - (y1 + t * vy))) <= margin:
            return True
    return False


def _track_ball_r(path: list[dict[str, Any]]) -> float:
    """This flight's own ball radius. The most direct scale available — no need
    to infer ball size from frame dimensions when the track measured it."""
    rs = [float(p.get("r") or 0) for p in path if float(p.get("r") or 0) > 0]
    return float(np.median(rs)) if rs else 8.0


def _longest_continuous(path: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the longest run with no teleport between consecutive samples.

    A ball's step between frames changes smoothly. A lock that jumps to a
    floodlight flare or a post moves hundreds of pixels in one frame while the
    real flight moved thirty — so the run is cut there rather than fitted
    through, which would bend the parabola and corrupt every value read off it.
    """
    if len(path) < 4:
        return path
    pts = sorted(path, key=lambda p: int(p["frame"]))
    steps = [
        _dist(a, b) / max(1, int(b["frame"]) - int(a["frame"]))
        for a, b in zip(pts, pts[1:])
    ]
    # Typical step = median of the *smaller* half. Median of every step lets two
    # teleports (lost ball, wrong blob) inflate the cut so a later 400 px lock-on
    # still counts as "continuous" and the speed fit is then refused as garbage.
    ordered = sorted(steps)
    typical = float(np.median(ordered[: max(1, (len(ordered) + 1) // 2)]))
    # A floor in absolute pixels loosens this test exactly where it matters: on a
    # high-frame-rate clip the ball steps only 5-12 px, so a 40 px floor lets a
    # flare eight times the real step splice into the track.
    limit = max(2.5 * typical, 2.0 * _track_ball_r(pts))
    runs: list[list[dict[str, Any]]] = [[pts[0]]]
    for step, nxt in zip(steps, pts[1:]):
        if step <= limit:
            runs[-1].append(nxt)
        else:
            runs.append([nxt])
    return max(runs, key=len)


def _trim_lost_lock(path: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop the tail once the lock stops travelling along its own heading.

    After the ball blurs out (three-quarter / 4K / 120 fps), the nearest blob
    is often turf or a bounce. Consecutive steps are still small, so
    `_longest_continuous` will not cut, but x-velocity collapses and the speed
    fit then reports garbage (or refuses the number). Cut at the first stall
    after a stable head of in-air samples.
    """
    if len(path) < 5:
        return path
    pts = sorted(
        (p for p in path if p.get("source") != "interpolated"),
        key=lambda p: int(p["frame"]),
    )
    if len(pts) < 5:
        pts = sorted(path, key=lambda p: int(p["frame"]))
    if len(pts) < 5:
        return path
    dxs: list[float] = []
    for a, b in zip(pts, pts[1:]):
        df = max(1, int(b["frame"]) - int(a["frame"]))
        dxs.append((float(b["x"]) - float(a["x"])) / df)
    typical = float(np.median(dxs[: min(4, len(dxs))]))
    if abs(typical) < 8.0:
        return path
    sign = 1.0 if typical >= 0 else -1.0
    cut = len(pts)
    for i, dx in enumerate(dxs):
        # Need five in-air samples before judging a stall, so a short blurry
        # flight is kept rather than discarded as "too short to be a ball".
        if i + 1 < 5:
            continue
        if sign * dx < 0.22 * abs(typical):
            cut = i + 1
            break
    kept = pts[:cut]
    if len(kept) < 5:
        return path
    # A first sample that sits well off the flight in y is usually a lock-on
    # from the hand or a flare, not leave-hand. Dropping it is what lets a
    # 9-point lost-ball clip (y=600 then a flat y=703 flight) quote a speed.
    rest = kept[1 : min(6, len(kept))]
    if len(kept) > 5 and len(rest) >= 4:
        ys = np.array([float(p["y"]) for p in rest], dtype=float)
        med_y = float(np.median(ys))
        mad_y = float(np.median(np.abs(ys - med_y))) or 1.0
        if abs(float(kept[0]["y"]) - med_y) > max(48.0, 4.0 * mad_y):
            kept = kept[1:]
    last_f = int(kept[-1]["frame"])
    first_f = int(kept[0]["frame"])
    out = [p for p in path if first_f <= int(p["frame"]) <= last_f]
    return out if len(out) >= 5 else path


def _reassociate(
    per_frame: list[dict[str, Any]],
    path: list[dict[str, Any]],
    frame_w: int,
    frame_h: int,
    iterations: int = 2,
) -> list[dict[str, Any]]:
    """Re-pick one candidate per frame against the flight's own fitted model.

    Greedy chaining is sequential, so one wrong pick drags the rest. Fitting
    x(t) linear and y(t) quadratic to the surviving points and then choosing the
    candidate nearest that curve is order-independent: it recovers frames the
    chain skipped and drops the ones it took from the arm. Ball size is used as
    a tie-break, so a torso blob sitting on the curve loses to the small round
    thing beside it.
    """
    if len(path) < 5:
        return path
    tol = max(12.0, 0.009 * float(np.hypot(frame_w, frame_h)))
    current = sorted(path, key=lambda p: int(p["frame"]))
    for _ in range(max(1, iterations)):
        ts = np.array([float(p["frame"]) for p in current])
        xs = np.array([float(p["x"]) for p in current])
        ys = np.array([float(p["y"]) for p in current])
        if len(current) < 5 or ts[-1] <= ts[0]:
            return current
        try:
            cx, cy, t0 = _fit_xy(ts, xs, ys)
        except Exception:
            return current
        med_r = float(np.median([float(p.get("r") or 8) for p in current])) or 8.0
        picked: list[dict[str, Any]] = []
        for item in per_frame:
            fr = int(item["frame"])
            t = float(fr) - t0
            px, py = float(np.polyval(cx, t)), float(np.polyval(cy, t))
            best, best_cost = None, 1e18
            for c in item["candidates"]:
                d = float(np.hypot(c["x"] - px, c["y"] - py))
                if d > tol:
                    continue
                # Size consistency matters as much as proximity here: a nearby
                # blob of the wrong size is a different object, and letting it win
                # on distance alone makes the track alternate between the two.
                cost = d + 1.5 * abs(float(c.get("r") or med_r) - med_r)
                if cost < best_cost:
                    best_cost, best = cost, c
            if best is not None:
                picked.append(_point(fr, best))
        if len(picked) < 5:
            return current
        current = _largest_contiguous(picked, max_gap=3)
        if len(current) < 5:
            return sorted(path, key=lambda p: int(p["frame"]))
    return current


def _extend_backward(
    per_frame: list[dict[str, Any]],
    path: list[dict[str, Any]],
    hand_x: float,
    hand_y: float,
    frame_w: int,
    frame_h: int,
    fps: float | None = None,
) -> list[dict[str, Any]]:
    """Prepend earlier detections that sit on the flight's own back-projection.

    Stops at the hand: a candidate within a hand's reach of the wrist is the
    hand (or the ball still in it), not a ball in flight.
    """
    if len(path) < 5:
        return path
    pts = sorted(path, key=lambda p: int(p["frame"]))
    ts = np.array([float(p["frame"]) for p in pts])
    xs = np.array([float(p["x"]) for p in pts])
    ys = np.array([float(p["y"]) for p in pts])
    try:
        cx, cy, t0 = _fit_xy(ts, xs, ys)
    except Exception:
        return path
    tol = max(14.0, 0.010 * float(np.hypot(frame_w, frame_h)))
    hand_reach = max(30.0, 2.0 * float(pts[0].get("r") or 12.0))
    # How far back to look is a duration, not a frame count: 13 frames is 54 ms
    # at 240 fps but 433 ms at 30 fps, and it sets how much leave-hand flight can
    # be recovered — which is what release snapping and release speed use.
    # This runs before the timebase, so `fps` is the container's claim and only a
    # lower bound on the real rate; size the window for the fastest capture that
    # claim could be hiding, or a slow-motion clip gets a window several times
    # too short and loses the frames nearest release.
    if fps and fps > 1:
        back_span = int(np.clip(round(float(fps) * 0.06 * SLOWMO_HEADROOM), 6, 24))
    else:
        back_span = 13
    by_frame = {int(item["frame"]): item["candidates"] for item in per_frame}
    first = int(pts[0]["frame"])
    added: list[dict[str, Any]] = []
    for fr in range(first - 1, first - 1 - back_span, -1):
        cands = by_frame.get(fr)
        if not cands:
            break
        px, py = float(np.polyval(cx, fr - t0)), float(np.polyval(cy, fr - t0))
        if float(np.hypot(px - hand_x, py - hand_y)) < hand_reach:
            break  # back-projection has reached the hand — the ball was still held
        pick, best = None, tol
        for c in cands:
            d = float(np.hypot(c["x"] - px, c["y"] - py))
            if d < best:
                best, pick = d, c
        if pick is None:
            break
        if float(np.hypot(pick["x"] - hand_x, pick["y"] - hand_y)) < hand_reach:
            break
        added.append(_point(fr, pick))
    if not added:
        return pts
    return sorted(added + pts, key=lambda p: int(p["frame"]))


def _point(frame: int, c: dict[str, Any], source: str = "detected") -> dict[str, Any]:
    return {
        "frame": int(frame),
        "x": float(c["x"]),
        "y": float(c["y"]),
        "r": float(c.get("r") or 6),
        "source": source,
    }


def _greedy_chain(
    per_frame: list[dict[str, Any]],
    start_i: int,
    seed: dict[str, Any],
    max_step: float,
    compact_r: float = 36.0,
    cricket_r: float = 10.0,
) -> list[dict[str, Any]]:
    path = [_point(per_frame[start_i]["frame"], seed)]
    vx = vy = 0.0
    have_v = False
    misses = 0
    last_r = float(seed.get("r") or 6)
    last_streak = bool(seed.get("streak"))
    for item in per_frame[start_i + 1 :]:
        df_guess = max(1, int(item["frame"]) - int(path[-1]["frame"]))
        pred = (
            path[-1]["x"] + vx * df_guess,
            path[-1]["y"] + vy * df_guess,
        ) if have_v else (path[-1]["x"], path[-1]["y"])
        picked = None
        best = 1e18
        step_lim = max_step * (1.6 if have_v else 2.4)
        for c in item["candidates"]:
            d = float(np.hypot(c["x"] - pred[0], c["y"] - pred[1]))
            if d > step_lim:
                continue
            r = float(c.get("r") or 6)
            if (
                last_r >= 10
                and not (last_streak or c.get("streak"))
                and (r < last_r * 0.35 or r > last_r * 2.8)
            ):
                continue
            cost = d + 0.15 * abs(r - last_r)
            if cost < best:
                best = cost
                picked = c
        if picked is None:
            misses += 1
            if misses >= 4:
                break
            continue
        # After a stable lock, a pick that stops travelling downrange is turf /
        # bounce / a blurred smear — not the ball. Treating it as a miss cuts
        # the lost-lock tail that otherwise poisons the speed fit.
        if len(path) >= 5:
            step_dx = [
                (float(path[j]["x"]) - float(path[j - 1]["x"]))
                / max(1, int(path[j]["frame"]) - int(path[j - 1]["frame"]))
                for j in range(1, len(path))
            ]
            typical_dx = float(np.median(step_dx[: min(4, len(step_dx))]))
            if abs(typical_dx) >= 8.0:
                df_obs = max(1, int(item["frame"]) - int(path[-1]["frame"]))
                obs_dx = (float(picked["x"]) - float(path[-1]["x"])) / df_obs
                if obs_dx * (1.0 if typical_dx >= 0 else -1.0) < 0.22 * abs(typical_dx):
                    picked = None
        if picked is None:
            misses += 1
            if misses >= 4:
                break
            continue
        misses = 0
        df = max(1, int(item["frame"]) - int(path[-1]["frame"]))
        vx = (picked["x"] - path[-1]["x"]) / df
        vy = (picked["y"] - path[-1]["y"]) / df
        have_v = True
        last_r = float(picked.get("r") or last_r)
        last_streak = bool(picked.get("streak"))
        path.append(_point(item["frame"], picked))
    return path


def _best_flight_path(
    per_frame: list[dict[str, Any]],
    hand_x: float,
    hand_y: float,
    dx: float,
    dy: float,
    fps: float,
    mpp: float | None,
    frame_w: int,
    frame_h: int,
    directional: bool = True,
) -> list[list[dict[str, Any]]]:
    """Pick moving downrange objects — not the hand, not a tree.

    Returns up to four distinct greedy chains, fastest-travel first, so a long
    poster crawl that later fails optical flow does not hide the real flight.
    """
    max_step = _max_step_px(fps, mpp, frame_w, frame_h)
    px = _px_scale(frame_w, frame_h)
    min_dim = float(min(frame_w, frame_h) or 1)
    cricket_r = max(6.0, 0.010 * min_dim)
    # One ball-size scale, not two: this sat two lines from `cricket_r` and was
    # 2.5x it at 720p, so the "tiny" pool meant a different thing per resolution.
    size_r = cricket_r * 1.8
    cell = max(25, int(round(0.013 * float(frame_w))))
    seed_cell = max(40, int(round(0.021 * float(frame_w))))
    min_down = 20.0 * px
    # Match the per-frame hand exclusion. 36 px·scale (~72 px on 4K) skipped
    # the just-released streak that still sits next to the fingers.
    min_dh = 16.0 * px
    min_net = 70.0 * px
    # Looser than flight_geometry_ok — a hybrid ball+poster chain is slower than
    # the ball until ballistic_clean strips the crawl, so the tight step belongs
    # after cleaning, not at seeding.
    min_step = min_ball_step_px(frame_w, frame_h, fps, mpp, fraction=0.5)
    turf_below_hand = 100.0 * px
    f0 = int(per_frame[0]["frame"]) if per_frame else 0
    rel_est = f0 + 8  # track window opens 8 frames before release
    max_above = max(220.0, 0.20 * float(frame_h))
    # Score "in the air" from torso height, not the raised bowling wrist. A
    # leave-hand wrist sits next to the ball; air≈0 then and posters win the
    # seed ranking. A collapsed chest-wrist accidentally boosted the real
    # flight — do that on purpose without moving the hand gates.
    rank_y = max(hand_y, min(frame_h * 0.62, hand_y + 0.14 * float(frame_h)))
    # Seed in *frames*, not seconds of container fps — an early pose-REL sits at
    # cocking, and the white ball only becomes a free blob 8–20 frames later.
    seed_horizon = min(len(per_frame), max(24, int(round(fps * 0.35))))
    raw: list[tuple[float, int, dict[str, Any]]] = []
    for i, item in enumerate(per_frame[:seed_horizon]):
        if int(item["frame"]) < rel_est - 2:
            continue
        elapsed = max(0, int(item["frame"]) - rel_est)
        # How far the ball can have got from the hand after `elapsed` frames is a
        # velocity question, so it must come from the same per-frame step bound
        # the chain uses. A fixed 55 px/frame is below a real 30 fps delivery
        # (which clears ~90 px/frame at 1080p, so the true ball leaves the
        # allowance and can never seed) and far above a 240 fps one (where the
        # allowance opens faster than the ball moves and floods the pool with
        # posters) — it fails at both ends of the input range at once.
        max_seed_dh = max(240.0, 0.14 * float(frame_w)) + max_step * elapsed
        mid = [c for c in item["candidates"] if c["y"] < frame_h * 0.70]
        ranked = sorted(item["candidates"], key=lambda c: c.get("score") or 0, reverse=True)
        compact = sorted(
            [c for c in item["candidates"] if float(c.get("r") or 0) <= cricket_r * 1.6 or c.get("streak")],
            key=lambda c: c.get("compact_score") or c.get("score") or 0,
            reverse=True,
        )
        tiny = sorted(
            [
                c
                for c in item["candidates"]
                if 2.5 <= float(c.get("r") or 0) <= size_r
            ],
            key=lambda c: (
                (1.4 if c.get("source") == "motion" else 1.0)
                * (c.get("compact_score") or c.get("score") or 0)
            ),
            reverse=True,
        )
        pool = ranked[:4] + compact[:4] + tiny[:6] + sorted(mid, key=lambda c: c.get("r") or 0, reverse=True)[:3]
        # Always consider compact movers still near the hand — the 4K/120fps
        # ball is a 5 px speck that loses a global score fight with posters.
        for c in item["candidates"]:
            dh_c = float(np.hypot(c["x"] - hand_x, c["y"] - hand_y))
            r_c = float(c.get("r") or 0)
            if min_dh <= dh_c <= max_seed_dh and (
                r_c <= size_r * 1.5 or c.get("streak") or r_c <= cricket_r * 1.6
            ):
                pool.append(c)
        seen_local = set()
        for c in pool:
            key = (int(c["x"]) // cell, int(c["y"]) // cell)
            if key in seen_local:
                continue
            seen_local.add(key)
            down = (c["x"] - hand_x) * dx + (c["y"] - hand_y) * dy
            if not directional:
                down = min_down  # unknown downrange sign — do not filter on it
            dh = float(np.hypot(c["x"] - hand_x, c["y"] - hand_y))
            turf_y = min(frame_h * 0.92, max(hand_y + turf_below_hand, frame_h * 0.82))
            if down < min_down or c["y"] > turf_y:
                continue
            if dh < min_dh or dh > max_seed_dh:
                continue
            if c["y"] < hand_y - max_above:
                continue
            air = max(0.0, rank_y - c["y"])
            r = float(c.get("r") or 4)
            # Prefer cricket-ball sized movers over poster-sized blobs that happen
            # to sit downrange of the hand. Streaks (120 fps blur) enclose a
            # larger circle than a disc — do not punish them as poster-sized.
            if c.get("streak") or r <= size_r:
                size_fit = 1.0
            else:
                size_fit = max(0.25, size_r / r)
            score = float(c.get("score") or r) * (1.0 + air / (80.0 * px)) * size_fit
            if c.get("streak") or c.get("source") == "motion":
                score *= 1.8
            raw.append((score, i, c))
    raw.sort(key=lambda t: t[0], reverse=True)
    seeds: list[tuple[float, int, dict[str, Any]]] = []
    used = set()
    for item in raw:
        key = (int(item[2]["x"]) // seed_cell, int(item[2]["y"]) // seed_cell)
        if key in used:
            continue
        used.add(key)
        seeds.append(item)
        if len(seeds) >= 24:
            break

    ranked_paths: list[tuple[float, list[dict[str, Any]]]] = []
    for _, i, c in seeds:
        chain = _greedy_chain(
            per_frame, i, c, max_step,
            compact_r=max(24.0, cricket_r * 2.5), cricket_r=cricket_r,
        )
        if len(chain) < 5:
            continue
        net = _dist(chain[0], chain[-1])
        if net < min_net:
            continue
        steps = [
            _dist(a, b) / max(1, int(b["frame"]) - int(a["frame"]))
            for a, b in zip(chain, chain[1:])
        ]
        med_step = float(np.median(steps)) if steps else 0.0
        span = max(1, int(chain[-1]["frame"]) - int(chain[0]["frame"]))
        speed = net / span
        if med_step < min_step or speed < min_step:
            continue
        # Alignment is measured over the first few frames only. `dx, dy` comes from
        # the bowling wrist's own velocity, which at release can point almost
        # straight up; judging a whole arc against it rejects perfectly good
        # flights. Over the arc's end the ball has also begun to fall, which drags
        # the direction further away. Here it only has to not travel backwards.
        head = chain[: min(len(chain), 8)]
        move = (head[-1]["x"] - head[0]["x"], head[-1]["y"] - head[0]["y"])
        mag = float(np.hypot(move[0], move[1])) or 1.0
        align = (move[0] * dx + move[1] * dy) / mag
        if align < -0.20:
            continue
        # Speed × travel, not length: a long poster crawl outranks a short real
        # flight if `len` is in the product (and extra miss-tolerance makes those
        # crawls even longer).
        sc = speed * net * (0.4 + 0.6 * max(0.0, align))
        ranked_paths.append((sc, chain))
    ranked_paths.sort(key=lambda t: t[0], reverse=True)
    out: list[list[dict[str, Any]]] = []
    used_starts: set[tuple[int, int]] = set()
    for _, chain in ranked_paths:
        key = (int(chain[0]["x"]) // seed_cell, int(chain[0]["y"]) // seed_cell)
        if key in used_starts:
            continue
        used_starts.add(key)
        out.append(chain)
        if len(out) >= 4:
            break
    if ranked_paths:
        longest = max(ranked_paths, key=lambda t: len(t[1]))[1]
        if longest not in out:
            out.append(longest)
    return out


def _ballistic_clean(path: list[dict[str, Any]], frame_w: int, frame_h: int) -> list[dict[str, Any]]:
    """Drop points that do not sit on one flight parabola.

    The tolerance comes from the fit's own residuals rather than a fixed number,
    and the fit is repeated once on the survivors. A fixed tolerance has to be
    loose enough for the worst clip, which on a sharp high-frame-rate track is
    wide enough to admit a second blob alongside the ball; the detector then
    alternates between the two and the per-frame speeds swing wildly. Scaling to
    the residuals lets a clean track reject tightly and a noisy one stay
    tolerant.
    """
    if len(path) < 5:
        return path
    current = sorted(path, key=lambda p: int(p["frame"]))
    diag = float(np.hypot(frame_w, frame_h))
    floor, ceiling = max(10.0, 0.006 * diag), max(28.0, 0.018 * diag)
    for _ in range(2):
        ts = np.array([p["frame"] for p in current], dtype=float)
        xs = np.array([p["x"] for p in current], dtype=float)
        ys = np.array([p["y"] for p in current], dtype=float)
        try:
            cx, cy, t0 = _fit_xy(ts, xs, ys)
        except Exception:
            return current
        res = np.array([
            float(np.hypot(p["x"] - np.polyval(cx, p["frame"] - t0),
                           p["y"] - np.polyval(cy, p["frame"] - t0)))
            for p in current
        ])
        mad = float(np.median(np.abs(res - np.median(res))))
        tol = float(np.clip(float(np.median(res)) + 3.0 * 1.4826 * mad, floor, ceiling))
        kept = [p for p, d in zip(current, res) if d <= tol]
        if len(kept) < 5:
            return current
        if len(kept) == len(current):
            return current
        current = kept
    return current


def _blob_centroid(bgr: np.ndarray, x: float, y: float, rad: float) -> tuple[float, float] | None:
    """Sub-pixel centre of the locked blob — dark *or* white cricket ball."""
    h, w = bgr.shape[:2]
    r = int(max(8.0, rad))
    x0, y0 = max(0, int(x) - r), max(0, int(y) - r)
    x1, y1 = min(w, int(x) + r), min(h, int(y) + r)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    gray = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    # White leather is a bright disc; a red/dark ball is the opposite. Otsu-inv
    # on a white ball walks the centroid onto a nearby shadow and kills the lock.
    if float(np.mean(gray)) >= 140:
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    m = cv2.moments(mask)
    if m["m00"] < 16:
        return None
    cx = float(m["m10"] / m["m00"]) + x0
    cy = float(m["m01"] / m["m00"]) + y0
    if float(np.hypot(cx - x, cy - y)) > rad * 1.15:
        return None
    return cx, cy


def _refine_centroids(video_path, path: list[dict[str, Any]], frame_w: int, frame_h: int) -> list[dict[str, Any]]:
    from app.pipeline import extract

    det = {int(p["frame"]): p for p in path if p.get("source") == "detected"}
    if len(det) < 4:
        return path
    f0, f1 = min(det), max(det)
    for idx, bgr in extract.iter_frame_range(video_path, f0, f1):
        p = det.get(idx)
        if p is None:
            continue
        hit = _blob_centroid(bgr, p["x"], p["y"], max(10.0, float(p.get("r") or 10) * 1.6))
        if hit is None:
            continue
        p["x"], p["y"] = float(hit[0]), float(hit[1])
    return path


def _init_tracker(bgr: np.ndarray, point: dict[str, Any]):
    maker = getattr(cv2, "TrackerCSRT_create", None)
    if maker is None:
        legacy = getattr(cv2, "legacy", None)
        maker = getattr(legacy, "TrackerCSRT_create", None) if legacy is not None else None
    if maker is None:
        return None
    r = max(8.0, float(point.get("r") or 6))
    x = int(point["x"] - r)
    y = int(point["y"] - r)
    w = int(r * 2)
    h = int(r * 2)
    ih, iw = bgr.shape[:2]
    x = max(0, min(x, iw - 4))
    y = max(0, min(y, ih - 4))
    w = max(8, min(w, iw - x))
    h = max(8, min(h, ih - y))
    tr = maker()
    try:
        tr.init(bgr, (x, y, w, h))
        return tr
    except Exception:
        return None


def fill_every_frame(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Insert a breakpoint on every missing frame using a ballistic fit."""
    if len(points) < 3:
        return points
    pts = sorted(points, key=lambda p: p["frame"])
    by_f = {int(p["frame"]): p for p in pts}
    f0, f1 = int(pts[0]["frame"]), int(pts[-1]["frame"])
    ts = np.array([p["frame"] for p in pts], dtype=float)
    xs = np.array([p["x"] for p in pts], dtype=float)
    ys = np.array([p["y"] for p in pts], dtype=float)
    try:
        cx, cy, t0 = _fit_xy(ts, xs, ys)
    except Exception:
        return pts
    filled = []
    for f in range(f0, f1 + 1):
        if f in by_f:
            filled.append(by_f[f])
            continue
        # Blend radius from neighbours.
        r0 = float(pts[0].get("r") or 4)
        r1 = float(pts[-1].get("r") or 4)
        t = (f - f0) / max(1, f1 - f0)
        filled.append({
            "frame": f,
            "x": float(np.polyval(cx, f - t0)),
            "y": float(np.polyval(cy, f - t0)),
            "r": float(r0 * (1 - t) + r1 * t),
            "source": "interpolated",
        })
    return filled


def annotate_frame_motion(
    points: list[dict[str, Any]],
    fps: float,
    meters_per_pixel: float | None,
) -> list[dict[str, Any]]:
    """Attach distance and speed between consecutive breakpoints."""
    out: list[dict[str, Any]] = []
    prev = None
    for p in sorted(points, key=lambda d: d["frame"]):
        q = dict(p)
        if prev is not None:
            df = max(1, int(q["frame"]) - int(prev["frame"]))
            dist_px = float(np.hypot(q["x"] - prev["x"], q["y"] - prev["y"]))
            q["dx_px"] = float(q["x"] - prev["x"])
            q["dy_px"] = float(q["y"] - prev["y"])
            q["dist_px"] = dist_px
            q["dt_s"] = df / max(fps, 1e-6)
            q["speed_px_per_frame"] = dist_px / df
            if meters_per_pixel:
                q["dist_m"] = dist_px * float(meters_per_pixel)
                q["speed_mps"] = q["dist_m"] / q["dt_s"]
                q["speed_kmh"] = q["speed_mps"] * 3.6
        out.append(q)
        prev = q
    return out


def ballistic_release_speed_px_per_frame(
    ball_track: list[dict[str, Any]],
    fps: float | None = None,
) -> float | None:
    """Image-plane speed of the ball *at the moment it leaves the hand*.

    Two things make a naive fit under-read badly:

    1. The ball recedes from the camera, so its pixel speed decays along the
       flight — x(t) is not a straight line. Fitting x linearly over half the arc
       and taking the median returns the mid-flight speed, not the release speed.
       Here x and y are both fitted as quadratics and differentiated at the first
       in-air sample.
    2. The window must stay near release. We use the first ~0.12 s of flight,
       where the ball is still close to the bowler's own depth plane.

    The quadratic derivative at an endpoint is noise-sensitive, so it is bounded
    against the raw early per-frame steps and falls back to their median if the
    fit runs away.
    """
    pts = sorted(
        [p for p in ball_track if p.get("source") != "interpolated"],
        key=lambda p: p["frame"],
    )
    if len(pts) < 5:
        pts = sorted(ball_track, key=lambda p: p["frame"])
    if len(pts) < 5:
        return flight_release_speed_px_per_frame(ball_track, fps)

    t0 = int(pts[0]["frame"])
    if fps and fps > 1:
        span = max(6, int(round(float(fps) * 0.12)))
        early = [p for p in pts if int(p["frame"]) - t0 <= span]
    else:
        early = []
    if len(early) < 6:
        early = pts[: max(6, min(12, len(pts)))]
    if len(early) < 5:
        return flight_release_speed_px_per_frame(ball_track, fps)

    ts = np.array([float(p["frame"]) for p in early], dtype=float)
    xs = np.array([float(p["x"]) for p in early], dtype=float)
    ys = np.array([float(p["y"]) for p in early], dtype=float)
    if ts[-1] <= ts[0]:
        return None

    raw_steps = [
        _dist(a, b) / max(1, int(b["frame"]) - int(a["frame"]))
        for a, b in zip(early, early[1:])
    ]
    raw_med = float(np.median(raw_steps)) if raw_steps else None

    deg = 2 if len(early) >= 6 else 1
    try:
        cx = np.polyfit(ts - ts[0], xs, deg)
        cy = np.polyfit(ts - ts[0], ys, deg)
    except Exception:
        return raw_med if raw_med and raw_med > 0.4 else None

    def _slope_at_zero(c: np.ndarray) -> float:
        return float(c[-2]) if len(c) >= 2 else 0.0

    v = float(np.hypot(_slope_at_zero(cx), _slope_at_zero(cy)))
    if raw_med and raw_med > 0:
        # An endpoint derivative can bolt; the raw early steps cannot.
        if not (0.75 * raw_med <= v <= 1.35 * raw_med):
            v = raw_med
    return v if v > 0.4 else None


def flight_release_speed_px_per_frame(
    ball_track: list[dict[str, Any]], fps: float | None = None
) -> float | None:
    """Release speed = robust mean of per-frame distance/time at the start of flight."""
    pts = sorted(ball_track, key=lambda p: p["frame"])
    speeds = [float(p["speed_px_per_frame"]) for p in pts if p.get("speed_px_per_frame")]
    if len(speeds) >= 3:
        # First ~12 intervals after the ball is moving; median kills a single jump.
        # Match ballistic_release_speed_px_per_frame's window: a fixed 12
        # intervals is 50 ms at 240 fps but 400 ms at 30 fps — mid-flight, after
        # the receding ball's pixel speed has already decayed.
        span = max(6, int(round(float(fps) * 0.12))) if fps and fps > 1 else 12
        window = speeds[: min(span, len(speeds))]
        return float(np.median(window))
    if len(pts) < 4:
        return None
    n = min(10, len(pts))
    ts = np.array([p["frame"] for p in pts[:n]], dtype=float)
    xs = np.array([p["x"] for p in pts[:n]], dtype=float)
    ys = np.array([p["y"] for p in pts[:n]], dtype=float)
    if ts[-1] <= ts[0]:
        return None
    try:
        cx = np.polyfit(ts, xs, 1)
        cy = np.polyfit(ts, ys, 1)
    except Exception:
        return None
    v = float(np.hypot(cx[0], cy[0]))
    return v if v > 0.3 else None


def _max_step_px(fps: float, mpp: float | None, frame_w: int, frame_h: int) -> float:
    """Max plausible ball displacement per frame (~50 m/s / 180 km/h)."""
    diag = float(np.hypot(frame_w, frame_h)) or 1.0
    cap = diag * 0.12
    if mpp and mpp > 0 and fps > 1:
        return float(min(cap, 50.0 / (mpp * fps)))
    return float(min(cap, max(18.0, min(frame_w, frame_h) * 0.08)))


def _pick_candidate(
    cands,
    px,
    py,
    search_r,
    max_step,
    hand_xy: tuple[float, float] | None = None,
    min_hand_dist: float = 0.0,
) -> dict[str, Any] | None:
    best = None
    best_cost = 1e9
    for c in cands:
        if float(c.get("r") or 0) > 70:
            continue
        d = float(np.hypot(c["x"] - px, c["y"] - py))
        if d > search_r or d > max_step:
            continue
        if hand_xy is not None and min_hand_dist > 0:
            dh = float(np.hypot(c["x"] - hand_xy[0], c["y"] - hand_xy[1]))
            if dh < min_hand_dist:
                continue
        cost = d / (0.35 + float(c.get("score") or 0.5))
        if cost < best_cost:
            best_cost = cost
            best = c
    return best


# --------------------------------------------------------------------------- #
# Release speed — physics-constrained fit
# --------------------------------------------------------------------------- #

GRAVITY_MPS2 = 9.80665


def _gravity_px_per_frame2(fps: float, mpp: float) -> float:
    """Gravity expressed in this clip's own pixel/frame units.

    Once the frame rate and the metres-per-pixel scale are known, the ball's
    vertical acceleration in the image is not a free parameter — it is fixed by
    physics. Everything below leans on that.
    """
    return GRAVITY_MPS2 / (float(mpp) * float(fps) ** 2)


def robust_release_velocity_px_per_frame(
    ball_track: list[dict[str, Any]],
    fps: float,
    meters_per_pixel: float,
    release_frame: int | None = None,
) -> dict[str, Any] | None:
    """Image-plane velocity at release, fitted with gravity pinned to physics.

    Why not `hypot(dx, dy)` per frame, which is what this replaces: a ball's
    horizontal pixel track is clean, but its *vertical* one is not. At 4K the
    ball is a motion-blurred streak several hundred pixels long, and the
    detector's centroid slides along that streak from frame to frame. That puts
    tens of pixels of noise on `dy` while `dx` stays good to a few pixels — and
    because `hypot` cannot tell the two apart, every y-glitch reads as the ball
    briefly going faster. On the clip this was written for, `dy` jumped
    16.9 -> 86.8 -> 2.9 px between consecutive frames and once went *negative*;
    the frames where it spiked are exactly the frames that reported 130+ km/h
    beside a true speed near 86.

    The fix is to stop letting `y` carry velocity information it does not have:

    * `x(t)` is fitted linearly — it is the trustworthy axis, and over the short
      window used here a receding ball's decay is far below the noise floor.
    * `y(t)` is fitted as `y0 + vy*t + 0.5*g*t**2` with **g pinned** to
      `_gravity_px_per_frame2`, not free. Only `y0` and `vy` are solved for, so
      a y-outlier can shift the fit slightly but can no longer invent
      acceleration — and therefore can no longer invent speed.
    * Points are trimmed iteratively against the fit's own residual spread, so a
      single bad detection is dropped rather than averaged in.

    Returns a dict carrying the fitted velocity *and* the evidence for it
    (residuals, points kept/dropped, a 0-1 quality). Callers are expected to
    refuse a speed whose quality is poor rather than display it — an honest
    "could not measure" beats a confident wrong number.
    """
    if not ball_track or not fps or not meters_per_pixel:
        return None

    # Interpolated points are fabricated by `interpolate_gaps` to keep the drawn
    # path continuous. They are a drawing aid, not observations, and must never
    # be fitted as if they were measurements.
    pts = sorted(
        (p for p in ball_track if p.get("source") != "interpolated"),
        key=lambda p: p["frame"],
    )
    if len(pts) < 5:
        return None

    # Cut a lost-lock tail before choosing the release window. Fitting the
    # downward crawl after x-velocity collapsed is what refused the last
    # three-quarter clip as "not a real flight path".
    pts = _trim_lost_lock(pts)
    if len(pts) < 5:
        return None

    first = int(pts[0]["frame"])
    t0 = first if release_frame is None else int(release_frame)
    # Stay near leave-hand: the ball is still at the bowler's own depth plane
    # there, which is the plane the height scale actually calibrates.
    span = max(6, int(round(float(fps) * 0.12)))
    # Pose REL is often a few frames after the ball is already free. Those
    # earlier detections are the real leave-hand samples.
    pre = max(4, int(round(float(fps) * 0.05)))
    early = [p for p in pts if -pre <= int(p["frame"]) - t0 <= span]
    if len(early) < 5:
        # Detections started after the window. Measuring from release
        # then falls through to the whole remaining path — including a lost-ball
        # tail. The first in-air sample is the nearest depth plane we have.
        t0 = first
        early = [p for p in pts if 0 <= int(p["frame"]) - t0 <= span]
    if len(early) < 5:
        early = pts[: max(5, min(14, len(pts)))]
    if len(early) < 5:
        return None

    g_px = _gravity_px_per_frame2(fps, meters_per_pixel)
    f = np.array([float(p["frame"]) for p in early])
    x = np.array([float(p["x"]) for p in early])
    y = np.array([float(p["y"]) for p in early])
    t = f - f[0]

    def _theil_sen(tt: np.ndarray, vv: np.ndarray) -> float:
        """Median of every pairwise slope.

        Least squares minimises *squared* error, so one detection sitting a
        hundred pixels off the path bends the whole line toward itself — which
        is exactly what happened here before this replaced it. Theil-Sen takes
        the median slope instead: an outlier corrupts only the pairs it appears
        in, and with n points that is n-1 of n(n-1)/2 pairs, so the median never
        moves. It costs an O(n^2) pass over at most ~15 points.
        """
        sl = [
            (vv[j] - vv[i]) / (tt[j] - tt[i])
            for i in range(len(tt))
            for j in range(i + 1, len(tt))
            if tt[j] > tt[i]
        ]
        return float(np.median(sl)) if sl else 0.0

    # y carries a known quadratic. Remove it first and what is left is a
    # straight line whose slope is vertical velocity and nothing else — so a
    # y-outlier can no longer be absorbed as "acceleration", i.e. as speed.
    y_lin = y - 0.5 * g_px * t**2
    vx = _theil_sen(t, x)
    vy = _theil_sen(t, y_lin)

    x0 = float(np.median(x - vx * t))
    y0 = float(np.median(y_lin - vy * t))
    rx = x - (x0 + vx * t)
    ry = y_lin - (y0 + vy * t)
    resid = np.hypot(rx, ry)

    # Drop points the robust line already disagrees with, then re-fit on what
    # survives so the reported velocity reflects only real observations.
    med = float(np.median(resid))
    mad = float(np.median(np.abs(resid - med))) or 1.0
    keep = resid <= max(med + 3.0 * mad, 8.0)
    if keep.sum() >= 4 and not keep.all():
        vx = _theil_sen(t[keep], x[keep])
        vy = _theil_sen(t[keep], y_lin[keep])
        x0 = float(np.median(x[keep] - vx * t[keep]))
        y0 = float(np.median(y_lin[keep] - vy * t[keep]))
        rx = x - (x0 + vx * t)
        ry = y_lin - (y0 + vy * t)
    if keep.sum() < 4:
        keep = np.ones(len(t), dtype=bool)

    x_rms = float(np.sqrt(np.mean(rx[keep] ** 2)))
    y_rms = float(np.sqrt(np.mean(ry[keep] ** 2)))

    speed = float(np.hypot(vx, vy))
    if not np.isfinite(speed) or speed <= 0.4:
        return None

    # Quality: how well the physics-constrained model actually described these
    # points, relative to the distance the ball moves in one frame. A fit whose
    # residuals approach a frame of travel is not measuring the ball.
    resid_rel = float(np.hypot(x_rms, y_rms)) / speed if speed > 0 else 1.0
    quality = max(0.0, min(1.0, 1.0 - resid_rel / 0.35))
    if keep.sum() < 5:
        quality *= 0.6

    return {
        "vx_px_per_frame": vx,
        "vy_px_per_frame": vy,
        "speed_px_per_frame": speed,
        "points_used": int(keep.sum()),
        "points_dropped": int(len(t) - keep.sum()),
        "x_residual_px": x_rms,
        "y_residual_px": y_rms,
        "gravity_px_per_frame2": g_px,
        "quality": quality,
        "first_frame": int(f[0]),
        "last_frame": int(f[-1]),
    }
