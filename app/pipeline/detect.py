"""Foreground-based player/ball detection for MVP.

Strategy: MOG2 background subtraction isolates moving foreground. The player is
the largest tall blob. Ball candidates are small, roughly-circular fast blobs.
We return MULTIPLE ball candidates per frame and let the tracker pick the one
consistent trajectory (this is what removes grass / cloud / camera-shake noise).
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from app.pipeline.cv_vision import enhance_bgr, hough_ball_candidates

# Blob finders were tuned on 720–1080p. On 4K, a 15 px DoG and a 4 px min-radius
# light up poster lettering and net sparkle, and CLAHE on 8 Mpx is too slow to
# finish a delivery scan. Detect at this cap and map coordinates back.
DETECT_MAX_SIDE = 1920


def _cricket_r(h: int, w: int) -> float:
    """Typical in-air cricket-ball radius in *this* image's pixels.

    `0.028 * min(h,w)` was torso/forearm scale on 1080p (~30 px) and on a 4K
    working copy still ~3× a real white ball (~7–12 px). Compact lists then
    filled with arm chunks and the 5 px streak never seeded.
    """
    return max(4.0, min(float(h), float(w)) * 0.010)


def make_background_subtractor() -> Any:
    return cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=32, detectShadows=False)


def foreground_mask(bg, frame: np.ndarray) -> np.ndarray:
    mask = bg.apply(frame)
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def detect_player_bbox(mask: np.ndarray, frame_shape: tuple[int, int]) -> dict[str, Any] | None:
    """Largest reasonably-tall foreground blob = bowler proxy (for scale)."""
    h, w = frame_shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < (w * h) * 0.004:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if bh < bw * 0.9:  # want upright person-like shape
            continue
        if area > best_area:
            best_area = area
            best = {"x": x, "y": y, "w": bw, "h": bh, "cx": x + bw / 2, "cy": y + bh / 2}
    return best


def detect_ball_candidates(
    mask: np.ndarray,
    frame_shape: tuple[int, int],
    player: dict[str, Any] | None,
    max_candidates: int = 12,
) -> list[dict[str, Any]]:
    """Return small round foreground blobs as ball hypotheses (unranked-ish)."""
    h, w = frame_shape
    min_r = max(2.0, min(w, h) * 0.004)
    max_r = min(w, h) * 0.045
    min_area = np.pi * min_r * min_r * 0.4
    max_area = np.pi * max_r * max_r * 2.5

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cands: list[dict[str, Any]] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(c)
        if radius < min_r or radius > max_r:
            continue
        circularity = area / (np.pi * radius * radius + 1e-6)
        if circularity < 0.45:
            continue
        # Prefer compact blobs away from the player's large torso mass
        score = circularity
        if player is not None:
            inside_x = player["x"] - 4 <= cx <= player["x"] + player["w"] + 4
            inside_y = player["y"] - 4 <= cy <= player["y"] + player["h"] + 4
            if inside_x and inside_y:
                score *= 0.6  # de-prioritize but keep (release happens near body)
        cands.append(
            {"x": float(cx), "y": float(cy), "r": float(radius), "score": float(score)}
        )

    cands.sort(key=lambda d: d["score"], reverse=True)
    return cands[:max_candidates]


def working_view(bgr: np.ndarray, max_side: int = DETECT_MAX_SIDE) -> tuple[np.ndarray, float, float]:
    """Downscale a 4K (or similar) frame to the detection working size.

    Returns `(frame, sx, sy)` mapping working-frame pixels → original
    (`x_orig = x_work * sx`). Both scales are `1.0` when the frame is already
    small enough.
    """
    h, w = bgr.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return bgr, 1.0, 1.0
    scale = max_side / float(side)
    nw = max(2, int(round(w * scale)) & ~1)
    nh = max(2, int(round(h * scale)) & ~1)
    work = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    return work, w / float(nw), h / float(nh)


def _scale_candidates(cands: list[dict[str, Any]], sx: float, sy: float) -> list[dict[str, Any]]:
    if sx == 1.0 and sy == 1.0:
        return cands
    sm = (sx + sy) * 0.5
    out: list[dict[str, Any]] = []
    for c in cands:
        d = dict(c)
        d["x"] = float(c["x"]) * sx
        d["y"] = float(c["y"]) * sy
        d["r"] = float(c.get("r") or 0) * sm
        out.append(d)
    return out


def collect_flight_candidates(
    bgr: np.ndarray,
    prev_work_gray: np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """All ball hypotheses for one frame, in *original* pixel coordinates.

    Detection runs on a ≤1920 working copy so 4K night posters do not drown a
    motion-blurred cricket ball. The gray returned is that working copy, to be
    passed back as `prev_work_gray` on the next frame.
    """
    work, sx, sy = working_view(bgr)
    enhanced = enhance_bgr(work)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    dark = detect_dark_flight_candidates(enhanced, enhance=False)
    color = detect_ball_color_candidates(enhanced, enhance=False)
    bright = detect_bright_flight_candidates(enhanced, enhance=False)
    motion: list[dict[str, Any]] = []
    fast: list[dict[str, Any]] = []
    faint: list[dict[str, Any]] = []
    if prev_work_gray is not None:
        cricket_r = _cricket_r(*gray.shape[:2])
        motion = detect_ball_motion_candidates(prev_work_gray, gray, max_r_frac=0.055)
        fast = detect_ball_motion_candidates(
            prev_work_gray, gray, threshold=36, max_r_frac=0.032
        )
        faint = detect_ball_motion_candidates(
            prev_work_gray, gray, threshold=14, max_r_frac=0.028
        )
        for c in motion + fast:
            c["source"] = "motion"
            r = float(c.get("r") or 0)
            # Tiny movers are the 4K/120fps ball; huge body/poster blobs are not.
            if r <= cricket_r * 1.6 or c.get("streak"):
                c["score"] = float(c.get("score") or 0) + 4.0
            else:
                c["score"] = float(c.get("score") or 0) + 0.4
        kept_faint: list[dict[str, Any]] = []
        for c in faint:
            c["source"] = "motion"
            r = float(c.get("r") or 0)
            if c.get("streak") or r <= cricket_r * 1.4:
                c["score"] = float(c.get("score") or 0) + 3.2
                kept_faint.append(c)
        faint = kept_faint
    hough = hough_ball_candidates(enhanced, enhance=False)
    merged = merge_ball_candidates(
        dark, color, bright, motion, fast, faint, hough, cricket_r=_cricket_r(*gray.shape[:2])
    )
    return _scale_candidates(merged, sx, sy), gray


def _blob_candidates(
    mask: np.ndarray,
    origin_xy: tuple[int, int] = (0, 0),
    max_candidates: int = 24,
    *,
    max_r_frac: float = 0.08,
) -> list[dict[str, Any]]:
    h, w = mask.shape[:2]
    min_r = max(2.0, min(w, h) * 0.002)
    max_r = min(w, h) * float(max_r_frac)
    cricket_r = _cricket_r(h, w)  # white/red cricket ball; training ovals stay on the size path
    ox, oy = origin_xy
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cands: list[dict[str, Any]] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 4:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(c)
        if radius < min_r or radius > max_r:
            continue
        circularity = area / (np.pi * radius * radius + 1e-6)
        bx, by, bw, bh = cv2.boundingRect(c)
        thin = float(min(bw, bh))
        long = float(max(bw, bh))
        aspect = long / max(thin, 1.0)
        # A 120 fps white ball is a streak, not a disc — especially on a
        # three-quarter / blurry angle where the ball smears along the path.
        streak = (
            aspect >= 1.75
            and thin <= cricket_r * 1.85
            and long <= min(w, h) * 0.18
            and circularity >= 0.04
        )
        if circularity < 0.08 and not streak:
            continue
        # Size-favoring score keeps large training ovals; compact score keeps a
        # 4–16 px cricket ball that posters and floodlight flares would drown.
        size_score = float(radius * (0.25 + circularity))
        compact_score = float(circularity * (1.2 + cricket_r / max(radius, 2.0)))
        if streak:
            compact_score = max(compact_score, 1.4 + thin / max(cricket_r, 1.0))
        cands.append({
            "x": float(cx + ox),
            "y": float(cy + oy),
            "r": float(radius),
            "score": max(size_score, compact_score),
            "size_score": size_score,
            "compact_score": compact_score,
            "streak": streak,
        })
    if not cands:
        return []
    keep: dict[tuple[int, int], dict[str, Any]] = {}
    by_size = sorted(cands, key=lambda d: d["size_score"], reverse=True)
    by_compact = sorted(
        [c for c in cands if c["r"] <= cricket_r * 1.6 or c.get("streak")],
        key=lambda d: d["compact_score"],
        reverse=True,
    )
    for c in by_size[: max(8, max_candidates // 2)]:
        keep[(int(c["x"]), int(c["y"]))] = c
    for c in by_compact[: max(8, max_candidates // 2)]:
        keep[(int(c["x"]), int(c["y"]))] = c
    return list(keep.values())[: max_candidates]


def detect_dark_flight_candidates(
    bgr: np.ndarray, *, mask_ground: bool = True, enhance: bool = True
) -> list[dict[str, Any]]:
    """Dark ball against sky/background (red cricket ball or dark football)."""
    if enhance:
        bgr = enhance_bgr(bgr)
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    dark = ((hsv[:, :, 2] < 125) & (gray < 135)).astype(np.uint8) * 255
    # Turf only — track.py still does the hand-relative cut. 0.62 of a 4K
    # frame (or a 1080p bowler in the lower half) deleted the in-air ball.
    if mask_ground:
        dark[int(h * 0.88) :, :] = 0
    kernel = np.ones((3, 3), np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel, iterations=1)
    return _blob_candidates(dark, max_candidates=24)


def detect_ball_color_candidates(
    bgr: np.ndarray, *, enhance: bool = True, mask_edges: bool = True
) -> list[dict[str, Any]]:
    """Red or white cricket-ball coloured blobs.

    White leather is often V~150–180 (not a clipped 185 highlight), and floodlight
    flares / posters are huge — cap radius at cricket-ball scale so they cannot
    crowd the candidate list. Dark training ovals still come from the dark detector.
    """
    if enhance:
        bgr = enhance_bgr(bgr)
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    red = cv2.bitwise_or(
        cv2.inRange(hsv, (0, 70, 50), (12, 255, 255)),
        cv2.inRange(hsv, (168, 70, 50), (180, 255, 255)),
    )
    white = cv2.inRange(hsv, (0, 0, 132), (180, 100, 255))
    mask = cv2.bitwise_or(red, white)
    # Floodlight streaks live in the top of night clips. Skipped on ROI crops,
    # where "the top of the frame" is not the top of the scene.
    if mask_edges:
        mask[: int(h * 0.08), :] = 0
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return _blob_candidates(mask, max_r_frac=0.038)


def detect_bright_flight_candidates(
    bgr: np.ndarray, *, mask_edges: bool = True, enhance: bool = True
) -> list[dict[str, Any]]:
    """White cricket ball as a small local brightness peak (night turf / grey wall).

    HSV-white alone matches shoes, stumps and poster lettering. A ball in flight
    is also brighter than its immediate surround — DoG keeps those discs and
    drops extended glare.
    """
    if enhance:
        bgr = enhance_bgr(bgr)
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    white = cv2.inRange(hsv, (0, 0, 132), (180, 100, 255))
    k = int(round(min(h, w) * 0.014))
    if k % 2 == 0:
        k += 1
    k = int(np.clip(k, 9, 31))
    surround = cv2.GaussianBlur(gray, (k, k), 0)
    peak = cv2.subtract(gray, surround)
    _, peak_m = cv2.threshold(peak, 10, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_and(white, peak_m)
    if mask_edges:
        mask[: int(h * 0.08), :] = 0
        mask[int(h * 0.90) :, :] = 0  # planted white ball from a previous delivery
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return _blob_candidates(mask, max_candidates=16, max_r_frac=0.032)


def detect_ball_motion_candidates(
    prev_gray: np.ndarray,
    gray: np.ndarray,
    *,
    threshold: int = 22,
    max_r_frac: float = 0.045,
) -> list[dict[str, Any]]:
    """Small moving blobs from frame-to-frame difference.

    Cap radius so the bowler silhouette cannot occupy the whole candidate list
    and hide a 4–8 px cricket ball.
    """
    diff = cv2.absdiff(gray, prev_gray)
    _, mask = cv2.threshold(diff, int(threshold), 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return _blob_candidates(mask, max_candidates=32, max_r_frac=max_r_frac)


def detect_ball_in_roi(
    bgr: np.ndarray,
    prev_gray: np.ndarray | None,
    cx: float,
    cy: float,
    radius: float,
) -> list[dict[str, Any]]:
    """Ball hypotheses in a window around the predicted position."""
    h, w = bgr.shape[:2]
    r = int(max(24, radius))
    x0 = max(0, int(cx) - r)
    y0 = max(0, int(cy) - r)
    x1 = min(w, int(cx) + r)
    y1 = min(h, int(cy) + r)
    if x1 - x0 < 12 or y1 - y0 < 12:
        return []
    crop = bgr[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    color = detect_ball_color_candidates(crop, mask_edges=False)
    for c in color:
        c["x"] += x0
        c["y"] += y0
    dark = detect_dark_flight_candidates(crop, mask_ground=False)
    for c in dark:
        c["x"] += x0
        c["y"] += y0
    bright = detect_bright_flight_candidates(crop, mask_edges=False)
    for c in bright:
        c["x"] += x0
        c["y"] += y0
    motion: list[dict[str, Any]] = []
    if prev_gray is not None:
        prev_crop = prev_gray[y0:y1, x0:x1]
        if prev_crop.shape[:2] == gray.shape[:2]:
            motion = detect_ball_motion_candidates(prev_crop, gray)
            for c in motion:
                c["x"] += x0
                c["y"] += y0
    return merge_ball_candidates(
        color, dark, bright, motion, cricket_r=_cricket_r(*gray.shape[:2])
    )


def merge_ball_candidates(
    *groups: list[dict[str, Any]],
    merge_px: float | None = None,
    per_group: int = 12,
    total: int = 48,
    compact_r: float | None = None,
    cricket_r: float | None = None,
) -> list[dict[str, Any]]:
    """Spatial merge with a per-source quota.

    Colour posters score higher than a small white ball (radius-weighted), so
    taking the global top-N dropped the ball. Each detector keeps its own best
    hits *and* its compact/streak hits, then we collapse duplicates.

    Every size decision here is a multiple of `cricket_r`, the expected in-air
    ball radius for this image. As raw pixels they meant different things to the
    two callers — `collect_flight_candidates` works on a downscaled view while
    `detect_ball_in_roi` works on a full-resolution crop — so the same "14 px"
    was 1.3x a ball in one and 0.7x in the other on 4K input.
    """
    r_c = float(cricket_r) if cricket_r else 10.0
    if merge_px is None:
        merge_px = max(4.0, 0.8 * r_c)
    if compact_r is None:
        compact_r = max(6.0, 1.4 * r_c)
    merged: list[dict[str, Any]] = []

    def _absorb(c: dict[str, Any]) -> None:
        hit = None
        for m in merged:
            if np.hypot(c["x"] - m["x"], c["y"] - m["y"]) <= merge_px:
                hit = m
                break
        if hit is None:
            merged.append(dict(c))
            return
        # A later compact/motion hit is the cricket ball; keep its centroid,
        # not the first poster sparkle that happened to sit 8 px away.
        c_compact = float(c.get("compact_score") or 0)
        h_compact = float(hit.get("compact_score") or 0)
        prefer = c.get("source") == "motion" and hit.get("source") != "motion"
        if prefer or c_compact > h_compact:
            hit["x"] = float(c["x"])
            hit["y"] = float(c["y"])
            hit["r"] = float(c.get("r") or hit.get("r") or 0)
            if c.get("source"):
                hit["source"] = c["source"]
            if c_compact:
                hit["compact_score"] = c_compact
        hit["score"] = max(float(hit.get("score") or 0), float(c.get("score") or 0))
        if c.get("streak"):
            hit["streak"] = True

    for group in groups:
        ranked = sorted(group, key=lambda d: d.get("score") or 0, reverse=True)[:per_group]
        compact = sorted(
            [
                c
                for c in group
                if float(c.get("r") or 0) <= compact_r or c.get("streak")
            ],
            key=lambda d: d.get("compact_score") or d.get("score") or 0,
            reverse=True,
        )[:per_group]
        for c in ranked + compact:
            _absorb(c)
    compact_kept = [
        m for m in merged if float(m.get("r") or 0) <= compact_r or m.get("streak")
    ]
    bulky = [
        m
        for m in merged
        if float(m.get("r") or 0) > compact_r and not m.get("streak")
    ]
    bulky.sort(key=lambda d: d.get("score") or 0, reverse=True)
    motion_c = [m for m in compact_kept if m.get("source") == "motion"]
    other_c = [m for m in compact_kept if m.get("source") != "motion"]
    # Specks below half a ball radius are net sparkle at any resolution — but at
    # high frame rates the real ball *is* nearly that small, so keep a few of
    # them rather than letting them occupy every compact slot.
    tiny_cut = 0.45 * r_c
    tiny_m = [m for m in motion_c if float(m.get("r") or 0) < tiny_cut]
    rest_m = [m for m in motion_c if float(m.get("r") or 0) >= tiny_cut]
    other_mid = [m for m in other_c if float(m.get("r") or 0) >= 0.55 * r_c]
    tiny_m.sort(key=lambda d: d.get("compact_score") or d.get("score") or 0, reverse=True)
    rest_m.sort(key=lambda d: d.get("score") or 0, reverse=True)
    other_mid.sort(key=lambda d: d.get("score") or 0, reverse=True)
    compact_out = rest_m[:10] + tiny_m[:8] + other_mid[:10]
    n_b = max(0, total - len(compact_out))
    return compact_out + bulky[:n_b]
