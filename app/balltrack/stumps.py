"""Find two cricket wicket sets in a behind-the-arm still (lab OpenCV, not on-device)."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from app.balltrack.calibrate import homography_from_boxes


def detect_stump_sets(
    bgr: np.ndarray,
    hints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if bgr is None or bgr.size == 0:
        raise ValueError("Empty frame")
    src_h, src_w = bgr.shape[:2]
    scale = min(1.0, 720.0 / max(src_w, src_h))
    small = cv2.resize(bgr, (int(src_w * scale), int(src_h * scale))) if scale < 0.999 else bgr
    mask = _stump_mask(small)
    h, w = small.shape[:2]

    near_hint = (hints or {}).get("bowler") or {"x": 0.22, "y": 0.62, "w": 0.56, "h": 0.30}
    far_hint = (hints or {}).get("batter") or {"x": 0.36, "y": 0.05, "w": 0.28, "h": 0.22}

    near = _wicket_in_hint(mask, near_hint, kind="near")
    far = _wicket_in_hint(mask, far_hint, kind="far")

    if near is None or far is None:
        global_hits = _wickets_global(mask)
        if near is None:
            near = _pick_kind(global_hits, "near", h)
        if far is None:
            far = _pick_kind(global_hits, "far", h)

    if near is None or far is None:
        raise ValueError(
            "Could not see both stump sets. Put the near wickets in the bottom box and the far wickets in the top box, then try again."
        )

    if far["cy"] >= near["cy"] - h * 0.08:
        raise ValueError("The two wicket sets overlap in the frame. Step back so both ends of the pitch are visible.")

    bowler = _to_norm(near, w, h)
    batter = _to_norm(far, w, h)
    cal = homography_from_boxes(bowler, batter, src_w, src_h)
    return {
        "bowler": bowler,
        "batter": batter,
        "confidence": {
            "bowler": round(float(near["score"]), 3),
            "batter": round(float(far["score"]), 3),
        },
        "pitch_length_m": cal["pitch_length_m"],
        "found": True,
    }


def _stump_mask(bgr: np.ndarray) -> np.ndarray:
    h, _w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    grass = cv2.inRange(hsv, (32, 35, 35), (95, 255, 255))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    dark = cv2.inRange(hsv, (0, 0, 0), (180, 90, 95))
    wood = cv2.inRange(hsv, (8, 25, 90), (40, 180, 230))
    pale = cv2.inRange(hsv, (0, 0, 150), (180, 55, 255))
    color = cv2.bitwise_or(dark, cv2.bitwise_or(wood, pale))
    color = cv2.bitwise_and(color, cv2.bitwise_not(grass))

    sobel = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    vert_e = cv2.convertScaleAbs(sobel)
    _, edges = cv2.threshold(vert_e, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, max(11, h // 28)))
    posts = cv2.morphologyEx(cv2.bitwise_or(color, edges), cv2.MORPH_OPEN, k, iterations=1)
    posts = cv2.morphologyEx(posts, cv2.MORPH_CLOSE, k, iterations=1)
    posts = cv2.bitwise_and(posts, cv2.bitwise_not(grass))
    return posts


def _wicket_in_hint(mask: np.ndarray, box: dict[str, Any], kind: str) -> dict[str, Any] | None:
    h, w = mask.shape[:2]
    pad = 0.45 if kind == "far" else 0.28
    x0 = int(max(0, (float(box["x"]) - pad * float(box["w"])) * w))
    y0 = int(max(0, (float(box["y"]) - pad * float(box["h"])) * h))
    x1 = int(min(w, (float(box["x"]) + float(box["w"]) * (1 + pad)) * w))
    y1 = int(min(h, (float(box["y"]) + float(box["h"]) * (1 + pad)) * h))
    if x1 - x0 < 12 or y1 - y0 < 12:
        return None
    roi = mask[y0:y1, x0:x1]
    hit = _wicket_from_mask(roi, kind)
    if hit is None:
        return None
    hit["x"] += x0
    hit["y"] += y0
    hit["cx"] += x0
    hit["cy"] += y0
    return hit


def _wickets_global(mask: np.ndarray) -> list[dict[str, Any]]:
    h, w = mask.shape[:2]
    hits: list[dict[str, Any]] = []
    for kind, y0n, y1n in (("near", 0.52, 1.0), ("far", 0.0, 0.48)):
        y0, y1 = int(h * y0n), int(h * y1n)
        x0, x1 = int(w * 0.12), int(w * 0.88)
        roi = mask[y0:y1, x0:x1]
        hit = _wicket_from_mask(roi, kind)
        if hit:
            hit["x"] += x0
            hit["y"] += y0
            hit["cx"] += x0
            hit["cy"] += y0
            hits.append(hit)
    return hits


def _pick_kind(hits: list[dict[str, Any]], kind: str, frame_h: int) -> dict[str, Any] | None:
    if not hits:
        return None
    if kind == "near":
        lower = [h for h in hits if h["cy"] > frame_h * 0.48]
        pool = lower or hits
        return max(pool, key=lambda d: d["h"] * d["w"] * d["score"])
    upper = [item for item in hits if item["cy"] < frame_h * 0.5]
    pool = upper or hits
    return max(pool, key=lambda d: d["score"])


def _wicket_from_mask(roi: np.ndarray, kind: str) -> dict[str, Any] | None:
    if roi.size == 0:
        return None
    rh, rw = roi.shape[:2]
    col = roi.sum(axis=0).astype(np.float64)
    if col.max() < rh * 4:
        return None
    win = max(3, rw // 40)
    kernel = np.ones(win, dtype=np.float64) / win
    smooth = np.convolve(col, kernel, mode="same")
    peaks = _peaks(smooth, min_prom=smooth.max() * (0.18 if kind == "near" else 0.12))
    rows = roi.sum(axis=1).astype(np.float64)
    active = np.where(rows > rows.max() * 0.12)[0]
    if active.size < 4:
        return None
    y0, y1 = int(active[0]), int(active[-1])
    height = y1 - y0 + 1

    if 2 <= len(peaks) <= 5:
        xs = np.array(peaks, dtype=np.float64)
        span = float(xs.max() - xs.min())
        if span < 6:
            return None
        gaps = np.diff(np.sort(xs))
        if gaps.size and (gaps.max() / max(gaps.min(), 1.0)) > 4.5:
            return None
        x0 = int(max(0, xs.min() - span * 0.18))
        x1 = int(min(rw - 1, xs.max() + span * 0.18))
        n = len(peaks)
        score = 0.45 + 0.12 * n + 0.2 * min(1.0, height / max(rh * 0.25, 1))
    else:
        nz = np.where(smooth > smooth.max() * 0.28)[0]
        if nz.size < 4:
            return None
        x0, x1 = int(nz[0]), int(nz[-1])
        span = x1 - x0
        if span < 4 or span > rw * 0.85:
            return None
        aspect = height / max(span, 1)
        if kind == "far" and not (0.7 <= aspect <= 8.0):
            return None
        if kind == "near" and not (0.45 <= aspect <= 6.0):
            return None
        score = 0.32 + 0.15 * min(1.0, height / max(rh * 0.3, 1))

    pad_y = int(height * 0.08)
    y0 = max(0, y0 - pad_y)
    y1 = min(rh - 1, y1 + int(height * 0.04))
    box_w = max(8, x1 - x0)
    box_h = max(8, y1 - y0)
    return {
        "x": x0,
        "y": y0,
        "w": box_w,
        "h": box_h,
        "cx": x0 + box_w / 2,
        "cy": y0 + box_h / 2,
        "score": float(min(0.95, score)),
    }


def _peaks(signal: np.ndarray, min_prom: float) -> list[int]:
    out: list[int] = []
    n = len(signal)
    if n < 5:
        return out
    min_dist = max(4, n // 28)
    for i in range(2, n - 2):
        if signal[i] < min_prom:
            continue
        if signal[i] >= signal[i - 1] and signal[i] >= signal[i + 1] and signal[i] >= signal[i - 2]:
            if out and i - out[-1] < min_dist:
                if signal[i] > signal[out[-1]]:
                    out[-1] = i
                continue
            out.append(i)
    return out


def _to_norm(hit: dict[str, Any], w: int, h: int) -> dict[str, float]:
    return {
        "x": round(float(hit["x"]) / w, 4),
        "y": round(float(hit["y"]) / h, 4),
        "w": round(float(hit["w"]) / w, 4),
        "h": round(float(hit["h"]) / h, 4),
    }
