"""Overlay video, per-ball clips, and a pitch-map PNG."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from app.balltrack.calibrate import STUMP_WIDTH_M, pitch_to_image


def _metric_ok(metric: dict[str, Any] | None) -> bool:
    if not metric:
        return False
    if metric.get("value") is None:
        return False
    status = metric.get("status")
    return status in (None, "ok")


def _draw_hud(frame: np.ndarray, delivery: dict[str, Any] | None) -> None:
    if not delivery:
        return
    h, w = frame.shape[:2]
    speed = delivery.get("speed_kmh") or {}
    length = delivery.get("length_m") or {}
    line = delivery.get("line_m") or {}
    chips: list[tuple[str, str]] = []
    if _metric_ok(speed):
        chips.append(("SPEED", f"{float(speed['value']):.0f} km/h"))
    if _metric_ok(length):
        chips.append(("LENGTH", f"{float(length['value']):.1f} m"))
    if _metric_ok(line):
        chips.append(("LINE", f"{float(line['value']):+.2f} m"))
    if not chips:
        return
    x0, y0 = int(w * 0.04), int(h * 0.04)
    pad = max(8, w // 80)
    for i, (label, value) in enumerate(chips):
        y = y0 + i * int(h * 0.075)
        text = f"{label}  {value}"
        scale = max(0.45, w / 900)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        x1, y1 = x0 + tw + pad * 2, y + th + pad * 2
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y), (x1, y1), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        cv2.putText(frame, text, (x0 + pad, y1 - pad), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)


def _draw_scene(
    frame: np.ndarray,
    *,
    pitch: np.ndarray | None,
    bowler_box: dict[str, Any] | None,
    batter_box: dict[str, Any] | None,
    trails: dict[int, list[tuple[int, int]]],
    hits: list[tuple[int, dict[str, Any]]],
    hud_delivery: dict[str, Any] | None,
) -> None:
    if pitch is not None:
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pitch], (210, 90, 30))
        cv2.addWeighted(overlay, 0.32, frame, 0.68, 0, frame)
        cv2.polylines(frame, [pitch], True, (255, 170, 50), 3, cv2.LINE_AA)
    _draw_wicket_set(frame, bowler_box)
    _draw_wicket_set(frame, batter_box)
    for i, p in hits:
        trails[i].append((int(p["x"]), int(p["y"])))
        cv2.circle(frame, (int(p["x"]), int(p["y"])), 7, (0, 0, 230), -1)
    for pts in trails.values():
        if len(pts) > 1:
            cv2.polylines(frame, [np.array(pts, dtype=np.int32)], False, (0, 0, 255), 4, cv2.LINE_AA)
    _draw_hud(frame, hud_delivery)


def _pitch_quad(H_inv: np.ndarray | None, pitch_length_m: float, stump_width_m: float) -> np.ndarray | None:
    if H_inv is None:
        return None
    corners = [
        pitch_to_image(H_inv, 0.0, 0.0),
        pitch_to_image(H_inv, 0.0, stump_width_m),
        pitch_to_image(H_inv, pitch_length_m, stump_width_m),
        pitch_to_image(H_inv, pitch_length_m, 0.0),
    ]
    if any(not np.isfinite(x) or not np.isfinite(y) for x, y in corners):
        return None
    return np.array(corners, dtype=np.int32)


def _draw_wicket_set(frame: np.ndarray, box: dict[str, Any] | None) -> None:
    if not box:
        return
    h, w = frame.shape[:2]
    x = float(box["x"]) * w
    y = float(box["y"]) * h
    bw = float(box["w"]) * w
    bh = float(box["h"]) * h
    base = y + bh
    post_w = max(3.0, bw * 0.17)
    gap = (bw - post_w * 3) / 2.0
    bail_h = max(3.0, bh * 0.08)
    post_h = max(10.0, bh - bail_h)
    top = base - post_h
    gold = (80, 200, 245)
    edge = (20, 110, 180)
    for i in range(3):
        left = x + i * (post_w + gap)
        top_w = post_w * 0.78
        inset = (post_w - top_w) / 2.0
        pts = np.array(
            [
                [left, base],
                [left + post_w, base],
                [left + post_w - inset, top],
                [left + inset, top],
            ],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(frame, pts, gold)
        cv2.polylines(frame, [pts], True, edge, 1, cv2.LINE_AA)
    cv2.rectangle(
        frame,
        (int(x), int(top - bail_h * 0.2)),
        (int(x + bw), int(top + bail_h * 0.55)),
        gold,
        -1,
        cv2.LINE_AA,
    )


def write_overlay(
    video_path: Path,
    out_path: Path,
    deliveries: list[dict[str, Any]],
    fps: float,
    max_seconds: float = 180.0,
    H_inv: np.ndarray | None = None,
    pitch_length_m: float = 20.12,
    pitch_width_m: float = 3.05,
    stump_width_m: float = STUMP_WIDTH_M,
    bowler_box: dict[str, Any] | None = None,
    batter_box: dict[str, Any] | None = None,
) -> Path:
    cap = cv2.VideoCapture(str(video_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    rate = fps if fps > 1 else 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, rate, (w, h))
    by_frame: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    for i, d in enumerate(deliveries):
        for p in d.get("track") or []:
            by_frame.setdefault(int(p["frame"]), []).append((i, p))
    idx = 0
    limit = int(max_seconds * rate)
    trails: dict[int, list[tuple[int, int]]] = {i: [] for i in range(len(deliveries))}
    pitch = _pitch_quad(H_inv, pitch_length_m, stump_width_m)
    while idx < limit:
        ok, frame = cap.read()
        if not ok:
            break
        active = None
        for d in deliveries:
            if int(d.get("start_frame") or 0) <= idx <= int(d.get("end_frame") or 0):
                active = d
                break
        _draw_scene(
            frame,
            pitch=pitch,
            bowler_box=bowler_box,
            batter_box=batter_box,
            trails=trails,
            hits=by_frame.get(idx, []),
            hud_delivery=active,
        )
        writer.write(frame)
        idx += 1
    cap.release()
    writer.release()
    return out_path


def write_clip(
    video_path: Path,
    out_path: Path,
    start_frame: int,
    end_frame: int,
    fps: float,
    delivery: dict[str, Any] | None = None,
    H_inv: np.ndarray | None = None,
    pitch_length_m: float = 20.12,
    stump_width_m: float = STUMP_WIDTH_M,
    bowler_box: dict[str, Any] | None = None,
    batter_box: dict[str, Any] | None = None,
) -> Path:
    cap = cv2.VideoCapture(str(video_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    rate = fps if fps > 1 else 30.0
    pad = int(rate * 0.25)
    a = max(0, start_frame - pad)
    b = end_frame + pad
    cap.set(cv2.CAP_PROP_POS_FRAMES, a)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, rate, (w, h))
    pitch = _pitch_quad(H_inv, pitch_length_m, stump_width_m)
    trails: dict[int, list[tuple[int, int]]] = {0: []}
    track = {int(p["frame"]): p for p in (delivery or {}).get("track") or []}
    for fr in range(a, b + 1):
        ok, frame = cap.read()
        if not ok:
            break
        hits = [(0, track[fr])] if fr in track else []
        _draw_scene(
            frame,
            pitch=pitch,
            bowler_box=bowler_box,
            batter_box=batter_box,
            trails=trails,
            hits=hits,
            hud_delivery=delivery if start_frame <= fr <= end_frame else None,
        )
        writer.write(frame)
    cap.release()
    writer.release()
    return out_path


def write_pitch_map(
    out_path: Path,
    deliveries: list[dict[str, Any]],
    pitch_length_m: float,
    pitch_width_m: float,
) -> Path:
    W, H = 420, 760
    img = Image.new("RGB", (W, H), (11, 61, 46))
    draw = ImageDraw.Draw(img)
    pad = 36
    field = [pad, pad, W - pad, H - pad]
    draw.rectangle(field, fill=(18, 92, 58), outline=(232, 245, 233), width=2)
    # Creases
    def y_at(metres: float) -> float:
        t = metres / max(pitch_length_m, 0.01)
        return field[3] - t * (field[3] - field[1])

    draw.line([(field[0], y_at(1.22)), (field[2], y_at(1.22))], fill=(251, 239, 231), width=2)
    draw.line([(field[0], y_at(pitch_length_m - 1.22)), (field[2], y_at(pitch_length_m - 1.22))], fill=(251, 239, 231), width=2)
    cx = (field[0] + field[2]) / 2
    for i, d in enumerate(deliveries):
        b = d.get("bounce") or {}
        if b.get("length_m") is None:
            continue
        x = cx + (float(b.get("width_m") if b.get("width_m") is not None else STUMP_WIDTH_M / 2) - STUMP_WIDTH_M / 2) / max(pitch_width_m, 0.01) * (field[2] - field[0]) * 0.85
        y = y_at(float(b["length_m"]))
        r = 7
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(228, 87, 46), outline=(255, 255, 255))
        draw.text((x + 8, y - 8), str(i + 1), fill=(255, 255, 255))
    draw.text((pad, 8), "Batter", fill=(232, 245, 233))
    draw.text((pad, H - 24), "Bowler", fill=(232, 245, 233))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "PNG")
    return out_path
