"""Camera-view quality — used to refuse speeds that would be false.

A single camera cannot see motion toward/away from the lens. Front-on and
steep three-quarter footage therefore cannot yield a truthful km/h. We
classify the view from pose (apparent shoulder width vs stature) and from
how much the bowling wrist travels sideways in the image.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.pipeline import pose as posemod
from app.pipeline.action import frame_by_index

# Apparent shoulder breadth / nose–ankle height.
# True side-on stacks the shoulders (~0.06–0.13). Front-on shows full breadth (~0.23+).
# Open-chested bowling actions sit ~0.18–0.22 and still show the arm wheel in plane.
FRONT_ON_RATIO = 0.23
THREE_QUARTER_RATIO = 0.145

# Slowest thing in the world we would still call a delivery. Anything below this
# is a walked-in throw or a false lock, whatever the camera.
MIN_BALL_MPS = 8.0
# A container frame rate is a *lower* bound on the capture rate: a slow-motion
# export understates it, never overstates it. When converting a real-world speed
# floor into pixels per frame we therefore assume the clip could be this many
# times slower than it claims, which makes the floor conservative — it will not
# reject a genuine flight just because the file lies about its rate.
SLOWMO_HEADROOM = 8.0


def min_ball_step_px(
    frame_w: int,
    frame_h: int,
    fps: float | None = None,
    meters_per_pixel: float | None = None,
    *,
    fraction: float = 1.0,
    rate_is_certain: bool = False,
) -> float:
    """Slowest image motion, in px per frame, that could still be a bowled ball.

    Pixels per frame is not a speed: it falls with the capture rate and rises
    with how much of the frame the bowler fills. A fixed px/frame floor — even
    one scaled by frame width — therefore means a different real-world speed on
    every clip, and on 4K high-frame-rate footage it lands *above* a genuine
    delivery and rejects it. Convert a real speed instead, whenever we have the
    scale to do so, and fall back to a small fraction of the diagonal only when
    we do not.
    """
    if meters_per_pixel and meters_per_pixel > 0 and fps and fps > 1:
        # Headroom only while the rate is still the container's claim. Once the
        # timebase has recovered the real capture rate the floor can be exact,
        # and it needs to be: with headroom it drops to ~1 px/frame and stops
        # rejecting the slow background crawls it exists to catch.
        headroom = 1.0 if rate_is_certain else SLOWMO_HEADROOM
        px = (MIN_BALL_MPS * fraction) / (
            float(meters_per_pixel) * float(fps) * headroom
        )
        return max(1.0, float(px))
    diag = float(np.hypot(frame_w or 1280, frame_h or 720))
    return max(1.5, 0.0025 * diag * fraction)


def classify_camera_view(
    pose_track: dict[str, Any],
    action: dict[str, Any],
) -> dict[str, Any]:
    """Return view class and whether image-plane km/h would be a false number."""
    frames = pose_track.get("frames") or []
    fps = float(pose_track.get("fps") or 30.0)
    side = action.get("throwing_side")
    release = action.get("release_frame")
    result: dict[str, Any] = {
        "view": "unknown",
        "shoulder_width_ratio": None,
        "wrist_lateral_ratio": None,
        "speed_ok": False,
        "stride_ok": False,
        "note": "Could not classify camera angle from pose",
    }
    if not frames or side is None or release is None:
        return result

    ratios: list[float] = []
    span = max(2, int(round(fps * 0.12)))
    for f in frames:
        if abs(int(f["frame"]) - int(release)) > span:
            continue
        lsh = posemod.point(f, "left_shoulder")
        rsh = posemod.point(f, "right_shoulder")
        bh = posemod.body_pixel_height(f)
        if lsh is None or rsh is None or not bh:
            continue
        ratios.append(float(np.hypot(lsh[0] - rsh[0], lsh[1] - rsh[1])) / float(bh))
    shoulder_ratio = float(np.median(ratios)) if ratios else None

    wrist_lateral = _wrist_lateral_ratio(frames, side, int(release), fps)

    if shoulder_ratio is None:
        result["wrist_lateral_ratio"] = wrist_lateral
        if wrist_lateral is not None and wrist_lateral < 0.28:
            result.update(
                view="front_on",
                speed_ok=False,
                stride_ok=False,
                note="Bowling arm moves mostly toward the camera — km/h from this clip would be false",
            )
            return result
        result["note"] = "Shoulders not visible enough to classify camera angle"
        return result

    result["shoulder_width_ratio"] = round(shoulder_ratio, 3)
    result["wrist_lateral_ratio"] = None if wrist_lateral is None else round(wrist_lateral, 3)

    frontish_wrist = wrist_lateral is not None and wrist_lateral < 0.30

    if shoulder_ratio >= FRONT_ON_RATIO or (shoulder_ratio >= 0.17 and frontish_wrist):
        result.update(
            view="front_on",
            speed_ok=False,
            stride_ok=False,
            note="Front-on camera — image-plane km/h and stride would be false. Film side-on.",
        )
        return result

    if shoulder_ratio >= THREE_QUARTER_RATIO:
        # Three-quarter: depth is missing. Allow speed only if the arm clearly
        # travels across the frame (true-ish side component).
        if frontish_wrist or wrist_lateral is None:
            result.update(
                view="three_quarter",
                speed_ok=False,
                stride_ok=False,
                note="Three-quarter view — too much motion is toward the camera. Film side-on for speed.",
            )
            return result
        result.update(
            view="three_quarter",
            speed_ok=True,
            stride_ok=False,
            note="Three-quarter view — stride is foreshortened; speed is an image-plane estimate",
        )
        return result

    result.update(
        view="side_on",
        speed_ok=True,
        stride_ok=True,
        note="Side-on view — image-plane speed is the best this camera can do (not a radar gun)",
    )
    return result


def _wrist_lateral_ratio(
    frames: list[dict[str, Any]],
    side: str,
    release_frame: int,
    fps: float,
) -> float | None:
    """|Δx| / (|Δx|+|Δy|) of the bowling wrist in the 150 ms before release."""
    pre = max(3, int(round(fps * 0.15)))
    start = frame_by_index(frames, release_frame - pre)
    end = frame_by_index(frames, release_frame)
    if start is None or end is None:
        return None
    w0 = posemod.point(start, f"{side}_wrist")
    w1 = posemod.point(end, f"{side}_wrist")
    if w0 is None or w1 is None:
        return None
    dx = abs(float(w1[0] - w0[0]))
    dy = abs(float(w1[1] - w0[1]))
    den = dx + dy
    # Scale the "did the wrist actually travel" floor to the bowler, not the
    # sensor — 8 px is a real move at 480p and landmark noise at 4K.
    body = posemod.body_pixel_height(end) or posemod.body_pixel_height(start)
    if den < max(4.0, 0.02 * float(body or 400.0)):
        return None
    return dx / den


def flight_is_trackable(
    ball_track: list[dict[str, Any]] | None,
    frame_w: int,
    frame_h: int,
    fps: float | None = None,
    meters_per_pixel: float | None = None,
) -> tuple[bool, str]:
    """Is this a real moving object that left the hand — regardless of km/h?

    Deliberately separate from `flight_geometry_ok`. Whether we found the ball
    and whether its direction lets us quote a speed are different questions, and
    conflating them throws away a perfectly good flight on any clip filmed from
    behind or down the pitch: the overlay then draws no ball, the release cannot
    snap to it, and the gravity timebase loses its only input. Speed is refused
    separately, by `flight_geometry_ok`, with its own reason.
    """
    if not ball_track or len(ball_track) < 5:
        return False, "Ball path too short to be a flight"
    pts = sorted(ball_track, key=lambda p: int(p["frame"]))
    net = float(np.hypot(
        float(pts[-1]["x"]) - float(pts[0]["x"]),
        float(pts[-1]["y"]) - float(pts[0]["y"]),
    ))
    span = max(1, int(pts[-1]["frame"]) - int(pts[0]["frame"]))
    # A ball recedes hard on a down-the-pitch view, so its net image travel can
    # be small; what it never does is sit still. Rate, not distance, is the test.
    min_step = min_ball_step_px(frame_w, frame_h, fps, meters_per_pixel, fraction=0.5)
    if (net / span) < min_step:
        return False, "Tracked object is too slow in the image to be a cricket ball"
    return True, ""


def flight_geometry_ok(
    ball_track: list[dict[str, Any]] | None,
    frame_w: int,
    frame_h: int,
    fps: float | None = None,
    meters_per_pixel: float | None = None,
    rate_is_certain: bool = False,
) -> tuple[bool, str]:
    """Can this flight support an image-plane km/h? (Not: is it a real flight.)"""
    if not ball_track or len(ball_track) < 5:
        return False, "Ball path too short to measure speed"
    pts = sorted(ball_track, key=lambda p: int(p["frame"]))
    net_x = abs(float(pts[-1]["x"]) - float(pts[0]["x"]))
    net_y = abs(float(pts[-1]["y"]) - float(pts[0]["y"]))
    net = float(np.hypot(net_x, net_y))
    min_net = 0.04 * float(np.hypot(frame_w or 1280, frame_h or 720))
    if net < min_net:
        return False, "Ball did not travel far enough in the image to measure speed"
    # Mostly vertical / toward camera: reporting km/h would under-read badly.
    if net_x < 0.03 * float(np.hypot(frame_w or 1280, frame_h or 720)) and net_x < net_y * 0.55:
        return False, "Ball motion is mostly toward the camera — a km/h figure would be false"
    # Stationary-ish blob (tree/post) that barely moves. Expressed as a real
    # speed, not a pixel count — see min_ball_step_px.
    span = max(1, int(pts[-1]["frame"]) - int(pts[0]["frame"]))
    # Two floors, whichever is higher. The speed-derived one needs the frame rate
    # to be known; the radius-derived one does not — a ball in flight clears a
    # good fraction of its own radius every frame at any capture rate, while a
    # poster or fence crawl moves a small fraction of its own size. That keeps a
    # useful floor in place before the timebase has run, when the rate-derived
    # one has to be generous.
    rs = [float(p.get("r") or 0) for p in pts if float(p.get("r") or 0) > 0]
    ball_r = float(np.median(rs)) if rs else 0.0
    min_step = max(
        min_ball_step_px(
            frame_w, frame_h, fps, meters_per_pixel, rate_is_certain=rate_is_certain
        ),
        0.35 * ball_r,
    )
    if (net / span) < min_step:
        return False, "Tracked object is too slow in the image to be a cricket ball"
    return True, ""
