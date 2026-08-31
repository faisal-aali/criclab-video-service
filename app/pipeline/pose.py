"""Human pose extraction (MediaPipe BlazePose) for bowling-action analysis.

This is the measurement engine for the biomechanics metrics. MediaPipe returns
33 body landmarks per frame; we keep them in *pixel* coordinates plus a
visibility score. Everything downstream (release detection, joint angles, arm
speed, the overlay skeleton) is derived from this track — the LLM never sees raw
frames.

Requires the Python 3.12 environment (`.venv312`) because MediaPipe does
not support Python 3.13/3.14.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:  # mediapipe is only importable on the 3.12 venv
    import mediapipe as mp

    _MP_POSE = mp.solutions.pose
    _LM = _MP_POSE.PoseLandmark
    POSE_CONNECTIONS = list(_MP_POSE.POSE_CONNECTIONS)
    MEDIAPIPE_AVAILABLE = True
except Exception:  # pragma: no cover - environment guard
    mp = None
    _MP_POSE = None
    _LM = None
    POSE_CONNECTIONS = []
    MEDIAPIPE_AVAILABLE = False


# Landmark name -> BlazePose index. Kept explicit so the rest of the codebase is
# readable and does not depend on mediapipe being importable.
LANDMARKS: dict[str, int] = {
    "nose": 0,
    "left_eye": 2,
    "right_eye": 5,
    "left_ear": 7,
    "right_ear": 8,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}


def extract_pose_track(
    video_path: Path,
    *,
    max_frames: int = 1200,
    model_complexity: int = 1,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Run pose estimation over the clip and return a serialisable pose track.

    Returns a dict with fps/size and `frames`: a list of
    ``{"frame": int, "landmarks": [[x, y, visibility], ... 33]}`` for every frame
    where a pose was detected.
    """
    if not MEDIAPIPE_AVAILABLE:
        raise RuntimeError(
            "mediapipe is not installed in this interpreter. Run the backend with "
            "the Python 3.12 environment (.venv312)."
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # If the clip is very long, sample evenly so we stay under max_frames.
    stride = 1
    if total > max_frames > 0:
        stride = max(1, total // max_frames)

    pose = _MP_POSE.Pose(
        model_complexity=model_complexity,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frames: list[dict[str, Any]] = []
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(rgb)
                if res.pose_landmarks:
                    lms = res.pose_landmarks.landmark
                    landmarks = [
                        [float(lm.x * width), float(lm.y * height), float(lm.visibility)]
                        for lm in lms
                    ]
                    frames.append({"frame": int(idx), "landmarks": landmarks})
            idx += 1
            if on_progress:
                if total > 0:
                    on_progress(min(idx, total), total)
                else:
                    # OpenCV sometimes reports FRAME_COUNT=0. Never claim 100%.
                    on_progress(idx, idx + 40)
    finally:
        pose.close()
        cap.release()

    frames, dropped = _drop_identity_flickers(frames, width, height)

    return {
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": total,
        "sampled_stride": stride,
        "detected_frames": len(frames),
        "dropped_outlier_frames": dropped,
        "frames": frames,
    }


def _frame_anchor(frame: dict[str, Any]) -> np.ndarray | None:
    """Mid-hip (fallback mid-shoulder) — a stable per-person anchor point."""
    lh = point(frame, "left_hip")
    rh = point(frame, "right_hip")
    if lh is not None and rh is not None:
        return (lh + rh) / 2
    ls = point(frame, "left_shoulder")
    rs = point(frame, "right_shoulder")
    if ls is not None and rs is not None:
        return (ls + rs) / 2
    return None


def _drop_identity_flickers(
    frames: list[dict[str, Any]],
    width: int,
    height: int,
) -> tuple[list[dict[str, Any]], int]:
    """Drop frames where the tracked person teleports (lock flicked to a bystander).

    Smooth motion barely deviates from a short local median of the mid-hip
    trajectory; a swap to another person in frame is an instant jump of
    hundreds of pixels. Reject frames whose anchor sits far from the local
    median — never smooth or move landmarks, only drop the frame.

    Limitation: this catches flickers up to ~4 samples. A sustained swap that
    dominates the 9-sample window would win the median instead — appearance
    cues (not just geometry) would be needed to catch that case.
    """
    if len(frames) < 9 or not width or not height:
        return frames, 0
    anchors: list[np.ndarray | None] = [_frame_anchor(f) for f in frames]
    valid = [(i, a) for i, a in enumerate(anchors) if a is not None]
    if len(valid) < 9:
        return frames, 0
    thresh = 0.08 * float(np.hypot(width, height))
    drop: set[int] = set()
    idxs = [i for i, _ in valid]
    pts = np.array([a for _, a in valid], dtype=float)
    half = 4
    for j in range(len(valid)):
        lo = max(0, j - half)
        hi = min(len(valid), j + half + 1)
        med = np.median(pts[lo:hi], axis=0)
        if float(np.linalg.norm(pts[j] - med)) > thresh:
            drop.add(idxs[j])
    if not drop:
        return frames, 0
    kept = [f for i, f in enumerate(frames) if i not in drop]
    return kept, len(drop)


# --------------------------------------------------------------------------- #
# Geometry helpers — all operate on pixel coordinates from a single frame.
# --------------------------------------------------------------------------- #

def point(frame: dict[str, Any], name: str) -> np.ndarray | None:
    """Return the (x, y) pixel position of a named landmark, if visible."""
    i = LANDMARKS.get(name)
    if i is None:
        return None
    lm = frame["landmarks"][i]
    if lm[2] < 0.25:  # too low visibility to trust
        return None
    return np.array([lm[0], lm[1]], dtype=float)


def angle_3pt(a: np.ndarray | None, b: np.ndarray | None, c: np.ndarray | None) -> float | None:
    """Interior angle ABC in degrees (b is the vertex)."""
    if a is None or b is None or c is None:
        return None
    ba = a - b
    bc = c - b
    nba = np.linalg.norm(ba)
    nbc = np.linalg.norm(bc)
    if nba < 1e-6 or nbc < 1e-6:
        return None
    cosang = float(np.dot(ba, bc) / (nba * nbc))
    cosang = max(-1.0, min(1.0, cosang))
    return math.degrees(math.acos(cosang))


def segment_angle_deg(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    """Angle of segment a->b vs the horizontal, in degrees.

    Uses image convention flipped so that up is positive (image y grows down).
    """
    if a is None or b is None:
        return None
    dx = float(b[0] - a[0])
    dy = float(-(b[1] - a[1]))
    if dx == 0 and dy == 0:
        return None
    return math.degrees(math.atan2(dy, dx))


def body_pixel_height(frame: dict[str, Any]) -> float | None:
    """Vertical extent from the head to the lower of the two ankles."""
    nose = point(frame, "nose")
    la = point(frame, "left_ankle")
    ra = point(frame, "right_ankle")
    ankles = [p for p in (la, ra) if p is not None]
    if nose is None or not ankles:
        return None
    ankle_y = max(p[1] for p in ankles)
    return float(abs(ankle_y - nose[1]))
