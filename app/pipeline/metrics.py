"""Bowling biomechanics metrics — measured from pose + player profile.

Physical units (km/h, m) use the bowler's stated height to scale pixels.
The LLM never invents measurements.

Formulas (image plane; reject rather than clamp):
  ball_speed   = ballistic fit on detected in-air points (x linear, y quadratic);
                 speed = hypot(vx, dy/dt) at first in-air sample × mpp × fps × 3.6
  arm_speed    = 90th-pct bowling-wrist px/frame from MER→REL on the pose track
                 (downswing on the video, not bowling_style, not a stalled REL frame)
  release_time = (release_frame − FFC) / fps × 1000 ms
  release_ht   = (planted_lead_ankle_y − wrist_y) × meters_per_pixel
  elbow        = interior shoulder–elbow–wrist on the most in-plane frame near release
  arm_swing    = unwrapped bowling-arm segment ω near release (deg/s)
  stride       = |lead_ankle − trail_ankle| / body_height_px at FFC
  hip–shoulder = 2D shoulder-line vs hip-line proxy (always estimated)
"""

from __future__ import annotations

import logging
import math

from typing import Any

import numpy as np

log = logging.getLogger("criclab.metrics")

from app.pipeline import pose as posemod
from app.pipeline.action import frame_by_index
from app.pipeline import track as ball_track_mod
from app.pipeline.view import classify_camera_view, flight_geometry_ok

MIN_PLAUSIBLE_KMH = 25.0
MAX_PLAUSIBLE_KMH = 160.0
MIN_PLAUSIBLE_BALL_KMH = 25.0
# Below this, the physics-constrained fit did not describe the tracked points
# well enough to stand behind a number. Refuse rather than display.
MIN_BALL_FIT_QUALITY = 0.25

PHASE_ORDER = [
    "back_foot_contact",
    "front_foot_contact",
    "hip_rotation",
    "max_external_rotation",
    "arm_horizontal",
    "release",
    "follow_through",
]
PHASE_LABELS = {
    "back_foot_contact": "Back-foot contact",
    "front_foot_contact": "Front-foot contact",
    "hip_rotation": "Hip rotation",
    "max_external_rotation": "Max arm cocking",
    "arm_horizontal": "Arm horizontal",
    "release": "Release",
    "follow_through": "Follow-through",
}


def _metric(
    value,
    unit: str,
    confidence: float,
    note: str | None = None,
    *,
    estimated: bool = True,
    status: str = "ok",
    raw_computed: float | None = None,
) -> dict[str, Any]:
    """Never attach a display value unless status is ok — failed gates must be —."""
    if status != "ok" or value is None:
        status = "unavailable" if status == "ok" else status
        value = None
        confidence = min(float(confidence), 0.08)
    out: dict[str, Any] = {
        "value": None if value is None else (round(float(value), 2) if isinstance(value, (int, float)) else value),
        "unit": unit,
        "confidence": round(float(confidence), 3),
        "estimated": estimated,
        "status": status,
        "note": note,
    }
    if raw_computed is not None:
        out["raw_computed"] = round(float(raw_computed), 2)
    return out


def _wrist_px_near_release(action: dict[str, Any], release_frame: int | None, fps: float) -> float | None:
    """Bowling-wrist px/frame through the throw, from the pose track.

    A single sample *at* the leave-hand frame is often already follow-through
    (the arm has slowed). That produced bogus ~15–20 km/h next to a 70 km/h ball.
    Use the 90th percentile from MER (or 150 ms before REL) through REL — the
    downswing on the video, not the cocking peak and not a stalled wrist.
    """
    series = action.get("wrist_speed_series") or []
    if not series or release_frame is None:
        return None
    rel = int(release_frame)
    mer = (action.get("phases") or {}).get("max_external_rotation")
    pre = max(2, int(round(fps * 0.15)))
    start = rel - pre
    if mer is not None:
        start = max(start, int(mer))
    near = [
        float(p["speed_px"])
        for p in series
        if p.get("speed_px") is not None and start <= int(p["frame"]) <= rel
    ]
    if len(near) < 2:
        return None
    return float(np.percentile(near, 90))


def _clamp_score(x: float) -> float:
    return float(max(0.0, min(100.0, round(x, 0))))


def _lerp_score(value: float | None, lo: float, hi: float, out_lo=40.0, out_hi=100.0) -> float | None:
    if value is None:
        return None
    t = (value - lo) / (hi - lo) if hi != lo else 0.0
    t = max(0.0, min(1.0, t))
    return _clamp_score(out_lo + t * (out_hi - out_lo))


def _odd_win(n: int, lo: int = 3) -> int:
    n = max(lo, int(n))
    return n if n % 2 else n + 1


def _moving_avg(values, win: int):
    if win < 3 or len(values) < 3:
        return values
    win = min(win, len(values) if len(values) % 2 else len(values) - 1)
    if win < 3:
        return values
    kernel = np.ones(win) / win
    padded = np.pad(values, (win // 2, win // 2), mode="edge")
    return list(np.convolve(padded, kernel, mode="valid")[: len(values)])


def _joint_angles(frame: dict[str, Any] | None, side: str) -> dict[str, float | None]:
    if frame is None:
        return {k: None for k in (
            "elbow_extension", "shoulder_abduction", "trunk_flexion",
            "front_knee_flexion", "hip_shoulder_separation", "bowling_arm_angle",
        )}
    o = "left" if side == "right" else "right"

    sh = posemod.point(frame, f"{side}_shoulder")
    el = posemod.point(frame, f"{side}_elbow")
    wr = posemod.point(frame, f"{side}_wrist")
    hip = posemod.point(frame, f"{side}_hip")
    fhip = posemod.point(frame, f"{o}_hip")
    fknee = posemod.point(frame, f"{o}_knee")
    fankle = posemod.point(frame, f"{o}_ankle")
    lsh = posemod.point(frame, "left_shoulder")
    rsh = posemod.point(frame, "right_shoulder")
    lhip = posemod.point(frame, "left_hip")
    rhip = posemod.point(frame, "right_hip")

    elbow_ext = posemod.angle_3pt(sh, el, wr)
    shoulder_abd = posemod.angle_3pt(el, sh, hip)
    front_knee = posemod.angle_3pt(fhip, fknee, fankle)

    mid_sh = mid_hip = None
    if lsh is not None and rsh is not None:
        mid_sh = (lsh + rsh) / 2
    if lhip is not None and rhip is not None:
        mid_hip = (lhip + rhip) / 2
    trunk_flexion = None
    if mid_sh is not None and mid_hip is not None:
        seg = posemod.segment_angle_deg(mid_hip, mid_sh)
        if seg is not None:
            trunk_flexion = round(90.0 - seg, 1)

    sep = None
    sh_line = posemod.segment_angle_deg(rsh, lsh)
    hip_line = posemod.segment_angle_deg(rhip, lhip)
    # Foreshortened lines (pointing at the camera) give meaningless angles —
    # scale the guard by torso length so it holds at any resolution/framing.
    sh_len = float(np.linalg.norm(lsh - rsh)) if lsh is not None and rsh is not None else 0.0
    hip_len = float(np.linalg.norm(lhip - rhip)) if lhip is not None and rhip is not None else 0.0
    torso_len = float(np.linalg.norm(mid_sh - mid_hip)) if mid_sh is not None and mid_hip is not None else 0.0
    min_line = max(8.0, 0.22 * torso_len)
    if sh_line is not None and hip_line is not None and sh_len >= min_line and hip_len >= min_line:
        sep = round(((sh_line - hip_line + 180) % 360) - 180, 1)

    bowling_arm = posemod.segment_angle_deg(sh, wr)

    return {
        "elbow_extension": None if elbow_ext is None else round(elbow_ext, 1),
        "shoulder_abduction": None if shoulder_abd is None else round(shoulder_abd, 1),
        "trunk_flexion": trunk_flexion,
        "front_knee_flexion": None if front_knee is None else round(front_knee, 1),
        "hip_shoulder_separation": sep,
        "bowling_arm_angle": None if bowling_arm is None else round(bowling_arm, 1),
    }


def _arm_inplane_length(frame: dict[str, Any], side: str) -> float:
    sh = posemod.point(frame, f"{side}_shoulder")
    el = posemod.point(frame, f"{side}_elbow")
    wr = posemod.point(frame, f"{side}_wrist")
    if sh is None or el is None or wr is None:
        return 0.0
    return float(np.linalg.norm(sh - el) + np.linalg.norm(el - wr))


def _frames_near(frames: list[dict[str, Any]], center: int | None, fps: float, window_s: float) -> list[dict[str, Any]]:
    if center is None:
        return []
    span = max(1, int(round(fps * window_s)))
    return [f for f in frames if abs(int(f["frame"]) - center) <= span]


def _best_elbow_near_release(frames, side: str, release_frame: int | None, fps: float) -> tuple[float | None, dict[str, Any] | None]:
    """Elbow interior angle on the most in-plane, most-extended pose near release.

    2D collapse (arm pointing at the camera) can report ~10° on a nearly straight arm.
    Prefer the frame with the longest shoulder–elbow–wrist projection, then the largest
    interior angle (180° = straight).
    """
    nearby = _frames_near(frames, release_frame, fps, 0.08)
    if not nearby:
        return None, None
    ranked = sorted(nearby, key=lambda f: _arm_inplane_length(f, side), reverse=True)
    best_ang = None
    best_frame = None
    for f in ranked[: max(3, len(ranked) // 3)]:
        ang = posemod.angle_3pt(
            posemod.point(f, f"{side}_shoulder"),
            posemod.point(f, f"{side}_elbow"),
            posemod.point(f, f"{side}_wrist"),
        )
        if ang is None:
            continue
        if best_ang is None or ang > best_ang:
            best_ang = ang
            best_frame = f
    if best_ang is None:
        return None, None
    return round(float(best_ang), 1), best_frame


# ICC Article 6 / Law 21: an action is illegal when the elbow *straightens* by
# more than 15° between the upper arm reaching horizontal and the ball leaving
# the hand. 10° is the point where a hand-eye judgement starts to be unreliable
# — we use it to mark "worth filming properly", not to accuse anyone.
CHUCK_LIMIT_DEG = 15.0
CHUCK_WATCH_DEG = 10.0

# Pace bands in km/h, by their usual cricket names. Ordered fastest first; the
# first band whose floor the delivery clears wins.
PACE_BANDS = (
    (142.0, "Fast"),
    (130.0, "Fast-medium"),
    (120.0, "Medium-fast"),
    (100.0, "Medium"),
    (80.0, "Medium-slow"),
    (0.0, "Slow / spin pace"),
)


def _elbow_at(frames, side: str, target: int | None, fps: float) -> tuple[float | None, float | None]:
    """Interior elbow angle at a phase frame, with the arm's in-plane length.

    The length is returned so the caller can refuse a reading taken while the
    arm points at the camera, where a straight arm projects as a bent one.
    """
    if target is None:
        return None, None
    nearby = _frames_near(frames, target, fps, 0.03)
    if not nearby:
        return None, None
    best = max(nearby, key=lambda f: _arm_inplane_length(f, side))
    ang = posemod.angle_3pt(
        posemod.point(best, f"{side}_shoulder"),
        posemod.point(best, f"{side}_elbow"),
        posemod.point(best, f"{side}_wrist"),
    )
    if ang is None:
        return None, None
    return float(ang), _arm_inplane_length(best, side)


def _upper_arm_horizontal_frame(
    frames,
    side: str,
    start_frame: int | None,
    release_frame: int | None,
    fps: float,
) -> tuple[int | None, float]:
    """Frame in the delivery swing where the upper arm passes horizontal.

    ICC Article 6 measures elbow extension from this instant to release. Only the
    second half of the front-foot-contact-to-release interval is searched: earlier
    in the action the arm is still winding up and also passes horizontal, and
    picking that crossing would compare a deeply cocked elbow against a straight
    one and manufacture a huge false "extension".

    Returns the frame and how many degrees off horizontal it actually was, so the
    caller can refuse a swing that never really passes horizontal in view.
    """
    if release_frame is None:
        return None, 999.0
    lo = int(start_frame) if start_frame is not None else int(release_frame) - max(4, int(round(fps * 0.15)))
    lo = int(lo + (int(release_frame) - lo) * 0.5)
    best_fr, best_err = None, 999.0
    for f in frames:
        fr = int(f["frame"])
        if fr < lo or fr > int(release_frame):
            continue
        ang = posemod.segment_angle_deg(
            posemod.point(f, f"{side}_shoulder"), posemod.point(f, f"{side}_elbow")
        )
        if ang is None:
            continue
        err = min(abs(ang), abs(abs(ang) - 180.0))
        if err < best_err:
            best_err, best_fr = err, fr
    return best_fr, best_err


def _elbow_extension_range(
    frames,
    side: str,
    phases: dict[str, Any],
    release_frame: int | None,
    elbow_at_release: float | None,
    fps: float,
    body_px: float | None,
    side_on_ok: bool,
) -> dict[str, Any]:
    """Elbow straightening from upper-arm-horizontal to release — the chuck test.

    This is deliberately hard to satisfy. The ICC test is done with marker-based
    3D capture for a reason: from one camera, an arm angled towards the lens
    projects a straight elbow as a bent one, and the error runs to tens of
    degrees — comfortably more than the 15 degrees the whole test turns on. A
    verdict built on that would accuse a legal bowler of throwing, so we only
    report one when the swing genuinely happens across the image, and we call it
    screening even then.

    `elbow_at_release_deg` is reported whenever the elbow is visible, because a
    straight arm at release is worth showing on its own.
    """
    out: dict[str, Any] = {
        "value": None, "verdict": None, "note": None,
        "elbow_at_arm_horizontal": None,
        "elbow_at_release": None if elbow_at_release is None else round(float(elbow_at_release), 1),
    }
    if not side_on_ok:
        out["note"] = (
            "The 15° throwing test is not run from this camera angle. Filmed anywhere but square "
            "to the bowling arm, a straight elbow projects as a bent one by more than the 15° the "
            "test turns on, so any verdict would be guesswork. Elbow angle at release is still shown."
        )
        return out
    if release_frame is None:
        out["note"] = "Release not detected — the 15° throwing test needs both ends of the swing"
        return out

    start_fr, horiz_err = _upper_arm_horizontal_frame(
        frames, side, phases.get("front_foot_contact"), release_frame, fps
    )
    if start_fr is None or horiz_err > 20.0:
        out["note"] = (
            "The upper arm never passes clearly through horizontal in this view, so there is no "
            "defined start point for the 15° throwing test"
        )
        return out

    ang_start, len_start = _elbow_at(frames, side, start_fr, fps)
    _, len_rel = _elbow_at(frames, side, release_frame, fps)
    if ang_start is None or elbow_at_release is None:
        out["note"] = "Bowling elbow not visible through the swing — extension not measured"
        return out

    # Both readings must come from an arm that projects to a real length, or the
    # angles are projection artefacts rather than joint angles.
    min_len = max(24.0, 0.16 * float(body_px)) if body_px else 40.0
    if (len_start or 0) < min_len or (len_rel or 0) < min_len:
        out["note"] = (
            "Bowling arm points at the camera through the swing — a straight arm projects as a bent "
            "one, so the 15° test would be false"
        )
        return out

    extension = float(elbow_at_release) - float(ang_start)
    out["elbow_at_arm_horizontal"] = round(float(ang_start), 1)
    out["value"] = round(extension, 1)
    tail = (
        " Screening from one camera, not an accredited test: only a 3D, marker-based assessment "
        "can call an action."
    )
    if extension > CHUCK_LIMIT_DEG:
        out["verdict"] = "above_limit_screening"
        out["note"] = (
            f"Elbow straightens about {extension:.0f}° from upper-arm-horizontal to release, above "
            f"the ICC {CHUCK_LIMIT_DEG:.0f}° limit on this view." + tail
        )
    elif extension > CHUCK_WATCH_DEG:
        out["verdict"] = "borderline"
        out["note"] = (
            f"Elbow straightens about {extension:.0f}° — inside the ICC {CHUCK_LIMIT_DEG:.0f}° limit, "
            "but close enough that camera angle alone could account for the gap." + tail
        )
    elif extension < -CHUCK_WATCH_DEG:
        out["verdict"] = "flexing"
        out["note"] = (
            f"The elbow bends {abs(extension):.0f}° into release rather than straightening, so there "
            "is no extension to test. Clear on this measure." + tail
        )
    else:
        out["verdict"] = "within_limit"
        out["note"] = (
            f"Elbow straightens about {extension:.0f}° from upper-arm-horizontal to release, well "
            f"inside the ICC {CHUCK_LIMIT_DEG:.0f}° limit." + tail
        )
    return out



# Physically plausible bands for a bowled delivery. These are cross-checks
# between independently measured quantities, not gates on any single one — a
# metric can be individually in range and still be impossible alongside its
# neighbours, and that is what these catch.
COHERENCE_BANDS: dict[str, tuple[float, float, str]] = {
    "ffc_to_release_ms": (55.0, 400.0, "Front-foot contact to release"),
    "bfc_to_ffc_ms": (60.0, 500.0, "Back-foot to front-foot contact"),
    "ball_over_arm_ratio": (1.0, 2.4, "Ball speed vs bowling-hand speed"),
    "release_height_over_stature": (0.70, 1.55, "Release height vs the bowler's own height"),
    "wrist_speed_vs_arm_swing": (0.45, 2.2, "Hand speed vs arm-swing rate"),
}


def _cross_validate(
    *,
    phases: dict[str, Any],
    fps: float,
    ball_kmh: float | None,
    arm_kmh: float | None,
    release_height_m: float | None,
    stature_m: float | None,
    arm_swing_deg_s: float | None,
    arm_length_m: float | None,
) -> dict[str, Any]:
    """Check the measurements against each other, not just against their own bands.

    Each headline number is already gated on its own plausibility. That is not
    enough: a release time, an arm speed and a ball speed can each sit inside a
    sensible range while describing a delivery that could not have happened. The
    checks here compare quantities that were measured *independently* — pose
    timing against ball timing, hand speed against the arm's angular rate,
    release height against the bowler's own stature — so a wrong frame rate, a
    mis-detected event or a lock on the wrong object shows up as a contradiction
    rather than a plausible-looking figure.

    Nothing is corrected here. Failing checks are reported so the caller can
    lower confidence and say what disagreed.
    """
    checks: list[dict[str, Any]] = []

    def _add(key: str, value: float | None, *, detail: str = "") -> None:
        if value is None:
            return
        lo, hi, label = COHERENCE_BANDS[key]
        ok = lo <= value <= hi
        checks.append({
            "check": key,
            "label": label,
            "value": round(float(value), 2),
            "expected": [lo, hi],
            "ok": ok,
            "detail": detail,
        })

    def _gap_ms(a: str, b: str) -> float | None:
        fa, fb = phases.get(a), phases.get(b)
        if fa is None or fb is None:
            return None
        return (int(fb) - int(fa)) / max(float(fps), 1e-6) * 1000.0

    _add("ffc_to_release_ms", _gap_ms("front_foot_contact", "release"),
         detail="Timing from the pose track, divided by the capture rate")
    _add("bfc_to_ffc_ms", _gap_ms("back_foot_contact", "front_foot_contact"),
         detail="Delivery-stride duration from the pose track")

    if ball_kmh is not None and arm_kmh:
        _add("ball_over_arm_ratio", float(ball_kmh) / float(arm_kmh),
             detail="Ball speed comes from the flight, hand speed from the pose track")

    if release_height_m is not None and stature_m:
        _add("release_height_over_stature", float(release_height_m) / float(stature_m),
             detail="Release height against the stature used to scale the image")

    # The wrist rides the arm: its speed should be the arm's angular rate times
    # the arm's length. Both sides are measured separately, so agreement is real
    # evidence the capture rate and the pixel scale are right.
    if arm_swing_deg_s and arm_length_m:
        implied_mps = math.radians(float(arm_swing_deg_s)) * float(arm_length_m)
        if implied_mps > 0.1 and arm_kmh:
            _add("wrist_speed_vs_arm_swing", (float(arm_kmh) / 3.6) / implied_mps,
                 detail=f"Arm swing implies about {implied_mps * 3.6:.0f} km/h at the hand")

    failed = [c for c in checks if not c["ok"]]
    out: dict[str, Any] = {
        "checks": checks,
        "passed": len(checks) - len(failed),
        "total": len(checks),
        "ok": not failed if checks else None,
        "failed": [c["check"] for c in failed],
        "note": None,
    }
    if not checks:
        out["note"] = "Not enough independent measurements on this clip to cross-check them"
    elif failed:
        first = failed[0]
        out["note"] = (
            f"{len(failed)} of {len(checks)} cross-checks disagree — "
            f"{first['label'].lower()} came out at {first['value']}, outside the "
            f"{first['expected'][0]}-{first['expected'][1]} a real delivery sits in. "
            "Treat the affected numbers as indicative and re-film square-on in good light."
        )
    else:
        out["note"] = (
            f"All {len(checks)} cross-checks agree — timing, speeds and geometry describe "
            "one consistent delivery"
        )
    return out


def _classify_delivery_pace(
    ball_kmh: float | None,
    arm_kmh: float | None,
    profile_style: str | None,
    *,
    foreshortened: bool = False,
) -> dict[str, Any]:
    """Name the delivery's pace band from what was actually measured.

    Ball speed decides it when the ball was tracked. Failing that the bowling
    hand's own speed stands in — the hand is slower than the ball it releases,
    so that reading is explicitly labelled as the weaker basis. The player's
    stated style is carried alongside for context and never overrides a
    measurement.
    """
    out: dict[str, Any] = {"value": None, "basis": None, "speed_kmh": None,
                           "profile_style": profile_style or None, "note": None}
    speed = None
    if ball_kmh is not None:
        speed, out["basis"] = float(ball_kmh), "ball_speed"
    elif arm_kmh is not None:
        speed, out["basis"] = float(arm_kmh), "arm_speed"
    if speed is None:
        out["note"] = (
            "No measured ball or arm speed on this clip — pace band not named. "
            "A stated bowling style is not a measurement."
        )
        return out

    out["speed_kmh"] = round(speed, 1)
    out["value"] = next(name for floor, name in PACE_BANDS if speed >= floor)
    if out["basis"] == "ball_speed":
        out["note"] = f"From the tracked ball at {speed:.0f} km/h (image-plane estimate, not a radar gun)"
    else:
        out["note"] = (
            f"Ball flight was not tracked — band inferred from the bowling hand at {speed:.0f} km/h, "
            "which runs slower than the ball it releases, so the true band may be quicker"
        )
    if foreshortened:
        out["band_edge_caveat"] = True
        out["note"] += (
            ". The delivery also travels away from the lens on this angle, so the image-plane "
            "speed reads low and the true band may be one step quicker"
        )
    if profile_style and out["value"]:
        out["note"] += f". Stated style: {profile_style}"
    return out


def _line_angle_series(frames, a_name, b_name, *, fps: float = 30.0):
    """Segment angle over time, skipping frames where the segment is degenerate.

    On side-on footage a body line (shoulders, hips) can point almost straight
    at the camera; its projected length collapses and the 2D angle flips wildly
    — thousands of fake deg/s. Angles from a near-zero-length segment are
    meaningless, so those frames are dropped, not smoothed into the series.
    """
    idxs, a_pts, b_pts, lens = [], [], [], []
    for f in frames:
        a = posemod.point(f, a_name)
        b = posemod.point(f, b_name)
        if a is not None and b is not None:
            idxs.append(int(f["frame"]))
            a_pts.append(a)
            b_pts.append(b)
            lens.append(float(np.linalg.norm(a - b)))
    if len(idxs) < 3:
        return [], []
    min_len = max(10.0, 0.30 * float(np.percentile(lens, 90)))
    kept = [i for i, L in enumerate(lens) if L >= min_len]
    if len(kept) < 3:
        return [], []
    idxs = [idxs[i] for i in kept]
    a_pts = [a_pts[i] for i in kept]
    b_pts = [b_pts[i] for i in kept]

    smooth_win = _odd_win(max(3, int(round(fps * 0.025))))
    a_arr = np.array(a_pts, dtype=float)
    b_arr = np.array(b_pts, dtype=float)
    a_arr[:, 0] = _moving_avg(list(a_arr[:, 0]), smooth_win)
    a_arr[:, 1] = _moving_avg(list(a_arr[:, 1]), smooth_win)
    b_arr[:, 0] = _moving_avg(list(b_arr[:, 0]), smooth_win)
    b_arr[:, 1] = _moving_avg(list(b_arr[:, 1]), smooth_win)

    angs = []
    for i in range(len(idxs)):
        ang = posemod.segment_angle_deg(a_arr[i], b_arr[i])
        if ang is not None:
            angs.append(ang)
        else:
            angs.append(angs[-1] if angs else 0.0)
    return idxs, angs


def _angular_velocity(idxs, angs, fps, smooth_win: int = 0):
    if len(angs) < 2:
        return [], []
    unwrapped = np.degrees(np.unwrap(np.radians(angs)))
    if smooth_win:
        unwrapped = np.array(_moving_avg(list(unwrapped), smooth_win))
    # A sample that bridges a dropped run (degenerate-segment filter, missed
    # pose) would be fiction — the unwrap branch across the gap is arbitrary.
    dfs = [max(1, int(idxs[i]) - int(idxs[i - 1])) for i in range(1, len(idxs))]
    max_df = max(3.0, 3.0 * float(np.median(dfs))) if dfs else 3.0
    vel, vidx = [], []
    for i in range(1, len(unwrapped)):
        df = max(1, idxs[i] - idxs[i - 1])
        if df > max_df:
            continue
        vel.append((unwrapped[i] - unwrapped[i - 1]) / df * fps)
        vidx.append(idxs[i])
    return vidx, vel


def _peak_near_release(vidx, vel, release_frame: int | None, fps: float) -> tuple[float | None, float | None]:
    if not vel:
        return None, None
    if release_frame is None:
        raw = float(np.percentile(np.abs(vel), 90))
        return round(raw, 0), raw
    span = max(2, int(round(fps * 0.08)))
    near = [abs(v) for i, v in zip(vidx, vel) if abs(i - release_frame) <= span]
    if not near:
        near = list(np.abs(vel))
    raw = float(np.percentile(near, 90))
    return round(raw, 0), raw


def _peak_in_window(
    vidx, vel, start_f: int | None, end_f: int | None
) -> tuple[float | None, float | None, int | None]:
    """Robust peak magnitude + its frame over [start_f, end_f].

    Hip and trunk rotation peak *before* release in a well-sequenced action
    (hip → torso → arm), so a near-release sample misses the real peak.
    95th percentile kills a single-sample spike for the *value*; the *frame*
    comes from a median-of-3 of the magnitudes so that same spike cannot
    become the event frame either. Too few samples inside a requested window
    → None (never silently widen to the whole series — the "stride peak"
    would come from the run-up or follow-through).
    """
    if not vel:
        return None, None, None
    windowed = start_f is not None or end_f is not None
    pairs = [
        (int(i), float(v))
        for i, v in zip(vidx, vel)
        if (start_f is None or int(i) >= int(start_f)) and (end_f is None or int(i) <= int(end_f))
    ]
    if len(pairs) < 3:
        if windowed:
            return None, None, None
        pairs = [(int(i), float(v)) for i, v in zip(vidx, vel)]
    if len(pairs) < 3:
        return None, None, None
    mags = [abs(v) for _, v in pairs]
    med3 = [float(np.median(mags[max(0, i - 1): i + 2])) for i in range(len(mags))]
    peak_fr = pairs[int(np.argmax(med3))][0]
    raw = float(np.percentile(mags, 95))
    return round(raw, 0), raw, int(peak_fr)


def _smooth_series_column(rows: list[dict[str, Any]], key: str, win: int) -> None:
    """Moving-average one numeric column in place, per contiguous non-None run.

    High-fps landmark jitter otherwise dominates the PDF angle charts. Gaps
    (None) are preserved — never interpolated into fake measurements.
    """
    if win < 3 or len(rows) < 3:
        return
    run_start = None
    for i in range(len(rows) + 1):
        has = i < len(rows) and rows[i].get(key) is not None
        if has and run_start is None:
            run_start = i
        elif not has and run_start is not None:
            run = [float(rows[j][key]) for j in range(run_start, i)]
            if len(run) >= 3:
                sm = _moving_avg(run, min(win, len(run) if len(run) % 2 else len(run) - 1))
                for j, v in zip(range(run_start, i), sm):
                    rows[j][key] = round(float(v), 1)
            run_start = None


def compute_metrics(
    *,
    fps: float,
    pose_track: dict[str, Any],
    action: dict[str, Any],
    scale: dict[str, Any],
    ball_track: list[dict[str, Any]] | None = None,
    player_profile: dict[str, Any] | None = None,
    timebase_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    frames = pose_track.get("frames") or []
    mpp = scale.get("meters_per_pixel")
    calibrated = bool(scale.get("calibrated"))
    scale_conf = float(scale.get("confidence") or 0.0)
    scale_note = scale.get("note")
    body_px = scale.get("body_px_height")
    profile = player_profile or {}
    height_m = profile.get("height_m") or scale.get("reference_height_m")

    side = action.get("throwing_side")
    release_frame = action.get("release_frame")
    phases = action.get("phases") or {}
    act_conf = float(action.get("confidence") or 0.1)
    rel_pose = frame_by_index(frames, release_frame)
    camera_view = classify_camera_view(pose_track, action)
    frame_w = int(pose_track.get("width") or 1280)
    frame_h = int(pose_track.get("height") or 720)
    speed_view_ok = bool(camera_view.get("speed_ok"))
    stride_view_ok = bool(camera_view.get("stride_ok"))

    # A tracked flight that measurably crosses the image is direct evidence that
    # this delivery happens in the image plane — stronger than the pose-based
    # guess, which reads a mixed/open action as front-on and then refuses every
    # speed on a clip whose ball plainly travels across frame. The flight had to
    # pass optical-flow validation and flight_geometry_ok to get here, so this
    # widens what we measure without lowering the bar for what counts as a ball.
    # `fps` here is the recovered capture rate, so the speed floor can be exact.
    ball_geo_ok, _ball_geo_reason = flight_geometry_ok(
        ball_track, frame_w, frame_h, fps, mpp, rate_is_certain=True
    )
    view_overridden_by_ball = bool(ball_geo_ok) and not speed_view_ok
    if view_overridden_by_ball:
        speed_view_ok = True
        camera_view = dict(camera_view)
        camera_view["speed_ok"] = True
        camera_view["speed_ok_source"] = "tracked_ball_flight"
        camera_view["note"] = (
            f"{camera_view.get('view', 'angled')} camera by pose, but the tracked ball crosses the "
            "image — speeds are measured from that flight and read low if the delivery also travels "
            "away from the lens. Film square-on for the truest km/h."
        )

    # --- Arm / hand speed: throw on the pose track (not bowling_style, not a stalled REL frame) ---
    arm_speed_mps = arm_speed_kmh = None
    speed_status = "unavailable"
    speed_note = None
    speed_conf = 0.0
    speed_raw = None
    hand_speed_px = _wrist_px_near_release(action, release_frame, fps)

    if not speed_view_ok:
        speed_note = camera_view.get("note") or "Camera angle cannot yield a truthful arm speed in km/h"
        speed_status = "unavailable"
    elif not hand_speed_px or hand_speed_px <= 0:
        speed_note = "No bowling-hand speed at release"
        speed_status = "unavailable"
    elif not mpp or not calibrated:
        speed_note = scale_note or "Enter bowler height to compute arm speed in km/h"
        speed_status = "unavailable"
    else:
        v_mps = hand_speed_px * mpp * fps
        v_kmh = v_mps * 3.6
        speed_raw = v_kmh
        if v_kmh > MAX_PLAUSIBLE_KMH:
            speed_note = (
                f"Computed {v_kmh:.0f} km/h is above a realistic bowling-arm speed "
                f"(>{MAX_PLAUSIBLE_KMH:.0f}) — likely camera angle or tracking error"
            )
            speed_status = "unavailable"
            speed_conf = 0.05
        elif v_kmh < MIN_PLAUSIBLE_KMH:
            speed_note = (
                f"Computed {v_kmh:.1f} km/h is below a realistic bowling-arm speed "
                f"(<{MIN_PLAUSIBLE_KMH:.0f}) — bowling arm may not be in view"
            )
            speed_status = "unavailable"
            speed_conf = 0.05
        else:
            arm_speed_mps, arm_speed_kmh = v_mps, v_kmh
            speed_status = "ok"
            speed_conf = min(0.85, 0.35 + scale_conf * 0.4 + act_conf * 0.3)
            speed_note = (
                "Bowling-wrist speed on the video through the throw "
                "(pose track × your height — not bowling style, not a radar gun)"
            )

    # --- Ball speed: in-air path only; reject if view/geometry/estimators fail ---
    ball_speed_mps = ball_speed_kmh = None
    ball_status = "unavailable"
    ball_note = "Ball flight not tracked — use a side-on clip with a visible red or white ball after release"
    ball_conf = 0.0
    ball_raw = None
    ball_px = None
    ball_scale_basis = None
    ball_depth_ratio = None
    geo_ok, geo_reason = ball_geo_ok, _ball_geo_reason
    if not speed_view_ok:
        ball_note = camera_view.get("note") or "Camera angle cannot yield a truthful ball speed"
        ball_status = "unavailable"
    elif not geo_ok:
        ball_note = geo_reason
        ball_status = "unavailable"
    elif ball_track and len(ball_track) >= 4:
        # Release speed comes from one physics-constrained robust fit, not from
        # reconciling two estimators that measure different things. The old path
        # blended a quadratic endpoint-derivative against the median of raw
        # per-frame `hypot(dx, dy)` — and because `dy` is where the detector's
        # noise lives (see `robust_release_velocity_px_per_frame`), both inputs
        # and therefore the blend were inflated by it. On the clip this was
        # rewritten for that produced 100 km/h against a true 86.
        if not mpp or not calibrated:
            ball_note = scale_note or "Ball tracked — enter bowler height to convert to km/h"
            ball_status = "unavailable"
        else:
            b_kmh = None
            b_mps = None
            # Release speed is read off the first frames of flight, where the
            # ball is still at the bowler's own distance from the camera — the
            # plane the height scale calibrates. The ball's own gravity-derived
            # scale describes where it ends up, not where it left the hand, and
            # is carried as a diagnostic only.
            ball_mpp = (timebase_info or {}).get("ball_meters_per_pixel")
            depth_ratio = (ball_mpp / mpp) if (ball_mpp and mpp) else None
            use_mpp, scale_basis = mpp, "bowler_height_plane"

            ball_fit = ball_track_mod.robust_release_velocity_px_per_frame(
                ball_track, fps, use_mpp, release_frame=action.get("release_frame")
            )
            if ball_fit:
                ball_px = float(ball_fit["speed_px_per_frame"])
                b_mps = ball_px * use_mpp * fps
                b_kmh = b_mps * 3.6

            if ball_fit is None:
                ball_note = "Ball path too short or too broken to fit a release speed"
                ball_status = "unavailable"
            elif ball_fit["quality"] < MIN_BALL_FIT_QUALITY:
                # An honest "could not measure" beats a confident wrong number.
                b_kmh = None
                b_mps = None
                ball_note = (
                    "The tracked ball positions do not fit a real flight path closely enough to "
                    "quote a speed — the detector likely lost the ball or locked onto something "
                    "else. Re-film side-on with the ball clearly visible after release."
                )
                ball_status = "unavailable"
                ball_conf = 0.08
            ball_raw = b_kmh

            # Reproducibility trail. Logged from the same values the metric is
            # built from — not recomputed — so it can never describe a different
            # calculation than the one that produced the displayed number.
            if ball_fit:
                _k = use_mpp * fps * 3.6
                _first = ball_fit["first_frame"]
                log.info(
                    "ball-speed | fps=%.3f mpp=%.7f g_px=%.4f/frame^2 | "
                    "release_frame=%s release_t=%.4fs fit_window=%s-%s "
                    "points_used=%d dropped=%d | resid x=%.1fpx y=%.1fpx | "
                    "vx=%.2f vy=%.2f |v|=%.2f px/frame | "
                    "vx=%.1f vy=%.1f release_speed=%.1f km/h | quality=%.2f -> %s",
                    fps, use_mpp, ball_fit["gravity_px_per_frame2"],
                    action.get("release_frame"), (action.get("release_frame") or 0) / max(fps, 1e-6),
                    _first, ball_fit["last_frame"],
                    ball_fit["points_used"], ball_fit["points_dropped"],
                    ball_fit["x_residual_px"], ball_fit["y_residual_px"],
                    ball_fit["vx_px_per_frame"], ball_fit["vy_px_per_frame"],
                    ball_fit["speed_px_per_frame"],
                    ball_fit["vx_px_per_frame"] * _k, ball_fit["vy_px_per_frame"] * _k,
                    ball_fit["speed_px_per_frame"] * _k, ball_fit["quality"],
                    "REPORTED" if b_kmh is not None else "REFUSED (low quality)",
                )

            if b_kmh is None:
                ball_note = ball_note or "Ball path too short to measure speed"
                ball_status = "unavailable"
            elif b_kmh > MAX_PLAUSIBLE_KMH:
                ball_note = (
                    f"Tracked path gave {b_kmh:.0f} km/h — not a realistic cricket ball speed "
                    "(likely the wrong blob). Re-film side-on with the ball clearly leaving the hand."
                )
                ball_status = "unavailable"
                ball_conf = 0.08
            elif b_kmh < MIN_PLAUSIBLE_BALL_KMH:
                ball_note = (
                    f"Tracked path gave {b_kmh:.1f} km/h — too slow to be release speed. "
                    "The ball may be out of view or lost after release."
                )
                ball_status = "unavailable"
                ball_conf = 0.08
            else:
                ball_speed_mps, ball_speed_kmh = b_mps, b_kmh
                ball_status = "ok"
                ball_conf = min(0.8, 0.35 + 0.04 * len(ball_track) + scale_conf * 0.25)
                ball_note = (
                    "Release speed measured over the first frames after the ball leaves the hand, "
                    "where it is still at your own distance from the camera. This is the speed "
                    "across the image only — a single camera cannot see how fast the ball also "
                    "travels away from the lens, so on anything but a square-on view the true "
                    "release speed is higher than this. Not a radar gun."
                )
                if depth_ratio and depth_ratio > 1.15:
                    ball_note += (
                        f" On this clip the ball recedes to about {depth_ratio:.1f}x its release "
                        "distance during the tracked flight, so expect a real gap."
                    )
                ball_scale_basis = scale_basis
                ball_depth_ratio = depth_ratio
    else:
        ball_note = "Ball flight not tracked — use a side-on clip with a visible red or white ball after release"

    # --- Release point / height / angle ---
    release_point = None
    release_height_m = None
    release_height_status = "unavailable"
    release_height_note = None
    release_angle = None

    ffc = phases.get("front_foot_contact")
    ffc_pose = frame_by_index(frames, ffc)
    ground_y = None
    if ffc_pose is not None:
        lead = "left" if side == "right" else "right"
        planted = posemod.point(ffc_pose, f"{lead}_ankle") if side else None
        if planted is not None:
            ground_y = float(planted[1])
        else:
            ankles = [posemod.point(ffc_pose, n) for n in ("left_ankle", "right_ankle")]
            ankles = [p for p in ankles if p is not None]
            if ankles:
                ground_y = max(float(p[1]) for p in ankles)

    height_frame = rel_pose
    if rel_pose is not None and side:
        wr = posemod.point(rel_pose, f"{side}_wrist")
        if wr is not None:
            # REL is the bowling wrist at leave-hand — never the in-air ball.
            release_point = {"x": float(wr[0]), "y": float(wr[1])}

    if height_frame is not None and side:
        wr = posemod.point(height_frame, f"{side}_wrist")
        sh = posemod.point(height_frame, f"{side}_shoulder")
        el = posemod.point(height_frame, f"{side}_elbow")
        if wr is not None and release_point is None:
            release_point = {"x": float(wr[0]), "y": float(wr[1])}
        # Release height is a *vertical* distance, and turning the camera around the
        # bowler does not foreshorten vertical pixels the way it foreshortens a
        # stride down the pitch. The height scale comes from this same bowler's own
        # pixel stature, so it stays valid from any yaw — only the sanity band gates it.
        if wr is not None and mpp and calibrated and ground_y is not None:
            h_m = (ground_y - float(wr[1])) * mpp
            if 0.8 <= h_m <= 3.2:
                release_height_m = h_m
                release_height_status = "ok"
                release_height_note = (
                    "Wrist height above the planted front foot, using your height to scale pixels"
                )
            else:
                release_height_note = f"Computed {h_m:.2f} m is outside a realistic release height"
                release_height_status = "unavailable"
        elif wr is not None and mpp and calibrated:
            la = posemod.point(height_frame, "left_ankle")
            ra = posemod.point(height_frame, "right_ankle")
            ankles = [p for p in (la, ra) if p is not None]
            if ankles:
                h_m = (max(float(p[1]) for p in ankles) - float(wr[1])) * mpp
                if 0.8 <= h_m <= 3.2:
                    release_height_m = h_m
                    release_height_status = "ok"
                    release_height_note = "Wrist height above ankles at release (pose + your height)"
                else:
                    release_height_note = f"Computed {h_m:.2f} m is outside a realistic release height"
        elif not calibrated:
            release_height_note = scale_note or "Enter bowler height to compute release height"
        else:
            release_height_note = "Wrist or planted ankle not visible"
        ref = el if el is not None else sh
        ang = posemod.segment_angle_deg(ref, wr) if wr is not None else None
        if ang is not None:
            release_angle = round(ang, 1)

    # --- Timing: FFC → release ---
    release_time_ms = None
    time_note = None
    time_status = "unavailable"
    if ffc is not None and release_frame is not None and release_frame >= ffc:
        release_time_ms = (release_frame - ffc) / max(fps, 1e-6) * 1000.0
        time_status = "ok"
        time_note = "Front-foot contact (lead ankle plant) → release"
    else:
        time_note = "Front-foot contact not detected from lead ankle — release time unavailable"

    win = action.get("delivery_window") or {}
    w_start, w_end = win.get("start"), win.get("end")

    # Chart/series span: the wrist-speed window starts after FFC, but the charts
    # must show the whole delivery (BFC → follow-through) like SpinLab's.
    series_pad = int(round(fps * 0.25))
    lo_cands = [v for v in (phases.get("back_foot_contact"), phases.get("front_foot_contact"), w_start) if v is not None]
    hi_cands = [v for v in (phases.get("follow_through"), w_end) if v is not None]
    series_lo = (int(min(lo_cands)) - series_pad) if lo_cands else None
    series_hi = (int(max(hi_cands)) + series_pad) if hi_cands else None
    series_frames = [
        f for f in frames
        if (series_lo is None or f["frame"] >= series_lo) and (series_hi is None or f["frame"] <= series_hi)
    ]

    ang_smooth = _odd_win(max(3, int(round(fps * 0.03))))
    line_win = _odd_win(max(5, int(round(fps * 0.05))))

    arm_vel: list[float] = []
    arm_vidx: list[int] = []
    if side and series_frames:
        arm_idx, arm_ang = _line_angle_series(
            series_frames, f"{side}_shoulder", f"{side}_wrist", fps=fps,
        )
        arm_vidx, arm_vel = _angular_velocity(arm_idx, arm_ang, fps, ang_smooth)
    arm_swing, arm_swing_raw = _peak_near_release(arm_vidx, arm_vel, release_frame, fps)
    arm_swing_note = None
    if arm_swing is not None and (arm_swing > 4000 or arm_swing < 40):
        arm_swing_note = (
            f"Computed {arm_swing:.0f} deg/s is outside a cricket-plausible arm-swing band — not reported"
        )
        arm_swing = None
    elif arm_swing is not None:
        arm_swing_note = "Bowling-arm angular speed near release (image plane, deg/s — not a static angle)"

    hip_idx, hip_ang = _line_angle_series(series_frames, "right_hip", "left_hip", fps=fps)
    sh_idx, sh_ang = _line_angle_series(series_frames, "right_shoulder", "left_shoulder", fps=fps)
    hip_vidx, hip_vel = _angular_velocity(hip_idx, hip_ang, fps, line_win)
    sh_vidx, sh_vel = _angular_velocity(sh_idx, sh_ang, fps, line_win)
    # Hip and trunk peak during the delivery stride, before the arm — search the
    # whole stride window (SpinLab-style kinematic chain), not ±80 ms of release.
    rot_start = phases.get("back_foot_contact")
    if rot_start is None:
        rot_start = phases.get("front_foot_contact")
    if rot_start is not None:
        rot_start = int(rot_start) - max(2, int(round(fps * 0.10)))
    elif w_start is not None:
        rot_start = int(w_start)
    # Bounded AT release: hip and trunk lead the arm in the chain, so a "peak"
    # in the follow-through is not a stride peak — it would put sequence item 2
    # after item 4 on the overlay and print a negative hip→REL split.
    rot_end = None
    if release_frame is not None:
        rot_end = int(release_frame)
    elif w_end is not None:
        rot_end = int(w_end)
    hip_rot, hip_raw, hip_peak_fr = _peak_in_window(hip_vidx, hip_vel, rot_start, rot_end)
    trunk_rot, trunk_raw, trunk_peak_fr = _peak_in_window(sh_vidx, sh_vel, rot_start, rot_end)
    hip_note = trunk_note = None
    if hip_rot is not None and (hip_rot > 1200 or hip_rot < 20):
        hip_note = f"Computed {hip_rot:.0f} deg/s is outside a realistic 2D pelvis-line band — not reported"
        hip_rot = None
        hip_peak_fr = None
    elif hip_rot is not None:
        hip_note = "Peak 2D pelvis-line speed during the delivery stride — not true 3D hip rotation"
    else:
        hip_note = "Too few clean pelvis-line samples in the stride window — hip line foreshortened from this angle"
    if trunk_rot is not None and (trunk_rot > 1200 or trunk_rot < 20):
        trunk_note = f"Computed {trunk_rot:.0f} deg/s is outside a realistic 2D shoulder-line band — not reported"
        trunk_rot = None
        trunk_peak_fr = None
    elif trunk_rot is not None:
        trunk_note = "Peak 2D shoulder-line speed during the delivery stride — not true 3D trunk rotation"
    else:
        trunk_note = "Too few clean shoulder-line samples in the stride window — shoulders foreshortened from this angle"

    # The hip-rotation event everywhere (overlay disc, timeline, stills, PDF)
    # is the measured hip peak — the same frame as the reported deg/s.
    if hip_peak_fr is not None and hip_rot is not None:
        phases["hip_rotation"] = int(hip_peak_fr)
        sources = action.get("phase_sources")
        if isinstance(sources, dict):
            sources["hip_rotation"] = "peak_hip_line_angular_velocity"

    hip_trunk_gap_ms = None
    if hip_peak_fr is not None and trunk_peak_fr is not None and hip_rot is not None and trunk_rot is not None:
        hip_trunk_gap_ms = (int(trunk_peak_fr) - int(hip_peak_fr)) / max(fps, 1e-6) * 1000.0

    # --- Stride length using stated height (not crouched pose height at FFC) ---
    stride_pct = None
    stride_note = None
    stride_status = "unavailable"
    if ffc_pose is not None:
        la = posemod.point(ffc_pose, "left_ankle")
        ra = posemod.point(ffc_pose, "right_ankle")
        if la is not None and ra is not None:
            stride_px = float(np.linalg.norm(la - ra))
            full_h_px = None
            if mpp and calibrated and height_m:
                full_h_px = float(height_m) / float(mpp)
            else:
                bh = posemod.body_pixel_height(ffc_pose)
                if bh:
                    full_h_px = bh / 0.87
            if full_h_px:
                stride_pct = round(stride_px / full_h_px * 100.0, 1)
                if not stride_view_ok:
                    stride_note = (
                        camera_view.get("note")
                        or "Stride would be foreshortened from this camera angle — not reported"
                    )
                    stride_pct = None
                    stride_status = "unavailable"
                elif 20 <= stride_pct <= 120:
                    stride_status = "ok"
                    stride_note = "Ankle-to-ankle distance at front-foot contact, as % of your height"
                else:
                    stride_note = f"Computed {stride_pct:.0f}% height is outside a realistic stride"
                    stride_pct = None
            else:
                stride_note = "Need bowler height to convert stride into % of stature"
        else:
            stride_note = "Ankles not visible at front-foot contact"
    else:
        stride_note = "Front-foot contact not detected"

    joint_table: dict[str, dict[str, float | None]] = {}
    for ph in PHASE_ORDER:
        if ph in phases:
            joint_table[ph] = _joint_angles(frame_by_index(frames, phases.get(ph)), side or "right")

    rel_angles = joint_table.get("release", {}) if "release" in joint_table else _joint_angles(rel_pose, side or "right")
    elbow_val, _elbow_frame = _best_elbow_near_release(frames, side or "right", release_frame, fps)
    if elbow_val is not None:
        rel_angles = dict(rel_angles)
        rel_angles["elbow_extension"] = elbow_val
        if "release" in joint_table:
            joint_table["release"]["elbow_extension"] = elbow_val

    ffc_angles = joint_table.get("front_foot_contact") or _joint_angles(ffc_pose, side or "right")
    front_knee = ffc_angles.get("front_knee_flexion")
    knee_note = "Front-leg angle at front-foot contact (180° = straight)" if front_knee is not None else "Front knee not visible at plant"
    if front_knee is None:
        front_knee = rel_angles.get("front_knee_flexion")
        knee_note = "Front-leg angle at release (180° = straight)" if front_knee is not None else "Front knee not visible"

    base_t = phases.get("front_foot_contact")
    if base_t is None:
        base_t = w_start

    def _t_ms(fr: int | None) -> float | None:
        # base_t == 0 is a valid base frame — compare with None, never truthiness.
        if fr is None or base_t is None:
            return None
        return round((int(fr) - int(base_t)) / max(fps, 1e-6) * 1000.0, 1)

    angle_series: list[dict[str, Any]] = []
    for f in series_frames:
        fr = int(f["frame"])
        a = _joint_angles(f, side or "right")
        angle_series.append({
            "frame": fr,
            "t_ms": _t_ms(fr),
            "elbow_extension": a["elbow_extension"],
            "shoulder_abduction": a["shoulder_abduction"],
            "trunk_flexion": a["trunk_flexion"],
            "front_knee_flexion": a["front_knee_flexion"],
        })
    # ~30 ms position smoothing so high-fps landmark jitter doesn't dominate
    # the PDF charts and per-phase velocity table (smooth, never clamp).
    for col in ("elbow_extension", "shoulder_abduction", "trunk_flexion", "front_knee_flexion"):
        _smooth_series_column(angle_series, col, ang_smooth)

    # Rotation-speed series for the SpinLab-style sequencing chart (page 2).
    # Per-sample plausibility uses the SAME bands that gate the headline
    # metrics: a 3000 deg/s "pelvis line" in the follow-through is projection
    # collapse, and plotting it as signal would contradict the number we
    # refused to report. Out-of-band samples become None (a gap), never clamped.
    def _in_band(v: float, cap: float) -> float | None:
        return round(float(v), 1) if abs(float(v)) <= cap else None

    hip_v_by_f = {int(i): float(v) for i, v in zip(hip_vidx, hip_vel)}
    sh_v_by_f = {int(i): float(v) for i, v in zip(sh_vidx, sh_vel)}
    arm_v_by_f = {int(i): float(v) for i, v in zip(arm_vidx, arm_vel)}
    rotation_series: list[dict[str, Any]] = []
    for fr in sorted(set(hip_v_by_f) | set(sh_v_by_f) | set(arm_v_by_f)):
        rotation_series.append({
            "frame": fr,
            "t_ms": _t_ms(fr),
            "hip_deg_s": None if fr not in hip_v_by_f else _in_band(hip_v_by_f[fr], 1200.0),
            "trunk_deg_s": None if fr not in sh_v_by_f else _in_band(sh_v_by_f[fr], 1200.0),
            "arm_deg_s": None if fr not in arm_v_by_f else _in_band(arm_v_by_f[fr], 4000.0),
        })

    event_t_ms = {
        key: _t_ms(phases.get(key))
        for key in ("back_foot_contact", "front_foot_contact", "hip_rotation",
                    "max_external_rotation", "arm_horizontal", "release", "follow_through")
        if phases.get(key) is not None
    }

    def rng(vel, clamp):
        if not vel:
            return {"min": None, "max": None}
        lo = float(np.percentile(vel, 5))
        hi = float(np.percentile(vel, 95))
        if abs(lo) > clamp or abs(hi) > clamp:
            return {"min": None, "max": None, "note": "outside realistic range — not clamped"}
        return {
            "min": round(lo, 0),
            "max": round(hi, 0),
        }

    angular_velocity_range = {
        "bowling_arm": rng(arm_vel, 4000),
        "hip_line": rng(hip_vel, 1600),
        "shoulder_line": rng(sh_vel, 1600),
    }

    # Sequencing is judged on the same peak frames the hip/trunk metrics report.
    seq_score = None
    order_ok = None
    f_hip, f_tor, f_arm = hip_peak_fr, trunk_peak_fr, release_frame
    if f_hip is not None and f_tor is not None and f_arm is not None:
        order_ok = f_hip <= f_tor <= f_arm
        seq_score = 100.0 if order_ok else (70.0 if f_hip <= f_arm else 45.0)

    # Wrist at the leave-hand *frame* can already be follow-through. If a real
    # in-air ball is ~4× faster than the measured arm, the arm sample is wrong.
    if (
        arm_speed_kmh is not None
        and ball_speed_kmh is not None
        and ball_speed_kmh > 0
        and arm_speed_kmh < 0.35 * ball_speed_kmh
    ):
        speed_note = (
            f"Wrist on the video had already slowed ({arm_speed_kmh:.0f} km/h) while the "
            f"ball was tracked at {ball_speed_kmh:.0f} km/h — not reporting that arm speed"
        )
        arm_speed_kmh = arm_speed_mps = None
        speed_status = "unavailable"

    arm_score = _lerp_score(arm_speed_kmh, 40, 120) if arm_speed_kmh is not None else None
    ball_score = _lerp_score(ball_speed_kmh, 60, 140) if ball_speed_kmh is not None else None
    brace_score = _lerp_score(front_knee, 120, 178) if front_knee is not None else None
    # |sep| beyond ~75° is projection collapse, not anatomy — reject, never score it.
    sep = rel_angles.get("hip_shoulder_separation")
    if sep is not None and abs(float(sep)) > 75:
        sep = None
    sep_score = _lerp_score(abs(sep), 5, 45) if sep is not None else None
    # Missing ball speed must not silently score from arm speed.
    components = [s for s in [ball_score, arm_score, seq_score, brace_score, sep_score] if s is not None]
    overall = _clamp_score(sum(components) / len(components)) if components else None

    scores = {
        "overall": overall,
        "arm_speed": arm_score,
        "ball_speed": ball_score,
        "sequencing": seq_score,
        "front_leg_brace": brace_score,
        "hip_shoulder_separation": sep_score,
        "label": "Heuristic indicators from measured angles/speed — not clinical grades",
    }

    elbow_note = (
        "Interior elbow angle near release (180° = straight bowling arm)"
        if elbow_val is not None else "Elbow not visible near release"
    )

    # Overlay / results share this 4-item sequence (omit a row if the frame was not seen).
    # --- Consistency: the ball cannot leave slower than the wrist that threw it ---
    # The ball sits beyond the wrist on the same rotating arm, so its release
    # speed is always the greater of the two (typically 1.1-1.5x). If the numbers
    # come out the other way round, one of them is foreshortened — almost always
    # the ball, which travels downrange away from the lens while the wrist stays
    # in the bowler's plane. We do not "correct" it; we say so and mark the ball
    # figure as a floor.
    speed_consistency: dict[str, Any] = {"ok": None, "ratio": None, "note": None}
    if ball_speed_kmh is not None and arm_speed_kmh:
        ratio = float(ball_speed_kmh) / float(arm_speed_kmh)
        speed_consistency["ratio"] = round(ratio, 2)
        speed_consistency["ok"] = ratio >= 1.0
        if ratio < 1.0:
            speed_consistency["note"] = (
                f"Measured ball speed ({ball_speed_kmh:.0f} km/h) came out below bowling-hand speed "
                f"({arm_speed_kmh:.0f} km/h), which cannot happen physically — the ball leaves from "
                "beyond the wrist on the same arm. The ball is travelling partly away from the camera, "
                "so its on-screen speed under-reads. Treat the ball figure as a lower bound and re-film "
                "square-on to the delivery for a truer number."
            )
            ball_conf = min(ball_conf, 0.3)
            ball_note = f"{ball_note} {speed_consistency['note']}"
        else:
            speed_consistency["note"] = (
                f"Ball leaves {ratio:.2f}x the bowling-hand speed — consistent with the ball sitting "
                "beyond the wrist on the same arm"
            )

    # --- Cross-validation across independently measured quantities ---
    # The arm's in-plane length at release, converted to metres, lets the wrist's
    # measured speed be checked against the arm's measured angular rate — two
    # numbers that came from different places and must agree.
    arm_len_m: float | None = None
    if rel_pose is not None and side and mpp and calibrated:
        px_len = _arm_inplane_length(rel_pose, side)
        if px_len > 0:
            arm_len_m = float(px_len) * float(mpp)
            if not (0.35 <= arm_len_m <= 1.05):
                arm_len_m = None  # foreshortened at release; not a usable lever arm

    cross_validation = _cross_validate(
        phases=phases,
        fps=fps,
        ball_kmh=ball_speed_kmh,
        arm_kmh=arm_speed_kmh,
        release_height_m=release_height_m,
        stature_m=height_m,
        arm_swing_deg_s=arm_swing,
        arm_length_m=arm_len_m,
    )
    # A measurement contradicted by its neighbours is not trustworthy just
    # because it sits inside its own band. Lower the confidence of what the
    # failing checks touch, and say so in the note rather than silently.
    if cross_validation.get("failed"):
        failed = set(cross_validation["failed"])
        if {"ball_over_arm_ratio"} & failed:
            ball_conf = min(ball_conf, 0.3)
        if {"wrist_speed_vs_arm_swing", "ffc_to_release_ms"} & failed:
            speed_conf = min(speed_conf, 0.35)

    # --- Action legality (chuck screening) and pace band ---
    legality = _elbow_extension_range(
        frames, side or "right", phases, release_frame,
        elbow_val if elbow_val is not None else rel_angles.get("elbow_extension"),
        fps, body_px, stride_view_ok,
    )
    pace = _classify_delivery_pace(
        ball_speed_kmh, arm_speed_kmh, profile.get("bowling_style"),
        foreshortened=not stride_view_ok,
    )

    kinematic_sequence: list[dict[str, Any]] = []
    seq_spec = [
        (1, "front_foot_contact", "Front-foot contact", False),
        (2, "hip_rotation", "Hip rotation", True),
        (3, "max_external_rotation", "Max arm cocking", False),
        (4, "release", "Elbow extension", False),
    ]
    for n, key, label, estimated in seq_spec:
        fr = phases.get(key)
        if key == "max_external_rotation" and fr is None:
            fr = phases.get("arm_horizontal")
            label = "Arm horizontal" if fr is not None else label
        kinematic_sequence.append({
            "n": n,
            "key": key,
            "label": label,
            "frame": int(fr) if fr is not None else None,
            "estimated": estimated,
        })

    return {
        "throwing_side": side,
        "player_profile": {
            "player_name": profile.get("player_name"),
            "age_years": profile.get("age_years"),
            "height_m": profile.get("height_m"),
            "height_ft": profile.get("height_ft"),
            "height_in": profile.get("height_in"),
            "weight_lbs": profile.get("weight_lbs"),
            "weight_kg": profile.get("weight_kg"),
            "bowling_arm": profile.get("bowling_arm") or side,
            "bowling_style": profile.get("bowling_style"),
        },
        "release_frame": release_frame,
        "release_point": release_point,
        "phases": phases,
        "phase_order": [p for p in PHASE_ORDER if p in phases],
        "phase_labels": {k: PHASE_LABELS[k] for k in PHASE_ORDER if k in phases},
        "phase_sources": action.get("phase_sources") or {},
        "kinematic_sequence": kinematic_sequence,
        "fps": fps,
        "timebase": timebase_info or {
            "fps": fps, "container_fps": fps, "slow_motion": False,
            "source": "container_metadata", "note": "Frame rate taken from the video file",
        },

        "ball_speed_kmh": _metric(
            ball_speed_kmh, "km/h", ball_conf, ball_note,
            estimated=True, status=ball_status, raw_computed=ball_raw,
        ),
        "ball_speed_mps": _metric(
            ball_speed_mps, "m/s", ball_conf, ball_note,
            estimated=True, status=ball_status,
            raw_computed=(ball_raw / 3.6 if ball_raw is not None else None),
        ),
        "ball_speed_px_per_frame": _metric(
            ball_px, "px/frame", ball_conf if ball_px else 0.0,
            "Ball displacement in the image just after release" if ball_px else ball_note,
            estimated=False, status="ok" if ball_px else "unavailable",
        ),
        "arm_speed_kmh": _metric(
            arm_speed_kmh, "km/h", speed_conf, speed_note,
            estimated=True, status=speed_status, raw_computed=speed_raw,
        ),
        "arm_speed_mps": _metric(
            arm_speed_mps, "m/s", speed_conf, speed_note,
            estimated=True, status=speed_status, raw_computed=(speed_raw / 3.6 if speed_raw is not None else None),
        ),
        "hand_speed_px_per_frame": _metric(
            hand_speed_px, "px/frame", act_conf if hand_speed_px else 0.0,
            "Bowling-hand speed at leave-hand in image pixels (not the cocking peak; no scale needed)",
            estimated=False, status="ok" if hand_speed_px else "unavailable",
        ),
        "release_time_ms": _metric(
            release_time_ms, "ms", act_conf if time_status == "ok" else 0.0, time_note,
            estimated=False, status=time_status,
        ),
        "hip_to_trunk_peak_gap_ms": _metric(
            hip_trunk_gap_ms, "ms", 0.25 if hip_trunk_gap_ms is not None else 0.0,
            "Time from peak hip-line to peak shoulder-line speed (2D proxies; positive = hip first)"
            if hip_trunk_gap_ms is not None
            else "Needs both hip and trunk rotation peaks within plausible bands",
            estimated=True,
            status="ok" if hip_trunk_gap_ms is not None else "unavailable",
        ),
        "arm_swing_speed_deg_s": _metric(
            arm_swing, "deg/s", 0.55 if arm_swing is not None else 0.0,
            arm_swing_note or "Bowling-arm angular speed (image plane)",
            estimated=True,
            status="ok" if arm_swing is not None else "unavailable",
            raw_computed=arm_swing_raw,
        ),
        "hip_rotation_speed_deg_s": _metric(
            hip_rot, "deg/s", 0.25 if hip_rot is not None else 0.0, hip_note,
            estimated=True,
            status="ok" if hip_rot is not None else "unavailable",
            raw_computed=hip_raw,
        ),
        "trunk_rotation_speed_deg_s": _metric(
            trunk_rot, "deg/s", 0.25 if trunk_rot is not None else 0.0, trunk_note,
            estimated=True,
            status="ok" if trunk_rot is not None else "unavailable",
            raw_computed=trunk_raw,
        ),
        "release_height_m": _metric(
            release_height_m, "m", scale_conf if release_height_status == "ok" else 0.0,
            release_height_note, estimated=True, status=release_height_status,
        ),
        "release_angle_deg": _metric(
            release_angle, "deg", 0.5 if release_angle is not None else 0.0,
            "Forearm angle vs horizontal at release (90° ≈ vertical, image plane)" if release_angle is not None else "Forearm angle unavailable",
            estimated=True, status="ok" if release_angle is not None else "unavailable",
        ),
        "stride_length_pct_height": _metric(
            stride_pct, "% height", 0.55 if stride_status == "ok" else 0.0, stride_note,
            estimated=False, status=stride_status,
        ),
        "elbow_extension_deg": _metric(
            elbow_val if elbow_val is not None else rel_angles.get("elbow_extension"), "deg",
            0.6 if elbow_val is not None else 0.0,
            elbow_note,
            estimated=False,
            status="ok" if (elbow_val is not None or rel_angles.get("elbow_extension") is not None) else "unavailable",
        ),
        "elbow_extension_range_deg": _metric(
            legality["value"], "deg",
            0.45 if legality["value"] is not None else 0.0,
            legality["note"],
            estimated=True,
            status="ok" if legality["value"] is not None else "unavailable",
        ),
        "action_legality": {
            "verdict": legality["verdict"],
            "assessable": legality["verdict"] is not None,
            "limit_deg": CHUCK_LIMIT_DEG,
            "extension_deg": legality["value"],
            "elbow_at_arm_horizontal_deg": legality["elbow_at_arm_horizontal"],
            "elbow_at_release_deg": legality["elbow_at_release"],
            "note": legality["note"],
            "status": "ok" if legality["verdict"] is not None else "unavailable",
        },
        "speed_consistency": speed_consistency,
        "cross_validation": cross_validation,
        "ball_speed_scale_basis": ball_scale_basis,
        "ball_depth_ratio": ball_depth_ratio,
        "delivery_type": {
            "value": pace["value"],
            "basis": pace["basis"],
            "speed_kmh": pace["speed_kmh"],
            "profile_style": pace["profile_style"],
            "band_edge_caveat": pace.get("band_edge_caveat", False),
            "note": pace["note"],
            "status": "ok" if pace["value"] is not None else "unavailable",
        },
        "front_knee_flexion_deg": _metric(
            front_knee, "deg",
            0.6 if front_knee is not None else 0.0,
            knee_note,
            estimated=False,
            status="ok" if front_knee is not None else "unavailable",
        ),
        "hip_shoulder_separation_deg": _metric(
            sep, "deg",
            0.4 if sep is not None else 0.0,
            "Shoulder line vs hip line at release (image-plane estimate, not 3D rotation)"
            if sep is not None
            else (
                f"Computed {rel_angles.get('hip_shoulder_separation'):.0f}° is projection collapse "
                "from this camera angle — not reported"
                if rel_angles.get("hip_shoulder_separation") is not None
                else "Shoulder or hip line not measurable at release — not visible, or pointing at the camera"
            ),
            estimated=True,
            status="ok" if sep is not None else "unavailable",
            raw_computed=rel_angles.get("hip_shoulder_separation"),
        ),

        "scores": scores,
        "sequencing_ok": order_ok,
        "joint_angle_table": joint_table,
        "angle_series": angle_series,
        "rotation_series": rotation_series,
        "event_t_ms": event_t_ms,
        "rotation_peaks": {
            "hip_frame": hip_peak_fr,
            "trunk_frame": trunk_peak_fr,
            "hip_t_ms": _t_ms(hip_peak_fr),
            "trunk_t_ms": _t_ms(trunk_peak_fr),
        },
        "angular_velocity_range": angular_velocity_range,
        "wrist_speed_series": action.get("wrist_speed_series", []),

        "trajectory_points": [
            {
                "frame": int(p["frame"]),
                "x": float(p["x"]),
                "y": float(p["y"]),
                "source": p.get("source"),
                "dist_px": p.get("dist_px"),
                "speed_kmh": p.get("speed_kmh"),
            }
            for p in (ball_track or [])
        ],
        "ball_speed_series": [
            {
                "frame": int(p["frame"]),
                "speed_kmh": p.get("speed_kmh"),
                "dist_px": p.get("dist_px"),
                "source": p.get("source"),
            }
            for p in (ball_track or [])
            if p.get("speed_kmh") is not None or p.get("dist_px") is not None
        ],
        "scale": scale,
        "quality": {
            "pose_frames": len(frames),
            "detected_ratio": round(len(frames) / max(1, pose_track.get("frame_count") or len(frames)), 2),
            "has_release": release_frame is not None,
            "has_front_foot_contact": ffc is not None,
            "calibrated": calibrated,
            "tracking_ok": len(frames) >= 10 and release_frame is not None,
            "ball_points": len(ball_track or []),
            "bowling_arm_source": action.get("bowling_arm_source"),
            "camera_view": camera_view.get("view"),
            "camera_view_note": camera_view.get("note"),
            "speed_view_ok": speed_view_ok,
            "speed_view_from_ball": view_overridden_by_ball,
            "capture_fps": (timebase_info or {}).get("fps", fps),
            "slow_motion": bool((timebase_info or {}).get("slow_motion")),
            "shoulder_width_ratio": camera_view.get("shoulder_width_ratio"),
            "opencv_flow_ok": bool((ball_track or [{}])[0].get("flow_stats")) if ball_track else False,
            "opencv_flow_agree": ((ball_track or [{}])[0].get("flow_stats") or {}).get("agree_frac"),
        },
    }
