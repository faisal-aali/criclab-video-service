"""Capture-rate recovery — the timebase every speed and duration rests on.

A phone slow-motion clip exported to a 30 fps container reports 30 fps in its
metadata while each frame really holds 1/120 s or 1/240 s of real time. Nothing
in the file says so. Every km/h and every millisecond derived from the container
rate is then wrong by the slow-down factor, silently and confidently.

Gravity is the ruler that catches it. A ball in free flight accelerates downward
at g = 9.81 m/s²; in image pixels that is

    a_px = g / (mpp · fps²)      pixels per frame²

so a parabola fitted to the tracked in-air path, plus the pixel scale we already
have from the bowler's height, solves for fps directly. Nothing here is a prior
about how fast people bowl — it is the same measurement a physicist would make.

We only override the container rate when the fit is strong (a statistically
significant quadratic term over enough of the flight) and the two disagree by
far more than fit noise could explain. Otherwise the container rate stands and
we say why.
"""

from __future__ import annotations

from typing import Any

import numpy as np

G_MPS2 = 9.81

# How far a candidate capture rate may sit from the release-plane estimate.
# The cubic term this estimate rests on is precise to roughly this much.
RATE_TOL = 0.22
# The cubic-at-release estimate is the fallback centre and carries more noise,
# so it is allowed a wider window.
RATE_TOL_CUBIC = 0.34

MIN_FPS = 20.0
MAX_FPS = 1200.0
# Below this the two rates agree inside what the fit can resolve — leave it alone.
DISAGREE_RATIO = 1.5
MIN_POINTS = 8
MIN_SPAN_FRAMES = 8
MIN_SIGNIFICANCE = 3.0  # sigma on the quadratic coefficient


def measure_capture_fps(
    ball_track: list[dict[str, Any]] | None,
    meters_per_pixel: float | None,
    container_fps: float,
) -> dict[str, Any]:
    """Solve the in-air path for the rate the clip was actually captured at.

    Returns a report that always carries `fps` (the rate to use downstream),
    `source`, and `note`; `slow_motion` is True only when we measured a rate
    materially faster than the container claims.
    """
    out: dict[str, Any] = {
        "ball_meters_per_pixel": None,
        "ball_depth_growth": None,
        "raw_debiased_fps": None,
        "fps": float(container_fps),
        "container_fps": float(container_fps),
        "measured_fps": None,
        "raw_measured_fps": None,
        "slow_motion": False,
        "slow_factor": None,
        "source": "container_metadata",
        "note": "Frame rate taken from the video file",
        "gravity_px_per_frame2": None,
        "significance": None,
        "points_used": 0,
    }
    if not meters_per_pixel or meters_per_pixel <= 0 or container_fps <= 0:
        out["note"] = "No pixel scale — cannot check the clip's frame rate against gravity"
        return out

    pts = _flight_points(ball_track)
    if len(pts) < MIN_POINTS:
        out["note"] = "Frame rate from the video file — no tracked ball flight to check it against"
        return out

    ts = np.array([float(p["frame"]) for p in pts])
    ys = np.array([float(p["y"]) for p in pts])
    if ts[-1] - ts[0] < MIN_SPAN_FRAMES:
        out["note"] = "Frame rate from the video file — tracked flight too short to check"
        return out

    fits = _curvature_pair(ts, ys)
    if fits is None:
        out["note"] = "Frame rate from the video file — could not fit the ball's flight"
        return out
    (a_mean, e_mean), (a_rel, e_rel), curve_source = fits
    out["gravity_px_per_frame2"] = round(a_rel, 5)
    out["gravity_px_per_frame2_mean"] = round(a_mean, 5)
    out["curvature_source"] = curve_source
    out["points_used"] = len(pts)

    if a_rel <= 0 or a_mean <= 0:
        out["note"] = "Frame rate from the video file — tracked path does not curve like free flight"
        return out
    sig = a_rel / e_rel if e_rel > 0 else 0.0
    sig_mean = a_mean / e_mean if e_mean > 0 else 0.0
    out["significance"] = round(float(sig), 2)
    out["significance_mean"] = round(float(sig_mean), 2)
    if sig < MIN_SIGNIFICANCE and sig_mean < MIN_SIGNIFICANCE:
        out["note"] = (
            "Frame rate from the video file — the flight's curvature is too weak to "
            "measure the capture rate from gravity"
        )
        return out

    mpp = float(meters_per_pixel)
    fps_rel = float(np.sqrt(G_MPS2 / (mpp * a_rel)))    # at release: unbiased, noisier
    fps_mean = float(np.sqrt(G_MPS2 / (mpp * a_mean)))  # arc average: biased high by recession
    if not np.isfinite(fps_rel) or not (MIN_FPS <= fps_rel <= MAX_FPS):
        out["note"] = "Frame rate from the video file — gravity check gave an impossible rate"
        return out
    out["raw_measured_fps"] = round(fps_rel, 1)
    out["raw_mean_arc_fps"] = round(fps_mean, 1)

    # De-bias the arc estimate with the recession the ball's own size reveals.
    # fps_arc = fps_true x sqrt(mean depth / release depth), so dividing that
    # square root out recovers the release-plane rate from the *precise* arc fit
    # instead of the noisy cubic one. Depth grows roughly linearly along the
    # flight, so the mean over the arc is the midpoint of start and end.
    growth = _depth_growth(pts)
    fps_debiased: float | None = None
    if growth is not None:
        depth_avg = (1.0 + growth) / 2.0
        fps_debiased = fps_mean / float(np.sqrt(depth_avg))
        out["ball_depth_growth"] = round(growth, 2)
        out["raw_debiased_fps"] = round(fps_debiased, 1)

    if fps_rel / float(container_fps) < DISAGREE_RATIO:
        out["note"] = (
            f"Frame rate {container_fps:.0f} fps from the video file, confirmed against "
            f"the ball's fall ({fps_rel:.0f} fps measured)"
        )
        out["ball_meters_per_pixel"] = _ball_scale(a_mean, float(container_fps))
        return out

    used = _pick_rate(fps_rel, fps_mean, float(container_fps), fps_debiased)
    if used is None:
        out["note"] = (
            f"The ball's fall implies about {fps_rel:.0f} fps, which does not match any rate this "
            f"{container_fps:.0f} fps file could have been exported from — leaving the file's rate alone"
        )
        return out
    out["ball_meters_per_pixel"] = _ball_scale(a_mean, used)
    out.update(
        fps=used,
        measured_fps=used,
        slow_motion=True,
        slow_factor=round(used / float(container_fps), 2),
        source="ball_gravity",
        note=(
            f"Slow-motion clip: the file says {container_fps:.0f} fps but the ball falls at "
            f"{G_MPS2:.2f} m/s² only if it was captured at ~{used:.0f} fps "
            f"({used / float(container_fps):.1f}× slow motion). Speeds and timings use the "
            "measured rate — at the file's rate they would read that many times too slow."
        ),
    )
    return out


def _pick_rate(
    fps_rel: float,
    fps_mean: float,
    container_fps: float,
    fps_debiased: float | None = None,
) -> float | None:
    """Choose the capture rate the clip could actually have been exported from.

    Two measurements, with known and opposite error directions:

    * `fps_mean`, from curvature averaged over the whole arc, is biased *high* —
      the ball recedes, so late in the flight it appears to fall more slowly than
      gravity. Depth only ever increases, so this is an upper bound.
    * `fps_rel`, from curvature at the first sample, removes that drift and is
      unbiased, but reading a cubic's middle term costs precision.

    And a structural fact: a slow-motion export writes each captured frame as one
    container frame, so the capture rate is a whole multiple of the container
    rate. Among those multiples we take the fastest one the upper bound allows
    that is still consistent with the release-plane estimate.
    """
    upper = fps_mean * 1.05
    # Prefer the size-de-biased estimate as the centre: it inherits the arc fit's
    # precision while removing that fit's known bias. The cubic stands in when
    # the ball's radii were unusable, but it is noisy enough that a small shift
    # in the tracked points could flip the answer between two adjacent
    # multiples — which is why it is no longer the first choice.
    centre = fps_debiased if fps_debiased else fps_rel
    tol = RATE_TOL if fps_debiased else RATE_TOL_CUBIC
    best: float | None = None
    # Phones export super-slow-motion at large whole factors too: 960 fps written
    # into a 30 fps container is 32x. Stopping at 10 left those clips falling
    # through to "no rate this file could have come from" and silently keeping
    # the container rate — the exact failure this module exists to catch.
    for k in (2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32):
        rate = container_fps * k
        if not (MIN_FPS <= rate <= MAX_FPS) or rate > upper:
            continue
        if abs(rate - centre) / centre > tol:
            continue
        if best is None or rate > best:
            best = rate
    return best


def _flight_points(ball_track: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Detected (never interpolated) in-air samples, time-ordered.

    Interpolated points come from the very parabola we are about to fit, so
    including them would manufacture the agreement we are testing for.
    """
    if not ball_track:
        return []
    pts = [p for p in ball_track if p.get("source") != "interpolated"]
    return sorted(pts, key=lambda p: int(p["frame"]))


def _depth_growth(pts: list[dict[str, Any]]) -> float | None:
    """How much further from the camera the ball gets across the tracked arc.

    A ball's apparent radius is inversely proportional to its distance, so the
    ratio of its size at the start of the flight to its size at the end is the
    ratio of those distances. That is a *direct* measurement of the recession
    that biases the arc-averaged gravity fit — and it comes from every frame,
    where the cubic's correction term is a weak signal read off three or four.

    Returns None when the radii cannot support it: motion blur changes apparent
    size too, so a ratio outside a plausible band is treated as unusable rather
    than trusted.
    """
    rs = [float(p.get("r") or 0.0) for p in pts]
    if len(rs) < 10 or any(r <= 0 for r in rs):
        return None
    k = max(3, len(rs) // 5)
    r_start = float(np.median(rs[:k]))
    r_end = float(np.median(rs[-k:]))
    if r_start <= 0 or r_end <= 0:
        return None
    growth = r_start / r_end
    if not (0.95 <= growth <= 4.0):
        return None
    return growth


def _quadratic_with_error(ts: np.ndarray, ys: np.ndarray) -> tuple[float, float] | None:
    """Return (d²y/dt², 1-sigma error on it) for y(t) = y0 + v·t + ½a·t²."""
    # Centre time to keep the Vandermonde matrix well-conditioned.
    t = ts - float(ts.mean())
    try:
        coeffs, cov = np.polyfit(t, ys, 2, cov=True)
    except Exception:
        return None
    if not np.all(np.isfinite(cov)):
        return None
    a_px = 2.0 * float(coeffs[0])
    a_err = 2.0 * float(np.sqrt(max(cov[0, 0], 0.0)))
    if not np.isfinite(a_px) or not np.isfinite(a_err):
        return None
    return a_px, a_err


def _curvature_pair(ts: np.ndarray, ys: np.ndarray):
    """Downward pixel acceleration measured two ways: arc-average and at release.

    The average over the whole arc is biased low (so its implied frame rate is
    biased high) because the ball recedes from the camera and its pixels come to
    be worth more metres. A cubic absorbs that drift in its third term, leaving
    the quadratic term as the curvature at the first sample, where the ball is
    still at the bowler's own distance. Both are returned so the caller can use
    the bias direction rather than guess at it.
    """
    t = ts - float(ts[0])
    quad = _quadratic_with_error(ts, ys)
    if quad is None:
        return None
    if len(ts) >= 10:
        try:
            coeffs, cov = np.polyfit(t, ys, 3, cov=True)
            if np.all(np.isfinite(cov)):
                a_px = 2.0 * float(coeffs[1])
                a_err = 2.0 * float(np.sqrt(max(cov[1, 1], 0.0)))
                if np.isfinite(a_px) and np.isfinite(a_err) and a_px > 0 and a_err > 0:
                    return quad, (a_px, a_err), "cubic_at_release"
        except Exception:
            pass
    return quad, quad, "quadratic_mean"


def _ball_scale(a_px: float, fps: float) -> float | None:
    """Metres per pixel *in the ball's own depth plane*, from how fast it falls.

    The body-height scale assumes the ball flies at the bowler's distance from
    the camera. It does not: a delivery travels downrange, away from the lens,
    so its pixels are worth more metres than the bowler's. Gravity is a known
    constant, so the ball's measured fall in px/frame² is a ruler for the plane
    it is actually flying in — no assumption about where that plane is.

    This is an average over the tracked arc, and it still cannot see motion
    directly toward or away from the lens, so a speed built on it remains a
    lower bound on the true release speed.
    """
    if a_px <= 0 or fps <= 0:
        return None
    mpp = G_MPS2 / (a_px * fps * fps)
    return round(mpp, 8) if 1e-5 < mpp < 1.0 else None
