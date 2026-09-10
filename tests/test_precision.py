"""Side-on precision pass: robust derivatives, sub-frame events, streak-aware ball path.

Synthetic pose tracks and images only — no video, no Mongo. Each test pins one
behaviour the metrics rely on: a Theil–Sen derivative that ignores one bad
landmark, a foot plant timed inside the frame, a ball-leave fit that survives a
streak outlier and a collapsed wrist, and a centroid that re-centres a fragment
on the whole smear.
"""

from __future__ import annotations

import math
import unittest

import cv2
import numpy as np

from app.pipeline import action as action_mod
from app.pipeline import detect, metrics, render, runner, track
from app.pipeline.pose import LANDMARKS

FPS = 120.0
FRAME_W, FRAME_H = 1920, 1080
# Nose at y=200, ankles at y=1000 → body 800 px; stature 1.80 m.
HEIGHT_M = 1.80
BODY_PX = 800.0
MPP = HEIGHT_M / (BODY_PX / 0.87)
G_PX = 9.80665 / (MPP * FPS * FPS)


def _blank_frame(idx: int) -> dict:
    lms = [[0.0, 0.0, 0.0] for _ in range(33)]
    lms[LANDMARKS["nose"]] = [500.0, 200.0, 0.99]
    lms[LANDMARKS["left_ankle"]] = [470.0, 1000.0, 0.99]
    lms[LANDMARKS["right_ankle"]] = [530.0, 1000.0, 0.99]
    lms[LANDMARKS["right_shoulder"]] = [520.0, 420.0, 0.99]
    lms[LANDMARKS["left_shoulder"]] = [480.0, 420.0, 0.99]
    lms[LANDMARKS["right_hip"]] = [520.0, 640.0, 0.99]
    lms[LANDMARKS["left_hip"]] = [480.0, 640.0, 0.99]
    return {"frame": int(idx), "landmarks": lms}


def _ball_at(t: float) -> tuple[float, float]:
    """Ball position for a flight released at t=20: 30 px/frame across, up then falling."""
    dt = t - 20.0
    return 600.0 + 30.0 * dt, 300.0 - 8.0 * dt + 0.5 * G_PX * dt * dt


def _synthetic_pose(n: int = 41, *, collapse_at: int | None = None) -> dict:
    frames = []
    for i in range(n):
        f = _blank_frame(i)
        if i <= 20:
            bx, by = _ball_at(float(i))
            wx, wy = bx, by + 8.0  # hand under the ball
        else:
            wx, wy = 600.0 + 20.0 * (i - 20), 300.0 + 12.0 * (i - 20)  # hand slows and drops
        ex, ey = wx - 60.0, wy + 100.0
        if collapse_at is not None and i == collapse_at:
            wx, wy = ex + 5.0, ey - 5.0  # MediaPipe parks the wrist on the elbow
        f["landmarks"][LANDMARKS["right_wrist"]] = [wx, wy, 0.99]
        f["landmarks"][LANDMARKS["right_elbow"]] = [ex, ey, 0.99]
        frames.append(f)
    return {"fps": FPS, "width": FRAME_W, "height": FRAME_H, "frames": frames}


def _synthetic_ball(first: int = 23, last: int = 38, *, outlier_at: int | None = 24) -> list[dict]:
    pts = []
    for fr in range(first, last + 1):
        x, y = _ball_at(float(fr))
        if outlier_at is not None and fr == outlier_at:
            y += 60.0  # a streak-centroid jump
        pts.append({"frame": fr, "x": x, "y": y, "r": 10.0, "source": "detected"})
    return pts


class TheilSenTests(unittest.TestCase):
    def test_slope_ignores_one_outlier(self) -> None:
        t = np.arange(10, dtype=float)
        v = 3.0 * t + 1.0
        v[4] += 200.0
        self.assertAlmostEqual(action_mod.theil_sen_slope(t, v), 3.0, places=6)

    def test_robust_speed_no_spike_on_teleport(self) -> None:
        idxs = list(range(41))
        pts = np.array([[4.0 * i, -2.0 * i] for i in idxs], dtype=float)
        pts[20] += [150.0, 150.0]
        speeds, resids = action_mod._robust_speed(idxs, pts, FPS)
        truth = math.hypot(4.0, 2.0)
        inner = speeds[3:-3]
        self.assertLess(max(abs(s - truth) for s in inner), 0.8)
        # The residual is a median: one teleport does not move it, jitter does.
        rng = np.random.default_rng(0)
        noisy = pts + rng.normal(0.0, 3.0, size=pts.shape)
        _, noisy_res = action_mod._robust_speed(idxs, noisy, FPS)
        self.assertGreater(float(np.median(noisy_res)), float(np.median(resids)))

    def test_wrist_speed_peak_lands_on_true_peak(self) -> None:
        frames = []
        for i in range(60):
            f = _blank_frame(i)
            # Velocity profile peaking at i=30.
            v = 30.0 * math.exp(-((i - 30) / 6.0) ** 2)
            x = 100.0 + sum(30.0 * math.exp(-((k - 30) / 6.0) ** 2) for k in range(i))
            f["landmarks"][LANDMARKS["right_wrist"]] = [x, 500.0 + 0.3 * v, 0.99]
            frames.append(f)
        idxs, speeds, _ = action_mod._wrist_speed(frames, "right_wrist", fps=FPS)
        peak_frame, peak = action_mod._robust_peak_speed(idxs, speeds)
        self.assertLessEqual(abs(peak_frame - 30), 2)
        self.assertAlmostEqual(peak, 30.0, delta=1.5)


class AngularVelocityTests(unittest.TestCase):
    def test_unwrap_and_spike(self) -> None:
        idxs = list(range(30))
        angs = [((170.0 + 15.0 * i + 180.0) % 360.0) - 180.0 for i in idxs]  # crosses ±180
        angs[12] += 40.0
        vidx, vel, res = metrics._angular_velocity(idxs, angs, FPS)
        truth = 15.0 * FPS
        for i, v in zip(vidx, vel):
            if 2 <= i <= 27:
                self.assertAlmostEqual(v, truth, delta=0.1 * truth, msg=f"frame {i}")
        rng = np.random.default_rng(0)
        noisy = [a + float(rng.normal(0.0, 2.0)) for a in angs]
        _, _, noisy_res = metrics._angular_velocity(idxs, noisy, FPS)
        self.assertGreater(float(np.median(noisy_res)), float(np.median(res)))

    def test_window_does_not_bridge_gap(self) -> None:
        idxs = [0, 1, 2, 3, 20, 21, 22, 23]
        angs = [0, 10, 20, 30, 0, 10, 20, 30]
        vidx, vel, _ = metrics._angular_velocity(idxs, angs, FPS)
        self.assertEqual(sorted(vidx), idxs)
        for v in vel:
            self.assertAlmostEqual(v, 10.0 * FPS, delta=1.0)


class PlantSubframeTests(unittest.TestCase):
    def test_subframe_from_descent_line(self) -> None:
        ys = [100, 106, 112, 118, 124, 130, 130, 130, 130, 130, 130]
        cands = [(i, np.array([500.0, float(y)])) for i, y in enumerate(ys)]
        plant = action_mod._plant_frame(cands, fps=FPS, body_px=BODY_PX)
        self.assertIsNotNone(plant)
        sub, fit = action_mod._plant_subframe(cands, plant, fps=FPS, body_px=BODY_PX)
        self.assertEqual(fit["method"], "line")
        self.assertGreaterEqual(sub, 4.0)
        self.assertLessEqual(sub, float(plant))
        self.assertAlmostEqual(fit["residual_px"], 0.0, places=6)

    def test_front_foot_contact_returns_subframe(self) -> None:
        pose = _synthetic_pose()
        # Lead (left) ankle descends and plants 12 frames before release at 20.
        for f in pose["frames"]:
            i = int(f["frame"])
            y = 940.0 + 6.0 * min(i, 8) + (12.0 if i >= 8 else 0.0)
            f["landmarks"][LANDMARKS["left_ankle"]] = [470.0, y, 0.99]
        hit = action_mod._detect_front_foot_contact(pose["frames"], "right", 20, fps=FPS)
        self.assertIsNotNone(hit)
        plant, sub, fit = hit
        self.assertLessEqual(abs(sub - plant), 2.0)
        self.assertIn(fit["method"], ("line", "interp"))


class BallLeaveTests(unittest.TestCase):
    def test_confident_leave_with_streak_outlier(self) -> None:
        pose = _synthetic_pose()
        leave = action_mod.ball_leave_frame(
            pose, "right", _synthetic_ball(), 18, fps=FPS, meters_per_pixel=MPP
        )
        self.assertEqual(leave["frame"], 20)
        self.assertGreaterEqual(leave["confidence"], 0.5)
        self.assertGreaterEqual(leave["subframe"], 20.0)
        self.assertLess(leave["subframe"], 21.5)
        self.assertEqual(leave["points_dropped"], 1)
        self.assertTrue(leave["gravity_pinned"])

    def test_collapsed_wrist_is_skipped(self) -> None:
        pose = _synthetic_pose(collapse_at=20)
        leave = action_mod.ball_leave_frame(
            pose, "right", _synthetic_ball(), 18, fps=FPS, meters_per_pixel=MPP
        )
        self.assertIn(20, leave["skipped_frames"])
        self.assertEqual(leave["frame"], 19)
        self.assertGreaterEqual(leave["confidence"], 0.5)
        self.assertGreater(leave["subframe"], 19.5)

    def test_hand_lock_is_not_confident(self) -> None:
        pose = _synthetic_pose()
        # "Ball" that keeps riding the wrist after release: no departure.
        pts = []
        for fr in range(23, 39):
            wr = pose["frames"][fr]["landmarks"][LANDMARKS["right_wrist"]]
            pts.append({"frame": fr, "x": wr[0], "y": wr[1] - 8.0, "r": 10.0, "source": "detected"})
        leave = action_mod.ball_leave_frame(pose, "right", pts, 18, fps=FPS, meters_per_pixel=MPP)
        self.assertLess(leave["confidence"], 0.5)

    def test_snap_only_when_confident(self) -> None:
        pose = _synthetic_pose()
        action = {"throwing_side": "right", "release_frame": 18, "phases": {"release": 18}, "phase_sources": {}}
        leave = action_mod.snap_release_to_ball_leave(pose, action, _synthetic_ball(), fps=FPS, meters_per_pixel=MPP)
        self.assertEqual(action["release_frame"], 20)
        self.assertEqual(action["phase_sources"]["release"], "wrist_closest_to_ball_path")
        self.assertIn("release", action["phase_subframes"])
        self.assertAlmostEqual(action["release_confidence"], leave["confidence"])

        action2 = {"throwing_side": "right", "release_frame": 18, "phases": {"release": 18}, "phase_sources": {}}
        rng = np.random.default_rng(1)
        junk = [{"frame": fr, "x": float(rng.uniform(0, 1900)), "y": float(rng.uniform(0, 1000)), "r": 10.0,
                 "source": "detected"} for fr in range(23, 39)]
        action_mod.snap_release_to_ball_leave(pose, action2, junk, fps=FPS, meters_per_pixel=MPP)
        self.assertEqual(action2["release_frame"], 18)
        self.assertLess(action2["release_confidence"], 0.5)

    def test_settle_after_track_pins_release(self) -> None:
        pose = _synthetic_pose()
        for f in pose["frames"]:
            i = int(f["frame"])
            y = 940.0 + 6.0 * min(i, 8) + (12.0 if i >= 8 else 0.0)
            f["landmarks"][LANDMARKS["left_ankle"]] = [470.0, y, 0.99]
        action = action_mod.analyze_action(pose, bowling_arm="right")
        scale = {"meters_per_pixel": MPP, "calibrated": True, "confidence": 0.65, "body_px_height": BODY_PX}
        ball = track.annotate_frame_motion(_synthetic_ball(), FPS, MPP)
        action, ball, tb, fps, leave = runner.settle_after_track(
            pose_track=pose, action=action, scale=scale, ball_track=ball, container_fps=FPS,
            frame_w=FRAME_W, frame_h=FRAME_H, bowling_arm="right", job_id="test",
        )
        self.assertEqual(action["release_frame"], 20)
        self.assertEqual(action["phase_sources"]["release"], "wrist_closest_to_ball_path")
        self.assertGreaterEqual(action["release_confidence"], 0.5)
        self.assertFalse(tb["slow_motion"])
        self.assertTrue(ball)


class StreakGeometryTests(unittest.TestCase):
    def test_rotated_bar_is_a_streak_with_axis(self) -> None:
        mask = np.zeros((1080, 1920), np.uint8)  # detection working size; gates scale with it
        # The streak flag is the bounding-box aspect rule, so the bar is
        # near-horizontal (a 45° bar boxes as a square and is not flagged).
        theta = math.radians(10.0)
        p0 = (900, 500)
        p1 = (int(900 + 60 * math.cos(theta)), int(500 + 60 * math.sin(theta)))
        cv2.line(mask, p0, p1, 255, 6)
        cands = detect._blob_candidates(mask)
        self.assertEqual(len(cands), 1)
        c = cands[0]
        self.assertTrue(c["streak"])
        self.assertAlmostEqual(c["streak_angle_deg"], 10.0, delta=5.0)
        self.assertAlmostEqual(c["streak_len"], 62.0, delta=12.0)
        q0, q1 = c["streak_p0"], c["streak_p1"]
        self.assertLess(math.hypot(q0[0] - p0[0], q0[1] - p0[1]), 12.0)
        self.assertLess(math.hypot(q1[0] - p1[0], q1[1] - p1[1]), 12.0)

    def test_scaling_and_point_carry_geometry(self) -> None:
        c = {"x": 10.0, "y": 20.0, "r": 5.0, "streak": True, "streak_angle_deg": 0.0,
             "streak_len": 40.0, "streak_width": 6.0, "streak_p0": (-10.0, 20.0), "streak_p1": (30.0, 20.0)}
        s = detect._scale_candidates([c], 2.0, 2.0)[0]
        self.assertAlmostEqual(s["streak_len"], 80.0)
        self.assertAlmostEqual(s["streak_p1"][0], 60.0)
        pt = track._point(7, s)
        self.assertTrue(pt["streak"])
        self.assertAlmostEqual(pt["streak_len"], 80.0)
        plain = track._point(8, {"x": 1.0, "y": 2.0, "r": 3.0})
        self.assertNotIn("streak", plain)


class BlobCentroidTests(unittest.TestCase):
    """`_blob_centroid` keeps its validated behaviour: the Otsu polarity is
    decided by the crop mean, which assumes the crop (1.6x the detected
    radius) is dominated by the ball. That is what the speed fit was tuned
    against; a thin streak in a wide crop leaves the point effectively where
    the detector put it, by design."""

    def test_bright_ball_dominating_crop(self) -> None:
        bgr = np.full((300, 300, 3), 40, np.uint8)
        cv2.circle(bgr, (150, 150), 12, (240, 240, 240), -1)
        hit = track._blob_centroid(bgr, 147.0, 152.0, 12.0)
        self.assertIsNotNone(hit)
        self.assertAlmostEqual(hit["x"], 150.0, delta=1.5)
        self.assertAlmostEqual(hit["y"], 150.0, delta=1.5)
        self.assertGreater(hit["len"], 15.0)
        self.assertLess(abs(hit["len"] - hit["width"]), 6.0)  # a disc has no long axis

    def test_dark_ball_dominating_crop(self) -> None:
        bgr = np.full((120, 120, 3), 200, np.uint8)
        cv2.circle(bgr, (70, 60), 12, (30, 20, 120), -1)
        hit = track._blob_centroid(bgr, 67.0, 58.0, 12.0)
        self.assertIsNotNone(hit)
        self.assertAlmostEqual(hit["x"], 70.0, delta=1.5)
        self.assertAlmostEqual(hit["y"], 60.0, delta=1.5)

    def test_far_centroid_is_rejected(self) -> None:
        bgr = np.full((300, 300, 3), 40, np.uint8)
        cv2.circle(bgr, (150, 150), 8, (240, 240, 240), -1)
        self.assertIsNone(track._blob_centroid(bgr, 100.0, 100.0, 24.0))


class MetricsPrecisionTests(unittest.TestCase):
    def _run(self, action_overrides: dict | None = None):
        pose = _synthetic_pose()
        for f in pose["frames"]:
            i = int(f["frame"])
            y = 940.0 + 6.0 * min(i, 8) + (12.0 if i >= 8 else 0.0)
            f["landmarks"][LANDMARKS["left_ankle"]] = [130.0 + 0.5 * i, y, 0.99]  # ~400 px stride
        action = action_mod.analyze_action(pose, bowling_arm="right", release_override=20, release_subframe=20.4)
        if action_overrides:
            action.update(action_overrides)
        scale = {"meters_per_pixel": MPP, "calibrated": True, "confidence": 0.65, "body_px_height": BODY_PX}
        ball = track.annotate_frame_motion(_synthetic_ball(), FPS, MPP)
        return metrics.compute_metrics(
            fps=FPS, pose_track=pose, action=action, scale=scale, ball_track=ball,
            player_profile={"height_m": HEIGHT_M, "bowling_arm": "right"},
        ), action

    def test_subframe_release_time_and_height(self) -> None:
        m, action = self._run()
        self.assertEqual(m["precision"]["release_time_basis"], "sub_frame")
        subs = action["phase_subframes"]
        expect = (20.4 - subs["front_foot_contact"]) / FPS * 1000.0
        self.assertAlmostEqual(m["release_time_ms"]["value"], expect, delta=0.05)
        self.assertEqual(m["precision"]["release_height_basis"], "sub_frame")
        self.assertEqual(m["release_height_m"]["status"], "ok")
        # wrist y at 20.4 interpolates between frames 20 and 21
        w20 = _ball_at(20.0)[1] + 8.0
        w21 = 300.0 + 12.0
        wy = w20 * 0.6 + w21 * 0.4
        self.assertAlmostEqual(m["release_point"]["y"], wy, delta=0.05)
        self.assertEqual(m["stride_length_pct_height"]["status"], "ok")
        self.assertGreaterEqual(m["precision"]["stride"]["samples"], 3)

    def test_outputs_are_bson_safe(self) -> None:
        from bson import BSON

        m, action = self._run()
        BSON.encode({"metrics": m, "action": runner._strip_series(action)})

    def test_integer_fallback_without_subframes(self) -> None:
        m, action = self._run({"phase_subframes": {}})
        self.assertEqual(m["precision"]["release_time_basis"], "integer_frames")
        ffc = action["phases"]["front_foot_contact"]
        self.assertAlmostEqual(m["release_time_ms"]["value"], (20 - ffc) / FPS * 1000.0, delta=0.05)


class RenderTests(unittest.TestCase):
    def test_interpolated_trail_is_drawn_differently(self) -> None:
        frame = np.zeros((400, 600, 3), np.uint8)
        wrist = [(i, (100 + 5 * i, 300 - 3 * i)) for i in range(25)]
        pts = {}
        for fr in range(20, 40):
            pts[fr] = {"pt": (200 + 10 * (fr - 20), 200), "r": 8.0,
                       "source": "interpolated" if 25 <= fr <= 30 else "detected"}
        phases = {"release": 20}
        render._draw_trail(frame, 39, wrist, pts, phases, [], 20, release_speed_kmh=90.0)
        solid = frame[195:206, 210:250].sum()
        dashed = frame[195:206, 255:295].sum()
        self.assertGreater(solid, 0)
        self.assertGreater(dashed, 0)
        self.assertLess(dashed, solid)


if __name__ == "__main__":
    unittest.main()


class FlowAnchorTests(unittest.TestCase):
    def test_streak_points_anchor_on_leading_end(self) -> None:
        pts = []
        for i in range(6):
            x = 100.0 + 40.0 * i
            pts.append({"frame": i, "x": x, "y": 200.0, "r": 20.0, "source": "detected", "streak": True,
                        "streak_len": 40.0, "streak_p0": (x - 20.0, 200.0), "streak_p1": (x + 20.0, 200.0)})
        out = track._flow_anchors(pts)
        for p in out:
            self.assertAlmostEqual(p["flow_x"], p["x"] + 20.0)  # ahead of the centre, along +x travel
            self.assertAlmostEqual(p["flow_y"], 200.0)

    def test_compact_points_keep_centre(self) -> None:
        pts = [{"frame": i, "x": 10.0 * i, "y": 5.0, "r": 6.0, "source": "detected"} for i in range(5)]
        out = track._flow_anchors(pts)
        self.assertTrue(all("flow_x" not in p for p in out))
