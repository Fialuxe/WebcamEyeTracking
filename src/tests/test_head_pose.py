"""Tests for HeadPoseEstimator and gaze normalisation utilities."""
import math
import numpy as np
import pytest

from main.gaze.head_pose import (
    HeadPoseEstimator,
    iris_to_gaze_cam,
    normalize_gaze,
    _LM_INDICES,
    _MODEL_3D,
)


# ---------------------------------------------------------------------------
# iris_to_gaze_cam
# ---------------------------------------------------------------------------

class TestIrisToGazeCam:
    def test_centre_points_forward(self):
        """Iris at image centre should produce a vector pointing straight ahead (+Z)."""
        v = iris_to_gaze_cam(0.5, 0.5, frame_w=640, frame_h=480)
        assert v.shape == (3,)
        # Z component dominates when iris is at centre
        assert v[2] > 0.9

    def test_unit_length(self):
        v = iris_to_gaze_cam(0.3, 0.7, frame_w=1280, frame_h=720)
        assert abs(np.linalg.norm(v) - 1.0) < 1e-6

    def test_left_shift_gives_negative_x(self):
        """Iris shifted left of centre → negative X component."""
        v = iris_to_gaze_cam(0.2, 0.5, frame_w=640, frame_h=480)
        assert v[0] < 0.0

    def test_down_shift_gives_positive_y(self):
        """Iris shifted below centre → positive Y component (image Y increases downward)."""
        v = iris_to_gaze_cam(0.5, 0.7, frame_w=640, frame_h=480)
        assert v[1] > 0.0


# ---------------------------------------------------------------------------
# normalize_gaze with identity rotation
# ---------------------------------------------------------------------------

class TestNormalizeGaze:
    def test_identity_rotation_passthrough(self):
        """With R_head = I, head-local == camera-space components."""
        R_id = np.eye(3)
        lx, ly = normalize_gaze(0.5, 0.5, R_id, frame_w=640, frame_h=480)
        cam = iris_to_gaze_cam(0.5, 0.5, 640, 480)
        assert abs(lx - cam[0]) < 1e-6
        assert abs(ly - cam[1]) < 1e-6

    def test_yaw_rotation_removes_lateral_offset(self):
        """
        If the iris is shifted right AND we rotate R_head to match that lateral
        head turn, the head-local X component should be smaller than the camera
        X component.
        """
        R_id = np.eye(3)
        lx_id, _ = normalize_gaze(0.7, 0.5, R_id, frame_w=640, frame_h=480)

        # Build a 10-degree yaw rotation (head turned left, so iris appears right)
        theta = math.radians(10)
        R_yaw = np.array([
            [ math.cos(theta), 0, math.sin(theta)],
            [0,                1, 0              ],
            [-math.sin(theta), 0, math.cos(theta)],
        ])
        lx_yaw, _ = normalize_gaze(0.7, 0.5, R_yaw, frame_w=640, frame_h=480)

        # After removing the yaw, local X should be smaller
        assert abs(lx_yaw) < abs(lx_id)


# ---------------------------------------------------------------------------
# HeadPoseEstimator — synthetic landmark test
# ---------------------------------------------------------------------------

class TestHeadPoseEstimator:
    def _make_frontal_landmarks(self, frame_w=640, frame_h=480) -> np.ndarray:
        """
        Project the canonical 3D model through a frontal camera pose (identity
        rotation, looking straight ahead) to create synthetic normalised landmarks.
        """
        focal = float(frame_w)
        cx, cy = frame_w / 2.0, frame_h / 2.0
        # Camera space = model space for frontal pose (Z = 600 mm depth)
        depth = 600.0
        pts_norm = np.zeros((478, 2), dtype=np.float64)
        for idx, pt3d in zip(_LM_INDICES, _MODEL_3D):
            px = pt3d[0] * focal / (depth + pt3d[2]) + cx
            py = pt3d[1] * focal / (depth + pt3d[2]) + cy
            pts_norm[idx, 0] = px / frame_w
            pts_norm[idx, 1] = py / frame_h
        return pts_norm

    def test_frontal_pose_returns_near_identity(self):
        """Frontal synthetic landmarks should yield R_head close to identity."""
        est = HeadPoseEstimator(640, 480)
        lm = self._make_frontal_landmarks(640, 480)
        R, tvec = est.estimate(lm, 640, 480)

        assert R.shape == (3, 3)
        assert tvec.shape == (3, 1)
        # Diagonal elements of R should be close to 1 for near-frontal pose
        assert abs(R[0, 0]) > 0.8
        assert abs(R[1, 1]) > 0.8
        assert abs(R[2, 2]) > 0.8

    def test_estimate_returns_valid_shapes(self):
        est = HeadPoseEstimator(1280, 720)
        lm = self._make_frontal_landmarks(1280, 720)
        R, tvec = est.estimate(lm, 1280, 720)
        assert R.shape == (3, 3)
        assert tvec.shape == (3, 1)

    def test_reset_clears_prev_guess(self):
        est = HeadPoseEstimator(640, 480)
        lm = self._make_frontal_landmarks(640, 480)
        est.estimate(lm, 640, 480)
        assert est._prev_rvec is not None
        est.reset()
        assert est._prev_rvec is None

    def test_consecutive_calls_stable(self):
        """Two calls with the same landmarks should give similar R_head (EMA smoothing)."""
        est = HeadPoseEstimator(640, 480)
        lm = self._make_frontal_landmarks(640, 480)
        R1, _ = est.estimate(lm, 640, 480)
        R2, _ = est.estimate(lm, 640, 480)
        # Should be very close after EMA settles
        assert np.max(np.abs(R1 - R2)) < 0.1
