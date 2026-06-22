"""
Head pose estimation from MediaPipe Face Mesh landmarks via cv2.solvePnP.

Provides R_head (3×3 rotation matrix) and diagnostic utilities.
The main gaze feature is now iris-relative-to-eye-corners (see webcam_source.py);
R_head is retained for logging/diagnostics.
"""
from __future__ import annotations

import math

import numpy as np
import cv2

# MediaPipe Face Mesh landmark indices used for PnP
# nose tip, chin, left-eye outer, right-eye outer, left-mouth, right-mouth
_LM_INDICES: list[int] = [4, 152, 263, 33, 287, 57]

# Canonical 3-D face model (mm), centred at nose tip
_MODEL_3D = np.array([
    [  0.0,   0.0,   0.0],   # nose tip
    [  0.0, -63.6, -12.5],   # chin
    [-43.3,  32.7, -26.0],   # left eye outer corner
    [ 43.3,  32.7, -26.0],   # right eye outer corner
    [-28.9, -28.9, -24.1],   # left mouth corner
    [ 28.9, -28.9, -24.1],   # right mouth corner
], dtype=np.float64)

# EMA smoothing factor for rotation matrix (reduces per-frame jitter)
_R_ALPHA: float = 0.7


class HeadPoseEstimator:
    """
    Estimates head rotation (R_head) and translation (tvec) from 6 facial
    landmarks using solvePnP.  Assumes a simple pinhole camera model with
    focal length = frame width (heuristic; replace with calibrated values
    for higher accuracy).
    """

    def __init__(self, frame_w: int, frame_h: int) -> None:
        focal = float(frame_w)
        cx, cy = frame_w / 2.0, frame_h / 2.0
        self._cam_mat = np.array(
            [[focal, 0.0, cx],
             [0.0, focal, cy],
             [0.0, 0.0,  1.0]],
            dtype=np.float64,
        )
        self._dist = np.zeros((4, 1), dtype=np.float64)
        self._prev_rvec: np.ndarray | None = None
        self._prev_tvec: np.ndarray | None = None
        self._r_smooth: np.ndarray = np.eye(3)
        self._rvec_smooth: np.ndarray = np.zeros((3, 1), dtype=np.float64)

    def estimate(
        self, landmarks_norm: np.ndarray, frame_w: int, frame_h: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Args:
            landmarks_norm: (N, 2) or (N, 3) array of MediaPipe normalized
                            landmark coordinates in [0, 1].
            frame_w, frame_h: frame dimensions for pixel conversion.

        Returns:
            R_head : (3, 3) rotation matrix — camera → head-local frame.
            tvec   : (3, 1) face centre in camera coordinates [mm].

        Usage:
            gaze_local = R_head.T @ gaze_cam_unit
        """
        img_pts = np.array(
            [[landmarks_norm[i, 0] * frame_w,
              landmarks_norm[i, 1] * frame_h]
             for i in _LM_INDICES],
            dtype=np.float64,
        )

        use_guess = self._prev_rvec is not None
        ok, rvec, tvec = cv2.solvePnP(
            _MODEL_3D,
            img_pts,
            self._cam_mat,
            self._dist,
            rvec=self._prev_rvec,
            tvec=self._prev_tvec,
            useExtrinsicGuess=use_guess,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return self._r_smooth.copy(), np.zeros((3, 1))

        self._prev_rvec = rvec.copy()
        self._prev_tvec = tvec.copy()

        # SLERP in Rodrigues (axis-angle) space — result is guaranteed to stay in SO(3)
        self._rvec_smooth = _R_ALPHA * rvec + (1.0 - _R_ALPHA) * self._rvec_smooth
        self._r_smooth, _ = cv2.Rodrigues(self._rvec_smooth)

        return self._r_smooth.copy(), tvec

    def reset(self) -> None:
        self._prev_rvec = None
        self._prev_tvec = None
        self._r_smooth = np.eye(3)
        self._rvec_smooth = np.zeros((3, 1), dtype=np.float64)


def rotation_to_euler_deg(R: np.ndarray) -> tuple[float, float, float]:
    """
    Decompose a 3×3 rotation matrix to (yaw, pitch, roll) in degrees.

    Uses ZYX intrinsic convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    Intended for diagnostic logging of head pose from solvePnP.

    Returns:
        (yaw_deg, pitch_deg, roll_deg)
    """
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -R[2, 0]))))
    roll  = math.degrees(math.atan2(R[2, 1], R[2, 2]))
    yaw   = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    return yaw, pitch, roll


def iris_to_gaze_cam(
    iris_x_norm: float,
    iris_y_norm: float,
    frame_w: int,
    frame_h: int,
) -> np.ndarray:
    """
    Convert a normalised iris centroid position to a unit gaze direction
    vector in camera space.

    Uses a simple pinhole model (focal = frame_w).  The resulting vector
    points from the camera origin toward where the iris is looking.

    Returns:
        (3,) unit vector in camera space.
    """
    focal = float(frame_w)
    cx, cy = frame_w / 2.0, frame_h / 2.0
    dx = iris_x_norm * frame_w - cx
    dy = iris_y_norm * frame_h - cy
    v = np.array([dx, dy, focal], dtype=np.float64)
    return v / np.linalg.norm(v)


def normalize_gaze(
    iris_x_norm: float,
    iris_y_norm: float,
    R_head: np.ndarray,
    frame_w: int,
    frame_h: int,
) -> tuple[float, float]:
    """
    Convert normalised iris position to head-local gaze direction components.

    Returns:
        (local_x, local_y): first two components of R_head.T @ gaze_cam.
        These are the features fed into the Ridge calibration mapping.
    """
    gaze_cam = iris_to_gaze_cam(iris_x_norm, iris_y_norm, frame_w, frame_h)
    gaze_local = R_head.T @ gaze_cam
    return float(gaze_local[0]), float(gaze_local[1])
