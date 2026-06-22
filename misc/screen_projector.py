"""
Geometric screen projector for gaze estimation.

Uses solvePnP tvec (face position in camera coords) and head rotation matrix
to ray-cast the estimated gaze direction onto a virtual screen plane placed
in front of the camera.

This is a fallback / complement to calibration-based gaze estimation.
The geometry is an approximation: the screen plane is modelled as a flat
surface perpendicular to the camera Z axis at a fixed depth, with the camera
assumed to sit at a known offset from the screen centre.

Physical coordinate convention (camera coords, right-hand):
  X — points right
  Y — points down
  Z — points toward user (depth)

Screen plane is at Z = screen_distance_mm from camera origin.
cam_offset_x_mm / cam_offset_y_mm give the camera's position relative to
the screen centre (e.g. cam_offset_y_mm = -150 means camera is 150 mm above
screen centre).
"""
from __future__ import annotations

import numpy as np


class ScreenProjector:
    """
    Projects estimated gaze to normalised screen coordinates [0, 1] using
    ray–plane intersection.

    Parameters
    ----------
    screen_distance_mm  : distance from camera to screen plane along Z (mm).
    screen_width_mm     : physical screen width (mm).
    screen_height_mm    : physical screen height (mm).
    cam_offset_x_mm     : camera X offset from screen centre (positive = right, mm).
    cam_offset_y_mm     : camera Y offset from screen centre (positive = down, mm).
                          Typically negative (camera above screen).
    """

    def __init__(
        self,
        screen_distance_mm: float = 600.0,
        screen_width_mm: float = 530.0,
        screen_height_mm: float = 300.0,
        cam_offset_x_mm: float = 0.0,
        cam_offset_y_mm: float = -150.0,
    ) -> None:
        self._screen_z = screen_distance_mm
        self._screen_w = screen_width_mm
        self._screen_h = screen_height_mm
        self._cam_ox = cam_offset_x_mm
        self._cam_oy = cam_offset_y_mm

    def project(
        self,
        tvec: np.ndarray,
        R_head: np.ndarray,
        local_x: float,
        local_y: float,
    ) -> tuple[float, float] | None:
        """
        Project gaze to normalised screen coordinates.

        Parameters
        ----------
        tvec    : (3, 1) or (3,) face/eye position in camera coords [mm]
                  (output of solvePnP).
        R_head  : (3, 3) head rotation matrix (camera-to-head, from solvePnP).
        local_x : iris-relative-to-corners X feature (dimensionless ~[-0.5, 0.5]).
        local_y : iris-relative-to-corners Y feature (dimensionless ~[-0.5, 0.5]).

        Returns
        -------
        (norm_x, norm_y) in [0, 1] if the ray intersects the screen plane,
        or None if the face is behind or on the screen plane (Z <= 0 intersection).
        """
        # ── Origin: face/eye position in camera coords ───────────────────────
        origin = np.array(tvec, dtype=np.float64).ravel()  # (3,)

        # ── Gaze direction: head-local → camera space ────────────────────────
        # local_x / local_y are the iris offset features (dimensionless).
        # We treat them as the X/Y components of a gaze vector in head-local
        # space with Z = 1 (forward), then rotate to camera space.
        #
        # Convention (confirmed against solvePnP and head_pose.normalize_gaze):
        #   R_head  maps head-local → camera  (i.e. p_cam = R_head @ p_head)
        #   R_head.T maps camera   → head-local
        # Therefore to convert a head-local gaze vector to camera space: R_head @ v.
        gaze_head_local = np.array([local_x, local_y, 1.0], dtype=np.float64)
        gaze_cam = R_head @ gaze_head_local
        norm = np.linalg.norm(gaze_cam)
        if norm < 1e-10:
            return None
        direction = gaze_cam / norm  # unit ray direction in camera coords

        # ── Ray–plane intersection ────────────────────────────────────────────
        # Plane: Z = screen_z (constant depth plane in camera coords).
        # Ray:   P(t) = origin + t * direction
        # Solve: origin[2] + t * direction[2] = screen_z
        dz = direction[2]
        if abs(dz) < 1e-10:
            return None  # ray parallel to screen plane

        t = (self._screen_z - origin[2]) / dz
        if t <= 0.0:
            return None  # intersection is behind the camera

        hit = origin + t * direction  # (3,) intersection point in camera coords

        # ── Convert hit to screen-relative coords ─────────────────────────────
        # Camera sits at (cam_offset_x, cam_offset_y) relative to screen centre,
        # so screen centre in camera coords is at (-cam_ox, -cam_oy, screen_z).
        screen_centre_x = -self._cam_ox
        screen_centre_y = -self._cam_oy

        rel_x = hit[0] - screen_centre_x   # mm from screen centre (X rightward)
        rel_y = hit[1] - screen_centre_y   # mm from screen centre (Y downward)

        # Normalise to [0, 1]: 0 = left/top, 1 = right/bottom
        norm_x = 0.5 + rel_x / self._screen_w
        norm_y = 0.5 + rel_y / self._screen_h

        return float(norm_x), float(norm_y)
