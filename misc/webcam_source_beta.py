"""
Beta gaze source with optional roll correction, TPS calibration, tvec features,
and geometric screen projection.

WebcamGazeSourceBeta inherits WebcamGazeSource and overrides only
_process_landmarks() — the parent's _run_solutions() / _run_tasks() loops
call self._process_landmarks(...) via dynamic dispatch so the override runs
automatically without duplicating the camera loop.

Feature flags
-------------
roll_correction     : de-rotate iris features by solvePnP roll angle before
                      averaging.  Reduces apparent lateral drift during head tilts.
use_tps             : use TPSCalibrationManager (thin plate spline) instead of
                      the default RidgeCV.
use_tvec_features   : add solvePnP tvec (X, Y, Z) to TPS feature vector.
                      Only effective when use_tps=True; silently ignored otherwise.
screen_projector    : ScreenProjector instance.  When provided and calibration
                      has NOT been performed, the projector is used to produce
                      gaze coordinates geometrically (no calibration required).
                      When the calibration IS active the calibration takes precedence.

tvec and calibration
--------------------
calib_window.py calls:
    source.calibration.add_point(local_x, local_y, tx, ty)
with exactly 4 args and cannot pass tvec.  When use_tvec_features=True and
use_tps=True, each frame calls TPSCalibrationManager.set_current_tvec(tvec)
so the stashed tvec is read inside add_point().
"""
from __future__ import annotations

import logging
import math
import time

import numpy as np

from .webcam_source import (
    WebcamGazeSource, FaceMeshBackend,
    _iris_rel, _IRIS_RIGHT, _IRIS_LEFT, _EYE_CORNERS_RIGHT, _EYE_CORNERS_LEFT,
    _LOG_EVERY, _mesh_certainty, _eye_certainty, _eye_certainty_blendshapes,
)
from .calibration import CalibrationManager
from .tps_calibration import TPSCalibrationManager
from .head_pose import HeadPoseEstimator, rotation_to_euler_deg
from .base import GazeSample, _put_drop_oldest
from .screen_projector import ScreenProjector
from .. import config

_log = logging.getLogger(__name__)


class WebcamGazeSourceBeta(WebcamGazeSource):
    """
    Drop-in replacement for WebcamGazeSource with four optional beta features.

    All parent public API is preserved; the only change is the internal
    _process_landmarks() implementation.
    """

    def __init__(
        self,
        condition: str = "Webcam",
        camera_id: int = 0,
        calibration: CalibrationManager | TPSCalibrationManager | None = None,
        use_filter: bool = False,
        backend: FaceMeshBackend = FaceMeshBackend.SOLUTIONS,
        model_path: str = config.FACEMESH_MODEL_PATH,
        # ── Beta feature flags ────────────────────────────────────────────────
        roll_correction: bool = True,
        use_tps: bool = True,
        use_tvec_features: bool = True,
        screen_projector: ScreenProjector | None = None,
    ) -> None:
        # Choose calibration manager based on flags
        if calibration is None:
            if use_tps:
                calibration = TPSCalibrationManager(use_tvec=use_tvec_features)
            else:
                calibration = CalibrationManager()

        super().__init__(
            condition=condition,
            camera_id=camera_id,
            calibration=calibration,
            use_filter=use_filter,
            backend=backend,
            model_path=model_path,
        )

        self._roll_correction = roll_correction
        self._use_tps = use_tps
        self._use_tvec_features = use_tvec_features
        self._screen_projector = screen_projector

        # True when both TPS and tvec features are active; gates per-frame
        # set_current_tvec() and tvec-aware predict() calls.
        self._tvec_capable: bool = (
            isinstance(self._calibration, TPSCalibrationManager)
            and use_tvec_features
        )

        # Latest raw tvec for projector and tvec feature injection
        self._latest_tvec: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Override: core landmark processing
    # ------------------------------------------------------------------

    def _process_landmarks(
        self,
        lm,
        frame_w: int,
        frame_h: int,
        head_estimator: HeadPoseEstimator,
        out_queue,
        blendshapes=None,
    ) -> None:
        # Face metrics for UI guide and Unity /face/metrics (same as parent)
        self._update_face_metrics(lm, frame_w)

        # ── Stage 1: iris-relative-to-corners (same as parent) ──────────────
        rel_r = _iris_rel(lm, _IRIS_RIGHT, *_EYE_CORNERS_RIGHT)
        rel_l = _iris_rel(lm, _IRIS_LEFT,  *_EYE_CORNERS_LEFT)

        # ── Head pose ────────────────────────────────────────────────────────
        lm_arr = np.array([[l.x, l.y] for l in lm])
        R_head, tvec = head_estimator.estimate(lm_arr, frame_w, frame_h)

        # Stash tvec for projector and calibration injection
        self._latest_tvec = tvec

        yaw, pitch, roll = rotation_to_euler_deg(R_head)  # computed every frame for CSV

        # ── Feature 1: Roll correction ────────────────────────────────────────
        if self._roll_correction:
            roll_deg = roll
            roll_rad = math.radians(roll_deg)
            cos_r = math.cos(roll_rad)
            sin_r = math.sin(roll_rad)
            # De-rotate by -roll: multiply by 2-D rotation matrix for -roll_deg
            rx_r =  cos_r * rel_r[0] + sin_r * rel_r[1]
            ry_r = -sin_r * rel_r[0] + cos_r * rel_r[1]
            rx_l =  cos_r * rel_l[0] + sin_r * rel_l[1]
            ry_l = -sin_r * rel_l[0] + cos_r * rel_l[1]
            rel_r = (rx_r, ry_r)
            rel_l = (rx_l, ry_l)

        local_x = (rel_r[0] + rel_l[0]) / 2.0
        local_y = (rel_r[1] + rel_l[1]) / 2.0

        # ── Feature 2: tvec injection into TPS calibration ───────────────────
        # Stash tvec in the manager so add_point() (called from calib_window.py
        # on the Tk thread) can pick it up without a signature change.
        if self._tvec_capable:
            tvec_flat = np.array(tvec, dtype=np.float64).ravel()
            self._calibration.set_current_tvec(tvec_flat)  # type: ignore[attr-defined]

        # ── Periodic debug logging ────────────────────────────────────────────
        self._frame_count += 1
        if self._frame_count % _LOG_EVERY == 0:
            tvec_flat_log = np.array(tvec).ravel()
            _log.debug(
                "beta rel_r=(%.4f,%.4f) rel_l=(%.4f,%.4f) local=(%.4f,%.4f) "
                "roll_corr=%s yaw=%.1f° pitch=%.1f° roll=%.1f° "
                "tvec=(%.0f,%.0f,%.0f)mm",
                rel_r[0], rel_r[1], rel_l[0], rel_l[1],
                local_x, local_y,
                self._roll_correction,
                yaw, pitch, roll,
                tvec_flat_log[0], tvec_flat_log[1], tvec_flat_log[2],
            )

        # Expose to calibration UI (2-tuple only; calib_window expects this)
        with self._local_lock:
            self._latest_local = (local_x, local_y)

        # ── Gaze output: calibration first, then projector, then nothing ─────
        gx: float | None = None
        gy: float | None = None
        source_tag: str = "webcam"

        if self._calibration.is_calibrated:
            # ── Stage 2a: calibration (TPS or Ridge) ─────────────────────────
            if self._tvec_capable:
                tvec_arr = np.array(tvec, dtype=np.float64).ravel()
                gx, gy = self._calibration.predict(local_x, local_y, tvec=tvec_arr)  # type: ignore[call-arg]
            else:
                gx, gy = self._calibration.predict(local_x, local_y)

            if isinstance(self._calibration, TPSCalibrationManager):
                source_tag = "webcam:tps"
            else:
                source_tag = "webcam:ridge"

            if self._frame_count % _LOG_EVERY == 0:
                _log.debug("screen=(%.4f,%.4f) [calibration]", gx, gy)

        elif self._screen_projector is not None:
            # ── Stage 2b: geometric projection (no calibration) ───────────────
            result = self._screen_projector.project(tvec, R_head, local_x, local_y)
            if result is not None:
                gx, gy = result
                source_tag = "webcam:geo"
                if self._frame_count % _LOG_EVERY == 0:
                    _log.debug("screen=(%.4f,%.4f) [projector]", gx, gy)

        if gx is None:
            return  # no gaze output available

        # ── Optional filter ───────────────────────────────────────────────────
        if self._use_filter:
            t  = time.monotonic()
            gx = self._filter_x.filter(gx, t)
            gy = self._filter_y.filter(gy, t)

        eye_cert = (
            _eye_certainty_blendshapes(blendshapes)
            if blendshapes is not None
            else _eye_certainty(lm)
        )
        sample = GazeSample(
            x=gx, y=gy,
            mesh_certainty=_mesh_certainty(lm),
            eye_certainty=eye_cert,
            source=source_tag,
            condition=self._condition,
            ts_wall_ms=time.time() * 1000,
            ts_mono_ns=time.monotonic_ns(),
            head_yaw=yaw,
            head_pitch=pitch,
            head_roll=roll,
        )
        _put_drop_oldest(out_queue, sample)
