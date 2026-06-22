"""
WebcamGazeSourceBeta2 — extends WebcamGazeSource with PACE (click-based RLS) calibration.

Prediction priority (per frame):
  1. CalibrationManager calibrated AND PACE ready  → PACE.predict()
  2. CalibrationManager calibrated, PACE not ready  → CalibrationManager.predict()
  3. pure_implicit AND PACE ready                   → PACE.predict()
  4. Otherwise                                       → drop frame (not calibrated enough)

NOTE (pure_implicit mode): PACE output is unreliable until min_updates clicks have
been registered. During that initial period frames are dropped entirely — the caller
should inform the user that ~min_updates clicks are needed before gaze appears.
"""
from __future__ import annotations

import logging
import queue
import threading
import time

import numpy as np

from .base import GazeSample, _put_drop_oldest
from .calibration import CalibrationManager
from .pace_calibration import PACECalibration
from .head_pose import rotation_to_euler_deg
from .webcam_source import (
    WebcamGazeSource,
    FaceMeshBackend,
    _iris_rel,
    _IRIS_RIGHT,
    _IRIS_LEFT,
    _EYE_CORNERS_RIGHT,
    _EYE_CORNERS_LEFT,
    _mesh_certainty,
    _eye_certainty,
    _eye_certainty_blendshapes,
)
from .. import config

_log = logging.getLogger(__name__)

# Log one line every N processed frames (matches parent constant)
_LOG_EVERY = 10


class WebcamGazeSourceBeta2(WebcamGazeSource):
    """
    WebcamGazeSource extended with PACE (Passive Adaptive Calibration Extension)
    via recursive-least-squares updates on every user mouse click.

    Parameters
    ----------
    use_pace : bool
        When True, PACE prediction is used once min_updates is reached.
    pace_min_updates : int
        Number of clicks required before PACE kicks in as the predictor.
    pure_implicit : bool
        When True, skip explicit (grid) calibration and rely on text calibration
        and/or PACE from identity prior.  When False (default), explicit
        calibration is required first and PACE refines from there.
    All other parameters are passed through to WebcamGazeSource.
    """

    def __init__(
        self,
        condition: str = "Webcam",
        camera_id: int = 0,
        calibration: CalibrationManager | None = None,
        use_filter: bool = False,
        backend: FaceMeshBackend = FaceMeshBackend.SOLUTIONS,
        model_path: str = config.FACEMESH_MODEL_PATH,
        use_pace: bool = True,
        pace_min_updates: int = 5,
        pure_implicit: bool = False,
    ) -> None:
        super().__init__(
            condition=condition,
            camera_id=camera_id,
            calibration=calibration,
            use_filter=use_filter,
            backend=backend,
            model_path=model_path,
        )
        self._use_pace = use_pace
        self._pace_min_updates = pace_min_updates
        self._pure_implicit = pure_implicit

        # PACE state — lock guards concurrent read (camera thread) / write (UI thread)
        self._pace_lock = threading.Lock()
        self._pace = PACECalibration(min_updates=pace_min_updates)

        _log.info(
            "WebcamGazeSourceBeta2 init: use_pace=%s pace_min_updates=%d pure_implicit=%s",
            use_pace,
            pace_min_updates,
            pure_implicit,
        )

    # ------------------------------------------------------------------
    # PACE public API
    # ------------------------------------------------------------------

    def on_click(self, screen_x_norm: float, screen_y_norm: float) -> None:
        """
        Called by the UI on every mouse click.  Gets the latest local gaze
        feature and feeds it to PACE as a labelled sample.

        Thread-safe: called on the Tk main thread, reads _latest_local which
        is written by the camera thread under _local_lock.
        """
        local = self.get_local_gaze()
        if local is None:
            _log.debug(
                "PACE: click at (%.3f,%.3f) ignored — no local gaze available",
                screen_x_norm,
                screen_y_norm,
            )
            return
        with self._pace_lock:
            self._pace.update(local[0], local[1], screen_x_norm, screen_y_norm)
            count = self._pace.update_count
        _log.debug(
            "PACE update #%d via click (%.3f,%.3f)", count, screen_x_norm, screen_y_norm
        )

    def promote_to_pace(self) -> None:
        """
        Re-initialise PACE using the fitted CalibrationManager as a warm-start
        ridge prior.  Call this after explicit or text calibration succeeds so
        PACE begins from a good linear approximation rather than from identity.
        """
        warm = self._calibration if self._calibration.is_calibrated else None
        with self._pace_lock:
            self._pace = PACECalibration(
                min_updates=self._pace_min_updates,
                warm_ridge=warm,
            )
        _log.info(
            "PACE promoted to warm-start (warm_ridge=%s)", "yes" if warm else "no"
        )

    def get_pace(self) -> PACECalibration:
        """Return the PACE instance (for status polling by UI)."""
        return self._pace

    # ------------------------------------------------------------------
    # Override: gaze prediction stage
    # ------------------------------------------------------------------

    def _process_landmarks(
        self,
        lm,
        frame_w: int,
        frame_h: int,
        head_estimator,
        out_queue: queue.Queue,
        blendshapes=None,
    ) -> None:
        """
        Override of parent Stage 2 only.  Stage 1 (feature extraction) and
        storage of _latest_local are reproduced here so that calibration UIs
        and on_click() always have a fresh feature regardless of calib state.

        Head-pose logging from the parent is intentionally omitted to reduce
        overhead; the parent's version runs every _LOG_EVERY frames if needed.
        """
        # Face metrics for UI guide and Unity /face/metrics (same as parent)
        self._update_face_metrics(lm, frame_w)

        # Stage 1: iris-relative-to-corners feature, averaged across both eyes
        rel_r = _iris_rel(lm, _IRIS_RIGHT, *_EYE_CORNERS_RIGHT)
        rel_l = _iris_rel(lm, _IRIS_LEFT, *_EYE_CORNERS_LEFT)
        local_x = (rel_r[0] + rel_l[0]) / 2.0
        local_y = (rel_r[1] + rel_l[1]) / 2.0

        # Head pose — computed every frame for CSV output
        lm_arr = np.array([[l.x, l.y] for l in lm])
        R_head, _tvec = head_estimator.estimate(lm_arr, frame_w, frame_h)
        yaw, pitch, roll = rotation_to_euler_deg(R_head)

        # Always store local gaze BEFORE any calibration gate so that
        # get_local_gaze() and on_click() always see a current sample.
        with self._local_lock:
            self._latest_local = (local_x, local_y)

        self._frame_count += 1
        if self._frame_count % _LOG_EVERY == 0:
            _log.debug(
                "rel_r=(%.4f,%.4f) rel_l=(%.4f,%.4f) local=(%.4f,%.4f)",
                rel_r[0], rel_r[1], rel_l[0], rel_l[1],
                local_x, local_y,
            )

        # Stage 2: select predictor
        # predict() is called inside the lock to prevent a race with on_click()'s
        # update() — both operate on the same numpy arrays in PACECalibration.
        predicted: tuple[float, float] | None = None
        pace_used: bool = False

        with self._pace_lock:
            pace_ready = self._use_pace and self._pace.is_ready
            if pace_ready and (self._calibration.is_calibrated or self._pure_implicit):
                predicted = self._pace.predict(local_x, local_y)
                pace_used = True

        if predicted is None:
            if self._calibration.is_calibrated:
                predicted = self._calibration.predict(local_x, local_y)
            # pure_implicit + PACE not ready: drop frame

        if predicted is None:
            return

        gx, gy = predicted

        if self._frame_count % _LOG_EVERY == 0:
            _log.debug("screen=(%.4f,%.4f) [beta2]", gx, gy)

        if self._use_filter:
            t = time.monotonic()
            gx = self._filter_x.filter(gx, t)
            gy = self._filter_y.filter(gy, t)

        source_tag = "webcam:pace" if pace_used else "webcam:ridge"
        eye_cert = (
            _eye_certainty_blendshapes(blendshapes)
            if blendshapes is not None
            else _eye_certainty(lm)
        )
        sample = GazeSample(
            x=gx,
            y=gy,
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
