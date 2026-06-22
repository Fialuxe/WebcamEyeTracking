"""
Webcam gaze source using MediaPipe Face Mesh.

Gaze feature: iris centroid relative to eye-corner midpoint, normalised by
inter-corner distance.  Averaged across both eyes.  This feature is
head-translation-invariant and avoids the parallax errors of the old
iris-absolute-position approach.

Two-stage pipeline:
  Stage 1: iris-relative-to-corners feature (both eyes, normalised)
  Stage 2 (CalibrationManager / Ridge): feature → normalised screen coords

Head pose (solvePnP) is estimated every frame for diagnostic logging only.

Debug mode: when enabled via set_debug_mode(True), the camera preview is
annotated with iris circles, IOD line, and estimated face position
(depth from IOD / iris diameter, lateral X/Y offset from camera axis).

Optionally applies One-Euro filtering for the WebcamFiltered condition.
"""
from __future__ import annotations

import logging
import queue
import sys
import threading
import time
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .base import GazeSource, GazeSample, _put_drop_oldest
from .calibration import CalibrationManager
from .filters import OneEuroFilter
from .head_pose import HeadPoseEstimator, rotation_to_euler_deg
from .. import config

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Iris / eye-corner landmark indices (MediaPipe 478-point model)
# ---------------------------------------------------------------------------

# Iris landmark groups — 5 points each (centre + 4 border)
_IRIS_RIGHT = [468, 469, 470, 471, 472]   # right iris (subject right = camera left)
_IRIS_LEFT  = [473, 474, 475, 476, 477]   # left iris  (subject left  = camera right)

# Eye corner pairs: (outer_temporal, inner_nasal) per eye
_EYE_CORNERS_RIGHT = (33, 133)
_EYE_CORNERS_LEFT  = (263, 362)

# Log one debug line every N processed frames (≈3 lines/sec at 30 fps)
_LOG_EVERY = 10

# ---------------------------------------------------------------------------
# Landmark certainty — Eye Aspect Ratio (Soukupová & Čech, 2016)
# ---------------------------------------------------------------------------

# Six-point EAR index tuples: (outer, upper-outer, upper-inner, inner, lower-inner, lower-outer)
_EAR_IDX_R: tuple = (33, 160, 158, 133, 153, 144)
_EAR_IDX_L: tuple = (263, 387, 385, 362, 380, 374)
# EAR value for a normally-open eye (used as normalisation denominator)
_EAR_OPEN: float = 0.25


def _ear(lm, p1: int, p2: int, p3: int, p4: int, p5: int, p6: int) -> float:
    """Eye Aspect Ratio from six landmark indices (Soukupová & Čech 2016)."""
    def d(a: int, b: int) -> float:
        return ((lm[a].x - lm[b].x) ** 2 + (lm[a].y - lm[b].y) ** 2) ** 0.5
    horiz = d(p1, p4)
    return (d(p2, p6) + d(p3, p5)) / (2.0 * horiz) if horiz > 1e-6 else 0.0


# Representative landmarks spread across the face for mesh quality estimation
_MESH_QUALITY_LMS: tuple = (1, 17, 33, 61, 133, 234, 263, 291, 362, 454, 468, 473)


def _mesh_certainty(lm) -> float:
    """
    Face mesh detection quality [0, 1].

    Tasks API: mean presence score of representative face landmarks (official
    per-landmark confidence from FaceLandmarker).
    Solutions API: mean visibility (typically ~1.0 — the binary detection gate
    already filters bad frames to None before this is called).
    Falls back to 1.0 if neither attribute carries a numeric value (some model
    versions return None for presence/visibility even when the attribute exists).
    """
    presence_vals = [v for i in _MESH_QUALITY_LMS
                     if (v := getattr(lm[i], "presence", None)) is not None]
    if presence_vals:
        return float(sum(presence_vals) / len(presence_vals))

    visibility_vals = [v for i in _MESH_QUALITY_LMS
                       if (v := getattr(lm[i], "visibility", None)) is not None]
    if visibility_vals:
        return float(sum(visibility_vals) / len(visibility_vals))

    return 1.0


def _eye_certainty(lm) -> float:
    """
    Eye openness certainty [0, 1] from EAR (Solutions API fallback only).

    Returns ~0 when eyes are fully closed, ~1.0 when open.
    Note: EAR is confounded with downward gaze (lid-following); prefer
    _eye_certainty_blendshapes() when Tasks API blendshapes are available.
    """
    ear_mean = (_ear(lm, *_EAR_IDX_R) + _ear(lm, *_EAR_IDX_L)) / 2.0
    return float(min(ear_mean / _EAR_OPEN, 1.0))


def _eye_certainty_blendshapes(blendshapes) -> float:
    """
    Eye openness certainty [0, 1] from MediaPipe FaceBlendshapes (Tasks API).

    Uses eyeBlinkLeft / eyeBlinkRight scores, which are trained to distinguish
    genuine blinks from lid-following during downward gaze — eliminating the
    systematic bias of EAR-based approaches.

    blendshapes: result.face_blendshapes[0] — list of Category objects with
                 .category_name (str) and .score (float).
    """
    scores = {b.category_name: b.score for b in blendshapes}
    blink = (scores.get("eyeBlinkLeft", 0.0) + scores.get("eyeBlinkRight", 0.0)) / 2.0
    return float(1.0 - blink)

# ---------------------------------------------------------------------------
# Face position estimation constants
# ---------------------------------------------------------------------------

_IOD_MM: float       = 63.0   # average adult inter-pupillary distance (mm)
_IRIS_DIAM_MM: float = 11.8   # average iris diameter (mm)
_NOSE_TIP_IDX: int   = 1      # MediaPipe nose-tip landmark


# ---------------------------------------------------------------------------
# Pure helpers (module-level, easily unit-tested)
# ---------------------------------------------------------------------------

def _iris_rel(lm, iris_ids: list[int], outer: int, inner: int) -> tuple[float, float]:
    """
    Iris centroid position relative to the midpoint of two eye corner landmarks,
    normalised by the inter-corner distance (approximately eye width).

    Returns (rel_x, rel_y): dimensionless, ~[-0.5, 0.5] range.
    Head-translation-invariant because iris and corners shift together.
    """
    iris_x = sum(lm[i].x for i in iris_ids) / len(iris_ids)
    iris_y = sum(lm[i].y for i in iris_ids) / len(iris_ids)
    mid_x = (lm[outer].x + lm[inner].x) / 2.0
    mid_y = (lm[outer].y + lm[inner].y) / 2.0
    dx = lm[outer].x - lm[inner].x
    dy = lm[outer].y - lm[inner].y
    eye_w = (dx * dx + dy * dy) ** 0.5
    if eye_w < 1e-6:
        return 0.0, 0.0
    return (iris_x - mid_x) / eye_w, (iris_y - mid_y) / eye_w


@dataclass
class FacePosition:
    """3-D face/eye position estimated from landmark geometry."""
    dist_iod_mm:  float   # depth from IOD method
    dist_iris_mm: float   # depth from iris-diameter method
    x_mm:         float   # lateral offset from camera optical axis (+ = right)
    y_mm:         float   # vertical offset from camera optical axis (+ = down)
    iod_px:       float   # inter-ocular distance in pixels
    iris_r_px:    float   # right iris radius in pixels
    iris_l_px:    float   # left iris radius in pixels


@dataclass
class FaceMetrics:
    """Lightweight face summary for the UI positioning guide (updated every frame)."""
    iod_norm: float   # IOD as fraction of camera frame width  (0→1)
    face_cx:  float   # iris-midpoint X in normalised [0, 1] space
    face_cy:  float   # iris-midpoint Y in normalised [0, 1] space


def _estimate_face_pos(lm, frame_w: int, frame_h: int) -> FacePosition:
    """
    Estimate face 3-D position from landmark geometry using two independent
    depth cues:

    1. IOD (inter-ocular distance): measures pixel distance between the two
       iris centres.  Physical IOD ≈ 63 mm ⇒ depth = IOD_mm * f / iod_px.

    2. Iris diameter: measures pixel radius of each iris.  Physical iris
       diameter ≈ 11.8 mm ⇒ depth = iris_diam_mm * f / (2 * r_px).

    Both cues use the pinhole model with focal length ≈ frame_w.

    Lateral X/Y position is derived from the nose-tip pixel offset from the
    image centre, scaled by the average depth estimate.
    """
    focal = float(frame_w)
    cx_px = frame_w / 2.0
    cy_px = frame_h / 2.0

    # Iris centres in pixels
    r_x = lm[468].x * frame_w
    r_y = lm[468].y * frame_h
    l_x = lm[473].x * frame_w
    l_y = lm[473].y * frame_h

    # ── Depth from IOD ──────────────────────────────────────────────────────
    iod_px   = ((r_x - l_x) ** 2 + (r_y - l_y) ** 2) ** 0.5
    dist_iod = (_IOD_MM * focal / iod_px) if iod_px > 1.0 else 0.0

    # ── Depth from iris diameter ─────────────────────────────────────────────
    def _iris_r(center_x, center_y, border_ids):
        return sum(
            ((lm[i].x * frame_w - center_x) ** 2 + (lm[i].y * frame_h - center_y) ** 2) ** 0.5
            for i in border_ids
        ) / len(border_ids)

    iris_r_px = _iris_r(r_x, r_y, [469, 470, 471, 472])
    iris_l_px = _iris_r(l_x, l_y, [474, 475, 476, 477])
    iris_avg  = (iris_r_px + iris_l_px) / 2.0
    dist_iris = (_IRIS_DIAM_MM * focal / (2.0 * iris_avg)) if iris_avg > 0.5 else 0.0

    # ── Lateral position from nose tip ───────────────────────────────────────
    dist_avg = (dist_iod + dist_iris) / 2.0
    nose_x   = lm[_NOSE_TIP_IDX].x * frame_w
    nose_y   = lm[_NOSE_TIP_IDX].y * frame_h
    x_mm = (nose_x - cx_px) * dist_avg / focal
    y_mm = (nose_y - cy_px) * dist_avg / focal

    return FacePosition(
        dist_iod_mm=dist_iod,
        dist_iris_mm=dist_iris,
        x_mm=x_mm,
        y_mm=y_mm,
        iod_px=iod_px,
        iris_r_px=iris_r_px,
        iris_l_px=iris_l_px,
    )


# ---------------------------------------------------------------------------
# Backend enum
# ---------------------------------------------------------------------------

class FaceMeshBackend(Enum):
    SOLUTIONS = "solutions"   # legacy mp.solutions.face_mesh (default)
    TASKS     = "tasks"       # mp.tasks.vision.FaceLandmarker (requires model file)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class WebcamGazeSource(GazeSource):
    """
    Runs MediaPipe Face Mesh on a background thread at camera frame rate.
    Emits GazeSample only when calibrated.
    """

    def __init__(
        self,
        condition: str = "Webcam",
        camera_id: int = 0,
        calibration: CalibrationManager | None = None,
        use_filter: bool = False,
        backend: FaceMeshBackend = FaceMeshBackend.SOLUTIONS,
        model_path: str = config.FACEMESH_MODEL_PATH,
    ) -> None:
        self._condition    = condition
        self._camera_id    = camera_id
        self._calibration  = calibration if calibration is not None else CalibrationManager()
        self._use_filter   = use_filter
        self._backend      = backend
        self._model_path   = model_path
        self._filter_x     = OneEuroFilter(config.ONE_EURO_MIN_CUTOFF, config.ONE_EURO_BETA)
        self._filter_y     = OneEuroFilter(config.ONE_EURO_MIN_CUTOFF, config.ONE_EURO_BETA)
        self._thread: threading.Thread | None = None
        self._stop_event   = threading.Event()
        self._frame_count: int = 0
        self._debug_mode: bool = False

        # Latest gaze feature — shared with calibration UI
        self._latest_local: tuple[float, float] | None = None
        self._local_lock = threading.Lock()

        # Latest face position metrics — shared with face-guide UI overlay
        self._latest_face_metrics: FaceMetrics | None = None
        self._metrics_lock = threading.Lock()

        # Frame callback for UI camera preview
        self._frame_callback = None
        self._frame_lock     = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def calibration(self) -> CalibrationManager:
        return self._calibration

    def start(self, out_queue: queue.Queue) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, args=(out_queue,), daemon=True, name="webcam-gaze"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def get_local_gaze(self) -> tuple[float, float] | None:
        """Return the most recent gaze feature (local_x, local_y) or None."""
        with self._local_lock:
            return self._latest_local

    def get_face_metrics(self) -> FaceMetrics | None:
        """Return the most recent face-position metrics, or None if no face detected."""
        with self._metrics_lock:
            return self._latest_face_metrics

    def _update_face_metrics(self, lm, frame_w: int) -> None:
        """Cache normalised IOD and face-centre from iris landmarks (cheap, runs every frame)."""
        r_x, r_y = lm[468].x, lm[468].y
        l_x, l_y = lm[473].x, lm[473].y
        iod_norm = ((r_x - l_x) ** 2 + (r_y - l_y) ** 2) ** 0.5
        with self._metrics_lock:
            self._latest_face_metrics = FaceMetrics(
                iod_norm=iod_norm,
                face_cx=(r_x + l_x) / 2.0,
                face_cy=(r_y + l_y) / 2.0,
            )

    def _clear_face_metrics(self) -> None:
        with self._metrics_lock:
            self._latest_face_metrics = None

    def set_frame_callback(self, cb: "Callable[[np.ndarray], None] | None") -> None:
        """Register a BGR-frame callback for the UI preview. Thread-safe."""
        with self._frame_lock:
            self._frame_callback = cb

    def set_debug_mode(self, enabled: bool) -> None:
        """
        Enable/disable the visual debug overlay on the camera preview.

        When enabled, each preview frame is annotated with:
          • Green circles on both irises
          • Cyan line showing the IOD measurement
          • Text panel: depth (from IOD and iris diameter), lateral X/Y offset
        """
        self._debug_mode = enabled

    # ------------------------------------------------------------------
    # Internal: camera thread entry
    # ------------------------------------------------------------------

    def _run(self, out_queue: queue.Queue) -> None:
        import cv2
        # CAP_DSHOW is significantly faster to open on Windows than the
        # default MSMF backend (typically <500 ms vs 5-15 s).
        if sys.platform == "win32":
            cap = cv2.VideoCapture(self._camera_id, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(self._camera_id)

        if not cap.isOpened():
            print(
                f"[webcam] ERROR: cannot open camera {self._camera_id}",
                file=sys.stderr,
            )
            return
        try:
            if self._backend == FaceMeshBackend.TASKS:
                self._run_tasks(cap, out_queue)
            else:
                self._run_solutions(cap, out_queue)
        finally:
            cap.release()

    # ------------------------------------------------------------------
    # Internal: debug overlay
    # ------------------------------------------------------------------

    def _draw_debug_overlay(
        self, frame: np.ndarray, lm, frame_w: int, frame_h: int
    ) -> np.ndarray:
        """Return an annotated copy of frame with face position info."""
        import cv2
        out = frame.copy()
        fp  = _estimate_face_pos(lm, frame_w, frame_h)

        # ── Iris circles ────────────────────────────────────────────────────
        for centre_id, border_ids, col in (
            (468, [469, 470, 471, 472], (0, 230, 0)),
            (473, [474, 475, 476, 477], (0, 180, 0)),
        ):
            cx = int(lm[centre_id].x * frame_w)
            cy = int(lm[centre_id].y * frame_h)
            r  = int(sum(
                ((lm[i].x * frame_w - cx) ** 2 + (lm[i].y * frame_h - cy) ** 2) ** 0.5
                for i in border_ids
            ) / len(border_ids))
            cv2.circle(out, (cx, cy), max(r, 2), col, 2, cv2.LINE_AA)
            cv2.circle(out, (cx, cy), 2, (0, 60, 255), -1)

        # ── IOD line ────────────────────────────────────────────────────────
        pt_r = (int(lm[468].x * frame_w), int(lm[468].y * frame_h))
        pt_l = (int(lm[473].x * frame_w), int(lm[473].y * frame_h))
        cv2.line(out, pt_r, pt_l, (0, 220, 220), 1, cv2.LINE_AA)

        # ── Text panel ──────────────────────────────────────────────────────
        lines = [
            f"Dist IOD : {fp.dist_iod_mm:5.0f} mm   Dist iris: {fp.dist_iris_mm:5.0f} mm",
            f"X : {fp.x_mm:+6.0f} mm        Y : {fp.y_mm:+6.0f} mm",
            f"IOD: {fp.iod_px:5.1f} px   iris R:{fp.iris_r_px:4.1f}  L:{fp.iris_l_px:4.1f} px",
        ]
        font    = cv2.FONT_HERSHEY_SIMPLEX
        fscale  = 0.50
        thick   = 1
        line_h  = 22
        pad     = 6
        tw      = max(cv2.getTextSize(ln, font, fscale, thick)[0][0] for ln in lines)
        y0      = 8
        cv2.rectangle(
            out,
            (4, y0 - pad),
            (4 + tw + 2 * pad, y0 + len(lines) * line_h + pad),
            (0, 0, 0), -1,
        )
        for i, ln in enumerate(lines):
            cv2.putText(
                out, ln,
                (4 + pad, y0 + i * line_h + 16),
                font, fscale, (0, 230, 0), thick, cv2.LINE_AA,
            )
        return out

    # ------------------------------------------------------------------
    # Internal: MediaPipe Solutions backend loop
    # ------------------------------------------------------------------

    def _run_solutions(self, cap, out_queue: queue.Queue) -> None:
        import cv2
        import mediapipe as mp

        mp_face_mesh    = mp.solutions.face_mesh
        head_estimator: HeadPoseEstimator | None = None

        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_mesh:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.001)
                    continue

                frame_h, frame_w = frame.shape[:2]
                if head_estimator is None:
                    head_estimator = HeadPoseEstimator(frame_w, frame_h)

                rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = face_mesh.process(rgb)
                lm_list = (
                    result.multi_face_landmarks[0].landmark
                    if result.multi_face_landmarks else None
                )

                with self._frame_lock:
                    cb = self._frame_callback
                if cb is not None:
                    if self._debug_mode and lm_list is not None:
                        display = self._draw_debug_overlay(frame, lm_list, frame_w, frame_h)
                    else:
                        display = frame.copy()
                    cb(display)

                if lm_list is None:
                    self._clear_face_metrics()
                    continue
                self._process_landmarks(lm_list, frame_w, frame_h, head_estimator, out_queue)

    # ------------------------------------------------------------------
    # Internal: gaze feature extraction and calibration
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
        # Update face-position metrics for the UI guide (cheap, every frame)
        self._update_face_metrics(lm, frame_w)

        # Stage 1: iris-relative-to-corners feature, averaged across both eyes
        rel_r  = _iris_rel(lm, _IRIS_RIGHT, *_EYE_CORNERS_RIGHT)
        rel_l  = _iris_rel(lm, _IRIS_LEFT,  *_EYE_CORNERS_LEFT)
        local_x = (rel_r[0] + rel_l[0]) / 2.0
        local_y = (rel_r[1] + rel_l[1]) / 2.0

        # Head pose — computed every frame for CSV output
        lm_arr        = np.array([[l.x, l.y] for l in lm])
        R_head, _tvec = head_estimator.estimate(lm_arr, frame_w, frame_h)
        yaw, pitch, roll = rotation_to_euler_deg(R_head)

        # Periodic debug logging
        self._frame_count += 1
        if self._frame_count % _LOG_EVERY == 0:
            fp = _estimate_face_pos(lm, frame_w, frame_h)
            _log.debug(
                "rel_r=(%.4f,%.4f) rel_l=(%.4f,%.4f) local=(%.4f,%.4f) "
                "yaw=%.1f° pitch=%.1f° roll=%.1f° "
                "dist=%.0fmm x=%+.0fmm y=%+.0fmm",
                rel_r[0], rel_r[1], rel_l[0], rel_l[1],
                local_x, local_y,
                yaw, pitch, roll,
                fp.dist_iod_mm, fp.x_mm, fp.y_mm,
            )

        # Expose to calibration UI
        with self._local_lock:
            self._latest_local = (local_x, local_y)

        if not self._calibration.is_calibrated:
            return

        # Stage 2: Ridge calibration → screen coords
        gx, gy = self._calibration.predict(local_x, local_y)

        if self._frame_count % _LOG_EVERY == 0:
            _log.debug("screen=(%.4f,%.4f) [calibrated]", gx, gy)

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
            source="webcam",
            condition=self._condition,
            ts_wall_ms=time.time() * 1000,
            ts_mono_ns=time.monotonic_ns(),
            head_yaw=yaw,
            head_pitch=pitch,
            head_roll=roll,
        )
        _put_drop_oldest(out_queue, sample)

    # ------------------------------------------------------------------
    # Internal: MediaPipe Tasks backend loop
    # ------------------------------------------------------------------

    def _run_tasks(self, cap, out_queue: queue.Queue) -> None:
        import cv2
        import os
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        if not os.path.exists(self._model_path):
            print(
                f"[webcam] Tasks backend: model file not found: {self._model_path}",
                file=sys.stderr,
            )
            print(
                "  Download: https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/1/face_landmarker.task",
                file=sys.stderr,
            )
            print("[webcam] Falling back to Solutions API.", file=sys.stderr)
            self._run_solutions(cap, out_queue)
            return

        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=self._model_path),
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,
        )
        head_estimator = None
        with mp_vision.FaceLandmarker.create_from_options(options) as landmarker:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.001)
                    continue

                frame_h, frame_w = frame.shape[:2]
                if head_estimator is None:
                    head_estimator = HeadPoseEstimator(frame_w, frame_h)

                rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result      = landmarker.detect(mp_image)
                lm_list     = result.face_landmarks[0] if result.face_landmarks else None
                blendshapes = result.face_blendshapes[0] if result.face_blendshapes else None

                with self._frame_lock:
                    cb = self._frame_callback
                if cb is not None:
                    if self._debug_mode and lm_list is not None:
                        display = self._draw_debug_overlay(frame, lm_list, frame_w, frame_h)
                    else:
                        display = frame.copy()
                    cb(display)

                if lm_list is None:
                    self._clear_face_metrics()
                    continue
                self._process_landmarks(lm_list, frame_w, frame_h, head_estimator, out_queue, blendshapes)
