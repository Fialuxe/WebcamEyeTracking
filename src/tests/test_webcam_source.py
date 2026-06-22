"""
Unit tests for webcam_source pure-logic helpers.

Hardware tests (require a physical camera) are marked @pytest.mark.slow and
are skipped by default.  Run them explicitly with:

    pytest -m slow src/tests/test_webcam_source.py -s
"""
import sys
import time

import pytest

from main.gaze.webcam_source import (
    FacePosition,
    _IOD_MM,
    _IRIS_DIAM_MM,
    _estimate_face_pos,
    _iris_rel,
)


# ---------------------------------------------------------------------------
# Minimal landmark mock
# ---------------------------------------------------------------------------

class _Lm:
    __slots__ = ("x", "y")

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


def _lm_grid(overrides: dict | None = None, n: int = 478) -> list:
    """Return *n* landmarks defaulting to (0.5, 0.5) with {index: (x, y)} overrides."""
    lm = [_Lm(0.5, 0.5) for _ in range(n)]
    for idx, (x, y) in (overrides or {}).items():
        lm[idx] = _Lm(x, y)
    return lm


# ---------------------------------------------------------------------------
# _iris_rel
# ---------------------------------------------------------------------------

class TestIrisRel:
    """Gaze feature: iris centroid relative to eye-corner midpoint."""

    def _lm_eye(self, iris_x, iris_y, outer_x=0.4, inner_x=0.6, y=0.5):
        """Build landmarks for right eye with given iris and corner positions."""
        ov = {i: (iris_x, iris_y) for i in [468, 469, 470, 471, 472]}
        ov[33]  = (outer_x, y)
        ov[133] = (inner_x, y)
        return _lm_grid(ov)

    def test_centred_iris_near_zero_x(self):
        """Iris at midpoint of corners → |rel_x| < 0.05."""
        lm = self._lm_eye(0.5, 0.5)
        rx, _ = _iris_rel(lm, [468, 469, 470, 471, 472], 33, 133)
        assert abs(rx) < 0.05

    def test_iris_toward_outer_gives_negative_x(self):
        """Iris shifted toward outer corner (smaller x) → rel_x < 0."""
        lm = self._lm_eye(iris_x=0.43, iris_y=0.5)
        rx, _ = _iris_rel(lm, [468, 469, 470, 471, 472], 33, 133)
        assert rx < 0.0

    def test_iris_toward_inner_gives_positive_x(self):
        """Iris shifted toward inner corner (larger x) → rel_x > 0."""
        lm = self._lm_eye(iris_x=0.57, iris_y=0.5)
        rx, _ = _iris_rel(lm, [468, 469, 470, 471, 472], 33, 133)
        assert rx > 0.0

    def test_iris_below_midpoint_gives_positive_y(self):
        """Iris below corner midpoint (higher y) → rel_y > 0."""
        lm = self._lm_eye(iris_x=0.5, iris_y=0.56)
        _, ry = _iris_rel(lm, [468, 469, 470, 471, 472], 33, 133)
        assert ry > 0.0

    def test_collapsed_corners_safe_return(self):
        """outer == inner → eye_w = 0 → safe (0, 0)."""
        lm = self._lm_eye(0.5, 0.5, outer_x=0.5, inner_x=0.5)
        assert _iris_rel(lm, [468, 469, 470, 471, 472], 33, 133) == (0.0, 0.0)

    def test_translation_invariant(self):
        """Shifting all landmarks by the same offset should not change rel."""
        def make(shift):
            ov = {i: (0.45 + shift, 0.5) for i in [468, 469, 470, 471, 472]}
            ov[33]  = (0.40 + shift, 0.5)
            ov[133] = (0.60 + shift, 0.5)
            return _lm_grid(ov)

        rx0, _ = _iris_rel(make(0.00), [468, 469, 470, 471, 472], 33, 133)
        rx1, _ = _iris_rel(make(0.10), [468, 469, 470, 471, 472], 33, 133)
        assert abs(rx0 - rx1) < 1e-9

    def test_scale_invariant(self):
        """Doubling eye width while proportionally shifting iris keeps rel constant."""
        def make(scale):
            outer = 0.5 - scale * 0.1
            inner = 0.5 + scale * 0.1
            iris  = 0.5 - scale * 0.05
            ov = {i: (iris, 0.5) for i in [468, 469, 470, 471, 472]}
            ov[33]  = (outer, 0.5)
            ov[133] = (inner, 0.5)
            return _lm_grid(ov)

        rx1, _ = _iris_rel(make(1.0), [468, 469, 470, 471, 472], 33, 133)
        rx2, _ = _iris_rel(make(2.0), [468, 469, 470, 471, 472], 33, 133)
        assert abs(rx1 - rx2) < 1e-9


# ---------------------------------------------------------------------------
# _estimate_face_pos
# ---------------------------------------------------------------------------

def _make_face_lm(dist_mm: float = 600.0, frame_w: int = 640, frame_h: int = 480) -> list:
    """
    Build landmarks encoding a face at *dist_mm* from camera.

    Iris centres placed symmetrically around the image centre.
    IOD = 63 mm, iris radius = 11.8/2 mm (both from physical constants).
    """
    focal = float(frame_w)
    cx    = frame_w / 2.0
    cy    = frame_h / 2.0

    iod_px    = _IOD_MM * focal / dist_mm
    iris_r_px = (_IRIS_DIAM_MM / 2.0) * focal / dist_mm

    r_x = (cx - iod_px / 2.0) / frame_w
    l_x = (cx + iod_px / 2.0) / frame_w
    ir  = iris_r_px / frame_w   # normalised iris radius

    lm = [_Lm(0.5, 0.5) for _ in range(478)]

    lm[468] = _Lm(r_x, cy / frame_h)
    for bid in [469, 470, 471, 472]:
        lm[bid] = _Lm(r_x + ir, cy / frame_h)

    lm[473] = _Lm(l_x, cy / frame_h)
    for bid in [474, 475, 476, 477]:
        lm[bid] = _Lm(l_x + ir, cy / frame_h)

    lm[1] = _Lm(0.5, 0.5)   # nose tip at image centre
    return lm


class TestEstimateFacePos:
    def test_returns_face_position_dataclass(self):
        lm = _make_face_lm()
        assert isinstance(_estimate_face_pos(lm, 640, 480), FacePosition)

    def test_iod_depth_roundtrip(self):
        """IOD-based depth should match synthetic distance within 5%."""
        for d in (400.0, 600.0, 800.0):
            lm = _make_face_lm(dist_mm=d)
            fp = _estimate_face_pos(lm, 640, 480)
            assert abs(fp.dist_iod_mm - d) / d < 0.05, (
                f"dist={d} mm → got {fp.dist_iod_mm:.1f} mm"
            )

    def test_iris_depth_roundtrip(self):
        """Iris-diameter-based depth should match synthetic distance within 5%."""
        d  = 600.0
        fp = _estimate_face_pos(_make_face_lm(dist_mm=d), 640, 480)
        assert abs(fp.dist_iris_mm - d) / d < 0.05

    def test_centred_face_zero_lateral(self):
        """Symmetric face with nose at image centre → |x_mm|, |y_mm| < 5 mm."""
        fp = _estimate_face_pos(_make_face_lm(), 640, 480)
        assert abs(fp.x_mm) < 5.0
        assert abs(fp.y_mm) < 5.0

    def test_iod_px_value(self):
        """iod_px should match IOD_MM * focal / dist within 1 px."""
        d        = 600.0
        expected = _IOD_MM * 640.0 / d
        fp       = _estimate_face_pos(_make_face_lm(dist_mm=d, frame_w=640), 640, 480)
        assert abs(fp.iod_px - expected) < 1.0

    def test_closer_face_larger_iod_px(self):
        """A face at 400 mm should have a larger IOD in pixels than at 800 mm."""
        fp_near = _estimate_face_pos(_make_face_lm(dist_mm=400), 640, 480)
        fp_far  = _estimate_face_pos(_make_face_lm(dist_mm=800), 640, 480)
        assert fp_near.iod_px > fp_far.iod_px

    def test_closer_face_smaller_dist_estimate(self):
        """Closer face → smaller depth estimate."""
        fp_near = _estimate_face_pos(_make_face_lm(dist_mm=400), 640, 480)
        fp_far  = _estimate_face_pos(_make_face_lm(dist_mm=800), 640, 480)
        assert fp_near.dist_iod_mm < fp_far.dist_iod_mm


# ---------------------------------------------------------------------------
# Camera startup timing  (hardware, skipped unless -m slow)
# ---------------------------------------------------------------------------

class TestCameraStartup:
    @pytest.mark.slow
    def test_cap_dshow_opens_and_delivers_frame(self):
        """CAP_DSHOW should open and deliver the first frame in < 5 s."""
        if sys.platform != "win32":
            pytest.skip("CAP_DSHOW is Windows-only")
        import cv2

        t0  = time.monotonic()
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            pytest.skip("No camera available")

        got = False
        for _ in range(60):
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                got = True
                break
        elapsed = time.monotonic() - t0
        cap.release()

        assert got, "Did not receive a valid frame"
        print(f"\n[CAP_DSHOW] first frame in {elapsed * 1000:.0f} ms")
        assert elapsed < 5.0

    @pytest.mark.slow
    def test_dshow_vs_default_speed(self):
        """Print a comparison of CAP_DSHOW vs default backend open time."""
        if sys.platform != "win32":
            pytest.skip("Windows-only")
        import cv2

        def _time_open(backend):
            t0  = time.monotonic()
            cap = cv2.VideoCapture(0) if backend is None else cv2.VideoCapture(0, backend)
            ok  = cap.isOpened()
            if ok:
                for _ in range(5):           # wait for first real frame
                    ret, f = cap.read()
                    if ret and f is not None and f.size > 0:
                        break
            cap.release()
            return time.monotonic() - t0, ok

        t_ds, ok_ds  = _time_open(cv2.CAP_DSHOW)
        t_def, ok_def = _time_open(None)

        if not (ok_ds and ok_def):
            pytest.skip("Camera unavailable for both backends")

        print(
            f"\nCAP_DSHOW : {t_ds  * 1000:6.0f} ms\n"
            f"default   : {t_def * 1000:6.0f} ms"
        )
        # No strict assertion — timing is machine-dependent.
        # The result is printed so the user can see the speedup.
        assert ok_ds and ok_def
