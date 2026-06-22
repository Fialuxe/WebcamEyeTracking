"""
Unit tests for calib_window.py pure logic.

These tests do NOT open a real Tk window — they only exercise module-level
constants and pure functions that can be imported without initialising Tk.
"""
import sys
import os

# Ensure src/ is on path (mirrors conftest.py)
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

from main.ui.calib_window import TARGETS, MARGIN, _average_samples


class TestTargets:
    """Test the TARGETS constant describes a valid 16-point non-uniform grid."""

    def test_targets_has_sixteen_elements(self):
        assert len(TARGETS) == 16

    def test_targets_margin_is_small(self):
        """MARGIN should be 0.05 (corners pushed toward screen edges)."""
        assert abs(MARGIN - 0.05) < 1e-9

    def test_targets_true_corners_present(self):
        """The four true corners at MARGIN must be in TARGETS."""
        corners = {
            (MARGIN, MARGIN),
            (1.0 - MARGIN, MARGIN),
            (MARGIN, 1.0 - MARGIN),
            (1.0 - MARGIN, 1.0 - MARGIN),
        }
        actual = {(round(x, 6), round(y, 6)) for x, y in TARGETS}
        for c in corners:
            assert c in actual, f"Corner {c} missing from TARGETS"

    def test_targets_are_within_unit_square(self):
        for x, y in TARGETS:
            assert 0.0 <= x <= 1.0, f"x={x} out of [0, 1]"
            assert 0.0 <= y <= 1.0, f"y={y} out of [0, 1]"

    def test_targets_all_unique(self):
        assert len(set(TARGETS)) == len(TARGETS), "Duplicate targets detected"

    def test_targets_cover_edges(self):
        """At least one point should lie on each of the four edges (at MARGIN)."""
        xs = [x for x, _ in TARGETS]
        ys = [y for _, y in TARGETS]
        assert any(abs(x - MARGIN) < 1e-9 for x in xs), "No left-edge point"
        assert any(abs(x - (1.0 - MARGIN)) < 1e-9 for x in xs), "No right-edge point"
        assert any(abs(y - MARGIN) < 1e-9 for y in ys), "No top-edge point"
        assert any(abs(y - (1.0 - MARGIN)) < 1e-9 for y in ys), "No bottom-edge point"


class TestAverageSamples:
    """Test the _average_samples() pure helper function."""

    def test_single_sample_returns_itself(self):
        result = _average_samples([(0.3, 0.7)])
        assert result == (0.3, 0.7)

    def test_two_equal_samples(self):
        result = _average_samples([(0.5, 0.5), (0.5, 0.5)])
        assert result == (0.5, 0.5)

    def test_mean_of_two_different_samples(self):
        result = _average_samples([(0.0, 0.0), (1.0, 1.0)])
        avg_x, avg_y = result
        assert abs(avg_x - 0.5) < 1e-9
        assert abs(avg_y - 0.5) < 1e-9

    def test_mean_of_three_samples(self):
        samples = [(0.1, 0.2), (0.3, 0.4), (0.5, 0.6)]
        avg_x, avg_y = _average_samples(samples)
        assert abs(avg_x - 0.3) < 1e-9
        assert abs(avg_y - 0.4) < 1e-9

    def test_returns_tuple(self):
        result = _average_samples([(0.1, 0.9)])
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_asymmetric_xy(self):
        """Ensure x and y are averaged independently."""
        samples = [(0.0, 1.0), (0.0, 0.0)]
        avg_x, avg_y = _average_samples(samples)
        assert abs(avg_x - 0.0) < 1e-9
        assert abs(avg_y - 0.5) < 1e-9


class TestModuleImportDoesNotOpenWindow:
    """
    Importing calib_window must not trigger Tk window creation.
    This test passes by virtue of the module being already imported above
    without any Tk instance having been created.
    """

    def test_import_succeeded_without_tk(self):
        # If we got here, the import completed without raising TclError
        import main.ui.calib_window as cw
        assert hasattr(cw, "run_calibration")
        assert hasattr(cw, "TARGETS")
        assert hasattr(cw, "_average_samples")
