"""Tests for CalibrationManager (Ridge regression, Challenge #6)."""
import numpy as np
import pytest
from main.gaze.calibration import CalibrationManager, CalibrationResult


def _add_grid_points(cal: CalibrationManager, n: int = 9) -> None:
    """Add n calibration points on a regular grid where local == target (identity mapping)."""
    side = int(n ** 0.5) + (1 if int(n ** 0.5) ** 2 < n else 0)
    count = 0
    for i in range(side):
        for j in range(side):
            if count >= n:
                break
            x = i / max(side - 1, 1)
            y = j / max(side - 1, 1)
            cal.add_point(x, y, x, y)
            count += 1


def _add_linear_points(cal: CalibrationManager, n: int = 9) -> None:
    """Add n points with a known affine mapping: target = 0.5*local + 0.1."""
    side = int(n ** 0.5) + (1 if int(n ** 0.5) ** 2 < n else 0)
    count = 0
    for i in range(side):
        for j in range(side):
            if count >= n:
                break
            lx = i / max(side - 1, 1)
            ly = j / max(side - 1, 1)
            tx = np.clip(0.5 * lx + 0.1, 0.0, 1.0)
            ty = np.clip(0.5 * ly + 0.1, 0.0, 1.0)
            cal.add_point(lx, ly, tx, ty)
            count += 1


class TestCalibrationManager:
    def test_initial_state(self):
        cal = CalibrationManager()
        assert not cal.is_calibrated
        assert cal.is_pre_calibration
        assert cal.point_count == 0

    def test_insufficient_points_fails(self):
        cal = CalibrationManager()
        for i in range(5):
            cal.add_point(i / 5, i / 5, i / 5, i / 5)
        result = cal.fit()
        assert not result.success

    def test_identity_mapping_succeeds(self):
        cal = CalibrationManager()
        _add_grid_points(cal, 9)
        result = cal.fit()
        assert result.success
        assert cal.is_calibrated
        assert not cal.is_pre_calibration

    def test_identity_low_validation_error(self):
        cal = CalibrationManager()
        _add_grid_points(cal, 9)
        result = cal.fit()
        assert result.validation_error_x < 0.1
        assert result.validation_error_y < 0.1

    def test_predict_accuracy_on_identity(self):
        cal = CalibrationManager()
        _add_grid_points(cal, 12)
        cal.fit()
        px, py = cal.predict(0.5, 0.5)
        assert abs(px - 0.5) < 0.05
        assert abs(py - 0.5) < 0.05

    def test_predict_clamps_output(self):
        cal = CalibrationManager()
        _add_grid_points(cal, 9)
        cal.fit()
        px, py = cal.predict(10.0, -5.0)
        assert 0.0 <= px <= 1.0
        assert 0.0 <= py <= 1.0

    def test_predict_before_fit_raises(self):
        cal = CalibrationManager()
        with pytest.raises(RuntimeError, match="Not calibrated"):
            cal.predict(0.5, 0.5)

    def test_reset_clears_state(self):
        cal = CalibrationManager()
        _add_grid_points(cal, 9)
        cal.fit()
        assert cal.is_calibrated
        cal.reset()
        assert not cal.is_calibrated
        assert cal.point_count == 0
        assert cal.is_pre_calibration

    def test_add_point_increments_count(self):
        cal = CalibrationManager()
        cal.add_point(0.1, 0.2, 0.1, 0.2)
        cal.add_point(0.3, 0.4, 0.3, 0.4)
        assert cal.point_count == 2

    def test_linear_mapping_accuracy(self):
        """Ridge should learn an affine transform accurately."""
        cal = CalibrationManager()
        _add_linear_points(cal, 12)
        cal.fit()
        px, py = cal.predict(0.5, 0.5)
        # Expected: 0.5 * 0.5 + 0.1 = 0.35
        assert abs(px - 0.35) < 0.05
        assert abs(py - 0.35) < 0.05

    def test_result_has_success_and_errors(self):
        cal = CalibrationManager()
        _add_grid_points(cal, 9)
        result = cal.fit()
        assert isinstance(result.success, bool)
        assert isinstance(result.validation_error_x, float)
        assert isinstance(result.validation_error_y, float)
