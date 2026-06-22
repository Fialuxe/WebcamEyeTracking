"""
Ridge regression calibration: head-local gaze direction → normalised screen coords.

This is Stage 2 of the two-stage pipeline:
  Stage 1 (head_pose.py): iris pixel → head-local gaze direction (removes head rotation)
  Stage 2 (here):         head-local (local_x, local_y) → screen (x, y)

After head-pose normalisation the residual mapping is nearly linear, so Ridge
regression with L2 regularisation is at the Pareto frontier of accuracy vs.
complexity for 9-point calibration (Severitt et al., 2023).

Holdout validation (33%) guards against overfitting (Challenge #6).
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import RidgeCV

_log = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    success: bool
    validation_error_x: float
    validation_error_y: float


class CalibrationManager:
    """
    Collects (local_x, local_y) → (target_x, target_y) pairs where
    local_x/local_y are head-local gaze direction components from
    head_pose.normalize_gaze(), and fits a Ridge regression model.

    API is intentionally identical to the previous polynomial manager so
    that callers (WebcamGazeSource, UI, tests) require minimal changes.
    """

    HOLDOUT_FRACTION: float = 0.33
    MIN_POINTS: int = 6
    # Candidate alphas for cross-validated selection; spans 4 orders of magnitude
    RIDGE_ALPHAS: tuple = (1e-3, 1e-2, 0.1, 1.0, 10.0)

    def __init__(self) -> None:
        self._points: list[tuple[float, float, float, float]] = []
        self._ridge_x: RidgeCV | None = None
        self._ridge_y: RidgeCV | None = None
        # Feature standardisation params (fit on all calibration points)
        self._feat_mean: np.ndarray = np.zeros(2)
        self._feat_std: np.ndarray = np.ones(2)
        self._calibrated: bool = False

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def add_point(
        self,
        local_x: float,
        local_y: float,
        target_x: float,
        target_y: float,
    ) -> None:
        """Add one calibration sample.

        Args:
            local_x, local_y : head-local gaze direction components
                               (from head_pose.normalize_gaze()).
            target_x, target_y : ground-truth screen position in [0, 1].
        """
        self._points.append((local_x, local_y, target_x, target_y))
        _log.debug(
            "calib point #%d: local=(%.4f,%.4f) target=(%.4f,%.4f)",
            len(self._points), local_x, local_y, target_x, target_y,
        )

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self) -> CalibrationResult:
        """Fit Ridge regression and validate on a held-out subset."""
        if len(self._points) < self.MIN_POINTS:
            _log.warning(
                "calibration failed: only %d points (need %d)",
                len(self._points), self.MIN_POINTS,
            )
            return CalibrationResult(
                success=False,
                validation_error_x=float("inf"),
                validation_error_y=float("inf"),
            )

        pts = self._points[:]
        # Fixed seed so holdout split is reproducible across runs (removes
        # spatial bias from row-major point ordering).
        random.Random(0).shuffle(pts)
        n_holdout = max(1, int(len(pts) * self.HOLDOUT_FRACTION))
        train = pts[:-n_holdout]
        holdout = pts[-n_holdout:]

        # Feature standardisation: fit scaler on all points so inference uses
        # the same transform. Small local gaze values (≈ ±0.2) would otherwise
        # cause Ridge to over-regularise coefficients and bias predictions.
        X_all = np.array([[p[0], p[1]] for p in pts])
        y_all_x = np.array([p[2] for p in pts])
        y_all_y = np.array([p[3] for p in pts])
        self._feat_mean = X_all.mean(axis=0)
        self._feat_std = X_all.std(axis=0)
        self._feat_std[self._feat_std < 1e-8] = 1.0  # avoid div-by-zero

        def _scale(X):
            return (X - self._feat_mean) / self._feat_std

        X_train = _scale(np.array([[p[0], p[1]] for p in train]))
        y_tx = np.array([p[2] for p in train])
        y_ty = np.array([p[3] for p in train])
        X_ho = _scale(np.array([[p[0], p[1]] for p in holdout]))
        y_ho_x = np.array([p[2] for p in holdout])
        y_ho_y = np.array([p[3] for p in holdout])

        # RidgeCV selects alpha via leave-one-out cross-validation on the
        # training set.
        ridge_x = RidgeCV(alphas=self.RIDGE_ALPHAS, fit_intercept=True)
        ridge_y = RidgeCV(alphas=self.RIDGE_ALPHAS, fit_intercept=True)
        ridge_x.fit(X_train, y_tx)
        ridge_y.fit(X_train, y_ty)

        err_x = float(np.mean(np.abs(ridge_x.predict(X_ho) - y_ho_x)))
        err_y = float(np.mean(np.abs(ridge_y.predict(X_ho) - y_ho_y)))

        ridge_x_full = RidgeCV(alphas=self.RIDGE_ALPHAS, fit_intercept=True)
        ridge_y_full = RidgeCV(alphas=self.RIDGE_ALPHAS, fit_intercept=True)
        ridge_x_full.fit(_scale(X_all), y_all_x)
        ridge_y_full.fit(_scale(X_all), y_all_y)
        self._ridge_x = ridge_x_full
        self._ridge_y = ridge_y_full
        self._calibrated = True

        _log.info(
            "calibration fit: %d points  alpha_x=%.4g alpha_y=%.4g  "
            "holdout_err_x=%.4f holdout_err_y=%.4f",
            len(pts),
            float(ridge_x_full.alpha_), float(ridge_y_full.alpha_),
            err_x, err_y,
        )
        return CalibrationResult(
            success=True,
            validation_error_x=err_x,
            validation_error_y=err_y,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, local_x: float, local_y: float) -> tuple[float, float]:
        """Map head-local gaze direction to normalised screen coordinates."""
        if not self._calibrated:
            raise RuntimeError("Not calibrated — call fit() first")
        if self._ridge_x is None or self._ridge_y is None:
            raise RuntimeError("Ridge models are None despite _calibrated=True — state corrupted")
        X = (np.array([[local_x, local_y]]) - self._feat_mean) / self._feat_std
        x = float(self._ridge_x.predict(X)[0])
        y = float(self._ridge_y.predict(X)[0])
        return max(0.0, min(1.0, x)), max(0.0, min(1.0, y))

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._points.clear()
        self._ridge_x = None
        self._ridge_y = None
        self._feat_mean = np.zeros(2)
        self._feat_std = np.ones(2)
        self._calibrated = False

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    @property
    def is_pre_calibration(self) -> bool:
        return not self._calibrated

    @property
    def point_count(self) -> int:
        return len(self._points)
