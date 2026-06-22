"""
TPS (Thin Plate Spline) calibration manager.

Uses scipy.interpolate.RBFInterpolator with kernel="thin_plate_spline" as a
drop-in replacement for CalibrationManager (Ridge regression).

Supports optional 5-D feature vector: (local_x, local_y, tvec_x, tvec_y, tvec_z)
when tvec is provided at add_point / predict time.

tvec integration via stashed state
-----------------------------------
The calibration window (calib_window.py) calls:
    source.calibration.add_point(local_x, local_y, tx, ty)
with exactly 4 positional arguments and no tvec — it cannot be changed without
touching the existing UI code.  To inject tvec into calibration points, the
owning WebcamGazeSourceBeta calls set_current_tvec(tvec) on every processed
frame before the calibration window may call add_point().  The stashed tvec is
then read inside add_point().  This is the only way to get tvec into the point
set without modifying calib_window.py.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass

import numpy as np

from .calibration import CalibrationResult

_log = logging.getLogger(__name__)

# Smoothing parameter for RBFInterpolator.  A value of 0 means exact
# interpolation (overfit risk); 0.1 gives gentle smoothing on typical 9-point
# calibration sets.  If the initial fit raises a numerical exception (can
# happen with 5-D input at minimum point count), we retry at 1.0.
_SMOOTHING_DEFAULT: float = 0.1
_SMOOTHING_FALLBACK: float = 1.0


class TPSCalibrationManager:
    """
    Thin Plate Spline calibration: (local_x, local_y[, tvec_x, tvec_y, tvec_z])
    → (screen_x, screen_y).

    Compatible with CalibrationManager's external API so it can be swapped in
    without changes to the UI or test scaffolding:
        add_point(local_x, local_y, target_x, target_y)
        fit()  → CalibrationResult
        predict(local_x, local_y)  → (float, float)
        reset()
        is_calibrated  (bool property)
        point_count    (int property)

    When use_tvec=True (set in __init__), the manager uses a 5-D feature vector.
    tvec is captured via set_current_tvec(); add_point() reads it automatically.
    """

    HOLDOUT_FRACTION: float = 0.33
    MIN_POINTS: int = 6

    def __init__(self, use_tvec: bool = False) -> None:
        """
        Parameters
        ----------
        use_tvec : when True, augment features with solvePnP tvec (X, Y, Z).
                   Requires set_current_tvec() to be called each frame by the
                   owning source before calibration points are added.
        """
        self._use_tvec = use_tvec
        self._points: list[tuple] = []   # (local_x, local_y, tx, ty, tvec | None)
        self._interp = None              # RBFInterpolator or None
        self._feat_mean: np.ndarray = np.zeros(2)
        self._feat_std: np.ndarray = np.ones(2)
        self._n_dims: int = 2            # feature dimensionality; set by fit()
        self._calibrated: bool = False
        # Per-frame tvec stash: written by WebcamGazeSourceBeta, read by add_point
        self._current_tvec: np.ndarray | None = None

    # ------------------------------------------------------------------
    # tvec injection (called every frame from the camera thread)
    # ------------------------------------------------------------------

    def set_current_tvec(self, tvec: np.ndarray | None) -> None:
        """Stash the latest solvePnP tvec for use at the next add_point() call.

        Thread note: in CPython a reference assignment is atomic under the GIL,
        so no lock is needed here.  This is a single-ref store, not a read-
        modify-write, so it is safe for the camera→Tk cross-thread access.
        """
        if tvec is not None:
            self._current_tvec = np.array(tvec, dtype=np.float64).ravel()
        else:
            self._current_tvec = None

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

        If use_tvec=True, the latest tvec stashed by set_current_tvec() is
        attached to this point.  If no tvec is available the point is stored
        with tvec=None and feature dimensionality falls back to 2-D for fit.
        """
        tvec = self._current_tvec.copy() if self._current_tvec is not None else None
        self._points.append((local_x, local_y, target_x, target_y, tvec))
        _log.debug(
            "TPS calib point #%d: local=(%.4f,%.4f) target=(%.4f,%.4f) tvec=%s",
            len(self._points), local_x, local_y, target_x, target_y,
            "yes" if tvec is not None else "none",
        )

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self) -> CalibrationResult:
        """Fit TPS interpolator and validate on a held-out subset."""
        from scipy.interpolate import RBFInterpolator

        if len(self._points) < self.MIN_POINTS:
            _log.warning(
                "TPS calibration failed: only %d points (need %d).%s",
                len(self._points), self.MIN_POINTS,
                " Note: tvec mode requires >6 points due to 5-D polynomial"
                " constraints." if self._use_tvec else "",
            )
            return CalibrationResult(
                success=False,
                validation_error_x=float("inf"),
                validation_error_y=float("inf"),
            )

        pts = self._points[:]
        random.Random(0).shuffle(pts)
        n_holdout = max(1, int(len(pts) * self.HOLDOUT_FRACTION))
        train = pts[:-n_holdout]
        holdout = pts[-n_holdout:]

        def _build_X(subset):
            """Build feature matrix; uses 5-D if tvec available and use_tvec=True."""
            rows = []
            for (lx, ly, _tx, _ty, tvec) in subset:
                if self._use_tvec and tvec is not None:
                    rows.append([lx, ly, tvec[0], tvec[1], tvec[2]])
                else:
                    rows.append([lx, ly])
            return np.array(rows, dtype=np.float64)

        X_all = _build_X(pts)
        n_dims = X_all.shape[1]

        # Feature standardisation (fit on all points so inference uses same params)
        self._feat_mean = X_all.mean(axis=0)
        self._feat_std  = X_all.std(axis=0)
        self._feat_std[self._feat_std < 1e-8] = 1.0

        def _scale(X):
            return (X - self._feat_mean) / self._feat_std

        X_train_raw = _build_X(train)
        X_ho_raw    = _build_X(holdout)

        # Trim to consistent dimensionality if tvec is missing for some points
        min_d = min(X_train_raw.shape[1], X_ho_raw.shape[1])
        X_train_raw = X_train_raw[:, :min_d]
        X_ho_raw    = X_ho_raw[:, :min_d]
        X_all_trim  = X_all[:, :min_d]

        # Re-compute standardisation params for the trimmed dimensionality
        feat_mean = X_all_trim.mean(axis=0)
        feat_std  = X_all_trim.std(axis=0)
        feat_std[feat_std < 1e-8] = 1.0

        def _sc(X):
            return (X - feat_mean) / feat_std

        self._feat_mean = feat_mean
        self._feat_std  = feat_std

        X_train_s = _sc(X_train_raw)
        X_ho_s    = _sc(X_ho_raw)
        X_all_s   = _sc(X_all_trim)

        y_tx = np.array([p[2] for p in train])
        y_ty = np.array([p[3] for p in train])
        y_ho_x = np.array([p[2] for p in holdout])
        y_ho_y = np.array([p[3] for p in holdout])
        y_all_x = np.array([p[2] for p in pts])
        y_all_y = np.array([p[3] for p in pts])

        # Fit train set for holdout validation
        interp_ho, smoothing_used = self._fit_rbf(
            X_train_s,
            np.column_stack([y_tx, y_ty]),
        )
        if interp_ho is None:
            _log.warning("TPS fit failed at all smoothing levels; returning failure.")
            return CalibrationResult(
                success=False,
                validation_error_x=float("inf"),
                validation_error_y=float("inf"),
            )

        ho_pred = interp_ho(X_ho_s)
        err_x = float(np.mean(np.abs(ho_pred[:, 0] - y_ho_x)))
        err_y = float(np.mean(np.abs(ho_pred[:, 1] - y_ho_y)))

        # Final fit on all points
        interp_full, _ = self._fit_rbf(
            X_all_s,
            np.column_stack([y_all_x, y_all_y]),
            smoothing=smoothing_used,
        )
        if interp_full is None:
            return CalibrationResult(
                success=False,
                validation_error_x=float("inf"),
                validation_error_y=float("inf"),
            )

        self._interp = interp_full
        self._n_dims = min_d
        self._calibrated = True

        _log.info(
            "TPS calibration fit: %d points  dims=%d  smoothing=%.3g  "
            "holdout_err_x=%.4f holdout_err_y=%.4f",
            len(pts), min_d, smoothing_used, err_x, err_y,
        )
        return CalibrationResult(
            success=True,
            validation_error_x=err_x,
            validation_error_y=err_y,
        )

    @staticmethod
    def _fit_rbf(
        X: np.ndarray,
        y: np.ndarray,
        smoothing: float | None = None,
    ) -> tuple:
        """
        Attempt to fit RBFInterpolator with thin_plate_spline kernel.

        Tries _SMOOTHING_DEFAULT first, then _SMOOTHING_FALLBACK if the first
        attempt raises any numerical exception.

        Returns (interpolator, smoothing_used) or (None, None) on failure.
        """
        from scipy.interpolate import RBFInterpolator

        levels = [smoothing] if smoothing is not None else [_SMOOTHING_DEFAULT, _SMOOTHING_FALLBACK]
        for s in levels:
            try:
                interp = RBFInterpolator(X, y, kernel="thin_plate_spline", smoothing=s)
                if smoothing is None and s == _SMOOTHING_FALLBACK:
                    _log.warning(
                        "TPS calibration: smoothing fallback used (smoothing=%.3g). "
                        "Consider collecting more calibration points.", s
                    )
                return interp, s
            except Exception as exc:
                _log.warning("TPS fit failed at smoothing=%.3g: %s", s, exc)
        return None, None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, local_x: float, local_y: float, tvec: np.ndarray | None = None) -> tuple[float, float]:
        """Map gaze features to normalised screen coordinates.

        Parameters
        ----------
        local_x, local_y : iris-relative-to-corners features.
        tvec             : optional (3,) solvePnP tvec; used when fitted with
                           5-D features (use_tvec=True and tvec was available).
        """
        if not self._calibrated or self._interp is None:
            raise RuntimeError("Not calibrated — call fit() first")

        if self._use_tvec and tvec is not None and self._n_dims == 5:
            tvec_flat = np.array(tvec, dtype=np.float64).ravel()
            feat = np.array([[local_x, local_y, tvec_flat[0], tvec_flat[1], tvec_flat[2]]])
        else:
            feat = np.array([[local_x, local_y]])
            if self._n_dims > 2:
                # Fitted with 5-D but no tvec at inference — pad with zeros
                # (suboptimal but avoids a crash).
                pad = np.zeros((1, self._n_dims - 2))
                feat = np.hstack([feat, pad])

        feat_s = (feat - self._feat_mean) / self._feat_std
        pred = self._interp(feat_s)[0]   # (2,)
        x = float(max(0.0, min(1.0, pred[0])))
        y = float(max(0.0, min(1.0, pred[1])))
        return x, y

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._points.clear()
        self._interp = None
        self._feat_mean = np.zeros(2)
        self._feat_std = np.ones(2)
        self._calibrated = False
        self._current_tvec = None

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    @property
    def is_pre_calibration(self) -> bool:
        return not self._calibrated

    @property
    def point_count(self) -> int:
        return len(self._points)
