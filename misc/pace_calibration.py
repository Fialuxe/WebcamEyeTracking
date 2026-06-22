"""
PACE calibration: click-based implicit eye tracker calibration via RLS.

CHI 2016: "Building a Personalized, Auto-Calibrating Eye Tracker from User Interactions"
Precision: ~2.56° with geometric features, no explicit calibration required.

Usage:
    pace = PACECalibration()
    # On mouse click at (norm_x, norm_y):
    local = webcam_source.get_local_gaze()
    if local:
        pace.update(*local, norm_x, norm_y)
    # Predict:
    if pace.is_ready:
        gx, gy = pace.predict(local_x, local_y)
"""
from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

from .calibration import CalibrationManager, CalibrationResult  # noqa: F401

_log = logging.getLogger(__name__)

# Number of polynomial features: [1, lx, ly, lx*ly, lx^2, ly^2]
_PHI_DIM = 6


def _make_phi(local_x: float, local_y: float) -> np.ndarray:
    """Build the 6-dimensional polynomial feature vector phi."""
    return np.array(
        [1.0, local_x, local_y, local_x * local_y, local_x ** 2, local_y ** 2],
        dtype=np.float64,
    )


class PACECalibration:
    """
    Recursive Least Squares (RLS) calibration from implicit click signals.

    The model maps phi(local_x, local_y) → (screen_x, screen_y) where phi is
    a 6-dimensional polynomial feature vector.  One shared 6×6 covariance
    matrix P is maintained (it depends only on the features, not on the
    targets), alongside two independent weight vectors w_x and w_y.

    Parameters
    ----------
    feature_dim : int
        Raw input dimension (default 2 for local_x, local_y).  Not used to
        size internal matrices — those are always sized to _PHI_DIM (6).
    min_updates : int
        Minimum number of click updates before is_ready returns True.
    forgetting_factor : float
        RLS forgetting factor λ ∈ (0, 1].  Values < 1 discount older
        observations, allowing adaptation to drift (e.g. user repositions).
    init_variance : float
        Diagonal value used to initialise P = init_variance * I.  Large values
        express high initial uncertainty.
    warm_ridge : CalibrationManager | None
        Optional calibrated CalibrationManager whose Ridge coefficients seed
        the initial weight vectors.  Allows faster convergence after explicit
        calibration.
    """

    def __init__(
        self,
        feature_dim: int = 2,
        min_updates: int = 5,
        forgetting_factor: float = 0.99,
        init_variance: float = 1e4,
        warm_ridge: CalibrationManager | None = None,
    ) -> None:
        self.feature_dim = feature_dim
        self.min_updates = min_updates
        self.forgetting_factor = forgetting_factor
        self.init_variance = init_variance
        self._warm_ridge = warm_ridge

        # Ring buffer for accuracy tracking (a-priori errors before update)
        self._error_buffer: deque[float] = deque(maxlen=10)
        self._n_updates: int = 0

        self._init_state()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_state(self) -> None:
        """Initialise (or re-initialise) RLS state, applying warm start if set."""
        # Shared covariance matrix P — same for both output dimensions
        self._P: np.ndarray = np.eye(_PHI_DIM, dtype=np.float64) * self.init_variance

        if self._warm_ridge is not None and self._warm_ridge.is_calibrated:
            self._w_x, self._w_y = self._warm_start_weights(self._warm_ridge)
            _log.debug("PACE warm-start applied from CalibrationManager")
        else:
            self._w_x: np.ndarray = np.zeros(_PHI_DIM, dtype=np.float64)
            self._w_y: np.ndarray = np.zeros(_PHI_DIM, dtype=np.float64)

    def _warm_start_weights(
        self, mgr: CalibrationManager
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert Ridge regression coefficients (in standardised 2-D space) into
        RLS weight vectors in the raw 6-D polynomial space.

        Ridge learned:  screen = cx · ((lx - mean) / std) + intercept
        We need:        screen = w · phi(lx, ly)
                                 where phi = [1, lx, ly, lx*ly, lx², ly²]

        Cross and quadratic terms have no Ridge counterpart, so w[3:6] = 0.
        """
        ridge_x = mgr._ridge_x
        ridge_y = mgr._ridge_y
        mean = mgr._feat_mean  # shape (2,)
        std = mgr._feat_std    # shape (2,)

        # Ridge coef_ shape is (n_features,) = (2,) here
        cx = ridge_x.coef_   # [c_lx, c_ly]
        cy = ridge_y.coef_

        w_x = np.zeros(_PHI_DIM, dtype=np.float64)
        w_y = np.zeros(_PHI_DIM, dtype=np.float64)

        # Bias: intercept - cx0*(mean0/std0) - cx1*(mean1/std1)
        w_x[0] = float(ridge_x.intercept_) - cx[0] * mean[0] / std[0] - cx[1] * mean[1] / std[1]
        w_y[0] = float(ridge_y.intercept_) - cy[0] * mean[0] / std[0] - cy[1] * mean[1] / std[1]

        # Linear terms
        w_x[1] = cx[0] / std[0]  # phi[1] = lx
        w_x[2] = cx[1] / std[1]  # phi[2] = ly
        w_y[1] = cy[0] / std[0]
        w_y[2] = cy[1] / std[1]

        # phi[3..5] = [lx*ly, lx^2, ly^2] — no Ridge counterpart
        # w_x[3:6] and w_y[3:6] remain 0

        return w_x, w_y

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        local_x: float,
        local_y: float,
        screen_x: float,
        screen_y: float,
    ) -> None:
        """
        Process a single click event via RLS update.

        Call this each time the user clicks at (screen_x, screen_y) and the
        current head-local gaze is (local_x, local_y).

        Args:
            local_x, local_y : head-local gaze direction (from head_pose).
            screen_x, screen_y : click position in normalised [0, 1] coords.
        """
        phi = _make_phi(local_x, local_y)
        lam = self.forgetting_factor

        # --- RLS covariance update (shared for both output dims) ---
        # Pphi = P @ phi : (6,)
        Pphi = self._P @ phi                          # (6,)
        denom = lam + phi @ Pphi                      # scalar
        # Numerator is outer(Pphi, Pphi) : (6,6)
        self._P = (self._P - np.outer(Pphi, Pphi) / denom) / lam

        # Kalman gain
        K = self._P @ phi                             # (6,)

        # A-priori prediction errors (using OLD weights for logging/accuracy)
        err_x = screen_x - float(self._w_x @ phi)
        err_y = screen_y - float(self._w_y @ phi)

        # Record combined MAE for accuracy tracking
        self._error_buffer.append((abs(err_x) + abs(err_y)) / 2.0)

        # Weight update
        self._w_x = self._w_x + K * err_x
        self._w_y = self._w_y + K * err_y

        self._n_updates += 1
        _log.debug(
            "PACE update #%d: err=%.4f,%.4f  mae=%.4f",
            self._n_updates,
            err_x,
            err_y,
            self.accuracy_estimate,
        )

    def predict(self, local_x: float, local_y: float) -> tuple[float, float]:
        """
        Predict normalised screen coordinates for a given head-local gaze.

        Returns clamped values in [0, 1] regardless of is_ready state, so
        callers may optionally use warm-start predictions before min_updates.

        Args:
            local_x, local_y : head-local gaze direction.

        Returns:
            (screen_x, screen_y) each clamped to [0, 1].
        """
        phi = _make_phi(local_x, local_y)
        x = float(self._w_x @ phi)
        y = float(self._w_y @ phi)
        return max(0.0, min(1.0, x)), max(0.0, min(1.0, y))

    def reset(self) -> None:
        """
        Reset RLS state.

        If a warm_ridge was provided at construction, it is re-applied so the
        model returns to its initial prior rather than to zero weights.
        """
        self._n_updates = 0
        self._error_buffer.clear()
        self._init_state()
        _log.debug("PACE calibration reset")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """True once at least min_updates click events have been processed."""
        return self._n_updates >= self.min_updates

    @property
    def update_count(self) -> int:
        """Total number of click updates processed since construction or reset."""
        return self._n_updates

    @property
    def accuracy_estimate(self) -> float:
        """
        Mean absolute error (in normalised screen units) over the most recent
        10 click updates, computed from a-priori prediction errors.

        Returns -1.0 if fewer than 10 updates have been recorded.
        """
        if len(self._error_buffer) < 10:
            return -1.0
        return float(np.mean(list(self._error_buffer)))
