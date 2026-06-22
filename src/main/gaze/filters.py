"""
One-Euro filter for smooth, low-latency 1D gaze signal filtering.

Parameters (set via config.py — TBD before data collection):
  min_cutoff : Hz  — baseline smoothing; lower = smoother at rest
  beta       : 1/Hz/s — speed coefficient; higher = less lag during fast movement
  d_cutoff   : Hz  — cutoff for the derivative low-pass filter
"""
from __future__ import annotations

import math


class LowPassFilter:
    def __init__(self) -> None:
        self._value: float | None = None

    def filter(self, x: float, alpha: float) -> float:
        if self._value is None:
            self._value = x
        else:
            self._value = alpha * x + (1.0 - alpha) * self._value
        return self._value

    @property
    def last(self) -> float | None:
        return self._value

    def reset(self) -> None:
        self._value = None


def _alpha(cutoff_hz: float, dt_s: float) -> float:
    """Compute smoothing coefficient for a given cutoff and time step."""
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / max(dt_s, 1e-9))


class OneEuroFilter:
    """
    1D One-Euro filter.
    Call filter(value, t) where t is monotonic time in seconds.
    Thread-unsafe — instantiate one per dimension per thread.
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_lp = LowPassFilter()
        self._dx_lp = LowPassFilter()
        self._prev_t: float | None = None

    def filter(self, x: float, t: float) -> float:
        if self._prev_t is None:
            self._prev_t = t
            return self._x_lp.filter(x, alpha=1.0)

        dt = max(t - self._prev_t, 1e-9)
        self._prev_t = t

        x_prev = self._x_lp.last
        dx = (x - x_prev) / dt if x_prev is not None else 0.0

        a_dx = _alpha(self.d_cutoff, dt)
        dx_hat = self._dx_lp.filter(dx, alpha=a_dx)

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = _alpha(cutoff, dt)
        return self._x_lp.filter(x, alpha=a)

    def reset(self) -> None:
        self._x_lp.reset()
        self._dx_lp.reset()
        self._prev_t = None
