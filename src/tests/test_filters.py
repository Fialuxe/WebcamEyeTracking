"""Tests for OneEuroFilter."""
import pytest
from main.gaze.filters import OneEuroFilter, LowPassFilter, _alpha


class TestAlpha:
    def test_high_cutoff_high_alpha(self):
        # Higher cutoff → less smoothing → alpha closer to 1
        a_high = _alpha(100.0, 1 / 60)
        a_low = _alpha(1.0, 1 / 60)
        assert a_high > a_low

    def test_alpha_bounded(self):
        for cutoff in [0.01, 1.0, 100.0]:
            a = _alpha(cutoff, 1 / 60)
            assert 0.0 < a < 1.0


class TestLowPassFilter:
    def test_first_value_returned_unchanged(self):
        f = LowPassFilter()
        assert f.filter(0.7, alpha=0.5) == pytest.approx(0.7)

    def test_smoothing_toward_new_value(self):
        f = LowPassFilter()
        f.filter(1.0, alpha=0.5)
        out = f.filter(0.0, alpha=0.5)
        assert out == pytest.approx(0.5)

    def test_reset_clears_state(self):
        f = LowPassFilter()
        f.filter(0.9, alpha=0.5)
        f.reset()
        assert f.last is None


class TestOneEuroFilter:
    def test_first_call_returns_input(self):
        f = OneEuroFilter()
        out = f.filter(0.5, t=0.0)
        assert out == pytest.approx(0.5)

    def test_constant_signal_converges(self):
        """With a constant 0.0 signal after starting at 1.0, output should approach 0."""
        f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
        t = 0.0
        dt = 1 / 60
        f.filter(1.0, t=t)
        t += dt
        outputs = []
        for _ in range(20):
            outputs.append(f.filter(0.0, t=t))
            t += dt
        assert outputs[-1] < 0.5

    def test_reset_clears_history(self):
        f = OneEuroFilter()
        f.filter(0.9, t=0.0)
        f.filter(0.8, t=1 / 60)
        f.reset()
        assert f._prev_t is None

    def test_high_beta_less_lag_on_fast_movement(self):
        """Higher beta → higher cutoff on fast motion → less lag."""
        target = 1.0
        f_hi = OneEuroFilter(min_cutoff=1.0, beta=10.0)
        f_lo = OneEuroFilter(min_cutoff=1.0, beta=0.0)
        t, dt = 0.0, 1 / 60
        for _ in range(30):
            f_hi.filter(target, t)
            f_lo.filter(target, t)
            t += dt
        hi_val = f_hi.filter(target, t)
        lo_val = f_lo.filter(target, t)
        assert abs(hi_val - target) <= abs(lo_val - target) + 1e-6

    def test_output_stable_on_constant_signal(self):
        """After convergence, output should not drift."""
        f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
        t, dt = 0.0, 1 / 60
        for _ in range(100):
            f.filter(0.5, t)
            t += dt
        out1 = f.filter(0.5, t)
        t += dt
        out2 = f.filter(0.5, t)
        assert abs(out1 - out2) < 1e-6
