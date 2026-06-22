"""Tests for Session (Challenges #8 and #9)."""
import pytest
from main.session.session import Session, VALID_CONDITIONS


class TestSession:
    def test_valid_conditions_accepted(self):
        for cond in VALID_CONDITIONS:
            s = Session("P01", cond)
            assert s.condition == cond

    def test_invalid_condition_raises(self):
        with pytest.raises(ValueError, match="condition must be one of"):
            Session("P01", "InvalidCondition")

    def test_initial_not_locked(self):
        s = Session("P01", "IR")
        assert not s.is_locked

    def test_lock(self):
        s = Session("P01", "IR")
        s.lock()
        assert s.is_locked

    def test_unlock(self):
        s = Session("P01", "IR")
        s.lock()
        s.unlock()
        assert not s.is_locked

    def test_initial_pre_calibration(self):
        s = Session("P01", "IR")
        assert s.is_pre_calibration
        assert not s.is_calibrated

    def test_mark_calibrated(self):
        s = Session("P01", "IR")
        s.mark_calibrated()
        assert s.is_calibrated
        assert not s.is_pre_calibration

    def test_sanitize_participant_id_formula(self):
        s = Session("=EXPLOIT()", "IR")
        assert s.participant_id.startswith("'")

    def test_sanitize_leaves_normal_id(self):
        s = Session("P01", "IR")
        assert s.participant_id == "P01"

    def test_ir_required_all_conditions(self):
        for cond in VALID_CONDITIONS - {"NoGaze"}:
            s = Session("P01", cond)
            assert s.ir_required()

    def test_ir_not_required_for_no_gaze(self):
        s = Session("P01", "NoGaze")
        assert not s.ir_required()

    def test_session_id_auto_assigned(self):
        s = Session("P01", "IR")
        assert s.session_id != ""

    def test_session_id_explicit(self):
        s = Session("P01", "IR", session_id="SID42")
        assert s.session_id == "SID42"
