"""Tests for CSVLogger (Challenge #10 and #13)."""
import csv
import os
import tempfile
import time

import pytest

from main.gaze.base import GazeSample
from main.recording.csv_logger import CSVLogger, sanitize, FIELDNAMES


def _sample(**kwargs) -> GazeSample:
    defaults = dict(
        x=0.5, y=0.3, mesh_certainty=0.9, eye_certainty=0.8, source="ir", condition="IR",
        ts_wall_ms=1234.5, ts_mono_ns=9_876_543_210,
    )
    defaults.update(kwargs)
    return GazeSample(**defaults)


def _run_logger(path: str, pid: str, sid: str, samples_and_flags):
    logger = CSVLogger(path, pid, sid)
    logger.start()
    for sample, pre_calib in samples_and_flags:
        logger.log(sample, pre_calibration=pre_calib)
        time.sleep(0.005)
    logger.stop()


class TestSanitize:
    @pytest.mark.parametrize("inp,expected", [
        ("=SUM(A1)", "'=SUM(A1)"),
        ("+SHELL()", "'+SHELL()"),
        ("-foo", "'-foo"),
        ("@bar", "'@bar"),
        ("normal", "normal"),
        ("P01", "P01"),
        ("", ""),
    ])
    def test_injection_chars(self, inp, expected):
        assert sanitize(inp) == expected


class TestCSVLogger:
    def test_writes_header(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            _run_logger(path, "P01", "S01", [(_sample(), False)])
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                assert set(reader.fieldnames or []) == set(FIELDNAMES)
        finally:
            os.unlink(path)

    def test_writes_one_row(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            _run_logger(path, "P01", "S01", [(_sample(), False)])
            with open(path, newline="") as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == 1
        finally:
            os.unlink(path)

    def test_writes_multiple_rows(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            items = [(_sample(), False)] * 5
            _run_logger(path, "P01", "S01", items)
            with open(path, newline="") as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == 5
        finally:
            os.unlink(path)

    def test_calibrated_flag(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            _run_logger(path, "P01", "S01", [
                (_sample(), True),   # pre_calibration=True  → calibrated=0
                (_sample(), False),  # pre_calibration=False → calibrated=1
            ])
            with open(path, newline="") as f:
                rows = list(csv.DictReader(f))
            assert rows[0]["calibrated"] == "0"
            assert rows[1]["calibrated"] == "1"
        finally:
            os.unlink(path)

    def test_formula_injection_in_participant_id(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            _run_logger(path, "=EXPLOIT()", "S01", [(_sample(), False)])
            with open(path, newline="") as f:
                rows = list(csv.DictReader(f))
            assert rows[0]["participant_id"].startswith("'")
        finally:
            os.unlink(path)

    def test_condition_in_every_row(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            samples = [(_sample(condition="Webcam"), False)] * 3
            _run_logger(path, "P02", "S02", samples)
            with open(path, newline="") as f:
                rows = list(csv.DictReader(f))
            for row in rows:
                assert row["condition"] == "Webcam"
                assert row["participant_id"] == "P02"
        finally:
            os.unlink(path)

    def test_dual_timestamps_present(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            _run_logger(path, "P01", "S01", [(_sample(), False)])
            with open(path, newline="") as f:
                rows = list(csv.DictReader(f))
            assert float(rows[0]["ts_wall_ms"]) > 0
            assert int(rows[0]["ts_mono_ns"]) > 0
        finally:
            os.unlink(path)

    def test_creates_output_directory(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "subdir", "output.csv")
            _run_logger(path, "P01", "S01", [(_sample(), False)])
            assert os.path.exists(path)


class TestHeadPoseColumns:
    def test_head_pose_none_writes_empty_string(self):
        """IR source rows (head_yaw=None) should write empty string, not 'None'."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            sample = GazeSample(
                x=0.5, y=0.3, mesh_certainty=1.0, eye_certainty=1.0,
                source="ir", condition="IR",
                ts_wall_ms=1000.0, ts_mono_ns=1_000_000_000,
                # head_yaw/pitch/roll not set → None
            )
            _run_logger(path, "P01", "S01", [(sample, False)])
            with open(path, newline="") as f:
                rows = list(csv.DictReader(f))
            assert rows[0]["head_yaw"] == ""
            assert rows[0]["head_pitch"] == ""
            assert rows[0]["head_roll"] == ""
        finally:
            os.unlink(path)

    def test_head_pose_float_writes_formatted_value(self):
        """Webcam source rows should write numeric head_yaw/pitch/roll."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            sample = GazeSample(
                x=0.5, y=0.3, mesh_certainty=0.9, eye_certainty=0.8,
                source="webcam", condition="Webcam",
                ts_wall_ms=1000.0, ts_mono_ns=1_000_000_000,
                head_yaw=12.5, head_pitch=-3.0, head_roll=1.25,
            )
            _run_logger(path, "P01", "S01", [(sample, False)])
            with open(path, newline="") as f:
                rows = list(csv.DictReader(f))
            assert float(rows[0]["head_yaw"]) == pytest.approx(12.5, abs=1e-3)
            assert float(rows[0]["head_pitch"]) == pytest.approx(-3.0, abs=1e-3)
            assert float(rows[0]["head_roll"]) == pytest.approx(1.25, abs=1e-3)
        finally:
            os.unlink(path)
