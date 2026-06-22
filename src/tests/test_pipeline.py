"""Integration tests for Pipeline (Challenge #12).

Verifies end-to-end routing: GazeSource → dispatcher → OSCSender + CSVLogger.
Uses MockGazeSource + temp-file CSVLogger so no hardware is required.
"""
import csv
import os
import socket
import tempfile
import time

import pytest

from main.gaze.base import GazeSample, MockGazeSource
from main.osc.sender import OSCSender
from main.recording.csv_logger import CSVLogger
from main.session.session import Session
from main.orchestrator import Pipeline


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _read_rows(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _make_pipeline(path: str, condition: str = "IR"):
    session = Session("P01", condition)
    osc = OSCSender("127.0.0.1", _free_port())
    csv_logger = CSVLogger(path, "P01", "S_test")
    pipeline = Pipeline(session, osc, csv_logger)
    return pipeline, session


class TestPipelineIntegration:
    def test_single_source_writes_rows(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            pipeline, _ = _make_pipeline(path)
            pipeline.add_source(MockGazeSource())
            pipeline.start()
            time.sleep(0.3)
            pipeline.stop()

            rows = _read_rows(path)
            assert len(rows) >= 5, f"Expected >= 5 rows, got {len(rows)}"
        finally:
            os.unlink(path)

    def test_two_sources_both_write(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            pipeline, _ = _make_pipeline(path)
            pipeline.add_source(MockGazeSource(source="ir", condition="IR"))
            pipeline.add_source(MockGazeSource(source="webcam", condition="Webcam"))
            pipeline.start()
            time.sleep(0.3)
            pipeline.stop()

            rows = _read_rows(path)
            sources = {r["source"] for r in rows}
            assert "ir" in sources
            assert "webcam" in sources
        finally:
            os.unlink(path)

    def test_csv_has_required_columns(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            pipeline, _ = _make_pipeline(path)
            pipeline.add_source(MockGazeSource())
            pipeline.start()
            time.sleep(0.15)
            pipeline.stop()

            rows = _read_rows(path)
            assert len(rows) > 0
            required = {
                "ts_wall_ms", "ts_mono_ns", "x", "y", "mesh_certainty", "eye_certainty",
                "source", "condition", "participant_id", "session_id", "calibrated",
            }
            assert required <= set(rows[0].keys())
        finally:
            os.unlink(path)

    def test_calibrated_flag_is_zero_before_calibration(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            pipeline, _ = _make_pipeline(path)
            pipeline.add_source(MockGazeSource())
            pipeline.start()
            time.sleep(0.15)
            pipeline.stop()

            rows = _read_rows(path)
            assert len(rows) > 0
            for row in rows:
                assert row["calibrated"] == "0"
        finally:
            os.unlink(path)

    def test_mark_calibrated_flips_flag(self):
        """calibrated must be 0 before and 1 after mark_calibrated()."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            pipeline, _ = _make_pipeline(path)
            pipeline.add_source(MockGazeSource())
            pipeline.start()
            time.sleep(0.15)
            pipeline.mark_calibrated()
            time.sleep(0.15)
            pipeline.stop()

            rows = _read_rows(path)
            flags = [r["calibrated"] for r in rows]
            assert "0" in flags, "Expected uncalibrated rows before mark_calibrated()"
            assert "1" in flags, "Expected calibrated rows after mark_calibrated()"
        finally:
            os.unlink(path)

    def test_trial_id_written_to_csv(self):
        """set_trial_id() causes subsequent CSV rows to have that trial_id."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            pipeline, _ = _make_pipeline(path)
            pipeline.add_source(MockGazeSource())
            pipeline.start()
            pipeline.set_trial_id("T01")
            time.sleep(0.3)
            pipeline.stop()

            rows = _read_rows(path)
            assert len(rows) > 0
            # At least some rows (after set_trial_id) should have trial_id == "T01"
            trial_ids = {r["trial_id"] for r in rows}
            assert "T01" in trial_ids, f"Expected 'T01' in trial_ids, got {trial_ids}"
        finally:
            os.unlink(path)

    def test_trial_id_clears(self):
        """clear_trial_id() causes subsequent CSV rows to have trial_id == ''."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            pipeline, _ = _make_pipeline(path)
            pipeline.add_source(MockGazeSource())
            pipeline.start()
            pipeline.set_trial_id("T01")
            time.sleep(0.15)
            pipeline.clear_trial_id()
            time.sleep(0.15)
            pipeline.stop()

            rows = _read_rows(path)
            assert len(rows) > 0
            trial_ids = [r["trial_id"] for r in rows]
            # After clear, there should be rows with empty trial_id
            assert "" in trial_ids, f"Expected empty trial_id after clear, got {trial_ids}"
        finally:
            os.unlink(path)

    def test_stop_drains_queued_samples(self):
        """Samples sitting in per-source queues when the dispatcher exits must not be lost.

        MockGazeSource emits 10 items without delay; with maxsize=2 the queue
        retains only the 2 most-recent items.  The dispatcher exits on stop_event
        before it can process them.  The drain pass in Pipeline.stop() must rescue
        those 2 remaining items and log them to CSV.
        """
        samples = [
            GazeSample(x=i / 9, y=0.5, mesh_certainty=1.0, eye_certainty=1.0, source="mock",
                       condition="IR", ts_wall_ms=float(i), ts_mono_ns=i)
            for i in range(10)
        ]
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            pipeline, _ = _make_pipeline(path)
            pipeline.add_source(MockGazeSource(samples=samples))
            pipeline.start()
            time.sleep(0.05)   # let mock finish emitting before stop
            pipeline.stop()

            rows = _read_rows(path)
            # Dispatcher + drain together must have written something;
            # drain rescues at least the 2 items left in the queue at shutdown.
            assert len(rows) >= 2, f"Expected drain to rescue queued samples, got {len(rows)}"
        finally:
            os.unlink(path)
