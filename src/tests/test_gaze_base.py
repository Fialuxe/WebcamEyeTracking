"""Tests for GazeSample, MockGazeSource."""
import queue
import time

import pytest

from main.gaze.base import GazeSample, MockGazeSource


def _sample(**kwargs) -> GazeSample:
    defaults = dict(
        x=0.5, y=0.5, mesh_certainty=1.0, eye_certainty=1.0, source="ir", condition="IR",
        ts_wall_ms=1000.0, ts_mono_ns=1_000_000_000,
    )
    defaults.update(kwargs)
    return GazeSample(**defaults)


class TestGazeSample:
    def test_values_clamped_above(self):
        s = _sample(x=1.5, y=2.0, mesh_certainty=3.0, eye_certainty=2.5)
        assert s.x == 1.0
        assert s.y == 1.0
        assert s.mesh_certainty == 1.0
        assert s.eye_certainty == 1.0

    def test_values_clamped_below(self):
        s = _sample(x=-0.5, y=-1.0, mesh_certainty=-0.1, eye_certainty=-0.5)
        assert s.x == 0.0
        assert s.y == 0.0
        assert s.mesh_certainty == 0.0
        assert s.eye_certainty == 0.0

    def test_valid_values_unchanged(self):
        s = _sample(x=0.3, y=0.7, mesh_certainty=0.8, eye_certainty=0.9)
        assert s.x == pytest.approx(0.3)
        assert s.y == pytest.approx(0.7)
        assert s.mesh_certainty == pytest.approx(0.8)
        assert s.eye_certainty == pytest.approx(0.9)

    def test_certainty_property_is_product(self):
        s = _sample(mesh_certainty=0.8, eye_certainty=0.5)
        assert s.certainty == pytest.approx(0.4)

    def test_fields_stored(self):
        s = _sample(source="webcam", condition="Webcam", ts_wall_ms=42.5, ts_mono_ns=99)
        assert s.source == "webcam"
        assert s.condition == "Webcam"
        assert s.ts_wall_ms == pytest.approx(42.5)
        assert s.ts_mono_ns == 99


class TestMockGazeSource:
    def test_emits_given_samples(self):
        samples = [_sample(x=0.1), _sample(x=0.9)]
        src = MockGazeSource(samples=samples)
        q: queue.Queue[GazeSample] = queue.Queue(maxsize=10)
        src.start(q)
        time.sleep(0.15)
        src.stop()
        assert not q.empty()
        got = q.get_nowait()
        assert got.x == pytest.approx(0.1)

    def test_continuous_mode_emits_multiple(self):
        src = MockGazeSource(source="mock", condition="IR")
        q: queue.Queue[GazeSample] = queue.Queue(maxsize=10)
        src.start(q)
        time.sleep(0.2)
        src.stop()
        assert q.qsize() > 0

    def test_stop_terminates_thread(self):
        src = MockGazeSource()
        q: queue.Queue[GazeSample] = queue.Queue(maxsize=4)
        src.start(q)
        time.sleep(0.05)
        src.stop()
        assert src._thread is not None
        assert not src._thread.is_alive()

    def test_source_and_condition_propagated(self):
        samples = [_sample(source="ir", condition="Webcam")]
        src = MockGazeSource(samples=samples)
        q: queue.Queue[GazeSample] = queue.Queue(maxsize=4)
        src.start(q)
        time.sleep(0.1)
        src.stop()
        got = q.get_nowait()
        assert got.source == "ir"
        assert got.condition == "Webcam"


class TestGazeSampleHeadPose:
    def test_head_pose_defaults_to_none(self):
        s = GazeSample(
            x=0.5, y=0.5, mesh_certainty=1.0, eye_certainty=1.0,
            source="webcam", condition="Webcam",
            ts_wall_ms=0.0, ts_mono_ns=0,
        )
        assert s.head_yaw is None
        assert s.head_pitch is None
        assert s.head_roll is None

    def test_head_pose_accepts_float_values(self):
        s = GazeSample(
            x=0.5, y=0.5, mesh_certainty=1.0, eye_certainty=1.0,
            source="webcam", condition="Webcam",
            ts_wall_ms=0.0, ts_mono_ns=0,
            head_yaw=15.3, head_pitch=-5.0, head_roll=2.1,
        )
        assert abs(s.head_yaw - 15.3) < 1e-9
        assert abs(s.head_pitch - (-5.0)) < 1e-9
        assert abs(s.head_roll - 2.1) < 1e-9

    def test_head_pose_not_clamped(self):
        """head_yaw/pitch/roll are not clamped to [0, 1] unlike x/y."""
        s = GazeSample(
            x=0.5, y=0.5, mesh_certainty=1.0, eye_certainty=1.0,
            source="webcam", condition="Webcam",
            ts_wall_ms=0.0, ts_mono_ns=0,
            head_yaw=90.0, head_pitch=-45.0, head_roll=180.0,
        )
        assert s.head_yaw == 90.0
        assert s.head_pitch == -45.0
        assert s.head_roll == 180.0
