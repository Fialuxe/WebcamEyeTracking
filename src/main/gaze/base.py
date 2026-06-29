"""
Core gaze data contract: GazeSample, GazeSource ABC, MockGazeSource.
All other modules depend on these types.
"""
from __future__ import annotations

import abc
import math
import queue
import threading
import time
from dataclasses import dataclass


@dataclass
class GazeSample:
    """Normalized gaze point shared across all sources and conditions.

    ts_mono_ns: int       # monotonic ns for cross-stream alignment
    head_yaw: float | None    # degrees, head rotation; None for IR
    head_pitch: float | None
    head_roll: float | None
    """
    x: float              # [0, 1]  left → right
    y: float              # [0, 1]  top  → bottom
    mesh_certainty: float # [0, 1]  face mesh detection quality
    eye_certainty: float  # [0, 1]  eye openness (EAR-based)
    source: str           # "ir" | "webcam"
    condition: str        # "IR" | "Webcam" | "WebcamFiltered"
    ts_wall_ms: float     # wall-clock ms at capture
    ts_mono_ns: int       # monotonic ns for cross-stream alignment
    head_yaw: float | None = None    # degrees; None for IR source
    head_pitch: float | None = None  # degrees; None for IR source
    head_roll: float | None = None   # degrees; None for IR source

    def __post_init__(self) -> None:
        self.x = max(0.0, min(1.0, self.x))
        self.y = max(0.0, min(1.0, self.y))
        self.mesh_certainty = max(0.0, min(1.0, self.mesh_certainty))
        self.eye_certainty = max(0.0, min(1.0, self.eye_certainty))

    @property
    def certainty(self) -> float:
        """Combined certainty (product of mesh and eye); used for CSV logging."""
        return self.mesh_certainty * self.eye_certainty


class GazeSource(abc.ABC):
    """Abstract interface for a gaze data source."""

    @abc.abstractmethod
    def start(self, out_queue: queue.Queue) -> None:
        """Start emitting GazeSample objects into out_queue on a background thread."""

    @abc.abstractmethod
    def stop(self) -> None:
        """Stop the source and release resources."""


class MockGazeSource(GazeSource):
    """
    Deterministic mock source for testing and demo without hardware.
    If samples is provided, emits those once then stops.
    Otherwise generates a sine-wave gaze pattern at ~60 Hz.
    """

    def __init__(
        self,
        samples: list[GazeSample] | None = None,
        source: str = "mock",
        condition: str = "IR",
    ) -> None:
        self._samples = samples
        self._source = source
        self._condition = condition
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self, out_queue: queue.Queue) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, args=(out_queue,), daemon=True, name="mock-gaze"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def _run(self, out_queue: queue.Queue) -> None:
        if self._samples is not None:
            for s in self._samples:
                if self._stop_event.is_set():
                    break
                _put_drop_oldest(out_queue, s)
            return

        t = 0.0
        while not self._stop_event.is_set():
            x = 0.5 + 0.3 * math.sin(t)
            y = 0.5 + 0.2 * math.sin(t * 1.3)
            sample = GazeSample(
                x=x,
                y=y,
                mesh_certainty=1.0,
                eye_certainty=1.0,
                source=self._source,
                condition=self._condition,
                ts_wall_ms=time.time() * 1000,
                ts_mono_ns=time.monotonic_ns(),
            )
            _put_drop_oldest(out_queue, sample)
            t += 0.05
            time.sleep(1 / 60)


class DemoGazeSource(GazeSource):
    """Fixed center gaze source for Demo mode — emits (0.5, 0.5) with certainty=1.0 at ~30 Hz."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self, out_queue: queue.Queue) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, args=(out_queue,), daemon=True, name="demo-gaze"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def _run(self, out_queue: queue.Queue) -> None:
        while not self._stop_event.is_set():
            sample = GazeSample(
                x=0.5,
                y=0.5,
                mesh_certainty=1.0,
                eye_certainty=1.0,
                source="demo",
                condition="Demo",
                ts_wall_ms=time.time() * 1000,
                ts_mono_ns=time.monotonic_ns(),
            )
            _put_drop_oldest(out_queue, sample)
            time.sleep(1 / 30)


def _put_drop_oldest(q: queue.Queue, item) -> None:
    """Put item into queue; if full, drop oldest then retry."""
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
        except queue.Full:
            pass
