"""
Pipeline: wires gaze sources → OSC sender + CSV logger (Challenge #12).

Thread model:
  - Each GazeSource runs its own background thread, pushing to a per-source Queue(maxsize=2).
  - The dispatcher thread wakes immediately when any source produces a sample via a shared
    threading.Event; falls back to a 5 ms timeout to handle source death gracefully.
  - OSCSender and CSVLogger each run on their own daemon threads.
"""
from __future__ import annotations

import queue
import threading
from typing import TYPE_CHECKING

from .gaze.base import GazeSource, GazeSample
from .osc.sender import OSCSender
from .recording.csv_logger import CSVLogger
from .session.session import Session

if TYPE_CHECKING:
    from .osc.receiver import OSCReceiver


class _SignalingQueue(queue.Queue):
    """Queue that sets a shared threading.Event whenever an item is put successfully."""

    def __init__(self, maxsize: int = 0, event: threading.Event | None = None) -> None:
        super().__init__(maxsize=maxsize)
        self._event = event or threading.Event()

    def put_nowait(self, item) -> None:
        super().put_nowait(item)  # raises queue.Full on overflow → event not set
        self._event.set()


class Pipeline:
    """
    Central coordinator.  add_source() registers gaze sources; start() launches
    all threads; stop() tears everything down in order.
    """

    def __init__(
        self,
        session: Session,
        osc_sender: OSCSender,
        csv_logger: CSVLogger,
        osc_receiver: "OSCReceiver | None" = None,
    ) -> None:
        self._session = session
        self._osc = osc_sender
        self._csv = csv_logger
        self._receiver = osc_receiver
        self._sources: list[tuple[GazeSource, queue.Queue]] = []
        self._dispatch_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._new_sample_event = threading.Event()
        self._current_trial_id: str = ""
        self._latest_gaze: GazeSample | None = None

    def add_source(self, source: GazeSource) -> None:
        q: _SignalingQueue[GazeSample] = _SignalingQueue(maxsize=2, event=self._new_sample_event)
        self._sources.append((source, q))

    def add_osc_receiver(self, receiver: "OSCReceiver") -> None:
        """Attach an OSCReceiver; must be called before start()."""
        self._receiver = receiver

    def set_trial_id(self, tid: str) -> None:
        """Set the current trial ID written into CSV rows."""
        self._current_trial_id = tid

    def clear_trial_id(self) -> None:
        """Clear the current trial ID (rows will have trial_id == '')."""
        self._current_trial_id = ""

    def start(self) -> None:
        self._stop_event.clear()
        self._osc.start()
        self._csv.start()
        if self._receiver is not None:
            self._receiver.start()
        for source, q in self._sources:
            source.start(q)
        self._dispatch_thread = threading.Thread(
            target=self._dispatch, daemon=True, name="gaze-dispatcher"
        )
        self._dispatch_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        for source, _ in self._sources:
            source.stop()
        if self._dispatch_thread:
            self._dispatch_thread.join(timeout=2.0)
        # Drain samples that arrived in per-source queues after the dispatcher exited
        pre_calib = self._session.is_pre_calibration
        trial_id = self._current_trial_id
        for _, q in self._sources:
            while True:
                try:
                    sample = q.get_nowait()
                    self._csv.log(sample, pre_calibration=pre_calib, trial_id=trial_id)
                except queue.Empty:
                    break
        if self._receiver is not None:
            self._receiver.stop()
        self._csv.stop()
        # OSCSender lifecycle is managed by main(); do NOT stop it here.

    def mark_calibrated(self) -> None:
        """
        Call after successful calibration to clear the pre-calibration flag
        on subsequent CSV rows (Challenge #9).
        """
        self._session.mark_calibrated()

    def get_latest_gaze(self) -> "GazeSample | None":
        """Return the most recent GazeSample seen by the dispatcher. Not thread-safe for writes, but safe for UI polling (stale read is acceptable)."""
        return self._latest_gaze

    def _dispatch(self) -> None:
        """Poll per-source queues, route samples to OSC and CSV."""
        while not self._stop_event.is_set():
            got_any = False
            pre_calib = self._session.is_pre_calibration
            trial_id = self._current_trial_id
            for _, q in self._sources:
                try:
                    sample = q.get_nowait()
                    self._osc.send(sample)
                    self._csv.log(sample, pre_calibration=pre_calib, trial_id=trial_id)
                    if self._receiver is not None:
                        self._receiver.set_latest_gaze(sample)
                    self._latest_gaze = sample
                    got_any = True
                except queue.Empty:
                    pass
            if not got_any:
                self._new_sample_event.wait(timeout=0.005)
                self._new_sample_event.clear()
