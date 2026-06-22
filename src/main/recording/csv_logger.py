"""
CSV logger thread (Challenge #10).

Every row includes condition, participant ID, session ID, and calibrated flag.
Flushes every CSV_FLUSH_EVERY rows to survive crashes.
Leading formula-injection characters are sanitized (Challenge #13).
Dual timestamps (wall-clock ms + monotonic ns) support cross-stream alignment (Challenge #4).
"""
from __future__ import annotations

import csv
import logging
import os
import queue
import threading

from ..gaze.base import GazeSample
from .. import config

_log = logging.getLogger(__name__)

_FORMULA_CHARS = frozenset("=+-@")

FIELDNAMES = [
    "ts_wall_ms",
    "ts_mono_ns",
    "x",
    "y",
    "mesh_certainty",
    "eye_certainty",
    "head_yaw",
    "head_pitch",
    "head_roll",
    "source",
    "condition",
    "participant_id",
    "session_id",
    "calibrated",
    "trial_id",
]

_QueueItem = tuple[GazeSample, bool, str] | None  # (sample, pre_calibration, trial_id) or sentinel


def sanitize(value: str) -> str:
    """Prepend apostrophe to block spreadsheet formula execution (Challenge #13)."""
    if value and value[0] in _FORMULA_CHARS:
        return "'" + value
    return value


class CSVLogger:
    """
    Writes GazeSamples to CSV on a dedicated daemon thread.
    log() is non-blocking and thread-safe.
    """

    def __init__(self, path: str, participant_id: str, session_id: str) -> None:
        self._path = path
        self._pid = sanitize(participant_id)
        self._sid = sanitize(session_id)
        self._queue: queue.Queue[_QueueItem] = queue.Queue(maxsize=200)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="csv-logger"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        try:
            self._queue.put(None, timeout=1.0)
        except queue.Full:
            pass
        if self._thread:
            self._thread.join(timeout=5.0)

    def log(self, sample: GazeSample, pre_calibration: bool = False, trial_id: str = "") -> None:
        """Non-blocking; silently drops if logger queue is full."""
        try:
            self._queue.put_nowait((sample, pre_calibration, trial_id))
        except queue.Full:
            pass

    def _run(self) -> None:
        dir_ = os.path.dirname(self._path)
        if dir_:
            os.makedirs(dir_, exist_ok=True)

        try:
            f_handle = open(self._path, "w", newline="", encoding="utf-8")
        except OSError:
            _log.exception("CSVLogger: cannot open %s — logging disabled", self._path)
            return

        with f_handle as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            count = 0

            while not self._stop_event.is_set():
                try:
                    item = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if item is None:
                    break

                sample, pre_calib, trial_id = item
                try:
                    self._write_row(writer, sample, pre_calib, trial_id)
                    count += 1
                    if count % config.CSV_FLUSH_EVERY == 0:
                        f.flush()
                except Exception:
                    _log.exception(
                        "CSVLogger: failed to write row ts_mono_ns=%s — skipping",
                        sample.ts_mono_ns,
                    )

            # Drain remaining items before closing
            while True:
                try:
                    item = self._queue.get_nowait()
                    if item is not None:
                        try:
                            self._write_row(writer, item[0], item[1], item[2])
                        except Exception:
                            _log.exception("CSVLogger: failed to write drain row — skipping")
                except queue.Empty:
                    break
            f.flush()

    def _write_row(
        self, writer: csv.DictWriter, sample: GazeSample, pre_calib: bool, trial_id: str = ""
    ) -> None:
        writer.writerow(
            {
                "ts_wall_ms": f"{sample.ts_wall_ms:.3f}",
                "ts_mono_ns": sample.ts_mono_ns,
                "x": f"{sample.x:.6f}",
                "y": f"{sample.y:.6f}",
                "mesh_certainty": f"{sample.mesh_certainty:.4f}",
                "eye_certainty": f"{sample.eye_certainty:.4f}",
                "head_yaw":   "" if sample.head_yaw   is None else f"{sample.head_yaw:.4f}",
                "head_pitch": "" if sample.head_pitch is None else f"{sample.head_pitch:.4f}",
                "head_roll":  "" if sample.head_roll  is None else f"{sample.head_roll:.4f}",
                "source": sanitize(sample.source),
                "condition": sanitize(sample.condition),
                "participant_id": self._pid,
                "session_id": self._sid,
                "calibrated": "0" if pre_calib else "1",
                "trial_id": sanitize(trial_id),
            }
        )
