"""
Tobii EyeX gaze source via Core SDK (Tobii.Interaction) and pythonnet.
Requires 64-bit Python, EyeX Engine running, and device connected.
x64 native DLL is placed next to the process working directory at startup.

DLL loading and Host() initialisation run on a background thread so the
Tk main thread is not blocked (clr.AddReference + Host() can take 5–15 s).
"""
from __future__ import annotations

import logging
import os
import queue
import shutil
import struct
import sys
import threading
import time

from .base import GazeSource, GazeSample, _put_drop_oldest
from .. import config

_log = logging.getLogger(__name__)


def _ensure_x64_dll() -> None:
    """Copy x64 native DLL to cwd so the .NET runtime can locate it."""
    dst = os.path.join(os.getcwd(), "Tobii.EyeX.Client.dll")
    if not os.path.exists(dst):
        shutil.copy2(config.TOBII_X64_DLL_SRC, dst)


class IRGazeSource(GazeSource):
    """
    Tobii EyeX gaze source.
    Emits GazeSample(source="ir") into out_queue via the SDK callback thread.

    start() is non-blocking: DLL loading and Host() creation happen on a
    dedicated background thread so the Tk main thread stays responsive.
    Gaze samples flow once the SDK is connected (~5–15 s on first call).
    """

    def __init__(
        self,
        condition: str = "IR",
        screen_w: int = config.SCREEN_W_PX,
        screen_h: int = config.SCREEN_H_PX,
    ) -> None:
        if struct.calcsize("P") * 8 != 64:
            raise RuntimeError("IRGazeSource requires 64-bit Python")
        self._condition = condition
        self._screen_w = screen_w
        self._screen_h = screen_h
        self._host = None
        self._stream = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self, out_queue: queue.Queue) -> None:
        """Non-blocking: starts SDK initialisation on a background thread."""
        _ensure_x64_dll()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._connect, args=(out_queue,), daemon=True, name="ir-gaze"
        )
        self._thread.start()

    def _connect(self, out_queue: queue.Queue) -> None:
        """Load Tobii SDK, register gaze callback, then wait for stop()."""
        try:
            import clr  # type: ignore[import]
            clr.AddReference(os.path.join(config.TOBII_LIB_PATH, "Tobii.Interaction.Model"))
            clr.AddReference(os.path.join(config.TOBII_LIB_PATH, "Tobii.Interaction.Net"))
            from Tobii.Interaction import Host  # type: ignore[import]
            from Tobii.Interaction.Framework import GazePointDataMode  # type: ignore[import]
        except Exception as exc:
            _log.error("IRGazeSource: failed to load Tobii SDK: %s", exc)
            print(f"[ir_source] SDK load failed: {exc}", file=sys.stderr)
            return

        w = self._screen_w
        h = self._screen_h
        cond = self._condition
        stop = self._stop_event

        def _on_gaze(gaze_point, _async_data):
            if stop.is_set():
                return
            sample = GazeSample(
                x=gaze_point.X / w,
                y=gaze_point.Y / h,
                mesh_certainty=1.0 if gaze_point.IsValid else 0.0,
                eye_certainty=1.0 if gaze_point.IsValid else 0.0,
                source="ir",
                condition=cond,
                ts_wall_ms=time.time() * 1000,
                ts_mono_ns=time.monotonic_ns(),
            )
            _put_drop_oldest(out_queue, sample)

        try:
            self._host = Host()
            self._stream = self._host.Streams.CreateGazePointDataStream(
                GazePointDataMode.LightlyFiltered
            )
            self._stream.Next += _on_gaze
            _log.info("IRGazeSource: Tobii SDK connected, streaming gaze data")
        except Exception as exc:
            _log.error("IRGazeSource: failed to connect to Tobii host: %s", exc)
            return

        # Keep the thread alive; gaze callbacks fire on .NET thread pool.
        self._stop_event.wait()

    def stop(self) -> None:
        self._stop_event.set()
        # Join BEFORE checking _host: _connect() sets self._host on the ir-gaze
        # thread.  Disposing before joining could race with a still-running
        # _connect() and leave an undisposed Host if stop() was called while SDK
        # loading was still in progress.
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        if self._host is not None:
            try:
                self._host.Dispose()
            except Exception as exc:
                _log.warning("IRGazeSource: Dispose() raised: %s", exc)
            self._host = None
            self._stream = None
