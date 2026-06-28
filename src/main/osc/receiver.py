"""
OSC receiver thread — Unity → Python command channel.

Routes incoming OSC messages from Unity to registered callbacks and sends
acknowledgements back via OSCSender.

OSC routes handled:
  /session/start              [participant_id: str] [condition: str]
  /experiment/trial_start     [trial_id: str]
  /experiment/trial_end
  /experiment/session_end
  /gaze/query
  /calibration/start          (legacy — opens local Tkinter window; deprecated)
  /calibration/abort
  /calibration/reset          (Unity-driven flow: clear accumulated points)
  /calibration/sample         [target_x: float] [target_y: float]
  /calibration/compute        (fit model, send /calibration/result)
  /ping
"""
from __future__ import annotations

import logging
import threading
from typing import Callable

from pythonosc.dispatcher import Dispatcher  # type: ignore[import]
from pythonosc.osc_server import ThreadingOSCUDPServer  # type: ignore[import]

from ..gaze.base import GazeSample
from .sender import OSCSender

_log = logging.getLogger(__name__)

_ACK_ADDRESS = "/experiment/ack"
_PONG_ADDRESS = "/pong"


class OSCReceiver:
    """
    Background OSC server that listens for commands from Unity.

    All command routes except /gaze/query and /ping send an
    /experiment/ack [command, status] reply via the OSCSender.

    Replies are always sent to the OSCSender's configured host/port — not
    back to the UDP source address of the incoming packet.
    """

    def __init__(self, host: str, port: int, sender: OSCSender) -> None:
        self._host = host
        self._port = port
        self._sender = sender
        self._server: ThreadingOSCUDPServer | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest_gaze: GazeSample | None = None
        # External callbacks keyed by OSC address
        self._handlers: dict[str, Callable] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_handler(self, address: str, callback: Callable) -> None:
        """Register an external callback for the given OSC address."""
        self._handlers[address] = callback

    def remove_handler(self, address: str) -> None:
        """Unregister the callback for the given OSC address (no-op if not set)."""
        self._handlers.pop(address, None)

    def set_latest_gaze(self, sample: GazeSample) -> None:
        """Thread-safe update of the cached gaze sample (used by /gaze/query)."""
        with self._lock:
            self._latest_gaze = sample

    def start(self) -> None:
        """Create the Dispatcher, register all routes, start the server thread."""
        if self._server is not None:
            return  # already running

        dispatcher = Dispatcher()

        # Register all routes
        dispatcher.map("/session/start", self._handle_session_start)
        dispatcher.map("/experiment/trial_start", self._handle_trial_start)
        dispatcher.map("/experiment/trial_end", self._handle_trial_end)
        dispatcher.map("/experiment/session_end", self._handle_session_end)
        dispatcher.map("/gaze/query", self._handle_gaze_query)
        dispatcher.map("/calibration/start",   self._handle_calibration_start)
        dispatcher.map("/calibration/abort",   self._handle_calibration_abort)
        dispatcher.map("/calibration/reset",   self._handle_calibration_reset)
        dispatcher.map("/calibration/sample",  self._handle_calibration_sample)
        dispatcher.map("/calibration/compute", self._handle_calibration_compute)
        dispatcher.map("/ping", self._handle_ping)

        self._server = ThreadingOSCUDPServer(
            (self._host, self._port), dispatcher
        )
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="osc-receiver",
        )
        self._thread.start()

    def stop(self) -> None:
        """Shut down the server. Safe to call multiple times."""
        if self._server is None:
            return
        try:
            self._server.shutdown()
            self._server.server_close()
        except Exception:
            pass
        finally:
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # ------------------------------------------------------------------
    # Internal handlers
    # ------------------------------------------------------------------

    def _send_ack(self, command: str, status: str) -> None:
        self._sender.send_message(_ACK_ADDRESS, [command, status])

    def _fire_handler(self, address: str, *args) -> None:
        """Call the externally registered callback for *address*, if any."""
        cb = self._handlers.get(address)
        if cb is not None:
            cb(*args)

    def _handle_session_start(self, address: str, *args) -> None:
        pid = str(args[0]) if len(args) > 0 else ""
        condition = str(args[1]) if len(args) > 1 else ""
        try:
            self._fire_handler(address, pid, condition)
            # ACK is deferred: on_session_start() sends it after pipeline.start() so
            # Unity does not receive "ok" before handlers for trial_start/calibration
            # are registered.  Errors are still sent immediately.
        except Exception as exc:
            self._send_ack("session_start", f"error: {exc}")

    def _handle_trial_start(self, address: str, *args) -> None:
        if self._handlers.get(address) is None:
            self._send_ack("trial_start", "error: no active session")
            return
        trial_id = str(args[0]) if args else ""
        try:
            self._fire_handler(address, trial_id)
            self._send_ack("trial_start", "ok")
        except Exception as exc:
            self._send_ack("trial_start", f"error: {exc}")

    def _handle_trial_end(self, address: str, *args) -> None:
        if self._handlers.get(address) is None:
            self._send_ack("trial_end", "error: no active session")
            return
        try:
            self._fire_handler(address)
            self._send_ack("trial_end", "ok")
        except Exception as exc:
            self._send_ack("trial_end", f"error: {exc}")

    def _handle_session_end(self, address: str, *args) -> None:
        if self._handlers.get(address) is None:
            self._send_ack("session_end", "error: no active session")
            return
        try:
            self._fire_handler(address)
            self._send_ack("session_end", "ok")
        except Exception as exc:
            self._send_ack("session_end", f"error: {exc}")

    def _handle_gaze_query(self, address: str, *args) -> None:
        with self._lock:
            sample = self._latest_gaze
        if sample is not None:
            self._sender.send(sample)

    def _handle_calibration_start(self, address: str, *args) -> None:
        if self._handlers.get(address) is None:
            # No handler means no webcam session is active (e.g. IR or NoGaze condition).
            # Send an explicit error ack instead of a misleading "ok".
            self._send_ack("calibration_start", "error: no handler")
            return
        try:
            self._fire_handler(address)
            self._send_ack("calibration_start", "ok")
        except Exception as exc:
            self._send_ack("calibration_start", f"error: {exc}")

    def _handle_calibration_abort(self, address: str, *args) -> None:
        try:
            self._fire_handler(address)
            self._send_ack("calibration_abort", "ok")
        except Exception as exc:
            self._send_ack("calibration_abort", f"error: {exc}")

    def _handle_calibration_reset(self, address: str, *args) -> None:
        """Unity-driven flow: clear accumulated calibration points."""
        try:
            self._fire_handler(address)
            self._send_ack("calibration_reset", "ok")
        except Exception as exc:
            self._send_ack("calibration_reset", f"error: {exc}")

    def _handle_calibration_sample(self, address: str, *args) -> None:
        """Unity-driven flow: capture one gaze sample for target (x, y).

        High-frequency (every ~100 ms during dwell); skipped silently if no
        handler is registered (non-webcam session).  No ACK — too chatty.
        """
        if self._handlers.get(address) is None:
            return
        target_x = float(args[0]) if len(args) > 0 else 0.5
        target_y = float(args[1]) if len(args) > 1 else 0.5
        try:
            self._fire_handler(address, target_x, target_y)
        except Exception as exc:
            _log.warning("calibration_sample handler error: %s", exc)

    def _handle_calibration_compute(self, address: str, *args) -> None:
        """Unity-driven flow: fit the Ridge model and send /calibration/result."""
        if self._handlers.get(address) is None:
            self._send_ack("calibration_compute", "error: no handler")
            return
        try:
            self._fire_handler(address)
            self._send_ack("calibration_compute", "ok")
        except Exception as exc:
            self._send_ack("calibration_compute", f"error: {exc}")

    def _handle_ping(self, address: str, *args) -> None:
        if not self._sender.is_alive:
            _log.warning("_handle_ping: sender not ready, dropping /pong")
            return
        self._sender.send_message(_PONG_ADDRESS, [])
