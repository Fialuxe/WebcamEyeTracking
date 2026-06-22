"""
OSC UDP sender thread (Challenge #1).

Message format: /gaze  x(f)  y(f)  mesh_certainty(f)  eye_certainty(f)  source(s)  condition(s)

Status callback fires True when first packet sends successfully, False when
the queue drains without a send (i.e., dropped connection or idle — Challenge #11).
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Callable

from pythonosc import udp_client  # type: ignore[import]

from ..gaze.base import GazeSample

_log = logging.getLogger(__name__)

_SENTINEL = None


class OSCSender:
    """
    Background thread that reads GazeSample from its internal queue and
    sends OSC /gaze messages over UDP.
    Call send() from any thread; old samples are dropped when queue is full.
    """

    ADDRESS = "/gaze"
    _IDLE_TIMEOUT_S = 0.5

    def __init__(self, host: str = "127.0.0.1", port: int = 9000) -> None:
        self._host = host
        self._port = port
        self._queue: queue.Queue[GazeSample | None] = queue.Queue(maxsize=2)
        self._client: udp_client.SimpleUDPClient | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._status_cb: Callable[[bool], None] | None = None

    def set_status_callback(self, cb: Callable[[bool], None]) -> None:
        self._status_cb = cb

    @property
    def is_alive(self) -> bool:
        """True if the sender thread is running and the UDP client is ready."""
        return (
            self._thread is not None
            and self._thread.is_alive()
            and self._client is not None
        )

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return  # already running — idempotent
        self._stop_event.clear()
        client = udp_client.SimpleUDPClient(self._host, self._port)
        # Non-blocking socket: control messages (pong, ack) are tiny and must
        # never stall the caller.  A full send buffer on localhost UDP is
        # virtually unreachable; if it happens the send raises BlockingIOError
        # which the broad except in send_message will catch and log.
        client._sock.setblocking(False)
        self._client = client
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="osc-sender"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        try:
            self._queue.put_nowait(_SENTINEL)
        except queue.Full:
            pass
        if self._thread:
            self._thread.join(timeout=2.0)

    def send_message(self, address: str, args: list) -> None:
        """Send an arbitrary OSC message immediately (not queued). Safe to call from any thread."""
        if self._client is None:
            _log.warning("send_message(%s): client not ready, dropping", address)
            return
        try:
            self._client.send_message(address, args)
        except Exception as exc:
            _log.warning("send_message(%s) failed: %s", address, exc)

    def send_face_metrics(
        self, iod_norm: float, face_cx: float, face_cy: float, status: int
    ) -> None:
        """Send real-time face-position data to Unity: /face/metrics [iod_norm] [cx] [cy] [status].

        status: 0=no face, 1=too far, 2=good, 3=too close.
        iod_norm: IOD as fraction of camera frame width.
        cx, cy: face centre in [0, 1] normalised screen space.
        """
        self.send_message(
            "/face/metrics",
            [float(iod_norm), float(face_cx), float(face_cy), int(status)],
        )

    def send_ready(self, state: str = "1.0") -> None:
        """Notify Unity: /ready [state: str].

        Sent when Python OSC is fully initialised ('1.0') or when a session
        pipeline has started and calibration commands can be accepted ('session').
        """
        self.send_message("/ready", [state])

    def send_calibration_result(self, quality: int, err_x: float, err_y: float) -> None:
        """Notify Unity: /calibration/result [quality] [err_x] [err_y].

        quality: 2=PASS, 1=MARGINAL, 0=FAIL or aborted.
        err values are clamped: -1.0 replaces inf/NaN (aborted with too few points).
        """
        _safe = lambda v: v if (v == v and v != float("inf") and v != float("-inf")) else -1.0
        self.send_message("/calibration/result", [quality, _safe(err_x), _safe(err_y)])

    def send(self, sample: GazeSample) -> None:
        """Non-blocking send; drops oldest frame if queue is full."""
        try:
            self._queue.put_nowait(sample)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(sample)
            except queue.Full:
                pass

    def _run(self) -> None:
        alive = False

        while not self._stop_event.is_set():
            try:
                sample = self._queue.get(timeout=self._IDLE_TIMEOUT_S)
            except queue.Empty:
                if alive:
                    alive = False
                    self._notify(False)
                continue

            if sample is _SENTINEL:
                break

            try:
                if self._client is None:
                    break
                self._client.send_message(
                    self.ADDRESS,
                    [sample.x, sample.y, sample.mesh_certainty, sample.eye_certainty,
                     sample.source, sample.condition],
                )
                if not alive:
                    alive = True
                    self._notify(True)
            except Exception:
                if alive:
                    alive = False
                    self._notify(False)

    def _notify(self, live: bool) -> None:
        if self._status_cb:
            try:
                self._status_cb(live)
            except Exception:
                pass
