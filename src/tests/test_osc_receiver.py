"""Tests for OSCReceiver — bidirectional OSC communication (Unity → Python)."""
from __future__ import annotations

import socket
import threading
import time

import pytest

from pythonosc import udp_client  # type: ignore[import]

from main.gaze.base import GazeSample
from main.osc.sender import OSCSender
from main.osc.receiver import OSCReceiver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _sample() -> GazeSample:
    return GazeSample(
        x=0.5, y=0.3, mesh_certainty=0.9, eye_certainty=0.8,
        source="ir", condition="IR",
        ts_wall_ms=1000.0, ts_mono_ns=1_000_000_000,
    )


def _recv_udp(port: int, timeout: float = 1.0) -> bytes | None:
    """Bind a socket on *port* and wait for a single UDP packet."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", port))
    sock.settimeout(timeout)
    try:
        data, _ = sock.recvfrom(4096)
        return data
    except socket.timeout:
        return None
    finally:
        sock.close()


def _make_receiver(recv_port: int, reply_port: int) -> tuple[OSCSender, OSCReceiver]:
    """
    Build an OSCSender pointed at *reply_port* (where the test listens) and
    an OSCReceiver that listens on *recv_port* (where we send commands).
    """
    sender = OSCSender("127.0.0.1", reply_port)
    sender.start()
    receiver = OSCReceiver("127.0.0.1", recv_port, sender)
    receiver.start()
    time.sleep(0.05)  # let server thread bind
    return sender, receiver


def _send_osc(port: int, address: str, *args) -> None:
    """Send a single OSC message to localhost:port."""
    client = udp_client.SimpleUDPClient("127.0.0.1", port)
    client.send_message(address, list(args) if args else [])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOSCReceiverPing:
    def test_ping_sends_pong_packet(self):
        recv_port = _free_port()
        reply_port = _free_port()

        received: list[bytes] = []

        def listen():
            data = _recv_udp(reply_port, timeout=2.0)
            if data:
                received.append(data)

        t = threading.Thread(target=listen, daemon=True)
        t.start()

        sender, receiver = _make_receiver(recv_port, reply_port)
        try:
            _send_osc(recv_port, "/ping")
            t.join(timeout=2.5)
        finally:
            receiver.stop()
            sender.stop()

        assert len(received) > 0, "Expected a /pong UDP packet back"
        assert b"pong" in received[0]


class TestOSCReceiverTrialStart:
    def test_trial_start_fires_callback_with_trial_id(self):
        recv_port = _free_port()
        reply_port = _free_port()

        fired: list[str] = []

        sender, receiver = _make_receiver(recv_port, reply_port)
        receiver.set_handler("/experiment/trial_start", lambda tid: fired.append(tid))
        try:
            _send_osc(recv_port, "/experiment/trial_start", "T42")
            time.sleep(0.2)
        finally:
            receiver.stop()
            sender.stop()

        assert fired == ["T42"], f"Expected callback with 'T42', got {fired}"


class TestOSCReceiverTrialEnd:
    def test_trial_end_fires_callback(self):
        recv_port = _free_port()
        reply_port = _free_port()

        fired: list[bool] = []

        sender, receiver = _make_receiver(recv_port, reply_port)
        receiver.set_handler("/experiment/trial_end", lambda: fired.append(True))
        try:
            _send_osc(recv_port, "/experiment/trial_end")
            time.sleep(0.2)
        finally:
            receiver.stop()
            sender.stop()

        assert fired == [True], f"Expected callback to fire, got {fired}"


class TestOSCReceiverGazeQuery:
    def test_gaze_query_sends_udp_packet_back(self):
        recv_port = _free_port()
        reply_port = _free_port()

        received: list[bytes] = []

        def listen():
            data = _recv_udp(reply_port, timeout=2.0)
            if data:
                received.append(data)

        t = threading.Thread(target=listen, daemon=True)
        t.start()

        sender, receiver = _make_receiver(recv_port, reply_port)
        receiver.set_latest_gaze(_sample())
        try:
            _send_osc(recv_port, "/gaze/query")
            t.join(timeout=2.5)
        finally:
            receiver.stop()
            sender.stop()

        assert len(received) > 0, "Expected /gaze reply UDP packet after /gaze/query"
        assert b"gaze" in received[0]


class TestOSCReceiverAck:
    def test_trial_start_sends_ack_ok_when_handler_registered(self):
        """With a session handler registered, trial_start should ACK 'ok'."""
        recv_port = _free_port()
        reply_port = _free_port()

        received: list[bytes] = []

        def listen():
            data = _recv_udp(reply_port, timeout=2.0)
            if data:
                received.append(data)

        t = threading.Thread(target=listen, daemon=True)
        t.start()

        sender, receiver = _make_receiver(recv_port, reply_port)
        receiver.set_handler("/experiment/trial_start", lambda tid: None)
        try:
            _send_osc(recv_port, "/experiment/trial_start", "T01")
            t.join(timeout=2.5)
        finally:
            receiver.stop()
            sender.stop()

        assert len(received) > 0, "Expected an /experiment/ack packet"
        payload = received[0]
        assert b"trial_start" in payload
        assert b"ok" in payload

    def test_trial_start_sends_error_when_no_handler(self):
        """Without a session handler, trial_start should ACK 'error: no active session'."""
        recv_port = _free_port()
        reply_port = _free_port()

        received: list[bytes] = []

        def listen():
            data = _recv_udp(reply_port, timeout=2.0)
            if data:
                received.append(data)

        t = threading.Thread(target=listen, daemon=True)
        t.start()

        sender, receiver = _make_receiver(recv_port, reply_port)
        try:
            _send_osc(recv_port, "/experiment/trial_start", "T01")
            t.join(timeout=2.5)
        finally:
            receiver.stop()
            sender.stop()

        assert len(received) > 0, "Expected an /experiment/ack error packet"
        payload = received[0]
        assert b"trial_start" in payload
        assert b"no active session" in payload


class TestOSCReceiverIdempotentStop:
    def test_stop_twice_does_not_raise(self):
        recv_port = _free_port()
        reply_port = _free_port()

        sender, receiver = _make_receiver(recv_port, reply_port)
        try:
            receiver.stop()
            receiver.stop()  # second stop must not raise
        finally:
            sender.stop()
