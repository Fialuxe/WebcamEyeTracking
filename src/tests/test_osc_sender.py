"""Tests for OSCSender (Challenge #1 and #11)."""
import socket
import threading
import time

import pytest

from main.gaze.base import GazeSample
from main.osc.sender import OSCSender


def _sample() -> GazeSample:
    return GazeSample(
        x=0.5, y=0.3, mesh_certainty=0.9, eye_certainty=0.8,
        source="ir", condition="IR",
        ts_wall_ms=1000.0, ts_mono_ns=1_000_000_000,
    )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestOSCSender:
    def test_sends_udp_packet(self):
        port = _free_port()
        received: list[bytes] = []

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("127.0.0.1", port))
        sock.settimeout(1.0)

        def listen():
            try:
                data, _ = sock.recvfrom(1024)
                received.append(data)
            except socket.timeout:
                pass

        t = threading.Thread(target=listen, daemon=True)
        t.start()

        sender = OSCSender("127.0.0.1", port)
        sender.start()
        sender.send(_sample())
        t.join(timeout=2.0)
        sender.stop()
        sock.close()

        assert len(received) > 0

    def test_status_callback_live(self):
        port = _free_port()
        statuses: list[bool] = []
        sender = OSCSender("127.0.0.1", port)
        sender.set_status_callback(statuses.append)
        sender.start()
        sender.send(_sample())
        time.sleep(0.3)
        sender.stop()
        assert True in statuses

    def test_status_goes_dead_on_idle(self):
        port = _free_port()
        statuses: list[bool] = []
        sender = OSCSender("127.0.0.1", port)
        sender.set_status_callback(statuses.append)
        sender.start()
        sender.send(_sample())
        time.sleep(1.0)  # wait past _IDLE_TIMEOUT_S (0.5s)
        sender.stop()
        # Should have transitioned live → dead
        assert True in statuses
        assert False in statuses

    def test_stop_is_idempotent(self):
        port = _free_port()
        sender = OSCSender("127.0.0.1", port)
        sender.start()
        sender.stop()
        sender.stop()  # second stop must not raise

    def test_send_before_start_does_not_crash(self):
        sender = OSCSender("127.0.0.1", _free_port())
        # Calling send before start should not raise (queue accepts items)
        sender.send(_sample())
