"""
CoGaze entry point.

Run:
  python -m main                      # full UI, real hardware
  python -m main --mock               # mock gaze sources (no hardware)
  python -m main --no-ir              # webcam only
  python -m main --osc-port 9001      # custom OSC port
"""
from __future__ import annotations

import argparse
import datetime
import logging
import os
import time
import warnings

# Suppress protobuf legacy API warning from MediaPipe's internal imports
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
# Suppress absl/glog INFO and WARNING messages (TFLite XNNPACK, inference_feedback_manager, etc.)
os.environ.setdefault("GLOG_minloglevel", "2")

from . import config
from .gaze.base import MockGazeSource
from .osc.sender import OSCSender
from .osc.receiver import OSCReceiver
from .orchestrator import Pipeline
from .recording.csv_logger import CSVLogger
from .session.session import Session

_log = logging.getLogger(__name__)


def _write_no_gaze_meta(pid: str, condition: str, session, csv_path: str) -> None:
    """Write session metadata JSON for NoGaze condition (no calibration event fires)."""
    import json
    meta = {
        "participant_id": pid,
        "condition": condition,
        "session_id": session.session_id,
        "calib_success": None,
        "calib_aborted": None,
        "calib_ts": None,
        "note": "NoGaze condition — no gaze data recorded",
    }
    meta_path = csv_path.replace(".csv", "_meta.json")
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    except OSError:
        pass


def _setup_logging(output_dir: str) -> None:
    """Configure file logging for gaze diagnostics.

    Writes DEBUG-level detail to data/gaze_debug_<timestamp>.log.
    Console output is suppressed to WARNING to avoid clutter.
    """
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, f"gaze_debug_{ts}.log")

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)-5s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    ))
    root.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    root.addHandler(ch)

    logging.getLogger(__name__).info("Gaze debug log: %s", log_path)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CoGaze eye tracking system")
    p.add_argument("--mock", action="store_true", help="Use mock gaze (no hardware)")
    p.add_argument("--no-ir", action="store_true", help="Disable IR tracker")
    p.add_argument("--no-webcam", action="store_true", help="Disable webcam")
    p.add_argument("--osc-host", default=config.OSC_HOST)
    p.add_argument("--osc-port", type=int, default=config.OSC_PORT)
    p.add_argument("--osc-recv-port", type=int, default=9001,
                   help="UDP port to receive OSC commands from Unity (default: 9001)")
    p.add_argument("--output-dir", default="data")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    _setup_logging(args.output_dir)

    # OSCSender lives for the entire app lifetime so /ping, ACKs, and
    # /calibration/result can be sent even before/between sessions.
    osc = OSCSender(host=args.osc_host, port=args.osc_port)

    pipeline: Pipeline | None = None
    main_win = None  # set by _start() closure below
    _pipeline_stopped = [False]  # guard against double-stop
    _session_gen = [0]           # incremented on restart to kill stale poll loops
    _active_dialog = [None]      # reference to the open SessionSetupDialog (if any)

    # Calibration state shared across on_session_start and _do_restart
    _calibrating = [False]          # True while CalibWindow is open
    _active_calib_win = [None]      # _CalibrationWindow instance, or None

    # OSCReceiver lives for the entire app lifetime so Unity can send /session/start
    # at any time, including before a session has been started from the Python UI.
    osc_receiver = OSCReceiver(args.osc_host, args.osc_recv_port, osc)

    import tkinter as tk
    root = tk.Tk()
    root.withdraw()

    from .ui.app import SessionSetupDialog, MainWindow

    def _start(pid: str, cond: str, _via_osc: bool = False) -> None:
        nonlocal main_win
        # Reject if a session is already running (Unity should send /experiment/session_end first)
        if pipeline is not None:
            _log.warning("/session/start ignored: session already active (pid=%s)", pid)
            osc.send_message("/experiment/ack", ["session_start", "error: session already active"])
            return
        # Validate condition before destroying the dialog (avoids broken UI state)
        from .session.session import VALID_CONDITIONS
        if cond not in VALID_CONDITIONS:
            _log.warning(
                "/session/start rejected: unknown condition %r (valid: %s)",
                cond, sorted(VALID_CONDITIONS),
            )
            if _via_osc:
                osc.send_message("/experiment/ack", ["session_start", f"error: unknown condition {cond!r}"])
            return
        # Close dialog if it is still open (e.g. triggered via /session/start from Unity)
        if _active_dialog[0] is not None:
            try:
                _active_dialog[0].destroy()
            except Exception:
                pass
            _active_dialog[0] = None
        main_win = MainWindow(root, pid, cond)
        main_win.set_restart_callback(_do_restart)
        root.deiconify()
        on_session_start(pid, cond, _via_osc=_via_osc)

    # Allow Unity to start a session remotely via /session/start [pid] [condition].
    # Pass _via_osc=True so on_session_start() sends the deferred ACK after pipeline
    # is fully up (handlers registered, sources started).
    osc_receiver.set_handler(
        "/session/start",
        lambda pid, cond: root.after(0, lambda p=pid, c=cond: _start(p, c, _via_osc=True)),
    )

    # Start OSC sender BEFORE the receiver so /ping and pre-session ACKs work.
    osc.start()
    osc_receiver.start()
    # Signal Unity that Python OSC is fully up — resolves the startup race where
    # Unity times out waiting for /pong during the IR DLL loading window.
    osc.send_ready("1.0")
    _log.info("Sent /ready 1.0 — OSC channel open")

    def on_session_start(pid: str, condition: str, _via_osc: bool = False) -> None:
        nonlocal pipeline, _pipeline_stopped

        session = Session(pid, condition)
        session.lock()

        ts = int(time.time())
        os.makedirs(args.output_dir, exist_ok=True)
        csv_path = os.path.join(args.output_dir, f"{pid}_{condition}_{ts}.csv")
        csv_logger = CSVLogger(csv_path, pid, session.session_id)
        pipeline = Pipeline(session, osc, csv_logger)

        # osc_receiver is already running — just register session-specific handlers.

        webcam_source = None
        _is_no_gaze = (condition == "NoGaze")
        _is_demo = (condition == "Demo")

        if _is_no_gaze:
            # NoGaze: IR hardware present but we deliberately do not connect
            # to any gaze SDK, so no samples are recorded.
            # Write session meta immediately (no calibration event will fire).
            _write_no_gaze_meta(pid, condition, session, csv_path)
        elif _is_demo:
            from .gaze.base import DemoGazeSource
            pipeline.add_source(DemoGazeSource())
        elif args.mock:
            pipeline.add_source(MockGazeSource(source="ir", condition=condition))
            if not args.no_webcam and condition in ("Webcam", "WebcamFiltered"):
                pipeline.add_source(MockGazeSource(source="webcam", condition=condition))
        else:
            if not args.no_ir:
                from .gaze.ir_source import IRGazeSource
                pipeline.add_source(IRGazeSource(condition=condition))
            if not args.no_webcam and condition in ("Webcam", "WebcamFiltered"):
                from .gaze.webcam_source import WebcamGazeSource, FaceMeshBackend
                from . import config as _config
                webcam_source = WebcamGazeSource(
                    condition=condition,
                    use_filter=(condition == "WebcamFiltered"),
                    backend=FaceMeshBackend(_config.FACEMESH_BACKEND),
                    model_path=_config.FACEMESH_MODEL_PATH,
                )
                pipeline.add_source(webcam_source)

        # 1. OSC status → UI (must use after() for thread safety)
        osc.set_status_callback(
            lambda live: root.after(
                0, lambda l=live: main_win is not None and main_win.update_osc_status(l)
            )
        )

        # 1b. OSCReceiver → Pipeline wiring
        def _on_trial_start(tid: str) -> None:
            # Snapshot pipeline locally: this runs on the osc-receiver thread while
            # _do_restart() on the Tk thread may concurrently set pipeline = None.
            # Checking and then calling in two steps risks AttributeError; a local
            # snapshot makes the check and call refer to the same object.
            p = pipeline
            if p is not None:
                p.set_trial_id(tid)
            root.after(0, lambda t=tid: main_win is not None and main_win.update_trial_id(t))

        def _on_trial_end() -> None:
            p = pipeline
            if p is not None:
                p.clear_trial_id()
            root.after(0, lambda: main_win is not None and main_win.update_trial_id(""))

        osc_receiver.set_handler("/experiment/trial_start", _on_trial_start)
        osc_receiver.set_handler("/experiment/trial_end", _on_trial_end)
        osc_receiver.set_handler(
            "/experiment/session_end",
            lambda: root.after(0, _do_restart),
        )

        # 2. Calibration button + OSC /calibration/start (only for webcam conditions)
        if _is_no_gaze:
            main_win.calib_panel.set_not_applicable("NoGaze: 視線追跡なし")
        elif _is_demo:
            main_win.calib_panel.set_not_applicable("Demo: キャリブレーション不要")
        elif webcam_source is not None:
            def _do_calibration():
                if _calibrating[0]:
                    _log.warning("/calibration/start ignored: calibration already in progress")
                    osc.send_message("/experiment/ack", ["calibration_start", "error: already calibrating"])
                    return
                _calibrating[0] = True
                osc.send_message("/calibration/started", [])
                # Reset accumulated points so a retry starts clean
                webcam_source.calibration.reset()
                from .ui.calib_window import run_calibration

                def _on_calib_done(success: bool, err_x: float, err_y: float) -> None:
                    _calibrating[0] = False
                    _active_calib_win[0] = None

                    # Detect ESC abort vs. fit failure
                    # ESC sends (False, 0.0, 0.0); fit failure sends (False, inf, inf)
                    calib_aborted = not success and err_x == 0.0 and err_y == 0.0

                    # Clamp inf/-inf/NaN before sending over OSC
                    _safe = lambda v: v if (v == v and abs(v) != float("inf")) else -1.0
                    safe_x, safe_y = _safe(err_x), _safe(err_y)

                    # 3-state quality: 2=PASS, 1=MARGINAL, 0=FAIL/aborted
                    if success and err_x <= config.CALIB_THRESHOLD_X and err_y <= config.CALIB_THRESHOLD_Y:
                        quality = 2
                    elif success:
                        quality = 1
                    else:
                        quality = 0

                    if main_win is not None:
                        main_win.calib_panel.show_result(success, err_x, err_y)
                    if success and pipeline is not None:
                        pipeline.mark_calibrated()
                    osc.send_calibration_result(quality, safe_x, safe_y)

                    # Write session metadata JSON next to the CSV
                    meta = {
                        "participant_id": pid,
                        "condition": condition,
                        "session_id": session.session_id,
                        "calib_success": success,
                        "calib_aborted": calib_aborted,
                        "calib_err_x": round(safe_x, 6),
                        "calib_err_y": round(safe_y, 6),
                        "calib_threshold_x": config.CALIB_THRESHOLD_X,
                        "calib_threshold_y": config.CALIB_THRESHOLD_Y,
                        "calib_ts": datetime.datetime.now().isoformat(),
                    }
                    meta_path = csv_path.replace(".csv", "_meta.json")
                    try:
                        import json
                        with open(meta_path, "w", encoding="utf-8") as f:
                            json.dump(meta, f, indent=2)
                    except OSError:
                        pass

                _active_calib_win[0] = run_calibration(root, webcam_source, on_done=_on_calib_done)

            main_win.calib_panel.set_callback(_do_calibration)

            # /calibration/abort: cancel any in-progress local calibration window
            def _abort_calibration() -> None:
                win = _active_calib_win[0]
                if win is not None:
                    win._on_esc()

            osc_receiver.set_handler(
                "/calibration/abort",
                lambda: root.after(0, _abort_calibration),
            )

            # ── Unity-driven calibration ──────────────────────────────────────
            # Unity shows the calibration dots and drives timing.
            # Python only captures gaze on demand and fits the model.

            def _on_calib_reset() -> None:
                webcam_source.calibration.reset()
                _log.info("Unity-driven calibration reset.")

            def _on_calib_sample(target_x: float, target_y: float) -> None:
                local = webcam_source.get_local_gaze()
                if local is None:
                    _log.debug("calibration/sample: face not detected — skipping")
                    return
                webcam_source.calibration.add_point(local[0], local[1], target_x, target_y)

            def _on_calib_compute() -> None:
                _log.info("calibration/compute: %d points collected", webcam_source.calibration.point_count)
                result = webcam_source.calibration.fit()
                _safe = lambda v: v if (v == v and abs(v) != float("inf")) else -1.0
                safe_x = _safe(result.validation_error_x)
                safe_y = _safe(result.validation_error_y)
                if (result.success
                        and result.validation_error_x <= config.CALIB_THRESHOLD_X
                        and result.validation_error_y <= config.CALIB_THRESHOLD_Y):
                    quality = 2
                elif result.success:
                    quality = 1
                else:
                    quality = 0
                _log.info(
                    "calibration/compute: success=%s err_x=%.3f err_y=%.3f quality=%d (threshold %.2f/%.2f)",
                    result.success, safe_x, safe_y, quality,
                    config.CALIB_THRESHOLD_X, config.CALIB_THRESHOLD_Y,
                )
                if main_win is not None:
                    main_win.calib_panel.show_result(
                        result.success, result.validation_error_x, result.validation_error_y)
                if result.success and pipeline is not None:
                    pipeline.mark_calibrated()
                osc.send_calibration_result(quality, safe_x, safe_y)
                calib_meta = {
                    "participant_id": pid,
                    "condition": condition,
                    "session_id": session.session_id,
                    "calib_success": result.success,
                    "calib_aborted": False,
                    "calib_err_x": round(safe_x, 6),
                    "calib_err_y": round(safe_y, 6),
                    "calib_threshold_x": config.CALIB_THRESHOLD_X,
                    "calib_threshold_y": config.CALIB_THRESHOLD_Y,
                    "calib_ts": datetime.datetime.now().isoformat(),
                }
                meta_path = csv_path.replace(".csv", "_meta.json")
                try:
                    import json
                    with open(meta_path, "w", encoding="utf-8") as f:
                        json.dump(calib_meta, f, indent=2)
                except OSError:
                    pass

            osc_receiver.set_handler(
                "/calibration/reset",
                lambda: root.after(0, _on_calib_reset),
            )
            osc_receiver.set_handler(
                "/calibration/sample",
                lambda tx, ty: root.after(0, lambda x=tx, y=ty: _on_calib_sample(x, y)),
            )
            osc_receiver.set_handler(
                "/calibration/compute",
                lambda: root.after(0, _on_calib_compute),
            )

        elif not _is_no_gaze and not _is_demo:
            # IR condition: hardware tracks but no calibration step
            main_win.calib_panel.set_not_applicable("IR: no calibration step")

        # 3. Camera preview + face-position guide (webcam only)
        if webcam_source is not None:
            main_win.start_preview(webcam_source)
            main_win.enable_face_guide()
            # Sync "Debug Mode" checkbox → visual overlay on camera preview
            def _on_debug_toggle(*_):
                if main_win is not None:
                    webcam_source.set_debug_mode(main_win._debug_var.get())
            main_win._debug_var.trace_add("write", _on_debug_toggle)

        # 4. Gaze update polling loop (~30 fps).
        # gen is captured per-session so restarting increments _session_gen and
        # kills this loop without needing an explicit after_cancel().
        gen = _session_gen[0]

        def _poll_gaze():
            if _session_gen[0] != gen:
                return  # session restarted; stop this loop
            if pipeline is not None:
                sample = pipeline.get_latest_gaze()
                if sample is not None:
                    if main_win is not None:
                        main_win.update_gaze(sample.x, sample.y, sample.source)
                        main_win.update_certainty(sample.mesh_certainty, sample.eye_certainty)
                    osc_receiver.set_latest_gaze(sample)
            # Face-position guide: update Python UI overlay AND send to Unity via OSC
            if webcam_source is not None:
                metrics = webcam_source.get_face_metrics()
                if main_win is not None:
                    main_win.update_face_guide(metrics)
                # Compute status code for Unity (mirrors FaceGuideOverlay colour logic)
                if metrics is None:
                    osc.send_face_metrics(0.0, 0.0, 0.0, 0)  # 0 = no face
                else:
                    diff = metrics.iod_norm - config.IOD_TARGET_NORM
                    tol  = config.IOD_TOLERANCE_NORM
                    if abs(diff) <= tol:
                        status = 2   # good
                    elif diff > tol:
                        status = 3   # too close
                    else:
                        status = 1   # too far
                    osc.send_face_metrics(
                        metrics.iod_norm, metrics.face_cx, metrics.face_cy, status
                    )
            root.after(33, _poll_gaze)

        root.after(33, _poll_gaze)

        # 5. Window close → pipeline stop (guard against double-stop from finally)
        def _stop_once():
            if not _pipeline_stopped[0]:
                _pipeline_stopped[0] = True
                if pipeline is not None:
                    pipeline.stop()

        main_win.set_close_callback(_stop_once)

        pipeline.start()
        # Signal Unity that the session pipeline is live and /calibration/start
        # can now be accepted.
        osc.send_ready("session")
        _log.info("Sent /ready session — pipeline started (pid=%s, cond=%s)", pid, condition)
        # Deferred ACK: sent here (after all handlers registered + pipeline started)
        # instead of immediately in _handle_session_start, so Unity's next message
        # does not arrive before trial_start/calibration_start handlers are in place.
        if _via_osc:
            osc.send_message("/experiment/ack", ["session_start", "ok"])

    def _do_restart() -> None:
        """Stop current session and show the session setup dialog again."""
        nonlocal pipeline, main_win, _pipeline_stopped
        # Guard against double invocation: /session_end OSC (root.after) and
        # the user's "Change Session" button can both enqueue a restart in the
        # same Tk event cycle.  After the first call all three are None, so a
        # second call would destroy the newly-created SessionSetupDialog.
        if pipeline is None and main_win is None and _active_calib_win[0] is None:
            return
        _session_gen[0] += 1  # kill active poll loop

        # Cancel any running calibration window before tearing down the session.
        # Calling _on_esc() triggers _on_calib_done (resets _calibrating/_active_calib_win)
        # and destroys the Toplevel cleanly.
        win = _active_calib_win[0]
        if win is not None:
            try:
                win._on_esc()
            except Exception:
                pass
            _calibrating[0] = False
            _active_calib_win[0] = None

        if not _pipeline_stopped[0]:
            _pipeline_stopped[0] = True
            if pipeline is not None:
                pipeline.stop()
        pipeline = None
        main_win = None
        _pipeline_stopped = [False]
        # Clear session-specific OSC handlers; keep /session/start alive
        for addr in (
            "/experiment/trial_start", "/experiment/trial_end",
            "/experiment/session_end",
            "/calibration/start", "/calibration/abort",
            "/calibration/reset", "/calibration/sample", "/calibration/compute",
        ):
            osc_receiver.remove_handler(addr)
        root.withdraw()
        for w in root.winfo_children():
            try:
                w.destroy()
            except Exception:
                pass
        dialog = SessionSetupDialog(root, on_start=_start)
        _active_dialog[0] = dialog

    dialog = SessionSetupDialog(root, on_start=_start)
    _active_dialog[0] = dialog

    try:
        root.mainloop()
    finally:
        if pipeline is not None and not _pipeline_stopped[0]:
            _pipeline_stopped[0] = True
            pipeline.stop()
        osc_receiver.stop()
        osc.stop()


if __name__ == "__main__":
    main()
