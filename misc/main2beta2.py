"""
main2beta2 — Implicit calibration: text reading + PACE (click-based RLS).

Run:
  python src/main2beta2.py [args]

Additional args:
  --pure-implicit       Skip explicit (grid) calibration, start from text calib
                        then PACE from identity prior.
  --md-text PATH        Path to a .md file for text calibration screen.
  --pace-min-updates N  Minimum clicks before PACE kicks in (default: 5).
  --no-pace             Disable PACE (explicit or text calibration only).

Calibration flow (default, --pure-implicit NOT set):
  1. Explicit 9-point grid calibration   → warm-start PACE from calibrated ridge
  2. PACE refines automatically on every mouse click inside any window

Pure-implicit flow (--pure-implicit):
  1. Text calibration window (reading passage) → fit CalibrationManager → warm-start PACE
  2. PACE refines on every click; gaze output appears after pace-min-updates clicks
  NOTE: until pace-min-updates clicks are registered, no gaze output is produced.

Threading: PACE.update() runs on the Tk main thread (click callback).
           PACE.predict() runs on the camera background thread (_process_landmarks).
           Both access PACECalibration under an internal lock in WebcamGazeSourceBeta2.
"""
from __future__ import annotations

import argparse
import datetime
import logging
import os
import sys
import time
import warnings

# Suppress protobuf legacy API warning from MediaPipe's internal imports
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
# Suppress absl/glog INFO and WARNING messages (TFLite XNNPACK etc.)
os.environ.setdefault("GLOG_minloglevel", "2")

# Ensure src/ is on sys.path when run as: python src/main2beta2.py
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from main import config
from main.gaze.base import MockGazeSource
from main.osc.sender import OSCSender
from main.orchestrator import Pipeline
from main.recording.csv_logger import CSVLogger
from main.session.session import Session


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging(output_dir: str) -> None:
    """Configure file logging for gaze diagnostics (same as __main__.py)."""
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, f"gaze_debug_beta2_{ts}.log")

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

    logging.getLogger(__name__).info("Beta2 gaze debug log: %s", log_path)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CoGaze beta2 — implicit calibration (PACE + text reading)"
    )
    p.add_argument("--mock", action="store_true", help="Use mock gaze (no hardware)")
    p.add_argument("--no-ir", action="store_true", help="Disable IR tracker")
    p.add_argument("--no-webcam", action="store_true", help="Disable webcam")
    p.add_argument("--osc-host", default=config.OSC_HOST)
    p.add_argument("--osc-port", type=int, default=config.OSC_PORT)
    p.add_argument("--output-dir", default="data")
    # Beta2-specific
    p.add_argument(
        "--pure-implicit",
        action="store_true",
        help="Skip explicit grid calibration; use text calib + PACE from identity prior",
    )
    p.add_argument(
        "--md-text",
        default=None,
        metavar="PATH",
        help="Path to a .md file used for text calibration (default: built-in passage)",
    )
    p.add_argument(
        "--pace-min-updates",
        type=int,
        default=5,
        metavar="N",
        help="Clicks required before PACE prediction kicks in (default: 5)",
    )
    p.add_argument(
        "--no-pace",
        action="store_true",
        help="Disable PACE; calibration is explicit or text only",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: C901
    args = _parse_args()
    _setup_logging(args.output_dir)

    _log = logging.getLogger(__name__)

    osc = OSCSender(host=args.osc_host, port=args.osc_port)
    pipeline: Pipeline | None = None
    webcam_source = None          # nonlocal reference kept so click handler is safe
    main_win = None
    _pipeline_stopped = [False]
    _session_gen = [0]
    _active_calib_win = [None]    # _CalibrationWindow or _TextCalibWindow, or None
    _pace_bind_id = [None]        # funcid of the per-session <Button-1> binding

    # ------------------------------------------------------------------
    # Global click hook (captures Tk window clicks for PACE updates)
    # ------------------------------------------------------------------

    def _on_global_click(event) -> None:
        """
        Converts a Tk button-1 event to normalised screen coords and
        feeds it to PACE via webcam_source.on_click().

        UI widget clicks are excluded to avoid contaminating PACE training
        data with button/checkbox positions.
        Runs on the Tk main thread — safe to call webcam_source.on_click().
        """
        import tkinter as tk
        from tkinter import ttk
        # UI ウィジェット上のクリックは PACE の学習データから除外
        if isinstance(event.widget, (tk.Button, ttk.Button, ttk.Checkbutton,
                                      tk.Checkbutton, ttk.Radiobutton)):
            return
        if webcam_source is None:
            return
        sw = event.widget.winfo_screenwidth()
        sh = event.widget.winfo_screenheight()
        if sw <= 0 or sh <= 0:
            return
        sx = event.x_root / sw
        sy = event.y_root / sh
        webcam_source.on_click(sx, sy)

    # ------------------------------------------------------------------
    # Session start
    # ------------------------------------------------------------------

    def on_session_start(pid: str, condition: str) -> None:
        nonlocal pipeline, _pipeline_stopped, webcam_source

        session = Session(pid, condition)
        session.lock()

        ts = int(time.time())
        os.makedirs(args.output_dir, exist_ok=True)
        csv_path = os.path.join(args.output_dir, f"{pid}_{condition}_{ts}_beta2.csv")
        csv_logger = CSVLogger(csv_path, pid, session.session_id)
        pipeline = Pipeline(session, osc, csv_logger)

        webcam_source = None

        if args.mock:
            pipeline.add_source(MockGazeSource(source="ir", condition=condition))
            if not args.no_webcam and condition in ("Webcam", "WebcamFiltered"):
                pipeline.add_source(MockGazeSource(source="webcam", condition=condition))
        else:
            if not args.no_ir:
                from main.gaze.ir_source import IRGazeSource
                pipeline.add_source(IRGazeSource(condition=condition))

            if not args.no_webcam and condition in ("Webcam", "WebcamFiltered"):
                from main.gaze.webcam_source import FaceMeshBackend
                from main.gaze.webcam_source_beta2 import WebcamGazeSourceBeta2
                webcam_source = WebcamGazeSourceBeta2(
                    condition=condition,
                    use_filter=(condition == "WebcamFiltered"),
                    backend=FaceMeshBackend(config.FACEMESH_BACKEND),
                    model_path=config.FACEMESH_MODEL_PATH,
                    use_pace=not args.no_pace,
                    pace_min_updates=args.pace_min_updates,
                    pure_implicit=args.pure_implicit,
                )
                pipeline.add_source(webcam_source)

        # 1. OSC status → UI
        osc.set_status_callback(
            lambda live: root.after(
                0, lambda l=live: main_win is not None and main_win.update_osc_status(l)
            )
        )

        # 2. Calibration button
        if webcam_source is not None:
            def _do_calibration() -> None:
                if args.pure_implicit:
                    _run_text_calibration()
                else:
                    _run_explicit_calibration()

            main_win.calib_panel.set_callback(_do_calibration)

            # Bind global click to PACE; save funcid so _do_restart() can remove
            # only this binding without disturbing other application-level bindings.
            _pace_bind_id[0] = root.bind_all("<Button-1>", _on_global_click, add="+")

        # 3. Camera preview
        if webcam_source is not None:
            main_win.start_preview(webcam_source)

            def _on_debug_toggle(*_):
                if main_win is not None:
                    webcam_source.set_debug_mode(main_win._debug_var.get())

            main_win._debug_var.trace_add("write", _on_debug_toggle)

        # 4. Gaze polling loop (~30 fps) with PACE status display
        gen = _session_gen[0]
        _pace_log_counter = [0]

        def _poll_gaze() -> None:
            if _session_gen[0] != gen:
                return  # session restarted
            if pipeline is not None:
                sample = pipeline.get_latest_gaze()
                warming = False
                # pure_implicit モード専用: PACE 未準備期間は gaze 表示の代わりに状態を表示
                if args.pure_implicit and webcam_source is not None and not args.no_pace:
                    pace = webcam_source.get_pace()
                    if not pace.is_ready:
                        warming = True
                        if main_win is not None:
                            remaining = max(0, args.pace_min_updates - pace.update_count)
                            main_win._gaze_var.set(f"PACE暖機中: あと{remaining}クリック必要")
                if sample is not None and main_win is not None and not warming:
                    main_win.update_gaze(sample.x, sample.y, sample.source)

            # PACE status logging (~once per second at 30 fps) + UI update
            if webcam_source is not None and not args.no_pace:
                _pace_log_counter[0] += 1
                if _pace_log_counter[0] % 30 == 0:
                    pace = webcam_source.get_pace()
                    if main_win is not None:
                        main_win.update_pace_status(
                            pace.update_count, pace.accuracy_estimate, pace.is_ready
                        )
                    acc = pace.accuracy_estimate
                    _log.info(
                        "PACE status: updates=%d accuracy_estimate=%s",
                        pace.update_count,
                        f"{acc:.4f}" if acc >= 0 else "n/a",
                    )

            root.after(33, _poll_gaze)

        root.after(33, _poll_gaze)

        # 5. Window close
        def _stop_once() -> None:
            if not _pipeline_stopped[0]:
                _pipeline_stopped[0] = True
                if pipeline is not None:
                    pipeline.stop()

        main_win.set_close_callback(_stop_once)
        pipeline.start()

    # ------------------------------------------------------------------
    # Calibration helpers (defined after on_session_start so they close
    # over the nonlocal `webcam_source` and `pipeline` after assignment)
    # ------------------------------------------------------------------

    def _run_explicit_calibration() -> None:
        """9-point grid calibration → warm-start PACE."""
        from main.ui.calib_window import run_calibration

        def _on_calib_done(success: bool, err_x: float, err_y: float) -> None:
            _active_calib_win[0] = None
            if main_win is not None:
                main_win.calib_panel.show_result(success, err_x, err_y)
            if success:
                if webcam_source is not None:
                    webcam_source.promote_to_pace()
                if pipeline is not None:
                    pipeline.mark_calibrated()

        _active_calib_win[0] = run_calibration(root, webcam_source, on_done=_on_calib_done)

    def _run_text_calibration() -> None:
        """Text reading calibration → warm-start PACE (pure_implicit mode)."""
        from main.ui.text_calib_window import run_text_calibration, DEFAULT_CALIB_TEXT

        md_text = DEFAULT_CALIB_TEXT
        if args.md_text:
            try:
                with open(args.md_text, encoding="utf-8") as fh:
                    md_text = fh.read()
            except OSError as exc:
                _log.warning("Could not read --md-text file: %s", exc)

        def _on_text_done(success: bool, n_samples: int) -> None:
            _active_calib_win[0] = None
            if main_win is not None:
                # Show result; err values not available from text calib
                main_win.calib_panel.show_result(success, 0.0, 0.0)
            if success:
                if webcam_source is not None:
                    # CalibrationManager was fitted inside text_calib_window
                    webcam_source.promote_to_pace()
                if pipeline is not None:
                    pipeline.mark_calibrated()
            _log.info(
                "Text calibration done: success=%s n_samples=%d", success, n_samples
            )

        _active_calib_win[0] = run_text_calibration(
            root,
            webcam_source,
            _on_text_done,
            md_text=md_text,
        )

    # ------------------------------------------------------------------
    # Session restart
    # ------------------------------------------------------------------

    def _do_restart() -> None:
        nonlocal pipeline, main_win, _pipeline_stopped, webcam_source
        # Guard: /session_end (root.after) and user "Change Session" can both
        # enqueue a restart in the same Tk cycle; prevent destroying the
        # SessionSetupDialog the first call created.
        if pipeline is None and main_win is None and _active_calib_win[0] is None:
            return
        _session_gen[0] += 1

        # Close any open calibration window before tearing down the session.
        # force_close() / _on_esc() avoids the confirmation messagebox that
        # would otherwise block the restart when samples have been collected.
        win = _active_calib_win[0]
        if win is not None:
            try:
                if hasattr(win, "force_close"):
                    win.force_close()   # _TextCalibWindow — no messagebox
                else:
                    win._on_esc()       # _CalibrationWindow — already safe
            except Exception:
                pass
            _active_calib_win[0] = None

        if not _pipeline_stopped[0]:
            _pipeline_stopped[0] = True
            if pipeline is not None:
                pipeline.stop()
        pipeline = None
        webcam_source = None   # guard click handler during dialog
        main_win = None
        _pipeline_stopped = [False]
        # Remove only the per-session PACE binding (by funcid) so any other
        # application-level <Button-1> bindings are not disturbed.
        fid = _pace_bind_id[0]
        if fid is not None:
            try:
                root.unbind_all("<Button-1>", fid)
            except Exception:
                pass
            _pace_bind_id[0] = None
        root.withdraw()
        for w in root.winfo_children():
            try:
                w.destroy()
            except Exception:
                pass
        from main.ui.app import SessionSetupDialog
        SessionSetupDialog(root, on_start=_start)

    # ------------------------------------------------------------------
    # Tk root and startup
    # ------------------------------------------------------------------

    import tkinter as tk
    root = tk.Tk()
    root.withdraw()

    from main.ui.app import SessionSetupDialog, MainWindow

    def _start(pid: str, cond: str) -> None:
        nonlocal main_win
        main_win = MainWindow(root, pid, cond)
        main_win.set_restart_callback(_do_restart)
        root.deiconify()
        on_session_start(pid, cond)

    SessionSetupDialog(root, on_start=_start)

    try:
        root.mainloop()
    finally:
        if pipeline is not None and not _pipeline_stopped[0]:
            _pipeline_stopped[0] = True
            pipeline.stop()


if __name__ == "__main__":
    main()
