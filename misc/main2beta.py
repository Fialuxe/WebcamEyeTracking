"""
main2beta — CoGaze with beta features.

Extends the standard CoGaze pipeline with:
  - Head roll correction (de-rotate iris features by solvePnP roll angle)
  - TPS calibration (RBFInterpolator thin_plate_spline vs. RidgeCV)
  - tvec features (solvePnP X/Y/Z added to TPS feature vector)
  - ScreenProjector (geometric gaze estimation without calibration)

All beta features are ON by default.  Use flags to disable individual ones.

Run:
    python src/main2beta.py                    # all beta features ON
    python src/main2beta.py --no-roll-correction
    python src/main2beta.py --no-tps
    python src/main2beta.py --no-tvec-features
    python src/main2beta.py --screen-projector   # enable geometric projector
    python src/main2beta.py --mock               # mock gaze (no hardware)
    python src/main2beta.py --no-ir
    python src/main2beta.py --osc-port 9001
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

# Suppress protobuf legacy API warning from MediaPipe's internal imports
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
# Suppress absl/glog INFO and WARNING messages (TFLite XNNPACK, etc.)
os.environ.setdefault("GLOG_minloglevel", "2")

# Ensure src/ is on the path when run as "python src/main2beta.py"
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from main.__main__ import _setup_logging
from main import config
from main.gaze.base import MockGazeSource
from main.osc.sender import OSCSender
from main.orchestrator import Pipeline
from main.recording.csv_logger import CSVLogger
from main.session.session import Session
from main.gaze.webcam_source import FaceMeshBackend
from main.gaze.webcam_source_beta import WebcamGazeSourceBeta
from main.gaze.screen_projector import ScreenProjector


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CoGaze beta — roll corr + TPS + tvec + projector")
    # Standard flags (same as main)
    p.add_argument("--mock",        action="store_true", help="Use mock gaze (no hardware)")
    p.add_argument("--no-ir",       action="store_true", help="Disable IR tracker")
    p.add_argument("--no-webcam",   action="store_true", help="Disable webcam")
    p.add_argument("--osc-host",    default=config.OSC_HOST)
    p.add_argument("--osc-port",    type=int, default=config.OSC_PORT)
    p.add_argument("--output-dir",  default="data")
    # Beta feature flags (ON by default except projector)
    p.add_argument("--no-roll-correction",  action="store_true",
                   help="Disable solvePnP roll de-rotation of iris features")
    p.add_argument("--no-tps",              action="store_true",
                   help="Use Ridge regression calibration instead of TPS")
    p.add_argument("--no-tvec-features",    action="store_true",
                   help="Do not add solvePnP tvec to TPS feature vector")
    p.add_argument("--screen-projector",    action="store_true",
                   help="Enable geometric ScreenProjector as fallback when not calibrated")
    # ScreenProjector geometry (only used if --screen-projector)
    p.add_argument("--screen-distance-mm",  type=float, default=600.0,
                   help="Camera-to-screen distance in mm (default 600)")
    p.add_argument("--screen-width-mm",     type=float, default=530.0,
                   help="Physical screen width in mm (default 530)")
    p.add_argument("--screen-height-mm",    type=float, default=300.0,
                   help="Physical screen height in mm (default 300)")
    p.add_argument("--cam-offset-x-mm",     type=float, default=0.0,
                   help="Camera X offset from screen centre in mm (default 0)")
    p.add_argument("--cam-offset-y-mm",     type=float, default=-150.0,
                   help="Camera Y offset from screen centre in mm (default -150)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    _setup_logging(args.output_dir)

    osc = OSCSender(host=args.osc_host, port=args.osc_port)
    pipeline = None
    main_win = None
    _pipeline_stopped = [False]
    _session_gen = [0]

    # Resolve beta flags
    roll_correction   = not args.no_roll_correction
    use_tps           = not args.no_tps
    use_tvec_features = not args.no_tvec_features

    # Build ScreenProjector only when requested
    projector: ScreenProjector | None = None
    if args.screen_projector:
        projector = ScreenProjector(
            screen_distance_mm=args.screen_distance_mm,
            screen_width_mm=args.screen_width_mm,
            screen_height_mm=args.screen_height_mm,
            cam_offset_x_mm=args.cam_offset_x_mm,
            cam_offset_y_mm=args.cam_offset_y_mm,
        )

    def on_session_start(pid: str, condition: str) -> None:
        nonlocal pipeline, _pipeline_stopped

        session = Session(pid, condition)
        session.lock()

        ts = int(time.time())
        os.makedirs(args.output_dir, exist_ok=True)
        csv_path = os.path.join(args.output_dir, f"{pid}_{condition}_{ts}.csv")
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
                webcam_source = WebcamGazeSourceBeta(
                    condition=condition,
                    use_filter=(condition == "WebcamFiltered"),
                    backend=FaceMeshBackend(config.FACEMESH_BACKEND),
                    model_path=config.FACEMESH_MODEL_PATH,
                    roll_correction=roll_correction,
                    use_tps=use_tps,
                    use_tvec_features=use_tvec_features,
                    screen_projector=projector,
                )
                pipeline.add_source(webcam_source)

        # OSC status → UI
        osc.set_status_callback(
            lambda live: root.after(
                0, lambda l=live: main_win is not None and main_win.update_osc_status(l)
            )
        )

        # Calibration button (only for webcam conditions)
        if webcam_source is not None:
            def _do_calibration():
                from main.ui.calib_window import run_calibration
                def _on_calib_done(success: bool, err_x: float, err_y: float) -> None:
                    if main_win is not None:
                        main_win.calib_panel.show_result(success, err_x, err_y)
                    if success and pipeline is not None:
                        pipeline.mark_calibrated()
                run_calibration(root, webcam_source, on_done=_on_calib_done)
            main_win.calib_panel.set_callback(_do_calibration)

        # Camera preview
        if webcam_source is not None:
            main_win.start_preview(webcam_source)
            def _on_debug_toggle(*_):
                if main_win is not None:
                    webcam_source.set_debug_mode(main_win._debug_var.get())
            main_win._debug_var.trace_add("write", _on_debug_toggle)

        # Gaze polling (~30 fps)
        gen = _session_gen[0]

        def _poll_gaze():
            if _session_gen[0] != gen:
                return
            if pipeline is not None:
                sample = pipeline.get_latest_gaze()
                if sample is not None and main_win is not None:
                    main_win.update_gaze(sample.x, sample.y, sample.source)
            root.after(33, _poll_gaze)

        root.after(33, _poll_gaze)

        def _stop_once():
            if not _pipeline_stopped[0]:
                _pipeline_stopped[0] = True
                if pipeline is not None:
                    pipeline.stop()

        main_win.set_close_callback(_stop_once)
        pipeline.start()

    def _do_restart() -> None:
        nonlocal pipeline, main_win, _pipeline_stopped
        _session_gen[0] += 1
        if not _pipeline_stopped[0]:
            _pipeline_stopped[0] = True
            if pipeline is not None:
                pipeline.stop()
        pipeline = None
        main_win = None
        _pipeline_stopped = [False]
        root.withdraw()
        for w in root.winfo_children():
            w.destroy()
        SessionSetupDialog(root, on_start=_start)

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
