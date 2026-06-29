"""
CoGaze UI: session setup dialog + main window (Challenges #7 and #8).

SessionSetupDialog  — collects participant ID + condition before recording
MainWindow          — camera preview, calibration controls, OSC status bar
launch()            — entry point; call with a callback for when session starts
"""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable

from ..session.session import VALID_CONDITIONS
from .. import config as _config


class SessionSetupDialog(tk.Toplevel):
    """Modal dialog: participant ID + condition selection (Challenge #8)."""

    _CONDITIONS = sorted(VALID_CONDITIONS)

    def __init__(self, parent: tk.Tk, on_start: Callable[[str, str], None]) -> None:
        super().__init__(parent)
        self.title("CoGaze — Session Setup")
        self.resizable(False, False)
        self.grab_set()
        self._on_start = on_start
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build()

    def _on_close(self) -> None:
        self.master.destroy()

    def _build(self) -> None:
        frame = ttk.Frame(self, padding=20)
        frame.grid(sticky="nsew")

        ttk.Label(frame, text="Participant ID:").grid(row=0, column=0, sticky="w", pady=4)
        self._pid_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self._pid_var, width=22).grid(
            row=0, column=1, sticky="ew", padx=8
        )

        ttk.Label(frame, text="Condition:").grid(row=1, column=0, sticky="nw", pady=4)
        self._cond_var = tk.StringVar(value=self._CONDITIONS[0])
        for i, cond in enumerate(self._CONDITIONS):
            ttk.Radiobutton(
                frame, text=cond, value=cond, variable=self._cond_var
            ).grid(row=1 + i, column=1, sticky="w")

        ttk.Button(frame, text="Start Recording", command=self._submit).grid(
            row=1 + len(self._CONDITIONS), column=0, columnspan=2, pady=14
        )

    def _submit(self) -> None:
        pid = self._pid_var.get().strip()
        if not pid:
            messagebox.showerror("Input Error", "Participant ID cannot be empty.", parent=self)
            return
        cond = self._cond_var.get()
        self._on_start(pid, cond)
        self.destroy()


class OSCStatusBar(ttk.Frame):
    """Persistent live/dead OSC indicator (Challenge #11)."""

    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent)
        self._dot = tk.Label(self, text="●", fg="gray", font=("Arial", 18))
        self._dot.pack(side="left", padx=4)
        self._label = ttk.Label(self, text="OSC: --")
        self._label.pack(side="left")

    def set_status(self, live: bool) -> None:
        if live:
            self._dot.configure(fg="green")
            self._label.configure(text="OSC: LIVE")
        else:
            self._dot.configure(fg="red")
            self._label.configure(text="OSC: DEAD")


class CalibrationPanel(ttk.LabelFrame):
    """
    Calibration controls (Challenge #7):
    - Button triggers calibration (experimenter-initiated, not auto-timed).
    - Shows pass/fail verdict, not raw numbers, so experimenter can decide fast.
    """

    def __init__(self, parent: tk.Widget, on_calibrate: Callable[[], None]) -> None:
        super().__init__(parent, text="Calibration", padding=10)
        self._on_calibrate = on_calibrate
        self._verdict_var = tk.StringVar(value="Not calibrated")

        self._btn = ttk.Button(self, text="Start Calibration", command=lambda: self._on_calibrate())
        self._btn.pack(fill="x", pady=4)
        self._verdict_label = ttk.Label(self, textvariable=self._verdict_var, font=("Arial", 11, "bold"))
        self._verdict_label.pack(pady=4)

    def show_result(self, passed: bool, err_x: float = 0.0, err_y: float = 0.0) -> None:
        thr_x = _config.CALIB_THRESHOLD_X
        thr_y = _config.CALIB_THRESHOLD_Y
        within = err_x <= thr_x and err_y <= thr_y
        if passed and within:
            self._verdict_var.set(
                f"PASS  x={err_x:.3f}  y={err_y:.3f}  [thr {thr_x:.3f}]"
            )
            self._verdict_label.configure(foreground="green")
            self._btn.configure(text="Start Calibration")
        elif passed:
            self._verdict_var.set(
                f"MARGINAL  x={err_x:.3f}  y={err_y:.3f}  [thr {thr_x:.3f}/{thr_y:.3f}]"
            )
            self._verdict_label.configure(foreground="orange")
            self._btn.configure(text="Retry Calibration")
        else:
            self._verdict_var.set("FAIL — please recalibrate")
            self._verdict_label.configure(foreground="red")
            self._btn.configure(text="Retry Calibration")

    def set_callback(self, cb: Callable[[], None]) -> None:
        self._on_calibrate = cb

    def set_not_applicable(self, reason: str = "Calibration not required") -> None:
        """Disable the calibration button (e.g. for IR condition)."""
        self._btn.configure(state="disabled")
        self._verdict_var.set(reason)
        self._verdict_label.configure(foreground="gray")


class FaceGuideOverlay:
    """
    Singapore-style face positioning guide drawn as canvas overlay items.

    Shows:
      • Dashed reference oval at the ideal face size/position (colour = distance status)
      • Blue outline oval tracking the actual detected face
      • One-line Japanese guidance text below the oval

    Call enable_face_guide() on MainWindow, then update_face_guide(metrics) each poll tick.
    """

    _CANVAS_W = 480
    _CANVAS_H = 360
    _IOD_TO_FACE_W = 0.45   # IOD / face_width anatomical ratio
    _FACE_ASPECT   = 1.35   # face_height / face_width

    def __init__(self, canvas: tk.Canvas) -> None:
        self._canvas = canvas
        cw, ch = self._CANVAS_W, self._CANVAS_H
        cx, cy = cw // 2, ch // 2

        # Reference oval: dashed, centred, sized to target IOD
        iod_c = _config.IOD_TARGET_NORM * cw
        fw    = iod_c / self._IOD_TO_FACE_W
        fh    = fw * self._FACE_ASPECT
        self._fw, self._fh = fw, fh
        self._ref_oval = canvas.create_oval(
            cx - fw / 2, cy - fh / 2, cx + fw / 2, cy + fh / 2,
            outline="#555555", width=2, dash=(10, 5), fill="",
            tags="face_guide",
        )
        # "target" label just above the oval
        canvas.create_text(
            cx, cy - fh / 2 - 11,
            text="ここに顔を合わせてください",
            fill="#555555", font=("Arial", 10),
            tags="face_guide",
        )
        # Live face oval — tracks actual face position each frame
        self._live_oval = canvas.create_oval(
            -200, -200, -180, -180,
            outline="#4488ff", width=1, fill="",
            tags="face_guide",
        )
        # Status text below the reference oval
        self._text_id = canvas.create_text(
            cx, cy + fh / 2 + 16,
            text="", fill="#888888", font=("Arial", 11, "bold"),
            tags="face_guide",
        )

    def update(self, metrics) -> None:
        cw, ch = self._CANVAS_W, self._CANVAS_H

        if metrics is None:
            self._canvas.itemconfig(self._ref_oval, outline="#444444")
            self._canvas.itemconfig(self._text_id, text="顔が見えません", fill="#666666")
            self._canvas.coords(self._live_oval, -200, -200, -180, -180)
            return

        iod_n  = metrics.iod_norm
        target = _config.IOD_TARGET_NORM
        tol    = _config.IOD_TOLERANCE_NORM
        diff   = iod_n - target

        if abs(diff) <= tol:
            color = "#00cc44"
            if abs(metrics.face_cx - 0.5) > 0.12:
                msg = ("← 左に動いてください" if metrics.face_cx > 0.5
                       else "右に動いてください →")
            elif abs(metrics.face_cy - 0.5) > 0.10:
                msg = ("↑ 上に動いてください" if metrics.face_cy > 0.5
                       else "↓ 下に動いてください")
            else:
                msg = "✓ 適切な位置"
        elif diff > tol:
            # IOD too large → participant too close
            frac  = min((diff - tol) / (tol * 2), 1.0)
            color = "#ff8c00" if frac < 0.5 else "#ff3333"
            msg   = "もっと離れてください"
        else:
            # IOD too small → participant too far
            frac  = min((tol - diff) / (tol * 2), 1.0)
            color = "#ff8c00" if frac < 0.5 else "#ff3333"
            msg   = "もっと近づいてください"

        self._canvas.itemconfig(self._ref_oval, outline=color)
        self._canvas.itemconfig(self._text_id, text=msg, fill=color)

        # Live oval: actual face size and position
        iod_c = iod_n * cw
        fw    = iod_c / self._IOD_TO_FACE_W
        fh    = fw * self._FACE_ASPECT
        lx    = metrics.face_cx * cw
        ly    = metrics.face_cy * ch
        self._canvas.coords(
            self._live_oval,
            lx - fw / 2, ly - fh / 2,
            lx + fw / 2, ly + fh / 2,
        )


class MainWindow:
    """Primary window shown during data collection."""

    def __init__(self, root: tk.Tk, participant_id: str, condition: str) -> None:
        self._root = root
        root.title(f"CoGaze  |  {condition}  [{participant_id}]")
        root.geometry("820x540")
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._on_close_cb: Callable[[], None] | None = None
        self._on_restart_cb: Callable[[], None] | None = None
        self._build()

    def set_close_callback(self, cb: Callable[[], None]) -> None:
        self._on_close_cb = cb

    def set_restart_callback(self, cb: Callable[[], None]) -> None:
        self._on_restart_cb = cb

    def _on_close(self) -> None:
        if self._on_close_cb:
            self._on_close_cb()
        self._root.destroy()

    def _on_restart(self) -> None:
        if not messagebox.askyesno(
            "Change Session",
            "Stop recording and change session?\nThis cannot be undone.",
            parent=self._root,
        ):
            return
        if self._on_restart_cb:
            self._on_restart_cb()

    def _build(self) -> None:
        # Top bar: OSC status + lock indicator
        top = ttk.Frame(self._root, padding=(8, 4))
        top.pack(fill="x")
        self.osc_bar = OSCStatusBar(top)
        self.osc_bar.pack(side="left")
        self._lock_label = ttk.Label(top, text="[ LOCKED ]", foreground="green")
        self._lock_label.pack(side="right", padx=8)

        # Body: preview canvas + right controls
        body = ttk.Frame(self._root)
        body.pack(fill="both", expand=True, padx=8, pady=4)

        self.canvas = tk.Canvas(body, width=480, height=360, background="#111")
        self.canvas.pack(side="left")

        right = ttk.Frame(body, padding=(12, 0))
        right.pack(side="left", fill="y")

        self.calib_panel = CalibrationPanel(right, on_calibrate=lambda: None)
        self.calib_panel.pack(fill="x", pady=6)

        ttk.Separator(right).pack(fill="x", pady=4)

        ttk.Button(right, text="Lock Condition", command=self._lock_condition).pack(
            fill="x", pady=2
        )
        ttk.Button(right, text="Change Session", command=self._on_restart).pack(
            fill="x", pady=2
        )

        self._preview_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(right, text="Show Preview", variable=self._preview_var,
                        command=self._on_preview_toggle).pack(anchor="w", pady=2)

        self._debug_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(right, text="Debug Mode", variable=self._debug_var).pack(
            anchor="w", pady=2
        )

        self._gaze_var = tk.StringVar(value="gaze: --")
        ttk.Label(right, textvariable=self._gaze_var, foreground="gray").pack(
            anchor="w", pady=4
        )

        self._trial_var = tk.StringVar(value="trial: --")
        ttk.Label(right, textvariable=self._trial_var, foreground="gray").pack(
            anchor="w", pady=2
        )

        self._certainty_var = tk.StringVar(value="certainty: --")
        ttk.Label(right, textvariable=self._certainty_var, foreground="gray").pack(
            anchor="w", pady=2
        )

        self._pace_var = tk.StringVar(value="")
        ttk.Label(right, textvariable=self._pace_var, foreground="blue",
                  font=("Arial", 10)).pack(anchor="w", pady=2)

    def _lock_condition(self) -> None:
        self._lock_label.configure(text="[ LOCKED ]", foreground="green")

    def _on_preview_toggle(self) -> None:
        active = self._preview_var.get()
        # Sync the plain-bool flag read by the camera thread in start_preview().
        if hasattr(self, "_preview_active"):
            self._preview_active = active
        if not active:
            self.canvas.delete("preview")

    def update_trial_id(self, tid: str) -> None:
        """Display current trial ID; shows '--' when tid is empty."""
        self._trial_var.set(f"trial: {tid or '--'}")

    def update_certainty(self, mesh_c: float, eye_c: float) -> None:
        """Display real-time mesh and eye certainty values."""
        self._certainty_var.set(f"mesh={mesh_c:.2f}  eye={eye_c:.2f}")

    def update_gaze(self, x: float, y: float, source: str) -> None:
        self._gaze_var.set(f"{source}: ({x:.3f}, {y:.3f})")
        if self._debug_var.get():
            px = int(x * 480)
            py = int(y * 360)
            self.canvas.delete("gaze_dot")
            self.canvas.create_oval(px - 6, py - 6, px + 6, py + 6, fill="red", tags="gaze_dot")

    def update_osc_status(self, live: bool) -> None:
        self.osc_bar.set_status(live)

    def update_pace_status(self, update_count: int, accuracy: float, is_ready: bool) -> None:
        """PACE キャリブレーション状態を UI に反映する。beta2 専用。"""
        if not hasattr(self, '_pace_var'):
            return  # 非 beta2 環境では無視
        acc_str = f"{accuracy:.3f}" if accuracy >= 0 else "蓄積中"
        if is_ready:
            self._pace_var.set(f"PACE: READY  MAE={acc_str}")
        else:
            self._pace_var.set(f"PACE: 暖機中 ({update_count}クリック済み)  MAE={acc_str}")

    def enable_face_guide(self) -> None:
        """Activate the face positioning overlay on the camera canvas."""
        self._face_guide: FaceGuideOverlay | None = FaceGuideOverlay(self.canvas)

    def update_face_guide(self, metrics) -> None:
        guide = getattr(self, "_face_guide", None)
        if guide is None:
            return
        try:
            guide.update(metrics)
            self.canvas.tag_raise("face_guide")  # keep above camera image
        except tk.TclError:
            self._face_guide = None

    def start_preview(self, webcam_source) -> None:
        """
        Register a frame callback on webcam_source and start the canvas preview loop.
        Converts BGR→RGB, resizes to canvas size (480×360), displays via ImageTk.
        Requires Pillow (PIL). Fails silently if PIL is not available.
        """
        try:
            from PIL import Image
        except ImportError:
            self.canvas.create_text(
                240, 180, text="Preview unavailable\npip install Pillow",
                fill="red", font=("Arial", 14), justify="center",
            )
            return

        self._preview_img_ref = None  # keep reference to prevent GC
        # Plain Python bool: safe to read from camera thread (GIL-atomic).
        # _on_preview_toggle keeps it in sync with _preview_var on the Tk thread.
        self._preview_active: bool = True

        def _on_frame(bgr_frame):
            if not self._preview_active:  # read plain bool, not BooleanVar (thread-safe)
                return
            import cv2
            rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb).resize((480, 360))
            # Pass PIL Image (not PhotoImage) — PhotoImage must be created on main thread
            self._root.after(0, lambda i=img: self._update_canvas(i))

        webcam_source.set_frame_callback(_on_frame)

    def _update_canvas(self, img) -> None:
        try:
            from PIL import ImageTk
            photo = ImageTk.PhotoImage(img)
            self._preview_img_ref = photo  # prevent GC
            self.canvas.delete("preview")
            self.canvas.create_image(0, 0, anchor="nw", image=photo, tags="preview")
            self.canvas.tag_lower("preview")  # keep camera image below guide overlay
        except tk.TclError:
            pass  # canvas destroyed during session restart


