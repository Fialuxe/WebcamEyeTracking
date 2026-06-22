"""
Full-screen calibration window for CoGaze.

Shows a 16-point non-uniform dot sequence, collects head-local gaze samples via
WebcamGazeSource.get_local_gaze(), and fits Ridge regression calibration.

Visual design (UX-optimised):
  - Smooth ease-in-out travel between dots (800 ms) — fovea arrives relaxed
  - Converging ring collapses toward the centre during dwell — draws gaze inward
  - Arc progress ring sweeps 0→360° over DWELL_SECONDS — eliminates uncertainty
  - Completed dots leave small ghost markers + bottom progress strip

Public API
----------
run_calibration(parent, webcam_source, on_done, n_points=16)
    Open calibration window, run dot sequence, call on_done() when complete.
"""
from __future__ import annotations

import tkinter as tk
from typing import Callable
import time
import math

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DWELL_SECONDS: float = 1.5        # seconds of dwell per dot (collecting samples)
MAX_WAIT_SECONDS: float = 3.0     # seconds before skipping a dot with no face detected
SAMPLE_INTERVAL_MS: int = 50      # polling interval for get_local_gaze()
DOT_RADIUS: int = 12              # fixed centre dot radius (px)
RING_RADIUS_START: int = 38       # converging ring start radius (px)
TRAVEL_SECONDS: float = 0.8       # smooth travel duration between dots (s)
ARC_RADIUS: int = 24              # progress arc radius (outside the fixed dot)

MARGIN: float = 0.05

# 16-point non-uniform corner-heavy grid (normalised [0, 1] screen space).
TARGETS: list[tuple[float, float]] = [
    # True corners
    (MARGIN,        MARGIN       ),
    (1.0 - MARGIN,  MARGIN       ),
    (MARGIN,        1.0 - MARGIN ),
    (1.0 - MARGIN,  1.0 - MARGIN ),
    # Edge midpoints (top, bottom, left, right)
    (0.5,           MARGIN       ),
    (0.5,           1.0 - MARGIN ),
    (MARGIN,        0.5          ),
    (1.0 - MARGIN,  0.5          ),
    # Near-corner ring at one-third from each corner
    (0.25,          0.25         ),
    (0.75,          0.25         ),
    (0.25,          0.75         ),
    (0.75,          0.75         ),
    # Inner ring
    (0.25,          0.5          ),
    (0.75,          0.5          ),
    (0.5,           0.25         ),
    (0.5,           0.75         ),
]


# ---------------------------------------------------------------------------
# Pure helpers (importable without Tk)
# ---------------------------------------------------------------------------

def _average_samples(buffer: list[tuple[float, float]]) -> tuple[float, float]:
    """Return the component-wise mean of a non-empty list of (x, y) samples."""
    n = len(buffer)
    return sum(s[0] for s in buffer) / n, sum(s[1] for s in buffer) / n


def _ease(t: float) -> float:
    """Cubic ease-in-out. t ∈ [0, 1]."""
    return t * t * (3.0 - 2.0 * t)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_calibration(
    parent: tk.Tk,
    webcam_source,
    on_done: Callable[[bool, float, float], None],
    n_points: int = 16,
) -> "_CalibrationWindow":
    """
    Open a full-screen Toplevel calibration window.
    Runs the calibration dot sequence, calls on_done() when complete,
    then destroys the window.
    Does NOT call pipeline.mark_calibrated() — the caller handles that.
    Returns the window instance so the caller can cancel via _on_esc().
    """
    return _CalibrationWindow(parent, webcam_source, on_done, n_points)


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------

class _CalibrationWindow:
    """
    Full-screen calibration window (internal class, not part of public API).

    Per-dot state machine:
        TRAVEL  → dot glides from previous position (ease-in-out, TRAVEL_SECONDS)
        DWELL   → converging ring + arc progress; gaze samples collected
        REWARD  → dot shrinks away, ghost marker placed, progress strip updated
        → next dot

    ESC: abort, call on_done(False, 0.0, 0.0), destroy.

    Accuracy notes:
        - Centre dot is stationary throughout DWELL (ring/arc never touch it).
        - Sampling (_collect_sample) starts only AFTER travel is complete.
        - Ring and arc animations are driven by separate after() chains;
          they use itemconfig/coords so there is no flicker from delete("all").
    """

    _PHASE_TRAVEL = "travel"
    _PHASE_DWELL  = "dwell"
    _PHASE_REWARD = "reward"

    def __init__(
        self,
        parent: tk.Tk,
        webcam_source,
        on_done: Callable[[bool, float, float], None],
        n_points: int,
    ) -> None:
        self._source     = webcam_source
        self._on_done    = on_done
        self._targets    = TARGETS[:n_points]
        self._n          = len(self._targets)
        self._idx        = 0
        self._phase      = self._PHASE_TRAVEL
        self._buffer:    list[tuple[float, float]] = []
        self._last_seen: tuple[float, float] | None = None
        self._point_start: float = 0.0
        self._travel_start: float = 0.0
        self._after_ids: list[str] = []
        self._done = False

        # Build window
        self._win = tk.Toplevel(parent)
        self._win.attributes("-fullscreen", True, "-topmost", True)
        self._win.configure(bg="black")
        self._win.bind("<Escape>", self._on_esc)
        self._win.protocol("WM_DELETE_WINDOW", self._on_esc)

        sw = self._win.winfo_screenwidth()
        sh = self._win.winfo_screenheight()
        self._sw, self._sh = sw, sh

        self._canvas = tk.Canvas(
            self._win, width=sw, height=sh,
            bg="black", highlightthickness=0,
        )
        self._canvas.pack(fill="both", expand=True)

        # Counter label (top-left)
        self._counter_var = tk.StringVar(value="")
        tk.Label(
            self._win, textvariable=self._counter_var,
            fg="#888888", bg="black", font=("Arial", 13),
        ).place(x=14, y=12)

        # ── Create persistent canvas items ──────────────────────────────────
        # Converging ring (visible during dwell)
        r0 = RING_RADIUS_START
        self._ring_id = self._canvas.create_oval(
            -r0, -r0, r0, r0,
            outline="#555555", width=2, fill="",
        )
        # Progress arc (visible during dwell, sweeps CW from 90°)
        ra = ARC_RADIUS
        self._arc_id = self._canvas.create_arc(
            -ra, -ra, ra, ra,
            start=90, extent=0,
            outline="#aaaaaa", width=2, style="arc",
        )
        # Centre dot (visible always)
        rd = DOT_RADIUS
        self._dot_id = self._canvas.create_oval(
            -rd, -rd, rd, rd,
            fill="white", outline="white",
        )
        # Hide them all off-screen until first point starts
        self._canvas.move(self._ring_id, -200, -200)
        self._canvas.move(self._arc_id,  -200, -200)
        self._canvas.move(self._dot_id,  -200, -200)

        # Progress strip (n_points dots along the bottom)
        strip_y = sh - 20
        strip_total_w = self._n * 14
        strip_x0 = (sw - strip_total_w) // 2
        self._prog_ids: list[int] = []
        for i in range(self._n):
            cx = strip_x0 + i * 14 + 7
            pid = self._canvas.create_oval(
                cx - 4, strip_y - 4, cx + 4, strip_y + 4,
                fill="#333333", outline="",
            )
            self._prog_ids.append(pid)

        # Track current dot pixel position (starts off-screen, first point handled below)
        self._cx: float = sw / 2
        self._cy: float = sh / 2

        # Place everything at centre and start
        self._move_items_to(int(self._cx), int(self._cy))
        self._win.after(120, self._show_instruction)

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _target_px(self, idx: int) -> tuple[int, int]:
        tx, ty = self._targets[idx]
        return int(tx * self._sw), int(ty * self._sh)

    def _move_items_to(self, px: int, py: int) -> None:
        """Teleport all persistent items so their centre is at (px, py)."""
        for item_id in (self._ring_id, self._arc_id, self._dot_id):
            x1, y1, x2, y2 = self._canvas.coords(item_id)
            half_w = (x2 - x1) / 2
            half_h = (y2 - y1) / 2
            self._canvas.coords(item_id,
                                px - half_w, py - half_h,
                                px + half_w, py + half_h)

    def _recentre_ring(self, px: int, py: int, r: int) -> None:
        self._canvas.coords(self._ring_id, px - r, py - r, px + r, py + r)

    def _recentre_arc(self, px: int, py: int) -> None:
        ra = ARC_RADIUS
        self._canvas.coords(self._arc_id, px - ra, py - ra, px + ra, py + ra)

    def _recentre_dot(self, px: int, py: int) -> None:
        rd = DOT_RADIUS
        self._canvas.coords(self._dot_id, px - rd, py - rd, px + rd, py + rd)

    # ------------------------------------------------------------------
    # Instruction overlay (shown before first dot)
    # ------------------------------------------------------------------

    def _show_instruction(self) -> None:
        """Show a 2-second instruction overlay before the first dot."""
        if self._done:
            return
        cx, cy = self._sw // 2, self._sh // 2
        self._canvas.create_text(
            cx, cy - 50,
            text="Follow the dot with your eyes.",
            fill="white", font=("Arial", 28), tags="instr",
        )
        self._canvas.create_text(
            cx, cy + 10,
            text="Keep your head still.",
            fill="#aaaaaa", font=("Arial", 20), tags="instr",
        )
        self._canvas.create_text(
            cx, cy + 70,
            text=f"Starting in 2 seconds…  ({self._n} points)",
            fill="#666666", font=("Arial", 16), tags="instr",
        )
        aid = self._win.after(2000, self._clear_instruction_and_start)
        self._after_ids.append(aid)

    def _clear_instruction_and_start(self) -> None:
        self._canvas.delete("instr")
        self._begin_travel()

    # ------------------------------------------------------------------
    # Phase: TRAVEL — smooth ease-in-out glide to next dot
    # ------------------------------------------------------------------

    def _begin_travel(self) -> None:
        if self._done:
            return
        if self._idx >= self._n:
            self._finish()
            return

        total = self._n
        self._counter_var.set(f"{self._idx + 1} / {total}")

        self._canvas.itemconfig(self._ring_id, outline="#333333")
        self._canvas.itemconfig(self._arc_id, outline="#333333", extent=0)
        self._canvas.itemconfig(self._dot_id, fill="#aaaaaa", outline="#aaaaaa")

        self._from_cx = self._cx
        self._from_cy = self._cy
        self._to_px, self._to_py = self._target_px(self._idx)
        self._travel_start = time.monotonic()
        self._phase = self._PHASE_TRAVEL
        self._travel_frame()

    def _travel_frame(self) -> None:
        if self._done:
            return
        elapsed = time.monotonic() - self._travel_start
        frac = min(elapsed / TRAVEL_SECONDS, 1.0)
        f = _ease(frac)

        cx = self._from_cx + (self._to_px - self._from_cx) * f
        cy = self._from_cy + (self._to_py - self._from_cy) * f
        self._cx, self._cy = cx, cy

        px, py = int(cx), int(cy)
        self._recentre_dot(px, py)
        self._recentre_ring(px, py, RING_RADIUS_START)
        self._recentre_arc(px, py)

        if frac < 1.0:
            aid = self._win.after(16, self._travel_frame)
            self._after_ids.append(aid)
        else:
            # Snap to exact pixel
            self._cx, self._cy = float(self._to_px), float(self._to_py)
            self._begin_dwell()

    # ------------------------------------------------------------------
    # Phase: DWELL — converging ring + arc; sample collection
    # ------------------------------------------------------------------

    def _begin_dwell(self) -> None:
        if self._done:
            return
        self._phase = self._PHASE_DWELL
        px, py = int(self._cx), int(self._cy)

        # Activate visuals
        self._canvas.itemconfig(self._ring_id, outline="#555555")
        self._canvas.itemconfig(self._arc_id, outline="#aaaaaa", extent=0)
        self._canvas.itemconfig(self._dot_id, fill="white", outline="white")

        # Reset ring to full size
        self._recentre_ring(px, py, RING_RADIUS_START)
        self._recentre_arc(px, py)
        self._recentre_dot(px, py)

        self._buffer = []
        self._last_seen = None
        self._point_start = time.monotonic()

        self._dwell_anim_frame()
        aid = self._win.after(SAMPLE_INTERVAL_MS, self._collect_sample)
        self._after_ids.append(aid)

    def _dwell_anim_frame(self) -> None:
        """Animate converging ring + arc. Runs independently of sampling."""
        if self._done or self._phase != self._PHASE_DWELL:
            return
        elapsed = time.monotonic() - self._point_start
        frac = min(elapsed / DWELL_SECONDS, 1.0)

        px, py = int(self._cx), int(self._cy)

        # Converging ring: radius shrinks from RING_RADIUS_START → 0
        r = int(RING_RADIUS_START * (1.0 - frac))
        self._recentre_ring(px, py, max(r, 1))

        # Progress arc: sweeps clockwise 0 → 360
        self._canvas.itemconfig(self._arc_id, extent=-(frac * 360))

        if frac < 1.0:
            aid = self._win.after(20, self._dwell_anim_frame)
            self._after_ids.append(aid)

    def _collect_sample(self) -> None:
        """Called every SAMPLE_INTERVAL_MS; collects gaze and triggers reward when ready."""
        if self._done or self._phase != self._PHASE_DWELL:
            return

        sample = self._source.get_local_gaze()
        if sample is not None and sample != self._last_seen:
            self._buffer.append(sample)
            self._last_seen = sample
            # Restore dot colour when face reappears
            self._canvas.itemconfig(self._dot_id, fill="white", outline="white")
        else:
            # No face: tint dot orange so participant knows to recentre
            self._canvas.itemconfig(self._dot_id, fill="#ff8c00", outline="#ff8c00")

        elapsed = time.monotonic() - self._point_start

        if self._buffer and elapsed >= DWELL_SECONDS:
            local_x, local_y = _average_samples(self._buffer)
            tx, ty = self._targets[self._idx]
            self._source.calibration.add_point(local_x, local_y, tx, ty)
            self._begin_reward()
        elif elapsed >= MAX_WAIT_SECONDS:
            if self._buffer:
                local_x, local_y = _average_samples(self._buffer)
                tx, ty = self._targets[self._idx]
                self._source.calibration.add_point(local_x, local_y, tx, ty)
            self._show_skip_warning()
            self._begin_reward()
        else:
            aid = self._win.after(SAMPLE_INTERVAL_MS, self._collect_sample)
            self._after_ids.append(aid)

    # ------------------------------------------------------------------
    # Skip warning
    # ------------------------------------------------------------------

    def _show_skip_warning(self) -> None:
        """Flash a warning when a dot is skipped due to no face detected."""
        warn_id = self._canvas.create_text(
            self._sw // 2, self._sh - 55,
            text=f"⚠ Point {self._idx + 1}: no face detected — skipped",
            fill="#ff6633", font=("Arial", 14), tags="skipwarn",
        )
        aid = self._win.after(1200, lambda: self._canvas.delete(warn_id))
        self._after_ids.append(aid)

    # ------------------------------------------------------------------
    # Phase: REWARD — dot shrinks, ghost marker placed, progress strip lit
    # ------------------------------------------------------------------

    def _begin_reward(self) -> None:
        if self._done:
            return
        self._phase = self._PHASE_REWARD
        px, py = int(self._cx), int(self._cy)

        # Hide ring and arc immediately
        self._canvas.itemconfig(self._ring_id, outline="")
        self._canvas.itemconfig(self._arc_id, outline="")

        # Place ghost marker at completed position
        self._canvas.create_oval(
            px - 4, py - 4, px + 4, py + 4,
            fill="", outline="#3a3a3a", width=1,
        )

        # Light up progress strip
        if self._idx < len(self._prog_ids):
            self._canvas.itemconfig(self._prog_ids[self._idx], fill="#888888")

        # Shrink dot animation
        self._shrink_dot(px, py, DOT_RADIUS)

    def _shrink_dot(self, px: int, py: int, r: int) -> None:
        if self._done:
            return
        if r <= 0:
            self._canvas.itemconfig(self._dot_id, fill="", outline="")
            self._idx += 1
            aid = self._win.after(80, self._begin_travel)
            self._after_ids.append(aid)
            return
        rd = max(r, 1)
        self._canvas.coords(self._dot_id, px - rd, py - rd, px + rd, py + rd)
        next_r = r - 2
        aid = self._win.after(18, lambda: self._shrink_dot(px, py, next_r))
        self._after_ids.append(aid)

    # ------------------------------------------------------------------
    # Finish / abort
    # ------------------------------------------------------------------

    def _finish(self) -> None:
        if self._done:
            return
        self._done = True

        # Flash all progress dots white
        for pid in self._prog_ids:
            self._canvas.itemconfig(pid, fill="white")

        aid = self._win.after(300, self._do_fit)
        self._after_ids.append(aid)

    def _do_fit(self) -> None:
        result = self._source.calibration.fit()
        self._call_done(result.success, result.validation_error_x, result.validation_error_y)

    def _call_done(self, success: bool, err_x: float, err_y: float) -> None:
        try:
            self._on_done(success, err_x, err_y)
        finally:
            self._win.destroy()

    def _on_esc(self, event=None) -> None:
        # Always cancel pending afters — when called from _do_restart() after
        # _finish() already set _done=True, this cancels the pending _do_fit so
        # a spurious /calibration/result is not sent for the dead session.
        for aid in self._after_ids:
            try:
                self._win.after_cancel(aid)
            except Exception:
                pass
        self._after_ids.clear()
        if self._done:
            return
        self._done = True
        self._call_done(False, 0.0, 0.0)
