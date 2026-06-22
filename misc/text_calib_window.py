"""
Text reading calibration window for CoGaze.

Displays Markdown-formatted text and asks the user to read it naturally.
Detects left-to-right horizontal gaze sweeps and generates calibration
samples by mapping each sweep's gaze feature values to the corresponding
text line's normalised screen Y coordinate.

Line-to-sweep assignment uses reading order (the k-th detected sweep maps
to the k-th text line in top-to-bottom order), because pre-calibration
there is no mapping from feature space to screen space — matching
local_y values directly to screen-normalised line positions would always
snap to the topmost line.

Public API
----------
run_text_calibration(parent, webcam_source, on_done, md_text, ...)
    Open calibration window, collect samples, call on_done(success, n_samples).
"""
from __future__ import annotations

import logging
import time
import tkinter as tk
from tkinter import ttk
from typing import Callable

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum wall-clock duration (seconds) of a sustained gaze run to be
# considered a valid L→R (or R→L) sweep rather than a saccade or blink.
_SWEEP_MIN_DURATION_S: float = 0.30

# Hysteresis margin in local_x feature units.  The sweep continues as long as
# local_x stays within this margin of its running peak (not retreating past
# peak ± margin).  Suppresses per-sample jitter that would otherwise cause
# constant false reversals at 100 ms polling without One-Euro filtering.
# Tune empirically: ~0.02–0.03 works for iris-relative feature values (±0.1–0.2 range).
_SWEEP_HYSTERESIS: float = 0.025

# Sample polling interval feeds both gaze collection AND sweep detection.
# Smaller = more samples per line but more CPU; 100 ms matches the default.
_DEFAULT_SAMPLE_INTERVAL_MS: int = 100

# Fraction of the canvas width used for text (centred).
_TEXT_WIDTH_FRACTION: float = 0.60

# Line-height multiplier relative to point size.
_LINE_HEIGHT_FACTOR: float = 1.4

# Font sizes (pt) per element type.
_FONT_H1 = ("Arial", 18, "bold")
_FONT_H2 = ("Arial", 14, "bold")
_FONT_BULLET = ("Arial", 12)
_FONT_PARA = ("Arial", 12)

# Vertical margins (pixels) added above/below block types.
_MARGIN_H1 = (16, 8)   # (above, below)
_MARGIN_H2 = (12, 6)
_MARGIN_BLANK = 8      # paragraph separator height

# Status bar height (pixels) — reserved at the top.
_STATUS_HEIGHT = 36

# Bottom bar height (pixels) — reserved for buttons.
_BOTTOM_HEIGHT = 48

# ---------------------------------------------------------------------------
# Default calibration text (Markdown subset)
# ---------------------------------------------------------------------------

DEFAULT_CALIB_TEXT: str = """
# 読んでください

このテキストを最初の行から最後の行まで自然なペースで読んでください。
特別な動作は必要ありません。ただし目で追うように読んでください。

目のトラッキングシステムはあなたの視線の動きから自動的に調整を行います。
読み終わったら画面の下部にある「完了」ボタンをクリックしてください。

## 注意

- 各行を左端から右端まで、目でなぞりながら読んでください（左→右の視線移動が重要です）
- テキストを指で追わないでください
- 頭を動かさず、目だけで読んでください
- 自然なペースで読んでください（急ぐ必要はありません）
"""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_text_calibration(
    parent: tk.Tk,
    webcam_source,          # WebcamGazeSource — has .get_local_gaze() → tuple|None
                            # and .calibration: CalibrationManager
    on_done: Callable[[bool, int], None],  # (success, n_samples_collected)
    md_text: str = DEFAULT_CALIB_TEXT,
    sample_interval_ms: int = _DEFAULT_SAMPLE_INTERVAL_MS,
    min_samples: int = 20,
) -> "_TextCalibWindow":
    """
    Open a full-screen text reading calibration window.

    Renders md_text as formatted text (supports #/## headings, bullet lists,
    blank-line paragraph separators).  Collects gaze samples from horizontal
    scan detection and maps them to text line Y positions.

    Calls on_done(success, n_samples) when the user clicks 完了 or presses ESC.
    Returns the window instance so the caller can call force_close() for a
    graceful abort without the confirmation messagebox.
    """
    return _TextCalibWindow(parent, webcam_source, on_done, md_text, sample_interval_ms, min_samples)


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------

class _TextCalibWindow:
    """
    Full-screen text calibration window.

    Sweep detection state machine
    ─────────────────────────────
    Every sample_interval_ms ms we call get_local_gaze() and append the
    result to _gaze_history.  _detect_scan() runs after each append and
    looks for a "sustained monotonic run" in local_x:

      • Track the *dominant direction* of the current run (positive = L→R
        when viewed from subject perspective; sign may be negative depending
        on camera flip — we tolerate both).
      • When the run has lasted ≥ _SWEEP_MIN_DURATION_S AND a directional
        reversal is detected, we close the sweep.
      • All gaze samples from that sweep are added as calibration points
        targeting (screen_x=0.5, screen_y=line_y) where line_y is the
        normalised Y of the next un-consumed text line.
      • Multiple points per sweep satisfy min_samples without requiring an
        impractically large number of text lines.
    """

    def __init__(
        self,
        parent: tk.Tk,
        webcam_source,
        on_done: Callable[[bool, int], None],
        md_text: str,
        sample_interval_ms: int,
        min_samples: int,
    ) -> None:
        self._source = webcam_source
        self._on_done = on_done
        self._sample_interval_ms = sample_interval_ms
        self._min_samples = min_samples
        self._samples_collected = 0
        self._done = False

        # Face-not-detected state (initialized here to avoid getattr/hasattr)
        self._no_face_count: int = 0
        self._was_no_face: bool = False

        # Gaze history for sweep detection: list of (local_x, local_y, wall_time)
        self._gaze_history: list[tuple[float, float, float]] = []

        # Text line normalised screen Y-values (populated by _render_text).
        # Each entry corresponds to one renderable line in reading order.
        self._text_line_ys: list[float] = []

        # Index of the next un-consumed text line in reading order.
        self._next_line_idx: int = 0

        # Sweep detector state
        self._sweep_start_idx: int = 0   # index into _gaze_history
        self._sweep_dir: int | None = None  # +1 or -1 (dominant direction)
        self._sweep_start_time: float = 0.0
        # Running extremum of local_x in the current sweep (max for +1 dir, min for -1).
        # Reversal is declared only when local_x retreats past extremum by _SWEEP_HYSTERESIS.
        self._sweep_peak: float = 0.0

        # Pending after() ID
        self._after_id: str | None = None

        # Build window
        self._win = tk.Toplevel(parent)
        self._win.attributes("-fullscreen", True, "-topmost", True)
        self._win.configure(bg="white")
        self._win.bind("<Escape>", self._on_esc)
        self._win.protocol("WM_DELETE_WINDOW", self._on_esc)

        sw = self._win.winfo_screenwidth()
        sh = self._win.winfo_screenheight()
        self._sw = sw
        self._sh = sh

        # ── Status bar (top) ────────────────────────────────────────────────
        self._status_var = tk.StringVar(value="キャリブレーション中...  サンプル数: 0")
        status_bar = tk.Label(
            self._win,
            textvariable=self._status_var,
            fg="black",
            bg="#f0f0f0",
            font=("Arial", 12),
            anchor="w",
            padx=16,
        )
        status_bar.place(x=0, y=0, width=sw, height=_STATUS_HEIGHT)

        # ── Text canvas ─────────────────────────────────────────────────────
        canvas_h = sh - _STATUS_HEIGHT - _BOTTOM_HEIGHT
        self._canvas = tk.Canvas(
            self._win,
            width=sw,
            height=canvas_h,
            bg="white",
            highlightthickness=0,
        )
        self._canvas.place(x=0, y=_STATUS_HEIGHT, width=sw, height=canvas_h)
        self._canvas_h = canvas_h

        # ── Bottom bar (buttons + sample counter) ──────────────────────────
        bottom = tk.Frame(self._win, bg="#f0f0f0")
        bottom.place(x=0, y=sh - _BOTTOM_HEIGHT, width=sw, height=_BOTTOM_HEIGHT)

        self._done_btn = tk.Button(
            bottom,
            text="完了",
            font=("Arial", 12, "bold"),
            bg="#4caf50",
            fg="white",
            padx=20,
            command=self._finish,
            state="disabled",
        )
        self._done_btn.pack(side="left", padx=16, pady=8)

        cancel_btn = tk.Button(
            bottom,
            text="キャンセル",
            font=("Arial", 12),
            padx=16,
            command=self._on_esc,
        )
        cancel_btn.pack(side="left", padx=4, pady=8)

        self._counter_var = tk.StringVar(value="収集: 0 サンプル")
        tk.Label(
            bottom,
            textvariable=self._counter_var,
            font=("Arial", 11),
            bg="#f0f0f0",
            fg="#555",
        ).pack(side="left", padx=24, pady=8)

        # ── Render text & start sampling ────────────────────────────────────
        self._parsed = self._parse_md(md_text)
        self._win.update_idletasks()   # measure canvas size
        self._render_text()
        self._win.after(200, self._collect_sample)

    # ------------------------------------------------------------------
    # Markdown parsing
    # ------------------------------------------------------------------

    def _parse_md(self, md_text: str) -> list[tuple[str, str]]:
        """
        Parse a Markdown subset into a list of (tag, text) tuples.
        tag values: 'h1', 'h2', 'bullet', 'para', 'blank'
        """
        blocks: list[tuple[str, str]] = []
        for raw_line in md_text.splitlines():
            line = raw_line.rstrip()
            if not line.strip():
                blocks.append(("blank", ""))
            elif line.startswith("# ") and not line.startswith("## "):
                blocks.append(("h1", line[2:].strip()))
            elif line.startswith("## "):
                blocks.append(("h2", line[3:].strip()))
            elif line.lstrip().startswith("- ") or line.lstrip().startswith("* "):
                # strip leading spaces then the marker
                stripped = line.lstrip()
                content = stripped[2:].strip()
                blocks.append(("bullet", content))
            else:
                blocks.append(("para", line.strip()))
        # Remove leading/trailing blank lines
        while blocks and blocks[0][0] == "blank":
            blocks.pop(0)
        while blocks and blocks[-1][0] == "blank":
            blocks.pop()
        return blocks

    # ------------------------------------------------------------------
    # Text rendering
    # ------------------------------------------------------------------

    def _render_text(self) -> None:
        """
        Render parsed Markdown to the canvas, recording each text line's
        normalised screen Y in self._text_line_ys.
        """
        sw = self._sw
        sh_canvas = self._canvas_h
        text_width = int(sw * _TEXT_WIDTH_FRACTION)
        x_left = (sw - text_width) // 2
        x_center = sw // 2

        y = 40  # top padding inside canvas

        for tag, text in self._parsed:
            if tag == "blank":
                y += _MARGIN_BLANK
                continue

            if tag == "h1":
                y += _MARGIN_H1[0]
                font = _FONT_H1
                x = x_center
                anchor = "n"
            elif tag == "h2":
                y += _MARGIN_H2[0]
                font = _FONT_H2
                x = x_center
                anchor = "n"
            elif tag == "bullet":
                font = _FONT_BULLET
                text = "• " + text
                x = x_left + 20   # indent
                anchor = "nw"
            else:  # para
                font = _FONT_PARA
                x = x_left
                anchor = "nw"

            # Wrap text to fit text_width
            lines = self._wrap_text(text, font, text_width)

            for wrapped_line in lines:
                self._canvas.create_text(
                    x, y,
                    text=wrapped_line,
                    font=font,
                    fill="black",
                    anchor=anchor,
                    width=text_width,
                )
                # Only record sweep-target Y for lines a reader actually scans
                # left-to-right (paragraphs and bullets).  Headings are short /
                # centred and produce no reliable horizontal sweep, so including
                # them would misalign subsequent sweep→line assignments.
                if tag in ("para", "bullet"):
                    screen_y_px = _STATUS_HEIGHT + y
                    norm_y = screen_y_px / self._sh
                    self._text_line_ys.append(norm_y)

                line_h = self._line_height(font)
                y += line_h

            # Bottom margin
            if tag == "h1":
                y += _MARGIN_H1[1]
            elif tag == "h2":
                y += _MARGIN_H2[1]

        _log.debug("Rendered %d text lines; Y positions: %s",
                   len(self._text_line_ys), self._text_line_ys)

    def _line_height(self, font: tuple) -> int:
        """Return the pixel line height for a given font tuple (name, size, style?)."""
        size_pt = font[1]
        return int(size_pt * _LINE_HEIGHT_FACTOR)

    def _wrap_text(self, text: str, font: tuple, max_width_px: int) -> list[str]:
        """
        Naively wrap text into lines that fit within max_width_px.
        Uses a temporary canvas text item to measure widths.
        """
        words = text.split()
        if not words:
            return [""]

        lines: list[str] = []
        current = words[0]
        for word in words[1:]:
            candidate = current + " " + word
            if self._text_pixel_width(candidate, font) <= max_width_px:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return lines

    def _text_pixel_width(self, text: str, font: tuple) -> int:
        """Measure text width in pixels using a temporary canvas item."""
        tmp = self._canvas.create_text(0, 0, text=text, font=font, anchor="nw")
        bbox = self._canvas.bbox(tmp)
        self._canvas.delete(tmp)
        if bbox is None:
            return 0
        return bbox[2] - bbox[0]

    # ------------------------------------------------------------------
    # Gaze sampling loop
    # ------------------------------------------------------------------

    def _collect_sample(self) -> None:
        """Called every sample_interval_ms ms; collects gaze and runs sweep detection."""
        if self._done:
            return

        sample = self._source.get_local_gaze()
        if sample is not None:
            lx, ly = sample
            self._gaze_history.append((lx, ly, time.monotonic()))
            self._detect_scan()
            self._no_face_count = 0
            # 顔検出再開時にステータスを通常に戻す（walk away 後に戻った場合）
            if self._was_no_face:
                self._was_no_face = False
                self._update_ui()
        else:
            self._no_face_count += 1
            if self._no_face_count == 20:  # 約2秒間未検出
                self._was_no_face = True
                self._status_var.set("カメラに顔が映っていません。正面を向いてください。")

        self._after_id = self._win.after(self._sample_interval_ms, self._collect_sample)

    # ------------------------------------------------------------------
    # Sweep detection
    # ------------------------------------------------------------------

    def _detect_scan(self) -> None:
        """
        Detect reading sweeps using hysteresis on the running extremum of local_x.

        Instead of reacting to per-frame sign changes (which flips every sample
        due to unfiltered iris-relative noise), we track the *peak* local_x seen
        during the current sweep and declare a reversal only when local_x retreats
        past (peak - _SWEEP_HYSTERESIS) for a rising sweep, or rises past
        (peak + _SWEEP_HYSTERESIS) for a falling sweep.  This means a brief jitter
        that doesn't eclipse the margin extends the sweep; a true return saccade
        (much larger amplitude) crosses the threshold and closes it.

        On closure: if the sweep lasted ≥ _SWEEP_MIN_DURATION_S, emit calibration
        points; otherwise discard (saccade too short to be a line read).
        """
        h = self._gaze_history
        n = len(h)
        if n < 2:
            return

        # Loop back once all text lines have been consumed so that continued
        # reading still generates calibration samples toward min_samples.
        if not self._text_line_ys:
            return
        self._next_line_idx %= len(self._text_line_ys)

        lx_curr, _ly, t_curr = h[-1]

        if self._sweep_dir is None:
            # No active sweep — start one from the last sample
            lx_prev = h[-2][0]
            dx = lx_curr - lx_prev
            if abs(dx) < 1e-6:
                return  # stationary; wait
            self._sweep_dir = 1 if dx > 0 else -1
            self._sweep_start_idx = n - 2
            self._sweep_start_time = h[-2][2]
            self._sweep_peak = lx_curr
            return

        # Update running extremum
        if self._sweep_dir == 1:
            # Rising sweep: track maximum
            if lx_curr > self._sweep_peak:
                self._sweep_peak = lx_curr
                return  # still advancing — no reversal
            # Check if retreat exceeds hysteresis threshold
            if self._sweep_peak - lx_curr < _SWEEP_HYSTERESIS:
                return  # within noise margin — continue sweep
        else:
            # Falling sweep: track minimum
            if lx_curr < self._sweep_peak:
                self._sweep_peak = lx_curr
                return
            if lx_curr - self._sweep_peak < _SWEEP_HYSTERESIS:
                return

        # Hysteresis threshold crossed — sweep ended at sample n-2 (before current)
        sweep_duration = h[-2][2] - self._sweep_start_time
        if sweep_duration >= _SWEEP_MIN_DURATION_S:
            self._emit_sweep(self._sweep_start_idx, n - 2)
        else:
            _log.debug(
                "Sweep discarded (duration %.3fs < %.3fs)",
                sweep_duration, _SWEEP_MIN_DURATION_S,
            )

        # Start a new sweep from the current sample
        lx_prev = h[-2][0]
        new_dir = 1 if (lx_curr - lx_prev) >= 0 else -1
        self._sweep_dir = new_dir
        self._sweep_start_idx = n - 2
        self._sweep_start_time = h[-2][2]
        self._sweep_peak = lx_curr

    def _emit_sweep(self, start_idx: int, end_idx: int) -> None:
        """
        Emit calibration points for the sweep spanning
        gaze_history[start_idx..end_idx] (inclusive) against the next
        unconsumed text line.
        """
        if self._next_line_idx >= len(self._text_line_ys):
            _log.debug("No more text lines to consume — sweep ignored.")
            return

        screen_y = self._text_line_ys[self._next_line_idx]
        screen_x = 0.5  # L→R full-line sweep: use horizontal midpoint
        # NOTE: using screen_x=0.5 for all sweeps means the x-axis of the
        # calibration model will be trained on a single-column target; this
        # degeneracy is intentional per the spec — implicit calibration
        # primarily recovers the y-axis mapping.

        h = self._gaze_history
        added = 0
        for i in range(start_idx, min(end_idx + 1, len(h))):
            lx, ly, _ = h[i]
            self._add_sample(lx, ly, screen_x, screen_y)
            added += 1

        self._next_line_idx += 1
        _log.debug(
            "Sweep → line %d (screen_y=%.3f): %d points added  total=%d",
            self._next_line_idx - 1, screen_y, added, self._samples_collected,
        )
        self._flash_sweep_feedback(screen_y)

    def _flash_sweep_feedback(self, norm_y: float) -> None:
        """走査完了をキャンバス上の緑ラインで通知（400ms後に消える）。"""
        y_px = norm_y * self._sh - _STATUS_HEIGHT
        y_px = max(0, min(self._canvas_h, int(y_px)))
        line_id = self._canvas.create_line(
            0, y_px, self._sw, y_px,
            fill="#4caf50", width=2, tags="sweep_flash",
        )
        def _delete_flash(lid=line_id):
            try:
                self._canvas.delete(lid)
            except Exception:
                pass  # window already destroyed (force_close / restart)
        self._win.after(400, _delete_flash)

    # ------------------------------------------------------------------
    # Sample management
    # ------------------------------------------------------------------

    def _add_sample(
        self,
        local_x: float,
        local_y: float,
        screen_x: float,
        screen_y: float,
    ) -> None:
        """Add one calibration point and refresh the UI counter."""
        self._source.calibration.add_point(local_x, local_y, screen_x, screen_y)
        self._samples_collected += 1
        self._update_ui()

    def _update_ui(self) -> None:
        """Refresh status bar and button state."""
        n = self._samples_collected
        n_lines = self._next_line_idx
        total_lines = len(self._text_line_ys)
        pct = min(100, int(n / self._min_samples * 100)) if self._min_samples > 0 else 0

        if self._was_no_face:
            return  # 顔未検出メッセージ表示中は上書きしない

        if n < self._min_samples:
            self._status_var.set(
                f"走査: {n_lines}/{total_lines} 行完了  |  キャリブ点: {n}"
            )
        else:
            self._status_var.set(
                f"走査: {n_lines}/{total_lines} 行完了  |  キャリブ完了 — 「完了」を押してください"
            )
        self._counter_var.set(f"進捗: {n}/{self._min_samples} ({pct}%)")
        if n >= self._min_samples:
            self._done_btn.configure(state="normal", bg="#1b5e20", text="完了")

    # ------------------------------------------------------------------
    # Finish / cancel
    # ------------------------------------------------------------------

    def _finish(self) -> None:
        """Fit calibration and call on_done(success, n_samples)."""
        if self._done:
            return
        self._done = True
        self._cancel_after()

        # Flush any open sweep that qualified but never got a closing reversal
        # (the user clicks 完了 at the end of the last line without a return saccade).
        h = self._gaze_history
        if (
            self._sweep_dir is not None
            and h
            and self._next_line_idx < len(self._text_line_ys)
        ):
            sweep_duration = h[-1][2] - self._sweep_start_time
            if sweep_duration >= _SWEEP_MIN_DURATION_S:
                _log.debug("Flushing open sweep at finish (duration %.3fs)", sweep_duration)
                self._emit_sweep(self._sweep_start_idx, len(h) - 1)

        result = self._source.calibration.fit()
        _log.info(
            "Text calibration complete: %d samples, fit success=%s",
            self._samples_collected, result.success,
        )
        self._call_done(result.success, self._samples_collected)

    def force_close(self) -> None:
        """Close without confirmation — used by session restart to avoid a blocking messagebox."""
        if self._done:
            return
        self._done = True
        self._cancel_after()
        _log.info("Text calibration force-closed by session restart.")
        self._call_done(False, 0)

    def _on_esc(self, event=None) -> None:
        """ESC or window close: cancel without fitting (with confirmation if samples exist)."""
        if self._done:
            return
        if self._samples_collected > 0:
            from tkinter import messagebox
            if not messagebox.askyesno(
                "キャンセル確認",
                f"キャリブレーションを中止しますか？\n"
                f"収集済みの {self._samples_collected} サンプルは破棄されます。",
                parent=self._win,
            ):
                return
        self._done = True
        self._cancel_after()
        _log.info("Text calibration cancelled by user (ESC).")
        self._call_done(False, 0)

    def _cancel_after(self) -> None:
        """Cancel any pending after() callback."""
        if self._after_id is not None:
            try:
                self._win.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def _call_done(self, success: bool, n_samples: int) -> None:
        """Invoke the on_done callback then destroy the window."""
        try:
            self._on_done(success, n_samples)
        finally:
            self._win.destroy()
