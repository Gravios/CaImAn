#!/usr/bin/env python3
"""
tiff_viewer.py — lightweight TIFF stack viewer
===============================================
Opens arbitrarily large TIFF stacks without loading the whole file into RAM.
Each frame is read on demand from the tifffile page index; a small LRU cache
keeps recently-visited frames warm.

Usage
-----
    python utilities/tiff_viewer.py                     # file-open dialog
    python utilities/tiff_viewer.py recording.tif
    python utilities/tiff_viewer.py recording.tif 500   # open at frame 500

Keyboard shortcuts
------------------
    Left / Right        previous / next frame
    Shift+Left/Right    −/+ 10 frames
    Ctrl+Left/Right     −/+ 100 frames
    Home / End          first / last frame
    Space               play / pause
    + / -               playback speed ×1.5 / ÷1.5
    A                   toggle auto-contrast (1–99th percentile)
    R                   reset contrast to full dtype range
    M                   toggle measure tool (draw line → pixel length)
    C                   clear all measurements
    Z                   reset zoom
    S                   save current frame as PNG
    Q / Escape          quit

Scroll wheel
------------
    Scroll up/down      previous/next frame
    Ctrl+Scroll         zoom in/out centred on cursor

Measure tool (press M)
----------------------
    Click and drag to draw a line.  The Euclidean pixel length is shown
    alongside the line and listed in the sidebar.  Multiple measurements
    accumulate; press C to clear all.

Requirements
------------
    tifffile   numpy   matplotlib   tkinter   pillow
    (all present in the caiman conda environment)
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import tifffile

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# ── Palette ───────────────────────────────────────────────────────────────────

BG         = "#1c1c1c"
BG_PANEL   = "#242424"
FG         = "#d0d0d0"
FG_DIM     = "#707070"
ACCENT     = "#4fc3f7"
DANGER     = "#ef5350"
SUCCESS    = "#69f0ae"

MEASURE_COLORS = [
    "#ff6b6b", "#ffd93d", "#6bcb77", "#4d96ff",
    "#ff922b", "#cc5de8", "#20c997", "#f06595",
]

CMAPS = [
    "gray", "inferno", "hot", "viridis", "plasma",
    "magma", "cividis", "RdBu_r",
]

PLAY_FPS_DEFAULT = 30
CACHE_FRAMES     = 64
AUTO_LO_PCT      = 1.0
AUTO_HI_PCT      = 99.0


# ── LRU frame cache ───────────────────────────────────────────────────────────

class FrameCache:
    def __init__(self, tf: tifffile.TiffFile, maxsize: int = CACHE_FRAMES):
        self._tf      = tf
        self._maxsize = maxsize
        self._data:   dict[int, np.ndarray] = {}
        self._order:  list[int]             = []
        self._lock    = threading.Lock()

    def get(self, idx: int) -> np.ndarray:
        with self._lock:
            if idx in self._data:
                self._order.remove(idx)
                self._order.append(idx)
                return self._data[idx]
            frame = self._tf.pages[idx].asarray().astype(np.float32)
            if len(self._order) >= self._maxsize:
                evict = self._order.pop(0)
                del self._data[evict]
            self._data[idx] = frame
            self._order.append(idx)
            return frame

    def prefetch(self, indices: list[int]) -> None:
        for i in indices:
            with self._lock:
                if i in self._data:
                    continue
            self.get(i)


# ── Measurement ───────────────────────────────────────────────────────────────

class Measurement:
    _counter = itertools.count(1)

    def __init__(self, x0, y0, x1, y1, color):
        self.id     = next(Measurement._counter)
        self.x0, self.y0 = x0, y0
        self.x1, self.y1 = x1, y1
        self.color  = color
        self.length = np.hypot(x1 - x0, y1 - y0)
        self.line_artist  = None
        self.text_artist  = None
        self.dot0_artist  = None
        self.dot1_artist  = None


# ── Viewer ────────────────────────────────────────────────────────────────────

class TiffViewer:

    def __init__(self, root: tk.Tk, path: str, start_frame: int = 0):
        self.root = root
        self.path = Path(path)

        self.tf       = tifffile.TiffFile(str(self.path))
        self.n_frames = len(self.tf.pages)
        if self.n_frames == 0:
            messagebox.showerror("Error", "TIFF has no pages.")
            root.destroy()
            return

        probe      = self.tf.pages[0].asarray()
        self.h     = probe.shape[0]
        self.w     = probe.shape[1] if probe.ndim > 1 else 1
        self.dtype = probe.dtype
        del probe

        self.cache = FrameCache(self.tf)

        # state
        self._idx           = max(0, min(start_frame, self.n_frames - 1))
        self._playing       = False
        self._play_fps      = PLAY_FPS_DEFAULT
        self._play_thread   = None
        self._auto_contrast = True
        if np.issubdtype(self.dtype, np.integer):
            info = np.iinfo(self.dtype)
            self._vmin = float(info.min)
            self._vmax = float(info.max)
        else:
            self._vmin, self._vmax = 0.0, 1.0
        self._cmap          = "gray"

        # measure state
        self._measure_mode  = False
        self._drag_start    = None
        self._drag_line     = None
        self._measurements: list[Measurement] = []
        self._color_cycle   = itertools.cycle(MEASURE_COLORS)

        self._build_ui()
        self._bind_keys()
        self._load_frame(self._idx, reset_contrast=True)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.root.title(f"TIFF Viewer — {self.path.name}")
        self.root.configure(bg=BG)
        self.root.minsize(860, 600)

        pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL,
                              bg=BG, sashwidth=5, sashrelief=tk.FLAT)
        pane.pack(fill=tk.BOTH, expand=True)

        # ── LEFT: image + controls ────────────────────────────────────────────
        left = tk.Frame(pane, bg=BG)
        pane.add(left, minsize=500)

        self.fig = Figure(figsize=(7, 5.5), dpi=100, facecolor="black")
        self.ax  = self.fig.add_axes([0, 0, 1, 1], facecolor="black")
        self.ax.set_axis_off()
        self.im  = self.ax.imshow(
            np.zeros((self.h, self.w), dtype=np.float32),
            cmap=self._cmap, aspect="equal",
            interpolation="nearest", origin="upper",
            vmin=self._vmin, vmax=self._vmax,
        )
        self.fig.subplots_adjust(0, 0, 1, 1)

        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        self.canvas.draw()

        # bottom control bar
        ctrl = tk.Frame(left, bg=BG_PANEL, pady=5)
        ctrl.pack(fill=tk.X, side=tk.BOTTOM)

        self._frame_var = tk.IntVar(value=self._idx)
        self.slider = ttk.Scale(
            ctrl, from_=0, to=max(0, self.n_frames - 1),
            orient=tk.HORIZONTAL, variable=self._frame_var,
            command=self._on_slider,
        )
        self.slider.pack(fill=tk.X, side=tk.LEFT, expand=True, padx=(8, 4))

        self._frame_label = tk.Label(
            ctrl, text=self._frame_text(),
            bg=BG_PANEL, fg=FG, font=("Courier", 9), width=18, anchor="w",
        )
        self._frame_label.pack(side=tk.LEFT, padx=(2, 6))

        self._play_btn = tk.Button(
            ctrl, text="▶", bg=BG_PANEL, fg=ACCENT, relief=tk.FLAT,
            activebackground=BG_PANEL, activeforeground=ACCENT,
            font=("TkDefaultFont", 12), width=2,
            command=self._toggle_play,
        )
        self._play_btn.pack(side=tk.LEFT, padx=2)

        tk.Label(ctrl, text="fps:", bg=BG_PANEL, fg=FG_DIM,
                 font=("TkDefaultFont", 8)).pack(side=tk.LEFT, padx=(6, 0))
        self._fps_var = tk.StringVar(value=str(self._play_fps))
        fps_e = tk.Entry(ctrl, textvariable=self._fps_var, width=4,
                         bg="#2a2a2a", fg=FG, insertbackground=FG,
                         relief=tk.FLAT, font=("Courier", 9))
        fps_e.pack(side=tk.LEFT, padx=(2, 8))
        fps_e.bind("<Return>", self._on_fps_change)

        # ── RIGHT: sidebar ────────────────────────────────────────────────────
        right = tk.Frame(pane, bg=BG_PANEL, width=220)
        pane.add(right, minsize=190)

        def _section(title):
            tk.Label(right, text=title, bg=BG_PANEL, fg=ACCENT,
                     font=("TkDefaultFont", 9, "bold"), anchor="w"
                     ).pack(fill=tk.X, padx=8, pady=(10, 1))
            tk.Frame(right, bg="#383838", height=1).pack(
                fill=tk.X, padx=8, pady=(0, 5))

        def _kv(label, value):
            row = tk.Frame(right, bg=BG_PANEL)
            row.pack(fill=tk.X, padx=8, pady=1)
            tk.Label(row, text=label + ":", bg=BG_PANEL, fg=FG_DIM,
                     font=("TkDefaultFont", 8), width=7, anchor="w"
                     ).pack(side=tk.LEFT)
            tk.Label(row, text=value, bg=BG_PANEL, fg=FG,
                     font=("Courier", 8), anchor="w",
                     wraplength=130, justify=tk.LEFT
                     ).pack(side=tk.LEFT, fill=tk.X)

        # File info
        _section("FILE")
        _kv("Name",   self.path.name)
        _kv("Frames", str(self.n_frames))
        _kv("Size",   f"{self.w} × {self.h} px")
        _kv("Dtype",  str(self.dtype))

        # Display
        _section("DISPLAY")
        cm_row = tk.Frame(right, bg=BG_PANEL)
        cm_row.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(cm_row, text="Colormap:", bg=BG_PANEL, fg=FG_DIM,
                 font=("TkDefaultFont", 8), anchor="w").pack(anchor="w")
        self._cmap_var = tk.StringVar(value=self._cmap)
        cm_box = ttk.Combobox(cm_row, textvariable=self._cmap_var,
                              values=CMAPS, state="readonly", width=14)
        cm_box.pack(anchor="w", pady=2)
        cm_box.bind("<<ComboboxSelected>>", self._on_cmap_change)

        self._ac_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            right, text="Auto-contrast (1–99%)", variable=self._ac_var,
            bg=BG_PANEL, fg=FG, selectcolor="#333",
            activebackground=BG_PANEL, activeforeground=FG,
            font=("TkDefaultFont", 8),
            command=self._on_ac_toggle,
        ).pack(anchor="w", padx=8, pady=2)

        for attr, label in [("_vmin_var", "Min"), ("_vmax_var", "Max")]:
            row = tk.Frame(right, bg=BG_PANEL)
            row.pack(fill=tk.X, padx=8, pady=1)
            tk.Label(row, text=label + ":", bg=BG_PANEL, fg=FG_DIM,
                     font=("TkDefaultFont", 8), width=4).pack(side=tk.LEFT)
            var = tk.DoubleVar(value=self._vmin if label == "Min" else self._vmax)
            setattr(self, attr, var)
            e = tk.Entry(row, textvariable=var, width=9,
                         bg="#2a2a2a", fg=FG, insertbackground=FG,
                         relief=tk.FLAT, font=("Courier", 8))
            e.pack(side=tk.LEFT, padx=2)
            e.bind("<Return>", self._on_manual_contrast)

        # Cursor / pixel info
        _section("CURSOR")
        self._pixel_var = tk.StringVar(value="—")
        tk.Label(right, textvariable=self._pixel_var, bg=BG_PANEL, fg=FG,
                 font=("Courier", 8), anchor="w", justify=tk.LEFT
                 ).pack(fill=tk.X, padx=8)

        # Measure
        _section("MEASURE  (M)")
        self._measure_btn = tk.Button(
            right, text="Measure: OFF", bg="#2a2a2a", fg=FG_DIM,
            relief=tk.FLAT, font=("TkDefaultFont", 8),
            command=self._toggle_measure,
        )
        self._measure_btn.pack(fill=tk.X, padx=8, pady=2)

        tk.Button(
            right, text="Clear all  (C)", bg="#2a2a2a", fg=FG_DIM,
            relief=tk.FLAT, font=("TkDefaultFont", 8),
            command=self._clear_measurements,
        ).pack(fill=tk.X, padx=8, pady=(0, 4))

        self._mlist = tk.Frame(right, bg=BG_PANEL)
        self._mlist.pack(fill=tk.BOTH, expand=True, padx=8)

        # Help
        _section("KEYS")
        tk.Label(right, text=(
            "←/→        ±1 frame\n"
            "Shift+←/→  ±10 frames\n"
            "Ctrl+←/→   ±100 frames\n"
            "Home/End    first/last\n"
            "Space       play/pause\n"
            "+/-         speed ×1.5\n"
            "Scroll      ±1 frame\n"
            "Ctrl+Scroll zoom\n"
            "Z           reset zoom\n"
            "M           measure tool\n"
            "C           clear measures\n"
            "A           auto-contrast\n"
            "R           reset contrast\n"
            "S           save PNG\n"
            "Q/Esc       quit"
        ), bg=BG_PANEL, fg=FG_DIM, font=("Courier", 7),
            justify=tk.LEFT, anchor="w").pack(fill=tk.X, padx=8, pady=2)

    # ── Frame loading ─────────────────────────────────────────────────────────

    def _frame_text(self) -> str:
        return f"{self._idx:6d} / {self.n_frames - 1}"

    def _load_frame(self, idx: int, reset_contrast: bool = False) -> None:
        self._idx = max(0, min(int(idx), self.n_frames - 1))
        frame = self.cache.get(self._idx)

        if reset_contrast or self._ac_var.get():
            lo = float(np.percentile(frame, AUTO_LO_PCT))
            hi = float(np.percentile(frame, AUTO_HI_PCT))
            if lo == hi:
                hi = lo + 1.0
            self._vmin, self._vmax = lo, hi
            self._vmin_var.set(round(lo, 2))
            self._vmax_var.set(round(hi, 2))

        self.im.set_data(frame)
        self.im.set_clim(self._vmin, self._vmax)
        self.canvas.draw_idle()

        self._frame_var.set(self._idx)
        self._frame_label.config(text=self._frame_text())

        # background prefetch
        fwd  = [min(self._idx + i, self.n_frames - 1) for i in range(1, 6)]
        bwd  = [max(self._idx - i, 0)                  for i in range(1, 3)]
        threading.Thread(
            target=self.cache.prefetch, args=(fwd + bwd,), daemon=True
        ).start()

    # ── Navigation ────────────────────────────────────────────────────────────

    def _go(self, delta: int) -> None:
        self._load_frame(self._idx + delta)

    def _on_slider(self, val) -> None:
        idx = int(float(val))
        if idx != self._idx:
            self._load_frame(idx)

    # ── Playback ──────────────────────────────────────────────────────────────

    def _toggle_play(self) -> None:
        if self._playing:
            self._playing = False
            self._play_btn.config(text="▶", fg=ACCENT)
        else:
            self._playing = True
            self._play_btn.config(text="⏸", fg=DANGER)
            threading.Thread(target=self._play_loop, daemon=True).start()

    def _play_loop(self) -> None:
        while self._playing:
            t0       = time.perf_counter()
            next_idx = (self._idx + 1) % self.n_frames
            self.root.after(0, self._load_frame, next_idx)
            sleep = max(0.0, 1.0 / self._play_fps - (time.perf_counter() - t0))
            time.sleep(sleep)

    def _on_fps_change(self, _event=None) -> None:
        try:
            self._play_fps = max(1, min(int(float(self._fps_var.get())), 200))
        except ValueError:
            pass
        self._fps_var.set(str(self._play_fps))

    def _scale_fps(self, factor: float) -> None:
        self._play_fps = max(1, min(int(self._play_fps * factor), 200))
        self._fps_var.set(str(self._play_fps))

    # ── Contrast ──────────────────────────────────────────────────────────────

    def _on_ac_toggle(self) -> None:
        if self._ac_var.get():
            self._load_frame(self._idx, reset_contrast=True)

    def _on_manual_contrast(self, _event=None) -> None:
        try:
            lo = float(self._vmin_var.get())
            hi = float(self._vmax_var.get())
        except ValueError:
            return
        if lo >= hi:
            return
        self._vmin, self._vmax = lo, hi
        self._ac_var.set(False)
        self.im.set_clim(lo, hi)
        self.canvas.draw_idle()

    def _reset_contrast(self) -> None:
        if np.issubdtype(self.dtype, np.integer):
            info = np.iinfo(self.dtype)
            lo, hi = float(info.min), float(info.max)
        else:
            lo, hi = 0.0, 1.0
        self._vmin, self._vmax = lo, hi
        self._vmin_var.set(round(lo, 2))
        self._vmax_var.set(round(hi, 2))
        self._ac_var.set(False)
        self.im.set_clim(lo, hi)
        self.canvas.draw_idle()

    def _on_cmap_change(self, _event=None) -> None:
        self._cmap = self._cmap_var.get()
        self.im.set_cmap(self._cmap)
        self.canvas.draw_idle()

    # ── Zoom ──────────────────────────────────────────────────────────────────

    def _zoom_at(self, cx: float, cy: float, factor: float) -> None:
        xl = list(self.ax.get_xlim())
        yl = list(self.ax.get_ylim())
        xl = [cx + (v - cx) / factor for v in xl]
        yl = [cy + (v - cy) / factor for v in yl]
        xl[0] = max(-0.5,          xl[0])
        xl[1] = min(self.w - 0.5,  xl[1])
        yl[0] = max(-0.5,          yl[0])
        yl[1] = min(self.h - 0.5,  yl[1])
        self.ax.set_xlim(xl)
        self.ax.set_ylim(yl)
        self.canvas.draw_idle()

    def _reset_zoom(self) -> None:
        self.ax.set_xlim(-0.5, self.w - 0.5)
        self.ax.set_ylim(self.h - 0.5, -0.5)
        self.canvas.draw_idle()

    # ── Measure tool ─────────────────────────────────────────────────────────

    def _toggle_measure(self) -> None:
        self._measure_mode = not self._measure_mode
        if self._measure_mode:
            self._measure_btn.config(text="Measure: ON ✓", fg=SUCCESS)
            self.canvas_widget.config(cursor="crosshair")
        else:
            self._measure_btn.config(text="Measure: OFF", fg=FG_DIM)
            self.canvas_widget.config(cursor="")
            self._cancel_drag()

    def _cancel_drag(self) -> None:
        if self._drag_line is not None:
            try:
                self._drag_line.remove()
            except Exception:
                pass
            self._drag_line = None
            self._drag_start = None
            self.canvas.draw_idle()

    def _clear_measurements(self) -> None:
        for m in self._measurements:
            for artist in (m.line_artist, m.text_artist,
                           m.dot0_artist, m.dot1_artist):
                if artist is not None:
                    try:
                        artist.remove()
                    except Exception:
                        pass
        self._measurements.clear()
        for w in self._mlist.winfo_children():
            w.destroy()
        self.canvas.draw_idle()

    def _data_coords(self, event) -> Optional[tuple[float, float]]:
        if event.xdata is None or event.ydata is None:
            return None
        return event.xdata, event.ydata

    def _on_mpl_press(self, event) -> None:
        if not self._measure_mode or event.inaxes != self.ax:
            return
        if event.button != 1:
            return
        pt = self._data_coords(event)
        if pt is None:
            return
        self._drag_start = pt

    def _on_mpl_motion(self, event) -> None:
        # pixel readout
        if event.inaxes == self.ax and event.xdata is not None:
            x = int(round(event.xdata))
            y = int(round(event.ydata))
            if 0 <= x < self.w and 0 <= y < self.h:
                val = self.cache.get(self._idx)[y, x]
                self._pixel_var.set(f"x={x}  y={y}\nval={val:.2f}")
            else:
                self._pixel_var.set("—")
        else:
            self._pixel_var.set("—")

        # drag rubber-band
        if not self._measure_mode or self._drag_start is None:
            return
        pt = self._data_coords(event)
        if pt is None:
            return
        x0, y0 = self._drag_start
        x1, y1 = pt
        length  = np.hypot(x1 - x0, y1 - y0)

        color = MEASURE_COLORS[len(self._measurements) % len(MEASURE_COLORS)]
        if self._drag_line is None:
            self._drag_line, = self.ax.plot(
                [x0, x1], [y0, y1], color=color,
                lw=1.5, ls="--", alpha=0.75, zorder=10,
            )
            self._drag_text = self.ax.text(
                (x0 + x1) / 2, (y0 + y1) / 2,
                f"{length:.1f} px",
                color=color, fontsize=8, fontweight="bold",
                ha="center", va="bottom", zorder=12,
                bbox=dict(boxstyle="round,pad=0.2", fc="#00000099", ec="none"),
            )
        else:
            self._drag_line.set_data([x0, x1], [y0, y1])
            self._drag_text.set_position(((x0 + x1) / 2, (y0 + y1) / 2))
            self._drag_text.set_text(f"{length:.1f} px")

        self.canvas.draw_idle()

    def _on_mpl_release(self, event) -> None:
        if not self._measure_mode or self._drag_start is None:
            return
        if event.button != 1:
            return

        # remove rubber-band
        if self._drag_line is not None:
            self._drag_line.remove()
            self._drag_line = None
        if hasattr(self, "_drag_text") and self._drag_text is not None:
            self._drag_text.remove()
            self._drag_text = None

        pt = self._data_coords(event)
        start = self._drag_start
        self._drag_start = None

        if pt is None or start is None:
            self.canvas.draw_idle()
            return

        x0, y0 = start
        x1, y1 = pt

        if np.hypot(x1 - x0, y1 - y0) < 2.0:
            self.canvas.draw_idle()
            return

        color = next(self._color_cycle)
        m = Measurement(x0, y0, x1, y1, color)

        # permanent line
        line, = self.ax.plot(
            [x0, x1], [y0, y1],
            color=color, lw=1.8, alpha=0.92, zorder=10,
            solid_capstyle="round",
        )
        m.line_artist = line

        # end-point dots
        d0, = self.ax.plot(x0, y0, "o", color=color, ms=5, zorder=11)
        d1, = self.ax.plot(x1, y1, "o", color=color, ms=5, zorder=11)
        m.dot0_artist, m.dot1_artist = d0, d1

        # annotation at midpoint
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        txt = self.ax.annotate(
            f"#{m.id}  {m.length:.1f} px",
            xy=(mx, my),
            xytext=(0, 10),
            textcoords="offset points",
            color=color, fontsize=8, fontweight="bold",
            ha="center", va="bottom", zorder=12,
            bbox=dict(boxstyle="round,pad=0.25", fc="#00000099", ec="none"),
        )
        m.text_artist = txt

        self._measurements.append(m)
        self._add_to_sidebar(m)
        self.canvas.draw_idle()

    def _add_to_sidebar(self, m: Measurement) -> None:
        row = tk.Frame(self._mlist, bg=BG_PANEL)
        row.pack(fill=tk.X, pady=1)
        tk.Label(row, text="■", fg=m.color, bg=BG_PANEL,
                 font=("TkDefaultFont", 10)).pack(side=tk.LEFT)
        tk.Label(row,
                 text=(f"#{m.id:02d}  {m.length:.1f} px\n"
                       f"    ({m.x0:.0f},{m.y0:.0f})"
                       f"→({m.x1:.0f},{m.y1:.0f})"),
                 fg=FG, bg=BG_PANEL, font=("Courier", 7),
                 justify=tk.LEFT, anchor="w"
                 ).pack(side=tk.LEFT, fill=tk.X)

    # ── Scroll → navigate / zoom ──────────────────────────────────────────────

    def _on_scroll_win(self, event) -> None:
        ctrl = bool(event.state & 0x4)
        if ctrl:
            factor = 1.15 if event.delta > 0 else 1 / 1.15
            self._zoom_from_tk(event, factor)
        else:
            self._go(-1 if event.delta > 0 else 1)

    def _on_scroll_up(self, event) -> None:
        if bool(event.state & 0x4):
            self._zoom_from_tk(event, 1.15)
        else:
            self._go(-1)

    def _on_scroll_down(self, event) -> None:
        if bool(event.state & 0x4):
            self._zoom_from_tk(event, 1 / 1.15)
        else:
            self._go(+1)

    def _zoom_from_tk(self, event, factor: float) -> None:
        try:
            cx, cy = self.ax.transData.inverted().transform(
                (event.x, event.y)
            )
        except Exception:
            return
        self._zoom_at(cx, cy, factor)

    # ── Save ──────────────────────────────────────────────────────────────────

    def _save_frame(self) -> None:
        out = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=f"{self.path.stem}_frame{self._idx:05d}.png",
            filetypes=[("PNG", "*.png"), ("TIFF", "*.tif"), ("All", "*")],
        )
        if not out:
            return
        frame = self.cache.get(self._idx)
        norm  = np.clip((frame - self._vmin) /
                        max(self._vmax - self._vmin, 1e-9), 0, 1)
        try:
            from PIL import Image as _PIL_Image
            cmap = matplotlib.colormaps[self._cmap]
            rgba = (cmap(norm) * 255).astype(np.uint8)
            _PIL_Image.fromarray(rgba).save(out)
        except ImportError:
            # Fallback: save raw normalised frame via matplotlib
            import matplotlib.image as _mimg
            _mimg.imsave(out, norm, cmap=self._cmap)

    # ── Bindings ──────────────────────────────────────────────────────────────

    def _bind_keys(self) -> None:
        r = self.root

        # Navigation
        r.bind("<Left>",          lambda e: self._go(-1))
        r.bind("<Right>",         lambda e: self._go(+1))
        r.bind("<Shift-Left>",    lambda e: self._go(-10))
        r.bind("<Shift-Right>",   lambda e: self._go(+10))
        r.bind("<Control-Left>",  lambda e: self._go(-100))
        r.bind("<Control-Right>", lambda e: self._go(+100))
        r.bind("<Home>",          lambda e: self._load_frame(0))
        r.bind("<End>",           lambda e: self._load_frame(self.n_frames - 1))

        # Playback
        r.bind("<space>",  lambda e: self._toggle_play())
        r.bind("<plus>",   lambda e: self._scale_fps(1.5))
        r.bind("<equal>",  lambda e: self._scale_fps(1.5))
        r.bind("<minus>",  lambda e: self._scale_fps(1 / 1.5))

        # Display
        r.bind("<a>", lambda e: (self._ac_var.set(not self._ac_var.get()),
                                 self._on_ac_toggle()))
        r.bind("<A>", lambda e: (self._ac_var.set(not self._ac_var.get()),
                                 self._on_ac_toggle()))
        r.bind("<r>", lambda e: self._reset_contrast())
        r.bind("<R>", lambda e: self._reset_contrast())
        r.bind("<z>", lambda e: self._reset_zoom())
        r.bind("<Z>", lambda e: self._reset_zoom())

        # Measure
        r.bind("<m>", lambda e: self._toggle_measure())
        r.bind("<M>", lambda e: self._toggle_measure())
        r.bind("<c>", lambda e: self._clear_measurements())
        r.bind("<C>", lambda e: self._clear_measurements())

        # Save / quit
        r.bind("<s>",      lambda e: self._save_frame())
        r.bind("<S>",      lambda e: self._save_frame())
        r.bind("<q>",      lambda e: self._quit())
        r.bind("<Q>",      lambda e: self._quit())
        r.bind("<Escape>", lambda e: self._quit())

        # Scroll (frame navigation + Ctrl+Scroll = zoom)
        self.canvas_widget.bind("<MouseWheel>", self._on_scroll_win)   # Win/Mac
        self.canvas_widget.bind("<Button-4>",   self._on_scroll_up)    # Linux
        self.canvas_widget.bind("<Button-5>",   self._on_scroll_down)  # Linux

        # matplotlib mouse events
        self.canvas.mpl_connect("button_press_event",   self._on_mpl_press)
        self.canvas.mpl_connect("motion_notify_event",  self._on_mpl_motion)
        self.canvas.mpl_connect("button_release_event", self._on_mpl_release)

    # ── Quit ─────────────────────────────────────────────────────────────────

    def _quit(self) -> None:
        self._playing = False
        try:
            self.tf.close()
        except Exception:
            pass
        self.root.destroy()


# ── Entry point ───────────────────────────────────────────────────────────────

def _pick_file() -> Optional[str]:
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Open TIFF stack",
        filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")],
    )
    root.destroy()
    return path or None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lightweight TIFF stack viewer — on-demand frame loading",
    )
    parser.add_argument("tiff",  nargs="?", help="TIFF file to open")
    parser.add_argument("frame", nargs="?", type=int, default=0,
                        help="Frame index to start at (default 0)")
    args = parser.parse_args()

    path = args.tiff or _pick_file()
    if path is None:
        print("No file selected.")
        sys.exit(0)
    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)

    root = tk.Tk()
    TiffViewer(root, path, start_frame=args.frame or 0)
    root.mainloop()


if __name__ == "__main__":
    main()
