"""Plot widgets: cell viewer, movie bar, trace / event views, matrices."""

import os
from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg

from PyQt6.QtCore import pyqtSignal, Qt, QTimer
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QToolButton,
    QWidget,
)

from .constants import CORR_A, CORR_B, CROSS_MIN_ARM, PALETTE_HEX
from .theme import CMAP_RDBU_R, CMAP_VIRIDIS_R, PT, UI
from .loading import _os_dirname
from .movie import MovieSource, find_session_movie
from .overlay import _build_overlay, _build_roi_overlay

if TYPE_CHECKING:                      # annotations only
    from .store import ComponentStore


# ── Cell viewer ───────────────────────────────────────────────────────────────

class CellViewer(pg.GraphicsLayoutWidget):
    """
    Top-left pane — Cn greyscale background with RGBA footprint overlay
    and per-component crosses on selected / pair components.

    Interaction (pyqtgraph built-in)
    ---------------------------------
      Right-drag     → pan
      Scroll wheel   → zoom centred on cursor
      Right-click    → context menu (includes "View All" / reset zoom)
      Left-click     → select component under cursor (footprint hit-test)
      Ctrl+Left      → toggle component in/out of selection
      Shift+Scroll   → step through movie frames (when a movie is shown)
    """

    component_clicked = pyqtSignal(int, bool)
    frame_scrolled    = pyqtSignal(int)      # ±steps from Shift+wheel

    def __init__(self, store: 'ComponentStore', parent=None):
        super().__init__(parent)
        self.store      = store
        self._sel       = []
        self._pair      = None
        self._centroids = None
        self._cross_params_cache = None
        self._invert_bg = False
        self._movie_img    = None     # current movie frame (float32, dims) or None
        self._movie_levels = None

        self.setBackground(PT["bg"])

        # ViewBox: aspect locked, Y not inverted → row 0 at bottom (origin='lower')
        self.vb = self.addViewBox(row=0, col=0)
        self.vb.setAspectLocked(True)
        self.vb.invertY(False)

        # Background (Cn)
        self._img_bg = pg.ImageItem()
        self._img_bg.setZValue(0)
        self.vb.addItem(self._img_bg)

        # Persistent "all ROIs" layer (dim cyan, every footprint; excluded
        # components in red once events exist, optionally hidden)
        self._show_all = True
        self._hide_excluded = False
        self._img_all  = pg.ImageItem()
        self._img_all.setZValue(0.5)
        self.vb.addItem(self._img_all)

        # Footprint overlay (RGBA) — selected / pair only, on top
        self._img_ov = pg.ImageItem()
        self._img_ov.setZValue(1)
        self.vb.addItem(self._img_ov)

        # Cross lines — managed as a list of PlotDataItems
        self._cross_items: list = []

        # Click detection via scene
        self.vb.scene().sigMouseClicked.connect(self._on_scene_click)

        self.refresh_rois()
        self._redraw()

    # ── Public interface ──────────────────────────────────────────────────────

    def set_selection(self, sel):
        self._sel  = list(sel)
        self._pair = None
        self._redraw()

    def set_pair(self, pair):
        self._pair = pair
        self._redraw()

    def refresh_rois(self):
        """
        Rebuild the persistent "all ROIs" base layer from the current store.

        Must be called after any component mutation (merge / delete / undo /
        redo / store swap) so the cell viewer reflects the new component set.
        Also drops the centroid / cross-arm caches, which are keyed on
        component count and would otherwise go stale when undo / redo restores
        a state with a coincidentally equal count.
        """
        self._centroids          = None
        self._cross_params_cache = None
        if self._show_all:
            self._img_all.setImage(
                _build_roi_overlay(self.store, self._hide_excluded))
            self._img_all.setVisible(True)
        else:
            self._img_all.setVisible(False)

    def set_show_all(self, flag: bool):
        self._show_all = bool(flag)
        self.refresh_rois()

    def set_hide_excluded(self, flag: bool):
        self._hide_excluded = bool(flag)
        self.refresh_rois()

    def refresh_background(self):
        """Redraw after the store's active background image has changed."""
        self._redraw()

    def apply_theme(self):
        self.setBackground(PT["bg"])
        self._redraw()

    def set_movie_frame(self, img, levels):
        """Show a movie frame as the background (fixed display *levels*).
        Only the background layer is redrawn, so playback stays cheap."""
        self._movie_img, self._movie_levels = img, levels
        self._draw_background()

    def clear_movie(self):
        """Return to the static background image."""
        if self._movie_img is not None:
            self._movie_img = self._movie_levels = None
            self._draw_background()

    @property
    def showing_movie(self) -> bool:
        return self._movie_img is not None

    def wheelEvent(self, ev):
        if (self._movie_img is not None
                and ev.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            d = ev.angleDelta()
            delta = d.y() or d.x()          # some platforms turn Shift+wheel horizontal
            if delta:
                self.frame_scrolled.emit(-1 if delta > 0 else 1)
            ev.accept()
            return
        super().wheelEvent(ev)

    def refresh_visibility(self):
        """Redraw after per-component visibility (hidden flags) has changed."""
        self.refresh_rois()
        self._redraw()


    def toggle_bg(self):
        self._invert_bg = not self._invert_bg
        self._redraw()

    def reset_zoom(self):
        self.vb.autoRange()

    # ── Scene click → component selection ────────────────────────────────────

    def _on_scene_click(self, event):
        if event.double():
            self.reset_zoom()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return

        pt  = self.vb.mapSceneToView(event.scenePos())
        col = int(round(pt.x()))
        row = int(round(pt.y()))

        s   = self.store
        d1, d2 = s.dims
        if not (0 <= row < d1 and 0 <= col < d2):
            return

        best_idx   = None
        best_value = 0.0
        peaks = s.peaks
        for i, val in zip(*s.components_at(row, col)):
            i = int(i)
            if not s.is_visible(i):            # hidden ROIs are not clickable
                continue
            if peaks[i] >= 1e-9 and val >= peaks[i] * 0.10 and val > best_value:
                best_value = float(val)
                best_idx   = i

        if best_idx is None:
            return

        ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        self.component_clicked.emit(best_idx, ctrl)

    # ── Centroid / cross-arm caches ───────────────────────────────────────────

    def _get_centroids(self):
        s = self.store
        if self._centroids is None or len(self._centroids) != s.n:
            self._centroids = [tuple(map(float, c)) for c in s.centroids]
        return self._centroids

    def _get_cross_params(self):
        s = self.store
        if (self._centroids is not None and len(self._centroids) == s.n
                and self._cross_params_cache is not None
                and len(self._cross_params_cache) == s.n):
            return self._cross_params_cache

        centroids = self._get_centroids()
        params    = []
        d1 = s.dims[0]
        for i in range(s.n):
            cy, cx = centroids[i]
            idx, val = s.fp_pixels(i)
            peak   = float(val.max()) if val.size else 0.0
            if peak < 1e-9:
                params.append((cy, cx, CROSS_MIN_ARM))
                continue
            on      = idx[val > peak * 0.2]
            ry, cx_ = on % d1, on // d1
            if ry.size == 0:
                params.append((cy, cx, CROSS_MIN_ARM))
                continue
            h   = int(ry.max() - ry.min()) + 1
            w   = int(cx_.max() - cx_.min()) + 1
            arm = max(CROSS_MIN_ARM, 0.75 * max(h, w))
            params.append((cy, cx, arm))

        self._cross_params_cache = params
        return self._cross_params_cache

    # ── Draw ──────────────────────────────────────────────────────────────────

    def _draw_background(self):
        if self._movie_img is not None:
            lo, hi = self._movie_levels
            img = self._movie_img
            if self._invert_bg:
                img = (lo + hi) - img
            self._img_bg.setImage(img, levels=(lo, hi), autoLevels=False)
            return
        cn = self.store.Cn.copy()
        if self._invert_bg:
            cn = cn.max() - cn
        lo, hi = cn.min(), cn.max()
        if hi > lo:
            cn8 = ((cn - lo) / (hi - lo) * 255).astype(np.uint8)
        else:
            cn8 = np.zeros_like(cn, dtype=np.uint8)
        self._img_bg.setImage(cn8, levels=(0, 255), autoLevels=False)

    def _redraw(self):
        s = self.store

        # Background (static image, or the current movie frame)
        self._draw_background()

        # RGBA overlay (selected + pair only)
        rgba = _build_overlay(s, self._sel, self._pair)
        self._img_ov.setImage(rgba)

        # Crosses
        self._draw_crosses()

    def _draw_crosses(self):
        # Remove old cross line items
        for item in self._cross_items:
            self.vb.removeItem(item)
        self._cross_items = []

        sel_set  = set(self._sel)
        pair_set = set(self._pair) if self._pair else set()
        active   = {i for i in (sel_set | pair_set)
                    if self.store.is_visible(i)}
        if not active:
            return

        cross_p = self._get_cross_params()
        for i in active:
            if i >= len(cross_p):
                continue
            cy, cx, arm = cross_p[i]

            if self._pair and i in pair_set:
                color = CORR_A if i == self._pair[0] else CORR_B
            else:
                k     = self._sel.index(i) if i in sel_set else 0
                color = PALETTE_HEX[k % len(PALETTE_HEX)]

            pen = pg.mkPen(color, width=1.5)
            h = pg.PlotDataItem([cx - arm, cx + arm], [cy,       cy      ], pen=pen)
            v = pg.PlotDataItem([cx,       cx      ], [cy - arm, cy + arm], pen=pen)
            h.setZValue(2)
            v.setZValue(2)
            self.vb.addItem(h)
            self.vb.addItem(v)
            self._cross_items.extend([h, v])


# ── Movie playback bar (under the cell viewer) ──────────────────────────────

class MovieBar(QWidget):
    """Play / scrub the session's raw .tif / .msr stack as the cell-view
    background, with the ROI overlay and crosses drawn on top.

    The movie next to the results file (``<session>.tif`` / ``.msr``) is
    found automatically and opened only when first used.  Keys while the
    image or this bar has focus: Space play/pause · ←/→ ±1 frame ·
    Shift+←/→ ±10 · Home/End.  Shift+wheel over the image steps frames.
    """

    def __init__(self, cell_view: 'CellViewer', trace_view: 'TraceViewer',
                 store: 'ComponentStore', parent=None):
        super().__init__(parent)
        self.cell_view, self.trace_view = cell_view, trace_view
        self.store = store
        self.src: MovieSource | None = None
        self.path = None
        self.cur  = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

        def _btn(text, tip, slot, checkable=False):
            b = QToolButton(self)
            b.setText(text)
            b.setToolTip(tip)
            b.setCheckable(checkable)
            b.setAutoRaise(True)
            (b.toggled if checkable else b.clicked).connect(slot)
            return b

        self.b_open  = _btn("⏏", "Open a movie (.tif / .msr) to play behind the ROIs…",
                            self.open_dialog)
        self.b_movie = _btn("Movie", "Show the movie frame as background "
                            "(off: the static BG image)", self.set_active, True)
        self.b_play  = _btn("▶", "Play / pause (Space)", self.toggle_play)
        self.slider  = QSlider(Qt.Orientation.Horizontal, self)
        self.slider.setToolTip("Scrub frames (←/→, Shift+←/→ ×10, Home/End; "
                               "Shift+wheel over the image)")
        self.slider.valueChanged.connect(self.seek)
        self.lbl = QLabel(self)
        self.lbl.setMinimumWidth(96)
        self.lbl.setStyleSheet(f"color: {UI['muted']}; font-family: monospace;")
        self.fps = QSpinBox(self)
        self.fps.setRange(1, 1000)
        self.fps.setSuffix(" fps")
        self.fps.setToolTip("Playback rate (default: the recording's frame rate)")
        self.fps.valueChanged.connect(self._on_fps)
        self.avg = QSpinBox(self)
        self.avg.setRange(1, MovieSource.MAX_AVG)
        self.avg.setPrefix("avg ")
        self.avg.setToolTip("Show the running mean of the last N frames "
                            "(denoises raw 2P frames)")
        self.avg.valueChanged.connect(lambda _: self.src and self.cell_view.showing_movie
                                      and self.seek(self.cur))

        # Take the cell pane's width instead of asking for one: a width hint
        # here pulls the window's splitters off their even split.
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 2, 4, 2)
        lay.setSpacing(4)
        for w in (self.b_open, self.b_movie, self.b_play):
            lay.addWidget(w)
        lay.addWidget(self.slider, 1)
        for w in (self.lbl, self.fps, self.avg):
            lay.addWidget(w)

        cell_view.frame_scrolled.connect(self.step)
        trace_view.cursor_dragged.connect(self.seek)
        self.set_store(store)

    # ── Store / file ──────────────────────────────────────────────────────────

    def set_store(self, store: 'ComponentStore'):
        """Reset for a new results file and look for its movie."""
        self.stop()
        self._close()
        self.store = store
        self.cur   = 0
        self.cell_view.clear_movie()
        self.trace_view.set_cursor(None)
        self.path = find_session_movie(store)
        fr = getattr(store, "fr", None)
        self.fps.blockSignals(True)
        self.fps.setValue(int(round(fr)) if fr and np.isfinite(fr) and fr >= 1 else 30)
        self.fps.blockSignals(False)
        self._update_ui()

    def open_dialog(self):
        win = self.window()
        start = self.path or (_os_dirname(self.store.fpath) if self.store.fpath else "")
        if not start and hasattr(win, "_start_dir"):
            start = win._start_dir()
        path, _ = QFileDialog.getOpenFileName(
            self, "Open movie", start,
            "Stacks (*.tif *.tiff *.msr);;TIFF (*.tif *.tiff);;MSR (*.msr);;All files (*)")
        if path and hasattr(win, "_remember_dir"):
            win._remember_dir(path)
        if path and self.open_path(path):
            self.set_active(True)

    def open_path(self, path: str) -> bool:
        try:
            src = MovieSource(path, self.store.dims)
        except Exception as exc:
            QMessageBox.warning(self, "Movie",
                                f"Cannot open {os.path.basename(path)}:\n{exc}")
            return False
        self.stop()
        self._close()
        self.src, self.path, self.cur = src, path, 0
        self.slider.blockSignals(True)
        self.slider.setRange(0, src.n_frames - 1)
        self.slider.setPageStep(max(1, src.n_frames // 50))
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self._update_ui()
        win = self.window()
        if hasattr(win, "statusBar"):
            win.statusBar().showMessage(
                f"Movie: {path}  ({src.n_frames} frames)", 8000)
        return True

    def _ensure_open(self) -> bool:
        if self.src is not None:
            return True
        return bool(self.path) and self.open_path(self.path)

    def _close(self):
        if self.src is not None:
            self.src.close()
            self.src = None

    def shutdown(self):
        self.stop()
        self._close()

    # ── Playback ──────────────────────────────────────────────────────────────

    def set_active(self, on: bool):
        """Movie frame (on) or the static background image (off)."""
        if on:
            if not self._ensure_open():
                self._set_movie_btn(False)
                return
            self.seek(self.cur)
        else:
            self.stop()
            self.cell_view.clear_movie()
            self.trace_view.set_cursor(None)
            self._set_movie_btn(False)
            self._update_ui()

    def seek(self, i: int):
        if not self._ensure_open():
            return
        src = self.src
        self.cur = int(min(max(int(i), 0), src.n_frames - 1))
        if self.slider.value() != self.cur:
            self.slider.blockSignals(True)
            self.slider.setValue(self.cur)
            self.slider.blockSignals(False)
        try:
            img = src.mean(self.cur, self.avg.value())
        except Exception as exc:
            self.stop()
            QMessageBox.warning(self, "Movie", f"Frame {self.cur}: {exc}")
            return
        self.cell_view.set_movie_frame(img, src.levels())
        self.sync_cursor()
        self._set_movie_btn(True)
        self._update_ui()

    def sync_cursor(self):
        """Trace cursor only when movie frames and trace samples align 1:1."""
        if self.src is not None and self.cell_view.showing_movie \
                and self.src.n_frames == self.store.disp_T:
            self.trace_view.set_cursor(self.cur)
        else:
            self.trace_view.set_cursor(None)

    def step(self, d: int):
        self.seek(self.cur + int(d))

    def seek_end(self):
        if self._ensure_open():
            self.seek(self.src.n_frames - 1)

    def toggle_play(self):
        if self._timer.isActive():
            self.stop()
        else:
            self.play()

    def play(self):
        if not self._ensure_open():
            return
        if not self.cell_view.showing_movie:
            self.seek(self.cur)
        self._timer.start(max(1, int(round(1000 / self.fps.value()))))
        self.b_play.setText("⏸")

    def stop(self):
        self._timer.stop()
        self.b_play.setText("▶")

    @property
    def playing(self) -> bool:
        return self._timer.isActive()

    def _tick(self):
        nxt = self.cur + 1
        self.seek(0 if nxt >= self.src.n_frames else nxt)    # loops

    def _on_fps(self, v):
        if self._timer.isActive():
            self._timer.setInterval(max(1, int(round(1000 / v))))

    # ── UI state ──────────────────────────────────────────────────────────────

    def _set_movie_btn(self, on: bool):
        self.b_movie.blockSignals(True)
        self.b_movie.setChecked(on)
        self.b_movie.blockSignals(False)

    def _update_ui(self):
        have = bool(self.path)
        for w in (self.b_movie, self.b_play, self.slider, self.fps, self.avg):
            w.setEnabled(have)
        open_tip = "Open a movie (.tif / .msr) to play behind the ROIs…"
        if not have:
            self.lbl.setText("no movie")
            self.b_open.setToolTip(open_tip)
            self.b_movie.setToolTip("No movie found next to the results — ⏏ to open one")
            return
        self.b_open.setToolTip(f"{open_tip}\nCurrent: {self.path}")
        self.b_movie.setToolTip(f"Show {os.path.basename(self.path)} as the background "
                                "(off: the static BG image)")
        if self.src is None:
            self.lbl.setText("not loaded")
        else:
            self.lbl.setText(f"{self.cur:>6d} / {self.src.n_frames - 1}")


# ── Trace viewer ──────────────────────────────────────────────────────────────

class TraceViewer(pg.PlotWidget):
    """Bottom-left — stacked normalised traces for selected components.
    While a movie plays, a yellow frame cursor marks the shown frame; drag
    it to seek."""

    cursor_dragged = pyqtSignal(int)

    def __init__(self, store: 'ComponentStore', parent=None):
        super().__init__(parent, background=PT["bg"])
        self.store = store
        self._sel  = []
        self._cursor_frame = None
        self._cursor = pg.InfiniteLine(angle=90, movable=True)
        self._cursor.setZValue(10)
        self._cursor.sigDragged.connect(
            lambda line: self.cursor_dragged.emit(int(round(line.value()))))
        self.getAxis('left').hide()
        self._style()
        self._legend = self._add_legend()
        self._redraw()

    def _style(self):
        self.setBackground(PT["bg"])
        self._cursor.setPen(pg.mkPen(PT["cursor"], width=1.5))
        self._cursor.setHoverPen(pg.mkPen(PT["cursor_hover"], width=3))
        ax = self.getAxis('bottom')
        ax.setLabel('Frame', color=PT["fg"])
        ax.setTextPen(pg.mkPen(PT["fg"]))
        ax.setPen(pg.mkPen(PT["fg"]))

    def _add_legend(self):
        return self.addLegend(labelTextColor=PT["legend_text"],
                              brush=pg.mkBrush(PT["legend_bg"]),
                              pen=pg.mkPen(PT["legend_pen"]))

    def apply_theme(self):
        self._style()
        self._redraw()

    def set_selection(self, sel):
        self._sel = list(sel)
        self._redraw()

    def refresh(self):
        """Redraw after the active display trace has changed."""
        self._redraw()

    def set_cursor(self, frame):
        """Show the movie frame cursor at *frame* (None hides it)."""
        self._cursor_frame = frame
        if frame is None:
            if self._cursor.scene() is not None:
                self.removeItem(self._cursor)
            return
        if self._cursor.scene() is None and self._sel:
            self.addItem(self._cursor, ignoreBounds=True)
        self._cursor.setBounds((0, max(self.store.disp_T - 1, 0)))
        self._cursor.setValue(frame)

    def _redraw(self):
        self.clear()
        self._legend = self._add_legend()
        s = self.store
        if not self._sel:
            ti = pg.TextItem("Select component(s)", color=PT["placeholder"],
                             anchor=(0.5, 0.5))
            self.addItem(ti)
            self.getViewBox().setRange(xRange=(0, 1), yRange=(0, 1))
            return

        try:
            T      = s.disp_T
            t      = np.arange(T, dtype=np.float32)
            offset = 0.0
            for k, i in enumerate(self._sel):
                color = PALETTE_HEX[k % len(PALETTE_HEX)]
                trace = s.disp_trace(i)
                span  = trace.max() - trace.min()
                tr    = (trace - trace.min()) / (span + 1e-9)
                self.plot(t, (tr + offset).astype(np.float32),
                          pen=pg.mkPen(color, width=1),
                          name=s.labels[i])
                offset += 1.3
            self.setXRange(0, T, padding=0)
            self.setYRange(-0.3, offset, padding=0)
            if self._cursor_frame is not None:     # clear() dropped it
                self.set_cursor(self._cursor_frame)
        except Exception as exc:
            self.clear()
            ti = pg.TextItem(f"Draw error: {exc}", color=PT["error"],
                             anchor=(0.5, 0.5))
            self.addItem(ti)


# ── Event viewer ──────────────────────────────────────────────────────────────

class EventViewer(pg.PlotWidget):
    """Below the traces — event detection for the most recently selected
    component (SLab's dF/f, filtered + peaks and binary panels in one):
    raw ΔF/F (grey), filtered ΔF/F (green), threshold (dashed), binary
    onset→peak plateaus (filled white) and detected peaks (red lines)."""

    def __init__(self, store: 'ComponentStore', parent=None):
        super().__init__(parent, background=PT["bg"])
        self.store = store
        self._sel  = []
        self.getAxis('left').enableAutoSIPrefix(False)   # no "(x0.001)"
        self.setMouseEnabled(x=True, y=False)
        self._style()
        self._redraw()

    def _style(self):
        self.setBackground(PT["bg"])
        for side in ('bottom', 'left'):
            self.getAxis(side).setTextPen(pg.mkPen(PT["fg"]))
            self.getAxis(side).setPen(pg.mkPen(PT["fg"]))
        self.getAxis('left').setLabel('ΔF/F', color=PT["fg"])

    def apply_theme(self):
        self._style()
        self._redraw()

    def set_selection(self, sel):
        self._sel = list(sel)
        self._redraw()

    def refresh(self):
        self._redraw()

    def _message(self, text):
        # Title only: this plot is x-linked to the trace viewer, so changing
        # the range here would squash the traces.
        self.setTitle(f"<span style='color:{PT['placeholder']};font-size:9pt'>{text}</span>")

    def _redraw(self):
        self.clear()
        self.setTitle(None)
        s = self.store
        if s.events is None:
            self._message("No events — Analysis ▸ Detect events…"
                          if s.can_detect else
                          "No events (file has no ΔF/F to detect on)")
            return
        if not self._sel:
            self._message("Select a component to see its events")
            return
        i  = self._sel[-1]
        ev = s.events
        t  = np.arange(ev.binary.shape[1], dtype=np.float32)
        if s.dff is not None:
            self.plot(t, s.dff[i], pen=pg.mkPen(PT["dff"], width=1))
        self.plot(t, ev.binary[i], pen=pg.mkPen(PT["binary"], width=1),
                  fillLevel=0, brush=pg.mkBrush(*PT["binary_fill"]))
        self.plot(t, ev.filtered[i], pen=pg.mkPen(PT["filtered"], width=1.2))
        thr = float(ev.thresholds[i])
        if thr > 0:
            self.addItem(pg.InfiniteLine(
                pos=thr, angle=0,
                pen=pg.mkPen(PT["threshold"], width=1, style=Qt.PenStyle.DashLine)))
        for pk in ev.peaks[i][:, 0]:
            self.addItem(pg.InfiniteLine(pos=float(pk), angle=90,
                                         pen=pg.mkPen(PT["peak"], width=1)))
        flag = "✓ included" if s.approved[i] else "✗ excluded"
        auto = "auto ✓" if ev.auto_approved[i] else "auto ✗"
        self.setTitle(
            f"<span style='color:{PT['title']};font-size:9pt'>{s.labels[i]} — "
            f"{ev.n_events(i)} events ({ev.frequencies[i]:.2f}/min), "
            f"thr {thr:.3f} — {flag} ({auto})</span>")
        self.setXRange(0, len(t), padding=0)
        self.enableAutoRange(axis='y')


# ── Shared matrix widget base ─────────────────────────────────────────────────

class _MatrixWidget(pg.GraphicsLayoutWidget):
    """
    Base for CorrMatrix and DistMatrix.
    Displays a K×K symmetric matrix as a heat-map with a fixed colorbar.
    Ctrl+clicking an off-diagonal cell emits pair_clicked(i, j).
    """

    pair_clicked = pyqtSignal(int, int)

    def __init__(self, store: 'ComponentStore', cmap: pg.ColorMap,
                 parent=None):
        super().__init__(parent)
        self.store      = store
        self._cmap      = cmap
        self._highlight = None
        self._hl_items: list = []

        self.setBackground(PT["bg"])

        # Column 0: matrix ViewBox (no mouse zoom/pan — leave for table/corr UX)
        self.vb = self.addViewBox(row=0, col=0)
        self.vb.invertY(True)            # row 0 at top (matrix convention)
        self.vb.setAspectLocked(True)
        self.vb.setMouseEnabled(x=True, y=True)
        self.vb.setMenuEnabled(True)    # right-click → "View All" reset zoom

        self._img = pg.ImageItem()
        self.vb.addItem(self._img)

        # Column 1: colorbar — do NOT pass insert_in (expects PlotItem,
        # not ViewBox).  Link the image manually and place in the grid.
        self._cbar = pg.ColorBarItem(
            colorMap=self._cmap, width=12,
            pen='#333333', hoverPen='#888888', hoverBrush='#888888'
        )
        self._cbar.setImageItem(self._img)   # no insert_in
        self.addItem(self._cbar, row=0, col=1)
        self.ci.layout.setColumnFixedWidth(1, 60)

        # Click
        self.vb.scene().sigMouseClicked.connect(self._on_scene_click)

    def apply_theme(self):
        self.setBackground(PT["bg"])

    def refresh(self):
        """Re-render after the component set changed; clears the highlight."""
        self._highlight = None
        self._render()

    def clear_highlight(self):
        """Remove the pair highlight (no matrix recompute)."""
        if self._highlight is not None or self._hl_items:
            self._highlight = None
            self._draw_highlight()

    def highlight_pair(self, i: int, j: int):
        """Highlight (i, j) without emitting pair_clicked (avoids signal loop)."""
        self._highlight = (i, j)
        self._draw_highlight()

    def _levels(self, mat: np.ndarray):
        raise NotImplementedError

    def _matrix(self) -> np.ndarray:
        raise NotImplementedError

    def redraw(self):
        """Re-render from the cached matrix (visibility changed); keeps the
        highlight unless it involves a now-invisible component."""
        if self._highlight and not all(self.store.is_visible(k) for k in self._highlight):
            self._highlight = None
        self._render()

    def _render(self):
        mat    = self._matrix()          # (K, K) float32, diag=NaN
        vis    = self.store.visible_mask()
        if not vis.all():                # hidden / hidden-rejected rows: neutral, not
            mat = mat.copy()             # in the colour levels, not clickable
            mat[~vis, :] = np.nan
            mat[:, ~vis] = np.nan
        vmin, vmax = self._levels(mat)
        # NaN → midpoint colour so diagonal is neutral
        display = np.where(np.isnan(mat), (vmin + vmax) / 2.0, mat)
        self._img.setImage(display.astype(np.float32))
        self._img.setLookupTable(self._cmap.getLookupTable(nPts=256))
        self._img.setLevels((vmin, vmax))
        self._cbar.setLevels((vmin, vmax))
        self._draw_highlight()

    def _draw_highlight(self):
        # Each pixel [row, col] is placed at corner (col, row) by ImageItem
        # and spans to (col+1, row+1), so its centre is at (col+0.5, row+0.5).
        # All highlight coordinates are offset by +0.5 accordingly.
        for item in self._hl_items:
            self.vb.removeItem(item)
        self._hl_items = []
        if self._highlight:
            hi, hj = self._highlight
            K = self.store.n
            for band, col in [(hi, CORR_A), (hj, CORR_B)]:
                c = band + 0.5   # centre of the band-th row/col
                lh = pg.PlotDataItem(
                    [-0.5, K + 0.5], [c, c],
                    pen=pg.mkPen(col, width=1.5))
                lv = pg.PlotDataItem(
                    [c, c], [-0.5, K + 0.5],
                    pen=pg.mkPen(col, width=1.5))
                for li in (lh, lv):
                    li.setZValue(2)
                    self.vb.addItem(li)
                    self._hl_items.append(li)
            for (ri, ci_), col in [((hi, hj), CORR_A), ((hj, hi), CORR_B)]:
                box = pg.PlotDataItem(
                    [ci_, ci_ + 1, ci_ + 1, ci_, ci_],
                    [ri,  ri,      ri  + 1, ri + 1, ri],
                    pen=pg.mkPen(col, width=2))
                box.setZValue(3)
                self.vb.addItem(box)
                self._hl_items.append(box)

    def _on_scene_click(self, event):
        # Require Ctrl+Left-click for pair selection so plain left-drag
        # is free for panning (handled by ViewBox natively).
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if not (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            return
        pt = self.vb.mapSceneToView(event.scenePos())
        j  = int(pt.x())   # floor: pixel [r,c] occupies [c, c+1)
        i  = int(pt.y())   # floor: pixel [r,c] occupies [r, r+1)
        n  = self.store.n
        if (0 <= i < n and 0 <= j < n and i != j
                and self.store.is_visible(i) and self.store.is_visible(j)):
            self._highlight = (i, j)
            self._draw_highlight()
            self.pair_clicked.emit(i, j)


# ── Correlation matrix ────────────────────────────────────────────────────────

class CorrMatrix(_MatrixWidget):
    """Bottom-right — pairwise Pearson correlation heat-map (RdBu_r)."""

    def __init__(self, store: 'ComponentStore', parent=None):
        super().__init__(store, CMAP_RDBU_R, parent)
        self._render()

    def _matrix(self) -> np.ndarray:
        return self.store.corr.copy()

    def _levels(self, mat):
        off  = mat[~np.isnan(mat)]
        clim = float(np.clip(np.percentile(np.abs(off), 99), 0.05, 1.0)) \
               if off.size > 0 else 1.0
        return -clim, clim


# ── Distance matrix ───────────────────────────────────────────────────────────

class DistMatrix(_MatrixWidget):
    """Top-right — pairwise centroid distance heat-map (viridis_r)."""

    def __init__(self, store: 'ComponentStore', parent=None):
        super().__init__(store, CMAP_VIRIDIS_R, parent)
        self._log = False
        self._render()

    def set_log(self, log: bool):
        if log != self._log:
            self._log = log
            self._render()

    def _matrix(self) -> np.ndarray:
        dist = self.store.dist
        if self._log:
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(dist > 0, np.log10(dist), np.nan)
        return dist

    def _levels(self, mat):
        off = mat[~np.isnan(mat)]
        if off.size == 0:
            return 0.0, 1.0
        # in log mode `mat` is already log10(distance) (see _matrix)
        vmax = float(np.clip(np.percentile(off, 99), 1e-3, None)) \
               if off.size > 0 else 1.0
        vmin = float(off.min()) if self._log and off.size > 0 else 0.0
        return vmin, vmax


__all__ = ['CellViewer', 'MovieBar', 'TraceViewer', 'EventViewer', '_MatrixWidget', 'CorrMatrix', 'DistMatrix']
