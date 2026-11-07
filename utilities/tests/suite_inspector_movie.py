"""Offscreen test: movie playback behind the ROI overlay in caiman_inspector."""
from _harness import REPO, load_inspector, ok, require, stub_caiman  # noqa: F401
require('tifffile')
import importlib.util, shutil, sys, tempfile, time, types
from pathlib import Path

import numpy as np
import tifffile

# light stand-in for the caiman package: only stack_io (+ imspectorreader) needed

insp = load_inspector()

from PyQt6.QtCore import QEventLoop, QPoint, QPointF, Qt, QTimer
from PyQt6.QtGui import QWheelEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QMessageBox
app = QApplication.instance() or QApplication([])
warnings = []
QMessageBox.warning = staticmethod(lambda *a, **k: warnings.append(a[2] if len(a) > 2 else a))

H, W, T, K = 40, 50, 60, 5               # non-square: catches transposes
dims = (H, W)
rng = np.random.default_rng(0)
movie = (rng.random((T, H, W)) * 100 + np.arange(T)[:, None, None] * 10).astype(np.uint16)


def make_store(folder: Path, name="S", t=T):
    A = np.zeros((H * W, K), np.float32)
    for k in range(K):
        A[rng.choice(H * W, 8, replace=False), k] = 1.0
    st = insp.ComponentStore(A, rng.random((K, t)).astype(np.float32), rng.random(dims), dims, fr=40.0)
    st.fpath = str(folder / f"{name}_results.hdf5")
    return st


def wait(ms):
    loop = QEventLoop(); QTimer.singleShot(ms, loop.quit); loop.exec()


root = Path(tempfile.mkdtemp())
try:
    sess = root / "S"; sess.mkdir()
    tifffile.imwrite(sess / "S.tif", movie, bigtiff=True, metadata=None)
    store = make_store(sess)
    assert insp.find_session_movie(store) == str(sess / "S.tif")
    win = insp.InspectorWindow(store); win.resize(1400, 900); win.show(); app.processEvents()
    bar, cv, tv = win.movie_bar, win.cell_view, win.trace_view
    assert bar.path == str(sess / "S.tif") and bar.src is None and bar.lbl.text() == "not loaded"
    assert not cv.showing_movie and bar.fps.value() == 40
    ok("session movie found next to the results; opened lazily; fps = recording rate")

    win._on_cell_click(1, False); app.processEvents()
    ov_before = cv._img_ov.image.copy()
    bar.seek(5); app.processEvents()
    assert bar.src is not None and bar.src.n_frames == T and cv.showing_movie
    assert np.array_equal(cv._img_bg.image, movie[5].astype(np.float32)), "frame 5 shown"
    assert bar.slider.value() == 5 and bar.lbl.text().split() == ["5", "/", str(T - 1)]
    assert bar.b_movie.isChecked()
    lo, hi = bar.src.levels()
    assert tuple(cv._img_bg.levels) == (lo, hi)
    assert np.array_equal(cv._img_ov.image, ov_before), "ROI overlay untouched"
    ok("seek shows the frame (native orientation) under an unchanged ROI overlay")

    bar.seek(30); l1 = tuple(cv._img_bg.levels); bar.seek(31)
    assert tuple(cv._img_bg.levels) == l1, "levels fixed across frames (no flicker)"
    bar.avg.setValue(4); app.processEvents()
    assert np.allclose(cv._img_bg.image, movie[28:32].astype(np.float32).mean(0))
    bar.avg.setValue(1); bar.seek(1); bar.avg.setValue(4)
    assert np.allclose(cv._img_bg.image, movie[0:2].astype(np.float32).mean(0)), "window clipped at 0"
    bar.avg.setValue(1)
    ok("fixed display levels; running mean over the last N frames")

    # trace cursor follows / drags
    bar.seek(12)
    assert tv._cursor.scene() is not None and tv._cursor.value() == 12
    tv.cursor_dragged.emit(20); app.processEvents()
    assert bar.cur == 20 and np.array_equal(cv._img_bg.image, movie[20].astype(np.float32))
    win._on_cell_click(3, True); app.processEvents()        # trace redraw clears plot items
    assert tv._cursor.scene() is not None and tv._cursor.value() == 20, "cursor survives redraw"
    ok("trace cursor tracks the frame, survives redraws, dragging it seeks")

    # play / loop / stop
    bar.fps.setValue(200); bar.seek(0); bar.play()
    assert bar.playing and bar.b_play.text() == "⏸"
    wait(400)
    assert bar.cur > 5, f"playback advanced only to {bar.cur}"
    bar.stop(); c = bar.cur; wait(100)
    assert bar.cur == c and not bar.playing and bar.b_play.text() == "▶"
    bar.seek(T - 1); bar._tick()
    assert bar.cur == 0, "loops at the end"
    ok(f"play advances, stop halts, end loops to 0")

    # Shift+wheel over the image steps; plain wheel does not
    bar.seek(10)
    def wheel(delta, mods):
        ev = QWheelEvent(QPointF(50, 50), QPointF(50, 50), QPoint(0, 0), QPoint(0, delta),
                         Qt.MouseButton.NoButton, mods, Qt.ScrollPhase.NoScrollPhase, False)
        cv.wheelEvent(ev)
    wheel(-120, Qt.KeyboardModifier.ShiftModifier); assert bar.cur == 11
    wheel(120, Qt.KeyboardModifier.ShiftModifier); wheel(120, Qt.KeyboardModifier.ShiftModifier)
    assert bar.cur == 9
    wheel(120, Qt.KeyboardModifier.NoModifier); assert bar.cur == 9, "plain wheel = zoom, not seek"
    ok("Shift+wheel steps frames; plain wheel still zooms")

    # keyboard shortcuts while the image has focus
    win.activateWindow(); cv.setFocus(); app.processEvents()
    if cv.hasFocus():
        QTest.keyClick(cv, Qt.Key.Key_Right); assert bar.cur == 10
        QTest.keyClick(cv, Qt.Key.Key_Right, Qt.KeyboardModifier.ShiftModifier); assert bar.cur == 20
        QTest.keyClick(cv, Qt.Key.Key_End); assert bar.cur == T - 1
        QTest.keyClick(cv, Qt.Key.Key_Home); assert bar.cur == 0
        QTest.keyClick(cv, Qt.Key.Key_Space); assert bar.playing
        QTest.keyClick(cv, Qt.Key.Key_Space); assert not bar.playing
        ok("keys: →, Shift+→, End, Home, Space")
    else:
        print("skip keys: offscreen platform gave no focus")
    # table focus: arrows must not move the movie
    win.table.setFocus(); app.processEvents(); c = bar.cur
    QTest.keyClick(win.table, Qt.Key.Key_Right)
    assert bar.cur == c, "shortcut leaked into the table"
    ok("movie keys do not fire while the table has focus")

    # invert, merge-refresh keep the movie
    bar.seek(7); win.act_invert_bg.trigger(); app.processEvents()
    lo, hi = bar.src.levels()
    assert np.allclose(cv._img_bg.image, (lo + hi) - movie[7].astype(np.float32))
    win.act_invert_bg.trigger()
    win._on_cell_click(0, False); win._on_cell_click(2, True); win._do_merge(); app.processEvents()
    assert cv.showing_movie and np.array_equal(cv._img_bg.image, movie[7].astype(np.float32))
    ok("invert and merge keep showing the movie frame")

    # picking a static BG leaves movie mode
    win.bg_combo.activated.emit(0); app.processEvents()
    assert not cv.showing_movie and not bar.b_movie.isChecked() and tv._cursor.scene() is None
    assert cv._img_bg.image.dtype == np.uint8, "static BG back"
    bar.b_movie.setChecked(True); app.processEvents()
    assert cv.showing_movie and bar.cur == 7, "toggle back resumes at the same frame"
    ok("BG combo → static background; Movie toggle → back at the same frame")

    # store swap to a session without a movie
    other = root / "O"; other.mkdir()
    win._swap_store(make_store(other, "O")); app.processEvents()
    assert win.movie_bar.path is None and bar.src is None and not cv.showing_movie
    assert not bar.b_play.isEnabled() and bar.lbl.text() == "no movie"
    ok("opening a file without a movie resets the bar")

    # transposed movie file (W×H frames) is auto-oriented
    tifffile.imwrite(other / "flipped.tif", movie.transpose(0, 2, 1), bigtiff=True, metadata=None)
    assert bar.open_path(str(other / "flipped.tif"))
    bar.set_active(True); bar.seek(3)
    assert np.array_equal(cv._img_bg.image, movie[3].astype(np.float32))
    ok("transposed stack auto-oriented to the results FOV")

    # trace length ≠ movie length → no cursor
    tifffile.imwrite(other / "long.tif", np.concatenate([movie, movie[:10]]), bigtiff=True, metadata=None)
    bar.open_path(str(other / "long.tif")); win._on_cell_click(0, False); bar.seek(4)
    assert cv.showing_movie and tv._cursor.scene() is None
    ok("no trace cursor when movie and trace lengths differ")

    # wrong FOV → warning, nothing breaks
    tifffile.imwrite(other / "wrong.tif", np.zeros((5, 33, 33), np.uint16), metadata=None)
    n = len(warnings)
    assert not bar.open_path(str(other / "wrong.tif"))
    assert len(warnings) == n + 1 and "33×33" in str(warnings[-1]) and "40×50" in str(warnings[-1])
    assert bar.src is not None and bar.path.endswith("long.tif"), "previous movie kept"
    ok("FOV mismatch is refused with a clear message")

    # the dir-named file is found too (caiman-inspector run inside a channel dir)
    ch = root / "sess-C00-fc000060"; ch.mkdir()
    tifffile.imwrite(ch / "sess-C00-fc000060.tif", movie[:3], metadata=None)
    st = make_store(ch, "sess-C00-fc000060"); assert insp.find_session_movie(st).endswith("sess-C00-fc000060.tif")
    st.fpath = str(ch / "renamed_results.hdf5")
    assert insp.find_session_movie(st).endswith("sess-C00-fc000060.tif"), "folder-name fallback"
    ok("movie found by session prefix or folder name")

    win.close(); app.processEvents()
    assert bar.src is None
    ok("closing the window closes the movie")
    print("ALL OK")
finally:
    shutil.rmtree(root)
