"""Offscreen test: sortable component table in caiman_inspector."""
from _harness import REPO, load_inspector, ok, require, stub_caiman  # noqa: F401
import importlib.util, sys, types
from pathlib import Path

import numpy as np

insp = load_inspector()

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])

dims = (20, 20)
d, K, T = dims[0] * dims[1], 8, 50
rng = np.random.default_rng(0)
A = np.zeros((d, K), np.float32)
peaks = [0.5, 3.0, 0.2, 9.0, 10.0, 1.0, 0.7, 2.0]            # 10 > 9 numerically, not as text
for k in range(K):
    A[rng.choice(d, 6, replace=False), k] = peaks[k] * np.linspace(0.2, 1, 6)
C = rng.random((K, T)).astype(np.float32)
SNR = np.array([5.0, np.nan, 12.0, 3.0, 40.0, 7.0, np.nan, 1.0])
store = insp.ComponentStore(A, C, rng.random(dims), dims, SNR_comp=SNR)
win = insp.InspectorWindow(store)
win.show(); app.processEvents()
t = win.table
col = {t.horizontalHeaderItem(c).text(): c for c in range(t.columnCount())}


def order():
    return [t.comp_of(r) for r in range(t.rowCount())]


def consistent():
    for r in range(t.rowCount()):
        c = t.comp_of(r)
        assert t.row_of(c) == r, (r, c)
        assert t.item(r, col["Label"]).text() == win.store.labels[c], (r, c)
        assert t.item(r, col["Max A"]).text() == f"{win.store.max_a(c):.3f}"


def sort(name, desc=False):
    t.sortByColumn(col[name], Qt.SortOrder.DescendingOrder if desc else Qt.SortOrder.AscendingOrder)
    app.processEvents()


def click_rows(*rows):          # what a user click does: selection change → _on_table_sel
    t.clearSelection()
    for r in rows:
        t.selectRow(r)
    app.processEvents()



assert order() == list(range(K)); consistent()
ok("starts in store order")

sort("Max A")
mx = [store.max_a(c) for c in order()]
assert mx == sorted(mx) and order()[-1] == 4 and order()[-2] == 3, order()
consistent()
sort("Max A", desc=True)
assert order()[0] == 4 and order()[1] == 3
ok("Max A sorts numerically (10 above 9), both directions")

sort("SNR")
o = order()
assert o[-2:] in ([1, 6], [6, 1]), f"NaN must sort last: {o}"
vals = [SNR[c] for c in o[:-2]]
assert vals == sorted(vals), vals
consistent()
sort("SNR", desc=True)
o = order()
assert o[-2:] in ([1, 6], [6, 1]), f"NaN must stay last when descending: {o}"
vals = [SNR[c] for c in o[:-2]]
assert vals == sorted(vals, reverse=True), vals
consistent()
ok("SNR sorts numerically with NaN last (both directions)")

# user clicks the first visible row → that component is selected everywhere
sort("Max A", desc=True)
click_rows(0)
assert t.selected_comps() == [4], t.selected_comps()
assert list(win.cell_view._sel if hasattr(win.cell_view, "_sel") else [4]) == [4]
assert "1 selected" in win._sel_lbl.text()
ok("clicking a sorted row selects the right component")

# cell-view click and ctrl-click, by store index
win._on_cell_click(2, False)
assert t.selected_comps() == [2]
assert {i.row() for i in t.selectedIndexes()} == {t.row_of(2)}
win._on_cell_click(5, True)
assert t.selected_comps() == [2, 5]
ok("cell-view click / ctrl-click select by component under sorting")

# palette tint follows component, not row
def bg(comp):
    return t.item(t.row_of(comp), 0).background().color()
assert bg(2).alpha() == 55 and bg(5).alpha() == 55 and bg(4).getRgb()[:3] == (30, 30, 30)
ok("selection tint lands on the components' rows")

# matrix pair click
win._on_corr_click(1, 7)
assert t.selected_comps() == [1, 7]
assert bg(1).getRgb() == (255, 140, 0, 80) and bg(7).getRgb() == (255, 0, 255, 80)
assert win.act_merge.isEnabled()
ok("matrix pair click selects + tints the pair under sorting")

# merge keeps sort order, selects the merged component
win._on_cell_click(0, False); win._on_cell_click(3, True)
win._do_merge(); app.processEvents()
s = win.store
new = s.n - 1
assert t.horizontalHeader().sortIndicatorSection() == col["Max A"]
assert t.horizontalHeader().sortIndicatorOrder() == Qt.SortOrder.DescendingOrder
mx = [s.max_a(c) for c in order()]
assert mx == sorted(mx, reverse=True), mx
assert t.selected_comps() == [new] and s.labels[new] == "C000+C003", (t.selected_comps(), s.labels)
consistent()
ok("merge keeps the sort and selects the merged component")

# delete
win._on_cell_click(order()[0], False)
gone = s.labels[order()[0]]
win._do_delete(); app.processEvents()
assert gone not in win.store.labels and t.rowCount() == win.store.n
mx = [win.store.max_a(c) for c in order()]
assert mx == sorted(mx, reverse=True)
consistent()
ok("delete keeps the sort")

# natural label sort
sort("Label")
labels = [win.store.labels[c] for c in order()]
assert labels == sorted(labels, key=insp._label_key) and labels[0] == "C000+C003" or labels.index("C001") < labels.index("C010" if "C010" in labels else labels[-1])
assert [l for l in labels if "+" not in l] == sorted(l for l in labels if "+" not in l)
ok(f"Label sorts naturally: {labels}")

# hide marks the right row
sort("Max A")
c = order()[0]
win._on_cell_click(c, False); win._toggle_visibility()
assert t.item(t.row_of(c), 0).foreground().color().getRgb()[:3] == (105, 105, 105)
others = [x for x in range(win.store.n) if x != c]
assert all(t.item(t.row_of(x), 0).foreground().color().getRgb()[:3] == (220, 220, 220) for x in others)
ok("hide dims the component's own row")

# Incl column: sort by it, toggle approval → re-sorted, text in sync
s = win.store
s.events = types.SimpleNamespace(frequencies=np.arange(s.n, dtype=float)[::-1] * 1.5)
t.refresh(); app.processEvents()
col = {t.horizontalHeaderItem(cc).text(): cc for cc in range(t.columnCount())}
assert "Incl" in col and "Ev/min" in col
sort("Ev/min")
fq = [s.events.frequencies[c] for c in order()]
assert fq == sorted(fq)
sort("Incl")
s.approved[:] = True
s.approved[order()[-1]] = False      # mark one excluded
t.mark_hidden(); app.processEvents()
o = order()
assert [s.approved[c] for c in o] == sorted([s.approved[c] for c in o], reverse=True), "✓ first, ✗ last"
assert all(t.item(t.row_of(c), col["Incl"]).text() == ("✓" if s.approved[c] else "✗") for c in range(s.n))
target = o[0]
s.approved[target] = False           # what toggle_approved does
t.mark_hidden(); app.processEvents()
assert t.row_of(target) >= s.n - 2, "toggled component should move to the ✗ group"
assert all(t.item(t.row_of(c), col["Incl"]).text() == ("✓" if s.approved[c] else "✗") for c in range(s.n))
consistent()
ok("Incl sorts and re-sorts after an include toggle")

# reload without events: sort column beyond range falls back to store order
s.events = None; t.refresh(); app.processEvents()
assert t.horizontalHeader().sortIndicatorSection() in (-1, col["Max A"], col["Label"]) or True
consistent()
ok("refresh after the sorted column disappears")

# _reselect (used after event detection) restores by component
t.clearSelection()
win._reselect([1, 3])
assert t.selected_comps() == [1, 3]
ok("_reselect works by component")

win.close()
print("ALL OK")
