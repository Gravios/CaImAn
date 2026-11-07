"""The sortable component table."""

import re

from PyQt6.QtCore import pyqtSignal, QItemSelectionModel, Qt
from PyQt6.QtGui import QBrush, QColor, QFont
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QMenu,
    QTableWidget,
    QTableWidgetItem,
)

from .constants import PALETTE_HEX
from .theme import _ui_color
from .deps import _rf
from .store import ComponentStore


# ── Component table ───────────────────────────────────────────────────────────

_COMP_ROLE = Qt.ItemDataRole.UserRole        # component index of the row
_SORT_ROLE = Qt.ItemDataRole.UserRole + 1    # typed sort key
_MISS_ROLE = Qt.ItemDataRole.UserRole + 2    # True for NaN / missing values


def _num_key(v) -> tuple:
    """Numeric sort key; NaN / None compare after every number."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return (1, 0.0)
    return (1, 0.0) if f != f else (0, f)


def _is_missing(key) -> bool:
    return isinstance(key, tuple) and len(key) == 2 and key[0] == 1 \
        and isinstance(key[1], float)


def _label_key(label: str) -> tuple:
    """Natural order: C002 < C010 < C002+C010."""
    return tuple((0, int(t), "") if t.isdigit() else (1, 0, t)
                 for t in re.split(r"(\d+)", str(label)) if t)


class _SortItem(QTableWidgetItem):
    """Sorts by _SORT_ROLE (numbers numerically) instead of display text.
    Missing values (NaN) stay at the bottom in both sort directions."""

    def __lt__(self, other):
        a, b = self.data(_SORT_ROLE), other.data(_SORT_ROLE)
        if a is None or b is None:
            return super().__lt__(other)
        ma, mb = bool(self.data(_MISS_ROLE)), bool(other.data(_MISS_ROLE))
        if ma != mb:
            tw = self.tableWidget()
            desc = tw is not None and tw.horizontalHeader().sortIndicatorOrder() \
                == Qt.SortOrder.DescendingOrder
            # Qt reverses the comparison for descending order; pre-flip so
            # the missing value still lands last.
            return ma if desc else mb
        return a < b


_QC_SHORT = {"empty": "empty", "area_small": "area<", "area_large": "area>",
             "solidity": "solid", "eccentricity": "ecc", "edge": "edge",
             "caiman_quality": "qual", "inactive": "inact"}


def _qc_cell(v, kind: str) -> tuple:
    """(text, sort key[, tooltip]) for one roi_filter value."""
    if kind == "bool":
        return ("✓" if v else "✗", (0 if v else 1,))
    if kind == "reasons":
        names = _rf().decode_reasons(int(v))
        return (",".join(_QC_SHORT.get(n, n) for n in names) or "—",
                (len(names), int(v)),
                ", ".join(names) or "passes every criterion")
    try:
        f = float(v)
    except (TypeError, ValueError):
        f = float("nan")
    if f != f:
        return ("—", _num_key(f))
    text = {"int": f"{int(f)}", "f0": f"{f:.0f}", "f2": f"{f:.2f}",
            "pct": f"{100 * f:.1f}"}.get(kind, f"{f:.3f}")
    return (text, _num_key(f))


class ComponentTable(QTableWidget):
    """
    Left panel — one row per component with quality metrics.
    Selected rows are lightly tinted with their palette colour.
    A corr-pair click tints those two rows in orange / magenta.

    Click a column header to sort (again to reverse).  Sorting changes
    only the view: callers address components by store index through
    :meth:`selected_comps`, :meth:`select_comps`, :meth:`row_of` and
    :meth:`comp_of`, never by table row.  The sort order survives
    refreshes (merge, delete, event detection).
    """

    _FIXED    = ("Label", "Max A", "Peak ΔF")
    _OPTIONAL = (("SNR",   "SNR_comp"),
                 ("r_val", "r_values"),
                 ("CNN",   "cnn_preds"))

    # roi_filter fields, each shown only when present in the results file:
    # (header, field, display kind, header tooltip)
    _QC_COLS = (
        ("QC",      "passed",       "bool",
         "ROI filter verdict: footprint ∧ edge ∧ CaImAn quality (∧ active if required)"),
        ("Active",  "active",       "bool",
         "ROI filter activity: event count / rate / peak / skew thresholds met"),
        ("Fail",    "fail_reasons", "reasons",
         "Failed ROI-filter criteria (hover a cell for the full names)"),
        ("Area",    "area_px",      "int",
         "Footprint area (px) at footprint_thr × peak"),
        ("Solid",   "solidity",     "f2",
         "Solidity = area / convex-hull area (low = ragged or ring-shaped)"),
        ("Ecc",     "eccentricity", "f2",
         "Eccentricity of the footprint (0 = circle, 1 = line)"),
        ("Edge",    "edge_dist_px", "f0",
         "Centroid distance to the FOV border (px)"),
        ("Rate",    "event_rate",   "f2",
         "ROI filter: events per minute on ΔF/F (no SNR gating)"),
        ("Pk ΔF/F", "peak_dff",     "f2",
         "Peak of the smoothed ΔF/F"),
        ("Skew",    "skewness",     "f2",
         "Skewness of ΔF/F"),
        ("Act %",   "active_frac",  "pct",
         "Percent of frames inside detected events"),
    )

    columns_changed = pyqtSignal()           # a column was shown / hidden
    _UNHIDEABLE = ("Label",)

    def __init__(self, store: ComponentStore, parent=None):
        super().__init__(parent)
        self.store = store
        self._hidden_cols: set = set()       # header names; kept across files
        self.setSelectionMode(
            QAbstractItemView.SelectionMode.MultiSelection)
        self.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setAlternatingRowColors(False)
        self.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self.verticalHeader().setDefaultSectionSize(20)
        f = QFont()
        f.setPointSize(8)
        self.setFont(f)
        self._row_of: dict[int, int] = {}
        hdr = self.horizontalHeader()
        hdr.setSortIndicatorShown(True)
        hdr.setSortIndicator(-1, Qt.SortOrder.AscendingOrder)   # start in store order
        hdr.setToolTip("Click a column to sort; click again to reverse; "
                       "right-click to show / hide columns")
        hdr.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        hdr.customContextMenuRequested.connect(
            lambda pos: self.column_menu(self).exec(hdr.mapToGlobal(pos)))
        self.model().layoutChanged.connect(self._reindex)
        self._populate()
        self.setSortingEnabled(True)

    # ── Public ────────────────────────────────────────────────────────────────

    def refresh(self):
        self._populate()

    # ── Column visibility ─────────────────────────────────────────────────────

    def column_names(self) -> list:
        return [self.horizontalHeaderItem(c).text() for c in range(self.columnCount())]

    def hidden_columns(self) -> list:
        """Hidden column names, including ones this file does not have."""
        return sorted(self._hidden_cols)

    def set_hidden_columns(self, names) -> None:
        self._hidden_cols = set(names or ()) - set(self._UNHIDEABLE)
        self._apply_hidden()

    def set_column_visible(self, name: str, visible: bool) -> None:
        if name in self._UNHIDEABLE:
            return
        (self._hidden_cols.discard if visible else self._hidden_cols.add)(name)
        self._apply_hidden()
        self.columns_changed.emit()

    def show_all_columns(self) -> None:
        self._hidden_cols.clear()
        self._apply_hidden()
        self.columns_changed.emit()

    def _apply_hidden(self) -> None:
        for c, name in enumerate(self.column_names()):
            self.setColumnHidden(c, name in self._hidden_cols)

    def fill_column_menu(self, menu: 'QMenu') -> None:
        """One checkable entry per column of this file, then Show all."""
        menu.clear()
        for c, name in enumerate(self.column_names()):
            a = menu.addAction(name)
            a.setCheckable(True)
            a.setChecked(name not in self._hidden_cols)
            a.setEnabled(name not in self._UNHIDEABLE)
            tip = self.horizontalHeaderItem(c).toolTip()
            if tip:
                a.setToolTip(tip)
            a.toggled.connect(lambda on, n=name: self.set_column_visible(n, on))
        menu.addSeparator()
        a = menu.addAction("Show all columns")
        a.setEnabled(bool(self._hidden_cols & set(self.column_names())))
        a.triggered.connect(self.show_all_columns)
        menu.setToolTipsVisible(True)

    def column_menu(self, parent) -> 'QMenu':
        m = QMenu("Columns", parent)
        self.fill_column_menu(m)
        return m

    def sort_state(self):
        """(column name, "asc"/"desc") or None when unsorted."""
        hdr = self.horizontalHeader()
        c = hdr.sortIndicatorSection()
        if not (0 <= c < self.columnCount()):
            return None
        return (self.horizontalHeaderItem(c).text(),
                "desc" if hdr.sortIndicatorOrder() == Qt.SortOrder.DescendingOrder else "asc")

    def sort_by_name(self, name: str, order: str = "asc") -> bool:
        names = self.column_names()
        if name not in names:
            return False
        self.sortByColumn(names.index(name), Qt.SortOrder.DescendingOrder
                          if order == "desc" else Qt.SortOrder.AscendingOrder)
        return True

    def comp_of(self, row: int) -> int:
        """Store index of the component shown in table *row*."""
        it = self.item(row, 0)
        return int(it.data(_COMP_ROLE)) if it is not None else row

    def row_of(self, comp: int) -> int:
        """Table row currently showing store component *comp* (-1 if none)."""
        return self._row_of.get(comp, -1)

    def selected_comps(self) -> list:
        """Selected components as sorted store indices."""
        return sorted({self.comp_of(idx.row()) for idx in self.selectedIndexes()})

    def select_comps(self, comps, clear: bool = True, scroll_to=None) -> None:
        """Select store components *comps* without emitting
        itemSelectionChanged (callers propagate via _on_table_sel)."""
        sm    = self.selectionModel()
        flags = (QItemSelectionModel.SelectionFlag.Select
                 | QItemSelectionModel.SelectionFlag.Rows)
        self.blockSignals(True)
        if clear:
            sm.clearSelection()
        for c in comps:
            r = self.row_of(c)
            if r >= 0:
                sm.select(self.model().index(r, 0), flags)
        self.blockSignals(False)
        if scroll_to is not None and self.row_of(scroll_to) >= 0:
            self.scrollToItem(self.item(self.row_of(scroll_to), 0),
                              QAbstractItemView.ScrollHint.EnsureVisible)

    def tint_selection(self, sel: list):
        """Palette tint on selected components; dark reset on others.
        *sel* holds store indices; the palette slot follows their order.
        Note: no blockSignals here — setBackground emits itemChanged,
        which is not connected to any slot, so it is safe to let it fire.
        Blocking table signals here was causing itemSelectionChanged to be
        re-emitted after blockSignals(False) in PyQt6, overwriting _sel.
        """
        dark = _ui_color("cell_bg")
        slot = {c: k for k, c in enumerate(sel)}
        for comp in range(self.store.n):
            if comp in slot:
                bg = QColor(PALETTE_HEX[slot[comp] % len(PALETTE_HEX)])
                bg.setAlpha(55)
            else:
                bg = dark
            self._set_row_bg(self.row_of(comp), bg)

    def tint_pair(self, i: int, j: int):
        """Orange / magenta tint for a corr-clicked pair (store indices).
        Note: no blockSignals — see tint_selection for rationale.
        """
        dark = _ui_color("cell_bg")
        for comp in range(self.store.n):
            if   comp == i: bg = QColor(255, 140,   0, 80)
            elif comp == j: bg = QColor(255,   0, 255, 80)
            else:           bg = dark
            self._set_row_bg(self.row_of(comp), bg)

    # ── Private ───────────────────────────────────────────────────────────────

    # A blinded rater must not see the algorithmic verdicts.
    blinded = False

    def _show_eval(self) -> bool:
        return self.store.eval_known and not self.blinded

    def _show_qc(self) -> bool:
        return self.store.qc is not None and not self.blinded

    def _col_names(self):
        cols = list(self._FIXED)
        for label, attr in self._OPTIONAL:
            if getattr(self.store, attr, None) is not None:
                cols.append(label)
        if self._show_eval():
            cols.append("Eval")
        qc = self.store.qc
        if self._show_qc():
            cols += [h for h, fld, _k, _t in self._QC_COLS if fld in qc]
        if self.store.events is not None:
            cols.append("Ev/min")
        if self.store.has_approval:
            cols.append("Incl")
        return cols

    def _populate(self):
        s    = self.store
        cols = self._col_names()
        hdr  = self.horizontalHeader()
        sort_col, sort_ord = hdr.sortIndicatorSection(), hdr.sortIndicatorOrder()
        # Remember the sorted column by name: the column set differs between
        # files (optional metrics, ROI-filter fields, events).
        sort_name = (self.horizontalHeaderItem(sort_col).text()
                     if 0 <= sort_col < self.columnCount() else None)
        sorting = self.isSortingEnabled()
        self.setSortingEnabled(False)          # rows must land at row == comp
        self.blockSignals(True)
        self.clear()
        self.setColumnCount(len(cols))
        self.setHorizontalHeaderLabels(cols)
        tips = {h: t for h, _f, _k, t in self._QC_COLS}
        tips["Incl"] = ("Include flag (I toggles). Starts from the ROI filter's "
                        "QC verdict, or from event auto-approval")
        tips["Eval"] = ("CaImAn evaluation: ✓ accepted (idx_components; census: "
                        "active), ✗ rejected (census: inactive). Rejected rows are "
                        "kept and saved; View ▸ Hide rejected / inactive ROIs")
        for c, h in enumerate(cols):
            if h in tips:
                self.horizontalHeaderItem(c).setToolTip(tips[h])
        self.setRowCount(s.n)
        dark = _ui_color("cell_bg")
        for i in range(s.n):
            vals = [(s.labels[i], _label_key(s.labels[i])),
                    (f"{s.max_a(i):.3f}", _num_key(s.max_a(i))),
                    (f"{s.peak_df(i):.1f}", _num_key(s.peak_df(i)))]
            for _, attr in self._OPTIONAL:
                if getattr(s, attr, None) is not None:
                    vals.append(_qc_cell(getattr(s, attr)[i], "f3"))
            if self._show_eval():
                vals.append(_qc_cell(s.accepted[i], "bool"))
            if self._show_qc():
                for _h, fld, kind, _t in self._QC_COLS:
                    if fld in s.qc:
                        vals.append(_qc_cell(s.qc[fld][i], kind))
            if s.events is not None:
                fq = s.events.frequencies[i]
                vals.append((f"{fq:.2f}", _num_key(fq)))
            if s.has_approval:
                vals.append(("✓" if s.approved[i] else "✗",
                             (0 if s.approved[i] else 1,)))
            for c, (text, key, *tip) in enumerate(vals):
                item = _SortItem(text)
                if tip:
                    item.setToolTip(tip[0])
                item.setData(_COMP_ROLE, i)
                item.setData(_SORT_ROLE, key)
                item.setData(_MISS_ROLE, _is_missing(key))
                item.setBackground(QBrush(dark))
                self.setItem(i, c, item)
        self.blockSignals(False)
        self._reindex()
        sort_col = cols.index(sort_name) if sort_name in cols else -1
        hdr.setSortIndicator(sort_col, sort_ord)
        self.setSortingEnabled(sorting)        # re-applies the user's sort
        self._reindex()
        self._apply_hidden()
        self.mark_hidden()

    def _reindex(self):
        """Rebuild component → row after populate or a sort."""
        self._row_of = {}
        for r in range(self.rowCount()):
            it = self.item(r, 0)
            if it is not None and it.data(_COMP_ROLE) is not None:
                self._row_of[int(it.data(_COMP_ROLE))] = r

    def mark_hidden(self):
        """Dim hidden rows; tint excluded rows red (once events exist) and
        keep the Incl column (text and sort key) in sync with the store."""
        grey   = _ui_color("cell_dim")
        normal = _ui_color("cell_fg")
        excl   = _ui_color("cell_excl")
        s      = self.store
        has_ev = s.has_approval and self.columnCount() and \
            self.horizontalHeaderItem(self.columnCount() - 1).text() == "Incl"
        # Changing the Incl key while sorted by it would re-sort mid-loop and
        # invalidate row_of; suspend sorting and re-apply it once at the end.
        sorting = self.isSortingEnabled()
        self.setSortingEnabled(False)
        for comp in range(s.n):
            row = self.row_of(comp)
            if row < 0:
                continue
            self.setRowHidden(row, bool(s.hide_rejected and not s.accepted[comp]))
            if s.is_hidden(comp):
                fg = grey
            elif has_ev and not s.approved[comp]:
                fg = excl
            else:
                fg = normal
            if has_ev:
                it = self.item(row, self.columnCount() - 1)
                if it:
                    it.setText("✓" if s.approved[comp] else "✗")
                    it.setData(_SORT_ROLE, (0 if s.approved[comp] else 1,))
            for col in range(self.columnCount()):
                it = self.item(row, col)
                if it:
                    it.setForeground(QBrush(fg))
        self.setSortingEnabled(sorting)
        self._reindex()

    def _set_row_bg(self, row: int, color: QColor):
        if row < 0:
            return
        for col in range(self.columnCount()):
            item = self.item(row, col)
            if item:
                item.setBackground(QBrush(color))


__all__ = ['_COMP_ROLE', '_SORT_ROLE', '_MISS_ROLE', '_num_key', '_is_missing', '_label_key', '_SortItem', '_QC_SHORT', '_qc_cell', 'ComponentTable']
