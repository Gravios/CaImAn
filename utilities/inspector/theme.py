"""Themes: UI / plot colour tables, colour maps, palette and font switching."""

import pyqtgraph as pg

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication

from .constants import OVERLAY_ALPHA, OVERLAY_ALPHA_DEFAULTS, PALETTE_HEX
from .prefs import _ALPHA_KEYS, appearance_prefs


# ── Colour maps ───────────────────────────────────────────────────────────────

def _cmap_rdbu_r():
    """Red-white-blue diverging, red = +1, blue = -1."""
    stops  = [0.0, 0.25, 0.5, 0.75, 1.0]
    colors = [(  5,  48,  97, 255),
              ( 67, 147, 195, 255),
              (247, 247, 247, 255),
              (214,  96,  77, 255),
              (103,   0,  31, 255)]
    return pg.ColorMap(pos=stops, color=colors)

def _cmap_viridis_r():
    """Reversed viridis: yellow (near) → purple (far)."""
    stops  = [0.0, 0.25, 0.5, 0.75, 1.0]
    colors = [(253, 231,  37, 255),
              ( 94, 201,  98, 255),
              ( 33, 144, 141, 255),
              ( 59,  82, 139, 255),
              ( 68,   1,  84, 255)]
    return pg.ColorMap(pos=stops, color=colors)

CMAP_RDBU_R    = _cmap_rdbu_r()
CMAP_VIRIDIS_R = _cmap_viridis_r()


# ── Dark palette (module-level so it can be called before QMainWindow) ────────

# ── Themes ────────────────────────────────────────────────────────────────────
# Two tables: UI (widgets, table cells, labels) and PT (plots).  They are
# chosen separately so a light interface can keep dark plots.  Views read
# UI / PT at draw time; InspectorWindow.apply_theme() re-styles live widgets.

PALETTE_DARK_BG = list(PALETTE_HEX)          # bright: for dark plot backgrounds
PALETTE_LIGHT_BG = [                          # saturated / darker: for white plots
    "#D62728", "#E67E00", "#B8A000", "#2CA02C",
    "#C000C0", "#E0559A", "#7B3FD0", "#4A5BD0",
    "#C8452A", "#6B9A00", "#C2187A", "#D9822B",
]

UI_THEMES = {
    "dark": dict(
        window=(28, 28, 28), window_text=(210, 210, 210), base=(18, 18, 18),
        alt_base=(32, 32, 32), text=(210, 210, 210), button=(45, 45, 45),
        button_text=(210, 210, 210), highlight=(55, 95, 155),
        highlight_text=(255, 255, 255), tooltip_base=(40, 40, 40),
        tooltip_text=(210, 210, 210), placeholder=(120, 120, 120),
        cell_bg=(30, 30, 30), cell_fg=(220, 220, 220), cell_dim=(105, 105, 105),
        cell_excl=(255, 110, 110),
        muted="#aaaaaa", hint="#888888", warn="#e0a030"),
    "light": dict(
        window=(239, 239, 239), window_text=(25, 25, 25), base=(255, 255, 255),
        alt_base=(246, 246, 246), text=(25, 25, 25), button=(225, 225, 225),
        button_text=(25, 25, 25), highlight=(48, 110, 200),
        highlight_text=(255, 255, 255), tooltip_base=(255, 255, 225),
        tooltip_text=(0, 0, 0), placeholder=(140, 140, 140),
        cell_bg=(255, 255, 255), cell_fg=(25, 25, 25), cell_dim=(160, 160, 160),
        cell_excl=(198, 40, 40),
        muted="#555555", hint="#777777", warn="#a05a00"),
}

PLOT_THEMES = {
    "dark": dict(
        bg="#111111", fg="#888888", title="#bbbbbb", placeholder="#555555",
        legend_text="#ffffff", legend_bg="#111111cc", legend_pen="#333333",
        dff="#666666", binary="#dddddd", binary_fill=(255, 255, 255, 70),
        filtered="#44dd66", threshold="#ccaa00", peak="#ff4444",
        peak_bad="#8a5a5a", fit="#4fc3f7",
        cursor="#ffd23f", cursor_hover="#ffe680", error="#ff4444",
        palette=PALETTE_DARK_BG),
    "light": dict(
        bg="#ffffff", fg="#444444", title="#333333", placeholder="#999999",
        legend_text="#000000", legend_bg="#ffffffcc", legend_pen="#bbbbbb",
        dff="#9a9a9a", binary="#333333", binary_fill=(0, 0, 0, 45),
        filtered="#1a8f3a", threshold="#a07800", peak="#d62728",
        peak_bad="#b59a9a", fit="#0277bd",
        cursor="#d08a00", cursor_hover="#f0a000", error="#c62828",
        palette=PALETTE_LIGHT_BG),
}

UI = dict(UI_THEMES["dark"])                 # active tables (mutated in place)
PT = dict(PLOT_THEMES["dark"])
_THEME_STATE = {"ui": "dark", "plot": "dark", "default_font_pt": None}


def _system_scheme() -> str:
    """'dark' or 'light' from the desktop (Qt colour scheme, else palette)."""
    app = QApplication.instance()
    try:
        cs = app.styleHints().colorScheme()
        if cs == Qt.ColorScheme.Dark:
            return "dark"
        if cs == Qt.ColorScheme.Light:
            return "light"
    except Exception:
        pass
    try:
        win = app.style().standardPalette().color(QPalette.ColorRole.Window)
        return "dark" if win.lightness() < 128 else "light"
    except Exception:
        return "dark"


def resolve_theme(appearance: dict) -> tuple:
    """(ui scheme, plot scheme), each 'dark' or 'light'."""
    ui = appearance.get("theme") or "dark"
    if ui == "system" or ui not in UI_THEMES:
        ui = _system_scheme() if ui == "system" else "dark"
    plot = appearance.get("plot_background") or "auto"
    if plot not in PLOT_THEMES:
        plot = ui
    return ui, plot


def set_theme(appearance: dict = None) -> tuple:
    """Activate a theme: app palette + font, UI / PT tables and the
    selection palette.  Widgets already built are re-styled by
    InspectorWindow.apply_theme()."""
    app = QApplication.instance()
    appearance = appearance_prefs() if appearance is None else appearance
    ui, plot = resolve_theme(appearance)
    UI.clear(); UI.update(UI_THEMES[ui])
    PT.clear(); PT.update(PLOT_THEMES[plot])
    PALETTE_HEX[:] = PT["palette"]
    for key, slot in _ALPHA_KEYS.items():          # percent → 0–1, clamped
        try:
            v = float(appearance.get(key))
        except (TypeError, ValueError):
            v = OVERLAY_ALPHA_DEFAULTS[slot] * 100
        OVERLAY_ALPHA[slot] = min(max(v, 0.0), 100.0) / 100.0
    _THEME_STATE["ui"], _THEME_STATE["plot"] = ui, plot
    if app is None:
        return ui, plot
    app.setStyle("Fusion")
    pal = QPalette()
    CR = QPalette.ColorRole
    for role, key in ((CR.Window, "window"), (CR.WindowText, "window_text"),
                      (CR.Base, "base"), (CR.AlternateBase, "alt_base"),
                      (CR.Text, "text"), (CR.Button, "button"),
                      (CR.ButtonText, "button_text"), (CR.Highlight, "highlight"),
                      (CR.HighlightedText, "highlight_text"),
                      (CR.ToolTipBase, "tooltip_base"),
                      (CR.ToolTipText, "tooltip_text"),
                      (CR.PlaceholderText, "placeholder")):
        pal.setColor(role, QColor(*UI[key]))
    dis = QColor(*UI["placeholder"])
    for role in (CR.WindowText, CR.Text, CR.ButtonText):
        pal.setColor(QPalette.ColorGroup.Disabled, role, dis)
    app.setPalette(pal)
    if _THEME_STATE["default_font_pt"] is None:
        _THEME_STATE["default_font_pt"] = app.font().pointSizeF()
    f = app.font()
    size = appearance.get("font_size")
    f.setPointSizeF(float(size) if size else _THEME_STATE["default_font_pt"])
    app.setFont(f)
    return ui, plot


def _apply_dark_palette():
    """Kept for callers outside this module; applies the saved appearance."""
    set_theme()


def _default_font_pt() -> int:
    pt = _THEME_STATE.get("default_font_pt")
    if pt is None and QApplication.instance() is not None:
        pt = QApplication.instance().font().pointSizeF()
    return int(round(pt)) if pt else 10


def _ui_color(key: str) -> QColor:
    return QColor(*UI[key])


__all__ = ['_cmap_rdbu_r', '_cmap_viridis_r', 'CMAP_RDBU_R', 'CMAP_VIRIDIS_R', 'PALETTE_DARK_BG', 'PALETTE_LIGHT_BG', 'UI_THEMES', 'PLOT_THEMES', 'UI', 'PT', '_THEME_STATE', '_system_scheme', 'resolve_theme', 'set_theme', '_apply_dark_palette', '_default_font_pt', '_ui_color']
