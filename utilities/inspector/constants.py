"""Shared constants: trace keys, history depth, overlay colours / opacity."""


HISTORY_MAXLEN = 5   # maximum undo / redo steps retained

TRACE_DENOISED = "C (denoised)"   # canonical trace key; this one is saved back
TRACE_DFF      = "ΔF/F"           # estimates.F_dff; input to event detection

# ── Colour constants ──────────────────────────────────────────────────────────

PALETTE_HEX = [
    "#FF4444", "#FF8C00", "#FFEE00", "#88FF00",
    "#FF00FF", "#FF66BB", "#AA44FF", "#AAAAFF",
    "#FF6644", "#CCFF00", "#FF44CC", "#FF9944",
]

CORR_A   = "#FF8C00"   # orange  – first of a clicked corr pair
CORR_B   = "#FF00FF"   # magenta – second
CYAN     = "#00FFFF"   # default (unselected) footprint colour
EXCLUDED = "#FF3B3B"   # footprints of components not approved (events mode)

# Footprint overlay opacity (0–1), read at draw time; set from the
# Appearance preferences (percent) by set_theme().
OVERLAY_ALPHA_DEFAULTS = {"selected": 0.55,   # selected / matrix-pair components
                          "all":      0.12,   # the all-ROIs layer
                          "excluded": 0.22}   # excluded: a little stronger, so red reads
OVERLAY_ALPHA = dict(OVERLAY_ALPHA_DEFAULTS)
CROSS_MIN_ARM = 25    # minimum cross arm half-length in pixels


def _hex_rgb(hex_c: str) -> tuple:
    h = hex_c.lstrip('#')
    return tuple(int(h[k:k + 2], 16) / 255.0 for k in (0, 2, 4))


__all__ = ['HISTORY_MAXLEN', 'TRACE_DENOISED', 'TRACE_DFF', 'PALETTE_HEX', 'CORR_A', 'CORR_B', 'CYAN', 'EXCLUDED', 'OVERLAY_ALPHA_DEFAULTS', 'OVERLAY_ALPHA', 'CROSS_MIN_ARM', '_hex_rgb']
