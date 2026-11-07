"""Preferences file ($CAIMAN_INSPECTOR_PREFS or ~/.config/caiman/inspector_prefs.json)
and the field tables the preference editor is built from."""

import json
import os
import sys

from .deps import _ek, _rf


# ── Preferences (JSON, per user) ─────────────────────────────────────────────

def prefs_path() -> str:
    """``$CAIMAN_INSPECTOR_PREFS``, else ``$XDG_CONFIG_HOME/caiman/
    inspector_prefs.json`` (``~/.config`` when XDG is unset)."""
    p = os.environ.get("CAIMAN_INSPECTOR_PREFS")
    if p:
        return p
    base = (os.environ.get("XDG_CONFIG_HOME")
            or os.path.join(os.path.expanduser("~"), ".config"))
    return os.path.join(base, "caiman", "inspector_prefs.json")


def load_prefs() -> dict:
    try:
        with open(prefs_path()) as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as exc:
        print(f"[prefs] ignoring unreadable {prefs_path()}: {exc}", file=sys.stderr)
        return {}


def save_prefs(prefs: dict) -> str:
    path = prefs_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(prefs, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)
    return path


def roi_filter_prefs(prefs: dict = None) -> dict:
    """roi_filter DEFAULTS (+ gSig override) overlaid with the saved section;
    unknown keys are dropped."""
    d = dict(_rf().DEFAULTS)
    d["gSig"] = None
    sec = (load_prefs() if prefs is None else prefs).get("roi_filter", {})
    for k, v in (sec if isinstance(sec, dict) else {}).items():
        if k in d:
            d[k] = v
    return d


# Field specs for the preference editor, one list per section:
#   (group, key, label, kind, (min, max, step, decimals), null_text, tooltip)
# kind: "float" | "int" | "bool" | "tristate";  null_text=None → not nullable,
# else the checkbox-off meaning ("off", "from run", …).
_RF_PREF_FIELDS = (
    ("Footprint", "footprint_thr", "Mask threshold (× peak)", "float", (0.01, 1.0, 0.05, 2), None,
     "Footprint mask = connected pixels ≥ this fraction of the footprint's peak."),
    ("Footprint", "gSig", "Expected cell radius gSig (px)", "float", (0.5, 200.0, 0.5, 1), "from run",
     "Sets the expected area π·gSig² for the area fractions.  Off: the run's init.gSig."),
    ("Footprint", "min_area_frac", "Min area (× π·gSig²)", "float", (0.0, 100.0, 0.1, 2), "off",
     "Fail footprints smaller than this multiple of the expected area."),
    ("Footprint", "max_area_frac", "Max area (× π·gSig²)", "float", (0.0, 1000.0, 0.5, 1), "off",
     "Fail footprints larger than this multiple of the expected area."),
    ("Footprint", "min_area_px", "Min area (px)", "int", (0, 10_000_000, 1, 0), "off",
     "Absolute limit; overrides the fraction."),
    ("Footprint", "max_area_px", "Max area (px)", "int", (0, 10_000_000, 10, 0), "off",
     "Absolute limit; overrides the fraction."),
    ("Footprint", "min_solidity", "Min solidity", "float", (0.0, 1.0, 0.05, 2), "off",
     "Area / convex-hull area; low for ragged or ring-shaped footprints."),
    ("Footprint", "max_eccentricity", "Max eccentricity", "float", (0.0, 1.0, 0.01, 2), "off",
     "0 = circle, → 1 = line (dendrites, scan artefacts)."),
    ("Edge", "edge_px", "Min distance to border (px)", "float", (0.0, 10_000.0, 1.0, 0), "off",
     "Fail ROIs whose centroid is closer than this to the FOV border."),
    ("CaImAn quality", "use_caiman_quality", "Apply CaImAn quality rule", "bool", None, None,
     "Pass any upper threshold and no *_lowest floor (CaImAn's evaluate_components)."),
    ("CaImAn quality", "min_SNR", "min_SNR", "float", (0.0, 1000.0, 0.1, 2), "from run", ""),
    ("CaImAn quality", "SNR_lowest", "SNR_lowest", "float", (0.0, 1000.0, 0.1, 2), "from run", ""),
    ("CaImAn quality", "rval_thr", "rval_thr", "float", (-1.0, 1.0, 0.05, 2), "from run", ""),
    ("CaImAn quality", "rval_lowest", "rval_lowest", "float", (-1.0, 1.0, 0.05, 2), "from run", ""),
    ("CaImAn quality", "use_cnn", "use_cnn", "tristate", None, "from run", ""),
    ("CaImAn quality", "min_cnn_thr", "min_cnn_thr", "float", (0.0, 1.0, 0.05, 2), "from run", ""),
    ("CaImAn quality", "cnn_lowest", "cnn_lowest", "float", (0.0, 1.0, 0.05, 2), "from run", ""),
    ("Activity", "event_threshold", "Event cutoff (ΔF/F)", "float", (0.001, 10.0, 0.05, 3), None,
     "SLab detection cap: threshold = min(10·noise, this)."),
    ("Activity", "lowpass_hz", "Low-pass cutoff (Hz)", "float", (0.1, 1000.0, 1.0, 1), "SLab default",
     "Off: SLab's normalised cutoff."),
    ("Activity", "min_events", "Min events", "int", (0, 1_000_000, 1, 0), "off",
     "Active ⇔ every set minimum is met."),
    ("Activity", "min_event_rate", "Min event rate (/min)", "float", (0.0, 1000.0, 0.1, 2), "off", ""),
    ("Activity", "min_peak_dff", "Min peak ΔF/F", "float", (0.0, 100.0, 0.05, 2), "off", ""),
    ("Activity", "min_skewness", "Min skewness", "float", (-100.0, 100.0, 0.1, 2), "off", ""),
    ("Verdict", "require_activity", "Inactive ROIs fail QC", "bool", None, None,
     "Off: activity is reported separately and silent cells can pass."),
)

_KIN_PREF_FIELDS = (
    ("Expected transient", "decay_time", "Decay time τ (s)", "float", (0.01, 10.0, 0.05, 3),
     "from events",
     "Indicator decay time of the template.  Off: the decay time the events "
     "were detected with (the file's params)."),
    ("Criteria (an event is valid when all set ones hold)", "min_shape_r",
     "Min shape r", "float", (-1.0, 1.0, 0.05, 2), "off",
     "Pearson r of raw ΔF/F around the event with the expected fast-rise / "
     "exponential-decay template."),
    ("Criteria (an event is valid when all set ones hold)", "tau_min_frac",
     "Min fitted τ (× expected)", "float", (0.0, 100.0, 0.1, 2), "off",
     "Best-fitting decay on the raw trace, relative to the expected τ."),
    ("Criteria (an event is valid when all set ones hold)", "tau_max_frac",
     "Max fitted τ (× expected)", "float", (0.0, 100.0, 0.1, 2), "off",
     "Slow drifts and neuropil decay far slower than the indicator."),
    ("Criteria (an event is valid when all set ones hold)", "max_rise_s",
     "Max rise 20→100 % (s)", "float", (0.0, 10.0, 0.01, 3), "off",
     "Raw rise time from 20 % of the peak above the preceding trough."),
    ("Criteria (an event is valid when all set ones hold)", "min_amp_snr",
     "Min amplitude / noise", "float", (0.0, 1000.0, 0.5, 2), "off",
     "Fitted amplitude over 1.4826·MAD of the raw ΔF/F outside events."),
)


def kinetics_prefs(prefs: dict = None) -> dict:
    """event_kinetics DEFAULTS overlaid with the saved section."""
    d = dict(_ek().DEFAULTS)
    sec = (load_prefs() if prefs is None else prefs).get("kinetics", {})
    for k, v in (sec if isinstance(sec, dict) else {}).items():
        if k in d:
            d[k] = v
    return d


_APPEARANCE_PREF_FIELDS = (
    ("Theme", "theme", "Theme", "choice",
     [("system", "System (follow the desktop)"), ("dark", "Dark"), ("light", "Light")], None,
     "Colours of windows, menus, the table and dialogs."),
    ("Theme", "plot_background", "Plot background", "choice",
     [("auto", "Follow theme"), ("dark", "Dark"), ("light", "Light")], None,
     "Cell viewer, traces, events and matrices.  Dark plots keep the bright "
     "selection colours; light plots use a darker palette."),
    ("Theme", "font_size", "Font size (pt)", "int", (6, 32, 1, 0), "default",
     "Base font size of the whole inspector."),
    ("ROI overlay", "alpha_selected", "Selected ROIs (%)", "int", (0, 100, 5, 0), None,
     "Opacity of selected and matrix-pair footprints in the cell viewer."),
    ("ROI overlay", "alpha_all", "All other ROIs (%)", "int", (0, 100, 2, 0), None,
     "Opacity of the all-ROIs layer (toolbar: All ROIs)."),
    ("ROI overlay", "alpha_excluded", "Excluded ROIs (%)", "int", (0, 100, 2, 0), None,
     "Opacity of excluded (red) footprints in the all-ROIs layer."),
)

# section key → (tab title, field specs, defaults loader)
PREF_SECTIONS = {
    "appearance": ("Appearance", _APPEARANCE_PREF_FIELDS,
                   lambda: dict(APPEARANCE_DEFAULTS)),
    "roi_filter": ("ROI filter", _RF_PREF_FIELDS, lambda: roi_filter_prefs({})),
    "kinetics":   ("Event kinetics", _KIN_PREF_FIELDS, lambda: kinetics_prefs({})),
}

APPEARANCE_DEFAULTS = {"theme": "dark", "plot_background": "auto", "font_size": None,
                       "alpha_selected": 55, "alpha_all": 12, "alpha_excluded": 22}
_ALPHA_KEYS = {"alpha_selected": "selected", "alpha_all": "all",
               "alpha_excluded": "excluded"}


def appearance_prefs(prefs: dict = None) -> dict:
    d = dict(APPEARANCE_DEFAULTS)
    sec = (load_prefs() if prefs is None else prefs).get("appearance", {})
    for k, v in (sec if isinstance(sec, dict) else {}).items():
        if k in d:
            d[k] = v
    return d


__all__ = ['prefs_path', 'load_prefs', 'save_prefs', 'roi_filter_prefs', '_RF_PREF_FIELDS',
           '_KIN_PREF_FIELDS', 'kinetics_prefs', '_APPEARANCE_PREF_FIELDS', 'PREF_SECTIONS', 'APPEARANCE_DEFAULTS', '_ALPHA_KEYS', 'appearance_prefs']
