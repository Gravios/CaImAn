"""
utilities/session_name.py
=========================
Session directory name → YAML fields, shared by stack-sessions,
stack-sessions-gui (session_meta) and rename-sessions.  Standard library
only, so naming tools work without the CaImAn stack.
"""
from __future__ import annotations

import re
from datetime import datetime

# Session directory name pattern (same schema as utilities/rename_stem.py):
#   strohA-ia-000000-20150709-TL001_134044-25x-spont
#   strohA-ia-STO0647-20150113-TL001_122919-25x-spont
#   {prefix}-{subject_id}-{date8}-{trial}_{time6}-{mag}-{condition}
# Subject ids may be alphanumeric; trial and magnification numbers are taken
# from the digits in their tokens (TL001 → 1, 25x → 25).
_SESSION_RE = re.compile(
    r'^(?P<prefix>.+?)'            # strohA-ia
    r'-(?P<subject_id>[^-]+)'      # 000000 | STO0647
    r'-(?P<date>\d{8})'            # 20150709
    r'-(?P<trial>[^-_]+)'          # TL001
    r'_(?P<time>\d{6})'            # 134044
    r'-(?P<mag>[^-]+)'             # 25x
    r'-(?P<condition>[^/]+)$',     # spont
    re.IGNORECASE,
)
_DIGITS_RE = re.compile(r'\d+')

# Condition keywords found anywhere in session name after the magnification
_CONDITION_KEYWORDS = {
    "spont":  "spontaneous",
    "vstim":  "visual_stimulus",
    "vs":     "visual_stimulus",
    "astim":  "auditory_stimulus",
    "loco":   "locomotion",
    "dark":   "dark",
    "default": "default",
}


def parse_session_name(name: str) -> dict:
    """Extract structured metadata from a session directory name.

    Returns a flat dict with keys matching template_acquisition.yaml fields:
    ``subject_id``, ``experiment_datetime``, ``experiment_id``,
    ``condition``, ``magnification``.
    """
    out: dict = {}
    m = _SESSION_RE.match(name)
    if not m:
        return out

    # subject id: numeric ids as int (000082 → 82, as before), others verbatim
    sid = m.group("subject_id")
    out["subject_id"] = int(sid) if sid.isdigit() else sid

    # datetime
    try:
        dt = datetime.strptime(
            m.group("date") + m.group("time"), "%Y%m%d%H%M%S"
        )
        out["experiment_datetime"] = dt
    except ValueError:
        pass

    # trial number → experiment id (last digit run: TL001 → 1)
    digits = _DIGITS_RE.findall(m.group("trial"))
    if digits:
        out["experiment_id"] = int(digits[-1])

    # magnification (leading digit run: 25x → 25)
    mm = _DIGITS_RE.match(m.group("mag"))
    if mm:
        out["magnification"] = int(mm.group())

    # condition - normalise known keywords
    cond_raw = m.group("condition").lower()
    for kw, label in _CONDITION_KEYWORDS.items():
        if kw in cond_raw:
            out["condition"] = label
            break
    else:
        out["condition"] = m.group("condition")

    return out
