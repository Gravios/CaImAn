"""
utilities/session_meta.py
=========================
Qt-free model behind ``stack-sessions-gui``: discover stack-sessions session
directories, edit their Trial YAML, and rename their naming elements.

Session naming follows :mod:`utilities.rename_stem`::

  <sourceId>-<userId>-<subjectId>-<date>-<trialId>_<time>-<magnification>-<experimentId>
  strohA-ia-000000-20150709-TL001_134044-25x-spont

YAML values are parsed with a small, predictable scalar grammar (see
:func:`coerce`) rather than full YAML 1.1, so ``000070`` stays the string
``"000070"`` instead of becoming octal 56, and ``no`` stays ``"no"``.
"""
from __future__ import annotations

import copy
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

from utilities.rename_stem import FIELDS, build_stem, parse_stem

# ── Naming elements ───────────────────────────────────────────────────────────

FIELD_LABELS = {
    "sourceId":      "Source / lab",
    "userId":        "Experimenter",
    "subjectId":     "Subject",
    "date":          "Date (YYYYMMDD)",
    "trialId":       "Trial",
    "time":          "Time (HHMMSS)",
    "magnification": "Magnification",
    "experimentId":  "Experiment / condition",
}

# Per-field token rules: what may appear so that build_stem → parse_stem
# round-trips to the same parts.
FIELD_PATTERNS = {
    "sourceId":      r"[^-_\s/]+",
    "userId":        r"[^-_\s/]+",
    "subjectId":     r"[^-_\s/]+",
    "date":          r"\d{8}",
    "trialId":       r"[^-_\s/]+",
    "time":          r"\d{6}",
    "magnification": r"[^-_\s/]+",
    "experimentId":  r"[^\s/]+",
}


def validate_parts(parts: dict) -> dict[str, str]:
    """Return ``{field: error}`` for invalid naming elements (empty if valid)."""
    errors: dict[str, str] = {}
    for f in FIELDS:
        v = parts.get(f) or ""
        if not re.fullmatch(FIELD_PATTERNS[f], v):
            errors[f] = "required" if not v else "invalid characters"
    if "date" not in errors:
        try:
            datetime.strptime(parts["date"], "%Y%m%d")
        except ValueError:
            errors["date"] = "not a calendar date"
    if "time" not in errors:
        try:
            datetime.strptime(parts["time"], "%H%M%S")
        except ValueError:
            errors["time"] = "not a clock time"
    if not errors and parse_stem(build_stem(parts)) != {f: parts[f] for f in FIELDS}:
        errors["experimentId"] = "name does not round-trip"
    return errors


# ── Session discovery ─────────────────────────────────────────────────────────

_CH_SUBDIR_RE = re.compile(r"-C(\d{2})-fc\d+$")


@dataclass
class Session:
    path: Path
    parts: dict
    stacked: list[str] = field(default_factory=list)   # channel ids with a finished stack
    partial: bool = False                              # a *.partial.tif is present
    raw_frames: int = 0

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def yaml_path(self) -> Path:
        return self.path / f"{self.name}.yaml"


def scan_session(path: Path) -> Session | None:
    parts = parse_stem(path.name)
    if parts is None:
        return None
    s = Session(path=path, parts=parts)
    try:
        entries = list(os.scandir(path))
    except OSError:
        return s
    for e in entries:
        if e.is_dir():
            m = _CH_SUBDIR_RE.search(e.name)
            if m and e.name.startswith(path.name):
                d = Path(e.path)
                if (d / f"{e.name}.tif").exists():
                    s.stacked.append(m.group(1))
                if any(d.glob("*.partial.tif")):
                    s.partial = True
        elif "_C" in e.name and "_t" in e.name and e.name.endswith(".tif"):
            s.raw_frames += 1
    s.stacked.sort()
    return s


def discover_sessions(parent: Path, prefix: str | None = None) -> list[Session]:
    """All immediate subdirectories of *parent* whose name matches the stem schema."""
    out = []
    for d in sorted(Path(parent).iterdir()):
        if not d.is_dir() or (prefix and not d.name.startswith(prefix)):
            continue
        s = scan_session(d)
        if s is not None:
            out.append(s)
    return out


# ── YAML I/O ──────────────────────────────────────────────────────────────────

def template_path() -> Path:
    return Path(__file__).resolve().parent / "pipelines" / "template_acquisition.yaml"


def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    doc = yaml.safe_load(path.read_text())
    if doc is None:
        return {}
    if not isinstance(doc, dict):
        raise ValueError(f"{path.name}: top level is not a mapping")
    return doc


def save_yaml(path: Path, doc: dict) -> None:
    """Write *doc* atomically, keeping the previous version as ``<name>.yaml.bak``."""
    text = yaml.dump(doc, default_flow_style=False, sort_keys=False, allow_unicode=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    if path.exists():
        bak = path.with_name(path.name + ".bak")
        bak.write_bytes(path.read_bytes())
    tmp.replace(path)


# ── Value text ↔ Python ───────────────────────────────────────────────────────

_NULL_WORDS = {"", "null", "~", "none"}
_INT_RE   = re.compile(r"[-+]?(0|[1-9]\d*)")
_FLOAT_RE = re.compile(r"[-+]?(\d+\.\d*|\.\d+|\d+(\.\d*)?[eE][-+]?\d+)")


def value_to_text(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, list):
        return yaml.safe_dump(v, default_flow_style=True).strip()
    return str(v)


def coerce(text: str, original=None):
    """Parse an edited cell back to a Python value.

    Grammar: null/~/empty → None · true/false → bool · plain integer (no
    leading zeros) → int · decimal/exponent → float · ``[...]`` → list ·
    anything else → str.  The original value's type constrains the result:
    a string stays a string, a float stays a float, a datetime is re-parsed.
    """
    t = text.strip()
    if t.lower() in _NULL_WORDS:
        return None
    if isinstance(original, str):
        return text
    if isinstance(original, datetime):
        try:
            return datetime.fromisoformat(t)
        except ValueError:
            return t
    if t.lower() in ("true", "false"):
        return t.lower() == "true"
    if _INT_RE.fullmatch(t):
        return float(t) if isinstance(original, float) else int(t)
    if _FLOAT_RE.fullmatch(t):
        return float(t)
    if t.startswith("["):
        try:
            v = yaml.safe_load(t)
        except yaml.YAMLError as exc:
            raise ValueError(f"invalid list: {exc}") from None
        if not isinstance(v, list):
            raise ValueError("invalid list")
        return v
    return t


# ── Nested-dict paths ─────────────────────────────────────────────────────────

Path_ = tuple   # key path, e.g. ("acquisition_system", "settings", "fa", "value")
DELETED = object()


def leaves(doc: dict, prefix: Path_ = ()) -> dict[Path_, object]:
    """Flatten to ``{path: value}``; lists and empty dicts are leaves."""
    out: dict[Path_, object] = {}
    for k, v in doc.items():
        p = prefix + (k,)
        if isinstance(v, dict) and v:
            out.update(leaves(v, p))
        else:
            out[p] = v
    return out


def _all_paths(doc: dict, prefix: Path_ = ()) -> set[Path_]:
    out: set[Path_] = set()
    for k, v in doc.items():
        p = prefix + (k,)
        out.add(p)
        if isinstance(v, dict):
            out |= _all_paths(v, p)
    return out


def get_path(doc: dict, path: Path_, default=None):
    cur = doc
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def set_path(doc: dict, path: Path_, value) -> None:
    cur = doc
    for k in path[:-1]:
        nxt = cur.get(k)
        if not isinstance(nxt, dict):
            nxt = cur[k] = {}
        cur = nxt
    cur[path[-1]] = copy.deepcopy(value)


def del_path(doc: dict, path: Path_) -> None:
    parent = get_path(doc, path[:-1]) if len(path) > 1 else doc
    if isinstance(parent, dict):
        parent.pop(path[-1], None)


def diff(old: dict, new: dict) -> list[tuple[Path_, object]]:
    """Edits turning *old* into *new*: ``[(path, value | DELETED), ...]``.

    Deletions come first and name the shallowest removed node, so applying
    the list to a different document removes whole subtrees cleanly.
    """
    ol, nl = leaves(old), leaves(new)
    new_all = _all_paths(new)
    dels: list[Path_] = []
    for p in ol:
        if p in nl:
            continue
        for k in range(1, len(p) + 1):
            q = p[:k]
            if q not in new_all:
                if q not in dels:
                    dels.append(q)
                break
    sets = [(p, v) for p, v in nl.items()
            if p not in ol or ol[p] != v or type(ol[p]) is not type(v)]
    return [(p, DELETED) for p in dels] + sets


def apply_changes(doc: dict, changes: list[tuple[Path_, object]]) -> None:
    for p, v in changes:
        if v is DELETED:
            del_path(doc, p)
        else:
            set_path(doc, p, v)


def merge_missing(doc: dict, template: dict) -> list[Path_]:
    """Add keys present in *template* but missing from *doc*; return added paths."""
    added: list[Path_] = []

    def _walk(d, t, prefix):
        for k, tv in t.items():
            p = prefix + (k,)
            if k not in d:
                d[k] = copy.deepcopy(tv)
                added.append(p)
            elif isinstance(tv, dict) and isinstance(d[k], dict):
                _walk(d[k], tv, p)
    _walk(doc, template, ())
    return added


def fill_nulls(doc: dict, source: dict) -> list[Path_]:
    """Copy non-null leaves of *source* into *doc* where *doc* is null or missing."""
    filled = []
    for p, v in leaves(source).items():
        if v is None:
            continue
        if get_path(doc, p) is None:
            set_path(doc, p, v)
            filled.append(p)
    return filled


# ── Inference (stack_sessions parsing + OME header) ──────────────────────────

def sync_from_name(doc: dict, session_name: str) -> list[Path_]:
    """Overwrite the YAML fields derived from the session name; return changed paths."""
    from utilities.stack_sessions import _parse_session_name
    dn = _parse_session_name(session_name)
    updates: list[tuple[Path_, object]] = []
    if dn.get("experiment_datetime"):
        updates.append((("experiment", "datetime"),
                        dn["experiment_datetime"].strftime("%Y-%m-%d %H:%M:%S")))
    if dn.get("experiment_id") is not None:
        updates.append((("experiment", "id"), dn["experiment_id"]))
    if dn.get("condition"):
        updates.append((("experiment", "condition"), dn["condition"]))
    if dn.get("subject_id") is not None:
        updates.append((("subject", "id"), dn["subject_id"]))
    if dn.get("magnification"):
        updates.append((("acquisition_system", "settings", "magnification", "value"),
                        dn["magnification"]))
    changed = []
    for p, v in updates:
        if get_path(doc, p, DELETED) != v:
            set_path(doc, p, v)
            changed.append(p)
    return changed


def infer_doc(session: Session) -> dict:
    """What stack-sessions would write for *session* (template + name + frame + OME)."""
    from utilities.stack_sessions import _discover_channels, _find_master, build_yaml
    px: dict = {}
    master = _find_master(session.path)
    if master is not None:
        try:
            from caiman.utils.ome_meta import extract_pixels
            px = extract_pixels(master)
        except SystemExit:
            px = {}
    else:
        master = next(iter(sorted(session.path.glob("*_C*_t*.tif"))), Path(""))
    return build_yaml(session.name, master, px, len(_discover_channels(session.path)))


# ── Rename ────────────────────────────────────────────────────────────────────

DOWNSTREAM_GLOBS = ("*_pipeline.json", "*_results.hdf5", "*.mmap")


def _owned(name: str, stem: str) -> bool:
    return name == stem or (name.startswith(stem) and name[len(stem)] in "-_.")


def plan_rename(session_dir: Path, new_stem: str) -> list[tuple[Path, Path]]:
    """``(old, new)`` pairs renaming *session_dir* and every entry inside it whose
    name starts with the session stem, deepest first.  Raw frame files
    (``tl1-…_C00_t0000000000.ome.tif``) do not carry the stem and are untouched.
    """
    old_stem = session_dir.name
    if new_stem == old_stem:
        return []
    plan = []
    for p in sorted(session_dir.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if _owned(p.name, old_stem):
            plan.append((p, p.with_name(new_stem + p.name[len(old_stem):])))
    plan.append((session_dir, session_dir.with_name(new_stem)))
    for _old, new in plan:
        if new.exists():
            raise FileExistsError(f"target already exists: {new}")
    return plan


def downstream_files(session_dir: Path) -> list[Path]:
    """Pipeline artefacts that embed absolute paths a rename will not rewrite."""
    out: list[Path] = []
    for g in DOWNSTREAM_GLOBS:
        out.extend(session_dir.rglob(g))
    return sorted(out)


def apply_rename(plan: list[tuple[Path, Path]]) -> None:
    """Execute *plan*; on any failure undo the renames already done and re-raise."""
    done: list[tuple[Path, Path]] = []
    try:
        for old, new in plan:
            old.rename(new)
            done.append((old, new))
    except OSError:
        for old, new in reversed(done):
            try:
                new.rename(old)
            except OSError:
                pass
        raise
