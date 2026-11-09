"""
utilities/rename_rules.py
=========================
The rules rename-sessions reads legacy directory names with, loaded from a
YAML rules file (format and built-in rules: ``utilities/pipelines/
rename_rules.yaml``).

    rules = load_rules()                       # built-in
    rules = load_rules(Path("rename-rules.yaml"))   # built-in + a dataset's
    rules_for(session_dir)                     # nearest rename-rules.yaml above

Dependency-free apart from PyYAML (no caiman import).
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

DEFAULT_PATH = Path(__file__).resolve().parent / "pipelines" / "rename_rules.yaml"
FILENAME = "rename-rules.yaml"          # a dataset's rules, found in a folder above
MERGED = ("paradigms", "defaults")      # partial files merge these (null removes)
EXTRA = {"extra_pre": "pre", "extra_tokens": "tokens"}   # tried first

# field -> (default type, default format, collects a list)
FIELDS = {
    "trialId": ("int", "TL{:03d}", False),
    "magnification": ("int", "{}x", False),
    "experimentId": ("lower", "{}", False),
    "laser_power": ("float", None, False),
    "fa": ("int", None, False),
    "depth_um": ("int", None, False),
    "gain": ("int", None, False),
    "n_frames": ("int", None, False),
    "sample_rate": ("float", None, False),
    "system": ("int", None, False),
    "side": ("lower", None, False),
    "protocol": ("str", None, True),
    "notes": ("str", None, True),
}
TYPES = ("int", "float", "decimal", "str", "lower")
KINDS = (None, "trial", "paradigm")
RULE_KEYS = {"field", "pattern", "match", "value", "type", "rest", "map", "default",
             "format", "repeat", "after", "kind"}


class RulesError(ValueError):
    """A rules file that can't be used (names the file and the rule)."""


@dataclass
class Rule:
    field: str
    regex: Optional[re.Pattern] = None
    search: bool = False
    value: Any = None          # group number, [int, frac] groups, or a literal
    type: str = "str"
    rest: Any = None           # group number or "around"
    map: dict = field(default_factory=dict)
    default: Optional[str] = None
    format: Optional[str] = None
    repeat_drop: bool = False
    after: Optional[str] = None
    kind: Optional[str] = None
    source: str = ""           # for messages: "tokens[3]"

    @property
    def collects(self) -> bool:
        return FIELDS[self.field][2]

    def find(self, text: str) -> Optional[re.Match]:
        if self.regex is None:
            return None
        return self.regex.search(text) if self.search else self.regex.fullmatch(text)

    def read(self, m: re.Match, token: str):
        """The (converted, mapped) value a match gives."""
        v = self.value
        if isinstance(v, list):
            whole, frac = m.group(v[0]), (m.group(v[1]) if len(v) > 1 else None)
            raw = f"{int(whole)}.{frac}" if frac else whole
        elif isinstance(v, int):
            raw = token if v == 0 else m.group(v)
        elif v is None:
            raw = m.group(0)
        else:
            return v                                    # literal
        if self.map or self.default is not None:
            raw = self.map.get(raw.lower(), self.default if self.default is not None else raw)
        if self.type in ("int",):
            return int(raw)
        if self.type in ("float", "decimal"):
            return float(raw)
        return raw.lower() if self.type == "lower" else raw

    def leftover(self, m: re.Match, token: str) -> list:
        """Tokens handed back after a match."""
        if self.rest == "around":
            return [r for r in (token[:m.start()], token[m.end():]) if r]
        if isinstance(self.rest, int) and m.group(self.rest):
            return [m.group(self.rest)]
        return []


@dataclass
class Rules:
    path: Optional[Path]               # the dataset file (None = built-in only)
    legacy_name: re.Pattern
    separators: re.Pattern
    ancestor_name: re.Pattern
    system_folder: re.Pattern
    max_trial: int
    defaults: dict
    pre: list
    tokens: list
    paradigms: dict
    prefixes: list                     # paradigm words usable as a prefix, longest first
    single_word_experiment: bool

    @property
    def label(self) -> str:
        return str(self.path) if self.path else "built-in"

    def paradigm(self, word: str):
        """(experimentId, rest) for a word that is or starts with a known
        paradigm, else None."""
        w = word.lower()
        if w in self.paradigms:
            return self.paradigms[w], ""
        for k in self.prefixes:
            if w.startswith(k):
                return self.paradigms[k], word[len(k):]
        return None


# ── loading ──────────────────────────────────────────────────────────────────

def _read(path: Path) -> dict:
    try:
        doc = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise RulesError(f"{path}: {exc}") from None
    if doc is None:
        return {}
    if not isinstance(doc, dict):
        raise RulesError(f"{path}: expected a mapping at the top")
    return doc


def merge(base: dict, over: dict) -> dict:
    """*over* on top of *base*: keys replace, except MERGED dicts (merged;
    a null value removes the key) and EXTRA lists (prepended to theirs)."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        if k in MERGED and isinstance(v, dict):
            d = dict(out.get(k) or {})
            for kk, vv in v.items():
                if vv is None:
                    d.pop(str(kk).lower() if k == "paradigms" else kk, None)
                else:
                    d[str(kk).lower() if k == "paradigms" else kk] = vv
            out[k] = d
        elif k in EXTRA:
            out[k] = list(v or []) + list(out.get(k) or [])
        else:
            out[k] = copy.deepcopy(v)
    return out


def _regex(where: str, pat, flags=re.I) -> re.Pattern:
    if not isinstance(pat, str):
        raise RulesError(f"{where}: pattern must be a string")
    try:
        return re.compile(pat, flags)
    except re.error as exc:
        raise RulesError(f"{where}: bad pattern {pat!r}: {exc}") from None


def _rule(where: str, r) -> Rule:
    if not isinstance(r, dict):
        raise RulesError(f"{where}: a rule is a mapping")
    bad = set(r) - RULE_KEYS
    if bad:
        raise RulesError(f"{where}: unknown key(s) {', '.join(sorted(bad))}")
    f = r.get("field")
    if f not in FIELDS:
        raise RulesError(f"{where}: field {f!r} is not one of {', '.join(FIELDS)}")
    kind = r.get("kind")
    if kind not in KINDS:
        raise RulesError(f"{where}: kind {kind!r} is not trial or paradigm")
    if kind == "paradigm":
        return Rule(f, kind=kind, source=where)
    if "pattern" not in r:
        raise RulesError(f"{where}: no pattern")
    rx = _regex(where, r["pattern"])
    typ = r.get("type") or ("decimal" if isinstance(r.get("value"), list) else FIELDS[f][0])
    if typ not in TYPES:
        raise RulesError(f"{where}: type {typ!r} is not one of {', '.join(TYPES)}")
    value, rest = r.get("value"), r.get("rest")
    for name, g in (("value", value), ("rest", rest)):
        gs = g if isinstance(g, list) else [g]
        for x in gs:
            if isinstance(x, int) and not isinstance(x, bool) and x > rx.groups:
                raise RulesError(f"{where}: {name} group {x} but the pattern has {rx.groups}")
    if rest is not None and rest != "around" and not isinstance(rest, int):
        raise RulesError(f"{where}: rest must be a group number or 'around'")
    if kind == "trial" and not isinstance(value, int):
        raise RulesError(f"{where}: a trial rule needs value: <group>")
    match = r.get("match", "full")
    if match not in ("full", "search"):
        raise RulesError(f"{where}: match must be full or search")
    after = r.get("after")
    if after is not None and after not in FIELDS:
        raise RulesError(f"{where}: after {after!r} is not a field")
    return Rule(f, regex=rx, search=match == "search", value=value, type=typ, rest=rest,
                map={str(k).lower(): str(v) for k, v in (r.get("map") or {}).items()},
                default=r.get("default"), format=r.get("format") or FIELDS[f][1],
                repeat_drop=r.get("repeat") == "drop", after=after, kind=kind, source=where)


def build(doc: dict, path: Optional[Path] = None) -> Rules:
    """Rules from a complete (already merged) document."""
    where = str(path) if path else str(DEFAULT_PATH)
    need = ("legacy_name", "separators", "ancestor_name", "system_folder", "max_trial",
            "defaults", "tokens", "paradigms")
    miss = [k for k in need if k not in doc]
    if miss:
        raise RulesError(f"{where}: missing {', '.join(miss)}")
    legacy = _regex(f"{where}: legacy_name", doc["legacy_name"], 0)
    if not {"date", "hh", "mm", "ss"} <= set(legacy.groupindex):
        raise RulesError(f"{where}: legacy_name needs groups date, hh, mm, ss")
    anc = _regex(f"{where}: ancestor_name", doc["ancestor_name"], 0)
    if not {"src", "user", "subj"} <= set(anc.groupindex):
        raise RulesError(f"{where}: ancestor_name needs groups src, user, subj")
    pre = [_rule(f"{where}: {k}[{i}]", r) for k in ("extra_pre", "pre")
           for i, r in enumerate(doc.get(k) or [])]
    tokens = [_rule(f"{where}: {k}[{i}]", r) for k in ("extra_tokens", "tokens")
              for i, r in enumerate(doc.get(k) or [])]
    for r in pre:
        if r.kind:
            raise RulesError(f"{r.source}: pre rules can't have a kind")
    paradigms = {str(k).lower(): str(v).lower() for k, v in (doc["paradigms"] or {}).items()}
    pmin = int(doc.get("paradigm_prefix_min", 4))
    try:
        max_trial = int(doc["max_trial"])
    except (TypeError, ValueError):
        raise RulesError(f"{where}: max_trial must be a number") from None
    return Rules(
        path=path, legacy_name=legacy, separators=_regex(f"{where}: separators",
                                                         doc["separators"], 0),
        ancestor_name=anc, system_folder=_regex(f"{where}: system_folder",
                                                doc["system_folder"]),
        max_trial=max_trial, defaults={str(k): str(v) for k, v in doc["defaults"].items()},
        pre=pre, tokens=tokens, paradigms=paradigms,
        prefixes=sorted((k for k in paradigms if len(k) >= pmin), key=len, reverse=True),
        single_word_experiment=bool(doc.get("single_word_experiment", True)))


_CACHE: dict = {}


def load_rules(path: Optional[Path] = None) -> Rules:
    """The built-in rules, or a rules file on top of them (cached by mtime)."""
    key = None
    if path is not None:
        path = Path(path).resolve()
        try:
            key = (path, path.stat().st_mtime_ns)
        except OSError as exc:
            raise RulesError(f"{path}: {exc}") from None
    if key in _CACHE:
        return _CACHE[key]
    doc = _read(DEFAULT_PATH)
    if path is not None:
        over = _read(path)
        ver = over.get("version", 1)
        if ver != 1:
            raise RulesError(f"{path}: version {ver} not supported (1)")
        doc = merge(doc, over)
    rules = _CACHE[key] = build(doc, path)
    return rules


def find_rules_file(d: Path, stop: Optional[Path] = None) -> Optional[Path]:
    """The nearest rename-rules.yaml in *d* or a folder above it (up to *stop*,
    the filesystem root by default)."""
    for a in [d, *d.parents]:
        f = a / FILENAME
        if f.is_file():
            return f
        if stop is not None and a == stop:
            break
    return None


def rules_for(d: Path) -> Rules:
    """The rules that apply to directory *d* (nearest rename-rules.yaml above
    it, else built-in)."""
    return load_rules(find_rules_file(d.parent))


def write_rules(dest: Path) -> Path:
    """Copy the built-in rules (with their comments) to *dest*, a file or a
    folder (-> <folder>/rename-rules.yaml).  Never overwrites."""
    dest = Path(dest)
    if dest.is_dir():
        dest = dest / FILENAME
    if dest.exists():
        raise FileExistsError(f"{dest} exists")
    dest.write_text(DEFAULT_PATH.read_text())
    return dest


__all__ = ["Rule", "Rules", "RulesError", "load_rules", "rules_for", "find_rules_file",
           "write_rules", "merge", "build", "FILENAME", "DEFAULT_PATH", "FIELDS"]
