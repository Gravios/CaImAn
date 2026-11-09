#!/usr/bin/env python3
"""
utilities/rename_sessions.py
============================
Rename legacy acquisition directories to the session naming schema and
write their Trial YAML.

    151106_tl1-25xspont-FA1-LP5-104um_13-34-40
      -> strohA-ia-000000-20151106-TL001_133440-25x-spont
    140225_tl1_fpstim_z90_p1_50_15-56-08
      -> strohA-ia-000000-20140225-TL001_155608-00x-fpstim

Schema (:mod:`utilities.rename_stem`)::

    <sourceId>-<userId>-<subjectId>-<date>-<trialId>_<time>-<magnification>-<experimentId>

Where each field comes from
---------------------------
Tokens are separated by - _ . and read in order; a rule may hand back the
rest of a glued token (tl1fpstim → tl1 + fpstim, z125fpstim → z125 + fpstim).

  date, time      leading YYMMDD (or YYYYMMDD) and trailing HH-MM-SS
  trialId         tl<N> (also glued: tl1fpstim, TL1new) → TL<NNN>; the index of
                  <paradigm><N> (3Dscan2, zstack1, morph2, test3) when there is
                  no tl<N>; none → TL000
                  (or --number-missing)
  magnification   <N>x, also glued to a word (25xspont, vs25x) → <N>x; none → 00x
  experimentId    a known word or a word starting with one (PARADIGMS: spont /
                  spontan, vstim / visstim, fpstim, stimfp, pawstim, fiberstim,
                  3dscan, zstack, morph, test, persup (persUp, persUpstate),
                  updown (UpDown, UpDownstate, Upown), …); one other plain word
                  is taken as is;
                  none → default
  sourceId-userId from the nearest ancestor named <src>-<user>-<subj>[-<date>[_note]]
  subjectId       (e.g. strohA-ia-000000-20140225, …-20150121_2Pexp_micfailure),
                  or --prefix / --subject

YAML only (not in the name):
  sample_rate     15_4Hz, 30.4Hz, 30Hz (a separate token; stack-sessions
                  replaces it with the OME header's)
  acquisition_system.system   sys<N>, or a System<N> folder above the session
  experiment.folder_note      the date folder's suffix (2Pexp_micfailure)
  laserPower      LP<N>[_<frac>], P<N>[_<frac>], P<N>sb   (p7_50 → 7.5, p10 → 10)
  fa              FA<N>, favg<N>
  depth           <N>um, depth<N>, z<N>  (stored negative, as stack-sessions
                  does: below the surface)
  gain            gain<N>
  n_frames        <N>img (requested; stack-sessions overwrites it with SizeT)
  experiment.protocol   the rest of a paradigm word or a protocol-like word
                        (pawstim10pulses3Hz → 10pulses3Hz, 1pulse50msfp,
                        pinzettealle10s)
  experiment.side       ipsi / contra (ipsi…, ispsi…, …ipsil)
  experiment.notes      every token no rule explains (sb, cloud25, 75 195, …),
                        verbatim — nothing is guessed or dropped

The Trial YAML ``<new>/<new>.yaml`` is the acquisition template filled from
the new name (as stack-sessions would) plus laser power, FA and depth, and
``experiment.legacy_name`` (the original directory name).  An existing YAML
is never overwritten: only its null fields are filled.  stack-sessions later
adds the OME fields (sample rate, frame size, …) without touching these.

Safety
------
  * Dry run by default; --apply renames.
  * Every target name is validated against the schema; a target that exists
    or that two directories would both get is refused (nothing for either).
  * Each directory is renamed together with the entries inside it that carry
    its old name; a failure rolls that directory back.
  * --log writes a TSV (src, dst, confidence, unparsed, yaml); --undo LOG
    reverses an applied log.
  * Date folders <src>-<user>-<subj>-<date> whose date disagrees with every
    session inside (e.g. 20150302 holding 150203 sessions, or 20152020) are
    reported, and renamed with --fix-date-folders after their sessions
    (folders with sessions from several dates are only reported).
  * Pipeline files (*_pipeline.json, *_results.hdf5, *.mmap) embed absolute
    paths a rename does not rewrite: they are listed as warnings.

Usage
-----
  rename-sessions --root /data/source/strohA/strohA-ia            # dry run
  rename-sessions --root … --apply --log rename.tsv
  rename-sessions --root … --min-confidence 4 --apply
  rename-sessions --root … --number-missing --apply   # TL by time for dirs without tl<N>
  rename-sessions --root … --fix-date-folders --apply # also fix wrong date folders
  rename-sessions --root … --yaml-only --apply        # YAML for already-named dirs
  rename-sessions --undo rename.tsv --apply
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from utilities.rename_stem import build_stem, parse_stem
from utilities import session_meta as sm

# ── Legacy-name grammar ─────────────────────────────────────────────────────

# <date>[_-]<body>[_-]<HH>-<MM>-<SS>    (body may be empty)
_LEGACY_RE = re.compile(
    r"^(?P<date>\d{6}|\d{8})"
    r"(?:[_-](?P<body>.*?))?"
    r"[_-](?P<hh>\d{2})-(?P<mm>\d{2})-(?P<ss>\d{2})$")

# Token rules, tried in order on each token (separators: - _ . space).  A
# rule may hand back the rest of a glued token (tl1fpstim → tl1 + fpstim).
_LP_FRAC_RE = re.compile(r"(?<![A-Za-z0-9])(lp|p)(\d+)_(\d+)(?![A-Za-z0-9])", re.I)
# frame rate, also across a separator: 15_4Hz, 30.4Hz, 30Hz (not inside a word:
# 10pulses3Hz stays a protocol)
_RATE_RE    = re.compile(r"(?<![A-Za-z0-9])(\d+)(?:[_.](\d+))?hz(?![A-Za-z0-9])", re.I)
_SYS_RE     = re.compile(r"sys(?:tem)?(\d+)$", re.I)              # sys3
_TRIAL_RE   = re.compile(r"tl(\d+)(.*)$", re.I)                  # tl5, tl1fpstim, TL1new
_LP_RE      = re.compile(r"(?:lp|p)(\d+)(sb)?$", re.I)           # LP5, p10, p25sb
_FA_RE      = re.compile(r"(?:fa|favg)(\d+)$", re.I)             # FA1, favg2
_GAIN_RE    = re.compile(r"gain(\d+)$", re.I)                    # gain61
_IMG_RE     = re.compile(r"(\d+)img$", re.I)                     # 4000img
_DEPTH_RE   = re.compile(r"(?:(\d+)um|depth(\d+)|z(\d+))(.*)$", re.I)   # 104um, depth116, z125fpstim
_MAG_RE     = re.compile(r"(?<![0-9])(\d+)x(?![0-9])", re.I)    # 25x, 25xspont, vs25x
_SIDE_RE    = re.compile(r"(ispsi|ipsil|ipsi|contra)", re.I)
_PROTOCOL_RE = re.compile(r"puls|pules|\d+(?:ms|hz)|alle\d", re.I)  # 10pulses3Hz, 50ms, alle10s

# experimentId vocabulary (lower case → name token).  Words that only *start*
# with one of these (pawstim10pulses3Hz, fpstimsinglepuleses50msl) give the
# paradigm, and the rest becomes the YAML protocol.
PARADIGMS = {
    "spont": "spont", "spontact": "spont", "spontaneous": "spont", "spontan": "spont",
    "vs": "vstim", "vstim": "vstim", "visstim": "vstim", "stim": "stim",
    "fpstim": "fpstim", "stimfp": "stimfp", "pawstim": "pawstim",
    "fiberstim": "fiberstim", "astim": "astim", "chrimson": "chrimson",
    "opo": "opo", "tisa": "tisa", "3d": "3dscan", "3dscan": "3dscan",
    "zstack": "zstack", "loco": "loco", "dark": "dark", "test": "test",
    "morph": "morph",
    # Up / Down states (incl. the spellings in the 2015 names)
    "persup": "persup", "persupstate": "persup",
    "updown": "updown", "updownstate": "updown", "upown": "updown",
    "default": "default",
}
_PREFIXES = sorted((k for k in PARADIGMS if len(k) >= 4), key=len, reverse=True)
DEFAULTS = {"subjectId": "000000", "trialId": "TL000",
            "magnification": "00x", "experimentId": "default"}
# fields that count towards the confidence score
SCORED = ("date", "time", "trialId", "magnification", "experimentId",
          "laser_power", "fa", "depth_um")

# ancestor carrying <src>-<user>-<subj>[-<date8>]
_ANCESTOR_RE = re.compile(r"^(?P<src>[^-_\s]+)-(?P<user>[^-_\s]+)-(?P<subj>[^-_\s]+)"
                          r"(?:-(?P<date>\d{8})(?P<suffix>[_-].+)?)?$")
_SYSTEM_DIR_RE = re.compile(r"^system[-_ ]?(\d+)$", re.I)            # System3


@dataclass
class Parsed:
    """A legacy directory name taken apart."""
    name: str
    parts: dict = field(default_factory=dict)       # schema fields (strings)
    laser_power: Optional[float] = None
    fa: Optional[int] = None
    depth_um: Optional[int] = None                   # positive µm below surface
    gain: Optional[int] = None
    folder_note: str = ""                            # date-folder suffix (_2Pexp_micfailure)
    n_frames: Optional[int] = None                   # requested (<N>img)
    side: Optional[str] = None                       # ipsi / contra
    sample_rate: Optional[float] = None              # Hz, from 15_4Hz / 30.4Hz
    system: Optional[int] = None                     # acquisition system (sys3)
    protocol: list = field(default_factory=list)     # stimulation protocol words
    found: set = field(default_factory=set)          # SCORED fields actually read
    unparsed: list = field(default_factory=list)     # kept verbatim as YAML notes

    @property
    def confidence(self) -> int:
        return len(self.found)


def _year(d: str) -> str:
    if len(d) == 8:
        return d
    yy = int(d[:2])
    return f"{1900 + yy if yy >= 70 else 2000 + yy}{d[2:]}"


def is_legacy(name: str) -> bool:
    return bool(_LEGACY_RE.match(name)) and parse_stem(name) is None


def _paradigm(word: str):
    """(experimentId, protocol rest) for a word that is or starts with a
    known paradigm, else None."""
    w = word.lower()
    if w in PARADIGMS:
        return PARADIGMS[w], ""
    for k in _PREFIXES:
        if w.startswith(k):
            return PARADIGMS[k], word[len(k):]
    return None


def parse_legacy(name: str) -> Optional[Parsed]:
    """Parse a legacy directory name, or None if it isn't one."""
    m = _LEGACY_RE.match(name)
    if not m or parse_stem(name) is not None:
        return None
    p = Parsed(name)
    date = _year(m.group("date"))
    time = m.group("hh") + m.group("mm") + m.group("ss")
    try:
        datetime.strptime(date + time, "%Y%m%d%H%M%S")
    except ValueError:
        return None
    p.parts.update(date=date, time=time)
    p.found |= {"date", "time"}
    body = m.group("body") or ""

    # frame rate and laser power with a fraction span a separator
    mm = _RATE_RE.search(body)
    if mm:
        p.sample_rate = float(f"{int(mm.group(1))}.{mm.group(2)}" if mm.group(2) else mm.group(1))
        body = body[:mm.start()] + " " + body[mm.end():]
    mm = _LP_FRAC_RE.search(body)
    if mm:
        p.laser_power = float(f"{int(mm.group(2))}.{mm.group(3)}")
        body = body[:mm.start()] + " " + body[mm.end():]
    todo = [w for w in re.split(r"[-_.\s]+", body) if w]
    scan_index = None
    paradigms: list = []
    while todo:
        w = todo.pop(0)
        if (mm := _TRIAL_RE.fullmatch(w)) and "trialId" not in p.found:
            p.parts["trialId"] = f"TL{int(mm.group(1)):03d}"
            p.found.add("trialId")
            if mm.group(2):
                todo.insert(0, mm.group(2))
        elif (mm := _LP_RE.fullmatch(w)) and p.laser_power is None:
            p.laser_power = float(mm.group(1))
            if mm.group(2):
                todo.insert(0, mm.group(2))
        elif (mm := _FA_RE.fullmatch(w)) and p.fa is None:
            p.fa = int(mm.group(1))
        elif (mm := _SYS_RE.fullmatch(w)) and p.system is None:
            p.system = int(mm.group(1))
        elif (mm := _GAIN_RE.fullmatch(w)) and p.gain is None:
            p.gain = int(mm.group(1))
        elif (mm := _IMG_RE.fullmatch(w)) and p.n_frames is None:
            p.n_frames = int(mm.group(1))
        elif (mm := _DEPTH_RE.fullmatch(w)) and p.depth_um is None:
            p.depth_um = int(next(g for g in mm.groups()[:3] if g))
            if mm.group(4):
                todo.insert(0, mm.group(4))
        elif (mm := _MAG_RE.search(w)) and ("magnification" not in p.found or
                                            p.parts["magnification"] == f"{int(mm.group(1))}x"):
            p.parts["magnification"] = f"{int(mm.group(1))}x"     # a repeat is dropped
            p.found.add("magnification")
            rest = w[:mm.start()] + w[mm.end():]
            if rest:
                todo.insert(0, rest)
        elif (mm := _SIDE_RE.search(w)) and p.side is None:
            p.side = "contra" if mm.group(1).lower() == "contra" else "ipsi"
            rest = [r for r in (w[:mm.start()], w[mm.end():]) if r]
            todo[:0] = rest
        elif (par := _paradigm(w)) is not None:
            paradigms.append(par[0])
            if par[1].isdigit():                 # 3Dscan2, zstack1, morph2, test3
                scan_index = int(par[1]) if scan_index is None else scan_index
            elif par[1]:
                todo.insert(0, par[1])
        elif _PROTOCOL_RE.search(w) and paradigms and not w.isdigit():
            p.protocol.append(w)
        else:
            p.unparsed.append(w)
    if paradigms:
        p.parts["experimentId"] = paradigms[0]
        p.found.add("experimentId")
        p.unparsed += [x for x in paradigms[1:] if x != paradigms[0]]
    elif len(p.unparsed) == 1 and re.fullmatch(r"[A-Za-z]+", p.unparsed[0]):
        p.parts["experimentId"] = p.unparsed.pop().lower()     # e.g. a new paradigm
        p.found.add("experimentId")
    if scan_index is not None and "trialId" not in p.found:
        p.parts["trialId"] = f"TL{scan_index:03d}"
        p.found.add("trialId")
    for key, v in (("laser_power", p.laser_power), ("fa", p.fa), ("depth_um", p.depth_um)):
        if v is not None:
            p.found.add(key)
    for k, v in DEFAULTS.items():
        p.parts.setdefault(k, v)
    return p


@dataclass
class Ancestry:
    """What the folders above a session say."""
    source: Optional[str] = None
    user: Optional[str] = None
    subject: Optional[str] = None
    date: Optional[str] = None           # of the date folder <src>-<user>-<subj>-<date8>
    date_dir: Optional[Path] = None
    suffix: str = ""                     # after the date: _2Pexp_micfailure
    system: Optional[int] = None         # a System<N> folder on the way up


def ancestor_ids(path: Path) -> Ancestry:
    """Ids from the nearest ancestor named <src>-<user>-<subj>[-<date8>[_note]]
    that is not itself a session; a System<N> folder below it gives the
    acquisition system."""
    anc = Ancestry()
    for a in path.parents:
        if parse_stem(a.name) is not None:
            continue
        ms = _SYSTEM_DIR_RE.match(a.name)
        if ms and anc.system is None:
            anc.system = int(ms.group(1))
            continue
        m = _ANCESTOR_RE.match(a.name)
        if m:
            anc.source, anc.user, anc.subject = m.group("src"), m.group("user"), m.group("subj")
            if m.group("date"):
                anc.date, anc.date_dir = m.group("date"), a
                anc.suffix = (m.group("suffix") or "").lstrip("_-")
            return anc
    return anc


# ── Plan ─────────────────────────────────────────────────────────────────────

@dataclass
class Item:
    src: Path
    parsed: Optional[Parsed]
    dst: Optional[Path] = None
    problems: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    ancestry: Optional["Ancestry"] = None

    @property
    def ok(self) -> bool:
        return self.dst is not None and not self.problems


def find_legacy_dirs(root: Path) -> list[Path]:
    """Legacy-named directories below *root* (not descending into them)."""
    out: list[Path] = []

    def walk(d: Path) -> None:
        try:
            children = sorted(c for c in d.iterdir() if c.is_dir() and not c.is_symlink())
        except OSError:
            return
        for c in children:
            if is_legacy(c.name):
                out.append(c)
            elif parse_stem(c.name) is None:      # sessions don't nest
                walk(c)
    if is_legacy(root.name):
        return [root]
    walk(root)
    return out


def plan(root: Path, *, prefix: Optional[str] = None, subject: Optional[str] = None,
         min_confidence: int = 0, number_missing: bool = False) -> list[Item]:
    """What would be renamed, and why not where it can't be."""
    items: list[Item] = []
    for d in find_legacy_dirs(root):
        p = parse_legacy(d.name)
        it = Item(d, p)
        items.append(it)
        if p is None:
            it.problems.append("not a legacy name (date or time invalid)")
            continue
        anc = ancestor_ids(d)
        it.ancestry = anc
        src_id, user_id, subj_id, anc_date = anc.source, anc.user, anc.subject, anc.date
        if p.system is None:
            p.system = anc.system
        if anc.suffix:
            p.folder_note = anc.suffix
        if prefix:
            src_id, _, user_id = prefix.partition("-")
            if not user_id:
                it.problems.append(f"--prefix {prefix!r} is not <sourceId>-<userId>")
        if not (src_id and user_id):
            it.problems.append("no <source>-<user> ancestor; pass --prefix")
        p.parts.update(sourceId=src_id or "", userId=user_id or "",
                       subjectId=subject or subj_id or DEFAULTS["subjectId"])
        if anc_date and anc_date != p.parts["date"]:
            it.warnings.append(f"date {p.parts['date']} differs from the folder's {anc_date}")
        if p.confidence < min_confidence:
            it.problems.append(f"confidence {p.confidence} < {min_confidence}")
        if p.unparsed:
            it.warnings.append("kept as YAML notes: " + " ".join(p.unparsed))

    if number_missing:                     # TL by acquisition time per folder / date
        groups = defaultdict(list)
        for it in items:
            if it.parsed and "trialId" in it.parsed.parts:
                groups[(it.src.parent, it.parsed.parts["date"])].append(it)
        for grp in groups.values():
            used = {int(it.parsed.parts["trialId"][2:]) for it in grp
                    if "trialId" in it.parsed.found}
            n = 0
            for it in sorted(grp, key=lambda it: it.parsed.parts["time"]):
                if "trialId" in it.parsed.found:
                    continue
                n += 1
                while n in used:
                    n += 1
                it.parsed.parts["trialId"] = f"TL{n:03d}"
                it.warnings.append(f"trial numbered {it.parsed.parts['trialId']} by time")

    for it in items:
        if it.parsed is None or it.problems:
            continue
        errs = sm.validate_parts(it.parsed.parts)
        if errs:
            it.problems.append("invalid name: " + ", ".join(f"{k} {v}" for k, v in errs.items()))
            continue
        it.dst = it.src.with_name(build_stem(it.parsed.parts))
        if it.dst.exists():
            it.problems.append(f"target exists: {it.dst.name}")
        down = sm.downstream_files(it.src)
        if down:
            it.warnings.append(f"{len(down)} pipeline file(s) keep the old path, e.g. {down[0].name}")
    targets = defaultdict(list)
    for it in items:
        if it.dst is not None and not it.problems:
            targets[it.dst].append(it)
    for dst, its in targets.items():
        if len(its) > 1:
            for it in its:
                it.problems.append(f"{len(its)} directories would become {dst.name}")
    return items


@dataclass
class FolderFix:
    src: Path
    dst: Path
    reason: str


def plan_date_folders(items: list[Item]) -> tuple[list[FolderFix], list[str]]:
    """Date folders whose name disagrees with *every* session in them on one
    date: (fixes, notes on folders left alone because their sessions
    disagree among themselves or the target exists)."""
    by_dir = defaultdict(set)
    for it in items:
        anc = it.ancestry
        if it.parsed is not None and anc is not None and anc.date_dir is not None:
            by_dir[anc.date_dir].add(it.parsed.parts["date"])
    fixes, notes = [], []
    for d, dates in sorted(by_dir.items()):
        m = _ANCESTOR_RE.match(d.name)
        folder_date = m.group("date")
        if dates == {folder_date}:
            continue
        if len(dates) > 1:
            notes.append(f"{d.name}: sessions from {', '.join(sorted(dates))} — left as is")
            continue
        (date,) = dates
        new = d.with_name(f"{m.group('src')}-{m.group('user')}-{m.group('subj')}-{date}"
                          f"{m.group('suffix') or ''}")
        try:
            datetime.strptime(folder_date, "%Y%m%d")
            why = f"its sessions are from {date}"
        except ValueError:
            why = f"{folder_date} is not a date; its sessions are from {date}"
        if new.exists():
            notes.append(f"{d.name}: {why}, but {new.name} exists — left as is")
            continue
        fixes.append(FolderFix(d, new, why))
    return fixes, notes


# ── YAML ─────────────────────────────────────────────────────────────────────

def trial_doc(new_name: str, p: Optional[Parsed]) -> dict:
    """Template + fields from the new name (as stack-sessions fills them) +
    laser power / FA / depth / legacy name from the old one."""
    from utilities.session_name import parse_session_name
    doc = sm.load_yaml(sm.template_path())
    sm.sync_from_name(doc, new_name)
    dn = parse_session_name(new_name)
    if dn.get("magnification") == 0:      # 00x = unknown, not zero
        sm.set_path(doc, ("acquisition_system", "settings", "magnification", "value"), None)
    if p is not None:
        s = ("acquisition_system", "settings")
        if p.laser_power is not None:
            sm.set_path(doc, s + ("laserPower", "value"), p.laser_power)
        if p.fa is not None:
            sm.set_path(doc, s + ("fa", "value"), p.fa)
        if p.depth_um is not None:
            sm.set_path(doc, s + ("depth", "value"), -abs(p.depth_um))
        if p.gain is not None:
            sm.set_path(doc, s + ("gain", "value"), p.gain)
        if p.sample_rate is not None:       # from the name; stack-sessions writes OME's
            sm.set_path(doc, s + ("sample_rate", "value"), p.sample_rate)
        if p.system is not None:
            doc.setdefault("acquisition_system", {})["system"] = p.system
        if p.n_frames is not None:          # requested; stack-sessions writes SizeT
            sm.set_path(doc, s + ("n_frames",), p.n_frames)
        exp = doc.setdefault("experiment", {})
        if p.protocol:
            exp["protocol"] = " ".join(p.protocol)
        if p.side:
            exp["side"] = p.side
        if p.unparsed:
            exp["notes"] = list(p.unparsed)
        if p.folder_note:
            exp["folder_note"] = p.folder_note
        exp["legacy_name"] = p.name
    return doc


def write_yaml(session_dir: Path, p: Optional[Parsed]) -> str:
    """Create <dir>/<dir>.yaml, or fill only the null fields of an existing
    one.  Returns what was done."""
    path = session_dir / f"{session_dir.name}.yaml"
    doc = trial_doc(session_dir.name, p)
    if path.exists():
        cur = sm.load_yaml(path)
        filled = sm.fill_nulls(cur, doc) + sm.merge_missing(cur, doc)
        if p is not None and "legacy_name" not in cur.get("experiment", {}):
            cur.setdefault("experiment", {})["legacy_name"] = p.name
            filled.append(("experiment", "legacy_name"))
        if filled:
            sm.save_yaml(path, cur)
            return f"yaml: filled {len(filled)} field(s) in {path.name}"
        return f"yaml: {path.name} up to date"
    sm.save_yaml(path, doc)
    return f"yaml: wrote {path.name}"


# ── Apply / log / undo ───────────────────────────────────────────────────────

LOG_FIELDS = ("src", "dst", "confidence", "unparsed", "yaml")   # yaml: wrote|filled|


def apply(items: list[Item], *, yaml: bool = True, log: Optional[Path] = None,
          out=print, folders: Optional[list] = None) -> int:
    """Rename every OK item; returns the number of failures.  Each rename is
    logged as soon as it is done, so the log stays complete (and --undo
    works) even if the run stops part-way."""
    failed = 0
    fh = writer = None
    if log is not None:
        new = not log.exists()
        fh = open(log, "a", newline="")
        writer = csv.writer(fh, delimiter="\t")
        if new:
            writer.writerow(LOG_FIELDS)
            fh.flush()
    try:
        for it in items:
            if not it.ok:
                continue
            try:
                sm.apply_rename(sm.plan_rename(it.src, it.dst.name))
            except OSError as exc:
                failed += 1
                out(f"  FAILED {it.src.name}: {exc}")
                continue
            done = ""
            if yaml:
                try:
                    msg = write_yaml(it.dst, it.parsed)
                    done = msg.split()[1].rstrip(":") if msg.split()[1] in ("wrote", "filled") else ""
                    out("    " + msg)
                except (Exception, SystemExit) as exc:   # the rename stands
                    failed += 1
                    out(f"    yaml: FAILED for {it.dst.name}: {exc}")
            if writer is not None:
                writer.writerow((str(it.src), str(it.dst), it.parsed.confidence,
                                 " ".join(it.parsed.unparsed), done))
                fh.flush()
        for fx in folders or []:              # after the sessions inside them
            try:
                # the folder only: sessions inside already carry the new date,
                # and must not be touched by a stem rename
                if fx.dst.exists():
                    raise FileExistsError(f"target already exists: {fx.dst}")
                fx.src.rename(fx.dst)
            except OSError as exc:
                failed += 1
                out(f"  FAILED folder {fx.src.name}: {exc}")
                continue
            out(f"  folder {fx.src.name} -> {fx.dst.name}")
            if writer is not None:
                writer.writerow((str(fx.src), str(fx.dst), "folder", "", ""))
                fh.flush()
    finally:
        if fh is not None:
            fh.close()
    return failed


def undo(log: Path, *, dry_run: bool = True, out=print) -> int:
    """Reverse the renames of an applied log (last first).  A YAML the run
    created (log column ``yaml`` = wrote) is removed; a filled one is kept."""
    with open(log, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    failed = 0
    for r in reversed(rows):
        src, dst = Path(r["src"]), Path(r["dst"])
        if not dst.exists():
            out(f"  skip (gone): {dst}"); failed += 1
            continue
        if src.exists():
            out(f"  skip (occupied): {src}"); failed += 1
            continue
        made = dst / f"{dst.name}.yaml"
        drop = r.get("yaml") == "wrote" and made.exists()
        out(f"  {dst.name}\n    -> {src.name}" + (f"\n    remove {made.name}" if drop else ""))
        if not dry_run:
            try:
                if drop:
                    made.unlink()
                if r.get("confidence") == "folder":     # a date folder: itself only
                    dst.rename(src)
                else:
                    sm.apply_rename(sm.plan_rename(dst, src.name))
            except OSError as exc:
                out(f"  FAILED: {exc}"); failed += 1
    return failed


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="rename-sessions",
        description="Rename legacy acquisition directories to the session schema "
                    "and write their Trial YAML (dry run unless --apply).",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--root", type=Path, default=Path.cwd(),
                    help="directory to search (default: current directory)")
    ap.add_argument("--apply", action="store_true", help="rename (default: dry run)")
    ap.add_argument("--prefix", help="<sourceId>-<userId> when no ancestor folder gives it")
    ap.add_argument("--subject", help="subjectId when no ancestor folder gives it")
    ap.add_argument("--min-confidence", type=int, default=0,
                    help=f"skip names with fewer recognised fields (of {len(SCORED)})")
    ap.add_argument("--number-missing", action="store_true",
                    help="number directories without tl<N> TL001, TL002, … by time "
                         "(per folder and date, skipping numbers in use)")
    ap.add_argument("--fix-date-folders", action="store_true",
                    help="also rename a date folder <src>-<user>-<subj>-<date> whose "
                         "date disagrees with all sessions in it (they agree on one "
                         "date); mixed folders are only reported")
    ap.add_argument("--no-yaml", action="store_true", help="rename only")
    ap.add_argument("--yaml-only", action="store_true",
                    help="no renames: write / fill the YAML of directories already "
                         "named by the schema")
    ap.add_argument("--log", type=Path, help="append applied renames to this TSV")
    ap.add_argument("--undo", type=Path, metavar="LOG", help="reverse an applied log")
    ap.add_argument("-q", "--quiet", action="store_true", help="only problems and totals")
    return ap.parse_args(argv)


def _yaml_only(root: Path, apply_: bool, out) -> int:
    dirs = []
    for d in sorted(p for p in root.rglob("*") if p.is_dir()):
        if parse_stem(d.name) is not None and not re.search(r"-C\d{2}-fc\d+$", d.name):
            dirs.append(d)
    if parse_stem(root.name) is not None:
        dirs.insert(0, root)
    for d in dirs:
        y = d / f"{d.name}.yaml"
        if apply_:
            out(f"{d.name}\n    " + write_yaml(d, None))
        else:
            out(f"{d.name}\n    yaml: would {'fill' if y.exists() else 'write'} {y.name}")
    out(f"\n{len(dirs)} session(s){'' if apply_ else ' (dry run; --apply to write)'}")
    return 0


def main(argv=None) -> int:
    a = parse_args(argv)
    out = print
    if a.undo:
        failed = undo(a.undo, dry_run=not a.apply, out=out)
        if not a.apply:
            out("\ndry run; --apply to undo")
        return 1 if failed else 0
    root = a.root.resolve()
    if a.yaml_only:
        return _yaml_only(root, a.apply, out)
    items = plan(root, prefix=a.prefix, subject=a.subject,
                 min_confidence=a.min_confidence, number_missing=a.number_missing)
    n_ok = sum(it.ok for it in items)
    for it in items:
        if a.quiet and it.ok and not it.warnings:
            continue
        conf = f"[{it.parsed.confidence}/{len(SCORED)}]" if it.parsed else ""
        out(f"{it.src.relative_to(root) if it.src != root else it.src.name}  {conf}")
        if it.dst is not None:
            out(f"    -> {it.dst.name}")
        p = it.parsed
        if p is not None:
            extra = [f"laserPower {p.laser_power}" if p.laser_power is not None else "",
                     f"fa {p.fa}" if p.fa is not None else "",
                     f"depth -{p.depth_um} um" if p.depth_um is not None else "",
                     f"gain {p.gain}" if p.gain is not None else "",
                     f"n_frames {p.n_frames}" if p.n_frames is not None else "",
                     f"side {p.side}" if p.side else "",
                     f"protocol '{' '.join(p.protocol)}'" if p.protocol else "",
                     f"sample_rate {p.sample_rate} Hz" if p.sample_rate is not None else "",
                     f"system {p.system}" if p.system is not None else ""]
            if any(extra):
                out("    yaml: " + ", ".join(e for e in extra if e))
        for w in it.warnings:
            out(f"    note: {w}")
        for pr in it.problems:
            out(f"    SKIP: {pr}")
    fixes, fnotes = plan_date_folders(items)
    if fixes or fnotes:
        out("\ndate folders:")
        for fx in fixes:
            out(f"  {fx.src.name}\n    -> {fx.dst.name}  ({fx.reason})"
                + ("" if a.fix_date_folders else "  [--fix-date-folders]"))
        for n in fnotes:
            out(f"  note: {n}")
    if not a.fix_date_folders:
        fixes = []
    out(f"\n{len(items)} legacy director{'y' if len(items) == 1 else 'ies'}: "
        f"{n_ok} to rename, {len(items) - n_ok} skipped"
        + (f"; {len(fixes)} date folder(s) to rename" if fixes else ""))
    if not a.apply:
        out("dry run; --apply to rename" + ("" if a.no_yaml else " and write the YAML"))
        return 0
    failed = apply(items, yaml=not a.no_yaml, log=a.log, out=out, folders=fixes)
    out(f"renamed {n_ok - failed}" + (f", {failed} failed" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
