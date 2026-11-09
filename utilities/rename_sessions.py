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
  date, time      leading YYMMDD (or YYYYMMDD) and trailing HH-MM-SS
  trialId         tl<N> → TL<NNN>; none → TL000 (or --number-missing)
  magnification   <N>x, also glued to a word (25xspont, vs25x) → <N>x; none → 00x
  experimentId    known words (spont, vstim, fpstim, …, see PARADIGMS); one
                  other unknown word is taken as is; none → default
  sourceId-userId from the nearest ancestor named <src>-<user>-<subj>[-<date>]
  subjectId       (e.g. strohA-ia-000000-20140225), or --prefix / --subject
  laser power     LP<N>[_<frac>], P<N>_<frac>   (LP5 → 5.0, p1_50 → 1.5) ┐
  frame averaging FA<N>                                                  ├ YAML
  depth           <N>um, depth<N>, z<N>   (stored negative, as stack-   │ only
                  sessions does: below the surface)                     ┘

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
  * --log writes a TSV (src, dst, confidence, unparsed); --undo LOG reverses
    an applied log.
  * Pipeline files (*_pipeline.json, *_results.hdf5, *.mmap) embed absolute
    paths a rename does not rewrite: they are listed as warnings.

Usage
-----
  rename-sessions --root /data/source/strohA/strohA-ia            # dry run
  rename-sessions --root … --apply --log rename.tsv
  rename-sessions --root … --min-confidence 4 --apply
  rename-sessions --root … --number-missing --apply   # TL by time for dirs without tl<N>
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

_B = r"(?<![A-Za-z0-9])"            # token start (separators are - _ .)
_E = r"(?![A-Za-z0-9])"             # token end
_TRIAL_RE = re.compile(_B + r"tl(\d+)" + _E, re.I)
_LP_RE    = re.compile(_B + r"(?:lp(\d+)(?:_(\d+))?|p(\d+)_(\d+))%?" + _E, re.I)
_FA_RE    = re.compile(_B + r"fa(\d+)" + _E, re.I)
_DEPTH_RE = re.compile(_B + r"(?:(\d+)um|depth[-_]?(\d+)|z(\d+))" + _E, re.I)
_MAG_RE   = re.compile(r"(?<![0-9])(\d+)x(?![0-9])", re.I)    # 25x, 25xspont, vs25x

# experimentId vocabulary (lower case → name token)
PARADIGMS = {
    "spont": "spont", "spontact": "spont", "spontaneous": "spont",
    "vs": "vstim", "vstim": "vstim", "stim": "stim", "fpstim": "fpstim",
    "astim": "astim", "chrimson": "chrimson", "opo": "opo", "tisa": "tisa",
    "3d": "3dscan", "3dscan": "3dscan", "loco": "loco", "dark": "dark",
    "default": "default",
}
DEFAULTS = {"subjectId": "000000", "trialId": "TL000",
            "magnification": "00x", "experimentId": "default"}
# fields that count towards the confidence score
SCORED = ("date", "time", "trialId", "magnification", "experimentId",
          "laser_power", "fa", "depth_um")

# ancestor carrying <src>-<user>-<subj>[-<date8>]
_ANCESTOR_RE = re.compile(r"^(?P<src>[^-_\s]+)-(?P<user>[^-_\s]+)-(?P<subj>[^-_\s]+)"
                          r"(?:-(?P<date>\d{8}))?$")


@dataclass
class Parsed:
    """A legacy directory name taken apart."""
    name: str
    parts: dict = field(default_factory=dict)       # schema fields (strings)
    laser_power: Optional[float] = None
    fa: Optional[int] = None
    depth_um: Optional[int] = None                   # positive µm below surface
    found: set = field(default_factory=set)          # SCORED fields actually read
    unparsed: list = field(default_factory=list)     # tokens left over

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

    def take(rx, fn):
        nonlocal body
        mm = rx.search(body)
        if not mm:
            return None
        body = body[:mm.start()] + " " + body[mm.end():]
        return fn(mm)

    tl = take(_TRIAL_RE, lambda mm: int(mm.group(1)))
    if tl is not None:
        p.parts["trialId"] = f"TL{tl:03d}"
        p.found.add("trialId")

    def lp(mm):
        whole, frac = (mm.group(1), mm.group(2)) if mm.group(1) else (mm.group(3), mm.group(4))
        return float(f"{int(whole)}.{frac}") if frac else float(whole)
    p.laser_power = take(_LP_RE, lp)
    p.fa = take(_FA_RE, lambda mm: int(mm.group(1)))
    p.depth_um = take(_DEPTH_RE, lambda mm: int(next(g for g in mm.groups() if g)))
    for key, v in (("laser_power", p.laser_power), ("fa", p.fa), ("depth_um", p.depth_um)):
        if v is not None:
            p.found.add(key)

    mag = take(_MAG_RE, lambda mm: int(mm.group(1)))
    if mag is not None:
        p.parts["magnification"] = f"{mag}x"
        p.found.add("magnification")

    words = [w for w in re.split(r"[-_.\s]+", body) if w]
    known = [PARADIGMS[w.lower()] for w in words if w.lower() in PARADIGMS]
    other = [w for w in words if w.lower() not in PARADIGMS]
    if known:
        p.parts["experimentId"] = known[0]
        p.found.add("experimentId")
        p.unparsed = other + [w for w in words if w.lower() in PARADIGMS][1:]
    elif len(other) == 1 and re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", other[0]):
        p.parts["experimentId"] = other[0].lower()      # e.g. a new paradigm
        p.found.add("experimentId")
    else:
        p.unparsed = other
    for k, v in DEFAULTS.items():
        p.parts.setdefault(k, v)
    return p


def ancestor_ids(path: Path) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """(sourceId, userId, subjectId, date) from the nearest ancestor named
    <src>-<user>-<subj>[-<date8>] that is not itself a session."""
    for a in path.parents:
        if parse_stem(a.name) is not None:
            continue
        m = _ANCESTOR_RE.match(a.name)
        if m:
            return m.group("src"), m.group("user"), m.group("subj"), m.group("date")
    return None, None, None, None


# ── Plan ─────────────────────────────────────────────────────────────────────

@dataclass
class Item:
    src: Path
    parsed: Optional[Parsed]
    dst: Optional[Path] = None
    problems: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

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
        src_id, user_id, subj_id, anc_date = ancestor_ids(d)
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
            it.warnings.append("not understood: " + " ".join(p.unparsed))

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
        doc.setdefault("experiment", {})["legacy_name"] = p.name
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
          out=print) -> int:
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
        if p is not None and (p.laser_power is not None or p.fa is not None or p.depth_um is not None):
            extra = [f"laserPower {p.laser_power}" if p.laser_power is not None else "",
                     f"fa {p.fa}" if p.fa is not None else "",
                     f"depth -{p.depth_um} um" if p.depth_um is not None else ""]
            out("    yaml: " + ", ".join(e for e in extra if e))
        for w in it.warnings:
            out(f"    note: {w}")
        for pr in it.problems:
            out(f"    SKIP: {pr}")
    out(f"\n{len(items)} legacy director{'y' if len(items) == 1 else 'ies'}: "
        f"{n_ok} to rename, {len(items) - n_ok} skipped")
    if not a.apply:
        out("dry run; --apply to rename" + ("" if a.no_yaml else " and write the YAML"))
        return 0
    failed = apply(items, yaml=not a.no_yaml, log=a.log, out=out)
    out(f"renamed {n_ok - failed}" + (f", {failed} failed" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
