#!/usr/bin/env python3
"""
insert_caiman_defaults.py
─────────────────────────────────────────────────────────────────────────────
Insert missing recommended fields into acquisition YAML files.
Existing values are never overwritten — only absent keys are added.

Fields inserted per section
---------------------------
experiment:
  class       — recording class (e.g. spontaneous, stimulus, behavior)
  condition   — experimental condition (e.g. baseline, drug, recovery)

subject:
  sex         — M / F
  age         — age in weeks at time of recording
  genotype    — e.g. wild_type, Thy1-GCaMP6s
  weight      — body weight in grams

acquisition_system.settings:
  indicator   — calcium indicator (e.g. GCaMP6s, GCaMP6f, jGCaMP8s)
  brain_area  — target brain region (e.g. V1, S1, M1, CA1)
  wavelength:
    excitation: null   — excitation wavelength [nm]
    emission:   null   — emission wavelength [nm]

Idempotent: keys that already have a non-null value are left untouched.
Backs up original as .yaml.bak before writing.

Usage:
    python insert_caiman_defaults.py /data/source/strohA/strohA-ia/strohA-ia-000000-20140813
    python insert_caiman_defaults.py path/to/file.yaml --dry-run
"""

from __future__ import annotations
import argparse, shutil, sys
from pathlib import Path
import yaml


# ── Recommended fields per section ────────────────────────────────────────────

_EXPERIMENT_FIELDS = {
    'class':     None,   # spontaneous | stimulus | behavior | ...
    'condition': None,   # baseline | drug | recovery | ...
}

_SUBJECT_FIELDS = {
    'sex':      None,   # M | F
    'age':      None,   # weeks
    'genotype': None,   # wild_type | Thy1-GCaMP6s | ...
    'weight':   None,   # grams
}

_SETTINGS_FIELDS = {
    'indicator':  None,   # GCaMP6s | GCaMP6f | jGCaMP8s | ...
    'brain_area': None,   # V1 | S1 | M1 | CA1 | ...
    'wavelength': {
        'excitation': None,   # nm
        'emission':   None,   # nm
    },
}


def _insert_missing(target: dict, defaults: dict) -> int:
    """Insert keys from defaults that are absent or null in target.
    Returns count of keys added/set."""
    added = 0
    for k, v in defaults.items():
        if k not in target or target[k] is None:
            target[k] = v
            added += 1
        elif isinstance(v, dict) and isinstance(target[k], dict):
            added += _insert_missing(target[k], v)
    return added


def _process(path: Path, *, dry_run: bool, force: bool) -> str:
    try:
        text = path.read_text()
        doc  = yaml.safe_load(text)
    except Exception as e:
        print(f"  ERROR {path.name}: {e}"); return 'error'

    if not isinstance(doc, dict):
        return 'skipped'

    # Work on a copy for dry-run reporting
    import copy
    target = copy.deepcopy(doc)

    added = 0
    if 'experiment' in target:
        added += _insert_missing(target['experiment'], _EXPERIMENT_FIELDS)
    if 'subject' in target:
        added += _insert_missing(target['subject'], _SUBJECT_FIELDS)
    if 'acquisition_system' in target:
        settings = target['acquisition_system'].setdefault('settings', {})
        added += _insert_missing(settings, _SETTINGS_FIELDS)

    if added == 0 and not force:
        return 'skipped'

    if dry_run:
        preview = yaml.dump(target, default_flow_style=False,
                            allow_unicode=True, sort_keys=False)
        print(f"\n  ── {path.name}  ({added} field(s) added) ──")
        # Show only lines with null to keep output readable
        for line in preview.splitlines():
            if 'null' in line or any(
                    k in line for k in ('indicator', 'brain_area', 'wavelength',
                                        'sex:', 'age:', 'genotype', 'weight',
                                        'class:', 'condition:')):
                print(f"    {line}")
        return 'dry_run'

    shutil.copy2(path, path.with_suffix('.yaml.bak'))
    path.write_text(yaml.dump(target, default_flow_style=False,
                              allow_unicode=True, sort_keys=False))
    print(f"  {path.name}  ({added} field(s) added)")
    return 'updated'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('targets', nargs='+', metavar='PATH')
    p.add_argument('--dry-run', action='store_true',
        help='Preview without modifying files')
    p.add_argument('--force', action='store_true',
        help='Re-insert even if no fields are missing')
    args = p.parse_args(argv)

    yamls = []
    for t in args.targets:
        pt = Path(t)
        if pt.is_dir():
            yamls += sorted(y for y in pt.rglob('*.yaml')
                            if '.bak' not in y.suffixes
                            and not any(part.startswith('.') for part in y.parts))
        elif pt.is_file():
            yamls.append(pt)

    if not yamls:
        print("No YAML files found."); return 0

    print(f"{'DRY RUN — ' if args.dry_run else ''}{len(yamls)} file(s)\n")
    counts = {'updated': 0, 'skipped': 0, 'dry_run': 0, 'error': 0}
    for y in yamls:
        counts[_process(y, dry_run=args.dry_run, force=args.force)] += 1

    print()
    if args.dry_run:
        print(f"Would update {counts['dry_run']}, skip {counts['skipped']}")
    else:
        print(f"Updated {counts['updated']}, skipped {counts['skipped']}, errors {counts['error']}")
        if counts['updated']:
            print("Originals backed up as .yaml.bak")
    return 1 if counts['error'] else 0

if __name__ == '__main__':
    sys.exit(main())
