#!/usr/bin/env python3
"""
smaps_report.py — snapshot /proc/<pid>/smaps for a running CaImAn pipeline.

Usage:
    python smaps_report.py --quick          # top processes by swap, no PID needed
    python smaps_report.py <pid> [<pid2>]   # detailed per-process breakdown
    python smaps_report.py --all            # all python processes
"""

import sys
import os
import re
from pathlib import Path
from collections import defaultdict


def parse_smaps(pid):
    path = Path(f"/proc/{pid}/smaps")
    if not path.exists():
        return None, None
    regions = []
    current = {}
    try:
        for line in path.read_text().splitlines():
            m = re.match(r'^([0-9a-f]+)-([0-9a-f]+)\s+(\S+)\s+\S+\s+\S+\s+\S+\s*(.*)', line)
            if m:
                if current:
                    regions.append(current)
                current = {
                    'addr_start': int(m.group(1), 16),
                    'addr_end':   int(m.group(2), 16),
                    'perms':      m.group(3),
                    'name':       m.group(4).strip(),
                }
            elif ':' in line:
                key, _, val = line.partition(':')
                val = val.strip()
                if val.endswith(' kB'):
                    current[key.strip()] = int(val[:-3])
                else:
                    current[key.strip()] = val
        if current:
            regions.append(current)
    except (PermissionError, FileNotFoundError):
        return None, None
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b'\x00', b' ').decode(errors='replace')[:80]
    except Exception:
        cmd = f"pid {pid}"
    return regions, cmd


def categorise(name, perms):
    if not name:
        return 'anon'
    if '/dev/shm' in name:
        return 'dev/shm'
    if 'CAIMAN_TEMP' in name or '/data/caiman' in name or '.mmap' in name:
        return 'mmap-data'
    if name.endswith('.so') or '/lib/' in name or ('/anaconda3/' in name and '.so' in name):
        return 'shared-lib'
    if 'python' in name.lower() or '.py' in name:
        return 'python'
    if '[stack' in name:
        return 'stack'
    if '[heap]' in name:
        return 'heap'
    return 'other-file'


def quick_swap_check():
    """Show top processes by swap usage — no PID needed."""
    print("\n" + "="*60)
    print("QUICK SWAP CHECK — top processes by swap usage")
    print("="*60)
    procs = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            status = (p / 'status').read_text()
            vmswap = re.search(r'VmSwap:\s+(\d+)', status)
            vmrss  = re.search(r'VmRSS:\s+(\d+)',  status)
            cmdline = (p / 'cmdline').read_bytes().replace(b'\x00', b' ').decode(errors='replace')[:60]
            if vmswap:
                swap_kb = int(vmswap.group(1))
                rss_kb  = int(vmrss.group(1)) if vmrss else 0
                if swap_kb > 1024:
                    procs.append((swap_kb, rss_kb, int(p.name), cmdline))
        except Exception:
            pass
    procs.sort(key=lambda x: x[0], reverse=True)
    print(f"{'PID':>8}  {'RSS MB':>8}  {'Swap MB':>8}  Command")
    print("─"*60)
    for swap_kb, rss_kb, pid, cmd in procs[:20]:
        print(f"{pid:>8}  {rss_kb//1024:>8,}  {swap_kb//1024:>8,}  {cmd[:50]}")
    total = sum(s for s, _, _, _ in procs)
    print(f"\nTotal swap in top processes: {total//1024:,} MB")
    print(f"\nSystem swap summary:")
    os.system("cat /proc/swaps")
    print()
    os.system("free -h")


def report_pid(pid):
    regions, cmd = parse_smaps(pid)
    if regions is None:
        print(f"  PID {pid}: cannot read smaps (permission denied or no such process)")
        return 0

    totals = defaultdict(lambda: defaultdict(int))
    anon_dirty = []
    file_rss   = []

    for r in regions:
        cat = categorise(r.get('name', ''), r.get('perms', ''))
        for metric in ('Rss', 'Pss', 'Private_Dirty', 'Swap', 'Size'):
            totals[cat][metric] += r.get(metric, 0)
        pdirty = r.get('Private_Dirty', 0)
        swap   = r.get('Swap', 0)
        rss    = r.get('Rss', 0)
        if not r.get('name') and (pdirty + swap) > 1024:
            anon_dirty.append((pdirty + swap, pdirty, swap, r))
        if r.get('name') and rss > 1024:
            file_rss.append((rss, r))

    print(f"\n{'─'*70}")
    print(f"PID {pid}  {cmd}")
    print(f"{'─'*70}")

    cats = ['anon', 'heap', 'dev/shm', 'mmap-data', 'stack', 'shared-lib', 'python', 'other-file']
    print(f"{'Category':<16} {'RSS':>8} {'Private':>8} {'Swap':>8}  MB")
    print(f"{'─'*48}")
    grand_rss = grand_pdirty = grand_swap = 0
    for cat in cats:
        d = totals[cat]
        rss_mb    = d['Rss']           // 1024
        pdirty_mb = d['Private_Dirty'] // 1024
        swap_mb   = d['Swap']          // 1024
        if rss_mb + swap_mb > 1:
            print(f"  {cat:<14} {rss_mb:>7,}  {pdirty_mb:>7,}  {swap_mb:>7,}")
        grand_rss    += d['Rss']
        grand_pdirty += d['Private_Dirty']
        grand_swap   += d['Swap']
    print(f"  {'TOTAL':<14} {grand_rss//1024:>7,}  {grand_pdirty//1024:>7,}  {grand_swap//1024:>7,}")

    anon_dirty.sort(key=lambda x: x[0], reverse=True)
    if anon_dirty:
        print(f"\n  Top anonymous regions (dirty+swap > 1 MB):")
        print(f"  {'Addr':>36}  {'Size':>7}  {'Dirty':>7}  {'Swap':>7}  MB")
        for total_kb, pdirty, swap, r in anon_dirty[:15]:
            addr    = f"0x{r['addr_start']:x}-0x{r['addr_end']:x}"
            size_mb = (r['addr_end'] - r['addr_start']) // 2**20
            print(f"  {addr:>36}  {size_mb:>6,}  {pdirty//1024:>6,}  {swap//1024:>6,}")

    file_rss.sort(key=lambda x: x[0], reverse=True)
    if file_rss:
        print(f"\n  Top file-backed regions (RSS > 1 MB):")
        print(f"  {'File':<42}  {'RSS':>7}  MB")
        seen = set()
        count = 0
        for rss, r in file_rss:
            name = r.get('name', '')
            if name in seen:
                continue
            seen.add(name)
            print(f"  {name[-42:]:42}  {rss//1024:>6,}")
            count += 1
            if count >= 12:
                break

    # /dev/shm breakdown
    shm_agg = defaultdict(lambda: [0, 0])
    for r in regions:
        name = r.get('name', '')
        if '/dev/shm' in name:
            shm_agg[name][0] += r.get('Rss', 0)
            shm_agg[name][1] += r.get('Swap', 0)
    if shm_agg:
        print(f"\n  /dev/shm files (by swap):")
        print(f"  {'File':<50}  {'RSS':>7}  {'Swap':>7}  MB")
        for name, (rss_kb, swap_kb) in sorted(shm_agg.items(), key=lambda x: x[1][1], reverse=True)[:20]:
            if rss_kb + swap_kb < 64:
                continue
            print(f"  {name[-50:]:50}  {rss_kb//1024:>6,}  {swap_kb//1024:>6,}")

    return grand_swap // 1024


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ('--quick', '-q'):
        quick_swap_check()
        return

    if sys.argv[1] == '--all':
        pids = []
        for p in Path('/proc').iterdir():
            if p.name.isdigit():
                try:
                    if b'python' in (p / 'cmdline').read_bytes():
                        pids.append(int(p.name))
                except Exception:
                    pass
    else:
        pids = [int(a) for a in sys.argv[1:]]

    total_swap = 0
    for pid in sorted(pids):
        swap = report_pid(pid)
        if swap:
            total_swap += swap

    print(f"\n{'='*70}")
    print(f"TOTAL SWAP across reported processes: {total_swap:,} MB")
    print(f"\nSystem swap summary:")
    os.system("cat /proc/swaps")
    print()
    os.system("free -h")


if __name__ == '__main__':
    main()
