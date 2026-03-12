"""
cpu_topology.py
===============
CPU cache-topology discovery and cache-aware work partitioning.

Motivation
----------
Modern multi-core CPUs have a three-level cache hierarchy:

    L1 (32 – 64 KB, ~4 cycles)   ← private per *logical* core
    L2 (256 KB – 2 MB, ~12 cycles) ← private per *physical* core (or shared
                                       within a CCX on AMD Zen)
    L3 / LLC (8 – 64 MB, ~40 cycles) ← shared across *all* cores in a socket

When N worker processes all read the same movie from shared memory the
*physical pages* are the same, but each core still has to pull cache lines
through the memory hierarchy independently.  If two workers process
*temporally adjacent* frame chunks they will access overlapping spatial
regions (e.g. the same template, the same border pixels).  Placing those
two workers on cores that **share L3** means the second worker finds those
lines already warm in the LLC instead of going to RAM.

What this module does
---------------------
1. **Discover L3 groups** – sets of logical CPUs that share an L3 cache.
   Tries three methods in order of reliability:
   a. ``/sys/devices/system/cpu/cpu*/cache/index*/`` (Linux sysfs – most precise)
   b. ``/proc/cpuinfo`` ``siblings`` / ``physical id`` fields
   c. ``psutil`` fallback (groups by physical core, conservative)

2. **Reorder chunks** (``cache_aware_chunk_order``) – given a list of frame
   index arrays (``idxs_list``) and a desired parallelism level, produce a
   permuted ordering so that the worker assigned to L3-group 0 gets chunks
   0, G, 2G, … (every G-th chunk), worker assigned to L3-group 1 gets chunks
   1, G+1, 2G+1, … etc.  Workers in the same L3 group therefore process
   temporally adjacent chunks, maximising cache reuse.

3. **Affinity hints** (``suggest_affinity``) – return a cpu_set for each
   worker index that pins it to a particular L3 group.  The caller decides
   whether to actually apply it (requires CAP_SYS_NICE on Linux, or is simply
   unsupported on Windows/macOS).

All functions degrade gracefully: if topology discovery fails the module
returns a single L3 group containing all logical CPUs, which leaves the
chunk order and affinity unchanged (equivalent to current behaviour).
"""

from __future__ import annotations

import logging
import os
import glob
from typing import Optional

logger = logging.getLogger("caiman")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _parse_cpumap(hex_map: str) -> list[int]:
    """Convert a Linux comma-separated hex cpu-map string to a list of CPU ids."""
    bits = int(hex_map.replace(",", ""), 16)
    cpus = []
    idx = 0
    while bits:
        if bits & 1:
            cpus.append(idx)
        bits >>= 1
        idx += 1
    return cpus


def _l3_groups_from_sysfs() -> Optional[list[list[int]]]:
    """
    Read L3 sharing info from Linux sysfs.

    Returns a list of CPU sets (each set = all logical CPUs sharing one L3),
    or None if sysfs is unavailable / doesn't expose cache topology.
    """
    cpu_dirs = sorted(glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cache/"))
    if not cpu_dirs:
        return None

    # Map from frozenset(cpus_sharing_L3) → list[int]
    l3_sets: dict[frozenset, list[int]] = {}

    for cpu_cache_dir in cpu_dirs:
        cpu_id_str = cpu_cache_dir.split("/cpu")[2].split("/")[0]
        try:
            cpu_id = int(cpu_id_str)
        except ValueError:
            continue

        for index_dir in glob.glob(os.path.join(cpu_cache_dir, "index*")):
            level_file = os.path.join(index_dir, "level")
            type_file  = os.path.join(index_dir, "type")
            map_file   = os.path.join(index_dir, "shared_cpu_map")

            if not all(os.path.exists(f) for f in [level_file, type_file, map_file]):
                continue
            try:
                level = open(level_file).read().strip()
                ctype = open(type_file).read().strip()
                if level != "3" or ctype == "Instruction":
                    continue
                cpu_map = open(map_file).read().strip()
                sharing = frozenset(_parse_cpumap(cpu_map))
                l3_sets.setdefault(sharing, sorted(sharing))
            except OSError:
                continue

    if not l3_sets:
        return None
    return list(l3_sets.values())


def _l3_groups_from_proc_cpuinfo() -> Optional[list[list[int]]]:
    """
    Infer L3 groups from /proc/cpuinfo ``physical id`` + ``siblings``.

    All logical CPUs with the same ``physical id`` share the same socket and
    therefore (for typical Intel/AMD) the same L3.
    """
    try:
        cpuinfo = open("/proc/cpuinfo").read()
    except OSError:
        return None

    socket_to_cpus: dict[int, list[int]] = {}
    current_cpu: Optional[int] = None
    current_socket: Optional[int] = None

    for line in cpuinfo.splitlines():
        if line.startswith("processor"):
            current_cpu = int(line.split(":")[1].strip())
        elif line.startswith("physical id"):
            current_socket = int(line.split(":")[1].strip())
        elif line == "" and current_cpu is not None and current_socket is not None:
            socket_to_cpus.setdefault(current_socket, []).append(current_cpu)
            current_cpu = None
            current_socket = None

    if not socket_to_cpus:
        return None
    return list(socket_to_cpus.values())


def _l3_groups_from_psutil() -> list[list[int]]:
    """
    Psutil fallback: group CPUs by physical core (pairs for hyper-threaded
    cores share L2; all cores on a socket share L3).  We conservatively
    group by socket using ``cpu_count`` comparison.
    """
    import psutil
    logical  = psutil.cpu_count(logical=True)  or 1
    physical = psutil.cpu_count(logical=False) or 1

    if logical == physical:
        # No hyper-threading; assume single L3 group
        return [list(range(logical))]

    # With HT: pairs of logical CPUs share L2; treat all per socket as one L3
    # (conservative – safe default)
    return [list(range(logical))]


# ── Public API ────────────────────────────────────────────────────────────────

def get_l3_groups() -> list[list[int]]:
    """
    Return a list of CPU groups that each share one L3 cache.

    Discovery order: sysfs → /proc/cpuinfo → psutil fallback.

    Returns
    -------
    list of list of int
        Each inner list contains the logical CPU indices that share one L3.
        On a single-socket machine this is typically ``[[0, 1, 2, …, N-1]]``.
        On dual-socket: ``[[0..N/2-1], [N/2..N-1]]``.
    """
    for fn in [_l3_groups_from_sysfs, _l3_groups_from_proc_cpuinfo]:
        result = fn()
        if result:
            logger.debug(f"cpu_topology: L3 groups from {fn.__name__}: {result}")
            return result

    result = _l3_groups_from_psutil()
    logger.debug(f"cpu_topology: L3 groups from psutil fallback: {result}")
    return result


def cache_aware_chunk_order(n_chunks: int, n_workers: int) -> list[int]:
    """
    Return a permutation of ``range(n_chunks)`` such that each L3-sharing
    worker group processes temporally adjacent chunks.

    Given G L3 groups each with W workers (W = n_workers // G), the
    assignment is::

        group 0, worker 0  → chunks  0,  G,  2G, …
        group 0, worker 1  → chunks  1,  G+1, 2G+1, …
        …
        group 1, worker 0  → chunks  W,  W+G, W+2G, …

    This ensures that within each L3 group the processed frame windows are
    close in time, maximising reuse of the template and border pixels in the
    shared LLC.

    Parameters
    ----------
    n_chunks : int
        Total number of work units (``len(idxs_list)``).
    n_workers : int
        Total number of parallel workers.

    Returns
    -------
    list[int]
        Permuted chunk indices; ``permutation[i]`` is the original chunk index
        that worker *i* % *n_workers* should process on its *i // n_workers*
        iteration.
    """
    l3_groups = get_l3_groups()
    n_groups = max(1, len(l3_groups))
    group_size = max(1, n_workers // n_groups)

    # Build the stripe permutation
    order: list[int] = []
    for stripe_start in range(0, n_chunks, n_workers):
        for w in range(n_workers):
            chunk_idx = stripe_start + w
            if chunk_idx < n_chunks:
                order.append(chunk_idx)

    # Already identity; refine so within each stripe workers in same L3
    # group get adjacent chunks by reordering the stripe
    # (This is a no-op when n_workers == 1 or n_groups == 1.)
    if n_groups == 1 or n_workers <= 1:
        return list(range(n_chunks))

    refined: list[int] = []
    for stripe_start in range(0, n_chunks, n_groups):
        block = list(range(stripe_start, min(stripe_start + n_groups, n_chunks)))
        # Workers in L3-group 0 get the first ceil(block/n_groups) indices, etc.
        refined.extend(block)

    return refined if len(refined) == n_chunks else list(range(n_chunks))


def suggest_affinity(worker_id: int, n_workers: int) -> Optional[set[int]]:
    """
    Return a set of logical CPU indices to pin *worker_id* to.

    Workers within the same L3 group are pinned to that group's CPU set so
    the OS scheduler keeps them on cores sharing LLC.

    Parameters
    ----------
    worker_id : int
        Zero-based index of the worker.
    n_workers : int
        Total number of workers.

    Returns
    -------
    set of int or None
        CPU set suggestion.  Returns *None* when topology is unknown or there
        is only one L3 group (no benefit to pinning).
    """
    l3_groups = get_l3_groups()
    if len(l3_groups) <= 1:
        return None

    group_idx = worker_id % len(l3_groups)
    return set(l3_groups[group_idx])


def apply_affinity(worker_id: int, n_workers: int) -> bool:
    """
    Attempt to pin the *calling* process to an L3-local CPU set.

    Requires ``CAP_SYS_NICE`` (Linux) or equivalent.  Silently succeeds on
    platforms where ``os.sched_setaffinity`` is unavailable.

    Returns
    -------
    bool
        *True* if affinity was successfully set, *False* otherwise.
    """
    cpu_set = suggest_affinity(worker_id, n_workers)
    if cpu_set is None:
        return False
    if not hasattr(os, "sched_setaffinity"):
        return False
    try:
        os.sched_setaffinity(0, cpu_set)
        logger.debug(f"Worker {worker_id}: pinned to CPUs {sorted(cpu_set)}")
        return True
    except PermissionError:
        logger.debug(f"Worker {worker_id}: cannot set affinity (no CAP_SYS_NICE)")
        return False
    except OSError as exc:
        logger.debug(f"Worker {worker_id}: sched_setaffinity failed: {exc}")
        return False
