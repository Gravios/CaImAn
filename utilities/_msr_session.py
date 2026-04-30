"""
utilities/_msr_session.py
=========================
MSR branch for new_session.py.

Bridges the organize_msr.py output (one .msr at the TL_dir level) to the
TIF-equivalent channel-subdir layout so batch_sessions can iterate it.

Input layout (output of organize_msr.py)::

    <TL_dir>/<TL_dir>.msr

Output layout (matches TIF convention)::

    <TL_dir>/<TL_dir>.yaml                                  (written by new_session)
    <TL_dir>/<TL_dir>-C<NN>-fc<NNNNNN>/
        <TL_dir>-C<NN>-fc<NNNNNN>.msr                       (moved/symlinked)

`-C<NN>-fc<NNNNNN>` is derived from MSR metadata: NN = channel index,
NNNNNN = frame count. For a multi-channel MSR, one subdir per channel is
created; the primary channel gets the original file, others get symlinks.

Public API
----------
setup_msr_session(msr_path, channels=None, dry_run=False) -> Path
    Returns the channel subdir of the primary (first) channel, which
    becomes `dest` for the rest of new_session.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

# imspectorreader must be importable from the caller's environment
# (same setup as organize_msr.py)


# Metadata-key candidates — extend with --inspect output once verified.
FRAME_KEYS   = ("NumberOfFrames", "NFrames", "TimeSteps", "TimeFrames",
                "NTime", "frame_count", "frames", "Frames",
                "NumberOfTimepoints", "n_frames")
CHANNEL_KEYS = ("NumberOfChannels", "NChannels", "Channels", "channels",
                "ChannelCount", "n_channels")


def _open_reader(path: Path):
    """Return (reader, metadata_dict). metadata_dict may be empty."""
    from imspectorreader import IMSpectorReader
    r = IMSpectorReader(str(path))
    md = next((getattr(r, a) for a in ("metadata", "meta", "info", "header")
               if isinstance(getattr(r, a, None), dict)), {})
    return r, md


def _parse_frame_count(reader, meta: dict) -> int:
    # Primary: direct attribute populated by IMSpectorReader._parse_file()
    for attr in ("slices_count", "size_z", "size_t"):
        v = getattr(reader, attr, None)
        if isinstance(v, int) and v > 0:
            return v
    # Fallback: scan metadata dict
    for k in FRAME_KEYS:
        if k in meta:
            try:
                return int(meta[k])
            except (TypeError, ValueError):
                continue
    raise ValueError(
        f"No frame count via reader.slices_count/size_z/size_t or metadata "
        f"(tried {FRAME_KEYS})"
    )


def _parse_channel_count(meta: dict) -> int:
    for k in CHANNEL_KEYS:
        if k in meta:
            try:
                return int(meta[k])
            except (TypeError, ValueError):
                continue
    return 1   # default: assume single channel if metadata is silent


def setup_msr_session(
    msr_path:  Path,
    channels:  list[int] | None = None,
    *,
    dry_run:   bool = False,
) -> Path:
    """Build the channel-subdir layout around an MSR file.

    Parameters
    ----------
    msr_path  : Path
        Absolute path to ``<TL_dir>/<TL_dir>.msr``.
    channels  : list[int] | None
        Channel indices to materialise. ``None`` → all channels detected
        in metadata (or [0] if metadata is silent).
    dry_run   : bool
        Print actions without touching disk.

    Returns
    -------
    Path
        The channel subdir for the primary (first) channel — this becomes
        ``dest`` for new_session.py's pipeline.py / pipeline.json output.
    """
    msr_path = msr_path.resolve()
    if msr_path.suffix.lower() != ".msr":
        raise ValueError(f"Not an MSR file: {msr_path}")
    if not msr_path.exists():
        raise FileNotFoundError(msr_path)

    tl_dir  = msr_path.parent
    tl_stem = tl_dir.name
    if msr_path.stem != tl_stem:
        raise ValueError(
            f"MSR stem {msr_path.stem!r} does not match TL_dir {tl_stem!r}. "
            f"Expected layout <TL_dir>/<TL_dir>.msr — run organize_msr.py first."
        )

    meta_reader, meta = _open_reader(msr_path)
    fc   = _parse_frame_count(meta_reader, meta)
    n_ch = _parse_channel_count(meta)

    if channels is None:
        channels = list(range(n_ch))
    else:
        bad = [c for c in channels if c < 0 or c >= n_ch]
        if bad:
            raise ValueError(
                f"Requested channels {bad} out of range; MSR has {n_ch} channels"
            )

    print(f"[msr] {msr_path.name}")
    print(f"[msr]   frames   = {fc}")
    print(f"[msr]   channels = {n_ch}  (using {channels})")

    primary_dir: Path | None = None
    for i, ch in enumerate(channels):
        ch_stem = f"{tl_stem}-C{ch:02d}-fc{fc:06d}"
        ch_dir  = tl_dir / ch_stem
        ch_msr  = ch_dir / f"{ch_stem}.msr"

        if ch_msr.exists():
            print(f"[msr]   exists   {ch_msr}")
        elif i == 0:
            print(f"[msr]   move     {msr_path.name} -> {ch_msr.relative_to(tl_dir)}")
            if not dry_run:
                ch_dir.mkdir(parents=True, exist_ok=True)
                msr_path.replace(ch_msr)
        else:
            target = (primary_dir / f"{primary_dir.name}.msr") if primary_dir else ch_msr
            print(f"[msr]   symlink  {ch_msr.relative_to(tl_dir)} -> "
                  f"{target.relative_to(tl_dir)}")
            if not dry_run:
                ch_dir.mkdir(parents=True, exist_ok=True)
                rel = Path("..") / target.parent.name / target.name
                if ch_msr.is_symlink() or ch_msr.exists():
                    ch_msr.unlink()
                ch_msr.symlink_to(rel)

        if primary_dir is None:
            primary_dir = ch_dir

    assert primary_dir is not None
    return primary_dir


# Allow direct CLI use: python _msr_session.py <msr_path> [--channels 0,1] [--dry-run]
def _cli(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 2)[1])
    ap.add_argument("msr", type=Path, help="Path to <TL_dir>/<TL_dir>.msr")
    ap.add_argument("--channels", default=None,
                    help="Comma-separated channel indices (default: all)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    chans = [int(c) for c in args.channels.split(",")] if args.channels else None
    primary = setup_msr_session(args.msr, channels=chans, dry_run=args.dry_run)
    print(f"\nprimary channel dir: {primary}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
