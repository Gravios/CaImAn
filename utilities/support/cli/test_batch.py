"""
``support test-batch`` — denoise a directory of TIFF stacks with a
single trained model.

Discovers ``*.tif*`` (by default) in ``--input-dir`` and writes
denoised versions to ``--output-dir`` with ``_denoised`` appended to
each stem. The model + sidecar are loaded once and reused.
"""

import argparse
import logging
import time
from pathlib import Path

import torch

from .paths import (default_output_path, resolve_checkpoint)
from .test import (_resolve_config, add_architecture_arguments,
                    add_inference_arguments, add_model_selection_arguments,
                    denoise_stack, load_model)


log = logging.getLogger("support.test_batch")


def add_arguments(p: argparse.ArgumentParser) -> None:
    """Populate an argparse subparser with batch-inference flags."""
    g = p.add_argument_group("batch IO")
    g.add_argument("--input-dir", type=Path, required=True,
                    help="directory of TIFFs to denoise (each one separately)")
    g.add_argument("--output-dir", type=Path, required=True,
                    help="directory for denoised outputs")
    g.add_argument("--pattern", type=str, default="*.tif*",
                    help="glob pattern for input discovery (default *.tif*)")
    add_model_selection_arguments(p)
    add_architecture_arguments(p, group_title=(
        "architecture overrides (only needed for checkpoints without "
        "sidecar JSON)"))
    add_inference_arguments(p)


def _discover(input_dir: Path, pattern: str) -> list[Path]:
    """Files matching ``pattern`` in ``input_dir`` (non-recursive),
    sorted, deduplicated, files only."""
    seen: set[Path] = set()
    out: list[Path] = []
    for p in sorted(input_dir.glob(pattern)):
        if p.is_file() and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def run(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s %(levelname)s %(message)s")

    in_dir = Path(args.input_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    if not in_dir.exists() or not in_dir.is_dir():
        log.error(f"--input-dir does not exist or is not a directory: {in_dir}")
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = _discover(in_dir, args.pattern)
    if not inputs:
        log.error(f"no files matching {args.pattern!r} in {in_dir}")
        return 2
    log.info(f"found {len(inputs)} input(s) under {in_dir}")

    cp = resolve_checkpoint(args.checkpoint, args.results_dir,
                             args.exp_name, args.epoch)
    log.info(f"checkpoint: {cp}")
    config = _resolve_config(cp, args)
    patch_size = args.patch_size if args.patch_size else config.patch_size

    use_cuda = torch.cuda.is_available() and not args.cpu
    model = load_model(cp, config, use_cuda)

    t0 = time.time()
    failed: list[tuple[Path, str]] = []
    for i, inp in enumerate(inputs, 1):
        log.info(f"[{i}/{len(inputs)}] {inp.name}")
        out = default_output_path(inp, output_dir=out_dir)
        try:
            denoise_stack(inp, out, model, list(patch_size),
                           list(args.patch_interval), args.batch_size,
                           args.include_edges)
        except (ValueError, RuntimeError) as e:
            # Continue on individual failures; report at the end.
            log.error(f"  failed on {inp.name}: {e}")
            failed.append((inp, str(e)))

    elapsed = time.time() - t0
    n_ok = len(inputs) - len(failed)
    log.info(f"batch done: {n_ok}/{len(inputs)} OK in {elapsed:.1f}s")
    if failed:
        log.error("the following inputs failed:")
        for p, msg in failed:
            log.error(f"  {p}: {msg}")
        return 1
    return 0
