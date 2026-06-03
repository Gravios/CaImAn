"""
``support test`` — denoise a single TIFF stack.

Two ways to specify the model:

    1. ``--checkpoint /path/to/model_N.pth``
       Explicit. Loads architecture from the JSON sidecar next to it
       (or falls back to ``--architecture`` flags if no sidecar — for
       shipped pretrained models).

    2. ``--exp-name NAME --results-dir DIR [--epoch N]``
       Auto-find the checkpoint. Defaults to the latest epoch.

Examples
--------
    # Smoke-test on shipped pretrained model:
    python -m src.cli test \\
        --checkpoint ./src/GUI/trained_models/bs3.pth \\
        --bs-size 3 3 \\
        --input /data/.../recording.tif \\
        --output /data/.../recording_denoised.tif

    # After training (auto-loads architecture from sidecar):
    python -m src.cli test \\
        --exp-name strohA_sa_82 \\
        --results-dir /data/support/results \\
        --input /data/.../recording.tif
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
import skimage.io as skio
from tqdm import tqdm

from ..dataset import DatasetSUPPORT_test_stitch
from ..network import SUPPORT

from .paths import (ModelConfig, add_architecture_arguments,
                     default_output_path, resolve_checkpoint,
                     resolve_input_paths)


log = logging.getLogger("support.test")


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def add_model_selection_arguments(p: argparse.ArgumentParser) -> None:
    """Add the ``--checkpoint`` / ``--exp-name`` lookup flags. Shared
    between ``test`` and ``test-batch``."""
    g = p.add_argument_group("model selection (one method required)")
    g.add_argument("--checkpoint", type=Path, default=None,
                    help="explicit checkpoint .pth")
    g.add_argument("--exp-name", type=str, default=None,
                    help="experiment name (auto-finds latest checkpoint)")
    g.add_argument("--results-dir", type=Path, default=Path("./results"),
                    help="results root for --exp-name lookup")
    g.add_argument("--epoch", type=int, default=None,
                    help="specific epoch (default: latest)")


def add_inference_arguments(p: argparse.ArgumentParser) -> None:
    """Add the inference-specific flags (patch size, batch, edges)
    common to ``test`` and ``test-batch``."""
    g = p.add_argument_group("inference")
    # patch_size default is None so we can fall back to the sidecar
    # value when the user doesn't override.
    g.add_argument("--patch-size", type=int, nargs=3, default=None,
                    help="patch [T X Y] (default: sidecar value or "
                         "61 64 64 if no sidecar)")
    g.add_argument("--patch-interval", type=int, nargs=3,
                    default=[1, 32, 32],
                    help="patch stride [T X Y]")
    g.add_argument("--batch-size", type=int, default=8)
    g.add_argument("--cpu", action="store_true",
                    help="force CPU inference")
    g.add_argument("--include-edges", choices=["none", "repeat", "mirror"],
                    default="none",
                    help="how to handle frames within input-frames/2 of "
                         "the stack edges. Default 'none' drops them.")


def add_arguments(p: argparse.ArgumentParser) -> None:
    """Populate an argparse subparser with the test-mode flags."""
    g_in = p.add_argument_group("input")
    g_in.add_argument("--input", "-i", type=Path, required=True,
                       help="path to noisy TIFF stack")
    g_in.add_argument("--output", "-o", type=Path, default=None,
                       help="output TIFF path (default: <stem>_denoised.tif "
                            "alongside input)")
    add_model_selection_arguments(p)
    add_architecture_arguments(p, group_title=(
        "architecture overrides (only needed for checkpoints without "
        "sidecar JSON)"))
    add_inference_arguments(p)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_config(checkpoint: Path,
                     args: argparse.Namespace) -> ModelConfig:
    """Pick a ModelConfig for the given checkpoint: sidecar JSON if
    present, otherwise build from CLI architecture flags."""
    cfg = ModelConfig.load_for_checkpoint(checkpoint)
    if cfg is not None:
        log.info(f"loaded architecture from sidecar: "
                  f"{ModelConfig.sidecar_for(checkpoint)}")
        if cfg.training_data:
            log.info(f"  model was trained on: {cfg.training_data}")
        return cfg
    log.warning(f"no sidecar JSON next to {checkpoint} — "
                 "using --architecture flags")
    return ModelConfig.from_namespace(args)


def _expand_edges(stack: torch.Tensor, mode: str, pad: int) -> torch.Tensor:
    """Pad the temporal axis so the network produces output for the
    first/last ``pad`` frames too. Caller crops after inference.

    Returns the input unchanged when ``mode=='none'`` or ``pad<=0``.
    Raises ``ValueError`` for unknown modes or when the stack is too
    short for the requested mirror padding.
    """
    if pad <= 0:
        return stack
    T = stack.shape[0]
    match mode:
        case "none":
            return stack
        case "repeat":
            head = stack[0:1].repeat((pad, 1, 1))
            tail = stack[-1:].repeat((pad, 1, 1))
        case "mirror":
            if pad >= T:
                raise ValueError(
                    f"--include-edges=mirror requires T >= pad+1 "
                    f"(got T={T}, pad={pad}); use --include-edges=repeat "
                    "or drop the flag")
            head = stack[1:pad + 1].flip(0)
            tail = stack[-pad - 1:-1].flip(0)
        case _:
            # argparse 'choices' restricts mode to {none,repeat,mirror};
            # if we reach this branch the parser was bypassed.
            raise ValueError(f"unknown --include-edges mode {mode!r}")
    return torch.cat([head, stack, tail], dim=0)


def load_model(checkpoint: Path, config: ModelConfig,
                use_cuda: bool) -> SUPPORT:
    """Build, move, and load weights for a SUPPORT model."""
    model = SUPPORT(**config.to_model_kwargs())
    if use_cuda:
        model = model.cuda()
    model.load_state_dict(
        torch.load(checkpoint, map_location="cuda" if use_cuda else "cpu"))
    model.eval()
    return model


def denoise_array(stack: np.ndarray, model: SUPPORT,
                   patch_size: list[int], patch_interval: list[int],
                   batch_size: int,
                   include_edges: str = "none") -> np.ndarray:
    """Core inference: take an in-memory (T, H, W) numpy array, return
    the denoised float32 array. No file I/O.

    Use this when the caller already has the data in memory — e.g. the
    pipeline integration that operates on the F→C-converted mmap and
    wants to skip the TIFF round-trip.

    Length semantics:
      - ``include_edges='none'``: output has ``T - 2*pad`` frames.
      - ``include_edges='repeat'`` / ``'mirror'``: output preserves
        input ``T`` via temporal padding with replicated or mirrored
        boundary frames.
    """
    stack = np.asarray(stack, dtype=np.float32)
    if stack.ndim == 2:
        stack = stack[None, ...]
    if stack.ndim != 3:
        raise ValueError(f"expected 2- or 3-D array; got shape {stack.shape}")

    pad = (model.in_channels - 1) // 2
    if stack.shape[0] < model.in_channels:
        raise ValueError(
            f"stack has {stack.shape[0]} frames but model needs at least "
            f"{model.in_channels} per temporal window; cannot denoise")

    t = torch.from_numpy(stack)
    t = _expand_edges(t, include_edges, pad)

    ds = DatasetSUPPORT_test_stitch(t, patch_size=patch_size,
                                     patch_interval=patch_interval)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size)

    on_cuda = next(model.parameters()).is_cuda

    denoised = np.zeros(ds.noisy_image.shape, dtype=np.float32)
    with torch.no_grad():
        for noisy, _, coords in tqdm(loader, desc="denoise"):
            if on_cuda:
                noisy = noisy.cuda()
            out = model(noisy)
            T = noisy.size(1)
            for bi in range(noisy.size(0)):
                sw = int(coords["stack_start_w"][bi])
                ew = int(coords["stack_end_w"][bi])
                pw = int(coords["patch_start_w"][bi])
                pew = int(coords["patch_end_w"][bi])
                sh = int(coords["stack_start_h"][bi])
                eh = int(coords["stack_end_h"][bi])
                ph = int(coords["patch_start_h"][bi])
                peh = int(coords["patch_end_h"][bi])
                ss = int(coords["init_s"][bi])
                denoised[ss + T // 2, sh:eh, sw:ew] = (
                    out[bi].squeeze()[ph:peh, pw:pew].cpu())

    # Denormalize using the global mean/std the Dataset captured
    denoised = denoised * ds.std_image.numpy() + ds.mean_image.numpy()

    # Crop edge padding (pad>0 in 3D mode); see docstring length semantics
    if pad > 0:
        denoised = denoised[pad:-pad]

    return denoised


def denoise_stack(input_path: Path, output_path: Path, model: SUPPORT,
                   patch_size: list[int], patch_interval: list[int],
                   batch_size: int, include_edges: str = "none") -> Path:
    """TIFF wrapper around :func:`denoise_array` — read input from disk,
    run inference, write output TIFF. Returns the output path.

    Length semantics: see :func:`denoise_array`.
    """
    log.info(f"loading {input_path}")
    stack = skio.imread(str(input_path)).astype(np.float32)
    log.info(f"  shape={stack.shape} dtype={stack.dtype} "
              f"range={stack.min():.1f}-{stack.max():.1f}")

    denoised = denoise_array(stack, model, patch_size, patch_interval,
                              batch_size, include_edges)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    skio.imsave(str(output_path), denoised, metadata={"axes": "TYX"})
    log.info(f"wrote {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")
    return output_path


def denoise_mmap(input_mmap_path: Path, output_mmap_path: Path,
                  model: SUPPORT, dims: tuple[int, int], T: int,
                  patch_size: list[int], patch_interval: list[int],
                  batch_size: int, include_edges: str = "mirror") -> Path:
    """C-order mmap wrapper around :func:`denoise_array`. Reads from
    CaImAn's ``(n_px, T)`` C-order mmap, denoises, writes a new mmap.

    Parameters
    ----------
    input_mmap_path
        Existing C-order mmap (e.g. CaImAn's ``*_cnmf_*_order_C_*.mmap``).
    output_mmap_path
        Where to write the denoised mmap. Created in mode ``w+``.
    model
        Pre-loaded SUPPORT network.
    dims, T
        Spatial ``(d1, d2)`` and frame count, as returned by
        ``cm.mmapping.load_memmap``.
    include_edges
        Default ``'mirror'`` here (vs ``'none'`` for the TIFF wrapper)
        so the output mmap has exactly ``T`` frames and is drop-in
        compatible with downstream stages that expect the same frame
        count. ``'none'`` would write a shorter mmap (``T - 2*pad``)
        which would force ``T`` to change in downstream params.

    Returns
    -------
    Path to the new mmap.
    """
    n_px = int(np.prod(dims))
    d1, d2 = dims

    log.info(f"loading mmap {input_mmap_path}")
    Yr_in = np.memmap(input_mmap_path, mode="r", dtype=np.float32,
                       shape=(n_px, T), order="C")
    log.info(f"  Yr_in shape={Yr_in.shape} dtype={Yr_in.dtype}")

    # Reshape to (T, d1, d2). The reshape view is F-order over (T, n_px)
    # then re-interpreted as (T, d1, d2) F-order. Each images[t] is a
    # strided 2-D view into the mmap; copying to a contiguous array up
    # front avoids cache-miss penalties during patch sampling.
    log.info(f"  copying mmap to contiguous (T, d1, d2) RAM buffer "
              f"({4 * n_px * T / 1e9:.2f} GB)")
    images_view = np.reshape(Yr_in.T, (T, d1, d2), order="F")
    images = np.ascontiguousarray(images_view, dtype=np.float32)
    del images_view, Yr_in

    # Run the denoiser
    denoised = denoise_array(images, model, patch_size, patch_interval,
                              batch_size, include_edges)
    log.info(f"  denoised shape={denoised.shape} "
              f"range={denoised.min():.1f}-{denoised.max():.1f}")

    if denoised.shape[0] != T:
        log.warning(
            f"output has {denoised.shape[0]} frames (input had {T}). "
            "Downstream stages query T from the new mmap and will "
            "adapt, but params recorded in the JSON config may now be "
            "stale. Consider include_edges='mirror' to preserve T.")
        T_out = denoised.shape[0]
    else:
        T_out = T

    # Write the new C-order mmap (n_px, T_out)
    output_mmap_path.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"writing mmap {output_mmap_path}  shape=({n_px}, {T_out})")
    Yr_out = np.memmap(output_mmap_path, mode="w+", dtype=np.float32,
                        shape=(n_px, T_out), order="C")
    # denoised is (T_out, d1, d2) C-order; we need (n_px, T_out) C-order.
    # Equivalent to reshape (T_out, n_px) F-order, then transpose.
    denoised_flat_F = np.reshape(denoised, (T_out, n_px), order="F")
    Yr_out[:] = denoised_flat_F.T
    Yr_out.flush()
    del Yr_out, denoised, denoised_flat_F

    log.info(f"wrote {output_mmap_path} "
              f"({output_mmap_path.stat().st_size / 1e9:.2f} GB)")
    return output_mmap_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s %(levelname)s %(message)s")

    cp = resolve_checkpoint(args.checkpoint, args.results_dir,
                             args.exp_name, args.epoch)
    log.info(f"checkpoint: {cp}")
    inp = resolve_input_paths([args.input])[0]
    out = args.output or default_output_path(inp)

    config = _resolve_config(cp, args)
    # Sidecar wins for patch_size unless user explicitly overrode
    patch_size = args.patch_size if args.patch_size else config.patch_size

    use_cuda = torch.cuda.is_available() and not args.cpu
    model = load_model(cp, config, use_cuda)

    t0 = time.time()
    denoise_stack(inp, out, model, list(patch_size),
                   list(args.patch_interval), args.batch_size,
                   args.include_edges)
    log.info(f"done in {time.time() - t0:.1f}s")
    return 0
