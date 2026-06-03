"""
``support train`` — self-supervised training of a SUPPORT model.

Wraps the algorithmic training loop with:

* pathlib-based path handling
* architecture sidecar JSON saved with each checkpoint (so inference
  doesn't need to re-specify ``--unet-channels`` etc.)
* sensible defaults that match the shipped ``bs3.pth`` architecture
* mixed-precision on by default (disable with ``--no-amp``)

Example
-------
    python -m src.cli train \\
        --noisy-data /data/.../recording.tif \\
        --exp-name strohA_sa_82 \\
        --results-dir /data/support/results \\
        --n-epochs 30
"""

import argparse
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Algorithmic code is reused unchanged from the upstream repo
from ..dataset import gen_train_dataloader, random_transform
from ..network import SUPPORT

from .paths import (ModelConfig, add_architecture_arguments,
                     checkpoint_path, experiment_dir)


log = logging.getLogger("support.train")


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def add_arguments(p: argparse.ArgumentParser) -> None:
    """Populate an argparse subparser with the train-mode flags."""
    g_data = p.add_argument_group("data")
    g_data.add_argument("--noisy-data", type=Path, nargs="+", required=True,
                         help="path(s) to noisy TIFF stack(s) to train on")
    g_data.add_argument("--exp-name", type=str, required=True,
                         help="experiment name; outputs land under "
                              "<results-dir>/saved_models/<exp-name>/")
    g_data.add_argument("--results-dir", type=Path, default=Path("./results"),
                         help="root for checkpoints, logs, tensorboard")

    # Architecture flags shared with `test` — single source of truth
    add_architecture_arguments(p, group_title="architecture")

    g_train = p.add_argument_group("training")
    g_train.add_argument("--patch-size", type=int, nargs=3,
                          default=[61, 64, 64],
                          help="patch [T X Y] (default 61 64 64); "
                               "patch_size[0] must equal --input-frames")
    g_train.add_argument("--patch-interval", type=int, nargs=3,
                          default=[1, 32, 32],
                          help="patch stride [T X Y] (default 1 32 32)")
    g_train.add_argument("--batch-size", type=int, default=8)
    g_train.add_argument("--n-epochs", type=int, default=30)
    g_train.add_argument("--lr", type=float, default=5e-4)
    g_train.add_argument("--loss-coef", type=float, nargs=2,
                          default=[0.5, 0.5],
                          help="L1, L2 loss coefficients (default 0.5 0.5)")
    g_train.add_argument("--start-epoch", type=int, default=0,
                          help="resume from this epoch (loads epoch-1 "
                               "model + sidecar)")
    g_train.add_argument("--random-seed", type=int, default=0)
    g_train.add_argument("--no-amp", action="store_true",
                          help="disable mixed precision (on by default)")
    g_train.add_argument("--cpu", action="store_true",
                          help="force CPU training (slow)")

    g_io = p.add_argument_group("io")
    g_io.add_argument("--n-cpu", type=int, default=8,
                       help="dataloader workers")
    g_io.add_argument("--prefetch-factor", type=int, default=2)
    g_io.add_argument("--checkpoint-interval", type=int, default=1,
                       help="epochs between checkpoint saves (-1 disables "
                            "intermediate saves; final epoch always saved)")
    g_io.add_argument("--checkpoint-interval-batch", type=int, default=10000,
                       help="batches between mid-epoch checkpoint saves")
    g_io.add_argument("--logging-interval", type=int, default=1,
                       help="epochs between logging stats")
    g_io.add_argument("--logging-interval-batch", type=int, default=50)
    g_io.add_argument("--is-zarr", action="store_true",
                       help="input data is a zarr array (out-of-core)")
    g_io.add_argument("--is-folder", action="store_true",
                       help="input data is a directory of TIFFs")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _train_one_epoch(loader, model, optimizer, scaler, rng, writer,
                      epoch: int, args: argparse.Namespace
                      ) -> tuple[list[float], list[float], list[float]]:
    """One pass over the training data."""
    L1 = torch.nn.L1Loss()
    L2 = torch.nn.MSELoss()
    is_rotate = (model.bs_size[0] == model.bs_size[1])
    use_amp = not args.no_amp

    model.train()
    losses: list[float] = []
    l1_losses: list[float] = []
    l2_losses: list[float] = []

    for i, data in enumerate(tqdm(loader, desc=f"epoch {epoch}")):
        if args.is_zarr:
            noisy, _, _, noisy_avg, noisy_std = data
            noisy_avg = noisy_avg.reshape(-1, 1, 1, 1).cuda()
            noisy_std = noisy_std.reshape(-1, 1, 1, 1).cuda()
        else:
            noisy, _, _ = data

        _, T, _, _ = noisy.shape
        noisy = noisy.cuda()
        noisy, _ = random_transform(noisy, None, rng, is_rotate)
        if args.is_zarr:
            noisy = (noisy - noisy_avg) / noisy_std
        target = noisy[:, T // 2, :, :].unsqueeze(1)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_amp):
            denoised = model(noisy)
            l1 = L1(denoised, target)
            l2 = L2(denoised, target)
            loss = args.loss_coef[0] * l1 + args.loss_coef[1] * l2

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())
        l1_losses.append(l1.item())
        l2_losses.append(l2.item())

        if (epoch % args.logging_interval == 0
                and i % args.logging_interval_batch == 0):
            m = float(np.mean(losses))
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            step = epoch * len(loader) + i
            writer.add_scalar("Loss/train_batch", m, step)
            writer.add_scalar("Loss_l1/train_batch",
                               float(np.mean(l1_losses)), step)
            writer.add_scalar("Loss_l2/train_batch",
                               float(np.mean(l2_losses)), step)
            log.info(f"[{ts}] epoch [{epoch}/{args.n_epochs}] "
                      f"batch [{i + 1}/{len(loader)}] loss={m:.4f}")

    return losses, l1_losses, l2_losses


def _save_checkpoint(model, optimizer, scaler, args: argparse.Namespace,
                      config: ModelConfig, epoch: int) -> None:
    """Save state_dict + JSON sidecar (sidecar last so a partial write
    leaves the sidecar absent rather than stale)."""
    cp = checkpoint_path(args.results_dir, args.exp_name, epoch)
    torch.save(model.state_dict(), cp)
    torch.save(optimizer.state_dict(), cp.with_name(f"optimizer_{epoch}.pth"))
    if not args.no_amp:
        torch.save(scaler.state_dict(), cp.with_name(f"scaler_{epoch}.pth"))
    config.epoch = epoch
    config.save(ModelConfig.sidecar_for(cp))
    log.info(f"saved checkpoint: {cp} (+ sidecar)")


def _load_resume_state(model, optimizer, scaler, args: argparse.Namespace
                        ) -> ModelConfig | None:
    """When resuming with ``--start-epoch N > 0``, load the epoch-(N-1)
    checkpoint, its optimizer/scaler, and (if available) its sidecar
    config. Returns the loaded ModelConfig or None.

    Raises ValueError if the user passed CLI architecture flags that
    don't match the sidecar — sidecar wins, but the mismatch is loud."""
    prev = args.start_epoch - 1
    prev_cp = checkpoint_path(args.results_dir, args.exp_name, prev)
    if not prev_cp.exists():
        raise FileNotFoundError(
            f"--start-epoch={args.start_epoch} requested but no "
            f"checkpoint at {prev_cp}")

    model.load_state_dict(torch.load(prev_cp))
    log.info(f"resumed model from {prev_cp}")

    opt_path = prev_cp.with_name(f"optimizer_{prev}.pth")
    if opt_path.exists():
        optimizer.load_state_dict(torch.load(opt_path))
        log.info(f"resumed optimizer from {opt_path}")

    if not args.no_amp:
        scaler_path = prev_cp.with_name(f"scaler_{prev}.pth")
        if scaler_path.exists():
            scaler.load_state_dict(torch.load(scaler_path))
            log.info(f"resumed AMP scaler from {scaler_path}")

    return ModelConfig.load_for_checkpoint(prev_cp)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    rng = np.random.default_rng(args.random_seed)

    # Resolve paths
    args.noisy_data = [Path(p).expanduser().resolve() for p in args.noisy_data]
    for p in args.noisy_data:
        if not p.exists():
            raise FileNotFoundError(f"--noisy-data path missing: {p}")
    args.results_dir = Path(args.results_dir).expanduser().resolve()

    # Directory layout
    exp_dir = experiment_dir(args.results_dir, args.exp_name)
    (args.results_dir / "logs").mkdir(parents=True, exist_ok=True)
    tb_dir = args.results_dir / "tsboard" / args.exp_name
    tb_dir.mkdir(parents=True, exist_ok=True)

    # File + stderr logging
    log_file = args.results_dir / "logs" / f"{args.exp_name}.log"
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.setLevel(logging.INFO)
    log.addHandler(file_handler)
    log.addHandler(stream_handler)
    log.propagate = False
    log.info("=== train start ===")
    log.info(f"args: {vars(args)}")
    log.info(f"checkpoints -> {exp_dir}")
    log.info(f"tensorboard -> {tb_dir}")

    writer = SummaryWriter(str(tb_dir))

    # Sanity checks
    if args.input_frames != args.patch_size[0]:
        raise ValueError(
            f"--input-frames ({args.input_frames}) must equal "
            f"patch_size[0] ({args.patch_size[0]})")

    cuda = torch.cuda.is_available() and not args.cpu
    if not cuda:
        log.warning("training on CPU — this will be very slow")

    # Data loader (gen_train_dataloader expects str paths and a namespace
    # exposing n_cpu, prefetch_factor, is_zarr)
    loader = gen_train_dataloader(
        list(args.patch_size), list(args.patch_interval),
        args.batch_size, [str(p) for p in args.noisy_data],
        args, is_zarr=args.is_zarr,
    )

    # Build the architecture config from CLI args (single source of truth)
    config = ModelConfig.from_namespace(
        args, training_data=[str(p) for p in args.noisy_data])

    # Construct the model from the config
    model = SUPPORT(**config.to_model_kwargs())
    if cuda:
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=not args.no_amp)

    # Resume?
    if args.start_epoch > 0:
        existing_config = _load_resume_state(model, optimizer, scaler, args)
        if existing_config is not None:
            # Warn loudly on architecture drift between resume target and
            # current CLI args. We KEEP the user's CLI args (they may
            # have intentionally changed something benign like notes),
            # but the user should know.
            cli_kwargs = config.to_model_kwargs()
            old_kwargs = existing_config.to_model_kwargs()
            if cli_kwargs != old_kwargs:
                log.warning("architecture mismatch resuming from epoch "
                             f"{args.start_epoch - 1}:")
                log.warning(f"  sidecar: {old_kwargs}")
                log.warning(f"  CLI:     {cli_kwargs}")
                log.warning("continuing with CLI architecture — verify this "
                             "is intentional")

    # Train
    saved_epochs: set[int] = set()
    for epoch in range(args.start_epoch, args.n_epochs):
        loader.dataset.precompute_indices()
        losses, l1l, l2l = _train_one_epoch(loader, model, optimizer, scaler,
                                              rng, writer, epoch, args)
        if epoch % args.logging_interval == 0:
            writer.add_scalar("Loss/train", float(np.mean(losses)), epoch)
            writer.add_scalar("Loss_l1/train", float(np.mean(l1l)), epoch)
            writer.add_scalar("Loss_l2/train", float(np.mean(l2l)), epoch)
        if (args.checkpoint_interval != -1
                and epoch % args.checkpoint_interval == 0):
            _save_checkpoint(model, optimizer, scaler, args, config, epoch)
            saved_epochs.add(epoch)

    # Always save the final epoch, even if interval would have skipped it
    final = args.n_epochs - 1
    if final not in saved_epochs and final >= args.start_epoch:
        _save_checkpoint(model, optimizer, scaler, args, config, final)

    writer.close()
    log.info("=== train done ===")
    return 0
