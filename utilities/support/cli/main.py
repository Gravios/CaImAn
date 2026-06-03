"""
``support`` — unified CLI for the SUPPORT denoising package.

Usage::

    support train       --noisy-data PATH --exp-name NAME [...]
    support test        --input PATH  (--checkpoint PATH | --exp-name NAME) [...]
    support test-batch  --input-dir DIR --output-dir DIR \\
                        (--checkpoint PATH | --exp-name NAME) [...]
    support info        --checkpoint PATH       # show architecture & provenance
    support list        --results-dir DIR       # list experiments + checkpoints

Equivalent invocations when not installed::

    python -m src.cli train ...
    python -m src.cli test ...
"""
import argparse
import sys
from pathlib import Path

from . import train as _train
from . import test as _test
from . import test_batch as _test_batch
from .paths import ModelConfig, find_checkpoints


def _info(args: argparse.Namespace) -> int:
    """Inspect a single checkpoint."""
    cp = Path(args.checkpoint).expanduser().resolve()
    if not cp.exists():
        print(f"error: checkpoint not found: {cp}", file=sys.stderr)
        return 2
    print(f"checkpoint: {cp}")
    print(f"   size: {cp.stat().st_size / 1e6:.2f} MB")
    cfg = ModelConfig.load_for_checkpoint(cp)
    if cfg is None:
        print(f"   (no sidecar JSON found at {ModelConfig.sidecar_for(cp)})")
        print("   architecture unknown — must be specified via "
              "--unet-channels etc. when running test")
        return 0
    print(f"sidecar: {ModelConfig.sidecar_for(cp)}")
    print(f"   exp_name: {cfg.exp_name}")
    print(f"   epoch: {cfg.epoch}")
    print(f"   in_channels (T window): {cfg.in_channels}")
    print(f"   mid_channels (UNet): {cfg.mid_channels}")
    print(f"   bs_size: {cfg.bs_size}")
    print(f"   bp: {cfg.bp}")
    print("   training_data:")
    for p in cfg.training_data:
        print(f"      {p}")
    return 0


def _list(args: argparse.Namespace) -> int:
    """List experiments + their checkpoints under a results dir."""
    results = Path(args.results_dir).expanduser().resolve()
    sm = results / "saved_models"
    if not sm.exists():
        print(f"no experiments at {sm}")
        return 0
    exps = sorted(d for d in sm.iterdir() if d.is_dir())
    if not exps:
        print(f"no experiments at {sm}")
        return 0
    print(f"experiments under {sm}:")
    for exp in exps:
        cps = find_checkpoints(exp)
        latest_epoch = (cps[-1].stem.split("_")[-1] if cps else "—")
        print(f"  {exp.name:30s}  {len(cps):3d} checkpoint(s)  "
              f"latest: epoch {latest_epoch}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="support",
        description="SUPPORT — self-supervised denoising for fluorescence "
                     "imaging. See subcommand --help for details.")
    sub = p.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p_train = sub.add_parser("train", help="train a model on noisy data",
                              formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _train.add_arguments(p_train)
    p_train.set_defaults(func=_train.run)

    p_test = sub.add_parser("test", help="denoise a single TIFF",
                             formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _test.add_arguments(p_test)
    p_test.set_defaults(func=_test.run)

    p_batch = sub.add_parser("test-batch",
                              help="denoise a directory of TIFFs",
                              formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _test_batch.add_arguments(p_batch)
    p_batch.set_defaults(func=_test_batch.run)

    p_info = sub.add_parser("info",
                             help="show architecture/provenance of a checkpoint")
    p_info.add_argument("--checkpoint", type=Path, required=True)
    p_info.set_defaults(func=_info)

    p_list = sub.add_parser("list",
                             help="list experiments under a results dir")
    p_list.add_argument("--results-dir", type=Path, default=Path("./results"))
    p_list.set_defaults(func=_list)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
