"""
Path and model-configuration utilities.

Centralises filesystem layout so train and test agree on where things
live, and provides a JSON sidecar so inference no longer needs to
manually re-specify the architecture used at training time.

Layout produced by training (rooted at ``results_dir``)::

    results_dir/
    ├── saved_models/
    │   └── <exp_name>/
    │       ├── model_0.pth
    │       ├── model_0.json          ← architecture sidecar
    │       ├── optimizer_0.pth
    │       ├── ...
    ├── logs/
    │   └── <exp_name>.log
    └── tsboard/
        └── <exp_name>/

Inference (``support test``) finds models by either
  (a) ``--checkpoint /path/to/model_N.pth`` — explicit, reads sidecar
      next to it for architecture; or
  (b) ``--exp-name NAME --results-dir DIR`` — auto-picks the latest
      ``model_N.pth`` in ``DIR/saved_models/NAME/``.

`ModelConfig` is the single source of truth for model architecture:
both training (writes it) and inference (reads it) construct the model
through this dataclass, never via free-floating kwargs.
"""

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Self


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Default architecture matches the shipped pretrained models (bs3.pth etc).
# Used as the fallback when a checkpoint has no sidecar JSON.
_DEFAULT_MID_CHANNELS = [16, 32, 64, 128, 256]
_DEFAULT_ONE_BY_ONE = [32, 16]
_DEFAULT_LAST_LAYER = [64, 32, 16]
_DEFAULT_BS_SIZE = [3, 3]
_DEFAULT_PATCH_SIZE = [61, 64, 64]


# ---------------------------------------------------------------------------
# Model architecture sidecar
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ModelConfig:
    """All architecture + key training params needed to rebuild the model
    at inference time. Saved as JSON next to each checkpoint.

    The defaults match the shipped ``bs3.pth`` so a freshly-constructed
    ``ModelConfig()`` produces a model compatible with the upstream
    pretrained checkpoints.
    """
    in_channels: int = 61
    mid_channels: list[int] = field(
        default_factory=lambda: list(_DEFAULT_MID_CHANNELS))
    depth: int = 5
    blind_conv_channels: int = 64
    one_by_one_channels: list[int] = field(
        default_factory=lambda: list(_DEFAULT_ONE_BY_ONE))
    last_layer_channels: list[int] = field(
        default_factory=lambda: list(_DEFAULT_LAST_LAYER))
    bs_size: list[int] = field(
        default_factory=lambda: list(_DEFAULT_BS_SIZE))
    bp: bool = False
    # Provenance (not strictly required to rebuild the net, but invaluable
    # for debugging "which model was this and what was it trained on?").
    exp_name: str = "unnamed"
    epoch: int = 0
    training_data: list[str] = field(default_factory=list)
    patch_size: list[int] = field(
        default_factory=lambda: list(_DEFAULT_PATCH_SIZE))
    notes: str = ""

    # ---- I/O ----

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(asdict(self), f, indent=2)
        return path

    @classmethod
    def load(cls, path: str | Path) -> Self:
        path = Path(path)
        with path.open() as f:
            data = json.load(f)
        # Tolerate older / future sidecars carrying unknown fields
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

    @classmethod
    def sidecar_for(cls, checkpoint_path: str | Path) -> Path:
        """Return the sidecar JSON path for a given .pth checkpoint."""
        return Path(checkpoint_path).with_suffix(".json")

    @classmethod
    def load_for_checkpoint(cls, checkpoint_path: str | Path) -> Self | None:
        """Read sidecar if it exists; return ``None`` otherwise (legacy
        checkpoints without sidecars get ``None`` and the CLI falls back
        to defaults + explicit user flags)."""
        sidecar = cls.sidecar_for(checkpoint_path)
        if sidecar.exists():
            return cls.load(sidecar)
        return None

    # ---- Build from argparse ----

    @classmethod
    def from_namespace(cls, args: argparse.Namespace,
                        training_data: list[str] | None = None) -> Self:
        """Build from a parsed argparse Namespace populated by
        :func:`add_architecture_arguments`. Architecture fields are
        copied verbatim; provenance fields are filled from
        ``exp_name`` / ``training_data`` if available."""
        return cls(
            in_channels=args.input_frames,
            mid_channels=list(args.unet_channels),
            depth=args.depth,
            blind_conv_channels=args.blind_conv_channels,
            one_by_one_channels=list(args.one_by_one_channels),
            last_layer_channels=list(args.last_layer_channels),
            bs_size=list(args.bs_size),
            bp=args.bp,
            exp_name=getattr(args, "exp_name", "unnamed") or "unnamed",
            epoch=getattr(args, "start_epoch", 0),
            training_data=list(training_data or []),
            patch_size=list(getattr(args, "patch_size", _DEFAULT_PATCH_SIZE)
                              or _DEFAULT_PATCH_SIZE),
        )

    def to_model_kwargs(self) -> dict:
        """The architecture-only kwargs ready to splat into
        ``model.SUPPORT.SUPPORT(**kwargs)``."""
        return {
            "in_channels": self.in_channels,
            "mid_channels": self.mid_channels,
            "depth": self.depth,
            "blind_conv_channels": self.blind_conv_channels,
            "one_by_one_channels": self.one_by_one_channels,
            "last_layer_channels": self.last_layer_channels,
            "bs_size": self.bs_size,
            "bp": self.bp,
        }


# ---------------------------------------------------------------------------
# Shared argparse helpers
# ---------------------------------------------------------------------------

def add_architecture_arguments(p: argparse.ArgumentParser,
                                 group_title: str = "architecture") -> None:
    """Add the eight architecture flags that ``train`` and ``test``
    share. ``ModelConfig.from_namespace`` consumes them."""
    g = p.add_argument_group(group_title)
    g.add_argument("--input-frames", type=int, default=61,
                    help="temporal window size; must equal patch_size[0]")
    g.add_argument("--unet-channels", type=int, nargs="+",
                    default=list(_DEFAULT_MID_CHANNELS),
                    help="U-Net feature map channels per depth "
                         "(default matches shipped bs3.pth)")
    g.add_argument("--depth", type=int, default=5)
    g.add_argument("--blind-conv-channels", type=int, default=64)
    g.add_argument("--one-by-one-channels", type=int, nargs="+",
                    default=list(_DEFAULT_ONE_BY_ONE))
    g.add_argument("--last-layer-channels", type=int, nargs="+",
                    default=list(_DEFAULT_LAST_LAYER))
    g.add_argument("--bs-size", type=int, nargs=2,
                    default=list(_DEFAULT_BS_SIZE),
                    help="blind-spot size, two ints (default 3 3)")
    g.add_argument("--bp", action="store_true",
                    help="blind-plane mode (for fast voltage data)")


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def resolve_input_paths(paths: list[str | Path]) -> list[Path]:
    """Expand and verify a list of input data paths. Accepts files only
    (directories should be iterated by the caller — see ``test-batch``)."""
    out: list[Path] = []
    for p in paths:
        p = Path(p).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"input path does not exist: {p}")
        if not p.is_file():
            raise ValueError(f"input path is not a file: {p}")
        out.append(p)
    return out


def experiment_dir(results_dir: str | Path, exp_name: str) -> Path:
    """``<results_dir>/saved_models/<exp_name>/`` — where checkpoints
    land. Created on demand."""
    d = Path(results_dir).expanduser().resolve() / "saved_models" / exp_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def checkpoint_path(results_dir: str | Path, exp_name: str,
                     epoch: int) -> Path:
    return experiment_dir(results_dir, exp_name) / f"model_{epoch}.pth"


_MODEL_RE = re.compile(r"^model_(\d+)\.pth$")


def find_checkpoints(exp_dir: str | Path) -> list[Path]:
    """All ``model_N.pth`` files in an experiment dir, sorted by epoch
    number (oldest first). Returns ``[]`` if the dir doesn't exist."""
    exp_dir = Path(exp_dir).expanduser().resolve()
    if not exp_dir.exists():
        return []
    pairs: list[tuple[int, Path]] = []
    for f in exp_dir.iterdir():
        m = _MODEL_RE.match(f.name)
        if m:
            pairs.append((int(m.group(1)), f))
    pairs.sort(key=lambda x: x[0])
    return [f for _, f in pairs]


def latest_checkpoint(exp_dir: str | Path) -> Path | None:
    """Most recent (highest-epoch) checkpoint, or ``None`` if none."""
    cps = find_checkpoints(exp_dir)
    return cps[-1] if cps else None


def resolve_checkpoint(checkpoint: str | Path | None = None,
                       results_dir: str | Path | None = None,
                       exp_name: str | None = None,
                       epoch: int | None = None) -> Path:
    """Resolve a checkpoint specification into a concrete file path.

    Caller can specify one of:
      - ``checkpoint``: explicit .pth path
      - ``results_dir`` + ``exp_name`` (+ optional ``epoch``): look up
        in experiment dir; ``epoch`` defaults to latest

    Raises ``ValueError`` if neither is given, ``FileNotFoundError`` if
    the resolved path doesn't exist.
    """
    if checkpoint is not None:
        p = Path(checkpoint).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"checkpoint not found: {p}")
        return p
    if results_dir is None or exp_name is None:
        raise ValueError("either --checkpoint, or both --results-dir and "
                          "--exp-name, must be provided")
    exp_dir = Path(results_dir).expanduser().resolve() / "saved_models" / exp_name
    if epoch is not None:
        p = exp_dir / f"model_{epoch}.pth"
        if not p.exists():
            raise FileNotFoundError(f"checkpoint not found: {p}")
        return p
    p = latest_checkpoint(exp_dir)
    if p is None:
        raise FileNotFoundError(f"no checkpoints found under {exp_dir}")
    return p


# ---------------------------------------------------------------------------
# Default output naming
# ---------------------------------------------------------------------------

def default_output_path(input_path: str | Path,
                         output_dir: str | Path | None = None,
                         suffix: str = "denoised") -> Path:
    """If user doesn't supply ``--output``, derive one alongside the
    input or in ``output_dir`` with ``_<suffix>`` appended to the stem."""
    p = Path(input_path)
    name = f"{p.stem}_{suffix}{p.suffix or '.tif'}"
    return ((Path(output_dir).expanduser().resolve() / name)
             if output_dir else p.parent / name)
