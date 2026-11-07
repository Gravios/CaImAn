"""Command-line entry point (caiman-inspector)."""

import argparse
import os
import sys

from PyQt6.QtWidgets import QApplication

from .theme import _apply_dark_palette
from .loading import _load_image_file, _make_demo_store, load_any
from .window import InspectorWindow


# ── Entry point ───────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="caiman-inspector",
        description="CaImAn Component Inspector (Qt6 / PyQt6)")
    parser.add_argument("results", nargs="?",
                        help="CaImAn .hdf5 results file, a raw .tif/.tiff stack "
                             "(its <stem>_results.hdf5 is found), or a session "
                             "folder.  Default: the current directory — opens "
                             "<dir>_results.hdf5 and <dir>.tif, where <dir> is "
                             "the folder's name")
    parser.add_argument("--results", dest="results_override", default=None,
                        help="Results file to use with a .tif stack or "
                             "session folder")
    parser.add_argument("--demo", action="store_true",
                        help="Open synthetic demo data")
    parser.add_argument("--background", "--bg", dest="background", default=None,
                        help="Image to show behind the footprints "
                             "(.npy/.npz/.tif/.png; a 3-D stack is max-projected)")
    args = parser.parse_args(argv)

    app = QApplication.instance() or QApplication(sys.argv)
    _apply_dark_palette()

    if args.demo:
        store = _make_demo_store()
    else:
        target = args.results or os.getcwd()
        try:
            store = load_any(target, args.results_override)
            print(f"Loaded {store.n} components from {store.fpath}")
        except Exception as exc:
            print(f"Error loading '{target}': {exc}", file=sys.stderr)
            if not args.results:
                print("Run inside a session folder, pass a results file / "
                      "stack / folder, or use --demo.", file=sys.stderr)
            sys.exit(1)

    if args.background:
        try:
            img  = _load_image_file(args.background)
            name = os.path.splitext(os.path.basename(args.background))[0]
            store.add_background(name, img, make_active=True)
            print(f"Background image: {args.background}")
        except Exception as exc:
            print(f"Warning: could not load background "
                  f"'{args.background}': {exc}", file=sys.stderr)

    win = InspectorWindow(store)
    win.show()
    sys.exit(app.exec())


__all__ = ['main']
