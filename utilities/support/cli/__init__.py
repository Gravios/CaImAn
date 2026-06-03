"""SUPPORT command-line interface.

Subcommands:
    train       — self-supervised training
    test        — denoise a single TIFF
    test-batch  — denoise a directory of TIFFs
    info        — inspect a checkpoint's architecture
    list        — list experiments under a results dir
"""
from .main import build_parser, main

__all__ = ["build_parser", "main"]
