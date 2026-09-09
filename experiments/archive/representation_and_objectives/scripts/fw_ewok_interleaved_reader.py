#!/usr/bin/env python3
"""Target-only EWoK four-cell reader for the interleaved FW arm.

The completed-result reader uses compact+row-block breadth and writes the shared
research output paths. This wrapper avoids collisions by importing the validated
research reader, overriding its output locations, and running only the interleaved
checkpoint. It performs no training and no official-example corpus shaping.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path.cwd()
WS = ROOT / "experiments/archive/representation_and_objectives"
BASE = WS / "scripts/fw_ewok_interaction_reader.py"
OUT = WS / "data/fw_ewok_interleaved_reader"
NOTE = WS / "notes/fw_ewok_interleaved_reader.md"
TARGET = "fw_breadth_interleaved_fullbatch_seed43022"


def main() -> None:
    spec = importlib.util.spec_from_file_location("fw_ewok_interaction_reader_interleaved", BASE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {BASE}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    mod.OUT_ROOT = OUT
    mod.NOTE = NOTE
    old_argv = sys.argv
    try:
        sys.argv = [str(BASE), "--targets", TARGET, "--device", "cpu", "--threads", "8", "--row_batch_size", "64", "--masked_batch_size", "128"]
        mod.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
