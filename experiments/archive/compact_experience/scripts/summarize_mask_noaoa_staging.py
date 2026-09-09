#!/usr/bin/env python3
"""Summarize the research mask no-AoA staging run without writing data."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import pathlib
import sys

SCRIPT_DIR = _public_path('experiments/archive/compact_experience/scripts')
sys.path.insert(0, str(SCRIPT_DIR))
import summarize_mask_eval as base  # noqa: E402

STUDY = _public_path('experiments/archive/compact_experience')
DEFAULT_OUT = _public_path('data/external/mask_noaoa_eval')
DEFAULT_NOTE = _public_path('data/external/mask_noaoa_result.md')


def main() -> None:
    # Reuse the base summarizer with both output root and note in staging.
    sys.argv = [sys.argv[0], "--out_root", str(DEFAULT_OUT), "--note", str(DEFAULT_NOTE)]
    base.main()


if __name__ == "__main__":
    main()
