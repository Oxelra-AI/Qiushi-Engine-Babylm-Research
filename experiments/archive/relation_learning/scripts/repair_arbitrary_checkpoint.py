#!/usr/bin/env python3
"""research: repair AutoModel registration for arbitrary FrozenSlowPrivate checkpoints.

Creates a repaired copy of a checkpoint by using the repair helper, then
validates that AutoModel exposes the same hidden states as the trusted MLM backbone.
Use for newly trained O62065/(M,S)62065 before SuperGLUE or other AutoModel-based
paths.  For MLM cheap7 this is not strictly required, but a repaired copy keeps all
coordinates faithful and reusable.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import pathlib
import sys
import time
from typing import Any

ROOT = _public_path('.')
A01_SCRIPTS = _public_path('experiments/archive/functional_learning/scripts')
if str(A01_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(A01_SCRIPTS))

import repair_automodel_clean_preservation as repair091  # noqa: E402


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=pathlib.Path, required=True)
    ap.add_argument("--dest", type=pathlib.Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out-dir", type=pathlib.Path, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    source = args.source if args.source.is_absolute() else ROOT / args.source
    dest = args.dest if args.dest.is_absolute() else ROOT / args.dest
    out_dir = args.out_dir if args.out_dir is not None else dest.parent / f"validation_{args.label}"
    out_dir = out_dir if out_dir.is_absolute() else ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"event": "repair_start", "label": args.label, "source": rel(source), "dest": rel(dest)}, ensure_ascii=False), flush=True)
    repair = repair091.create_repaired(source, dest, bool(args.force))
    print(json.dumps({"event": "validate_start", "label": args.label, "dest": rel(dest)}, ensure_ascii=False), flush=True)
    validation = repair091.validate_pair(source, dest, out_dir, str(args.label))
    result: dict[str, Any] = {
        "status": "ARBITRARY_CHECKPOINT_AUTOMODEL_REPAIR_DONE",
        "created_utc": now(),
        "label": str(args.label),
        "repair": repair,
        "validation": validation,
        "all_valid": bool(validation.get("valid")),
        "scientific_use": "Repaired destination is valid for AutoModel-based official fine-tuning and remains the faithful MLM function for AutoModelForMaskedLM.",
    }
    result_path = out_dir / "repair_validation.json"
    md_path = out_dir / "repair_validation.md"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    lines = [
        f"# research AutoModel repair: {args.label}\n\n",
        f"All valid: `{result['all_valid']}`\n\n",
        f"Source: `{rel(source)}`\n\n",
        f"Destination: `{rel(dest)}`\n\n",
        f"AutoModel class: `{validation.get('automodel', {}).get('class')}`; params `{validation.get('automodel', {}).get('param_count')}`; max hidden diff `{validation.get('hidden_max_abs_diff')}`.\n\n",
        f"Full JSON: `{rel(result_path)}`\n",
    ]
    md_path.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": result["status"], "label": args.label, "all_valid": result["all_valid"], "dest": rel(dest), "out_json": rel(result_path), "out_md": rel(md_path)}, indent=2, ensure_ascii=False), flush=True)
    if not result["all_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
