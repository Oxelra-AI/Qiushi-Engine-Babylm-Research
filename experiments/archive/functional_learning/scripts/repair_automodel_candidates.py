#!/usr/bin/env python3
"""research: repair AutoModel loading for clean-seed65 and acquisition-only (M,S) endpoints.

Scientific purpose
------------------
The next official comparisons require the exact endpoints to be evaluated through
BabyLM's AutoModel-based SuperGLUE path.  These checkpoints are MLM checkpoints
whose config may only register AutoModelForMaskedLM; if unrepaired, AutoModel can
silently load a stock DeBERTa encoder and drop the adapter paths.  This script
reuses the research repair/validation functions, but makes the target set explicit
for research:

* clean_pres_lambda1_eval_seed62065_u0080: fixed-policy eval-mode preservation
  replicate endpoint;
* densemask_sparselabel_seed62064_u0080: acquisition-only (M,S) endpoint whose
  official behavior must be compared directly against clean preservation.

It creates local repaired copies and validates real AutoModel hidden
states against the trusted MLM backbone on a fixed batch.
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
from typing import Any, Dict, List, Tuple

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
SCRIPTS = _public_path('experiments/archive/functional_learning/scripts')
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import repair_automodel_clean_preservation as repair091  # noqa: E402

DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/automodel_repair_candidates')
TARGETS: List[Tuple[str, pathlib.Path]] = [
    (
        "clean_pres_lambda1_eval_seed62065_u0080",
        _public_path('experiments/archive/functional_learning/data/clean_preservation_lambda1_eval_seed62065_full80/checkpoints/update_0080'),
    ),
    (
        "densemask_sparselabel_seed62064_u0080",
        _public_path('experiments/archive/functional_learning/data/densemask_sparselabel_train_seed62064/correspondence_focus_weighted/checkpoints/update_0080'),
    ),
]


def rel(path: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    repairs: List[Dict[str, Any]] = []
    validations: List[Dict[str, Any]] = []
    for label, src in TARGETS:
        if not src.exists():
            raise FileNotFoundError(f"missing endpoint for {label}: {src}")
        dst = args.out_dir / f"repaired_{label}"
        print(json.dumps({"event": "repair_start", "label": label, "source": rel(src), "dest": rel(dst)}, ensure_ascii=False), flush=True)
        repairs.append({"label": label, **repair091.create_repaired(src, dst, bool(args.force))})
        print(json.dumps({"event": "validate_start", "label": label, "repaired": rel(dst)}, ensure_ascii=False), flush=True)
        val = repair091.validate_pair(src, dst, args.out_dir / f"validation_{label}", label)
        validations.append(val)
        print(json.dumps({
            "event": "validate_done",
            "label": label,
            "valid": val.get("valid"),
            "automodel_class": val.get("automodel", {}).get("class"),
            "private_adapter_params": val.get("automodel", {}).get("private_adapter_params"),
            "hidden_max_abs_diff": val.get("hidden_state_comparison", {}).get("max_abs_diff"),
        }, ensure_ascii=False), flush=True)

    result: Dict[str, Any] = {
        "status": "AUTOMODEL_REPAIR_CANDIDATES",
        "created_utc": now(),
        "script": rel(_public_path('experiments/archive/functional_learning/scripts/repair_automodel_candidates.py')),
        "repairs": repairs,
        "validations": validations,
        "all_valid": all(bool(v.get("valid")) for v in validations),
        "scientific_use": "Use these repaired copies for AutoModel-based SuperGLUE and compatible official zero-shot/Reading evaluation of clean seed62065 and acquisition-only (M,S).",
    }
    out_json = args.out_dir / "repair_validation.json"
    out_md = args.out_dir / "repair_validation.md"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    lines = ["# research candidate AutoModel repair\n\n", f"Status: `{result['status']}`; all valid `{result['all_valid']}`.\n\n"]
    for val in validations:
        lines.append(f"## {val['label']}\n")
        lines.append(f"- repaired: `{val['repaired']}`\n")
        lines.append(f"- AutoModel class: `{val['automodel'].get('class')}` total params `{val['automodel'].get('total_params')}` adapter `{val['automodel'].get('adapter_params')}` private `{val['automodel'].get('private_adapter_params')}`\n")
        lines.append(f"- private scales: `{val['automodel'].get('private_adapter_scales')}`\n")
        lines.append(f"- hidden max diff vs trusted MLM: `{val['hidden_state_comparison'].get('max_abs_diff')}` exact `{val['hidden_state_comparison'].get('exact_match')}`\n")
        neg = val.get("negative_original_automodel", {})
        lines.append(f"- original AutoModel negative loaded `{neg.get('loaded')}` identity `{neg.get('identity')}` adapterless/stock `{neg.get('is_stock_or_adapterless')}`\n\n")
    out_md.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": result["status"], "all_valid": result["all_valid"], "out_json": rel(out_json), "out_md": rel(out_md)}, indent=2), flush=True)
    if not result["all_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
