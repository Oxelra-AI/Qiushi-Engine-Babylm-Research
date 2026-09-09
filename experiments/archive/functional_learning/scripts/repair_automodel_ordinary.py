#!/usr/bin/env python3
"""research: repair AutoModel loading for the verified ordinary-continuation endpoint.

The ordinary-control endpoint is the tested research `inherited_wwm` continuation
identified in research, not the flawed research reimplementation.  It must be
evaluated through the same repaired AutoModel coordinate as coherent86, exact
(M,S), and clean preservation, because the BabyLM SuperGLUE path calls
AutoModel.from_pretrained.  This script creates a local repaired copy
and validates the AutoModel hidden states against the trusted MLM backbone on a
fixed batch.
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

DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/automodel_repair_ordinary')
TARGETS: List[Tuple[str, pathlib.Path]] = [
    (
        "ordinary_inherited_wwm_seed62064_u0080",
        _public_path('experiments/archive/functional_learning/data/unchanged_focus_weighted_train/inherited_wwm/checkpoints/update_0080'),
    )
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
        "status": "AUTOMODEL_REPAIR_ORDINARY_CONTROL",
        "created_utc": now(),
        "script": rel(_public_path('experiments/archive/functional_learning/scripts/repair_automodel_ordinary.py')),
        "ordinary_control_identity": rel(_public_path('experiments/archive/functional_learning/data/ordinary_control_identity/ordinary_control_identity.json')),
        "repairs": repairs,
        "validations": validations,
        "all_valid": all(bool(v.get("valid")) for v in validations),
        "scientific_use": "Use this repaired copy for full official zero-shot/Reading and AutoModel-based SuperGLUE evaluation of the verified research ordinary-continuation control.",
    }
    out_json = args.out_dir / "repair_validation.json"
    out_md = args.out_dir / "repair_validation.md"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    lines = ["# research ordinary-control AutoModel repair\n\n", f"Status: `{result['status']}`; all valid `{result['all_valid']}`.\n\n"]
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
