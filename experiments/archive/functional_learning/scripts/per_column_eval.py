#!/usr/bin/env python3
"""research: Per-column parallel evaluation for paired-seed endpoints.

Evaluates ONE column for ONE endpoint with correct physical GPU mapping.
Uses isolated out-root per (endpoint, column) to prevent race conditions.

The --gpu N argument propagates to setup_env which sets CUDA_VISIBLE_DEVICES=N.
Do NOT set CUDA_VISIBLE_DEVICES externally.

Usage:
  python per_column_eval.py --endpoint o62065 --column BLiMP --gpu 0
  python per_column_eval.py --endpoint ms62065 --column SuperGLUE --gpu 1
"""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import argparse, json, os, pathlib, shutil, sys, time

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
OUT_BASE = _public_path('experiments/archive/functional_learning/data/parallel_eval')

ENDPOINTS = {
    "o62065": {
        "model": _public_path('experiments/archive/relation_learning/data/repair_ordinary62065_bundle/repaired_ordinary62065_u0080'),
        "label": "ordinary_inherited_wwm_seed62065",
    },
    "ms62065": {
        "model": _public_path('experiments/archive/relation_learning/data/repair_ms62065_bundle/repaired_ms62065_u0080'),
        "label": "ms_acquisition_seed62065",
    },
}

def rel(p):
    return str(pathlib.Path(p).relative_to(ROOT))

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--endpoint", choices=list(ENDPOINTS.keys()), required=True)
    ap.add_argument("--column", required=True)
    ap.add_argument("--gpu", type=int, required=True, help="Physical GPU index (0 or 1)")
    args = ap.parse_args()

    ep = ENDPOINTS[args.endpoint]
    col = args.column
    gpu = args.gpu
    model_path = ep["model"]

    # Isolated out-root per (endpoint, column) — no race conditions
    out_root = OUT_BASE / f"{args.endpoint}_{col}"
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "per_target").mkdir(exist_ok=True)

    target = f"step109_{ep['label']}"

    print(json.dumps({
        "event": "eval_start", "endpoint": args.endpoint, "column": col,
        "physical_gpu": gpu, "target": target, "model": rel(model_path),
    }), flush=True)

    # Import and configure research which sets up research base runner
    sys.path.insert(0, str(_public_path('experiments/archive/functional_learning/scripts')))
    import official_eval_arbitrary as s29

    # Stage model
    run_dir = s29.ensure_staged_run(model_path, target, "final", out_root,
                                    f"research {args.endpoint} {col}", "paired_seed")

    # Configure base module
    s29.configure_base(out_root, target, run_dir, "final",
                       f"research {args.endpoint} {col}", "paired_seed")

    base = s29.base

    # Load/create payload
    payload = base.load_or_new_payload(target, gpu, True)
    payload.setdefault("tasks", {})

    # Evaluate the single column
    if col == "SuperGLUE":
        base.eval_superglue(target, payload, gpu, True)
        s29.patch_superglue_primary_metric(target, payload)
    elif col == "Reading":
        base.eval_reading(target, payload, gpu, True)
    elif col in base.ZERO_BY_COL:
        base.eval_zero_shot_column(target, payload, col, gpu, True)
    else:
        raise ValueError(f"Unknown column: {col}")

    base.save_payload(target, payload)

    # Extract and save result
    score = None
    task_data = {}
    if col in payload.get("tasks", {}):
        task_data = payload["tasks"][col]
        score = task_data.get("score")

    result = {
        "status": "COLUMN_EVAL_DONE",
        "endpoint": args.endpoint,
        "endpoint_label": ep["label"],
        "column": col,
        "physical_gpu": gpu,
        "target": target,
        "score": score,
        "task_data": task_data,
        "payload_path": rel(base.per_target_path(target)),
    }
    out_json = out_root / f"result_{col}.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))

    print(json.dumps({
        "event": "eval_done", "endpoint": args.endpoint, "column": col,
        "physical_gpu": gpu, "score": score, "out_json": rel(out_json),
    }), flush=True)

if __name__ == "__main__":
    main()
