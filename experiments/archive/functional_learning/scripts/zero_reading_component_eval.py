#!/usr/bin/env python3
"""research: one official zero-shot/Reading component for one paired-seed endpoint.

This is the corrected completion wrapper for the paired-seed endpoints.  It runs
exactly one independent official zero-shot column or Reading component in an
isolated output root.  GlobalPIQA is intentionally represented by its two
independent official subcolumns, `GlobalPIQA_parallel` and
`GlobalPIQA_nonparallel`; final aggregation takes their mean.

The --gpu argument is the physical GPU id passed into research setup_env.  Do not
try to steer physical placement by setting CUDA_VISIBLE_DEVICES externally.
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
from typing import Any, Dict

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
OUT_BASE = _public_path('experiments/archive/functional_learning/data/parallel_zero_reading')

ENDPOINTS: Dict[str, Dict[str, Any]] = {
    "o62065": {
        "model": _public_path('experiments/archive/relation_learning/data/repair_ordinary62065_bundle/repaired_ordinary62065_u0080'),
        "label": "ordinary_inherited_wwm_seed62065",
    },
    "ms62065": {
        "model": _public_path('experiments/archive/relation_learning/data/repair_ms62065_bundle/repaired_ms62065_u0080'),
        "label": "ms_acquisition_seed62065",
    },
}

COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"]


def rel(path: pathlib.Path | str | None) -> str | None:
    if path is None:
        return None
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--endpoint", choices=sorted(ENDPOINTS), required=True)
    ap.add_argument("--column", choices=COLUMNS, required=True)
    ap.add_argument("--gpu", type=int, required=True, help="Physical GPU index passed to setup_env")
    ap.add_argument("--out-base", type=pathlib.Path, default=OUT_BASE)
    args = ap.parse_args()

    ep = ENDPOINTS[args.endpoint]
    model_path = pathlib.Path(ep["model"])
    if not (model_path / "model.safetensors").is_file():
        raise FileNotFoundError(model_path / "model.safetensors")

    sys.path.insert(0, str(_public_path('experiments/archive/functional_learning/scripts')))
    import official_eval_arbitrary as s29  # noqa: E402

    out_root = pathlib.Path(args.out_base) / f"{args.endpoint}_{args.column}"
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "per_target").mkdir(exist_ok=True)
    target = f"step110_{ep['label']}_{args.column}"
    desc = f"research independent official {args.column} for {args.endpoint}"

    run_dir = s29.ensure_staged_run(model_path, target, "final", out_root, desc, "paired_seed_independent_zero_reading")
    s29.configure_base(out_root, target, run_dir, "final", desc, "paired_seed_independent_zero_reading")
    base = s29.base
    payload = base.load_or_new_payload(target, int(args.gpu), True)
    payload.setdefault("tasks", {})
    payload.setdefault("component_source", {})
    payload["component_source"].update({
        "source_model_path": rel(model_path),
        "staged_run_dir": rel(run_dir),
        "component": args.column,
        "physical_gpu_arg": int(args.gpu),
        "created_by": rel(_public_path('experiments/archive/functional_learning/scripts/zero_reading_component_eval.py')),
    })
    base.save_payload(target, payload)

    print(json.dumps({
        "event": "zero_reading_component_start",
        "endpoint": args.endpoint,
        "endpoint_label": ep["label"],
        "column": args.column,
        "physical_gpu_arg": int(args.gpu),
        "target": target,
        "staged_model": rel(base.model_path_for(target)),
        "source_model": rel(model_path),
    }, ensure_ascii=False), flush=True)

    if args.column == "Reading":
        base.eval_reading(target, payload, int(args.gpu), True)
        task_data = payload.get("tasks", {}).get("Reading", {})
        score = (task_data.get("scores") or {}).get("Reading")
    elif args.column in base.ZERO_BY_COL:
        base.eval_zero_shot_column(target, payload, args.column, int(args.gpu), True)
        task_data = payload.get("tasks", {}).get(args.column, {})
        score = task_data.get("score")
    else:
        raise ValueError(args.column)
    base.save_payload(target, payload)
    valid = bool(task_data and task_data.get("returncode") == 0 and score is not None)
    result = {
        "status": "ZERO_READING_COMPONENT_DONE" if valid else "ZERO_READING_COMPONENT_INCOMPLETE",
        "created_utc": now(),
        "endpoint": args.endpoint,
        "endpoint_label": ep["label"],
        "column": args.column,
        "physical_gpu_arg": int(args.gpu),
        "target": target,
        "source_model": rel(model_path),
        "staged_model": rel(base.model_path_for(target)),
        "run_dir": rel(run_dir),
        "out_root": rel(out_root),
        "score": score,
        "task_data": task_data,
        "payload_path": rel(base.per_target_path(target)),
        "valid_for_component_aggregation": valid,
    }
    out_json = out_root / f"result_{args.column}.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "endpoint": args.endpoint,
        "column": args.column,
        "physical_gpu_arg": int(args.gpu),
        "score": score,
        "out_json": rel(out_json),
    }, ensure_ascii=False), flush=True)
    if not valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
