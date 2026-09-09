#!/usr/bin/env python3
"""research: one official SuperGLUE fine-tuning task for one paired-seed endpoint.

This repairs research's remaining serial SuperGLUE branch.  It exposes the same
official-compatible finetune invocation used by the inherited research runner, but
runs exactly one independent SuperGLUE task in an isolated output root.  The
--gpu argument is the physical GPU id passed to research setup_env; do not rely on
an external CUDA_VISIBLE_DEVICES label.
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
from statistics import mean
from typing import Any, Dict

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
OUT_BASE = _public_path('experiments/archive/functional_learning/data/parallel_superglue')

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

PRIMARY = {
    "boolq": "accuracy",
    "multirc": "accuracy",
    "rte": "accuracy",
    "wsc": "accuracy",
    "mrpc": "f1",
    "qqp": "f1",
    "mnli": "accuracy",
}
TASK_ORDER = ["boolq", "multirc", "rte", "wsc", "mrpc", "qqp", "mnli"]


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
    ap.add_argument("--task", choices=TASK_ORDER, required=True)
    ap.add_argument("--gpu", type=int, required=True, help="Physical GPU index passed to setup_env")
    ap.add_argument("--out-base", type=pathlib.Path, default=OUT_BASE)
    args = ap.parse_args()

    ep = ENDPOINTS[args.endpoint]
    model_path = pathlib.Path(ep["model"])
    if not (model_path / "model.safetensors").is_file():
        raise FileNotFoundError(model_path / "model.safetensors")

    sys.path.insert(0, str(_public_path('experiments/archive/functional_learning/scripts')))
    import official_eval_arbitrary as s29  # noqa: E402

    spec = None
    for cand in s29.base.SUPERGLUE_TASKS:
        if cand.get("task") == args.task:
            spec = dict(cand)
            break
    if spec is None:
        raise KeyError(args.task)

    out_root = pathlib.Path(args.out_base) / f"{args.endpoint}_{args.task}"
    out_root.mkdir(parents=True, exist_ok=True)
    target = f"step110_{ep['label']}_{args.task}"
    desc = f"research independent official SuperGLUE {args.task} for {args.endpoint}"
    run_dir = s29.ensure_staged_run(model_path, target, "final", out_root, desc, "paired_seed_independent_superglue")
    s29.configure_base(out_root, target, run_dir, "final", desc, "paired_seed_independent_superglue")
    base = s29.base
    env = base.setup_env(target, int(args.gpu))
    staged_model = base.model_path_for(target)

    print(json.dumps({
        "event": "superglue_task_start",
        "endpoint": args.endpoint,
        "endpoint_label": ep["label"],
        "task": args.task,
        "physical_gpu_arg": int(args.gpu),
        "target": target,
        "staged_model": rel(staged_model),
        "source_model": rel(model_path),
        "cuda_visible_devices_in_child_env": env.get("CUDA_VISIBLE_DEVICES"),
    }, ensure_ascii=False), flush=True)

    rec = base.run_superglue_task(target, staged_model, spec, env, int(args.gpu), True)
    results_root = pathlib.Path(base.OUT_ROOT) / "superglue_results" / target / args.task
    results_txt = s29.latest_file(results_root, "results.txt")
    pred = s29.latest_file(results_root, "predictions.json")
    primary_metric = PRIMARY[args.task]
    primary_score = None
    if results_txt is not None:
        primary_score = s29.parse_results_txt_metric(results_txt, primary_metric)
    valid = bool(rec.get("returncode") == 0 and primary_score is not None and pred is not None and pathlib.Path(pred).is_file())
    result = {
        "status": "SUPERGLUE_TASK_DONE" if valid else "SUPERGLUE_TASK_INCOMPLETE",
        "created_utc": now(),
        "endpoint": args.endpoint,
        "endpoint_label": ep["label"],
        "task": args.task,
        "physical_gpu_arg": int(args.gpu),
        "target": target,
        "source_model": rel(model_path),
        "staged_model": rel(staged_model),
        "run_dir": rel(run_dir),
        "out_root": rel(out_root),
        "primary_metric": primary_metric,
        "primary_score": float(primary_score) if primary_score is not None else None,
        "accuracy_from_predictions": rec.get("accuracy"),
        "returncode": rec.get("returncode"),
        "elapsed_sec": rec.get("elapsed_sec"),
        "results_txt": rel(results_txt),
        "predictions": rel(pred),
        "log": rec.get("log"),
        "record": rec,
        "valid_for_primary_superglue_aggregation": valid,
        "note": "Single-task official-compatible finetune using research arguments; aggregate with f1 for MRPC/QQP and accuracy otherwise.",
    }
    out_json = out_root / "superglue_task_result.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "endpoint": args.endpoint,
        "task": args.task,
        "physical_gpu_arg": int(args.gpu),
        "primary_metric": primary_metric,
        "primary_score": primary_score,
        "out_json": rel(out_json),
    }, ensure_ascii=False), flush=True)
    if not valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
