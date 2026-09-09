#!/usr/bin/env python3
"""research: official-style BabyLM Strict-Small evaluation for an arbitrary local MLM checkpoint.

Scientific purpose
------------------
research produced a small selected cheap7 gain for the materialized
pa_87M_alpha050 checkpoint. This wrapper evaluates such a single local checkpoint
on the stronger full official-compatible surface without pretending it has a
complete AoA ladder. It reuses the hardened official-evaluation runner from the
prior BabyLM sessions, but stages the arbitrary checkpoint under a temporary
`run_dir/hf_model/<endpoint>` layout so the inherited runner can address it
unambiguously.

The output is a research measurement, not a submission package:
  * full zero-shot / Reading predictions are produced with the current pristine
    official coordinate used in REPRESENTATION_FRONTIER_STUDIES research;
  * SuperGLUE is finetuned with the inherited official-compatible wrapper and
    patched to the current primary metric convention (F1 for MRPC/QQP, accuracy
    otherwise);
  * AoA is recorded as unavailable when the staged model root lacks the required
    19 strict-small checkpoints. The provisional Overall can use AoA=0.0, but
    `submit_ready_aoa` remains false.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import hashlib
import json
import os
import pathlib
import shutil
import sys
import time
from typing import Any, Dict, List


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
STUDY = USER_ROOT / "experiments/archive/functional_learning"
WORKSPACE = STUDY
COMPACT_EXPERIENCE_SCRIPTS = USER_ROOT / "experiments/archive/compact_experience/scripts"
if str(COMPACT_EXPERIENCE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(COMPACT_EXPERIENCE_SCRIPTS))

import full_overall_eval_runner as base  # noqa: E402
from babylm_official_scoring import normalize_aoa_record  # noqa: E402

CURRENT_STRICT = USER_ROOT / "experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict"
PRISTINE_FULL = CURRENT_STRICT / "evaluation_data/full_eval"
# This follows frontier_consolidation research exactly: GlobalPIQA full evaluation is read from
# the INITIAL_MODEL_STUDIES strict repository while the other official-coordinate data come from
# the pristine representation_and_objectives coordinate.
GLOBALPIQA_FULL = USER_ROOT / "experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval"
DEFAULT_OUT_ROOT = WORKSPACE / "data/pa87_alpha050_official_eval"

SUPERGLUE_PRIMARY_METRIC = {
    "boolq": "accuracy",
    "mnli": "accuracy",
    "mrpc": "f1",
    "multirc": "accuracy",
    "qqp": "f1",
    "rte": "accuracy",
    "wsc": "accuracy",
}

ZERO_SHOT_TASKS = [
    {"column": "BLiMP", "task": "blimp", "data_path": str((PRISTINE_FULL / "blimp_filtered").resolve()), "batch_size": 128},
    {"column": "Supplement", "task": "blimp", "data_path": str((PRISTINE_FULL / "supplement_filtered").resolve()), "batch_size": 128},
    {"column": "EWoK", "task": "ewok", "data_path": str((PRISTINE_FULL / "ewok_filtered").resolve()), "batch_size": 64},
    {"column": "Entity", "task": "entity_tracking", "data_path": str((PRISTINE_FULL / "entity_tracking").resolve()), "batch_size": 128},
    {"column": "COMPS", "task": "comps", "data_path": str((PRISTINE_FULL / "comps").resolve()), "batch_size": 128},
    {"column": "GlobalPIQA_parallel", "task": "global_piqa_parallel", "data_path": str((GLOBALPIQA_FULL / "global_piqa_parallel").resolve()), "batch_size": 128},
    {"column": "GlobalPIQA_nonparallel", "task": "global_piqa_nonparallel", "data_path": str((GLOBALPIQA_FULL / "global_piqa_nonparallel").resolve()), "batch_size": 128},
]
AOA_STEPS = [f"chck_{i}M" for i in range(1, 10)] + [f"chck_{i * 10}M" for i in range(1, 11)]


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(USER_ROOT))
    except Exception:
        return str(path)


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def latest_file(root: pathlib.Path, name: str) -> pathlib.Path | None:
    hits = sorted(root.rglob(name), key=lambda p: (p.stat().st_mtime, str(p))) if root.exists() else []
    return hits[-1] if hits else None


def parse_results_txt_metric(path: pathlib.Path, metric: str) -> float:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip() == metric:
            return float(value.strip()) * 100.0
    raise ValueError(f"metric {metric} not found in {path}")


def patch_superglue_primary_metric(target: str, payload: Dict[str, Any]) -> None:
    rec = payload.get("tasks", {}).get("SuperGLUE")
    if not isinstance(rec, dict):
        return
    details = []
    vals = []
    for task, metric in SUPERGLUE_PRIMARY_METRIC.items():
        root = pathlib.Path(base.OUT_ROOT) / "superglue_results" / target / task
        results_txt = latest_file(root, "results.txt")
        if results_txt is None:
            raise FileNotFoundError(f"Missing SuperGLUE results.txt for {task} under {root}")
        score = parse_results_txt_metric(results_txt, metric)
        details.append({"task": task, "metric": metric, "score": score, "results_txt": rel(results_txt)})
        vals.append(score)
    rec["superglue_mean_accuracy_only_legacy"] = rec.get("superglue_mean")
    rec["superglue_primary_metric_details"] = details
    rec["superglue_mean"] = sum(vals) / len(vals)
    rec["superglue_coordinate"] = "current official primary metrics: f1 for MRPC/QQP, accuracy otherwise"
    payload["tasks"]["SuperGLUE"] = rec
    base.save_payload(target, payload)
    print(json.dumps({"event": "superglue_primary_metric_patched", "target": target, "superglue_mean": rec["superglue_mean"], "legacy_accuracy_mean": rec.get("superglue_mean_accuracy_only_legacy")}), flush=True)


def configure_base(out_root: pathlib.Path, target: str, run_dir: pathlib.Path, endpoint: str, description: str, family: str) -> None:
    base.WORKSPACE = WORKSPACE
    base.STUDY = STUDY
    base.USER_ROOT = USER_ROOT
    base.ROOT_INITIAL_MODEL_STUDIES = USER_ROOT / "experiments/archive/initial_model_studies"
    base.STRICT = CURRENT_STRICT
    base.RUN_BASE = WORKSPACE / "training/runs"
    base.OUT_ROOT = out_root
    base.PER_TARGET_DIR = out_root / "per_target"
    base.TARGETS = {
        target: {
            "run_dir": run_dir,
            "endpoint": endpoint,
            "description": description,
            "family": family,
        }
    }
    base.ZERO_SHOT_TASKS = ZERO_SHOT_TASKS
    base.ZERO_BY_COL = {t["column"]: t for t in ZERO_SHOT_TASKS}

    def run_or_record_aoa_single_or_missing(target_name: str, payload: Dict[str, Any], gpu: int, force: bool) -> None:
        name = "AoA"
        if base.task_done(payload, name) and not force:
            print(json.dumps({"event": "skip_existing", "target": target_name, "column": name}), flush=True)
            return
        model_root = base.model_root_for(target_name)
        available = {p.name for p in model_root.iterdir() if p.is_dir()} if model_root.exists() else set()
        missing = [x for x in AOA_STEPS if x not in available]
        rec: Dict[str, Any] = {
            "column": name,
            "required_strict_small_steps": AOA_STEPS,
            "available_checkpoint_count": len(available),
            "available_checkpoints": sorted(available),
            "missing_required_steps": missing,
            "model_root": rel(model_root),
            "gpu": gpu,
        }
        if missing:
            rec.update(normalize_aoa_record({
                "status": "not_official_missing_checkpoints",
                "aoa_official": None,
                "aoa_raw_correlation": None,
                "aoa_leaderboard_score": 0.0,
                "aoa_for_provisional_overall": 0.0,
                "interpretation": "The staged artifact lacks at least one required strict-small AoA checkpoint. A leaderboard-unit AoA of 0.0 is recorded only for provisional arithmetic; this is not an official AoA measurement and submit_ready_aoa remains false.",
                "returncode": 0,
            }))
            payload["tasks"][name] = rec
            base.save_payload(target_name, payload)
            print(json.dumps({"event": "aoa_unavailable", "target": target_name, "missing_count": len(missing), "available_checkpoint_count": len(available)}), flush=True)
            return
        raise RuntimeError("research arbitrary evaluator currently supports honest missing-AoA recording; full AoA ladder evaluation should use the hardened research/AoA helper path after staging all 19 checkpoints.")

    base.run_or_record_aoa = run_or_record_aoa_single_or_missing


def ensure_staged_run(model_path: pathlib.Path, target: str, endpoint: str, out_root: pathlib.Path, description: str, family: str) -> pathlib.Path:
    model_path = model_path.resolve()
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not (model_path / "model.safetensors").exists():
        raise FileNotFoundError(model_path / "model.safetensors")
    run_dir = out_root / "staged_model_roots" / target
    model_root = run_dir / "hf_model"
    staged_endpoint = model_root / endpoint
    model_root.mkdir(parents=True, exist_ok=True)
    if staged_endpoint.exists() or staged_endpoint.is_symlink():
        if staged_endpoint.is_symlink() or staged_endpoint.is_file():
            staged_endpoint.unlink()
        else:
            # If it is an old directory from a copy-based run, remove only this staged endpoint.
            shutil.rmtree(staged_endpoint)
    os.symlink(model_path, staged_endpoint, target_is_directory=True)
    model_sha = sha256_file(model_path / "model.safetensors")
    # Keep a small metrics/provenance file so inherited wrappers that expect a run
    # directory can still describe the model faithfully.
    metrics = {
        "status": "STAGED_ARBITRARY_MODEL",
        "created_utc": now_utc(),
        "target": target,
        "description": description,
        "family": family,
        "source_model_path": rel(model_path),
        "staged_run_dir": rel(run_dir),
        "staged_endpoint": endpoint,
        "model_safetensors_sha256": model_sha,
        "parameter_count": None,
        "tokenizer_label": "inherited_from_source_checkpoint",
        "saved_checkpoints": [{"name": endpoint, "source": rel(model_path), "sha256": model_sha}],
        "research_note": "This is a symlink-staged evaluation wrapper for a materialized checkpoint, not a new training run.",
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "scientific_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {
        "status": "STAGED_MODEL_READY",
        "created_utc": now_utc(),
        "target": target,
        "run_dir": rel(run_dir),
        "model_root": rel(model_root),
        "endpoint": endpoint,
        "staged_endpoint_symlink": rel(staged_endpoint),
        "source_model_path": rel(model_path),
        "source_model_safetensors_sha256": model_sha,
        "description": description,
        "family": family,
    }
    (run_dir / "staged_model_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return run_dir


def preflight_payload(out_root: pathlib.Path, target: str, run_dir: pathlib.Path, endpoint: str, model_path: pathlib.Path, columns: List[str], description: str, family: str) -> dict[str, Any]:
    required = {
        "current_strict": CURRENT_STRICT,
        "current_full_eval": PRISTINE_FULL,
        "globalpiqa_full_eval": GLOBALPIQA_FULL,
        "compact_experience_base_runner": COMPACT_EXPERIENCE_SCRIPTS / "full_overall_eval_runner.py",
        "babylm_official_scoring": COMPACT_EXPERIENCE_SCRIPTS / "babylm_official_scoring.py",
        "model_path": model_path,
        "staged_run_dir": run_dir,
        "staged_model_path": run_dir / "hf_model" / endpoint,
        "scientific_metrics": run_dir / "scientific_metrics.json",
    }
    return {
        "status": "ARBITRARY_OFFICIAL_EVAL_PREFLIGHT",
        "created_utc": now_utc(),
        "target": target,
        "description": description,
        "family": family,
        "source_model_path": rel(model_path),
        "run_dir": rel(run_dir),
        "model_root": rel(run_dir / "hf_model"),
        "endpoint": endpoint,
        "out_root": rel(out_root),
        "columns_requested": columns,
        "required_paths": {k: {"path": rel(v), "exists": pathlib.Path(v).exists()} for k, v in required.items()},
        "zero_shot_specs": ZERO_SHOT_TASKS,
        "reading_data_path": str((PRISTINE_FULL / "reading/reading_data.csv").resolve()),
        "superglue_data_dir": str((PRISTINE_FULL / "glue_filtered").resolve()),
        "aoa_handling": "Records not_official_missing_checkpoints and provisional leaderboard AoA=0.0 unless all 19 strict-small checkpoints are staged.",
        "official_like_arithmetic": "mean(BLiMP, Supplement, EWoK, Entity, COMPS, SuperGLUE primary metric mean, GlobalPIQA mean, Reading, AoA leaderboard score); submit_ready_aoa remains false if AoA ladder is missing.",
    }


def run_one(args: argparse.Namespace) -> None:
    out_root = pathlib.Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    model_path = pathlib.Path(args.model_path)
    target = args.target
    endpoint = args.endpoint
    description = args.description or f"research official-style evaluation of arbitrary checkpoint {target}"
    family = args.family or "arbitrary_single_checkpoint"
    run_dir = ensure_staged_run(model_path, target, endpoint, out_root, description, family)
    configure_base(out_root, target, run_dir, endpoint, description, family)
    columns = args.columns or [t["column"] for t in ZERO_SHOT_TASKS] + ["Reading", "SuperGLUE", "AoA"]

    if args.preflight_only:
        payload = preflight_payload(out_root, target, run_dir, endpoint, model_path.resolve(), columns, description, family)
        out = out_root / f"{target}_preflight.json"
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        missing = [k for k, v in payload["required_paths"].items() if not v["exists"]]
        print(json.dumps({"status": payload["status"], "out_json": rel(out), "target": target, "missing_required": missing, "columns": columns}, indent=2, ensure_ascii=False), flush=True)
        if missing:
            raise SystemExit(2)
        return

    payload = base.load_or_new_payload(target, int(args.gpu), args.force)
    payload.setdefault("tasks", {})
    payload.setdefault("source", {})
    payload["source"].update({
        "source_model_path": rel(model_path.resolve()),
        "staged_run_dir": rel(run_dir),
        "staged_endpoint": endpoint,
        "description": description,
        "family": family,
        "created_by": rel(_public_path('experiments/archive/functional_learning/scripts/official_eval_arbitrary.py')),
    })
    base.save_payload(target, payload)
    print(json.dumps({"event": "official_eval_start", "target": target, "gpu": args.gpu, "columns": columns, "model_path": rel(run_dir / "hf_model" / endpoint), "source_model": rel(model_path.resolve())}, ensure_ascii=False), flush=True)
    for col in columns:
        if col in base.ZERO_BY_COL:
            base.eval_zero_shot_column(target, payload, col, int(args.gpu), args.force)
        elif col == "Reading":
            base.eval_reading(target, payload, int(args.gpu), args.force)
        elif col == "SuperGLUE":
            base.eval_superglue(target, payload, int(args.gpu), args.force)
            patch_superglue_primary_metric(target, payload)
        elif col == "AoA":
            base.run_or_record_aoa(target, payload, int(args.gpu), args.force)
        else:
            raise ValueError(f"Unknown column {col}")
    base.finalize_payload(target, payload)
    print(json.dumps({"status": "ARBITRARY_OFFICIAL_EVAL_DONE", "target": target, "per_target_json": rel(base.per_target_path(target)), "official_overall": payload.get("official_overall")}, indent=2, ensure_ascii=False), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--endpoint", default="final")
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--columns", nargs="*", default=None, help="Subset among zero-shot columns, Reading, SuperGLUE, AoA. Default: all official-style columns.")
    ap.add_argument("--description", default="")
    ap.add_argument("--family", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()
    run_one(args)


if __name__ == "__main__":
    main()
