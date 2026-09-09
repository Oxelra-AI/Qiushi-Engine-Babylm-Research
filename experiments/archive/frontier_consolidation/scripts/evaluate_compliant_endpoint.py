#!/usr/bin/env python3
"""research: full official-compatible evaluation harness for compliant-tokenizer models.

This script is intentionally parameterized because the compliant-tokenizer retrain
is load-bearing and must be isolated from the invalid old-tokenizer endpoint.
It runs the same official-style task wrappers as the inherited full evaluator, but
points EWoK at the current pristine 7,618-row official coordinate and uses the
repaired min_context=0 AoA helper so the AoA ladder has 8,005 rows/checkpoint.

Use `--preflight-only` while the retrain is pending.  Once `hf_model/chck_100M`
exists, run selected columns or the full surface.  After all predictions exist,
run `experiments/archive/representation_and_objectives/scripts/stage_pristine_collate.py` with
this script's full-root, target, EWoK predictions path, AoA output root, and model
root to obtain the same official-coordinate scalar as Steps 37/38 in representation_and_objectives.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any, Dict, List


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
STUDY = USER_ROOT / "experiments/archive/frontier_consolidation"
WORKSPACE = STUDY
COMPACT_EXPERIENCE_SCRIPTS = USER_ROOT / "experiments/archive/compact_experience/scripts"
if str(COMPACT_EXPERIENCE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(COMPACT_EXPERIENCE_SCRIPTS))

import full_overall_eval_runner as base  # noqa: E402
from babylm_official_scoring import normalize_aoa_record, compute_overall_from_tasks  # noqa: E402

CURRENT_STRICT = USER_ROOT / "experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict"
PRISTINE_FULL = CURRENT_STRICT / "evaluation_data/full_eval"
GLOBALPIQA_FULL = USER_ROOT / "experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval"
AOA_HELPER = WORKSPACE / "scripts/aoa_local_ckpts_minctx.py"
DEFAULT_OUT_ROOT = WORKSPACE / "data/compliant_full_eval"
DEFAULT_COLLATE_ROOT = WORKSPACE / "data/compliant_pristine_collate"

SUPERGLUE_PRIMARY_METRIC = {"boolq": "accuracy", "mnli": "accuracy", "mrpc": "f1", "multirc": "accuracy", "qqp": "f1", "rte": "accuracy", "wsc": "accuracy"}
ARM_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "reinvest": {
        "target": "complianttok_reinvest_seed43022",
        "run_dir": WORKSPACE / "training/runs/complianttok_reinvest_seed43022_r2",
        "endpoint": "chck_100M",
        "description": "compact_view_reinvest retrained from random weights with a 16k BPE tokenizer trained only on the same 10M Strict-Small pool used for its pretraining; submission-relevant end-to-end corpus-budget-valid endpoint if training/evaluation complete",
        "family": "end_to_end_compliant_tokenizer_density_reinvestment",
    },
    "clean_qwen": {
        "target": "complianttok_clean_qwen_seed43022",
        "run_dir": WORKSPACE / "training/runs/complianttok_cleanqwen_seed43022_r2",
        "endpoint": "chck_100M",
        "description": "matched clean-Qwen aligned control retrained with the same fixed 10M-reinvest tokenizer and frozen recipe; scientific fixed-tokenizer control only, not an independently end-to-end Strict-Small-valid system because tokenizer-fitting text plus clean-Qwen pretraining text has a >10M union",
        "family": "fixed_reinvest_tokenizer_clean_qwen_scientific_control",
    },
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
    """Replace inherited accuracy-only mean with official primary metric mean.

    The inherited full-eval runner computes SuperGLUE from validation accuracy for
    every task.  The current official-coordinate convention uses F1 for MRPC/QQP
    and accuracy for the other SuperGLUE tasks.  Final pristine collation also
    parses `results.txt`; this patch makes the per-target JSON and projection
    arithmetic agree with that coordinate.
    """
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
        details.append({"task": task, "metric": metric, "score": score, "results_txt": str(results_txt)})
        vals.append(score)
    rec["superglue_mean_accuracy_only_legacy"] = rec.get("superglue_mean")
    rec["superglue_primary_metric_details"] = details
    rec["superglue_mean"] = sum(vals) / len(vals)
    rec["superglue_coordinate"] = "current official primary metrics: f1 for MRPC/QQP, accuracy otherwise"
    payload["tasks"]["SuperGLUE"] = rec
    base.save_payload(target, payload)
    print(json.dumps({"event": "superglue_primary_metric_patched", "target": target, "superglue_mean": rec["superglue_mean"], "legacy_accuracy_mean": rec.get("superglue_mean_accuracy_only_legacy")}), flush=True)

def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def configure_base(out_root: pathlib.Path, target: str, run_dir: pathlib.Path, endpoint: str, description: str, family: str) -> None:
    # Patch the inherited runner globals before calling any of its functions.
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

    def run_or_record_aoa_minctx0(target_name: str, payload: Dict[str, Any], gpu: int, force: bool) -> None:
        name = "AoA"
        if base.task_done(payload, name) and not force:
            print(json.dumps({"event": "skip_existing", "target": target_name, "column": name}), flush=True)
            return
        model_root = base.model_root_for(target_name)
        available = {p.name for p in model_root.iterdir() if p.is_dir()} if model_root.exists() else set()
        missing = [x for x in base.AOA_STEPS if x not in available]
        rec: Dict[str, Any] = {
            "column": name,
            "required_strict_small_steps": base.AOA_STEPS,
            "available_checkpoint_count": len(available),
            "missing_required_steps": missing,
            "model_root": str(model_root),
            "aoa_helper": str(AOA_HELPER),
            "min_context": 0,
            "expected_rows_per_step": 8005,
        }
        if missing:
            rec.update({
                "status": "not_official_missing_checkpoints",
                "aoa_official": None,
                "aoa_raw_correlation": None,
                "aoa_leaderboard_score": 0.0,
                "aoa_for_provisional_overall": 0.0,
                "interpretation": "Missing at least one required strict-small AoA checkpoint; no official AoA is inferred.",
                "returncode": 0,
            })
            payload["tasks"][name] = rec
            base.save_payload(target_name, payload)
            print(json.dumps({"event": "aoa_unavailable", "target": target_name, "missing_count": len(missing)}), flush=True)
            return
        out_dir = out_root / "aoa_outputs" / target_name
        out_json = out_dir / "aoa_local_ckpts_minctx0.json"
        out_note = out_dir / "aoa_local_ckpts_minctx0.md"
        log = out_root / "logs" / target_name / "aoa_minctx0.log"
        out_dir.mkdir(parents=True, exist_ok=True)
        log.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, str(AOA_HELPER.resolve()),
            "--model_root", str(model_root.resolve()),
            "--out_dir", str(out_dir.resolve()),
            "--out_json", str(out_json.resolve()),
            "--out_note", str(out_note.resolve()),
            "--log", str(log.resolve()),
            "--gpu", str(gpu),
            "--min_context", "0",
            "--expected_rows_per_step", "8005",
        ]
        env = os.environ.copy()
        cache_root = out_dir / "runtime_cache_preimport"
        cache_map = {
            "HF_HOME": cache_root / "hf_home",
            "HF_HUB_CACHE": cache_root / "hf_home" / "hub",
            "HUGGINGFACE_HUB_CACHE": cache_root / "hf_home" / "hub",
            "TRANSFORMERS_CACHE": cache_root / "transformers",
            "HF_MODULES_CACHE": cache_root / "modules",
            "HF_DATASETS_CACHE": cache_root / "datasets",
            "TMPDIR": cache_root / "tmp",
        }
        for key, path in cache_map.items():
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)
            env[key] = str(path.resolve())
        env["TOKENIZERS_PARALLELISM"] = "false"
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        t0 = time.time()
        with log.open("a", encoding="utf-8") as f:
            f.write(f"\n[{now_utc()}] $ {' '.join(cmd)}\n")
        proc = subprocess.run(cmd, cwd=str(USER_ROOT), env=env, capture_output=True, text=True, timeout=14400)
        with log.open("a", encoding="utf-8") as f:
            f.write(proc.stdout)
            f.write("\n--- STDERR ---\n")
            f.write(proc.stderr)
            f.write(f"\n[returncode={proc.returncode} elapsed_sec={time.time()-t0:.2f}]\n")
        rec.update({"returncode": proc.returncode, "elapsed_sec": round(time.time() - t0, 3), "log": str(log), "out_json": str(out_json), "out_note": str(out_note)})
        if proc.returncode != 0:
            rec["error"] = proc.stderr[-4000:] or proc.stdout[-4000:]
            payload["tasks"][name] = rec
            base.save_payload(target_name, payload)
            raise RuntimeError(rec)
        data = json.loads(out_json.read_text(encoding="utf-8"))
        if data.get("status") != "AOA_LOCAL_CKPTS_MINCTX_DONE" or data.get("row_count_values") != [8005] or data.get("num_steps") != len(base.AOA_STEPS) or data.get("finite_surprisals") is not True:
            rec["error"] = {"helper_status": data.get("status"), "row_count_values": data.get("row_count_values"), "num_steps": data.get("num_steps"), "finite_surprisals": data.get("finite_surprisals")}
            payload["tasks"][name] = rec
            base.save_payload(target_name, payload)
            raise RuntimeError(rec)
        curve = data.get("curve_fitness_record") if isinstance(data, dict) else None
        rec.update(normalize_aoa_record({
            "status": "official_aoa_done",
            "helper_status": data.get("status"),
            "aoa_helper_runner": str(AOA_HELPER.resolve()),
            "aoa_official": float(data.get("aoa", 0.0)),
            "surprisal_path": data.get("surprisal_path"),
            "score_path": data.get("score_path"),
            "score_tokenizer_path": data.get("score_tokenizer_path"),
            "num_rows": data.get("num_rows"),
            "num_steps": data.get("num_steps"),
            "step_counts": data.get("step_counts"),
            "expected_steps": data.get("expected_steps"),
            "row_count_values": data.get("row_count_values"),
            "finite_surprisals": data.get("finite_surprisals"),
            "curve_fitness_record": curve,
            "aoa_p_value": curve.get("p_value") if isinstance(curve, dict) else None,
            "aoa_n_words": curve.get("n_words") if isinstance(curve, dict) else None,
        }))
        payload["tasks"][name] = rec
        base.save_payload(target_name, payload)
        print(json.dumps({"event": "aoa_done", "target": target_name, "aoa_raw": rec.get("aoa_raw_correlation"), "aoa_leaderboard": rec.get("aoa_leaderboard_score"), "gpu": gpu}), flush=True)

    base.run_or_record_aoa = run_or_record_aoa_minctx0


def preflight_payload(out_root: pathlib.Path, collate_root: pathlib.Path, target: str, run_dir: pathlib.Path, endpoint: str, description: str, family: str, columns: List[str]) -> dict[str, Any]:
    model_root = run_dir / "hf_model"
    model_path = model_root / endpoint
    full_required = {
        "current_strict": CURRENT_STRICT,
        "current_full_eval": PRISTINE_FULL,
        "globalpiqa_full_eval": GLOBALPIQA_FULL,
        "aoa_helper_minctx0": AOA_HELPER,
        "compact_experience_base_runner": COMPACT_EXPERIENCE_SCRIPTS / "full_overall_eval_runner.py",
        "representation_and_objectives_pristine_collator": USER_ROOT / "experiments/archive/representation_and_objectives/scripts/stage_pristine_collate.py",
    }
    return {
        "status": "COMPLIANT_EVAL_PREFLIGHT",
        "created_utc": now_utc(),
        "target": target,
        "description": description,
        "family": family,
        "run_dir": str(run_dir),
        "model_root": str(model_root),
        "model_path": str(model_path),
        "endpoint": endpoint,
        "model_root_exists": model_root.exists(),
        "model_path_exists": model_path.exists(),
        "metrics_exists": (run_dir / "scientific_metrics.json").exists(),
        "out_root": str(out_root),
        "collate_root": str(collate_root / target),
        "columns_requested": columns,
        "required_paths": {k: {"path": str(v), "exists": pathlib.Path(v).exists()} for k, v in full_required.items()},
        "zero_shot_specs": ZERO_SHOT_TASKS,
        "reading_data_path": str((PRISTINE_FULL / "reading/reading_data.csv").resolve()),
        "superglue_data_dir": str((PRISTINE_FULL / "glue_filtered").resolve()),
        "aoa_expected": {"steps": base.AOA_STEPS, "row_count_per_step": 8005, "min_context": 0},
        "budget_semantics": {
            "reinvest": "end-to-end submission-relevant only for the reinvest arm because tokenizer training text and pretraining 10M pool are the same frozen pool",
            "clean_qwen": "fixed-tokenizer scientific control only: it isolates pretraining-corpus effects under the reinvest tokenizer but its tokenizer-training text plus clean-Qwen pretraining text exceeds the 10M union"
        },
        "collate_command_template_after_predictions_exist": [
            sys.executable,
            "experiments/archive/representation_and_objectives/scripts/stage_pristine_collate.py",
            "--full-root", str(out_root.resolve()),
            "--target", target,
            "--model-root", str(model_root.resolve()),
            "--out-dir", str((collate_root / target).resolve()),
            "--pristine-ewok-predictions", f"{out_root}/official_outputs/{target}/EWoK/<latest>/zero_shot/mlm/ewok/ewok_filtered/predictions.json",
            "--aoa-dir", f"{out_root}/aoa_outputs/{target}",
            "--endpoint", endpoint,
            "--tag", target,
        ],
    }


def run_one(args: argparse.Namespace) -> None:
    defaults = ARM_DEFAULTS[args.arm]
    target = args.target or defaults["target"]
    run_dir = pathlib.Path(args.run_dir) if args.run_dir else pathlib.Path(defaults["run_dir"])
    endpoint = args.endpoint or defaults["endpoint"]
    out_root = pathlib.Path(args.out_root)
    collate_root = pathlib.Path(args.collate_root)
    columns = args.columns or [t["column"] for t in ZERO_SHOT_TASKS] + ["Reading", "SuperGLUE", "AoA"]
    configure_base(out_root, target, run_dir, endpoint, defaults["description"], defaults["family"])
    out_root.mkdir(parents=True, exist_ok=True)

    if args.preflight_only:
        payload = preflight_payload(out_root, collate_root, target, run_dir, endpoint, defaults["description"], defaults["family"], columns)
        out = out_root / f"{target}_preflight.json"
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": payload["status"],
            "out_json": str(out),
            "target": target,
            "model_path_exists": payload["model_path_exists"],
            "missing_required": [k for k, v in payload["required_paths"].items() if not v["exists"]],
            "columns": columns,
        }, indent=2), flush=True)
        return

    model_path = run_dir / "hf_model" / endpoint
    if not model_path.exists():
        raise FileNotFoundError(f"Cannot evaluate before endpoint exists: {model_path}")
    if not (run_dir / "scientific_metrics.json").exists():
        raise FileNotFoundError(f"Training metrics missing; inspect run before evaluation: {run_dir / 'scientific_metrics.json'}")

    payload = base.load_or_new_payload(target, int(args.gpu), args.force)
    payload.setdefault("tasks", {})
    print(json.dumps({"event": "compliant_eval_start", "target": target, "gpu": args.gpu, "columns": columns, "model_path": str(model_path)}), flush=True)
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
    print(json.dumps({"status": "COMPLIANT_EVAL_DONE", "target": target, "per_target_json": str(base.per_target_path(target)), "official_overall": payload.get("official_overall")}, indent=2, ensure_ascii=False), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARM_DEFAULTS))
    ap.add_argument("--target", default="")
    ap.add_argument("--run-dir", default="")
    ap.add_argument("--endpoint", default="chck_100M")
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--collate-root", default=str(DEFAULT_COLLATE_ROOT))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--columns", nargs="*", default=None, help="Subset among zero-shot columns, Reading, SuperGLUE, AoA")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()
    run_one(args)


if __name__ == "__main__":
    main()
