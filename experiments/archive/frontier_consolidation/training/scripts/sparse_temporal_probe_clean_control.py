#!/usr/bin/env python3
"""research sparse temporal probe for clean-Qwen seed-control ladders.

Runs the same selected task slices used by the compact_view_reinvest sparse probe,
but on the inherited clean-Qwen seed43022/seed43122 checkpoint ladders from
compact_experience. This is checkpoint-only evaluation: no pretraining and no full official
surface. It provides the matched control needed to interpret whether reinvest seed
spread is ordinary base-recipe behavior or excess/changed spread under compact
view reinvestment.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import gc
import json
import os
import pathlib
import shutil
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForMaskedLM

USER_ROOT = pathlib.Path.cwd()
if not (USER_ROOT / "experiments").exists():
    p = _public_path('experiments/archive/frontier_consolidation/training/scripts/sparse_temporal_probe_clean_control.py')
    for parent in [p] + list(p.parents):
        if (parent / "experiments").exists():
            USER_ROOT = parent
            break

STUDY = USER_ROOT / "experiments/archive" / 'frontier_consolidation'
STRICT = USER_ROOT / "experiments/archive" / 'initial_model_studies' / "repos" / "babylm-eval" / "strict"
if str(STRICT) not in sys.path:
    sys.path.insert(0, str(STRICT))

from evaluation_pipeline.sentence_zero_shot.dataset import get_dataloader  # noqa: E402
from evaluation_pipeline.sentence_zero_shot.compute_results import compute_results  # noqa: E402
from evaluation_pipeline.sentence_zero_shot.run import process_results  # noqa: E402

SEED_RUNS = {
    "43022": USER_ROOT / "experiments/archive" / 'compact_experience' / "training" / "runs" / "qwen_clean_aligned_16k_seed43022",
    "43122": USER_ROOT / "experiments/archive" / 'compact_experience' / "training" / "runs" / "qwen_clean_aligned_16k_seed43122",
}

BLIMP_WORST = [
    "principle_A_reconstruction",
    "superlative_quantifiers_1",
    "principle_A_c_command",
    "wh_questions_object_gap",
    "sentential_subject_island",
    "distractor_agreement_relative_clause",
    "left_branch_island_simple_question",
    "sentential_negation_npi_scope",
    "matrix_question_npi_licensor_present",
    "principle_A_case_2",
]
BLIMP_CONTROL = [
    "principle_A_domain_1",
    "left_branch_island_echo_question",
    "wh_island",
    "only_npi_licensor_present",
    "drop_argument",
    "superlative_quantifiers_2",
    "principle_A_domain_2",
    "wh_vs_that_with_gap",
]
SUPPLEMENT_ALL = [
    "qa_congruence_easy",
    "qa_congruence_tricky",
    "hypernym",
    "subject_aux_inversion",
    "turn_taking",
]
EWOK_ALL = [
    "agent-properties",
    "material-dynamics",
    "material-properties",
    "physical-dynamics",
    "physical-interactions",
    "physical-relations",
    "quantitative-properties",
    "social-interactions",
    "social-properties",
    "social-relations",
    "spatial-relations",
]
ENTITY_FILES = ["regular", "ambiref", "move_contents"]


@dataclass(frozen=True)
class TaskGroup:
    name: str
    task: str
    source_dir: pathlib.Path
    stems: Optional[List[str]]
    batch_size: int
    non_causal_batch_size: int
    description: str


TASK_GROUPS: Dict[str, TaskGroup] = {
    "blimp_worst": TaskGroup(
        name="blimp_worst",
        task="blimp",
        source_dir=STRICT / "evaluation_data" / "fast_eval" / "blimp_fast",
        stems=BLIMP_WORST,
        batch_size=96,
        non_causal_batch_size=96,
        description="BLiMP files where reinvest seed43122 was much below seed43022 at 100M.",
    ),
    "blimp_control": TaskGroup(
        name="blimp_control",
        task="blimp",
        source_dir=STRICT / "evaluation_data" / "fast_eval" / "blimp_fast",
        stems=BLIMP_CONTROL,
        batch_size=96,
        non_causal_batch_size=96,
        description="BLiMP files where reinvest seed43122 matched or exceeded seed43022 at 100M.",
    ),
    "supplement_all": TaskGroup(
        name="supplement_all",
        task="blimp",
        source_dir=STRICT / "evaluation_data" / "fast_eval" / "supplement_fast",
        stems=SUPPLEMENT_ALL,
        batch_size=128,
        non_causal_batch_size=128,
        description="All fast BLiMP Supplement slices.",
    ),
    "ewok_all": TaskGroup(
        name="ewok_all",
        task="ewok",
        source_dir=STRICT / "evaluation_data" / "fast_eval" / "evaluation_data" / "fast_eval" / "ewok_fast",
        stems=EWOK_ALL,
        batch_size=96,
        non_causal_batch_size=96,
        description="All EWoK fast domains.",
    ),
    "entity_full": TaskGroup(
        name="entity_full",
        task="entity_tracking",
        source_dir=STRICT / "evaluation_data" / "full_eval" / "entity_tracking",
        stems=ENTITY_FILES,
        batch_size=96,
        non_causal_batch_size=96,
        description="Full Entity Tracking rows except the options skipped by the evaluator.",
    ),
}


class Args:
    pass


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", default=["43022", "43122"], choices=sorted(SEED_RUNS))
    ap.add_argument("--exposures", nargs="+", type=int, default=[1, 10, 40, 100], help="Checkpoint exposures in M words")
    ap.add_argument("--task-groups", nargs="+", default=["blimp_worst", "blimp_control", "supplement_all", "ewok_all", "entity_full"], choices=sorted(TASK_GROUPS))
    ap.add_argument("--gpu", default="1", help="CUDA_VISIBLE_DEVICES value, or cpu")
    ap.add_argument("--run-dir", default=os.environ.get("QIUSHI_AI_LAB_RUN_DIR"), help="Output directory")
    return ap.parse_args()


def checkpoint_path(seed: str, exposure_m: int) -> pathlib.Path:
    run = SEED_RUNS[seed]
    ckpt = run / "hf_model" / f"chck_{exposure_m}M"
    if exposure_m == 100:
        ckpt = run / "hf_model" / "chck_100M"
    return ckpt


def safe_copy_slice(task: TaskGroup, dest: pathlib.Path) -> pathlib.Path:
    out = dest / task.name
    out.mkdir(parents=True, exist_ok=True)
    stems = task.stems if task.stems is not None else [p.stem for p in sorted(task.source_dir.glob("*.jsonl"))]
    copied = []
    for stem in stems:
        src = task.source_dir / f"{stem}.jsonl"
        if not src.exists():
            raise FileNotFoundError(f"missing source slice for {task.name}: {src}")
        dst = out / src.name
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copyfile(src, dst)
        copied.append(src.name)
    manifest = {
        "task_group": task.name,
        "task": task.task,
        "source_dir": str(task.source_dir),
        "files": copied,
        "description": task.description,
    }
    (out / "slice_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return out


def make_eval_args(model_path: pathlib.Path, data_path: pathlib.Path, task: TaskGroup) -> Args:
    a = Args()
    a.data_path = data_path
    a.task = task.task
    a.model_path_or_name = str(model_path)
    a.backend = "mlm"
    a.output_dir = pathlib.Path("unused_step027_clean_sparse_temporal_probe")
    a.images_path = None
    a.image_split = None
    a.image_template = None
    a.revision_name = None
    a.min_temperature = 1.0
    a.max_temperature = None
    a.temperature_interval = 0.05
    a.batch_size = task.batch_size
    a.non_causal_batch_size = task.non_causal_batch_size
    a.full_sentence_scores = False
    a.save_predictions = False
    return a


def average_score(task: TaskGroup, accuracies: dict, averages: dict) -> tuple[float, dict[str, float]]:
    temp = sorted(averages.keys(), key=float)[0]
    avg = float(averages[temp])
    uid_scores = {str(k): float(v) for k, v in accuracies[temp].get("UID", {}).items()}
    return avg, uid_scores


def evaluate_task_group(model, model_path: pathlib.Path, task: TaskGroup, data_dir: pathlib.Path) -> dict[str, Any]:
    a = make_eval_args(model_path, data_dir, task)
    dataloader = get_dataloader(a)
    temps = [1.0]
    t0 = time.time()
    with torch.no_grad():
        results, _preds = compute_results(a, model, dataloader, temps)
    accuracies, averages = process_results(a, results)
    avg, by_uid = average_score(task, accuracies, averages)
    elapsed = time.time() - t0
    return {
        "task_group": task.name,
        "task": task.task,
        "average_score": avg,
        "subtask_scores": by_uid,
        "n_subtasks": len(by_uid),
        "elapsed_sec": round(elapsed, 3),
        "description": task.description,
    }


def load_model(model_path: pathlib.Path, device: torch.device):
    model = AutoModelForMaskedLM.from_pretrained(str(model_path), trust_remote_code=True)
    model.to(device)
    model.eval()
    return model


def build_comparisons(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by = {(r["seed"], r["exposure_m"]): r for r in rows}
    exposures = sorted({r["exposure_m"] for r in rows})
    out: dict[str, Any] = {"by_exposure": {}, "task_group_summary": {}}
    for exp in exposures:
        r0 = by.get(("43022", exp))
        r1 = by.get(("43122", exp))
        if not r0 or not r1:
            continue
        task_groups = sorted(set(r0["scores"]) & set(r1["scores"]))
        diffs = {}
        for name in task_groups:
            s0 = r0["scores"][name]["average_score"]
            s1 = r1["scores"][name]["average_score"]
            diffs[name] = s1 - s0
        out["by_exposure"][str(exp)] = diffs
    for name in sorted({k for r in rows for k in r["scores"]}):
        diffs_by_exp = {int(e): d[name] for e, d in out["by_exposure"].items() if name in d}
        if not diffs_by_exp:
            continue
        early_exps = [e for e in diffs_by_exp if e <= 10]
        late_exps = [e for e in diffs_by_exp if e >= 40]
        early_mean = sum(diffs_by_exp[e] for e in early_exps) / len(early_exps) if early_exps else None
        late_mean = sum(diffs_by_exp[e] for e in late_exps) / len(late_exps) if late_exps else None
        final = diffs_by_exp.get(100, diffs_by_exp[max(diffs_by_exp)])
        out["task_group_summary"][name] = {
            "seed43122_minus_43022_by_exposure": diffs_by_exp,
            "early_mean_1_10M": early_mean,
            "late_mean_40_100M": late_mean,
            "final_100M": final,
            "max_abs_difference": max(abs(x) for x in diffs_by_exp.values()),
        }
    return out


def write_note(summary: dict[str, Any], out_json: pathlib.Path, out_md: pathlib.Path) -> None:
    lines: list[str] = []
    lines.append("# research clean-Qwen sparse temporal seed-control\n\n")
    lines.append("Existing inherited clean-Qwen checkpoint ladders only; same selected slices as the reinvest sparse temporal run; not a full official evaluation and no training.\n\n")
    lines.append("## Purpose\n")
    lines.append("This measures whether the seed43022/seed43122 slice-time spread observed for compact_view_reinvest is already present in the inherited clean-Qwen recipe or is enlarged/changed by the compact-view reinvestment corpus.\n\n")
    if summary.get("failures"):
        lines.append("## Failures\n")
        for f in summary["failures"]:
            lines.append(f"- {f}\n")
        lines.append("\n")
    lines.append("## Seed43122 minus seed43022 by exposure\n")
    for exp, diffs in summary["comparisons"]["by_exposure"].items():
        ordered = ", ".join(f"{k}={v:.3f}" for k, v in sorted(diffs.items()))
        lines.append(f"- {exp}M: {ordered}\n")
    lines.append("\n## Task-group summaries\n")
    for name, rec in sorted(summary["comparisons"]["task_group_summary"].items()):
        lines.append(f"- {name}: early_mean={rec['early_mean_1_10M']}, late_mean={rec['late_mean_40_100M']}, final={rec['final_100M']}, max_abs={rec['max_abs_difference']}\n")
    lines.append(f"\nMachine-readable output: `{out_json}`\n")
    out_md.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    ns = parse_args()
    run_dir = pathlib.Path(ns.run_dir) if ns.run_dir else STUDY / "training" / "runs" / "clean_sparse_temporal_probe"
    run_dir.mkdir(parents=True, exist_ok=True)
    if ns.gpu.lower() != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ns.gpu
    device = torch.device("cuda") if (ns.gpu.lower() != "cpu" and torch.cuda.is_available()) else torch.device("cpu")

    data_root = run_dir / "data_slices"
    task_dirs = {name: safe_copy_slice(TASK_GROUPS[name], data_root) for name in ns.task_groups}
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started = time.time()
    manifest = {
        "status": "CLEAN_SPARSE_TEMPORAL_PROBE_RUNNING",
        "purpose": "Same sparse task-slice temporal comparison on clean-Qwen seeds 43022/43122, used as a matched seed-control for the compact_view_reinvest sparse probe.",
        "seeds": ns.seeds,
        "exposures_m": ns.exposures,
        "task_groups": {name: TASK_GROUPS[name].description for name in ns.task_groups},
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "seed_runs": {k: str(v) for k, v in SEED_RUNS.items()},
        "run_dir": str(run_dir),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    for seed in ns.seeds:
        for exp in ns.exposures:
            model_path = checkpoint_path(seed, exp)
            rec: dict[str, Any] = {
                "seed": seed,
                "exposure_m": exp,
                "checkpoint_path": str(model_path),
                "scores": {},
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            if not (model_path / "model.safetensors").exists():
                failures.append({"seed": seed, "exposure_m": exp, "error": f"missing checkpoint {model_path}"})
                continue
            t_load = time.time()
            model = load_model(model_path, device)
            rec["model_load_sec"] = round(time.time() - t_load, 3)
            try:
                for name in ns.task_groups:
                    task = TASK_GROUPS[name]
                    print(json.dumps({"event": "task_start", "seed": seed, "exposure_m": exp, "task_group": name}, ensure_ascii=False), flush=True)
                    try:
                        tres = evaluate_task_group(model, model_path, task, task_dirs[name])
                        rec["scores"][name] = tres
                        print(json.dumps({"event": "task_done", "seed": seed, "exposure_m": exp, "task_group": name, "score": tres["average_score"], "elapsed_sec": tres["elapsed_sec"]}, ensure_ascii=False), flush=True)
                    except Exception as e:
                        err = {"seed": seed, "exposure_m": exp, "task_group": name, "error": repr(e)}
                        failures.append(err)
                        print(json.dumps({"event": "task_error", **err}, ensure_ascii=False), flush=True)
                rec["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                rows.append(rec)
                partial = {"rows": rows, "failures": failures, "comparisons": build_comparisons(rows), "elapsed_sec": round(time.time() - started, 3)}
                (run_dir / "partial_summary.json").write_text(json.dumps(partial, indent=2) + "\n", encoding="utf-8")
            finally:
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    summary = {
        "status": "CLEAN_SPARSE_TEMPORAL_PROBE_COMPLETE",
        "purpose": manifest["purpose"],
        "seeds": ns.seeds,
        "exposures_m": ns.exposures,
        "task_groups": {name: {"task": TASK_GROUPS[name].task, "description": TASK_GROUPS[name].description, "stems": TASK_GROUPS[name].stems} for name in ns.task_groups},
        "rows": rows,
        "failures": failures,
        "comparisons": build_comparisons(rows),
        "elapsed_sec": round(time.time() - started, 3),
    }
    out_json = run_dir / "clean_sparse_temporal_probe_summary.json"
    out_md = run_dir / "clean_sparse_temporal_probe_note.md"
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_note(summary, out_json, out_md)
    print(json.dumps({"status": summary["status"], "out_json": str(out_json), "out_md": str(out_md), "run_dir": str(run_dir), "elapsed_sec": summary["elapsed_sec"], "failures": len(failures)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
