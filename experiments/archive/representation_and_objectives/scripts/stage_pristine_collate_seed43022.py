#!/usr/bin/env python3
"""research: stage compact_view_reinvest seed43022 predictions into one pristine official coordinate.

This script does not edit official code or evaluation data.  It creates a new
results tree containing:
  * research seed43022 full predictions for unchanged columns,
  * research official-data EWoK predictions (7618-row pristine EWoK),
  * research official min_context=0 AoA surprisals/scores (8005 rows/checkpoint),
then runs the unmodified upstream collator from the fresh research checkout and
scores the resulting collated JSON on the same pristine evaluation data.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any


def find_user_root() -> Path:
    return _PUBLIC_ROOT

ROOT = find_user_root()
STUDY = ROOT / "experiments/archive" / 'representation_and_objectives'
WORKSPACE = STUDY
OUT_DIR = WORKSPACE / "data" / "pristine_collate_seed43022"
PRISTINE_STRICT = WORKSPACE / "data" / "pristine_official_coordinate" / "babylm-eval" / "strict"
LOCAL_STRICT_WITH_GENERATED_DATA = ROOT / "experiments/archive" / 'initial_model_studies' / "repos" / "babylm-eval" / "strict"
research = WORKSPACE / "data" / "compact_reinvest_full_eval"
AOA = WORKSPACE / "data" / "official_aoa_min0_reinvest"
EWOK = WORKSPACE / "data" / "official_ewok_reeval"
MODEL_ROOT = ROOT / "experiments/archive" / 'frontier_consolidation' / "training" / "runs" / "cleanqwen_fineweb_compact_view_reinvest_16k_seed43022" / "hf_model"
MODEL_PATH = MODEL_ROOT / "chck_100M"

ZERO_SOURCES = {
    "blimp/blimp_filtered/predictions.json": research / "official_outputs" / "compact_view_reinvest" / "BLiMP" / "chck_100M" / "full_compact_view_reinvest_BLiMP" / "zero_shot" / "mlm" / "blimp" / "blimp_filtered" / "predictions.json",
    "blimp/supplement_filtered/predictions.json": research / "official_outputs" / "compact_view_reinvest" / "Supplement" / "chck_100M" / "full_compact_view_reinvest_Supplement" / "zero_shot" / "mlm" / "blimp" / "supplement_filtered" / "predictions.json",
    # Corrected official-data EWoK predictions; do not use research stale-local EWoK here.
    "ewok/ewok_filtered/predictions.json": EWOK / "official_outputs" / "EWoK" / "chck_100M" / "official_ewok_reinvest" / "zero_shot" / "mlm" / "ewok" / "ewok_filtered" / "predictions.json",
    "entity_tracking/entity_tracking/predictions.json": research / "official_outputs" / "compact_view_reinvest" / "Entity" / "chck_100M" / "full_compact_view_reinvest_Entity" / "zero_shot" / "mlm" / "entity_tracking" / "entity_tracking" / "predictions.json",
    "comps/comps/predictions.json": research / "official_outputs" / "compact_view_reinvest" / "COMPS" / "chck_100M" / "full_compact_view_reinvest_COMPS" / "zero_shot" / "mlm" / "comps" / "comps" / "predictions.json",
    "global_piqa_parallel/global_piqa_parallel/predictions.json": research / "official_outputs" / "compact_view_reinvest" / "GlobalPIQA_parallel" / "chck_100M" / "full_compact_view_reinvest_GlobalPIQA_parallel" / "zero_shot" / "mlm" / "global_piqa_parallel" / "global_piqa_parallel" / "predictions.json",
    "global_piqa_nonparallel/global_piqa_nonparallel/predictions.json": research / "official_outputs" / "compact_view_reinvest" / "GlobalPIQA_nonparallel" / "chck_100M" / "full_compact_view_reinvest_GlobalPIQA_nonparallel" / "zero_shot" / "mlm" / "global_piqa_nonparallel" / "global_piqa_nonparallel" / "predictions.json",
    "reading/predictions.json": research / "official_outputs" / "compact_view_reinvest" / "Reading" / "chck_100M" / "full_compact_view_reinvest_Reading" / "zero_shot" / "mlm" / "reading" / "predictions.json",
    "AoA_word/surprisal.json": AOA / "hf_model_local_ckpts" / "main" / "zero_shot" / "mlm" / "AoA_word" / "surprisal.json",
    "AoA_word/aoa_score.json": AOA / "hf_model_local_ckpts" / "main" / "zero_shot" / "mlm" / "AoA_word" / "aoa_score.json",
}
SUPERGLUE = ["boolq", "mnli", "mrpc", "multirc", "qqp", "rte", "wsc"]
FINETUNE_METRIC = {"boolq": "accuracy", "mnli": "accuracy", "mrpc": "f1", "multirc": "accuracy", "qqp": "f1", "rte": "accuracy", "wsc": "accuracy"}
OVERALL_KEYS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "SuperGLUE", "GlobalPIQA", "Reading", "AoA"]
NLP_KEYS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "SuperGLUE"]
HUMAN_KEYS = ["Reading", "AoA"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> dict[str, Any]:
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return {"cmd": cmd, "cwd": str(cwd), "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr, "elapsed_sec": round(time.time() - t0, 3)}


def copy_or_link(src: Path, dst: Path, symlink: bool) -> dict[str, Any]:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if symlink:
        os.symlink(src.resolve(), dst)
        mode = "symlink"
    else:
        shutil.copy2(src, dst)
        mode = "copy"
    return {"src": str(src), "dst": str(dst), "mode": mode, "size_bytes": src.stat().st_size, "sha256": sha256_file(src)}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def prediction_lengths(pred_obj: Any) -> dict[str, int] | int | None:
    if not isinstance(pred_obj, dict):
        return None
    # GlobalPIQA is keyed by example ids and each value has exactly one prediction.
    if pred_obj and all(isinstance(v, dict) and "predictions" in v for v in pred_obj.values()):
        return {str(k).lower(): len(v.get("predictions", [])) for k, v in pred_obj.items()}
    return None


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return sum(1 for _ in f)


def parse_results_txt(path: Path, metric: str) -> float:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        k, _, v = line.partition(":")
        if k.strip() == metric:
            return float(v.strip()) * 100.0
    raise ValueError(f"metric {metric} not found in {path}")


def score_zero_shot_with_official_functions(collated: dict[str, Any], env: dict[str, str]) -> dict[str, Any]:
    sys.path.insert(0, str((PRISTINE_STRICT / "evaluation_pipeline").resolve()))
    import calculate_results_from_pred as calc  # type: ignore
    from transformers import AutoTokenizer  # type: ignore

    full = PRISTINE_STRICT / "evaluation_data" / "full_eval"
    scores: dict[str, float] = {}
    details: dict[str, Any] = {}

    scores["BLiMP"] = float(calc._calculate_blimp_results(collated["blimp"], full / "blimp_filtered"))
    scores["Supplement"] = float(calc._calculate_blimp_results(collated["blimp_supplement"], full / "supplement_filtered"))
    scores["EWoK"] = float(calc._calculate_ewok_results(collated["ewok"], full / "ewok_filtered"))
    # Collator key is entity_tracking_filtered; score against all full Entity files with the same
    # skip-`nothing` filtering used by sentence_zero_shot/read_files.py and documented in collate_preds.py.
    import collate_preds as coll  # type: ignore
    subtask_to_targets: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for data_path in sorted((full / "entity_tracking").glob("*.jsonl")):
        with data_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.strip():
                    continue
                ex = json.loads(line)
                if any("nothing" in option for option in ex["options"]):
                    continue
                subtask_to_targets[f'{data_path.stem}_{ex["numops"]}_ops'].append(ex)
    et_frac: dict[str, float] = {}
    et_counts: dict[str, dict[str, int]] = {}
    for subtask, pred_block in collated["entity_tracking_filtered"].items():
        preds = pred_block["predictions"]
        targets = subtask_to_targets[subtask]
        if len(preds) != len(targets):
            raise RuntimeError(f"Entity length mismatch for {subtask}: {len(preds)} predictions vs {len(targets)} targets")
        correct = 0
        for pred, ex in zip(preds, targets):
            pv = pred["pred"].strip() if isinstance(pred["pred"], str) else pred["pred"]
            tv = ex["options"][0].strip() if isinstance(ex["options"][0], str) else ex["options"][0]
            correct += int(pv == tv)
        et_frac[subtask] = correct / len(targets)
        et_counts[subtask] = {"correct": correct, "total": len(targets)}
    scores["Entity"] = 100.0 * sum(et_frac.values()) / len(et_frac)
    details["Entity"] = {"subtask_fraction_scores": et_frac, "subtask_counts": et_counts}
    scores["COMPS"] = float(calc._calculate_comps_results(collated["comps"], full / "comps"))

    # GlobalPIQA is scored by the unmodified collator helper against official eng_latn files.
    gp_scores = []
    for label, key, data_file, task_name in [
        ("GlobalPIQA_parallel", "global_piqa_parallel", LOCAL_STRICT_WITH_GENERATED_DATA / "evaluation_data" / "full_eval" / "global_piqa_parallel" / "eng_latn.jsonl", "global_piqa_parallel"),
        ("GlobalPIQA_nonparallel", "global_piqa_nonparallel", LOCAL_STRICT_WITH_GENERATED_DATA / "evaluation_data" / "full_eval" / "global_piqa_nonparallel" / "eng_latn.jsonl", "global_piqa_nonparallel"),
    ]:
        frac = coll._calculate_global_piqa_results(collated[key], data_file, task_name)[task_name]
        score = 100.0 * frac
        scores[label] = score
        gp_scores.append(score)
        details[label] = {"fraction": float(frac), "score": score, "data_file": str(data_file)}
    scores["GlobalPIQA"] = sum(gp_scores) / 2.0

    reading_frac = coll._calculate_reading_results(collated["reading"], full / "reading" / "reading_data.csv")
    scores["Reading"] = 100.0 * (float(reading_frac["spr"]) + float(reading_frac["rt"])) / 2.0
    details["Reading"] = {"self_paced_fraction": float(reading_frac["spr"]), "eye_tracking_fraction": float(reading_frac["rt"]), "leaderboard_score": scores["Reading"]}

    sg_scores = []
    sg_detail = {}
    for task in SUPERGLUE:
        metric = FINETUNE_METRIC[task]
        # Scoring function in calculate_results_from_pred computes accuracy for all tasks, but the official
        # leaderboard/print_results_table uses f1 for MRPC and QQP. Use results.txt for those primary metrics
        # and also compute prediction accuracy for transparency below.
        results_path = research / "superglue_results" / "compact_view_reinvest" / task / "chck_100M" / "main" / "finetune" / task / "results.txt"
        primary = parse_results_txt(results_path, metric)
        sg_scores.append(primary)
        sg_detail[task] = {"metric": metric, "score": primary, "results_txt": str(results_path)}
    scores["SuperGLUE"] = sum(sg_scores) / len(sg_scores)
    details["SuperGLUE"] = sg_detail

    # AoA: the collated JSON contains the official AoA score object separately from the
    # 152095-row `aoa_surprisals` object.  The unmodified collator has already checked the
    # surprisal block size; leaderboard arithmetic uses 100 * raw curve fitness from aoa_score.json.
    aoa_obj = collated["aoa"]
    aoa_raw = float(aoa_obj.get("aoa", aoa_obj.get("curve_fitness", 0.0)))
    scores["AoA"] = float(100.0 * aoa_raw)
    details["AoA"] = {"raw_curve_fitness": float(aoa_raw), "leaderboard_score": scores["AoA"], "source": "collated aoa_score.json"}

    vals = [scores[k] for k in OVERALL_KEYS]
    official = {
        "scores": {k: scores[k] for k in OVERALL_KEYS},
        "Overall": sum(vals) / len(vals),
        "NLP_average": sum(scores[k] for k in NLP_KEYS) / len(NLP_KEYS),
        "Human_like_average": sum(scores[k] for k in HUMAN_KEYS) / len(HUMAN_KEYS),
        "official_like_arithmetic": "mean(BLiMP, Supplement, EWoK, Entity, COMPS, SuperGLUE, GlobalPIQA, Reading, AoA_leaderboard_score)",
        "aoa_unit": "leaderboard_score=100*raw_correlation",
    }
    return {"official_overall": official, "details": details}


def summarize_collated(path: Path) -> dict[str, Any]:
    collated = load_json(path)
    out: dict[str, Any] = {
        "top_keys": sorted(collated.keys()),
        "null_keys": sorted([k for k, v in collated.items() if v is None]),
    }
    # Prediction lengths by benchmark.
    for key in ["blimp", "blimp_supplement", "ewok", "entity_tracking_filtered", "comps", "global_piqa_parallel", "global_piqa_nonparallel", "reading", "glue"]:
        v = collated.get(key)
        if isinstance(v, dict):
            if key == "glue":
                out["glue_lengths"] = {t: len(tv.get("predictions", [])) for t, tv in v.items()}
            elif key in ["global_piqa_parallel", "global_piqa_nonparallel"]:
                out[f"{key}_example_count"] = len(v)
                out[f"{key}_prediction_total"] = sum(len(x.get("predictions", [])) for x in v.values())
            elif key == "reading":
                out["reading_lengths"] = {t: len(tv.get("predictions", [])) for t, tv in v.items()}
            else:
                out[f"{key}_lengths"] = {t: len(tv.get("predictions", [])) for t, tv in v.items()}
    aoa_s = collated.get("aoa_surprisals")
    if isinstance(aoa_s, dict):
        rows = aoa_s.get("results", [])
        c = collections.Counter(r.get("step") for r in rows)
        out["aoa_surprisal_num_rows"] = len(rows)
        out["aoa_surprisal_num_steps"] = len(c)
        out["aoa_surprisal_row_count_values"] = sorted(set(c.values()))
        out["aoa_surprisal_steps"] = sorted(c.keys(), key=lambda s: (len(str(s)), str(s)))
    aoa = collated.get("aoa")
    if isinstance(aoa, dict):
        out["aoa_score_keys"] = sorted(aoa.keys())
        out["aoa_score_results_keys"] = sorted(aoa.get("results", {}).keys()) if isinstance(aoa.get("results"), dict) else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--copy", action="store_true", help="Copy instead of symlink")
    args = ap.parse_args()

    if args.clean and OUT_DIR.exists():
        for child in OUT_DIR.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
    elif OUT_DIR.exists():
        # Remove any incomplete children from a failed earlier attempt; keep this inside OUT_DIR.
        for child in OUT_DIR.iterdir():
            if child.name in {"results", "hf_home", "datasets", "modules", "transformers", "tmp"}:
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child)
                else:
                    child.unlink()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_dir = OUT_DIR / "results"
    if results_dir.exists():
        shutil.rmtree(results_dir)
    model_stem = MODEL_ROOT.stem
    zero_root = results_dir / model_stem / "main" / "zero_shot" / "mlm"
    fine_root = results_dir / model_stem / "main" / "finetune"

    env = os.environ.copy()
    cache = OUT_DIR / "hf_cache"
    tmp = OUT_DIR / "tmp"
    for k, rel in [
        ("HF_HOME", "hf_home"), ("HF_HUB_CACHE", "hf_home/hub"), ("HF_DATASETS_CACHE", "datasets"),
        ("TRANSFORMERS_CACHE", "transformers"), ("HF_MODULES_CACHE", "modules"), ("TMPDIR", "tmp")]:
        p = OUT_DIR / rel
        p.mkdir(parents=True, exist_ok=True)
        env[k] = str(p.resolve())
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONPATH"] = str(PRISTINE_STRICT.resolve()) + os.pathsep + str((PRISTINE_STRICT / "evaluation_pipeline").resolve()) + os.pathsep + env.get("PYTHONPATH", "")

    staged = []
    for rel, src in ZERO_SOURCES.items():
        staged.append(copy_or_link(src, zero_root / rel, symlink=not args.copy))
    for task in SUPERGLUE:
        src = research / "superglue_results" / "compact_view_reinvest" / task / "chck_100M" / "main" / "finetune" / task / "predictions.json"
        staged.append(copy_or_link(src, fine_root / task / "predictions.json", symlink=not args.copy))

    collate_cmd = [
        sys.executable, "evaluation_pipeline/collate_preds.py",
        "--model_path_or_name", str(MODEL_ROOT.resolve()),
        "--backend", "mlm",
        "--results_dir", str(results_dir.resolve()),
        "--revision_name", "main",
        "--track", "strict-small",
    ]
    collate_run = run(collate_cmd, cwd=PRISTINE_STRICT, env=env)
    collated_path = results_dir / model_stem / "all_full_preds_and_fast_scores_mlm.json"
    # Also create the path expected by calculate_results_from_pred.py.
    calc_path = results_dir / model_stem / "main" / "all_full_preds_mlm.json"
    if collated_path.exists():
        calc_path.parent.mkdir(parents=True, exist_ok=True)
        if calc_path.exists() or calc_path.is_symlink():
            calc_path.unlink()
        os.symlink(collated_path.resolve(), calc_path)

    calc_cmd = [
        sys.executable, "evaluation_pipeline/calculate_results_from_pred.py",
        "--model_path_or_name", str(MODEL_ROOT.resolve()),
        "--backend", "mlm",
        "--results_dir", str(results_dir.resolve()),
        "--evaluation_data_dir", str((PRISTINE_STRICT / "evaluation_data").resolve()),
        "--revision_name", "main",
    ]
    calc_run = run(calc_cmd, cwd=PRISTINE_STRICT, env=env) if calc_path.exists() else {"skipped": True}

    collated_summary: dict[str, Any] = {}
    score_summary: dict[str, Any] = {}
    if collated_path.exists():
        collated_summary = summarize_collated(collated_path)
        collated = load_json(collated_path)
        if not collated_summary.get("null_keys"):
            score_summary = score_zero_shot_with_official_functions(collated, env)
        else:
            score_summary = {"skipped_due_to_null_keys": collated_summary.get("null_keys")}

    leader = 41.8
    corrected_prior = 42.07234127469371
    if score_summary.get("official_overall"):
        overall = score_summary["official_overall"]["Overall"]
        score_summary["official_overall"]["visible_leader_overall"] = leader
        score_summary["official_overall"]["margin_over_visible_leader"] = overall - leader
        score_summary["official_overall"]["delta_vs_step036_corrected_arithmetic"] = overall - corrected_prior

    result = {
        "status": "PRISTINE_COLLATE_SEED43022",
        "created_utc": now(),
        "purpose": "One unmodified-collator submission coordinate for compact_view_reinvest seed43022 using pristine official EWoK and official min_context=0 AoA.",
        "paths": {"out_dir": str(OUT_DIR), "results_dir": str(results_dir), "collated_path": str(collated_path), "calc_compatible_symlink": str(calc_path)},
        "official_coordinate": {
            "pristine_strict": str(PRISTINE_STRICT),
            "local_strict_with_official_generated_globalpiqa_data": str(LOCAL_STRICT_WITH_GENERATED_DATA),
            "model_root": str(MODEL_ROOT),
            "model_path": str(MODEL_PATH),
            "uses_step036_pristine_ewok": True,
            "uses_step035_min_context0_aoa": True,
            "official_code_unmodified_by_this_script": True,
            "globalpiqa_data_note": "GlobalPIQA eng_latn.jsonl files are not present in the BabyLM-2026-Strict-Evals snapshot; the official global_piqa/dl.py generates them from mrlbenchmarks/global-piqa-* datasets. This scorer uses the already-generated INITIAL_MODEL_STUDIES files and records their hashes.",
        },
        "staged_files": staged,
        "collate_run": collate_run,
        "calc_run_stdout_parser": calc_run,
        "collated_summary": collated_summary,
        "score_summary": score_summary,
        "input_corrections": {
            "overall": 42.086785719138156,
            "local_ewok": 53.67,
            "official_pristine_ewok": 53.54,
            "min_context20_aoa": 0.0,
            "official_min_context0_aoa": 0.0,
            "corrected_arithmetic": corrected_prior,
        },
    }
    out_json = OUT_DIR / "pristine_collate_seed43022_summary.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "collate_returncode": collate_run.get("returncode"),
        "collated_exists": collated_path.exists(),
        "null_keys": collated_summary.get("null_keys"),
        "aoa_row_count_values": collated_summary.get("aoa_surprisal_row_count_values"),
        "ewok_lengths_total": sum(collated_summary.get("ewok_lengths", {}).values()) if isinstance(collated_summary.get("ewok_lengths"), dict) else None,
        "overall": score_summary.get("official_overall", {}).get("Overall") if isinstance(score_summary, dict) else None,
        "margin_over_visible_leader": score_summary.get("official_overall", {}).get("margin_over_visible_leader") if isinstance(score_summary, dict) else None,
        "out_json": str(out_json),
    }, indent=2))


if __name__ == "__main__":
    main()
