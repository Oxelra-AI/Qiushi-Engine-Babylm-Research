#!/usr/bin/env python3
"""research/039: stage compact_view_reinvest seed43122 predictions into the same official coordinate as seed43022.

This script is intentionally a post-processing artifact for the already-running managed
full seed43122 evaluation. It does not launch training. It requires that the full eval
has produced all non-EWoK zero-shot columns, Reading, SuperGLUE predictions/results,
official min_context=0 AoA, and a pristine-EWoK seed43122 prediction file. If the
managed full eval produced EWoK only against the old INITIAL_MODEL_STUDIES local data, a separate
pristine-EWoK reeval must be run first, exactly like research for seed43022.
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
import shutil
import subprocess
import sys
import time
from typing import Any


def find_user_root() -> Path:
    return _PUBLIC_ROOT

ROOT = find_user_root()
STUDY = ROOT / "experiments/archive/representation_and_objectives"
WORKSPACE = STUDY
PRISTINE_STRICT = WORKSPACE / "data/pristine_official_coordinate/babylm-eval/strict"
LOCAL_STRICT_WITH_GENERATED_DATA = ROOT / "experiments/archive/initial_model_studies/repos/babylm-eval/strict"
FULL = WORKSPACE / "data/seed43122_full_eval"
MODEL_ROOT = WORKSPACE / "training/runs/repl_compact_view_reinvest_seed43122/hf_model"
OUT_DIR = WORKSPACE / "data/pristine_collate_seed43122"
TARGET = "compact_view_reinvest_seed43122"
ENDPOINT = "chck_100M"
SUPERGLUE = ["boolq", "mnli", "mrpc", "multirc", "qqp", "rte", "wsc"]
FINETUNE_METRIC = {"boolq": "accuracy", "mnli": "accuracy", "mrpc": "f1", "multirc": "accuracy", "qqp": "f1", "rte": "accuracy", "wsc": "accuracy"}
OVERALL_KEYS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "SuperGLUE", "GlobalPIQA", "Reading", "AoA"]
NLP_KEYS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "SuperGLUE"]
HUMAN_KEYS = ["Reading", "AoA"]

ZERO_REL_BY_COLUMN = {
    "BLiMP": "blimp/blimp_filtered/predictions.json",
    "Supplement": "blimp/supplement_filtered/predictions.json",
    "EWoK": "ewok/ewok_filtered/predictions.json",
    "Entity": "entity_tracking/entity_tracking/predictions.json",
    "COMPS": "comps/comps/predictions.json",
    "GlobalPIQA_parallel": "global_piqa_parallel/global_piqa_parallel/predictions.json",
    "GlobalPIQA_nonparallel": "global_piqa_nonparallel/global_piqa_nonparallel/predictions.json",
    "Reading": "reading/predictions.json",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def latest_file(root: Path, name: str) -> Path | None:
    cands = sorted(root.rglob(name), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    return cands[0] if cands else None


def find_zero_prediction(column: str, pristine_ewok_path: Path | None) -> Path:
    if column == "EWoK" and pristine_ewok_path is not None:
        return pristine_ewok_path
    task_root = FULL / "official_outputs" / TARGET / column
    p = latest_file(task_root, "predictions.json")
    if not p:
        raise FileNotFoundError(f"Missing {column} predictions under {task_root}")
    return p


def find_aoa(aoa_dir: Path | None, which: str) -> Path:
    if aoa_dir is None:
        aoa_dir = FULL / "official_aoa_min0_seed43122"
    p = latest_file(aoa_dir, which)
    if not p:
        raise FileNotFoundError(f"Missing {which} under {aoa_dir}")
    return p


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


def run(cmd: list[str], cwd: Path, env: dict[str, str]) -> dict[str, Any]:
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return {"cmd": cmd, "cwd": str(cwd), "returncode": p.returncode, "stdout_tail": p.stdout[-4000:], "stderr_tail": p.stderr[-8000:], "elapsed_sec": round(time.time() - t0, 3)}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_collated(path: Path) -> dict[str, Any]:
    c = load_json(path)
    out: dict[str, Any] = {"top_keys": sorted(c.keys()), "null_keys": sorted(k for k, v in c.items() if v is None)}
    for key in ["blimp", "blimp_supplement", "ewok", "entity_tracking_filtered", "comps", "reading", "glue"]:
        v = c.get(key)
        if isinstance(v, dict):
            if key in {"reading", "glue"}:
                out[f"{key}_lengths"] = {t: len(tv.get("predictions", [])) for t, tv in v.items()}
            else:
                out[f"{key}_lengths"] = {t: len(tv.get("predictions", [])) for t, tv in v.items()}
    for key in ["global_piqa_parallel", "global_piqa_nonparallel"]:
        v = c.get(key)
        if isinstance(v, dict):
            out[f"{key}_example_count"] = len(v)
            out[f"{key}_prediction_total"] = sum(len(x.get("predictions", [])) for x in v.values())
    aoa_s = c.get("aoa_surprisals")
    if isinstance(aoa_s, dict):
        rows = aoa_s.get("results", [])
        cnt = collections.Counter(r.get("step") for r in rows)
        out["aoa_surprisal_num_rows"] = len(rows)
        out["aoa_surprisal_num_steps"] = len(cnt)
        out["aoa_surprisal_row_count_values"] = sorted(set(cnt.values()))
    return out


def parse_results_txt(path: Path, metric: str) -> float:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        k, _, v = line.partition(":")
        if k.strip() == metric:
            return float(v.strip()) * 100.0
    raise ValueError(f"metric {metric} not found in {path}")


def score_collated(collated_path: Path, staged_superglue_results: dict[str, Path]) -> dict[str, Any]:
    sys.path.insert(0, str((PRISTINE_STRICT / "evaluation_pipeline").resolve()))
    import calculate_results_from_pred as calc  # type: ignore
    import collate_preds as coll  # type: ignore

    c = load_json(collated_path)
    full = PRISTINE_STRICT / "evaluation_data/full_eval"
    scores: dict[str, float] = {}
    details: dict[str, Any] = {}
    scores["BLiMP"] = float(calc._calculate_blimp_results(c["blimp"], full / "blimp_filtered"))
    scores["Supplement"] = float(calc._calculate_blimp_results(c["blimp_supplement"], full / "supplement_filtered"))
    scores["EWoK"] = float(calc._calculate_ewok_results(c["ewok"], full / "ewok_filtered"))
    # Entity score using filtered entity targets, same as research seed43022.
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
    et_scores = []
    et_counts = {}
    for subtask, block in c["entity_tracking_filtered"].items():
        preds = block["predictions"]
        targets = subtask_to_targets[subtask]
        if len(preds) != len(targets):
            raise RuntimeError(f"Entity length mismatch {subtask}: {len(preds)} vs {len(targets)}")
        correct = sum(1 for pred, ex in zip(preds, targets) if pred["pred"].strip() == ex["options"][0].strip())
        et_scores.append(correct / len(targets))
        et_counts[subtask] = {"correct": correct, "total": len(targets)}
    scores["Entity"] = 100.0 * sum(et_scores) / len(et_scores)
    details["Entity"] = et_counts
    scores["COMPS"] = float(calc._calculate_comps_results(c["comps"], full / "comps"))
    gp = []
    for key, task_name in [("global_piqa_parallel", "global_piqa_parallel"), ("global_piqa_nonparallel", "global_piqa_nonparallel")]:
        f = LOCAL_STRICT_WITH_GENERATED_DATA / "evaluation_data/full_eval" / task_name / "eng_latn.jsonl"
        frac = coll._calculate_global_piqa_results(c[key], f, task_name)[task_name]
        scores[task_name] = 100.0 * float(frac)
        gp.append(scores[task_name])
    scores["GlobalPIQA"] = sum(gp) / 2.0
    reading_frac = coll._calculate_reading_results(c["reading"], full / "reading/reading_data.csv")
    scores["Reading"] = 100.0 * (float(reading_frac["spr"]) + float(reading_frac["rt"])) / 2.0
    sg_scores = []
    sg_detail = {}
    for task in SUPERGLUE:
        metric = FINETUNE_METRIC[task]
        p = staged_superglue_results[task]
        val = parse_results_txt(p, metric)
        sg_scores.append(val)
        sg_detail[task] = {"metric": metric, "score": val, "results_txt": str(p)}
    scores["SuperGLUE"] = sum(sg_scores) / len(sg_scores)
    details["SuperGLUE"] = sg_detail
    aoa_raw = float(c["aoa"].get("aoa", 0.0))
    scores["AoA"] = 100.0 * aoa_raw
    overall = sum(scores[k] for k in OVERALL_KEYS) / len(OVERALL_KEYS)
    return {
        "scores": {k: scores[k] for k in OVERALL_KEYS},
        "GlobalPIQA_parallel": scores["global_piqa_parallel"],
        "GlobalPIQA_nonparallel": scores["global_piqa_nonparallel"],
        "Overall": overall,
        "NLP_average": sum(scores[k] for k in NLP_KEYS) / len(NLP_KEYS),
        "Human_like_average": sum(scores[k] for k in HUMAN_KEYS) / len(HUMAN_KEYS),
        "margin_over_visible_leader_41p8": overall - 41.8,
        "details": details,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pristine-ewok-predictions", type=Path, default=None, help="Optional seed43122 EWoK predictions against research pristine 7618-row EWoK. Required if full-run EWoK used stale INITIAL_MODEL_STUDIES data.")
    ap.add_argument("--aoa-dir", type=Path, default=None, help="Directory containing official min_context=0 seed43122 surprisal.json and aoa_score.json")
    ap.add_argument("--copy", action="store_true")
    args = ap.parse_args()

    # The runtime may mount OUT_DIR itself as the exact writable target.  Do not
    # delete the mount root; clean its contents instead so reruns remain fresh.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for child in list(OUT_DIR.iterdir()):
        if child.is_symlink() or child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)
    results_dir = OUT_DIR / "results"
    model_stem = MODEL_ROOT.stem
    zero_root = results_dir / model_stem / "main/zero_shot/mlm"
    fine_root = results_dir / model_stem / "main/finetune"

    staged = []
    columns = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"]
    for col in columns:
        src = find_zero_prediction(col, args.pristine_ewok_predictions)
        staged.append(copy_or_link(src, zero_root / ZERO_REL_BY_COLUMN[col], symlink=not args.copy))
    staged.append(copy_or_link(find_aoa(args.aoa_dir, "surprisal.json"), zero_root / "AoA_word/surprisal.json", symlink=not args.copy))
    staged.append(copy_or_link(find_aoa(args.aoa_dir, "aoa_score.json"), zero_root / "AoA_word/aoa_score.json", symlink=not args.copy))

    sg_results = {}
    for task in SUPERGLUE:
        root = FULL / "superglue_results" / TARGET / task / ENDPOINT / "main/finetune" / task
        pred = root / "predictions.json"
        res = root / "results.txt"
        if not pred.exists():
            alt = latest_file(FULL / "superglue_results" / TARGET / task, "predictions.json")
            if not alt:
                raise FileNotFoundError(pred)
            pred = alt
            res = alt.parent / "results.txt"
        if not res.exists():
            raise FileNotFoundError(res)
        sg_results[task] = res
        staged.append(copy_or_link(pred, fine_root / task / "predictions.json", symlink=not args.copy))

    env = os.environ.copy()
    for rel, key in [("hf_home", "HF_HOME"), ("hf_home/hub", "HF_HUB_CACHE"), ("datasets", "HF_DATASETS_CACHE"), ("transformers", "TRANSFORMERS_CACHE"), ("modules", "HF_MODULES_CACHE"), ("tmp", "TMPDIR")]:
        p = OUT_DIR / rel
        p.mkdir(parents=True, exist_ok=True)
        env[key] = str(p.resolve())
    env["PYTHONPATH"] = str(PRISTINE_STRICT.resolve()) + os.pathsep + str((PRISTINE_STRICT / "evaluation_pipeline").resolve()) + os.pathsep + env.get("PYTHONPATH", "")
    env["TOKENIZERS_PARALLELISM"] = "false"

    collate_cmd = [
        sys.executable, "evaluation_pipeline/collate_preds.py",
        "--model_path_or_name", str(MODEL_ROOT.resolve()),
        "--backend", "mlm",
        "--results_dir", str(results_dir.resolve()),
        "--revision_name", "main",
        "--track", "strict-small",
    ]
    collate_run = run(collate_cmd, PRISTINE_STRICT, env)
    collated = results_dir / model_stem / "all_full_preds_and_fast_scores_mlm.json"
    summary = summarize_collated(collated) if collated.exists() else {}
    score = score_collated(collated, sg_results) if collated.exists() and not summary.get("null_keys") else {}
    out = {
        "status": "PRISTINE_COLLATE_SEED43122",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Apply the same official-coordinate collation/scoring convention as seed43022 to the independent seed43122 full evaluation.",
        "requires_pristine_ewok_if_old_full_eval_used_stale_local_ewok": args.pristine_ewok_predictions is None,
        "paths": {"out_dir": str(OUT_DIR), "results_dir": str(results_dir), "collated_path": str(collated), "full_eval_root": str(FULL), "model_root": str(MODEL_ROOT)},
        "staged_files": staged,
        "collate_run": collate_run,
        "collated_summary": summary,
        "score_summary": score,
        "coordinate_notes": {
            "pristine_strict": str(PRISTINE_STRICT),
            "globalpiqa_lineage": "See experiments/archive/representation_and_objectives/data/globalpiqa_official_lineage/globalpiqa_official_lineage.json; inherited generated files are byte-identical to current official dl.py output.",
            "superglue_primary_metric": "f1 for MRPC/QQP, accuracy for boolq/mnli/multirc/rte/wsc, matching research seed43022 coordinate.",
            "aoa_expected": "official min_context=0, 8005 rows/checkpoint over 19 checkpoints",
        },
    }
    out_json = OUT_DIR / "pristine_collate_seed43122_summary.json"
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": out["status"],
        "collate_returncode": collate_run.get("returncode"),
        "null_keys": summary.get("null_keys"),
        "ewok_total": sum(summary.get("ewok_lengths", {}).values()) if isinstance(summary.get("ewok_lengths"), dict) else None,
        "aoa_row_count_values": summary.get("aoa_surprisal_row_count_values"),
        "overall": score.get("Overall"),
        "margin_over_visible_leader_41p8": score.get("margin_over_visible_leader_41p8"),
        "out_json": str(out_json),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
