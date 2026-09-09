#!/usr/bin/env python3
"""research: official-coordinate zero-shot columns for raw reinvest seed43022 90M.

This existing-checkpoint evaluation tests whether the 90M fast-screen gain for
compact_view_reinvest seed43022 survives the full/pristine zero-shot coordinate
before spending SuperGLUE finetuning or AoA compute.  It evaluates only:
  - full BLiMP on pristine blimp_filtered
  - full BLiMP Supplement on pristine supplement_filtered
  - current official 7,618-row EWoK on pristine ewok_filtered

Entity_full, COMPS, GlobalPIQA, and Reading are already available from the research
broad screen for this same checkpoint.  No training or corpus changes occur.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Dict, Optional


def discover_root() -> Path:
    root = Path.cwd()
    if (root / "experiments").exists():
        return root
    here = _public_path('experiments/archive/frontier_consolidation/training/scripts/official_zeroshot_r43022_90m.py')
    for parent in [here] + list(here.parents):
        if (parent / "experiments").exists():
            return parent
    raise RuntimeError("Could not find user root containing Sessions/")


ROOT = discover_root()
STUDY = ROOT / "experiments/archive/frontier_consolidation"
STRICT_LOCAL = ROOT / "experiments/archive/initial_model_studies/repos/babylm-eval/strict"
STRICT_PRISTINE = ROOT / "experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict"
MODEL_90M = ROOT / "experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/hf_model/chck_90M"
BROAD_90M = STUDY / "data/broad_noaoa_late_checkpoint_merge/broad_noaoa_late_checkpoint_merge.json"
SEED43022_OFFICIAL_100M = ROOT / "experiments/archive/representation_and_objectives/data/pristine_collate_seed43022/pristine_collate_seed43022_summary.json"

TASKS = {
    "BLiMP": {"task": "blimp", "data_path": STRICT_PRISTINE / "evaluation_data/full_eval/blimp_filtered", "batch": 128},
    "Supplement": {"task": "blimp", "data_path": STRICT_PRISTINE / "evaluation_data/full_eval/supplement_filtered", "batch": 128},
    "EWoK": {"task": "ewok", "data_path": STRICT_PRISTINE / "evaluation_data/full_eval/ewok_filtered", "batch": 64},
}
KNOWN_RAW90_FAST = {
    "Entity_full": 27.62,
    "COMPS": 52.05,
    "GlobalPIQA": 35.12,
    "Reading": 8.17,
}
VISIBLE_LEADER = 41.8


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--columns", nargs="+", default=["BLiMP", "Supplement", "EWoK"], choices=sorted(TASKS))
    ap.add_argument("--out-root", type=Path, default=STUDY / "data/official_zeroshot_r43022_90m")
    ap.add_argument("--batch-size-override", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def parse_sentence_score(text: str) -> Optional[float]:
    for pat in [
        r"### AVERAGE [A-Z_ '\\-]*\s*\n\s*([+-]?[0-9]+(?:\.[0-9]+)?)",
        r"AVERAGE ACCURACY\s*\n\s*([+-]?[0-9]+(?:\.[0-9]+)?)",
    ]:
        m = re.search(pat, text)
        if m:
            val = float(m.group(1))
            if math.isfinite(val) and -1000 < val < 1000:
                return val
    vals = re.findall(r"(?:accuracy|score|acc|acc_norm)[^0-9+\-]*([+-]?[0-9]+(?:\.[0-9]+)?)", text, flags=re.I)
    if vals:
        val = float(vals[-1])
        return val * (100 if 0 <= val <= 1 else 1)
    return None


def read_report_score(task_out: Path) -> tuple[Optional[float], Optional[str]]:
    candidates = sorted(task_out.rglob("best_temperature_report.txt"), key=lambda p: (p.stat().st_mtime, str(p)))
    if not candidates:
        candidates = sorted(task_out.rglob("*.txt"), key=lambda p: (p.stat().st_mtime, str(p)))
    for p in reversed(candidates):
        score = parse_sentence_score(p.read_text(encoding="utf-8", errors="replace"))
        if score is not None:
            return score, str(p)
    return None, None


def count_jsonl_rows(path: Path) -> int:
    if path.is_dir():
        return sum(count_jsonl_rows(p) for p in path.glob("*.jsonl"))
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return sum(1 for _ in f)


def count_prediction_items(pred_path: Optional[Path]) -> Optional[int]:
    if pred_path is None or not pred_path.exists():
        return None
    try:
        payload = json.loads(pred_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    total = 0
    for v in payload.values():
        if isinstance(v, dict) and isinstance(v.get("predictions"), list):
            total += len(v["predictions"])
    return total


def setup_env(out_root: Path, gpu: str) -> Dict[str, str]:
    env = os.environ.copy()
    hf = out_root / "hf_cache"
    tmp = out_root / "tmp"
    env["HF_HOME"] = str((hf / "home").resolve())
    env["HF_HUB_CACHE"] = str((hf / "hub").resolve())
    env["HF_DATASETS_CACHE"] = str((hf / "datasets").resolve())
    env["TRANSFORMERS_CACHE"] = str((hf / "transformers").resolve())
    env["HF_MODULES_CACHE"] = str((hf / "modules").resolve())
    env["TMPDIR"] = str(tmp.resolve())
    env["NLTK_DATA"] = str((ROOT / "experiments/archive/initial_model_studies/data/nltk_data").resolve())
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    for k in ["HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE", "HF_MODULES_CACHE", "TMPDIR"]:
        Path(env[k]).mkdir(parents=True, exist_ok=True)
    return env


def run_column(col: str, out_root: Path, env: Dict[str, str], batch_override: Optional[int], force: bool, dry_run: bool) -> Dict[str, Any]:
    spec = TASKS[col]
    data_path = Path(spec["data_path"])
    out_dir = out_root / "official_outputs" / "r43022_90M" / col
    log = out_root / "logs" / f"r43022_90M_{col}.log"
    out_dir.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    batch = int(batch_override or spec["batch"])
    revision = f"official_r43022_90M_{col}"
    cmd = [
        sys.executable, "-m", "evaluation_pipeline.sentence_zero_shot.run",
        "--model_path_or_name", str(MODEL_90M.resolve()),
        "--backend", "mlm",
        "--task", str(spec["task"]),
        "--data_path", str(data_path.resolve()),
        "--revision_name", revision,
        "--save_predictions",
        "--batch_size", str(batch),
        "--non_causal_batch_size", str(batch),
        "--output_dir", str(out_dir.resolve()),
    ]
    rec: Dict[str, Any] = {
        "column": col,
        "task": spec["task"],
        "data_path": str(data_path),
        "data_rows": count_jsonl_rows(data_path),
        "batch_size": batch,
        "output_dir": str(out_dir),
        "log": str(log),
        "command": cmd,
        "dry_run": dry_run,
    }
    if dry_run:
        rec.update({"returncode": None, "score": None, "prediction_item_total": None})
        return rec
    # Resume-safe: if a previous successful score exists and force is false, reuse it.
    if not force:
        score, report = read_report_score(out_dir)
        preds = sorted(out_dir.rglob("predictions.json"), key=lambda p: (p.stat().st_mtime, str(p)))
        if score is not None and preds:
            rec.update({"returncode": 0, "score": score, "stdout_score": None, "report_score": score, "report_path": report,
                        "predictions_path": str(preds[-1]), "prediction_item_total": count_prediction_items(preds[-1]), "reused_existing": True})
            return rec
    t0 = time.time()
    with log.open("a", encoding="utf-8") as f:
        f.write(f"\n[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] $ {' '.join(cmd)}\n")
    proc = subprocess.run(cmd, cwd=str(STRICT_PRISTINE.resolve()), env=env, capture_output=True, text=True, timeout=3600)
    with log.open("a", encoding="utf-8") as f:
        f.write(proc.stdout)
        f.write("\n--- STDERR ---\n")
        f.write(proc.stderr)
        f.write(f"\n[returncode={proc.returncode} elapsed_sec={time.time()-t0:.1f}]\n")
    stdout_score = parse_sentence_score(proc.stdout)
    report_score, report_path = read_report_score(out_dir)
    score = stdout_score if stdout_score is not None else report_score
    preds = sorted(out_dir.rglob("predictions.json"), key=lambda p: (p.stat().st_mtime, str(p)))
    pred_path = preds[-1] if preds else None
    rec.update({
        "returncode": proc.returncode,
        "score": score,
        "stdout_score": stdout_score,
        "report_score": report_score,
        "report_path": report_path,
        "predictions_path": str(pred_path) if pred_path else None,
        "prediction_item_total": count_prediction_items(pred_path),
        "elapsed_sec": round(time.time() - t0, 3),
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    })
    return rec


def read_official_100m_scores() -> Dict[str, Any]:
    if not SEED43022_OFFICIAL_100M.exists():
        return {}
    d = json.loads(SEED43022_OFFICIAL_100M.read_text(encoding="utf-8"))
    return d.get("score_summary", {}).get("official_overall", {}).get("scores", {})


def read_raw90_fast_table() -> Dict[str, Any]:
    if not BROAD_90M.exists():
        return {}
    d = json.loads(BROAD_90M.read_text(encoding="utf-8"))
    return d.get("table", {}).get("r43022_90M", {})


def write_note(payload: Dict[str, Any], out_md: Path) -> None:
    lines = [
        "# research — raw seed43022 90M official-coordinate zero-shot probe\n\n",
        "Existing checkpoint only. Full BLiMP, full Supplement, and pristine 7,618-row EWoK were evaluated before any SuperGLUE or AoA work.\n\n",
        f"Summary JSON: `{payload['out_json']}`\n\n",
        "## Measured official-coordinate zero-shot columns\n",
        "| column | score | data rows | prediction items | returncode |\n",
        "|---|---:|---:|---:|---:|\n",
    ]
    for col in payload["columns"]:
        r = payload["records"].get(col, {})
        score = r.get("score")
        lines.append(f"| {col} | {score if score is not None else ''} | {r.get('data_rows')} | {r.get('prediction_item_total')} | {r.get('returncode')} |\n")
    proj = payload.get("projection", {})
    if proj:
        lines += [
            "\n## Projection using measured zero-shot plus known raw90 no-AoA columns\n",
            f"- Seven measured/known columns excluding SuperGLUE and AoA sum: {proj.get('seven_column_sum_excluding_superglue_aoa'):.6f}.\n",
            f"- Required SuperGLUE+AoA for 41.8: {proj.get('required_superglue_plus_aoa_for_41p8'):.6f}.\n",
            f"- Required SuperGLUE+AoA for 42.0: {proj.get('required_superglue_plus_aoa_for_42p0'):.6f}.\n",
            f"- Required SuperGLUE+AoA to exceed seed43022 100M official 42.033135: {proj.get('required_superglue_plus_aoa_to_exceed_100m_official'):.6f}.\n",
            f"- If SuperGLUE matches the seed43022 100M official value and AoA remains 0.0, projected Overall is {proj.get('overall_if_superglue_matches_100m_and_aoa0'):.6f}.\n",
        ]
    lines.append("\n## Scientific read\n")
    for msg in payload.get("scientific_read", {}).values():
        lines.append(f"- {msg}\n")
    out_md.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    t0 = time.time()
    args.out_root.mkdir(parents=True, exist_ok=True)
    for p in [STRICT_LOCAL, STRICT_PRISTINE, MODEL_90M, MODEL_90M / "model.safetensors"]:
        if not p.exists():
            raise FileNotFoundError(p)
    env = setup_env(args.out_root, args.gpu)
    records: Dict[str, Any] = {}
    for col in args.columns:
        print(json.dumps({"event": "eval_start", "column": col, "model": str(MODEL_90M), "gpu": args.gpu}), flush=True)
        rec = run_column(col, args.out_root, env, args.batch_size_override, args.force, args.dry_run)
        records[col] = rec
        print(json.dumps({"event": "eval_done", "column": col, "score": rec.get("score"), "rc": rec.get("returncode"), "prediction_items": rec.get("prediction_item_total")}), flush=True)
        partial = {"status": "OFFICIAL_ZEROSHOT_R43022_90M_PARTIAL", "records": records}
        (args.out_root / "partial_official_zeroshot_r43022_90m.json").write_text(json.dumps(partial, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    failures = []
    for col, rec in records.items():
        if rec.get("returncode") not in [0, None] or rec.get("score") is None:
            failures.append({"column": col, "returncode": rec.get("returncode"), "score": rec.get("score"), "log": rec.get("log")})
    official_100m = read_official_100m_scores()
    raw90_fast = read_raw90_fast_table()
    measured = {col: records[col].get("score") for col in ["BLiMP", "Supplement", "EWoK"] if col in records}
    projection = {}
    if not failures and all(k in measured and measured[k] is not None for k in ["BLiMP", "Supplement", "EWoK"]):
        seven = float(measured["BLiMP"]) + float(measured["Supplement"]) + float(measured["EWoK"]) + KNOWN_RAW90_FAST["Entity_full"] + KNOWN_RAW90_FAST["COMPS"] + KNOWN_RAW90_FAST["GlobalPIQA"] + KNOWN_RAW90_FAST["Reading"]
        req_418 = VISIBLE_LEADER * 9.0 - seven
        req_420 = 42.0 * 9.0 - seven
        req_100m = 42.0331347900748 * 9.0 - seven
        sg100 = float(official_100m.get("SuperGLUE", 71.03604952825312))
        projection = {
            "measured_columns": measured,
            "known_raw90_columns_from_step032_broad_screen": KNOWN_RAW90_FAST,
            "raw90_fast_table": raw90_fast,
            "seven_column_sum_excluding_superglue_aoa": seven,
            "required_superglue_plus_aoa_for_41p8": req_418,
            "required_superglue_plus_aoa_for_42p0": req_420,
            "required_superglue_plus_aoa_to_exceed_100m_official": req_100m,
            "seed43022_100m_official_superglue_reference": sg100,
            "overall_if_superglue_matches_100m_and_aoa0": (seven + sg100 + 0.0) / 9.0,
            "delta_vs_100m_official_if_superglue_matches_and_aoa0": (seven + sg100) / 9.0 - 42.0331347900748,
        }
    sci: Dict[str, str] = {}
    if failures:
        sci["failure"] = f"Some zero-shot columns failed or lacked scores: {failures}. Do not use this as a route result until repaired."
    elif projection:
        sci["sota_potential"] = (
            f"With full BLiMP/Supplement/EWoK measured and research raw90 Entity/COMPS/GlobalPIQA/Reading reused, raw90 needs "
            f"SuperGLUE+AoA {projection['required_superglue_plus_aoa_for_41p8']:.3f} to clear 41.8 and "
            f"{projection['required_superglue_plus_aoa_to_exceed_100m_official']:.3f} to exceed the frozen 100M official endpoint."
        )
        sci["superglue_scenario"] = (
            f"If SuperGLUE equals the frozen 100M seed43022 value {projection['seed43022_100m_official_superglue_reference']:.3f} and AoA remains 0.0, "
            f"raw90 projects to Overall {projection['overall_if_superglue_matches_100m_and_aoa0']:.4f}."
        )
        if projection["required_superglue_plus_aoa_to_exceed_100m_official"] < projection["seed43022_100m_official_superglue_reference"]:
            sci["next"] = "The raw90 checkpoint remains plausible enough for a targeted SuperGLUE/AoA official-coordinate follow-up; it should not trigger new training."
        else:
            sci["next"] = "The raw90 checkpoint does not improve enough on official zero-shot columns to justify SuperGLUE/AoA escalation unless another measured column changes."
    payload = {
        "status": "OFFICIAL_ZEROSHOT_R43022_90M_DRY_RUN" if args.dry_run else "OFFICIAL_ZEROSHOT_R43022_90M_COMPLETE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Cheapest official-coordinate zero-shot test for raw compact_view_reinvest seed43022 90M before SuperGLUE/AoA work.",
        "model_path": str(MODEL_90M),
        "strict_pristine": str(STRICT_PRISTINE),
        "strict_local": str(STRICT_LOCAL),
        "gpu": args.gpu,
        "columns": args.columns,
        "records": records,
        "failures": failures,
        "official_100m_reference_scores": official_100m,
        "projection": projection,
        "scientific_read": sci,
        "elapsed_sec": round(time.time() - t0, 3),
    }
    out_json = args.out_root / "official_zeroshot_r43022_90m_summary.json"
    payload["out_json"] = str(out_json)
    payload["out_md"] = str(args.out_root / "official_zeroshot_r43022_90m.md")
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_note(payload, Path(payload["out_md"]))
    print(json.dumps({
        "status": payload["status"],
        "failures": failures,
        "projection": projection,
        "out_json": payload["out_json"],
        "out_md": payload["out_md"],
        "elapsed_sec": payload["elapsed_sec"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
