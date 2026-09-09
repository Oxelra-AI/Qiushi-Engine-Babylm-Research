#!/usr/bin/env python3
"""research: broad no-AoA evaluation of selected existing reinvest checkpoints.

Purpose: test the existing-checkpoint stopping hypothesis before any new training,
corpus generation, SuperGLUE, AoA, or full official evaluation.  The selected
checkpoints come from the already trained compact_view_reinvest ladders:
  - seed43022: 80M and 90M (100M is the frozen official endpoint reference)
  - seed43122: 45M and 80M (100M is the weak replication reference)

This is a fast official-compatible research surface matching the research/25
no-AoA screen: BLiMP, BLiMP Supplement, fast EWoK, Entity fast/full, COMPS,
GlobalPIQA parallel/nonparallel, and Reading.  It is not a leaderboard collation.
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
from typing import Any, Dict, Iterable, List, Optional


def discover_root() -> Path:
    root = Path.cwd()
    if (root / "experiments").exists():
        return root
    here = _public_path('experiments/archive/frontier_consolidation/training/scripts/broad_noaoa_late_checkpoints.py')
    for parent in [here] + list(here.parents):
        if (parent / "experiments").exists():
            return parent
    raise RuntimeError("Could not discover user root containing Sessions/")


ROOT = discover_root()
STUDY = ROOT / "experiments/archive" / 'frontier_consolidation'
STRICT = ROOT / "experiments/archive" / 'initial_model_studies' / "repos" / "babylm-eval" / "strict"
PYTHON_EXE = sys.executable


TARGETS: Dict[str, Dict[str, Any]] = {
    "r43022_80M": {
        "family": "compact_view_reinvest",
        "seed": "43022",
        "exposure_m": 80,
        "model_path": ROOT / "experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/hf_model/chck_80M",
        "reference_role": "80M common stopping candidate; research best across-seed Supplement minimum/mean.",
    },
    "r43022_90M": {
        "family": "compact_view_reinvest",
        "seed": "43022",
        "exposure_m": 90,
        "model_path": ROOT / "experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/hf_model/chck_90M",
        "reference_role": "90M seed43022 fast-Supplement maximum; tests whether it preserves broad balance.",
    },
    "r43122_45M": {
        "family": "compact_view_reinvest",
        "seed": "43122",
        "exposure_m": 45,
        "model_path": ROOT / "experiments/archive/representation_and_objectives/training/runs/repl_compact_view_reinvest_seed43122/hf_model/chck_45M",
        "reference_role": "45M seed43122 fast-Supplement maximum; tests whether per-seed early stopping is broad or narrow.",
    },
    "r43122_80M": {
        "family": "compact_view_reinvest",
        "seed": "43122",
        "exposure_m": 80,
        "model_path": ROOT / "experiments/archive/representation_and_objectives/training/runs/repl_compact_view_reinvest_seed43122/hf_model/chck_80M",
        "reference_role": "80M common stopping candidate; research best across-seed Supplement minimum/mean.",
    },
}

TASKS = [
    ("BLiMP", "blimp", "evaluation_data/fast_eval/blimp_fast", 128),
    ("Supplement", "blimp", "evaluation_data/fast_eval/supplement_fast", 128),
    ("EWoK", "ewok", "evaluation_data/fast_eval/evaluation_data/fast_eval/ewok_fast", 64),
    ("Entity", "entity_tracking", "evaluation_data/fast_eval/entity_tracking_fast", 128),
    ("Entity_full", "entity_tracking", "evaluation_data/full_eval/entity_tracking", 128),
    ("COMPS", "comps", "evaluation_data/full_eval/comps", 128),
    ("GlobalPIQA_parallel", "global_piqa_parallel", "evaluation_data/fast_eval/global_piqa_parallel", 128),
    ("GlobalPIQA_nonparallel", "global_piqa_nonparallel", "evaluation_data/fast_eval/global_piqa_nonparallel", 128),
]
TASK_BY_COL = {c: (t, d, b) for c, t, d, b in TASKS}
DEFAULT_COLUMNS = [c for c, _, _, _ in TASKS] + ["Reading"]
TABLE_KEYS = [
    "BLiMP", "Supplement", "EWoK", "Entity", "Entity_full", "COMPS",
    "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "GlobalPIQA_mean",
    "Reading", "Reading_eye", "Reading_self_paced", "equal7_mean", "equal7_full_entity",
]

# Known 100M fast/no-AoA references, not reevaluated here.  They are included
# only for interpretation of selected checkpoints.
KNOWN_FAST_REFERENCES = {
    "r43022_100M_fast": {
        "family": "compact_view_reinvest", "seed": "43022", "exposure_m": 100,
        "source": "experiments/archive/frontier_consolidation/data/density_noaoa_eval_reinvest/per_target/compact_view_reinvest.json",
        "BLiMP": 66.63, "Supplement": 66.4, "EWoK": 53.09, "Entity": 28.07,
        "Entity_full": 27.75, "COMPS": 51.97, "GlobalPIQA_parallel": 25.24,
        "GlobalPIQA_nonparallel": 46.0, "GlobalPIQA_mean": 35.62,
        "Reading": 8.24, "Reading_eye": 11.15, "Reading_self_paced": 5.33,
    },
    "r43122_100M_fast": {
        "family": "compact_view_reinvest", "seed": "43122", "exposure_m": 100,
        "source": "experiments/archive/representation_and_objectives/data/compact_reinvest_seed43122_fast/compact_view_reinvest_seed43122_fast_summary.json",
        "BLiMP": 65.75, "Supplement": 63.2, "EWoK": 49.36, "Entity": 26.68,
        "Entity_full": 26.29, "COMPS": 51.54, "GlobalPIQA_parallel": 24.27,
        "GlobalPIQA_nonparallel": 46.0, "GlobalPIQA_mean": 35.135,
        "Reading": 8.865, "Reading_eye": 12.87, "Reading_self_paced": 4.86,
    },
}
REFERENCE_COLUMNS = {
    "visible_leader_card": {
        "BLiMP": 67.20, "Supplement": 56.01, "EWoK": 56.07, "Entity": 28.45,
        "COMPS": 53.57, "GlobalPIQA_mean": 39.67, "SuperGLUE": 69.79,
        "Reading": 5.42, "AoA": 0.0, "Overall": 41.80,
    },
    "official_seed43022_100M": {
        "source": "experiments/archive/representation_and_objectives/data/pristine_collate_seed43022/pristine_collate_seed43022_summary.json",
        "Overall": 42.0331347900748,
        "BLiMP": 66.8723, "Supplement": 63.2758, "EWoK": 53.5366,
        "Entity": 27.7457, "COMPS": 51.9688, "GlobalPIQA_mean": 35.6214,
        "SuperGLUE": 71.0360, "Reading": 8.2416, "AoA": 0.0,
    },
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", choices=sorted(TARGETS), default=sorted(TARGETS))
    ap.add_argument("--columns", nargs="+", default=DEFAULT_COLUMNS)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--work-tag", default="broad_noaoa_late_checkpoints")
    ap.add_argument("--out-root", type=Path, default=STUDY / "data/broad_noaoa_late_checkpoints")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def setup_env(out_root: Path, work_tag: str, gpu: str) -> Dict[str, str]:
    env = os.environ.copy()
    hf = out_root / "hf_cache" / work_tag
    env["HF_HOME"] = str((hf / "home").resolve())
    env["HF_HUB_CACHE"] = str((hf / "hub").resolve())
    env["HF_DATASETS_CACHE"] = str((hf / "datasets").resolve())
    env["TRANSFORMERS_CACHE"] = str((hf / "transformers").resolve())
    env["HF_MODULES_CACHE"] = str((hf / "modules").resolve())
    env["NLTK_DATA"] = str((ROOT / "experiments/archive/initial_model_studies/data/nltk_data").resolve())
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    tmp = out_root / "tmp" / work_tag
    env["TMPDIR"] = str(tmp.resolve())
    for k in ["HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE", "HF_MODULES_CACHE", "TMPDIR"]:
        Path(env[k]).mkdir(parents=True, exist_ok=True)
    return env


def parse_sentence_score(text: str) -> Optional[float]:
    patterns = [
        r"### AVERAGE [A-Z_ '\\-]*\s*\n\s*([+-]?[0-9]+(?:\.[0-9]+)?)",
        r"AVERAGE ACCURACY\s*\n\s*([+-]?[0-9]+(?:\.[0-9]+)?)",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            val = float(m.group(1))
            if math.isfinite(val) and -1000 < val < 1000:
                return val
    for line in reversed(text.splitlines()):
        s = line.strip()
        m = re.match(r"^(?:[0-9.]+\s+)?([+-]?[0-9]+(?:\.[0-9]+)?)$", s)
        if m:
            val = float(m.group(1))
            if math.isfinite(val) and -1000 < val < 1000:
                return val
    vals = re.findall(r"(?:accuracy|score|acc|acc_norm)[^0-9+\-]*([+-]?[0-9]+(?:\.[0-9]+)?)", text, flags=re.I)
    if vals:
        val = float(vals[-1])
        return val * (100 if 0 <= val <= 1 else 1)
    return None


def parse_reading_scores(text: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for label, key in [("EYE TRACKING SCORE", "Reading_eye"), ("SELF-PACED READING SCORE", "Reading_self_paced")]:
        m = re.search(re.escape(label) + r":\s*([+-]?[0-9]+(?:\.[0-9]+)?)", text)
        if m:
            out[key] = float(m.group(1))
    if "Reading_eye" in out and "Reading_self_paced" in out:
        out["Reading"] = (out["Reading_eye"] + out["Reading_self_paced"]) / 2.0
    return out


def read_report_score(task_out: Path) -> Optional[float]:
    candidates = sorted(task_out.rglob("best_temperature_report.txt"), key=lambda p: (p.stat().st_mtime, str(p)))
    if not candidates:
        candidates = sorted(task_out.rglob("*.txt"), key=lambda p: (p.stat().st_mtime, str(p)))
    for p in reversed(candidates):
        score = parse_sentence_score(p.read_text(encoding="utf-8", errors="replace"))
        if score is not None:
            return score
    return None


def read_reading_report(task_out: Path) -> Dict[str, float]:
    candidates = sorted(task_out.rglob("report.txt"), key=lambda p: (p.stat().st_mtime, str(p)))
    for p in reversed(candidates):
        scores = parse_reading_scores(p.read_text(encoding="utf-8", errors="replace"))
        if scores:
            return scores
    return {}


def run_cmd(cmd: List[str], env: Dict[str, str], log_path: Path, timeout: int = 3600) -> subprocess.CompletedProcess[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] $ " + " ".join(cmd) + "\n")
    p = subprocess.run(cmd, cwd=str(STRICT.resolve()), env=env, capture_output=True, text=True, timeout=timeout)
    elapsed = time.time() - t0
    with log_path.open("a", encoding="utf-8") as f:
        f.write(p.stdout)
        f.write("\n--- STDERR ---\n")
        f.write(p.stderr)
        f.write(f"\n[returncode={p.returncode} elapsed_sec={elapsed:.1f}]\n")
    return p


def eval_sentence_column(target: str, model_path: Path, column: str, env: Dict[str, str], out_root: Path, work_tag: str) -> Dict[str, Any]:
    task, data_path, batch = TASK_BY_COL[column]
    task_out = out_root / "eval_outputs" / work_tag / target / column
    task_out.mkdir(parents=True, exist_ok=True)
    log = out_root / "logs" / work_tag / f"{target}_{column}.log"
    revision = f"step032_{target}_{column}"
    cmd = [
        PYTHON_EXE, "-m", "evaluation_pipeline.sentence_zero_shot.run",
        "--model_path_or_name", str(model_path.resolve()),
        "--backend", "mlm",
        "--task", task,
        "--data_path", data_path,
        "--revision_name", revision,
        "--save_predictions",
        "--batch_size", str(batch),
        "--output_dir", str(task_out.resolve()),
    ]
    t0 = time.time()
    p = run_cmd(cmd, env, log, timeout=3600)
    stdout_score = parse_sentence_score(p.stdout)
    report_score = read_report_score(task_out)
    score = stdout_score if stdout_score is not None else report_score
    return {"column": column, "task": task, "data_path": data_path, "score": score,
            "stdout_score": stdout_score, "report_score": report_score,
            "returncode": p.returncode, "elapsed_sec": round(time.time() - t0, 3),
            "log": str(log), "output_dir": str(task_out)}


def eval_reading(target: str, model_path: Path, env: Dict[str, str], out_root: Path, work_tag: str) -> Dict[str, Any]:
    task_out = out_root / "eval_outputs" / work_tag / target / "Reading"
    task_out.mkdir(parents=True, exist_ok=True)
    log = out_root / "logs" / work_tag / f"{target}_Reading.log"
    revision = f"step032_{target}_Reading"
    cmd = [
        PYTHON_EXE, "-m", "evaluation_pipeline.reading.run",
        "--model_path_or_name", str(model_path.resolve()),
        "--backend", "mlm",
        "--data_path", "evaluation_data/fast_eval/reading/reading_data.csv",
        "--revision_name", revision,
        "--output_dir", str(task_out.resolve()),
    ]
    t0 = time.time()
    p = run_cmd(cmd, env, log, timeout=3600)
    scores = parse_reading_scores(p.stdout)
    if not scores:
        scores = read_reading_report(task_out)
    return {"column": "Reading", "scores": scores, "returncode": p.returncode,
            "elapsed_sec": round(time.time() - t0, 3), "log": str(log), "output_dir": str(task_out)}


def per_target_path(out_root: Path, target: str) -> Path:
    return out_root / "per_target" / f"{target}.json"


def save_target(out_root: Path, target: str, payload: Dict[str, Any]) -> None:
    p = per_target_path(out_root, target)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_run_summary(target: str) -> Dict[str, Any]:
    cfg = TARGETS[target]
    model_path = Path(cfg["model_path"])
    run_dir = model_path.parent.parent
    p = run_dir / "scientific_metrics.json"
    out: Dict[str, Any] = {
        "target": target,
        "family": cfg["family"],
        "seed": cfg["seed"],
        "exposure_m": cfg["exposure_m"],
        "model_path": str(model_path),
        "run_dir": str(run_dir),
        "reference_role": cfg["reference_role"],
        "scientific_metrics_path": str(p),
        "endpoint_exists": (model_path / "model.safetensors").exists(),
    }
    if p.exists():
        m = json.loads(p.read_text(encoding="utf-8"))
        for k in [
            "variant", "word_exposure", "loss_first", "loss_last", "actual_training_steps",
            "parameter_count", "vocab_size", "tokenizer_label", "tokenizer_path",
            "masking_curriculum", "mask_prob_start", "mask_prob_end", "learning_rate",
            "warmup_fraction", "weight_decay", "n_layer", "hidden_size", "n_head",
            "max_position_embeddings", "max_relative_positions", "seed", "extra_init_seed", "train_rng_seed",
            "data_source_type", "example_jsonl", "example_jsonl_label",
        ]:
            if k in m:
                out[k] = m[k]
    return out


def eval_target(target: str, gpu: str, work_tag: str, columns: Iterable[str], out_root: Path, force: bool = False, dry_run: bool = False) -> Dict[str, Any]:
    cfg = TARGETS[target]
    model_path = Path(cfg["model_path"])
    if not (model_path / "model.safetensors").exists():
        raise FileNotFoundError(f"checkpoint missing for {target}: {model_path}")
    p = per_target_path(out_root, target)
    payload = None if force or not p.exists() else json.loads(p.read_text(encoding="utf-8"))
    if payload is None:
        payload = {"target": target, "model_path": str(model_path),
                   "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "gpu": gpu, "work_tag": work_tag, "tasks": {}, "run_summary": read_run_summary(target)}
    if dry_run:
        payload["dry_run"] = True
        save_target(out_root, target, payload)
        return payload
    env = setup_env(out_root, f"{work_tag}_{target}", gpu)
    for col in columns:
        if col == "Reading":
            if "Reading" not in payload["tasks"] or force:
                rec = eval_reading(target, model_path, env, out_root, work_tag)
                payload["tasks"]["Reading"] = rec
                print(json.dumps({"event": "eval_done", "target": target, "column": col, "scores": rec.get("scores"), "rc": rec["returncode"]}), flush=True)
        else:
            if col not in TASK_BY_COL:
                raise ValueError(f"unknown column {col}")
            if col not in payload["tasks"] or force:
                rec = eval_sentence_column(target, model_path, col, env, out_root, work_tag)
                payload["tasks"][col] = rec
                print(json.dumps({"event": "eval_done", "target": target, "column": col, "score": rec.get("score"), "rc": rec["returncode"]}), flush=True)
        save_target(out_root, target, payload)
    payload["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save_target(out_root, target, payload)
    return payload


def scores_from_payload(raw: Dict[str, Any]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {k: None for k in TABLE_KEYS}
    tasks = raw.get("tasks", {})
    for col in [c for c, _, _, _ in TASKS]:
        if col in tasks:
            out[col] = tasks[col].get("score")
    if "Reading" in tasks:
        for k, v in tasks["Reading"].get("scores", {}).items():
            out[k] = v
    if out.get("GlobalPIQA_parallel") is not None and out.get("GlobalPIQA_nonparallel") is not None:
        out["GlobalPIQA_mean"] = (out["GlobalPIQA_parallel"] + out["GlobalPIQA_nonparallel"]) / 2.0  # type: ignore[operator]
    eq_keys = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_mean", "Reading"]
    if all(out.get(k) is not None for k in eq_keys):
        out["equal7_mean"] = sum(float(out[k]) for k in eq_keys) / len(eq_keys)  # type: ignore[arg-type]
    eq_full_keys = ["BLiMP", "Supplement", "EWoK", "Entity_full", "COMPS", "GlobalPIQA_mean", "Reading"]
    if all(out.get(k) is not None for k in eq_full_keys):
        out["equal7_full_entity"] = sum(float(out[k]) for k in eq_full_keys) / len(eq_full_keys)  # type: ignore[arg-type]
    return out


def add_composites(scores: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(scores)
    if "GlobalPIQA_mean" not in out and out.get("GlobalPIQA_parallel") is not None and out.get("GlobalPIQA_nonparallel") is not None:
        out["GlobalPIQA_mean"] = (float(out["GlobalPIQA_parallel"]) + float(out["GlobalPIQA_nonparallel"])) / 2.0
    eq_keys = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_mean", "Reading"]
    if all(k in out and out[k] is not None for k in eq_keys):
        out["equal7_mean"] = sum(float(out[k]) for k in eq_keys) / len(eq_keys)
    eq_full_keys = ["BLiMP", "Supplement", "EWoK", "Entity_full", "COMPS", "GlobalPIQA_mean", "Reading"]
    if all(k in out and out[k] is not None for k in eq_full_keys):
        out["equal7_full_entity"] = sum(float(out[k]) for k in eq_full_keys) / len(eq_full_keys)
    return out


def diff(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, float]:
    out = {}
    for k in TABLE_KEYS:
        if a.get(k) is not None and b.get(k) is not None:
            out[k] = round(float(a[k]) - float(b[k]), 4)
    return out


def build_summary(targets: List[str], out_root: Path, work_tag: str, dry_run: bool) -> Dict[str, Any]:
    raw = {t: json.loads(per_target_path(out_root, t).read_text(encoding="utf-8")) for t in targets if per_target_path(out_root, t).exists()}
    table = {t: scores_from_payload(raw[t]) for t in raw}
    known = {k: add_composites(v) for k, v in KNOWN_FAST_REFERENCES.items()}
    comparisons: Dict[str, Dict[str, float]] = {}
    for t, s in table.items():
        seed = TARGETS[t]["seed"]
        ref_key = f"r{seed}_100M_fast"
        if ref_key in known:
            comparisons[f"{t}_minus_{ref_key}"] = diff(s, known[ref_key])
    # Common 80M across-seed comparison, if both ran.
    if "r43022_80M" in table and "r43122_80M" in table:
        comparisons["r43122_80M_minus_r43022_80M"] = diff(table["r43122_80M"], table["r43022_80M"])
    if "r43022_90M" in table and "r43022_80M" in table:
        comparisons["r43022_90M_minus_r43022_80M"] = diff(table["r43022_90M"], table["r43022_80M"])
    if "r43122_45M" in table and "r43122_80M" in table:
        comparisons["r43122_45M_minus_r43122_80M"] = diff(table["r43122_45M"], table["r43122_80M"])

    candidate_scores = []
    for t, s in table.items():
        candidate_scores.append((t, s.get("equal7_full_entity")))
    best_equal7_full = max([x for x in candidate_scores if x[1] is not None], key=lambda x: x[1], default=None)
    failures = []
    for t, payload in raw.items():
        for col, rec in payload.get("tasks", {}).items():
            if rec.get("returncode") not in [0, None]:
                failures.append({"target": t, "column": col, "returncode": rec.get("returncode"), "log": rec.get("log")})
            if col != "Reading" and rec.get("score") is None:
                failures.append({"target": t, "column": col, "error": "score_missing", "log": rec.get("log")})
            if col == "Reading" and not rec.get("scores"):
                failures.append({"target": t, "column": col, "error": "reading_scores_missing", "log": rec.get("log")})

    result = {
        "status": "BROAD_NOAOA_LATE_CHECKPOINTS_DRY_RUN" if dry_run else "BROAD_NOAOA_LATE_CHECKPOINTS_COMPLETE",
        "purpose": "Minimal broad no-AoA surface over selected existing compact_view_reinvest checkpoints to decide whether earlier stopping deserves averaging/full-eval follow-up before any new training.",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "targets_requested": targets,
        "columns": DEFAULT_COLUMNS,
        "work_tag": work_tag,
        "strict_repo": str(STRICT),
        "table": table,
        "known_100M_fast_references": known,
        "reference_columns": REFERENCE_COLUMNS,
        "comparisons": comparisons,
        "best_by_equal7_full_entity_in_this_run": best_equal7_full,
        "failures": failures,
        "raw_paths": {t: str(per_target_path(out_root, t)) for t in raw},
        "out_root": str(out_root),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    out_json = out_root / "broad_noaoa_late_checkpoints_summary.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_note(result, out_root / "broad_noaoa_late_checkpoints.md")
    return result


def fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def write_note(summary: Dict[str, Any], out_md: Path) -> None:
    lines = [
        "# research — broad no-AoA late-checkpoint screen\n\n",
        "Existing compact_view_reinvest checkpoints only; no training, corpus changes, SuperGLUE, AoA, or official leaderboard collation.\n\n",
        f"Summary JSON: `{out_md.parent / 'broad_noaoa_late_checkpoints_summary.json'}`\n\n",
        "## Scores\n",
        "| target | BLiMP | Supp | EWoK-fast | Entity fast | Entity full | COMPS | GPIQA | Reading | equal7 fast | equal7 fullEnt |\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for t, scores in summary.get("table", {}).items():
        lines.append("| " + t + " | " + " | ".join(fmt(scores.get(k)) for k in [
            "BLiMP", "Supplement", "EWoK", "Entity", "Entity_full", "COMPS", "GlobalPIQA_mean", "Reading", "equal7_mean", "equal7_full_entity",
        ]) + " |\n")
    if summary.get("known_100M_fast_references"):
        lines.append("\n## Known 100M fast references (not reevaluated here)\n")
        lines.append("| reference | BLiMP | Supp | EWoK-fast | Entity full | COMPS | GPIQA | Reading | equal7 fullEnt |\n")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for t, scores in summary["known_100M_fast_references"].items():
            lines.append("| " + t + " | " + " | ".join(fmt(scores.get(k)) for k in [
                "BLiMP", "Supplement", "EWoK", "Entity_full", "COMPS", "GlobalPIQA_mean", "Reading", "equal7_full_entity",
            ]) + " |\n")
    lines.append("\n## Contrasts\n")
    keep = ["equal7_full_entity", "equal7_mean", "BLiMP", "Supplement", "EWoK", "Entity_full", "COMPS", "GlobalPIQA_mean", "Reading"]
    for name, delta in summary.get("comparisons", {}).items():
        vals = ", ".join(f"{k} {delta[k]:+.3f}" for k in keep if k in delta)
        lines.append(f"- **{name}**: {vals}\n")
    lines.append("\n## Scientific read\n")
    lines.append("- This screen can select candidate existing checkpoints for official or averaging follow-up, but it cannot establish an Overall score because SuperGLUE, AoA, and current official EWoK/full collation are absent.\n")
    lines.append("- Continue the stopping/averaging route only if a candidate improves Supplement without erasing EWoK, Entity, GlobalPIQA, Reading, and the broad no-AoA balance relative to its 100M reference.\n")
    if summary.get("failures"):
        lines.append(f"- Failures or missing scores present: {summary['failures']}\n")
    out_md.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    if not STRICT.exists():
        raise FileNotFoundError(STRICT)
    for t in args.targets:
        model_path = Path(TARGETS[t]["model_path"])
        if not (model_path / "model.safetensors").exists():
            raise FileNotFoundError(f"missing {model_path / 'model.safetensors'}")
    manifest = {
        "status": "BROAD_NOAOA_LATE_CHECKPOINTS_STARTING",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Evaluate four selected existing reinvest checkpoints on a broad fast/no-AoA surface before any new training or official full evaluation.",
        "targets": {t: {**{k: (str(v) if isinstance(v, Path) else v) for k, v in TARGETS[t].items()}} for t in args.targets},
        "columns": args.columns,
        "gpu": args.gpu,
        "dry_run": bool(args.dry_run),
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    t0 = time.time()
    for target in args.targets:
        print(json.dumps({"event": "target_start", "target": target, "gpu": args.gpu, "model_path": str(TARGETS[target]["model_path"])}, ensure_ascii=False), flush=True)
        eval_target(target, str(args.gpu), args.work_tag, args.columns, out_root, force=args.force, dry_run=args.dry_run)
    summary = build_summary(args.targets, out_root, args.work_tag, args.dry_run)
    summary["elapsed_sec"] = round(time.time() - t0, 3)
    out_json = out_root / "broad_noaoa_late_checkpoints_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_note(summary, out_root / "broad_noaoa_late_checkpoints.md")
    print(json.dumps({
        "status": summary["status"],
        "targets": args.targets,
        "out_json": str(out_json),
        "out_md": str(out_root / "broad_noaoa_late_checkpoints.md"),
        "failures": summary.get("failures"),
        "best_by_equal7_full_entity_in_this_run": summary.get("best_by_equal7_full_entity_in_this_run"),
        "elapsed_sec": summary["elapsed_sec"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
