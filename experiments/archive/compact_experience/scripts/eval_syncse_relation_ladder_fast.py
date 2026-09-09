#!/usr/bin/env python3
"""research: evaluate the SynCSE semantic-relation ladder.

Expected trained arms:
  - hardneg_coherent_exact10: sent0 + hard_neg, same row
  - hardneg_mismatched_exact10: sent0 + hard_neg, deranged row
  - paraphrase_aligned_exact10: sent0 + sent1, same row
  - paraphrase_mismatched_exact10: sent0 + sent1, deranged row

  The two primary scientific contrasts are:
    - hardneg_coherent - hardneg_mismatched: SynCSE hard-negative relation adjacency,
      often lexical/entity recurrence with possible contradiction or role reversal,
      versus random hard-negative adjacency.
    - paraphrase_aligned - paraphrase_mismatched: local paraphrase/rewrite adjacency
      versus random paraphrase-side adjacency.
  The cross-relation interaction is a relation-type probe only. It must not be
  interpreted as clean rewrite correspondence beyond generic coherence without a
  true document-adjacent or topic-matched non-rewrite control.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT_INITIAL_MODEL_STUDIES = pathlib.Path("experiments/archive/initial_model_studies")
ROOT_COMPACT_EXPERIENCE = pathlib.Path("experiments/archive/compact_experience")
STRICT = ROOT_INITIAL_MODEL_STUDIES / "repos/babylm-eval/strict"
RUN_BASE = ROOT_COMPACT_EXPERIENCE / "training/runs"
OUT_ROOT = ROOT_COMPACT_EXPERIENCE / "data/syncse_relation_ladder_eval"
NOTE_PATH = (ROOT_COMPACT_EXPERIENCE.parents[2] / 'research/notes/compact_experience/syncse_relation_ladder_eval.md')
FINAL_JSON_PATH = OUT_ROOT / "relation_ladder_eval_summary.json"

RUNS: Dict[str, pathlib.Path] = {
    "hardneg_coherent": RUN_BASE / "relation_ladder_hardneg_coherent_exact10_seed43",
    "hardneg_mismatched": RUN_BASE / "relation_ladder_hardneg_mismatched_exact10_seed43",
    "paraphrase_aligned": RUN_BASE / "relation_ladder_paraphrase_aligned_exact10_seed43",
    "paraphrase_mismatched": RUN_BASE / "relation_ladder_paraphrase_mismatched_exact10_seed43",
    "initial_model_baseline": ROOT_INITIAL_MODEL_STUDIES / "training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256",
}
DEFAULT_TARGETS = ["hardneg_coherent", "hardneg_mismatched", "paraphrase_aligned", "paraphrase_mismatched", "initial_model_baseline"]

TASKS = [
    ("BLiMP", "blimp", "evaluation_data/fast_eval/blimp_fast", 128),
    ("Supplement", "blimp", "evaluation_data/fast_eval/supplement_fast", 128),
    ("EWoK", "ewok", "evaluation_data/fast_eval/evaluation_data/fast_eval/ewok_fast", 64),
    ("Entity", "entity_tracking", "evaluation_data/fast_eval/entity_tracking_fast", 128),
    ("COMPS", "comps", "evaluation_data/full_eval/comps", 128),
    ("GlobalPIQA_parallel", "global_piqa_parallel", "evaluation_data/fast_eval/global_piqa_parallel", 128),
    ("GlobalPIQA_nonparallel", "global_piqa_nonparallel", "evaluation_data/fast_eval/global_piqa_nonparallel", 128),
]
TASK_BY_COL = {c: (t, d, b) for c, t, d, b in TASKS}
TABLE_KEYS = [
    "BLiMP", "Supplement", "EWoK", "Entity", "COMPS",
    "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "GlobalPIQA_mean",
    "Reading", "Reading_eye", "Reading_self_paced", "equal7_mean",
]
NLP_PROXY_KEYS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_mean"]


def parse_target(spec: str) -> Tuple[str, str]:
    if spec not in RUNS:
        raise KeyError(f"unknown target {spec}; choices={sorted(RUNS)}")
    return spec, spec


def model_path_for(spec: str) -> pathlib.Path:
    # Evaluate final root hf_model so this works even when exact-pass checkpoint names are word-count-based.
    return RUNS[spec] / "hf_model"


def setup_env(work_tag: str, gpu: int) -> Dict[str, str]:
    env = os.environ.copy()
    hf = OUT_ROOT / "hf_cache" / work_tag
    env["HF_HOME"] = str(hf.resolve())
    env["HF_HUB_CACHE"] = str((hf / "hub").resolve())
    env["HF_DATASETS_CACHE"] = str((hf / "datasets").resolve())
    env["TRANSFORMERS_CACHE"] = str((hf / "transformers").resolve())
    env["HF_MODULES_CACHE"] = str((hf / "modules").resolve())
    env["NLTK_DATA"] = str((ROOT_INITIAL_MODEL_STUDIES / "data/nltk_data").resolve())
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    for k in ["HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE", "HF_MODULES_CACHE"]:
        pathlib.Path(env[k]).mkdir(parents=True, exist_ok=True)
    return env


def parse_sentence_score(text: str) -> Optional[float]:
    m = re.search(r"AVERAGE ACCURACY\s*\n\s*([+-]?[0-9]+(?:\.[0-9]+)?)", text)
    if m:
        return float(m.group(1))
    for line in reversed(text.splitlines()):
        s = line.strip()
        m = re.match(r"^(?:[0-9.]+\s+)?([+-]?[0-9]+(?:\.[0-9]+)?)$", s)
        if m:
            val = float(m.group(1))
            if -1000 < val < 1000:
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


def read_report_score(task_out: pathlib.Path) -> Optional[float]:
    candidates = sorted(task_out.rglob("best_temperature_report.txt"), key=lambda p: (p.stat().st_mtime, str(p)))
    if not candidates:
        candidates = sorted(task_out.rglob("*.txt"), key=lambda p: (p.stat().st_mtime, str(p)))
    for p in reversed(candidates):
        score = parse_sentence_score(p.read_text(encoding="utf-8", errors="replace"))
        if score is not None:
            return score
    return None


def read_reading_report(task_out: pathlib.Path) -> Dict[str, float]:
    candidates = sorted(task_out.rglob("report.txt"), key=lambda p: (p.stat().st_mtime, str(p)))
    for p in reversed(candidates):
        scores = parse_reading_scores(p.read_text(encoding="utf-8", errors="replace"))
        if scores:
            return scores
    return {}


def run_cmd(cmd: List[str], env: Dict[str, str], log_path: pathlib.Path, timeout: int = 1800) -> subprocess.CompletedProcess[str]:
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


def eval_sentence_column(target: str, model_path: pathlib.Path, column: str, work_tag: str, env: Dict[str, str]) -> Dict[str, Any]:
    task, data_path, batch = TASK_BY_COL[column]
    task_out = OUT_ROOT / "eval_outputs" / work_tag / target / column
    task_out.mkdir(parents=True, exist_ok=True)
    log = OUT_ROOT / "logs" / work_tag / f"{target}_{column}.log"
    revision = f"step019_{target}_{column}"
    cmd = [
        sys.executable, "-m", "evaluation_pipeline.sentence_zero_shot.run",
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
    p = run_cmd(cmd, env, log, timeout=1800)
    stdout_score = parse_sentence_score(p.stdout)
    report_score = read_report_score(task_out)
    score = stdout_score if stdout_score is not None else report_score
    return {"column": column, "task": task, "data_path": data_path, "score": score, "stdout_score": stdout_score, "report_score": report_score, "returncode": p.returncode, "elapsed_sec": round(time.time() - t0, 3), "log": str(log), "output_dir": str(task_out)}


def eval_reading(target: str, model_path: pathlib.Path, work_tag: str, env: Dict[str, str]) -> Dict[str, Any]:
    task_out = OUT_ROOT / "eval_outputs" / work_tag / target / "Reading"
    task_out.mkdir(parents=True, exist_ok=True)
    log = OUT_ROOT / "logs" / work_tag / f"{target}_Reading.log"
    revision = f"step019_{target}_Reading"
    cmd = [
        sys.executable, "-m", "evaluation_pipeline.reading.run",
        "--model_path_or_name", str(model_path.resolve()),
        "--backend", "mlm",
        "--data_path", "evaluation_data/fast_eval/reading/reading_data.csv",
        "--revision_name", revision,
        "--output_dir", str(task_out.resolve()),
    ]
    t0 = time.time()
    p = run_cmd(cmd, env, log, timeout=1800)
    scores = parse_reading_scores(p.stdout)
    if not scores:
        scores = read_reading_report(task_out)
    return {"column": "Reading", "scores": scores, "returncode": p.returncode, "elapsed_sec": round(time.time() - t0, 3), "log": str(log), "output_dir": str(task_out)}


def per_target_path(target: str) -> pathlib.Path:
    return OUT_ROOT / "per_target" / f"{target}.json"


def save_target(target: str, payload: Dict[str, Any]) -> None:
    p = per_target_path(target)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_run_summary(target: str) -> Dict[str, Any]:
    run_dir = RUNS[target]
    p = run_dir / "scientific_metrics.json"
    out: Dict[str, Any] = {"run_dir": str(run_dir), "scientific_metrics_path": str(p)}
    if p.exists():
        m = json.loads(p.read_text(encoding="utf-8"))
        for k in ["word_exposure", "loss_first", "loss_last", "actual_training_steps", "parameter_count", "example_jsonl_total_words", "selected_for_training_words", "mask_mode", "mask_prob", "source_words_consumed"]:
            if k in m:
                out[k] = m[k]
        out["saved_checkpoint_names"] = [x.get("name") for x in m.get("saved_checkpoints", [])]
    return out


def eval_target(target: str, gpu: int, work_tag: str, columns: Iterable[str], force: bool = False) -> Dict[str, Any]:
    model_path = model_path_for(target)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    p = per_target_path(target)
    payload = None if force or not p.exists() else json.loads(p.read_text(encoding="utf-8"))
    if payload is None:
        payload = {"target": target, "model_path": str(model_path), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "gpu": gpu, "work_tag": work_tag, "tasks": {}, "reading": None, "run_summary": read_run_summary(target)}
    env = setup_env(work_tag, gpu)
    for column in list(columns):
        if column == "Reading":
            if payload.get("reading") and not force:
                print(json.dumps({"event": "resume", "target": target, "column": column}), flush=True)
                continue
            print(json.dumps({"event": "eval_start", "target": target, "column": column, "gpu": gpu}), flush=True)
            rec = eval_reading(target, model_path, work_tag, env)
            payload["reading"] = rec
            save_target(target, payload)
            print(json.dumps({"event": "eval_done", "target": target, "column": column, "scores": rec.get("scores"), "rc": rec["returncode"]}), flush=True)
        else:
            if payload.get("tasks", {}).get(column) and not force:
                print(json.dumps({"event": "resume", "target": target, "column": column}), flush=True)
                continue
            print(json.dumps({"event": "eval_start", "target": target, "column": column, "gpu": gpu}), flush=True)
            rec = eval_sentence_column(target, model_path, column, work_tag, env)
            payload.setdefault("tasks", {})[column] = rec
            save_target(target, payload)
            print(json.dumps({"event": "eval_done", "target": target, "column": column, "score": rec.get("score"), "rc": rec["returncode"]}), flush=True)
    payload["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    payload["run_summary"] = read_run_summary(target)
    save_target(target, payload)
    return payload


def target_to_table(rec: Dict[str, Any]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    tasks = rec.get("tasks", {})
    for col, _, _, _ in TASKS:
        out[col] = tasks.get(col, {}).get("score")
    gp = out.get("GlobalPIQA_parallel")
    gn = out.get("GlobalPIQA_nonparallel")
    out["GlobalPIQA_mean"] = (gp + gn) / 2.0 if gp is not None and gn is not None else None
    rs = (rec.get("reading") or {}).get("scores") or {}
    out["Reading"] = rs.get("Reading")
    out["Reading_eye"] = rs.get("Reading_eye")
    out["Reading_self_paced"] = rs.get("Reading_self_paced")
    vals = [out.get(k) for k in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_mean", "Reading"]]
    valid = [v for v in vals if v is not None]
    out["equal7_mean"] = sum(valid) / len(valid) if valid else None
    return out


def weighted_proxy(row: Dict[str, Optional[float]]) -> Optional[float]:
    vals = [row.get(k) for k in NLP_PROXY_KEYS]
    if any(v is None for v in vals):
        return None
    return (3.0 / 28.0) * sum(v for v in vals if v is not None) + (1.0 / 8.0) * (row.get("Reading") or 0.0)


def delta(a: Dict[str, Optional[float]], b: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    return {k: (a.get(k) - b.get(k) if a.get(k) is not None and b.get(k) is not None else None) for k in TABLE_KEYS}


def fmt(x: Optional[float]) -> str:
    return "NA" if x is None else f"{x:.3f}"


def build_summary(targets: List[str]) -> Dict[str, Any]:
    table: Dict[str, Dict[str, Optional[float]]] = {}
    raw: Dict[str, Any] = {}
    for t in targets:
        p = per_target_path(t)
        if p.exists():
            rec = json.loads(p.read_text(encoding="utf-8"))
            raw[t] = rec
            table[t] = target_to_table(rec)
    contrasts: Dict[str, Dict[str, Optional[float]]] = {}
    for name, a, b in [
        ("hardneg_coherent_minus_hardneg_mismatched", "hardneg_coherent", "hardneg_mismatched"),
        ("paraphrase_aligned_minus_paraphrase_mismatched", "paraphrase_aligned", "paraphrase_mismatched"),
        ("paraphrase_aligned_minus_hardneg_coherent", "paraphrase_aligned", "hardneg_coherent"),
        ("paraphrase_mismatched_minus_hardneg_mismatched", "paraphrase_mismatched", "hardneg_mismatched"),
        ("paraphrase_aligned_minus_initial_model_studies_baseline", "paraphrase_aligned", "initial_model_baseline"),
        ("hardneg_coherent_minus_initial_model_studies_baseline", "hardneg_coherent", "initial_model_baseline"),
    ]:
        if a in table and b in table:
            contrasts[name] = delta(table[a], table[b])
      # Cross-relation interaction: (paraphrase aligned-mismatch) - (hardneg relation-mismatch).
      # Because SynCSE hard_neg often includes contradiction/role reversal, this is
      # not a clean coherence-controlled causal estimate of rewrite correspondence.
    did: Dict[str, Optional[float]] = {}
    pa = contrasts.get("paraphrase_aligned_minus_paraphrase_mismatched")
    hc = contrasts.get("hardneg_coherent_minus_hardneg_mismatched")
    if pa and hc:
        did = {k: (pa.get(k) - hc.get(k) if pa.get(k) is not None and hc.get(k) is not None else None) for k in TABLE_KEYS}
        contrasts["relation_type_interaction_paraphrase_minus_hardneg"] = did
    payload = {
        "status": "RELATION_LADDER_FAST_EVAL_DONE" if len(raw) == len(targets) else "RELATION_LADDER_FAST_EVAL_PARTIAL",
        "targets_requested": targets,
        "targets_completed": sorted(raw),
        "table": table,
        "absolute_weighted_fast_proxy": {k: weighted_proxy(v) for k, v in table.items()},
        "contrasts": contrasts,
          "weighted_fast_proxy_delta": {k: weighted_proxy(v) for k, v in contrasts.items()},
          "interpretation_caveat": "SynCSE hard_neg is not a pure same-topic non-synonymous coherent control; samples can contain negation, role reversal, and factual contradiction. Treat hardneg contrasts as relation-type probes and do not use the cross-relation interaction as direct attribution to rewrite correspondence without a true document-adjacent or topic-matched non-rewrite control.",
          "run_summaries": {t: read_run_summary(t) for t in targets if t in RUNS},
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    FINAL_JSON_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_note(payload)
    return payload


def write_note(payload: Dict[str, Any]) -> None:
    table = payload.get("table", {})
    abs_proxy = payload.get("absolute_weighted_fast_proxy", {})
    lines = [
        "# research — SynCSE semantic-relation ladder evaluation",
        "",
        f"Summary JSON: `{FINAL_JSON_PATH}`",
        "",
          "This evaluation is a relation-type probe. SynCSE `hard_neg` is not a pure same-topic non-synonymous coherent control; examples often preserve entities/lexical fields while introducing semantic incompatibility, negation, or role reversal. Therefore the hard-neg contrast is informative about hard-negative adjacency, but the cross-relation interaction must not be used as direct attribution to rewrite correspondence beyond generic coherence.",
        "",
        "## Scores",
        "",
        "| target | BLiMP | Supplement | EWoK | Entity | COMPS | GPIQA mean | Reading | equal-7 | weighted fast proxy | loss_last |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    runs = payload.get("run_summaries", {})
    for t in ["hardneg_coherent", "hardneg_mismatched", "paraphrase_aligned", "paraphrase_mismatched", "initial_model_baseline"]:
        if t not in table:
            continue
        s = table[t]
        run = runs.get(t, {})
        lines.append(f"| {t} | {fmt(s.get('BLiMP'))} | {fmt(s.get('Supplement'))} | {fmt(s.get('EWoK'))} | {fmt(s.get('Entity'))} | {fmt(s.get('COMPS'))} | {fmt(s.get('GlobalPIQA_mean'))} | {fmt(s.get('Reading'))} | {fmt(s.get('equal7_mean'))} | {fmt(abs_proxy.get(t))} | {fmt(run.get('loss_last'))} |")
    lines += ["", "## Contrasts", "", "| contrast | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | Reading | equal-7 | weighted fast proxy |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, d in payload.get("contrasts", {}).items():
        lines.append(f"| {name} | {fmt(d.get('BLiMP'))} | {fmt(d.get('Supplement'))} | {fmt(d.get('EWoK'))} | {fmt(d.get('Entity'))} | {fmt(d.get('COMPS'))} | {fmt(d.get('GlobalPIQA_mean'))} | {fmt(d.get('Reading'))} | {fmt(d.get('equal7_mean'))} | {fmt(payload.get('weighted_fast_proxy_delta', {}).get(name))} |")
    NOTE_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*", default=DEFAULT_TARGETS)
    ap.add_argument("--columns", nargs="*", default=[c for c, _, _, _ in TASKS] + ["Reading"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--work_tag", default="relation_ladder")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    targets = args.targets or DEFAULT_TARGETS
    if args.summary_only:
        print(json.dumps(build_summary(targets), indent=2, ensure_ascii=False))
        return
    for target in targets:
        print(json.dumps({"event": "target_start", "target": target, "gpu": args.gpu}), flush=True)
        eval_target(target, args.gpu, args.work_tag, args.columns, force=args.force)
        print(json.dumps({"event": "target_done", "target": target}), flush=True)
    print(json.dumps(build_summary(targets), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
