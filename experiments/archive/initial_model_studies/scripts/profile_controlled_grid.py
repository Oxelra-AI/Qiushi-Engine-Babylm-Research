#!/usr/bin/env python3
"""Profile the research controlled dense-vs-memory grid with official BabyLM fast/local tasks."""
from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import sys
from typing import Any

STUDY = pathlib.Path("experiments/archive/initial_model_studies")
STRICT_DIR = STUDY / "repos/babylm-eval/strict"
AI_RUNS = STUDY / "training/runs"
HF_MODULES_CACHE = STUDY / "training/hf_modules_cache"
OUT_JSON = STUDY / "data/controlled_grid_profile.json"
OUT_NOTE = (STUDY.parents[2] / 'research/notes/initial_model_studies/controlled_grid_profile_interpretation.md')

TASKS = [
    ("blimp_fast", "blimp", "evaluation_data/fast_eval/blimp_fast"),
    ("supplement_fast", "blimp", "evaluation_data/fast_eval/supplement_fast"),
    ("ewok_fast", "ewok", "evaluation_data/fast_eval/evaluation_data/fast_eval/ewok_fast"),
    ("entity_tracking_fast", "entity_tracking", "evaluation_data/fast_eval/entity_tracking_fast"),
    ("comps", "comps", "evaluation_data/full_eval/comps"),
]
COLS = ["blimp_fast", "supplement_fast", "ewok_fast", "entity_tracking_fast", "comps", "reading_eye_tracking", "reading_self_paced"]
GRID = []
for pool_name in ["pool1M", "pool10M"]:
    for seed in [42, 43]:
        for short, variant in [("dense", "dense_untied_causal"), ("memory", "memory_causal")]:
            GRID.append({
                "pool_name": pool_name,
                "seed": seed,
                "short": short,
                "variant": variant,
                "run_id": f"babylm_step26_ctrl_{short}_{pool_name}_seed{seed}_1M",
            })


def run_cmd(cmd: list[str], cwd: pathlib.Path, env: dict[str, str]) -> None:
    print("CMD", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def read_avg(report: pathlib.Path) -> float | None:
    if not report.exists():
        return None
    txt = report.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"AVERAGE ACCURACY\s*\n([0-9.\-]+)", txt)
    return float(m.group(1)) if m else None


def read_reading(report: pathlib.Path) -> dict[str, float | None]:
    if not report.exists():
        return {"eye_tracking": None, "self_paced": None}
    txt = report.read_text(encoding="utf-8", errors="replace")
    out: dict[str, float | None] = {}
    for label, key in [("EYE TRACKING SCORE", "eye_tracking"), ("SELF-PACED READING SCORE", "self_paced")]:
        m = re.search(re.escape(label) + r":\s*([0-9.\-]+)", txt)
        out[key] = float(m.group(1)) if m else None
    return out


def load_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def profile_cell(cell: dict[str, Any], env: dict[str, str]) -> dict[str, Any]:
    run_dir = AI_RUNS / cell["run_id"]
    model_path = (run_dir / "hf_model").resolve()
    outdir = (run_dir / "eval_results_controlled_grid").resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    metrics = load_json(run_dir / "scientific_metrics.json")
    order = load_json(run_dir / "example_order_manifest.json")
    result: dict[str, Any] = {
        **cell,
        "model_path": str(model_path),
        "outdir": str(outdir),
        "parameter_count": metrics.get("parameter_count"),
        "loss_first": metrics.get("loss_first"),
        "loss_last": metrics.get("loss_last"),
        "word_exposure": metrics.get("word_exposure"),
        "example_pool_words_actual": metrics.get("example_pool_words_actual"),
        "lr_schedule_total_steps": metrics.get("lr_schedule_total_steps"),
        "shared_core_init_applied": metrics.get("shared_core_init_applied"),
        "source_words_consumed": order.get("source_words_consumed"),
        "scores": {},
        "reports": {},
    }
    for task_name, task, data_path in TASKS:
        dp = STRICT_DIR / data_path
        if not dp.exists():
            result["scores"][task_name] = None
            result["reports"][task_name] = f"MISSING_DATA:{dp}"
            continue
        run_cmd([
            sys.executable, "-m", "evaluation_pipeline.sentence_zero_shot.run",
            "--model_path_or_name", str(model_path),
            "--backend", "causal",
            "--task", task,
            "--data_path", data_path,
            "--save_predictions",
            "--revision_name", "chck_1M",
            "--batch_size", "64",
            "--output_dir", str(outdir),
        ], STRICT_DIR, env)
        report = outdir / "hf_model" / "chck_1M" / "zero_shot" / "causal" / task / task_name / "best_temperature_report.txt"
        result["scores"][task_name] = read_avg(report)
        result["reports"][task_name] = str(report)
    reading_data = STRICT_DIR / "evaluation_data/fast_eval/reading/reading_data.csv"
    if reading_data.exists():
        run_cmd([
            sys.executable, "-m", "evaluation_pipeline.reading.run",
            "--model_path_or_name", str(model_path),
            "--backend", "causal",
            "--data_path", "evaluation_data/fast_eval/reading/reading_data.csv",
            "--revision_name", "chck_1M",
            "--output_dir", str(outdir),
        ], STRICT_DIR, env)
        rreport = outdir / "hf_model" / "chck_1M" / "zero_shot" / "causal" / "reading" / "report.txt"
        reading = read_reading(rreport)
        result["scores"].update({f"reading_{k}": v for k, v in reading.items()})
        result["reports"]["reading"] = str(rreport)
    else:
        result["scores"].update({"reading_eye_tracking": None, "reading_self_paced": None})
        result["reports"]["reading"] = f"MISSING_DATA:{reading_data}"
    return result


def delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return round(a - b, 4)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = []
    for pool_name in ["pool1M", "pool10M"]:
        for seed in [42, 43]:
            dense = next(r for r in rows if r["pool_name"] == pool_name and r["seed"] == seed and r["short"] == "dense")
            memory = next(r for r in rows if r["pool_name"] == pool_name and r["seed"] == seed and r["short"] == "memory")
            pairs.append({
                "pool_name": pool_name,
                "seed": seed,
                "memory_minus_dense": {c: delta(memory["scores"].get(c), dense["scores"].get(c)) for c in COLS},
                "dense_scores": {c: dense["scores"].get(c) for c in COLS},
                "memory_scores": {c: memory["scores"].get(c) for c in COLS},
            })
    by_col: dict[str, Any] = {}
    for c in COLS:
        vals1 = [p["memory_minus_dense"][c] for p in pairs if p["pool_name"] == "pool1M" and p["memory_minus_dense"][c] is not None]
        vals10 = [p["memory_minus_dense"][c] for p in pairs if p["pool_name"] == "pool10M" and p["memory_minus_dense"][c] is not None]
        by_col[c] = {
            "pool1M_mean_delta": round(sum(vals1) / len(vals1), 4) if vals1 else None,
            "pool10M_mean_delta": round(sum(vals10) / len(vals10), 4) if vals10 else None,
            "pool_interaction_pool10M_minus_pool1M": round((sum(vals10) / len(vals10)) - (sum(vals1) / len(vals1)), 4) if vals1 and vals10 else None,
            "all_cell_deltas": {f"{p['pool_name']}_seed{p['seed']}": p["memory_minus_dense"][c] for p in pairs},
        }
    return {"pairs": pairs, "by_metric": by_col}


def write_note(rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    def fmt(x):
        return "NA" if x is None else f"{x:.2f}" if isinstance(x, float) else str(x)
    lines = []
    lines.append("# research — Controlled pool/seed memory profile\n")
    lines.append("Evidence JSON: `experiments/archive/initial_model_studies/data/controlled_grid_profile.json`\n")
    lines.append("All runs use 1M word exposure, `lr_total_steps=98`, paired shared GPT-2 core initialization, identical consumed example order within each pool/seed dense-memory pair, official tokenizer/corpus, and `chck_1M` evaluation.\n")
    lines.append("## Scores\n")
    lines.append("| pool | seed | model | BLiMP | Supp | EWoK | Entity | COMPS | Reading eye | Reading SPR |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for pool in ["pool1M", "pool10M"]:
        for seed in [42, 43]:
            for short in ["dense", "memory"]:
                r = next(x for x in rows if x["pool_name"] == pool and x["seed"] == seed and x["short"] == short)
                s = r["scores"]
                lines.append(f"| {pool} | {seed} | {short} | {fmt(s.get('blimp_fast'))} | {fmt(s.get('supplement_fast'))} | {fmt(s.get('ewok_fast'))} | {fmt(s.get('entity_tracking_fast'))} | {fmt(s.get('comps'))} | {fmt(s.get('reading_eye_tracking'))} | {fmt(s.get('reading_self_paced'))} |")
    lines.append("\n## Memory minus dense deltas\n")
    lines.append("| pool | seed | BLiMP | Supp | EWoK | Entity | COMPS | Reading eye | Reading SPR |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for p in summary["pairs"]:
        d = p["memory_minus_dense"]
        lines.append(f"| {p['pool_name']} | {p['seed']} | {fmt(d.get('blimp_fast'))} | {fmt(d.get('supplement_fast'))} | {fmt(d.get('ewok_fast'))} | {fmt(d.get('entity_tracking_fast'))} | {fmt(d.get('comps'))} | {fmt(d.get('reading_eye_tracking'))} | {fmt(d.get('reading_self_paced'))} |")
    lines.append("\n## Mean deltas and pool interaction\n")
    lines.append("| metric | mean delta pool1M | mean delta pool10M | pool10M minus pool1M |")
    lines.append("|---|---:|---:|---:|")
    for c in COLS:
        m = summary["by_metric"][c]
        lines.append(f"| {c} | {fmt(m['pool1M_mean_delta'])} | {fmt(m['pool10M_mean_delta'])} | {fmt(m['pool_interaction_pool10M_minus_pool1M'])} |")
    lines.append("\n## Immediate scientific reading\n")
    ent = summary["by_metric"]["entity_tracking_fast"]
    ewok = summary["by_metric"]["ewok_fast"]
    read_eye = summary["by_metric"]["reading_eye_tracking"]
    lines.append(f"Entity memory effect: pool1M mean {fmt(ent['pool1M_mean_delta'])}, pool10M mean {fmt(ent['pool10M_mean_delta'])}, interaction {fmt(ent['pool_interaction_pool10M_minus_pool1M'])}.\n")
    lines.append(f"EWoK memory effect: pool1M mean {fmt(ewok['pool1M_mean_delta'])}, pool10M mean {fmt(ewok['pool10M_mean_delta'])}, interaction {fmt(ewok['pool_interaction_pool10M_minus_pool1M'])}.\n")
    lines.append(f"Reading-eye memory effect: pool1M mean {fmt(read_eye['pool1M_mean_delta'])}, pool10M mean {fmt(read_eye['pool10M_mean_delta'])}, interaction {fmt(read_eye['pool_interaction_pool10M_minus_pool1M'])}.\n")
    lines.append("Next work: train/profile `memory_nopersist_causal` in the cell whose memory advantage is scientifically most informative, so the surviving effect can be separated from extra per-token capacity.\n")
    OUT_NOTE.parent.mkdir(parents=True, exist_ok=True)
    OUT_NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    HF_MODULES_CACHE.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["HF_MODULES_CACHE"] = str(HF_MODULES_CACHE.resolve())
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    rows = [profile_cell(cell, env) for cell in GRID]
    summary = summarize(rows)
    payload = {"rows": rows, "summary": summary}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_note(rows, summary)
    print("WROTE", OUT_JSON)
    print("WROTE", OUT_NOTE)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
