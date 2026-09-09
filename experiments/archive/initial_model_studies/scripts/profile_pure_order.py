#!/usr/bin/env python3
"""Profile the research pure-order dense checkpoints and compare vs identical-data random full-pool baselines."""
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
JSON = STUDY / "data/controlled_grid_profile.json"
OUT_JSON = STUDY / "data/pure_order_profile.json"
OUT_NOTE = (STUDY.parents[2] / 'research/notes/initial_model_studies/pure_order_profile_interpretation.md')

TASKS = [
    ("blimp_fast", "blimp", "evaluation_data/fast_eval/blimp_fast"),
    ("supplement_fast", "blimp", "evaluation_data/fast_eval/supplement_fast"),
    ("ewok_fast", "ewok", "evaluation_data/fast_eval/evaluation_data/fast_eval/ewok_fast"),
    ("entity_tracking_fast", "entity_tracking", "evaluation_data/fast_eval/entity_tracking_fast"),
    ("comps", "comps", "evaluation_data/full_eval/comps"),
]
COLS = ["blimp_fast", "supplement_fast", "ewok_fast", "entity_tracking_fast", "comps", "reading_eye_tracking", "reading_self_paced"]

ORDER_RUNS = []
for seed in [42, 43]:
    for mode in ["source_stage", "readability_interleave"]:
        ORDER_RUNS.append({
            "seed": seed,
            "order_mode": mode,
            "run_id": f"babylm_step30_order_{mode}_pool10M_seed{seed}_1M",
            "baseline_run_id": f"babylm_step26_ctrl_dense_pool10M_seed{seed}_1M",
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


def profile_run(run_id: str, env: dict[str, str]) -> dict[str, Any]:
    run_dir = AI_RUNS / run_id
    model_path = (run_dir / "hf_model").resolve()
    outdir = (run_dir / "eval_results_pure_order").resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    metrics = load_json(run_dir / "scientific_metrics.json")
    scores: dict[str, float | None] = {}
    reports: dict[str, str] = {}
    for task_name, task, data_path in TASKS:
        dp = STRICT_DIR / data_path
        if not dp.exists():
            scores[task_name] = None
            reports[task_name] = f"MISSING_DATA:{dp}"
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
        scores[task_name] = read_avg(report)
        reports[task_name] = str(report)
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
        scores.update({f"reading_{k}": v for k, v in reading.items()})
        reports["reading"] = str(rreport)
    else:
        scores.update({"reading_eye_tracking": None, "reading_self_paced": None})
        reports["reading"] = f"MISSING_DATA:{reading_data}"
    return {
        "run_id": run_id,
        "loss_first": metrics.get("loss_first"),
        "loss_last": metrics.get("loss_last"),
        "word_exposure": metrics.get("word_exposure"),
        "example_pool_words_actual": metrics.get("example_pool_words_actual"),
        "scores": scores,
        "reports": reports,
    }


def baseline_scores_map() -> dict[str, dict[str, float | None]]:
    data = load_json(JSON)
    out = {}
    for row in data["rows"]:
        out[row["run_id"]] = row["scores"]
    return out


def delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return round(a - b, 4)


def main() -> None:
    HF_MODULES_CACHE.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["HF_MODULES_CACHE"] = str(HF_MODULES_CACHE.resolve())
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    baselines = baseline_scores_map()

    rows = []
    for spec in ORDER_RUNS:
        prof = profile_run(spec["run_id"], env)
        base = baselines.get(spec["baseline_run_id"])
        prof.update({
            "seed": spec["seed"],
            "order_mode": spec["order_mode"],
            "baseline_run_id": spec["baseline_run_id"],
            "baseline_scores": base,
            "order_minus_random": {c: delta(prof["scores"].get(c), base.get(c) if base else None) for c in COLS},
        })
        rows.append(prof)

    by_mode: dict[str, Any] = {}
    for mode in ["source_stage", "readability_interleave"]:
        mode_rows = [r for r in rows if r["order_mode"] == mode]
        by_mode[mode] = {}
        for c in COLS:
            vals = [r["order_minus_random"][c] for r in mode_rows if r["order_minus_random"][c] is not None]
            by_mode[mode][c] = {
                "mean_delta": round(sum(vals) / len(vals), 4) if vals else None,
                "per_seed": {r["seed"]: r["order_minus_random"][c] for r in mode_rows},
            }
    payload = {"rows": rows, "by_mode": by_mode}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def fmt(x):
        return "NA" if x is None else (f"{x:.2f}" if isinstance(x, float) else str(x))

    lines = []
    lines.append("# research — Pure-order dense profile vs identical-data random full-pool baseline\n")
    lines.append("Evidence JSON: `experiments/archive/initial_model_studies/data/pure_order_profile.json`\n")
    lines.append("Each order run consumes exactly the same selected example set and source mix as its random full-pool dense baseline (research/27), changing only training order. 1M exposure, `lr_total_steps=98`, shared-core init, `chck_1M` eval.\n")
    lines.append("## Scores (order runs)\n")
    lines.append("| seed | order | BLiMP | Supp | EWoK | Entity | COMPS | Read eye | Read SPR | loss_last |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        s = r["scores"]
        lines.append(f"| {r['seed']} | {r['order_mode']} | {fmt(s.get('blimp_fast'))} | {fmt(s.get('supplement_fast'))} | {fmt(s.get('ewok_fast'))} | {fmt(s.get('entity_tracking_fast'))} | {fmt(s.get('comps'))} | {fmt(s.get('reading_eye_tracking'))} | {fmt(s.get('reading_self_paced'))} | {fmt(r.get('loss_last'))} |")
    lines.append("\n## Random full-pool dense baselines (from research)\n")
    lines.append("| seed | BLiMP | Supp | EWoK | Entity | COMPS | Read eye | Read SPR |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for seed in [42, 43]:
        b = baselines.get(f"babylm_step26_ctrl_dense_pool10M_seed{seed}_1M", {})
        lines.append(f"| {seed} | {fmt(b.get('blimp_fast'))} | {fmt(b.get('supplement_fast'))} | {fmt(b.get('ewok_fast'))} | {fmt(b.get('entity_tracking_fast'))} | {fmt(b.get('comps'))} | {fmt(b.get('reading_eye_tracking'))} | {fmt(b.get('reading_self_paced'))} |")
    lines.append("\n## Order minus random deltas\n")
    lines.append("| seed | order | BLiMP | Supp | EWoK | Entity | COMPS | Read eye | Read SPR |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        d = r["order_minus_random"]
        lines.append(f"| {r['seed']} | {r['order_mode']} | {fmt(d.get('blimp_fast'))} | {fmt(d.get('supplement_fast'))} | {fmt(d.get('ewok_fast'))} | {fmt(d.get('entity_tracking_fast'))} | {fmt(d.get('comps'))} | {fmt(d.get('reading_eye_tracking'))} | {fmt(d.get('reading_self_paced'))} |")
    lines.append("\n## Mean order-minus-random by mode\n")
    lines.append("| order | BLiMP | Supp | EWoK | Entity | COMPS | Read eye | Read SPR |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for mode in ["source_stage", "readability_interleave"]:
        m = by_mode[mode]
        lines.append(f"| {mode} | {fmt(m['blimp_fast']['mean_delta'])} | {fmt(m['supplement_fast']['mean_delta'])} | {fmt(m['ewok_fast']['mean_delta'])} | {fmt(m['entity_tracking_fast']['mean_delta'])} | {fmt(m['comps']['mean_delta'])} | {fmt(m['reading_eye_tracking']['mean_delta'])} | {fmt(m['reading_self_paced']['mean_delta'])} |")
    lines.append("\n## Note\n")
    lines.append("Single-pass source_stage places hard Gutenberg/SimpleWiki text at the end of training, so a higher loss_last and any BLiMP/Supplement drop may be an end-of-pass artifact rather than a curriculum failure. Judge order effects by seed-stable profile deltas, and if source_stage looks penalized mainly on syntax, test a multi-epoch or repeated-interleave variant that removes the end-of-pass artifact.\n")
    OUT_NOTE.parent.mkdir(parents=True, exist_ok=True)
    OUT_NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("WROTE", OUT_JSON)
    print("WROTE", OUT_NOTE)
    print(json.dumps(by_mode, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
