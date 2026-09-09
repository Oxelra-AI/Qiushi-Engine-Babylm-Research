#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess

from training_process import run_training
import sys

from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
TRAIN = ROOT / "training/scripts/babylm_masked_train.py"
STRICT_DIR = ROOT / "repos/babylm-eval/strict"
RUN_ROOT = ROOT / "training/runs"
OUT_JSON = ROOT / "data/masked_1m_grid_profile.json"
OUT_TRAIN_JSON = ROOT / "data/masked_1m_training_summary.json"
OUT_NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/42_masked_1m_token_vs_wwm_profile.md')
LOG = (ROOT.parents[2] / 'research/notes/initial_model_studies/masked_1m_train_and_profile.log')

RUN_SPECS = [("token", "babylm_masked_token_1M"), ("wwm", "babylm_masked_wwm_1M")]
TASKS = [
    ("blimp_fast", "blimp", "evaluation_data/fast_eval/blimp_fast"),
    ("supplement_fast", "blimp", "evaluation_data/fast_eval/supplement_fast"),
    ("ewok_fast", "ewok", "evaluation_data/fast_eval/evaluation_data/fast_eval/ewok_fast"),
    ("entity_tracking_fast", "entity_tracking", "evaluation_data/fast_eval/entity_tracking_fast"),
    ("comps", "comps", "evaluation_data/full_eval/comps"),
]
COLS = ["blimp_fast", "supplement_fast", "ewok_fast", "entity_tracking_fast", "comps", "reading_eye_tracking", "reading_self_paced"]
COMMON = [
    "--max_word_exposure", "1000000",
    "--example_pool_words", "10000000",
    "--checkpoint_words", "1000000",
    "--words_per_example", "160",
    "--mask_prob", "0.15",
    "--seq_length", "128",
    "--max_seq_length", "128",
    "--batch_size", "64",
    "--lr_total_steps", "98",
    "--hidden_size", "256",
    "--n_layer", "8",
    "--n_head", "8",
    "--learning_rate", "0.001",
    "--seed", "42",
    "--extra_init_seed", "456",
    "--train_rng_seed", "789",
    "--log_every", "50",
]


def setup_env() -> dict[str, str]:
    env = os.environ.copy()
    hf_home = ROOT / "training/hf_home"
    env["HF_HOME"] = str(hf_home.resolve())
    env["HF_HUB_CACHE"] = str((hf_home / "hub").resolve())
    env["TRANSFORMERS_CACHE"] = str((hf_home / "transformers").resolve())
    env["HF_MODULES_CACHE"] = str((ROOT / "training/hf_modules_cache").resolve())
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    for key in ["HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_MODULES_CACHE"]:
        pathlib.Path(env[key]).mkdir(parents=True, exist_ok=True)
    return env


def run(cmd: list[str], env: dict[str, str], cwd: pathlib.Path | None = None, logf=None) -> subprocess.CompletedProcess:
    line = "$ " + " ".join(cmd)
    print(line, flush=True)
    if logf:
        logf.write("\n" + line + "\n"); logf.flush()
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=str(cwd) if cwd else None)
    print(p.stdout[-6000:], flush=True)
    if logf:
        logf.write(p.stdout + f"\n[returncode={p.returncode}]\n"); logf.flush()
    if p.returncode != 0:
        raise RuntimeError(f"command failed with {p.returncode}: {' '.join(cmd)}\n{p.stdout[-12000:]}")
    return p


def load_json(path: pathlib.Path):
    return json.loads(path.read_text(encoding="utf-8"))


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
    out = {}
    for label, key in [("EYE TRACKING SCORE", "eye_tracking"), ("SELF-PACED READING SCORE", "self_paced")]:
        m = re.search(re.escape(label) + r":\s*([0-9.\-]+)", txt)
        out[key] = float(m.group(1)) if m else None
    return out


def train_and_check(mask_mode: str, run_id: str, env: dict[str, str], logf) -> dict:
    run_dir = RUN_ROOT / run_id
    cmd = [
        sys.executable, str(TRAIN),
        "--output_dir", str(run_dir), "--mask_mode", mask_mode, *COMMON,
    ]
    run_training(cmd, output_dir=run_dir, timeout=1800, env=env, logf=logf)
    checks = []
    for rel in ["hf_model", "hf_model/chck_1M"]:
        p = run_dir / rel
        tok = AutoTokenizer.from_pretrained(p)
        model = AutoModelForMaskedLM.from_pretrained(p)
        checks.append({"rel": rel, "ok": True, "model_class": model.__class__.__name__, "tokenizer_class": tok.__class__.__name__, "params": sum(x.numel() for x in model.parameters())})
    metrics = load_json(run_dir / "scientific_metrics.json")
    manifest = load_json(run_dir / "example_order_manifest.json")
    assert metrics["word_exposure"] == 1_000_000
    assert metrics["mask_mode"] == mask_mode
    assert metrics["backend"] == "mlm"
    return {"run_id": run_id, "mask_mode": mask_mode, "metrics": metrics, "manifest": manifest, "load_checks": checks}


def profile(run_id: str, env: dict[str, str], logf) -> dict:
    run_dir = RUN_ROOT / run_id
    model_path = (run_dir / "hf_model").resolve()
    outdir = (run_dir / "eval_results_masked_1m").resolve()
    scores = {}
    reports = {}
    for task_name, task, data_path in TASKS:
        run([
            sys.executable, "-m", "evaluation_pipeline.sentence_zero_shot.run",
            "--model_path_or_name", str(model_path), "--backend", "mlm", "--task", task,
            "--data_path", data_path, "--save_predictions", "--revision_name", "chck_1M",
            "--batch_size", "64", "--output_dir", str(outdir),
        ], env, cwd=STRICT_DIR, logf=logf)
        report = outdir / "hf_model" / "chck_1M" / "zero_shot" / "mlm" / task / task_name / "best_temperature_report.txt"
        scores[task_name] = read_avg(report)
        reports[task_name] = str(report)
    run([
        sys.executable, "-m", "evaluation_pipeline.reading.run",
        "--model_path_or_name", str(model_path), "--backend", "mlm",
        "--data_path", "evaluation_data/fast_eval/reading/reading_data.csv",
        "--revision_name", "chck_1M", "--output_dir", str(outdir),
    ], env, cwd=STRICT_DIR, logf=logf)
    rreport = outdir / "hf_model" / "chck_1M" / "zero_shot" / "mlm" / "reading" / "report.txt"
    reading = read_reading(rreport)
    scores.update({f"reading_{k}": v for k, v in reading.items()})
    reports["reading"] = str(rreport)
    return {"run_id": run_id, "scores": scores, "reports": reports}


def diff(a, b):
    if a is None or b is None:
        return None
    return round(a - b, 4)


def main() -> None:
    env = setup_env()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    train_rows = []
    profile_rows = []
    with LOG.open("a", encoding="utf-8") as logf:
        for mask_mode, run_id in RUN_SPECS:
            train_rows.append(train_and_check(mask_mode, run_id, env, logf))
        # assert identical data condition
        assert train_rows[0]["manifest"]["consumed_example_ids_in_order"] == train_rows[1]["manifest"]["consumed_example_ids_in_order"]
        assert train_rows[0]["manifest"]["source_words_consumed"] == train_rows[1]["manifest"]["source_words_consumed"]
        for _, run_id in RUN_SPECS:
            profile_rows.append(profile(run_id, env, logf))

    train_summary = []
    for r in train_rows:
        m = r["metrics"]
        train_summary.append({
            "run_id": r["run_id"], "mask_mode": r["mask_mode"], "parameter_count": m["parameter_count"],
            "loss_first": m["loss_first"], "loss_last": m["loss_last"], "word_exposure": m["word_exposure"],
            "steps": m["actual_training_steps"], "source_words": r["manifest"]["source_words_consumed"],
            "first12_examples": r["manifest"]["consumed_example_ids_in_order"][:12], "load_checks": r["load_checks"],
        })
    OUT_TRAIN_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_TRAIN_JSON.write_text(json.dumps(train_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    row_by_mode = {r["run_id"].split("masked_")[1].split("_1M")[0]: r for r in profile_rows}
    token = row_by_mode["token"]
    wwm = row_by_mode["wwm"]
    wwm_minus_token = {c: diff(wwm["scores"].get(c), token["scores"].get(c)) for c in COLS}
    payload = {"train_summary": train_summary, "profile_rows": profile_rows, "wwm_minus_token": wwm_minus_token}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def fmt(x):
        return "NA" if x is None else (f"{x:.2f}" if isinstance(x, float) else str(x))
    lines = [
        "# research — 1M masked-LM token masking vs whole-word masking",
        "",
        f"Evidence JSON: `{OUT_JSON}`",
        "",
        "Both runs use the same official-corpus selected examples/order/source mix, same baseline tokenizer, same BERT-like masked model size, same optimizer schedule, same seeds, 1M whitespace-word exposure, and official `mlm` backend at `chck_1M`.",
        "",
        "| mode | BLiMP | Supp | EWoK | Entity | COMPS | Read eye | Read SPR | loss_last |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    mode_by_run = {run_id: mode for mode, run_id in RUN_SPECS}
    for r in profile_rows:
        s = r["scores"]
        mode = mode_by_run[r["run_id"]]
        loss = next(t["loss_last"] for t in train_summary if t["run_id"] == r["run_id"])
        lines.append(f"| {mode} | {fmt(s.get('blimp_fast'))} | {fmt(s.get('supplement_fast'))} | {fmt(s.get('ewok_fast'))} | {fmt(s.get('entity_tracking_fast'))} | {fmt(s.get('comps'))} | {fmt(s.get('reading_eye_tracking'))} | {fmt(s.get('reading_self_paced'))} | {fmt(loss)} |")
    lines += ["", "## WWM minus token", "", "| BLiMP | Supp | EWoK | Entity | COMPS | Read eye | Read SPR |", "|---:|---:|---:|---:|---:|---:|---:|", f"| {fmt(wwm_minus_token['blimp_fast'])} | {fmt(wwm_minus_token['supplement_fast'])} | {fmt(wwm_minus_token['ewok_fast'])} | {fmt(wwm_minus_token['entity_tracking_fast'])} | {fmt(wwm_minus_token['comps'])} | {fmt(wwm_minus_token['reading_eye_tracking'])} | {fmt(wwm_minus_token['reading_self_paced'])} |"]
    OUT_NOTE.parent.mkdir(parents=True, exist_ok=True)
    OUT_NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"wwm_minus_token": wwm_minus_token, "out": str(OUT_JSON)}, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
