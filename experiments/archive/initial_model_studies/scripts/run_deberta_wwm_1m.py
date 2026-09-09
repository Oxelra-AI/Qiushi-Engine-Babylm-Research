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
OUT_JSON = ROOT / "data/debertav2_wwm_1m_profile.json"
OUT_TRAIN_JSON = ROOT / "data/debertav2_wwm_1m_training_summary.json"
OUT_NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/debertav2_wwm_1m_profile.md')
LOG = (ROOT.parents[2] / 'research/notes/initial_model_studies/debertav2_wwm_1m_train_and_profile.log')

SEED_SPECS = [
    {
        "seed": 42,
        "extra_init_seed": 456,
        "train_rng_seed": 789,
        "run_id": "babylm_masked_debertav2_wwm_seed42_1M",
        "baseline_run_id": "babylm_masked_wwm_pos512_1M",
        "baseline_profile": ROOT / "data/masked_1m_grid_pos512_profile.json",
    },
    {
        "seed": 43,
        "extra_init_seed": 457,
        "train_rng_seed": 790,
        "run_id": "babylm_masked_debertav2_wwm_seed43_1M",
        "baseline_run_id": "babylm_masked_wwm_pos512_seed43_1M",
        "baseline_profile": ROOT / "data/masked_1m_seed43_profile.json",
    },
]

TASKS = [
    ("blimp_fast", "blimp", "evaluation_data/fast_eval/blimp_fast"),
    ("supplement_fast", "blimp", "evaluation_data/fast_eval/supplement_fast"),
    ("ewok_fast", "ewok", "evaluation_data/fast_eval/evaluation_data/fast_eval/ewok_fast"),
    ("entity_tracking_fast", "entity_tracking", "evaluation_data/fast_eval/entity_tracking_fast"),
    ("comps", "comps", "evaluation_data/full_eval/comps"),
]
COLS = ["blimp_fast", "supplement_fast", "ewok_fast", "entity_tracking_fast", "comps", "reading_eye_tracking", "reading_self_paced"]
COMMON = [
    "--model_type", "deberta_v2",
    "--hidden_size", "240",
    "--n_layer", "8",
    "--n_head", "6",
    "--ffn_mult", "4",
    "--position_buckets", "256",
    "--max_relative_positions", "256",
    "--deberta_relative_attention", "true",
    "--deberta_pos_att_type", "p2c,c2p",
    "--max_word_exposure", "1000000",
    "--example_pool_words", "10000000",
    "--checkpoint_words", "1000000",
    "--words_per_example", "160",
    "--tokenizer_label", "baseline16k",
    "--tokenization_summary_limit", "0",
    "--mask_mode", "wwm",
    "--mask_prob", "0.15",
    "--seq_length", "256",
    "--max_seq_length", "256",
    "--max_position_embeddings", "512",
    "--batch_size", "64",
    "--lr_total_steps", "98",
    "--learning_rate", "0.001",
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
    os.environ.update({k: env[k] for k in ["HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_MODULES_CACHE", "TOKENIZERS_PARALLELISM"] if k in env})
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


def read_avg(report: pathlib.Path) -> float:
    txt = report.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"AVERAGE ACCURACY\s*\n([0-9.\-]+)", txt)
    if not m:
        raise RuntimeError(f"Could not parse {report}\n{txt[:500]}")
    return float(m.group(1))


def read_reading(report: pathlib.Path) -> dict[str, float]:
    txt = report.read_text(encoding="utf-8", errors="replace")
    out = {}
    for label, key in [("EYE TRACKING SCORE", "reading_eye_tracking"), ("SELF-PACED READING SCORE", "reading_self_paced")]:
        m = re.search(re.escape(label) + r":\s*([0-9.\-]+)", txt)
        if not m:
            raise RuntimeError(f"Could not parse {label} from {report}\n{txt}")
        out[key] = float(m.group(1))
    return out


def baseline_scores(profile_path: pathlib.Path, baseline_run_id: str) -> dict[str, float]:
    data = load_json(profile_path)
    for row in data["profile_rows"]:
        if row["run_id"] == baseline_run_id:
            return row["scores"]
    raise KeyError((profile_path, baseline_run_id))


def train_and_check(spec: dict, env: dict[str, str], logf) -> dict:
    run_id = spec["run_id"]
    run_dir = RUN_ROOT / run_id
    cmd = [
        sys.executable, str(TRAIN),
        "--output_dir", str(run_dir),
        "--seed", str(spec["seed"]), "--extra_init_seed", str(spec["extra_init_seed"]), "--train_rng_seed", str(spec["train_rng_seed"]),
        *COMMON,
    ]
    metrics_path = run_dir / "scientific_metrics.json"
    if metrics_path.exists() and (run_dir / "hf_model" / "config.json").exists() and (run_dir / "hf_model" / "chck_1M" / "config.json").exists():
        print(f"REUSE_COMPLETED_TRAINING {run_id}", flush=True)
        if logf:
            logf.write(f"\nREUSE_COMPLETED_TRAINING {run_id}\n"); logf.flush()
    else:
        run_training(cmd, output_dir=run_dir, timeout=2400, env=env, logf=logf)
    checks = []
    for rel in ["hf_model", "hf_model/chck_1M"]:
        p = run_dir / rel
        tok = AutoTokenizer.from_pretrained(p)
        model = AutoModelForMaskedLM.from_pretrained(p)
        assert model.__class__.__name__ == "DebertaV2ForMaskedLM", model.__class__.__name__
        checks.append({
            "rel": rel,
            "ok": True,
            "model_class": model.__class__.__name__,
            "tokenizer_class": tok.__class__.__name__,
            "parameter_count": sum(x.numel() for x in model.parameters()),
            "input_embedding_params": model.get_input_embeddings().weight.numel(),
            "hidden_size": int(model.config.hidden_size),
            "layers": int(model.config.num_hidden_layers),
            "heads": int(model.config.num_attention_heads),
            "intermediate_size": int(model.config.intermediate_size),
            "relative_attention": bool(getattr(model.config, "relative_attention", False)),
            "pos_att_type": list(getattr(model.config, "pos_att_type", [])),
            "position_buckets": int(getattr(model.config, "position_buckets", 0)),
            "max_relative_positions": int(getattr(model.config, "max_relative_positions", 0)),
            "max_position_embeddings": int(model.config.max_position_embeddings),
        })
    metrics = load_json(run_dir / "scientific_metrics.json")
    manifest = load_json(run_dir / "example_order_manifest.json")
    baseline_manifest = load_json(RUN_ROOT / spec["baseline_run_id"] / "example_order_manifest.json")
    assert metrics["word_exposure"] == 1_000_000
    assert metrics["model_type"] == "deberta_v2"
    assert metrics["model_family"] == "DebertaV2ForMaskedLM"
    assert metrics["parameter_count"] == 10733104
    assert metrics["hidden_size"] == 240 and metrics["n_layer"] == 8 and metrics["n_head"] == 6
    assert metrics["mask_mode"] == "wwm"
    assert metrics["max_seq_length"] == 256 and metrics["max_position_embeddings"] >= 512
    assert manifest["consumed_example_ids_in_order"] == baseline_manifest["consumed_example_ids_in_order"]
    assert manifest["source_words_consumed"] == baseline_manifest["source_words_consumed"]
    coupling = load_json(run_dir / "tokenization_coupling_summary.json")
    return {"run_id": run_id, "seed": spec["seed"], "baseline_run_id": spec["baseline_run_id"], "metrics": metrics, "manifest": manifest, "load_checks": checks, "coupling": coupling}


def profile(run_id: str, env: dict[str, str], logf) -> dict:
    run_dir = RUN_ROOT / run_id
    model_path = (run_dir / "hf_model").resolve()
    outdir = (run_dir / "eval_results_debertav2_wwm_1m").resolve()
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
        assert report.exists(), report
        scores[task_name] = read_avg(report)
        reports[task_name] = str(report)
    run([
        sys.executable, "-m", "evaluation_pipeline.reading.run",
        "--model_path_or_name", str(model_path), "--backend", "mlm",
        "--data_path", "evaluation_data/fast_eval/reading/reading_data.csv",
        "--revision_name", "chck_1M", "--output_dir", str(outdir),
    ], env, cwd=STRICT_DIR, logf=logf)
    rreport = outdir / "hf_model" / "chck_1M" / "zero_shot" / "mlm" / "reading" / "report.txt"
    assert rreport.exists(), rreport
    scores.update(read_reading(rreport))
    reports["reading"] = str(rreport)
    return {"run_id": run_id, "scores": scores, "reports": reports}


def diff(a: float, b: float) -> float:
    return round(a - b, 4)


def fmt(x):
    return f"{x:.2f}" if isinstance(x, float) else str(x)


def main() -> None:
    env = setup_env()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    train_rows = []
    profile_rows = []
    with LOG.open("a", encoding="utf-8") as logf:
        for spec in SEED_SPECS:
            train_rows.append(train_and_check(spec, env, logf))
        for row in train_rows:
            profile_rows.append(profile(row["run_id"], env, logf))

    train_summary = []
    rows = []
    deltas = {}
    for spec, tr, pr in zip(SEED_SPECS, train_rows, profile_rows):
        m = tr["metrics"]
        bscore = baseline_scores(spec["baseline_profile"], spec["baseline_run_id"])
        d = {c: diff(pr["scores"][c], bscore[c]) for c in COLS}
        deltas[str(spec["seed"])] = d
        train_summary.append({
            "run_id": tr["run_id"], "seed": spec["seed"], "baseline_run_id": spec["baseline_run_id"],
            "parameter_count": m["parameter_count"], "embedding_parameter_count": m["embedding_parameter_count"],
            "non_embedding_parameter_count": m["non_embedding_parameter_count"], "vocab_size": m["vocab_size"],
            "loss_first": m["loss_first"], "loss_last": m["loss_last"],
            "masked_tokens_per_whitespace_word": m["masked_tokens_per_whitespace_word"],
            "word_exposure": m["word_exposure"], "steps": m["actual_training_steps"],
            "model_family": m["model_family"], "model_type": m["model_type"],
            "hidden_size": m["hidden_size"], "n_layer": m["n_layer"], "n_head": m["n_head"],
            "intermediate_size": m["intermediate_size"], "position_buckets": m["position_buckets"],
            "max_relative_positions": m["max_relative_positions"], "deberta_pos_att_type": m["deberta_pos_att_type"],
            "source_words": tr["manifest"]["source_words_consumed"], "first12_examples": tr["manifest"]["consumed_example_ids_in_order"][:12],
            "load_checks": tr["load_checks"], "tokenization_coupling_summary": tr["coupling"],
        })
        rows.append({
            "run_id": tr["run_id"], "seed": spec["seed"], "baseline_run_id": spec["baseline_run_id"],
            "scores": pr["scores"], "baseline_scores_bert_wwm": bscore, "debertav2_minus_bert_wwm": d, "reports": pr["reports"],
        })
    mean_delta = {c: round(sum(deltas[str(s["seed"])][c] for s in SEED_SPECS) / len(SEED_SPECS), 4) for c in COLS}
    payload = {"train_summary": train_summary, "profile_rows": rows, "debertav2_minus_bert_wwm_by_seed": deltas, "mean_debertav2_minus_bert_wwm": mean_delta}
    OUT_TRAIN_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_TRAIN_JSON.write_text(json.dumps(train_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# research — WWM fixed base: parameter-matched DeBERTa-v2 vs BERT at 1M",
        "",
        f"Evidence JSON: `{OUT_JSON}`",
        "",
        "Only the architecture changes. The raw official-corpus examples/order/source mix, baseline 16k tokenizer, WWM objective, fixed length 256, optimizer schedule, word exposure, and seeds are matched to each seed's existing BERT-WWM baseline. The DeBERTa-v2 configuration is hidden 240, 8 layers, 6 heads, intermediate 960, relative attention p2c/c2p, total parameters 10,733,104 (+0.055% vs BERT).",
        "",
        "| seed | arch | BLiMP | Supp | EWoK | Entity | COMPS | Read eye | Read SPR | loss_last | params | emb params | non-emb params |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row, tr in zip(rows, train_summary):
        s = row["scores"]
        lines.append(f"| {row['seed']} | DeBERTaV2 | {fmt(s['blimp_fast'])} | {fmt(s['supplement_fast'])} | {fmt(s['ewok_fast'])} | {fmt(s['entity_tracking_fast'])} | {fmt(s['comps'])} | {fmt(s['reading_eye_tracking'])} | {fmt(s['reading_self_paced'])} | {fmt(tr['loss_last'])} | {tr['parameter_count']} | {tr['embedding_parameter_count']} | {tr['non_embedding_parameter_count']} |")
    lines += ["", "## DeBERTaV2 minus matched BERT-WWM baseline", "", "| seed | BLiMP | Supp | EWoK | Entity | COMPS | Read eye | Read SPR |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for spec in SEED_SPECS:
        d = deltas[str(spec["seed"])]
        lines.append(f"| {spec['seed']} | {fmt(d['blimp_fast'])} | {fmt(d['supplement_fast'])} | {fmt(d['ewok_fast'])} | {fmt(d['entity_tracking_fast'])} | {fmt(d['comps'])} | {fmt(d['reading_eye_tracking'])} | {fmt(d['reading_self_paced'])} |")
    lines += ["", "## Mean score delta", "", "| BLiMP | Supp | EWoK | Entity | COMPS | Read eye | Read SPR |", "|---:|---:|---:|---:|---:|---:|---:|", f"| {fmt(mean_delta['blimp_fast'])} | {fmt(mean_delta['supplement_fast'])} | {fmt(mean_delta['ewok_fast'])} | {fmt(mean_delta['entity_tracking_fast'])} | {fmt(mean_delta['comps'])} | {fmt(mean_delta['reading_eye_tracking'])} | {fmt(mean_delta['reading_self_paced'])} |"]
    OUT_NOTE.parent.mkdir(parents=True, exist_ok=True)
    OUT_NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"mean_debertav2_minus_bert_wwm": mean_delta, "by_seed": deltas, "out": str(OUT_JSON)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
