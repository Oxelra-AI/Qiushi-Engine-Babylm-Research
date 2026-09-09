#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pathlib
import re
import shutil
import subprocess

from training_process import run_training
import sys

from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
TRAIN = ROOT / "training/scripts/babylm_masked_train.py"
STRICT_DIR = ROOT / "repos/babylm-eval/strict"
RUN_ROOT = ROOT / "training/runs"
MIX_META = ROOT / "data/mixture_revision_61/mixture_materialization_all_seeds.json"
OUT_JSON = ROOT / "data/mixture_adjacent_vs_shuffled_1m_profile.json"
OUT_TRAIN_JSON = ROOT / "data/mixture_adjacent_vs_shuffled_1m_training_summary.json"
OUT_NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/mixture_adjacent_vs_shuffled_1m_profile.md')
LOG = (ROOT.parents[2] / 'research/notes/initial_model_studies/mixture_adjacent_vs_shuffled_1m_train_and_profile.log')

SEED_SPECS = {
    42: {"extra_init_seed": 456, "train_rng_seed": 789},
    43: {"extra_init_seed": 457, "train_rng_seed": 790},
}
TASKS = [
    ("blimp_fast", "blimp", "evaluation_data/fast_eval/blimp_fast"),
    ("supplement_fast", "blimp", "evaluation_data/fast_eval/supplement_fast"),
    ("ewok_fast", "ewok", "evaluation_data/fast_eval/evaluation_data/fast_eval/ewok_fast"),
    ("entity_tracking_fast", "entity_tracking", "evaluation_data/fast_eval/entity_tracking_fast"),
    ("comps", "comps", "evaluation_data/full_eval/comps"),
]
COLS = ["blimp_fast", "supplement_fast", "ewok_fast", "entity_tracking_fast", "comps", "reading_eye_tracking", "reading_self_paced"]
COMMON_BASE = [
    "--tokenizer_label", "baseline16k",
    "--tokenization_summary_limit", "0",
    "--mask_mode", "wwm",
    "--mask_prob", "0.15",
    "--seq_length", "256",
    "--max_seq_length", "256",
    "--max_position_embeddings", "512",
    "--lr_total_steps", "98",
    "--hidden_size", "256",
    "--n_layer", "8",
    "--n_head", "8",
    "--ffn_mult", "4",
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


def run(cmd: list[str], env: dict[str, str], cwd: pathlib.Path | None = None, logf=None) -> None:
    line = "$ " + " ".join(cmd)
    print(line, flush=True)
    if logf:
        logf.write("\n" + line + "\n"); logf.flush()
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=str(cwd) if cwd else None)
    print(p.stdout[-5000:], flush=True)
    if logf:
        logf.write(p.stdout + f"\n[returncode={p.returncode}]\n"); logf.flush()
    if p.returncode != 0:
        raise RuntimeError(f"command failed with {p.returncode}: {' '.join(cmd)}\n{p.stdout[-10000:]}")


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


def run_id_for(arm: str, seed: int) -> str:
    return f"babylm_step62_mixture_{arm}_wwm_seed{seed}_1M"


def arm_path(seed_meta: dict, arm: str) -> str:
    return seed_meta[f"mixture_{arm}_path"]


def arm_label(arm: str) -> str:
    return f"official500k_GEMwikiAuto500k_mixture_{arm}"


def validate_materialization(meta: dict) -> None:
    for seed_str, seed_meta in meta["seeds"].items():
        v = seed_meta["validation"]
        required = [
            "identical_official_records",
            "source_pair_ids_same_order",
            "identical_target_pair_id_multiset",
            "identical_combined_example_count",
            "identical_total_words",
            "no_identity_target_adjacency_in_shuffled_pairs",
            "pair_adjacent_zero_over_256",
            "pair_shuffled_zero_over_256",
            "hard_validation_ok",
            "aggregate_opportunity_identical",
        ]
        for k in required:
            assert v[k], (seed_str, k, v)
        assert v["combined_kept_token_delta_total"] == 0, (seed_str, v)
        assert v["combined_word_group_delta_total"] == 0, (seed_str, v)
        assert seed_meta["actual_total_words"] == 1_000_000, seed_meta
        assert seed_meta["recommended_batch"]["actual_steps"] == 98, seed_meta["recommended_batch"]


def train_and_check(seed: int, arm: str, seed_meta: dict, env: dict[str, str], logf) -> dict:
    run_id = run_id_for(arm, seed)
    run_dir = RUN_ROOT / run_id
    actual_words = int(seed_meta["actual_total_words"])
    batch_size = int(seed_meta["recommended_batch"]["batch_size"])
    meta_path = seed_meta["materialization_meta_path"]
    cmd = [
        sys.executable, str(TRAIN),
        "--output_dir", str(run_dir),
        "--example_jsonl", arm_path(seed_meta, arm),
        "--example_jsonl_label", arm_label(arm),
        "--example_jsonl_meta", meta_path,
        "--seed", str(seed),
        "--extra_init_seed", str(SEED_SPECS[seed]["extra_init_seed"]),
        "--train_rng_seed", str(SEED_SPECS[seed]["train_rng_seed"]),
        "--max_word_exposure", str(actual_words),
        "--example_pool_words", str(actual_words),
        "--checkpoint_words", str(actual_words),
        "--batch_size", str(batch_size),
        *COMMON_BASE,
    ]
    complete = (run_dir / "scientific_metrics.json").exists() and (run_dir / "hf_model" / "chck_1M" / "config.json").exists()
    if complete:
        print(f"REUSE_COMPLETED_TRAINING {run_id}", flush=True)
        if logf:
            logf.write(f"\nREUSE_COMPLETED_TRAINING {run_id}\n"); logf.flush()
    else:
        if run_dir.exists():
            print(f"REMOVE_INCOMPLETE_RUN_DIR {run_id}", flush=True)
            if logf:
                logf.write(f"\nREMOVE_INCOMPLETE_RUN_DIR {run_id}\n"); logf.flush()
            shutil.rmtree(run_dir)
        run_training(cmd, output_dir=run_dir, timeout=2400, env=env, logf=logf)
    for rel in ["hf_model", "hf_model/chck_1M"]:
        AutoTokenizer.from_pretrained(run_dir / rel)
        AutoModelForMaskedLM.from_pretrained(run_dir / rel)
    metrics = load_json(run_dir / "scientific_metrics.json")
    coupling = load_json(run_dir / "tokenization_coupling_summary.json")
    assert metrics["data_source_type"] == "example_jsonl", metrics
    assert metrics["word_exposure"] == actual_words, metrics
    assert metrics["mask_mode"] == "wwm", metrics
    assert metrics["actual_training_steps"] == 98, metrics
    assert coupling["total_words_summarized"] == actual_words, coupling
    return {"run_id": run_id, "seed": seed, "arm": arm, "metrics": metrics, "coupling": coupling}


def profile(run_id: str, env: dict[str, str], logf) -> dict:
    run_dir = RUN_ROOT / run_id
    model_path = (run_dir / "hf_model").resolve()
    outdir = (run_dir / "eval_results_revision_62").resolve()
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
    meta = load_json(MIX_META)
    validate_materialization(meta)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    train_rows = {}
    profile_rows = {}
    with LOG.open("a", encoding="utf-8") as logf:
        for seed_str, seed_meta in meta["seeds"].items():
            seed = int(seed_str)
            for arm in ["adjacent", "shuffled"]:
                row = train_and_check(seed, arm, seed_meta, env, logf)
                train_rows[row["run_id"]] = row
        for run_id in list(train_rows.keys()):
            profile_rows[run_id] = profile(run_id, env, logf)

    per_seed_scores = {}
    deltas = {}
    for seed_str in meta["seeds"]:
        seed = int(seed_str)
        adj_id = run_id_for("adjacent", seed)
        shuf_id = run_id_for("shuffled", seed)
        adj = profile_rows[adj_id]["scores"]
        shuf = profile_rows[shuf_id]["scores"]
        per_seed_scores[str(seed)] = {"mixture_adjacent": adj, "mixture_shuffled": shuf}
        deltas[str(seed)] = {c: diff(adj[c], shuf[c]) for c in COLS}
    mean_delta = {c: round(sum(deltas[s][c] for s in deltas) / len(deltas), 4) for c in COLS}

    train_summary = []
    for run_id, row in train_rows.items():
        m = row["metrics"]
        c = row["coupling"]
        train_summary.append({
            "run_id": run_id,
            "seed": row["seed"],
            "arm": row["arm"],
            "parameter_count": m["parameter_count"],
            "embedding_parameter_count": m["embedding_parameter_count"],
            "loss_first": m["loss_first"],
            "loss_last": m["loss_last"],
            "masked_tokens_per_whitespace_word": m["masked_tokens_per_whitespace_word"],
            "word_exposure": m["word_exposure"],
            "steps": m["actual_training_steps"],
            "data_source_type": m["data_source_type"],
            "example_jsonl_label": m["example_jsonl_label"],
            "truncated_examples": c["truncated_examples_at_max_seq_length"],
            "tokens_lost_to_truncation": c["total_tokens_lost_to_truncation"],
            "kept_tokens_per_word": c["kept_tokens_per_whitespace_word"],
            "word_groups_per_word": c["word_groups_kept_per_whitespace_word"],
        })

    payload = {
        "mixture_materialization_meta": str(MIX_META),
        "mixture_materialization_validation_by_seed": {s: meta["seeds"][s]["validation"] for s in meta["seeds"]},
        "train_summary": train_summary,
        "per_seed_scores": per_seed_scores,
        "mixture_adjacent_minus_mixture_shuffled_by_seed": deltas,
        "mean_mixture_adjacent_minus_mixture_shuffled": mean_delta,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_TRAIN_JSON.write_text(json.dumps(train_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# research — 50/50 official+rewrite mixture: adjacent vs shuffled",
        "",
        f"Evidence JSON: `{OUT_JSON}`",
        "",
        "Both arms per seed share identical 500k official examples, identical 500k rewrite source/target multisets, identical slot order, total words, kept tokens, WWM groups, and expected mask opportunity. Only rewrite target adjacency differs. Base: BERT-WWM, baseline 16k tokenizer, fixed 256, AdamW, paired seeds 42/43.",
        "",
        "| seed | arm | BLiMP | Supp | EWoK | Entity | COMPS | Read eye | Read SPR |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed_str in meta["seeds"]:
        for arm, key in [("adjacent", "mixture_adjacent"), ("shuffled", "mixture_shuffled")]:
            s = per_seed_scores[seed_str][key]
            lines.append(f"| {seed_str} | {key} | {fmt(s['blimp_fast'])} | {fmt(s['supplement_fast'])} | {fmt(s['ewok_fast'])} | {fmt(s['entity_tracking_fast'])} | {fmt(s['comps'])} | {fmt(s['reading_eye_tracking'])} | {fmt(s['reading_self_paced'])} |")
    lines += ["", "## mixture-adjacent minus mixture-shuffled", "", "| seed | BLiMP | Supp | EWoK | Entity | COMPS | Read eye | Read SPR |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for seed_str in meta["seeds"]:
        d = deltas[seed_str]
        lines.append(f"| {seed_str} | {fmt(d['blimp_fast'])} | {fmt(d['supplement_fast'])} | {fmt(d['ewok_fast'])} | {fmt(d['entity_tracking_fast'])} | {fmt(d['comps'])} | {fmt(d['reading_eye_tracking'])} | {fmt(d['reading_self_paced'])} |")
    lines += ["", "## mean delta", "", "| BLiMP | Supp | EWoK | Entity | COMPS | Read eye | Read SPR |", "|---:|---:|---:|---:|---:|---:|---:|", f"| {fmt(mean_delta['blimp_fast'])} | {fmt(mean_delta['supplement_fast'])} | {fmt(mean_delta['ewok_fast'])} | {fmt(mean_delta['entity_tracking_fast'])} | {fmt(mean_delta['comps'])} | {fmt(mean_delta['reading_eye_tracking'])} | {fmt(mean_delta['reading_self_paced'])} |"]
    OUT_NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"mean_mixture_adjacent_minus_mixture_shuffled": mean_delta, "by_seed": deltas, "out": str(OUT_JSON)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
