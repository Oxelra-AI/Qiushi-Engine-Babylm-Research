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
PAIR_DIR = ROOT / "data/rewrite_pairs_revision_57"
META = PAIR_DIR / "pair_materialization_target1000000_actual999995_n21080_sel5711_shuf5712.json"
ADJ_JSONL = PAIR_DIR / "pair_adjacent_target1000000_actual999995_n21080_sel5711_shuf5712.jsonl"
SHUF_JSONL = PAIR_DIR / "pair_shuffled_target1000000_actual999995_n21080_sel5711_shuf5712.jsonl"
WORDS = 999995
OUT_JSON = ROOT / "data/pair_vs_shuffle_1m_profile.json"
OUT_TRAIN_JSON = ROOT / "data/pair_vs_shuffle_1m_training_summary.json"
OUT_NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/pair_vs_shuffle_1m_profile.md')
LOG = (ROOT.parents[2] / 'research/notes/initial_model_studies/pair_vs_shuffle_1m_train_and_profile.log')

SEED_SPECS = [
    {"seed": 42, "extra_init_seed": 456, "train_rng_seed": 789},
    {"seed": 43, "extra_init_seed": 457, "train_rng_seed": 790},
]
ARMS = [
    {"arm": "pair_adjacent", "jsonl": ADJ_JSONL, "label": "GEM_wiki_auto_asset_turk_pair_adjacent"},
    {"arm": "pair_shuffled", "jsonl": SHUF_JSONL, "label": "GEM_wiki_auto_asset_turk_pair_shuffled"},
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
    "--max_word_exposure", str(WORDS),
    "--example_pool_words", str(WORDS),
    "--checkpoint_words", str(WORDS),
    "--example_jsonl_meta", str(META),
    "--tokenizer_label", "baseline16k",
    "--tokenization_summary_limit", "0",
    "--mask_mode", "wwm",
    "--mask_prob", "0.15",
    "--seq_length", "256",
    "--max_seq_length", "256",
    "--max_position_embeddings", "512",
    # The materialized pair examples are shorter and more numerous than the
    # official 160-word chunks. Use batch_size=216 so 21,080 examples produce
    # 98 optimizer updates, preserving the prior 1M WWM schedule/update count
    # and roughly the same words per update instead of silently running 330 steps.
    "--batch_size", "216",
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
    print(p.stdout[-4000:], flush=True)
    if logf:
        logf.write(p.stdout + f"\n[returncode={p.returncode}]\n"); logf.flush()
    if p.returncode != 0:
        raise RuntimeError(f"command failed with {p.returncode}: {' '.join(cmd)}\n{p.stdout[-8000:]}")


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
    return f"babylm_step58_{arm}_wwm_seed{seed}_999995w"


def train_and_check(arm_spec: dict, seed_spec: dict, env: dict[str, str], logf) -> dict:
    run_id = run_id_for(arm_spec["arm"], seed_spec["seed"])
    run_dir = RUN_ROOT / run_id
    cmd = [
        sys.executable, str(TRAIN),
        "--output_dir", str(run_dir),
        "--example_jsonl", str(arm_spec["jsonl"]),
        "--example_jsonl_label", arm_spec["label"],
        "--seed", str(seed_spec["seed"]), "--extra_init_seed", str(seed_spec["extra_init_seed"]), "--train_rng_seed", str(seed_spec["train_rng_seed"]),
        *COMMON,
    ]
    if (run_dir / "scientific_metrics.json").exists() and (run_dir / "hf_model" / "chck_1M" / "config.json").exists():
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
    assert metrics["word_exposure"] == WORDS, metrics
    assert metrics["mask_mode"] == "wwm"
    assert coupling["truncated_examples_at_max_seq_length"] == 0, coupling
    return {"run_id": run_id, "arm": arm_spec["arm"], "seed": seed_spec["seed"], "metrics": metrics, "coupling": coupling}


def profile(run_id: str, env: dict[str, str], logf) -> dict:
    run_dir = RUN_ROOT / run_id
    model_path = (run_dir / "hf_model").resolve()
    outdir = (run_dir / "eval_results_revision_58").resolve()
    scores = {}
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
    run([
        sys.executable, "-m", "evaluation_pipeline.reading.run",
        "--model_path_or_name", str(model_path), "--backend", "mlm",
        "--data_path", "evaluation_data/fast_eval/reading/reading_data.csv",
        "--revision_name", "chck_1M", "--output_dir", str(outdir),
    ], env, cwd=STRICT_DIR, logf=logf)
    rreport = outdir / "hf_model" / "chck_1M" / "zero_shot" / "mlm" / "reading" / "report.txt"
    assert rreport.exists(), rreport
    scores.update(read_reading(rreport))
    return {"run_id": run_id, "scores": scores}


def diff(a: float, b: float) -> float:
    return round(a - b, 4)


def fmt(x):
    return f"{x:.2f}" if isinstance(x, float) else str(x)


def main() -> None:
    env = setup_env()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    meta = load_json(META)
    v = meta["validation"]
    for k in ["identical_target_multiset", "identical_sentence_multiset", "identical_example_count", "identical_total_words", "no_adjacent_example_over_256", "no_shuffled_example_over_256", "no_identity_target_adjacency_in_shuffled"]:
        assert v[k], (k, v)
    assert v["token_count_delta_total"] == 0 and v["word_group_delta_total"] == 0, v

    train_rows = {}
    profile_rows = {}
    with LOG.open("a", encoding="utf-8") as logf:
        for seed_spec in SEED_SPECS:
            for arm_spec in ARMS:
                r = train_and_check(arm_spec, seed_spec, env, logf)
                train_rows[r["run_id"]] = r
        for run_id in list(train_rows.keys()):
            profile_rows[run_id] = profile(run_id, env, logf)

    deltas = {}
    per_arm_scores = {}
    for seed_spec in SEED_SPECS:
        seed = seed_spec["seed"]
        adj_id = run_id_for("pair_adjacent", seed)
        shuf_id = run_id_for("pair_shuffled", seed)
        adj = profile_rows[adj_id]["scores"]
        shuf = profile_rows[shuf_id]["scores"]
        per_arm_scores[seed] = {"pair_adjacent": adj, "pair_shuffled": shuf}
        deltas[str(seed)] = {c: diff(adj[c], shuf[c]) for c in COLS}
    mean_delta = {c: round(sum(deltas[str(s["seed"])][c] for s in SEED_SPECS) / len(SEED_SPECS), 4) for c in COLS}

    train_summary = []
    for run_id, r in train_rows.items():
        m = r["metrics"]
        train_summary.append({
            "run_id": run_id, "arm": r["arm"], "seed": r["seed"],
            "parameter_count": m["parameter_count"], "embedding_parameter_count": m["embedding_parameter_count"],
            "loss_first": m["loss_first"], "loss_last": m["loss_last"],
            "masked_tokens_per_whitespace_word": m["masked_tokens_per_whitespace_word"],
            "word_exposure": m["word_exposure"], "steps": m["actual_training_steps"],
            "data_source_type": m["data_source_type"], "example_jsonl_label": m["example_jsonl_label"],
            "truncated_examples": r["coupling"]["truncated_examples_at_max_seq_length"],
            "tokens_per_word": r["coupling"]["untruncated_tokens_per_whitespace_word"],
        })
    payload = {
        "materialization_meta": str(META),
        "materialization_validation": v,
        "train_summary": train_summary,
        "per_seed_scores": per_arm_scores,
        "pair_adjacent_minus_pair_shuffled_by_seed": deltas,
        "mean_pair_adjacent_minus_pair_shuffled": mean_delta,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_TRAIN_JSON.write_text(json.dumps(train_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# research — decisive pair-adjacent vs pair-shuffled at 1M (WWM base)",
        "",
        f"Evidence JSON: `{OUT_JSON}`",
        "",
        "Both arms share the exact same source/target sentence multiset, total words (999,995), example count (21,080), zero truncation at 256, and identical total tokens/WWM groups. Only whether each source is adjacent to its own rewrite (adjacent) or to an unrelated target (shuffled) differs. Base: BERT-WWM, baseline 16k tokenizer, fixed 256, AdamW, paired seeds 42/43.",
        "",
        "| seed | arm | BLiMP | Supp | EWoK | Entity | COMPS | Read eye | Read SPR |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed_spec in SEED_SPECS:
        seed = seed_spec["seed"]
        for arm in ["pair_adjacent", "pair_shuffled"]:
            s = per_arm_scores[seed][arm]
            lines.append(f"| {seed} | {arm} | {fmt(s['blimp_fast'])} | {fmt(s['supplement_fast'])} | {fmt(s['ewok_fast'])} | {fmt(s['entity_tracking_fast'])} | {fmt(s['comps'])} | {fmt(s['reading_eye_tracking'])} | {fmt(s['reading_self_paced'])} |")
    lines += ["", "## pair-adjacent minus pair-shuffled", "", "| seed | BLiMP | Supp | EWoK | Entity | COMPS | Read eye | Read SPR |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for seed_spec in SEED_SPECS:
        d = deltas[str(seed_spec["seed"])]
        lines.append(f"| {seed_spec['seed']} | {fmt(d['blimp_fast'])} | {fmt(d['supplement_fast'])} | {fmt(d['ewok_fast'])} | {fmt(d['entity_tracking_fast'])} | {fmt(d['comps'])} | {fmt(d['reading_eye_tracking'])} | {fmt(d['reading_self_paced'])} |")
    lines += ["", "## mean delta", "", "| BLiMP | Supp | EWoK | Entity | COMPS | Read eye | Read SPR |", "|---:|---:|---:|---:|---:|---:|---:|", f"| {fmt(mean_delta['blimp_fast'])} | {fmt(mean_delta['supplement_fast'])} | {fmt(mean_delta['ewok_fast'])} | {fmt(mean_delta['entity_tracking_fast'])} | {fmt(mean_delta['comps'])} | {fmt(mean_delta['reading_eye_tracking'])} | {fmt(mean_delta['reading_self_paced'])} |"]
    OUT_NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"mean_pair_adjacent_minus_pair_shuffled": mean_delta, "by_seed": deltas, "out": str(OUT_JSON)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
