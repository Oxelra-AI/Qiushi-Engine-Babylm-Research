#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess

from training_process import run_training
import sys

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
TRAIN = ROOT / "training/scripts/babylm_masked_train.py"
STRICT_DIR = ROOT / "repos/babylm-eval/strict"
META = ROOT / "data/rewrite_pairs_revision_57/pair_materialization_target10000_actual10211_n211_sel5701_shuf5702.json"
RUN_ID = "babylm_pair_adjacent_wwm_smoke10k"
RUN_DIR = ROOT / "training/runs" / RUN_ID
OUT_JSON = ROOT / "data/pair_adjacent_smoke_summary.json"
OUT_NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/pair_adjacent_smoke.md')


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


def run(cmd: list[str], env: dict[str, str], cwd: pathlib.Path | None = None) -> subprocess.CompletedProcess:
    print("$", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=str(cwd) if cwd else None)
    print(p.stdout[-6000:], flush=True)
    if p.returncode != 0:
        raise RuntimeError(f"command failed with {p.returncode}: {' '.join(cmd)}\n{p.stdout[-12000:]}")
    return p


def read_avg(report: pathlib.Path) -> float:
    txt = report.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"AVERAGE ACCURACY\s*\n([0-9.\-]+)", txt)
    if not m:
        raise RuntimeError(f"could not parse {report}\n{txt[:500]}")
    return float(m.group(1))


def main() -> None:
    env = setup_env()
    meta = json.loads(META.read_text(encoding="utf-8"))
    validation = meta["validation"]
    required_true = [
        "identical_target_multiset",
        "identical_sentence_multiset",
        "identical_example_count",
        "identical_total_words",
        "no_adjacent_example_over_256",
        "no_shuffled_example_over_256",
        "no_identity_target_adjacency_in_shuffled",
    ]
    assert all(validation[k] for k in required_true), validation
    assert validation["token_count_delta_total"] == 0, validation
    assert validation["word_group_delta_total"] == 0, validation
    actual_words = int(meta["actual_words"])
    pair_adjacent = meta["pair_adjacent_path"]

    run_training([
        sys.executable, str(TRAIN),
        "--output_dir", str(RUN_DIR),
        "--example_jsonl", pair_adjacent,
        "--example_jsonl_label", "GEM_wiki_auto_asset_turk_pair_adjacent_step57_smoke",
        "--example_jsonl_meta", str(META),
        "--max_word_exposure", str(actual_words),
        "--example_pool_words", str(actual_words),
        "--checkpoint_words", str(actual_words),
        "--tokenizer_label", "baseline16k",
        "--tokenization_summary_limit", "0",
        "--mask_mode", "wwm",
        "--mask_prob", "0.15",
        "--seq_length", "256",
        "--max_seq_length", "256",
        "--max_position_embeddings", "512",
        "--batch_size", "16",
        "--lr_total_steps", "16",
        "--hidden_size", "128",
        "--n_layer", "2",
        "--n_head", "4",
        "--learning_rate", "0.001",
        "--seed", "42",
        "--extra_init_seed", "456",
        "--train_rng_seed", "789",
        "--log_every", "4",
    ], output_dir=RUN_DIR, timeout=1200, env=env)

    checks = []
    for rel in ["hf_model", "hf_model/chck_1M"]:
        p = RUN_DIR / rel
        tok = AutoTokenizer.from_pretrained(p)
        model = AutoModelForMaskedLM.from_pretrained(p)
        ids = torch.randint(low=0, high=len(tok), size=(2, 180))
        y = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids)
        checks.append({
            "rel": rel,
            "model_class": model.__class__.__name__,
            "tokenizer_class": tok.__class__.__name__,
            "parameter_count": sum(x.numel() for x in model.parameters()),
            "seq_len_forward_tested": 180,
            "forward_loss": float(y.loss.detach()),
        })

    outdir = (RUN_DIR / "eval_results_pair_adjacent_smoke").resolve()
    model_path = (RUN_DIR / "hf_model").resolve()
    scores = {}
    reports = {}
    for task_name, task, data_path in [
        ("blimp_fast", "blimp", "evaluation_data/fast_eval/blimp_fast"),
        ("entity_tracking_fast", "entity_tracking", "evaluation_data/fast_eval/entity_tracking_fast"),
    ]:
        run([
            sys.executable, "-m", "evaluation_pipeline.sentence_zero_shot.run",
            "--model_path_or_name", str(model_path),
            "--backend", "mlm",
            "--task", task,
            "--data_path", data_path,
            "--save_predictions",
            "--revision_name", "chck_1M",
            "--batch_size", "64",
            "--output_dir", str(outdir),
        ], env, cwd=STRICT_DIR)
        report = outdir / "hf_model" / "chck_1M" / "zero_shot" / "mlm" / task / task_name / "best_temperature_report.txt"
        scores[task_name] = read_avg(report)
        reports[task_name] = str(report)

    metrics = json.loads((RUN_DIR / "scientific_metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((RUN_DIR / "data_manifest.json").read_text(encoding="utf-8"))
    coupling = json.loads((RUN_DIR / "tokenization_coupling_summary.json").read_text(encoding="utf-8"))
    assert metrics["data_source_type"] == "example_jsonl", metrics
    assert metrics["word_exposure"] == actual_words, metrics
    assert coupling["truncated_examples_at_max_seq_length"] == 0, coupling
    summary = {
        "status": "PAIR_ADJACENT_WWM_SMOKE_OK",
        "run_id": RUN_ID,
        "materialization_meta": str(META),
        "pair_adjacent_jsonl": pair_adjacent,
        "actual_words": actual_words,
        "checks": checks,
        "scores": scores,
        "reports": reports,
        "metrics_subset": {
            "data_source_type": metrics["data_source_type"],
            "example_jsonl_label": metrics["example_jsonl_label"],
            "parameter_count": metrics["parameter_count"],
            "embedding_parameter_count": metrics["embedding_parameter_count"],
            "loss_first": metrics["loss_first"],
            "loss_last": metrics["loss_last"],
            "word_exposure": metrics["word_exposure"],
            "masked_tokens_per_whitespace_word": metrics["masked_tokens_per_whitespace_word"],
            "actual_training_steps": metrics["actual_training_steps"],
        },
        "manifest_files": manifest["files"],
        "tokenization_coupling_summary": coupling,
        "materialization_validation": validation,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    OUT_NOTE.parent.mkdir(parents=True, exist_ok=True)
    OUT_NOTE.write_text(
        "# research — pair-adjacent rewrite JSONL smoke\n\n"
        f"Evidence JSON: `{OUT_JSON}`\n\n"
        "The materialized pair-adjacent JSONL trained through the repaired `--example_jsonl` path, saved root/`chck_1M`, loaded with `AutoModelForMaskedLM`, and completed official fast BLiMP and Entity with backend `mlm`.\n\n"
        f"Words: {actual_words}; examples: {coupling['num_examples_summarized']}; truncated examples: {coupling['truncated_examples_at_max_seq_length']}.\n"
        f"BLiMP fast: {scores['blimp_fast']:.2f}; Entity fast: {scores['entity_tracking_fast']:.2f}; loss {metrics['loss_first']:.4f}->{metrics['loss_last']:.4f}.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("WROTE", OUT_JSON)
    print("WROTE", OUT_NOTE)


if __name__ == "__main__":
    main()
