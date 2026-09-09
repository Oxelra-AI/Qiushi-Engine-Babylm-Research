#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys

from training_process import run_training
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
TRAIN = ROOT / "training/scripts/babylm_masked_train_fullcycle.py"
RUN_ROOT = ROOT / "training/runs"
OUT = ROOT / "data/34m_smoke_summary.json"
NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/34m_smoke_summary.md')

ARMS = [
    {
        "name": "bert8x512",
        "run_id": "babylm_smoke_bert8x512_wwm_20k",
        "model_type": "bert",
        "hidden": 512,
        "heads": 8,
        "extra": [],
    },
    {
        "name": "deberta8x480",
        "run_id": "babylm_smoke_deberta8x480_wwm_20k",
        "model_type": "deberta_v2",
        "hidden": 480,
        "heads": 8,
        "extra": [
            "--position_buckets", "256",
            "--max_relative_positions", "256",
            "--deberta_relative_attention", "true",
            "--deberta_pos_att_type", "p2c,c2p",
        ],
    },
]


def run(cmd: list[str]) -> str:
    print("$", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.stdout[-5000:], flush=True)
    if p.returncode != 0:
        raise RuntimeError(f"command failed {p.returncode}: {' '.join(cmd)}\n{p.stdout[-10000:]}")
    return p.stdout


def smoke_arm(arm: dict) -> dict:
    run_dir = RUN_ROOT / arm["run_id"]
    if run_dir.exists():
        shutil.rmtree(run_dir)
    cmd = [
        sys.executable, str(TRAIN),
        "--output_dir", str(run_dir),
        "--max_word_exposure", "20000",
        "--example_pool_words", "20000",
        "--checkpoint_words", "20000",
        "--model_type", arm["model_type"],
        "--hidden_size", str(arm["hidden"]),
        "--n_layer", "8",
        "--n_head", str(arm["heads"]),
        "--ffn_mult", "4",
        *arm["extra"],
        "--tokenizer_label", "baseline16k",
        "--mask_mode", "wwm",
        "--mask_prob", "0.15",
        "--seq_length", "256",
        "--max_seq_length", "256",
        "--max_position_embeddings", "512",
        "--batch_size", "4",
        "--lr_total_steps", "32",
        "--learning_rate", "0.001",
        "--seed", "42",
        "--extra_init_seed", "456",
        "--train_rng_seed", "789",
        "--log_every", "1",
    ]
    run_training(cmd, output_dir=run_dir, timeout=1200)
    load_checks = []
    for rel in ["hf_model", "hf_model/chck_1M"]:
        p = run_dir / rel
        tok = AutoTokenizer.from_pretrained(p)
        model = AutoModelForMaskedLM.from_pretrained(p)
        load_checks.append({
            "rel": rel,
            "tokenizer_class": tok.__class__.__name__,
            "model_class": model.__class__.__name__,
            "parameter_count": model.num_parameters(),
        })
    metrics = json.loads((run_dir / "scientific_metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "example_order_manifest.json").read_text(encoding="utf-8"))
    return {
        "name": arm["name"],
        "run_id": arm["run_id"],
        "run_dir": str(run_dir),
        "metrics": {
            "model_family": metrics["model_family"],
            "model_type": metrics["model_type"],
            "parameter_count": metrics["parameter_count"],
            "embedding_parameter_count": metrics["embedding_parameter_count"],
            "non_embedding_parameter_count": metrics["non_embedding_parameter_count"],
            "word_exposure": metrics["word_exposure"],
            "actual_training_steps": metrics["actual_training_steps"],
            "loss_first": metrics["loss_first"],
            "loss_last": metrics["loss_last"],
            "hidden_size": metrics["hidden_size"],
            "n_layer": metrics["n_layer"],
            "n_head": metrics["n_head"],
            "intermediate_size": metrics["intermediate_size"],
            "deberta_relative_attention": metrics.get("deberta_relative_attention"),
            "deberta_pos_att_type": metrics.get("deberta_pos_att_type"),
        },
        "manifest": {
            "data_source_type": manifest.get("data_source_type"),
            "selected_for_training_words": manifest.get("selected_for_training_words"),
            "unique_official_pool_words": manifest.get("unique_official_pool_words"),
            "selection_epochs": manifest.get("selection_epochs", []),
        },
        "load_checks": load_checks,
    }


def main():
    rows = [smoke_arm(arm) for arm in ARMS]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"status": "34M_SMOKE_OK", "arms": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research — 34M arm smoke summary", "", f"Evidence JSON: `{OUT}`", "", "| arm | model | params | non-emb | steps | loss first→last | load |", "|---|---|---:|---:|---:|---:|---|"]
    for row in rows:
        m = row["metrics"]
        ok = all(x["model_class"] for x in row["load_checks"])
        lines.append(f"| {row['name']} | {m['model_family']} | {m['parameter_count']} | {m['non_embedding_parameter_count']} | {m['actual_training_steps']} | {m['loss_first']:.3f}→{m['loss_last']:.3f} | {ok} |")
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "34M_SMOKE_OK", "out": str(OUT), "arms": [{"name": r["name"], "params": r["metrics"]["parameter_count"], "model": r["metrics"]["model_family"]} for r in rows]}, indent=2))

if __name__ == "__main__":
    main()
