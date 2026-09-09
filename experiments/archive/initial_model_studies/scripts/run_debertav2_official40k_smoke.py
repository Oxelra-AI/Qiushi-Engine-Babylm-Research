#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys

from training_process import run_training
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
TRAIN = ROOT / "training/scripts/babylm_masked_train_fullcycle.py"
TOK40 = ROOT / "training/tokenizers/official40k"
RUN = ROOT / "training/runs/babylm_smoke_debertav2_8x480_official40k_20k"
OUT = ROOT / "data/debertav2_official40k_smoke.json"
NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/debertav2_official40k_smoke.md')


def run(cmd: list[str]) -> str:
    print("$", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.stdout[-5000:], flush=True)
    if p.returncode != 0:
        raise RuntimeError(f"command failed {p.returncode}: {' '.join(cmd)}\n{p.stdout[-10000:]}")
    return p.stdout


def main():
    if RUN.exists():
        shutil.rmtree(RUN)
    tok = AutoTokenizer.from_pretrained(TOK40, use_fast=True)
    tok_info = {"tokenizer_path": str(TOK40), "class": tok.__class__.__name__, "len": len(tok), "vocab_size_attr": getattr(tok, "vocab_size", None), "mask_token": tok.mask_token, "pad_token": tok.pad_token}
    cmd = [
        sys.executable, str(TRAIN),
        "--output_dir", str(RUN),
        "--max_word_exposure", "20000",
        "--example_pool_words", "20000",
        "--checkpoint_words", "20000",
        "--model_type", "deberta_v2",
        "--hidden_size", "480",
        "--n_layer", "8",
        "--n_head", "8",
        "--ffn_mult", "4",
        "--position_buckets", "256",
        "--max_relative_positions", "256",
        "--deberta_relative_attention", "true",
        "--deberta_pos_att_type", "p2c,c2p",
        "--tokenizer_path", str(TOK40),
        "--tokenizer_label", "official40k",
        "--tokenization_summary_limit", "0",
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
    run_training(cmd, output_dir=RUN, timeout=1200)
    load_checks = []
    for rel in ["hf_model", "hf_model/chck_1M"]:
        p = RUN / rel
        lt = AutoTokenizer.from_pretrained(p)
        lm = AutoModelForMaskedLM.from_pretrained(p)
        load_checks.append({"rel": rel, "tokenizer_class": lt.__class__.__name__, "tokenizer_len": len(lt), "model_class": lm.__class__.__name__, "parameter_count": lm.num_parameters()})
    metrics = json.loads((RUN / "scientific_metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((RUN / "example_order_manifest.json").read_text(encoding="utf-8"))
    tok_sum = json.loads((RUN / "tokenization_coupling_summary.json").read_text(encoding="utf-8"))
    payload = {"status": "OFFICIAL40K_SMOKE_OK", "run_dir": str(RUN), "initial_tokenizer": tok_info, "metrics": {k: metrics.get(k) for k in ["model_family", "model_type", "parameter_count", "embedding_parameter_count", "non_embedding_parameter_count", "vocab_size", "tokenizer_label", "tokenizer_path", "word_exposure", "actual_training_steps", "loss_first", "loss_last", "deberta_relative_attention", "deberta_pos_att_type"]}, "manifest": {k: manifest.get(k) for k in ["tokenizer_label", "tokenizer_path", "tokenizer_vocab_size", "selected_for_training_words", "data_source_type"]}, "tokenization_summary": tok_sum, "load_checks": load_checks}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research — DeBERTa-v2 official40k smoke", "", f"Evidence JSON: `{OUT}`", "", "| property | value |", "|---|---:|", f"| tokenizer length | {tok_info['len']} |", f"| params | {payload['metrics']['parameter_count']} |", f"| embedding params | {payload['metrics']['embedding_parameter_count']} |", f"| non-embedding params | {payload['metrics']['non_embedding_parameter_count']} |", f"| loss first→last | {payload['metrics']['loss_first']:.4f}→{payload['metrics']['loss_last']:.4f} |", f"| tokens/word | {tok_sum['kept_tokens_per_word']:.4f} |", f"| truncated frac | {tok_sum['truncated_example_fraction']:.6f} |"]
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "OFFICIAL40K_SMOKE_OK", "out": str(OUT), "params": payload["metrics"]["parameter_count"], "vocab": payload["metrics"]["vocab_size"], "tokens_per_word": tok_sum["kept_tokens_per_word"]}, indent=2))

if __name__ == "__main__":
    main()
