#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib

from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
runs = {
    "official": ROOT / "training/runs/babylm_repaired_official_tiny_smoke",
    "jsonl": ROOT / "training/runs/babylm_repaired_jsonl_tiny_smoke",
}
summary = {}
for name, path in runs.items():
    tok = AutoTokenizer.from_pretrained(path / "hf_model")
    model = AutoModelForMaskedLM.from_pretrained(path / "hf_model")
    metrics = json.loads((path / "scientific_metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((path / "data_manifest.json").read_text(encoding="utf-8"))
    order = json.loads((path / "example_order_manifest.json").read_text(encoding="utf-8"))
    coupling = json.loads((path / "tokenization_coupling_summary.json").read_text(encoding="utf-8"))
    summary[name] = {
        "model_class": model.__class__.__name__,
        "tokenizer_class": tok.__class__.__name__,
        "word_exposure": metrics["word_exposure"],
        "data_source_type": metrics["data_source_type"],
        "loss_first": metrics["loss_first"],
        "loss_last": metrics["loss_last"],
        "num_examples": order["num_consumed_examples"],
        "selected_words": order["selected_for_training_words"],
        "total_words_manifest": manifest["total_dataset_whitespace_words_counted"],
        "truncated_examples": coupling["truncated_examples_at_max_seq_length"],
        "tokens_per_word": coupling["untruncated_tokens_per_whitespace_word"],
        "checkpoint_names": [c["name"] for c in metrics["saved_checkpoints"]],
    }
assert summary["official"]["data_source_type"] == "official_corpus", summary
assert summary["jsonl"]["data_source_type"] == "example_jsonl", summary
assert summary["jsonl"]["word_exposure"] == 28, summary
assert summary["jsonl"]["num_examples"] == 2, summary
out = ROOT / "data/trainer_repair_smoke_summary.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
print("WROTE", out)
