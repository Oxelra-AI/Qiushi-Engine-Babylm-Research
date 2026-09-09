#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
RUN = ROOT / "training/runs/babylm_fullcycle_bert8x512_wwm_seed42_100M_b256"
OUT = ROOT / "data/bert8x512_b256_training_validation.json"
NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/bert8x512_b256_training_validation.md')


def read_json(p: pathlib.Path):
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    metrics = read_json(RUN / "scientific_metrics.json")
    manifest = read_json(RUN / "example_order_manifest.json")
    tok_summary = read_json(RUN / "tokenization_coupling_summary.json")
    log_lines = [json.loads(x) for x in (RUN / "training_log.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    ckpts = sorted(
        [p.name for p in (RUN / "hf_model").iterdir() if p.is_dir() and p.name.startswith("chck_")],
        key=lambda x: int(x.split("_")[1][:-1]) if x.endswith("M") else -1,
    )
    load_checks = []
    for rel in ["hf_model", "hf_model/chck_1M", "hf_model/chck_10M", "hf_model/chck_50M", "hf_model/chck_100M"]:
        p = RUN / rel
        tok = AutoTokenizer.from_pretrained(p)
        model = AutoModelForMaskedLM.from_pretrained(p)
        load_checks.append({
            "rel": rel,
            "tokenizer_class": tok.__class__.__name__,
            "model_class": model.__class__.__name__,
            "parameter_count": model.num_parameters(),
        })
    payload = {
        "run_dir": str(RUN),
        "valid": True,
        "model_family": metrics["model_family"],
        "model_type": metrics["model_type"],
        "parameter_count": metrics["parameter_count"],
        "embedding_parameter_count": metrics["embedding_parameter_count"],
        "non_embedding_parameter_count": metrics["non_embedding_parameter_count"],
        "word_exposure": metrics["word_exposure"],
        "selected_for_training_words": metrics["selected_for_training_words"],
        "example_pool_words_actual": metrics["example_pool_words_actual"],
        "actual_training_steps": metrics["actual_training_steps"],
        "lr_schedule_total_steps": metrics["lr_schedule_total_steps"],
        "loss_first": metrics["loss_first"],
        "loss_last": metrics["loss_last"],
        "batch_words_first": log_lines[0]["batch_words"],
        "first_log": log_lines[0],
        "mid_log": log_lines[len(log_lines)//2],
        "last_log": log_lines[-1],
        "num_log_lines": len(log_lines),
        "data_source_type": manifest.get("data_source_type"),
        "num_selection_epochs": len(manifest.get("selection_epochs", [])),
        "selection_epochs_first2": manifest.get("selection_epochs", [])[:2],
        "selection_epochs_last2": manifest.get("selection_epochs", [])[-2:],
        "unique_official_pool_words": manifest.get("unique_official_pool_words"),
        "num_checkpoints": len(ckpts),
        "checkpoints_first5": ckpts[:5],
        "checkpoints_last5": ckpts[-5:],
        "load_checks": load_checks,
        "tokenization_coupling_summary": tok_summary,
        "interpretation": "BERT 8x512 batch-256/2442-step update-geometry control for DeBERTa-v2 b256; same architecture as BERT34 b512 but same batch/update geometry as DeBERTa rescue.",
    }
    # Basic hard checks.
    assert payload["word_exposure"] == 100000000
    assert payload["selected_for_training_words"] == 100000000
    assert payload["actual_training_steps"] == 2442
    assert payload["lr_schedule_total_steps"] == 2442
    assert payload["num_selection_epochs"] == 10
    assert payload["num_checkpoints"] == 100
    assert payload["last_log"]["cumulative_word_exposure"] == 100000000
    assert all(x["model_class"] == "BertForMaskedLM" for x in load_checks)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research — BERT 8x512 batch-256 update-geometry control validation", "",
        f"Evidence JSON: `{OUT}`", "",
        f"Run: `{RUN}`", "",
        "Validated as a full 100M official-corpus run matching the DeBERTa-v2 b256 update geometry.", "",
        "| property | value |", "|---|---:|",
        f"| parameters | {payload['parameter_count']} |",
        f"| non-embedding parameters | {payload['non_embedding_parameter_count']} |",
        f"| exposure words | {payload['word_exposure']} |",
        f"| optimizer steps | {payload['actual_training_steps']} |",
        f"| loss first | {payload['loss_first']:.4f} |",
        f"| loss last | {payload['loss_last']:.4f} |",
        f"| checkpoints | {payload['num_checkpoints']} |",
        f"| truncated example fraction | {tok_summary['truncated_example_fraction']:.6f} |",
    ]
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "BERT_B256_VALIDATED", "out": str(OUT), "params": payload["parameter_count"], "steps": payload["actual_training_steps"], "loss_first": payload["loss_first"], "loss_last": payload["loss_last"]}, indent=2))

if __name__ == "__main__":
    main()
