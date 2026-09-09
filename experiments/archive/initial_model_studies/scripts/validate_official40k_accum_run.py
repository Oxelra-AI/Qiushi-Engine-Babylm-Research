#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
RUN = ROOT / "training/runs/babylm_fullcycle_debertav2_8x480_official40k_wwm_seed42_100M_b128_acc2"
OUT = ROOT / "data/official40k_accum_training_validation.json"
NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/official40k_accum_training_validation.md')


def read_json(path: pathlib.Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    metrics = read_json(RUN / "scientific_metrics.json")
    manifest = read_json(RUN / "example_order_manifest.json")
    tok_summary = read_json(RUN / "tokenization_coupling_summary.json")
    log_lines = [json.loads(line) for line in (RUN / "training_log.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    hf = RUN / "hf_model"
    ckpts = sorted(
        [p.name for p in hf.iterdir() if p.is_dir() and p.name.startswith("chck_")],
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
            "tokenizer_len": len(tok),
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
        "vocab_size": metrics["vocab_size"],
        "tokenizer_label": metrics["tokenizer_label"],
        "tokenizer_path": metrics["tokenizer_path"],
        "word_exposure": metrics["word_exposure"],
        "selected_for_training_words": metrics["selected_for_training_words"],
        "example_pool_words_actual": metrics["example_pool_words_actual"],
        "actual_training_steps": metrics["actual_training_steps"],
        "lr_schedule_total_steps": metrics["lr_schedule_total_steps"],
        "loss_first": metrics["loss_first"],
        "loss_last": metrics["loss_last"],
        "masked_tokens_total": metrics["masked_tokens_total"],
        "masked_tokens_per_whitespace_word": metrics["masked_tokens_per_whitespace_word"],
        "first_log": log_lines[0],
        "mid_log": log_lines[len(log_lines)//2],
        "last_log": log_lines[-1],
        "num_log_lines": len(log_lines),
        "num_selection_epochs": len(manifest.get("selection_epochs", [])),
        "unique_official_pool_words": manifest.get("unique_official_pool_words"),
        "total_official_corpus_words": manifest.get("total_official_corpus_words"),
        "num_checkpoints": len(ckpts),
        "checkpoints_first5": ckpts[:5],
        "checkpoints_last5": ckpts[-5:],
        "load_checks": load_checks,
        "tokenization_coupling_summary": tok_summary,
        "interpretation": "Full official40k DeBERTa-v2 8x480 run repaired with microbatch 128 and grad_accum_steps 2; same intended effective batch/update schedule as baseline16k DeBERTa b256, but 40k tokenizer and larger embedding matrix.",
    }
    assert payload["word_exposure"] == 100000000
    assert payload["selected_for_training_words"] == 100000000
    assert payload["actual_training_steps"] == 2442
    assert payload["lr_schedule_total_steps"] == 2442
    assert payload["num_log_lines"] == 2442
    assert payload["last_log"]["cumulative_word_exposure"] == 100000000
    assert payload["num_selection_epochs"] == 10
    assert payload["num_checkpoints"] == 100
    assert payload["vocab_size"] == 40000
    assert payload["parameter_count"] == 45826720
    assert payload["non_embedding_parameter_count"] == 26626720
    assert all(c["tokenizer_len"] == 40000 and c["model_class"] == "DebertaV2ForMaskedLM" and c["parameter_count"] == 45826720 for c in load_checks)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research — official40k DeBERTa-v2 accumulation-run validation", "",
        f"Evidence JSON: `{OUT}`", "", f"Run: `{RUN}`", "",
        "Validated as a complete 100M official-corpus run repaired with gradient accumulation.", "",
        "| property | value |", "|---|---:|",
        f"| parameters | {payload['parameter_count']} |",
        f"| embedding parameters | {payload['embedding_parameter_count']} |",
        f"| non-embedding parameters | {payload['non_embedding_parameter_count']} |",
        f"| vocab size | {payload['vocab_size']} |",
        f"| exposure words | {payload['word_exposure']} |",
        f"| optimizer steps | {payload['actual_training_steps']} |",
        f"| microbatch / accumulation | {payload['first_log']['microbatch_size']} / {payload['first_log']['grad_accum_steps']} |",
        f"| loss first | {payload['loss_first']:.4f} |",
        f"| loss last | {payload['loss_last']:.4f} |",
        f"| masked tokens/word | {payload['masked_tokens_per_whitespace_word']:.6f} |",
        f"| checkpoints | {payload['num_checkpoints']} |",
        f"| kept tokens/word | {tok_summary['kept_tokens_per_whitespace_word']:.6f} |",
        f"| truncated example fraction | {tok_summary['truncated_example_fraction']:.6f} |",
    ]
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "OFFICIAL40K_ACCUM_VALIDATED",
        "out": str(OUT),
        "params": payload["parameter_count"],
        "steps": payload["actual_training_steps"],
        "loss_first": payload["loss_first"],
        "loss_last": payload["loss_last"],
        "truncated_fraction": tok_summary["truncated_example_fraction"],
    }, indent=2))

if __name__ == "__main__":
    main()
