#!/usr/bin/env python3
"""Compare INITIAL_MODEL_STUDIES research fixed-WWM run with COMPACT_EXPERIENCE b256 fixed-WWM rerun."""
from __future__ import annotations
import hashlib, json
from pathlib import Path

INITIAL_MODEL_STUDIES = Path("experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256")
COMPACT_EXPERIENCE = Path("experiments/archive/compact_experience/training/runs/wwm_fixed_100M_b256_seq256_seed43")
OUT = Path("experiments/archive/compact_experience/data/b256_pair_eval/alignment_comparison.json")
NOTE = Path("research/notes/compact_experience/alignment_comparison.md")


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def sha16(p: Path):
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()[:16]


def jsonl_first_last(path: Path):
    if not path.exists():
        return {"exists": False}
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    out = {"exists": True, "lines": len(lines)}
    if lines:
        out["first"] = json.loads(lines[0])
        out["last"] = json.loads(lines[-1])
    return out


def manifest_summary(run: Path):
    p = run / "example_order_manifest.json"
    if not p.exists():
        p = run / "data_manifest.json"
    if not p.exists():
        return {"exists": False}
    obj = load_json(p)
    keys = ["seed", "example_pool_words_actual", "selected_for_training_words", "words_per_example", "num_consumed_examples", "source_words_consumed", "data_source_type", "selection_epochs"]
    out = {"exists": True, "path": str(p), "sha16": sha16(p), "has_consumed_ids": "consumed_example_ids_in_order" in obj}
    for k in keys:
        if k in obj:
            out[k] = obj[k]
    if "consumed_example_ids_in_order" in obj:
        ids = obj["consumed_example_ids_in_order"]
        out["ids_len"] = len(ids)
        out["ids_head20"] = ids[:20]
        out["ids_tail20"] = ids[-20:]
    return out


def run_summary(run: Path):
    m = load_json(run / "scientific_metrics.json") or {}
    cfg = load_json(run / "hf_model/chck_100M/config.json") or load_json(run / "hf_model/config.json") or {}
    return {
        "run": str(run),
        "exists": run.exists(),
        "metrics_sha16": sha16(run / "scientific_metrics.json"),
        "config_sha16": sha16(run / "hf_model/chck_100M/config.json"),
        "root_config_sha16": sha16(run / "hf_model/config.json"),
        "tokenizer_sha16": sha16(run / "hf_model/chck_100M/tokenizer.json"),
        "root_tokenizer_sha16": sha16(run / "hf_model/tokenizer.json"),
        "model_sha16": sha16(run / "hf_model/chck_100M/model.safetensors"),
        "train_log": jsonl_first_last(run / "training_log.jsonl"),
        "manifest": manifest_summary(run),
        "metrics_core": {k: m.get(k) for k in [
            "variant", "backend", "model_family", "model_type", "parameter_count", "embedding_parameter_count", "non_embedding_parameter_count", "vocab_size", "tokenizer_label", "tokenizer_path", "word_exposure", "example_pool_words_actual", "selected_for_training_words", "loss_first", "loss_last", "lr_schedule_total_steps", "actual_training_steps", "mask_mode", "mask_prob", "masked_tokens_total", "masked_tokens_mean_per_step", "masked_tokens_per_whitespace_word", "seq_length", "max_seq_length", "hidden_size", "n_layer", "n_head", "ffn_mult", "intermediate_size", "position_buckets", "max_relative_positions", "deberta_relative_attention", "deberta_pos_att_type", "seed", "extra_init_seed", "train_rng_seed"
        ]},
        "config_core": {k: cfg.get(k) for k in [
            "model_type", "vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads", "intermediate_size", "max_position_embeddings", "position_buckets", "max_relative_positions", "relative_attention", "pos_att_type", "hidden_dropout_prob", "attention_probs_dropout_prob", "layer_norm_eps", "initializer_range", "pad_token_id"
        ]},
    }


def diff_dict(a: dict, b: dict):
    keys = sorted(set(a) | set(b))
    return {k: {"INITIAL_MODEL_STUDIES": a.get(k), "COMPACT_EXPERIENCE": b.get(k)} for k in keys if a.get(k) != b.get(k)}


def main():
    a = run_summary(INITIAL_MODEL_STUDIES)
    b = run_summary(COMPACT_EXPERIENCE)
    payload = {
        "status": "STEP263_ALIGNMENT_COMPARISON",
        "INITIAL_MODEL_STUDIES_step263": a,
        "COMPACT_EXPERIENCE_b256_fixed": b,
        "metrics_core_diff": diff_dict(a["metrics_core"], b["metrics_core"]),
        "config_core_diff": diff_dict(a["config_core"], b["config_core"]),
        "source_words_equal": a["manifest"].get("source_words_consumed") == b["manifest"].get("source_words_consumed"),
        "selection_epochs_equal": a["manifest"].get("selection_epochs") == b["manifest"].get("selection_epochs"),
        "tokenizer_sha_equal": a.get("tokenizer_sha16") == b.get("tokenizer_sha16"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research — research alignment comparison",
        "",
        f"JSON: `{OUT}`",
        "",
        "This compares the old INITIAL_MODEL_STUDIES research fixed-WWM run with the COMPACT_EXPERIENCE b256/fixed-seq fixed-WWM rerun after the research-style direct recheck showed an 8-point Supplement mismatch.",
        "",
        f"- Source-word accounting equal: `{payload['source_words_equal']}`.",
        f"- Selection-epoch summaries equal: `{payload['selection_epochs_equal']}`.",
        f"- Checkpoint tokenizer SHA equal: `{payload['tokenizer_sha_equal']}`.",
        f"- INITIAL_MODEL_STUDIES manifest stores exact consumed IDs: `{a['manifest'].get('has_consumed_ids')}`; COMPACT_EXPERIENCE manifest stores exact consumed IDs: `{b['manifest'].get('has_consumed_ids')}`.",
        "",
        "## Core metric differences",
        "",
        "```json",
        json.dumps(payload["metrics_core_diff"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Core config differences",
        "",
        "```json",
        json.dumps(payload["config_core_diff"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Interpretation",
        "",
        "The two runs match the high-level data source word counts and selection-epoch summary, and use the same checkpoint tokenizer. They do not have the same final weights, and the COMPACT_EXPERIENCE manifest lacks the full consumed-example ID list, so exact data order identity cannot be verified from COMPACT_EXPERIENCE artifacts. The old INITIAL_MODEL_STUDIES trainer and COMPACT_EXPERIENCE curriculum trainer must be compared at implementation/RNG level before treating the b256 rerun as the research coordinate.",
    ]
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT), "note": str(NOTE), "source_words_equal": payload["source_words_equal"], "selection_epochs_equal": payload["selection_epochs_equal"], "tokenizer_sha_equal": payload["tokenizer_sha_equal"]}, indent=2))

if __name__ == "__main__":
    main()
