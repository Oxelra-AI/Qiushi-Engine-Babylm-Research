#!/usr/bin/env python3
"""Audit source legal-40k tokenizer vs checkpoint-saved tokenizers.

The research accumulated trainer loads the source tokenizer and saves it inside HF
checkpoints. If save_pretrained reserializes tokenizer.json, raw SHA can differ
while the learned vocabulary and tokenization function remain identical. This
script distinguishes harmless serialization drift from real tokenizer drift.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

ROOT = Path(".").resolve()
WS = ROOT / "experiments/archive/representation_and_objectives"
OUT_DIR = WS / "data/tokenizer_checkpoint_serialization_audit"
EXPECTED_SOURCE_SHA = "94b44bcf5901d097e20294e8220740758eac29af3ec20d58d8da867c85197758"
SOURCE = WS / "data/legal_representation_route_map/tokenizers/legal_byte_bpe_40k"
CANDIDATES = {
    "source_40k": SOURCE,
    "pilot_chck1M": WS / "training/runs/legal40k_accum_pilot_seed43022_1step/hf_model/chck_1M",
    "seed43022_chck1M": WS / "training/runs/legal40k_accum_compact_view_reinvest_seed43022/hf_model/chck_1M",
    "seed43122_chck1M": WS / "training/runs/legal40k_accum_compact_view_reinvest_seed43122/hf_model/chck_1M",
}
TRAIN = ROOT / "experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl"
EVAL_SAMPLE_FILES = [
    WS / "data/globalpiqa_official_lineage/official_dl_scratch/generated_by_current_official_dl/evaluation_data/full_eval/global_piqa_parallel/validation.jsonl",
    WS / "data/globalpiqa_official_lineage/official_dl_scratch/generated_by_current_official_dl/evaluation_data/full_eval/global_piqa_nonparallel/validation.jsonl",
]


def sha_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_jsonable(obj: Any) -> str:
    text = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_text_samples(limit_train: int = 200, limit_eval: int = 200) -> list[str]:
    texts: list[str] = []
    if TRAIN.exists():
        with TRAIN.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if len(texts) >= limit_train:
                    break
                obj = json.loads(line)
                t = str(obj.get("text", ""))
                if t:
                    texts.append(t)
    # Add handcrafted relation/supplement-like strings to test word-starts and punctuation.
    texts.extend([
        "The material becomes brittle when temperature falls below freezing.",
        "A teacher gave the student a book because the lesson had just begun.",
        "Add whipped cream, nuts, and chocolate on top of ice cream.",
        "The child who the parents watched was running near the river.",
        "If the ball is heavier than the feather, it may fall differently in air.",
    ])
    n_eval = 0
    for path in EVAL_SAMPLE_FILES:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if n_eval >= limit_eval:
                    break
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                for v in obj.values():
                    if isinstance(v, str) and v.strip():
                        texts.append(v)
                        n_eval += 1
                        break
                if n_eval >= limit_eval:
                    break
    return texts


def tokenizer_record(label: str, path: Path, texts: list[str]) -> dict[str, Any]:
    tok_json = path / "tokenizer.json"
    if not path.exists() or not tok_json.exists():
        return {"label": label, "path": str(path), "exists": path.exists(), "tokenizer_json_exists": tok_json.exists()}
    tok = AutoTokenizer.from_pretrained(str(path), use_fast=True)
    vocab = tok.get_vocab()
    encodings = []
    unk_count = 0
    total_tokens = 0
    for t in texts:
        ids = tok(t, add_special_tokens=False).input_ids
        encodings.append(ids)
        total_tokens += len(ids)
        if tok.unk_token_id is not None:
            unk_count += sum(1 for x in ids if x == tok.unk_token_id)
    special = {name: getattr(tok, name) for name in ["bos_token", "eos_token", "unk_token", "pad_token", "mask_token"]}
    special_ids = {name: getattr(tok, name + "_id") for name in ["bos_token", "eos_token", "unk_token", "pad_token", "mask_token"]}
    return {
        "label": label,
        "path": str(path),
        "exists": True,
        "tokenizer_json_sha256": sha_file(tok_json),
        "vocab_size_property": tok.vocab_size,
        "len_tokenizer": len(tok),
        "is_fast": tok.is_fast,
        "special_tokens": special,
        "special_token_ids": special_ids,
        "vocab_hash": hash_jsonable(vocab),
        "sample_encodings_hash": hash_jsonable(encodings),
        "sample_count": len(texts),
        "total_sample_tokens": total_tokens,
        "unk_count": unk_count,
        "first_tokens": tok.convert_ids_to_tokens(tok(texts[0], add_special_tokens=False).input_ids[:20]) if texts else [],
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    texts = load_text_samples()
    records = {label: tokenizer_record(label, path, texts) for label, path in CANDIDATES.items()}
    source = records["source_40k"]
    comparisons = {}
    for label, rec in records.items():
        if label == "source_40k" or "vocab_hash" not in rec:
            continue
        comparisons[label] = {
            "raw_tokenizer_sha_matches_source_expected": rec.get("tokenizer_json_sha256") == EXPECTED_SOURCE_SHA,
            "raw_tokenizer_sha_matches_source_record": rec.get("tokenizer_json_sha256") == source.get("tokenizer_json_sha256"),
            "vocab_hash_matches_source": rec.get("vocab_hash") == source.get("vocab_hash"),
            "sample_encodings_hash_matches_source": rec.get("sample_encodings_hash") == source.get("sample_encodings_hash"),
            "special_ids_match_source": rec.get("special_token_ids") == source.get("special_token_ids"),
            "len_matches_source": rec.get("len_tokenizer") == source.get("len_tokenizer"),
            "unk_count": rec.get("unk_count"),
        }
    all_functional = all(
        c.get("vocab_hash_matches_source") and c.get("sample_encodings_hash_matches_source") and c.get("special_ids_match_source") and c.get("len_matches_source")
        for c in comparisons.values()
    ) if comparisons else False
    payload = {
        "status": "TOKENIZER_CHECKPOINT_SERIALIZATION_AUDIT",
        "expected_source_tokenizer_json_sha256": EXPECTED_SOURCE_SHA,
        "source_path": str(SOURCE),
        "sample_count": len(texts),
        "records": records,
        "comparisons_to_source": comparisons,
        "all_existing_checkpoint_tokenizers_functionally_match_source_on_vocab_specials_and_samples": all_functional,
        "interpretation": [
            "Raw tokenizer.json SHA can change under save_pretrained portable serialization; vocab hash, special-token IDs, and sample encodings distinguish harmless serialization drift from learned-tokenizer drift.",
            "If functional hashes match but raw SHA differs, future preflights should accept the checkpoint tokenizer while recording both source and saved tokenizer JSON SHAs.",
        ],
    }
    out_json = OUT_DIR / "tokenizer_checkpoint_serialization_audit.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "all_functional_match": all_functional,
        "comparisons": comparisons,
        "out_json": str(out_json),
    }, indent=2))


if __name__ == "__main__":
    main()
