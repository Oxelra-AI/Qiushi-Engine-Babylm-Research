#!/usr/bin/env python3
"""research: compare expected legal tokenizer with saved endpoint tokenizer.

This resolves whether a raw tokenizer.json SHA mismatch blocks evaluation or is only
serialization drift after HuggingFace save_pretrained.  The check is deliberately
semantic: vocab, BPE merges/model settings, normalizer/pre-tokenizer/decoder,
special token maps/config, and encode/decode agreement on stress strings and a
small deterministic corpus/evaluation sample.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import hashlib
import json
import pathlib
import random
import time
from typing import Any

from transformers import AutoTokenizer


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
STUDY = USER_ROOT / "experiments/archive/frontier_consolidation"
WORKSPACE = STUDY
DEFAULT_EXPECTED = WORKSPACE / "data/compliant_tokenizer"
DEFAULT_SAVED = WORKSPACE / "training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_100M"
DEFAULT_OUT = WORKSPACE / "data/tokenizer_semantic_equivalence"
DEFAULT_POOL = WORKSPACE / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.resolve().relative_to(USER_ROOT))
    except Exception:
        return str(p)


def token_json_summary(path: pathlib.Path) -> dict[str, Any]:
    obj = load_json(path)
    model = obj.get("model", {}) if isinstance(obj, dict) else {}
    return {
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "top_keys": sorted(obj.keys()) if isinstance(obj, dict) else None,
        "model_type": model.get("type"),
        "vocab_size": len(model.get("vocab", {})) if isinstance(model.get("vocab"), dict) else None,
        "merge_count": len(model.get("merges", [])) if isinstance(model.get("merges"), list) else None,
        "unk_token": model.get("unk_token"),
        "byte_fallback": model.get("byte_fallback"),
        "continuing_subword_prefix": model.get("continuing_subword_prefix"),
        "end_of_word_suffix": model.get("end_of_word_suffix"),
        "dropout": model.get("dropout"),
        "normalizer": obj.get("normalizer") if isinstance(obj, dict) else None,
        "pre_tokenizer": obj.get("pre_tokenizer") if isinstance(obj, dict) else None,
        "post_processor": obj.get("post_processor") if isinstance(obj, dict) else None,
        "decoder": obj.get("decoder") if isinstance(obj, dict) else None,
        "added_tokens": obj.get("added_tokens") if isinstance(obj, dict) else None,
    }


def canonical_model_fingerprint(tok_json: pathlib.Path) -> str:
    obj = load_json(tok_json)
    model = obj.get("model", {})
    keep = {
        "model": {k: model.get(k) for k in sorted(model.keys())},
        "normalizer": obj.get("normalizer"),
        "pre_tokenizer": obj.get("pre_tokenizer"),
        "post_processor": obj.get("post_processor"),
        "decoder": obj.get("decoder"),
        "added_tokens": obj.get("added_tokens"),
    }
    data = json.dumps(keep, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def sample_pool_lines(path: pathlib.Path, n: int = 512, seed: int = 48048) -> list[str]:
    rng = random.Random(seed)
    # Reservoir sample text fields without loading the whole file.
    sample: list[str] = []
    seen = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            seen += 1
            try:
                txt = str(json.loads(line).get("text", ""))
            except Exception:
                continue
            if len(sample) < n:
                sample.append(txt)
            else:
                j = rng.randrange(seen)
                if j < n:
                    sample[j] = txt
    return sample


def encoding_agreement(expected_dir: pathlib.Path, saved_dir: pathlib.Path, pool: pathlib.Path, max_pool: int) -> dict[str, Any]:
    tok_a = AutoTokenizer.from_pretrained(str(expected_dir), use_fast=True)
    tok_b = AutoTokenizer.from_pretrained(str(saved_dir), use_fast=True)
    stress = [
        "Hello world.",
        "A\nB",
        "Speaker A:\nSpeaker B:",
        "What is the answer?\nIt is not obvious.",
        "The box is on the table, beside the lamp, and under the shelf.",
        "I can't believe the 3.14-year-old co-operates—really!",
        "\t leading tab and trailing spaces   ",
        "中文 Русский عربى emoji🙂 newline\n",
        "<s> <mask> </s> <pad> <unk>",
    ]
    samples = stress + sample_pool_lines(pool, n=max_pool)
    mismatches = []
    for i, text in enumerate(samples):
        a = tok_a(text, add_special_tokens=False)["input_ids"]
        b = tok_b(text, add_special_tokens=False)["input_ids"]
        if a != b:
            mismatches.append({
                "index": i,
                "kind": "stress" if i < len(stress) else "pool",
                "text_head": text[:200],
                "ids_expected_head": a[:80],
                "ids_saved_head": b[:80],
                "len_expected": len(a),
                "len_saved": len(b),
            })
            if len(mismatches) >= 20:
                break
    special_equal = {
        "vocab_size_equal": len(tok_a.get_vocab()) == len(tok_b.get_vocab()),
        "vocab_dict_equal": tok_a.get_vocab() == tok_b.get_vocab(),
        "all_special_tokens_equal": tok_a.all_special_tokens == tok_b.all_special_tokens,
        "all_special_ids_equal": tok_a.all_special_ids == tok_b.all_special_ids,
        "special_tokens_map_equal": tok_a.special_tokens_map == tok_b.special_tokens_map,
        "model_max_length_equal": getattr(tok_a, "model_max_length", None) == getattr(tok_b, "model_max_length", None),
    }
    return {
        "sample_count": len(samples),
        "stress_count": len(stress),
        "pool_sample_count": len(samples) - len(stress),
        "mismatch_count_capped": len(mismatches),
        "mismatches_head": mismatches,
        "encoding_all_checked_equal": len(mismatches) == 0,
        "special_equal": special_equal,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expected", default=str(DEFAULT_EXPECTED))
    ap.add_argument("--saved", default=str(DEFAULT_SAVED))
    ap.add_argument("--pool", default=str(DEFAULT_POOL))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--max-pool-samples", type=int, default=512)
    args = ap.parse_args()

    expected = pathlib.Path(args.expected)
    saved = pathlib.Path(args.saved)
    pool = pathlib.Path(args.pool)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    expected_tok_json = expected / "tokenizer.json"
    saved_tok_json = saved / "tokenizer.json"
    expected_summary = token_json_summary(expected_tok_json)
    saved_summary = token_json_summary(saved_tok_json)
    expected_obj = load_json(expected_tok_json)
    saved_obj = load_json(saved_tok_json)
    expected_model = expected_obj.get("model", {})
    saved_model = saved_obj.get("model", {})

    checks = {
        "raw_sha_equal": expected_summary["sha256"] == saved_summary["sha256"],
        "canonical_fingerprint_equal": canonical_model_fingerprint(expected_tok_json) == canonical_model_fingerprint(saved_tok_json),
        "model_vocab_equal": expected_model.get("vocab") == saved_model.get("vocab"),
        "model_merges_equal": expected_model.get("merges") == saved_model.get("merges"),
        "model_settings_equal": {k: expected_model.get(k) == saved_model.get(k) for k in sorted(set(expected_model) | set(saved_model)) if k not in {"vocab", "merges"}},
        "normalizer_equal": expected_obj.get("normalizer") == saved_obj.get("normalizer"),
        "pre_tokenizer_equal": expected_obj.get("pre_tokenizer") == saved_obj.get("pre_tokenizer"),
        "post_processor_equal": expected_obj.get("post_processor") == saved_obj.get("post_processor"),
        "decoder_equal": expected_obj.get("decoder") == saved_obj.get("decoder"),
        "added_tokens_equal": expected_obj.get("added_tokens") == saved_obj.get("added_tokens"),
    }
    agreement = encoding_agreement(expected, saved, pool, args.max_pool_samples)
    semantic_equivalent = bool(
        checks["model_vocab_equal"]
        and checks["model_merges_equal"]
        and all(checks["model_settings_equal"].values())
        and checks["normalizer_equal"]
        and checks["pre_tokenizer_equal"]
        and checks["post_processor_equal"]
        and checks["decoder_equal"]
        and checks["added_tokens_equal"]
        and agreement["encoding_all_checked_equal"]
        and agreement["special_equal"].get("vocab_dict_equal")
        and agreement["special_equal"].get("all_special_ids_equal")
    )

    payload = {
        "status": "TOKENIZER_SEMANTIC_EQUIVALENCE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "expected_dir": rel(expected),
        "saved_dir": rel(saved),
        "pool_sample_source": rel(pool),
        "expected_tokenizer_json": expected_summary,
        "saved_tokenizer_json": saved_summary,
        "canonical_fingerprint_expected": canonical_model_fingerprint(expected_tok_json),
        "canonical_fingerprint_saved": canonical_model_fingerprint(saved_tok_json),
        "checks": checks,
        "encoding_agreement": agreement,
        "semantic_equivalent_for_evaluation": semantic_equivalent,
        "interpretation": "If semantic_equivalent_for_evaluation is true, the raw tokenizer.json SHA mismatch is save_pretrained serialization drift and should not block scoring, provided training-command tokenizer metadata still binds the tokenizer to the allowed 10M pool.",
    }
    out_json = out_dir / "tokenizer_semantic_equivalence.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_md = out_dir / "tokenizer_semantic_equivalence.md"
    out_md.write_text(
        "# research tokenizer semantic equivalence\n\n"
        f"Expected: `{payload['expected_dir']}`\n\n"
        f"Saved: `{payload['saved_dir']}`\n\n"
        f"Raw SHA equal: {checks['raw_sha_equal']}\n\n"
        f"Expected SHA: `{expected_summary['sha256']}`\n\n"
        f"Saved SHA: `{saved_summary['sha256']}`\n\n"
        f"Canonical fingerprint equal: {checks['canonical_fingerprint_equal']}\n\n"
        f"Vocab equal: {checks['model_vocab_equal']}\n\n"
        f"Merges equal: {checks['model_merges_equal']}\n\n"
        f"Encoding mismatches in checked strings: {agreement['mismatch_count_capped']} / {agreement['sample_count']} (capped at 20)\n\n"
        f"Semantic equivalent for evaluation: **{semantic_equivalent}**\n\n"
        f"Full JSON: `{rel(out_json)}`\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": payload["status"],
        "semantic_equivalent_for_evaluation": semantic_equivalent,
        "raw_sha_equal": checks["raw_sha_equal"],
        "canonical_fingerprint_equal": checks["canonical_fingerprint_equal"],
        "vocab_equal": checks["model_vocab_equal"],
        "merges_equal": checks["model_merges_equal"],
        "encoding_mismatch_count_capped": agreement["mismatch_count_capped"],
        "out_json": rel(out_json),
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
