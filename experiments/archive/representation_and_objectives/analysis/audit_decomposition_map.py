"""Quantify defects in research's decode/re-encode decomposition map."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer

from candidate_sgcr_module import build_prefix_decomposition_map


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tok40", type=Path, required=True)
    parser.add_argument("--tok16", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tok40_json = args.tok40 / "tokenizer.json"
    tok16_json = args.tok16 / "tokenizer.json"
    tok40 = Tokenizer.from_file(str(tok40_json))
    tok16 = Tokenizer.from_file(str(tok16_json))
    vocab40 = {token_id: piece for piece, token_id in tok40.get_vocab().items()}
    exact = build_prefix_decomposition_map(args.tok40, args.tok16)

    texts = [tok40.decode([token_id], skip_special_tokens=False) for token_id in range(len(vocab40))]
    encoded = []
    for start in range(0, len(texts), 1024):
        encoded.extend(tok16.encode_batch(texts[start : start + 1024], add_special_tokens=False))
    current = [encoding.ids for encoding in encoded]
    attention_filtered = [
        [token_id for token_id, keep in zip(encoding.ids, encoding.attention_mask) if keep]
        for encoding in encoded
    ]
    mismatches = [
        token_id
        for token_id in range(len(vocab40))
        if attention_filtered[token_id] != exact[token_id]
    ]
    total_slots = sum(map(len, current))
    padding_slots = sum(
        len(encoding.ids) - sum(encoding.attention_mask) for encoding in encoded
    )
    examples = []
    for token_id in mismatches[:25]:
        examples.append(
            {
                "legal40k_id": token_id,
                "legal40k_piece": vocab40[token_id],
                "isolated_decoded_text": texts[token_id],
                "exact_component_ids": exact[token_id],
                "decode_reencode_active_ids": attention_filtered[token_id],
            }
        )
    result = {
        "tokenizer_sha256": {
            "legal40k": sha256(tok40_json),
            "legal16k": sha256(tok16_json),
        },
        "raw_legal16k_runtime_settings": {
            "padding": tok16.padding,
            "truncation": tok16.truncation,
        },
        "current_step084_map": {
            "token_count": len(current),
            "stored_length_histogram": dict(sorted(Counter(map(len, current)).items())),
            "total_stored_component_slots": total_slots,
            "padding_component_slots": padding_slots,
            "padding_component_fraction": padding_slots / total_slots,
            "attention_filtered_length_histogram": dict(
                sorted(Counter(map(len, attention_filtered)).items())
            ),
        },
        "exact_merge_prefix_map": {
            "length_histogram": dict(sorted(Counter(map(len, exact.values())).items())),
            "maximum_components": max(map(len, exact.values())),
            "empty_decompositions": sum(not values for values in exact.values()),
            "all_component_ids_below_16384": all(
                0 <= component < 16384
                for values in exact.values()
                for component in values
            ),
        },
        "decode_reencode_vs_exact": {
            "mismatched_tokens": len(mismatches),
            "mismatch_fraction": len(mismatches) / len(current),
            "matching_tokens": len(current) - len(mismatches),
            "examples": examples,
        },
        "conclusion": (
            "research stores 256 ids per token because padding is enabled, and even "
            "filtering padding does not repair context-sensitive/lossy isolated byte-level "
            "decode/re-encode. Use exact recursive splitting at the shared 16k BPE merge prefix."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
