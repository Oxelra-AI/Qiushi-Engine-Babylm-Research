"""Recompute K=50 component support with exact merge-prefix decompositions."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from tokenizers import Tokenizer

from candidate_sgcr_module import build_prefix_decomposition_map


def count_pool(tokenizer: Tokenizer, pool: Path, vocab_size: int) -> list[int]:
    counts = [0] * vocab_size
    with pool.open(encoding="utf-8") as handle:
        batch: list[str] = []
        for line in handle:
            batch.append(json.loads(line)["text"])
            if len(batch) == 512:
                for encoding in tokenizer.encode_batch(batch, add_special_tokens=False):
                    for token_id in encoding.ids:
                        counts[token_id] += 1
                batch.clear()
        if batch:
            for encoding in tokenizer.encode_batch(batch, add_special_tokens=False):
                for token_id in encoding.ids:
                    counts[token_id] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tok40", type=Path, required=True)
    parser.add_argument("--tok16", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tok40 = Tokenizer.from_file(str(args.tok40 / "tokenizer.json"))
    tok16 = Tokenizer.from_file(str(args.tok16 / "tokenizer.json"))
    tok40.no_padding()
    tok40.no_truncation()
    tok16.no_padding()
    tok16.no_truncation()
    counts40 = count_pool(tok40, args.pool, 40000)
    counts16 = count_pool(tok16, args.pool, 16384)
    exact = build_prefix_decomposition_map(args.tok40, args.tok16)

    decoded = {}
    for piece, token_id in tok40.get_vocab().items():
        text = tok40.decode([token_id], skip_special_tokens=False)
        decoded[token_id] = tok16.encode(text, add_special_tokens=False).ids
    used = [token_id for token_id, count in enumerate(counts40) if count > 0]
    low = [token_id for token_id in used if counts40[token_id] < 50]

    def component_summary(mapping: dict[int, list[int]]) -> dict:
        minima = [min(counts16[c] for c in mapping[token_id]) for token_id in low]
        return {
            "low_support_types": len(low),
            "all_components_ge50_types": sum(value >= 50 for value in minima),
            "all_components_ge50_fraction": sum(value >= 50 for value in minima) / len(minima),
            "minimum_component_support_histogram_coarse": {
                "0-9": sum(value < 10 for value in minima),
                "10-49": sum(10 <= value < 50 for value in minima),
                "50-99": sum(50 <= value < 100 for value in minima),
                "100+": sum(value >= 100 for value in minima),
            },
        }

    rhos = [count / (count + 50.0) for count in counts40]
    total40 = sum(counts40[token_id] for token_id in used)
    type_mean = sum(rhos[token_id] for token_id in used) / len(used)
    mass_mean = sum(rhos[token_id] * counts40[token_id] for token_id in used) / total40
    result = {
        "pool": str(args.pool),
        "counts": {
            "legal40k_total_tokens": sum(counts40),
            "legal16k_total_tokens": sum(counts16),
            "legal40k_used_types": len(used),
            "K50_type_mean_rho": type_mean,
            "K50_mass_weighted_mean_rho": mass_mean,
            "uniform_type_mean_residual_multiplier": 1.0 - type_mean,
            "mass_matched_residual_multiplier": 1.0 - mass_mean,
            "type_mean_vs_mass_matched_residual_ratio": (1.0 - type_mean) / (1.0 - mass_mean),
        },
        "exact_merge_prefix_components": component_summary(exact),
        "unpadded_decode_reencode_components": component_summary(decoded),
        "notes": (
            "Counts disable both raw tokenizer padding and truncation. The exact mapping "
            "recursively splits legal40k merges at the shared legal16k merge frontier."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
