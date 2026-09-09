#!/usr/bin/env python3
"""Materialize controlled rewrite-pair JSONL arms for BabyLM research.

Goal: create pair-adjacent and pair-shuffled training examples from
GEM/wiki_auto_asset_turk such that the two arms retain exactly the same sentence
multiset, total whitespace-word budget, example count, and tokenized content
length under the baseline 16k tokenizer. This isolates semantic-equivalent
adjacency from distribution/content effects before any BabyLM training.

Construction used here (deliberately conservative):
- Select valid source->target simplification pairs deterministically.
- Each output example contains exactly two sentences separated by a delimiter.
- pair_adjacent example i: source_i + delimiter + target_i.
- pair_shuffled example i: source_i + delimiter + target_perm[i], with a fixed
  deranged permutation. Sources and targets are therefore the exact same
  multisets across arms; all sources remain in the same order; only target
  adjacency is destroyed.
- We retain only rows where both adjacent and shuffled two-sentence examples fit
  completely within max_seq_length under the baseline tokenizer. Because each
  source_i is identical across arms and the selected target multiset is identical,
  the total words are identical by construction; because every complete example
  fits, the trainer will not truncate retained content.

The script writes:
- pair_adjacent JSONL
- pair_shuffled JSONL
- metadata/validation JSON

It does not train. The repaired trainer's --example_jsonl path consumes these
examples exactly in file order with no shuffle/repacking/partial final example.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import pathlib
import random
from dataclasses import dataclass
from typing import Any

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

DATASET_ID = "GEM/wiki_auto_asset_turk"
DATASET_REVISION = "ac2b97468b38cb35fcebe327ac8e1cb6b55b6b99"
DATASET_FILE = "wiki_auto_asset_turk/train-00000-of-00001.parquet"
BASELINE_TOKENIZER_REPO = "BabyLM-community/BabyLM-2026-Baseline-GPT2-Strict"
LICENSES = {
    "WikiAuto": "CC BY-NC 3.0",
    "ASSET": "CC BY-NC 4.0",
    "TurkCorpus": "GPL v3.0",
}
DEFAULT_OUT_DIR = pathlib.Path("experiments/archive/initial_model_studies/data/rewrite_pairs_revision_57")
# Use ordinary sentence adjacency, not an artificial marker that would add a
# non-corpus whitespace word and a special-looking token to every example.
DELIM = " "


@dataclass(frozen=True)
class PairRow:
    pair_id: int
    source: str
    target: str
    source_words: int
    target_words: int


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_text(x: Any) -> str:
    s = " ".join(str(x).replace("\n", " ").split())
    # Avoid ambiguity with our explicit sentence delimiter.
    return s.replace("[SEP]", "SEP")


def word_count(s: str) -> int:
    return len(s.split())


def load_parquet_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    # Prefer pyarrow for exact parquet reading; fall back to pandas if necessary.
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(path)
        return table.to_pylist()
    except Exception:
        import pandas as pd
        return pd.read_parquet(path).to_dict(orient="records")


def token_ids(tok, text: str) -> list[int]:
    return tok(text, add_special_tokens=False, truncation=False)["input_ids"]


def is_word_start(token_str: str) -> bool:
    return token_str.startswith("Ġ") or token_str.startswith("▁")


def count_word_groups(tok, ids: list[int]) -> int:
    special = set(tok.all_special_ids)
    groups = 0
    for j, tid in enumerate(ids):
        if tid in special:
            continue
        token = str(tok.convert_ids_to_tokens(int(tid)))
        if groups == 0 or is_word_start(token) or j == 0:
            groups += 1
    return groups


def multiset_hash(values: list[str]) -> str:
    h = hashlib.sha256()
    for v in sorted(values):
        h.update(v.encode("utf-8")); h.update(b"\n")
    return h.hexdigest()


def write_jsonl(path: pathlib.Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def deranged_permutation(n: int, seed: int, forbidden_identity: set[int]) -> list[int]:
    if n < 2:
        raise ValueError("need at least two pairs")
    rng = random.Random(seed)
    perm = list(range(n))
    for _ in range(10000):
        rng.shuffle(perm)
        if all(perm[i] != i for i in forbidden_identity):
            return perm
    # Deterministic fallback: cyclic shift by an offset coprime enough; avoids identity for all selected rows.
    offset = 1 + (seed % (n - 1))
    perm = [(i + offset) % n for i in range(n)]
    if any(perm[i] == i for i in forbidden_identity):
        raise RuntimeError("failed to construct derangement")
    return perm


def summarize_records(records: list[dict[str, Any]], tok, max_seq_length: int) -> dict[str, Any]:
    total_words = 0
    total_tokens = 0
    total_groups = 0
    max_tokens = 0
    hist = {"<=64": 0, "65-128": 0, "129-256": 0, ">256": 0}
    over = []
    for r in records:
        w = int(r["words"])
        ids = token_ids(tok, r["text"])
        L = len(ids)
        total_words += w
        total_tokens += L
        total_groups += count_word_groups(tok, ids)
        max_tokens = max(max_tokens, L)
        if L <= 64:
            hist["<=64"] += 1
        elif L <= 128:
            hist["65-128"] += 1
        elif L <= 256:
            hist["129-256"] += 1
        else:
            hist[">256"] += 1
        if L > max_seq_length:
            over.append({"example_id": r["example_id"], "token_len": L, "words": w, "pair_ids": r["pair_ids"]})
    return {
        "num_examples": len(records),
        "total_words": total_words,
        "total_tokens": total_tokens,
        "total_word_groups": total_groups,
        "max_tokens": max_tokens,
        "tokens_per_word": total_tokens / max(1, total_words),
        "word_groups_per_word": total_groups / max(1, total_words),
        "expected_wwm_predicted_tokens_at_mask_prob_0p15": 0.15 * total_tokens,
        "expected_wwm_selected_groups_at_mask_prob_0p15": 0.15 * total_groups,
        "length_histogram": hist,
        "num_over_max_seq_length": len(over),
        "over_max_seq_length_examples": over[:20],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--target_words", type=int, default=10000)
    p.add_argument("--max_seq_length", type=int, default=256)
    p.add_argument("--selection_seed", type=int, default=5701)
    p.add_argument("--shuffle_seed", type=int, default=5702)
    p.add_argument("--max_candidate_pairs", type=int, default=200000)
    p.add_argument("--hf_cache_dir", default="experiments/archive/initial_model_studies/training/hf_home")
    args = p.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hf_home = pathlib.Path(args.hf_cache_dir)
    os.environ.setdefault("HF_HOME", str(hf_home.resolve()))
    os.environ.setdefault("HF_HUB_CACHE", str((hf_home / "hub").resolve()))
    os.environ.setdefault("TRANSFORMERS_CACHE", str((hf_home / "transformers").resolve()))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    pathlib.Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.environ["HF_HUB_CACHE"]).mkdir(parents=True, exist_ok=True)

    parquet_path = pathlib.Path(hf_hub_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        revision=DATASET_REVISION,
        filename=DATASET_FILE,
        cache_dir=os.environ["HF_HUB_CACHE"],
    ))
    tok = AutoTokenizer.from_pretrained(BASELINE_TOKENIZER_REPO, revision="main", use_fast=True)
    rows = load_parquet_rows(parquet_path)

    candidates: list[PairRow] = []
    seen = set()
    for idx, row in enumerate(rows):
        if len(candidates) >= args.max_candidate_pairs:
            break
        if "source" not in row or "target" not in row:
            raise RuntimeError(f"expected source/target fields, got {row.keys()}")
        src = normalize_text(row["source"])
        tgt = normalize_text(row["target"])
        if not src or not tgt or src == tgt:
            continue
        key = (src, tgt)
        if key in seen:
            continue
        seen.add(key)
        sw, tw = word_count(src), word_count(tgt)
        # Keep single pair examples compact; later validation checks both arms with shuffled targets.
        if sw == 0 or tw == 0 or sw + tw == 0:
            continue
        if len(token_ids(tok, src + DELIM + tgt)) <= args.max_seq_length:
            candidates.append(PairRow(pair_id=idx, source=src, target=tgt, source_words=sw, target_words=tw))
    if len(candidates) < 100:
        raise RuntimeError(f"too few candidates after filtering: {len(candidates)}")

    rng = random.Random(args.selection_seed)
    rng.shuffle(candidates)
    # Select first n pairs whose adjacent examples fit and whose cumulative words hit target exactly if possible.
    selected: list[PairRow] = []
    total_words = 0
    for pr in candidates:
        w = word_count(pr.source + DELIM + pr.target)
        if total_words + w > args.target_words:
            continue
        selected.append(pr)
        total_words += w
        if total_words == args.target_words:
            break
    # If exact target not reachable by greedy selection, use the achieved exact selected total and require trainer target to match.
    if total_words < max(100, int(0.9 * args.target_words)):
        raise RuntimeError(f"selected too few words {total_words} for target {args.target_words}")

    n = len(selected)
    # Construct a derangement, then filter any shuffled examples that would exceed max seq length. Iterate until stable.
    for round_idx in range(20):
        perm = deranged_permutation(n, args.shuffle_seed + round_idx, set(range(n)))
        keep = []
        for i, pr in enumerate(selected):
            shuffled_tgt = selected[perm[i]].target
            if len(token_ids(tok, pr.source + DELIM + shuffled_tgt)) <= args.max_seq_length:
                keep.append(i)
        if len(keep) == n:
            break
        selected = [selected[i] for i in keep]
        n = len(selected)
        if n < 2:
            raise RuntimeError("filtering for shuffled fit left too few examples")
    else:
        raise RuntimeError("could not find fit-preserving shuffled permutation")

    perm = deranged_permutation(n, args.shuffle_seed + 999, set(range(n)))
    # Final pass: remove any rare overlength shuffled example and rebuild once, keeping both arms identical in retained pairs.
    keep = []
    for i, pr in enumerate(selected):
        if len(token_ids(tok, pr.source + DELIM + selected[perm[i]].target)) <= args.max_seq_length:
            keep.append(i)
    if len(keep) != n:
        selected = [selected[i] for i in keep]
        n = len(selected)
        perm = deranged_permutation(n, args.shuffle_seed + 1000, set(range(n)))

    adjacent_records = []
    shuffled_records = []
    for out_i, pr in enumerate(selected):
        target_j = perm[out_i]
        tgt_shuf = selected[target_j]
        adj_text = pr.source + DELIM + pr.target
        shuf_text = pr.source + DELIM + tgt_shuf.target
        adj_words = word_count(adj_text)
        shuf_words = word_count(shuf_text)
        # DELIM is a single ordinary space, so words are exactly source+target words.
        adjacent_records.append({
            "example_id": out_i,
            "source": "GEM/wiki_auto_asset_turk_pair_adjacent",
            "text": adj_text,
            "words": adj_words,
            "pair_ids": [pr.pair_id],
            "source_pair_id": pr.pair_id,
            "target_pair_id": pr.pair_id,
            "mode": "pair_adjacent",
        })
        shuffled_records.append({
            "example_id": out_i,
            "source": "GEM/wiki_auto_asset_turk_pair_shuffled",
            "text": shuf_text,
            "words": shuf_words,
            "pair_ids": [pr.pair_id, tgt_shuf.pair_id],
            "source_pair_id": pr.pair_id,
            "target_pair_id": tgt_shuf.pair_id,
            "mode": "pair_shuffled",
        })

    adj_sum = summarize_records(adjacent_records, tok, args.max_seq_length)
    shuf_sum = summarize_records(shuffled_records, tok, args.max_seq_length)
    source_texts_adj = [selected[int(r["source_pair_id"]) if False else i].source for i, r in enumerate(adjacent_records)]
    # The selected list order gives all source and adjacent target texts. Shuffled target texts are a permutation.
    selected_sources = [pr.source for pr in selected]
    selected_targets = [pr.target for pr in selected]
    shuffled_targets = [selected[perm[i]].target for i in range(n)]

    validation = {
        "identical_source_multiset": collections.Counter(selected_sources) == collections.Counter(selected_sources),
        "identical_target_multiset": collections.Counter(selected_targets) == collections.Counter(shuffled_targets),
        "identical_sentence_multiset": collections.Counter(selected_sources + selected_targets) == collections.Counter(selected_sources + shuffled_targets),
        "identical_example_count": len(adjacent_records) == len(shuffled_records),
        "identical_total_words": adj_sum["total_words"] == shuf_sum["total_words"],
        "no_adjacent_example_over_256": adj_sum["num_over_max_seq_length"] == 0,
        "no_shuffled_example_over_256": shuf_sum["num_over_max_seq_length"] == 0,
        "no_identity_target_adjacency_in_shuffled": all(perm[i] != i for i in range(n)),
        "token_count_delta_total": shuf_sum["total_tokens"] - adj_sum["total_tokens"],
        "word_group_delta_total": shuf_sum["total_word_groups"] - adj_sum["total_word_groups"],
        "expected_predicted_token_delta_at_0p15": shuf_sum["expected_wwm_predicted_tokens_at_mask_prob_0p15"] - adj_sum["expected_wwm_predicted_tokens_at_mask_prob_0p15"],
        "expected_group_delta_at_0p15": shuf_sum["expected_wwm_selected_groups_at_mask_prob_0p15"] - adj_sum["expected_wwm_selected_groups_at_mask_prob_0p15"],
        "source_multiset_hash": multiset_hash(selected_sources),
        "target_multiset_hash": multiset_hash(selected_targets),
        "shuffled_target_multiset_hash": multiset_hash(shuffled_targets),
    }
    hard_ok = all(bool(validation[k]) for k in [
        "identical_target_multiset", "identical_sentence_multiset", "identical_example_count",
        "identical_total_words", "no_adjacent_example_over_256", "no_shuffled_example_over_256",
        "no_identity_target_adjacency_in_shuffled",
    ])
    if not hard_ok:
        raise RuntimeError(f"hard validation failed: {validation}")

    label = f"target{args.target_words}_actual{adj_sum['total_words']}_n{n}_sel{args.selection_seed}_shuf{args.shuffle_seed}"
    adj_path = out_dir / f"pair_adjacent_{label}.jsonl"
    shuf_path = out_dir / f"pair_shuffled_{label}.jsonl"
    meta_path = out_dir / f"pair_materialization_{label}.json"
    write_jsonl(adj_path, adjacent_records)
    write_jsonl(shuf_path, shuffled_records)

    meta = {
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "dataset_file": DATASET_FILE,
        "dataset_file_cache_path": str(parquet_path),
        "dataset_file_sha256": sha256_file(parquet_path),
        "licenses": LICENSES,
        "baseline_tokenizer_repo": BASELINE_TOKENIZER_REPO,
        "delimiter": DELIM,
        "target_words_requested": args.target_words,
        "actual_words": adj_sum["total_words"],
        "max_seq_length": args.max_seq_length,
        "selection_seed": args.selection_seed,
        "shuffle_seed": args.shuffle_seed,
        "num_selected_pairs": n,
        "pair_adjacent_path": str(adj_path),
        "pair_shuffled_path": str(shuf_path),
        "pair_adjacent_sha256": sha256_file(adj_path),
        "pair_shuffled_sha256": sha256_file(shuf_path),
        "adjacent_summary": adj_sum,
        "shuffled_summary": shuf_sum,
        "validation": validation,
        "selected_pair_ids_first20": [pr.pair_id for pr in selected[:20]],
        "shuffled_target_indices_first20": perm[:20],
        "note": "Both JSONL files are intended for babylm_masked_train.py --example_jsonl; set --max_word_exposure to actual_words to avoid partial examples.",
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "REWRITE_PAIR_JSONL_READY",
        "actual_words": adj_sum["total_words"],
        "num_examples": n,
        "pair_adjacent_path": str(adj_path),
        "pair_shuffled_path": str(shuf_path),
        "meta_path": str(meta_path),
        "validation": validation,
        "adjacent_summary": adj_sum,
        "shuffled_summary": shuf_sum,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
