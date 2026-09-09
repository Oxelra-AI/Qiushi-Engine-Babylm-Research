#!/usr/bin/env python3
"""Materialize official+rewrite 50/50 mixture JSONLs for BabyLM research.

This script is intentionally isolated from the trainer. It creates exact JSONL
examples for a controlled data-composition experiment after research/59 showed
that rewrite adjacency improves EWoK/Entity modestly but pure WikiAuto-pair data
hurts broad BabyLM performance.

For each seed:
- Reconstruct official BabyLM 2026 Strict-Small examples exactly like the masked
  trainer: 10M official pool, 160-word examples, seed-specific shuffle.
- Retain exactly 500,000 official words (3,125 complete official examples).
- Select about 500,000 complete GEM/wiki_auto_asset_turk source->target pairs.
- Build mixture_adjacent: official examples + source_i target_i pair examples.
- Build mixture_shuffled: exact same official examples + same source and target
  sentence multisets, but target_i is deranged within the selected pair subset.
- Use a shared condition-independent shuffled slot order for adjacent/shuffled so
  example order differs only in the target text attached to each pair source.
- Validate under the baseline 16k tokenizer: pair examples have zero >256-token
  cases, official examples are identical across arms, total words/example count
  match, and aggregate kept tokens/WWM groups/expected mask opportunity match.

The repaired trainer should consume these JSONLs with --example_jsonl and the
actual total word count recorded in the per-seed metadata.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import pathlib
import random
from dataclasses import dataclass
from typing import Any, Iterable

from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoTokenizer

OFFICIAL_DATASET_ID = "BabyLM-community/BabyLM-2026-Strict-Small"
OFFICIAL_DATASET_REVISION = "c92ab16b4f08858304b0815706065b3354d8fc0a"
TRAIN_FILES = [
    "bnc_spoken.train.txt",
    "childes.train.txt",
    "gutenberg.train.txt",
    "open_subtitles.train.txt",
    "simple_wiki.train.txt",
    "switchboard.train.txt",
]
GEM_DATASET_ID = "GEM/wiki_auto_asset_turk"
GEM_DATASET_REVISION = "ac2b97468b38cb35fcebe327ac8e1cb6b55b6b99"
GEM_DATASET_FILE = "wiki_auto_asset_turk/train-00000-of-00001.parquet"
BASELINE_TOKENIZER_REPO = "BabyLM-community/BabyLM-2026-Baseline-GPT2-Strict"
LICENSES = {
    "WikiAuto": "CC BY-NC 3.0",
    "ASSET": "CC BY-NC 4.0",
    "TurkCorpus": "GPL v3.0",
}
DEFAULT_OUT_DIR = pathlib.Path("experiments/archive/initial_model_studies/data/mixture_revision_61")
DELIM = " "


@dataclass
class Example:
    text: str
    words: int
    example_id: int = -1
    source: str = ""


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


def count_words_in_file(path: pathlib.Path) -> int:
    total = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            total += len(line.split())
    return total


def word_count(s: str) -> int:
    return len(s.split())


def normalize_text(x: Any) -> str:
    return " ".join(str(x).replace("\n", " ").replace("[SEP]", "SEP").split())


def iter_examples(files: list[pathlib.Path], max_words: int, words_per_example: int) -> Iterable[Example]:
    used = 0
    buf: list[str] = []
    buf_source = ""
    for fp in files:
        with fp.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                for w in line.split():
                    if used >= max_words:
                        break
                    if not buf:
                        buf_source = fp.name
                    buf.append(w)
                    used += 1
                    if len(buf) >= words_per_example:
                        yield Example(" ".join(buf), len(buf), source=buf_source)
                        buf = []
                        buf_source = ""
                if used >= max_words:
                    break
        if used >= max_words:
            break
    if buf:
        yield Example(" ".join(buf), len(buf), source=buf_source)


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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def load_parquet_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    except Exception:
        import pandas as pd
        return pd.read_parquet(path).to_dict(orient="records")


def download_official(out_dir: pathlib.Path) -> tuple[pathlib.Path, list[dict[str, Any]]]:
    raw_dir = out_dir / "official_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    local = pathlib.Path(snapshot_download(
        repo_id=OFFICIAL_DATASET_ID,
        repo_type="dataset",
        revision=OFFICIAL_DATASET_REVISION,
        allow_patterns=TRAIN_FILES + ["README.md"],
        local_dir=raw_dir,
        local_dir_use_symlinks=False,
    ))
    manifest = []
    for name in TRAIN_FILES:
        p = local / name
        if not p.exists():
            raise FileNotFoundError(p)
        manifest.append({
            "path": str(p),
            "name": name,
            "bytes": p.stat().st_size,
            "sha256": sha256_file(p),
            "whitespace_words": count_words_in_file(p),
        })
    return local, manifest


def select_official_examples(files: list[pathlib.Path], seed: int, pool_words: int, target_words: int, words_per_example: int) -> list[Example]:
    pool = list(iter_examples(files, pool_words, words_per_example))
    for i, ex in enumerate(pool):
        ex.example_id = i
    pool_actual = sum(ex.words for ex in pool)
    if pool_actual != pool_words:
        raise RuntimeError(f"official pool word mismatch {pool_actual} vs {pool_words}")
    rng = random.Random(seed)
    rng.shuffle(pool)
    selected: list[Example] = []
    total = 0
    for ex in pool:
        if total + ex.words > target_words:
            continue
        selected.append(ex)
        total += ex.words
        if total == target_words:
            break
    if total != target_words:
        raise RuntimeError(f"official selection total {total} != target {target_words}")
    return selected


def deranged_permutation(n: int, seed: int) -> list[int]:
    if n < 2:
        raise RuntimeError("need at least two pairs for derangement")
    rng = random.Random(seed)
    perm = list(range(n))
    for _ in range(10000):
        rng.shuffle(perm)
        if all(perm[i] != i for i in range(n)):
            return perm
    offset = 1 + (seed % (n - 1))
    perm = [(i + offset) % n for i in range(n)]
    if any(perm[i] == i for i in range(n)):
        raise RuntimeError("failed to construct derangement")
    return perm


def load_pair_candidates(parquet_path: pathlib.Path, tok, max_seq_length: int, max_candidate_pairs: int) -> list[PairRow]:
    rows = load_parquet_rows(parquet_path)
    candidates: list[PairRow] = []
    seen = set()
    for idx, row in enumerate(rows):
        if len(candidates) >= max_candidate_pairs:
            break
        src = normalize_text(row.get("source", ""))
        tgt = normalize_text(row.get("target", ""))
        if not src or not tgt or src == tgt:
            continue
        key = (src, tgt)
        if key in seen:
            continue
        seen.add(key)
        sw, tw = word_count(src), word_count(tgt)
        if sw <= 0 or tw <= 0:
            continue
        if len(token_ids(tok, src + DELIM + tgt)) <= max_seq_length:
            candidates.append(PairRow(idx, src, tgt, sw, tw))
    if len(candidates) < 100:
        raise RuntimeError(f"too few pair candidates: {len(candidates)}")
    return candidates


def select_pairs_for_seed(candidates: list[PairRow], tok, target_words: int, max_seq_length: int, selection_seed: int, shuffle_seed: int) -> tuple[list[PairRow], list[int], int]:
    shuffled = list(candidates)
    random.Random(selection_seed).shuffle(shuffled)
    selected: list[PairRow] = []
    total = 0
    for pr in shuffled:
        w = word_count(pr.source + DELIM + pr.target)
        if total + w > target_words:
            continue
        selected.append(pr)
        total += w
        if total == target_words:
            break
    if total < int(0.95 * target_words):
        raise RuntimeError(f"selected too few pair words {total} for target {target_words}")

    # Remove any selected pair whose shuffled target would make an overlength example, then rebuild derangement.
    for attempt in range(30):
        perm = deranged_permutation(len(selected), shuffle_seed + attempt)
        keep = []
        for i, pr in enumerate(selected):
            shuf_tgt = selected[perm[i]].target
            if len(token_ids(tok, pr.source + DELIM + shuf_tgt)) <= max_seq_length:
                keep.append(i)
        if len(keep) == len(selected):
            return selected, perm, sum(word_count(pr.source + DELIM + pr.target) for pr in selected)
        selected = [selected[i] for i in keep]
        if len(selected) < 2:
            raise RuntimeError("overlength filtering left too few selected pairs")
    raise RuntimeError("could not find fit-preserving pair derangement")


def summarize_records(records: list[dict[str, Any]], tok, max_seq_length: int) -> dict[str, Any]:
    total_words = 0
    total_untruncated = 0
    total_kept = 0
    total_groups = 0
    max_untruncated = 0
    max_kept = 0
    over = []
    hist = {"<=64": 0, "65-128": 0, "129-256": 0, ">256": 0}
    source_words: dict[str, int] = {}
    for r in records:
        w = int(r["words"])
        ids = token_ids(tok, r["text"])
        kept = ids[:max_seq_length]
        L = len(ids)
        K = len(kept)
        total_words += w
        total_untruncated += L
        total_kept += K
        total_groups += count_word_groups(tok, kept)
        max_untruncated = max(max_untruncated, L)
        max_kept = max(max_kept, K)
        source_words[r["source"]] = source_words.get(r["source"], 0) + w
        if L <= 64:
            hist["<=64"] += 1
        elif L <= 128:
            hist["65-128"] += 1
        elif L <= 256:
            hist["129-256"] += 1
        else:
            hist[">256"] += 1
            over.append({"example_id": r["example_id"], "source": r["source"], "tokens": L, "words": w})
    return {
        "num_examples": len(records),
        "total_words": total_words,
        "total_untruncated_tokens": total_untruncated,
        "total_kept_tokens_at_max_seq_length": total_kept,
        "total_word_groups_kept_at_max_seq_length": total_groups,
        "tokens_lost_to_truncation": total_untruncated - total_kept,
        "max_untruncated_tokens": max_untruncated,
        "max_kept_tokens": max_kept,
        "untruncated_tokens_per_word": total_untruncated / max(1, total_words),
        "kept_tokens_per_word": total_kept / max(1, total_words),
        "word_groups_per_word": total_groups / max(1, total_words),
        "expected_wwm_selected_groups_at_mask_prob_0p15": 0.15 * total_groups,
        "expected_wwm_predicted_tokens_at_mask_prob_0p15_if_selected_groups_cover_tokens": 0.15 * total_kept,
        "truncated_examples_at_max_seq_length": len(over),
        "truncated_example_fraction": len(over) / max(1, len(records)),
        "untruncated_length_histogram": hist,
        "source_words": source_words,
        "over_max_examples_first20": over[:20],
    }


def records_for_seed(seed: int, official_examples: list[Example], selected_pairs: list[PairRow], perm: list[int], slot_seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    official_records = []
    for j, ex in enumerate(official_examples):
        official_records.append({
            "kind": "official",
            "slot_key": f"official:{j}",
            "source": f"official::{ex.source}",
            "text": ex.text,
            "words": ex.words,
            "official_example_id": ex.example_id,
            "official_source_file": ex.source,
        })
    adj_pair_records = []
    shuf_pair_records = []
    for j, pr in enumerate(selected_pairs):
        tgt = selected_pairs[perm[j]]
        adj_text = pr.source + DELIM + pr.target
        shuf_text = pr.source + DELIM + tgt.target
        adj_pair_records.append({
            "kind": "pair",
            "slot_key": f"pair:{j}",
            "source": "GEM/wiki_auto_asset_turk_mixture_pair_adjacent",
            "text": adj_text,
            "words": word_count(adj_text),
            "source_pair_id": pr.pair_id,
            "target_pair_id": pr.pair_id,
            "mode": "mixture_adjacent",
        })
        shuf_pair_records.append({
            "kind": "pair",
            "slot_key": f"pair:{j}",
            "source": "GEM/wiki_auto_asset_turk_mixture_pair_shuffled",
            "text": shuf_text,
            "words": word_count(shuf_text),
            "source_pair_id": pr.pair_id,
            "target_pair_id": tgt.pair_id,
            "mode": "mixture_shuffled",
        })

    slots = [("official", i) for i in range(len(official_records))] + [("pair", i) for i in range(len(selected_pairs))]
    random.Random(slot_seed).shuffle(slots)
    adj = []
    shuf = []
    for out_i, (kind, idx) in enumerate(slots):
        if kind == "official":
            base_adj = dict(official_records[idx])
            base_shuf = dict(official_records[idx])
        else:
            base_adj = dict(adj_pair_records[idx])
            base_shuf = dict(shuf_pair_records[idx])
        for r in (base_adj, base_shuf):
            r["example_id"] = out_i
            r["mixture_seed"] = seed
        adj.append(base_adj)
        shuf.append(base_shuf)

    aux = {
        "slot_seed": slot_seed,
        "slots_first30": [f"{k}:{i}" for k, i in slots[:30]],
        "official_example_ids_first20": [ex.example_id for ex in official_examples[:20]],
        "selected_pair_ids_first20": [pr.pair_id for pr in selected_pairs[:20]],
        "shuffled_target_indices_first20": perm[:20],
    }
    return adj, shuf, aux


def recommended_batch_size(num_examples: int, target_steps: int) -> dict[str, int]:
    exact = []
    for b in range(1, num_examples + 1):
        if math.ceil(num_examples / b) == target_steps:
            exact.append(b)
    if exact:
        b = min(exact)
        return {"batch_size": b, "actual_steps": math.ceil(num_examples / b), "target_steps": target_steps, "exact_target_steps": True}
    b = math.ceil(num_examples / target_steps)
    return {"batch_size": b, "actual_steps": math.ceil(num_examples / b), "target_steps": target_steps, "exact_target_steps": False}


def validate_pair_records(adj: list[dict[str, Any]], shuf: list[dict[str, Any]], tok, max_seq_length: int) -> dict[str, Any]:
    adj_off = [r for r in adj if r["kind"] == "official"]
    shuf_off = [r for r in shuf if r["kind"] == "official"]
    adj_pair = [r for r in adj if r["kind"] == "pair"]
    shuf_pair = [r for r in shuf if r["kind"] == "pair"]
    official_same = [(r["slot_key"], r["text"], r["words"], r["official_example_id"]) for r in adj_off] == [(r["slot_key"], r["text"], r["words"], r["official_example_id"]) for r in shuf_off]
    adj_pair_sources = [r["text"].split(DELIM, 1)[0] for r in adj_pair]
    # More reliable from records metadata: source_pair_id sequence same across adjacent/shuffled pair slots.
    source_pair_ids_same = [r["source_pair_id"] for r in adj_pair] == [r["source_pair_id"] for r in shuf_pair]
    target_ids_adj = [r["target_pair_id"] for r in adj_pair]
    target_ids_shuf = [r["target_pair_id"] for r in shuf_pair]
    no_identity = all(r["source_pair_id"] != r["target_pair_id"] for r in shuf_pair)
    pair_adj_over = [r for r in adj_pair if len(token_ids(tok, r["text"])) > max_seq_length]
    pair_shuf_over = [r for r in shuf_pair if len(token_ids(tok, r["text"])) > max_seq_length]
    adj_summary = summarize_records(adj, tok, max_seq_length)
    shuf_summary = summarize_records(shuf, tok, max_seq_length)
    validation = {
        "identical_official_records": official_same,
        "source_pair_ids_same_order": source_pair_ids_same,
        "identical_target_pair_id_multiset": collections.Counter(target_ids_adj) == collections.Counter(target_ids_shuf),
        "identical_combined_example_count": len(adj) == len(shuf),
        "identical_total_words": adj_summary["total_words"] == shuf_summary["total_words"],
        "no_identity_target_adjacency_in_shuffled_pairs": no_identity,
        "pair_adjacent_zero_over_256": len(pair_adj_over) == 0,
        "pair_shuffled_zero_over_256": len(pair_shuf_over) == 0,
        "combined_kept_token_delta_total": shuf_summary["total_kept_tokens_at_max_seq_length"] - adj_summary["total_kept_tokens_at_max_seq_length"],
        "combined_word_group_delta_total": shuf_summary["total_word_groups_kept_at_max_seq_length"] - adj_summary["total_word_groups_kept_at_max_seq_length"],
        "expected_predicted_token_delta_at_0p15": shuf_summary["expected_wwm_predicted_tokens_at_mask_prob_0p15_if_selected_groups_cover_tokens"] - adj_summary["expected_wwm_predicted_tokens_at_mask_prob_0p15_if_selected_groups_cover_tokens"],
        "expected_group_delta_at_0p15": shuf_summary["expected_wwm_selected_groups_at_mask_prob_0p15"] - adj_summary["expected_wwm_selected_groups_at_mask_prob_0p15"],
        "official_example_id_hash": multiset_hash([str(r["official_example_id"]) for r in adj_off]),
        "target_pair_id_hash_adjacent": multiset_hash([str(x) for x in target_ids_adj]),
        "target_pair_id_hash_shuffled": multiset_hash([str(x) for x in target_ids_shuf]),
    }
    required = [
        "identical_official_records",
        "source_pair_ids_same_order",
        "identical_target_pair_id_multiset",
        "identical_combined_example_count",
        "identical_total_words",
        "no_identity_target_adjacency_in_shuffled_pairs",
        "pair_adjacent_zero_over_256",
        "pair_shuffled_zero_over_256",
    ]
    validation["hard_validation_ok"] = all(bool(validation[k]) for k in required)
    validation["aggregate_opportunity_identical"] = (validation["combined_kept_token_delta_total"] == 0 and validation["combined_word_group_delta_total"] == 0)
    return {"validation": validation, "adjacent_summary": adj_summary, "shuffled_summary": shuf_summary}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--official_words", type=int, default=500000)
    p.add_argument("--pair_words", type=int, default=500000)
    p.add_argument("--official_pool_words", type=int, default=10000000)
    p.add_argument("--words_per_official_example", type=int, default=160)
    p.add_argument("--max_seq_length", type=int, default=256)
    p.add_argument("--target_steps", type=int, default=98)
    p.add_argument("--seeds", default="42,43")
    p.add_argument("--pair_selection_seed_base", type=int, default=610000)
    p.add_argument("--pair_shuffle_seed_base", type=int, default=611000)
    p.add_argument("--slot_seed_base", type=int, default=612000)
    p.add_argument("--max_candidate_pairs", type=int, default=484000)
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

    tok = AutoTokenizer.from_pretrained(BASELINE_TOKENIZER_REPO, revision="main", use_fast=True)
    official_dir, official_manifest = download_official(out_dir)
    official_files = [official_dir / n for n in TRAIN_FILES]
    gem_path = pathlib.Path(hf_hub_download(
        repo_id=GEM_DATASET_ID,
        repo_type="dataset",
        revision=GEM_DATASET_REVISION,
        filename=GEM_DATASET_FILE,
        cache_dir=os.environ["HF_HUB_CACHE"],
    ))
    pair_candidates = load_pair_candidates(gem_path, tok, args.max_seq_length, args.max_candidate_pairs)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    all_meta: dict[str, Any] = {
        "official_dataset_id": OFFICIAL_DATASET_ID,
        "official_dataset_revision": OFFICIAL_DATASET_REVISION,
        "official_manifest": official_manifest,
        "gem_dataset_id": GEM_DATASET_ID,
        "gem_dataset_revision": GEM_DATASET_REVISION,
        "gem_dataset_file": GEM_DATASET_FILE,
        "gem_dataset_file_cache_path": str(gem_path),
        "gem_dataset_file_sha256": sha256_file(gem_path),
        "licenses": LICENSES,
        "baseline_tokenizer_repo": BASELINE_TOKENIZER_REPO,
        "official_words_target": args.official_words,
        "pair_words_target": args.pair_words,
        "official_pool_words": args.official_pool_words,
        "words_per_official_example": args.words_per_official_example,
        "max_seq_length": args.max_seq_length,
        "target_steps": args.target_steps,
        "num_pair_candidates": len(pair_candidates),
        "seeds": {},
    }

    for seed in seeds:
        official = select_official_examples(official_files, seed, args.official_pool_words, args.official_words, args.words_per_official_example)
        sel_seed = args.pair_selection_seed_base + seed
        shuf_seed = args.pair_shuffle_seed_base + seed
        slot_seed = args.slot_seed_base + seed
        pairs, perm, pair_actual_words = select_pairs_for_seed(pair_candidates, tok, args.pair_words, args.max_seq_length, sel_seed, shuf_seed)
        adj, shuf, aux = records_for_seed(seed, official, pairs, perm, slot_seed)
        checks = validate_pair_records(adj, shuf, tok, args.max_seq_length)
        if not checks["validation"]["hard_validation_ok"]:
            raise RuntimeError(f"hard validation failed seed {seed}: {checks['validation']}")
        if not checks["validation"]["aggregate_opportunity_identical"]:
            raise RuntimeError(f"aggregate token/group opportunity not identical seed {seed}: {checks['validation']}")
        actual_words = checks["adjacent_summary"]["total_words"]
        if actual_words != checks["shuffled_summary"]["total_words"]:
            raise RuntimeError("word mismatch after validation")
        batch = recommended_batch_size(len(adj), args.target_steps)
        label = f"seed{seed}_official{args.official_words}_pair{pair_actual_words}_total{actual_words}_n{len(adj)}"
        adj_path = out_dir / f"mixture_adjacent_{label}.jsonl"
        shuf_path = out_dir / f"mixture_shuffled_{label}.jsonl"
        meta_path = out_dir / f"mixture_materialization_{label}.json"
        write_jsonl(adj_path, adj)
        write_jsonl(shuf_path, shuf)
        seed_meta = {
            "seed": seed,
            "pair_selection_seed": sel_seed,
            "pair_shuffle_seed": shuf_seed,
            "slot_seed": slot_seed,
            "official_words": sum(x.words for x in official),
            "official_num_examples": len(official),
            "pair_words": pair_actual_words,
            "pair_num_examples": len(pairs),
            "actual_total_words": actual_words,
            "combined_num_examples": len(adj),
            "recommended_batch": batch,
            "mixture_adjacent_path": str(adj_path),
            "mixture_shuffled_path": str(shuf_path),
            "mixture_adjacent_sha256": sha256_file(adj_path),
            "mixture_shuffled_sha256": sha256_file(shuf_path),
            "materialization_meta_path": str(meta_path),
            **aux,
            **checks,
        }
        # The per-seed meta includes only this seed for convenient trainer --example_jsonl_meta.
        per_seed_meta = dict(all_meta)
        per_seed_meta["seeds"] = {str(seed): seed_meta}
        meta_path.write_text(json.dumps(per_seed_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        seed_meta["materialization_meta_sha256"] = sha256_file(meta_path)
        all_meta["seeds"][str(seed)] = seed_meta

    all_meta_path = out_dir / "mixture_materialization_all_seeds.json"
    all_meta_path.write_text(json.dumps(all_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "MIXTURE_JSONL_READY",
        "all_meta_path": str(all_meta_path),
        "seeds": {
            seed: {
                "actual_total_words": all_meta["seeds"][str(seed)]["actual_total_words"],
                "combined_num_examples": all_meta["seeds"][str(seed)]["combined_num_examples"],
                "recommended_batch": all_meta["seeds"][str(seed)]["recommended_batch"],
                "validation": all_meta["seeds"][str(seed)]["validation"],
                "adjacent_path": all_meta["seeds"][str(seed)]["mixture_adjacent_path"],
                "shuffled_path": all_meta["seeds"][str(seed)]["mixture_shuffled_path"],
                "meta_path": all_meta["seeds"][str(seed)]["materialization_meta_path"],
            }
            for seed in seeds
        },
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
