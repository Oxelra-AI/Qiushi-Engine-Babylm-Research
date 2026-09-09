#!/usr/bin/env python3
"""Build matched aligned-rewrite data arms for BabyLM research.

This materializer implements the route chosen in research after legal model-side
factor tests failed to reproduce the public leader's Entity/EWoK package.  It
constructs manifest-backed JSONL examples for five arms from the accessible
GEM/wiki_auto_asset_turk simplification-pair dataset:

  aligned       : source_i + target_i in one MLM window
  shuffled      : source_i + target_perm[i] with no identity target adjacency
  unpaired_mix  : the same source_i and target_i texts as independent examples
  source_only   : selected source texts repeated deterministically to the same
                  word budget
  rewrite_only  : selected target texts repeated deterministically to the same
                  word budget

Design invariants:
- no gated go76dof/Fineweb_simplification_pairs data is used;
- aligned, shuffled, and unpaired_mix contain exactly the same source and target
  text multisets at the selected-pair level;
- word counts are exact whitespace counts stored in each JSONL row;
- source/rewrite-only arms are deterministic full-example resamplings to the
  same total word budget, never partial final examples;
- tokenization and WWM word-group summaries under the BabyLM baseline16k
  tokenizer are saved before training;
- by default source and target side token lengths are each capped at 128, so any
  aligned or shuffled source+target window is guaranteed <=256 tokens.

The output JSONL files are consumed directly by
training/scripts/babylm_masked_train_leadershape.py --example_jsonl.
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
from typing import Any, Iterable

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
DEFAULT_OUT_DIR = pathlib.Path("experiments/archive/initial_model_studies/data/aligned_rewrite_revision_150")
DELIM = " "  # ordinary adjacency; no artificial whitespace word is inserted


@dataclass(frozen=True)
class PairRow:
    pair_id: int
    source: str
    target: str
    source_words: int
    target_words: int
    source_tokens: int
    target_tokens: int
    aligned_tokens: int
    source_sha256: str
    target_sha256: str

    @property
    def pair_words(self) -> int:
        return self.source_words + self.target_words


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_text(x: Any) -> str:
    s = " ".join(str(x).replace("\n", " ").split())
    return s.replace("[SEP]", "SEP")


def word_count(text: str) -> int:
    return len(text.split())


def load_parquet_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
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
        if int(tid) in special:
            continue
        token = str(tok.convert_ids_to_tokens(int(tid)))
        if groups == 0 or is_word_start(token) or j == 0:
            groups += 1
    return groups


def multiset_hash(values: Iterable[str]) -> str:
    h = hashlib.sha256()
    for value in sorted(values):
        h.update(value.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def write_jsonl(path: pathlib.Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def deranged_permutation(n: int, seed: int) -> list[int]:
    if n < 2:
        raise ValueError("need at least two selected pairs")
    rng = random.Random(seed)
    perm = list(range(n))
    for _ in range(20000):
        rng.shuffle(perm)
        if all(perm[i] != i for i in range(n)):
            return perm
    # deterministic cyclic fallback
    offset = 1 + (seed % (n - 1))
    perm = [(i + offset) % n for i in range(n)]
    if any(perm[i] == i for i in range(n)):
        raise RuntimeError("failed to construct target derangement")
    return perm


def choose_pairs_greedy(candidates: list[PairRow], target_words: int, seed: int) -> tuple[list[PairRow], int]:
    """Greedy selection of unique pairs up to target_words without exceeding it.

    Returns (selected_pairs, actual_total_words).  All five arms will use the
    actual total as their word budget; source_only and rewrite_only can hit any
    target through deterministic resampling. This avoids fragile exact-fill
    combinatorics for unique aligned pairs.
    """
    rng = random.Random(seed)
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    selected: list[PairRow] = []
    total = 0
    for pr in shuffled:
        if total + pr.pair_words <= target_words:
            selected.append(pr)
            total += pr.pair_words
        if total == target_words:
            break
    return selected, total


def solve_unbounded_length_fill(lengths: list[int], target: int) -> list[int] | None:
    """Return a list of lengths summing to target, allowing repeats.

    This helper is used only for source_only/rewrite_only full-example
    resampling, where repeated examples are intentional and recorded. It must
    not be used to select unique aligned pairs.
    """
    if target == 0:
        return []
    lengths = sorted({int(x) for x in lengths if 0 < int(x) <= target})
    if not lengths:
        return None
    prev = [-1] * (target + 1)
    prev[0] = 0
    for s in range(1, target + 1):
        for L in lengths:
            if L > s:
                break
            if prev[s - L] != -1:
                prev[s] = L
                break
    if prev[target] == -1:
        return None
    out: list[int] = []
    s = target
    while s > 0:
        L = prev[s]
        if L <= 0:
            return None
        out.append(L)
        s -= L
    return out


def make_single_side_records(
    side_name: str,
    selected: list[PairRow],
    target_words: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Repeat full source or target sentences deterministically to target_words."""
    assert side_name in {"source_only", "rewrite_only"}
    units = []
    for pr in selected:
        if side_name == "source_only":
            text, words, sha = pr.source, pr.source_words, pr.source_sha256
            source = "GEM/wiki_auto_asset_turk_source_only"
        else:
            text, words, sha = pr.target, pr.target_words, pr.target_sha256
            source = "GEM/wiki_auto_asset_turk_rewrite_only"
        if words > 0:
            units.append({"pair_id": pr.pair_id, "text": text, "words": words, "text_sha256": sha, "source": source})
    rng = random.Random(seed)
    order = list(range(len(units)))
    rng.shuffle(order)
    records: list[dict[str, Any]] = []
    total = 0
    repeat_round = 0
    cursor = 0
    # Append full examples until no unit in a complete scan fits the remaining budget.
    while total < target_words:
        appended = False
        remaining = target_words - total
        for _ in range(len(order)):
            idx = order[cursor]
            cursor = (cursor + 1) % len(order)
            if cursor == 0:
                repeat_round += 1
            u = units[idx]
            if u["words"] <= remaining:
                records.append({
                    "example_id": len(records),
                    "source": u["source"],
                    "text": u["text"],
                    "words": u["words"],
                    "mode": side_name,
                    "pair_ids": [u["pair_id"]],
                    "source_pair_id": u["pair_id"] if side_name == "source_only" else None,
                    "target_pair_id": u["pair_id"] if side_name == "rewrite_only" else None,
                    "text_sha256": u["text_sha256"],
                    "repeat_round": repeat_round,
                })
                total += u["words"]
                appended = True
                break
        if not appended:
            # Remaining budget is smaller than every fitting unit.  Use bounded
            # backtracking: pop the last few records to enlarge the gap, then
            # retry DP fill over available repeated full examples.
            available_lengths = sorted({u["words"] for u in units})
            filled = False
            max_backtrack = min(20, len(records))
            for bt in range(max_backtrack + 1):
                remaining = target_words - total
                if remaining == 0:
                    filled = True
                    break
                fill = solve_unbounded_length_fill(available_lengths, remaining)
                if fill is not None:
                    # Build a lookup for deterministic unit selection.
                    by_len: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
                    for u in units:
                        by_len[u["words"]].append(u)
                    len_counts = collections.Counter(fill)
                    for L, k in len_counts.items():
                        choices = by_len[L]
                        for j in range(k):
                            u = choices[j % len(choices)]
                            records.append({
                                "example_id": len(records),
                                "source": u["source"],
                                "text": u["text"],
                                "words": u["words"],
                                "mode": side_name,
                                "pair_ids": [u["pair_id"]],
                                "source_pair_id": u["pair_id"] if side_name == "source_only" else None,
                                "target_pair_id": u["pair_id"] if side_name == "rewrite_only" else None,
                                "text_sha256": u["text_sha256"],
                                "repeat_round": repeat_round,
                                "exact_fill": True,
                            })
                            total += L
                    filled = True
                    break
                # Backtrack: pop last record and enlarge gap.
                if records:
                    popped = records.pop()
                    total -= popped["words"]
                else:
                    break
            if not filled:
                raise RuntimeError(f"could not exactly fill {side_name} remainder {target_words - total} after {max_backtrack} backtracks")
            break
    if total != target_words:
        raise RuntimeError(f"{side_name} total {total} != target {target_words}")
    return records


def summarize_records(records: list[dict[str, Any]], tok, max_seq_length: int) -> dict[str, Any]:
    total_words = 0
    total_tokens = 0
    total_groups = 0
    max_tokens = 0
    hist = {"<=64": 0, "65-128": 0, "129-256": 0, ">256": 0}
    over = []
    mode_counts = collections.Counter()
    source_words = 0
    target_words = 0
    for r in records:
        text = str(r["text"])
        words = int(r["words"])
        actual = word_count(text)
        if words != actual:
            raise RuntimeError(f"word mismatch in arm row {r.get('example_id')}: {words} != {actual}")
        ids = token_ids(tok, text)
        L = len(ids)
        groups = count_word_groups(tok, ids)
        total_words += words
        total_tokens += L
        total_groups += groups
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
            over.append({"example_id": r.get("example_id"), "token_len": L, "words": words, "mode": r.get("mode"), "pair_ids": r.get("pair_ids")})
        mode = str(r.get("mode", "unknown"))
        mode_counts[mode] += 1
        if mode in {"aligned", "shuffled"}:
            source_words += int(r.get("source_words", 0))
            target_words += int(r.get("target_words", 0))
        elif mode == "unpaired_source" or mode == "source_only":
            source_words += words
        elif mode == "unpaired_target" or mode == "rewrite_only":
            target_words += words
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
        "over_max_seq_length_examples_first20": over[:20],
        "mode_counts": dict(mode_counts),
        "source_words_contribution": source_words,
        "target_words_contribution": target_words,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--target_words", type=int, default=100000)
    p.add_argument("--max_seq_length", type=int, default=256)
    p.add_argument("--max_source_tokens", type=int, default=128)
    p.add_argument("--max_target_tokens", type=int, default=128)
    p.add_argument("--selection_seed", type=int, default=15001)
    p.add_argument("--shuffle_seed", type=int, default=15002)
    p.add_argument("--unpaired_order_seed", type=int, default=15003)
    p.add_argument("--source_only_seed", type=int, default=15004)
    p.add_argument("--rewrite_only_seed", type=int, default=15005)
    p.add_argument("--reserve_words", type=int, default=20000)
    p.add_argument("--max_candidate_pairs", type=int, default=484000)
    p.add_argument("--hf_cache_dir", default="experiments/archive/initial_model_studies/training/hf_home")
    args = p.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hf_home = pathlib.Path(args.hf_cache_dir)
    # Use a local writable cache; the environment may define read-only
    # global HF cache variables that would otherwise override cache_dir choices.
    os.environ["HF_HOME"] = str(hf_home.resolve())
    os.environ["HF_HUB_CACHE"] = str((hf_home / "hub").resolve())
    os.environ["TRANSFORMERS_CACHE"] = str((hf_home / "transformers").resolve())
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    pathlib.Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.environ["HF_HUB_CACHE"]).mkdir(parents=True, exist_ok=True)

    parquet_path = pathlib.Path(hf_hub_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        revision=DATASET_REVISION,
        filename=DATASET_FILE,
        cache_dir=os.environ["HF_HUB_CACHE"],
    ))
    tok = AutoTokenizer.from_pretrained(
        BASELINE_TOKENIZER_REPO,
        revision="main",
        use_fast=True,
        cache_dir=os.environ["HF_HUB_CACHE"],
    )
    rows = load_parquet_rows(parquet_path)

    candidates: list[PairRow] = []
    seen: set[tuple[str, str]] = set()
    rejected = collections.Counter()
    for idx, row in enumerate(rows):
        if len(candidates) >= args.max_candidate_pairs:
            break
        if "source" not in row or "target" not in row:
            raise RuntimeError(f"expected source/target fields, got {row.keys()}")
        src = normalize_text(row["source"])
        tgt = normalize_text(row["target"])
        if not src or not tgt:
            rejected["empty"] += 1
            continue
        if src == tgt:
            rejected["identity_text"] += 1
            continue
        key = (src, tgt)
        if key in seen:
            rejected["duplicate_pair"] += 1
            continue
        seen.add(key)
        sw, tw = word_count(src), word_count(tgt)
        if sw <= 0 or tw <= 0:
            rejected["zero_words"] += 1
            continue
        src_ids = token_ids(tok, src)
        tgt_ids = token_ids(tok, tgt)
        src_tok, tgt_tok = len(src_ids), len(tgt_ids)
        aligned_tok = len(token_ids(tok, src + DELIM + tgt))
        if src_tok > args.max_source_tokens:
            rejected["source_over_side_token_cap"] += 1
            continue
        if tgt_tok > args.max_target_tokens:
            rejected["target_over_side_token_cap"] += 1
            continue
        if src_tok + tgt_tok > args.max_seq_length:
            rejected["side_sum_over_max_seq"] += 1
            continue
        if aligned_tok > args.max_seq_length:
            rejected["aligned_over_max_seq"] += 1
            continue
        candidates.append(PairRow(
            pair_id=idx,
            source=src,
            target=tgt,
            source_words=sw,
            target_words=tw,
            source_tokens=src_tok,
            target_tokens=tgt_tok,
            aligned_tokens=aligned_tok,
            source_sha256=sha256_text(src),
            target_sha256=sha256_text(tgt),
        ))
    if not candidates:
        raise RuntimeError("no candidate pairs after filtering")
    total_candidate_pair_words = sum(pr.pair_words for pr in candidates)
    if total_candidate_pair_words < args.target_words:
        raise RuntimeError(f"candidate pair words {total_candidate_pair_words} < target {args.target_words}")

    selected, effective_target_words = choose_pairs_greedy(candidates, args.target_words, args.selection_seed)
    if effective_target_words == 0:
        raise RuntimeError("no pairs selected; check target_words and candidate pool")
    if effective_target_words < args.target_words * 0.95:
        print(f"WARNING: achieved {effective_target_words} words from target {args.target_words} "
              f"(shortfall {args.target_words - effective_target_words})")
    # All five arms will use effective_target_words as the common budget.
    n = len(selected)
    perm = deranged_permutation(n, args.shuffle_seed)

    aligned_records: list[dict[str, Any]] = []
    shuffled_records: list[dict[str, Any]] = []
    unpaired_units: list[dict[str, Any]] = []
    for out_i, pr in enumerate(selected):
        tgt_shuf = selected[perm[out_i]]
        aligned_text = pr.source + DELIM + pr.target
        shuffled_text = pr.source + DELIM + tgt_shuf.target
        aligned_records.append({
            "example_id": out_i,
            "source": "GEM/wiki_auto_asset_turk_aligned",
            "text": aligned_text,
            "words": word_count(aligned_text),
            "mode": "aligned",
            "pair_ids": [pr.pair_id],
            "source_pair_id": pr.pair_id,
            "target_pair_id": pr.pair_id,
            "source_words": pr.source_words,
            "target_words": pr.target_words,
            "source_text_sha256": pr.source_sha256,
            "target_text_sha256": pr.target_sha256,
        })
        shuffled_records.append({
            "example_id": out_i,
            "source": "GEM/wiki_auto_asset_turk_shuffled",
            "text": shuffled_text,
            "words": word_count(shuffled_text),
            "mode": "shuffled",
            "pair_ids": [pr.pair_id, tgt_shuf.pair_id],
            "source_pair_id": pr.pair_id,
            "target_pair_id": tgt_shuf.pair_id,
            "source_words": pr.source_words,
            "target_words": tgt_shuf.target_words,
            "source_text_sha256": pr.source_sha256,
            "target_text_sha256": tgt_shuf.target_sha256,
        })
        unpaired_units.append({
            "source": "GEM/wiki_auto_asset_turk_unpaired_source",
            "text": pr.source,
            "words": pr.source_words,
            "mode": "unpaired_source",
            "pair_ids": [pr.pair_id],
            "source_pair_id": pr.pair_id,
            "target_pair_id": None,
            "text_sha256": pr.source_sha256,
        })
        unpaired_units.append({
            "source": "GEM/wiki_auto_asset_turk_unpaired_target",
            "text": pr.target,
            "words": pr.target_words,
            "mode": "unpaired_target",
            "pair_ids": [pr.pair_id],
            "source_pair_id": None,
            "target_pair_id": pr.pair_id,
            "text_sha256": pr.target_sha256,
        })

    rng_unpaired = random.Random(args.unpaired_order_seed)
    rng_unpaired.shuffle(unpaired_units)
    unpaired_records = []
    for i, r in enumerate(unpaired_units):
        rr = dict(r)
        rr["example_id"] = i
        unpaired_records.append(rr)

    source_only_records = make_single_side_records("source_only", selected, effective_target_words, args.source_only_seed)
    rewrite_only_records = make_single_side_records("rewrite_only", selected, effective_target_words, args.rewrite_only_seed)

    arms = {
        "aligned": aligned_records,
        "shuffled": shuffled_records,
        "unpaired_mix": unpaired_records,
        "source_only": source_only_records,
        "rewrite_only": rewrite_only_records,
    }
    summaries = {arm: summarize_records(records, tok, args.max_seq_length) for arm, records in arms.items()}
    for arm, summary in summaries.items():
        if summary["total_words"] != effective_target_words:
            raise RuntimeError(f"arm {arm} words {summary['total_words']} != effective target {effective_target_words}")

    selected_sources = [pr.source for pr in selected]
    selected_targets = [pr.target for pr in selected]
    shuffled_targets = [selected[perm[i]].target for i in range(n)]
    unpaired_texts = [r["text"] for r in unpaired_records]
    validation = {
        "target_words_exact_all_arms": all(s["total_words"] == effective_target_words for s in summaries.values()),
        "aligned_shuffled_same_source_multiset": collections.Counter(selected_sources) == collections.Counter(selected_sources),
        "aligned_shuffled_same_target_multiset": collections.Counter(selected_targets) == collections.Counter(shuffled_targets),
        "aligned_shuffled_same_text_multiset": collections.Counter(selected_sources + selected_targets) == collections.Counter(selected_sources + shuffled_targets),
        "unpaired_same_text_multiset_as_aligned_pair_sides": collections.Counter(unpaired_texts) == collections.Counter(selected_sources + selected_targets),
        "no_identity_target_adjacency_in_shuffled": all(perm[i] != i for i in range(n)),
        "no_overlength_examples_any_arm": all(s["num_over_max_seq_length"] == 0 for s in summaries.values()),
        "aligned_vs_shuffled_token_count_delta": summaries["shuffled"]["total_tokens"] - summaries["aligned"]["total_tokens"],
        "aligned_vs_shuffled_word_group_delta": summaries["shuffled"]["total_word_groups"] - summaries["aligned"]["total_word_groups"],
        "aligned_source_multiset_hash": multiset_hash(selected_sources),
        "aligned_target_multiset_hash": multiset_hash(selected_targets),
        "shuffled_target_multiset_hash": multiset_hash(shuffled_targets),
        "unpaired_text_multiset_hash": multiset_hash(unpaired_texts),
    }
    hard_bool_keys = [
        "target_words_exact_all_arms",
        "aligned_shuffled_same_target_multiset",
        "aligned_shuffled_same_text_multiset",
        "unpaired_same_text_multiset_as_aligned_pair_sides",
        "no_identity_target_adjacency_in_shuffled",
        "no_overlength_examples_any_arm",
    ]
    if not all(bool(validation[k]) for k in hard_bool_keys):
        raise RuntimeError(f"hard validation failed: {validation}")

    label = (
        f"w{effective_target_words}_n{n}_sel{args.selection_seed}_shuf{args.shuffle_seed}"
        f"_side{args.max_source_tokens}-{args.max_target_tokens}"
    )
    arm_paths = {}
    for arm, records in arms.items():
        path = out_dir / f"{arm}_{label}.jsonl"
        write_jsonl(path, records)
        arm_paths[arm] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "rows": len(records),
            "words": summaries[arm]["total_words"],
        }
    meta_path = out_dir / f"manifest_{label}.json"
    sample_pairs = []
    for i, pr in enumerate(selected[:10]):
        sample_pairs.append({
            "selected_index": i,
            "source_pair_id": pr.pair_id,
            "shuffled_target_pair_id": selected[perm[i]].pair_id,
            "source_words": pr.source_words,
            "target_words": pr.target_words,
            "source_tokens": pr.source_tokens,
            "target_tokens": pr.target_tokens,
            "source_preview": pr.source[:160],
            "target_preview": pr.target[:160],
            "shuffled_target_preview": selected[perm[i]].target[:160],
        })
    meta = {
        "status": "ALIGNED_REWRITE_ARMS_READY",
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "dataset_file": DATASET_FILE,
        "dataset_file_cache_path": str(parquet_path),
        "dataset_file_sha256": sha256_file(parquet_path),
        "licenses": LICENSES,
        "baseline_tokenizer_repo": BASELINE_TOKENIZER_REPO,
        "delimiter": DELIM,
        "target_words_requested": args.target_words,
        "effective_target_words": effective_target_words,
        "max_seq_length": args.max_seq_length,
        "max_source_tokens": args.max_source_tokens,
        "max_target_tokens": args.max_target_tokens,
        "selection_seed": args.selection_seed,
        "shuffle_seed": args.shuffle_seed,
        "unpaired_order_seed": args.unpaired_order_seed,
        "source_only_seed": args.source_only_seed,
        "rewrite_only_seed": args.rewrite_only_seed,
        "reserve_words": args.reserve_words,
        "num_raw_rows": len(rows),
        "candidate_pairs": len(candidates),
        "candidate_pair_words": total_candidate_pair_words,
        "rejected_counts": dict(rejected),
        "selected_pairs": n,
        "selected_source_words": sum(pr.source_words for pr in selected),
        "selected_target_words": sum(pr.target_words for pr in selected),
        "arm_paths": arm_paths,
        "summaries": summaries,
        "validation": validation,
        "sample_pairs_first10": sample_pairs,
        "trainer_note": "Use babylm_masked_train_leadershape.py --example_jsonl <arm_path> --example_jsonl_meta <this_manifest> --max_word_exposure target_words --example_pool_words target_words. JSONL selection rejects partial final examples.",
        "scientific_note": "The first training experiment should keep S1 12x384/baseline16k/AdamW/flat-WWM fixed. Aligned-vs-shuffled isolates correct semantic correspondence; unpaired/source/rewrite arms separate same-window and rewrite-distribution effects.",
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": meta["status"],
        "manifest": str(meta_path),
        "effective_target_words": effective_target_words,
        "selected_pairs": n,
        "arm_paths": arm_paths,
        "validation": validation,
        "summaries_compact": {
            arm: {
                "rows": summaries[arm]["num_examples"],
                "words": summaries[arm]["total_words"],
                "tokens_per_word": summaries[arm]["tokens_per_word"],
                "max_tokens": summaries[arm]["max_tokens"],
                "over_max": summaries[arm]["num_over_max_seq_length"],
            } for arm in arms
        },
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
