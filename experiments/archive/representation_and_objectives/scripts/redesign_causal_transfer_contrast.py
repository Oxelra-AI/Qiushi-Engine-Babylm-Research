#!/usr/bin/env python3
"""research: redesign the causal-GPT compact-view transfer contrast.

Purpose
-------
The causal-GPT compact-vs-repeat scaffold is not
scientifically symmetric: exact source-prefix repetition gives the repeat arm a
causal boundary-copy/restart advantage, while compact rewrites carry more BPE
mass. This script uses the clean aligned reference assets but constructs
CPU-only, no-training contrast assets whose treatment arms match the two
problematic factors before any full H100 transfer run:

1. own_compact vs adjbreak_compact
   - Same compact rewrite multiset, same filler, same tokenizer, same order seed.
   - Therefore treatment token mass and boundary novelty are matched.
   - Difference: own source--rewrite correspondence vs length-near broken
     correspondence. This isolates local semantic correspondence/consolidation
     under causal training without the repeat copy artifact.

2. semantic_extractive vs random_extractive
   - Both second sides are copied from the source, with equal word counts and
     near-matched BPE token counts per pair.
   - Therefore literal copy opportunity and treatment token mass are matched.
   - Difference: source-token selection guided by the compact rewrite vs a
     matched random source-token selection. This tests whether information-dense
     semantic selection matters when causal copy opportunity is held fixed.

The script materializes 10M-word JSONL pools locally and writes a
manifest/audit. It does not train a model, does not modify source assets, and is meant
as a repairable data design for the orthogonal transfer leg.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
import re
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

ROOT_A01 = Path("experiments/archive/representation_and_objectives")
ROOT_A02 = Path("experiments/archive/frontier_consolidation")
A02_SCAFFOLD = ROOT_A02 / "data/causal_transfer_scaffold"
PAIRS_FILE = ROOT_A02 / "data/density_core_reinvestment_medium_riskhard/selected_compact_reinvest_pairs.jsonl"
FILLER_FILE = A02_SCAFFOLD / "filler_rows.jsonl"
TOKENIZER_DIR = A02_SCAFFOLD / "neutral_tokenizer"
OUT = ROOT_A01 / "data/causal_transfer_redesign"
SEED = 171043
TARGET_TOTAL_WORDS = 10_000_000
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def wc(text: str) -> int:
    return len(text.split())


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> Tuple[int, int]:
    n = 0
    words = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            n += 1
            words += int(r["words"])
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    return n, words


def pct(xs: Sequence[float], q: float) -> float:
    if not xs:
        return float("nan")
    if len(xs) == 1:
        return float(xs[0])
    s = sorted(xs)
    pos = (len(s) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(s[lo])
    return float(s[lo] * (hi - pos) + s[hi] * (pos - lo))


def stats(xs: Sequence[float]) -> Dict[str, Any]:
    xs = list(xs)
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "mean": float(statistics.mean(xs)),
        "median": float(statistics.median(xs)),
        "p10": pct(xs, 0.10),
        "p90": pct(xs, 0.90),
        "min": float(min(xs)),
        "max": float(max(xs)),
    }


def norm_tokens(text: str) -> List[str]:
    out = []
    for m in WORD_RE.finditer(text.lower()):
        w = m.group(0).strip("'-")
        if w:
            out.append(w)
    return out


def simple_stems(words: Iterable[str]) -> set[str]:
    stems: set[str] = set()
    for w in words:
        stems.add(w)
        for suf in ("ing", "edly", "edly", "ed", "ly", "ies", "s"):
            if len(w) > len(suf) + 3 and w.endswith(suf):
                stems.add(w[: -len(suf)])
    return stems


def token_len(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def side_boundary_overlap(left_words: List[str], right_words: List[str]) -> int:
    """Longest k such that suffix(left,k) == prefix(right,k), case-sensitive."""
    m = min(len(left_words), len(right_words))
    best = 0
    for k in range(1, m + 1):
        if left_words[-k:] == right_words[:k]:
            best = k
    return best


def prefix_match_fraction(source_words: List[str], side_words: List[str]) -> float:
    if not side_words:
        return 0.0
    k = 0
    for a, b in zip(source_words, side_words):
        if a == b:
            k += 1
        else:
            break
    return k / len(side_words)


def choose_adjbreak_assignment(pairs: List[Dict[str, Any]], orders: List[str], seed: int) -> List[int]:
    """Assign a different rewrite to every source, preserving rewrite length in almost all cases.

    First rotate within (rewrite_words, order) buckets with size >1. Singleton
    buckets are rotated among themselves. This keeps the global rewrite multiset
    exactly identical to own_compact; the few singleton long lengths can differ
    locally but preserve total treatment words and tokens globally.
    """
    rng = random.Random(seed + 20501)
    groups: Dict[Tuple[int, str], List[int]] = collections.defaultdict(list)
    for i, (p, o) in enumerate(zip(pairs, orders)):
        groups[(int(p["rewrite_words"]), o)].append(i)

    assignment = [-1] * len(pairs)
    singletons: List[int] = []
    for key, idxs in sorted(groups.items()):
        if len(idxs) == 1:
            singletons.extend(idxs)
            continue
        local = idxs[:]
        rng.shuffle(local)
        shift = 1
        for pos, i in enumerate(local):
            assignment[i] = local[(pos + shift) % len(local)]

    # Rare long-length singleton groups. Rotate them together; this changes local
    # word length for only these items but keeps the global rewrite multiset.
    if singletons:
        if len(singletons) == 1:
            # Fallback should not happen here; use nearest non-self global item.
            only = singletons[0]
            candidates = [j for j in range(len(pairs)) if j != only]
            j = min(candidates, key=lambda x: abs(pairs[x]["rewrite_words"] - pairs[only]["rewrite_words"]))
            assignment[only] = j
            # The displaced item cannot keep exact multiset in this impossible
            # case; record through self check. Current assets have >1 singleton.
        else:
            singletons_sorted = sorted(singletons, key=lambda i: (pairs[i]["rewrite_words"], i))
            for pos, i in enumerate(singletons_sorted):
                assignment[i] = singletons_sorted[(pos + 1) % len(singletons_sorted)]

    assert all(j >= 0 for j in assignment)
    assert all(i != j for i, j in enumerate(assignment))
    assert sorted(assignment) == list(range(len(pairs))), "rewrite assignment must be a permutation"
    return assignment


def source_word_scores(source_words: List[str], rewrite_text: str, df: collections.Counter, n_docs: int) -> List[float]:
    rw_norm = set(norm_tokens(rewrite_text))
    rw_stems = simple_stems(rw_norm)
    scores: List[float] = []
    for idx, w in enumerate(source_words):
        nw_list = norm_tokens(w)
        nw = nw_list[0] if nw_list else w.lower().strip()
        score = 0.0
        if nw in rw_norm:
            score += 10.0
        if nw in rw_stems:
            score += 4.0
        if any(ch.isdigit() for ch in w):
            score += 3.0
        if w[:1].isupper() and idx > 0:
            score += 2.0
        # crude inverse document frequency from pair sources, to prefer content
        # words over grammatical filler without an external analyzer.
        if nw:
            score += min(4.0, math.log((n_docs + 1) / (1 + df.get(nw, 0))))
        # avoid over-selecting punctuation-only tokens
        if not nw:
            score -= 5.0
        # tiny deterministic position tie-break: keep early anchors but do not
        # always produce a prefix.
        score += (idx % 7) * 1e-4
        scores.append(score)
    return scores


def select_semantic_indices(source_words: List[str], rewrite_text: str, k: int, df: collections.Counter, n_docs: int) -> List[int]:
    k = min(k, len(source_words))
    scores = source_word_scores(source_words, rewrite_text, df, n_docs)
    ranked = sorted(range(len(source_words)), key=lambda i: (-scores[i], i))[:k]
    return sorted(ranked)


def sample_random_indices_matched(
    source_words: List[str],
    k: int,
    target_tok: int,
    semantic_indices: set[int],
    tokenizer,
    rng: random.Random,
    tries: int = 96,
) -> List[int]:
    """Random source-token selection with same k and near same tokenizer mass.

    Preference: low overlap with semantic indices and target rewrite lexical set is
    handled in audit; here choose nearest BPE token count and avoid always using
    the source prefix. If the source is too short, sampling is still exact-k.
    """
    n = len(source_words)
    k = min(k, n)
    all_idx = list(range(n))
    best: Tuple[float, List[int]] | None = None
    # Include a deterministic shifted-window candidate to stabilize short cases.
    if n >= k:
        starts = list(range(max(1, n - k + 1)))
        rng.shuffle(starts)
        for st in starts[: min(8, len(starts))]:
            cand = list(range(st, st + k))
            text = " ".join(source_words[i] for i in cand)
            tok = token_len(tokenizer, text)
            sem_overlap = len(set(cand) & semantic_indices) / max(k, 1)
            prefix_penalty = 1.0 if cand == list(range(k)) else 0.0
            score = abs(tok - target_tok) + 0.75 * sem_overlap + 0.25 * prefix_penalty
            if best is None or score < best[0]:
                best = (score, cand)
    for _ in range(tries):
        cand = sorted(rng.sample(all_idx, k)) if n > k else all_idx[:]
        text = " ".join(source_words[i] for i in cand)
        tok = token_len(tokenizer, text)
        sem_overlap = len(set(cand) & semantic_indices) / max(k, 1)
        prefix_penalty = 1.0 if cand == list(range(k)) else 0.0
        # Strongly prioritize token mass, then not selecting the exact same
        # semantic subset. This matches copy opportunity while weakening semantic
        # selection.
        score = abs(tok - target_tok) + 0.75 * sem_overlap + 0.25 * prefix_penalty
        if best is None or score < best[0]:
            best = (score, cand)
    assert best is not None
    return sorted(best[1])


def make_pair_text(first: str, second: str) -> str:
    return (first.strip() + " " + second.strip()).strip()


def build_pool(filler_rows: List[Dict[str, Any]], items: List[Dict[str, Any]], label: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    n_filler = len(filler_rows)
    n_pairs = len(items)
    pair_interval = n_filler / n_pairs if n_pairs else float("inf")
    item_idx = 0
    next_insert = pair_interval / 2
    for fi, fr in enumerate(filler_rows):
        rows.append({"text": fr["text"], "words": int(fr.get("words", wc(fr["text"]))), "source": fr.get("source", "filler")})
        while item_idx < n_pairs and fi >= next_insert:
            it = items[item_idx]
            rows.append({"text": it["text"], "words": int(it["words"]), "source": label, "pair_idx": it["pair_idx"], "order": it["order"]})
            item_idx += 1
            next_insert += pair_interval
    while item_idx < n_pairs:
        it = items[item_idx]
        rows.append({"text": it["text"], "words": int(it["words"]), "source": label, "pair_idx": it["pair_idx"], "order": it["order"]})
        item_idx += 1
    return rows


def overlap_metrics(source_text: str, side_text: str, rewrite_text: str) -> Dict[str, float]:
    src = set(norm_tokens(source_text))
    side = set(norm_tokens(side_text))
    rw = set(norm_tokens(rewrite_text))
    return {
        "side_source_jaccard": len(side & src) / max(1, len(side | src)),
        "side_words_in_source_frac": sum(1 for w in norm_tokens(side_text) if w in src) / max(1, len(norm_tokens(side_text))),
        "side_rewrite_jaccard": len(side & rw) / max(1, len(side | rw)),
        "side_words_in_rewrite_frac": sum(1 for w in norm_tokens(side_text) if w in rw) / max(1, len(norm_tokens(side_text))),
        "source_rewrite_jaccard": len(src & rw) / max(1, len(src | rw)),
    }


def summarize_arm(items: List[Dict[str, Any]], tokenizer) -> Dict[str, Any]:
    side_tokens = [it["side_tokens"] for it in items]
    pair_tokens = [token_len(tokenizer, it["text"]) for it in items]
    side_words = [it["side_words"] for it in items]
    boundary = [it["boundary_overlap_words"] for it in items]
    source_prefix = [it["source_prefix_match_frac"] for it in items]
    copy_frac = [it["side_words_in_source_frac"] for it in items]
    rw_frac = [it["side_words_in_rewrite_frac"] for it in items]
    return {
        "pair_items": len(items),
        "pair_words_total": int(sum(it["words"] for it in items)),
        "side_words_total": int(sum(side_words)),
        "side_tokens_total": int(sum(side_tokens)),
        "pair_tokens_total": int(sum(pair_tokens)),
        "side_tokens_per_word": stats([t / max(1, w) for t, w in zip(side_tokens, side_words)]),
        "side_words": stats(side_words),
        "boundary_suffix_prefix_overlap_words": stats(boundary),
        "source_prefix_match_fraction": stats(source_prefix),
        "side_words_in_source_fraction": stats(copy_frac),
        "side_words_in_rewrite_fraction": stats(rw_frac),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--write-pools", action="store_true", help="Materialize 10M JSONL pools; default only writes audit/table")
    args = ap.parse_args()

    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    pairs = read_jsonl(PAIRS_FILE)
    filler_rows = read_jsonl(FILLER_FILE)
    filler_words = sum(int(r.get("words", wc(r["text"]))) for r in filler_rows)
    pair_words = sum(int(p["pair_words"]) for p in pairs)
    assert filler_words + pair_words == TARGET_TOTAL_WORDS, (filler_words, pair_words)

    rng = random.Random(args.seed)
    orders = ["source_first" if rng.random() < 0.5 else "view_first" for _ in pairs]
    assignment = choose_adjbreak_assignment(pairs, orders, args.seed)

    # Pair-source document frequency for language-tool-free extractive selection.
    df: collections.Counter[str] = collections.Counter()
    for p in pairs:
        df.update(set(norm_tokens(p["source_text"])))
    n_docs = len(pairs)

    own_items: List[Dict[str, Any]] = []
    adj_items: List[Dict[str, Any]] = []
    sem_ext_items: List[Dict[str, Any]] = []
    rand_ext_items: List[Dict[str, Any]] = []
    table_rows: List[Dict[str, Any]] = []
    rand_rng = random.Random(args.seed + 20502)

    for i, (p, order, j) in enumerate(zip(pairs, orders, assignment)):
        src = p["source_text"].strip()
        own_side = p["rewrite_text"].strip()
        adj_side = pairs[j]["rewrite_text"].strip()
        src_words = src.split()
        rw_words = int(p["rewrite_words"])
        k = min(rw_words, len(src_words))

        sem_idx = select_semantic_indices(src_words, own_side, k, df, n_docs)
        sem_side = " ".join(src_words[x] for x in sem_idx)
        sem_tok = token_len(tokenizer, sem_side)
        rand_idx = sample_random_indices_matched(src_words, k, sem_tok, set(sem_idx), tokenizer, rand_rng)
        rand_side = " ".join(src_words[x] for x in rand_idx)

        arms = {
            "own_compact": own_side,
            "adjbreak_compact": adj_side,
            "semantic_extractive": sem_side,
            "random_extractive": rand_side,
        }
        per: Dict[str, Dict[str, Any]] = {}
        for arm, side in arms.items():
            if order == "source_first":
                text = make_pair_text(src, side)
                boundary = side_boundary_overlap(src.split(), side.split())
            else:
                text = make_pair_text(side, src)
                boundary = side_boundary_overlap(side.split(), src.split())
            side_words = wc(side)
            item_words = wc(text)
            # For adjbreak, side_words may differ locally but total remains exact.
            per[arm] = {
                "text": text,
                "side_text": side,
                "words": item_words,
                "side_words": side_words,
                "pair_idx": i,
                "order": order,
                "side_tokens": token_len(tokenizer, side),
                "text_tokens": token_len(tokenizer, text),
                "boundary_overlap_words": boundary,
                "source_prefix_match_frac": prefix_match_fraction(src.split(), side.split()),
                **overlap_metrics(src, side, own_side),
            }

        own_items.append(per["own_compact"])
        adj_items.append(per["adjbreak_compact"])
        sem_ext_items.append(per["semantic_extractive"])
        rand_ext_items.append(per["random_extractive"])

        table_rows.append({
            "pair_idx": i,
            "pair_id": p.get("pair_id"),
            "order": order,
            "source_words": int(p["source_words"]),
            "rewrite_words": int(p["rewrite_words"]),
            "assigned_adjbreak_pair_idx": j,
            "assigned_adjbreak_pair_id": pairs[j].get("pair_id"),
            "adjbreak_rewrite_words": int(pairs[j]["rewrite_words"]),
            "own_side_tokens": per["own_compact"]["side_tokens"],
            "adj_side_tokens": per["adjbreak_compact"]["side_tokens"],
            "semantic_ext_indices": sem_idx,
            "random_ext_indices": rand_idx,
            "semantic_ext_side_tokens": per["semantic_extractive"]["side_tokens"],
            "random_ext_side_tokens": per["random_extractive"]["side_tokens"],
            "semantic_ext_side": sem_side,
            "random_ext_side": rand_side,
            "own_side_rewrite_frac": per["own_compact"]["side_words_in_rewrite_frac"],
            "adj_side_rewrite_frac": per["adjbreak_compact"]["side_words_in_rewrite_frac"],
            "semantic_ext_rewrite_frac": per["semantic_extractive"]["side_words_in_rewrite_frac"],
            "random_ext_rewrite_frac": per["random_extractive"]["side_words_in_rewrite_frac"],
        })

    arms_items = {
        "own_compact": own_items,
        "adjbreak_compact": adj_items,
        "semantic_extractive": sem_ext_items,
        "random_extractive": rand_ext_items,
    }

    pools_written: Dict[str, Any] = {}
    if args.write_pools:
        for arm, items in arms_items.items():
            rows = build_pool(filler_rows, items, f"step205_{arm}")
            path = out / f"causal_{arm}_10M.jsonl"
            n_rows, n_words = write_jsonl(path, rows)
            assert n_words == TARGET_TOTAL_WORDS, (arm, n_words)
            pools_written[arm] = {
                "path": str(path),
                "rows": n_rows,
                "words": n_words,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }

    table_path = out / "pair_redesign_table.jsonl"
    write_jsonl(table_path, ({"text": json.dumps(r, ensure_ascii=False), "words": 0, **r} for r in table_rows))
    # The write_jsonl helper requires a text field and words; for this metadata
    # table those extra fields are harmless but noisy. Rewrite compactly.
    with table_path.open("w", encoding="utf-8") as f:
        for r in table_rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")

    arm_summaries = {arm: summarize_arm(items, tokenizer) for arm, items in arms_items.items()}

    def token_delta(a: str, b: str) -> Dict[str, Any]:
        ta = arm_summaries[a]["side_tokens_total"]
        tb = arm_summaries[b]["side_tokens_total"]
        return {
            "a": a,
            "b": b,
            "a_side_tokens": ta,
            "b_side_tokens": tb,
            "diff_a_minus_b": ta - tb,
            "pct_of_b": (ta - tb) / tb * 100 if tb else None,
        }

    # Derangement quality.
    same_adj_len = sum(1 for i, j in enumerate(assignment) if pairs[i]["rewrite_words"] == pairs[j]["rewrite_words"])
    adj_examples = []
    for i, j in enumerate(assignment[:200]):
        if len(adj_examples) >= 8:
            break
        if pairs[i]["rewrite_words"] != pairs[j]["rewrite_words"]:
            adj_examples.append({"i": i, "j": j, "rw_i": pairs[i]["rewrite_words"], "rw_j": pairs[j]["rewrite_words"]})

    # Per-pair token matching for extractive contrast.
    ext_abs_token_diffs = [abs(a["side_tokens"] - b["side_tokens"]) for a, b in zip(sem_ext_items, rand_ext_items)]
    ext_exact_tok = sum(1 for d in ext_abs_token_diffs if d == 0)
    ext_le1_tok = sum(1 for d in ext_abs_token_diffs if d <= 1)

    audit = {
        "status": "CAUSAL_TRANSFER_CONTRAST_REDESIGNED",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": round(time.time() - t0, 2),
        "purpose": "Repair the A02 causal-GPT transfer contrast before H100 spend: match treatment token mass and boundary/copy opportunity so compact/repeat-like outcomes are interpretable in both directions.",
        "inputs": {
            "pairs_file": str(PAIRS_FILE),
            "pairs_sha256": sha256_file(PAIRS_FILE),
            "filler_file": str(FILLER_FILE),
            "filler_sha256": sha256_file(FILLER_FILE),
            "tokenizer_dir": str(TOKENIZER_DIR),
            "tokenizer_json_sha256": sha256_file(TOKENIZER_DIR / "tokenizer.json"),
            "order_seed": args.seed,
        },
        "word_budget": {
            "filler_words": filler_words,
            "pair_words": pair_words,
            "total_words_each_pool": TARGET_TOTAL_WORDS,
            "pair_count": len(pairs),
            "filler_rows": len(filler_rows),
        },
        "design": {
            "own_vs_adjbreak_compact": "Same compact rewrite multiset and neutral tokenizer; own pairs preserve source--rewrite correspondence, adjbreak pairs derange rewrites to break local correspondence while preserving compact token mass and boundary novelty.",
            "semantic_vs_random_extractive": "Both sides are source-token extracts and thus copyable from the same source context; semantic extracts are selected using the compact rewrite as a language-tool-free guide, random extracts are sampled from the same source with same word count and near-matched BPE token count.",
            "not_a_training_result": "These are CPU data/audit assets only; any H100 run should begin with a small pilot and official-compatible causal evaluation after evaluator repair.",
        },
        "adjbreak_assignment": {
            "is_permutation": sorted(assignment) == list(range(len(pairs))),
            "self_assignments": sum(1 for i, j in enumerate(assignment) if i == j),
            "same_rewrite_word_length_pairs": same_adj_len,
            "same_rewrite_word_length_fraction": same_adj_len / len(pairs),
            "local_length_mismatch_count": len(pairs) - same_adj_len,
            "local_length_mismatch_examples_first200": adj_examples,
        },
        "arm_summaries": arm_summaries,
        "matched_factor_checks": {
            "own_vs_adjbreak_side_token_delta": token_delta("own_compact", "adjbreak_compact"),
            "semantic_vs_random_extractive_side_token_delta": token_delta("semantic_extractive", "random_extractive"),
            "semantic_vs_random_extractive_per_pair_abs_token_diff": stats(ext_abs_token_diffs),
            "semantic_vs_random_extractive_exact_token_matched_pairs": ext_exact_tok,
            "semantic_vs_random_extractive_le1_token_matched_pairs": ext_le1_tok,
            "semantic_vs_random_extractive_le1_fraction": ext_le1_tok / len(pairs),
        },
        "interpretation_rules": {
            "own_compact_gt_adjbreak": "Evidence that local source-own compact correspondence/consolidation transfers to causal training when token mass and boundary novelty are matched.",
            "adjbreak_ge_own_compact": "Evidence against local correspondence as the causal-transfer driver at this coordinate, because compact distribution/token mass and boundary novelty were held constant.",
            "semantic_extractive_gt_random_extractive": "Evidence that information-dense source-token selection helps beyond literal copyability under causal training.",
            "random_extractive_ge_semantic_extractive": "Evidence that causal transfer is dominated by generic source copy/repetition rather than semantic selection, because both arms are copyable and token-matched.",
            "do_not_compare_directly": "Do not directly read own_compact vs semantic_extractive as one mechanism verdict; they differ in abstractive novelty vs copyability. Use them as complementary factor tests.",
        },
        "outputs": {
            "out_dir": str(out),
            "pair_table": str(table_path),
            "pools_written": pools_written,
        },
    }

    audit_path = out / "causal_transfer_redesign_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md_path = out / "causal_transfer_redesign_summary.md"
    md = []
    md.append("# research causal-GPT transfer contrast redesign\n")
    md.append("This is a CPU-only redesign of the causal-transfer contrast after the confound audit. It preserves the clean aligned pair/filler/tokenizer assets but does not launch training.\n")
    md.append("## Arms\n")
    md.append("- `own_compact`: source paired with its own compact rewrite; semantic compression + correspondence, low literal copy.\n")
    md.append("- `adjbreak_compact`: source paired with a deranged compact rewrite; same rewrite multiset/token mass/boundary novelty, broken correspondence.\n")
    md.append("- `semantic_extractive`: source paired with source-token extract selected using the compact rewrite; copyable + semantically selected.\n")
    md.append("- `random_extractive`: source paired with a same-source random extract matched in word count and near-matched in BPE tokens; copyable + weak semantic selection.\n")
    md.append("## Matched-factor checks\n")
    md.append(f"- own vs adjbreak side-token delta: `{audit['matched_factor_checks']['own_vs_adjbreak_side_token_delta']}`\n")
    md.append(f"- semantic vs random extractive side-token delta: `{audit['matched_factor_checks']['semantic_vs_random_extractive_side_token_delta']}`\n")
    md.append(f"- extractive per-pair abs token diff: `{audit['matched_factor_checks']['semantic_vs_random_extractive_per_pair_abs_token_diff']}`; <=1 token pairs `{ext_le1_tok}/{len(pairs)}`.\n")
    md.append(f"- adjbreak local same rewrite-word length: `{same_adj_len}/{len(pairs)}`; local mismatches are rare long-tail cases while global rewrite multiset is exact.\n")
    md.append("## Reading the result if trained later\n")
    md.append("Only train after the causal evaluator is repaired to the official `evaluation_data` roots and fail-closed all-column scoring. A small pilot should precede 100M. The two contrasts should be read separately: own-vs-adjbreak for local correspondence under matched compact token mass and boundary novelty; semantic-vs-random extractive for information-dense selection under matched copy opportunity.\n")
    md_path.write_text("".join(md), encoding="utf-8")

    print(json.dumps({
        "status": audit["status"],
        "out_dir": str(out),
        "audit": str(audit_path),
        "summary_md": str(md_path),
        "pools_written": bool(args.write_pools),
        "own_adj_token_delta_pct": audit["matched_factor_checks"]["own_vs_adjbreak_side_token_delta"]["pct_of_b"],
        "ext_token_delta_pct": audit["matched_factor_checks"]["semantic_vs_random_extractive_side_token_delta"]["pct_of_b"],
        "ext_le1_fraction": audit["matched_factor_checks"]["semantic_vs_random_extractive_le1_fraction"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
