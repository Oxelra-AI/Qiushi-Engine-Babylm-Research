#!/usr/bin/env python3
"""Materialize 4M matched-content arms for the C/S screen.

Reads the official BabyLM Strict-Small 10M-word corpus, packs into 160-word
rows, scores every row with C-score (compositional structure density via spacy)
and S-score (recoverable signal density via tokenizer statistics), then selects
five source-matched arms of exactly 4,000,000 words each.

Output: JSONL files + metadata JSONs + screen summary JSON under
  experiments/archive/compact_experience/data/cs_4m_screen

Design: research/plans/compact_experience/cs_materializer_design.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TRAIN_FILES = [
    "bnc_spoken.train.txt",
    "childes.train.txt",
    "gutenberg.train.txt",
    "open_subtitles.train.txt",
    "simple_wiki.train.txt",
    "switchboard.train.txt",
]

FILLER_RE = re.compile(
    r"^(uh|um|erm|hmm|mhm|huh|yeah|yep|yup|nah|ok|okay|oh|ah|wow|hm|mm)$",
    re.IGNORECASE,
)
BOILERPLATE_RE = re.compile(r"^(= =|#|\*[A-Z]+:\t|[A-Z]:\t)")

CLAUSAL_DEPS = {"advcl", "ccomp", "xcomp", "relcl", "acl"}
CORE_ARG_DEPS = {"nsubj", "nsubjpass", "dobj", "obj", "iobj", "attr"}
MODIFIER_DEPS = {"amod", "advmod", "prep", "pobj", "det"}
FINITE_TAGS = {"VBZ", "VBP", "VBD", "VBN", "VBG", "MD"}
SUBJ_DEPS = {"nsubj", "nsubjpass", "expl"}


@dataclass
class PoolRow:
    example_id: int
    text: str
    words: int
    source: str
    c_score: float = 0.0
    s_score: float = 0.0
    token_count: int = 0
    sent_completeness: float = 0.0  # cached from the single spacy parse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--raw_dir",
        default="experiments/archive/initial_model_studies/training/runs"
        "fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/raw_dataset",
        help="Directory containing official .train.txt files",
    )
    p.add_argument(
        "--out_dir",
        default="experiments/archive/compact_experience/data/cs_4m_screen",
    )
    p.add_argument("--max_pool_words", type=int, default=10_000_000)
    p.add_argument("--words_per_example", type=int, default=160)
    p.add_argument("--target_words", type=int, default=4_000_000)
    p.add_argument("--selection_fraction", type=float, default=0.4)
    p.add_argument("--seed_ra", type=int, default=100)
    p.add_argument("--seed_rb", type=int, default=200)
    p.add_argument("--spacy_batch", type=int, default=256)
    p.add_argument(
        "--tokenizer_path",
        default="experiments/archive/initial_model_studies/training/runs"
        "fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model",
        help="Portable baseline16k tokenizer path from the verified research run",
    )
    p.add_argument("--dry_run", action="store_true", help="Score a small subset only")
    p.add_argument("--dry_run_rows", type=int, default=500)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Pool construction
# ---------------------------------------------------------------------------
def build_pool(raw_dir: Path, max_words: int, wpe: int) -> list[PoolRow]:
    """Reproduce iter_examples logic: pack words sequentially, 160 per row."""
    pool: list[PoolRow] = []
    used = 0
    buf: list[str] = []
    buf_source = ""
    eid = 0

    for fn in TRAIN_FILES:
        fp = raw_dir / fn
        if not fp.exists():
            raise FileNotFoundError(f"Missing: {fp}")
        with fp.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                for w in line.split():
                    if used >= max_words:
                        break
                    if not buf:
                        buf_source = fn
                    buf.append(w)
                    used += 1
                    if len(buf) >= wpe:
                        pool.append(PoolRow(
                            example_id=eid,
                            text=" ".join(buf),
                            words=len(buf),
                            source=buf_source,
                        ))
                        eid += 1
                        buf = []
                        buf_source = ""
                if used >= max_words:
                    break
        if used >= max_words:
            break

    if buf:
        pool.append(PoolRow(
            example_id=eid,
            text=" ".join(buf),
            words=len(buf),
            source=buf_source,
        ))

    return pool


# ---------------------------------------------------------------------------
# Scoring: C-score (spacy-based compositional structure density)
# ---------------------------------------------------------------------------
def score_c_batch(pool: list[PoolRow], nlp, batch_size: int = 256) -> None:
    """Score all rows in-place with C-score using spacy pipe."""
    texts = [r.text for r in pool]
    for i, doc in enumerate(nlp.pipe(texts, batch_size=batch_size, n_process=1)):
        row = pool[i]
        n_finite_verbs = 0
        n_clausal_deps = 0
        n_core_args = 0
        n_modifiers = 0
        n_entity_mentions = len(doc.ents)
        n_filler_tokens = 0

        for token in doc:
            tag = token.tag_
            dep = token.dep_
            if token.pos_ == "VERB" and tag in FINITE_TAGS:
                n_finite_verbs += 1
            if dep in CLAUSAL_DEPS:
                n_clausal_deps += 1
            if dep in CORE_ARG_DEPS:
                n_core_args += 1
            if dep in MODIFIER_DEPS:
                n_modifiers += 1
            if FILLER_RE.match(token.text):
                n_filler_tokens += 1

        # Sentence-level features
        n_complete_sents = 0
        n_fragment_sents = 0
        n_total_sents = 0
        for sent in doc.sents:
            n_total_sents += 1
            has_finite = False
            has_subj = False
            for t in sent:
                if t.pos_ == "VERB" and t.tag_ in FINITE_TAGS:
                    has_finite = True
                if t.dep_ in SUBJ_DEPS:
                    has_subj = True
            if has_finite and has_subj:
                n_complete_sents += 1
            if not has_finite:
                n_fragment_sents += 1

        # Boilerplate lines
        n_boilerplate = 0
        for line in row.text.split("\n") if "\n" in row.text else [row.text]:
            if BOILERPLATE_RE.match(line.strip()):
                n_boilerplate += 1

        c_raw = (
            n_finite_verbs
            + n_clausal_deps
            + n_core_args
            + 0.5 * n_modifiers
            + n_complete_sents
            + 0.5 * n_entity_mentions
        )
        c_penalty = (
            2.0 * n_fragment_sents
            + 1.0 * n_filler_tokens
            + 1.5 * n_boilerplate
        )
        row.c_score = (c_raw - c_penalty) / max(row.words, 1)
        row.sent_completeness = n_complete_sents / max(n_total_sents, 1)


# ---------------------------------------------------------------------------
# Scoring: S-score (tokenizer-based recoverable signal density)
# ---------------------------------------------------------------------------
def is_noise_word(w: str) -> bool:
    if w.isdigit():
        return True
    if all(not c.isalpha() for c in w):
        return True
    if len(w) == 1 and not w.isalpha():
        return True
    return False


def build_global_unigram_freq(pool: list[PoolRow], tokenizer) -> tuple[Counter, float, float]:
    """Build global token frequency table and compute p50/p95."""
    freq: Counter = Counter()
    for row in pool:
        ids = tokenizer.encode(row.text, add_special_tokens=False)
        freq.update(ids)
    counts = sorted(freq.values())
    if not counts:
        return freq, 0, 0
    p50 = float(np.percentile(counts, 50))
    p95 = float(np.percentile(counts, 95))
    return freq, p50, p95


def score_s_batch(
    pool: list[PoolRow],
    tokenizer,
    global_freq: Counter,
    p50: float,
    p95: float,
) -> None:
    """Score all rows in-place with S-score, reusing the cached spacy features."""
    for row in pool:
        ids = tokenizer.encode(row.text, add_special_tokens=False)
        token_count = len(ids)
        row.token_count = token_count

        if token_count == 0:
            row.s_score = 0.0
            continue

        unique_tokens = len(set(ids))
        # Moderate frequency ratio
        mod_count = sum(1 for tid in ids if p50 <= global_freq.get(tid, 0) <= p95)
        moderate_freq_ratio = mod_count / token_count

        # Local type-token ratio
        local_ttr = unique_tokens / token_count

        # Sentence completeness
        sent_completeness = row.sent_completeness

        # Bigram diversity
        if token_count >= 2:
            bigrams = [(ids[j], ids[j + 1]) for j in range(token_count - 1)]
            unique_bigrams = len(set(bigrams))
            bigram_diversity = unique_bigrams / len(bigrams)
        else:
            bigram_diversity = 1.0

        # Noise ratio
        words_list = row.text.split()
        noise_count = sum(1 for w in words_list if is_noise_word(w))
        noise_ratio = noise_count / max(row.words, 1)

        row.s_score = (
            0.35 * moderate_freq_ratio
            + 0.20 * local_ttr
            + 0.25 * sent_completeness
            + 0.15 * bigram_diversity
            - 0.30 * noise_ratio
        )


# ---------------------------------------------------------------------------
# Arm selection
# ---------------------------------------------------------------------------
def compute_source_quotas(
    pool: list[PoolRow], fraction: float
) -> dict[str, int]:
    """Compute per-source row quotas at the given fraction."""
    source_counts: Counter = Counter()
    for r in pool:
        source_counts[r.source] += 1

    quotas = {}
    for src, cnt in source_counts.items():
        quotas[src] = round(cnt * fraction)
    return quotas


def select_random_arm(
    pool: list[PoolRow], quotas: dict[str, int], seed: int
) -> list[PoolRow]:
    """Select rows uniformly at random within each source group."""
    import random

    rng = random.Random(seed)
    by_source: dict[str, list[PoolRow]] = {}
    for r in pool:
        by_source.setdefault(r.source, []).append(r)

    selected: list[PoolRow] = []
    for src in TRAIN_FILES:
        rows = by_source.get(src, [])
        q = quotas.get(src, 0)
        if q > len(rows):
            q = len(rows)
        chosen = rng.sample(rows, q)
        selected.extend(chosen)

    selected.sort(key=lambda r: r.example_id)
    return selected


def select_top_score_arm(
    pool: list[PoolRow],
    quotas: dict[str, int],
    score_key: str,
    ascending: bool = False,
) -> list[PoolRow]:
    """Select top (or bottom) scoring rows within each source group."""
    by_source: dict[str, list[PoolRow]] = {}
    for r in pool:
        by_source.setdefault(r.source, []).append(r)

    selected: list[PoolRow] = []
    for src in TRAIN_FILES:
        rows = by_source.get(src, [])
        q = quotas.get(src, 0)
        if q > len(rows):
            q = len(rows)
        rows_sorted = sorted(
            rows,
            key=lambda r: getattr(r, score_key),
            reverse=not ascending,
        )
        chosen = rows_sorted[:q]
        selected.extend(chosen)

    selected.sort(key=lambda r: r.example_id)
    return selected


# ---------------------------------------------------------------------------
# Statistics and validation
# ---------------------------------------------------------------------------
def score_stats(values: list[float]) -> dict:
    if not values:
        return {"mean": 0, "std": 0, "p10": 0, "p50": 0, "p90": 0, "min": 0, "max": 0}
    arr = np.array(values)
    return {
        "mean": round(float(arr.mean()), 6),
        "std": round(float(arr.std()), 6),
        "p10": round(float(np.percentile(arr, 10)), 6),
        "p50": round(float(np.percentile(arr, 50)), 6),
        "p90": round(float(np.percentile(arr, 90)), 6),
        "min": round(float(arr.min()), 6),
        "max": round(float(arr.max()), 6),
    }


def text_multiset_hash(rows: list[PoolRow]) -> str:
    """SHA256 of sorted word multiset for content identity."""
    h = hashlib.sha256()
    words = []
    for r in rows:
        words.extend(r.text.split())
    words.sort()
    h.update(" ".join(words).encode("utf-8"))
    return h.hexdigest()


def compute_arm_stats(
    rows: list[PoolRow], tokenizer
) -> dict:
    """Compute surface statistics for an arm."""
    total_words = sum(r.words for r in rows)
    total_tokens = sum(r.token_count for r in rows)

    source_quotas: dict[str, dict] = {}
    for r in rows:
        if r.source not in source_quotas:
            source_quotas[r.source] = {"rows": 0, "words": 0}
        source_quotas[r.source]["rows"] += 1
        source_quotas[r.source]["words"] += r.words

    # Type-token ratio and rare-word fraction
    all_words = []
    punct_digit_count = 0
    dialogue_marker_count = 0
    for r in rows:
        ws = r.text.split()
        all_words.extend(ws)
        for w in ws:
            if is_noise_word(w):
                punct_digit_count += 1
            if re.match(r"^\*[A-Z]+:$|^[A-Z]:$", w):
                dialogue_marker_count += 1

    word_freq = Counter(all_words)
    type_count = len(word_freq)
    ttr = type_count / max(total_words, 1)
    # Rare = appears only once
    rare_count = sum(1 for c in word_freq.values() if c == 1)
    rare_fraction = rare_count / max(type_count, 1)

    # Duplicate estimate
    text_set: set = set()
    exact_dups = 0
    for r in rows:
        if r.text in text_set:
            exact_dups += 1
        text_set.add(r.text)

    return {
        "total_words": total_words,
        "total_rows": len(rows),
        "source_quotas": source_quotas,
        "token_count": total_tokens,
        "tokens_per_word": round(total_tokens / max(total_words, 1), 4),
        "type_token_ratio": round(ttr, 6),
        "rare_word_fraction": round(rare_fraction, 6),
        "punct_digit_rate": round(punct_digit_count / max(total_words, 1), 6),
        "dialogue_marker_rate": round(dialogue_marker_count / max(total_words, 1), 6),
        "duplicate_estimate": {
            "exact_dup_rows": exact_dups,
            "near_dup_fraction": round(exact_dups / max(len(rows), 1), 6),
        },
    }


def validate_arm(rows: list[PoolRow], target_words: int, arm_name: str) -> list[str]:
    """Return list of validation errors (empty = pass)."""
    errors = []
    total = sum(r.words for r in rows)
    if total != target_words:
        errors.append(f"{arm_name}: total words {total} != target {target_words}")

    ids = [r.example_id for r in rows]
    if len(ids) != len(set(ids)):
        errors.append(f"{arm_name}: duplicate example_ids found")

    for r in rows:
        actual = len(r.text.split())
        if actual != r.words:
            errors.append(f"{arm_name}: row {r.example_id} words field {r.words} != actual {actual}")
            break  # one error is enough

    return errors


def validate_jsonl_dryrun(jsonl_path: Path, target_words: int) -> list[str]:
    """Simulate load_examples_jsonl validation."""
    errors = []
    selected = 0
    row_num = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            row_num += 1
            text = str(obj["text"])
            words = int(obj.get("words", len(text.split())))
            actual = len(text.split())
            if words != actual:
                errors.append(f"row {row_num}: words field {words} != actual {actual}")
                break
            if selected < target_words:
                if selected + words > target_words:
                    errors.append(
                        f"row {row_num}: partial example needed "
                        f"(selected={selected}, words={words}, target={target_words})"
                    )
                    break
                selected += words
    if selected != target_words and not errors:
        errors.append(f"selected {selected} != target {target_words}")
    return errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    t0 = time.time()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build pool
    print(json.dumps({"event": "building_pool", "raw_dir": str(raw_dir)}), flush=True)
    pool = build_pool(raw_dir, args.max_pool_words, args.words_per_example)
    total_pool_words = sum(r.words for r in pool)
    print(
        json.dumps({
            "event": "pool_built",
            "rows": len(pool),
            "total_words": total_pool_words,
            "elapsed_sec": round(time.time() - t0, 1),
        }),
        flush=True,
    )

    if args.dry_run:
        # Deterministic source-stratified sample rather than the first contiguous
        # rows (which would all come from bnc_spoken).
        by_source: dict[str, list[PoolRow]] = {}
        for row in pool:
            by_source.setdefault(row.source, []).append(row)
        base = args.dry_run_rows // len(TRAIN_FILES)
        remainder = args.dry_run_rows % len(TRAIN_FILES)
        sampled: list[PoolRow] = []
        for j, src in enumerate(TRAIN_FILES):
            take = base + (1 if j < remainder else 0)
            rows_src = by_source.get(src, [])
            if take >= len(rows_src):
                sampled.extend(rows_src)
            elif take > 0:
                idx = np.linspace(0, len(rows_src) - 1, num=take, dtype=int)
                sampled.extend(rows_src[int(k)] for k in idx)
        pool = sorted(sampled, key=lambda r: r.example_id)
        print(json.dumps({"event": "dry_run_stratified_sample", "rows": len(pool),
                          "source_rows": dict(Counter(r.source for r in pool))}), flush=True)

    # 2. Load spacy
    print(json.dumps({"event": "loading_spacy"}), flush=True)
    import spacy
    nlp = spacy.load("en_core_web_sm")
    # Disable NER if not needed for S-score to speed up; but we need it for C-score
    # Actually we need NER for entity mentions in C-score, so keep all components

    # 3. Score C
    print(json.dumps({"event": "scoring_c", "rows": len(pool)}), flush=True)
    tc = time.time()
    score_c_batch(pool, nlp, batch_size=args.spacy_batch)
    print(
        json.dumps({
            "event": "c_scored",
            "elapsed_sec": round(time.time() - tc, 1),
            "c_mean": round(np.mean([r.c_score for r in pool]), 4),
        }),
        flush=True,
    )

    # 4. Load tokenizer and build global freq
    print(json.dumps({"event": "loading_tokenizer"}), flush=True)
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    BASELINE_TOKENIZER_REPO = "BabyLM-community/BabyLM-2026-Baseline-GPT2-Strict"
    if args.tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True, local_files_only=True)
    else:
        # This network fallback is retained for portability, but the verified
        # research tokenizer path is the default to avoid the read-only shared cache.
        tokenizer = AutoTokenizer.from_pretrained(
            BASELINE_TOKENIZER_REPO, revision="main", use_fast=True
        )

    print(json.dumps({"event": "building_unigram_freq", "rows": len(pool)}), flush=True)
    tf = time.time()
    global_freq, p50, p95 = build_global_unigram_freq(pool, tokenizer)
    print(
        json.dumps({
            "event": "unigram_freq_built",
            "unique_tokens": len(global_freq),
            "p50": round(p50, 1),
            "p95": round(p95, 1),
            "elapsed_sec": round(time.time() - tf, 1),
        }),
        flush=True,
    )

    # 5. Score S
    print(json.dumps({"event": "scoring_s", "rows": len(pool)}), flush=True)
    ts = time.time()
    score_s_batch(pool, tokenizer, global_freq, p50, p95)
    print(
        json.dumps({
            "event": "s_scored",
            "elapsed_sec": round(time.time() - ts, 1),
            "s_mean": round(np.mean([r.s_score for r in pool]), 4),
        }),
        flush=True,
    )

    if args.dry_run:
        # In dry-run, just report score distributions and exit
        c_vals = [r.c_score for r in pool]
        s_vals = [r.s_score for r in pool]
        result = {
            "status": "dry_run_ok",
            "rows_scored": len(pool),
            "c_score_stats": score_stats(c_vals),
            "s_score_stats": score_stats(s_vals),
            "elapsed_sec": round(time.time() - t0, 1),
        }
        (out_dir / "dry_run_report.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print(json.dumps(result), flush=True)
        return

    # 6. Compute source quotas
    quotas = compute_source_quotas(pool, args.selection_fraction)
    quota_total_words = sum(q * args.words_per_example for q in quotas.values())
    print(
        json.dumps({
            "event": "quotas_computed",
            "quotas": quotas,
            "total_words": quota_total_words,
        }),
        flush=True,
    )
    if quota_total_words != args.target_words:
        print(
            json.dumps({
                "event": "ERROR",
                "msg": f"quota words {quota_total_words} != target {args.target_words}",
            }),
            flush=True,
        )
        sys.exit(1)

    # 7. Select arms
    arms: dict[str, list[PoolRow]] = {}
    print(json.dumps({"event": "selecting_arms"}), flush=True)

    arms["R-a"] = select_random_arm(pool, quotas, args.seed_ra)
    arms["R-b"] = select_random_arm(pool, quotas, args.seed_rb)
    arms["C-high"] = select_top_score_arm(pool, quotas, "c_score", ascending=False)
    arms["C-control"] = select_top_score_arm(pool, quotas, "c_score", ascending=True)
    arms["S-high"] = select_top_score_arm(pool, quotas, "s_score", ascending=False)

    # 8. Validate and write
    all_errors: list[str] = []
    arm_metas: dict[str, dict] = {}

    for arm_name, rows in arms.items():
        print(json.dumps({"event": "processing_arm", "arm": arm_name, "rows": len(rows)}), flush=True)

        # Validate
        errs = validate_arm(rows, args.target_words, arm_name)
        all_errors.extend(errs)

        # Write JSONL
        safe_name = arm_name.replace("-", "_").lower()
        jsonl_path = out_dir / f"{safe_name}.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for r in rows:
                obj = {
                    "text": r.text,
                    "words": r.words,
                    "example_id": r.example_id,
                    "source": r.source,
                    "kind": "official",
                    "c_score": round(r.c_score, 6),
                    "s_score": round(r.s_score, 6),
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

        # Dry-run JSONL validation
        jsonl_errs = validate_jsonl_dryrun(jsonl_path, args.target_words)
        all_errors.extend([f"{arm_name}/JSONL: {e}" for e in jsonl_errs])

        # Compute stats
        stats = compute_arm_stats(rows, tokenizer)
        c_vals = [r.c_score for r in rows]
        s_vals = [r.s_score for r in rows]

        meta = {
            "arm_name": arm_name,
            "total_words": stats["total_words"],
            "total_rows": stats["total_rows"],
            "source_quotas": stats["source_quotas"],
            "selection_method": {
                "R-a": "random sample per source, seed=100",
                "R-b": "random sample per source, seed=200",
                "C-high": "top c_score per source",
                "C-control": "bottom c_score per source",
                "S-high": "top s_score per source",
            }.get(arm_name, "unknown"),
            "c_score_stats": score_stats(c_vals),
            "s_score_stats": score_stats(s_vals),
            "token_count": stats["token_count"],
            "tokens_per_word": stats["tokens_per_word"],
            "type_token_ratio": stats["type_token_ratio"],
            "rare_word_fraction": stats["rare_word_fraction"],
            "punct_digit_rate": stats["punct_digit_rate"],
            "dialogue_marker_rate": stats["dialogue_marker_rate"],
            "duplicate_estimate": stats["duplicate_estimate"],
            "example_ids": [r.example_id for r in rows],
            "text_multiset_hash": text_multiset_hash(rows),
        }

        meta_path = out_dir / f"{safe_name}_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        arm_metas[arm_name] = meta

    # 9. Overlap analysis
    c_high_ids = set(arm_metas["C-high"]["example_ids"])
    s_high_ids = set(arm_metas["S-high"]["example_ids"])
    c_ctrl_ids = set(arm_metas["C-control"]["example_ids"])
    ra_ids = set(arm_metas["R-a"]["example_ids"])
    rb_ids = set(arm_metas["R-b"]["example_ids"])

    ch_sh_shared = c_high_ids & s_high_ids
    ch_cc_shared = c_high_ids & c_ctrl_ids
    ra_rb_shared = ra_ids & rb_ids

    overlap = {
        "c_high_s_high": {
            "shared_ids": len(ch_sh_shared),
            "fraction_of_c_high": round(len(ch_sh_shared) / max(len(c_high_ids), 1), 4),
            "fraction_of_s_high": round(len(ch_sh_shared) / max(len(s_high_ids), 1), 4),
        },
        "c_high_c_control": {
            "shared_ids": len(ch_cc_shared),
            "note": "should be 0 by construction",
        },
        "r_a_r_b": {
            "shared_ids": len(ra_rb_shared),
            "fraction_of_r_a": round(len(ra_rb_shared) / max(len(ra_ids), 1), 4),
        },
    }

    if len(ch_cc_shared) > 0:
        all_errors.append("C-high and C-control share rows — this should not happen")

    if overlap["c_high_s_high"]["fraction_of_c_high"] > 0.60:
        all_errors.append(
            f"WARNING: C-high/S-high overlap {overlap['c_high_s_high']['fraction_of_c_high']:.1%} > 60%"
        )

    # 10. Screen summary
    summary = {
        "screen_name": "cs_4m_matched_content",
        "pool_total_words": total_pool_words,
        "pool_total_rows": len(pool) if not args.dry_run else "N/A",
        "arms": list(arms.keys()),
        "words_per_arm": args.target_words,
        "source_matching": "proportional to official corpus",
        "source_quotas": quotas,
        "overlap": overlap,
        "global_unigram_freq_p50": round(p50, 1),
        "global_unigram_freq_p95": round(p95, 1),
        "scoring_notes": "spacy en_core_web_sm for C-score, baseline16k tokenizer for S-score",
        "pool_construction": f"iter_examples(TRAIN_FILES, max_words={args.max_pool_words}, "
        f"words_per_example={args.words_per_example})",
        "trainer_interface": "--example_jsonl <arm>.jsonl --max_word_exposure 4000000",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "elapsed_sec": round(time.time() - t0, 1),
        "validation_errors": all_errors,
        "validation_passed": len(all_errors) == 0,
    }
    (out_dir / "screen_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 11. Final status
    status = {
        "status": "ok" if not all_errors else "errors",
        "arms_written": list(arms.keys()),
        "out_dir": str(out_dir),
        "summary": str(out_dir / "screen_summary.json"),
        "validation_errors": all_errors,
        "overlap_c_high_s_high": overlap["c_high_s_high"]["fraction_of_c_high"],
        "elapsed_sec": round(time.time() - t0, 1),
    }
    print(json.dumps(status), flush=True)

    if all_errors:
        for e in all_errors:
            print(f"VALIDATION ERROR: {e}", file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
