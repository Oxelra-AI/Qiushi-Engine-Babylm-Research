#!/usr/bin/env python3
"""research: repair the FW source-breadth arm so independent FineWeb
companion material is packed as whole sentences inside each training example.

research correctly matched the compact and source-breadth arms in word totals,
row lengths, tokenizer, and pass order, but it filled each row's companion budget
by slicing a global word stream. That made many independent FineWeb sentences
cross example boundaries. This script preserves the same scientific comparison
while removing that avoidable confound:

  compact_view row = selected source span + compact rewrite span
  repaired source_breadth row = the same selected source span + whole independent
                                FineWeb sentences with the identical companion
                                word count for that row

The repaired arm keeps exact 10M total words, the exact compact-row word sequence,
the same filler rows, the same shared tokenizer, zero source-hash overlap with the
compact-pair sources, and the same 100M pass order for later training.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import math
import pathlib
import random
import statistics
import sys
import time
from typing import Any

ROOT = pathlib.Path("experiments/archive/representation_and_objectives")
SCRIPT = ROOT / "scripts/fw_source_breadth_arm.py"
COMPACT_100M = ROOT / "data/fw_source_breadth_arm/fw_preserved_compact_view_100M.jsonl"
MANIFEST = ROOT / "data/fw_source_breadth_arm/fw_source_breadth_arm_manifest.json"
TOKENIZER_DIR = ROOT / "data/shared_tokenizer/shared_16k_tokenizer"
OUT_DIR = ROOT / "data/fw_source_breadth_wholesentence_arm"
NOTE = (ROOT.parents[2] / 'research/notes/representation_and_objectives/fw_source_breadth_wholesentence_repair.md')

TOTAL_WORDS = 10_000_000
PASSES = 10
MAX_PARTS = 5
DEFAULT_SEED = 10482931
DEFAULT_STREAM_SEED = 10289931
FW_COMPACT_LABEL = "fw_preserved_compact_view"
FW_BREADTH_LABEL = "fw_preserved_source_breadth_wholesentence"
DOC_CAP = 6


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def wc(text: str) -> int:
    return len(" ".join((text or "").split()).split())


def norm_text(text: str) -> str:
    return " ".join((text or "").split())


def norm_hash(text: str) -> str:
    return hashlib.sha256(norm_text(text).lower().encode("utf-8")).hexdigest()[:32]


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def iter_jsonl(path: pathlib.Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_step102_module():
    spec = importlib.util.spec_from_file_location("fw_source_breadth_arm", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def stats(vals: list[int | float]) -> dict[str, Any]:
    if not vals:
        return {"n": 0}
    arr = sorted(float(x) for x in vals)
    def q(p: float) -> float:
        if len(arr) == 1:
            return arr[0]
        pos = p * (len(arr) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(arr) - 1)
        a = pos - lo
        return arr[lo] * (1 - a) + arr[hi] * a
    return {
        "n": len(arr),
        "sum": float(sum(arr)),
        "min": arr[0],
        "p01": q(0.01),
        "p05": q(0.05),
        "p10": q(0.10),
        "p25": q(0.25),
        "median": q(0.50),
        "p75": q(0.75),
        "p90": q(0.90),
        "p95": q(0.95),
        "p99": q(0.99),
        "max": arr[-1],
        "mean": float(statistics.mean(arr)),
    }


def sort_candidates(candidates: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for c in candidates:
        d = dict(c)
        d["_rand"] = rng.random()
        d["norm_hash"] = str(d.get("norm_hash") or norm_hash(str(d.get("text") or "")))
        d["text"] = norm_text(str(d.get("text") or ""))
        d["words"] = int(d.get("words") or wc(d["text"]))
        d["doc_id"] = str(d.get("doc_id") or f"docless::{d['norm_hash']}")
        if not d["text"] or d["words"] != wc(d["text"]):
            raise RuntimeError(f"bad candidate word count for {d.get('norm_hash')}")
        out.append(d)
    out.sort(key=lambda c: (int(c.get("source_priority") or 999), -float(c.get("quality_score") or 0.0), float(c["_rand"])))
    return out


def generate_length_combos(total: int, allowed_lengths: set[int], max_parts: int = MAX_PARTS) -> list[tuple[int, ...]]:
    allowed = sorted(x for x in allowed_lengths if 0 < x <= total)
    out: list[tuple[int, ...]] = []

    def rec(start_i: int, remaining: int, parts: list[int]) -> None:
        if remaining == 0:
            out.append(tuple(parts))
            return
        if len(parts) >= max_parts:
            return
        # Need nondecreasing lengths to avoid duplicate patterns.
        for k in range(start_i, len(allowed)):
            w = allowed[k]
            if w > remaining:
                break
            # Simple lower-bound pruning.
            rem_slots = max_parts - len(parts) - 1
            if remaining - w > 0 and rem_slots <= 0:
                continue
            parts.append(w)
            rec(k, remaining - w, parts)
            parts.pop()
    rec(0, total, [])
    return out


def make_buckets(candidates: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    buckets: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for c in candidates:
        buckets[int(c["words"])].append(c)
    return buckets


def count_available_by_length(buckets: dict[int, list[dict[str, Any]]], used: set[str], doc_counts: collections.Counter[str]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for L, arr in buckets.items():
        n = 0
        for c in arr:
            h = str(c["norm_hash"])
            if h in used:
                continue
            doc = str(c.get("doc_id") or f"docless::{h}")
            if doc_counts[doc] >= DOC_CAP:
                continue
            n += 1
        counts[L] = n
    return counts


def try_pattern(pattern: tuple[int, ...], buckets: dict[int, list[dict[str, Any]]], used: set[str], doc_counts: collections.Counter[str]) -> list[dict[str, Any]] | None:
    chosen: list[dict[str, Any]] = []
    chosen_hashes: set[str] = set()
    temp_docs: collections.Counter[str] = collections.Counter()
    need_by_len = collections.Counter(pattern)
    for L, need in sorted(need_by_len.items()):
        found: list[dict[str, Any]] = []
        for c in buckets.get(L, []):
            h = str(c["norm_hash"])
            if h in used or h in chosen_hashes:
                continue
            doc = str(c.get("doc_id") or f"docless::{h}")
            if doc_counts[doc] + temp_docs[doc] >= DOC_CAP:
                continue
            found.append(c)
            chosen_hashes.add(h)
            temp_docs[doc] += 1
            if len(found) == need:
                break
        if len(found) < need:
            return None
        chosen.extend(found)
    # Restore original pattern order roughly by length; content order within row is not an evaluation signal.
    chosen.sort(key=lambda c: (int(c["words"]), int(c.get("source_priority") or 999), -float(c.get("quality_score") or 0.0), float(c.get("_rand") or 0.0)))
    return chosen


def select_whole_sentence_by_row(row_metas: list[dict[str, Any]], candidates: list[dict[str, Any]], seed: int) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    candidates = sort_candidates(candidates, seed)
    buckets = make_buckets(candidates)
    allowed_lengths = set(buckets)
    budgets = [int(m["companion_words"]) for m in row_metas]
    pattern_cache = {b: generate_length_combos(b, allowed_lengths, MAX_PARTS) for b in sorted(set(budgets))}
    missing = {b: len(pats) for b, pats in pattern_cache.items() if not pats}
    if missing:
        raise RuntimeError(f"some budgets have no exact whole-sentence length pattern: {missing}")

    # Harder budgets first: rare budgets and fewer patterns; output order is restored later.
    budget_counts = collections.Counter(budgets)
    row_order = sorted(range(len(row_metas)), key=lambda i: (len(pattern_cache[budgets[i]]), budget_counts[budgets[i]], budgets[i], i))
    used: set[str] = set()
    doc_counts: collections.Counter[str] = collections.Counter()
    chosen_by_row: list[list[dict[str, Any]]] = [[] for _ in row_metas]
    selected_words = 0
    fail_trace: list[dict[str, Any]] = []

    # Initial abundance is used as a stabilizer so greedy choices do not consume rare lengths too early.
    initial_counts = collections.Counter(int(c["words"]) for c in candidates)

    for t, row_i in enumerate(row_order):
        b = budgets[row_i]
        pats = pattern_cache[b]
        avail = count_available_by_length(buckets, used, doc_counts)
        feasible_scored: list[tuple[tuple[float, ...], tuple[int, ...], list[dict[str, Any]]]] = []
        # Dynamic ranking: first use few whole sentences, then avoid scarce lengths, then prefer balanced pieces.
        for pat in pats:
            pc = collections.Counter(pat)
            if any(avail.get(L, 0) < need for L, need in pc.items()):
                continue
            chosen = try_pattern(pat, buckets, used, doc_counts)
            if chosen is None:
                continue
            scarcity = sum(pc[L] / max(1, avail.get(L, 0)) for L in pc)
            original_scarcity = sum(pc[L] / max(1, initial_counts.get(L, 0)) for L in pc)
            imbalance = max(pat) - min(pat) if len(pat) > 1 else 0
            quality = statistics.mean(float(c.get("quality_score") or 0.0) for c in chosen)
            priority = statistics.mean(float(c.get("source_priority") or 999) for c in chosen)
            # Prefer 1-2 sentence packs when available; for equal part count use abundant lengths and high-quality pools.
            score = (len(pat), scarcity, original_scarcity, imbalance / max(1, b), priority, -quality)
            feasible_scored.append((score, pat, chosen))
            # Keep search bounded once many good patterns are available.
            if len(feasible_scored) >= 250:
                break
        if not feasible_scored:
            fail_trace.append({"row_index": row_i, "budget": b, "remaining_by_length": {str(k): int(v) for k, v in sorted(avail.items()) if v}})
            raise RuntimeError(f"no feasible whole-sentence exact fill for row {row_i}, budget {b}; saved trace has {len(fail_trace)} failure(s)")
        feasible_scored.sort(key=lambda x: x[0])
        _, pat, chosen = feasible_scored[0]
        if sum(int(c["words"]) for c in chosen) != b:
            raise RuntimeError(f"chosen pattern sum mismatch row={row_i} pat={pat}")
        chosen_by_row[row_i] = chosen
        for c in chosen:
            h = str(c["norm_hash"])
            if h in used:
                raise RuntimeError(f"duplicate source selected: {h}")
            used.add(h)
            doc_counts[str(c.get("doc_id") or f"docless::{h}")] += 1
            selected_words += int(c["words"])

    if selected_words != sum(budgets):
        raise RuntimeError(f"selected words {selected_words} != budget {sum(budgets)}")

    selected_flat = [c for row in chosen_by_row for c in row]
    selected_by_pool = collections.Counter(str(c.get("source_pool") or "unknown") for c in selected_flat)
    selected_words_by_pool = collections.Counter()
    selected_words_by_domain = collections.Counter()
    selected_rows_by_domain = collections.Counter()
    for c in selected_flat:
        selected_words_by_pool[str(c.get("source_pool") or "unknown")] += int(c["words"])
        selected_words_by_domain[str(c.get("primary_domain") or "no_domain")] += int(c["words"])
        selected_rows_by_domain[str(c.get("primary_domain") or "no_domain")] += 1
    per_row_source_counts = [len(row) for row in chosen_by_row]
    per_row_source_words = [int(c["words"]) for row in chosen_by_row for c in row]
    summary = {
        "seed": seed,
        "row_fill_order": "harder_budgets_first_restore_original_output_order",
        "selected_rows": len(selected_flat),
        "selected_words": selected_words,
        "unique_docs": len(doc_counts),
        "doc_cap": DOC_CAP,
        "max_doc_use": max(doc_counts.values()) if doc_counts else 0,
        "per_row_source_count_stats": stats(per_row_source_counts),
        "selected_source_word_stats": stats(per_row_source_words),
        "selected_rows_by_pool": dict(selected_by_pool),
        "selected_words_by_pool": dict(selected_words_by_pool),
        "selected_rows_by_domain": dict(selected_rows_by_domain),
        "selected_words_by_domain": dict(selected_words_by_domain),
        "budgets_by_part_count": {str(k): int(v) for k, v in sorted(collections.Counter(per_row_source_counts).items())},
        "length_pattern_availability": {str(k): len(v) for k, v in sorted(pattern_cache.items())},
    }
    return chosen_by_row, summary


def build_repaired_rows(row_metas: list[dict[str, Any]], chosen_by_row: list[list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    fw_rows: list[dict[str, Any]] = []
    row_sidecar: list[dict[str, Any]] = []
    selected_sidecar: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    for i, (m, chosen) in enumerate(zip(row_metas, chosen_by_row)):
        budget = int(m["companion_words"])
        companion_words = sum(int(c["words"]) for c in chosen)
        if companion_words != budget:
            raise RuntimeError(f"row {i} companion sum {companion_words} != budget {budget}")
        companion_text = " ".join(c["text"] for c in chosen)
        source_text = norm_text(str(m["source_text"]))
        text = norm_text(f"{source_text} {companion_text}")
        if wc(text) != int(m["total_words"]):
            raise RuntimeError(f"row {i} total word mismatch: {wc(text)} != {m['total_words']}")
        fw_rows.append({
            "text": text,
            "words": int(m["total_words"]),
            "example_id": int(m["example_id"]),
            "source": FW_BREADTH_LABEL,
        })
        row_sidecar.append({
            "row_index": i,
            "example_id": int(m["example_id"]),
            "pair_ids": m["pair_ids"],
            "common_source_words": int(m["source_words"]),
            "breadth_companion_words": budget,
            "total_words": int(m["total_words"]),
            "breadth_source_ids": [str(c["norm_hash"]) for c in chosen],
            "breadth_source_words": [int(c["words"]) for c in chosen],
            "whole_sentences": True,
        })
        for c in chosen:
            h = str(c["norm_hash"])
            if h not in seen_sources:
                selected_sidecar.append({
                    "norm_hash": h,
                    "text": c["text"],
                    "words": int(c["words"]),
                    "doc_id": str(c.get("doc_id") or ""),
                    "domains": c.get("domains") or [],
                    "primary_domain": str(c.get("primary_domain") or "no_domain"),
                    "source_pool": str(c.get("source_pool") or "unknown"),
                    "source_path": str(c.get("source_path") or ""),
                    "source_priority": int(c.get("source_priority") or 999),
                    "quality_score": float(c.get("quality_score") or 0.0),
                    "sentence_index": c.get("sentence_index"),
                    "_rand": float(c.get("_rand") or 0.0),
                })
                seen_sources.add(h)
    return fw_rows, row_sidecar, selected_sidecar


def pool_word_count(rows: list[dict[str, Any]]) -> int:
    total = 0
    for r in rows:
        text = str(r.get("text") or "")
        w = int(r.get("words") or wc(text))
        if w != wc(text):
            raise RuntimeError(f"row word field mismatch source={r.get('source')} example_id={r.get('example_id')}")
        total += w
    return total


def write_training_stream(path: pathlib.Path, rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    n = len(rows)
    total = 0
    order_hashes: list[str] = []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for pass_i in range(PASSES):
            order = list(range(n))
            random.Random(seed + 1000 + pass_i).shuffle(order)
            order_hashes.append(sha256_text(",".join(map(str, order))))
            for idx in order:
                r = rows[idx]
                rec = {"text": r["text"], "words": int(r["words"]), "example_id": int(r.get("example_id") or 0), "source": str(r.get("source") or "")}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total += int(r["words"])
    if total != TOTAL_WORDS * PASSES:
        raise RuntimeError(f"training stream total {total} != {TOTAL_WORDS * PASSES}")
    return {
        "path": str(path),
        "rows": n * PASSES,
        "words": total,
        "sha256": sha256_file(path),
        "pass_order_hashes": order_hashes,
        "seed": seed,
    }


def arm_summary(path: pathlib.Path) -> dict[str, Any]:
    rows = 0
    words = 0
    source_words: collections.Counter[str] = collections.Counter()
    word_seq: list[int] = []
    mismatches: list[dict[str, Any]] = []
    for i, r in enumerate(iter_jsonl(path), 1):
        rows += 1
        text = str(r.get("text") or "")
        w = int(r.get("words") or wc(text))
        actual = wc(text)
        if w != actual and len(mismatches) < 20:
            mismatches.append({"line": i, "field_words": w, "actual_words": actual, "source": r.get("source")})
        words += w
        word_seq.append(w)
        source_words[str(r.get("source") or "")] += w
    return {"path": str(path), "rows": rows, "words": words, "sha256": sha256_file(path), "source_words": dict(source_words), "word_seq": word_seq, "word_mismatches": mismatches}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--stream-seed", type=int, default=DEFAULT_STREAM_SEED)
    ap.add_argument("--write-training-stream", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    breadth_builder = load_step102_module()
    pairs = breadth_builder.load_pairs()
    row_metas = breadth_builder.pack_pair_metas(pairs)
    compact_fw, filler_rows = breadth_builder.load_compact_arm()
    if len(compact_fw) != len(row_metas):
        raise RuntimeError(f"FW row count mismatch compact={len(compact_fw)} metas={len(row_metas)}")
    if [int(r["words"]) for r in compact_fw] != [int(m["total_words"]) for m in row_metas]:
        raise RuntimeError("FineWeb row word sequence differs from compact arm")

    pair_source_words = sum(int(p["source_words"]) for p in pairs)
    companion_budget = sum(int(p["rewrite_words"]) for p in pairs)
    pair_total_words = pair_source_words + companion_budget
    filler_words = pool_word_count(filler_rows)
    if filler_words + pair_total_words != TOTAL_WORDS:
        raise RuntimeError(f"filler+pair words {filler_words}+{pair_total_words} != {TOTAL_WORDS}")

    exclude_hashes = {str(p["norm_hash"]) for p in pairs}
    candidates, candidate_summary = breadth_builder.load_candidate_sources(exclude_hashes)
    chosen_by_row, selection_summary = select_whole_sentence_by_row(row_metas, candidates, args.seed)
    breadth_fw_rows, row_sidecar, selected_sidecar = build_repaired_rows(row_metas, chosen_by_row)

    breadth_pool = list(breadth_fw_rows) + list(filler_rows)
    compact_pool = list(compact_fw) + list(filler_rows)
    if [int(r["words"]) for r in breadth_pool] != [int(r["words"]) for r in compact_pool]:
        raise RuntimeError("complete arm row word sequence changed")
    total_words = pool_word_count(breadth_pool)
    if total_words != TOTAL_WORDS:
        raise RuntimeError(f"breadth arm words {total_words} != {TOTAL_WORDS}")
    selected_hashes = {str(r["norm_hash"]) for r in selected_sidecar}
    overlap = sorted(selected_hashes & exclude_hashes)
    if overlap:
        raise RuntimeError(f"selected breadth sources overlap compact pair sources: {overlap[:5]}")

    arm_path = OUT_DIR / "fw_preserved_source_breadth_wholesentence_10M.jsonl"
    selected_path = OUT_DIR / "source_breadth_wholesentence_companion_sources.jsonl"
    row_meta_path = OUT_DIR / "source_breadth_wholesentence_row_meta.jsonl"
    write_jsonl(arm_path, breadth_pool)
    write_jsonl(selected_path, selected_sidecar)
    write_jsonl(row_meta_path, row_sidecar)

    stream_rec: dict[str, Any] = {}
    if args.write_training_stream:
        stream_path = OUT_DIR / "fw_preserved_source_breadth_wholesentence_100M.jsonl"
        stream_rec = write_training_stream(stream_path, breadth_pool, args.stream_seed)

    compact_summary = arm_summary(breadth_builder.COMPACT_ARM)
    breadth_summary = arm_summary(arm_path)
    compact_seq = compact_summary.pop("word_seq")
    breadth_seq = breadth_summary.pop("word_seq")
    if compact_seq != breadth_seq:
        raise RuntimeError("arm word sequences differ after writing")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    compact_stream_rec = (manifest.get("training_streams") or {}).get("compact_view") or {}
    pass_orders_match_existing_compact = None
    if stream_rec and compact_stream_rec.get("pass_order_hashes"):
        pass_orders_match_existing_compact = stream_rec["pass_order_hashes"] == compact_stream_rec.get("pass_order_hashes")
        if not pass_orders_match_existing_compact:
            raise RuntimeError("repaired breadth pass orders do not match existing compact stream pass orders")

    manifest = {
        "status": "FW_SOURCE_BREADTH_WHOLESENTENCE_READY",
        "created_utc": now_utc(),
        "scientific_purpose": "Repair the source-breadth arm so the compact-vs-breadth comparison tests aligned compact re-expression against coherent independent FineWeb experience, not against sentence-fragmented breadth.",
        "inputs": {
            "script": str(SCRIPT),
            "manifest": str(MANIFEST),
            "compact_arm_10m": str(breadth_builder.COMPACT_ARM),
            "compact_stream_100m": str(COMPACT_100M),
            "shared_tokenizer": str(TOKENIZER_DIR),
        },
        "budgets": {
            "total_words": total_words,
            "rows_total": len(breadth_pool),
            "fineweb_rows": len(breadth_fw_rows),
            "filler_rows_reused": len(filler_rows),
            "filler_words_reused": filler_words,
            "common_fineweb_source_words": pair_source_words,
            "breadth_companion_words": companion_budget,
            "fineweb_block_words": pair_total_words,
            "row_word_sequence_matches_compact": True,
            "whole_sentence_companion_rows": len(breadth_fw_rows),
        },
        "candidate_sources": {
            "total_candidates_after_excluding_pair_sources": len(candidates),
            "total_candidate_words": int(sum(int(c["words"]) for c in candidates)),
            "by_input_pool": candidate_summary,
        },
        "selection": selection_summary,
        "integrity": {
            "selected_breadth_hash_overlap_with_compact_pair_sources": len(overlap),
            "source_sentences_split_across_rows": 0,
            "all_row_companions_whole_sentences": True,
            "compact_and_breadth_row_word_sequence_match": True,
            "compact_arm_summary": compact_summary,
            "breadth_arm_summary": breadth_summary,
        },
        "training_streams": {
            "compact_view_existing": compact_stream_rec,
            "source_breadth_wholesentence": stream_rec,
            "pass_orders_match_existing_compact": pass_orders_match_existing_compact,
        },
        "files": {
            "source_breadth_wholesentence_10m": str(arm_path),
            "selected_sources": str(selected_path),
            "row_meta": str(row_meta_path),
            "source_breadth_wholesentence_100m": str(stream_rec.get("path") or ""),
            "manifest": str(OUT_DIR / "fw_source_breadth_wholesentence_manifest.json"),
            "note": str(NOTE),
        },
        "hashes": {
            "source_breadth_wholesentence_10m": sha256_file(arm_path),
            "selected_sources": sha256_file(selected_path),
            "row_meta": sha256_file(row_meta_path),
        },
        "elapsed_sec": round(time.time() - t0, 2),
    }
    if stream_rec:
        manifest["hashes"]["source_breadth_wholesentence_100m"] = stream_rec["sha256"]
    manifest_path = OUT_DIR / "fw_source_breadth_wholesentence_manifest.json"
    manifest["files"]["manifest"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text(
        "# research — repaired FineWeb source-breadth arm with whole-sentence companions\n\n"
        "The research source-breadth arm preserved the compact arm's row word sequence by slicing a global stream of independent FineWeb words. "
        "The boundary check found that this made almost every breadth row contain a source-sentence fragment. The repaired arm keeps the same comparison but fills each row's companion budget with whole independent FineWeb sentences.\n\n"
        "## Preserved structure\n\n"
        f"- Total words: {total_words:,}; rows: {len(breadth_pool):,}; FineWeb rows: {len(breadth_fw_rows):,}.\n"
        f"- Common selected FineWeb source words: {pair_source_words:,}; repaired independent companion words: {companion_budget:,}; FineWeb block: {pair_total_words:,}.\n"
        f"- Row word sequence matches compact arm: yes; source-sentence splits across breadth rows: 0.\n"
        f"- Selected independent sources: {selection_summary['selected_rows']:,} sentences / {selection_summary['selected_words']:,} words from {selection_summary['unique_docs']:,} docs, max doc use {selection_summary['max_doc_use']} (cap {DOC_CAP}).\n"
        f"- Per-row companion source count: mean {selection_summary['per_row_source_count_stats']['mean']:.3f}, median {selection_summary['per_row_source_count_stats']['median']:.0f}, max {selection_summary['per_row_source_count_stats']['max']:.0f}.\n"
        f"- Hash overlap with compact-pair selected sources: {len(overlap)}.\n"
        f"- 100M stream written: {bool(stream_rec)}; pass order matches existing compact stream: {pass_orders_match_existing_compact}.\n\n"
        "## Use for training\n\n"
        "Use this repaired whole-sentence source-breadth arm rather than the research sliced source-breadth stream if the compact-vs-breadth H100 comparison is launched. The compact arm and shared tokenizer are unchanged.\n\n"
        f"Manifest: `{manifest_path}`\n"
        f"10M arm: `{arm_path}`\n"
        f"100M stream: `{stream_rec.get('path') or ''}`\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": manifest["status"],
        "total_words": total_words,
        "rows_total": len(breadth_pool),
        "fineweb_rows": len(breadth_fw_rows),
        "selected_sources": selection_summary["selected_rows"],
        "selected_words": selection_summary["selected_words"],
        "source_sentence_splits": 0,
        "row_word_sequence_matches_compact": True,
        "stream_written": bool(stream_rec),
        "pass_orders_match_existing_compact": pass_orders_match_existing_compact,
        "manifest": str(manifest_path),
        "note": str(NOTE),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
