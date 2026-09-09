#!/usr/bin/env python3
"""Materialize a FineWeb second-view density contrast without mixing regimes.

Three mechanisms are distinguished:
  1. exact / packet-local repetition of a source sentence;
  2. faithful near-length generated second views;
  3. faithful compact generated second views, where saved rewrite words are spent
     on additional distinct source facts within the same corpus word budget.

This script consumes analyzed generation rows for a near-length regime and a compact
regime.  It creates two matched families, each with its own official length-matched
control so FineWeb-vs-official effects remain separable from view-density effects:

  near family:
    official_lengthmatched_near
    fineweb_repeat_near
    fineweb_near_view

  compact family:
    official_lengthmatched_compact
    fineweb_repeat_compact_reinvest
    fineweb_compact_view_reinvest

Within each family, the repeat and view arms have identical row lengths and identical
filler; the official arm matches that family's row lengths.  Across families the
changed block has the same word budget, but compact can include more sources because
its accepted rewrites are shorter.  That is the intended information-density contrast.

No BabyLM evaluation labels or scores are read.
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import json
import pathlib
import random
import re
import statistics
import time
from typing import Any, Iterable

WORKSPACE = pathlib.Path("experiments/archive/frontier_consolidation")
OFFICIAL_POOL_DEFAULT = pathlib.Path("experiments/archive/compact_experience/data/mixture/official_pool.jsonl")
NEAR_ROWS_DEFAULT = WORKSPACE / "data/high_precision_near_analysis/fineweb_high_anchor_generation_rows.jsonl"
COMPACT_ROWS_DEFAULT = WORKSPACE / "data/high_precision_compact_analysis/fineweb_high_anchor_generation_rows.jsonl"
OUT_DEFAULT = WORKSPACE / "data/density_contrast_materialized"
NOTE_DEFAULT = (WORKSPACE.parents[2] / 'research/notes/frontier_consolidation/density_contrast_materialization.md')

TOTAL_WORDS = 10_000_000
PASSES = 10
OFFICIAL_ROW_WORDS = 160
MAX_PACKET_WORDS = 160
RNG_SEED = 82913013

WORD_RE = re.compile(r"[A-Za-z0-9]+")


@dataclasses.dataclass
class Pair:
    pair_id: str
    source_text: str
    rewrite_text: str
    source_words: int
    rewrite_words: int
    pair_words: int
    sentence_id: str
    doc_id: str
    domain_hits: list[str]
    regime: str
    content_recall: float | None = None
    content_overlap: float | None = None
    analyzer_length_ratio: float | None = None
    soft_flags: list[str] | None = None
    source_risks: list[str] | None = None


@dataclasses.dataclass
class Row:
    text: str
    words: int
    example_id: int
    source: str
    pair_ids: list[str] | None = None
    component_sources: dict[str, int] | None = None


def wc(text: str) -> int:
    return len((text or "").split())


def norm_text(text: str) -> str:
    return " ".join((text or "").replace("\u00a0", " ").split())


def norm_for_copy(text: str) -> str:
    return " ".join(WORD_RE.findall((text or "").lower()))


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def write_jsonl(path: pathlib.Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def source_key(row: dict[str, Any]) -> str:
    sid = row.get("sentence_id")
    did = row.get("doc_id")
    src = norm_text(str(row.get("source_text") or ""))
    if sid is not None and str(sid) != "":
        return f"sid:{sid}|doc:{did}"
    return "txt:" + hashlib.sha1(src.encode("utf-8")).hexdigest()[:16]


def coerce_pair(row: dict[str, Any], regime: str) -> Pair | None:
    if not row.get("accepted_for_next_construction", False):
        return None
    src = norm_text(str(row.get("source_text") or row.get("original") or ""))
    rew = norm_text(str(row.get("rewrite_text") or row.get("rewrite") or row.get("output") or ""))
    if not src or not rew:
        return None
    sw, rw = wc(src), wc(rew)
    if sw <= 0 or rw <= 0:
        return None
    pid_base = row.get("prompt_id") or row.get("pair_id") or source_key(row)
    return Pair(
        pair_id=f"{regime}:{pid_base}",
        source_text=src,
        rewrite_text=rew,
        source_words=sw,
        rewrite_words=rw,
        pair_words=sw + rw,
        sentence_id=str(row.get("sentence_id") or ""),
        doc_id=str(row.get("doc_id") or ""),
        domain_hits=[str(x) for x in (row.get("domain_hits") or [])],
        regime=regime,
        content_recall=float(row.get("content_recall")) if row.get("content_recall") is not None else None,
        content_overlap=float(row.get("content_overlap")) if row.get("content_overlap") is not None else None,
        analyzer_length_ratio=float(row.get("length_ratio")) if row.get("length_ratio") is not None else None,
        soft_flags=[str(x) for x in (row.get("soft_flags") or [])],
        source_risks=[str(x) for x in (row.get("source_risks") or [])],
    )


def filter_pairs(rows: list[dict[str, Any]], regime: str, *, near_min_ratio: float, near_max_ratio: float,
                 compact_min_ratio: float, compact_max_ratio: float, compact_min_recall: float,
                 reject_exact_copy: bool, reject_source_risk: bool) -> tuple[list[Pair], collections.Counter[str]]:
    out: list[Pair] = []
    reasons: collections.Counter[str] = collections.Counter()
    seen_sources: set[str] = set()
    for r in rows:
        p = coerce_pair(r, regime)
        if p is None:
            reasons["not_accepted_or_malformed"] += 1
            continue
        if p.pair_words > MAX_PACKET_WORDS:
            reasons["pair_too_long_for_single_seq256_packet"] += 1
            continue
        if reject_source_risk and p.source_risks:
            reasons["source_risk"] += 1
            continue
        ratio = p.rewrite_words / max(1, p.source_words)
        if reject_exact_copy and norm_for_copy(p.source_text) == norm_for_copy(p.rewrite_text):
            reasons["exact_copy"] += 1
            continue
        key = f"{p.sentence_id}|{p.doc_id}|{norm_for_copy(p.source_text)[:80]}"
        if key in seen_sources:
            reasons["duplicate_source"] += 1
            continue
        if regime == "near":
            if ratio < near_min_ratio or ratio > near_max_ratio:
                reasons["outside_near_ratio_band"] += 1
                continue
        elif regime == "compact":
            if ratio < compact_min_ratio or ratio > compact_max_ratio:
                reasons["outside_compact_ratio_band"] += 1
                continue
            if p.content_recall is not None and p.content_recall < compact_min_recall:
                reasons["compact_low_content_recall"] += 1
                continue
        else:
            raise ValueError(regime)
        seen_sources.add(key)
        out.append(p)
    return out, reasons


def doc_diverse_order(pairs: list[Pair], seed: int) -> list[Pair]:
    rng = random.Random(seed)
    by_doc: dict[str, list[Pair]] = collections.defaultdict(list)
    for p in pairs:
        by_doc[p.doc_id or p.pair_id].append(p)
    docs = list(by_doc)
    rng.shuffle(docs)
    ordered: list[Pair] = []
    for d in docs:
        cand = by_doc[d]
        cand.sort(key=lambda p: (-(len(p.domain_hits) > 0), abs(p.source_words - 24), p.pair_id))
        ordered.append(cand[0])
    extras = [p for ps in by_doc.values() for p in ps[1:]]
    rng.shuffle(extras)
    ordered.extend(extras)
    return ordered


def select_to_budget(pairs: list[Pair], budget: int, seed: int) -> tuple[list[Pair], int]:
    selected: list[Pair] = []
    total = 0
    for p in doc_diverse_order(pairs, seed):
        if p.pair_words > MAX_PACKET_WORDS:
            continue
        if total + p.pair_words <= budget:
            selected.append(p)
            total += p.pair_words
    return selected, total


def repeat_source_words(source_text: str, n_words: int) -> str:
    toks = source_text.split()
    if not toks or n_words <= 0:
        return ""
    out: list[str] = []
    while len(out) < n_words:
        out.extend(toks[: n_words - len(out)])
    return " ".join(out)


def pack_pair_rows(pairs: list[Pair], use_rewrite: bool, source_label: str, example_base: int) -> list[Row]:
    rows: list[Row] = []
    cur_segments: list[str] = []
    cur_words = 0
    cur_pair_ids: list[str] = []
    cur_domains: collections.Counter[str] = collections.Counter()
    for p in pairs:
        second = p.rewrite_text if use_rewrite else repeat_source_words(p.source_text, p.rewrite_words)
        text = f"{p.source_text} {second}".strip()
        L = wc(text)
        if L != p.pair_words:
            raise RuntimeError(f"pair {p.pair_id} expected {p.pair_words} words but built {L}")
        if cur_words and cur_words + L > MAX_PACKET_WORDS:
            rows.append(Row(text=" ".join(cur_segments), words=cur_words, example_id=example_base + len(rows),
                            source=source_label, pair_ids=list(cur_pair_ids), component_sources=dict(cur_domains)))
            cur_segments, cur_words, cur_pair_ids, cur_domains = [], 0, [], collections.Counter()
        cur_segments.append(text)
        cur_words += L
        cur_pair_ids.append(p.pair_id)
        for d in p.domain_hits or ["no_domain"]:
            cur_domains[d] += L
    if cur_segments:
        rows.append(Row(text=" ".join(cur_segments), words=cur_words, example_id=example_base + len(rows),
                        source=source_label, pair_ids=list(cur_pair_ids), component_sources=dict(cur_domains)))
    for r in rows:
        if r.words != wc(r.text):
            raise RuntimeError("packed row word mismatch")
        if r.words > MAX_PACKET_WORDS:
            raise RuntimeError("packed row exceeds max packet words")
    return rows


def read_official_pool(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    total = 0
    for i, r in enumerate(rows):
        r["words"] = int(r.get("words") or wc(str(r.get("text") or "")))
        actual = wc(str(r.get("text") or ""))
        if r["words"] != actual:
            raise RuntimeError(f"official row word mismatch row={i}: field={r['words']} actual={actual}")
        total += r["words"]
    if total != TOTAL_WORDS:
        raise RuntimeError(f"official pool has {total} words, expected {TOTAL_WORDS}")
    return rows


def official_word_stream(official: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    words: list[str] = []
    sources: list[str] = []
    for r in official:
        ws = str(r["text"]).split()
        words.extend(ws)
        sources.extend([str(r.get("source", "official"))] * len(ws))
    return words, sources


def official_stream_rows_from_words(words: list[str], sources: list[str], lengths: list[int], offset_words: int,
                                    source_prefix: str, example_base: int) -> list[Row]:
    if not lengths:
        return []
    offset_words %= len(words)
    wstream = words[offset_words:] + words[:offset_words]
    sstream = sources[offset_words:] + sources[:offset_words]
    out: list[Row] = []
    pos = 0
    for i, L in enumerate(lengths):
        if L <= 0:
            continue
        seg = wstream[pos:pos + L]
        seg_src = sstream[pos:pos + L]
        if len(seg) != L:
            raise RuntimeError("official stream exhausted")
        common = collections.Counter(seg_src).most_common(1)[0][0]
        out.append(Row(text=" ".join(seg), words=L, example_id=example_base + i, source=f"{source_prefix}::{common}"))
        pos += L
    return out


def official_filler_rows(official: list[dict[str, Any]], needed_words: int, seed: int) -> list[Row]:
    if needed_words % OFFICIAL_ROW_WORDS != 0:
        raise RuntimeError(f"needed filler words not divisible by {OFFICIAL_ROW_WORDS}: {needed_words}")
    needed = needed_words // OFFICIAL_ROW_WORDS
    rows = list(official)
    random.Random(seed).shuffle(rows)
    chosen = rows[:needed]
    if len(chosen) != needed:
        raise RuntimeError(f"need {needed} filler rows, got {len(chosen)}")
    return [Row(text=str(r["text"]), words=int(r["words"]), example_id=int(r.get("example_id") or (800000 + i)), source=str(r.get("source", "official"))) for i, r in enumerate(chosen)]


def row_to_record(r: Row) -> dict[str, Any]:
    return {"text": r.text, "words": r.words, "example_id": r.example_id, "source": r.source}


def write_pool(path: pathlib.Path, rows: list[Row], meta_path: pathlib.Path | None = None) -> None:
    write_jsonl(path, [row_to_record(r) for r in rows])
    if meta_path is not None:
        with meta_path.open("w", encoding="utf-8") as f:
            for i, r in enumerate(rows):
                if r.pair_ids:
                    f.write(json.dumps({
                        "row_index": i,
                        "example_id": r.example_id,
                        "words": r.words,
                        "pair_ids": r.pair_ids,
                        "component_sources": r.component_sources or {},
                    }, ensure_ascii=False) + "\n")


def write_training(path: pathlib.Path, rows: list[Row], pass_orders: list[list[int]]) -> None:
    total = 0
    with path.open("w", encoding="utf-8") as f:
        for order in pass_orders:
            for idx in order:
                r = rows[idx]
                f.write(json.dumps(row_to_record(r), ensure_ascii=False) + "\n")
                total += r.words
    if total != TOTAL_WORDS * PASSES:
        raise RuntimeError(f"{path} total {total}")


def rows_equal(a: list[Row], b: list[Row]) -> bool:
    if len(a) != len(b):
        return False
    return all(x.text == y.text and x.words == y.words and x.example_id == y.example_id and x.source == y.source for x, y in zip(a, b))


def source_word_counts(rows: list[Row]) -> dict[str, int]:
    c: collections.Counter[str] = collections.Counter()
    for r in rows:
        c[r.source] += r.words
    return dict(c.most_common())


def pair_summary(pairs: list[Pair], selected_words: int, changed_budget: int, topup_words: int) -> dict[str, Any]:
    src = sum(p.source_words for p in pairs)
    rew = sum(p.rewrite_words for p in pairs)
    ratio_vals = [p.rewrite_words / max(1, p.source_words) for p in pairs]
    return {
        "selected_pairs": len(pairs),
        "selected_pair_words": selected_words,
        "neutral_official_topup_words_inside_changed_block": topup_words,
        "changed_block_budget_words": changed_budget,
        "source_words": src,
        "rewrite_words": rew,
        "rewrite_to_source_ratio_weighted": rew / max(1, src),
        "rewrite_to_source_ratio_mean": statistics.fmean(ratio_vals) if ratio_vals else None,
        "unique_source_sentences": len({(p.sentence_id, p.doc_id, norm_for_copy(p.source_text)[:80]) for p in pairs}),
        "unique_docs": len({p.doc_id for p in pairs}),
        "source_sentences_per_100k_changed_words": len(pairs) * 100000 / max(1, changed_budget),
        "docs_per_100k_changed_words": len({p.doc_id for p in pairs}) * 100000 / max(1, changed_budget),
        "domain_hit_counts": dict(collections.Counter(d for p in pairs for d in (p.domain_hits or ["no_domain"])).most_common()),
        "near_copy_like_soft_count": sum(1 for p in pairs if p.soft_flags and "near_copy_view" in p.soft_flags),
    }


def make_family(regime: str, selected: list[Pair], changed_budget: int, official: list[dict[str, Any]],
                official_words: list[str], official_sources: list[str], seed: int) -> tuple[dict[str, list[Row]], dict[str, Any]]:
    if regime == "near":
        repeat_name = "fineweb_repeat_near"
        view_name = "fineweb_near_view"
        official_name = "official_lengthmatched_near"
        base = 710000
    elif regime == "compact":
        repeat_name = "fineweb_repeat_compact_reinvest"
        view_name = "fineweb_compact_view_reinvest"
        official_name = "official_lengthmatched_compact"
        base = 730000
    else:
        raise ValueError(regime)

    view_rows = pack_pair_rows(selected, use_rewrite=True, source_label=view_name, example_base=base)
    repeat_rows = pack_pair_rows(selected, use_rewrite=False, source_label=repeat_name, example_base=base)
    selected_words = sum(p.pair_words for p in selected)
    topup_words = changed_budget - selected_words
    if topup_words < 0:
        raise RuntimeError(f"{regime} selected words exceed changed budget")
    topup_lengths: list[int] = []
    rem = topup_words
    while rem > 0:
        L = min(MAX_PACKET_WORDS, rem)
        topup_lengths.append(L)
        rem -= L
    topup = official_stream_rows_from_words(official_words, official_sources, topup_lengths,
                                            offset_words=(seed + (11 if regime == "near" else 17)) * 37,
                                            source_prefix=f"neutral_topup_{regime}",
                                            example_base=base + 100000)
    changed_lengths = [r.words for r in view_rows] + [r.words for r in topup]
    official_changed = official_stream_rows_from_words(official_words, official_sources, changed_lengths,
                                                       offset_words=(seed + (23 if regime == "near" else 29)) * 41,
                                                       source_prefix=official_name,
                                                       example_base=base + 200000)
    filler = official_filler_rows(official, TOTAL_WORDS - changed_budget, seed=seed + 1000)
    arms = {
        official_name: official_changed + filler,
        repeat_name: repeat_rows + topup + filler,
        view_name: view_rows + topup + filler,
    }
    lengths = {name: [r.words for r in rows] for name, rows in arms.items()}
    if len({tuple(v) for v in lengths.values()}) != 1:
        raise RuntimeError(f"{regime} family row-length sequence mismatch")
    if not rows_equal(arms[repeat_name][len(repeat_rows):], arms[view_name][len(view_rows):]):
        raise RuntimeError(f"{regime} repeat/view suffix mismatch")
    for name, rows in arms.items():
        total = sum(r.words for r in rows)
        if total != TOTAL_WORDS:
            raise RuntimeError(f"{name} total {total}")
    meta = {
        "regime": regime,
        "arm_names": {"official": official_name, "repeat": repeat_name, "view": view_name},
        "selected_pair_rows": len(view_rows),
        "changed_block_rows": len(view_rows) + len(topup),
        "selected_pair_words": selected_words,
        "neutral_official_topup_words_inside_changed_block": topup_words,
        "changed_block_budget_words": changed_budget,
        "filler_rows": len(filler),
        "filler_words": sum(r.words for r in filler),
        "row_length_sequence_identical_within_family": len({tuple(v) for v in lengths.values()}) == 1,
        "repeat_view_suffix_identical_after_pair_rows": rows_equal(arms[repeat_name][len(repeat_rows):], arms[view_name][len(view_rows):]),
        "word_totals": {name: sum(r.words for r in rows) for name, rows in arms.items()},
        "row_counts": {name: len(rows) for name, rows in arms.items()},
    }
    return arms, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--near-rows", default=str(NEAR_ROWS_DEFAULT))
    ap.add_argument("--compact-rows", default=str(COMPACT_ROWS_DEFAULT))
    ap.add_argument("--official-pool", default=str(OFFICIAL_POOL_DEFAULT))
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--note", default=str(NOTE_DEFAULT))
    ap.add_argument("--changed-budget", type=int, default=0, help="Same source+view/repetition budget for near and compact families; 0 uses min available rounded down.")
    ap.add_argument("--min-changed-budget", type=int, default=160000)
    ap.add_argument("--near-min-ratio", type=float, default=0.75)
    ap.add_argument("--near-max-ratio", type=float, default=1.20)
    ap.add_argument("--compact-min-ratio", type=float, default=0.35)
    ap.add_argument("--compact-max-ratio", type=float, default=0.85)
    ap.add_argument("--compact-min-recall", type=float, default=0.45)
    ap.add_argument("--allow-exact-copy", action="store_true")
    ap.add_argument("--reject-source-risk", action="store_true")
    ap.add_argument("--write-training", action="store_true")
    ap.add_argument("--seed", type=int, default=RNG_SEED)
    args = ap.parse_args()

    t0 = time.time()
    near_path = pathlib.Path(args.near_rows)
    compact_path = pathlib.Path(args.compact_rows)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    near_rows = read_jsonl(near_path)
    compact_rows = read_jsonl(compact_path)
    near_candidates, near_reject = filter_pairs(
        near_rows, "near", near_min_ratio=args.near_min_ratio, near_max_ratio=args.near_max_ratio,
        compact_min_ratio=args.compact_min_ratio, compact_max_ratio=args.compact_max_ratio,
        compact_min_recall=args.compact_min_recall, reject_exact_copy=not args.allow_exact_copy,
        reject_source_risk=args.reject_source_risk)
    compact_candidates, compact_reject = filter_pairs(
        compact_rows, "compact", near_min_ratio=args.near_min_ratio, near_max_ratio=args.near_max_ratio,
        compact_min_ratio=args.compact_min_ratio, compact_max_ratio=args.compact_max_ratio,
        compact_min_recall=args.compact_min_recall, reject_exact_copy=not args.allow_exact_copy,
        reject_source_risk=args.reject_source_risk)

    near_available = sum(p.pair_words for p in near_candidates)
    compact_available = sum(p.pair_words for p in compact_candidates)
    if args.changed_budget > 0:
        changed_budget = args.changed_budget
    else:
        changed_budget = min(near_available, compact_available)
        changed_budget = (changed_budget // OFFICIAL_ROW_WORDS) * OFFICIAL_ROW_WORDS
    if changed_budget % OFFICIAL_ROW_WORDS != 0:
        raise RuntimeError("changed budget must be divisible by 160 so the common 10M filler is exact")
    if changed_budget < args.min_changed_budget:
        raise RuntimeError(
            f"Changed budget {changed_budget} below min {args.min_changed_budget}. Generate more accepted near/compact views before training; "
            "lower min only for a labeled construction smoke test."
        )

    near_sel, near_words = select_to_budget(near_candidates, changed_budget, args.seed + 1)
    compact_sel, compact_words = select_to_budget(compact_candidates, changed_budget, args.seed + 2)
    # If either greedy selection leaves too much neutral top-up, the contrast is mostly not generated views.
    for name, w in [("near", near_words), ("compact", compact_words)]:
        if w < 0.90 * changed_budget:
            raise RuntimeError(f"{name} selected only {w} of {changed_budget}; generate more or reduce changed budget")

    official = read_official_pool(pathlib.Path(args.official_pool))
    official_words, official_sources = official_word_stream(official)
    near_arms, near_meta = make_family("near", near_sel, changed_budget, official, official_words, official_sources, args.seed + 10)
    compact_arms, compact_meta = make_family("compact", compact_sel, changed_budget, official, official_words, official_sources, args.seed + 20)
    arms = {**near_arms, **compact_arms}

    pool_paths: dict[str, str] = {}
    train_paths: dict[str, str] = {}
    row_meta_paths: dict[str, str] = {}
    for name, rows in arms.items():
        p = out_dir / f"{name}_10M.jsonl"
        meta_p = out_dir / f"{name}_changed_block_rows_meta.jsonl"
        write_pool(p, rows, meta_p if name.startswith("fineweb_") else None)
        pool_paths[name] = str(p)
        if name.startswith("fineweb_"):
            row_meta_paths[name] = str(meta_p)

    n_rows_by_arm = {name: len(rows) for name, rows in arms.items()}
    if args.write_training:
        for name, rows in arms.items():
            n_rows = len(rows)
            pass_orders = []
            for pass_i in range(PASSES):
                order = list(range(n_rows))
                random.Random(args.seed + 1000 + pass_i).shuffle(order)
                pass_orders.append(order)
            p = out_dir / f"{name}_100M.jsonl"
            write_training(p, rows, pass_orders)
            train_paths[name] = str(p)

    selected_near_path = out_dir / "selected_near_pairs.jsonl"
    selected_compact_path = out_dir / "selected_compact_pairs.jsonl"
    write_jsonl(selected_near_path, [dataclasses.asdict(p) for p in near_sel])
    write_jsonl(selected_compact_path, [dataclasses.asdict(p) for p in compact_sel])

    hashes = {pathlib.Path(p).name: sha256_file(pathlib.Path(p)) for p in pool_paths.values()}
    hashes["selected_near_pairs.jsonl"] = sha256_file(selected_near_path)
    hashes["selected_compact_pairs.jsonl"] = sha256_file(selected_compact_path)
    for p in train_paths.values():
        hashes[pathlib.Path(p).name] = sha256_file(pathlib.Path(p))

    near_summary = pair_summary(near_sel, near_words, changed_budget, changed_budget - near_words)
    compact_summary = pair_summary(compact_sel, compact_words, changed_budget, changed_budget - compact_words)
    payload = {
        "status": "DENSITY_CONTRAST_MATERIALIZED",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scientific_purpose": "Separate repetition, near-length generated views, and compact generated views; compact reinvests saved rewrite words in more source sentences at fixed changed-block budget.",
        "official_constraint_basis": "Official 2026 guideline permits Strict-Small submissions trained on 10M words or less with flexible dataset construction and teacher-model feedback, provided constraints are satisfied; this is still only a research contrast, not a submission artifact.",
        "inputs": {
            "near_rows": str(near_path),
            "near_rows_sha256": sha256_file(near_path),
            "compact_rows": str(compact_path),
            "compact_rows_sha256": sha256_file(compact_path),
            "official_pool": str(args.official_pool),
            "official_pool_sha256": sha256_file(pathlib.Path(args.official_pool)),
        },
        "candidate_counts": {
            "near_analyzed_rows": len(near_rows),
            "compact_analyzed_rows": len(compact_rows),
            "near_candidates": len(near_candidates),
            "compact_candidates": len(compact_candidates),
            "near_candidate_pair_words": near_available,
            "compact_candidate_pair_words": compact_available,
            "near_reject_reasons": dict(near_reject.most_common()),
            "compact_reject_reasons": dict(compact_reject.most_common()),
        },
        "changed_block_budget_words": changed_budget,
        "total_words_per_pool": TOTAL_WORDS,
        "passes": PASSES,
        "write_training": bool(args.write_training),
        "near_family": near_meta,
        "compact_family": compact_meta,
        "near_selected_summary": near_summary,
        "compact_selected_summary": compact_summary,
        "information_density_contrast": {
            "additional_compact_source_sentences_vs_near": compact_summary["selected_pairs"] - near_summary["selected_pairs"],
            "additional_compact_docs_vs_near": compact_summary["unique_docs"] - near_summary["unique_docs"],
            "compact_source_sentence_multiplier_vs_near": compact_summary["selected_pairs"] / max(1, near_summary["selected_pairs"]),
            "compact_doc_multiplier_vs_near": compact_summary["unique_docs"] / max(1, near_summary["unique_docs"]),
            "near_rewrite_to_source_ratio_weighted": near_summary["rewrite_to_source_ratio_weighted"],
            "compact_rewrite_to_source_ratio_weighted": compact_summary["rewrite_to_source_ratio_weighted"],
            "rewrite_words_saved_per_near_source_word_at_equal_budget": near_summary["rewrite_to_source_ratio_weighted"] - compact_summary["rewrite_to_source_ratio_weighted"],
        },
        "source_word_counts_by_arm": {name: source_word_counts(rows) for name, rows in arms.items()},
        "files": {
            "selected_near_pairs": str(selected_near_path),
            "selected_compact_pairs": str(selected_compact_path),
            "pools": pool_paths,
            "training": train_paths,
            "row_meta": row_meta_paths,
            "note": str(args.note),
        },
        "sha256": hashes,
        "interpretation": {
            "near_view_effect": "fineweb_near_view minus fineweb_repeat_near, within the near-length family and row-length-matched controls",
            "compact_view_effect": "fineweb_compact_view_reinvest minus fineweb_repeat_compact_reinvest, within the compact family and row-length-matched controls",
            "density_reinvestment_effect": "compare compact and near families at the same changed-block budget, using repeat arms to estimate source-count/source-breadth changes independent of generated views",
            "fineweb_vs_official_separated": "official_lengthmatched_near and official_lengthmatched_compact are separate controls; do not mix their deltas into the view-density interpretation",
        },
        "elapsed_sec": round(time.time() - t0, 2),
    }
    meta_path = out_dir / "density_contrast_metadata.json"
    meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    note_path = pathlib.Path(args.note)
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_lines = ["# research FineWeb density contrast materialization\n\n"]
    note_lines.append("Near-length and compact generated view regimes are kept separate. Compact is allowed to spend saved rewrite words on additional source sentences at the same changed-block word budget.\n\n")
    note_lines.append(f"Changed-block budget: {changed_budget:,} words. Near selected {near_summary['selected_pairs']:,} source sentences; compact selected {compact_summary['selected_pairs']:,}.\n\n")
    note_lines.append("Families:\n\n")
    for fam_name, fam_meta in [("near", near_meta), ("compact", compact_meta)]:
        note_lines.append(f"- {fam_name}: {fam_meta['arm_names']} with {fam_meta['changed_block_rows']:,} changed rows and {fam_meta['filler_rows']:,} common filler rows.\n")
    note_lines.append(f"\nMetadata: `{meta_path}`\n")
    note_path.write_text("".join(note_lines), encoding="utf-8")

    print(json.dumps({
        "status": payload["status"],
        "metadata": str(meta_path),
        "changed_block_budget_words": changed_budget,
        "near_pairs": near_summary["selected_pairs"],
        "compact_pairs": compact_summary["selected_pairs"],
        "compact_source_sentence_multiplier_vs_near": payload["information_density_contrast"]["compact_source_sentence_multiplier_vs_near"],
        "write_training": bool(args.write_training),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
