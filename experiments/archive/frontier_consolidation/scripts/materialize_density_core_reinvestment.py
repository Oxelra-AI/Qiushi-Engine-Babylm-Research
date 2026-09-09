#!/usr/bin/env python3
"""Materialize a source-aligned compact-view reinvestment contrast.

This is the repaired version of the research density contrast after scientific
review.  It distinguishes three questions without merging near and compact views:

1. Near-length generated view value on a shared source core:
      fineweb_near_view_core - fineweb_repeat_near_core
2. Compact generated view value on the same shared source core, without spending
   saved words on extra FineWeb sources:
      fineweb_compact_view_core_neutral - fineweb_repeat_compact_core_neutral
3. Compact reinvestment: use the words saved by compact rewrites to include
   additional compact source+view packets at the same changed-block budget:
      fineweb_compact_view_reinvest - fineweb_compact_view_core_neutral
   and its source-only counterpart via repeat arms.

Official length-matched controls are produced separately for the near-core,
compact-core-neutral, and compact-reinvest families.  They are controls for row
length / official-vs-FineWeb effects, not evidence that cross-family deltas are
pure density without the repeat/view decomposition.

All corpus-word counts are whitespace counts.  No BabyLM evaluation data or labels
are read.
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
NEAR_ROWS_DEFAULT = WORKSPACE / "data/high_precision_near_analysis/high_precision_near_ws_rows.jsonl"
COMPACT_ROWS_DEFAULT = WORKSPACE / "data/high_precision_compact_analysis/high_precision_compact_ws_rows.jsonl"
OUT_DEFAULT = WORKSPACE / "data/density_core_reinvestment"
NOTE_DEFAULT = (WORKSPACE.parents[2] / 'research/notes/frontier_consolidation/13_density_core_reinvestment_materialization.md')

TOTAL_WORDS = 10_000_000
PASSES = 10
OFFICIAL_ROW_WORDS = 160
MAX_PACKET_WORDS = 160
RNG_SEED = 82913113
WORD_RE = re.compile(r"[A-Za-z0-9]+")


@dataclasses.dataclass
class Pair:
    pair_id: str
    key: str
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
    entity_recall: float | None = None
    number_recall: float | None = None
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


def source_key_from_fields(sentence_id: Any, doc_id: Any, source_text: str) -> str:
    if sentence_id is not None and str(sentence_id) != "":
        return f"sid:{sentence_id}|doc:{doc_id}"
    return "txt:" + hashlib.sha1(norm_for_copy(source_text).encode("utf-8")).hexdigest()[:20]


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: pathlib.Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def coerce_pair(row: dict[str, Any], regime: str) -> Pair | None:
    if not row.get("accepted_for_next_construction", False):
        return None
    src = norm_text(str(row.get("source_text") or ""))
    rew = norm_text(str(row.get("rewrite_text") or row.get("output") or ""))
    if not src or not rew:
        return None
    sw = wc(src)
    rw = wc(rew)
    if sw <= 0 or rw <= 0:
        return None
    key = source_key_from_fields(row.get("sentence_id"), row.get("doc_id"), src)
    return Pair(
        pair_id=f"{regime}:{row.get('prompt_id') or key}",
        key=key,
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
        entity_recall=float(row.get("entity_recall")) if row.get("entity_recall") is not None else None,
        number_recall=float(row.get("number_recall")) if row.get("number_recall") is not None else None,
        soft_flags=[str(x) for x in (row.get("soft_flags") or [])],
        source_risks=[str(x) for x in (row.get("source_risks") or [])],
    )


def filter_pair(row: dict[str, Any], regime: str, args: argparse.Namespace) -> tuple[Pair | None, str | None]:
    p = coerce_pair(row, regime)
    if p is None:
        return None, "not_accepted_or_malformed"
    if p.pair_words > MAX_PACKET_WORDS:
        return None, "pair_too_long_for_seq256_packet"
    if args.reject_source_risk and p.source_risks:
        return None, "source_risk"
    ratio = p.rewrite_words / max(1, p.source_words)
    if not args.allow_exact_copy and norm_for_copy(p.source_text) == norm_for_copy(p.rewrite_text):
        return None, "exact_copy"
    if regime == "near":
        if ratio < args.near_min_ratio or ratio > args.near_max_ratio:
            return None, "outside_near_ratio_band"
        if p.content_recall is not None and p.content_recall < args.near_min_recall:
            return None, "near_low_content_recall"
    elif regime == "compact":
        if ratio < args.compact_min_ratio or ratio > args.compact_max_ratio:
            return None, "outside_compact_ratio_band"
        if p.content_recall is not None and p.content_recall < args.compact_min_recall:
            return None, "compact_low_content_recall"
    else:
        raise ValueError(regime)
    return p, None


def candidate_map(rows: list[dict[str, Any]], regime: str, args: argparse.Namespace) -> tuple[dict[str, Pair], collections.Counter[str]]:
    out: dict[str, Pair] = {}
    reject: collections.Counter[str] = collections.Counter()
    for r in rows:
        p, reason = filter_pair(r, regime, args)
        if p is None:
            reject[reason or "unknown"] += 1
            continue
        # If duplicate source appears, keep the shorter, cleaner pair.
        old = out.get(p.key)
        if old is None:
            out[p.key] = p
        else:
            old_quality = ((old.content_recall or 0.0), -(old.pair_words), old.pair_id)
            new_quality = ((p.content_recall or 0.0), -(p.pair_words), p.pair_id)
            if new_quality > old_quality:
                out[p.key] = p
                reject["duplicate_source_replaced"] += 1
            else:
                reject["duplicate_source_dropped"] += 1
    return out, reject


def doc_diverse_keys(pairs_by_key: dict[str, Pair], keys: Iterable[str], seed: int) -> list[str]:
    rng = random.Random(seed)
    by_doc: dict[str, list[str]] = collections.defaultdict(list)
    for k in keys:
        p = pairs_by_key[k]
        by_doc[p.doc_id or k].append(k)
    docs = list(by_doc)
    rng.shuffle(docs)
    ordered: list[str] = []
    for d in docs:
        ks = by_doc[d]
        ks.sort(key=lambda k: (-(len(pairs_by_key[k].domain_hits) > 0), abs(pairs_by_key[k].source_words - 24), k))
        ordered.append(ks[0])
    extras = [k for ks in by_doc.values() for k in ks[1:]]
    rng.shuffle(extras)
    ordered.extend(extras)
    return ordered


def select_shared_core(near: dict[str, Pair], compact: dict[str, Pair], changed_budget: int, seed: int,
                       min_core_fraction: float) -> tuple[list[str], int, int]:
    keys = set(near) & set(compact)
    ordered = doc_diverse_keys(near, keys, seed)
    selected: list[str] = []
    near_words = 0
    compact_words = 0
    for k in ordered:
        n = near[k]
        c = compact[k]
        if near_words + n.pair_words <= changed_budget:
            selected.append(k)
            near_words += n.pair_words
            compact_words += c.pair_words
    if near_words < min_core_fraction * changed_budget:
        raise RuntimeError(
            f"Shared near core covers only {near_words}/{changed_budget} words ({near_words/max(1,changed_budget):.3f}); "
            "generate more shared accepted near/compact rows or lower budget only for a labelled smoke run."
        )
    return selected, near_words, compact_words


def select_added_compact(compact: dict[str, Pair], exclude: set[str], budget: int, seed: int) -> tuple[list[str], int]:
    if budget <= 0:
        return [], 0
    ordered = doc_diverse_keys(compact, [k for k in compact if k not in exclude], seed)
    selected: list[str] = []
    words = 0
    for k in ordered:
        p = compact[k]
        if words + p.pair_words <= budget:
            selected.append(k)
            words += p.pair_words
    return selected, words


def repeat_source_words(source_text: str, n_words: int, salt: str) -> str:
    toks = source_text.split()
    if not toks or n_words <= 0:
        return ""
    # Deterministic cyclic span avoids always repeating only the sentence prefix.
    h = int(hashlib.sha1(salt.encode("utf-8")).hexdigest()[:8], 16)
    start = h % len(toks)
    rot = toks[start:] + toks[:start]
    out: list[str] = []
    while len(out) < n_words:
        out.extend(rot[: n_words - len(out)])
    return " ".join(out)


def pack_pair_rows(pairs: list[Pair], use_rewrite: bool, source_label: str, example_base: int) -> list[Row]:
    rows: list[Row] = []
    cur_segments: list[str] = []
    cur_words = 0
    cur_pair_ids: list[str] = []
    cur_domains: collections.Counter[str] = collections.Counter()
    for p in pairs:
        second = p.rewrite_text if use_rewrite else repeat_source_words(p.source_text, p.rewrite_words, p.pair_id)
        text = f"{p.source_text} {second}".strip()
        L = wc(text)
        if L != p.pair_words:
            raise RuntimeError(f"pair {p.pair_id} expected {p.pair_words}, built {L}")
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
            raise RuntimeError("packed row exceeds max")
    return rows


def read_official_pool(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    total = 0
    for i, r in enumerate(rows):
        r["words"] = int(r.get("words") or wc(str(r.get("text") or "")))
        actual = wc(str(r.get("text") or ""))
        if r["words"] != actual:
            raise RuntimeError(f"official row word mismatch {i}")
        total += r["words"]
    if total != TOTAL_WORDS:
        raise RuntimeError(f"official pool has {total}, expected {TOTAL_WORDS}")
    return rows


def official_word_stream(official: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    words: list[str] = []
    sources: list[str] = []
    for r in official:
        ws = str(r["text"]).split()
        words.extend(ws)
        sources.extend([str(r.get("source", "official"))] * len(ws))
    return words, sources


def official_stream_rows(words: list[str], sources: list[str], lengths: list[int], offset_words: int,
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
        seg = wstream[pos:pos+L]
        seg_src = sstream[pos:pos+L]
        if len(seg) != L:
            raise RuntimeError("official stream exhausted")
        common = collections.Counter(seg_src).most_common(1)[0][0]
        out.append(Row(text=" ".join(seg), words=L, example_id=example_base+i, source=f"{source_prefix}::{common}"))
        pos += L
    return out


def official_filler_rows(official: list[dict[str, Any]], needed_words: int, seed: int) -> list[Row]:
    if needed_words % OFFICIAL_ROW_WORDS != 0:
        raise RuntimeError(f"needed filler {needed_words} not divisible by {OFFICIAL_ROW_WORDS}")
    needed = needed_words // OFFICIAL_ROW_WORDS
    rows = list(official)
    random.Random(seed).shuffle(rows)
    chosen = rows[:needed]
    if len(chosen) != needed:
        raise RuntimeError(f"need {needed} filler rows, got {len(chosen)}")
    return [Row(text=str(r["text"]), words=int(r["words"]), example_id=int(r.get("example_id") or (880000+i)), source=str(r.get("source", "official"))) for i, r in enumerate(chosen)]


def make_topup(words: list[str], sources: list[str], needed_words: int, seed: int, label: str, example_base: int) -> list[Row]:
    lengths: list[int] = []
    rem = needed_words
    while rem > 0:
        L = min(MAX_PACKET_WORDS, rem)
        lengths.append(L)
        rem -= L
    return official_stream_rows(words, sources, lengths, offset_words=seed * 53, source_prefix=label, example_base=example_base)


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


def write_training(path: pathlib.Path, rows: list[Row], seed: int) -> None:
    total = 0
    n_rows = len(rows)
    with path.open("w", encoding="utf-8") as f:
        for pass_i in range(PASSES):
            order = list(range(n_rows))
            random.Random(seed + 1000 + pass_i).shuffle(order)
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


def pair_quality_summary(pairs: list[Pair], budget: int, neutral_topup: int = 0) -> dict[str, Any]:
    src = sum(p.source_words for p in pairs)
    rew = sum(p.rewrite_words for p in pairs)
    ratios = [p.rewrite_words / max(1, p.source_words) for p in pairs]
    def stat(vals: list[float]) -> dict[str, Any]:
        vals = [float(v) for v in vals if v is not None]
        if not vals:
            return {"n": 0}
        vals.sort()
        return {"n": len(vals), "mean": statistics.fmean(vals), "median": statistics.median(vals), "min": vals[0], "max": vals[-1]}
    return {
        "pairs": len(pairs),
        "source_words": src,
        "rewrite_words": rew,
        "pair_words": src + rew,
        "neutral_official_topup_words_inside_changed_block": neutral_topup,
        "changed_block_budget_words": budget,
        "unique_docs": len({p.doc_id for p in pairs}),
        "rewrite_to_source_ratio_weighted": rew / max(1, src),
        "rewrite_to_source_ratio_stats": stat(ratios),
        "content_recall_stats": stat([p.content_recall for p in pairs if p.content_recall is not None]),
        "entity_recall_stats": stat([p.entity_recall for p in pairs if p.entity_recall is not None]),
        "number_recall_stats": stat([p.number_recall for p in pairs if p.number_recall is not None]),
        "domain_hit_counts": dict(collections.Counter(d for p in pairs for d in (p.domain_hits or ["no_domain"])).most_common()),
        "near_copy_like_soft_count": sum(1 for p in pairs if p.soft_flags and "near_copy_view" in p.soft_flags),
    }


def build_family(family_name: str, official_name: str, repeat_name: str, view_name: str,
                 pair_list: list[Pair], changed_budget: int, neutral_topup_words: int,
                 official_words: list[str], official_sources: list[str], filler: list[Row], seed: int,
                 example_base: int) -> tuple[dict[str, list[Row]], dict[str, Any]]:
    view_pair_rows = pack_pair_rows(pair_list, use_rewrite=True, source_label=view_name, example_base=example_base)
    repeat_pair_rows = pack_pair_rows(pair_list, use_rewrite=False, source_label=repeat_name, example_base=example_base)
    topup = make_topup(official_words, official_sources, neutral_topup_words, seed, f"neutral_topup_{family_name}", example_base + 100000)
    changed_lengths = [r.words for r in view_pair_rows] + [r.words for r in topup]
    official_changed = official_stream_rows(official_words, official_sources, changed_lengths, offset_words=seed * 67,
                                            source_prefix=official_name, example_base=example_base + 200000)
    arms = {
        official_name: official_changed + filler,
        repeat_name: repeat_pair_rows + topup + filler,
        view_name: view_pair_rows + topup + filler,
    }
    if len({tuple(r.words for r in rows) for rows in arms.values()}) != 1:
        raise RuntimeError(f"{family_name} row length mismatch")
    if not rows_equal(arms[repeat_name][len(repeat_pair_rows):], arms[view_name][len(view_pair_rows):]):
        raise RuntimeError(f"{family_name} repeat/view suffix mismatch")
    for name, rows in arms.items():
        total = sum(r.words for r in rows)
        if total != TOTAL_WORDS:
            raise RuntimeError(f"{name} total {total}")
    meta = {
        "family_name": family_name,
        "arm_names": {"official": official_name, "repeat": repeat_name, "view": view_name},
        "pair_rows": len(view_pair_rows),
        "changed_block_rows": len(view_pair_rows) + len(topup),
        "pair_words": sum(p.pair_words for p in pair_list),
        "neutral_official_topup_words_inside_changed_block": neutral_topup_words,
        "changed_block_budget_words": changed_budget,
        "row_length_sequence_identical_within_family": len({tuple(r.words for r in rows) for rows in arms.values()}) == 1,
        "repeat_view_suffix_identical_after_pair_rows": rows_equal(arms[repeat_name][len(repeat_pair_rows):], arms[view_name][len(view_pair_rows):]),
        "filler_rows": len(filler),
        "filler_words": sum(r.words for r in filler),
        "word_totals": {name: sum(r.words for r in rows) for name, rows in arms.items()},
        "row_counts": {name: len(rows) for name, rows in arms.items()},
    }
    return arms, meta


def choose_changed_budget(near_map: dict[str, Pair], compact_map: dict[str, Pair], args: argparse.Namespace) -> int:
    if args.changed_budget > 0:
        return args.changed_budget
    shared_keys = set(near_map) & set(compact_map)
    near_shared = sum(near_map[k].pair_words for k in shared_keys)
    compact_all = sum(p.pair_words for p in compact_map.values())
    # Leave a little headroom for greedy doc-diverse selection and exact 160 divisibility.
    raw = int(min(near_shared, compact_all) * args.auto_budget_fraction)
    return (raw // OFFICIAL_ROW_WORDS) * OFFICIAL_ROW_WORDS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--near-rows", default=str(NEAR_ROWS_DEFAULT))
    ap.add_argument("--compact-rows", default=str(COMPACT_ROWS_DEFAULT))
    ap.add_argument("--official-pool", default=str(OFFICIAL_POOL_DEFAULT))
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--note", default=str(NOTE_DEFAULT))
    ap.add_argument("--changed-budget", type=int, default=0)
    ap.add_argument("--auto-budget-fraction", type=float, default=0.92)
    ap.add_argument("--min-changed-budget", type=int, default=160000)
    ap.add_argument("--min-core-fraction", type=float, default=0.90)
    ap.add_argument("--near-min-ratio", type=float, default=0.75)
    ap.add_argument("--near-max-ratio", type=float, default=1.20)
    ap.add_argument("--near-min-recall", type=float, default=0.45)
    ap.add_argument("--compact-min-ratio", type=float, default=0.35)
    ap.add_argument("--compact-max-ratio", type=float, default=0.85)
    ap.add_argument("--compact-min-recall", type=float, default=0.45)
    ap.add_argument("--allow-exact-copy", action="store_true")
    ap.add_argument("--reject-source-risk", action="store_true")
    ap.add_argument("--write-training", action="store_true")
    ap.add_argument("--seed", type=int, default=RNG_SEED)
    args = ap.parse_args()

    t0 = time.time()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    near_path = pathlib.Path(args.near_rows)
    compact_path = pathlib.Path(args.compact_rows)
    near_rows = read_jsonl(near_path)
    compact_rows = read_jsonl(compact_path)
    near_map, near_reject = candidate_map(near_rows, "near", args)
    compact_map, compact_reject = candidate_map(compact_rows, "compact", args)
    changed_budget = choose_changed_budget(near_map, compact_map, args)
    if changed_budget % OFFICIAL_ROW_WORDS != 0:
        raise RuntimeError("changed budget must be divisible by 160")
    if changed_budget < args.min_changed_budget:
        raise RuntimeError(
            f"Changed budget {changed_budget} below min {args.min_changed_budget}; generate more accepted views before training."
        )

    core_keys, near_core_words, compact_core_words = select_shared_core(
        near_map, compact_map, changed_budget, args.seed + 1, args.min_core_fraction)
    core_set = set(core_keys)
    saved_words = changed_budget - compact_core_words
    added_keys, added_words = select_added_compact(compact_map, core_set, saved_words, args.seed + 2)
    if added_words <= 0 and saved_words > 1000:
        raise RuntimeError("Compact reinvestment could not select added sources despite saved budget")
    near_core_pairs = [near_map[k] for k in core_keys]
    compact_core_pairs = [compact_map[k] for k in core_keys]
    compact_added_pairs = [compact_map[k] for k in added_keys]
    compact_reinvest_pairs = compact_core_pairs + compact_added_pairs
    compact_reinvest_words = sum(p.pair_words for p in compact_reinvest_pairs)
    if compact_reinvest_words > changed_budget:
        raise RuntimeError("compact reinvest exceeds budget")
    near_topup = changed_budget - near_core_words
    compact_core_topup = changed_budget - compact_core_words
    compact_reinvest_topup = changed_budget - compact_reinvest_words

    official = read_official_pool(pathlib.Path(args.official_pool))
    official_words, official_sources = official_word_stream(official)
    filler = official_filler_rows(official, TOTAL_WORDS - changed_budget, seed=args.seed + 5000)

    family_specs = [
        ("near_core", "official_lengthmatched_near_core", "fineweb_repeat_near_core", "fineweb_near_view_core", near_core_pairs, near_topup, 710000),
        ("compact_core_neutral", "official_lengthmatched_compact_core_neutral", "fineweb_repeat_compact_core_neutral", "fineweb_compact_view_core_neutral", compact_core_pairs, compact_core_topup, 730000),
        ("compact_reinvest", "official_lengthmatched_compact_reinvest", "fineweb_repeat_compact_reinvest", "fineweb_compact_view_reinvest", compact_reinvest_pairs, compact_reinvest_topup, 750000),
    ]
    arms: dict[str, list[Row]] = {}
    family_meta: dict[str, Any] = {}
    for i, spec in enumerate(family_specs):
        fam, off, rep, view, pairs, topup, base = spec
        fam_arms, fam_meta = build_family(fam, off, rep, view, pairs, changed_budget, topup,
                                          official_words, official_sources, filler, args.seed + 100 + i, base)
        arms.update(fam_arms)
        family_meta[fam] = fam_meta

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
        if args.write_training:
            tp = out_dir / f"{name}_100M.jsonl"
            write_training(tp, rows, args.seed + 7000)
            train_paths[name] = str(tp)

    selected_paths = {
        "near_core_pairs": str(out_dir / "selected_near_core_pairs.jsonl"),
        "compact_core_pairs": str(out_dir / "selected_compact_core_pairs.jsonl"),
        "compact_added_pairs": str(out_dir / "selected_compact_added_pairs.jsonl"),
        "compact_reinvest_pairs": str(out_dir / "selected_compact_reinvest_pairs.jsonl"),
        "core_source_keys": str(out_dir / "core_source_keys.json"),
        "added_source_keys": str(out_dir / "added_source_keys.json"),
    }
    write_jsonl(pathlib.Path(selected_paths["near_core_pairs"]), [dataclasses.asdict(p) for p in near_core_pairs])
    write_jsonl(pathlib.Path(selected_paths["compact_core_pairs"]), [dataclasses.asdict(p) for p in compact_core_pairs])
    write_jsonl(pathlib.Path(selected_paths["compact_added_pairs"]), [dataclasses.asdict(p) for p in compact_added_pairs])
    write_jsonl(pathlib.Path(selected_paths["compact_reinvest_pairs"]), [dataclasses.asdict(p) for p in compact_reinvest_pairs])
    pathlib.Path(selected_paths["core_source_keys"]).write_text(json.dumps(core_keys, indent=2) + "\n", encoding="utf-8")
    pathlib.Path(selected_paths["added_source_keys"]).write_text(json.dumps(added_keys, indent=2) + "\n", encoding="utf-8")

    hashes = {pathlib.Path(p).name: sha256_file(pathlib.Path(p)) for p in pool_paths.values()}
    for p in selected_paths.values():
        hashes[pathlib.Path(p).name] = sha256_file(pathlib.Path(p))
    for p in train_paths.values():
        hashes[pathlib.Path(p).name] = sha256_file(pathlib.Path(p))

    near_summary = pair_quality_summary(near_core_pairs, changed_budget, near_topup)
    compact_core_summary = pair_quality_summary(compact_core_pairs, changed_budget, compact_core_topup)
    compact_added_summary = pair_quality_summary(compact_added_pairs, saved_words, saved_words - added_words)
    compact_reinvest_summary = pair_quality_summary(compact_reinvest_pairs, changed_budget, compact_reinvest_topup)

    payload = {
        "status": "DENSITY_CORE_REINVESTMENT_MATERIALIZED",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scientific_purpose": "Source-aligned test of near-length views, compact views without source reinvestment, and compact views with saved words reinvested in additional sources.",
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
            "near_candidates": len(near_map),
            "compact_candidates": len(compact_map),
            "shared_candidate_sources": len(set(near_map) & set(compact_map)),
            "near_candidate_pair_words": sum(p.pair_words for p in near_map.values()),
            "compact_candidate_pair_words": sum(p.pair_words for p in compact_map.values()),
            "near_reject_reasons": dict(near_reject.most_common()),
            "compact_reject_reasons": dict(compact_reject.most_common()),
        },
        "changed_block_budget_words": changed_budget,
        "total_words_per_pool": TOTAL_WORDS,
        "passes": PASSES,
        "write_training": bool(args.write_training),
        "selection": {
            "core_sources": len(core_keys),
            "compact_added_sources": len(added_keys),
            "near_core_pair_words": near_core_words,
            "compact_core_pair_words": compact_core_words,
            "saved_words_from_compact_core_vs_near_core_budget": compact_core_topup,
            "compact_added_pair_words": added_words,
            "compact_reinvest_pair_words": compact_reinvest_words,
            "near_neutral_topup_words": near_topup,
            "compact_core_neutral_topup_words": compact_core_topup,
            "compact_reinvest_neutral_topup_words": compact_reinvest_topup,
            "compact_reinvest_sources_total": len(core_keys) + len(added_keys),
            "source_multiplier_compact_reinvest_vs_near_core": (len(core_keys) + len(added_keys)) / max(1, len(core_keys)),
            "added_source_fraction_of_reinvest": len(added_keys) / max(1, len(core_keys) + len(added_keys)),
        },
        "near_core_summary": near_summary,
        "compact_core_summary": compact_core_summary,
        "compact_added_summary": compact_added_summary,
        "compact_reinvest_summary": compact_reinvest_summary,
        "families": family_meta,
        "audit": {
            "same_core_source_keys_for_near_and_compact_core": [p.key for p in near_core_pairs] == [p.key for p in compact_core_pairs],
            "common_filler_words_all_families": sum(r.words for r in filler),
            "common_filler_rows_all_families": len(filler),
            "all_pool_word_totals": {name: sum(r.words for r in rows) for name, rows in arms.items()},
            "all_pool_row_counts": {name: len(rows) for name, rows in arms.items()},
        },
        "source_word_counts_by_arm": {name: source_word_counts(rows) for name, rows in arms.items()},
        "files": {
            **selected_paths,
            "pools": pool_paths,
            "training": train_paths,
            "row_meta": row_meta_paths,
            "note": str(args.note),
        },
        "sha256": hashes,
        "interpretation": {
            "near_generated_view_value": "fineweb_near_view_core minus fineweb_repeat_near_core on the shared source core",
            "compact_generated_view_value_without_reinvestment": "fineweb_compact_view_core_neutral minus fineweb_repeat_compact_core_neutral on the same source core plus neutral official topup",
            "compact_reinvestment_value_with_views": "fineweb_compact_view_reinvest minus fineweb_compact_view_core_neutral, with both using compact views but only reinvest adds extra FineWeb sources",
            "compact_reinvestment_source_only_counterpart": "fineweb_repeat_compact_reinvest minus fineweb_repeat_compact_core_neutral estimates the added-source effect under repetition rather than generated compact views",
            "official_controls": "Each family has its own official lengthmatched arm; these control row-length / official-vs-FineWeb effects but should not be collapsed into a single cross-family scalar.",
        },
        "elapsed_sec": round(time.time() - t0, 2),
    }
    meta_path = out_dir / "density_core_reinvestment_metadata.json"
    meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    note_path = pathlib.Path(args.note)
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(
        "# research density core/reinvestment materialization\n\n"
        "This repaired construction uses a shared source core for near and compact views, then isolates compact saved-word reinvestment by adding compact-only FineWeb sources in a separate family.\n\n"
        f"Changed-block budget: {changed_budget:,} words. Shared core sources: {len(core_keys):,}. Compact added sources: {len(added_keys):,}. "
        f"Compact reinvest source multiplier: {payload['selection']['source_multiplier_compact_reinvest_vs_near_core']:.3f}.\n\n"
        f"Metadata: `{meta_path}`\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "status": payload["status"],
        "metadata": str(meta_path),
        "changed_block_budget_words": changed_budget,
        "core_sources": len(core_keys),
        "compact_added_sources": len(added_keys),
        "source_multiplier_compact_reinvest_vs_near_core": payload["selection"]["source_multiplier_compact_reinvest_vs_near_core"],
        "write_training": bool(args.write_training),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
