#!/usr/bin/env python3
"""research: build/audit candidate factorial controls for compact-view mechanisms.

Compact views are analyzed as a three-factor object:
  1. source-position spread,
  2. recurrent lexical content,
  3. fluent compressed syntax.

The research prefix and telegraphic skeleton arms cannot separate these factors.
This CPU-only script prototypes a more matched factorial family at pair and
changed-block level, without training and without constructing final pools.

Candidate controls:
  - compact: original generated compact view (reference).
  - compact_scrambled: same compact word multiset, deterministic shuffle; fixes
    recurrent words/exposure within compact while damaging coherence/order.
  - prefix_fluent: first k source words; contiguous, prefix-local, source words.
  - prefix_scrambled: same prefix word multiset, deterministic shuffle.
  - sourcewide_onegap: source-only, exact-k view made by deleting one contiguous
    low-value block from the source, so the kept words form at most two intact
    source spans.  This is a sentence-preserving-ish source-wide control, much
    less telegraphic than selecting isolated content words.
  - sourcewide_onegap_scrambled: same onegap word multiset, deterministic shuffle.
  - best_contiguous_span: a contiguous source span of length k chosen for compact
    lexical overlap/tail content; a fluency/control continuum point, not a main
    factorial cell.

The script computes text geometry and exact DeBERTa MLM-loader exposure on the
research changed-row packing. It does not train/evaluate models and does not read
pending experimental outputs.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import pathlib
import random
import re
import statistics
import time
from dataclasses import dataclass
from typing import Any, Iterable

from transformers import AutoTokenizer

ROOT = pathlib.Path("experiments/archive/frontier_consolidation")
DEFAULT_PAIRS = ROOT / "data/density_core_reinvestment_medium_riskhard/selected_compact_reinvest_pairs.jsonl"
DEFAULT_META = ROOT / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_changed_block_rows_meta.jsonl"
DEFAULT_BASE_POOL = ROOT / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"
DEFAULT_TOKENIZER = ROOT / "data/compliant_tokenizer"
DEFAULT_OUT = ROOT / "data/factorial_view_candidate_audit"
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['\u2019-][A-Za-z0-9]+)?")
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "so", "because", "as", "than",
    "to", "of", "in", "on", "for", "with", "without", "by", "from", "at", "into", "onto", "over",
    "under", "about", "between", "among", "through", "during", "before", "after", "above", "below",
    "is", "am", "are", "was", "were", "be", "been", "being", "do", "does", "did", "done", "doing",
    "have", "has", "had", "having", "can", "could", "may", "might", "must", "shall", "should", "will",
    "would", "this", "that", "these", "those", "there", "here", "it", "its", "they", "them", "their",
    "theirs", "he", "him", "his", "she", "her", "hers", "we", "us", "our", "ours", "you", "your",
    "yours", "i", "me", "my", "mine", "who", "whom", "whose", "which", "what", "where", "when",
    "why", "how", "not", "no", "nor", "only", "just", "also", "very", "more", "most", "less", "least",
    "much", "many", "some", "any", "all", "each", "every", "other", "another", "such", "own", "same",
    "too", "again", "still", "already", "yet", "up", "down", "out", "off", "back", "away",
}
CONNECTORS = {"because", "after", "before", "during", "while", "when", "although", "though", "since", "if", "then", "but", "and", "or", "which", "that", "who", "whose", "where", "therefore", "however"}
VARIANT_ORDER = [
    "compact",
    "compact_scrambled",
    "prefix_fluent",
    "prefix_scrambled",
    "sourcewide_onegap",
    "sourcewide_onegap_scrambled",
    "best_contiguous_span",
]


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def words(text: str) -> list[str]:
    return [w for w in text.split() if w]


def wc(text: str) -> int:
    return len(words(text))


def norm_word(word: str) -> str:
    parts = WORD_RE.findall(word)
    if not parts:
        return ""
    return "".join(parts).lower()


def is_content(n: str) -> bool:
    return bool(n) and (n.isdigit() or (len(n) >= 4 and n not in STOPWORDS))


def is_function(n: str) -> bool:
    return bool(n) and n in STOPWORDS


def punctish(w: str) -> bool:
    return any(ch in w for ch in ".,;:!?()[]{}\"'")


def counter_norm(ws: list[str]) -> collections.Counter[str]:
    c = collections.Counter()
    for w in ws:
        n = norm_word(w)
        if n:
            c[n] += 1
    return c


def multiset_intersection_size(a: collections.Counter[str], b: collections.Counter[str]) -> int:
    return sum(min(a[k], b[k]) for k in set(a) | set(b))


def jaccard(a: collections.Counter[str], b: collections.Counter[str]) -> float | None:
    keys = set(a) | set(b)
    if not keys:
        return None
    inter = sum(min(a[k], b[k]) for k in keys)
    union = sum(max(a[k], b[k]) for k in keys)
    return inter / union if union else None


def safe_mean(xs: Iterable[float | int | None]) -> float | None:
    vals = [float(x) for x in xs if isinstance(x, (int, float)) and math.isfinite(float(x))]
    return sum(vals) / len(vals) if vals else None


def safe_median(xs: Iterable[float | int | None]) -> float | None:
    vals = [float(x) for x in xs if isinstance(x, (int, float)) and math.isfinite(float(x))]
    return statistics.median(vals) if vals else None


def frac(num: float, den: float) -> float | None:
    return float(num) / float(den) if den else None


def shuffle_words(ws: list[str], salt: str) -> list[str]:
    out = list(ws)
    if len(out) <= 1:
        return out
    seed = int(hashlib.sha256(salt.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    rng.shuffle(out)
    # Avoid accidentally preserving original order for short rows.
    if out == ws and len(out) > 2:
        out = out[1:] + out[:1]
    return out


def best_contiguous_span(src: list[str], comp: list[str]) -> list[int]:
    k = len(comp)
    n = len(src)
    if k >= n:
        return list(range(n))
    comp_norms = counter_norm(comp)
    source_norms = [norm_word(w) for w in src]
    best: tuple[float, int] | None = None
    for start in range(0, n - k + 1):
        idxs = list(range(start, start + k))
        kept = [source_norms[i] for i in idxs]
        kept_c = collections.Counter(nm for nm in kept if nm)
        inter = multiset_intersection_size(kept_c, comp_norms)
        content_hits = sum(1 for i in idxs if is_content(source_norms[i]) and comp_norms.get(source_norms[i], 0) > 0)
        tail = sum(1 for i in idxs if i >= k and is_content(source_norms[i]))
        mean_pos = safe_mean([i / max(1, n - 1) for i in idxs]) or 0.0
        score = 4.0 * content_hits + 1.5 * inter + 1.2 * tail + 0.5 * mean_pos
        cand = (score, -start)
        if best is None or cand > best:
            best = cand
    start = -best[1] if best is not None else 0
    return list(range(start, start + k))


def sourcewide_onegap(src: list[str], comp: list[str]) -> list[int]:
    """Select exact-k source words by deleting one contiguous block.

    This is designed as a more sentence-preserving source-wide control than the
    research telegraphic skeletons: output is prefix-span + suffix-span, with at
    most one seam.  It tries to keep compact-overlapping content, numbers/names,
    connectors/glue, and late source content, while matching compact function rate.
    """
    k = len(comp)
    n = len(src)
    if k >= n:
        return list(range(n))
    m = n - k
    comp_norms = counter_norm(comp)
    comp_func = frac(sum(1 for w in comp if is_function(norm_word(w))), len(comp)) or 0.0
    source_norms = [norm_word(w) for w in src]
    possible_starts = list(range(0, n - m + 1))
    if k >= 4 and n - m + 1 > 2:
        # Prefer keeping at least two words on both sides when possible.
        possible_starts = [s for s in possible_starts if s >= 2 and (n - (s + m)) >= 2] or possible_starts
    best: tuple[float, int] | None = None
    for start in possible_starts:
        drop = set(range(start, start + m))
        kept = [i for i in range(n) if i not in drop]
        kept_norms = [source_norms[i] for i in kept]
        kept_counter = collections.Counter(nm for nm in kept_norms if nm)
        inter = multiset_intersection_size(kept_counter, comp_norms)
        compact_content_hits = sum(1 for i in kept if is_content(source_norms[i]) and comp_norms.get(source_norms[i], 0) > 0)
        content_kept = sum(1 for i in kept if is_content(source_norms[i]))
        tail_content_kept = sum(1 for i in kept if i >= k and is_content(source_norms[i]))
        important_kept = sum(1 for i in kept if (src[i][:1].isupper() and i != 0) or source_norms[i].isdigit())
        connectors_kept = sum(1 for i in kept if source_norms[i] in CONNECTORS)
        mean_pos = safe_mean([i / max(1, n - 1) for i in kept]) or 0.0
        width = (max(kept) - min(kept)) / max(1, n - 1)
        func = frac(sum(1 for i in kept if is_function(source_norms[i])), len(kept)) or 0.0
        func_penalty = abs(func - comp_func)
        # Penalize an ugly seam if neither side has punctuation/connective relief.
        seam_penalty = 0.0
        left = start - 1
        right = start + m
        if 0 <= left < n and 0 <= right < n:
            relief = punctish(src[left]) or punctish(src[right]) or source_norms[left] in CONNECTORS or source_norms[right] in CONNECTORS
            seam_penalty = 0.0 if relief else 0.35
        # Avoid degenerate all-prefix/all-suffix when source-wide alternatives exist.
        edge_penalty = 0.0
        if start == 0 or start + m == n:
            edge_penalty = 1.0
        score = (
            4.0 * compact_content_hits
            + 1.5 * inter
            + 0.8 * content_kept
            + 1.7 * tail_content_kept
            + 0.9 * important_kept
            + 0.3 * connectors_kept
            + 2.0 * width
            + 0.8 * mean_pos
            - 4.0 * func_penalty
            - seam_penalty
            - edge_penalty
        )
        cand = (score, -start)
        if best is None or cand > best:
            best = cand
    start = -best[1] if best is not None else 0
    drop = set(range(start, start + m))
    return [i for i in range(n) if i not in drop]


def prefix_indices(src: list[str], comp: list[str]) -> list[int]:
    k = len(comp)
    return list(range(min(k, len(src))))


def make_text(src: list[str], idxs: list[int]) -> str:
    return " ".join(src[i] for i in idxs if 0 <= i < len(src))


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_pairs(path: pathlib.Path) -> list[dict[str, Any]]:
    return [r for r in read_jsonl(path) if r.get("source_text") and r.get("rewrite_text")]


def is_word_start(token_str: str) -> bool:
    return token_str.startswith("Ġ") or token_str.startswith("▁")


def build_groups(ids: list[int], tokenizer, special_ids: set[int]) -> list[int]:
    groups = []
    gid = -1
    cache: dict[int, bool] = {}
    for i, tid in enumerate(ids):
        if tid in special_ids:
            groups.append(-1)
            continue
        v = cache.get(tid)
        if v is None:
            s = tokenizer.convert_ids_to_tokens(tid)
            v = bool(s is not None and is_word_start(str(s)))
            cache[tid] = v
        if gid < 0 or v or i == 0:
            gid += 1
        groups.append(gid)
    return groups


def token_info(text: str, tokenizer, seq_len: int, special_ids: set[int]) -> dict[str, Any]:
    enc = tokenizer(text, add_special_tokens=False, truncation=False)
    raw_ids = [int(x) for x in enc["input_ids"]]
    active_ids = raw_ids[:seq_len]
    candidate_ids = [tid for tid in active_ids if tid not in special_ids]
    groups = build_groups(active_ids, tokenizer, special_ids)
    group_count = len({g for g in groups if g >= 0})
    return {
        "raw_tokens": len(raw_ids),
        "active_tokens": len(active_ids),
        "truncated_tokens": max(0, len(raw_ids) - seq_len),
        "candidate_tokens": len(candidate_ids),
        "candidate_groups": group_count,
    }


def run_lengths(idxs: list[int]) -> list[int]:
    if not idxs:
        return []
    out = []
    cur = 1
    for a, b in zip(idxs, idxs[1:]):
        if b == a + 1:
            cur += 1
        else:
            out.append(cur)
            cur = 1
    out.append(cur)
    return out


def evaluate_pair_variant(pair: dict[str, Any], variant: str, view_words: list[str], source_idxs: list[int] | None, tokenizer, seq_len: int, special_ids: set[int]) -> dict[str, Any]:
    src = words(pair["source_text"])
    comp = words(pair["rewrite_text"])
    src_norms = [norm_word(w) for w in src]
    view_norms = [norm_word(w) for w in view_words]
    comp_norms = counter_norm(comp)
    view_counter = counter_norm(view_words)
    src_counter = counter_norm(src)
    content_positions = [i for i, n in enumerate(src_norms) if is_content(n)]
    tail_content = [i for i in content_positions if i >= len(comp)]
    view_norm_set = {n for n in view_norms if n}
    covered_content = [i for i in content_positions if src_norms[i] in view_norm_set]
    covered_tail = [i for i in tail_content if src_norms[i] in view_norm_set]
    idxs = source_idxs or []
    rl = run_lengths(sorted(idxs)) if idxs else []
    # Row is what DeBERTa MLM sees for a single pair if isolated; changed-block
    # packing is summarized separately.
    view_text = " ".join(view_words)
    row_text = str(pair["source_text"]).strip() + " " + view_text
    vtok = token_info(view_text, tokenizer, seq_len, special_ids)
    rtok = token_info(row_text, tokenizer, seq_len, special_ids)
    return {
        "pair_id": pair.get("pair_id"),
        "variant": variant,
        "source_words": len(src),
        "view_words": len(view_words),
        "exact_k": len(view_words) == len(comp),
        "view_text": view_text,
        "source_content_coverage": frac(len(covered_content), len(content_positions)),
        "tail_content_coverage": frac(len(covered_tail), len(tail_content)),
        "content_fraction": frac(sum(1 for n in view_norms if is_content(n)), len(view_norms)),
        "function_fraction": frac(sum(1 for n in view_norms if is_function(n)), len(view_norms)),
        "punctuation_per_word": frac(sum(1 for w in view_words if punctish(w)), len(view_words)),
        "jaccard_with_compact": jaccard(view_counter, comp_norms),
        "jaccard_with_source": jaccard(view_counter, src_counter),
        "compact_multiset_overlap": multiset_intersection_size(view_counter, comp_norms),
        "compact_overlap_fraction": frac(multiset_intersection_size(view_counter, comp_norms), len(view_words)),
        "source_overlap_fraction": frac(multiset_intersection_size(view_counter, src_counter), len(view_words)),
        "selected_source_width_frac": ((max(idxs) - min(idxs)) / max(1, len(src) - 1)) if idxs else None,
        "selected_tail_fraction": frac(sum(1 for i in idxs if i >= len(comp)), len(idxs)) if idxs else None,
        "mean_selected_source_pos_frac": safe_mean([i / max(1, len(src) - 1) for i in idxs]) if idxs else None,
        "num_source_runs": len(rl) if idxs else None,
        "mean_source_run_len": safe_mean(rl) if rl else None,
        "max_source_run_len": max(rl) if rl else None,
        "adjacent_pair_fraction": frac(sum(1 for a, b in zip(sorted(idxs), sorted(idxs)[1:]) if b == a + 1), max(0, len(idxs) - 1)) if len(idxs) > 1 else None,
        "view_raw_tokens": vtok["raw_tokens"],
        "view_tokens_per_word": frac(vtok["raw_tokens"], len(view_words)),
        "row_raw_tokens_single_pair": rtok["raw_tokens"],
        "row_truncated_tokens_single_pair": rtok["truncated_tokens"],
    }


@dataclass
class Segment:
    pair_id: str | None
    segment: str
    char_start: int
    char_end: int


def reconstruct_changed_row(meta_row: dict[str, Any], variant_views: dict[str, str], pair_by_id: dict[str, dict[str, Any]], topup_text: str) -> tuple[str, list[Segment]]:
    pair_ids = [str(x) for x in meta_row.get("pair_ids", [])]
    if not pair_ids:
        t = topup_text.strip()
        return t, [Segment(None, "other", 0, len(t))] if t else []
    parts: list[tuple[str, str, str]] = []
    for pid in pair_ids:
        source = str(pair_by_id[pid]["source_text"]).strip()
        view = variant_views[pid].strip()
        parts.append((pid, "source", source))
        parts.append((pid, "view", view))
    segments: list[Segment] = []
    out_parts: list[str] = []
    cur = 0
    for pid, seg, t in parts:
        if out_parts:
            cur += 1
        start = cur
        end = start + len(t)
        segments.append(Segment(pid, seg, start, end))
        out_parts.append(t)
        cur = end
    return " ".join(out_parts), segments


def offset_segment(offset: tuple[int, int], segs: list[Segment]) -> str:
    a, b = offset
    mid = (a + b - 1) / 2.0 if b > a else a
    best = "unknown"
    best_ov = -1
    for s in segs:
        if s.char_start <= mid < s.char_end:
            return s.segment
        ov = max(0, min(b, s.char_end) - max(a, s.char_start))
        if ov > best_ov:
            best_ov = ov
            best = s.segment
    return best if best_ov > 0 else "unknown"


def changed_block_loader_summary(meta_rows: list[dict[str, Any]], base_changed_rows: list[dict[str, Any]], variant_views: dict[str, str], pair_by_id: dict[str, dict[str, Any]], tokenizer, seq_len: int, special_ids: set[int]) -> dict[str, Any]:
    raw_tokens = active_tokens = trunc_tokens = candidate_tokens = candidate_groups = 0
    rows_truncated = 0
    seg_active = collections.Counter({"source": 0, "view": 0, "other": 0, "unknown": 0})
    seg_raw = collections.Counter({"source": 0, "view": 0, "other": 0, "unknown": 0})
    text_mismatch = 0
    word_mismatch = 0
    pair_full = []
    view_visible_fracs = []
    rows = []
    for i, meta in enumerate(meta_rows):
        text, segs = reconstruct_changed_row(meta, variant_views, pair_by_id, str(base_changed_rows[i].get("text", "")))
        if wc(text) != int(meta.get("words", -1)):
            word_mismatch += 1
        enc = tokenizer(text, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
        ids = [int(x) for x in enc["input_ids"]]
        offsets = [(int(a), int(b)) for a, b in enc.get("offset_mapping", [])]
        raw = len(ids); active = min(raw, seq_len); trunc = max(0, raw - seq_len)
        raw_tokens += raw; active_tokens += active; trunc_tokens += trunc
        if trunc:
            rows_truncated += 1
        active_ids = ids[:seq_len]
        candidate_tokens += sum(1 for tid in active_ids if tid not in special_ids)
        groups = build_groups(active_ids, tokenizer, special_ids)
        candidate_groups += len({g for g in groups if g >= 0})
        # Segment token counts.
        for ti, off in enumerate(offsets):
            seg = offset_segment(off, segs)
            if seg not in seg_raw:
                seg = "unknown"
            seg_raw[seg] += 1
            if ti < seq_len:
                seg_active[seg] += 1
        # Pair visibility.
        per_pair = collections.defaultdict(lambda: {"source_raw": 0, "view_raw": 0, "source_vis": 0, "view_vis": 0})
        for ti, off in enumerate(offsets):
            # Need pair id, not just segment.
            a, b = off
            mid = (a + b - 1) / 2.0 if b > a else a
            sid = None
            sseg = None
            best_ov = -1
            for s in segs:
                if s.pair_id is None:
                    continue
                if s.char_start <= mid < s.char_end:
                    sid = s.pair_id; sseg = s.segment; break
                ov = max(0, min(b, s.char_end) - max(a, s.char_start))
                if ov > best_ov:
                    best_ov = ov; sid = s.pair_id; sseg = s.segment
            if sid is None or sseg not in {"source", "view"}:
                continue
            per_pair[sid][f"{sseg}_raw"] += 1
            if ti < seq_len:
                per_pair[sid][f"{sseg}_vis"] += 1
        for d in per_pair.values():
            full = d["source_raw"] == d["source_vis"] and d["view_raw"] == d["view_vis"] and d["source_raw"] > 0 and d["view_raw"] > 0
            pair_full.append(full)
            if d["view_raw"]:
                view_visible_fracs.append(d["view_vis"] / d["view_raw"])
        rows.append({"row_index": int(meta.get("row_index", i)), "raw_tokens": raw, "active_tokens": active, "truncated_tokens": trunc, "pair_count": len(meta.get("pair_ids", [])), "text": text})
    return {
        "changed_rows": len(meta_rows),
        "changed_words": sum(int(r.get("words", 0)) for r in meta_rows),
        "raw_tokens": raw_tokens,
        "active_tokens": active_tokens,
        "truncated_tokens": trunc_tokens,
        "rows_truncated": rows_truncated,
        "candidate_tokens": candidate_tokens,
        "candidate_groups": candidate_groups,
        "expected_masked_tokens_p015": 0.15 * candidate_tokens,
        "active_segment_tokens": dict(seg_active),
        "raw_segment_tokens": dict(seg_raw),
        "pair_full_visible_fraction": frac(sum(1 for x in pair_full if x), len(pair_full)),
        "mean_view_visible_fraction": safe_mean(view_visible_fracs),
        "word_mismatch_rows": word_mismatch,
        "text_mismatch_rows": text_mismatch,
        "sample_rows": rows[:5],
    }


def summarize_pair_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    byv: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in records:
        byv[r["variant"]].append(r)
    for v, rows in sorted(byv.items()):
        def vals(k: str) -> list[float]:
            return [float(r[k]) for r in rows if isinstance(r.get(k), (int, float)) and math.isfinite(float(r[k]))]
        out[v] = {
            "n": len(rows),
            "all_exact_k": all(bool(r.get("exact_k")) for r in rows),
            "mean_source_content_coverage": safe_mean(vals("source_content_coverage")),
            "mean_tail_content_coverage": safe_mean(vals("tail_content_coverage")),
            "frac_tail_content_recovered": frac(sum(1 for r in rows if isinstance(r.get("tail_content_coverage"), (int, float)) and float(r["tail_content_coverage"]) > 0), sum(1 for r in rows if isinstance(r.get("tail_content_coverage"), (int, float)))),
            "mean_content_fraction": safe_mean(vals("content_fraction")),
            "mean_function_fraction": safe_mean(vals("function_fraction")),
            "mean_punctuation_per_word": safe_mean(vals("punctuation_per_word")),
            "mean_jaccard_with_compact": safe_mean(vals("jaccard_with_compact")),
            "mean_compact_overlap_fraction": safe_mean(vals("compact_overlap_fraction")),
            "mean_source_overlap_fraction": safe_mean(vals("source_overlap_fraction")),
            "mean_selected_source_width_frac": safe_mean(vals("selected_source_width_frac")),
            "mean_selected_tail_fraction": safe_mean(vals("selected_tail_fraction")),
            "mean_num_source_runs": safe_mean(vals("num_source_runs")),
            "mean_source_run_len": safe_mean(vals("mean_source_run_len")),
            "mean_max_source_run_len": safe_mean(vals("max_source_run_len")),
            "mean_adjacent_pair_fraction": safe_mean(vals("adjacent_pair_fraction")),
            "mean_view_tokens_per_word": safe_mean(vals("view_tokens_per_word")),
            "mean_row_raw_tokens_single_pair": safe_mean(vals("row_raw_tokens_single_pair")),
            "single_pair_truncation_rows": sum(1 for r in rows if int(r.get("row_truncated_tokens_single_pair", 0)) > 0),
        }
    return out


def fmt(x: Any, nd: int = 4, as_pct: bool = False) -> str:
    if x is None:
        return ""
    try:
        v = float(x)
    except Exception:
        return str(x)
    if as_pct:
        return f"{100*v:.2f}%"
    return f"{v:.{nd}f}"


def write_jsonl(rows: list[dict[str, Any]], path: pathlib.Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=pathlib.Path, default=DEFAULT_PAIRS)
    ap.add_argument("--meta", type=pathlib.Path, default=DEFAULT_META)
    ap.add_argument("--base-pool", type=pathlib.Path, default=DEFAULT_BASE_POOL)
    ap.add_argument("--tokenizer", type=pathlib.Path, default=DEFAULT_TOKENIZER)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--seq-len", type=int, default=256)
    args = ap.parse_args()
    t0 = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pairs = read_pairs(args.pairs)
    pair_by_id = {str(r["pair_id"]): r for r in pairs}
    meta_rows = read_jsonl(args.meta)
    base_changed_rows = read_jsonl(args.base_pool)[:len(meta_rows)]
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    special_ids = set(int(x) for x in tokenizer.all_special_ids)

    variant_views: dict[str, dict[str, str]] = {v: {} for v in VARIANT_ORDER}
    pair_metrics: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    for p in pairs:
        pid = str(p["pair_id"])
        src = words(p["source_text"])
        comp = words(p["rewrite_text"])
        pref_idx = prefix_indices(src, comp)
        onegap_idx = sourcewide_onegap(src, comp)
        bestspan_idx = best_contiguous_span(src, comp)
        variant_word_lists = {
            "compact": comp,
            "compact_scrambled": shuffle_words(comp, pid + ":compact_scrambled"),
            "prefix_fluent": [src[i] for i in pref_idx],
            "prefix_scrambled": shuffle_words([src[i] for i in pref_idx], pid + ":prefix_scrambled"),
            "sourcewide_onegap": [src[i] for i in onegap_idx],
            "sourcewide_onegap_scrambled": shuffle_words([src[i] for i in onegap_idx], pid + ":sourcewide_scrambled"),
            "best_contiguous_span": [src[i] for i in bestspan_idx],
        }
        idx_map = {
            "compact": None,
            "compact_scrambled": None,
            "prefix_fluent": pref_idx,
            "prefix_scrambled": pref_idx,
            "sourcewide_onegap": onegap_idx,
            "sourcewide_onegap_scrambled": onegap_idx,
            "best_contiguous_span": bestspan_idx,
        }
        for v, ws in variant_word_lists.items():
            view_text = " ".join(ws)
            variant_views[v][pid] = view_text
            m = evaluate_pair_variant(p, v, ws, idx_map[v], tokenizer, args.seq_len, special_ids)
            pair_metrics.append({k: val for k, val in m.items() if k != "view_text"})
        # Save high-leverage examples where onegap is source-wide and compact-like.
        m_one = [r for r in pair_metrics[-len(VARIANT_ORDER):] if r["variant"] == "sourcewide_onegap"][0]
        if (m_one.get("selected_tail_fraction") or 0) > 0.25 and (m_one.get("jaccard_with_compact") or 0) > 0.35:
            examples.append({
                "pair_id": pid,
                "source_text": p["source_text"],
                "compact": variant_views["compact"][pid],
                "prefix_fluent": variant_views["prefix_fluent"][pid],
                "sourcewide_onegap": variant_views["sourcewide_onegap"][pid],
                "compact_scrambled": variant_views["compact_scrambled"][pid],
                "sourcewide_onegap_scrambled": variant_views["sourcewide_onegap_scrambled"][pid],
                "sourcewide_metrics": {k: m_one.get(k) for k in ["tail_content_coverage", "selected_tail_fraction", "num_source_runs", "mean_adjacent_pair_fraction", "jaccard_with_compact", "function_fraction", "view_tokens_per_word"]},
            })

    # Write pair-level view files for inspectability, but these are not train pools.
    view_files: dict[str, str] = {}
    view_shas: dict[str, str] = {}
    for v in VARIANT_ORDER:
        rows = []
        for p in pairs:
            pid = str(p["pair_id"])
            rows.append({
                "pair_id": pid,
                "key": p.get("key"),
                "source_text": p.get("source_text"),
                "view_text": variant_views[v][pid],
                "variant": v,
                "source_words": wc(str(p.get("source_text", ""))),
                "view_words": wc(variant_views[v][pid]),
                "original_compact_text": p.get("rewrite_text"),
            })
        path = args.out_dir / f"{v}_candidate_pairs.jsonl"
        write_jsonl(rows, path)
        view_files[v] = str(path)
        view_shas[v] = sha256_file(path)

    pair_summary = summarize_pair_metrics(pair_metrics)
    block_summary: dict[str, Any] = {}
    for v in VARIANT_ORDER:
        block_summary[v] = changed_block_loader_summary(meta_rows, base_changed_rows, variant_views[v], pair_by_id, tokenizer, args.seq_len, special_ids)

    # Pair metrics CSV.
    csv_path = args.out_dir / "factorial_candidate_pair_metrics.csv"
    fieldnames = [k for k in pair_metrics[0].keys() if k != "view_text"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in pair_metrics:
            w.writerow({k: r.get(k) for k in fieldnames})

    ex_path = args.out_dir / "sourcewide_onegap_examples.jsonl"
    write_jsonl(examples[:80], ex_path)

    # Derived comparison checks for the intended factorial structure.
    comparisons: dict[str, Any] = {}
    def delta(a: str, b: str, key: str, summary: dict[str, dict[str, Any]]) -> float | None:
        if a not in summary or b not in summary:
            return None
        av = summary[a].get(key); bv = summary[b].get(key)
        if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
            return float(av) - float(bv)
        return None
    comparisons["coherence_fixed_words"] = {
        "compact_minus_compact_scrambled": {
            "delta_pair_view_tokens_per_word": delta("compact", "compact_scrambled", "mean_view_tokens_per_word", pair_summary),
            "delta_changed_active_tokens": block_summary["compact"]["active_tokens"] - block_summary["compact_scrambled"]["active_tokens"],
            "delta_changed_candidate_groups": block_summary["compact"]["candidate_groups"] - block_summary["compact_scrambled"]["candidate_groups"],
            "shared_word_multiset_by_construction": True,
        },
        "prefix_fluent_minus_prefix_scrambled": {
            "delta_pair_view_tokens_per_word": delta("prefix_fluent", "prefix_scrambled", "mean_view_tokens_per_word", pair_summary),
            "delta_changed_active_tokens": block_summary["prefix_fluent"]["active_tokens"] - block_summary["prefix_scrambled"]["active_tokens"],
            "delta_changed_candidate_groups": block_summary["prefix_fluent"]["candidate_groups"] - block_summary["prefix_scrambled"]["candidate_groups"],
            "shared_word_multiset_by_construction": True,
        },
        "sourcewide_onegap_minus_scrambled": {
            "delta_pair_view_tokens_per_word": delta("sourcewide_onegap", "sourcewide_onegap_scrambled", "mean_view_tokens_per_word", pair_summary),
            "delta_changed_active_tokens": block_summary["sourcewide_onegap"]["active_tokens"] - block_summary["sourcewide_onegap_scrambled"]["active_tokens"],
            "delta_changed_candidate_groups": block_summary["sourcewide_onegap"]["candidate_groups"] - block_summary["sourcewide_onegap_scrambled"]["candidate_groups"],
            "shared_word_multiset_by_construction": True,
        },
    }
    comparisons["position_spread_sentence_preserving"] = {
        "sourcewide_onegap_minus_prefix_fluent": {
            "delta_tail_content_coverage": delta("sourcewide_onegap", "prefix_fluent", "mean_tail_content_coverage", pair_summary),
            "delta_source_width_frac": delta("sourcewide_onegap", "prefix_fluent", "mean_selected_source_width_frac", pair_summary),
            "delta_num_source_runs": delta("sourcewide_onegap", "prefix_fluent", "mean_num_source_runs", pair_summary),
            "delta_function_fraction": delta("sourcewide_onegap", "prefix_fluent", "mean_function_fraction", pair_summary),
            "delta_changed_active_tokens": block_summary["sourcewide_onegap"]["active_tokens"] - block_summary["prefix_fluent"]["active_tokens"],
            "delta_changed_view_active_tokens": block_summary["sourcewide_onegap"]["active_segment_tokens"].get("view", 0) - block_summary["prefix_fluent"]["active_segment_tokens"].get("view", 0),
        }
    }
    comparisons["natural_compact_beyond_source_only"] = {
        "compact_minus_sourcewide_onegap": {
            "delta_tail_content_coverage": delta("compact", "sourcewide_onegap", "mean_tail_content_coverage", pair_summary),
            "delta_jaccard_with_compact": delta("compact", "sourcewide_onegap", "mean_jaccard_with_compact", pair_summary),
            "delta_function_fraction": delta("compact", "sourcewide_onegap", "mean_function_fraction", pair_summary),
            "delta_changed_active_tokens": block_summary["compact"]["active_tokens"] - block_summary["sourcewide_onegap"]["active_tokens"],
            "delta_changed_view_active_tokens": block_summary["compact"]["active_segment_tokens"].get("view", 0) - block_summary["sourcewide_onegap"]["active_segment_tokens"].get("view", 0),
        }
    }

    result = {
        "status": "FACTORIAL_VIEW_CANDIDATE_AUDIT",
        "purpose": "prototype matched controls for source-position spread, recurrent lexical content, and fluent compressed syntax before any H100 screen",
        "pairs_path": str(args.pairs),
        "pairs_sha256": sha256_file(args.pairs),
        "meta_path": str(args.meta),
        "meta_sha256": sha256_file(args.meta),
        "tokenizer_path": str(args.tokenizer),
        "tokenizer_json_sha256": sha256_file(args.tokenizer / "tokenizer.json") if (args.tokenizer / "tokenizer.json").exists() else None,
        "seq_len": args.seq_len,
        "n_pairs": len(pairs),
        "variants": VARIANT_ORDER,
        "variant_pair_files": view_files,
        "variant_pair_file_sha256": view_shas,
        "pair_summary": pair_summary,
        "changed_block_loader_summary": block_summary,
        "comparisons": comparisons,
        "pair_metrics_csv": str(csv_path),
        "examples_jsonl": str(ex_path),
        "scientific_warnings": [
            "These are candidate controls, not authorized training pools.",
            "Only within a word-multiset pair (compact/scrambled, prefix/scrambled, onegap/scrambled) is recurrent lexical content fixed exactly.",
            "The position-spread contrast sourcewide_onegap vs prefix_fluent still changes which source words recur; it must be exposure- and distribution-matched before training.",
            "Exact legal words alone are insufficient; changed-block active tokens, WWM candidate mass, truncation, and pair visibility must be close under the DeBERTa loader.",
        ],
        "elapsed_sec": time.time() - t0,
    }
    out_json = args.out_dir / "factorial_view_candidate_audit.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md_path = args.out_dir / "factorial_view_candidate_audit.md"
    lines = ["# research factorial view candidate audit", ""]
    lines.append("CPU-only candidate-control construction; no training/evaluation/upload/submission.")
    lines.append("")
    lines.append("## Pair-level geometry")
    lines.append("| variant | tail coverage | source width | runs | adjacencies | function frac | Jaccard compact | compact overlap | view tok/word |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for v in VARIANT_ORDER:
        s = pair_summary[v]
        lines.append(
            f"| {v} | {fmt(s.get('mean_tail_content_coverage'), as_pct=True)} | {fmt(s.get('mean_selected_source_width_frac'), as_pct=True)} | "
            f"{fmt(s.get('mean_num_source_runs'), 2)} | {fmt(s.get('mean_adjacent_pair_fraction'), as_pct=True)} | "
            f"{fmt(s.get('mean_function_fraction'), as_pct=True)} | {fmt(s.get('mean_jaccard_with_compact'), 4)} | "
            f"{fmt(s.get('mean_compact_overlap_fraction'), as_pct=True)} | {fmt(s.get('mean_view_tokens_per_word'), 4)} |"
        )
    lines.append("")
    lines.append("## Changed-block loader exposure")
    lines.append("| variant | active tokens | trunc tokens | trunc rows | candidate groups | E[masked tokens] | active source/view tokens | pair full visible |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for v in VARIANT_ORDER:
        b = block_summary[v]
        src = b["active_segment_tokens"].get("source", 0)
        view = b["active_segment_tokens"].get("view", 0)
        lines.append(
            f"| {v} | {b['active_tokens']} | {b['truncated_tokens']} | {b['rows_truncated']} | {b['candidate_groups']} | "
            f"{fmt(b['expected_masked_tokens_p015'], 1)} | {src}/{view} | {fmt(b.get('pair_full_visible_fraction'), as_pct=True)} |"
        )
    lines.append("")
    lines.append("## What the candidate family can and cannot identify")
    lines.append("- Coherence/order at fixed recurrent words is cleanest within compact vs compact_scrambled, prefix_fluent vs prefix_scrambled, and sourcewide_onegap vs sourcewide_onegap_scrambled: each pair shares the same word multiset and nearly the same loader exposure by construction.")
    lines.append("- Source-position spread is better isolated by sourcewide_onegap vs prefix_fluent than by the research telegraphic skeletons: onegap keeps at most two contiguous source spans, preserving local sentence fragments while increasing tail/source-width exposure. But it still changes which source words recur, so it is not a perfect fixed-identity contrast.")
    lines.append("- Natural compact style beyond source-only recurrence is read by compact vs sourcewide_onegap only after exposure and distribution differences are checked; it cannot be inferred from copy NLL alone.")
    lines.append("- These candidates are still not trainable routes until pending DeBERTa/A01 results are read and a final scaffold matches exact-loader exposure tightly enough.")
    lines.append("")
    lines.append("## Derived comparison highlights")
    for group, comp in comparisons.items():
        lines.append(f"### {group}")
        for name, vals in comp.items():
            lines.append(f"- {name}: `{json.dumps(vals, ensure_ascii=False)}`")
        lines.append("")
    lines.append(f"Pair metrics CSV: `{csv_path}`")
    lines.append(f"Examples: `{ex_path}`")
    lines.append("Pair files:")
    for v in VARIANT_ORDER:
        lines.append(f"- {v}: `{view_files[v]}` SHA `{view_shas[v]}`")
    lines.append(f"JSON: `{out_json}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "out_dir": str(args.out_dir), "n_pairs": len(pairs), "elapsed_sec": round(result["elapsed_sec"], 2)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
