#!/usr/bin/env python3
"""research: target-level recurrence/visibility/alignment audit for factorial view candidates.

This implements the strongest CPU-only follow-up from the research independent review memo.
For each candidate view word occurrence under the exact DeBERTa MLM row loader,
it records lexical copy zone, visibility, source counterpart visibility, BPE copy
status, WWM group size, source/view distance, and target-class mass.  The goal
is to expose whether proposed compact-view factor controls really match the
strata that carry the hypothesized mechanism before any H100 training.

No model is loaded. No training or evaluation is performed.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import pathlib
import re
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

from transformers import AutoTokenizer

ROOT = pathlib.Path("experiments/archive/frontier_consolidation")
CAND_DIR = ROOT / "data/factorial_view_candidate_audit"
DEFAULT_PAIRS = ROOT / "data/density_core_reinvestment_medium_riskhard/selected_compact_reinvest_pairs.jsonl"
DEFAULT_META = ROOT / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_changed_block_rows_meta.jsonl"
DEFAULT_BASE_POOL = ROOT / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"
DEFAULT_TOKENIZER = ROOT / "data/compliant_tokenizer"
DEFAULT_OUT = ROOT / "data/target_level_factor_alignment_audit"
VARIANTS = [
    "compact",
    "compact_scrambled",
    "prefix_fluent",
    "prefix_scrambled",
    "sourcewide_onegap",
    "sourcewide_onegap_scrambled",
    "best_contiguous_span",
]
WORD_RE = re.compile(r"\S+")
NORM_RE = re.compile(r"[A-Za-z0-9]+(?:['\u2019-][A-Za-z0-9]+)?")
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


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_pairs(path: pathlib.Path) -> dict[str, dict[str, Any]]:
    return {str(r["pair_id"]): r for r in load_jsonl(path) if r.get("pair_id")}


def load_views(candidate_dir: pathlib.Path, variants: list[str]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for v in variants:
        p = candidate_dir / f"{v}_candidate_pairs.jsonl"
        if not p.exists():
            raise FileNotFoundError(p)
        d = {}
        for r in load_jsonl(p):
            d[str(r["pair_id"])] = str(r["view_text"]).strip()
        out[v] = d
    return out


def norm_word(w: str) -> str:
    parts = NORM_RE.findall(w)
    return "".join(parts).lower() if parts else ""


def lex_class(word: str) -> str:
    n = norm_word(word)
    if not n:
        return "punct_or_symbol"
    if any(ch.isdigit() for ch in n):
        return "number"
    if n in STOPWORDS:
        return "function"
    if word[:1].isupper():
        return "capitalized_content"
    if len(n) <= 3:
        return "short_other"
    return "content"


def is_content_class(cls: str) -> bool:
    return cls in {"number", "capitalized_content", "content"}


def is_word_start(token_str: str) -> bool:
    return token_str.startswith("Ġ") or token_str.startswith("▁")


@dataclass
class WordOcc:
    pair_id: str | None
    segment: str
    word_pos: int
    text: str
    norm: str
    lex_class: str
    char_start: int
    char_end: int
    raw_token_indices: list[int] = field(default_factory=list)
    active_token_indices: list[int] = field(default_factory=list)
    active_token_ids: list[int] = field(default_factory=list)
    group_ids: list[int] = field(default_factory=list)

    def full_visible(self) -> bool:
        return bool(self.raw_token_indices) and len(self.raw_token_indices) == len(self.active_token_indices)

    def any_visible(self) -> bool:
        return bool(self.active_token_indices)

    def raw_token_count(self) -> int:
        return len(self.raw_token_indices)

    def active_token_count(self) -> int:
        return len(self.active_token_indices)

    def group_count(self) -> int:
        return len(set(g for g in self.group_ids if g >= 0))


def word_spans(part: str, base: int, pair_id: str | None, segment: str) -> list[WordOcc]:
    out = []
    for j, m in enumerate(WORD_RE.finditer(part)):
        text = m.group(0)
        out.append(WordOcc(
            pair_id=pair_id,
            segment=segment,
            word_pos=j,
            text=text,
            norm=norm_word(text),
            lex_class=lex_class(text),
            char_start=base + m.start(),
            char_end=base + m.end(),
        ))
    return out


def build_changed_row(meta: dict[str, Any], variant: str, pairs: dict[str, dict[str, Any]], views: dict[str, dict[str, str]], topup_text: str) -> tuple[str, list[WordOcc]]:
    pair_ids = [str(x) for x in meta.get("pair_ids", [])]
    if not pair_ids:
        t = topup_text.strip()
        return t, word_spans(t, 0, None, "other")
    text_parts = []
    occs: list[WordOcc] = []
    cursor = 0
    for pid in pair_ids:
        for seg, part in (("source", str(pairs[pid]["source_text"]).strip()), ("view", views[variant][pid].strip())):
            if text_parts:
                cursor += 1
            base = cursor
            occs.extend(word_spans(part, base, pid, seg))
            text_parts.append(part)
            cursor = base + len(part)
    return " ".join(text_parts), occs


def token_groups(active_ids: list[int], tokenizer, special_ids: set[int]) -> list[int]:
    groups = []
    gid = -1
    cache: dict[int, bool] = {}
    for i, tid in enumerate(active_ids):
        if tid in special_ids:
            groups.append(-1)
            continue
        flag = cache.get(tid)
        if flag is None:
            s = tokenizer.convert_ids_to_tokens(tid)
            flag = bool(s is not None and is_word_start(str(s)))
            cache[tid] = flag
        if gid < 0 or flag or i == 0:
            gid += 1
        groups.append(gid)
    return groups


def assign_token_to_word(a: int, b: int, words: list[WordOcc]) -> int | None:
    if b > a:
        mid = (a + b - 1) / 2.0
    else:
        mid = a
    best_i = None
    best_ov = -1
    for i, w in enumerate(words):
        if w.char_start <= mid < w.char_end:
            return i
        ov = max(0, min(b, w.char_end) - max(a, w.char_start))
        if ov > best_ov:
            best_ov = ov
            best_i = i
    return best_i if best_ov > 0 else None


def annotate_tokens(text: str, occs: list[WordOcc], tokenizer, seq_len: int, special_ids: set[int]) -> dict[str, Any]:
    enc = tokenizer(text, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
    ids = [int(x) for x in enc["input_ids"]]
    offsets = [(int(a), int(b)) for a, b in enc.get("offset_mapping", [])]
    active_len = min(len(ids), seq_len)
    groups = token_groups(ids[:active_len], tokenizer, special_ids)
    group_sizes = collections.Counter(g for g in groups if g >= 0)
    for ti, (a, b) in enumerate(offsets):
        wi = assign_token_to_word(a, b, occs)
        if wi is None:
            continue
        occ = occs[wi]
        occ.raw_token_indices.append(ti)
        if ti < active_len:
            occ.active_token_indices.append(ti)
            occ.active_token_ids.append(ids[ti])
            occ.group_ids.append(groups[ti] if ti < len(groups) else -1)
    return {
        "raw_tokens": len(ids),
        "active_tokens": active_len,
        "truncated_tokens": max(0, len(ids) - seq_len),
        "group_sizes": group_sizes,
    }


def zone_for_matches(match_positions: list[int], prefix_cut: int) -> str:
    if not match_positions:
        return "absent"
    has_prefix = any(i < prefix_cut for i in match_positions)
    has_tail = any(i >= prefix_cut for i in match_positions)
    if has_prefix and has_tail:
        return "both_prefix_tail"
    if has_tail:
        return "tail_only"
    return "prefix_only"


def source_match_bin(n: int) -> str:
    if n <= 0:
        return "0"
    if n == 1:
        return "1_unique"
    if n <= 3:
        return "2_3"
    return "4plus"


def safe_mean(xs: Iterable[float | int | None]) -> float | None:
    vals = [float(x) for x in xs if isinstance(x, (int, float)) and math.isfinite(float(x))]
    return sum(vals) / len(vals) if vals else None


def safe_median(xs: Iterable[float | int | None]) -> float | None:
    vals = [float(x) for x in xs if isinstance(x, (int, float)) and math.isfinite(float(x))]
    return statistics.median(vals) if vals else None


def compact_record(r: dict[str, Any]) -> dict[str, Any]:
    # Small enough for CSV while still carrying the target-level state.
    keys = [
        "variant", "row_index", "pair_id", "view_word_pos", "view_word", "norm", "lex_class",
        "source_match_count", "source_match_bin", "copy_zone", "source_decile_min", "source_decile_max",
        "view_full_visible", "view_any_visible", "counterpart_any_visible", "counterpart_full_visible",
        "complete_bpe_copy_any_source", "same_bpe_any_source", "frac_view_bpes_seen_in_source",
        "view_raw_tokens", "view_active_tokens", "wwm_group_count", "wwm_token_mass", "min_visible_source_to_view_token_gap",
    ]
    return {k: r.get(k) for k in keys}


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


def summarize_records(records: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    by_variant = collections.defaultdict(list)
    by_stratum = collections.defaultdict(list)
    by_zone = collections.defaultdict(list)
    for r in records:
        by_variant[r["variant"]].append(r)
        key = (r["variant"], r["copy_zone"], r["lex_class"])
        by_stratum[key].append(r)
        by_zone[(r["variant"], r["copy_zone"])].append(r)

    summary: dict[str, Any] = {}
    for v, rs in sorted(by_variant.items()):
        summary[v] = summarize_group(rs)
    strata_rows = []
    for (v, zone, cls), rs in sorted(by_stratum.items()):
        s = summarize_group(rs)
        s.update({"variant": v, "copy_zone": zone, "lex_class": cls})
        strata_rows.append(s)
    zone_rows = []
    for (v, zone), rs in sorted(by_zone.items()):
        s = summarize_group(rs)
        s.update({"variant": v, "copy_zone": zone})
        zone_rows.append(s)
    return summary, strata_rows, zone_rows


def summarize_group(rs: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rs)
    def count(pred) -> int:
        return sum(1 for r in rs if pred(r))
    def vals(k: str) -> list[float]:
        return [float(r[k]) for r in rs if isinstance(r.get(k), (int, float)) and math.isfinite(float(r[k]))]
    return {
        "targets": n,
        "view_raw_tokens": sum(int(r.get("view_raw_tokens") or 0) for r in rs),
        "view_active_tokens": sum(int(r.get("view_active_tokens") or 0) for r in rs),
        "wwm_token_mass": sum(int(r.get("wwm_token_mass") or 0) for r in rs),
        "wwm_groups": sum(int(r.get("wwm_group_count") or 0) for r in rs),
        "expected_masked_token_mass_p015": 0.15 * sum(int(r.get("wwm_token_mass") or 0) for r in rs),
        "full_visible_frac": count(lambda r: r.get("view_full_visible")) / n if n else None,
        "counterpart_any_visible_frac": count(lambda r: r.get("counterpart_any_visible")) / n if n else None,
        "counterpart_full_visible_frac": count(lambda r: r.get("counterpart_full_visible")) / n if n else None,
        "whole_word_copy_frac": count(lambda r: int(r.get("source_match_count") or 0) > 0) / n if n else None,
        "complete_bpe_copy_frac": count(lambda r: r.get("complete_bpe_copy_any_source")) / n if n else None,
        "same_bpe_any_frac": count(lambda r: r.get("same_bpe_any_source")) / n if n else None,
        "mean_frac_view_bpes_seen_in_source": safe_mean(vals("frac_view_bpes_seen_in_source")),
        "mean_min_visible_gap": safe_mean(vals("min_visible_source_to_view_token_gap")),
        "median_min_visible_gap": safe_median(vals("min_visible_source_to_view_token_gap")),
        "contentlike_frac": count(lambda r: is_content_class(str(r.get("lex_class")))) / n if n else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-dir", type=pathlib.Path, default=CAND_DIR)
    ap.add_argument("--pairs", type=pathlib.Path, default=DEFAULT_PAIRS)
    ap.add_argument("--meta", type=pathlib.Path, default=DEFAULT_META)
    ap.add_argument("--base-pool", type=pathlib.Path, default=DEFAULT_BASE_POOL)
    ap.add_argument("--tokenizer", type=pathlib.Path, default=DEFAULT_TOKENIZER)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--write-records", action="store_true", default=True)
    args = ap.parse_args()
    t0 = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs(args.pairs)
    views = load_views(args.candidate_dir, VARIANTS)
    meta_rows = load_jsonl(args.meta)
    base_rows = load_jsonl(args.base_pool)[:len(meta_rows)]
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    special_ids = set(int(x) for x in tokenizer.all_special_ids)

    all_records: list[dict[str, Any]] = []
    samples_by_need: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    row_issues = []
    for variant in VARIANTS:
        for row_i, meta in enumerate(meta_rows):
            text, occs = build_changed_row(meta, variant, pairs, views, str(base_rows[row_i].get("text", "")))
            if len(text.split()) != int(meta.get("words", -1)):
                row_issues.append({"variant": variant, "row_index": row_i, "words": len(text.split()), "meta_words": meta.get("words")})
            tokinfo = annotate_tokens(text, occs, tokenizer, args.seq_len, special_ids)
            source_by_pair: dict[str, list[WordOcc]] = collections.defaultdict(list)
            view_words: list[WordOcc] = []
            for occ in occs:
                if occ.pair_id is None:
                    continue
                if occ.segment == "source":
                    source_by_pair[occ.pair_id].append(occ)
                elif occ.segment == "view":
                    view_words.append(occ)
            source_by_norm: dict[str, dict[str, list[WordOcc]]] = {}
            for pid, sws in source_by_pair.items():
                d: dict[str, list[WordOcc]] = collections.defaultdict(list)
                for s in sws:
                    if s.norm:
                        d[s.norm].append(s)
                source_by_norm[pid] = d
            for vw in view_words:
                pid = str(vw.pair_id)
                pair = pairs[pid]
                prefix_cut = len(str(pair["rewrite_text"]).split())
                source_words = source_by_pair[pid]
                matches = source_by_norm[pid].get(vw.norm, []) if vw.norm else []
                match_positions = [m.word_pos for m in matches]
                zone = zone_for_matches(match_positions, prefix_cut)
                counterpart_any_visible = any(m.any_visible() for m in matches)
                counterpart_full_visible = any(m.full_visible() for m in matches)
                source_token_id_seqs = [tuple(m.active_token_ids) for m in matches if m.active_token_ids]
                view_seq = tuple(vw.active_token_ids)
                complete_bpe = bool(view_seq) and any(view_seq == s for s in source_token_id_seqs)
                source_token_set = {tid for m in matches for tid in m.active_token_ids}
                all_source_token_set = {tid for sw in source_words for tid in sw.active_token_ids}
                seen_tokens = sum(1 for tid in vw.active_token_ids if tid in all_source_token_set)
                same_bpe_any = bool(vw.active_token_ids) and seen_tokens > 0
                frac_seen = seen_tokens / len(vw.active_token_ids) if vw.active_token_ids else None
                source_deciles = [min(9, int(10 * m.word_pos / max(1, len(source_words)))) for m in matches]
                # Min visible token gap from any visible source occurrence with same normalized word.
                gaps = []
                if vw.active_token_indices:
                    v_start = min(vw.active_token_indices)
                    for m in matches:
                        if m.active_token_indices:
                            gaps.append(v_start - max(m.active_token_indices) - 1)
                group_sizes = tokinfo["group_sizes"]
                gset = {g for g in vw.group_ids if g >= 0}
                wwm_token_mass = sum(int(group_sizes[g]) for g in gset)
                rec = {
                    "variant": variant,
                    "row_index": row_i,
                    "pair_id": pid,
                    "view_word_pos": vw.word_pos,
                    "view_word": vw.text,
                    "norm": vw.norm,
                    "lex_class": vw.lex_class,
                    "source_match_count": len(matches),
                    "source_match_bin": source_match_bin(len(matches)),
                    "copy_zone": zone,
                    "source_decile_min": min(source_deciles) if source_deciles else None,
                    "source_decile_max": max(source_deciles) if source_deciles else None,
                    "view_full_visible": vw.full_visible(),
                    "view_any_visible": vw.any_visible(),
                    "counterpart_any_visible": counterpart_any_visible,
                    "counterpart_full_visible": counterpart_full_visible,
                    "complete_bpe_copy_any_source": complete_bpe,
                    "same_bpe_any_source": same_bpe_any,
                    "frac_view_bpes_seen_in_source": frac_seen,
                    "view_raw_tokens": vw.raw_token_count(),
                    "view_active_tokens": vw.active_token_count(),
                    "wwm_group_count": vw.group_count(),
                    "wwm_token_mass": wwm_token_mass,
                    "min_visible_source_to_view_token_gap": min(gaps) if gaps else None,
                }
                all_records.append(rec)
                # Keep a few representative records for human inspection.
                if len(samples_by_need[variant]) < 12 and zone in {"tail_only", "absent"} and vw.lex_class in {"content", "capitalized_content", "number"}:
                    samples_by_need[variant].append(compact_record(rec) | {"source_text": pair.get("source_text"), "view_text": views[variant][pid]})

    variant_summary, strata_rows, zone_rows = summarize_records(all_records)

    # CSV artifacts.
    records_csv = args.out_dir / "view_word_target_records.csv"
    if args.write_records:
        with records_csv.open("w", encoding="utf-8", newline="") as f:
            fieldnames = list(compact_record(all_records[0]).keys())
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in all_records:
                w.writerow(compact_record(r))
    strata_csv = args.out_dir / "target_strata_summary.csv"
    with strata_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["variant", "copy_zone", "lex_class", "targets", "view_active_tokens", "wwm_token_mass", "wwm_groups", "expected_masked_token_mass_p015", "full_visible_frac", "counterpart_any_visible_frac", "counterpart_full_visible_frac", "whole_word_copy_frac", "complete_bpe_copy_frac", "same_bpe_any_frac", "mean_frac_view_bpes_seen_in_source", "mean_min_visible_gap", "contentlike_frac"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in strata_rows:
            w.writerow({k: r.get(k) for k in fieldnames})
    zone_csv = args.out_dir / "copy_zone_summary.csv"
    with zone_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["variant", "copy_zone", "targets", "view_active_tokens", "wwm_token_mass", "wwm_groups", "expected_masked_token_mass_p015", "full_visible_frac", "counterpart_any_visible_frac", "complete_bpe_copy_frac", "mean_frac_view_bpes_seen_in_source", "contentlike_frac"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in zone_rows:
            w.writerow({k: r.get(k) for k in fieldnames})
    samples_path = args.out_dir / "target_alignment_samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as f:
        for variant, rows in samples_by_need.items():
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Derived pairwise contrasts important for design.
    def diff(a: str, b: str, key: str) -> float | None:
        av = variant_summary.get(a, {}).get(key); bv = variant_summary.get(b, {}).get(key)
        if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
            return float(av) - float(bv)
        return None
    design_contrasts = {
        "compact_vs_compact_scrambled_fixed_word_multiset": {
            "delta_targets": diff("compact", "compact_scrambled", "targets"),
            "delta_view_active_tokens": diff("compact", "compact_scrambled", "view_active_tokens"),
            "delta_wwm_token_mass": diff("compact", "compact_scrambled", "wwm_token_mass"),
            "delta_counterpart_visible_frac": diff("compact", "compact_scrambled", "counterpart_any_visible_frac"),
            "delta_complete_bpe_copy_frac": diff("compact", "compact_scrambled", "complete_bpe_copy_frac"),
        },
        "sourcewide_onegap_vs_scrambled_fixed_word_multiset": {
            "delta_targets": diff("sourcewide_onegap", "sourcewide_onegap_scrambled", "targets"),
            "delta_view_active_tokens": diff("sourcewide_onegap", "sourcewide_onegap_scrambled", "view_active_tokens"),
            "delta_wwm_token_mass": diff("sourcewide_onegap", "sourcewide_onegap_scrambled", "wwm_token_mass"),
            "delta_counterpart_visible_frac": diff("sourcewide_onegap", "sourcewide_onegap_scrambled", "counterpart_any_visible_frac"),
            "delta_complete_bpe_copy_frac": diff("sourcewide_onegap", "sourcewide_onegap_scrambled", "complete_bpe_copy_frac"),
        },
        "sourcewide_onegap_vs_prefix_fluent_position_confounded": {
            "delta_view_active_tokens": diff("sourcewide_onegap", "prefix_fluent", "view_active_tokens"),
            "delta_wwm_token_mass": diff("sourcewide_onegap", "prefix_fluent", "wwm_token_mass"),
            "delta_complete_bpe_copy_frac": diff("sourcewide_onegap", "prefix_fluent", "complete_bpe_copy_frac"),
            "delta_counterpart_visible_frac": diff("sourcewide_onegap", "prefix_fluent", "counterpart_any_visible_frac"),
        },
        "compact_vs_sourcewide_onegap_style_and_lexicon_confounded": {
            "delta_view_active_tokens": diff("compact", "sourcewide_onegap", "view_active_tokens"),
            "delta_wwm_token_mass": diff("compact", "sourcewide_onegap", "wwm_token_mass"),
            "delta_whole_word_copy_frac": diff("compact", "sourcewide_onegap", "whole_word_copy_frac"),
            "delta_complete_bpe_copy_frac": diff("compact", "sourcewide_onegap", "complete_bpe_copy_frac"),
        },
    }

    result = {
        "status": "TARGET_LEVEL_FACTOR_ALIGNMENT_AUDIT",
        "purpose": "target-level recurrence/copy-zone/visibility table for factorial design repair; no model training or evaluation",
        "pairs_path": str(args.pairs),
        "pairs_sha256": sha256_file(args.pairs),
        "candidate_dir": str(args.candidate_dir),
        "meta_path": str(args.meta),
        "meta_sha256": sha256_file(args.meta),
        "tokenizer_path": str(args.tokenizer),
        "tokenizer_json_sha256": sha256_file(args.tokenizer / "tokenizer.json") if (args.tokenizer / "tokenizer.json").exists() else None,
        "seq_len": args.seq_len,
        "variants": VARIANTS,
        "target_records": len(all_records),
        "variant_summary": variant_summary,
        "design_contrasts": design_contrasts,
        "row_issues": row_issues[:20],
        "n_row_issues": len(row_issues),
        "records_csv": str(records_csv) if args.write_records else None,
        "strata_csv": str(strata_csv),
        "zone_csv": str(zone_csv),
        "samples_jsonl": str(samples_path),
        "elapsed_sec": time.time() - t0,
        "scientific_reading": [
            "Within fixed-word-multiset ordered/scrambled pairs, target counts and active-token mass should be nearly identical; any large difference would confound an order/coherence probe.",
            "Cross-family comparisons such as sourcewide_onegap vs prefix_fluent or compact vs sourcewide_onegap remain confounded if target copy zones, whole-word/BPE copy fractions, or view-active target mass differ.",
            "This audit estimates expected WWM target opportunity, not realized stochastic masks; a frozen-model probe should mask aligned stable lexical occurrences explicitly.",
        ],
    }
    out_json = args.out_dir / "target_level_factor_alignment_audit.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md = args.out_dir / "target_level_factor_alignment_audit.md"
    lines = ["# research target-level factor alignment audit", ""]
    lines.append("CPU/tokenizer-only audit of view-word targets under exact DeBERTa row truncation and WWM grouping. No model was loaded.")
    lines.append("")
    lines.append(f"Target records: {len(all_records):,}; tokenizer SHA `{result['tokenizer_json_sha256']}`; seq_len={args.seq_len}.")
    lines.append("")
    lines.append("## Variant target mass")
    lines.append("| variant | targets | active target tokens | WWM token mass | WWM groups | full visible | counterpart visible | whole-word copy | complete-BPE copy | seen-BPE frac | contentlike |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for v in VARIANTS:
        s = variant_summary[v]
        lines.append(
            f"| {v} | {s['targets']} | {s['view_active_tokens']} | {s['wwm_token_mass']} | {s['wwm_groups']} | "
            f"{fmt(s['full_visible_frac'], as_pct=True)} | {fmt(s['counterpart_any_visible_frac'], as_pct=True)} | "
            f"{fmt(s['whole_word_copy_frac'], as_pct=True)} | {fmt(s['complete_bpe_copy_frac'], as_pct=True)} | "
            f"{fmt(s['mean_frac_view_bpes_seen_in_source'], as_pct=True)} | {fmt(s['contentlike_frac'], as_pct=True)} |"
        )
    lines.append("")
    lines.append("## Copy-zone target mass (expected WWM token opportunity)")
    lines.append("| variant | zone | targets | active tokens | WWM mass | counterpart visible | complete-BPE copy | contentlike |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in zone_rows:
        lines.append(
            f"| {r['variant']} | {r['copy_zone']} | {r['targets']} | {r['view_active_tokens']} | {r['wwm_token_mass']} | "
            f"{fmt(r['counterpart_any_visible_frac'], as_pct=True)} | {fmt(r['complete_bpe_copy_frac'], as_pct=True)} | {fmt(r['contentlike_frac'], as_pct=True)} |"
        )
    lines.append("")
    lines.append("## Design contrasts")
    for k, d in design_contrasts.items():
        lines.append(f"- {k}: `{json.dumps(d, ensure_ascii=False)}`")
    lines.append("")
    lines.append("## Scientific reading")
    lines.append("The table should be used to repair future controls, not to justify training. Fixed-word-multiset ordered/scrambled pairs are the cleanest current single-factor probes. Position-spread and natural-compact comparisons still require target-composition and exposure matching because lexical copy zones and active target mass can differ even when legal words and row order are fixed.")
    lines.append("")
    lines.append(f"Full target records CSV: `{records_csv}`")
    lines.append(f"Strata summary CSV: `{strata_csv}`")
    lines.append(f"Copy-zone summary CSV: `{zone_csv}`")
    lines.append(f"Samples: `{samples_path}`")
    lines.append(f"JSON: `{out_json}`")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"status": result["status"], "out_dir": str(args.out_dir), "target_records": len(all_records), "n_row_issues": len(row_issues), "elapsed_sec": round(result["elapsed_sec"], 2)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
