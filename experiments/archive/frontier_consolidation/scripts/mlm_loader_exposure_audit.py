#!/usr/bin/env python3
"""research: exact DeBERTa MLM loader exposure audit for skeleton-view pools.

Purpose: before any H100 screen, quantify the actual row-tokenized/truncated
exposure geometry used by the inherited DeBERTa WWM trainer.  The trainer
encodes each JSONL row independently with add_special_tokens=False,
truncation=True, max_length=256, padding=max_length, then selects whole-word
BPE groups with nominal probability 0.15.  This script reproduces the static
parts of that loader exactly enough to expose confounds: active tokens,
truncation, WWM candidate groups/tokens, source/view co-visibility, and segment
mass for the research pool scaffolds.

It does not train, score, or inspect pending experimental outputs.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import pathlib
import statistics
import time
from dataclasses import dataclass
from typing import Any, Iterable

from transformers import AutoTokenizer

ROOT = pathlib.Path("experiments/archive/frontier_consolidation")
DEFAULT_MANIFEST = ROOT / "data/skeleton_reinvest_pool_scaffold/skeleton_reinvest_pool_manifest.json"
DEFAULT_PAIRS = ROOT / "data/density_core_reinvestment_medium_riskhard/selected_compact_reinvest_pairs.jsonl"
DEFAULT_VARIANT_DIR = ROOT / "data/extractive_skeleton_variant_audit"
DEFAULT_META = ROOT / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_changed_block_rows_meta.jsonl"
DEFAULT_TOKENIZER = ROOT / "data/compliant_tokenizer"
DEFAULT_OUT = ROOT / "data/mlm_loader_exposure_audit_pools"

SEG_ORDER = ["source", "view", "other", "unknown"]


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def wc(text: str) -> int:
    return len(text.split())


def safe_mean(xs: Iterable[float | int | None]) -> float | None:
    vals = [float(x) for x in xs if isinstance(x, (int, float)) and math.isfinite(float(x))]
    return sum(vals) / len(vals) if vals else None


def safe_median(xs: Iterable[float | int | None]) -> float | None:
    vals = [float(x) for x in xs if isinstance(x, (int, float)) and math.isfinite(float(x))]
    return statistics.median(vals) if vals else None


def pct(num: float, den: float) -> float | None:
    return float(num) / float(den) if den else None


def is_word_start(token_str: str) -> bool:
    return token_str.startswith("Ġ") or token_str.startswith("▁")


def build_word_groups(input_ids: list[int], attention_mask: list[int], tokenizer, special_ids: set[int]) -> list[int]:
    groups: list[int] = []
    gid = -1
    cache: dict[int, bool] = {}
    for i, (tid, attn) in enumerate(zip(input_ids, attention_mask)):
        if not attn:
            groups.append(-1)
            continue
        tid_i = int(tid)
        if tid_i in special_ids:
            groups.append(-1)
            continue
        v = cache.get(tid_i)
        if v is None:
            s = tokenizer.convert_ids_to_tokens(tid_i)
            v = bool(s is not None and is_word_start(str(s)))
            cache[tid_i] = v
        if gid < 0 or v or i == 0:
            gid += 1
        groups.append(gid)
    return groups


def load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_pairs(path: pathlib.Path) -> dict[str, dict[str, Any]]:
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[str(r["pair_id"])] = r
    return out


def load_variant_views(variant_dir: pathlib.Path, variants: list[str], pairs: dict[str, dict[str, Any]]) -> dict[str, dict[str, str]]:
    views: dict[str, dict[str, str]] = {}
    for v in variants:
        if v == "compact":
            views[v] = {pid: str(r["rewrite_text"]).strip() for pid, r in pairs.items()}
            continue
        p = variant_dir / f"{v}_pairs.jsonl"
        if not p.exists():
            raise FileNotFoundError(f"variant pair file missing for {v}: {p}")
        d: dict[str, str] = {}
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    d[str(r["pair_id"])] = str(r["view_text"]).strip()
        views[v] = d
    return views


@dataclass
class Segment:
    pair_id: str | None
    segment: str
    char_start: int
    char_end: int
    text: str


def reconstruct_segments(meta_row: dict[str, Any], variant: str, pairs: dict[str, dict[str, Any]], views: dict[str, dict[str, str]], topup_text: str | None = None) -> tuple[str, list[Segment]]:
    pair_ids = [str(x) for x in meta_row.get("pair_ids", [])]
    if not pair_ids:
        text = (topup_text or "").strip()
        return text, [Segment(None, "other", 0, len(text), text)] if text else []
    parts: list[tuple[str | None, str, str]] = []
    for pid in pair_ids:
        p = pairs[pid]
        source_text = str(p["source_text"]).strip()
        view_text = str(views[variant][pid]).strip()
        parts.append((pid, "source", source_text))
        parts.append((pid, "view", view_text))
    text_parts: list[str] = []
    segments: list[Segment] = []
    cursor = 0
    for pid, seg, part in parts:
        if not part:
            continue
        if text_parts:
            cursor += 1  # one join-space before this part
        start = cursor
        end = start + len(part)
        segments.append(Segment(pid, seg, start, end, part))
        text_parts.append(part)
        cursor = end
    text = " ".join(text_parts)
    return text, segments


def assign_offset_to_segment(start: int, end: int, segments: list[Segment]) -> int | None:
    if not segments:
        return None
    if end <= start:
        # Fast tokenizers sometimes return zero offsets for special/empty tokens.
        mid = start
    else:
        mid = (start + end - 1) / 2.0
    best_i = None
    best_overlap = -1
    for i, s in enumerate(segments):
        if s.char_start <= mid < s.char_end:
            return i
        ov = max(0, min(end, s.char_end) - max(start, s.char_start))
        if ov > best_overlap:
            best_overlap = ov
            best_i = i
    return best_i if best_overlap > 0 else None


def row_token_info(text: str, tokenizer, seq_len: int, special_ids: set[int]) -> dict[str, Any]:
    # Raw tokenization is used to determine truncation.  Training then truncates
    # to seq_len and pads to seq_len; padding contributes no active candidate mass.
    enc_raw = tokenizer(text, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
    raw_ids = [int(x) for x in enc_raw["input_ids"]]
    raw_offsets = [(int(a), int(b)) for a, b in enc_raw.get("offset_mapping", [])]
    active_ids = raw_ids[:seq_len]
    active_mask = [1] * len(active_ids)
    if len(active_ids) < seq_len:
        active_mask += [0] * (seq_len - len(active_ids))
        active_ids += [int(tokenizer.pad_token_id)] * (seq_len - len(active_ids))
    groups = build_word_groups(active_ids, active_mask, tokenizer, special_ids)
    candidate_positions = [i for i, (tid, attn) in enumerate(zip(active_ids, active_mask)) if attn and int(tid) not in special_ids]
    group_to_tokens: dict[int, list[int]] = collections.defaultdict(list)
    for i in candidate_positions:
        gid = groups[i]
        if gid >= 0:
            group_to_tokens[gid].append(i)
    return {
        "raw_ids": raw_ids,
        "raw_offsets": raw_offsets,
        "raw_token_count": len(raw_ids),
        "active_token_count": min(len(raw_ids), seq_len),
        "truncated_token_count": max(0, len(raw_ids) - seq_len),
        "candidate_token_count": len(candidate_positions),
        "candidate_group_count": len(group_to_tokens),
        "group_to_tokens": group_to_tokens,
        "active_ids": active_ids,
        "active_mask": active_mask,
        "groups": groups,
    }


def analyze_changed_row(meta_row: dict[str, Any], row_obj: dict[str, Any], variant: str, tokenizer, seq_len: int, special_ids: set[int], pairs: dict[str, dict[str, Any]], views: dict[str, dict[str, str]]) -> dict[str, Any]:
    recon, segments = reconstruct_segments(meta_row, variant, pairs, views, topup_text=str(row_obj.get("text", "")))
    actual_text = str(row_obj.get("text", "")).strip()
    text_match = (recon == actual_text)
    info = row_token_info(actual_text, tokenizer, seq_len, special_ids)
    raw_seg_counts = collections.Counter({k: 0 for k in SEG_ORDER})
    active_seg_counts = collections.Counter({k: 0 for k in SEG_ORDER})
    raw_seg_by_index: list[str] = []
    for i, (a, b) in enumerate(info["raw_offsets"]):
        si = assign_offset_to_segment(a, b, segments)
        seg = segments[si].segment if si is not None else "unknown"
        if seg not in SEG_ORDER:
            seg = "unknown"
        raw_seg_by_index.append(seg)
        raw_seg_counts[seg] += 1
        if i < seq_len:
            active_seg_counts[seg] += 1
    group_seg_counts = collections.Counter({k: 0 for k in SEG_ORDER})
    group_token_mass = collections.Counter({k: 0 for k in SEG_ORDER})
    for gid, toks in info["group_to_tokens"].items():
        segs = [raw_seg_by_index[t] if t < len(raw_seg_by_index) else "unknown" for t in toks]
        seg = collections.Counter(segs).most_common(1)[0][0] if segs else "unknown"
        group_seg_counts[seg] += 1
        group_token_mass[seg] += len(toks)

    pair_stats: list[dict[str, Any]] = []
    by_pair: dict[str, dict[str, list[int]]] = collections.defaultdict(lambda: {"source_raw": [], "view_raw": [], "source_visible": [], "view_visible": []})
    for i, (a, b) in enumerate(info["raw_offsets"]):
        si = assign_offset_to_segment(a, b, segments)
        if si is None:
            continue
        s = segments[si]
        if s.pair_id is None or s.segment not in {"source", "view"}:
            continue
        by_pair[s.pair_id][f"{s.segment}_raw"].append(i)
        if i < seq_len:
            by_pair[s.pair_id][f"{s.segment}_visible"].append(i)
    for pid, d in by_pair.items():
        sr, vr = d["source_raw"], d["view_raw"]
        sv, vv = d["source_visible"], d["view_visible"]
        source_full = bool(sr) and len(sv) == len(sr)
        view_full = bool(vr) and len(vv) == len(vr)
        source_any = bool(sv)
        view_any = bool(vv)
        dist = None
        if sv and vv:
            dist = min(vv) - max(sv) - 1
        pair_stats.append({
            "pair_id": pid,
            "source_raw_tokens": len(sr),
            "view_raw_tokens": len(vr),
            "source_visible_tokens": len(sv),
            "view_visible_tokens": len(vv),
            "source_visible_fraction": len(sv) / len(sr) if sr else None,
            "view_visible_fraction": len(vv) / len(vr) if vr else None,
            "both_any_visible": source_any and view_any,
            "both_full_visible": source_full and view_full,
            "source_full_visible": source_full,
            "view_full_visible": view_full,
            "token_gap_source_to_view_visible": dist,
        })
    return {
        "row_index": int(meta_row.get("row_index", -1)),
        "example_id": row_obj.get("example_id"),
        "text_match": text_match,
        "word_match_meta": wc(actual_text) == int(meta_row.get("words", -1)),
        "row_words": int(row_obj.get("words", wc(actual_text))),
        "meta_words": int(meta_row.get("words", -1)),
        "pair_count": len(meta_row.get("pair_ids", [])),
        "raw_token_count": info["raw_token_count"],
        "active_token_count": info["active_token_count"],
        "truncated_token_count": info["truncated_token_count"],
        "candidate_token_count": info["candidate_token_count"],
        "candidate_group_count": info["candidate_group_count"],
        "raw_segment_tokens": dict(raw_seg_counts),
        "active_segment_tokens": dict(active_seg_counts),
        "wwm_candidate_groups_by_segment": dict(group_seg_counts),
        "wwm_candidate_token_mass_by_segment": dict(group_token_mass),
        "pair_stats": pair_stats,
    }


def summarize_variant(rows: list[dict[str, Any]], changed_details: list[dict[str, Any]], seq_len: int) -> dict[str, Any]:
    total_words = sum(int(r.get("words", wc(str(r.get("text", ""))))) for r in rows)
    raw_tokens = sum(int(r["raw_token_count"]) for r in rows)
    active_tokens = sum(int(r["active_token_count"]) for r in rows)
    trunc_tokens = sum(int(r["truncated_token_count"]) for r in rows)
    candidate_tokens = sum(int(r["candidate_token_count"]) for r in rows)
    candidate_groups = sum(int(r["candidate_group_count"]) for r in rows)
    rows_truncated = sum(1 for r in rows if int(r["truncated_token_count"]) > 0)
    changed_tokens = sum(int(r["active_token_count"]) for r in changed_details)
    changed_raw_tokens = sum(int(r["raw_token_count"]) for r in changed_details)
    seg_active = collections.Counter({k: 0 for k in SEG_ORDER})
    seg_raw = collections.Counter({k: 0 for k in SEG_ORDER})
    seg_groups = collections.Counter({k: 0 for k in SEG_ORDER})
    seg_group_mass = collections.Counter({k: 0 for k in SEG_ORDER})
    pair_stats = []
    for r in changed_details:
        seg_active.update({k: int(v) for k, v in r["active_segment_tokens"].items()})
        seg_raw.update({k: int(v) for k, v in r["raw_segment_tokens"].items()})
        seg_groups.update({k: int(v) for k, v in r["wwm_candidate_groups_by_segment"].items()})
        seg_group_mass.update({k: int(v) for k, v in r["wwm_candidate_token_mass_by_segment"].items()})
        pair_stats.extend(r["pair_stats"])
    visible_pairs = [p for p in pair_stats if p.get("both_any_visible")]
    return {
        "rows": len(rows),
        "words": total_words,
        "raw_tokens": raw_tokens,
        "active_tokens": active_tokens,
        "truncated_tokens": trunc_tokens,
        "rows_truncated": rows_truncated,
        "rows_truncated_fraction": pct(rows_truncated, len(rows)),
        "candidate_tokens": candidate_tokens,
        "candidate_groups": candidate_groups,
        "expected_wwm_masked_tokens_at_p015": 0.15 * candidate_tokens,
        "expected_wwm_masked_groups_at_p015": 0.15 * candidate_groups,
        "tokens_per_word_raw": pct(raw_tokens, total_words),
        "tokens_per_word_active": pct(active_tokens, total_words),
        "changed_active_tokens": changed_tokens,
        "changed_raw_tokens": changed_raw_tokens,
        "changed_active_token_fraction": pct(changed_tokens, active_tokens),
        "changed_truncated_tokens": sum(int(r["truncated_token_count"]) for r in changed_details),
        "changed_rows_truncated": sum(1 for r in changed_details if int(r["truncated_token_count"]) > 0),
        "changed_rows": len(changed_details),
        "changed_text_mismatches": sum(1 for r in changed_details if not r["text_match"]),
        "changed_word_mismatches": sum(1 for r in changed_details if not r["word_match_meta"]),
        "active_segment_tokens_changed": dict(seg_active),
        "raw_segment_tokens_changed": dict(seg_raw),
        "wwm_candidate_groups_by_segment_changed": dict(seg_groups),
        "wwm_candidate_token_mass_by_segment_changed": dict(seg_group_mass),
        "expected_masked_tokens_by_segment_changed_p015": {k: 0.15 * int(v) for k, v in seg_group_mass.items()},
        "expected_masked_groups_by_segment_changed_p015": {k: 0.15 * int(v) for k, v in seg_groups.items()},
        "pair_records": len(pair_stats),
        "pair_both_any_visible_fraction": pct(sum(1 for p in pair_stats if p.get("both_any_visible")), len(pair_stats)),
        "pair_both_full_visible_fraction": pct(sum(1 for p in pair_stats if p.get("both_full_visible")), len(pair_stats)),
        "source_full_visible_fraction": pct(sum(1 for p in pair_stats if p.get("source_full_visible")), len(pair_stats)),
        "view_full_visible_fraction": pct(sum(1 for p in pair_stats if p.get("view_full_visible")), len(pair_stats)),
        "mean_source_visible_fraction": safe_mean(p.get("source_visible_fraction") for p in pair_stats),
        "mean_view_visible_fraction": safe_mean(p.get("view_visible_fraction") for p in pair_stats),
        "mean_visible_source_to_view_token_gap": safe_mean(p.get("token_gap_source_to_view_visible") for p in visible_pairs),
        "median_visible_source_to_view_token_gap": safe_median(p.get("token_gap_source_to_view_visible") for p in visible_pairs),
    }


def fmt(x: Any, nd: int = 6, as_pct: bool = False) -> str:
    if x is None:
        return ""
    try:
        v = float(x)
    except Exception:
        return str(x)
    if as_pct:
        return f"{100*v:.2f}%"
    return f"{v:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=pathlib.Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--pairs", type=pathlib.Path, default=DEFAULT_PAIRS)
    ap.add_argument("--variant-dir", type=pathlib.Path, default=DEFAULT_VARIANT_DIR)
    ap.add_argument("--meta", type=pathlib.Path, default=DEFAULT_META)
    ap.add_argument("--tokenizer", type=pathlib.Path, default=DEFAULT_TOKENIZER)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--seq-len", type=int, default=256)
    args = ap.parse_args()
    t0 = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    variant_paths = {v: pathlib.Path(s["path"]) for v, s in manifest["variants"].items()}
    variants = list(variant_paths.keys())
    meta_rows = load_jsonl(args.meta)
    pairs = load_pairs(args.pairs)
    views = load_variant_views(args.variant_dir, variants, pairs)
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    special_ids = set(int(x) for x in tokenizer.all_special_ids)

    result: dict[str, Any] = {
        "status": "MLM_LOADER_EXPOSURE_AUDIT",
        "purpose": "static exact-loader exposure/topology audit before any skeleton H100 training",
        "manifest_path": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "pairs_path": str(args.pairs),
        "pairs_sha256": sha256_file(args.pairs),
        "meta_path": str(args.meta),
        "meta_sha256": sha256_file(args.meta),
        "tokenizer_path": str(args.tokenizer),
        "tokenizer_json_sha256": sha256_file(args.tokenizer / "tokenizer.json") if (args.tokenizer / "tokenizer.json").exists() else None,
        "seq_len": args.seq_len,
        "trainer_coordinate": {
            "row_tokenization": "tokenizer(text, add_special_tokens=False, truncation=True, max_length=256, padding=max_length)",
            "loader_order": "DataLoader shuffle=False over selected JSONL rows",
            "masking": "wwm_fixed, nominal p=0.15 over tokenizer word-start groups; expected target mass reported, not a realized CUDA RNG mask stream",
        },
        "variants": {},
        "comparisons_vs_compact": {},
        "comparisons_vs_prefix_repeat": {},
        "issues": [],
    }

    per_row_csv = args.out_dir / "changed_row_loader_exposure.csv"
    pair_csv = args.out_dir / "pair_visibility_loader_exposure.csv"
    row_fields = [
        "variant", "row_index", "example_id", "row_words", "pair_count", "text_match", "word_match_meta",
        "raw_token_count", "active_token_count", "truncated_token_count", "candidate_token_count", "candidate_group_count",
        "active_source_tokens", "active_view_tokens", "active_other_tokens",
        "wwm_source_token_mass", "wwm_view_token_mass", "wwm_source_groups", "wwm_view_groups",
    ]
    pair_fields = [
        "variant", "row_index", "pair_id", "source_raw_tokens", "view_raw_tokens", "source_visible_tokens", "view_visible_tokens",
        "source_visible_fraction", "view_visible_fraction", "both_any_visible", "both_full_visible", "token_gap_source_to_view_visible",
    ]
    with per_row_csv.open("w", encoding="utf-8", newline="") as rf, pair_csv.open("w", encoding="utf-8", newline="") as pf:
        rw = csv.DictWriter(rf, fieldnames=row_fields)
        pw = csv.DictWriter(pf, fieldnames=pair_fields)
        rw.writeheader(); pw.writeheader()
        for variant, path in variant_paths.items():
            if not path.exists():
                result["issues"].append(f"missing pool for {variant}: {path}")
                continue
            raw_rows = load_jsonl(path)
            all_row_static = []
            changed_details = []
            for i, row in enumerate(raw_rows):
                text = str(row.get("text", ""))
                info = row_token_info(text, tokenizer, args.seq_len, special_ids)
                base_static = {
                    "raw_token_count": info["raw_token_count"],
                    "active_token_count": info["active_token_count"],
                    "truncated_token_count": info["truncated_token_count"],
                    "candidate_token_count": info["candidate_token_count"],
                    "candidate_group_count": info["candidate_group_count"],
                }
                all_row_static.append({**row, **base_static})
                if i < len(meta_rows):
                    d = analyze_changed_row(meta_rows[i], row, variant, tokenizer, args.seq_len, special_ids, pairs, views)
                    changed_details.append(d)
                    rw.writerow({
                        "variant": variant,
                        "row_index": d["row_index"],
                        "example_id": d["example_id"],
                        "row_words": d["row_words"],
                        "pair_count": d["pair_count"],
                        "text_match": d["text_match"],
                        "word_match_meta": d["word_match_meta"],
                        "raw_token_count": d["raw_token_count"],
                        "active_token_count": d["active_token_count"],
                        "truncated_token_count": d["truncated_token_count"],
                        "candidate_token_count": d["candidate_token_count"],
                        "candidate_group_count": d["candidate_group_count"],
                        "active_source_tokens": d["active_segment_tokens"].get("source", 0),
                        "active_view_tokens": d["active_segment_tokens"].get("view", 0),
                        "active_other_tokens": d["active_segment_tokens"].get("other", 0),
                        "wwm_source_token_mass": d["wwm_candidate_token_mass_by_segment"].get("source", 0),
                        "wwm_view_token_mass": d["wwm_candidate_token_mass_by_segment"].get("view", 0),
                        "wwm_source_groups": d["wwm_candidate_groups_by_segment"].get("source", 0),
                        "wwm_view_groups": d["wwm_candidate_groups_by_segment"].get("view", 0),
                    })
                    for p in d["pair_stats"]:
                        pw.writerow({
                            "variant": variant,
                            "row_index": d["row_index"],
                            "pair_id": p["pair_id"],
                            "source_raw_tokens": p["source_raw_tokens"],
                            "view_raw_tokens": p["view_raw_tokens"],
                            "source_visible_tokens": p["source_visible_tokens"],
                            "view_visible_tokens": p["view_visible_tokens"],
                            "source_visible_fraction": p["source_visible_fraction"],
                            "view_visible_fraction": p["view_visible_fraction"],
                            "both_any_visible": p["both_any_visible"],
                            "both_full_visible": p["both_full_visible"],
                            "token_gap_source_to_view_visible": p["token_gap_source_to_view_visible"],
                        })
            summary = summarize_variant(all_row_static, changed_details, args.seq_len)
            summary.update({
                "pool_path": str(path),
                "pool_sha256": sha256_file(path),
            })
            result["variants"][variant] = summary

    def add_comp(label: str, base_label: str, target: dict[str, Any]) -> None:
        base = result["variants"].get(base_label)
        if not base:
            return
        for v, s in result["variants"].items():
            if v == base_label:
                continue
            target[v] = {
                "delta_raw_tokens": s["raw_tokens"] - base["raw_tokens"],
                "delta_active_tokens": s["active_tokens"] - base["active_tokens"],
                "delta_truncated_tokens": s["truncated_tokens"] - base["truncated_tokens"],
                "delta_candidate_tokens": s["candidate_tokens"] - base["candidate_tokens"],
                "delta_candidate_groups": s["candidate_groups"] - base["candidate_groups"],
                "delta_changed_active_tokens": s["changed_active_tokens"] - base["changed_active_tokens"],
                "delta_changed_source_active_tokens": s["active_segment_tokens_changed"].get("source", 0) - base["active_segment_tokens_changed"].get("source", 0),
                "delta_changed_view_active_tokens": s["active_segment_tokens_changed"].get("view", 0) - base["active_segment_tokens_changed"].get("view", 0),
                "delta_changed_view_wwm_token_mass": s["wwm_candidate_token_mass_by_segment_changed"].get("view", 0) - base["wwm_candidate_token_mass_by_segment_changed"].get("view", 0),
                "delta_pair_both_full_visible_fraction": (s.get("pair_both_full_visible_fraction") or 0.0) - (base.get("pair_both_full_visible_fraction") or 0.0),
                "delta_mean_view_visible_fraction": (s.get("mean_view_visible_fraction") or 0.0) - (base.get("mean_view_visible_fraction") or 0.0),
                "delta_tokens_per_word_active": (s.get("tokens_per_word_active") or 0.0) - (base.get("tokens_per_word_active") or 0.0),
            }
    add_comp("compact", "compact", result["comparisons_vs_compact"])
    add_comp("prefix", "prefix_repeat", result["comparisons_vs_prefix_repeat"])

    # Human-readable scientific summary.
    md = args.out_dir / "mlm_loader_exposure_audit.md"
    lines = ["# research MLM loader exposure audit for research pools", ""]
    lines.append("This is a CPU/static audit of the exact DeBERTa MLM data-loader geometry. It is not training or evaluation.")
    lines.append("")
    lines.append(f"Tokenizer JSON SHA: `{result['tokenizer_json_sha256']}`; seq_len={args.seq_len}.")
    lines.append("")
    lines.append("## Pool-level exposure")
    lines.append("| variant | rows | words | raw tokens | active tokens | tokens/word active | trunc rows | trunc tokens | candidate groups | E[masked tokens] p=.15 | pair full visible | changed active source/view tokens |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    order = ["compact", "prefix_repeat", "spread_even", "content_spread", "scored_source_skeleton"]
    for v in order:
        if v not in result["variants"]:
            continue
        s = result["variants"][v]
        src = s["active_segment_tokens_changed"].get("source", 0)
        view = s["active_segment_tokens_changed"].get("view", 0)
        lines.append(
            f"| {v} | {s['rows']} | {s['words']} | {s['raw_tokens']} | {s['active_tokens']} | {fmt(s['tokens_per_word_active'])} | "
            f"{s['rows_truncated']} ({fmt(s['rows_truncated_fraction'], as_pct=True)}) | {s['truncated_tokens']} | {s['candidate_groups']} | "
            f"{fmt(s['expected_wwm_masked_tokens_at_p015'], 1)} | {fmt(s['pair_both_full_visible_fraction'], as_pct=True)} | {src}/{view} |"
        )
    lines.append("")
    if result["comparisons_vs_compact"]:
        lines.append("## Differences versus compact")
        lines.append("| variant | Δ active tokens | Δ trunc tokens | Δ candidate groups | Δ changed view active tokens | Δ view WWM token mass | Δ pair full-visible fraction |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for v, d in result["comparisons_vs_compact"].items():
            lines.append(f"| {v} | {d['delta_active_tokens']} | {d['delta_truncated_tokens']} | {d['delta_candidate_groups']} | {d['delta_changed_view_active_tokens']} | {d['delta_changed_view_wwm_token_mass']} | {fmt(d['delta_pair_both_full_visible_fraction'], as_pct=True)} |")
        lines.append("")
    lines.append("## Scientific reading")
    lines.append("The current research scaffold fixes legal words and row positions, but it does not automatically fix model exposure. Large differences in active tokens, WWM group mass, truncation, or pair visibility mean the arms should not be treated as a clean factorial training design. A repaired design must match these quantities under this loader before H100 use.")
    if result["issues"]:
        lines.append("")
        lines.append("## Issues")
        for issue in result["issues"]:
            lines.append(f"- {issue}")
    lines.append("")
    lines.append(f"Changed-row CSV: `{per_row_csv}`")
    lines.append(f"Pair visibility CSV: `{pair_csv}`")
    lines.append(f"JSON: `{args.out_dir / 'mlm_loader_exposure_audit.json'}`")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result["elapsed_sec"] = time.time() - t0
    result["out_md"] = str(md)
    result["changed_row_csv"] = str(per_row_csv)
    result["pair_visibility_csv"] = str(pair_csv)
    out_json = args.out_dir / "mlm_loader_exposure_audit.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "out_dir": str(args.out_dir), "variants": list(result["variants"].keys()), "issues": result["issues"], "elapsed_sec": round(result["elapsed_sec"], 2)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
