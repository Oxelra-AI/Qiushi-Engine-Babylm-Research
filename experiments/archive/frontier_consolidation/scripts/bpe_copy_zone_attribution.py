#!/usr/bin/env python3
"""research: tokenizer-level source-zone attribution for research view-lift records.

The earlier text skeleton measurement is source-word geometry.  This script works
at the actual tokenizer granularity used by the DeBERTa checkpoints: tokenize the
source with offsets, map source BPE tokens to whitespace-word positions, and ask
whether a lifted target token is available in the repeated prefix or only in the
source tail that compact views can selectively re-expose.

No model is loaded and no scoring is performed.  Existing research NLL records are
only re-annotated.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import os
import pathlib
import re
import statistics
from typing import Any

ROOT = pathlib.Path("experiments/archive/frontier_consolidation")
DEFAULT_PAIRS = ROOT / "data/density_core_reinvestment_medium_riskhard/selected_compact_reinvest_pairs.jsonl"
DEFAULT_CHCK = ROOT / "training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M"
DEFAULT_RECORDS = ROOT / "data/reciprocal_view_lift_probe_chck82_128/reciprocal_view_lift_records.jsonl"
DEFAULT_OUT = ROOT / "data/bpe_copy_zone_attribution_chck82"
WORD_SPAN_RE = re.compile(r"\S+")
NORM_RE = re.compile(r"[A-Za-z0-9]+")
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "so", "because", "as", "than",
    "to", "of", "in", "on", "for", "with", "without", "by", "from", "at", "into", "onto", "over",
    "under", "about", "between", "among", "through", "during", "before", "after", "above", "below",
    "is", "am", "are", "was", "were", "be", "been", "being", "do", "does", "did", "done", "doing",
    "have", "has", "had", "having", "can", "could", "may", "might", "must", "shall", "should", "will",
    "would", "this", "that", "these", "those", "there", "here", "it", "its", "they", "them", "their",
    "he", "him", "his", "she", "her", "we", "us", "our", "you", "your", "i", "me", "my", "who",
    "which", "what", "where", "when", "why", "how", "not", "no", "only", "just", "also", "very",
    "more", "most", "less", "many", "some", "any", "all", "each", "every", "other", "another", "same",
}


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_piece(s: str) -> str:
    return "".join(NORM_RE.findall(s)).lower()


def is_content_norm(n: str) -> bool:
    return bool(n) and (n.isdigit() or (len(n) >= 4 and n not in STOPWORDS))


def word_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in WORD_SPAN_RE.finditer(text)]


def overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def offset_to_word_index(off: tuple[int, int], spans: list[tuple[int, int]]) -> int | None:
    s, e = int(off[0]), int(off[1])
    if e <= s:
        return None
    best_i = None
    best_ov = 0
    for i, (ws, we) in enumerate(spans):
        ov = overlap(s, e, ws, we)
        if ov > best_ov:
            best_ov = ov
            best_i = i
        if ws > e:
            break
    return best_i


def zone_from_positions(positions: list[int], repeat_len: int) -> str:
    if not positions:
        return "not_in_source"
    has_prefix = any(p < repeat_len for p in positions)
    has_tail = any(p >= repeat_len for p in positions)
    if has_prefix and has_tail:
        return "both_prefix_and_tail"
    if has_tail:
        return "tail_only"
    return "prefix_only"


def load_pairs(path: pathlib.Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            pid = str(r.get("pair_id"))
            if pid and r.get("source_text") and r.get("rewrite_text"):
                out[pid] = r
    return out


def build_source_token_maps(tokenizer, pairs: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    maps: dict[str, dict[str, Any]] = {}
    offset_supported = True
    for pid, r in pairs.items():
        source = r["source_text"]
        spans = word_spans(source)
        repeat_len = int(r.get("rewrite_words") or len(str(r.get("rewrite_text", "")).split()))
        try:
            enc = tokenizer(source, add_special_tokens=False, return_offsets_mapping=True)
            ids = [int(x) for x in enc["input_ids"]]
            offsets = enc["offset_mapping"]
        except Exception:
            offset_supported = False
            ids = [int(x) for x in tokenizer.encode(source, add_special_tokens=False)]
            offsets = [(0, 0)] * len(ids)
        id_positions: dict[int, list[int]] = collections.defaultdict(list)
        norm_positions: dict[str, list[int]] = collections.defaultdict(list)
        tok_rows = []
        for tid, off in zip(ids, offsets):
            wi = offset_to_word_index(tuple(off), spans)
            try:
                ttxt = tokenizer.decode([int(tid)], clean_up_tokenization_spaces=False)
            except Exception:
                ttxt = str(tid)
            tn = norm_piece(ttxt)
            if wi is not None:
                id_positions[int(tid)].append(int(wi))
                if tn:
                    norm_positions[tn].append(int(wi))
            tok_rows.append({"id": int(tid), "norm": tn, "offset": list(off), "word_i": wi})
        maps[pid] = {
            "repeat_len": repeat_len,
            "source_word_count": len(spans),
            "source_token_count": len(ids),
            "id_positions": {str(k): v for k, v in id_positions.items()},
            "norm_positions": dict(norm_positions),
            "offset_supported": offset_supported,
            "sample_tokens": tok_rows[:20],
        }
    return maps


def safe_mean(xs):
    ys = [float(x) for x in xs if isinstance(x, (int, float)) and math.isfinite(float(x))]
    return sum(ys) / len(ys) if ys else None


def safe_median(xs):
    ys = [float(x) for x in xs if isinstance(x, (int, float)) and math.isfinite(float(x))]
    return statistics.median(ys) if ys else None


def summarize(records: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = collections.defaultdict(list)
    for r in records:
        buckets[tuple(r.get(k) for k in keys)].append(r)
    rows = []
    for vals, rs in sorted(buckets.items(), key=lambda kv: str(kv[0])):
        lifts = [r.get("lift_nll_sideonly_minus_paired") for r in rs]
        finite_lifts = [float(x) for x in lifts if isinstance(x, (int, float)) and math.isfinite(float(x))]
        rows.append({
            "key": dict(zip(keys, vals)),
            "n": len(rs),
            "mean_lift": safe_mean(finite_lifts),
            "median_lift": safe_median(finite_lifts),
            "positive_lift_frac": sum(1 for x in finite_lifts if x > 0) / len(finite_lifts) if finite_lifts else None,
            "mean_paired_nll": safe_mean([r.get("paired_nll") for r in rs]),
            "mean_sideonly_nll": safe_mean([r.get("sideonly_nll") for r in rs]),
        })
    return rows


def fmt(x):
    return "" if x is None else f"{float(x):.6f}"


def annotate_records(records_path: pathlib.Path, maps: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    annotated = []
    with records_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            pid = str(r.get("pair_id"))
            m = maps.get(pid)
            tn = norm_piece(str(r.get("token_text", "")))
            zone_id = "unknown"
            zone_norm = "unknown"
            pos_id: list[int] = []
            pos_norm: list[int] = []
            repeat_len = None
            if m:
                repeat_len = int(m["repeat_len"])
                pos_id = [int(x) for x in m["id_positions"].get(str(int(r.get("orig_id"))), [])]
                pos_norm = [int(x) for x in m["norm_positions"].get(tn, [])] if tn else []
                zone_id = zone_from_positions(pos_id, repeat_len)
                zone_norm = zone_from_positions(pos_norm, repeat_len)
            rr = dict(r)
            rr.update({
                "token_norm": tn,
                "token_is_content_norm": is_content_norm(tn),
                "source_bpe_zone_by_id": zone_id,
                "source_bpe_zone_by_norm": zone_norm,
                "source_bpe_positions_by_id": pos_id[:12],
                "source_bpe_positions_by_norm": pos_norm[:12],
                "repeat_prefix_words": repeat_len,
                "source_word_count": m.get("source_word_count") if m else None,
                "source_token_count": m.get("source_token_count") if m else None,
            })
            annotated.append(rr)
    return annotated


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=pathlib.Path, default=DEFAULT_PAIRS)
    ap.add_argument("--checkpoint", type=pathlib.Path, default=DEFAULT_CHCK)
    ap.add_argument("--records", type=pathlib.Path, default=DEFAULT_RECORDS)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(args.out_dir / ".hf_cache")
    os.environ["TRANSFORMERS_CACHE"] = str(args.out_dir / ".hf_cache")
    os.environ["HF_MODULES_CACHE"] = str(args.out_dir / ".hf_cache/modules")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.checkpoint), trust_remote_code=True)
    pairs = load_pairs(args.pairs)
    maps = build_source_token_maps(tokenizer, pairs)
    annotated = annotate_records(args.records, maps)

    compact_other = [r for r in annotated if r.get("pair_type") == "compact" and r.get("target_segment") == "other"]
    compact_other_content = [r for r in compact_other if r.get("token_is_content_norm")]
    compact_other_copy_id = [r for r in compact_other if r.get("copy_by_id")]
    compact_other_copy_text = [r for r in compact_other if r.get("copy_by_text")]
    groups = {
        "all_by_pair_target_idzone": summarize(annotated, ["pair_type", "target_segment", "source_bpe_zone_by_id"]),
        "compact_other_by_idzone": summarize(compact_other, ["source_bpe_zone_by_id"]),
        "compact_other_content_by_idzone": summarize(compact_other_content, ["source_bpe_zone_by_id"]),
        "compact_other_step178_copy_by_idzone": summarize(compact_other_copy_id, ["source_bpe_zone_by_id"]),
        "compact_other_step178_copy_text_by_normzone": summarize(compact_other_copy_text, ["source_bpe_zone_by_norm"]),
        "compact_other_by_content_idzone": summarize(compact_other, ["token_is_content_norm", "source_bpe_zone_by_id"]),
    }
    result = {
        "status": "BPE_COPY_ZONE_ATTRIBUTION",
        "pairs_path": str(args.pairs),
        "pairs_sha256": sha256_file(args.pairs),
        "checkpoint_tokenizer_path": str(args.checkpoint),
        "records_path": str(args.records),
        "records_sha256": sha256_file(args.records),
        "n_pairs": len(pairs),
        "n_records": len(annotated),
        "subsets": {
            "compact_other": len(compact_other),
            "compact_other_content": len(compact_other_content),
            "compact_other_step178_copy_id": len(compact_other_copy_id),
            "compact_other_step178_copy_text": len(compact_other_copy_text),
        },
        "groups": groups,
        "notes": [
            "source_bpe_zone_by_id maps target orig_id to source tokenizer occurrences with offset-derived word positions",
            "tail_only means all same-id source BPE occurrences are after the repeat prefix length",
            "both_prefix_and_tail means the same BPE token occurs on both sides of the prefix cut",
            "norm-zone is looser and mainly used for research copy_by_text records",
            "no model is loaded; existing research NLL records are re-annotated only",
        ],
    }
    out_json = args.out_dir / "bpe_copy_zone_attribution.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_csv = args.out_dir / "bpe_copy_zone_annotated_records.csv"
    fieldnames = [
        "model_type", "model_label", "pair_id", "pair_type", "target_segment", "token_text", "token_norm",
        "token_is_content_norm", "orig_id", "copy_by_id", "copy_by_text", "source_bpe_zone_by_id",
        "source_bpe_zone_by_norm", "source_bpe_positions_by_id", "source_bpe_positions_by_norm",
        "repeat_prefix_words", "source_word_count", "source_token_count", "paired_nll", "sideonly_nll",
        "lift_nll_sideonly_minus_paired",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in annotated:
            row = {k: r.get(k) for k in fieldnames}
            row["source_bpe_positions_by_id"] = json.dumps(row["source_bpe_positions_by_id"])
            row["source_bpe_positions_by_norm"] = json.dumps(row["source_bpe_positions_by_norm"])
            w.writerow(row)
    out_md = args.out_dir / "bpe_copy_zone_attribution.md"
    lines = ["# research BPE copy-zone attribution", ""]
    lines.append("Tokenizer-offset re-annotation of existing research lift records; no model scoring.")
    lines.append("")
    lines.append(f"Records: `{args.records}` SHA `{result['records_sha256']}`")
    lines.append(f"Tokenizer checkpoint: `{args.checkpoint}`")
    lines.append("")
    for name, rows in groups.items():
        lines.append(f"## {name}")
        lines.append("| key | n | mean lift | median lift | positive frac | mean paired NLL | mean side-only NLL |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for row in rows:
            key = ", ".join(f"{k}={v}" for k, v in row["key"].items())
            lines.append(f"| {key} | {row['n']} | {fmt(row.get('mean_lift'))} | {fmt(row.get('median_lift'))} | {fmt(row.get('positive_lift_frac'))} | {fmt(row.get('mean_paired_nll'))} | {fmt(row.get('mean_sideonly_nll'))} |")
        lines.append("")
    lines.append("## Scientific reading")
    lines.append("The key rows are compact rewrite/other targets. A high mean lift in `tail_only` or `both_prefix_and_tail` same-id strata means the compact view makes the model exploit source BPE tokens beyond the matched repeat prefix, strengthening the source-wide skeleton recurrence interpretation. If high lift were confined to `prefix_only`, copied-token lift would be less distinct from exact prefix recurrence. This remains a small attribution of a 128-pair probe, not official competence evidence.")
    lines.append("")
    lines.append(f"Annotated CSV: `{out_csv}`")
    lines.append(f"JSON: `{out_json}`")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "out_dir": str(args.out_dir), "n_records": len(annotated)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
