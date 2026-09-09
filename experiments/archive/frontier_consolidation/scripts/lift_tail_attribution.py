#!/usr/bin/env python3
"""research: attribute research DeBERTa view-lift records to prefix/tail source copy.

This joins the existing reciprocal-view lift records with the compact-pair source
text.  For compact rewrite-side target tokens, copied-token lift can arise from
source tokens that lie inside the repeat arm's first-N prefix or from later source
positions unavailable to the repeat prefix.  The latter supports a source-wide
skeleton-recurrence reading of compact views.

No model is loaded and no new scoring is performed.
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
from typing import Any

ROOT = pathlib.Path("experiments/archive/frontier_consolidation")
DEFAULT_PAIRS = ROOT / "data/density_core_reinvestment_medium_riskhard/selected_compact_reinvest_pairs.jsonl"
DEFAULT_RECORDS = ROOT / "data/reciprocal_view_lift_probe_chck82_128/reciprocal_view_lift_records.jsonl"
DEFAULT_OUT = ROOT / "data/lift_tail_attribution_chck82"
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['\u2019-][A-Za-z0-9]+)?")
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


def norm(s: str) -> str:
    parts = WORD_RE.findall(s)
    if not parts:
        return ""
    return "".join(parts).lower()


def is_content(n: str) -> bool:
    return bool(n) and (n.isdigit() or (len(n) >= 4 and n not in STOPWORDS))


def load_pairs(path: pathlib.Path) -> dict[str, dict[str, Any]]:
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            pid = str(r.get("pair_id"))
            source_words = [w for w in r["source_text"].split() if w]
            rewrite_words = [w for w in r["rewrite_text"].split() if w]
            repeat_len = int(r.get("rewrite_words") or len(rewrite_words))
            norm_positions: dict[str, list[int]] = collections.defaultdict(list)
            for i, w in enumerate(source_words):
                n = norm(w)
                if n:
                    norm_positions[n].append(i)
            out[pid] = {
                "pair_id": pid,
                "source_words": source_words,
                "rewrite_words": rewrite_words,
                "repeat_len": repeat_len,
                "norm_positions": norm_positions,
                "source_len": len(source_words),
                "source_text": r.get("source_text"),
                "rewrite_text": r.get("rewrite_text"),
            }
    return out


def classify_token(pair: dict[str, Any] | None, token_text: str) -> dict[str, Any]:
    n = norm(token_text)
    if not pair or not n:
        return {"norm": n, "copy_zone": "unknown", "is_content": is_content(n), "source_positions": []}
    positions = list(pair["norm_positions"].get(n, []))
    repeat_len = int(pair["repeat_len"])
    has_prefix = any(p < repeat_len for p in positions)
    has_tail = any(p >= repeat_len for p in positions)
    if not positions:
        zone = "not_in_source"
    elif has_tail and not has_prefix:
        zone = "tail_only"
    elif has_tail and has_prefix:
        zone = "both_prefix_and_tail"
    else:
        zone = "prefix_only"
    return {"norm": n, "copy_zone": zone, "is_content": is_content(n), "source_positions": positions}


def safe_mean(xs):
    ys = [float(x) for x in xs if isinstance(x, (int, float)) and math.isfinite(float(x))]
    return sum(ys) / len(ys) if ys else None


def safe_median(xs):
    ys = [float(x) for x in xs if isinstance(x, (int, float)) and math.isfinite(float(x))]
    return statistics.median(ys) if ys else None


def summarize(records: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = collections.defaultdict(list)
    for r in records:
        buckets[tuple(r.get(k) for k in group_keys)].append(r)
    rows = []
    for vals, rs in sorted(buckets.items(), key=lambda kv: str(kv[0])):
        lifts = [r.get("lift_nll_sideonly_minus_paired") for r in rs]
        paired = [r.get("paired_nll") for r in rs]
        side = [r.get("sideonly_nll") for r in rs]
        rows.append({
            "key": dict(zip(group_keys, vals)),
            "n": len(rs),
            "mean_lift": safe_mean(lifts),
            "median_lift": safe_median(lifts),
            "positive_lift_frac": sum(1 for x in lifts if isinstance(x, (int, float)) and math.isfinite(float(x)) and x > 0) / len([x for x in lifts if isinstance(x, (int, float)) and math.isfinite(float(x))]) if any(isinstance(x, (int, float)) and math.isfinite(float(x)) for x in lifts) else None,
            "mean_paired_nll": safe_mean(paired),
            "mean_sideonly_nll": safe_mean(side),
        })
    return rows


def fmt(x: Any) -> str:
    if x is None:
        return ""
    return f"{float(x):.6f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=pathlib.Path, default=DEFAULT_PAIRS)
    ap.add_argument("--records", type=pathlib.Path, default=DEFAULT_RECORDS)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs(args.pairs)
    annotated = []
    with args.records.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            pair = pairs.get(str(r.get("pair_id")))
            c = classify_token(pair, str(r.get("token_text", "")))
            rr = dict(r)
            rr.update({
                "token_norm": c["norm"],
                "token_is_content_norm": c["is_content"],
                "source_copy_zone_by_text": c["copy_zone"],
                "source_positions_by_text": c["source_positions"][:12],
                "source_len": pair.get("source_len") if pair else None,
                "repeat_prefix_words": pair.get("repeat_len") if pair else None,
            })
            annotated.append(rr)

    # Key subsets for the scientific reading.
    compact_other = [r for r in annotated if r.get("pair_type") == "compact" and r.get("target_segment") == "other"]
    compact_source = [r for r in annotated if r.get("pair_type") == "compact" and r.get("target_segment") == "source"]
    content_compact_other = [r for r in compact_other if r.get("token_is_content_norm")]
    content_compact_source = [r for r in compact_source if r.get("token_is_content_norm")]

    groups = {
        "all_by_pair_target_zone": summarize(annotated, ["pair_type", "target_segment", "source_copy_zone_by_text"]),
        "compact_other_content_by_zone": summarize(content_compact_other, ["source_copy_zone_by_text"]),
        "compact_source_content_by_zone": summarize(content_compact_source, ["source_copy_zone_by_text"]),
        "compact_other_by_content_zone": summarize(compact_other, ["token_is_content_norm", "source_copy_zone_by_text"]),
    }
    result = {
        "status": "LIFT_TAIL_ATTRIBUTION",
        "pairs_path": str(args.pairs),
        "pairs_sha256": sha256_file(args.pairs),
        "records_path": str(args.records),
        "records_sha256": sha256_file(args.records),
        "n_records": len(annotated),
        "n_pairs_loaded": len(pairs),
        "subsets": {
            "compact_other": len(compact_other),
            "compact_other_content": len(content_compact_other),
            "compact_source": len(compact_source),
            "compact_source_content": len(content_compact_source),
        },
        "groups": groups,
        "interpretation_notes": [
            "source_copy_zone_by_text uses normalized token text against source whitespace words",
            "tail_only means token text appears in source after the repeated-prefix length and not inside that prefix",
            "both_prefix_and_tail is ambiguous copied content available in both prefix and later source",
            "the records are existing research chck82 model NLLs; this script does not rescore the model",
        ],
    }
    out_json = args.out_dir / "lift_tail_attribution.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_csv = args.out_dir / "lift_tail_annotated_records.csv"
    fieldnames = [
        "model_type", "model_label", "pair_id", "pair_type", "target_segment", "token_text", "token_norm",
        "token_is_content_norm", "copy_by_text", "source_copy_zone_by_text", "source_positions_by_text",
        "source_len", "repeat_prefix_words", "paired_nll", "sideonly_nll", "lift_nll_sideonly_minus_paired",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in annotated:
            row = {k: r.get(k) for k in fieldnames}
            row["source_positions_by_text"] = json.dumps(row["source_positions_by_text"])
            w.writerow(row)

    lines = ["# research lift tail-attribution", ""]
    lines.append("Existing research chck82 DeBERTa lift records joined to source-prefix/tail positions; no new scoring.")
    lines.append("")
    lines.append(f"Records: `{args.records}` SHA `{result['records_sha256']}`")
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
    lines.append("If compact rewrite-side lift for content tokens is substantial in `tail_only` or `both_prefix_and_tail` strata, then the copy-lift is not simply the first-N repeat prefix; compact views reintroduce source-wide content keys that the matched repeat prefix would not expose. If lift remains concentrated in `prefix_only`, the skeleton view is less distinct from exact recurrence. Counts are small because research sampled 128 pairs and at most a few targets per side, so this is attribution of existing probe evidence rather than final proof.")
    lines.append("")
    lines.append(f"Annotated records CSV: `{out_csv}`")
    lines.append(f"JSON: `{out_json}`")
    out_md = args.out_dir / "lift_tail_attribution.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "out_dir": str(args.out_dir), "n_records": len(annotated)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
