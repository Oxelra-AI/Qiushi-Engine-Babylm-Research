#!/usr/bin/env python3
"""Prepare and summarize manual quality labels for research selected-pair spotcheck.

Labels intended:
  faithful | minor_change | meaning_change | fragment_format | unsure
The script creates a TSV if absent; if labels are present, it summarizes counts by
cohort/source/bucket.  It does not use an LLM judge; this is for manual
inspection so corpus semantic-quality risk becomes measured rather than anecdotal.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import collections
import csv
import json
import pathlib

ROOT = _public_path('experiments/archive/compact_experience')
DATA = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned')
IN_JSON = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/selected_pairs_stratified_spotcheck.json')
OUT_TSV = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/selected_pairs_quality_labels.tsv')
SUMMARY = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/selected_pairs_quality_label_summary.json')

FIELDS = ["label", "pair_id", "cohort", "source", "bucket", "original_words", "rewrite_words", "len_ratio", "content_overlap", "entity_recall", "original", "rewrite", "notes"]
VALID = {"faithful", "minor_change", "meaning_change", "fragment_format", "unsure", ""}


def load_items():
    obj = json.loads(IN_JSON.read_text(encoding="utf-8"))
    if isinstance(obj, dict):
        return obj.get("sample") or obj.get("items") or obj.get("pairs") or []
    return obj


def create_sheet(items):
    with OUT_TSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t")
        w.writeheader()
        for x in items:
            w.writerow({
                "label": "",
                "pair_id": x.get("pair_id"),
                "cohort": x.get("cohort"),
                "source": x.get("source"),
                "bucket": str(x.get("bucket")),
                "original_words": x.get("original_words") or (x.get("words") or [None, None])[0],
                "rewrite_words": x.get("rewrite_words") or (x.get("words") or [None, None])[1],
                "len_ratio": x.get("len_ratio"),
                "content_overlap": x.get("content_overlap"),
                "entity_recall": x.get("entity_recall"),
                "original": x.get("original"),
                "rewrite": x.get("rewrite"),
                "notes": "",
            })


def summarize():
    rows = []
    with OUT_TSV.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            lab = (r.get("label") or "").strip()
            if lab not in VALID:
                raise SystemExit(f"invalid label {lab!r} for pair {r.get('pair_id')}")
            rows.append(r)
    total = len(rows)
    labeled = [r for r in rows if (r.get("label") or "").strip()]
    counts = collections.Counter(r["label"] for r in labeled)
    by_source = collections.defaultdict(collections.Counter)
    by_cohort = collections.defaultdict(collections.Counter)
    for r in labeled:
        by_source[r.get("source", "")][r["label"]] += 1
        by_cohort[r.get("cohort", "")][r["label"]] += 1
    payload = {
        "status": "PAIR_QUALITY_LABEL_SUMMARY",
        "sheet": str(OUT_TSV),
        "total_rows": total,
        "labeled_rows": len(labeled),
        "label_counts": dict(counts),
        "meaning_or_format_bad_count": counts.get("meaning_change", 0) + counts.get("fragment_format", 0),
        "meaning_or_format_bad_rate_over_labeled": None if not labeled else round((counts.get("meaning_change", 0) + counts.get("fragment_format", 0)) / len(labeled), 4),
        "by_source": {k: dict(v) for k, v in by_source.items()},
        "by_cohort": {k: dict(v) for k, v in by_cohort.items()},
    }
    SUMMARY.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def main():
    items = load_items()
    if not OUT_TSV.exists():
        create_sheet(items)
        print(json.dumps({"status": "created_label_sheet", "rows": len(items), "tsv": str(OUT_TSV)}, indent=2))
    summarize()

if __name__ == "__main__":
    main()
