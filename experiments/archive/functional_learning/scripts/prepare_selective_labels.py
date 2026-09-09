#!/usr/bin/env python3
"""research: extract reviewed selective-compaction semantic labels.

The research independent review verifier returned JSONL-like semantic labels inside a markdown
integration file.  This script extracts those labels and joins them to the 512-row
selective pilot so later learner tests can use individually reviewed rows rather than
unreviewed automatic admission.  It does not estimate corpus prevalence or certify a
promotion rule.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import pathlib
from collections import Counter
from typing import Any, Dict, List

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
DEFAULT_independent_review = _public_path('data/external/independent_review01_verifier1_integration.md')
DEFAULT_JOINED = _public_path('experiments/archive/functional_learning/data/selective_compact_pilot/selective_generation_joined.jsonl')
DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/selective_reviewed_labels')

VALID = {
    "faithful_shortening",
    "supported_partial_view",
    "correspondence_repair",
    "altered_meaning",
    "information_losing_summary",
    "unclear",
}
ALIASES = {
    "supported_summary_with_lost_detail": "information_losing_summary",
    "unclear_or_information_losing": "unclear",
    "unclear_or_unsafe_source": "unclear",
}


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def load_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_labels(path: pathlib.Path) -> List[Dict[str, Any]]:
    labels = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s.startswith("{") or "pair_id" not in s or "label" not in s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            continue
        lab = str(obj.get("label", "")).strip().lower().replace(" ", "_").replace("-", "_")
        lab = ALIASES.get(lab, lab)
        if lab not in VALID:
            lab = "unclear"
        labels.append({
            "pair_id": obj.get("pair_id"),
            "label": lab,
            "reason": obj.get("reason", ""),
            "source": rel(path),
        })
    # Keep last label if duplicate appears in the integration.
    by_id = {r["pair_id"]: r for r in labels if r.get("pair_id")}
    return [by_id[k] for k in sorted(by_id)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--independent_review", type=pathlib.Path, default=DEFAULT_independent_review)
    ap.add_argument("--joined", type=pathlib.Path, default=DEFAULT_JOINED)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    labels = extract_labels(args.independent_review)
    joined = {r["pair_id"]: r for r in load_jsonl(args.joined)}
    enriched = []
    missing = []
    for lab in labels:
        row = joined.get(lab["pair_id"])
        if row is None:
            missing.append(lab["pair_id"])
            continue
        current_words = len(str(row.get("current_rewrite", "")).split())
        compact_words = len(str(row.get("compact_rewrite", "")).split())
        enriched.append({
            **lab,
            "source_corpus": row.get("source", ""),
            "example_id": row.get("example_id"),
            "original": row.get("original", ""),
            "current_rewrite": row.get("current_rewrite", ""),
            "compact_rewrite": row.get("compact_rewrite", ""),
            "current_words": current_words,
            "compact_words": compact_words,
            "saved_words_if_used": max(0, current_words - compact_words),
            "selective_keep_current": str(row.get("compact_rewrite", "")).strip() == "KEEP_CURRENT" or bool(row.get("selective_keep_current", False)),
            "risk_score": row.get("risk_score"),
            "risk_tags": row.get("risk_tags", []),
        })

    labels_path = args.out_dir / "selective_semantic_labels.jsonl"
    with labels_path.open("w", encoding="utf-8") as f:
        for r in enriched:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    counts = Counter(r["label"] for r in enriched)
    by_label_saved = {lab: sum(r["saved_words_if_used"] for r in enriched if r["label"] == lab) for lab in counts}
    summary = {
        "status": "SELECTIVE_LABELS_EXTRACTED",
        "scientific_purpose": "Individually reviewed labels for bounded learner tests; not a corpus prevalence estimate and not automatic admission for unreviewed rows.",
        "independent_review_integration": rel(args.independent_review),
        "joined_selective_pilot": rel(args.joined),
        "n_labels_extracted": len(labels),
        "n_labels_joined": len(enriched),
        "missing_pair_ids": missing,
        "label_counts": dict(counts),
        "saved_words_by_label_on_reviewed_rows": by_label_saved,
        "labels_path": rel(labels_path),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
