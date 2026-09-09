#!/usr/bin/env python3
"""research: verify actual probe-clean dose selections have zero heldout/probe text hits.

This is the post-repair counterpart to audit_dose_probe_leakage.py.  It
audits the actual selected pair files emitted by
materialize_probe_clean_nested_doses.py and appends the result to the
probe-clean stream metadata.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import collections
import json
import pathlib
import sys
import time
from typing import Any

SCRIPT_DIR = _public_path('experiments/archive/relation_learning/scripts')
sys.path.insert(0, str(SCRIPT_DIR))
import audit_dose_probe_leakage as audit  # noqa: E402

ROOT = pathlib.Path.cwd()
STUDY = ROOT / "experiments/archive/relation_learning"
OUT_DEFAULT = STUDY / "data/dose_leakage_audit"
META = STUDY / "data/probe_clean_nested_dose_streams/probe_clean_nested_dose_materialization_metadata.json"
DOSE21 = STUDY / "data/probe_clean_nested_dose_streams/dose21_selected_pairs_probe_clean.jsonl"
DOSE25_EXTRA = STUDY / "data/probe_clean_nested_dose_streams/dose25_extra_selected_pairs_probe_clean_from_shard0.jsonl"
DOSE25_SUPER = STUDY / "data/probe_clean_nested_dose_streams/dose25_selected_pairs_probe_clean_superset.jsonl"


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def make_queries(rows: list[dict[str, Any]], dose21_ids: set[str], extra_ids: set[str]) -> list[audit.QueryText]:
    queries: list[audit.QueryText] = []
    for obj in rows:
        pid = str(obj.get("pair_id"))
        for field, key in [("original", "original"), ("rewrite", "rewrite")]:
            text = " ".join(str(obj.get(key, "")).split())
            if not text:
                continue
            ts = audit.tokens(text)
            queries.append(audit.QueryText(
                pair_id=pid,
                dose21=pid in dose21_ids,
                dose25_topup=pid in extra_ids,
                dose25=True,
                field=field,
                text=text,
                norm=audit.norm_text(text),
                toks=ts,
                tokset=audit.informative_tokens(ts),
                source=str(obj.get("source", "")),
                original_id=str(obj.get("original_id", "")),
            ))
    return queries


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--metadata", default=str(META))
    ap.add_argument("--jaccard-threshold", type=float, default=0.80)
    ap.add_argument("--containment-threshold", type=float, default=0.95)
    ap.add_argument("--update-metadata", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dose21_rows = audit.read_jsonl(DOSE21)
    extra_rows = audit.read_jsonl(DOSE25_EXTRA)
    super_rows = audit.read_jsonl(DOSE25_SUPER)
    dose21_ids = {str(r.get("pair_id")) for r in dose21_rows}
    extra_ids = {str(r.get("pair_id")) for r in extra_rows}
    queries = make_queries(super_rows, dose21_ids, extra_ids)
    refs, ref_meta = audit.load_references()
    hits = audit.screen_hits(queries, refs, jaccard_threshold=args.jaccard_threshold, containment_threshold=args.containment_threshold)
    by_pair = collections.defaultdict(list)
    for h in hits:
        by_pair[h["pair_id"]].append(h)
    hits_path = out_dir / "probe_clean_selected_leakage_hits.jsonl"
    write_jsonl(hits_path, hits)
    summary = {
        "status": "PROBE_CLEAN_SELECTED_LEAKAGE_AUDIT",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": round(time.time() - t0, 2),
        "selected_counts": {
            "dose21_pairs": len(dose21_rows),
            "dose25_extra_pairs": len(extra_rows),
            "dose25_superset_pairs": len(super_rows),
            "selected_texts_audited": len(queries),
            "dose21_pair_words": sum(int(r.get("pair_words", 0)) for r in dose21_rows),
            "dose25_extra_pair_words": sum(int(r.get("pair_words", 0)) for r in extra_rows),
            "dose25_superset_pair_words": sum(int(r.get("pair_words", 0)) for r in super_rows),
        },
        "reference_meta": ref_meta,
        "thresholds": {
            "jaccard_threshold": args.jaccard_threshold,
            "heldout_query_containment_threshold": args.containment_threshold,
        },
        "blocking_hit_count": len(hits),
        "blocking_pair_count": len(by_pair),
        "blocking_hits_by_screen": dict(sorted(collections.Counter(h["screen"] for h in hits).items())),
        "blocking_hits_by_type": dict(sorted(collections.Counter(h["hit_type"] for h in hits).items())),
        "outputs": {"hits_jsonl": audit.rel(hits_path)},
        "inputs": {
            "dose21_selected": audit.rel(DOSE21),
            "dose25_extra": audit.rel(DOSE25_EXTRA),
            "dose25_superset": audit.rel(DOSE25_SUPER),
        },
        "input_sha256": {
            "dose21_selected": audit.sha256_file(DOSE21),
            "dose25_extra": audit.sha256_file(DOSE25_EXTRA),
            "dose25_superset": audit.sha256_file(DOSE25_SUPER),
        },
        "scientific_interpretation": "Zero hits means the actual probe-clean selected source/rewrite texts do not overlap the ordinary heldout rows, compact probe pairs, or WikiLarge probe sentences under the same primary leakage screen used before training.",
    }
    summary_path = out_dir / "probe_clean_selected_leakage_audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.update_metadata:
        meta_path = pathlib.Path(args.metadata)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["probe_clean_selected_leakage_audit"] = summary
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        summary["metadata_updated"] = audit.rel(meta_path)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
