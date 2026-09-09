#!/usr/bin/env python3
"""research: filter accepted restatement-dose shards against heldout/probe text screens.

This repairs the research nested dose stream after the primary leakage audit found
that many *selected* source sentences were exact substrings of the ordinary
heldout row slices.  It runs the same exact/subsequence/high-overlap screen over
all accepted shard0/shard1 candidate pairs, writes leakage-hit evidence, and
exports clean accepted pools for re-materialization.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import collections
import csv
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
SHARD1 = STUDY / "data/dose_arm_validated/accepted_dose_arm_pairs_shard1.jsonl"
SHARD0 = STUDY / "data/dose_arm_validated/accepted_dose_arm_pairs_shard0.jsonl"


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    preferred = [
        "pair_id", "shard", "original_id", "query_field", "query_source", "hit_type", "screen", "ref_field", "ref_id",
        "jaccard", "contain_query_in_ref_tokenset", "query_tokens", "ref_tokens", "query_text", "ref_text", "ref_source_path",
    ]
    fields = [f for f in preferred if f in rows[0]] + [f for f in sorted(set().union(*(r.keys() for r in rows))) if f not in preferred]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def load_shard(path: pathlib.Path, shard: str) -> list[dict[str, Any]]:
    rows = audit.read_jsonl(path)
    for r in rows:
        r["_step060_shard"] = shard
    return rows


def make_queries(rows: list[dict[str, Any]]) -> list[audit.QueryText]:
    queries: list[audit.QueryText] = []
    for obj in rows:
        pid = str(obj.get("pair_id"))
        shard = str(obj.get("_step060_shard"))
        for field, key in [("original", "original"), ("rewrite", "rewrite")]:
            text = " ".join(str(obj.get(key, "")).split())
            if not text:
                continue
            ts = audit.tokens(text)
            q = audit.QueryText(
                pair_id=pid,
                dose21=(shard == "shard1"),
                dose25_topup=(shard == "shard0"),
                dose25=True,
                field=field,
                text=text,
                norm=audit.norm_text(text),
                toks=ts,
                tokset=audit.informative_tokens(ts),
                source=str(obj.get("source", "")),
                original_id=str(obj.get("original_id", "")),
            )
            queries.append(q)
    return queries


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--jaccard-threshold", type=float, default=0.80)
    ap.add_argument("--containment-threshold", type=float, default=0.95)
    args = ap.parse_args()

    t0 = time.time()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shard1 = load_shard(SHARD1, "shard1")
    shard0 = load_shard(SHARD0, "shard0")
    all_rows = shard1 + shard0
    id_to_row = {str(r.get("pair_id")): r for r in all_rows}
    id_to_shard = {str(r.get("pair_id")): str(r.get("_step060_shard")) for r in all_rows}
    queries = make_queries(all_rows)
    refs, ref_meta = audit.load_references()
    hits = audit.screen_hits(queries, refs, jaccard_threshold=args.jaccard_threshold, containment_threshold=args.containment_threshold)

    for h in hits:
        h["shard"] = id_to_shard.get(h["pair_id"], "")
    hit_pair_ids = sorted({h["pair_id"] for h in hits})
    hit_by_shard = collections.Counter(id_to_shard.get(pid, "") for pid in hit_pair_ids)
    hit_words_by_shard: dict[str, int] = collections.Counter()
    for pid in hit_pair_ids:
        r = id_to_row[pid]
        hit_words_by_shard[id_to_shard[pid]] += int(r.get("pair_words", 0))

    clean1 = [{k: v for k, v in r.items() if k != "_step060_shard"} for r in shard1 if str(r.get("pair_id")) not in hit_pair_ids]
    clean0 = [{k: v for k, v in r.items() if k != "_step060_shard"} for r in shard0 if str(r.get("pair_id")) not in hit_pair_ids]
    clean1_path = out_dir / "accepted_dose_arm_pairs_shard1_probe_clean.jsonl"
    clean0_path = out_dir / "accepted_dose_arm_pairs_shard0_probe_clean.jsonl"
    write_jsonl(clean1_path, clean1)
    write_jsonl(clean0_path, clean0)

    hits_csv = out_dir / "accepted_shard_probe_leakage_hits.csv"
    hits_jsonl = out_dir / "accepted_shard_probe_leakage_hits.jsonl"
    write_csv(hits_csv, hits)
    write_jsonl(hits_jsonl, hits)
    exclude_path = out_dir / "accepted_shard_blocking_pair_ids_to_exclude.txt"
    exclude_path.write_text("\n".join(hit_pair_ids) + ("\n" if hit_pair_ids else ""), encoding="utf-8")

    def row_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "pairs": len(rows),
            "pair_words": sum(int(r.get("pair_words", 0)) for r in rows),
            "source_counts": dict(collections.Counter(str(r.get("source", "")) for r in rows)),
        }

    summary = {
        "status": "ACCEPTED_SHARDS_PROBE_FILTERED",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": round(time.time() - t0, 2),
        "thresholds": {
            "jaccard_threshold": args.jaccard_threshold,
            "heldout_query_containment_threshold": args.containment_threshold,
        },
        "reference_meta": ref_meta,
        "original_shard_stats": {"shard1": row_stats(shard1), "shard0": row_stats(shard0)},
        "leakage_hit_rows": len(hits),
        "leakage_pair_count": len(hit_pair_ids),
        "leakage_pair_count_by_shard": dict(sorted(hit_by_shard.items())),
        "leakage_pair_words_by_shard": dict(sorted(hit_words_by_shard.items())),
        "leakage_hits_by_screen": dict(sorted(collections.Counter(h["screen"] for h in hits).items())),
        "leakage_hits_by_type": dict(sorted(collections.Counter(h["hit_type"] for h in hits).items())),
        "clean_shard_stats": {"shard1": row_stats(clean1), "shard0": row_stats(clean0)},
        "targets_available_after_filter": {
            "dose21_target_words": 443200,
            "dose25_topup_target_words": 400000,
            "shard1_enough_for_dose21": sum(int(r.get("pair_words", 0)) for r in clean1) >= 443200,
            "shard0_enough_for_topup": sum(int(r.get("pair_words", 0)) for r in clean0) >= 400000,
        },
        "outputs": {
            "clean_shard1": audit.rel(clean1_path),
            "clean_shard0": audit.rel(clean0_path),
            "hits_csv": audit.rel(hits_csv),
            "hits_jsonl": audit.rel(hits_jsonl),
            "exclude_pair_ids": audit.rel(exclude_path),
        },
        "input_sha256": {
            "shard1": audit.sha256_file(SHARD1),
            "shard0": audit.sha256_file(SHARD0),
            "heldout6992": audit.sha256_file(audit.HELDOUT_6992),
            "heldout2647": audit.sha256_file(audit.HELDOUT_2647),
            "wiki_pairs": audit.sha256_file(audit.WIKI_PAIRS),
        },
        "scientific_note": (
            "These clean shard pools remove any accepted pair whose original or rewrite matched ordinary row-holdout, compact-probe, "
            "or WikiLarge-probe text under exact normalized, token-subsequence, high-Jaccard, or heldout high-containment screens. "
            "Use only these clean pools for GPU dose training."
        ),
    }
    (out_dir / "accepted_shard_probe_filter_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
