#!/usr/bin/env python3
"""research: audit current base/dose streams against the research 2,647-row heldout set.

The 6,992-row set belongs to later research rowholdout pools and turned out not
to be truly held out from the compact-view-reinvest substrate.  The research
compact-view-reinvest stream was constructed with its own 2,647-row holdout.
This script checks whether those rows are absent from the current base and
repaired dose streams, and builds a pair-text-clean subset after removing rows
whose text appears as inherited COMPACT_EXPERIENCE ALN source/rewrite material.
"""
from __future__ import annotations

import collections
import csv
import json
import pathlib
import re
import time
from typing import Any

ROOT = pathlib.Path.cwd()
STUDY = ROOT / "experiments/archive/relation_learning"
OUT = STUDY / "data/current_dose_stream_heldout2647_exposure"
HELDOUT = ROOT / "experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/heldout_cleanqwen_rows.jsonl"
INHERITED_HITS = STUDY / "data/inherited_aln_leakage_audit/inherited_aln_blocking_hits.jsonl"
STREAMS = {
    "base_compact_view_reinvest_10M": ROOT / "experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl",
    "base_compact_view_reinvest_100M": ROOT / "experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl",
    "dose21_probe_clean_100M": STUDY / "data/probe_clean_nested_dose_streams/dose21/dose21_compact_view_reinvest_100M.jsonl",
    "dose25_probe_clean_100M": STUDY / "data/probe_clean_nested_dose_streams/dose25/dose25_compact_view_reinvest_100M.jsonl",
}
ROWS_PER_PASS = 64740
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[\u2019'][A-Za-z0-9]+)?")


def norm_text(s: str) -> str:
    return " ".join(m.group(0).lower().replace("\u2019", "'") for m in WORD_RE.finditer(str(s)))


def read_jsonl(path: pathlib.Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def parent_ref_id(ref_id: str) -> str:
    return ref_id.split(":chunk:", 1)[0] if ":chunk:" in ref_id else ref_id


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
    fields = list(rows[0].keys())
    extra = sorted(set().union(*(r.keys() for r in rows)) - set(fields))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields + extra)
        w.writeheader(); w.writerows(rows)


def load_heldout_and_pair_blocked() -> tuple[list[dict[str, Any]], set[str]]:
    rows = []
    for obj in read_jsonl(HELDOUT):
        rid = f"heldout2647:row:{obj.get('source')}:{obj.get('example_id')}"
        text = " ".join(str(obj.get("text", "")).split())
        rows.append({
            "heldout_row_id": rid,
            "source": str(obj.get("source", "")),
            "example_id": int(obj.get("example_id")),
            "words": int(obj.get("words", len(text.split()))),
            "text": text,
            "norm": norm_text(text),
            "raw": obj,
        })
    blocked = set()
    if INHERITED_HITS.exists():
        for h in read_jsonl(INHERITED_HITS):
            if h.get("screen") == "heldout2647":
                blocked.add(parent_ref_id(str(h.get("ref_id", ""))))
    return rows, blocked


def scan_stream(name: str, path: pathlib.Path, heldout_rows: list[dict[str, Any]], pair_blocked: set[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_key = {(r["source"], r["example_id"]): r for r in heldout_rows}
    by_norm = {r["norm"]: r for r in heldout_rows}
    hits = []
    source_counts = collections.Counter()
    source_words = collections.Counter()
    rows_scanned = 0
    total_words = 0
    for row_index, obj in enumerate(read_jsonl(path)):
        rows_scanned += 1
        src = str(obj.get("source", ""))
        words = int(obj.get("words", len(str(obj.get("text", "")).split())))
        source_counts[src] += 1
        source_words[src] += words
        total_words += words
        try:
            exid = int(obj.get("example_id"))
        except Exception:
            exid = None
        text = " ".join(str(obj.get("text", "")).split())
        nt = norm_text(text)
        matched: list[tuple[str, dict[str, Any]]] = []
        if exid is not None and (src, exid) in by_key:
            matched.append(("source_example_id", by_key[(src, exid)]))
        if nt in by_norm and (not matched or by_norm[nt]["heldout_row_id"] not in {m[1]["heldout_row_id"] for m in matched}):
            matched.append(("norm_text", by_norm[nt]))
        for hit_type, h in matched:
            hits.append({
                "stream": name,
                "stream_path": rel(path),
                "stream_row_index": row_index,
                "pass_index": row_index // ROWS_PER_PASS if rows_scanned > ROWS_PER_PASS else 0,
                "within_pass_row_index": row_index % ROWS_PER_PASS,
                "stream_source": src,
                "stream_example_id": exid,
                "stream_words": words,
                "hit_type": hit_type,
                "heldout_row_id": h["heldout_row_id"],
                "heldout_source": h["source"],
                "heldout_example_id": h["example_id"],
                "heldout_words": h["words"],
                "inherited_aln_pair_text_hit_row": h["heldout_row_id"] in pair_blocked,
                "norm_text_exact": nt == h["norm"],
                "text_preview": text[:300],
            })
    unique = {h["heldout_row_id"] for h in hits}
    meta = {
        "stream": name,
        "stream_path": rel(path),
        "rows_scanned": rows_scanned,
        "total_words": total_words,
        "hit_rows": len(hits),
        "unique_heldout_rows_hit": len(unique),
        "unique_heldout_rows_hit_without_inherited_pair_text_hit": len({h["heldout_row_id"] for h in hits if not h["inherited_aln_pair_text_hit_row"]}),
        "hit_rows_by_type": dict(collections.Counter(h["hit_type"] for h in hits)),
        "hit_rows_by_stream_source": dict(collections.Counter(h["stream_source"] for h in hits)),
        "hit_rows_by_pass_index": dict(sorted((str(k), v) for k, v in collections.Counter(h["pass_index"] for h in hits).items())),
        "top_source_row_counts": dict(source_counts.most_common(12)),
        "top_source_word_counts": dict(source_words.most_common(12)),
    }
    return hits, meta


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, pair_blocked = load_heldout_and_pair_blocked()
    all_hits = []
    metas = []
    for name, path in STREAMS.items():
        hits, meta = scan_stream(name, path, rows, pair_blocked)
        write_csv(OUT / f"{name}_heldout2647_identity_hits.csv", hits)
        all_hits.extend(hits)
        metas.append(meta)
    write_csv(OUT / "all_stream_heldout2647_identity_hits.csv", all_hits)
    stream_hit_ids = {h["heldout_row_id"] for h in all_hits}
    clean_pair = [r["raw"] for r in rows if r["heldout_row_id"] not in pair_blocked]
    clean_union = [r["raw"] for r in rows if r["heldout_row_id"] not in pair_blocked and r["heldout_row_id"] not in stream_hit_ids]
    blocked_pair = [r["raw"] for r in rows if r["heldout_row_id"] in pair_blocked]
    write_jsonl(OUT / "heldout2647_no_inherited_aln_pair_text_hits.jsonl", clean_pair)
    write_jsonl(OUT / "heldout2647_inherited_aln_pair_text_hit_rows.jsonl", blocked_pair)
    write_jsonl(OUT / "heldout2647_no_pair_text_or_stream_identity_hits.jsonl", clean_union)
    summary = {
        "status": "CURRENT_DOSE_STREAM_HELDOUT2647_EXPOSURE_AUDIT",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": round(time.time() - t0, 2),
        "heldout_rows": len(rows),
        "inherited_aln_pair_text_hit_rows": len(pair_blocked),
        "per_stream": metas,
        "clean_subsets": {
            "no_inherited_aln_pair_text_hits_rows": len(clean_pair),
            "no_pair_text_or_stream_identity_hits_rows": len(clean_union),
            "no_inherited_aln_pair_text_hits_path": rel(OUT / "heldout2647_no_inherited_aln_pair_text_hits.jsonl"),
            "no_pair_text_or_stream_identity_hits_path": rel(OUT / "heldout2647_no_pair_text_or_stream_identity_hits.jsonl"),
            "blocked_pair_text_hit_rows_path": rel(OUT / "heldout2647_inherited_aln_pair_text_hit_rows.jsonl"),
        },
        "set_relations": {
            "unique_stream_identity_hit_rows_union": len(stream_hit_ids),
            "stream_hit_intersection_pair_blocked": len(stream_hit_ids & pair_blocked),
            "pair_blocked_without_stream_identity_hit": len(pair_blocked - stream_hit_ids),
        },
        "inputs": {"heldout": rel(HELDOUT), "inherited_hits": rel(INHERITED_HITS), "streams": {k: rel(v) for k, v in STREAMS.items()}},
        "outputs": {"all_hits_csv": rel(OUT / "all_stream_heldout2647_identity_hits.csv")},
        "scientific_interpretation": "Rows in the research 2,647 heldout set are the natural heldout ordinate for compact-view-reinvest base/dose streams if stream identity hits are zero. Rows whose text appears as inherited ALN source/rewrite should be removed for a pair-text-clean ordinary fit readout.",
    }
    (OUT / "audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
