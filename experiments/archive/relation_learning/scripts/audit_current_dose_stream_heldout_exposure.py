#!/usr/bin/env python3
"""research: audit heldout-row exposure in the current base and repaired dose streams.

The inherited ALN pair-text audit tells whether heldout rows contain a source or
rewrite from the inherited qwen_pair_packed block.  A separate question is
whether the ordinary 160-word heldout rows themselves are present as ordinary
training rows in the compact-view-reinvest base or in the repaired up-dose
streams.  This script scans stream row identities and normalized text equality
against the 6,992-row primary heldout set, writes per-stream hit tables, and
materializes clean subsets for downstream ordinary-heldout loss scoring.
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
OUT = STUDY / "data/current_dose_stream_heldout_exposure"
HELDOUT = ROOT / "experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/heldout_cleanqwen_rows.jsonl"
PAIR_BLOCKED = STUDY / "data/inherited_aln_leakage_audit/heldout6992_inherited_aln_pair_text_hit_rows.jsonl"

STREAMS = {
    "base_compact_view_reinvest_10M": ROOT / "experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl",
    "base_compact_view_reinvest_100M": ROOT / "experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl",
    "dose21_probe_clean_100M": STUDY / "data/probe_clean_nested_dose_streams/dose21/dose21_compact_view_reinvest_100M.jsonl",
    "dose25_probe_clean_100M": STUDY / "data/probe_clean_nested_dose_streams/dose25/dose25_compact_view_reinvest_100M.jsonl",
}
ROWS_PER_PASS = 64740
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[\u2019'][A-Za-z0-9]+)?")
PAIR_LIKE_SOURCES = {"qwen_pair_packed", "dose21_qwen_pair_packed", "dose25_extra_qwen_pair_packed", "cleanqwen_fineweb_compact_view_reinvest"}


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


def load_heldout() -> tuple[list[dict[str, Any]], set[str]]:
    rows = []
    for obj in read_jsonl(HELDOUT):
        rid = f"heldout6992:row:{obj.get('source')}:{obj.get('example_id')}"
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
    if PAIR_BLOCKED.exists():
        for obj in read_jsonl(PAIR_BLOCKED):
            rid = f"heldout6992:row:{obj.get('source')}:{obj.get('example_id')}"
            blocked.add(rid)
    return rows, blocked


def scan_stream(name: str, path: pathlib.Path, heldout_rows: list[dict[str, Any]], pair_blocked_ids: set[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_key = {(r["source"], r["example_id"]): r for r in heldout_rows}
    by_norm = {r["norm"]: r for r in heldout_rows}
    hits: list[dict[str, Any]] = []
    source_counts = collections.Counter()
    source_words = collections.Counter()
    rows_scanned = 0
    total_words = 0
    for row_index, obj in enumerate(read_jsonl(path)):
        rows_scanned += 1
        src = str(obj.get("source", ""))
        words = int(obj.get("words", len(str(obj.get("text", "")).split())))
        total_words += words
        source_counts[src] += 1
        source_words[src] += words
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
            rid = h["heldout_row_id"]
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
                "heldout_row_id": rid,
                "heldout_source": h["source"],
                "heldout_example_id": h["example_id"],
                "heldout_words": h["words"],
                "norm_text_exact": nt == h["norm"],
                "pair_like_or_changed_source": src in PAIR_LIKE_SOURCES or src.startswith("qwen_pair_packed") or src.startswith("dose"),
                "inherited_aln_pair_text_hit_row": rid in pair_blocked_ids,
                "text_preview": text[:300],
            })
    unique = {h["heldout_row_id"] for h in hits}
    by_hit_type = collections.Counter(h["hit_type"] for h in hits)
    by_source = collections.Counter(h["stream_source"] for h in hits)
    by_pass = collections.Counter(h["pass_index"] for h in hits)
    unique_not_pair_blocked = {h["heldout_row_id"] for h in hits if not h["inherited_aln_pair_text_hit_row"]}
    meta = {
        "stream": name,
        "stream_path": rel(path),
        "rows_scanned": rows_scanned,
        "total_words": total_words,
        "hit_rows": len(hits),
        "unique_heldout_rows_hit": len(unique),
        "unique_heldout_rows_hit_without_inherited_pair_text_hit": len(unique_not_pair_blocked),
        "hit_rows_by_type": dict(sorted(by_hit_type.items())),
        "hit_rows_by_stream_source": dict(sorted(by_source.items())),
        "hit_rows_by_pass_index": dict(sorted((str(k), v) for k, v in by_pass.items())),
        "top_source_row_counts": dict(source_counts.most_common(12)),
        "top_source_word_counts": dict(source_words.most_common(12)),
    }
    return hits, meta


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    heldout_rows, pair_blocked_ids = load_heldout()
    all_hits: list[dict[str, Any]] = []
    metas = []
    for name, path in STREAMS.items():
        hits, meta = scan_stream(name, path, heldout_rows, pair_blocked_ids)
        write_csv(OUT / f"{name}_heldout_identity_hits.csv", hits)
        all_hits.extend(hits)
        metas.append(meta)
    write_csv(OUT / "all_stream_heldout_identity_hits.csv", all_hits)

    hit_union_all = {h["heldout_row_id"] for h in all_hits}
    hit_union_base_dose_100m = {h["heldout_row_id"] for h in all_hits if h["stream"] != "base_compact_view_reinvest_10M"}
    hit_union_stream_or_pair = hit_union_base_dose_100m | pair_blocked_ids
    clean_any = [r["raw"] for r in heldout_rows if r["heldout_row_id"] not in hit_union_stream_or_pair]
    blocked_any = [r["raw"] for r in heldout_rows if r["heldout_row_id"] in hit_union_stream_or_pair]
    clean_pair_only = [r["raw"] for r in heldout_rows if r["heldout_row_id"] not in pair_blocked_ids]
    write_jsonl(OUT / "heldout6992_no_inherited_pair_or_base_dose_stream_hits.jsonl", clean_any)
    write_jsonl(OUT / "heldout6992_inherited_pair_or_base_dose_stream_hit_rows.jsonl", blocked_any)
    write_jsonl(OUT / "heldout6992_no_inherited_pair_text_hits_copy.jsonl", clean_pair_only)

    stream_sets = {m["stream"]: {h["heldout_row_id"] for h in all_hits if h["stream"] == m["stream"]} for m in metas}
    pair_clean_ids = {r["heldout_row_id"] for r in heldout_rows if r["heldout_row_id"] not in pair_blocked_ids}
    clean_stream_counts = {name: len(ids & pair_clean_ids) for name, ids in stream_sets.items()}
    summary = {
        "status": "CURRENT_DOSE_STREAM_HELDOUT_EXPOSURE_AUDIT",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": round(time.time() - t0, 2),
        "heldout_rows": len(heldout_rows),
        "inherited_aln_pair_text_hit_rows": len(pair_blocked_ids),
        "per_stream": metas,
        "set_relations": {
            "base100M_vs_dose21_unique_hit_symmetric_difference": len(stream_sets.get("base_compact_view_reinvest_100M", set()) ^ stream_sets.get("dose21_probe_clean_100M", set())),
            "base100M_vs_dose25_unique_hit_symmetric_difference": len(stream_sets.get("base_compact_view_reinvest_100M", set()) ^ stream_sets.get("dose25_probe_clean_100M", set())),
            "dose21_vs_dose25_unique_hit_symmetric_difference": len(stream_sets.get("dose21_probe_clean_100M", set()) ^ stream_sets.get("dose25_probe_clean_100M", set())),
            "pair_clean_rows_hit_by_stream": clean_stream_counts,
        },
        "clean_subsets": {
            "no_inherited_pair_text_hits_rows": len(clean_pair_only),
            "no_inherited_pair_or_base_dose_stream_hits_rows": len(clean_any),
            "removed_by_union_pair_or_stream_hits_rows": len(blocked_any),
            "no_inherited_pair_or_base_dose_stream_hits_path": rel(OUT / "heldout6992_no_inherited_pair_or_base_dose_stream_hits.jsonl"),
            "blocked_union_path": rel(OUT / "heldout6992_inherited_pair_or_base_dose_stream_hit_rows.jsonl"),
        },
        "inputs": {"heldout": rel(HELDOUT), "pair_blocked": rel(PAIR_BLOCKED), "streams": {k: rel(v) for k, v in STREAMS.items()}},
        "outputs": {"all_hits_csv": rel(OUT / "all_stream_heldout_identity_hits.csv")},
        "scientific_interpretation": "If base and up-dose streams hit the same heldout rows, absolute ordinary-heldout loss is contaminated but within-up-dose deltas are not distorted by differential row exposure. If dose streams remove additional heldout rows relative to base, ordinary-heldout dose deltas must be read on a subset that excludes the union of inherited pair-text hits and any stream-exposed heldout rows.",
    }
    (OUT / "audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
