#!/usr/bin/env python3
"""research: audit inherited COMPACT_EXPERIENCE ALN material and COMPACT_EXPERIENCE streams for heldout/probe leakage.

research repaired the *new* dose pools after discovering that Qwen restatement
candidates drawn blindly from official text frequently overlapped the later
BabyLM ordinary row-holdout and probe text.  The inherited COMPACT_EXPERIENCE aligned
restatement block was drawn by an earlier blind process, so the same screen must
be applied before interpreting either (i) compact-view-reinvest baseline ordinary
heldout loss, which shares the inherited block, or (ii) the old COMPACT_EXPERIENCE OFF/ALN
ordinary-heldout improvement used as a calibration prior.

This script reuses the research text screen for selected ALN pair sources and
rewrites, and separately checks whether the COMPACT_EXPERIENCE OFF/ALN 10M streams themselves
contain the 6,992 primary heldout rows as ordinary official text.  The 100M
training files repeat/shuffle these 10M pools for ten passes, so the 10M audit is
sufficient for presence and pass exposure.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import csv
import hashlib
import importlib.util
import json
import pathlib
import sys
import time
from dataclasses import dataclass
from typing import Any

ROOT = pathlib.Path.cwd()
STUDY = ROOT / "experiments/archive/relation_learning"
SCRIPT = STUDY / "scripts/audit_dose_probe_leakage.py"
OUT_DEFAULT = STUDY / "data/inherited_aln_leakage_audit"
COMPACT_EXPERIENCE_DIR = ROOT / "experiments/archive/compact_experience/data/qwen_clean_aligned"
SELECTED_ALN = COMPACT_EXPERIENCE_DIR / "selected_pairs.jsonl"
QWEN_PAIR_META = COMPACT_EXPERIENCE_DIR / "qwen_pair_packed_rows_meta.jsonl"
OFFICIAL_ONLY_10M = COMPACT_EXPERIENCE_DIR / "training_corpora/official_only_10M.jsonl"
QWEN_ALIGNED_10M = COMPACT_EXPERIENCE_DIR / "training_corpora/qwen_aligned_10M.jsonl"
MATERIALIZATION_META = COMPACT_EXPERIENCE_DIR / "clean_materialization_metadata.json"


def import_step060():
    spec = importlib.util.spec_from_file_location("leakage", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


leak = import_step060()


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: pathlib.Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


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


def pair_id(obj: dict[str, Any], index: int) -> str:
    return str(obj.get("pair_id") or obj.get("id") or f"selected_aln_index_{index:05d}")


def load_selected_aln_queries() -> tuple[list[Any], dict[str, Any]]:
    queries = []
    n = 0
    fields_seen = collections.Counter()
    pair_words = 0
    for idx, obj in enumerate(read_jsonl(SELECTED_ALN)):
        n += 1
        pid = pair_id(obj, idx)
        original = " ".join(str(obj.get("original") or obj.get("source_text") or "").split())
        rewrite = " ".join(str(obj.get("rewrite") or obj.get("rewrite_text") or "").split())
        pair_words += int(obj.get("pair_words") or (len(original.split()) + len(rewrite.split())))
        for field, text in [("original", original), ("rewrite", rewrite)]:
            if not text:
                continue
            ts = leak.tokens(text)
            fields_seen[field] += 1
            queries.append(leak.QueryText(
                pair_id=pid,
                dose21=False,
                dose25_topup=False,
                dose25=False,
                field=field,
                text=text,
                norm=leak.norm_text(text),
                toks=ts,
                tokset=leak.informative_tokens(ts),
                source=str(obj.get("source", "")),
                original_id=str(obj.get("example_id") or obj.get("original_id") or ""),
            ))
    meta = {
        "selected_pairs": n,
        "selected_pair_words_recomputed_or_recorded": pair_words,
        "selected_texts_audited": len(queries),
        "query_fields": dict(fields_seen),
        "selected_pairs_path": rel(SELECTED_ALN),
    }
    return queries, meta


def blocking_summary(hits: list[dict[str, Any]]) -> dict[str, Any]:
    blocking_types = {"exact_norm", "exact_token_subsequence", "high_jaccard", "heldout_high_query_containment"}
    blocking = [h for h in hits if h.get("hit_type") in blocking_types]
    by_pair: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    by_ref: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for h in blocking:
        by_pair[str(h["pair_id"])].append(h)
        by_ref[str(h["ref_id"])].append(h)
    return {
        "blocking_hit_count": len(blocking),
        "blocking_pair_count": len(by_pair),
        "blocking_ref_count": len(by_ref),
        "blocking_hits_by_screen": dict(sorted(collections.Counter(h["screen"] for h in blocking).items())),
        "blocking_hits_by_type": dict(sorted(collections.Counter(h["hit_type"] for h in blocking).items())),
        "blocking_hits_by_query_field": dict(sorted(collections.Counter(h["query_field"] for h in blocking).items())),
    }


def heldout_rows(path: pathlib.Path, tag: str) -> list[dict[str, Any]]:
    rows = []
    for obj in read_jsonl(path):
        text = " ".join(str(obj.get("text", "")).split())
        ts = leak.tokens(text)
        rows.append({
            "tag": tag,
            "row_id": f"{tag}:row:{obj.get('source')}:{obj.get('example_id')}",
            "source": str(obj.get("source", "")),
            "example_id": int(obj.get("example_id")),
            "text": text,
            "norm": leak.norm_text(text),
            "tokens": ts,
            "words": int(obj.get("words", len(text.split()))),
            "raw": obj,
        })
    return rows


def parent_heldout_ref(ref_id: str) -> str:
    # research row chunks have ids like heldout6992:row:source:example_id:chunk:j.
    if ":chunk:" in ref_id:
        return ref_id.split(":chunk:", 1)[0]
    return ref_id


def write_clean_heldout_subset(out_dir: pathlib.Path, hits: list[dict[str, Any]]) -> dict[str, Any]:
    blocked_6992 = {
        parent_heldout_ref(str(h["ref_id"]))
        for h in hits
        if h.get("screen") == "heldout6992" and h.get("hit_type") in {"exact_norm", "exact_token_subsequence", "high_jaccard", "heldout_high_query_containment"}
    }
    rows = heldout_rows(leak.HELDOUT_6992, "heldout6992")
    clean = [r["raw"] for r in rows if r["row_id"] not in blocked_6992]
    blocked = [r["raw"] for r in rows if r["row_id"] in blocked_6992]
    clean_path = out_dir / "heldout6992_no_inherited_aln_pair_text_hits.jsonl"
    blocked_path = out_dir / "heldout6992_inherited_aln_pair_text_hit_rows.jsonl"
    write_jsonl(clean_path, clean)
    write_jsonl(blocked_path, blocked)
    return {
        "heldout6992_total_rows": len(rows),
        "rows_with_inherited_aln_pair_text_hits": len(blocked),
        "rows_without_inherited_aln_pair_text_hits": len(clean),
        "clean_subset_path": rel(clean_path),
        "blocked_subset_path": rel(blocked_path),
    }


def stream_row_identity_hits(stream_path: pathlib.Path, heldout: list[dict[str, Any]], *, stream_tag: str, ordinary_only: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_key = {(r["source"], int(r["example_id"])): r for r in heldout}
    hits = []
    n_rows = 0
    n_ordinary = 0
    n_pair = 0
    total_words = 0
    for row_index, obj in enumerate(read_jsonl(stream_path)):
        n_rows += 1
        total_words += int(obj.get("words", len(str(obj.get("text", "")).split())))
        source = str(obj.get("source", ""))
        is_pair = source == "qwen_pair_packed" or source.startswith("qwen_pair_packed")
        if is_pair:
            n_pair += 1
        else:
            n_ordinary += 1
        if ordinary_only and is_pair:
            continue
        try:
            exid = int(obj.get("example_id"))
        except Exception:
            continue
        key = (source, exid)
        if key in by_key:
            h = by_key[key]
            norm_match = leak.norm_text(str(obj.get("text", ""))) == h["norm"]
            hits.append({
                "stream": stream_tag,
                "stream_path": rel(stream_path),
                "stream_row_index": row_index,
                "stream_source": source,
                "stream_example_id": exid,
                "heldout_row_id": h["row_id"],
                "heldout_source": h["source"],
                "heldout_example_id": h["example_id"],
                "words": int(obj.get("words", 0)),
                "norm_text_exact_match": norm_match,
                "text_preview": str(obj.get("text", ""))[:300],
            })
    meta = {
        "stream": stream_tag,
        "stream_path": rel(stream_path),
        "rows_scanned": n_rows,
        "ordinary_rows_scanned": n_ordinary,
        "pair_rows_scanned": n_pair,
        "total_words": total_words,
        "identity_hit_count": len(hits),
        "unique_heldout_rows_by_identity": len({h["heldout_row_id"] for h in hits}),
        "norm_text_exact_identity_hits": sum(1 for h in hits if h["norm_text_exact_match"]),
        "ordinary_only": ordinary_only,
    }
    return hits, meta


def stream_sequence_hits(stream_path: pathlib.Path, heldout: list[dict[str, Any]], *, stream_tag: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # Streaming exact-token-subsequence search for heldout rows inside a 10M stream.
    candidates_by_prefix: dict[tuple[str, ...], list[dict[str, Any]]] = collections.defaultdict(list)
    usable = []
    for r in heldout:
        ts = r["tokens"]
        if len(ts) >= 5:
            candidates_by_prefix[tuple(ts[:5])].append(r)
            usable.append(r)
    max_len = max(len(r["tokens"]) for r in usable) if usable else 0
    found: dict[str, dict[str, Any]] = {}
    buf_toks: list[str] = []
    buf_rows: list[int] = []
    buf_sources: list[str] = []
    total_tokens = 0
    n_rows = 0
    n_pair_rows = 0
    n_ordinary_rows = 0
    total_words = 0

    def scan(scannable_end: int) -> None:
        nonlocal buf_toks, buf_rows, buf_sources
        for j in range(0, max(0, scannable_end)):
            if j + 5 > len(buf_toks):
                break
            gram = tuple(buf_toks[j:j+5])
            cand = candidates_by_prefix.get(gram)
            if not cand:
                continue
            for r in cand:
                rid = r["row_id"]
                if rid in found:
                    continue
                q = r["tokens"]
                L = len(q)
                if j + L <= len(buf_toks) and buf_toks[j:j+L] == q:
                    srcs = sorted(set(buf_sources[j:j+L]))
                    found[rid] = {
                        "stream": stream_tag,
                        "stream_path": rel(stream_path),
                        "heldout_row_id": rid,
                        "heldout_source": r["source"],
                        "heldout_example_id": r["example_id"],
                        "heldout_words": r["words"],
                        "start_stream_row_index": buf_rows[j],
                        "end_stream_row_index": buf_rows[j+L-1],
                        "stream_sources_spanned": ";".join(srcs[:8]) + (";..." if len(srcs) > 8 else ""),
                        "spans_qwen_pair_row": any(s == "qwen_pair_packed" or s.startswith("qwen_pair_packed") for s in srcs),
                        "heldout_text_preview": r["text"][:300],
                    }

    for row_index, obj in enumerate(read_jsonl(stream_path)):
        n_rows += 1
        source = str(obj.get("source", ""))
        is_pair = source == "qwen_pair_packed" or source.startswith("qwen_pair_packed")
        if is_pair:
            n_pair_rows += 1
        else:
            n_ordinary_rows += 1
        total_words += int(obj.get("words", len(str(obj.get("text", "")).split())))
        ts = leak.tokens(str(obj.get("text", "")))
        total_tokens += len(ts)
        buf_toks.extend(ts)
        buf_rows.extend([row_index] * len(ts))
        buf_sources.extend([source] * len(ts))
        # Keep max_len-1 tokens unscanned so a heldout row split across row boundaries can complete.
        scannable_end = max(0, len(buf_toks) - max_len + 1)
        scan(scannable_end)
        if scannable_end:
            del buf_toks[:scannable_end]
            del buf_rows[:scannable_end]
            del buf_sources[:scannable_end]
    # Final flush.
    scan(max(0, len(buf_toks) - 4))
    rows = list(found.values())
    meta = {
        "stream": stream_tag,
        "stream_path": rel(stream_path),
        "rows_scanned": n_rows,
        "ordinary_rows_scanned": n_ordinary_rows,
        "pair_rows_scanned": n_pair_rows,
        "total_words": total_words,
        "total_word_tokens_scanned": total_tokens,
        "heldout_rows_queried": len(usable),
        "exact_token_subsequence_hit_count": len(rows),
        "hits_spanning_qwen_pair_rows": sum(1 for r in rows if r["spans_qwen_pair_row"]),
        "hits_not_spanning_qwen_pair_rows": sum(1 for r in rows if not r["spans_qwen_pair_row"]),
    }
    return rows, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--jaccard-threshold", type=float, default=0.80)
    ap.add_argument("--containment-threshold", type=float, default=0.95)
    ap.add_argument("--skip-stream-sequence", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    refs, ref_meta = leak.load_references()
    aln_queries, aln_query_meta = load_selected_aln_queries()
    aln_hits = leak.screen_hits(aln_queries, refs, jaccard_threshold=args.jaccard_threshold, containment_threshold=args.containment_threshold)
    blocking_types = {"exact_norm", "exact_token_subsequence", "high_jaccard", "heldout_high_query_containment"}
    aln_blocking_hits = [h for h in aln_hits if h.get("hit_type") in blocking_types]
    leak.write_csv(out_dir / "inherited_aln_blocking_hits.csv", aln_blocking_hits)
    with (out_dir / "inherited_aln_blocking_hits.jsonl").open("w", encoding="utf-8") as f:
        for h in aln_blocking_hits:
            f.write(json.dumps(h, ensure_ascii=False) + "\n")
    clean_subset_meta = write_clean_heldout_subset(out_dir, aln_blocking_hits)

    heldout6992 = heldout_rows(leak.HELDOUT_6992, "heldout6992")
    heldout2647 = heldout_rows(leak.HELDOUT_2647, "heldout2647")
    all_heldout = heldout6992 + heldout2647

    qwen_identity_hits, qwen_identity_meta = stream_row_identity_hits(
        QWEN_ALIGNED_10M, all_heldout, stream_tag="compact_experience_qwen_aligned_10M_identity", ordinary_only=True
    )
    write_csv(out_dir / "paired_context_qwen_aligned_10M_ordinary_identity_hits.csv", qwen_identity_hits)

    sequence_metas = []
    if not args.skip_stream_sequence:
        off_seq_hits, off_seq_meta = stream_sequence_hits(
            OFFICIAL_ONLY_10M, heldout6992, stream_tag="compact_experience_official_only_10M_fullstream_heldout6992"
        )
        qwen_seq_hits, qwen_seq_meta = stream_sequence_hits(
            QWEN_ALIGNED_10M, heldout6992, stream_tag="compact_experience_qwen_aligned_10M_fullstream_heldout6992"
        )
        write_csv(out_dir / "paired_context_official_only_10M_heldout6992_sequence_hits.csv", off_seq_hits)
        write_csv(out_dir / "paired_context_qwen_aligned_10M_heldout6992_sequence_hits.csv", qwen_seq_hits)
        sequence_metas.extend([off_seq_meta, qwen_seq_meta])

    aln_summary = blocking_summary(aln_blocking_hits)
    materialization_meta = json.loads(MATERIALIZATION_META.read_text(encoding="utf-8"))
    summary = {
        "status": "INHERITED_ALN_AND_COMPACT_EXPERIENCE_STREAM_LEAKAGE_AUDIT",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": round(time.time() - t0, 2),
        "thresholds": {
            "jaccard_threshold": args.jaccard_threshold,
            "heldout_query_containment_threshold": args.containment_threshold,
        },
        "reference_meta": ref_meta,
        "inherited_aln_query_meta": aln_query_meta,
        "inherited_aln_pair_text_screen": aln_summary,
        "heldout6992_clean_subset_from_pair_text_hits": clean_subset_meta,
        "compact_experience_qwen_aligned_ordinary_identity_screen": qwen_identity_meta,
        "compact_experience_stream_sequence_screens": sequence_metas,
        "compact_experience_materialization_context": {
            "selected_pairs": materialization_meta.get("selected_pairs"),
            "selected_pair_words": materialization_meta.get("selected_pair_words"),
            "qwen_pair_rows": materialization_meta.get("qwen_pair_rows"),
            "official_filler_words_in_treatment": materialization_meta.get("official_filler_words_in_treatment"),
            "treatment_pool_rows": materialization_meta.get("treatment_pool_rows"),
            "control_pool_rows": materialization_meta.get("control_pool_rows"),
            "sha256": materialization_meta.get("sha256", {}),
        },
        "inputs": {
            "selected_aln_pairs": rel(SELECTED_ALN),
            "qwen_pair_meta": rel(QWEN_PAIR_META),
            "official_only_10M": rel(OFFICIAL_ONLY_10M),
            "qwen_aligned_10M": rel(QWEN_ALIGNED_10M),
            "materialization_meta": rel(MATERIALIZATION_META),
            "heldout6992": rel(leak.HELDOUT_6992),
            "heldout2647": rel(leak.HELDOUT_2647),
            "compact_all_accepted": rel(leak.ALL_ACCEPTED_PAIRS),
            "wiki_pairs": rel(leak.WIKI_PAIRS),
        },
        "input_sha256": {
            "selected_aln_pairs": sha256_file(SELECTED_ALN),
            "official_only_10M": sha256_file(OFFICIAL_ONLY_10M),
            "qwen_aligned_10M": sha256_file(QWEN_ALIGNED_10M),
            "heldout6992": sha256_file(leak.HELDOUT_6992),
            "heldout2647": sha256_file(leak.HELDOUT_2647),
        },
        "outputs": {
            "inherited_aln_blocking_hits_csv": rel(out_dir / "inherited_aln_blocking_hits.csv"),
            "inherited_aln_blocking_hits_jsonl": rel(out_dir / "inherited_aln_blocking_hits.jsonl"),
            "qwen_aligned_ordinary_identity_hits_csv": rel(out_dir / "paired_context_qwen_aligned_10M_ordinary_identity_hits.csv"),
            "official_only_sequence_hits_csv": rel(out_dir / "paired_context_official_only_10M_heldout6992_sequence_hits.csv"),
            "qwen_aligned_sequence_hits_csv": rel(out_dir / "paired_context_qwen_aligned_10M_heldout6992_sequence_hits.csv"),
            **{k: v for k, v in clean_subset_meta.items() if k.endswith("path")},
        },
        "scientific_interpretation": (
            "Inherited ALN pair-text hits contaminate absolute ordinary-heldout ordinates for any substrate that contains the inherited aligned block. "
            "The cleaned heldout subset removes rows whose text appears as an inherited source/rewrite under the same blocking screen. "
            "The COMPACT_EXPERIENCE stream screens separately establish whether old OFF/ALN ordinary-heldout comparisons trained on the later heldout rows as official text; up-dose contrasts within the compact-view-reinvest substrate remain shared-inherited-block contrasts."
        ),
    }
    (out_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
