#!/usr/bin/env python3
"""research: relation-stream exposure screen for corpus-derived heldout rows.

This screen classifies, for each relation-family stream set, whether each heldout
row is absent from every stream, present identically in every arm, or present in
only some arms.  The goal is to preserve within-family contrast validity while
using the word "held out" only where identity absence has actually been earned.

The match used here is deliberately row-level/exact: either matching
(source, example_id) or matching the entire normalized row text.  The research
COMPACT_EXPERIENCE audit already separately established exact full-text subsequence presence
for the old OFF/ALN streams.  This script is the broad row-identity companion
across relation families, architectures, and current dose streams.
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
OUT = STUDY / "data/relation_stream_exposure_screen"

HELDOUT_SPECS = {
    "heldout6992": ROOT / "experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/heldout_cleanqwen_rows.jsonl",
    "heldout2647": ROOT / "experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/heldout_cleanqwen_rows.jsonl",
}
INHERITED_HITS = STUDY / "data/inherited_aln_leakage_audit/inherited_aln_blocking_hits.jsonl"

STREAMS = {
    # COMPACT_EXPERIENCE relation-design family (10M pool is repeated to 100M).
    "compact_experience_OFF": ROOT / "experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/official_only_10M.jsonl",
    "compact_experience_ALN": ROOT / "experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl",
    "compact_experience_SHUF": ROOT / "experiments/archive/compact_experience/data/qwen_shuffled_control/training_corpora/qwen_shuffled_10M.jsonl",
    "compact_experience_DUP": ROOT / "experiments/archive/compact_experience/data/selected_original_dup_all_control/training_corpora/selected_original_dup_all_10M.jsonl",
    "compact_experience_SEP": ROOT / "experiments/archive/compact_experience/data/qwen_separated_pair_control/training_corpora/qwen_separated_pair_10M.jsonl",
    # REPRESENTATION_FRONTIER_STUDIES max-dose relation/locality streams.
    "max_C_clean": ROOT / "experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/cleanqwen_lengthmatched_dose2p64x_10M.jsonl",
    "max_R_repeat": ROOT / "experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/compact_repeat_dose2p64x_10M.jsonl",
    "max_V_view": ROOT / "experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/compact_view_dose2p64x_10M.jsonl",
    "split_RS_repeat": ROOT / "experiments/archive/relation_learning/data/split_inwindow_rowholdout_pools/compact_repeat_split_dose2p64x_10M.jsonl",
    "split_VS_view": ROOT / "experiments/archive/relation_learning/data/split_inwindow_rowholdout_pools/compact_view_split_dose2p64x_10M.jsonl",
    # Hash-mixed / half-view controls.
    "HM_hash_mixed": ROOT / "experiments/archive/relation_learning/data/hash_mixed_inwindow_pools/compact_hash_mixed_dose2p64x_10M.jsonl",
    "HV_half_view": ROOT / "experiments/archive/relation_learning/data/half_view_control_pools/compact_half_view_noexact_dose2p64x_10M.jsonl",
    # Current compact-view-reinvest practical substrate and repaired up-dose streams.
    "current_base_compact_view_reinvest": ROOT / "experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl",
    "current_dose21_probe_clean": ROOT / "experiments/archive/relation_learning/data/probe_clean_nested_dose_streams/dose21/dose21_compact_view_reinvest_100M.jsonl",
    "current_dose25_probe_clean": ROOT / "experiments/archive/relation_learning/data/probe_clean_nested_dose_streams/dose25/dose25_compact_view_reinvest_100M.jsonl",
}

FAMILIES = {
    "compact_experience_OFF_ALN_SHUF_DUP_SEP": {
        "OFF": "compact_experience_OFF",
        "ALN": "compact_experience_ALN",
        "SHUF": "compact_experience_SHUF",
        "DUP": "compact_experience_DUP",
        "SEP": "compact_experience_SEP",
    },
    "representation_frontier_studies_DeBERTa_CRV_shared_streams": {
        "C": "max_C_clean",
        "R": "max_R_repeat",
        "V": "max_V_view",
    },
    "representation_frontier_studies_DeBERTa_split_locality": {
        "C": "max_C_clean",
        "RS": "split_RS_repeat",
        "VS": "split_VS_view",
    },
    "RoBERTa_CRV_same_input_streams": {
        "C": "max_C_clean",
        "R": "max_R_repeat",
        "V": "max_V_view",
    },
    "RoBERTa_repeat_split_same_input_streams": {
        "C": "max_C_clean",
        "R": "max_R_repeat",
        "RS": "split_RS_repeat",
    },
    "causal_GPT_C_R_RS_same_input_streams": {
        "C": "max_C_clean",
        "R": "max_R_repeat",
        "RS": "split_RS_repeat",
    },
    "hash_mixed_and_half_view_controls": {
        "C": "max_C_clean",
        "R": "max_R_repeat",
        "V": "max_V_view",
        "HM": "HM_hash_mixed",
        "HV": "HV_half_view",
    },
    "current_compactview_base_dose21_dose25": {
        "BASE0": "current_base_compact_view_reinvest",
        "DOSE21": "current_dose21_probe_clean",
        "DOSE25": "current_dose25_probe_clean",
    },
}

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[\u2019'][A-Za-z0-9]+)?")


def norm_text(s: str) -> str:
    return " ".join(m.group(0).lower().replace("\u2019", "'") for m in WORD_RE.finditer(str(s)))


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


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


def row_id(screen: str, obj: dict[str, Any]) -> str:
    return f"{screen}:row:{obj.get('source')}:{obj.get('example_id')}"


def parent_ref_id(ref_id: str) -> str:
    return ref_id.split(":chunk:", 1)[0] if ":chunk:" in ref_id else ref_id


def load_heldout() -> tuple[list[dict[str, Any]], dict[tuple[str, str, int], list[str]], dict[str, list[str]], set[str]]:
    rows: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str, int], list[str]] = collections.defaultdict(list)
    by_norm: dict[str, list[str]] = collections.defaultdict(list)
    for screen, path in HELDOUT_SPECS.items():
        for obj in read_jsonl(path):
            text = " ".join(str(obj.get("text", "")).split())
            rid = row_id(screen, obj)
            try:
                exid = int(obj.get("example_id"))
            except Exception:
                continue
            rec = {
                "heldout_row_id": rid,
                "screen": screen,
                "source": str(obj.get("source", "")),
                "example_id": exid,
                "words": int(obj.get("words", len(text.split()))),
                "text": text,
                "norm": norm_text(text),
                "raw": obj,
            }
            rows.append(rec)
            by_key[(screen, rec["source"], rec["example_id"])].append(rid)
            by_norm[f"{screen}\t{rec['norm']}"].append(rid)
    pair_hits: set[str] = set()
    if INHERITED_HITS.exists():
        for h in read_jsonl(INHERITED_HITS):
            sid = str(h.get("screen", ""))
            if sid in HELDOUT_SPECS:
                pair_hits.add(parent_ref_id(str(h.get("ref_id", ""))))
    return rows, by_key, by_norm, pair_hits


def scan_stream(alias: str, path: pathlib.Path, by_key: dict[tuple[str, str, int], list[str]], by_norm: dict[str, list[str]], pair_hits: set[str]) -> tuple[list[dict[str, Any]], dict[str, Any], set[str]]:
    if not path.exists():
        return [], {"stream_alias": alias, "stream_path": rel(path), "exists": False}, set()
    hit_rows: list[dict[str, Any]] = []
    unique_ids: set[str] = set()
    counts_by_source = collections.Counter()
    hit_counts_by_type = collections.Counter()
    row_count = 0
    total_words = 0
    for ix, obj in enumerate(read_jsonl(path)):
        row_count += 1
        src = str(obj.get("source", ""))
        counts_by_source[src] += 1
        text = " ".join(str(obj.get("text", "")).split())
        words = int(obj.get("words", len(text.split())))
        total_words += words
        try:
            exid = int(obj.get("example_id"))
        except Exception:
            exid = None
        nt = norm_text(text)
        matches: dict[str, set[str]] = collections.defaultdict(set)
        if exid is not None:
            for screen in HELDOUT_SPECS:
                for rid in by_key.get((screen, src, exid), []):
                    matches[rid].add("source_example_id")
        for screen in HELDOUT_SPECS:
            for rid in by_norm.get(f"{screen}\t{nt}", []):
                matches[rid].add("norm_text")
        for rid, types in matches.items():
            unique_ids.add(rid)
            typ = "+".join(sorted(types))
            hit_counts_by_type[typ] += 1
            hit_rows.append({
                "stream_alias": alias,
                "stream_path": rel(path),
                "stream_row_index": ix,
                "stream_source": src,
                "stream_example_id": exid,
                "stream_words": words,
                "hit_type": typ,
                "heldout_row_id": rid,
                "screen": rid.split(":row:", 1)[0],
                "inherited_aln_pair_text_hit_row": rid in pair_hits,
                "text_preview": text[:220],
            })
    meta = {
        "stream_alias": alias,
        "stream_path": rel(path),
        "exists": True,
        "rows_scanned": row_count,
        "total_words": total_words,
        "hit_occurrences": len(hit_rows),
        "unique_heldout_rows_hit": len(unique_ids),
        "unique_heldout6992_rows_hit": sum(1 for rid in unique_ids if rid.startswith("heldout6992:")),
        "unique_heldout2647_rows_hit": sum(1 for rid in unique_ids if rid.startswith("heldout2647:")),
        "hit_occurrences_by_type": dict(sorted(hit_counts_by_type.items())),
        "top_stream_sources": dict(counts_by_source.most_common(12)),
    }
    return hit_rows, meta, unique_ids


def summarize_family(family: str, arms: dict[str, str], heldout_rows: list[dict[str, Any]], stream_unique_hits: dict[str, set[str]], pair_hits: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    arm_names = list(arms.keys())
    categories: list[dict[str, Any]] = []
    for r in heldout_rows:
        rid = r["heldout_row_id"]
        present = [arm for arm in arm_names if rid in stream_unique_hits.get(arms[arm], set())]
        if not present:
            category = "absent_from_every_stream"
        elif len(present) == len(arm_names):
            category = "present_identically_in_every_arm"
        else:
            category = "differentially_present_some_arms"
        categories.append({
            "family": family,
            "screen": r["screen"],
            "heldout_row_id": rid,
            "source": r["source"],
            "example_id": r["example_id"],
            "words": r["words"],
            "category": category,
            "present_arm_count": len(present),
            "arm_count": len(arm_names),
            "present_arms": ";".join(present),
            "absent_arms": ";".join([a for a in arm_names if a not in present]),
            "inherited_aln_pair_text_hit_row": rid in pair_hits,
            "text_preview": r["text"][:220],
        })
    summary_rows: list[dict[str, Any]] = []
    for screen in HELDOUT_SPECS:
        sub = [c for c in categories if c["screen"] == screen]
        cat_counts = collections.Counter(c["category"] for c in sub)
        for cat in ["absent_from_every_stream", "present_identically_in_every_arm", "differentially_present_some_arms"]:
            rows = [c for c in sub if c["category"] == cat]
            summary_rows.append({
                "family": family,
                "screen": screen,
                "category": cat,
                "rows": len(rows),
                "fraction": round(len(rows) / max(len(sub), 1), 6),
                "inherited_aln_pair_text_hit_rows": sum(1 for c in rows if c["inherited_aln_pair_text_hit_row"]),
                "arm_map": json.dumps(arms, sort_keys=True),
            })
        diff = [c for c in sub if c["category"] == "differentially_present_some_arms"]
        diff_patterns = collections.Counter(c["present_arms"] for c in diff)
        for pat, n in diff_patterns.most_common(10):
            summary_rows.append({
                "family": family,
                "screen": screen,
                "category": "differential_pattern_top",
                "rows": n,
                "fraction": round(n / max(len(sub), 1), 6),
                "inherited_aln_pair_text_hit_rows": sum(1 for c in diff if c["present_arms"] == pat and c["inherited_aln_pair_text_hit_row"]),
                "arm_map": pat,
            })
    return categories, summary_rows


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    heldout_rows, by_key, by_norm, pair_hits = load_heldout()

    all_hits: list[dict[str, Any]] = []
    stream_metas: list[dict[str, Any]] = []
    stream_unique_hits: dict[str, set[str]] = {}
    for alias, path in STREAMS.items():
        hits, meta, unique = scan_stream(alias, path, by_key, by_norm, pair_hits)
        all_hits.extend(hits)
        stream_metas.append(meta)
        stream_unique_hits[alias] = unique
        write_csv(OUT / f"{alias}_row_identity_hits.csv", hits)
        print(json.dumps({"event": "stream_scanned", "alias": alias, "unique_hits": len(unique), "rows": meta.get("rows_scanned"), "elapsed_sec": round(time.time() - t0, 1)}, ensure_ascii=False), flush=True)

    all_categories: list[dict[str, Any]] = []
    family_summaries: list[dict[str, Any]] = []
    for family, arms in FAMILIES.items():
        cats, summ = summarize_family(family, arms, heldout_rows, stream_unique_hits, pair_hits)
        all_categories.extend(cats)
        family_summaries.extend(summ)

    write_csv(OUT / "all_stream_row_identity_hits.csv", all_hits)
    write_csv(OUT / "family_row_categories.csv", all_categories)
    write_csv(OUT / "family_summary.csv", family_summaries)

    by_family_screen = collections.defaultdict(dict)
    for row in family_summaries:
        if row["category"] in {"absent_from_every_stream", "present_identically_in_every_arm", "differentially_present_some_arms"}:
            by_family_screen[(row["family"], row["screen"])][row["category"]] = row["rows"]
    compact = []
    for (family, screen), vals in sorted(by_family_screen.items()):
        compact.append({"family": family, "screen": screen, **vals})

    summary = {
        "status": "RELATION_STREAM_ROW_EXPOSURE_SCREEN_DONE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": round(time.time() - t0, 2),
        "match_definition": "row-level exact exposure: matching (source, example_id) or identical normalized full row text; exact subsequence exposure is not counted here except where research already separately recorded it for COMPACT_EXPERIENCE OFF/ALN.",
        "heldout_inputs": {k: rel(v) for k, v in HELDOUT_SPECS.items()},
        "heldout_rows": dict(collections.Counter(r["screen"] for r in heldout_rows)),
        "inherited_aln_pair_text_hit_rows_loaded": len(pair_hits),
        "stream_metas": stream_metas,
        "family_compact_summary": compact,
        "families": FAMILIES,
        "streams": {k: rel(v) for k, v in STREAMS.items()},
        "outputs": {
            "all_hits_csv": rel(OUT / "all_stream_row_identity_hits.csv"),
            "family_row_categories_csv": rel(OUT / "family_row_categories.csv"),
            "family_summary_csv": rel(OUT / "family_summary.csv"),
            "summary_json": rel(OUT / "summary.json"),
        },
        "scientific_interpretation": "For a family and row set, absent_from_every_stream earns row-identity heldout status for that family; present_identically_in_every_arm preserves within-family contrasts but changes absolute ordinary-loss wording to trained-text fit; differential exposure can bias the contrast itself and requires subseting or removal before using that ordinate.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
