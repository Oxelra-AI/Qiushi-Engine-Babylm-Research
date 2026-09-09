#!/usr/bin/env python3
"""research: Feasibility/readout for register-contrast displacement instrument.

CPU/file-only: inspect whether the existing clean changed block contains enough
child/subtitle-heavy and Gutenberg/SimpleWiki-heavy rows, with the same word-count
multiset as an admitted FineWeb view block, to build same-rho same-row-geometry
hybrid pools from existing material only.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json, pathlib, collections, statistics, hashlib, time
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT

ROOT = find_user_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
POOL256 = WS / "data/dose_2p64x_rowholdout_pools"
POOL280 = WS / "data/subdose_ladder_maxgeom_pools"
OUT_DIR = WS / "data/register_displacement_feasibility"

CLEAN_META = POOL256 / "cleanqwen_lengthmatched_dose2p64x_changed_block_rows_meta.jsonl"
VIEW_META = POOL256 / "compact_view_dose2p64x_changed_block_rows_meta.jsonl"
CLEAN_10M = POOL256 / "cleanqwen_lengthmatched_dose2p64x_10M.jsonl"
VIEW_10M = POOL256 / "compact_view_dose2p64x_10M.jsonl"
QUARTER_10M = POOL280 / "subdose_quarter_1x_view_10M.jsonl"
SUBDOSE_META = POOL280 / "subdose_ladder_metadata.json"
PAIR_ROWS = 7923
TOTAL_LINES = 65313

CHILD_SUB = {"childes", "open_subtitles"}
ADULT = {"gutenberg", "simple_wiki"}
OTHER = {"bnc_spoken", "switchboard"}


def load_jsonl(path: pathlib.Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def row_group(meta: dict[str, Any], threshold: float = 0.5) -> str:
    words = int(meta.get("words", 0))
    comps = meta.get("component_sources", {}) or {}
    cs = sum(int(comps.get(k, 0)) for k in CHILD_SUB)
    ad = sum(int(comps.get(k, 0)) for k in ADULT)
    oth = sum(int(comps.get(k, 0)) for k in OTHER)
    if words <= 0:
        return "unknown"
    if cs / words >= threshold and cs > ad:
        return "child_sub"
    if ad / words >= threshold and ad > cs:
        return "adult_gut_simple"
    if cs > ad:
        return "mixed_child_sub_lean"
    if ad > cs:
        return "mixed_adult_lean"
    if oth > 0:
        return "other_or_balanced"
    return "other_or_balanced"


def group_scores(meta: dict[str, Any]) -> dict[str, Any]:
    words = int(meta.get("words", 0))
    comps = meta.get("component_sources", {}) or {}
    cs = sum(int(comps.get(k, 0)) for k in CHILD_SUB)
    ad = sum(int(comps.get(k, 0)) for k in ADULT)
    return {
        "child_sub_words": cs,
        "adult_words": ad,
        "other_words": words - cs - ad,
        "child_sub_frac": cs / words if words else 0.0,
        "adult_frac": ad / words if words else 0.0,
        "dominance_gap_child_minus_adult": (cs - ad) / words if words else 0.0,
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    wc = [int(r["words"]) for r in rows]
    comps = collections.Counter()
    groups = collections.Counter()
    for r in rows:
        comps.update({k:int(v) for k,v in (r.get("component_sources", {}) or {}).items()})
        groups[row_group(r)] += 1
    total = sum(wc)
    return {
        "n_rows": len(rows),
        "words": total,
        "mean_words": statistics.mean(wc) if wc else None,
        "min_words": min(wc) if wc else None,
        "max_words": max(wc) if wc else None,
        "component_words": dict(comps),
        "group_rows_threshold_0p5": dict(groups),
        "child_sub_words": sum(comps.get(k,0) for k in CHILD_SUB),
        "adult_gut_simple_words": sum(comps.get(k,0) for k in ADULT),
        "other_words": total - sum(comps.get(k,0) for k in CHILD_SUB) - sum(comps.get(k,0) for k in ADULT),
    }


def greedy_subset_by_hist(candidates_by_len: dict[int, list[int]], target_hist: dict[int, int], score: dict[int, float], reverse: bool=True) -> tuple[set[int], list[dict[str, Any]]]:
    """Pick exact target_hist rows from candidates by word length, prioritizing score."""
    selected = set()
    deficits = []
    for w, need in sorted(target_hist.items()):
        cand = list(candidates_by_len.get(w, []))
        cand.sort(key=lambda i: score.get(i, 0.0), reverse=reverse)
        if len(cand) < need:
            deficits.append({"words": w, "need": need, "available": len(cand), "deficit": need-len(cand)})
            take = cand
        else:
            take = cand[:need]
        selected.update(take)
    return selected, deficits


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clean_meta = list(load_jsonl(CLEAN_META))
    view_meta = list(load_jsonl(VIEW_META))
    submeta = json.loads(SUBDOSE_META.read_text())
    assert len(clean_meta) == 7924 and len(view_meta) == 7924
    # Identify existing quarter active rows by line-level comparison between subdose_quarter and clean/view.
    active = []
    mismatches = []
    with open(QUARTER_10M, encoding="utf-8") as fq, open(CLEAN_10M, encoding="utf-8") as fc, open(VIEW_10M, encoding="utf-8") as fv:
        for idx, (lq, lc, lv) in enumerate(zip(fq, fc, fv)):
            if idx >= PAIR_ROWS:
                break
            rq, rc, rv = json.loads(lq), json.loads(lc), json.loads(lv)
            if rq["text"] == rv["text"]:
                active.append(idx)
            elif rq["text"] != rc["text"]:
                mismatches.append(idx)
    assert not mismatches[:10], f"unexpected quarter mismatches {mismatches[:10]}"
    target_rows = [view_meta[i] for i in active]
    target_hist = collections.Counter(int(r["words"]) for r in target_rows)

    # Basic group readout across clean changed block and existing quarter-displaced rows.
    clean_pair_rows = clean_meta[:PAIR_ROWS]
    original_displaced_rows = [clean_meta[i] for i in active]

    # Candidate definitions: strict/loose by dominance of desired register words.
    candidates = {}
    for name, predicate in {
        "child_sub_dominant_0p5": lambda r: group_scores(r)["child_sub_frac"] >= 0.5 and group_scores(r)["child_sub_words"] > group_scores(r)["adult_words"],
        "adult_dominant_0p5": lambda r: group_scores(r)["adult_frac"] >= 0.5 and group_scores(r)["adult_words"] > group_scores(r)["child_sub_words"],
        "child_sub_lean": lambda r: group_scores(r)["child_sub_words"] > group_scores(r)["adult_words"],
        "adult_lean": lambda r: group_scores(r)["adult_words"] > group_scores(r)["child_sub_words"],
    }.items():
        idxs = [i for i,r in enumerate(clean_pair_rows) if predicate(r)]
        by_len = collections.defaultdict(list)
        for i in idxs:
            by_len[int(clean_pair_rows[i]["words"])].append(i)
        candidates[name] = {"indices": idxs, "by_len_counts": {str(k):len(v) for k,v in sorted(by_len.items())}}

    child_by_len = collections.defaultdict(list)
    adult_by_len = collections.defaultdict(list)
    score_child = {}
    score_adult = {}
    for i, r in enumerate(clean_pair_rows):
        gs = group_scores(r)
        if gs["child_sub_words"] > gs["adult_words"]:
            child_by_len[int(r["words"])].append(i)
            score_child[i] = gs["dominance_gap_child_minus_adult"]
        if gs["adult_words"] > gs["child_sub_words"]:
            adult_by_len[int(r["words"])].append(i)
            score_adult[i] = -gs["dominance_gap_child_minus_adult"]
    child_sel, child_def = greedy_subset_by_hist(child_by_len, target_hist, score_child, True)
    adult_sel, adult_def = greedy_subset_by_hist(adult_by_len, target_hist, score_adult, True)

    out = {
        "status": "REGISTER_DISPLACEMENT_FEASIBILITY_COMPLETE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Check whether same-rho register-contrast displacement pools can be built from existing MAX clean/view pools only.",
        "source_files": {"clean_meta": rel(CLEAN_META), "view_meta": rel(VIEW_META), "quarter_10m": rel(QUARTER_10M)},
        "existing_quarter_active_rows": {"n_rows": len(active), "first": active[:5], "last": active[-5:], "words": sum(int(view_meta[i]["words"]) for i in active)},
        "target_fineweb_block": summarize_rows(target_rows),
        "original_clean_rows_displaced_by_existing_quarter": summarize_rows(original_displaced_rows),
        "all_clean_changed_pair_rows": summarize_rows(clean_pair_rows),
        "candidate_counts": {k: len(v["indices"]) for k,v in candidates.items()},
        "exact_word_count_multiset_match_using_lean_rows": {
            "child_sub_lean_selected_rows": len(child_sel),
            "adult_lean_selected_rows": len(adult_sel),
            "target_rows": len(active),
            "child_deficits": child_def[:20],
            "adult_deficits": adult_def[:20],
            "child_deficit_total": sum(d["deficit"] for d in child_def),
            "adult_deficit_total": sum(d["deficit"] for d in adult_def),
            "child_selected_summary": summarize_rows([clean_pair_rows[i] for i in sorted(child_sel)]),
            "adult_selected_summary": summarize_rows([clean_pair_rows[i] for i in sorted(adult_sel)]),
            "intersection_selected_rows": len(child_sel & adult_sel),
        },
        "scientific_note": "If exact word-count multiset matches exist, we can create two pools with identical FineWeb active rows and identical row-word sequence, differing only in which clean-register rows are absent and the corresponding remaining clean-row assignment.",
        "no_gpu_training_eval_generation_streaming_upload": True,
    }
    (OUT_DIR / "register_displacement_feasibility.json").write_text(json.dumps(out, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    md = ["# research register-displacement feasibility", "", "CPU/file-only; no model loading/training/eval/generation/streaming.", "", "## Existing quarter admitted FineWeb block", f"Rows: {len(active)}, words: {out['existing_quarter_active_rows']['words']}", "", "## Original clean rows displaced by existing quarter", "```json", json.dumps(out["original_clean_rows_displaced_by_existing_quarter"], indent=2, ensure_ascii=False), "```", "", "## Candidate exact length-match feasibility", "```json", json.dumps(out["exact_word_count_multiset_match_using_lean_rows"], indent=2, ensure_ascii=False), "```", ""]
    (OUT_DIR / "register_displacement_feasibility.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"status": out["status"], "active_rows": len(active), "active_words": out['existing_quarter_active_rows']['words'], "child_deficit_total": out['exact_word_count_multiset_match_using_lean_rows']['child_deficit_total'], "adult_deficit_total": out['exact_word_count_multiset_match_using_lean_rows']['adult_deficit_total'], "summary": rel(OUT_DIR / "register_displacement_feasibility.md")}, indent=2))

if __name__ == "__main__":
    main()
