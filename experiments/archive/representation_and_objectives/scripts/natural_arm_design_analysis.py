#!/usr/bin/env python3
"""research natural-arm design analysis for the next FineWeb study.

This is CPU-only and does not train or evaluate a model.  It reads the measured
representation and consolidation artifacts and writes a durable design calculation
for the next H100 experiment after the pending source-breadth trajectory returns.
"""
from __future__ import annotations

import json
import math
import pathlib
from typing import Any

ROOT = pathlib.Path("experiments/archive/representation_and_objectives")
OUT_DIR = ROOT / "data/natural_arm_design_analysis"
NOTE = (ROOT.parents[2] / 'research/notes/representation_and_objectives/natural_arm_scaled_fineweb_design.md')

A02_NOAOA = pathlib.Path("experiments/archive/frontier_consolidation/data/density_noaoa_eval_retry/density_noaoa_eval_summary.json")
A02_META = pathlib.Path("experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/density_cleanqwen_rowholdout_overlay_metadata.json")
A02_NEAR_SUMMARY = pathlib.Path("experiments/archive/frontier_consolidation/data/medium_near_analysis/medium_near_ws_summary.json")
A02_COMPACT_SUMMARY = pathlib.Path("experiments/archive/frontier_consolidation/data/medium_compact_analysis/medium_compact_ws_summary.json")
A01_SEQSAFE = ROOT / "training/data/cached_fineweb_seqsafe96_candidate/materialization_metadata.json"
LIVE_POOL = ROOT / "data/live_fineweb_relation_dense_pool/live_fineweb_relation_dense_pool_summary.json"
PUBLIC = ROOT / "data/public_component_tradeoffs/public_component_tradeoffs.json"

COLS7 = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_mean", "Reading"]
LEADER_KEYS = {
    "BLiMP": "BLiMP",
    "Supplement": "Supplement",
    "EWoK": "EWoK",
    "Entity": "Entity",
    "COMPS": "COMPS",
    "GlobalPIQA_mean": "GlobalPIQA",
    "Reading": "Reading",
}
CLEAN_QWEN = {
    "Overall": 41.34429066479573,
    "BLiMP": 66.84,
    "Supplement": 62.84,
    "EWoK": 50.19,
    "Entity": 25.76,
    "COMPS": 51.78,
    "GlobalPIQA_mean": 36.62,
    "SuperGLUE": 70.3086,
    "Reading": 7.76,
    "AoA": 0.0,
}


def load(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def f(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def diff(a: dict[str, Any], b: dict[str, Any], cols: list[str]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for c in cols:
        av = f(a.get(c)); bv = f(b.get(c))
        out[c] = None if av is None or bv is None else round(av - bv, 6)
    return out


def equal7(row: dict[str, Any], entity_key: str = "Entity") -> float | None:
    vals = []
    for c in ["BLiMP", "Supplement", "EWoK", entity_key, "COMPS", "GlobalPIQA_mean", "Reading"]:
        v = f(row.get(c))
        if v is None:
            return None
        vals.append(v)
    return sum(vals) / len(vals)


def get_summary_overall(path: pathlib.Path) -> dict[str, Any]:
    obj = load(path)
    # The compact-view summary stores numbers under summary/overall.
    return obj.get("summary", {}).get("overall", {})


def public_leader(pub: dict[str, Any]) -> dict[str, Any]:
    leader = {"Overall": pub.get("strict_small_leader", {}).get("overall", 41.8)}
    for r in pub.get("top15", []):
        if r.get("rank") == 1:
            for out_k, in_k in LEADER_KEYS.items():
                leader[out_k] = f(r.get(in_k))
            leader["SuperGLUE"] = f(r.get("SuperGLUE"))
            leader["AoA"] = f(r.get("AoA"))
            break
    # Fall back to recorded summary values if the public parse changed.
    defaults = {
        "BLiMP": 67.2, "Supplement": 56.01, "EWoK": 56.07, "Entity": 28.45,
        "COMPS": 53.57, "GlobalPIQA_mean": 39.67, "SuperGLUE": 69.79,
        "Reading": 5.42, "AoA": 0.0, "Overall": 41.8,
    }
    for k, v in defaults.items():
        if leader.get(k) is None:
            leader[k] = v
    return leader


def estimate_scale_requirements(live: dict[str, Any], target_source_words: int) -> dict[str, Any]:
    dense_yield = f(live.get("dense_word_yield_per_doc_word")) or 0.0
    strict_yield = f(live.get("strict_word_yield_per_doc_word")) or 0.0
    return {
        "target_source_words": target_source_words,
        "doc_words_needed_at_step015_dense_yield": None if dense_yield <= 0 else int(math.ceil(target_source_words / dense_yield)),
        "doc_words_needed_at_step015_strict_yield": None if strict_yield <= 0 else int(math.ceil(target_source_words / strict_yield)),
        "estimated_docs_at_step015_density": None if live.get("doc_words_scanned", 0) == 0 else int(math.ceil(target_source_words / max(dense_yield, 1e-9) / (live["doc_words_scanned"] / live["docs_scanned"]))),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    a02 = load(A02_NOAOA)
    a02_meta = load(A02_META)
    a01_meta = load(A01_SEQSAFE)
    live = load(LIVE_POOL)
    pub = load(PUBLIC)
    leader = public_leader(pub)
    near_summary = get_summary_overall(A02_NEAR_SUMMARY)
    compact_summary = get_summary_overall(A02_COMPACT_SUMMARY)

    # Normalize the compact-view table for comparison against clean-Qwen and the leader.
    a02_table = {}
    for name, row in a02.get("table", {}).items():
        norm = dict(row)
        if "Entity_full" in row:
            norm["Entity_full"] = row["Entity_full"]
        norm["equal7_fast_entity"] = equal7(norm, "Entity")
        norm["equal7_full_entity"] = equal7({**norm, "Entity_full_key_alias": norm.get("Entity_full")}, "Entity") if False else row.get("equal7_full_entity")
        # For full-Entity comparisons use the row's Entity_full as Entity.
        row_full = dict(norm); row_full["Entity"] = row.get("Entity_full", row.get("Entity"))
        norm["minus_cleanqwen_full_entity_components"] = diff(row_full, CLEAN_QWEN, COLS7)
        norm["minus_cleanqwen_equal7_full_entity"] = None if equal7(row_full) is None else round(equal7(row_full) - equal7(CLEAN_QWEN), 6)
        norm["minus_leader_full_entity_components"] = diff(row_full, leader, COLS7)
        a02_table[name] = norm

    seq_verify = a01_meta.get("verification", {})
    a01_current = {
        "fineweb_source_words": seq_verify.get("fineweb_words"),
        "fineweb_fraction": seq_verify.get("word_fractions", {}).get("seqsafe_fineweb"),
        "has_natural_lengthmatched_arm": True,
        "natural_arm_source_counts": seq_verify.get("source_word_counts_control"),
        "fineweb_source_arm_counts": seq_verify.get("source_word_counts_treatment"),
        "scientific_reading": "The pending A01 contrast already has a natural lengthmatched arm for source breadth, but no source-plus-view arm.",
    }

    # Existing view-generation resources.
    near_res = {
        "accepted_source_words": near_summary.get("accepted_source_words_whitespace"),
        "accepted_rewrite_words": near_summary.get("accepted_rewrite_words_whitespace"),
        "accepted_pair_words": near_summary.get("accepted_pair_words_whitespace"),
        "rewrite_to_source_ratio": near_summary.get("weighted_rewrite_to_source_ratio_whitespace"),
        "accepted_rate": near_summary.get("accepted_rate"),
        "content_recall_mean": near_summary.get("content_recall_stats", {}).get("mean"),
        "entity_recall_mean": near_summary.get("entity_recall_stats", {}).get("mean"),
        "number_recall_mean": near_summary.get("number_recall_stats", {}).get("mean"),
    }
    compact_res = {
        "accepted_source_words": compact_summary.get("accepted_source_words_whitespace"),
        "accepted_rewrite_words": compact_summary.get("accepted_rewrite_words_whitespace"),
        "accepted_pair_words": compact_summary.get("accepted_pair_words_whitespace"),
        "rewrite_to_source_ratio": compact_summary.get("weighted_rewrite_to_source_ratio_whitespace"),
        "accepted_rate": compact_summary.get("accepted_rate"),
        "content_recall_mean": compact_summary.get("content_recall_stats", {}).get("mean"),
        "entity_recall_mean": compact_summary.get("entity_recall_stats", {}).get("mean"),
        "number_recall_mean": compact_summary.get("number_recall_stats", {}).get("mean"),
    }

    # Candidate changed-block scales. These are planning calculations only; the
    # pending source-breadth trajectory decides which one is worth materializing.
    candidate_scales = []
    for changed in [1_750_000, 2_500_000, 3_500_000, 5_000_000]:
        near_source_need = int(round(changed / (1.0 + (near_res["rewrite_to_source_ratio"] or 0.93))))
        compact_source_need = int(round(changed / (1.0 + (compact_res["rewrite_to_source_ratio"] or 0.61))))
        candidate_scales.append({
            "changed_block_words": changed,
            "fraction_of_10M": round(changed / 10_000_000, 6),
            "near_view_source_words_needed_if_same_ratio": near_source_need,
            "compact_view_source_words_needed_if_same_ratio": compact_source_need,
            "live_dense_source_extraction_for_near": estimate_scale_requirements(live, near_source_need),
            "live_dense_source_extraction_for_compact": estimate_scale_requirements(live, compact_source_need),
            "common_filler_words_if_preserve_cleanqwen_pairs": 10_000_000 - 1_656_800 - changed,
            "fits_with_cleanqwen_pairs_preserved": (10_000_000 - 1_656_800 - changed) >= 0,
        })

    design = {
        "status": "NATURAL_ARM_FINEWEB_DESIGN_ANALYSIS",
        "inputs": {
            "a02_noaoa": str(A02_NOAOA),
            "a02_overlay_metadata": str(A02_META),
            "a01_pending_seqsafe_metadata": str(A01_SEQSAFE),
            "live_pool_summary": str(LIVE_POOL),
            "public_component_tradeoffs": str(PUBLIC),
        },
        "trusted_cleanqwen_reference": {**CLEAN_QWEN, "equal7_noaoa_full_entity": equal7(CLEAN_QWEN)},
        "public_leader_reference": leader,
        "a02_near_result_interpreted_against_natural_reference": {
            "near_view_minus_near_repeat": a02.get("contrasts", {}).get("near_view_minus_near_repeat"),
            "arms": a02_table,
            "reading": "A02 near-view improves strongly over the artificial near-repeat arm, but its full-Entity equal7 remains below the COMPACT_EXPERIENCE clean-Qwen natural allocation reference. The next scaled study must train a natural lengthmatched arm, not only repeat and view arms.",
        },
        "a01_current_pending_source_breadth_contrast": a01_current,
        "available_view_generation_resources": {
            "medium_near": near_res,
            "medium_compact": compact_res,
            "live_relation_dense_pool": {
                "docs_scanned": live.get("docs_scanned"),
                "doc_words_scanned": live.get("doc_words_scanned"),
                "strict_anchor_words": live.get("strict_anchor_words"),
                "relation_dense_words": live.get("relation_dense_words"),
                "strict_yield": live.get("strict_word_yield_per_doc_word"),
                "dense_yield": live.get("dense_word_yield_per_doc_word"),
                "important_quality_caveat": "Line-level samples still contain some web-advice, license, explore-further, and pronoun/context fragments; live scaling needs stricter source filtering before generation.",
            },
        },
        "candidate_scale_calculations": candidate_scales,
        "recommended_next_experiment_shape_after_s14": {
            "arms": [
                "A_natural_lengthmatched_cleanqwen_slice: inherited clean-Qwen qwen_pair_packed rows preserved; a coherent official/non-Qwen slice of the same word budget is repacked to the changed-block row lengths; no FineWeb, no source repetition.",
                "B_fineweb_source_only: same common filler and same changed-block word budget; selected FineWeb source rows only, arranged with the same row-length sequence as A.",
                "C_fineweb_source_plus_faithful_view: same common filler, same changed-block word budget, and preferably the same FineWeb source set as B; one faithful near or compact view is adjacent to its source.",
            ],
            "why_three_arms": "A→B reads source breadth, B→C reads generated-view utility, and A→C reads net movement over the strong natural allocation rather than over artificial repetition.",
            "compact_warning": "If compact views are used with reinvested extra sources, add the matching source-only reinvest arm or the source-diversity and view effects become inseparable.",
            "scale_from_pending_s14": {
                "strong_source_breadth": "Use 3.5M or larger changed block if protected columns are not damaged; train A/B/C together.",
                "small_positive_source_breadth": "Build cleaner live relation-dense sources and a 2.5M A/B/C pilot before a very large run.",
                "flat_or_damaging_source_only": "Do not extend cached source-only repetition; use A/B/C at moderate scale only if view evidence and source quality are improved enough to test a different mechanism.",
            },
        },
    }

    out_json = OUT_DIR / "natural_arm_scaled_fineweb_design_analysis.json"
    out_json.write_text(json.dumps(design, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    nv = design["a02_near_result_interpreted_against_natural_reference"]["arms"].get("near_view", {})
    nr = design["a02_near_result_interpreted_against_natural_reference"]["arms"].get("near_repeat", {})
    nvd = nv.get("minus_cleanqwen_full_entity_components", {})
    nrd = nr.get("minus_cleanqwen_full_entity_components", {})

    lines: list[str] = []
    lines.append("# research natural-arm FineWeb scale design\n\n")
    lines.append("## Design Correction\n\n")
    lines.append("A generated view can beat artificial same-source repetition and still fail to beat the trusted natural allocation. That happened in A01's SimpleWiki semantic-view endpoint, and the same risk is visible in A02's near-view FineWeb result. Therefore the next scaled FineWeb study must include a strong natural lengthmatched arm in the same materialization family.\n\n")
    lines.append("## A02 near-view result read against clean-Qwen\n\n")
    lines.append(f"A02 `near_view - near_repeat` gives Entity_full +{a02.get('contrasts', {}).get('near_view_minus_near_repeat', {}).get('Entity_full')} and equal7_full_entity +{a02.get('contrasts', {}).get('near_view_minus_near_repeat', {}).get('equal7_full_entity')}. But this is a repeat-control contrast.\n\n")
    lines.append(f"Against the COMPACT_EXPERIENCE clean-Qwen natural reference, `near_view` has full-Entity equal7 delta {nv.get('minus_cleanqwen_equal7_full_entity')}; component deltas: BLiMP {nvd.get('BLiMP')}, Supplement {nvd.get('Supplement')}, EWoK {nvd.get('EWoK')}, Entity {nvd.get('Entity')}, COMPS {nvd.get('COMPS')}, GlobalPIQA {nvd.get('GlobalPIQA_mean')}, Reading {nvd.get('Reading')}.\n\n")
    lines.append(f"`near_repeat` against the same reference has full-Entity equal7 delta {nr.get('minus_cleanqwen_equal7_full_entity')}; this shows that the repeat arm is itself a weak baseline for SOTA-facing judgment.\n\n")
    lines.append("## What A01's pending seqsafe96 run already tests\n\n")
    lines.append(f"The pending `s14_t33_tool1` source-breadth contrast has a natural lengthmatched control: qwen pairs {a01_current['natural_arm_source_counts'].get('qwen_pair_packed'):,} words, official lengthmatched block {a01_current['natural_arm_source_counts'].get('official_lengthmatched_to_seqsafe_fineweb'):,} words, identical official tail {a01_current['natural_arm_source_counts'].get('official_identical_tail_after_seqsafe_fineweb_block'):,} words. Its FineWeb source-only arm replaces the lengthmatched block with {a01_current['fineweb_source_words']:,} cached FineWeb words. It does not contain a source-plus-view arm.\n\n")
    lines.append("## Next experiment shape\n\n")
    lines.append("After the source-breadth result becomes available, the next H100 study should be one matched family with:\n\n")
    for arm in design["recommended_next_experiment_shape_after_s14"]["arms"]:
        lines.append(f"- {arm}\n")
    lines.append("\nThis separates source breadth (A→B), faithful-view utility (B→C), and net movement over the strong coordinate (A→C). A rewrite-versus-repeat-only study cannot answer the SOTA question.\n\n")
    lines.append("If compact views are chosen and the saved word budget is reinvested into additional sources, include the source-only reinvest counterpart as a fourth arm; otherwise source diversity and generated-view effects are mixed together.\n\n")
    lines.append("## Scale and source-resource calculation\n\n")
    lines.append(f"Current accepted A02 near resource: {near_res['accepted_source_words']:,} source words + {near_res['accepted_rewrite_words']:,} rewrite words = {near_res['accepted_pair_words']:,} pair words. Current accepted compact resource: {compact_res['accepted_source_words']:,} source + {compact_res['accepted_rewrite_words']:,} rewrite = {compact_res['accepted_pair_words']:,} pair words. These are useful but smaller than a 2.5M–3.5M changed-block study.\n\n")
    lines.append(f"research live relation-dense extraction found {live.get('relation_dense_words'):,} dense words from {live.get('doc_words_scanned'):,} scanned doc words (yield {100*(live.get('dense_word_yield_per_doc_word') or 0):.2f}%). Samples still show some web-fragment noise, so stricter source filtering must precede any large generation.\n\n")
    lines.append("| changed-block words | near source needed | compact source needed | dense doc words for near source | dense doc words for compact source | common filler if qwen pairs preserved |\n")
    lines.append("|---:|---:|---:|---:|---:|---:|\n")
    for srow in candidate_scales:
        lines.append(f"| {srow['changed_block_words']:,} | {srow['near_view_source_words_needed_if_same_ratio']:,} | {srow['compact_view_source_words_needed_if_same_ratio']:,} | {srow['live_dense_source_extraction_for_near']['doc_words_needed_at_step015_dense_yield']:,} | {srow['live_dense_source_extraction_for_compact']['doc_words_needed_at_step015_dense_yield']:,} | {srow['common_filler_words_if_preserve_cleanqwen_pairs']:,} |\n")
    lines.append("\n## How to use the pending A01 source-breadth result\n\n")
    lines.append("- If the 17.5% source-only arm gives a substantial EWoK+Entity+COMPS+GlobalPIQA gain without erasing BLiMP/Supplement/Reading, build a larger natural/source/source+view family, likely 3.5M changed-block words.\n")
    lines.append("- If it is small positive, do not call source breadth dead; first improve live FineWeb source quality and test a 2.5M natural/source/source+view family.\n")
    lines.append("- If it is flat or damaging, do not extend cached source-only repetition. A moderate natural/source/source+view study is only worth training after source quality or view mechanism has changed.\n\n")
    lines.append(f"JSON: `{out_json}`\n")
    NOTE.write_text("".join(lines), encoding="utf-8")

    print(json.dumps({"status": design["status"], "out_json": str(out_json), "note": str(NOTE), "near_view_minus_cleanqwen_equal7_full_entity": nv.get("minus_cleanqwen_equal7_full_entity")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
