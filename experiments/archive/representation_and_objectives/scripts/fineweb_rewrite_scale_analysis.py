#!/usr/bin/env python3
"""research: route-relevant scale analysis for FineWeb source-by-rewrite evidence.

Reads frontier_consolidation high-anchor generation outputs and local inherited score files.
Writes a quantitative JSON plus a research-facing note.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean, median

ROOT = Path("experiments/archive/representation_and_objectives")
OUT_DIR = ROOT / "data" / "fineweb_rewrite_scale"
OUT_JSON = OUT_DIR / "fineweb_rewrite_scale_analysis.json"
OUT_NOTE = (ROOT / 'notes'.parents[3] / 'research/notes/representation_and_objectives/fineweb_rewrite_scale_and_route.md')

PEER = Path("experiments/archive/frontier_consolidation")
GEN_ROWS = PEER / "data/high_anchor_fineweb_generation_analysis/fineweb_high_anchor_generation_rows.jsonl"
ACCEPTED = PEER / "data/high_anchor_fineweb_generation_analysis/fineweb_high_anchor_accepted_rewrites.jsonl"
HIGH_ANCHOR_META = PEER / "data/fineweb_high_anchor_slice/fineweb_high_anchor_slice_metadata.json"
REPAIR_META = PEER / "data/fineweb_source_by_rewrite_repair/fineweb_source_by_rewrite_repair_metadata.json"
TINY_FACTOR_META = PEER / "data/fineweb_factor_contrast/fineweb_factor_contrast_metadata.json"

A01_POOL = ROOT / "data/fineweb_source_pool_audit/fineweb_source_pool_audit.json"
QWEN_CLEAN = Path("experiments/archive/compact_experience/data/full_eval/per_target/qwen_clean_aligned.json")
ORIG_DUP = Path("experiments/archive/compact_experience/data/mechanism_eval/per_target/selected_original_dup_all.json")
MIX25 = Path("experiments/archive/compact_experience/data/full_overall_eval/per_target/mix25_16k_seed43.json")
LEADERBOARD = ROOT / "data/babylm2026_live_surface/strict_small_top30.json"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def stat(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return {"n": 0}
    vals_sorted = sorted(vals)
    def q(p):
        if len(vals_sorted) == 1:
            return vals_sorted[0]
        pos = p * (len(vals_sorted) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return vals_sorted[lo]
        return vals_sorted[lo] * (hi - pos) + vals_sorted[hi] * (pos - lo)
    return {
        "n": len(vals_sorted),
        "min": vals_sorted[0],
        "p05": q(0.05),
        "mean": mean(vals_sorted),
        "median": median(vals_sorted),
        "p95": q(0.95),
        "max": vals_sorted[-1],
        "sum": sum(vals_sorted),
    }


def column_scores(j):
    return j["official_overall"]["scores"]


def score_delta(a, b):
    cols = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "SuperGLUE", "GlobalPIQA", "Reading", "AoA"]
    sa = column_scores(a)
    sb = column_scores(b)
    out = {}
    for c in cols:
        out[c] = sa[c] - sb[c]
    out["Overall"] = a["official_overall"]["Overall"] - b["official_overall"]["Overall"]
    out["Ksum_EWoK_Entity_COMPS_GlobalPIQA"] = sum(out[c] for c in ["EWoK", "Entity", "COMPS", "GlobalPIQA"])
    out["preserve_sum_Supplement_Reading_SuperGLUE"] = sum(out[c] for c in ["Supplement", "Reading", "SuperGLUE"])
    return out


def leader_row():
    rows = load_json(LEADERBOARD)
    for r in rows:
        if r.get("Model_plain") == "wwm_curriculum_simplification_40k":
            return r
    return None


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gen_rows = load_jsonl(GEN_ROWS)
    acc_rows = load_jsonl(ACCEPTED)
    high_anchor = load_json(HIGH_ANCHOR_META)
    repair = load_json(REPAIR_META)
    tiny = load_json(TINY_FACTOR_META)
    pool = load_json(A01_POOL)
    qwen = load_json(QWEN_CLEAN)
    orig = load_json(ORIG_DUP)
    mix25 = load_json(MIX25)

    # Similarity/rewrite strength for generated rows.
    accepted = [r for r in gen_rows if r.get("accepted_for_next_construction")]
    rejected = [r for r in gen_rows if not r.get("accepted_for_next_construction")]
    hard = Counter(reason for r in rejected for reason in r.get("hard_reasons", []))
    soft_all = Counter(flag for r in gen_rows for flag in r.get("soft_flags", []))
    soft_acc = Counter(flag for r in accepted for flag in r.get("soft_flags", []))
    by_domain = defaultdict(lambda: {"n": 0, "accepted": 0, "exact_copy": 0, "near_copy": 0, "changed_enough": 0, "source_words": 0, "accepted_pair_words": 0})

    enriched = []
    for r in gen_rows:
        s = norm_text(r.get("source_text", ""))
        t = norm_text(r.get("rewrite_text", ""))
        sim = SequenceMatcher(None, s, t).ratio() if s or t else 0.0
        exact = s == t
        near = "near_copy_view" in r.get("soft_flags", []) or sim >= 0.94
        changed = bool(r.get("accepted_for_next_construction")) and (not exact) and sim <= 0.92 and r.get("length_ratio", 999) <= 1.05
        e = dict(r)
        e["char_similarity"] = sim
        e["exact_copy"] = exact
        e["near_copy_by_flag_or_similarity"] = near
        e["changed_enough_for_second_view"] = changed
        enriched.append(e)
        domains = r.get("domain_hits") or ["no_domain"]
        for d in domains:
            bd = by_domain[d]
            bd["n"] += 1
            bd["source_words"] += int(r.get("source_words") or 0)
            if r.get("accepted_for_next_construction"):
                bd["accepted"] += 1
                bd["accepted_pair_words"] += int(r.get("pair_words") or 0)
                if exact:
                    bd["exact_copy"] += 1
                if near:
                    bd["near_copy"] += 1
                if changed:
                    bd["changed_enough"] += 1

    acc_enriched = [r for r in enriched if r.get("accepted_for_next_construction")]
    exact_acc = [r for r in acc_enriched if r["exact_copy"]]
    near_acc = [r for r in acc_enriched if r["near_copy_by_flag_or_similarity"]]
    changed_acc = [r for r in acc_enriched if r["changed_enough_for_second_view"]]
    compressed_acc = [r for r in acc_enriched if (not r["exact_copy"]) and r.get("length_ratio", 999) <= 0.90]

    source_words_all = sum(int(r.get("source_words") or 0) for r in gen_rows)
    source_words_acc = sum(int(r.get("source_words") or 0) for r in acc_enriched)
    pair_words_acc = sum(int(r.get("pair_words") or 0) for r in acc_enriched)
    pair_per_all_source = pair_words_acc / source_words_all
    pair_per_accepted_source = pair_words_acc / source_words_acc
    accepted_source_fraction = source_words_acc / source_words_all
    changed_source_words = sum(int(r.get("source_words") or 0) for r in changed_acc)
    changed_pair_words = sum(int(r.get("pair_words") or 0) for r in changed_acc)
    compressed_pair_words = sum(int(r.get("pair_words") or 0) for r in compressed_acc)

    by_domain_out = {}
    for d, bd in sorted(by_domain.items()):
        n = bd["n"]
        accepted_n = bd["accepted"]
        by_domain_out[d] = dict(bd)
        by_domain_out[d].update({
            "accepted_rate": accepted_n / n if n else None,
            "exact_copy_rate_among_accepted": bd["exact_copy"] / accepted_n if accepted_n else None,
            "near_copy_rate_among_accepted": bd["near_copy"] / accepted_n if accepted_n else None,
            "changed_enough_rate_among_accepted": bd["changed_enough"] / accepted_n if accepted_n else None,
        })

    scale_inputs = {
        "peer_high_anchor_full": {
            "source_words": high_anchor["kept"]["words"],
            "rows": high_anchor["kept"]["rows"],
            "docs": high_anchor["kept"]["unique_docs"],
        },
        "peer_high_precision_full": {
            "source_words": repair["high_precision_from_strict_factual"]["words"],
            "rows": repair["high_precision_from_strict_factual"]["rows"],
            "docs": repair["high_precision_from_strict_factual"]["unique_docs"],
        },
        "peer_medium_repaired_full": {
            "source_words": repair["medium_repaired_from_strict_plus_balanced"]["words"],
            "rows": repair["medium_repaired_from_strict_plus_balanced"]["rows"],
            "docs": repair["medium_repaired_from_strict_plus_balanced"]["unique_docs"],
        },
        "a01a02_priority_dedup_all_available": {
            "source_words": pool["canonical_priority_summary"]["words"],
            "rows": pool["canonical_priority_summary"]["rows"],
            "docs": pool["canonical_priority_summary"]["unique_docs"],
        },
        "a01_seqsafe96_cached_source_block": {
            "source_words": 1753280,
            "rows": 21916,
            "docs": 5566,
        },
    }

    scale_estimates = {}
    for name, info in scale_inputs.items():
        sw = info["source_words"]
        est_pair = sw * pair_per_all_source
        est_accepted_source = sw * accepted_source_fraction
        est_changed_pair = sw * (changed_pair_words / source_words_all)
        scale_estimates[name] = {
            **info,
            "estimated_accepted_pair_words_using_pilot": est_pair,
            "estimated_accepted_source_words_using_pilot": est_accepted_source,
            "estimated_substantive_pair_words_using_pilot": est_changed_pair,
            "corpus_fraction_accepted_pair": est_pair / 10_000_000,
            "ten_pass_exposure_words": est_pair * 10,
            "relative_to_peer_tiny_selected_pair_words": est_pair / tiny["selected_pair_words"],
            "relative_to_compact_experience_clean_qwen_pair_words": est_pair / 1_656_800,
        }

    inherited = {
        "qwen_clean_minus_original_dup": score_delta(qwen, orig),
        "mix25_minus_qwen_clean": score_delta(mix25, qwen),
        "qwen_clean_overall": qwen["official_overall"],
        "original_dup_overall": orig["official_overall"],
        "mix25_overall": mix25["official_overall"],
        "strict_small_leader_row": leader_row(),
    }

    summary = {
        "status": "FINEWEB_REWRITE_SCALE_ANALYZED",
        "peer_generation": {
            "prompt_count": len(gen_rows),
            "accepted_count": len(acc_enriched),
            "accepted_rate": len(acc_enriched) / len(gen_rows),
            "source_words_all": source_words_all,
            "source_words_accepted": source_words_acc,
            "pair_words_accepted": pair_words_acc,
            "accepted_source_fraction": accepted_source_fraction,
            "pair_words_per_all_source_word": pair_per_all_source,
            "pair_words_per_accepted_source_word": pair_per_accepted_source,
            "exact_copy_accepted_count": len(exact_acc),
            "exact_copy_accepted_rate": len(exact_acc) / len(acc_enriched),
            "near_copy_accepted_count": len(near_acc),
            "near_copy_accepted_rate": len(near_acc) / len(acc_enriched),
            "changed_enough_accepted_count": len(changed_acc),
            "changed_enough_accepted_rate": len(changed_acc) / len(acc_enriched),
            "changed_enough_source_words": changed_source_words,
            "changed_enough_pair_words": changed_pair_words,
            "changed_enough_pair_fraction_of_all_source_words": changed_pair_words / source_words_all,
            "compressed_nonexact_accepted_count_len_le_0p90": len(compressed_acc),
            "compressed_nonexact_pair_words_len_le_0p90": compressed_pair_words,
            "length_ratio_stats_accepted": stat([r.get("length_ratio") for r in acc_enriched]),
            "char_similarity_stats_accepted": stat([r.get("char_similarity") for r in acc_enriched]),
            "hard_reason_counts_rejected": dict(hard.most_common()),
            "soft_flag_counts_all": dict(soft_all.most_common()),
            "soft_flag_counts_accepted": dict(soft_acc.most_common()),
            "by_domain": by_domain_out,
            "sample_exact_copies": [
                {"id": r.get("prompt_id"), "source_text": r.get("source_text"), "rewrite_text": r.get("rewrite_text"), "domain_hits": r.get("domain_hits", [])}
                for r in exact_acc[:8]
            ],
            "sample_changed_views": [
                {"id": r.get("prompt_id"), "source_text": r.get("source_text"), "rewrite_text": r.get("rewrite_text"), "length_ratio": r.get("length_ratio"), "char_similarity": r.get("char_similarity"), "domain_hits": r.get("domain_hits", [])}
                for r in changed_acc[:8]
            ],
        },
        "scale_estimates": scale_estimates,
        "tiny_peer_factor_contrast": {
            "selected_pairs": tiny["selected_pairs"],
            "selected_pair_words": tiny["selected_pair_words"],
            "changed_block_fraction": tiny["audit"]["changed_block_fraction"],
            "selected_source_words": tiny["selected_source_words"],
            "selected_rewrite_words": tiny["selected_rewrite_words"],
            "relative_to_compact_experience_clean_qwen_pair_block": tiny["selected_pair_words"] / 1_656_800,
        },
        "inherited_score_context": inherited,
        "source_files": {
            "gen_rows": str(GEN_ROWS),
            "accepted": str(ACCEPTED),
            "high_anchor_meta": str(HIGH_ANCHOR_META),
            "repair_meta": str(REPAIR_META),
            "tiny_factor_meta": str(TINY_FACTOR_META),
            "a01_pool_json": str(A01_POOL),
            "qwen_clean": str(QWEN_CLEAN),
            "orig_dup": str(ORIG_DUP),
            "mix25": str(MIX25),
        },
    }

    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    pg = summary["peer_generation"]
    se = scale_estimates
    qd = inherited["qwen_clean_minus_original_dup"]
    md = inherited["mix25_minus_qwen_clean"]

    lines = []
    lines.append("# research FineWeb rewrite scale and route")
    lines.append("")
    lines.append("## What the peer high-anchor pilot actually shows")
    lines.append("")
    lines.append(f"frontier_consolidation generated {pg['prompt_count']} high-anchor FineWeb simplification prompts and accepted {pg['accepted_count']} ({pg['accepted_rate']:.3f}). The accepted rows preserve entities and numbers well, but many accepted rows are not a new expression in a strong sense: exact copies among accepted = {pg['exact_copy_accepted_count']} ({pg['exact_copy_accepted_rate']:.3f}); near copies by flag or high string similarity = {pg['near_copy_accepted_count']} ({pg['near_copy_accepted_rate']:.3f}); accepted rows with a visibly changed but still source-faithful second view by the current text-similarity rule = {pg['changed_enough_accepted_count']} ({pg['changed_enough_accepted_rate']:.3f}).")
    lines.append("")
    lines.append(f"The pilot's accepted pair mass is {pg['pair_words_accepted']:,} words from {pg['source_words_all']:,} prompt-source words, so a direct scale-up gives {pg['pair_words_per_all_source_word']:.3f} accepted source+rewrite corpus words per input source word. The more substantive changed-view subset gives {pg['changed_enough_pair_words']:,} pair words, or {pg['changed_enough_pair_fraction_of_all_source_words']:.3f} per input source word. This means rewrite quality is safe enough to use, but the first prompt is often too conservative to be the main learning mechanism by itself.")
    lines.append("")
    lines.append("## Scale estimates")
    lines.append("")
    lines.append("| source set | source words | estimated accepted pair words | corpus fraction | relative to clean-Qwen 1.6568M pair block | estimated substantive pair words |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for key in ["peer_high_anchor_full", "peer_high_precision_full", "peer_medium_repaired_full", "a01a02_priority_dedup_all_available", "a01_seqsafe96_cached_source_block"]:
        v = se[key]
        lines.append(f"| {key} | {v['source_words']:,.0f} | {v['estimated_accepted_pair_words_using_pilot']:,.0f} | {100*v['corpus_fraction_accepted_pair']:.2f}% | {v['relative_to_compact_experience_clean_qwen_pair_words']:.3f}x | {v['estimated_substantive_pair_words_using_pilot']:,.0f} |")
    lines.append("")
    lines.append(f"The frontier_consolidation factorized corpus built from the first pilot uses only {tiny['selected_pair_words']:,} changed-block words ({100*tiny['audit']['changed_block_fraction']:.3f}% of the 10M pool), which is {tiny['selected_pair_words']/1_656_800:.3f}x the inherited clean-Qwen pair block. It is useful as a construction smoke test but should not consume a full 100M-word training slot as a main SOTA probe unless deliberately testing an ultra-small perturbation.")
    lines.append("")
    lines.append("## Inherited score context")
    lines.append("")
    lines.append(f"The inherited clean-Qwen pair block moved official Overall by {qd['Overall']:.3f} versus the selected-original-duplicate control with a 1.6568M-word paired block. Its task movement was Ksum={qd['Ksum_EWoK_Entity_COMPS_GlobalPIQA']:.3f}, Supplement+Reading+SuperGLUE sum={qd['preserve_sum_Supplement_Reading_SuperGLUE']:.3f}, and GlobalPIQA={qd['GlobalPIQA']:.3f}. Thus same-window official-source rewriting is a real sample-efficiency signal but not enough for the current 41.8 target, partly because GlobalPIQA moved down.")
    lines.append("")
    lines.append(f"The inherited mix25 endpoint remains numerically closer to the leader (Overall {mix25['official_overall']['Overall']:.4f}) but lacks the strict checkpoint ladder needed for a direct submit-ready AoA package. Relative to clean-Qwen it has Overall delta {md['Overall']:.3f}, EWoK {md['EWoK']:.3f}, GlobalPIQA {md['GlobalPIQA']:.3f}, Reading {md['Reading']:.3f}, and Entity {md['Entity']:.3f}; it improves some leader-gap columns while harming Entity and SuperGLUE. It is a useful signal about mixture direction, not the next artifact to submit.")
    lines.append("")
    lines.append("## Route implication")
    lines.append("")
    lines.append("The next large H100 work should not be the tiny 16.6k-word peer contrast. It should either (a) use the existing A01 seqsafe96 1.75M-word FineWeb source-only replacement to test broad source breadth at a scale comparable to the clean-Qwen intervention, or (b) first expand/repair the high-anchor/high-precision rewrite pool so the source+rewrite arm reaches a nontrivial unique-word scale. The existing seqsafe96 block is a source-breadth candidate, not a clean rewrite substrate; its cached 96-word chunks include residual web/document artifacts and would need sentence-level or paragraph-level repair before simplification. The cleanest scientific path is still three arms: protected slot control, FineWeb source repetition, and FineWeb source plus accepted faithful rewrite. The generation prompt should be improved toward information-dense faithful compression because the first high-anchor prompt produced many near-copy views.")
    lines.append("")
    lines.append("The pending semantic-view comparison remains decisive for whether source-conservative second views deserve further investment. Once it returns, read the component trajectories before any new H100 launch. If same-source views do not move EWoK/Entity/COMPS/GlobalPIQA, prioritize the scaled source-breadth contrast. If they do move those columns without damaging Supplement/Reading, construct a scaled three-arm FineWeb design using accepted high-anchor/high-precision rewrites plus a source-repeat arm.")
    lines.append("")
    lines.append(f"JSON: `{OUT_JSON}`")
    OUT_NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": summary["status"],
        "out_json": str(OUT_JSON),
        "out_note": str(OUT_NOTE),
        "accepted_rate": pg["accepted_rate"],
        "exact_copy_accepted_rate": pg["exact_copy_accepted_rate"],
        "near_copy_accepted_rate": pg["near_copy_accepted_rate"],
        "changed_enough_rate": pg["changed_enough_accepted_rate"],
        "high_precision_est_pair_words": se["peer_high_precision_full"]["estimated_accepted_pair_words_using_pilot"],
        "priority_est_pair_words": se["a01a02_priority_dedup_all_available"]["estimated_accepted_pair_words_using_pilot"],
        "tiny_block_fraction": tiny["audit"]["changed_block_fraction"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
