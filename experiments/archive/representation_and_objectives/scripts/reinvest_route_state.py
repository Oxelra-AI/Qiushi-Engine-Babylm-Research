#!/usr/bin/env python3
"""Durable research route state for compact-view reinvest.

This CPU-only synthesis records the rationale for moving from the older cached
FineWeb source-breadth branch and the core-only replication controller to the
SOTA-facing compact_view_reinvest endpoint. It does not read write-owned full-eval
outputs or infer results from running tasks.
"""
from __future__ import annotations

import json
import pathlib
from typing import Any

USER_ROOT = pathlib.Path(".").resolve()
A01 = USER_ROOT / "experiments/archive" / 'representation_and_objectives'
A02 = USER_ROOT / "experiments/archive" / 'frontier_consolidation'
OUT_DIR = A01 / "data" / "reinvest_route_state"
OUT_JSON = OUT_DIR / "reinvest_route_state.json"
NOTE = (USER_ROOT / 'research/notes/representation_and_objectives/reinvest_route_state.md')

DENSITY_ANALYSIS = A01 / "data" / "compact_view_density_review" / "compact_view_density_review.json"
FAST_REINVEST = A02 / "data" / "density_noaoa_eval_reinvest" / "density_noaoa_eval_summary.json"
OVERLAY_META = A02 / "data" / "density_cleanqwen_overlay_medium_riskhard" / "density_cleanqwen_rowholdout_overlay_metadata.json"
DENSITY_META = A02 / "data" / "density_core_reinvestment_medium_riskhard" / "density_core_reinvestment_metadata.json"
TRAINER_EXPOSURE = A02 / "data" / "trainer_exact_token_exposure_measurement" / "trainer_exact_token_exposure_summary.json"


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def pct_counts(counts: dict[str, Any]) -> dict[str, float]:
    total = sum(float(v) for v in counts.values()) or 1.0
    return {k: round(float(v) / total, 6) for k, v in sorted(counts.items())}


def roundf(x: Any, n: int = 6) -> Any:
    if isinstance(x, (int, float)):
        return round(float(x), n)
    return x


def main() -> None:
    density_analysis = read_json(DENSITY_ANALYSIS)
    fast = read_json(FAST_REINVEST)
    overlay = read_json(OVERLAY_META)
    density = read_json(DENSITY_META)
    exposure = read_json(TRAINER_EXPOSURE)

    ps = overlay.get("pair_summaries", {})
    core = ps.get("compact_core_neutral", {})
    reinvest = ps.get("compact_reinvest", {})
    added = density.get("compact_added_summary", {})

    core_domains = core.get("domain_hit_counts", {})
    reinvest_domains = reinvest.get("domain_hit_counts", {})
    added_domains = added.get("domain_hit_counts", {})

    seven = density_analysis.get("seven_column_summaries", {})
    comp_moves = density_analysis.get("component_movements", {})
    fast_table = fast.get("table", {})
    contrasts = fast.get("contrasts", {})

    route = {
        "status": "REINVEST_ROUTE_STATE",
        "sources": {
            "density_analysis_review": str(DENSITY_ANALYSIS),
            "fast_reinvest_summary": str(FAST_REINVEST),
            "overlay_metadata": str(OVERLAY_META),
            "density_metadata": str(DENSITY_META),
            "trainer_exact_exposure": str(TRAINER_EXPOSURE),
        },
        "scientific_pivot": {
            "mechanistic_anchor": "compact_view_core seed43022 vs same-seed compact_repeat_core: anchor-preserving compact generated views improve the fast task-family surface beyond exact repetition and beyond the inherited natural coordinate.",
            "sota_facing_endpoint": "compact_view_reinvest seed43022: same compact-view principle with the saved word budget used for additional distinct FineWeb source-view packets.",
            "reason_for_pivot_from_old_tasks": "The fast reinvest surface is stronger than core and needs only SuperGLUE+AoA 66.18 for Overall 41.8; older cached FineWeb source-breadth tasks were unresolved lower-priority evidence and were cancelled before producing BabyLM scores to free H100 attention for the stronger endpoint.",
        },
        "fast_surface": {
            "compact_view_core": fast_table.get("compact_view_core"),
            "compact_view_reinvest": fast_table.get("compact_view_reinvest"),
            "reinvest_minus_core": contrasts.get("compact_view_reinvest_minus_compact_view_core"),
            "reinvest_seven_summary": seven.get("compact_view_reinvest"),
            "core_seven_summary": seven.get("compact_view_core"),
            "reinvest_minus_compact_experience_clean_qwen": comp_moves.get("compact_view_reinvest_minus_compact_experience_clean_qwen"),
            "reinvest_minus_visible_leader_fast_columns": comp_moves.get("compact_view_reinvest_minus_visible_leader_fast_columns"),
        },
        "fixed_budget_compression_reinvestment": {
            "core_pairs": core.get("pairs"),
            "reinvest_pairs": reinvest.get("pairs"),
            "added_pairs": added.get("pairs"),
            "pair_multiplier_reinvest_vs_core": roundf(float(reinvest.get("pairs", 0)) / float(core.get("pairs", 1))),
            "core_source_words": core.get("source_words"),
            "reinvest_source_words": reinvest.get("source_words"),
            "added_source_words": added.get("source_words"),
            "core_rewrite_words": core.get("rewrite_words"),
            "reinvest_rewrite_words": reinvest.get("rewrite_words"),
            "added_rewrite_words": added.get("rewrite_words"),
            "core_pair_words": core.get("pair_words"),
            "reinvest_pair_words": reinvest.get("pair_words"),
            "core_neutral_topup_words": core.get("neutral_cleanqwen_topup_words_inside_changed_block"),
            "reinvest_neutral_topup_words": reinvest.get("neutral_cleanqwen_topup_words_inside_changed_block"),
            "core_rewrite_to_source_ratio_weighted": core.get("rewrite_to_source_ratio_weighted"),
            "reinvest_rewrite_to_source_ratio_weighted": reinvest.get("rewrite_to_source_ratio_weighted"),
            "core_content_recall_mean": core.get("content_recall_stats", {}).get("mean"),
            "reinvest_content_recall_mean": reinvest.get("content_recall_stats", {}).get("mean"),
            "core_entity_recall_mean": core.get("entity_recall_stats", {}).get("mean"),
            "reinvest_entity_recall_mean": reinvest.get("entity_recall_stats", {}).get("mean"),
            "core_number_recall_mean": core.get("number_recall_stats", {}).get("mean"),
            "reinvest_number_recall_mean": reinvest.get("number_recall_stats", {}).get("mean"),
            "core_near_copy_like_soft_count": core.get("near_copy_like_soft_count"),
            "reinvest_near_copy_like_soft_count": reinvest.get("near_copy_like_soft_count"),
            "added_domain_counts": added_domains,
            "core_domain_fractions": pct_counts(core_domains),
            "reinvest_domain_fractions": pct_counts(reinvest_domains),
            "added_domain_fractions": pct_counts(added_domains),
        },
        "trainer_exact_exposure_reading": density_analysis.get("trainer_exact_exposure_reading"),
        "running_step025_tasks": {
            "s25_t24_tool1": {
                "purpose": "A01-owned full official-compatible evaluation of already-trained A02 compact_view_reinvest seed43022",
                "output_root": "experiments/archive/representation_and_objectives/data/compact_reinvest_full_eval",
                "per_target_json": "experiments/archive/representation_and_objectives/data/compact_reinvest_full_eval/per_target/compact_view_reinvest.json",
                "summary_json": "experiments/archive/representation_and_objectives/data/compact_reinvest_full_eval/compact_reinvest_full_eval_summary.json",
            },
            "s25_t33_tool1": {
                "purpose": "Train compact_view_reinvest seed43122 from the frozen A02 corpus and run fast no-AoA endpoint screen",
                "run_dir": "experiments/archive/representation_and_objectives/training/runs/repl_compact_view_reinvest_seed43122",
                "output_root": "experiments/archive/representation_and_objectives/data/compact_reinvest_seed43122_fast",
            },
            "peer_running_core_full_eval": {
                "purpose": "A02 full official-compatible compact_view_core seed43022 evaluation",
                "output_root": "experiments/archive/frontier_consolidation/data/density_full_eval",
                "per_target_json": "experiments/archive/frontier_consolidation/data/density_full_eval/per_target/compact_view_core.json",
            },
        },
        "cancelled_lower_priority_tasks": {
            "s23_t22_tool1": "Cancelled in research before training; it had only verified the core-view frozen data hash and was waiting on A02 core full eval. It is not BabyLM evidence.",
            "s20_t25_tool1": "Cancelled/failed in research while only waiting for absent cached-FineWeb no-AoA summary. It produced no FineWeb full score.",
            "s17_t37_tool1": "Cancelled in research after preflight and training launch for cached-FineWeb seqsafe96; no repaired no-AoA summary or score should be inferred. Any partial run directories are not route evidence.",
        },
        "result_dependent_next_work": {
            "if_reinvest_full_crosses_41p8_and_seed43122_fast_holds": "Prioritize submit-ready packaging/checks, full evaluation for the seed43122 reinvest endpoint if needed, and same-seed repeat or lengthmatched controls only to explain mechanism, not to delay preserving the SOTA-facing model.",
            "if_reinvest_full_crosses_41p8_but_seed43122_collapses": "Treat seed43022 as potentially fragile; train a small targeted additional seed or inspect which fast component changed before claiming a stable principle.",
            "if_reinvest_full_misses_due_to_superglue_or_aoa_but_fast_hard_components_survive": "Keep the density route alive and test repair specifically for SuperGLUE/AoA or combine with validated masking/tokenization factors; do not discard the compact data mechanism.",
            "if_reinvest_full_loses_hard_components": "Analyze which component failed relative to fast screen/core and rebuild the source/view selection rather than extending the same corpus unchanged.",
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(route, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# research reinvest route state",
        "",
        "## Main scientific line",
        "",
        "The mechanistic anchor remains compact generated views on the shared source core, but the SOTA-facing endpoint is now `compact_view_reinvest`: the same anchor-preserving compression with the recovered word budget used for additional compact source-view packets.",
        "",
        "## Fast evidence",
        "",
    ]
    rv = fast_table.get("compact_view_reinvest", {})
    rc = contrasts.get("compact_view_reinvest_minus_compact_view_core", {})
    lines += [
        f"- `compact_view_reinvest` fast seven-column mean: {rv.get('equal7_mean'):.4f}; full-Entity mean: {rv.get('equal7_full_entity'):.4f}.",
        f"- Reinvest minus core: equal7 {rc.get('equal7_mean'):+.4f}, EWoK {rc.get('EWoK'):+.2f}, Entity {rc.get('Entity'):+.2f}, GlobalPIQA_mean {rc.get('GlobalPIQA_mean'):+.3f}, with BLiMP {rc.get('BLiMP'):+.2f} and COMPS {rc.get('COMPS'):+.2f}.",
        f"- From the research arithmetic, reinvest needs SuperGLUE+AoA {seven.get('compact_view_reinvest', {}).get('sg_plus_aoa_needed_for_overall_41p8'):.3f} for Overall 41.8.",
        "",
        "## Fixed-budget principle",
        "",
        f"- Core used {core.get('pairs')} source-view pairs; reinvest uses {reinvest.get('pairs')} pairs, adding {added.get('pairs')} compact pairs while staying within the same 423,520-word changed-block budget.",
        f"- Core pair words {core.get('pair_words')} plus {core.get('neutral_cleanqwen_topup_words_inside_changed_block')} neutral top-up words; reinvest pair words {reinvest.get('pair_words')} plus {reinvest.get('neutral_cleanqwen_topup_words_inside_changed_block')} top-up words.",
        f"- Reinvest source words {reinvest.get('source_words')} vs core {core.get('source_words')}; rewrite/source ratio remains ~{reinvest.get('rewrite_to_source_ratio_weighted'):.3f}; entity recall {reinvest.get('entity_recall_stats', {}).get('mean'):.3f}; number recall {reinvest.get('number_recall_stats', {}).get('mean'):.3f}; content recall {reinvest.get('content_recall_stats', {}).get('mean'):.3f}.",
        "",
        "## Active work after research",
        "",
        "- `s25_t24_tool1`: A01-owned full official-compatible evaluation of already-trained `compact_view_reinvest` seed43022; output root `experiments/archive/representation_and_objectives/data/compact_reinvest_full_eval`.",
        "- `s25_t33_tool1`: train `compact_view_reinvest` seed43122 and run fast no-AoA screen; output root `experiments/archive/representation_and_objectives/data/compact_reinvest_seed43122_fast`.",
        "- A02 still owns the core full-eval output directory `experiments/archive/frontier_consolidation/data/density_full_eval`; read it only after the runtime/group makes it available.",
        "",
        "## Closed work",
        "",
        "The older cached-FineWeb seqsafe96 branch and the core-only seed43122 controller were stopped before producing new BabyLM scores. They should not be interpreted as evidence for or against compact-view density.",
        "",
        f"JSON: `{OUT_JSON}`",
    ]
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": route["status"], "json": str(OUT_JSON), "note": str(NOTE)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
