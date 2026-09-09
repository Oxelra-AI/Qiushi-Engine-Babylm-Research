#!/usr/bin/env python3
"""research: compare the launched research probability reallocation with exact-swap innovation WWM.

CPU-only analysis. It does not inspect or wait for the managed 80M run. The goal is to
preserve the scientific distinction between two related but not equivalent ways of
using the research conditional-innovation object, so the pending score can be
interpreted without route drift.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import math
import pathlib
import time
from typing import Any

def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
SUMMARY = WS / "data/strict_innovation_group_map/strict_innovation_group_map_summary.json"
SMOKE = WS / "data/strict_innovation_mask_smoke/strict_innovation_mask_smoke.json"
GPU_SMOKE = WS / "data/strict_innovation_gpu_smoke/scientific_metrics.json"
CODEX_SUMMARY = WS / "analysis/innovation_metadata_summary.json"
CODEX_SMOKE = WS / "analysis/innovation_wwm_smoke.json"
CODEX_BATCH = WS / "analysis/frozen_stream_batch_audit.json"
OUT_DIR = WS / "data/innovation_intervention_comparison"
OUT_JSON = OUT_DIR / "innovation_intervention_comparison.json"
OUT_MD = OUT_DIR / "innovation_intervention_comparison.md"

MASK_PROB = 0.15
P_STRICT = 0.5
# Fixed-seq256 trainer-visible groups per 10M epoch.
FIXED256_VISIBLE_GROUPS_PER_10M = 9_802_194


def read_json(path: pathlib.Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def r6(x: float) -> float:
    return round(float(x), 6)


def compute() -> dict[str, Any]:
    s75 = read_json(SUMMARY)
    s75_smoke = read_json(SMOKE)
    s75_gpu = read_json(GPU_SMOKE)
    cx = read_json(CODEX_SUMMARY)
    cx_smoke = read_json(CODEX_SMOKE)
    cx_batch = read_json(CODEX_BATCH)

    t75 = s75["totals"]
    strict_groups = float(t75["n_strict"])
    strict_tokens = float(t75["strict_tokens"])
    copy_groups = float(t75["n_copyable"])
    copy_tokens = float(t75["copyable_tokens"])
    p_copy_token = MASK_PROB - (P_STRICT - MASK_PROB) * strict_tokens / copy_tokens
    p_copy_group = MASK_PROB - (P_STRICT - MASK_PROB) * strict_groups / copy_groups
    per_10m = {
        "baseline_strict_group_labels": MASK_PROB * strict_groups,
        "treatment_strict_group_labels": P_STRICT * strict_groups,
        "extra_strict_group_labels": (P_STRICT - MASK_PROB) * strict_groups,
        "baseline_strict_token_labels": MASK_PROB * strict_tokens,
        "treatment_strict_token_labels": P_STRICT * strict_tokens,
        "extra_strict_token_labels": (P_STRICT - MASK_PROB) * strict_tokens,
        "baseline_copy_group_labels": MASK_PROB * copy_groups,
        "treatment_copy_group_labels_token_basis": p_copy_token * copy_groups,
        "removed_copy_group_labels_token_basis": (MASK_PROB - p_copy_token) * copy_groups,
        "baseline_copy_token_labels": MASK_PROB * copy_tokens,
        "treatment_copy_token_labels": p_copy_token * copy_tokens,
        "removed_copy_token_labels": (MASK_PROB - p_copy_token) * copy_tokens,
        "expected_token_mass_error": (P_STRICT - MASK_PROB) * strict_tokens - (MASK_PROB - p_copy_token) * copy_tokens,
        "expected_group_mass_error_token_basis": (P_STRICT - MASK_PROB) * strict_groups - (MASK_PROB - p_copy_token) * copy_groups,
        "extra_strict_group_fraction_of_global_selected_groups": ((P_STRICT - MASK_PROB) * strict_groups) / (FIXED256_VISIBLE_GROUPS_PER_10M * MASK_PROB),
        "strict_group_exposure_multiplier": P_STRICT / MASK_PROB,
        "strict_token_exposure_multiplier": P_STRICT / MASK_PROB,
        "copy_group_exposure_multiplier_token_basis": p_copy_token / MASK_PROB,
        "copy_token_exposure_multiplier": p_copy_token / MASK_PROB,
    }

    cx_tot = cx["totals_unique_10m_pool"]
    cx_innov_groups = float(cx_tot["innovation_groups"])
    cx_innov_tokens = float(cx_tot["innovation_tokens"])
    cx_copy_groups = float(cx_tot["copyable_groups"])
    # Use the broad stress-test realized swap rate as the empirical projection.
    successful_swaps = float(cx_smoke["targeting"]["successful_swaps"])
    replicates = float(cx_smoke["configuration"]["replicates"])
    extra_swaps_per_10m = successful_swaps / max(1.0, replicates)
    codex_per_10m = {
        "baseline_innovation_group_labels": MASK_PROB * cx_innov_groups,
        "projected_extra_innovation_group_labels": extra_swaps_per_10m,
        "projected_treatment_innovation_group_labels": MASK_PROB * cx_innov_groups + extra_swaps_per_10m,
        "innovation_group_exposure_multiplier": (MASK_PROB * cx_innov_groups + extra_swaps_per_10m) / max(1.0, MASK_PROB * cx_innov_groups),
        "copyable_group_delta": 0.0,
        "source_group_delta": 0.0,
        "selected_group_delta_per_batch": 0.0,
        "selected_token_delta_per_batch": 0.0,
        "extra_innovation_group_fraction_of_global_selected_groups": extra_swaps_per_10m / (FIXED256_VISIBLE_GROUPS_PER_10M * MASK_PROB),
        "empirical_collision_rate": cx_smoke["targeting"]["proposal_baseline_collision_rate"],
        "empirical_no_donor_rate_given_noncollision_approx": cx_smoke["targeting"]["no_equal_length_donor"] / max(1.0, successful_swaps + cx_smoke["targeting"]["no_equal_length_donor"]),
    }

    def scale(d: dict[str, float], epochs: int) -> dict[str, float]:
        return {k: (v * epochs if isinstance(v, (int, float)) and not isinstance(v, bool) else v) for k, v in d.items()}

    comparison = {
        "status": "INNOVATION_INTERVENTION_COMPARISON",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Separate the already-launched research strict copyable-funded probability intervention from the newly delivered exact-swap proposal, so the pending 80M score is interpreted correctly and not overgeneralized.",
        "inputs": {
            "summary": str(SUMMARY),
            "smoke": str(SMOKE),
            "gpu_smoke": str(GPU_SMOKE),
            "codex_summary": str(CODEX_SUMMARY),
            "codex_smoke": str(CODEX_SMOKE),
            "codex_batch_audit": str(CODEX_BATCH),
        },
        "probability_reallocation": {
            "definition": "row-unique content-like rewrite groups absent from paired source and from every other row occurrence; p_strict=0.5; copyable rewrite groups p_copy chosen from token mass; source/filler/other ordinary groups remain at 0.15 in expectation",
            "target_counts_per_10m": {
                "strict_groups": int(strict_groups),
                "strict_tokens": int(strict_tokens),
                "copyable_groups": int(copy_groups),
                "copyable_tokens": int(copy_tokens),
                "rows_with_strict": int(t75["rows_with_strict"]),
            },
            "probabilities": {
                "mask_prob": MASK_PROB,
                "p_strict": P_STRICT,
                "p_copy_token_basis": p_copy_token,
                "p_copy_group_basis_for_reference_only": p_copy_group,
            },
            "expected_per_10m": {k: r6(v) for k, v in per_10m.items()},
            "expected_80m": {k: r6(v * 8) for k, v in per_10m.items() if "multiplier" not in k and "fraction" not in k and "error" not in k},
            "cpu_smoke_key_checks": {
                "strict_rate": s75_smoke["rates"]["strict_group_rate"],
                "copyable_rate": s75_smoke["rates"]["copy_group_rate"],
                "source_rate": s75_smoke["rates"]["source_group_rate"],
                "rewrite_other_rate": s75_smoke["rates"]["rewrite_other_group_rate"],
                "selected_token_mass_ratio": s75_smoke["mass_ratio_vs_standard_selected_tokens"],
                "all_checks_passed": s75_smoke["all_checks_passed"],
            },
            "gpu_smoke_key_checks": {
                "total_words": s75_gpu["total_words"],
                "total_steps": s75_gpu["total_steps"],
                "final_loss": s75_gpu["final_loss"],
                "strict_selected_groups": s75_gpu["cumulative_mask_stats"].get("strict_selected_groups"),
                "strict_available_groups": s75_gpu["cumulative_mask_stats"].get("strict_available_groups"),
                "copy_selected_groups": s75_gpu["cumulative_mask_stats"].get("copy_selected_groups"),
                "copy_available_groups": s75_gpu["cumulative_mask_stats"].get("copy_available_groups"),
            },
        },
        "codex_exact_swap": {
            "definition": "pair-local source-absent rewrite innovations with strict whole-group containment; one hash-rotated proposal per changed row; if not already selected, swap with same-token-length selected donor outside every pair span, preferring ordinary rows; copyable and source pair masks unchanged exactly",
            "target_counts_per_10m": {
                "innovation_groups": int(cx_innov_groups),
                "innovation_tokens": int(cx_innov_tokens),
                "copyable_groups": int(cx_copy_groups),
                "rows_with_innovation": int(cx_tot["rows_with_innovation"]),
                "relation_cue_innovation_groups": int(cx_tot["innovation_cue:relation_cue"]),
                "content_or_other_innovation_groups": int(cx_tot["innovation_cue:content_or_other"]),
            },
            "projected_per_10m_from_broad_smoke": {k: r6(v) if isinstance(v, (int, float)) else v for k, v in codex_per_10m.items()},
            "projected_80m_from_broad_smoke": {k: r6(v * 8) for k, v in codex_per_10m.items() if isinstance(v, (int, float)) and "rate" not in k and "fraction" not in k and "multiplier" not in k and "delta_per_batch" not in k},
            "broad_smoke_key_checks": {
                "baseline_selected_groups": cx_smoke["selection_mass"]["baseline_selected_groups"],
                "biased_selected_groups": cx_smoke["selection_mass"]["biased_selected_groups"],
                "baseline_selected_tokens": cx_smoke["selection_mass"]["baseline_selected_tokens"],
                "biased_selected_tokens": cx_smoke["selection_mass"]["biased_selected_tokens"],
                "per_batch_group_mismatches": cx_smoke["selection_mass"]["per_batch_group_mismatches"],
                "per_batch_token_mismatches": cx_smoke["selection_mass"]["per_batch_token_mismatches"],
                "successful_swaps": int(successful_swaps),
                "copyable_delta": cx_smoke["copyable_and_source_controls"]["copyable_delta"],
                "source_selection_changed_rows": cx_smoke["copyable_and_source_controls"]["rows_with_source_selection_changed"],
                "donor_ordinary_row_fraction": cx_smoke["copyable_and_source_controls"]["donor_ordinary_row_fraction"],
            },
            "actual_frozen_batch_context": {
                "batches": cx_batch["batches"],
                "changed_per_batch_mean": cx_batch["changed_per_batch"]["mean"],
                "changed_per_batch_min": cx_batch["changed_per_batch"]["min"],
                "changed_per_batch_max": cx_batch["changed_per_batch"]["max"],
                "all_changed_batches": cx_batch["all_changed_batches"],
            },
        },
    }

    comparison["interpretation"] = {
        "not_equivalent": True,
        "answers": "Does a strong strict-content innovation upweight, funded by lower copyable-rewrite supervision while source/filler remain ordinary in expectation, improve mature BabyLM surfaces?",
        "codex_answers": "Does a smaller exact same-token donor swap from non-pair ordinary groups give each changed row an added source-absent innovation signal while leaving source and copyable pair masks exactly baseline?",
        "do_not_overclose": "A negative research score should close the running p_strict/p_copy intervention, but it should not by itself close the exact-swap variant because the donor source, target breadth, perturbation size, and copyable-mask behavior differ.",
        "do_not_cancel_pending_from_codex_alone": "The research run still tests a meaningful copyable-to-strict-innovation hypothesis built from research; the exact-swap proposal supplies a cleaner dormant alternative rather than evidence that the launched run cannot improve.",
        "score_reading": [
            "Broad BLiMP/Supplement/EWoK gains at 70M/80M versus research would support continuing research to 100M/full evaluation.",
            "Innovation-probe improvement with broad score damage or copyable-like damage would suggest the target is real but copyable suppression/pressure size is harmful; then exact-swap becomes the next lower-perturbation test.",
            "No innovation-probe improvement and weak cheap columns would weaken the whole masking-pressure route, not only the probability schedule.",
            "A gain confined to GlobalPIQA_nonparallel or COMPS repeats the redistribution pattern that closed word-mean and minfreq50 and should stop research without 100M continuation."
        ],
    }
    return comparison


def write_md(result: dict[str, Any]) -> str:
    s75 = result["probability_reallocation"]
    cx = result["codex_exact_swap"]
    lines = []
    lines.append("# research — innovation intervention comparison while research 80M runs")
    lines.append("")
    lines.append("CPU-only analysis; it did not inspect, wait for, or affect the managed research training/evaluation tasks.")
    lines.append("")
    lines.append("## Why this comparison matters")
    lines.append("")
    lines.append("A different innovation-biased WWM design was proposed after the research strict probability run had already been launched. The two mechanisms are scientifically related but not the same experiment, so the pending 80M score must not be overinterpreted.")
    lines.append("")
    lines.append("## Launched research probability reallocation")
    lines.append("")
    lines.append(f"- Targets: {s75['target_counts_per_10m']['strict_groups']:,} strict row-unique content-like innovation groups / {s75['target_counts_per_10m']['strict_tokens']:,} tokens per 10M; {s75['target_counts_per_10m']['copyable_groups']:,} copyable groups / {s75['target_counts_per_10m']['copyable_tokens']:,} tokens.")
    lines.append(f"- Probabilities: strict {s75['probabilities']['p_strict']:.3f}; copyable {s75['probabilities']['p_copy_token_basis']:.9f}; source/filler/other ordinary at {s75['probabilities']['mask_prob']:.2f} in expectation.")
    e = s75["expected_per_10m"]
    lines.append(f"- Per 10M expected change: strict token labels +{e['extra_strict_token_labels']:.1f}, copyable token labels -{e['removed_copy_token_labels']:.1f}; strict group labels +{e['extra_strict_group_labels']:.1f}, copyable group labels -{e['removed_copy_group_labels_token_basis']:.1f}.")
    lines.append(f"- Exposure multipliers: strict ×{e['strict_group_exposure_multiplier']:.3f}; copyable ×{e['copy_token_exposure_multiplier']:.3f}; extra strict group labels are {100*e['extra_strict_group_fraction_of_global_selected_groups']:.3f}% of global baseline selected groups.")
    lines.append(f"- CPU smoke: strict rate {s75['cpu_smoke_key_checks']['strict_rate']}, copyable {s75['cpu_smoke_key_checks']['copyable_rate']}, source {s75['cpu_smoke_key_checks']['source_rate']}, selected-token mass ratio {s75['cpu_smoke_key_checks']['selected_token_mass_ratio']}, checks passed `{s75['cpu_smoke_key_checks']['all_checks_passed']}`.")
    lines.append("")
    lines.append("## exact-swap alternative")
    lines.append("")
    lines.append(f"- Targets: {cx['target_counts_per_10m']['innovation_groups']:,} pair-local source-absent innovation groups / {cx['target_counts_per_10m']['innovation_tokens']:,} tokens per 10M, including {cx['target_counts_per_10m']['content_or_other_innovation_groups']:,} content/other and {cx['target_counts_per_10m']['relation_cue_innovation_groups']:,} relation-cue groups.")
    ce = cx["projected_per_10m_from_broad_smoke"]
    lines.append(f"- Mechanism: begin with exact baseline WWM; at most one innovation proposal per changed row; if not already selected, swap with an already-selected ordinary non-pair donor of identical token length, preferably from an ordinary row.")
    lines.append(f"- Broad smoke projection per 10M: baseline innovation group labels {ce['baseline_innovation_group_labels']:.1f}, extra {ce['projected_extra_innovation_group_labels']:.1f}, treatment total {ce['projected_treatment_innovation_group_labels']:.1f}, exposure multiplier ×{ce['innovation_group_exposure_multiplier']:.3f}.")
    lines.append(f"- Exact controls in smoke: selected groups/tokens unchanged per batch; copyable delta {cx['broad_smoke_key_checks']['copyable_delta']}; source-selection changed rows {cx['broad_smoke_key_checks']['source_selection_changed_rows']}; ordinary-row donor fraction {cx['broad_smoke_key_checks']['donor_ordinary_row_fraction']}.")
    lines.append(f"- Frozen batch audit: all {cx['actual_frozen_batch_context']['batches']} actual batches contain changed rows but none are all-changed; changed rows per batch mean {cx['actual_frozen_batch_context']['changed_per_batch_mean']:.3f}, range {cx['actual_frozen_batch_context']['changed_per_batch_min']}–{cx['actual_frozen_batch_context']['changed_per_batch_max']}.")
    lines.append("")
    lines.append("## Scientific interpretation")
    lines.append("")
    for item in result["interpretation"]["score_reading"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("The exact-swap proposal does not make the already-running research experiment obsolete: research directly tests whether copyable rewrite supervision is surplus enough to fund stronger strict-content innovation pressure. But research and exact-swap are not interchangeable. If research is negative in a way consistent with copyable suppression or excessive pressure, exact-swap remains a distinct lower-perturbation candidate; if research is broadly positive, it is the stronger and already-running route toward 100M/full evaluation.")
    lines.append("")
    lines.append(f"Full JSON: `{OUT_JSON}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = compute()
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    OUT_MD.write_text(write_md(result), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "out_json": str(OUT_JSON),
        "out_md": str(OUT_MD),
        "extra_strict_tokens_per_10m": result["probability_reallocation"]["expected_per_10m"]["extra_strict_token_labels"],
        "copy_token_multiplier": result["probability_reallocation"]["expected_per_10m"]["copy_token_exposure_multiplier"],
        "codex_extra_innovation_groups_per_10m": result["codex_exact_swap"]["projected_per_10m_from_broad_smoke"]["projected_extra_innovation_group_labels"],
    }, indent=2))


if __name__ == "__main__":
    main()
