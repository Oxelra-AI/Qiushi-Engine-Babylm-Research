#!/usr/bin/env python3
"""research: cross-coordinate synthesis for compact source-absent channel evidence.

This script does not train, evaluate models, or inspect unfinished
run directories.  It reads completed representation and consolidation artifacts
and writes a compact structured synthesis that preserves what is established before
the pending 100M packed target-selective arms and RoBERTa official trajectory are delivered.

Scientific purpose
------------------
Separate three claims that have become easy to conflate:

1. Local denoising channel: compact source-absent content targets change compact-side
   predictive distributions.
2. Endpoint mediation: that local channel carries the historical BabyLM task gain.
3. Architecture/data transfer: the natural compact-vs-repeat marginal works outside
   the original DeBERTa-v2 coordinate.

The output is a decision matrix for the pending evidence, not a final result.
"""
from __future__ import annotations

import json
import pathlib
import time
from typing import Any

ROOT = pathlib.Path("experiments/archive/representation_and_objectives")
A02 = pathlib.Path("experiments/archive/frontier_consolidation")
OUT = ROOT / "data/compact_channel_cross_coordinate_synthesis"

PATHS = {
    "triangle": ROOT / "data/compact_triangle_noaoa_eval/triangle_noaoa_summary.json",
    "source_use": ROOT / "data/adjusted_source_use_probe/adjusted_source_use_fast_bootstrap.json",
    "crossview": ROOT / "data/crossview_interaction/crossview_partner_visibility_interaction.json",
    "wholeword": ROOT / "data/wholeword_control_readout/wholeword_control_readout.json",
    "source_disjoint": ROOT / "data/source_disjoint_target_probe/source_disjoint_target_type_probe.json",
    "order_integrated": A02 / "data/compact_order_integrated_readout/compact_order_integrated_readout.json",
    "roberta_local": A02 / "data/roberta_pair_stratified_response_probe_early_cpu/stratified_response.json",
    "roberta_note": (A02.parents[2] / 'research/notes/frontier_consolidation/compact_order_result_and_roberta_transfer_decision.md'),
    "a01_pending_note": (ROOT.parents[2] / 'research/notes/representation_and_objectives/packed_targetselect_readout_hardening.md'),
}


def load_json(name: str) -> Any:
    p = PATHS[name]
    return json.loads(p.read_text(encoding="utf-8"))


def get(d: Any, *keys: str, default: Any = None) -> Any:
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def f(x: Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def cite_path(name: str) -> str:
    return str(PATHS[name])


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    tri = load_json("triangle")
    src_use = load_json("source_use")
    cross = load_json("crossview")
    ww = load_json("wholeword")
    sd = load_json("source_disjoint")
    order = load_json("order_integrated")
    rob = load_json("roberta_local")

    triangle_view_repeat = get(tri, "contrasts", "view_minus_repeat_reinvest", "equal7_mean")
    triangle_view_adj = get(tri, "contrasts", "view_minus_adjbreak_reinvest", "equal7_mean")
    triangle_adj_repeat = -float(get(tri, "contrasts", "repeat_minus_adjbreak_reinvest", "equal7_mean"))
    seed_band = get(tri, "seed_spread_bands", "fast_equal7_spread")

    copied_boot = get(src_use, "copied_main_pooled_common", "bootstrap_contrasts", "compact_minus_mean_extracts")

    cross_abs_i = get(cross, "weighted_interaction", "rw_abs_content", "I_partner")
    cross_copy_i = get(cross, "weighted_interaction", "rw_copied", "I_partner")
    cross_abs_end_i = get(cross, "endwindow_interaction", "rw_abs_content", "I_partner")
    cross_copy_end_i = get(cross, "endwindow_interaction", "rw_copied", "I_partner")

    ww20_abs = get(ww, "contrast_bootstrap", "chck_20M", "drop_abs_minus_drop_copied_word", "source_absent_content")
    ww20_ret = get(ww, "contrast_bootstrap", "chck_20M", "drop_abs_minus_drop_copied_word", "retained_content")
    ww20_func = get(ww, "contrast_bootstrap", "chck_20M", "drop_abs_minus_drop_copied_word", "function_other")

    sd_quality = get(sd, "contrast_summary", "chck_20M", "source_disjoint_quality", "drop_abs_minus_drop_copied_word", "source_absent_content")
    if sd_quality is None:
        # Older/large research JSON nests by contrast first.
        sd_quality = get(sd, "eval_summary", "contrasts", "drop_abs_minus_drop_copied_word", "source_disjoint_quality", "source_absent_content")
    sd_full = get(sd, "contrast_summary", "chck_20M", "source_disjoint_quality", "drop_abs_minus_full", "source_absent_content")
    if sd_full is None:
        sd_full = get(sd, "eval_summary", "contrasts", "drop_abs_minus_full", "source_disjoint_quality", "source_absent_content")

    order20 = get(order, "decision_readout", "chck_20M")
    order40 = get(order, "decision_readout", "chck_40M")

    rob10 = get(rob, "per_checkpoint", "chck_10M", "summary")
    rob20 = get(rob, "per_checkpoint", "chck_20M", "summary")

    evidence = []
    evidence.append({
        "evidence": "Natural compact-view DeBERTa triangle",
        "supports": "Intact compact faithful views outperform literal source repetition in the original masked-denoising coordinate.",
        "key_numbers": {
            "view_minus_repeat_equal7_pp": triangle_view_repeat,
            "view_minus_adjbreak_equal7_pp": triangle_view_adj,
            "adjbreak_minus_repeat_equal7_pp": triangle_adj_repeat,
            "same_recipe_seed_band_fast_equal7_pp": seed_band,
        },
        "reading": "The robust fact is view > repeat; view-adjbreak and adjbreak-repeat remain seed-band-limited components rather than separate settled causes.",
        "path": cite_path("triangle"),
    })
    evidence.append({
        "evidence": "Frozen source-use copied-target audit",
        "supports": "Rejects stronger ordered copied-token retrieval as the compact advantage explanation.",
        "key_numbers": {
            "compact_minus_mean_extracts_copied_I_nats_median": get(copied_boot, "q500"),
            "q025": get(copied_boot, "q025"),
            "q975": get(copied_boot, "q975"),
            "p_gt_0": get(copied_boot, "p_gt_0"),
        },
        "reading": "After copy-opportunity matching, compact has lower copied-target source-conditioning than extractive families; copied retrieval is not the load-bearing compact benefit.",
        "path": cite_path("source_use"),
    })
    evidence.append({
        "evidence": "Correct-vs-wrong partner visibility MLM screen",
        "supports": "Direct correct-source reachability exists but is not dominant for source-absent compact content.",
        "key_numbers": {
            "rw_abs_content_I_partner_allrun_nats": cross_abs_i,
            "rw_copied_I_partner_allrun_nats": cross_copy_i,
            "rw_abs_content_I_partner_endwindow_nats": cross_abs_end_i,
            "rw_copied_I_partner_endwindow_nats": cross_copy_end_i,
        },
        "reading": "Correct partner visibility helps source-absent targets modestly, but copied-token interaction is larger, so the mechanism is not simple source-to-rewrite prediction.",
        "path": cite_path("crossview"),
    })
    evidence.append({
        "evidence": "A01 identical-text target-selective whole-word control",
        "supports": "Causal local source-absent target-label channel under DeBERTa pair geometry.",
        "key_numbers": {
            "chck20_drop_abs_minus_drop_copied_word_source_absent_nats": get(ww20_abs, "piece_weighted_delta"),
            "source_absent_p025": get(ww20_abs, "piece_cluster_bootstrap", "p025"),
            "source_absent_p975": get(ww20_abs, "piece_cluster_bootstrap", "p975"),
            "retained_content_nats": get(ww20_ret, "piece_weighted_delta"),
            "function_other_nats": get(ww20_func, "piece_weighted_delta"),
        },
        "reading": "Removing source-absent labels harms source-absent compact-side prediction relative to a whole-word copied-content deletion matched in BPE mass; retained/function categories move opposite at 20M.",
        "path": cite_path("wholeword"),
    })
    evidence.append({
        "evidence": "A01 source-disjoint held-out compact-pair probe",
        "supports": "The local target-channel effect is not just repeated-row or exact masked-instance memorization.",
        "key_numbers": {
            "source_disjoint_quality_drop_abs_minus_copied_word_source_absent_nats": get(sd_quality, "piece_weighted_delta"),
            "p025": get(sd_quality, "pair_cluster_bootstrap", "p025"),
            "p975": get(sd_quality, "pair_cluster_bootstrap", "p975"),
            "drop_abs_minus_full_source_absent_nats": get(sd_full, "piece_weighted_delta"),
        },
        "reading": "The same category-specific deficit appears on exact-source-disjoint compact rewrites, strengthening the local mechanism inside this data/model family.",
        "path": cite_path("source_disjoint"),
    })
    evidence.append({
        "evidence": "A02 compact ordered-vs-scrambled DeBERTa screen",
        "supports": "Shows local source-absent denoising can dissociate from selected BabyLM competence.",
        "key_numbers": {
            "chck40_official_cheap7_ordered_minus_scrambled_pp": get(order40, "official_delta_cheap7"),
            "chck40_cheap6_no_GlobalPIQA_pp": get(order40, "official_delta_cheap6_no_GlobalPIQA"),
            "chck40_cheap5_no_GlobalPIQA_Reading_pp": get(order40, "official_delta_cheap5_no_GlobalPIQA_Reading"),
            "chck40_EWoK_Entity_pp": get(order40, "official_delta_EWoK_Entity"),
            "chck40_source_absent_ordered_minus_scrambled_nll": get(order40, "source_absent_ordered_minus_scrambled_nll"),
            "chck40_source_absent_minus_controls_mean_nll": get(order40, "source_absent_minus_controls_mean_nll"),
        },
        "reading": "At 40M, ordered compact word order gives a large source-absent local NLL advantage but worse stable official readouts; therefore local channel presence alone cannot justify an endpoint or transfer claim.",
        "path": cite_path("order_integrated"),
    })
    evidence.append({
        "evidence": "A02 early RoBERTa compact-vs-repeat local bridge",
        "supports": "Early local source-absent compact response appears in a stock bidirectional MLM coordinate, pending official trajectory.",
        "key_numbers": {
            "chck10_compact_source_absent_advantage_nats": get(rob10, "by_view_category", "compact|source_absent_content", "repeat_minus_compact_advantage"),
            "chck20_compact_source_absent_advantage_nats": get(rob20, "by_view_category", "compact|source_absent_content", "repeat_minus_compact_advantage"),
            "chck10_content_density_high_minus_low_source_absent_nats": get(rob10, "feature_contrasts", "compact_content_fraction|compact|source_absent_content", "contrast"),
            "chck20_content_density_high_minus_low_source_absent_nats": get(rob20, "feature_contrasts", "compact_content_fraction|compact|source_absent_content", "contrast"),
            "chck20_tail_coverage_high_minus_low_source_absent_nats": get(rob20, "feature_contrasts", "compact_tail_content_coverage|compact|source_absent_content", "contrast"),
        },
        "reading": "RoBERTa already shows a compact-trained local advantage on compact source-absent events by 10M/20M, concentrated more by compact content density than by tail coverage, but no selected official readout is known yet.",
        "path": cite_path("roberta_local"),
    })

    claim_status = {
        "local_source_absent_denoising_channel": {
            "status": "supported_within_tested_bidirectional_MLM_coordinates",
            "why": [
                "A01 target-selective deletion causes category-specific source-absent compact loss increases under whole-word copied controls.",
                "A01 source-disjoint held-out compact pairs preserve the sign.",
                "A02 RoBERTa early compact-vs-repeat local probe has the same local source-absent direction.",
            ],
            "not_yet": "This is not by itself an official-task or architecture-general learning principle, because ordered/scrambled created local source-absent improvement while worsening stable official readouts.",
        },
        "endpoint_mediation_of_historical_compact_gain": {
            "status": "unresolved_pending_A01_100M_packed_targetselect",
            "required_evidence": [
                "The matched 100M packed `drop_abs_content` arm must show positive local `drop_abs_minus_drop_copied_word` on source_absent_content, including source-disjoint sets.",
                "The endpoint contrast `drop_copied_word_minus_drop_abs` must favor retaining source-absent labels on Supplement and research relational EWoK domains, not only on aggregate cheap7 or volatile GlobalPIQA/Reading movement.",
            ],
            "pending_tasks": ["s229_t42_tool1", "s229_t43_tool1"],
            "postrun_reader": str(ROOT / "scripts/packed_targetselect_postrun_readout.py"),
        },
        "architecture_transfer_of_natural_compact_marginal": {
            "status": "unresolved_pending_A02_RoBERTa_100M_official_trajectory",
            "required_evidence": [
                "Stable late-band RoBERTa compact-minus-repeat gains on cheap6 without GlobalPIQA, cheap5 without GlobalPIQA/Reading, EWoK+Entity, Supplement/Entity/COMPS.",
                "Agreement or principled disagreement with the local RoBERTa source-absent/content-density bridge.",
            ],
            "known_boundary": "Decoder-only GPT2 causal coordinate did not show broad transfer; negative RoBERTa would bound transfer further but would not identify the cause alone.",
        },
    }

    pending_matrix = {
        "A01_packed_positive__A02_RoBERTa_positive": "Strongest next scientific state: source-absent compact labels mediate the original DeBERTa trajectory and the natural compact marginal transfers to another bidirectional MLM. Next expensive work should be an independent-seed RoBERTa/DeBERTa replication or a natural-density data manipulation, not explicit target selection.",
        "A01_packed_positive__A02_RoBERTa_negative": "Source-absent labels likely mediate the DeBERTa compact trajectory, but transfer is architecture/implementation bounded. Next work should identify which DeBERTa feature or objective geometry lets the channel reach tasks before spending on new data arms.",
        "A01_packed_negative__A02_RoBERTa_positive": "Natural compact marginal transfers, but not through the isolated source-absent label channel. Next work should focus on the coupled natural marginal: faithful compression, content density, BPE burden, source-wide lexical coverage, and reinvested diversity.",
        "A01_packed_negative__A02_RoBERTa_negative": "Local source-absent denoising remains real but insufficient. The compact principle must be rebuilt around coordinate-specific DeBERTa effects or the broader natural marginal; no new H100 target-channel extension is justified from local losses alone.",
        "local_only_anywhere": "A positive local source-absent NLL channel without coherent Supplement/relational-EWoK or stable official-family movement should be preserved as mechanism evidence but not promoted to the transferable learning principle.",
    }

    payload = {
        "status": "COMPACT_CHANNEL_CROSS_COORDINATE_SYNTHESIS",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "meaning": "Completed-artifact synthesis separating local source-absent denoising, historical endpoint mediation, and cross-architecture transfer while A01/A02 100M jobs remain pending.",
        "inputs": {k: str(v) for k, v in PATHS.items()},
        "evidence_table": evidence,
        "claim_status": claim_status,
        "pending_result_matrix": pending_matrix,
        "immediate_next_use": "When A01 packed 100M tasks deliver, run packed_targetselect_postrun_readout.py and read it through claim_status['endpoint_mediation_of_historical_compact_gain']; when A02 RoBERTa selected trajectory delivers, read it through claim_status['architecture_transfer_of_natural_compact_marginal']. Do not launch a new expensive route from local denoising alone.",
        "elapsed_sec": round(time.time() - t0, 2),
    }
    out_json = OUT / "compact_channel_cross_coordinate_synthesis.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md = [
        "# research compact channel cross-coordinate synthesis",
        "",
        f"JSON: `{out_json}`",
        "",
        "## Three claims kept separate",
        "",
        "1. **Local source-absent compact denoising** is supported in tested bidirectional-MLM settings.",
        "2. **Mediation of the historical 100M compact-view task gain** is still unresolved until A01 packed target-selective 100M endpoints are read.",
        "3. **Architecture transfer of the natural compact marginal** is still unresolved until A02 RoBERTa 100M official-compatible trajectory is read.",
        "",
        "## Evidence table",
        "",
        "| Evidence | Key numbers | Reading |",
        "|---|---|---|",
    ]
    for e in evidence:
        nums = "; ".join(f"{k}={v}" for k, v in e["key_numbers"].items())
        md.append(f"| {e['evidence']} | {nums} | {e['reading']} |")
    md += [
        "",
        "## Pending-result matrix",
        "",
    ]
    for k, v in pending_matrix.items():
        md.append(f"- `{k}`: {v}")
    md += [
        "",
        "## Operational consequence",
        "",
        "No new H100 route is justified from local denoising alone. After both packed 100M arms complete, verify integrity and run `scripts/packed_targetselect_postrun_readout.py`. Compare the completed RoBERTa official-compatible trajectory with the early local bridge.",
        "",
    ]
    out_md = (OUT.parents[4] / 'research/documents/representation_and_objectives/data/compact_channel_cross_coordinate_synthesis/compact_channel_cross_coordinate_synthesis.md')
    out_md.write_text("\n".join(md), encoding="utf-8")

    print(json.dumps({
        "status": payload["status"],
        "out_json": str(out_json),
        "out_md": str(out_md),
        "claim_status": claim_status,
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
