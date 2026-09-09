#!/usr/bin/env python3
"""research: integrate RoBERTa aggregate selected scores with pair-stratified local response.

This is a lightweight reader, not a trainer/evaluator.  It encodes the scientific
specified scientific interpretation:

- Aggregate selected official-compatible trajectories decide whether the natural compact
  marginal transfers on stable BabyLM readouts.
- Pair strata from the research atlas can explain where the model response concentrates
  (tail coverage, content density, source-absent fraction), but strata are not causal.
- Concordance between stable downstream gains and feature-structured local gains can
  justify an independent-seed replication; local-only gains should not trigger an
  expensive route.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any


def find_user_root() -> Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
DEFAULT_READER = WS / "data/roberta_transfer_result_reader/reader_report.json"
DEFAULT_OFFICIAL = WS / "data/roberta_full100m_integrated/roberta_transfer_integrated.json"
DEFAULT_STRATIFIED = WS / "data/roberta_pair_stratified_response_probe/stratified_response.json"
DEFAULT_EARLY = WS / "data/roberta_pair_stratified_response_probe_early_cpu_repaired/stratified_response.json"
DEFAULT_MANIFEST = WS / "data/roberta_pair_stratified_response_probe/stratum_manifest.json"
DEFAULT_OUT = WS / "data/roberta_stratified_bridge_integrator"
LATE = ["chck_60M", "chck_70M", "chck_80M", "chck_90M", "chck_100M"]
STABLE_KEYS = ["cheap6_no_GlobalPIQA", "cheap5_no_GlobalPIQA_Reading", "EWoK_plus_Entity", "Supplement", "Entity", "COMPS"]
LOCAL_KEYS = [
    "compact|source_absent_content",
    "compact|retained_content",
    "compact|function_other",
    "repeat|retained_content",
    "repeat|function_other",
]
FEATURE_PATTERNS = [
    "compact_content_fraction|compact|source_absent_content",
    "compact_source_absent_content_fraction_of_content|compact|source_absent_content",
    "compact_tail_content_coverage|compact|source_absent_content",
    "compact_content_fraction|compact|retained_content",
    "compact_tail_content_coverage|compact|retained_content",
]


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def mean(vals: list[Any]) -> float | None:
    xs = [float(v) for v in vals if finite(v)]
    return sum(xs) / len(xs) if xs else None


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def summarize_official_from_reader(reader: dict[str, Any] | None, official: dict[str, Any] | None) -> dict[str, Any]:
    if reader and reader.get("ready_for_scientific_interpretation"):
        summary = reader.get("integrated_selected_readout", {}).get("summary", {})
        return {
            "source": "reader_report",
            "ready": True,
            "late_means": summary.get("late_means_recomputed", {}),
            "stable_late_positive": bool(summary.get("stable_late_positive")),
            "volatile_carried_readout": bool(summary.get("volatile_carried_readout")),
            "scientific_reading": summary.get("scientific_reading"),
            "late_rows_present": summary.get("late_rows_present"),
        }
    if official:
        rows = official.get("per_checkpoint", [])
        byck = {r.get("ck"): r for r in rows if isinstance(r, dict)}
        late_rows = [byck[ck] for ck in LATE if ck in byck and not byck[ck].get("missing")]
        late_means = {}
        for k in ["cheap7", *STABLE_KEYS, "GlobalPIQA", "Reading"]:
            late_means[k] = mean([r.get("compact_minus_repeat", {}).get(k) for r in late_rows])
        stable_positive = (
            finite(late_means.get("cheap6_no_GlobalPIQA")) and late_means["cheap6_no_GlobalPIQA"] > 0 and
            finite(late_means.get("cheap5_no_GlobalPIQA_Reading")) and late_means["cheap5_no_GlobalPIQA_Reading"] > 0 and
            finite(late_means.get("EWoK_plus_Entity")) and late_means["EWoK_plus_Entity"] >= 0 and
            sum(1 for k in STABLE_KEYS if finite(late_means.get(k)) and late_means[k] > 0) >= 4
        )
        volatile = finite(late_means.get("cheap7")) and late_means["cheap7"] > 0 and not stable_positive
        return {"source": "official_integrated_json", "ready": len(late_rows) == len(LATE), "late_means": late_means, "stable_late_positive": stable_positive, "volatile_carried_readout": volatile, "late_rows_present": [r.get("ck") for r in late_rows]}
    return {"source": "none", "ready": False, "reason": "No completed official selected trajectory reader/integrated JSON."}


def summarize_stratified(strat: dict[str, Any] | None) -> dict[str, Any]:
    if not strat:
        return {"ready": False, "ready_for_concordance": False, "reason": "No stratified local response JSON."}
    pcs = strat.get("per_checkpoint", {})
    present = [ck for ck, rec in pcs.items() if isinstance(rec, dict) and not rec.get("missing")]
    late_present = [ck for ck in LATE if ck in present]
    using_late_band = len(late_present) == len(LATE)
    ready_for_concordance = using_late_band
    # If the late local file is not present, keep early checkpoints as a local-only readout,
    # but never let them enter the mature concordance decision.
    use = late_present if using_late_band else present
    local_means: dict[str, float | None] = {}
    feature_means: dict[str, float | None] = {}
    compact_specific_did_means: dict[str, dict[str, float | None]] = {}
    local_bootstrap_by_key: dict[str, Any] = {}
    for key in LOCAL_KEYS:
        vals = []
        boots = []
        for ck in use:
            rec = pcs[ck].get("summary", {}).get("by_view_category", {}).get(key, {})
            vals.append(rec.get("repeat_minus_compact_advantage"))
            if rec.get("pair_cluster_bootstrap"):
                boots.append({"ck": ck, **rec["pair_cluster_bootstrap"]})
        local_means[key] = mean(vals)
        if boots:
            local_bootstrap_by_key[key] = boots
    for key in FEATURE_PATTERNS:
        vals = []
        for ck in use:
            rec = pcs[ck].get("summary", {}).get("feature_contrasts", {}).get(key, {})
            if rec:
                vals.append(rec.get("contrast"))
        feature_means[key] = mean(vals)
    for feature in ["compact_tail_content_coverage", "compact_content_fraction", "compact_source_absent_content_fraction_of_content"]:
        bins: dict[str, list[float]] = {}
        for ck in use:
            frec = pcs[ck].get("summary", {}).get("compact_specific_did_by_feature_bin", {}).get(feature, {})
            for b, rec in frec.items():
                val = rec.get("compact_specific_advantage_did")
                if finite(val):
                    bins.setdefault(b, []).append(float(val))
        compact_specific_did_means[feature] = {b: mean(vs) for b, vs in sorted(bins.items())}
    source_absent_adv = local_means.get("compact|source_absent_content")
    retained_adv = local_means.get("compact|retained_content")
    repeat_retained_adv = local_means.get("repeat|retained_content")
    control_vals = [v for v in [retained_adv, repeat_retained_adv] if finite(v)]
    selectivity_margin = (float(source_absent_adv) - max(float(v) for v in control_vals)) if finite(source_absent_adv) and control_vals else None
    source_absent_selective = finite(selectivity_margin) and float(selectivity_margin) > 0.03
    concentration = {
        "content_density_source_absent": feature_means.get("compact_content_fraction|compact|source_absent_content"),
        "source_absent_fraction_source_absent": feature_means.get("compact_source_absent_content_fraction_of_content|compact|source_absent_content"),
        "tail_coverage_source_absent": feature_means.get("compact_tail_content_coverage|compact|source_absent_content"),
    }
    positive_concentration_markers = [k for k, v in concentration.items() if finite(v) and float(v) > 0.02]
    return {
        "ready": True,
        "ready_for_concordance": ready_for_concordance,
        "source_path": strat.get("events"),
        "response_events_sha256": strat.get("events_sha256"),
        "present_checkpoints": present,
        "late_present_checkpoints": late_present,
        "checkpoints_used_for_means": use,
        "using_late_band": using_late_band,
        "local_view_category_advantage_means": local_means,
        "local_category_bootstrap_by_checkpoint": local_bootstrap_by_key,
        "feature_contrast_means": feature_means,
        "compact_specific_did_means": compact_specific_did_means,
        "source_absent_selectivity_margin_vs_controls": selectivity_margin,
        "source_absent_selective_local_advantage": bool(source_absent_selective),
        "source_absent_advantage_concentrated_in_measured_features": bool(positive_concentration_markers),
        "positive_concentration_markers": positive_concentration_markers,
        "concentration_markers": concentration,
        "interpretation_note": "Positive advantage means compact-trained local NLL is lower. These are event strata from an engineered probe sample, not natural-prevalence estimates or causal interventions; read with official selected scores.",
    }


def interpret(official: dict[str, Any], stratified: dict[str, Any]) -> str:
    if not official.get("ready") and stratified.get("ready"):
        return "Only local stratified response is available. Treat any compact local advantage as checkpoint-local mechanism evidence pending official-compatible selected trajectories; do not launch a new expensive arm from local response alone."
    if official.get("ready") and stratified.get("ready") and not stratified.get("ready_for_concordance"):
        return "Official selected trajectory is available but the stratified local bridge is early/incomplete rather than late-band; run the late local bridge before using stratum concordance, and do not let early local response veto or promote the official result."
    if not stratified.get("ready") and official.get("ready"):
        return "Official selected trajectory is available but the stratified local bridge is absent; run the local bridge before choosing a causal data intervention."
    if not official.get("ready") and not stratified.get("ready"):
        return "Neither official selected trajectory nor stratified local response is complete; wait for the managed RoBERTa results and preserve current closed-route boundaries."
    stable = bool(official.get("stable_late_positive"))
    volatile = bool(official.get("volatile_carried_readout"))
    local_sel = bool(stratified.get("source_absent_selective_local_advantage"))
    local_conc = bool(stratified.get("source_absent_advantage_concentrated_in_measured_features"))
    if stable and local_sel and local_conc:
        return "RoBERTa shows stable late downstream transfer and the mature local compact response is source-absent-selective with specified measured-feature concentration; the next expensive work should be a paired independent-seed replication, not a new factor sweep."
    if stable and not (local_sel or local_conc):
        return "RoBERTa shows stable downstream transfer but the mature local response is not source-absent/feature-concentrated; compact marginal may transfer through a different ingredient, so inspect item/family movement before designing one clean intervention."
    if volatile:
        return "RoBERTa compact movement is carried by volatile columns rather than stable families; local stratified advantages, if present, should not promote the route."
    if (not stable) and (local_sel or local_conc):
        return "Local compact response exists without stable downstream transfer, echoing the ordered/scrambled local-vs-selected dissociation; do not replicate RoBERTa solely, and choose one causal intervention from the discordant ingredient only after reading the full trajectory."
    return "RoBERTa is neutral/negative on stable downstream readouts and lacks a strong matching local feature-response structure; bound this coordinate and return to compact-marginal decomposition using one measured ingredient at a time."


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# research RoBERTa stratified bridge integration",
        "",
        f"Created: `{payload['created_utc']}`",
        "",
        "## Scientific reading",
        "",
        payload["scientific_reading"],
        "",
        "## Official selected trajectory",
        "",
        f"Ready: `{payload['official_selected']['ready']}` source: `{payload['official_selected'].get('source')}`",
        f"Late means: `{json.dumps(payload['official_selected'].get('late_means'), sort_keys=True)}`",
        "",
        "## Stratified local response",
        "",
        f"Ready: `{payload['stratified_local']['ready']}` ready for concordance: `{payload['stratified_local'].get('ready_for_concordance')}` using late band: `{payload['stratified_local'].get('using_late_band')}` checkpoints used: `{payload['stratified_local'].get('checkpoints_used_for_means')}`",
        f"Event SHA match: `{payload['event_manifest_consistency'].get('event_sha_matches')}` manifest `{payload['event_manifest_consistency'].get('manifest_events_sha256')}` response `{payload['event_manifest_consistency'].get('response_events_sha256')}`",
        f"Local view/category advantages: `{json.dumps(payload['stratified_local'].get('local_view_category_advantage_means'), sort_keys=True)}`",
        f"Concentration markers: `{json.dumps(payload['stratified_local'].get('concentration_markers'), sort_keys=True)}` positive markers: `{payload['stratified_local'].get('positive_concentration_markers')}`",
        "This integration is a route-reading aid. It does not submit, upload, evaluate official tasks, or launch training.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader_report", type=Path, default=DEFAULT_READER)
    ap.add_argument("--official_integrated", type=Path, default=DEFAULT_OFFICIAL)
    ap.add_argument("--stratified_response", type=Path, default=DEFAULT_STRATIFIED)
    ap.add_argument("--early_stratified_response", type=Path, default=DEFAULT_EARLY)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    reader = read_json(args.reader_report)
    official_payload = read_json(args.official_integrated)
    strat = read_json(args.stratified_response)
    strat_source = str(args.stratified_response)
    if strat is None and args.early_stratified_response.exists():
        strat = read_json(args.early_stratified_response)
        strat_source = str(args.early_stratified_response)
    official = summarize_official_from_reader(reader, official_payload)
    stratified = summarize_stratified(strat)
    stratified["json_source"] = strat_source if strat else None
    manifest = read_json(args.manifest)
    manifest_event_sha = manifest.get("frozen_events_sha256") if manifest else None
    response_event_sha = strat.get("events_sha256") if strat else None
    event_sha_matches = bool(manifest_event_sha and response_event_sha and manifest_event_sha == response_event_sha)
    payload = {
        "status": "ROBERTA_STRATIFIED_BRIDGE_INTEGRATION",
        "created_utc": now_utc(),
        "official_selected": official,
        "stratified_local": stratified,
        "event_manifest_consistency": {
            "manifest_exists": manifest is not None,
            "manifest_events_sha256": manifest_event_sha,
            "response_events_sha256": response_event_sha,
            "event_sha_matches": event_sha_matches,
            "category_consistency": manifest.get("category_consistency") if manifest else None,
        },
        "scientific_reading": interpret(official, stratified),
        "human_submission_boundary": "No leaderboard submission, upload, official evaluation, training launch, or managed-task polling is performed by this integrator.",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "stratified_bridge_integration.json", payload)
    (args.out_dir / "stratified_bridge_integration.md").write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out": str(args.out_dir / "stratified_bridge_integration.json"), "official_ready": official.get("ready"), "stratified_ready": stratified.get("ready"), "reading": payload["scientific_reading"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
