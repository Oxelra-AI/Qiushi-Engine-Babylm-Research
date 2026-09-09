#!/usr/bin/env python3
"""Synthesize RoBERTa compact-vs-repeat transfer and target-selective evidence.

This reader starts no training, no selected evaluation, no upload, and no leaderboard action.
It only reads files that already exist.  It is intended to be rerun after the existing
RoBERTa trajectory/evaluation/bridge jobs and target-selective jobs deliver.

Scientific object: whether the natural compact-view/reinvestment marginal transfers beyond
DeBERTa into a stock RoBERTa/BERT-style bidirectional MLM coordinate, and whether any
transfer aligns with the local source-absent compact-content response rather than being a
volatile-column movement or a local-loss-only dissociation.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any


def find_user_root() -> Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
A01_WS = ROOT / "experiments/archive/representation_and_objectives"

LATE = ["chck_60M", "chck_70M", "chck_80M", "chck_90M", "chck_100M"]
STABLE_KEYS = [
    "cheap6_no_GlobalPIQA",
    "cheap5_no_GlobalPIQA_Reading",
    "EWoK_plus_Entity",
    "Supplement",
    "Entity",
    "COMPS",
]
ALL_KEYS = ["cheap7", *STABLE_KEYS, "BLiMP", "EWoK", "GlobalPIQA", "Reading"]
LOCAL_KEYS = [
    "compact|source_absent_content",
    "compact|retained_content",
    "compact|function_other",
    "repeat|retained_content",
    "repeat|function_other",
]
FEATURE_KEYS = [
    "compact_content_fraction|compact|source_absent_content",
    "compact_tail_content_coverage|compact|source_absent_content",
    "compact_source_absent_content_fraction_of_content|compact|source_absent_content",
]

DEFAULT_OFFICIAL = WS / "data/roberta_full100m_integrated/roberta_transfer_integrated.json"
DEFAULT_READER_REPORTS = [
    WS / "data/roberta_result_reader_after_tasks/reader_report.json",
    WS / "data/roberta_transfer_result_reader/reader_report.json",
]
DEFAULT_BRIDGE_AFTER = WS / "data/roberta_stratified_bridge_after_tasks/stratified_bridge_integration.json"
DEFAULT_LATE_STRAT = WS / "data/roberta_pair_stratified_response_probe_late_cpu/stratified_response.json"
DEFAULT_EARLY_STRAT = WS / "data/roberta_pair_stratified_response_probe_early_cpu_repaired/stratified_response.json"
DEFAULT_MANIFEST = WS / "data/roberta_pair_stratified_response_probe/stratum_manifest.json"
DEFAULT_A01_STATIC = A01_WS / "data/packed_static_label_load_profile_v2/packed_static_label_load_profile_v2.json"
DEFAULT_A01_POSTRUN = A01_WS / "data/packed_targetselect_postrun_readout/postrun_readout_summary.json"
DEFAULT_OUT = WS / "data/roberta_compact_result_synthesis"


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def mean(vals: list[Any]) -> float | None:
    xs = [float(v) for v in vals if finite(v)]
    return sum(xs) / len(xs) if xs else None


def first_existing(paths: list[Path]) -> tuple[Path | None, dict[str, Any] | None]:
    for p in paths:
        obj = read_json(p)
        if obj is not None:
            return p, obj
    return None, None


def summarize_official(official_path: Path, reader_paths: list[Path]) -> dict[str, Any]:
    reader_path, reader = first_existing(reader_paths)
    if reader and reader.get("ready_for_scientific_interpretation"):
        s = reader.get("integrated_selected_readout", {}).get("summary", {})
        late_means = s.get("late_means_recomputed", {}) or {}
        stable_pos = bool(s.get("stable_late_positive"))
        volatile = bool(s.get("volatile_carried_readout"))
        return {
            "ready": True,
            "source": "reader_report",
            "path": str(reader_path),
            "late_rows_present": s.get("late_rows_present", []),
            "late_means": late_means,
            "stable_late_positive": stable_pos,
            "volatile_carried_readout": volatile,
            "stable_positive_keys": s.get("stable_positive_keys", []),
            "stable_nonpositive_keys": s.get("stable_negative_or_zero_keys", []),
            "scientific_reading": s.get("scientific_reading"),
        }

    official = read_json(official_path)
    if official is None:
        return {
            "ready": False,
            "source": "absent",
            "path": str(official_path),
            "reason": "RoBERTa selected official-compatible integrated JSON is not present yet.",
        }
    rows = official.get("per_checkpoint", [])
    byck = {r.get("ck"): r for r in rows if isinstance(r, dict)}
    late_rows = [byck[ck] for ck in LATE if ck in byck and not byck[ck].get("missing")]
    late_means = {k: mean([r.get("compact_minus_repeat", {}).get(k) for r in late_rows]) for k in ALL_KEYS}
    stable_positive_keys = [k for k in STABLE_KEYS if finite(late_means.get(k)) and float(late_means[k]) > 0]
    stable_nonpositive_keys = [k for k in STABLE_KEYS if finite(late_means.get(k)) and float(late_means[k]) <= 0]
    stable_pos = (
        len(late_rows) == len(LATE)
        and finite(late_means.get("cheap6_no_GlobalPIQA")) and float(late_means["cheap6_no_GlobalPIQA"]) > 0
        and finite(late_means.get("cheap5_no_GlobalPIQA_Reading")) and float(late_means["cheap5_no_GlobalPIQA_Reading"]) > 0
        and finite(late_means.get("EWoK_plus_Entity")) and float(late_means["EWoK_plus_Entity"]) >= 0
        and len(stable_positive_keys) >= 4
    )
    volatile = (
        len(late_rows) == len(LATE)
        and finite(late_means.get("cheap7")) and float(late_means["cheap7"]) > 0
        and not stable_pos
    )
    return {
        "ready": len(late_rows) == len(LATE),
        "source": "official_integrated_json",
        "path": str(official_path),
        "status": official.get("status"),
        "late_rows_present": [r.get("ck") for r in late_rows],
        "late_means": late_means,
        "stable_late_positive": stable_pos,
        "volatile_carried_readout": volatile,
        "stable_positive_keys": stable_positive_keys,
        "stable_nonpositive_keys": stable_nonpositive_keys,
    }


def summarize_stratified(late_path: Path, early_path: Path, manifest_path: Path) -> dict[str, Any]:
    strat = read_json(late_path)
    using_late = strat is not None
    chosen_path = late_path
    if strat is None:
        strat = read_json(early_path)
        chosen_path = early_path
    if strat is None:
        return {
            "ready": False,
            "using_late_band": False,
            "path": str(late_path),
            "reason": "Neither late nor repaired early stratified local-response JSON is present.",
        }
    manifest = read_json(manifest_path)
    manifest_sha = manifest.get("frozen_events_sha256") if manifest else None
    response_sha = strat.get("events_sha256")
    pcs = strat.get("per_checkpoint", {})
    present = [ck for ck, rec in pcs.items() if isinstance(rec, dict) and not rec.get("missing")]
    late_present = [ck for ck in LATE if ck in present]
    use = late_present if len(late_present) == len(LATE) else present

    local_means: dict[str, float | None] = {}
    local_boot: dict[str, Any] = {}
    for key in LOCAL_KEYS:
        vals = []
        boots = []
        for ck in use:
            rec = pcs.get(ck, {}).get("summary", {}).get("by_view_category", {}).get(key, {})
            vals.append(rec.get("repeat_minus_compact_advantage"))
            if rec.get("pair_cluster_bootstrap"):
                boots.append({"ck": ck, **rec["pair_cluster_bootstrap"]})
        local_means[key] = mean(vals)
        if boots:
            local_boot[key] = boots

    feature_means: dict[str, float | None] = {}
    for key in FEATURE_KEYS:
        vals = []
        for ck in use:
            rec = pcs.get(ck, {}).get("summary", {}).get("feature_contrasts", {}).get(key, {})
            vals.append(rec.get("contrast"))
        feature_means[key] = mean(vals)

    src_abs = local_means.get("compact|source_absent_content")
    controls = [
        local_means.get("compact|retained_content"),
        local_means.get("compact|function_other"),
        local_means.get("repeat|retained_content"),
    ]
    controls2 = [float(v) for v in controls if finite(v)]
    margin = float(src_abs) - max(controls2) if finite(src_abs) and controls2 else None
    source_absent_selective = finite(margin) and float(margin) > 0.03
    content_density = feature_means.get("compact_content_fraction|compact|source_absent_content")
    tail_cov = feature_means.get("compact_tail_content_coverage|compact|source_absent_content")
    source_abs_frac = feature_means.get("compact_source_absent_content_fraction_of_content|compact|source_absent_content")
    markers = []
    if finite(content_density) and float(content_density) > 0.02:
        markers.append("content_density")
    if finite(tail_cov) and float(tail_cov) > 0.02:
        markers.append("tail_coverage")
    if finite(source_abs_frac) and float(source_abs_frac) > 0.02:
        markers.append("source_absent_fraction")

    return {
        "ready": True,
        "path": str(chosen_path),
        "using_late_band": bool(len(late_present) == len(LATE)),
        "present_checkpoints": present,
        "late_present_checkpoints": late_present,
        "checkpoints_used_for_means": use,
        "events_sha256": response_sha,
        "manifest_events_sha256": manifest_sha,
        "event_sha_matches": bool(manifest_sha and response_sha and manifest_sha == response_sha),
        "local_view_category_advantage_means": local_means,
        "local_category_bootstrap_by_checkpoint": local_boot,
        "feature_contrast_means": feature_means,
        "source_absent_selectivity_margin_vs_controls": margin,
        "source_absent_selective_local_advantage": source_absent_selective,
        "positive_feature_markers": markers,
        "content_density_marker": content_density,
        "tail_coverage_marker": tail_cov,
        "source_absent_fraction_marker": source_abs_frac,
        "interpretation_warning": "Positive local advantage means compact-trained local NLL is lower. Event strata are engineered local probes, not natural prevalence estimates.",
    }


def summarize_a01_static(static_path: Path) -> dict[str, Any]:
    obj = read_json(static_path)
    if obj is None:
        return {"ready": False, "path": str(static_path), "reason": "A01 static label-load profile absent."}
    allg = obj.get("groups", {}).get("all", {})
    global_lr = obj.get("global_lr_weighted_fields", {})
    max_dev = global_lr.get("max_abs_batch_denom_ratio_deviation_from_1")
    lr_ratio = global_lr.get("lr_used_weighted_denom_ratio_abs_over_copied")
    expected_per_batch = global_lr.get("lr_used_weighted_mean_expected_selected_minus_abs_deleted_bpe_per_batch")
    weak_loss_load_alt = (
        finite(max_dev) and float(max_dev) < 0.002
        and finite(lr_ratio) and abs(float(lr_ratio) - 1.0) < 0.0001
        and finite(expected_per_batch) and abs(float(expected_per_batch)) < 0.5
    )
    return {
        "ready": True,
        "path": str(static_path),
        "status": obj.get("status"),
        "stream_sha256": obj.get("inputs", {}).get("stream_sha256"),
        "word_exposure_used": obj.get("word_exposure_used"),
        "abs_content_candidate_pieces": allg.get("abs_content_pieces"),
        "selected_copied_candidate_pieces": allg.get("selected_copied_pieces"),
        "selected_minus_abs_candidate_pieces": allg.get("selected_minus_abs_candidate_pieces"),
        "expected_realized_selected_minus_abs_bpe": allg.get("expected_realized_selected_minus_abs_bpe"),
        "lr_used_weighted_denom_ratio_abs_over_copied": lr_ratio,
        "max_abs_batch_denom_ratio_deviation_from_1": max_dev,
        "lr_used_weighted_expected_selected_minus_abs_deleted_bpe_per_batch": expected_per_batch,
        "loss_load_alternate_explanation_weakened": weak_loss_load_alt,
        "reading": "This static file only protects the training interface for A01 target-selective arms; it is not endpoint evidence.",
    }


def summarize_a01_postrun(postrun_path: Path) -> dict[str, Any]:
    obj = read_json(postrun_path)
    if obj is None:
        return {"ready": False, "path": str(postrun_path), "reason": "A01 target-selective postrun summary absent."}
    mech = obj.get("mechanism_readout", {})
    endpoint = mech.get("endpoint_reverse_accuracy_fields", {})
    local = mech.get("local_loss_fields", {})
    source_abs_strong_sets = local.get("source_absent_pair_interval_positive_sets", []) or []
    supp = endpoint.get("Supplement_pp")
    rel = endpoint.get("EWoK_step211_relational_pp")
    rel_adj = endpoint.get("relational_minus_adjacency_independent_pp")
    local_supported = bool(source_abs_strong_sets)
    endpoint_supported = finite(supp) and float(supp) > 0 and finite(rel) and float(rel) > 0
    aligned = local_supported and endpoint_supported
    return {
        "ready": obj.get("status") == "PACKED_TARGETSELECT_POSTRUN_READOUT_DONE",
        "path": str(postrun_path),
        "status": obj.get("status"),
        "integrity_ok": {k: v.get("ok") for k, v in (obj.get("integrity") or {}).items() if isinstance(v, dict)},
        "local_source_absent_pair_interval_positive_sets": source_abs_strong_sets,
        "endpoint_reverse_accuracy_fields": endpoint,
        "vector_similarity_reverse_to_historical": mech.get("vector_similarity_reverse_to_historical"),
        "local_supported": local_supported,
        "endpoint_supported_on_supplement_and_relational_ewok": endpoint_supported,
        "local_and_endpoint_aligned": aligned,
        "reading": "A source-absent mediation reading needs local source-absent loss movement together with Supplement and relational-EWoK endpoint movement.",
    }


def interpret(roberta: dict[str, Any], local: dict[str, Any], a01_static: dict[str, Any], a01_postrun: dict[str, Any]) -> str:
    roberta_ready = bool(roberta.get("ready"))
    local_ready = bool(local.get("ready"))
    local_late = bool(local.get("using_late_band"))
    if not roberta_ready:
        if local_ready and local_late:
            return "RoBERTa selected trajectory is not complete in the visible files. The late local source-absent response is ready and remains local pseudolikelihood evidence; no new expensive arm follows before the selected official-compatible trajectory is read. Use research semantic-local response for the relational/event concentration of this local channel."
        return "RoBERTa selected trajectory is not complete in the visible files. The repaired early local source-absent response remains only a local pseudolikelihood signal; no new expensive arm follows before the selected trajectory and late local bridge are read."
    stable = bool(roberta.get("stable_late_positive"))
    volatile = bool(roberta.get("volatile_carried_readout"))
    local_sel = bool(local.get("source_absent_selective_local_advantage")) if local_ready and local_late else False
    content_marker = "content_density" in (local.get("positive_feature_markers") or []) if local_ready and local_late else False
    a01_aligned = bool(a01_postrun.get("local_and_endpoint_aligned")) if a01_postrun.get("ready") else False

    if stable and local_sel and content_marker:
        extra = " A01 target-selective evidence is also aligned." if a01_aligned else " A01 target-selective endpoint evidence is not yet aligned or not yet visible."
        return "RoBERTa shows stable late compact-minus-repeat transfer and the late local response is source-absent-selective with content-density concentration. The next expensive work should be a paired independent-seed RoBERTa replication to test robustness before treating the compact marginal as transferable beyond one seed." + extra
    if stable and not (local_sel and content_marker):
        return "RoBERTa shows stable late compact-minus-repeat transfer, but the measured late local source-absent/content-density structure does not align. Treat the natural compact marginal as transferable in this coordinate while keeping the mechanism open; inspect item/family movement before choosing one clean data intervention."
    if volatile:
        return "RoBERTa compact movement is positive only through volatile columns. Local source-absent response, even if present, should be read as another local-vs-downstream dissociation rather than a reason to replicate."
    if (not stable) and local_ready and (local_sel or content_marker):
        return "RoBERTa has local compact source-absent/content-density response without stable downstream transfer, matching the earlier ordered/scrambled and explicit target-pressure dissociation. Bound this RoBERTa coordinate and use A01 target-selective endpoint evidence, if it becomes aligned, only to choose one natural compact-data intervention rather than another RoBERTa replication."
    return "RoBERTa is neutral or negative on stable late readouts and lacks a matching late local feature-response pattern. Bound the tested coordinate and return to compact-marginal decomposition around content density, source-wide coverage, lexical recurrence, and diversity reinvestment."


def render_md(payload: dict[str, Any]) -> str:
    r = payload["roberta_selected"]
    l = payload["roberta_local"]
    s = payload["a01_static_label_load"]
    p = payload["a01_targetselect_postrun"]
    lines = [
        "# research RoBERTa compact-result synthesis",
        "",
        f"Created: `{payload['created_utc']}`",
        "",
        "## Scientific reading",
        "",
        payload["scientific_reading"],
        "",
        "## RoBERTa selected trajectory",
        "",
        f"Ready: `{r.get('ready')}` source: `{r.get('source')}` path: `{r.get('path')}`",
        f"Late rows: `{r.get('late_rows_present')}`",
        f"Late means: `{json.dumps(r.get('late_means'), sort_keys=True)}`",
        f"Stable positive: `{r.get('stable_late_positive')}` volatile-column movement: `{r.get('volatile_carried_readout')}`",
        "",
        "## RoBERTa local pair-stratified response",
        "",
        f"Ready: `{l.get('ready')}` using late band: `{l.get('using_late_band')}` path: `{l.get('path')}`",
        f"Event SHA match: `{l.get('event_sha_matches')}` manifest: `{l.get('manifest_events_sha256')}` response: `{l.get('events_sha256')}`",
        f"Local means: `{json.dumps(l.get('local_view_category_advantage_means'), sort_keys=True)}`",
        f"Feature means: `{json.dumps(l.get('feature_contrast_means'), sort_keys=True)}`",
        "",
        "## A01 static target-label load profile",
        "",
        f"Ready: `{s.get('ready')}` weak loss-load alternative: `{s.get('loss_load_alternate_explanation_weakened')}` path: `{s.get('path')}`",
        f"Candidate pieces absent/copied: `{s.get('abs_content_candidate_pieces')}` / `{s.get('selected_copied_candidate_pieces')}`; expected realized copied-minus-absent BPE: `{s.get('expected_realized_selected_minus_abs_bpe')}`",
        f"LR-weighted denominator ratio: `{s.get('lr_used_weighted_denom_ratio_abs_over_copied')}` max batch deviation: `{s.get('max_abs_batch_denom_ratio_deviation_from_1')}`",
        "",
        "## A01 target-selective endpoint readout",
        "",
        f"Ready: `{p.get('ready')}` path: `{p.get('path')}` local+endpoint aligned: `{p.get('local_and_endpoint_aligned')}`",
        f"Local source-absent positive sets: `{p.get('local_source_absent_pair_interval_positive_sets')}`",
        f"Endpoint fields: `{json.dumps(p.get('endpoint_reverse_accuracy_fields'), sort_keys=True)}`",
        "",
        "This reader performs no training, selected evaluation, upload, or leaderboard action.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--official", type=Path, default=DEFAULT_OFFICIAL)
    ap.add_argument("--reader_report", action="append", type=Path, default=None)
    ap.add_argument("--late_stratified", type=Path, default=DEFAULT_LATE_STRAT)
    ap.add_argument("--early_stratified", type=Path, default=DEFAULT_EARLY_STRAT)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--a01_static", type=Path, default=DEFAULT_A01_STATIC)
    ap.add_argument("--a01_postrun", type=Path, default=DEFAULT_A01_POSTRUN)
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    reader_paths = args.reader_report if args.reader_report else DEFAULT_READER_REPORTS
    roberta_selected = summarize_official(args.official, reader_paths)
    roberta_local = summarize_stratified(args.late_stratified, args.early_stratified, args.manifest)
    a01_static = summarize_a01_static(args.a01_static)
    a01_postrun = summarize_a01_postrun(args.a01_postrun)
    payload = {
        "status": "ROBERTA_COMPACT_RESULT_SYNTHESIS",
        "created_utc": now_utc(),
        "meaning": "Synthesis of RoBERTa natural compact-vs-repeat transfer with local pair-stratified response and A01 source-absent target-selective evidence. Reads existing files only.",
        "inputs": {
            "official": str(args.official),
            "reader_reports": [str(p) for p in reader_paths],
            "late_stratified": str(args.late_stratified),
            "early_stratified": str(args.early_stratified),
            "manifest": str(args.manifest),
            "a01_static": str(args.a01_static),
            "a01_postrun": str(args.a01_postrun),
        },
        "roberta_selected": roberta_selected,
        "roberta_local": roberta_local,
        "a01_static_label_load": a01_static,
        "a01_targetselect_postrun": a01_postrun,
        "scientific_reading": interpret(roberta_selected, roberta_local, a01_static, a01_postrun),
        "no_leaderboard_submission": True,
        "no_training_or_evaluation_started": True,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "roberta_compact_result_synthesis.json", payload)
    (args.out_dir / "roberta_compact_result_synthesis.md").write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "out": str(args.out_dir / "roberta_compact_result_synthesis.json"),
        "roberta_selected_ready": roberta_selected.get("ready"),
        "roberta_local_ready": roberta_local.get("ready"),
        "roberta_local_using_late_band": roberta_local.get("using_late_band"),
        "a01_static_ready": a01_static.get("ready"),
        "a01_postrun_ready": a01_postrun.get("ready"),
        "scientific_reading": payload["scientific_reading"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
