#!/usr/bin/env python3
"""research: Score the state-margin probe on designed DeBERTa relation arms.

Runs the research T/U/N state-margin scorer on the frontier_consolidation DeBERTa relation arms
(CLEAN, REPEAT, VIEW) and relation_learning split arms (REPEAT_SPLIT, VIEW_SPLIT) at
seed43022 final checkpoint, using the same probe sets as the state-update
intervention scoring.

Purpose: connect the state-margin recency-reader characterization of the base
model to the established causal locality and relation-typed readout results.
If REPEAT shows stale-state retention and REPEAT_SPLIT collapses toward CLEAN,
the same within-window locality mechanism operates on entity-state reading,
giving the paper its fourth independent readout.

Predictions written before scoring:
  1. REPEAT T(orig_new - source) < CLEAN: exact recurrence training biases toward
     source-state retention even when a true update is present.
  2. REPEAT_SPLIT ≈ CLEAN: removing within-window locality collapses the effect.
  3. VIEW may differ from REPEAT in the direction of better changed-form handling
     (positive or at least less negative T(orig_new - source) than REPEAT),
     since VIEW was trained with varied restatements rather than exact copies.
  4. VIEW_SPLIT ≈ CLEAN: removing within-window locality collapses the VIEW effect.

Implementation: patches research MODULE_ROOTS with the relation arm paths and
calls the same scoring engine.  Only raw T/U/N margins matter here; paired
arm deltas will be empty because the arms do not share a "base"/"intervention"
label structure.  Post-processing computes arm-minus-CLEAN differences.
"""
from __future__ import annotations

import csv
import json
import math
import pathlib
import statistics
import sys
import time

ROOT = pathlib.Path.cwd()

# ── Pre-scoring prediction record ──────────────────────────────────────
PREDICTIONS = {
    "step": 55,
    "purpose": "Fourth readout: tie state-margin recency-reader to causal locality",
    "predictions": [
        "REPEAT T(orig_new - source) < CLEAN T(orig_new - source) for UPDATED_USE packets: stale attraction",
        "REPEAT_SPLIT T(orig_new - source) ≈ CLEAN T(orig_new - source): locality removed",
        "VIEW T(orig_new - source) > REPEAT T(orig_new - source): better changed-form handling",
        "VIEW_SPLIT ≈ CLEAN: locality removed",
        "On DISTRACTOR, REPEAT should show stronger source-state retention (T/U retention margins)",
    ],
    "decision_rule": "If predictions 1-2 hold, the state-margin probe confirms causal locality on state behavior; this joins the paper. If 1 fails, the state-margin probe is insensitive to the relation arm axis."
}

# ── Model paths ────────────────────────────────────────────────────────
ARM_MODELS = {
    "seed43022_clean": {
        "seed": "43022", "arm": "clean",
        "root": ROOT / "experiments/archive/frontier_consolidation/training/runs/full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43022/hf_model",
    },
    "seed43022_repeat": {
        "seed": "43022", "arm": "repeat",
        "root": ROOT / "experiments/archive/frontier_consolidation/training/runs/full_p2c_c2p_abs_repeat_dose2p64x_matched_rowholdout_deberta100M_seed43022/hf_model",
    },
    "seed43022_view": {
        "seed": "43022", "arm": "view",
        "root": ROOT / "experiments/archive/frontier_consolidation/training/runs/full_p2c_c2p_abs_view_dose2p64x_matched_rowholdout_deberta100M_seed43022/hf_model",
    },
    "seed43022_repeat_split": {
        "seed": "43022", "arm": "repeat_split",
        "root": ROOT / "experiments/archive/relation_learning/training/runs/full_p2c_c2p_abs_repeat_split_dose2p64x_rowholdout_deberta100M_seed43022/hf_model",
    },
    "seed43022_view_split": {
        "seed": "43022", "arm": "view_split",
        "root": ROOT / "experiments/archive/relation_learning/training/runs/full_p2c_c2p_abs_view_split_dose2p64x_rowholdout_deberta100M_seed43022/hf_model",
    },
}

OUT_DIR = ROOT / "experiments/archive/relation_learning/data/relation_arm_state_margin"


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def se(vals):
    xs = [float(x) for x in vals if math.isfinite(float(x))]
    if len(xs) <= 1:
        return float("nan")
    return statistics.stdev(xs) / math.sqrt(len(xs))


def mean_vals(vals):
    xs = [float(x) for x in vals if math.isfinite(float(x))]
    return float(statistics.mean(xs)) if xs else float("nan")


def run_scoring():
    """Patch research MODEL_ROOTS and call main()."""
    script_dir = str(ROOT / "experiments/archive/relation_learning/scripts")
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    import score_state_margin_probe as scorer

    # Patch MODEL_ROOTS with relation arms
    scorer.MODEL_ROOTS.update(ARM_MODELS)

    # Write prediction record before scoring
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = OUT_DIR / "pre_score_predictions.json"
    pred_path.write_text(json.dumps(PREDICTIONS, indent=2) + "\n")
    print(f"[PREDICTIONS] Written to {pred_path}", flush=True)

    # Run scorer with all 5 arms at final checkpoint
    model_names = list(ARM_MODELS.keys())
    sys.argv = [
        "scorer",
        "--models", *model_names,
        "--checkpoints", "final",
        "--out-dir", str(OUT_DIR / "raw_scores"),
        "--device", "cuda:0",
        "--batch-size", "256",
        "--skip-missing",
    ]
    print(f"[RUN] Scoring {len(model_names)} models at final checkpoint", flush=True)
    scorer.main()
    print("[DONE] Raw scoring complete", flush=True)


def compute_arm_minus_clean():
    """Post-process: compute arm-minus-CLEAN raw margin differences."""
    raw_dir = OUT_DIR / "raw_scores"
    margin_path = raw_dir / "raw_TUN_margin_rows.csv"
    if not margin_path.exists():
        print(f"[SKIP] No margin rows at {margin_path}", flush=True)
        return

    with margin_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Index by (packet_id, probe_set, condition, slot_mode) -> arm -> row
    from collections import defaultdict
    by_key = defaultdict(dict)
    for r in rows:
        key = (r["packet_id"], r["probe_set"], r["condition"], r["slot_mode"])
        by_key[key][r["arm"]] = r

    # Fields to compare
    fields = [
        "full_m_orig_new_minus_source", "full_m_source_minus_orig_new",
        "full_m_source_minus_swapped_new", "full_m_orig_new_minus_swapped_new",
        "full_gold_margin_vs_original_competitor", "full_retention_margin_vs_swapped_new",
        "content_m_orig_new_minus_source", "content_m_source_minus_orig_new",
        "content_m_source_minus_swapped_new", "content_m_orig_new_minus_swapped_new",
        "content_gold_margin_vs_original_competitor", "content_retention_margin_vs_swapped_new",
    ]

    # Compute per-packet arm-minus-clean deltas
    delta_rows = []
    for key, arms in by_key.items():
        if "clean" not in arms:
            continue
        clean = arms["clean"]
        for arm_name in ["repeat", "view", "repeat_split", "view_split"]:
            if arm_name not in arms:
                continue
            other = arms[arm_name]
            rec = {
                "packet_id": key[0], "probe_set": key[1], "condition": key[2],
                "slot_mode": key[3], "comparison": f"{arm_name}_minus_clean",
                "packet_type": other.get("packet_type", ""),
                "arm": arm_name,
            }
            for f in fields:
                cv, ov = clean.get(f), other.get(f)
                if cv is not None and ov is not None:
                    try:
                        rec[f"delta_{f}"] = float(ov) - float(cv)
                        rec[f"clean_{f}"] = float(cv)
                        rec[f"arm_{f}"] = float(ov)
                    except (ValueError, TypeError):
                        pass
            delta_rows.append(rec)

    delta_path = OUT_DIR / "arm_minus_clean_delta_rows.csv"
    if delta_rows:
        keys = list(delta_rows[0].keys())
        for r in delta_rows[1:]:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with delta_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(delta_rows)
        print(f"[DONE] Delta rows: {len(delta_rows)} -> {delta_path}", flush=True)

    # Summaries by (comparison, packet_type, condition)
    summary_groups = defaultdict(list)
    for r in delta_rows:
        gk = (r["comparison"], r["packet_type"], r["condition"], r["probe_set"])
        summary_groups[gk].append(r)

    summary_rows = []
    for (comp, ptype, cond, pset), vals in sorted(summary_groups.items()):
        rec = {
            "comparison": comp, "packet_type": ptype, "condition": cond,
            "probe_set": pset,
            "n_packets": len({v["packet_id"] for v in vals}),
        }
        for f in fields:
            dcol = f"delta_{f}"
            xs = [float(v[dcol]) for v in vals if dcol in v and math.isfinite(float(v.get(dcol, float("nan"))))]
            rec[f"mean_{dcol}"] = mean_vals(xs) if xs else float("nan")
            rec[f"se_{dcol}"] = se(xs) if xs else float("nan")
        summary_rows.append(rec)

    summary_path = OUT_DIR / "arm_minus_clean_delta_summary.csv"
    if summary_rows:
        keys = list(summary_rows[0].keys())
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(summary_rows)
        print(f"[DONE] Summary: {len(summary_rows)} -> {summary_path}", flush=True)

    # Also produce a per-arm raw margin summary for direct comparison
    raw_summary_path = raw_dir / "raw_TUN_margin_summary.csv"
    if raw_summary_path.exists():
        with raw_summary_path.open(encoding="utf-8") as f:
            raw_summary = list(csv.DictReader(f))
        # Filter to key entries for concise table
        key_fields = [
            "mean_full_m_orig_new_minus_source", "se_full_m_orig_new_minus_source",
            "mean_full_m_source_minus_orig_new", "se_full_m_source_minus_orig_new",
            "mean_content_m_orig_new_minus_source", "se_content_m_orig_new_minus_source",
        ]
        concise = []
        for r in raw_summary:
            c = {k: r.get(k) for k in ["arm", "checkpoint", "probe_set", "slot_mode",
                                         "packet_type", "condition", "grouping", "n_packets"]}
            for f in key_fields:
                c[f] = r.get(f)
            concise.append(c)
        concise_path = OUT_DIR / "per_arm_key_margins.csv"
        if concise:
            with concise_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(concise[0].keys()), extrasaction="ignore")
                w.writeheader()
                w.writerows(concise)
            print(f"[DONE] Key margins: {len(concise)} -> {concise_path}", flush=True)

    # Write note
    write_note(summary_rows, delta_rows)


def write_note(summary_rows, delta_rows):
    """Write interpretive note."""
    lines = [f"# research relation-arm state-margin result\n\n"]
    lines.append(f"Created UTC: {now_utc()}\n\n")
    lines.append("## Purpose\n\n")
    lines.append("Connect the state-margin recency-reader characterization to the established "
                 "causal locality result. If REPEAT shows stale-state retention and REPEAT_SPLIT "
                 "collapses toward CLEAN, the state-margin probe is a fourth independent readout.\n\n")
    lines.append("## Pre-scored predictions\n\n")
    for p in PREDICTIONS["predictions"]:
        lines.append(f"- {p}\n")
    lines.append(f"\nDecision rule: {PREDICTIONS['decision_rule']}\n\n")

    def fmt(x):
        try:
            return f"{float(x):+.4f}"
        except (ValueError, TypeError):
            return "NA"

    # Key comparison table
    lines.append("## Key result: T(orig_new - source) for UPDATED_USE packets\n\n")
    lines.append("This is the primary state-update-use margin. More positive = model prefers "
                 "new state after update. Less positive / negative = stale attraction.\n\n")
    lines.append("| comparison | probe_set | n | full Δ(arm-clean) | SE | content Δ | SE |\n")
    lines.append("|---|---|---:|---:|---:|---:|---:|\n")
    for r in summary_rows:
        if r.get("condition") == "T" and r.get("packet_type") == "UPDATED_USE":
            lines.append(
                f"| {r['comparison']} | {r['probe_set']} | {r['n_packets']} | "
                f"{fmt(r.get('mean_delta_full_m_orig_new_minus_source'))} | "
                f"{fmt(r.get('se_delta_full_m_orig_new_minus_source'))} | "
                f"{fmt(r.get('mean_delta_content_m_orig_new_minus_source'))} | "
                f"{fmt(r.get('se_delta_content_m_orig_new_minus_source'))} |\n"
            )

    lines.append("\n## Key result: source retention for UNCHANGED_DISTRACTOR_USE packets\n\n")
    lines.append("| comparison | condition | probe_set | n | full Δ(source-new) | SE | content Δ | SE |\n")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|\n")
    for r in summary_rows:
        if r.get("packet_type") == "UNCHANGED_DISTRACTOR_USE" and r.get("condition") in ("T", "U"):
            lines.append(
                f"| {r['comparison']} | {r['condition']} | {r['probe_set']} | {r['n_packets']} | "
                f"{fmt(r.get('mean_delta_full_m_source_minus_orig_new'))} | "
                f"{fmt(r.get('se_delta_full_m_source_minus_orig_new'))} | "
                f"{fmt(r.get('mean_delta_content_m_source_minus_orig_new'))} | "
                f"{fmt(r.get('se_delta_content_m_source_minus_orig_new'))} |\n"
            )

    lines.append("\n## Interpretation\n\n")
    lines.append("Read the delta tables against the predictions. If locality predictions hold, "
                 "this readout joins the paper alongside compact probes, Wikipedia probes, and "
                 "Entity official evaluation as the fourth independent confirmation of relation-typed "
                 "source-conditioned readout with causal locality dependence.\n")

    note_path = OUT_DIR / "relation_arm_state_margin_note.md"
    note_path.write_text("".join(lines), encoding="utf-8")
    print(f"[NOTE] {note_path}", flush=True)


def main():
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 1: run scoring engine
    run_scoring()

    # Phase 2: compute arm-minus-CLEAN deltas
    compute_arm_minus_clean()

    elapsed = time.time() - started
    meta = {
        "status": "RELATION_ARM_STATE_MARGIN_DONE",
        "created_utc": now_utc(),
        "elapsed_sec": round(elapsed, 1),
        "models_scored": list(ARM_MODELS.keys()),
        "outputs": {
            "raw_scores": str(OUT_DIR / "raw_scores"),
            "arm_minus_clean_deltas": str(OUT_DIR / "arm_minus_clean_delta_rows.csv"),
            "arm_minus_clean_summary": str(OUT_DIR / "arm_minus_clean_delta_summary.csv"),
            "key_margins": str(OUT_DIR / "per_arm_key_margins.csv"),
            "note": str(OUT_DIR / "relation_arm_state_margin_note.md"),
            "predictions": str(OUT_DIR / "pre_score_predictions.json"),
        },
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
