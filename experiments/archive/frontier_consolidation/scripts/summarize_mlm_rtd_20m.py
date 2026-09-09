#!/usr/bin/env python3
"""research: summarize the 20M MLM+RTD-GDES screen against legal research 20M.

This is a research-facing validator/summarizer, not a final evaluator.  It checks
training exposure/checkpoints and reads the official-compatible per-target JSON
produced by evaluate_compliant_endpoint.py for cheap columns.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

CHEAP_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
ZERO_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]
GP_COLS = ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]


def read_json(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_scores(payload: dict) -> dict[str, float | None]:
    tasks = payload.get("tasks", {})
    out: dict[str, float | None] = {}
    for c in ZERO_COLUMNS:
        rec = tasks.get(c, {})
        out[c] = float(rec["score"]) if "score" in rec and rec["score"] is not None else None
    gp_vals = []
    for c in GP_COLS:
        rec = tasks.get(c, {})
        if "score" in rec and rec["score"] is not None:
            gp_vals.append(float(rec["score"]))
    out["GlobalPIQA"] = mean(gp_vals) if len(gp_vals) == 2 else None
    rd = tasks.get("Reading", {})
    if isinstance(rd.get("scores"), dict) and rd["scores"].get("Reading") is not None:
        out["Reading"] = float(rd["scores"]["Reading"])
    elif rd.get("score") is not None:
        out["Reading"] = float(rd["score"])
    else:
        out["Reading"] = None
    return out


def cheap7(scores: dict[str, float | None]) -> float | None:
    vals = [scores.get(c) for c in CHEAP_COLUMNS]
    if any(v is None for v in vals):
        return None
    return float(mean(float(v) for v in vals))


def validate_training(run_dir: Path) -> dict:
    metrics_path = run_dir / "scientific_metrics.json"
    manifest_path = run_dir / "training_manifest.json"
    log_path = run_dir / "training_log.jsonl"
    ckpt_root = run_dir / "hf_model"
    res = {
        "run_dir": str(run_dir),
        "metrics_path": str(metrics_path),
        "manifest_path": str(manifest_path),
        "log_path": str(log_path),
        "metrics_exists": metrics_path.exists(),
        "manifest_exists": manifest_path.exists(),
        "log_exists": log_path.exists(),
    }
    if metrics_path.exists():
        metrics = read_json(metrics_path)
        res["metrics"] = metrics
        res["status"] = metrics.get("status")
        res["word_exposure"] = metrics.get("word_exposure")
        res["actual_steps"] = metrics.get("actual_steps")
        res["mlm_loss_last"] = metrics.get("mlm_loss_last")
        res["rtd_loss_last"] = metrics.get("rtd_loss_last")
        res["saved_checkpoint_names"] = [x.get("name") for x in metrics.get("saved_checkpoints", [])]
    else:
        res["status"] = "missing_metrics"
    expected = ["chck_5M", "chck_10M", "chck_15M", "chck_20M"]
    res["expected_checkpoints"] = expected
    res["checkpoint_exists"] = {name: (ckpt_root / name / "model.safetensors").exists() for name in expected}
    res["checkpoint_rtd_head_exists"] = {name: (ckpt_root / name / "rtd_head.pt").exists() for name in expected}
    res["final_model_exists"] = (ckpt_root / "model.safetensors").exists()
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-json", default="experiments/archive/frontier_consolidation/data/legal20m_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_20M.json")
    ap.add_argument("--candidate-json", default="experiments/archive/frontier_consolidation/data/mlm_rtd_20M_eval/per_target/mlm_rtd_lambda1_seed43022_20M.json")
    ap.add_argument("--run-dir", default="experiments/archive/frontier_consolidation/training/runs/mlm_rtd_lambda1_seed43022_20M")
    ap.add_argument("--out-json", default="experiments/archive/frontier_consolidation/data/mlm_rtd_20M_summary/mlm_rtd_20M_summary.json")
    ap.add_argument("--out-md", default="research/documents/frontier_consolidation/data/mlm_rtd_20M_summary/mlm_rtd_20M_summary.md")
    args = ap.parse_args()

    baseline_p = Path(args.baseline_json)
    cand_p = Path(args.candidate_json)
    run_dir = Path(args.run_dir)
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    training = validate_training(run_dir)
    baseline = read_json(baseline_p) if baseline_p.exists() else None
    candidate = read_json(cand_p) if cand_p.exists() else None

    baseline_scores = extract_scores(baseline) if baseline else None
    candidate_scores = extract_scores(candidate) if candidate else None
    baseline_cheap7 = cheap7(baseline_scores) if baseline_scores else None
    candidate_cheap7 = cheap7(candidate_scores) if candidate_scores else None

    deltas = None
    if baseline_scores and candidate_scores:
        deltas = {c: (None if baseline_scores[c] is None or candidate_scores[c] is None else candidate_scores[c] - baseline_scores[c]) for c in CHEAP_COLUMNS}
        deltas["cheap7"] = None if baseline_cheap7 is None or candidate_cheap7 is None else candidate_cheap7 - baseline_cheap7

    # Scientific decision language is intentionally direct and numerical.
    interpretation = []
    if deltas and deltas.get("cheap7") is not None:
        d = float(deltas["cheap7"])
        if d >= 0.35 and all((deltas.get(c) or -999) > -0.5 for c in ["Supplement", "EWoK", "Reading"]):
            decision = "continue_candidate_to_longer_endpoint"
            interpretation.append("20M cheap7 is broadly higher than the matched legal baseline, without damaging the historically fragile Supplement/EWoK/Reading trio.")
        elif d <= -0.15 or any((deltas.get(c) is not None and deltas[c] <= -1.0) for c in ["Supplement", "EWoK", "Reading"]):
            decision = "close_or_rebuild_before_more_training"
            interpretation.append("20M score movement is negative or damages a fragile column strongly enough that ordinary maturation should not justify a 100M run.")
        else:
            decision = "needs_scientific_review_before_continuation"
            interpretation.append("20M movement is small or mixed; continuation would need mechanism-specific reasoning, not routine maturation hope.")
    else:
        decision = "await_training_or_eval"
        interpretation.append("Candidate cheap7 is not yet complete; use this script again after training and cheap-column evaluation.")

    result = {
        "status": "MLM_RTD_20M_SUMMARY",
        "training": training,
        "baseline_json": str(baseline_p),
        "candidate_json": str(cand_p),
        "baseline_scores": baseline_scores,
        "candidate_scores": candidate_scores,
        "baseline_cheap7": baseline_cheap7,
        "candidate_cheap7": candidate_cheap7,
        "deltas": deltas,
        "decision": decision,
        "interpretation": interpretation,
    }
    out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = ["# research MLM+RTD-GDES 20M screen summary", "", f"Decision: `{decision}`", ""]
    lines.append("## Training validation")
    lines.append(f"- Run dir: `{run_dir}`")
    lines.append(f"- Status: `{training.get('status')}`")
    lines.append(f"- Word exposure: {training.get('word_exposure')}")
    lines.append(f"- Actual steps: {training.get('actual_steps')}")
    lines.append(f"- Last MLM loss: {training.get('mlm_loss_last')}")
    lines.append(f"- Last RTD loss: {training.get('rtd_loss_last')}")
    lines.append(f"- Checkpoints present: {training.get('checkpoint_exists')}")
    lines.append("")
    lines.append("## Cheap-column scores")
    lines.append("| column | baseline20M | MLM+RTD20M | delta |")
    lines.append("|---|---:|---:|---:|")
    for c in CHEAP_COLUMNS:
        b = None if baseline_scores is None else baseline_scores.get(c)
        v = None if candidate_scores is None else candidate_scores.get(c)
        dd = None if deltas is None else deltas.get(c)
        lines.append(f"| {c} | {'' if b is None else f'{b:.4f}'} | {'' if v is None else f'{v:.4f}'} | {'' if dd is None else f'{dd:+.4f}'} |")
    lines.append(f"| cheap7 | {'' if baseline_cheap7 is None else f'{baseline_cheap7:.4f}'} | {'' if candidate_cheap7 is None else f'{candidate_cheap7:.4f}'} | {'' if not deltas or deltas.get('cheap7') is None else f'{deltas['cheap7']:+.4f}'} |")
    lines.append("")
    lines.append("## Interpretation")
    for s in interpretation:
        lines.append(f"- {s}")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": result["status"],
        "decision": decision,
        "training_status": training.get("status"),
        "candidate_cheap7": candidate_cheap7,
        "baseline_cheap7": baseline_cheap7,
        "delta_cheap7": None if not deltas else deltas.get("cheap7"),
        "out_json": str(out_json),
        "out_md": str(out_md),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
