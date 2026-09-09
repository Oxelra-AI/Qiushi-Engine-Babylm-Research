#!/usr/bin/env python3
"""Audit the research dualpos untied comparison nonfit."""
from __future__ import annotations
import json, math
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path("experiments/archive/representation_and_objectives/data")
OUT = ROOT / "dualpos_untied_failure_audit"
RUNS = {
    "dualpos_shared_plus": ROOT / "dualpos_fullalpha_shared_bsplus_seed29930",
    "dualpos_shared_minus": ROOT / "dualpos_fullalpha_shared_bsminus_seed29930",
    "dualpos_untied_plus": ROOT / "dualpos_fullalpha_untied_bsplus_seed29930",
    "frozen_posalign_untied_plus": ROOT / "posalign_untied_bsplus",
}

def load_jsonl(path):
    rows = []
    if not path.exists(): return rows
    with path.open() as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def all_files(run, name):
    return sorted(run.rglob(name))

def stats(xs):
    if not xs: return {"n": 0}
    return {"n": len(xs), "mean": mean(xs), "std": pstdev(xs), "min": min(xs), "max": max(xs)}

def summarize_run(run):
    comp = []
    train_comp = []
    state = []
    train_state = []
    for p in all_files(run, "comparison_predictions.jsonl"):
        comp.extend(load_jsonl(p))
    for p in all_files(run, "train_comparison_predictions.jsonl"):
        train_comp.extend(load_jsonl(p))
    for p in all_files(run, "state_predictions.jsonl"):
        state.extend(load_jsonl(p))
    for p in all_files(run, "train_state_predictions.jsonl"):
        train_state.extend(load_jsonl(p))
    def comp_summary(rows):
        if not rows: return {"n": 0}
        return {
            "n": len(rows),
            "acc": sum(int(r.get("correct", False)) for r in rows) / len(rows),
            "prob_same": stats([float(r.get("prob_same", 0.5)) for r in rows]),
            "abs_logit": stats([abs(float(r.get("logit_same", 0.0))) for r in rows]),
            "signed_margin": stats([float(r.get("signed_margin", 0.0)) for r in rows]),
            "de1_abs": stats([abs(float(r.get("d_e1", 0.0))) for r in rows]),
            "de2_abs": stats([abs(float(r.get("d_e2", 0.0))) for r in rows]),
            "near_half_frac": sum(abs(float(r.get("prob_same", 0.5)) - 0.5) < 1e-3 for r in rows)/len(rows),
        }
    def state_summary(rows):
        # one row per candidate; use candidate_index==0 to avoid duplicates for d_e
        one = [r for r in rows if int(r.get("candidate_index", -1)) == 0]
        return {"n_choice": len(one), "de_abs": stats([abs(float(r.get("d_e", 0.0))) for r in one])}
    return {"eval_comp": comp_summary(comp), "train_comp": comp_summary(train_comp),
            "eval_state": state_summary(state), "train_state": state_summary(train_state)}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = {name: summarize_run(path) for name, path in RUNS.items()}
    (OUT / "dualpos_untied_failure_audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = ["# research dualpos untied comparison nonfit audit", "",
             "| run | train cmp acc | train prob mean | train near-half | train abs logit mean | eval cmp acc | eval abs logit mean | eval state abs d_e mean |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in result.items():
        tc = r["train_comp"]; ec = r["eval_comp"]; es = r["eval_state"]
        lines.append(f"| {name} | {tc.get('acc',0):.3f} | {tc.get('prob_same',{}).get('mean',0):.6f} | {tc.get('near_half_frac',0):.3f} | {tc.get('abs_logit',{}).get('mean',0):.6f} | {ec.get('acc',0):.3f} | {ec.get('abs_logit',{}).get('mean',0):.6f} | {es.get('de_abs',{}).get('mean',0):.3f} |")
    ((OUT.parents[4] / 'research/documents/representation_and_objectives/data/dualpos_untied_failure_audit/dualpos_untied_failure_audit.md')).write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "DUALPOS_UNTIED_FAILURE_AUDIT_COMPLETE", "md": str((OUT.parents[4] / 'research/documents/representation_and_objectives/data/dualpos_untied_failure_audit/dualpos_untied_failure_audit.md')), "json": str(OUT / "dualpos_untied_failure_audit.json")}, indent=2))

if __name__ == "__main__":
    main()
