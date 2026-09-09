#!/usr/bin/env python3
"""Focused analysis for research fit-repaired one-seed run and calibration."""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np

FIT = Path("experiments/archive/representation_and_objectives/data/independent_orientation_cross_fitrepair1/independent_orientation_cross_summary.json")
CAL = Path("experiments/archive/representation_and_objectives/data/independent_orientation_cross_calibration/independent_orientation_cross_summary.json")
OUT = Path("experiments/archive/representation_and_objectives/data/fitrepair_analysis")
OUT.mkdir(parents=True, exist_ok=True)

QUERIES = ["event_role", "focal_state", "untouched_state"]
ARMS = ["exposure", "Etrue_Rtrue", "Eflip_Rtrue", "Etrue_Rflip", "Eflip_Rflip"]
KEY_EVALS = [
    "atp_trainTrain_trainHyp", "atp_trainTrain_heldHyp", "atp_heldHeld_trainHyp", "atp_heldHeld_heldHyp",
    "atp_dirDir_trainHyp", "atp_dirDir_heldHyp",
    "atp_trainEvent_dirState_trainHyp", "atp_dirEvent_trainState_trainHyp",
    "atp_trainEvent_nonState_trainHyp", "atp_nonEvent_trainState_trainHyp",
]
PAIR_EVALS = [
    "pair_trainCtx_trainHyp", "pair_trainCtx_heldHyp", "pair_heldCtx_trainHyp", "pair_heldCtx_heldHyp",
    "pair_dirparaCtx_trainHyp", "pair_dirparaCtx_heldHyp", "pair_nonparaCtx_trainHyp", "pair_nonparaCtx_heldHyp",
]


def jload(p): return json.loads(p.read_text("utf-8"))

def getq(agg, arm, ev, q):
    return agg[arm]["evals"][ev]["contrastive_by_query"].get(q, {}).get("mean", float("nan"))

def getcf(agg, arm, ev, key):
    return agg[arm]["evals"][ev]["contrastive_by_conflict"].get(key, {}).get("mean", float("nan"))

def fmt(x):
    return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def main():
    fit = jload(FIT)
    cal = jload(CAL).get("semantic_calibration", {})
    agg = fit["aggregate"]
    out = {"train_acc": {a: agg[a]["train_acc"] for a in ARMS}, "key_metrics": {}, "pair_metrics": {}, "calibration": cal}
    for ev in KEY_EVALS:
        out["key_metrics"][ev] = {a: {q: getq(agg, a, ev, q) for q in QUERIES} for a in ARMS}
        out["key_metrics"][ev]["selective"] = {
            "event_flip_effect": {q: getq(agg, "Etrue_Rtrue", ev, q)-getq(agg, "Eflip_Rtrue", ev, q) for q in QUERIES},
            "rank_flip_effect": {q: getq(agg, "Etrue_Rtrue", ev, q)-getq(agg, "Etrue_Rflip", ev, q) for q in QUERIES},
            "joint_flip_effect": {q: getq(agg, "Etrue_Rtrue", ev, q)-getq(agg, "Eflip_Rflip", ev, q) for q in QUERIES},
        }
        out["key_metrics"][ev]["conflict_selective"] = {
            "event_flip_focal_conflict": getcf(agg, "Etrue_Rtrue", ev, "focal_state_conflict")-getcf(agg, "Eflip_Rtrue", ev, "focal_state_conflict"),
            "rank_flip_focal_conflict": getcf(agg, "Etrue_Rtrue", ev, "focal_state_conflict")-getcf(agg, "Etrue_Rflip", ev, "focal_state_conflict"),
            "event_flip_untouched_conflict": getcf(agg, "Etrue_Rtrue", ev, "untouched_state_conflict")-getcf(agg, "Eflip_Rtrue", ev, "untouched_state_conflict"),
            "rank_flip_untouched_conflict": getcf(agg, "Etrue_Rtrue", ev, "untouched_state_conflict")-getcf(agg, "Etrue_Rflip", ev, "untouched_state_conflict"),
        }
    for ev in PAIR_EVALS:
        out["pair_metrics"][ev] = {a: getq(agg, a, ev, "event_role") for a in ARMS}
    (OUT/"fitrepair_analysis.json").write_text(json.dumps(out, indent=2, ensure_ascii=False)+"\n", "utf-8")

    lines=[]
    lines.append("# research fit-repaired independent-orientation analysis\n\n")
    lines.append("This note analyzes the one-seed fit-repaired run in which all five arms reached training accuracy 1.0. It should be treated as a fit-verified pilot awaiting the 3-seed replication, not as final evidence.\n\n")
    lines.append("## Context wording calibration\n\n")
    lines.append("Mean-pooled pretrained DeBERTa cosine between train contexts and alternative contexts:\n\n")
    for k,v in cal.items():
        lines.append(f"- {k}: mean {v['mean']:.3f}, std {v['std']:.3f}, min {v['min']:.3f}, n {v['n']}\n")
    lines.append("\nThe nonparaphrase state context unexpectedly has high cosine (about 0.742), close to the directional state paraphrase (about 0.749), even though it does not encode higher/lower direction. Mean-pooled cosine therefore measures topical/lexical similarity more than relation-direction equivalence. Nonparaphrase performance, not cosine alone, is the important context-side calibration.\n\n")
    lines.append("## Fit-repaired arm training\n\n")
    for a in ARMS:
        tr=out['train_acc'][a]
        lines.append(f"- {a}: train acc {tr['mean']:.3f}\n")
    lines.append("\n## ATP compound contrastive metrics\n\n")
    for ev in KEY_EVALS:
        lines.append(f"### {ev}\n\n")
        lines.append("| arm | event | focal | secondary |\n|---|---:|---:|---:|\n")
        for a in ARMS:
            d=out['key_metrics'][ev][a]
            lines.append(f"| {a} | {fmt(d['event_role'])} | {fmt(d['focal_state'])} | {fmt(d['untouched_state'])} |\n")
        se=out['key_metrics'][ev]['selective']
        lines.append(f"\nEvent-flip effect on event/focal/secondary: {fmt(se['event_flip_effect']['event_role'])}/{fmt(se['event_flip_effect']['focal_state'])}/{fmt(se['event_flip_effect']['untouched_state'])}.\n")
        lines.append(f"Rank-flip effect on event/focal/secondary: {fmt(se['rank_flip_effect']['event_role'])}/{fmt(se['rank_flip_effect']['focal_state'])}/{fmt(se['rank_flip_effect']['untouched_state'])}.\n\n")
    lines.append("## Pair-domain event transfer\n\n")
    lines.append("| eval | exposure | Etrue_Rtrue | Eflip_Rtrue | Etrue_Rflip | Eflip_Rflip |\n|---|---:|---:|---:|---:|---:|\n")
    for ev in PAIR_EVALS:
        d=out['pair_metrics'][ev]
        lines.append(f"| {ev} | {fmt(d['exposure'])} | {fmt(d['Etrue_Rtrue'])} | {fmt(d['Eflip_Rtrue'])} | {fmt(d['Etrue_Rflip'])} | {fmt(d['Eflip_Rflip'])} |\n")
    lines.append("\n## Scientific reading\n\n")
    lines.append("1. Familiar trainTrain/trainHyp contexts show clear selective control: Etrue_Rtrue gives 1.000/1.000/1.000; Eflip_Rtrue gives event 0.004 with focal 1.000 and secondary 0.992; Etrue_Rflip gives event 1.000, focal 0.196, secondary 0.988. This is not a head-wide label convention. It demonstrates that sparse event orientation can be inverted without inverting ranking facts, and sparse focal-rank orientation can be inverted while the secondary ranking fact remains mostly conserved.\n\n")
    lines.append("2. The secondary ranking fact is not simply dragged by the focal-rank flip on familiar contexts: Etrue_Rflip preserves secondary at 0.988 and Eflip_Rflip preserves secondary at 0.967 while focal ranking drops. This is the first controlled evidence of query/position-specific conservation under an intentionally wrong focal orientation.\n\n")
    lines.append("3. Context side remains the hard boundary. Held/held and dirDir state contexts collapse for focal and secondary state despite full train fit; event often remains strong. This means research's failure was not merely head calibration. It also means the law should be stated as compositional use of oriented evidence only when the directional context interface is grounded by the learner, not as a cosine or raw-frequency threshold.\n\n")
    lines.append("4. Nonparaphrase controls behave as relation-removal controls: when state context is replaced by mention-only text, focal and secondary approach chance even if event is perfectly controlled; when event context is mention-only but state context remains train wording, event is chance while state readout is strong. This is better evidence of relation-specific context use than mean-pooled cosine.\n\n")
    lines.append("5. Pair-domain event transfer needs caution: in Eflip_Rtrue/Eflip_Rflip, train-context pair event rows invert on familiar context but held/dir/non contexts can revert or collapse depending on context wording. Event orientation is therefore not universally domain/wording invariant; it is strongest on the sparse-training context family and should be evaluated with context controls.\n\n")
    lines.append("The pending 3-seed run should test whether these selective effects survive seed variation. If it replicates, the next construction problem is to strengthen context-side grounding for ranking direction without training the final held wording directly, e.g. by adding directional paraphrase bridges or contrastive context objectives that preserve independent event/rank labels and secondary-state truth.\n")
    ((OUT.parents[4] / 'research/documents/representation_and_objectives/data/fitrepair_analysis/fitrepair_analysis.md')).write_text("".join(lines), "utf-8")
    print(json.dumps({"status":"FITREPAIR_ANALYSIS_DONE", "md":str((OUT.parents[4] / 'research/documents/representation_and_objectives/data/fitrepair_analysis/fitrepair_analysis.md')), "json":str(OUT/'fitrepair_analysis.json')}))

if __name__ == '__main__':
    main()
