#!/usr/bin/env python3
"""Analyze research replication and research independent-orientation pilot."""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np

S262 = Path("experiments/archive/representation_and_objectives/data/contrastive_wording_cross/contrastive_wording_cross_summary.json")
S263 = Path("experiments/archive/representation_and_objectives/data/independent_orientation_cross_pilot/independent_orientation_cross_summary.json")
OUT = Path("experiments/archive/representation_and_objectives/data/orientation_result_analysis")
OUT.mkdir(parents=True, exist_ok=True)


def load(p):
    return json.loads(p.read_text("utf-8"))


def stat(vals):
    vals = [float(v) for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "n": 0, "vals": []}
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals), "vals": vals}


def s262_vals(obj, arm, ev, q, where="query"):
    vals = []
    for r in obj["per_seed"]:
        if r["arm"] != arm:
            continue
        ce = r["evals"][ev]["contrastive"]
        if where == "query":
            vals.append(ce["by_query"].get(q))
        else:
            vals.append(ce["by_conflict"].get(q))
    return stat(vals)


def s263_metric(agg, arm, ev, q, conflict=None):
    if arm not in agg or ev not in agg[arm]["evals"]:
        return float("nan")
    e = agg[arm]["evals"][ev]
    if conflict is None:
        return e["contrastive_by_query"][q]["mean"]
    return e["contrastive_by_conflict"].get(f"{q}_{conflict}", {}).get("mean", float("nan"))


def fmt(x):
    if isinstance(x, dict):
        return f"{x['mean']:.3f}±{x['std']:.3f} (n={x['n']}; {', '.join(f'{v:.3f}' for v in x['vals'])})"
    return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def main():
    s262 = load(S262)
    s263 = load(S263)
    analysis = {"research": {}, "research": {}}

    key_evals_262 = ["atp_trainCtx_trainHyp", "atp_trainCtx_heldHyp", "atp_heldCtx_trainHyp", "atp_heldCtx_heldHyp"]
    arms262 = ["exposure", "aligned", "flipped"]
    queries = ["event_role", "focal_state", "untouched_state"]
    for ev in key_evals_262:
        analysis["research"][ev] = {}
        for arm in arms262:
            analysis["research"][ev][arm] = {q: s262_vals(s262, arm, ev, q) for q in queries}
        analysis["research"][ev]["aligned_minus_flipped"] = {
            q: stat([a - b for a, b in zip(
                analysis["research"][ev]["aligned"][q]["vals"],
                analysis["research"][ev]["flipped"][q]["vals"]
            )]) for q in queries
        }
        analysis["research"][ev]["aligned_minus_exposure"] = {
            q: stat([a - b for a, b in zip(
                analysis["research"][ev]["aligned"][q]["vals"],
                analysis["research"][ev]["exposure"][q]["vals"]
            )]) for q in queries
        }

    agg = s263["aggregate"]
    key_evals_263 = [
        "atp_trainTrain_trainHyp", "atp_trainTrain_heldHyp",
        "atp_heldHeld_trainHyp", "atp_heldHeld_heldHyp",
        "atp_dirDir_trainHyp", "atp_trainEvent_nonState_trainHyp",
        "atp_nonEvent_trainState_trainHyp",
    ]
    arms263 = ["exposure", "Etrue_Rtrue", "Eflip_Rtrue", "Etrue_Rflip", "Eflip_Rflip"]
    analysis["research"]["train_acc"] = {a: agg[a]["train_acc"] for a in arms263}
    analysis["research"]["metrics"] = {}
    for ev in key_evals_263:
        analysis["research"]["metrics"][ev] = {
            arm: {q: s263_metric(agg, arm, ev, q) for q in queries} for arm in arms263
        }
        analysis["research"]["metrics"][ev]["selective_effects"] = {
            "event_flip_on_event": s263_metric(agg, "Etrue_Rtrue", ev, "event_role") - s263_metric(agg, "Eflip_Rtrue", ev, "event_role"),
            "event_flip_on_focal": s263_metric(agg, "Etrue_Rtrue", ev, "focal_state") - s263_metric(agg, "Eflip_Rtrue", ev, "focal_state"),
            "event_flip_on_untouched": s263_metric(agg, "Etrue_Rtrue", ev, "untouched_state") - s263_metric(agg, "Eflip_Rtrue", ev, "untouched_state"),
            "rank_flip_on_event": s263_metric(agg, "Etrue_Rtrue", ev, "event_role") - s263_metric(agg, "Etrue_Rflip", ev, "event_role"),
            "rank_flip_on_focal": s263_metric(agg, "Etrue_Rtrue", ev, "focal_state") - s263_metric(agg, "Etrue_Rflip", ev, "focal_state"),
            "rank_flip_on_untouched": s263_metric(agg, "Etrue_Rtrue", ev, "untouched_state") - s263_metric(agg, "Etrue_Rflip", ev, "untouched_state"),
        }

    (OUT / "orientation_result_analysis.json").write_text(json.dumps(analysis, indent=2, ensure_ascii=False) + "\n", "utf-8")

    lines = []
    lines.append("# research analysis of research replication and independent-orientation pilot\n\n")
    lines.append("## research full 3-seed replication: contrastive accuracy by query\n\n")
    for ev in key_evals_262:
        lines.append(f"### {ev}\n\n")
        lines.append("| arm/contrast | event | focal_state | untouched_state |\n")
        lines.append("|---|---:|---:|---:|\n")
        for arm in arms262:
            d = analysis["research"][ev][arm]
            lines.append(f"| {arm} | {fmt(d['event_role'])} | {fmt(d['focal_state'])} | {fmt(d['untouched_state'])} |\n")
        for c in ["aligned_minus_flipped", "aligned_minus_exposure"]:
            d = analysis["research"][ev][c]
            lines.append(f"| {c} | {fmt(d['event_role'])} | {fmt(d['focal_state'])} | {fmt(d['untouched_state'])} |\n")
        lines.append("\n")

    lines.append("## research interpretation after replication\n\n")
    lines.append("The stable result is not a universal cosine/frequency law.  The train-context/held-hypothesis surface replicates for focal and secondary state even though the held state hypothesis word 'outranked' was rare/absent in the 10M corpus; the held-context surfaces remain weak and high-variance for state.  Thus the asymmetry is specifically context-side extraction/grounding under the compound ATP context, not a raw threshold on isolated string cosine.  Joint flipping in research only establishes that a trained classifier can invert the queried labels on familiar surfaces; it does not identify shared coordinates or untouched conservation because event, focal state, and secondary state were all flipped together.\n\n")

    lines.append("## research one-seed independent-orientation pilot\n\n")
    lines.append("Training labels: event and focal ranking sparse labels vary independently; secondary ranking labels remain true. Evaluation labels are always true facts.\n\n")
    lines.append("### Train accuracy\n\n")
    for arm, st in analysis["research"]["train_acc"].items():
        lines.append(f"- {arm}: {st['mean']:.3f} (std {st['std']:.3f}, n={st['n']})\n")
    lines.append("\n")
    lines.append("### Key contrastive metrics\n\n")
    for ev in key_evals_263:
        lines.append(f"#### {ev}\n\n")
        lines.append("| arm | event | focal_state | untouched_state |\n")
        lines.append("|---|---:|---:|---:|\n")
        for arm in arms263:
            d = analysis["research"]["metrics"][ev][arm]
            lines.append(f"| {arm} | {d['event_role']:.3f} | {d['focal_state']:.3f} | {d['untouched_state']:.3f} |\n")
        eff = analysis["research"]["metrics"][ev]["selective_effects"]
        lines.append(f"\nSelective effects relative to Etrue_Rtrue: event_flip(event/focal/untouched) = {eff['event_flip_on_event']:.3f}/{eff['event_flip_on_focal']:.3f}/{eff['event_flip_on_untouched']:.3f}; rank_flip(event/focal/untouched) = {eff['rank_flip_on_event']:.3f}/{eff['rank_flip_on_focal']:.3f}/{eff['rank_flip_on_untouched']:.3f}.\n\n")

    lines.append("## research pilot reading\n\n")
    lines.append("The pilot validates the new design only partially.  Eflip_Rtrue is the cleanest fit-interpretable arm: it reaches train accuracy 1.000 and on familiar trainTrain/trainHyp context inverts event (0.000) while preserving focal and secondary ranking at 1.000/1.000.  This rules out a purely head-wide polarity explanation for that arm and shows that event orientation can be changed without destroying ranking readouts when the context and hypothesis are familiar.  On trainTrain/heldHyp it still preserves ranking strongly (focal 0.850, secondary 0.994) while event is only partly inverted/transferred (0.275 true accuracy), consistent with held event hypotheses being less clean under flipped event evidence.\n\n")
    lines.append("The ranking-flip arms are not yet decisive: Etrue_Rtrue and Etrue_Rflip did not fully fit in this small 4-epoch pilot (0.869 and 0.844 train accuracy).  Their selective effects therefore cannot be used as evidence about rank-coordinate independence.  The Eflip_Rflip arm nearly fits (0.984) but focal/secondary behavior is mixed, likely because focal ranking labels are flipped while secondary ranking labels remain true, forcing the learner to use positional/query-specific information rather than a single ranking coordinate.\n\n")
    lines.append("The nonparaphrase context controls behave as intended in the clean Eflip_Rtrue arm: with event context preserved and state context replaced by mention-only text, event remains inverted (0.013 true accuracy) while focal/secondary ranking fall near chance; with event context mention-only and state context preserved, event is near chance (0.531) while focal/secondary ranking remain true (0.994/1.000).  This shows the model is not merely transferring one global output polarity; it uses whichever directional context is present.  However this is still one seed and small rows.\n\n")
    lines.append("Next experiment should be a fit-repaired version, not a larger BabyLM trajectory: keep the same crossed arms but raise epochs/row count or rebalance sparse/base mixture until Etrue_Rtrue and Etrue_Rflip fit, then repeat two or three seeds.  The primary readout should be selective movement on conflict worlds for trainTrain, heldHeld, dirDir, trainEvent_nonState, and nonEvent_trainState.\n")

    ((OUT.parents[4] / 'research/documents/representation_and_objectives/data/orientation_result_analysis/orientation_result_analysis.md')).write_text("".join(lines), "utf-8")
    print(json.dumps({"status": "ANALYSIS_DONE", "out_md": str((OUT.parents[4] / 'research/documents/representation_and_objectives/data/orientation_result_analysis/orientation_result_analysis.md')), "out_json": str(OUT / "orientation_result_analysis.json")}, indent=2))

if __name__ == "__main__":
    main()
