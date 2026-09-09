#!/usr/bin/env python3
"""research: corrected readout of research orientation evidence.

The purpose is to preserve the corrected scientific interpretation of the
research/279 symmetry-identification work from existing files only. No model
loading, training, official evaluation, or upload.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev

STUDY = Path("experiments/archive/representation_and_objectives")
WS = STUDY
SUB = WS / "data" / "equivariant_symmetry_substrate"
FINE = WS / "data" / "finetune_orientation_probe" / "finetune_orientation_results.json"
FROZEN_PRE = WS / "data" / "frozen_pretrained_probe" / "frozen_pretrained_results.json"
FROZEN_RAND = WS / "data" / "frozen_random_probe" / "frozen_random_results.json"
OUT = WS / "data" / "orientation_evidence_readout"

ARMS = [
    "exposure_only",
    "heldheld_only",
    "aligned_state_bridge",
    "inverted_state_bridge",
    "neutral_decoupled",
    "mixed_event_bridge",
]


def load_jsonl(path: Path):
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def finite_vals(rows, key):
    vals = []
    for r in rows:
        v = r.get(key)
        if v is None:
            continue
        try:
            x = float(v)
        except Exception:
            continue
        if math.isfinite(x):
            vals.append(x)
    return vals


def fmt(x, nd=4):
    if x is None:
        return "NA"
    if isinstance(x, float) and not math.isfinite(x):
        return "NA"
    return f"{x:.{nd}f}"


def mean_std(vals):
    if not vals:
        return None, None
    return mean(vals), pstdev(vals)


def ci_tiny(vals):
    # Descriptive normal-width interval for seed-to-seed spread, not a theorem.
    if len(vals) <= 1:
        return None
    m, s = mean(vals), pstdev(vals)
    return (m - 1.96 * s / math.sqrt(len(vals)), m + 1.96 * s / math.sqrt(len(vals)))


def arm_rows():
    out = {}
    for arm in ARMS:
        rows = load_jsonl(SUB / "arms" / arm / "train_supervised.jsonl")
        ctr_task = Counter(r.get("task") for r in rows)
        ctr_query = Counter(r.get("query_kind", "comparison") for r in rows)
        ctr_label_by_task = defaultdict(Counter)
        for r in rows:
            ctr_label_by_task[r.get("task")][bool(r.get("label"))] += 1
        out[arm] = {
            "n_rows": len(rows),
            "task_counts": dict(ctr_task),
            "query_kind_counts": dict(ctr_query),
            "label_counts_by_task": {k: dict(v) for k, v in ctr_label_by_task.items()},
            "row_presentations_at_step279_50_epochs": len(rows) * 50,
        }
    return out


def paired_differences(fine):
    rows = fine["all_results"]
    by = {(r["arm"], r["seed"]): r for r in rows}
    diffs = []
    for seed in sorted({r["seed"] for r in rows}):
        a = by.get(("aligned_state_bridge", seed))
        inv = by.get(("inverted_state_bridge", seed))
        if not a or not inv:
            continue
        rec = {"seed": seed}
        for key in ["mixed_acc", "state_changed", "state_unchanged", "state_pair_both", "xtempl_changed", "nperm_changed"]:
            rec[key + "_aligned"] = a.get(key)
            rec[key + "_inverted"] = inv.get(key)
            if a.get(key) is not None and inv.get(key) is not None:
                rec[key + "_aligned_minus_inverted"] = round(float(a[key]) - float(inv[key]), 6)
        diffs.append(rec)
    summary = {}
    for key in ["mixed_acc", "state_changed", "state_unchanged", "state_pair_both", "xtempl_changed", "nperm_changed"]:
        vals = [d[key + "_aligned_minus_inverted"] for d in diffs if key + "_aligned_minus_inverted" in d]
        m, s = mean_std(vals)
        c = ci_tiny(vals)
        summary[key] = {"mean_diff": m, "std_diff": s, "normal_width_interval": c, "values": vals}
    return diffs, summary


def implied_bridge_fit(fine, row_info):
    # If all relation-comparison training rows were fitted correctly, this is the
    # minimum possible accuracy on the state-query bridge rows. It is not a
    # measured split-specific training accuracy.
    out = {}
    agg = fine["arm_aggregates"]
    for arm in ARMS:
        total = row_info[arm]["n_rows"]
        comp = row_info[arm]["task_counts"].get("relation_comparison", 0)
        state = row_info[arm]["task_counts"].get("state_query", 0)
        train = agg.get(arm, {}).get("train_acc_mean")
        if train is None or not state:
            out[arm] = None
            continue
        correct = train * total
        lo = max(0.0, (correct - comp) / state)
        hi = min(1.0, correct / state)
        out[arm] = {
            "train_acc_mean": train,
            "n_total": total,
            "n_relation_comparison": comp,
            "n_state_query": state,
            "state_fit_range_given_only_aggregate_train_acc": [lo, hi],
            "note": "This range is derived from aggregate train accuracy only; it does not prove local inverted semantics were learned.",
        }
    return out


def conservation_readout(fine):
    out = {}
    agg = fine["arm_aggregates"]
    for arm, a in agg.items():
        c = a.get("state_changed_mean")
        u = a.get("state_unchanged_mean")
        both = a.get("state_pair_both_mean")
        if c is None or u is None or both is None:
            continue
        lower = max(0.0, c + u - 1.0)
        indep = c * u
        out[arm] = {
            "changed": c,
            "unchanged": u,
            "both": both,
            "minimum_overlap_from_marginals": lower,
            "independent_overlap_reference": indep,
            "both_minus_minimum": both - lower,
            "both_minus_independent": both - indep,
        }
    return out


def frozen_compare(pre, rand):
    out = {}
    for label, data in [("frozen_pretrained", pre), ("frozen_random", rand)]:
        agg = data["arm_aggregates"]
        out[label] = {
            arm: {
                "train": agg.get(arm, {}).get("train_acc_mean"),
                "hh": agg.get(arm, {}).get("hh_acc_mean"),
                "mixed": agg.get(arm, {}).get("mixed_acc_mean"),
                "state_changed": agg.get(arm, {}).get("state_changed_mean"),
                "state_unchanged": agg.get(arm, {}).get("state_unchanged_mean"),
                "pair_both": agg.get(arm, {}).get("state_pair_both_mean"),
            }
            for arm in ARMS
        }
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fine = load_json(FINE)
    pre = load_json(FROZEN_PRE)
    rand = load_json(FROZEN_RAND)
    rows = arm_rows()
    diffs, diff_summary = paired_differences(fine)
    bridge_fit = implied_bridge_fit(fine, rows)
    cons = conservation_readout(fine)
    froz = frozen_compare(pre, rand)

    result = {
        "source_files": {
            "substrate": str(SUB),
            "finetune": str(FINE),
            "frozen_pretrained": str(FROZEN_PRE),
            "frozen_random": str(FROZEN_RAND),
        },
        "arm_row_structure": rows,
        "aligned_inverted_seed_paired_differences": diffs,
        "aligned_inverted_difference_summary": diff_summary,
        "state_bridge_fit_range_from_aggregate_train": bridge_fit,
        "changed_unchanged_joint_readout": cons,
        "frozen_encoder_readout": froz,
        "central_numbers": {
            "fine_aligned_mixed": fine["arm_aggregates"]["aligned_state_bridge"].get("mixed_acc_mean"),
            "fine_inverted_mixed": fine["arm_aggregates"]["inverted_state_bridge"].get("mixed_acc_mean"),
            "fine_aligned_minus_inverted_mixed": fine["arm_aggregates"]["aligned_state_bridge"].get("mixed_acc_mean") - fine["arm_aggregates"]["inverted_state_bridge"].get("mixed_acc_mean"),
            "fine_aligned_changed_true": fine["arm_aggregates"]["aligned_state_bridge"].get("state_changed_mean"),
            "fine_inverted_changed_true": fine["arm_aggregates"]["inverted_state_bridge"].get("state_changed_mean"),
            "fine_neutral_changed_true": fine["arm_aggregates"]["neutral_decoupled"].get("state_changed_mean"),
        },
    }
    (OUT / "orientation_evidence_readout.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    lines = []
    lines.append("# research corrected readout of research orientation evidence")
    lines.append("")
    lines.append("This file re-reads the research/279 symmetry-identification work from saved outputs only. No model loading, training, official evaluation, or upload occurred.")
    lines.append("")
    lines.append("## Central readout")
    c = result["central_numbers"]
    lines.append(f"- Mixed held-seen orientation stays at chance: aligned {fmt(c['fine_aligned_mixed'])}, inverted {fmt(c['fine_inverted_mixed'])}, aligned-minus-inverted {fmt(c['fine_aligned_minus_inverted_mixed'])}.")
    lines.append(f"- True-labeled changed-state readout improves in state-trained arms: aligned {fmt(c['fine_aligned_changed_true'])}, inverted {fmt(c['fine_inverted_changed_true'])}, neutral {fmt(c['fine_neutral_changed_true'])}.")
    lines.append("- The direct orientation prediction from the formal slot-orbit solver is therefore absent in the fine-tuned DeBERTa readout; the state gain is real but not polarity-controlled on mixed relations.")
    lines.append("")
    lines.append("## Arm structure and exposure")
    lines.append("| arm | rows | relation rows | state rows | row presentations at 50 epochs |")
    lines.append("|---|---:|---:|---:|---:|")
    for arm in ARMS:
        r = rows[arm]
        lines.append(f"| {arm} | {r['n_rows']} | {r['task_counts'].get('relation_comparison',0)} | {r['task_counts'].get('state_query',0)} | {r['row_presentations_at_step279_50_epochs']} |")
    lines.append("")
    lines.append("The neutral arm has more than twice the supervised rows of the aligned/inverted arms, so its magnitude cannot be read as a matched state-format reference without a new matched run.")
    lines.append("")
    lines.append("## What aggregate train accuracy can and cannot show")
    lines.append("| arm | aggregate train acc | state-query rows | possible state-query train accuracy range |")
    lines.append("|---|---:|---:|---:|")
    for arm in ["aligned_state_bridge", "inverted_state_bridge", "neutral_decoupled"]:
        b = bridge_fit[arm]
        lo, hi = b["state_fit_range_given_only_aggregate_train_acc"]
        lines.append(f"| {arm} | {fmt(b['train_acc_mean'])} | {b['n_state_query']} | [{fmt(lo)}, {fmt(hi)}] |")
    lines.append("")
    lines.append("The inverted arm's aggregate train score does not by itself prove that a coherent inverted event-state semantics was learned on the bridge rows; it only constrains the possible bridge-row fit range.")
    lines.append("")
    lines.append("## Aligned-minus-inverted seed-paired differences")
    lines.append("| readout | mean diff | seed values |")
    lines.append("|---|---:|---|")
    for key in ["mixed_acc", "state_changed", "state_unchanged", "state_pair_both", "xtempl_changed", "nperm_changed"]:
        d = diff_summary[key]
        vals = ", ".join(fmt(v,3) for v in d["values"])
        lines.append(f"| {key} | {fmt(d['mean_diff'])} | {vals} |")
    lines.append("")
    lines.append("The changed-state aligned advantage is small relative to seed variation and reverses in one seed; several unchanged and joint readouts favor the inverted arm. This is not a clean bridge-polarity interaction.")
    lines.append("")
    lines.append("## Changed plus unchanged joint structure")
    lines.append("| arm | changed | unchanged | both | both - minimum overlap | both - independent reference |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for arm in ["exposure_only", "heldheld_only", "aligned_state_bridge", "inverted_state_bridge", "neutral_decoupled", "mixed_event_bridge"]:
        x = cons[arm]
        lines.append(f"| {arm} | {fmt(x['changed'])} | {fmt(x['unchanged'])} | {fmt(x['both'])} | {fmt(x['both_minus_minimum'])} | {fmt(x['both_minus_independent'])} |")
    lines.append("")
    lines.append("Aligned and inverted state arms raise both-correct above the no-state baseline, but the observed joint score is close to the overlap forced by marginal changed/unchanged accuracies and below the independence reference. This supports state-task transfer, not a protected conserved state record.")
    lines.append("")
    lines.append("## Corrected interpretation")
    lines.append("The durable result is: in this DeBERTa checkpoint and research surface, state-formatted supervision gives out-of-distribution state-query competence, including new names/templates, while relation-comparison orientation remains at chance and bridge polarity does not install opposite global orientations. The data do not yet identify whether the state gain is driven by pretrained semantic priors, generic state-format calibration, surface-readable grammatical roles, or optimization/credit assignment. Distinguishing those alternatives requires matched neutral/aligned/inverted runs, trainable random-init controls matched on local fit, pretraining-checkpoint comparisons, and contradictory-evidence dose curves with explicit local-vs-new support readouts.")
    lines.append("")
    lines.append("## Files")
    lines.append(f"- JSON readout: `{OUT / 'orientation_evidence_readout.json'}`")
    ((OUT.parents[4] / 'research/documents/representation_and_objectives/data/orientation_evidence_readout/orientation_evidence_readout_summary.md')).write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": "ORIENTATION_EVIDENCE_READOUT_COMPLETE",
        "out": str(OUT),
        "summary": str((OUT.parents[4] / 'research/documents/representation_and_objectives/data/orientation_evidence_readout/orientation_evidence_readout_summary.md')),
        "json": str(OUT / "orientation_evidence_readout.json"),
        "central_numbers": result["central_numbers"],
        "no_model_loading_training_official_eval_upload": True,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
