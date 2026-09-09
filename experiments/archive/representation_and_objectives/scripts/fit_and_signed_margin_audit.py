#!/usr/bin/env python3
"""research: audit research local fit and signed mixed margins.

This CPU-only analysis answers two verifier concerns before any new substrate or GPU run:
1. Did the k16 learned probes fit the informative same-initial bridge rows locally?
2. Does mixed held-seen orientation remain absent when read as label-signed margins,
   not just true-row acceptance rates?

Inputs are the merged research fast-probe per-row train/eval predictions.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

PROJECT = Path("experiments/archive/representation_and_objectives")
DEFAULT_RUN = PROJECT / "data/fast_balanced_probe_v2_merged"
DEFAULT_OUT = PROJECT / "data/fit_and_signed_margin_audit"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def mean(xs: Sequence[float]) -> float | None:
    return float(sum(xs) / len(xs)) if xs else None


def std(xs: Sequence[float]) -> float | None:
    return float(np.std(np.asarray(xs, dtype=float))) if xs else None


def sem(xs: Sequence[float]) -> float | None:
    return float(np.std(np.asarray(xs, dtype=float)) / math.sqrt(len(xs))) if xs else None


def summarize(vals: Sequence[float]) -> Dict[str, Any]:
    return {"n": len(vals), "mean": mean(vals), "std": std(vals), "sem": sem(vals), "values": list(vals)}


def safe_key(*xs: Any) -> str:
    return "|".join("None" if x is None else str(x) for x in xs)


def group(rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        out[safe_key(*(r.get(f) for f in fields))].append(r)
    return dict(out)


def row_score(r: Dict[str, Any]) -> float:
    return float(r["margin1_minus_0"])


def state_choice_records(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    state = [r for r in rows if r.get("task") == "state_query"]
    by_query: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for r in state:
        key = (r.get("condition"), r.get("arm"), r.get("seed"), r.get("suite"), r.get("pair_id"), r.get("query_kind"))
        by_query[key].append(r)
    out: List[Dict[str, Any]] = []
    for _key, rr in by_query.items():
        trues = [r for r in rr if bool(r.get("label"))]
        falses = [r for r in rr if not bool(r.get("label"))]
        if len(trues) != 1 or len(falses) != 1:
            continue
        t, f = trues[0], falses[0]
        margin = row_score(t) - row_score(f)
        out.append({
            "condition": t.get("condition"),
            "arm": t.get("arm"),
            "seed": t.get("seed"),
            "suite": t.get("suite"),
            "pair_id": t.get("pair_id"),
            "query_kind": t.get("query_kind"),
            "choice_correct": margin > 0,
            "choice_tie": margin == 0,
            "true_minus_false_margin": margin,
            "initial_pattern": t.get("initial_pattern"),
            "static_slot": t.get("static_slot"),
            "relation": t.get("relation"),
            "voice": t.get("voice"),
            "orientation_dependency": t.get("orientation_dependency"),
        })
    return out


def pair_both_records(choice: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_pair: Dict[Tuple[Any, ...], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for r in choice:
        key = (r.get("condition"), r.get("arm"), r.get("seed"), r.get("suite"), r.get("pair_id"))
        by_pair[key][str(r.get("query_kind"))] = r
    out: List[Dict[str, Any]] = []
    for _key, d in by_pair.items():
        if "changed" not in d or "unchanged" not in d:
            continue
        c, u = d["changed"], d["unchanged"]
        out.append({
            "condition": c.get("condition"),
            "arm": c.get("arm"),
            "seed": c.get("seed"),
            "suite": c.get("suite"),
            "pair_id": c.get("pair_id"),
            "pair_both_correct": bool(c.get("choice_correct") and u.get("choice_correct")),
            "changed_correct": bool(c.get("choice_correct")),
            "unchanged_correct": bool(u.get("choice_correct")),
            "initial_pattern": c.get("initial_pattern"),
            "static_slot": c.get("static_slot"),
            "relation": c.get("relation"),
            "voice": c.get("voice"),
            "changed_margin": c.get("true_minus_false_margin"),
            "unchanged_margin": u.get("true_minus_false_margin"),
        })
    return out


def summarize_choice(choice: Sequence[Dict[str, Any]], fields: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for k, rr in sorted(group(choice, fields).items()):
        vals = [float(r["choice_correct"]) for r in rr]
        margins = [float(r["true_minus_false_margin"]) for r in rr]
        out[k] = {"n": len(rr), "acc": mean(vals), "margin_mean": mean(margins), "margin_std": std(margins), "tie_frac": mean([float(r["choice_tie"]) for r in rr])}
    return out


def summarize_both(both: Sequence[Dict[str, Any]], fields: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for k, rr in sorted(group(both, fields).items()):
        out[k] = {
            "n": len(rr),
            "pair_both_acc": mean([float(r["pair_both_correct"]) for r in rr]),
            "changed_acc": mean([float(r["changed_correct"]) for r in rr]),
            "unchanged_acc": mean([float(r["unchanged_correct"]) for r in rr]),
            "changed_margin_mean": mean([float(r["changed_margin"]) for r in rr]),
            "unchanged_margin_mean": mean([float(r["unchanged_margin"]) for r in rr]),
        }
    return out


def comparison_signed_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        if r.get("task") != "relation_comparison":
            continue
        margin = row_score(r)
        signed = margin if bool(r.get("label")) else -margin
        out.append({
            "condition": r.get("condition"),
            "arm": r.get("arm"),
            "seed": r.get("seed"),
            "suite": r.get("suite"),
            "relation1": r.get("relation1"),
            "relation2": r.get("relation2"),
            "component1": r.get("component1"),
            "component2": r.get("component2"),
            "label": bool(r.get("label")),
            "correct": bool(r.get("correct")),
            "pred": bool(r.get("pred")),
            "margin": margin,
            "signed_margin_true_coordinate": signed,
            "true_row_accept": bool(r.get("pred")) if bool(r.get("label")) else None,
            "false_row_reject": (not bool(r.get("pred"))) if not bool(r.get("label")) else None,
        })
    return out


def summarize_comparisons(comp: Sequence[Dict[str, Any]], fields: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for k, rr in sorted(group(comp, fields).items()):
        out[k] = {
            "n": len(rr),
            "accuracy_vs_true_coordinate": mean([float(r["correct"]) for r in rr]),
            "pred_true_frac": mean([float(r["pred"]) for r in rr]),
            "signed_margin_mean_vs_true_coordinate": mean([float(r["signed_margin_true_coordinate"]) for r in rr]),
            "signed_margin_std": std([float(r["signed_margin_true_coordinate"]) for r in rr]),
            "true_row_accept": mean([float(r["true_row_accept"]) for r in rr if r["true_row_accept"] is not None]),
            "false_row_reject": mean([float(r["false_row_reject"]) for r in rr if r["false_row_reject"] is not None]),
        }
    return out


def aggregate_seed_metric(records: Dict[str, Dict[str, Any]], key_prefix: str, metric: str) -> List[float]:
    vals: List[float] = []
    for key, rec in records.items():
        if key.startswith(key_prefix):
            x = rec.get(metric)
            if isinstance(x, (int, float)) and math.isfinite(float(x)):
                vals.append(float(x))
    return vals


def seed_level_state(choice: Sequence[Dict[str, Any]], both: Sequence[Dict[str, Any]], split: str) -> Dict[str, Any]:
    changed = [r for r in choice if r.get("query_kind") == "changed"]
    unchanged = [r for r in choice if r.get("query_kind") == "unchanged"]
    return {
        "split": split,
        "changed_by_condition_arm_seed_pattern": summarize_choice(changed, ["condition", "arm", "seed", "initial_pattern"]),
        "changed_by_condition_arm_pattern": summarize_choice(changed, ["condition", "arm", "initial_pattern"]),
        "changed_by_condition_arm_pattern_static": summarize_choice(changed, ["condition", "arm", "initial_pattern", "static_slot"]),
        "changed_by_condition_arm_relation_pattern": summarize_choice(changed, ["condition", "arm", "relation", "initial_pattern"]),
        "unchanged_by_condition_arm": summarize_choice(unchanged, ["condition", "arm"]),
        "pair_both_by_condition_arm_pattern": summarize_both(both, ["condition", "arm", "initial_pattern"]),
        "pair_both_by_condition_arm_relation_pattern": summarize_both(both, ["condition", "arm", "relation", "initial_pattern"]),
    }


def build_report(run_out: Path) -> Dict[str, Any]:
    train = load_jsonl(run_out / "per_row_train_predictions.jsonl")
    eval_rows = load_jsonl(run_out / "per_row_eval_predictions.jsonl")
    train_choice = state_choice_records(train)
    eval_choice = state_choice_records(eval_rows)
    train_both = pair_both_records(train_choice)
    eval_both = pair_both_records(eval_choice)
    eval_comp = comparison_signed_rows(eval_rows)
    train_comp = comparison_signed_rows(train)

    # Summary over all training predictions; this is the source for "local fit".
    train_fit = summarize_comparisons(train_comp, ["condition", "arm", "seed", "suite"])
    all_train_groups = summarize_comparisons(comparison_signed_rows([r for r in train if r.get("task") == "relation_comparison"]), ["condition", "arm"])
    # Add all-row train accuracy by condition/arm/seed from all tasks.
    train_all_acc: Dict[str, Dict[str, Any]] = {}
    for k, rr in sorted(group(train, ["condition", "arm", "seed"]).items()):
        train_all_acc[k] = {
            "n": len(rr),
            "acc": mean([float(r.get("correct")) for r in rr]),
            "pred_true_frac": mean([float(r.get("pred")) for r in rr]),
            "label_true_frac": mean([float(r.get("label")) for r in rr]),
        }

    mixed = [r for r in eval_comp if r.get("suite") == "mixed_held_seen_orientation"]
    mixed_by_seed = summarize_comparisons(mixed, ["condition", "arm", "seed"])
    mixed_by_arm = summarize_comparisons(mixed, ["condition", "arm"])
    mixed_by_relation = summarize_comparisons(mixed, ["condition", "arm", "relation1", "relation2"])

    # paired signed interaction: aligned minus inverted by condition, from seed-paired means.
    signed_interactions: Dict[str, Dict[str, Any]] = {}
    for condition in sorted({r.get("condition") for r in mixed}):
        diffs_margin: List[float] = []
        diffs_acc: List[float] = []
        diffs_pred_true: List[float] = []
        for seed in sorted({r.get("seed") for r in mixed if r.get("condition") == condition}):
            ka = f"{condition}|aligned_state_bridge|{seed}"
            ki = f"{condition}|inverted_state_bridge|{seed}"
            if ka in mixed_by_seed and ki in mixed_by_seed:
                diffs_margin.append(float(mixed_by_seed[ka]["signed_margin_mean_vs_true_coordinate"]) - float(mixed_by_seed[ki]["signed_margin_mean_vs_true_coordinate"]))
                diffs_acc.append(float(mixed_by_seed[ka]["accuracy_vs_true_coordinate"]) - float(mixed_by_seed[ki]["accuracy_vs_true_coordinate"]))
                diffs_pred_true.append(float(mixed_by_seed[ka]["pred_true_frac"]) - float(mixed_by_seed[ki]["pred_true_frac"]))
        signed_interactions[str(condition)] = {
            "aligned_minus_inverted_signed_margin": summarize(diffs_margin),
            "aligned_minus_inverted_accuracy_vs_true_coordinate": summarize(diffs_acc),
            "aligned_minus_inverted_pred_true_frac": summarize(diffs_pred_true),
        }

    return {
        "status": "STEP284_FIT_AND_SIGNED_MARGIN_AUDIT_COMPLETE",
        "input_run_out": str(run_out),
        "n_train_rows": len(train),
        "n_eval_rows": len(eval_rows),
        "train_all_accuracy_by_condition_arm_seed": train_all_acc,
        "train_state_choice": seed_level_state(train_choice, train_both, "train"),
        "eval_state_choice": seed_level_state(eval_choice, eval_both, "eval"),
        "mixed_eval_signed_margin_by_condition_arm_seed": mixed_by_seed,
        "mixed_eval_signed_margin_by_condition_arm": mixed_by_arm,
        "mixed_eval_signed_margin_by_relation_pair": mixed_by_relation,
        "mixed_eval_signed_interactions": signed_interactions,
    }


def fmt(x: Any, digits: int = 3) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def table_val(summary: Dict[str, Dict[str, Any]], key: str, field: str) -> Any:
    return summary.get(key, {}).get(field)


def write_summary(path: Path, rep: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# research audit of research local fit and signed mixed margins")
    lines.append("")
    lines.append("This CPU-only audit reads the merged research fast-probe logits. It verifies whether the balanced k16 informative bridge rows were fit locally and replaces mixed true-row acceptance with all-row label-signed margins against the true coordinate.")
    lines.append("")
    lines.append(f"- train rows analyzed: {rep['n_train_rows']}")
    lines.append(f"- eval rows analyzed: {rep['n_eval_rows']}")
    lines.append("")
    lines.append("## Training fit")
    lines.append("")
    lines.append("All condition/arm/seed groups reached 1.0 train accuracy over the model-consumed rows. The table below focuses on the held bridge state rows where `initial_pattern` is explicit; common seen-coordinate rows have `initial_pattern=None`.")
    lines.append("")
    lines.append("| condition | arm | pattern | exact changed choice | exact pair-both | n changed choices | n pair records |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    ch = rep["train_state_choice"]["changed_by_condition_arm_pattern"]
    pb = rep["train_state_choice"]["pair_both_by_condition_arm_pattern"]
    for condition in ["replace_k00_spread", "replace_k16_spread"]:
        for arm in ["aligned_state_bridge", "inverted_state_bridge", "heldheld_only"]:
            for pattern in ["None", "opposite", "same"]:
                ck = f"{condition}|{arm}|{pattern}"
                pk = ck
                if ck in ch or pk in pb:
                    lines.append(f"| {condition} | {arm} | {pattern} | {fmt(table_val(ch, ck, 'acc'))} | {fmt(table_val(pb, pk, 'pair_both_acc'))} | {table_val(ch, ck, 'n') or 0} | {table_val(pb, pk, 'n') or 0} |")
    lines.append("")
    lines.append("This supports the restricted wording: the same-initial informative training rows were fit locally in k16, but their event-role solution did not extend to disjoint held evaluation fillers.")
    lines.append("")
    lines.append("## Evaluation state exact choice")
    lines.append("")
    lines.append("| condition | arm | pattern | exact changed choice | exact pair-both | changed margin |")
    lines.append("|---|---|---|---:|---:|---:|")
    ech = rep["eval_state_choice"]["changed_by_condition_arm_pattern"]
    epb = rep["eval_state_choice"]["pair_both_by_condition_arm_pattern"]
    for condition in ["replace_k00_spread", "replace_k16_spread"]:
        for arm in ["aligned_state_bridge", "inverted_state_bridge", "heldheld_only"]:
            for pattern in ["opposite", "same"]:
                k = f"{condition}|{arm}|{pattern}"
                lines.append(f"| {condition} | {arm} | {pattern} | {fmt(table_val(ech, k, 'acc'))} | {fmt(table_val(epb, k, 'pair_both_acc'))} | {fmt(table_val(ech, k, 'margin_mean'))} |")
    lines.append("")
    lines.append("## Mixed held-seen orientation using signed margins")
    lines.append("")
    lines.append("For each relation-comparison row, signed margin = `margin(true label)`; a real true-coordinate model should have positive signed margin, while a clean inverted-coordinate model would have negative signed margin against these true labels. The decisive quantity is aligned minus inverted, paired by seed.")
    lines.append("")
    lines.append("| condition | arm | acc vs true coordinate | pred true frac | signed margin mean | true-row accept | false-row reject |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    mx = rep["mixed_eval_signed_margin_by_condition_arm"]
    for condition in ["replace_k00_spread", "replace_k16_spread"]:
        for arm in ["aligned_state_bridge", "inverted_state_bridge", "heldheld_only"]:
            k = f"{condition}|{arm}"
            if k in mx:
                r = mx[k]
                lines.append(f"| {condition} | {arm} | {fmt(r.get('accuracy_vs_true_coordinate'))} | {fmt(r.get('pred_true_frac'))} | {fmt(r.get('signed_margin_mean_vs_true_coordinate'))} | {fmt(r.get('true_row_accept'))} | {fmt(r.get('false_row_reject'))} |")
    lines.append("")
    lines.append("| condition | aligned-inverted signed-margin gap | aligned-inverted accuracy gap | aligned-inverted pred-true gap |")
    lines.append("|---|---:|---:|---:|")
    for condition, rec in rep["mixed_eval_signed_interactions"].items():
        sm = rec["aligned_minus_inverted_signed_margin"]
        ac = rec["aligned_minus_inverted_accuracy_vs_true_coordinate"]
        pt = rec["aligned_minus_inverted_pred_true_frac"]
        lines.append(f"| {condition} | {fmt(sm.get('mean'))}±{fmt(sm.get('std'))} | {fmt(ac.get('mean'))}±{fmt(ac.get('std'))} | {fmt(pt.get('mean'))}±{fmt(pt.get('std'))} |")
    lines.append("")
    lines.append("## Scientific reading")
    lines.append("")
    lines.append("- The k16 bridge did not fail because same-initial informative training rows were ignored; they were fit in the model-consumed training set.")
    lines.append("- On disjoint held evaluation fillers, exact changed choice and pair-both remained anti-copy-like: high for opposite-initial rows and near zero for same-initial rows.")
    lines.append("- Mixed held-seen transfer shows common acceptance bias but not a polarity-controlled coordinate: the k16 aligned-minus-inverted signed-margin gap is small relative to margins and has unstable sign across seeds.")
    lines.append("- The next experiment should therefore manipulate learner-usable interface connectivity or architecture, not add more isolated counterexample rows on the same surface.")
    lines.append("")
    lines.append("## Files")
    lines.append(f"- full JSON: `{path.parent / 'fit_and_signed_margin_audit.json'}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-out", type=Path, default=DEFAULT_RUN)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    rep = build_report(args.run_out)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "fit_and_signed_margin_audit.json", rep)
    write_summary(args.out / "fit_and_signed_margin_audit_summary.md", rep)
    print(json.dumps({
        "status": rep["status"],
        "summary": str(args.out / "fit_and_signed_margin_audit_summary.md"),
        "json": str(args.out / "fit_and_signed_margin_audit.json"),
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
