#!/usr/bin/env python3
"""research: analyze learned balanced k0/k16 probe outputs.

Reads the per-row outputs produced by `balanced_budget_probe.py` and
creates cell-resolved summaries for deciding whether independently informative
state worlds induce task-local event-state learning or reusable signed relation
orientation.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

PROJECT = Path("experiments/archive/representation_and_objectives")
DEFAULT_RUN_OUT = PROJECT / "training/runs/balanced_k00_k16_probe/probe_outputs"
DEFAULT_DATA_ROOT = PROJECT / "data/information_budget_substrate"
DEFAULT_ANALYSIS_OUT = PROJECT / "data/balanced_probe_analysis"
DEFAULT_CONDITIONS = ["replace_k00_spread", "replace_k16_spread"]
DEFAULT_ARMS = ["aligned_state_bridge", "inverted_state_bridge", "heldheld_only"]
STATE_SUITES = ["paired_state_conservation", "cross_template_state_readout", "name_permutation_counterfactual"]
MIXED_SUITE = "mixed_held_seen_orientation"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path) as f:
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
    if not xs:
        return None
    return float(np.std(np.asarray(xs, dtype=float)))


def sem(xs: Sequence[float]) -> float | None:
    if not xs:
        return None
    return float(np.std(np.asarray(xs, dtype=float)) / math.sqrt(len(xs)))


def safe_key(*xs: Any) -> str:
    return "|".join("None" if x is None else str(x) for x in xs)


def group_rows(rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> Dict[str, List[Dict[str, Any]]]:
    g: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        g[safe_key(*(r.get(f) for f in fields))].append(r)
    return dict(g)


def summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"n": 0}
    out: Dict[str, Any] = {"n": len(rows), "acc": mean([float(r["correct"]) for r in rows])}
    if all("pred" in r for r in rows):
        out["pred_true_frac"] = mean([float(r["pred"]) for r in rows])
    if all("label" in r for r in rows):
        out["label_true_frac"] = mean([float(r["label"]) for r in rows])
    if any("margin1_minus_0" in r for r in rows):
        out["margin_mean"] = mean([float(r.get("margin1_minus_0", 0.0)) for r in rows])
    return out


def relation_anchor_map(data_root: Path, condition: str, arm: str) -> Dict[str, Any]:
    path = data_root / condition / "arms" / arm / "train_supervised.jsonl"
    if not path.exists():
        return {"state_anchor_relations": [], "state_anchor_cells": {}}
    rows = load_jsonl(path)
    changed_true = [r for r in rows if r.get("task") == "state_query" and r.get("query_kind") == "changed" and bool(r.get("label"))]
    rels = sorted({str(r.get("relation")) for r in changed_true if r.get("relation") is not None})
    cells = Counter(safe_key(r.get("relation"), r.get("voice"), r.get("static_slot"), r.get("initial_pattern")) for r in changed_true)
    return {"state_anchor_relations": rels, "state_anchor_cells": dict(sorted(cells.items()))}


def tag_direct_anchor(rows: Sequence[Dict[str, Any]], anchors: Sequence[str]) -> List[Dict[str, Any]]:
    anchor_set = set(anchors)
    out = []
    for r in rows:
        rr = dict(r)
        rel = rr.get("relation")
        rr["state_relation_anchor_status"] = "direct_state_anchor" if rel in anchor_set else ("unanchored_or_graph_transfer" if rel is not None else "none")
        out.append(rr)
    return out


def state_metrics(rows: Sequence[Dict[str, Any]], anchors: Sequence[str]) -> Dict[str, Any]:
    tagged = tag_direct_anchor(rows, anchors)
    state = [r for r in tagged if r.get("task") == "state_query"]
    true_state = [r for r in state if bool(r.get("label"))]
    changed = [r for r in true_state if r.get("query_kind") == "changed"]
    unchanged = [r for r in true_state if r.get("query_kind") == "unchanged"]
    out: Dict[str, Any] = {
        "state_all_rows": summarize(state),
        "true_state_choices": summarize(true_state),
        "true_changed": summarize(changed),
        "true_unchanged": summarize(unchanged),
        "changed_by_pattern": {k: summarize(v) for k, v in sorted(group_rows(changed, ["initial_pattern"]).items())},
        "changed_by_pattern_static": {k: summarize(v) for k, v in sorted(group_rows(changed, ["initial_pattern", "static_slot"]).items())},
        "changed_by_relation": {k: summarize(v) for k, v in sorted(group_rows(changed, ["relation"]).items())},
        "changed_by_anchor_status": {k: summarize(v) for k, v in sorted(group_rows(changed, ["state_relation_anchor_status"]).items())},
        "changed_by_pattern_static_relation_voice": {k: summarize(v) for k, v in sorted(group_rows(changed, ["initial_pattern", "static_slot", "relation", "voice"]).items())},
        "unchanged_by_pattern_static": {k: summarize(v) for k, v in sorted(group_rows(unchanged, ["initial_pattern", "static_slot"]).items())},
        "pair_both": pair_both(true_state),
    }
    return out


def pair_both(true_state_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_pair: Dict[Tuple[Any, Any, Any, Any], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for r in true_state_rows:
        # include seed/arm/condition/suite to avoid merging across runs
        key = (r.get("condition"), r.get("arm"), r.get("seed"), r.get("suite"), r.get("pair_id"))
        by_pair[key][str(r.get("query_kind"))] = r
    pairs: List[Dict[str, Any]] = []
    for key, d in by_pair.items():
        if "changed" not in d or "unchanged" not in d:
            continue
        c, u = d["changed"], d["unchanged"]
        pairs.append({
            "condition": c.get("condition"),
            "arm": c.get("arm"),
            "seed": c.get("seed"),
            "suite": c.get("suite"),
            "pair_id": c.get("pair_id"),
            "correct": bool(c.get("correct") and u.get("correct")),
            "changed_correct": bool(c.get("correct")),
            "unchanged_correct": bool(u.get("correct")),
            "initial_pattern": c.get("initial_pattern"),
            "static_slot": c.get("static_slot"),
            "relation": c.get("relation"),
            "voice": c.get("voice"),
            "state_relation_anchor_status": c.get("state_relation_anchor_status"),
        })
    if not pairs:
        return {"all": {"n": 0}}
    return {
        "all": summarize(pairs),
        "by_pattern": {k: summarize(v) for k, v in sorted(group_rows(pairs, ["initial_pattern"]).items())},
        "by_pattern_static": {k: summarize(v) for k, v in sorted(group_rows(pairs, ["initial_pattern", "static_slot"]).items())},
        "by_relation": {k: summarize(v) for k, v in sorted(group_rows(pairs, ["relation"]).items())},
        "by_anchor_status": {k: summarize(v) for k, v in sorted(group_rows(pairs, ["state_relation_anchor_status"]).items())},
        "by_pattern_static_relation_voice": {k: summarize(v) for k, v in sorted(group_rows(pairs, ["initial_pattern", "static_slot", "relation", "voice"]).items())},
    }


def mixed_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    comp = [r for r in rows if r.get("task") == "relation_comparison"]
    true_comp = [r for r in comp if bool(r.get("label"))]
    false_comp = [r for r in comp if not bool(r.get("label"))]
    return {
        "all_comparisons": summarize(comp),
        "true_statements": summarize(true_comp),
        "false_statements": summarize(false_comp),
        "true_by_component_pair": {k: summarize(v) for k, v in sorted(group_rows(true_comp, ["component1", "component2"]).items())},
        "true_by_relation_pair": {k: summarize(v) for k, v in sorted(group_rows(true_comp, ["relation1", "relation2"]).items())},
    }


def owner_slot(owner: Any, r: Dict[str, Any]) -> int | None:
    order = r.get("arg_order") or []
    for i, x in enumerate(order):
        if x == owner:
            return i
    return None


def rule_pred(rule: str, r: Dict[str, Any]) -> bool | None:
    if r.get("task") != "state_query":
        return None
    if r.get("query_kind") == "unchanged":
        return r.get("candidate") == r.get("static_owner")
    if r.get("query_kind") != "changed":
        return None
    init_slot = owner_slot(r.get("initial_changed_owner"), r)
    if init_slot is None:
        return None
    if rule == "anti_copy":
        target = 1 - init_slot
    elif rule == "copy_initial":
        target = init_slot
    elif rule == "static_slot_step282_switch":
        st = r.get("static_slot")
        if st is None:
            return None
        target = 1 - init_slot if int(st) == 0 else init_slot
    elif rule == "static_slot_opposite_switch":
        st = r.get("static_slot")
        if st is None:
            return None
        target = init_slot if int(st) == 0 else 1 - init_slot
    elif rule == "oracle_event_role_state":
        return r.get("candidate") == r.get("supervised_changed_owner")
    else:
        return None
    return r.get("candidate_slot") == target


def heuristic_fingerprint(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    state = [r for r in rows if r.get("task") == "state_query"]
    true_state = [r for r in state if bool(r.get("label"))]
    changed = [r for r in true_state if r.get("query_kind") == "changed"]
    out: Dict[str, Any] = {}
    for rule in ["anti_copy", "copy_initial", "static_slot_step282_switch", "static_slot_opposite_switch", "oracle_event_role_state"]:
        agree: List[float] = []
        correct_when_rule_correct: List[float] = []
        correct_when_rule_wrong: List[float] = []
        for r in changed:
            pred = rule_pred(rule, r)
            if pred is None:
                continue
            agree.append(float(bool(r.get("pred")) == bool(pred)))
            rule_correct = bool(pred) == bool(r.get("label"))
            if rule_correct:
                correct_when_rule_correct.append(float(r.get("correct")))
            else:
                correct_when_rule_wrong.append(float(r.get("correct")))
        out[rule] = {
            "n": len(agree),
            "agreement_with_model_pred": mean(agree),
            "model_acc_on_rule_correct_rows": mean(correct_when_rule_correct),
            "model_acc_on_rule_wrong_rows": mean(correct_when_rule_wrong),
            "n_rule_correct_rows": len(correct_when_rule_correct),
            "n_rule_wrong_rows": len(correct_when_rule_wrong),
        }
    return out


def aggregate_by_seed(rows: Sequence[Dict[str, Any]], data_root: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {"groups": {}, "condition_arm_seed_rows": {}}
    grouped = group_rows(rows, ["condition", "arm", "seed"])
    for key, gr in sorted(grouped.items()):
        condition, arm, seed_s = key.split("|")
        anchors = relation_anchor_map(data_root, condition, arm)["state_anchor_relations"]
        suites = group_rows(gr, ["suite"])
        rec = {
            "condition": condition,
            "arm": arm,
            "seed": int(seed_s),
            "state_anchor_relations": anchors,
            "n_eval_rows": len(gr),
            "suites": {},
            "heuristic_fingerprint_paired_state": {},
        }
        for suite, sr in sorted(suites.items()):
            if suite in STATE_SUITES:
                rec["suites"][suite] = state_metrics(sr, anchors)
                if suite == "paired_state_conservation":
                    rec["heuristic_fingerprint_paired_state"] = heuristic_fingerprint(sr)
            elif suite == MIXED_SUITE or any(r.get("task") == "relation_comparison" for r in sr):
                rec["suites"][suite] = mixed_metrics(sr)
            else:
                rec["suites"][suite] = {"overall": summarize(sr)}
        out["condition_arm_seed_rows"][key] = rec
    return out


def metric_get(d: Dict[str, Any], path: Sequence[str]) -> float | None:
    x: Any = d
    for p in path:
        if not isinstance(x, dict) or p not in x:
            return None
        x = x[p]
    return float(x) if isinstance(x, (int, float)) and math.isfinite(float(x)) else None


def collect_metric(seed_recs: Dict[str, Dict[str, Any]], condition: str, arm: str, path: Sequence[str]) -> List[float]:
    vals: List[float] = []
    for key, rec in seed_recs.items():
        if rec.get("condition") == condition and rec.get("arm") == arm:
            x = metric_get(rec, path)
            if x is not None:
                vals.append(x)
    return vals


def format_vals(vals: Sequence[float]) -> str:
    if not vals:
        return "n/a"
    m = mean(vals)
    s = std(vals)
    assert m is not None and s is not None
    if len(vals) == 1 or s < 0.0005:
        return f"{m:.3f}"
    return f"{m:.3f}±{s:.3f}"


def metric_stats(vals: Sequence[float]) -> Dict[str, Any]:
    return {"n": len(vals), "mean": mean(vals), "std": std(vals), "sem": sem(vals), "values": list(vals)}


def build_comparison(seed_recs: Dict[str, Dict[str, Any]], conditions: Sequence[str], arms: Sequence[str]) -> Dict[str, Any]:
    paths = {
        "psc_changed_same": ["suites", "paired_state_conservation", "true_changed", "acc"],
        "psc_changed_opposite": ["suites", "paired_state_conservation", "changed_by_pattern", "opposite", "acc"],
        "psc_changed_same_pattern": ["suites", "paired_state_conservation", "changed_by_pattern", "same", "acc"],
        "psc_pair_both_all": ["suites", "paired_state_conservation", "pair_both", "all", "acc"],
        "psc_pair_both_same": ["suites", "paired_state_conservation", "pair_both", "by_pattern", "same", "acc"],
        "psc_pair_both_opposite": ["suites", "paired_state_conservation", "pair_both", "by_pattern", "opposite", "acc"],
        "psc_offdiag_same_st0": ["suites", "paired_state_conservation", "changed_by_pattern_static", "same|0", "acc"],
        "psc_offdiag_opposite_st1": ["suites", "paired_state_conservation", "changed_by_pattern_static", "opposite|1", "acc"],
        "psc_direct_anchor_changed": ["suites", "paired_state_conservation", "changed_by_anchor_status", "direct_state_anchor", "acc"],
        "psc_graph_transfer_changed": ["suites", "paired_state_conservation", "changed_by_anchor_status", "unanchored_or_graph_transfer", "acc"],
        "mixed_true": ["suites", "mixed_held_seen_orientation", "true_statements", "acc"],
        "mixed_all": ["suites", "mixed_held_seen_orientation", "all_comparisons", "acc"],
        "name_perm_changed_same": ["suites", "name_permutation_counterfactual", "changed_by_pattern", "same", "acc"],
        "cross_template_changed_same": ["suites", "cross_template_state_readout", "changed_by_pattern", "same", "acc"],
    }
    out: Dict[str, Any] = {"metrics": {}, "signed_orientation": {}, "k16_minus_k0": {}}
    for condition in conditions:
        out["metrics"][condition] = {}
        for arm in arms:
            out["metrics"][condition][arm] = {name: metric_stats(collect_metric(seed_recs, condition, arm, path)) for name, path in paths.items()}
        if "aligned_state_bridge" in arms and "inverted_state_bridge" in arms:
            a = collect_metric(seed_recs, condition, "aligned_state_bridge", paths["mixed_true"])
            b = collect_metric(seed_recs, condition, "inverted_state_bridge", paths["mixed_true"])
            if len(a) == len(b) and a:
                out["signed_orientation"][condition] = metric_stats([x - y for x, y in zip(a, b)])
    if "replace_k00_spread" in conditions and "replace_k16_spread" in conditions:
        for arm in arms:
            out["k16_minus_k0"][arm] = {}
            for name, path in paths.items():
                a = collect_metric(seed_recs, "replace_k16_spread", arm, path)
                b = collect_metric(seed_recs, "replace_k00_spread", arm, path)
                if len(a) == len(b) and a:
                    out["k16_minus_k0"][arm][name] = metric_stats([x - y for x, y in zip(a, b)])
    return out


def write_summary(path: Path, seed_recs: Dict[str, Dict[str, Any]], comparison: Dict[str, Any], all_results: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    errors = [r for r in all_results if isinstance(r, dict) and "error" in r]
    lines: List[str] = []
    lines.append("# research learned balanced k0/k16 analysis")
    lines.append("")
    lines.append("This analysis is generated from per-row learned predictions.  It separates redundant versus independently informative state evidence, direct state-anchor relations versus graph-transfer relations, off-diagonal pattern×static cells, joint changed/unchanged conservation, and signed mixed-relation transfer.")
    lines.append("")
    lines.append("## Run completeness")
    lines.append("")
    lines.append(f"- all_results entries: {len(all_results)}")
    lines.append(f"- error entries: {len(errors)}")
    if errors:
        lines.append(f"- errors: `{errors}`")
    lines.append(f"- per-seed groups analyzed: {len(seed_recs)}")
    lines.append("")
    lines.append("## Central per-condition metrics")
    lines.append("")
    lines.append("| condition | arm | changed same | changed opposite | pair-both same | pair-both opposite | same/st0 | opposite/st1 | direct-anchor changed | graph-transfer changed | mixed true |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for condition in args.conditions:
        for arm in args.arms:
            md = comparison["metrics"].get(condition, {}).get(arm, {})
            def f(name: str) -> str:
                return format_vals(md.get(name, {}).get("values", []))
            lines.append(
                f"| {condition} | {arm} | {f('psc_changed_same_pattern')} | {f('psc_changed_opposite')} | "
                f"{f('psc_pair_both_same')} | {f('psc_pair_both_opposite')} | {f('psc_offdiag_same_st0')} | {f('psc_offdiag_opposite_st1')} | "
                f"{f('psc_direct_anchor_changed')} | {f('psc_graph_transfer_changed')} | {f('mixed_true')} |"
            )
    lines.append("")
    lines.append("## k16 minus k0 within each arm")
    lines.append("")
    lines.append("| arm | Δ changed same | Δ changed opposite | Δ pair-both same | Δ graph-transfer changed | Δ mixed true |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for arm in args.arms:
        d = comparison.get("k16_minus_k0", {}).get(arm, {})
        def f(name: str) -> str:
            return format_vals(d.get(name, {}).get("values", []))
        lines.append(f"| {arm} | {f('psc_changed_same_pattern')} | {f('psc_changed_opposite')} | {f('psc_pair_both_same')} | {f('psc_graph_transfer_changed')} | {f('mixed_true')} |")
    lines.append("")
    lines.append("## Signed mixed-relation orientation")
    lines.append("")
    lines.append("Positive means aligned bridge assigns higher probability/accuracy to true mixed statements than inverted bridge.  The slot-orbit symbolic control predicts a large positive gap; chance or tiny gaps mean no reusable signed coordinate reached the comparison surface.")
    lines.append("")
    lines.append("| condition | aligned mixed true − inverted mixed true |")
    lines.append("|---|---:|")
    for condition in args.conditions:
        vals = comparison.get("signed_orientation", {}).get(condition, {}).get("values", [])
        lines.append(f"| {condition} | {format_vals(vals)} |")
    lines.append("")
    lines.append("## Reading the result")
    lines.append("")
    lines.append("- If `replace_k16_spread` raises same-initial changed and pair-both cells relative to k0, including `same|0` and `opposite|1`, while unchanged rows remain strong, then independently varied state evidence induced an event-state rule on this surface.")
    lines.append("- If that state improvement is concentrated only in direct-anchor relations (`h0_dax`, `h2_norp`) and graph-transfer relations stay low, the result is relation-local rather than component-wide.")
    lines.append("- If aligned−inverted mixed true-statement separation remains near zero, state learning did not produce a reusable signed relation coordinate, even if state rows improve.")
    lines.append("- Only the combination of joint state success and signed mixed transfer supports moving from the binary experiment to a counterallocated information-efficiency study over k and relation coverage.")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append(f"- input run outputs: `{args.run_out}`")
    lines.append(f"- full analysis JSON: `{args.analysis_out / 'balanced_probe_analysis.json'}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-out", type=Path, default=DEFAULT_RUN_OUT)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--analysis-out", type=Path, default=DEFAULT_ANALYSIS_OUT)
    ap.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    ap.add_argument("--arms", nargs="+", default=DEFAULT_ARMS)
    ap.add_argument("--check-only", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    all_results_path = args.run_out / "all_results.json"
    per_row_path = args.run_out / "per_row_eval_predictions.jsonl"
    needed = [all_results_path, per_row_path]
    status = {"run_out": str(args.run_out), "needed": {str(p): p.exists() for p in needed}}
    if args.check_only:
        print(json.dumps(status, indent=2, sort_keys=True), flush=True)
        return
    missing = [p for p in needed if not p.exists()]
    if missing:
        raise SystemExit(f"Missing learned probe outputs: {[str(p) for p in missing]}")
    all_results = load_json(all_results_path)
    pred_rows = load_jsonl(per_row_path)
    seed_analysis = aggregate_by_seed(pred_rows, args.data_root)
    comparison = build_comparison(seed_analysis["condition_arm_seed_rows"], args.conditions, args.arms)
    report = {
        "run_out": str(args.run_out),
        "data_root": str(args.data_root),
        "n_per_row_eval_predictions": len(pred_rows),
        "all_results_error_entries": [r for r in all_results if isinstance(r, dict) and "error" in r],
        "seed_analysis": seed_analysis,
        "comparison": comparison,
    }
    args.analysis_out.mkdir(parents=True, exist_ok=True)
    write_json(args.analysis_out / "balanced_probe_analysis.json", report)
    write_summary(args.analysis_out / "balanced_probe_analysis_summary.md", seed_analysis["condition_arm_seed_rows"], comparison, all_results, args)
    print(json.dumps({
        "status": "BALANCED_PROBE_ANALYSIS_COMPLETE",
        "summary": str(args.analysis_out / "balanced_probe_analysis_summary.md"),
        "report": str(args.analysis_out / "balanced_probe_analysis.json"),
        "n_per_row_eval_predictions": len(pred_rows),
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
