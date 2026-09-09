#!/usr/bin/env python3
"""research merge/decompose shared-coordinate probe outputs.

Reads one or more research shared-coordinate output roots and summarizes train fit,
state exact choice, pair-both conservation, and relation-comparison signed margins.
No model loading or training.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

PROJECT = Path("experiments/archive/representation_and_objectives")
DEFAULT_INPUTS = [
    PROJECT / "data/shared_coordinate_aligned_seed28800",
    PROJECT / "data/shared_coordinate_inverted_seed28800",
]
DEFAULT_OUT = PROJECT / "data/shared_coordinate_merged_analysis"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def mean(xs: Sequence[float]) -> float | None:
    return float(sum(xs) / len(xs)) if xs else None


def std(xs: Sequence[float]) -> float | None:
    return float(np.std(np.asarray(xs, dtype=float))) if xs else None


def sem(xs: Sequence[float]) -> float | None:
    return float(np.std(np.asarray(xs, dtype=float)) / math.sqrt(len(xs))) if xs else None


def summ(xs: Sequence[float]) -> Dict[str, Any]:
    return {"n": len(xs), "mean": mean(xs), "std": std(xs), "sem": sem(xs), "values": [float(x) for x in xs]}


def discover_run_dirs(input_dirs: Sequence[Path]) -> List[Path]:
    dirs: List[Path] = []
    for root in input_dirs:
        if (root / "result.json").exists():
            dirs.append(root)
        if root.exists():
            for child in sorted(root.iterdir()):
                if child.is_dir() and (child / "result.json").exists():
                    dirs.append(child)
    # Deduplicate preserving order.
    seen = set(); out = []
    for d in dirs:
        s = str(d)
        if s not in seen:
            out.append(d); seen.add(s)
    return out


def state_choice_rows(rows: Sequence[Dict[str, Any]], target_mode: str) -> List[Dict[str, Any]]:
    by: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[(r["condition"], r["arm"], r["seed"], r.get("prediction_split"), r["suite"], r["query_key"])].append(r)
    out: List[Dict[str, Any]] = []
    label_key = "label_true" if target_mode == "true" else "label_arm_target"
    for key, xs in by.items():
        if len(xs) != 2:
            continue
        xs = sorted(xs, key=lambda r: int(r["candidate_index"]))
        pred_i = 0 if float(xs[0]["score"]) >= float(xs[1]["score"]) else 1
        target_i = None
        for i, r in enumerate(xs):
            if bool(r.get(label_key)):
                target_i = i
        if target_i is None:
            continue
        r0 = xs[0]
        out.append({
            "condition": r0["condition"], "arm": r0["arm"], "seed": int(r0["seed"]),
            "prediction_split": r0.get("prediction_split"), "suite": r0["suite"], "query_key": r0["query_key"],
            "target_mode": target_mode, "correct": float(pred_i == target_i),
            "margin": float(xs[target_i]["score"]) - float(xs[1 - target_i]["score"]),
            "is_changed": bool(r0.get("is_changed")), "query_kind": r0.get("query_kind"),
            "relation": r0.get("relation"), "relation_family": r0.get("relation_family"),
            "initial_pattern": r0.get("initial_pattern"), "static_slot": r0.get("static_slot"),
        })
    return out


def pair_both_rows(choice: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by: Dict[Tuple[Any, ...], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for r in choice:
        if r["suite"] not in {"paired_state_conservation", "cross_template_state_readout"}:
            continue
        parts = str(r["query_key"]).split("|")
        base = "|".join(parts[:-1]) if len(parts) >= 3 else re.sub(r"_(changed|unchanged)$", "", str(r["query_key"]))
        qk = "changed" if r["is_changed"] else "unchanged"
        by[(r["condition"], r["arm"], r["seed"], r["prediction_split"], r["suite"], r["target_mode"], base)][qk] = r
    out = []
    for key, d in by.items():
        if "changed" not in d or "unchanged" not in d:
            continue
        ch, un = d["changed"], d["unchanged"]
        out.append({
            "condition": key[0], "arm": key[1], "seed": key[2], "prediction_split": key[3],
            "suite": key[4], "target_mode": key[5], "base_key": key[6],
            "both_correct": float(ch["correct"] and un["correct"]),
            "changed_correct": float(ch["correct"]), "unchanged_correct": float(un["correct"]),
            "changed_margin": ch["margin"], "unchanged_margin": un["margin"],
            "relation": ch.get("relation"), "relation_family": ch.get("relation_family"),
            "initial_pattern": ch.get("initial_pattern"), "static_slot": ch.get("static_slot"),
        })
    return out


def comp_target_rows(rows: Sequence[Dict[str, Any]], target_mode: str) -> List[Dict[str, Any]]:
    return [r for r in rows if r.get("target_mode") == target_mode]


def group_summary(rows: Sequence[Dict[str, Any]], fields: Sequence[str], value: str) -> Dict[str, Any]:
    g: Dict[Tuple[Any, ...], List[float]] = defaultdict(list)
    for r in rows:
        if r.get(value) is None:
            continue
        g[tuple(r.get(f) for f in fields)].append(float(r[value]))
    return {json.dumps(dict(zip(fields, k)), sort_keys=True): summ(v) for k, v in sorted(g.items(), key=lambda kv: str(kv[0]))}


def central_for_run(result: Dict[str, Any], state_rows: Sequence[Dict[str, Any]], comp_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    cond = result["condition"]; arm = result["arm"]; seed = int(result["seed"])
    central: Dict[str, Any] = {"condition": cond, "arm": arm, "seed": seed, "vocab_scope": result.get("vocab_scope", "unknown")}
    for split in ["train", "eval"]:
        sr = [r for r in state_rows if r.get("prediction_split") == split]
        cr = [r for r in comp_rows if r.get("prediction_split") == split]
        for mode in ["true", "arm"]:
            choices = state_choice_rows(sr, mode)
            boths = pair_both_rows(choices)
            def filt(xs: Sequence[Dict[str, Any]], **kw: Any) -> List[Dict[str, Any]]:
                out = list(xs)
                for k, v in kw.items():
                    out = [x for x in out if x.get(k) == v]
                return out
            def acc(xs: Sequence[Dict[str, Any]], key: str = "correct") -> float | None:
                return mean([float(x[key]) for x in xs if x.get(key) is not None])
            prefix = f"{split}_{mode}"
            central[f"{prefix}_state_all"] = acc(choices)
            central[f"{prefix}_psc_changed_same_direct"] = acc(filt(choices, suite="paired_state_conservation", is_changed=True, relation_family="direct_anchor", initial_pattern="same"))
            central[f"{prefix}_psc_changed_same_graph"] = acc(filt(choices, suite="paired_state_conservation", is_changed=True, relation_family="graph_transfer", initial_pattern="same"))
            central[f"{prefix}_psc_changed_opp_graph"] = acc(filt(choices, suite="paired_state_conservation", is_changed=True, relation_family="graph_transfer", initial_pattern="opposite"))
            central[f"{prefix}_psc_unchanged"] = acc(filt(choices, suite="paired_state_conservation", is_changed=False))
            central[f"{prefix}_psc_pair_both_same_graph"] = acc(filt(boths, suite="paired_state_conservation", relation_family="graph_transfer", initial_pattern="same"), "both_correct")
            central[f"{prefix}_psc_pair_both_opp_graph"] = acc(filt(boths, suite="paired_state_conservation", relation_family="graph_transfer", initial_pattern="opposite"), "both_correct")
            central[f"{prefix}_xt_changed_same_graph"] = acc(filt(choices, suite="cross_template_state_readout", is_changed=True, relation_family="graph_transfer", initial_pattern="same"))
            central[f"{prefix}_xt_pair_both_same_graph"] = acc(filt(boths, suite="cross_template_state_readout", relation_family="graph_transfer", initial_pattern="same"), "both_correct")
            ctarget = comp_target_rows(cr, mode)
            for suite in ["mixed_held_seen_orientation", "heldheld_unseen_edge_closure"]:
                xs = filt(ctarget, suite=suite)
                central[f"{prefix}_{suite}_acc"] = acc(xs)
                central[f"{prefix}_{suite}_signed_margin"] = mean([float(x["signed_margin"]) for x in xs])
                central[f"{prefix}_{suite}_pred_true_frac"] = mean([float(bool(x["pred"])) for x in xs])
    central["final_train_state_acc"] = result.get("final_train_metrics", {}).get("train_state_acc")
    central["final_train_changed_acc"] = result.get("final_train_metrics", {}).get("train_changed_acc")
    central["final_train_unchanged_acc"] = result.get("final_train_metrics", {}).get("train_unchanged_acc")
    central["final_train_cmp_acc"] = result.get("final_train_metrics", {}).get("train_cmp_acc")
    central["epochs"] = result.get("epochs")
    central["elapsed_seconds"] = result.get("train_info", {}).get("elapsed_seconds")
    return central


def analyze(inputs: Sequence[Path]) -> Dict[str, Any]:
    run_dirs = discover_run_dirs(inputs)
    results = []
    central = []
    full_group: Dict[str, Any] = {"state_choice": {}, "pair_both": {}, "comparison": {}}
    for rd in run_dirs:
        result = load_json(rd / "result.json")
        state_rows = load_jsonl(rd / "train_state_predictions.jsonl") + load_jsonl(rd / "eval_state_predictions.jsonl")
        comp_rows = load_jsonl(rd / "train_comparison_predictions.jsonl") + load_jsonl(rd / "eval_comparison_predictions.jsonl")
        results.append({"run_dir": str(rd), **result})
        central.append(central_for_run(result, state_rows, comp_rows))
        for mode in ["true", "arm"]:
            choices = state_choice_rows(state_rows, mode)
            boths = pair_both_rows(choices)
            cr = comp_target_rows(comp_rows, mode)
            full_group["state_choice"].update(group_summary(choices, ["condition", "arm", "target_mode", "prediction_split", "suite", "relation", "initial_pattern", "is_changed"], "correct"))
            full_group["pair_both"].update(group_summary(boths, ["condition", "arm", "target_mode", "prediction_split", "suite", "relation_family", "initial_pattern"], "both_correct"))
            full_group["comparison"].update(group_summary(cr, ["condition", "arm", "target_mode", "prediction_split", "suite"], "correct"))
    # Aggregate central metrics by condition/arm and paired differences.
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in central:
        groups[(r["condition"], r["arm"], str(r.get("vocab_scope", "unknown")))].append(r)
    central_group: Dict[str, Any] = {}
    metric_names = sorted(k for k in central[0].keys() if k not in {"condition", "arm", "seed", "vocab_scope"}) if central else []
    for k, rows in groups.items():
        g = {"n": len(rows)}
        for m in metric_names:
            vals = [float(r[m]) for r in rows if r.get(m) is not None and not (isinstance(r.get(m), float) and math.isnan(r.get(m)))]
            g[m] = summ(vals)
        central_group[f"{k[0]}|{k[1]}|{k[2]}"] = g
    pair_by_arm_seed: Dict[Tuple[str, int, str], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for r in central:
        pair_by_arm_seed[(r["arm"], int(r["seed"]), str(r.get("vocab_scope", "unknown")))][r["condition"]] = r
    paired_diffs: Dict[str, Any] = {}
    for metric in [
        "eval_arm_psc_changed_same_graph", "eval_arm_psc_pair_both_same_graph", "eval_arm_psc_unchanged",
        "eval_arm_mixed_held_seen_orientation_acc", "eval_arm_mixed_held_seen_orientation_signed_margin",
        "eval_arm_heldheld_unseen_edge_closure_acc", "final_train_state_acc", "final_train_cmp_acc",
    ]:
        by_arm: Dict[str, List[float]] = defaultdict(list)
        for (arm, seed, vocab_scope), d in pair_by_arm_seed.items():
            if "tied" in d and "untied" in d and d["tied"].get(metric) is not None and d["untied"].get(metric) is not None:
                by_arm[arm].append(float(d["tied"][metric]) - float(d["untied"][metric]))
        for arm, vals in by_arm.items():
            paired_diffs[f"tied_minus_untied|{arm}|{metric}"] = summ(vals)
    return {
        "status": "SHARED_COORDINATE_MERGE_ANALYSIS_COMPLETE",
        "input_dirs": [str(p) for p in inputs],
        "run_dirs": [str(p) for p in run_dirs],
        "n_runs": len(run_dirs),
        "central_by_run": central,
        "central_groups": central_group,
        "paired_differences": paired_diffs,
        "full_group_summaries": full_group,
        "no_model_loading_training_evaluation_upload_or_leaderboard": True,
    }


def fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def write_summary(out: Path, report: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# research shared-coordinate merged analysis")
    lines.append("")
    lines.append("This file merges saved tied/untied raw-text factorization probe outputs. It reports train fit, direct-anchor state learning, graph-transfer h1/h3 state choices, joint changed+unchanged state conservation, and signed relation-comparison readouts. No model is loaded here.")
    lines.append("")
    lines.append(f"- runs read: {report['n_runs']}")
    lines.append("")
    cols = [
        "condition", "arm", "seed", "vocab_scope", "final_train_state_acc", "final_train_cmp_acc",
        "eval_arm_psc_changed_same_direct", "eval_arm_psc_changed_same_graph",
        "eval_arm_psc_pair_both_same_graph", "eval_arm_psc_unchanged",
        "eval_arm_mixed_held_seen_orientation_acc", "eval_arm_mixed_held_seen_orientation_signed_margin",
        "eval_arm_heldheld_unseen_edge_closure_acc",
    ]
    lines.append("## Central rows")
    lines.append("")
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for r in report.get("central_by_run", []):
        lines.append("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |")
    lines.append("")
    lines.append("## Tied minus untied paired differences")
    lines.append("")
    lines.append("| metric | n | mean | values |")
    lines.append("|---|---:|---:|---|")
    for k, d in sorted(report.get("paired_differences", {}).items()):
        lines.append(f"| {k} | {d.get('n')} | {fmt(d.get('mean'))} | {d.get('values')} |")
    lines.append("")
    lines.append("## Scientific reading")
    lines.append("")
    lines.append("The important pattern is whether tying the event-output coordinate raises same-initial graph-transfer h1/h3 state choice and pair-both conservation while preserving unchanged facts and producing the expected signed mixed held-seen relation orientation. If train comparison fit is weak, the run mainly shows that this raw-text scalar scorer failed to learn the comparison graph, not that the principle is false. If tied beats untied only on direct h0/h2 rows, the sharing constraint has not transported the coordinate through the held-held graph.")
    lines.append("")
    lines.append("## Files")
    lines.append(f"- full JSON: `{out / 'shared_coordinate_merged_analysis.json'}`")
    (out / "shared_coordinate_merged_analysis_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dirs", nargs="+", type=Path, default=DEFAULT_INPUTS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = analyze(args.input_dirs)
    write_json(args.out / "shared_coordinate_merged_analysis.json", report)
    write_summary(args.out, report)
    print(json.dumps({
        "status": report["status"],
        "n_runs": report["n_runs"],
        "json": str(args.out / "shared_coordinate_merged_analysis.json"),
        "summary": str(args.out / "shared_coordinate_merged_analysis_summary.md"),
        "no_model_loading_training_evaluation_upload_or_leaderboard": True,
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
