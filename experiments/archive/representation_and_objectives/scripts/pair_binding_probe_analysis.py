#!/usr/bin/env python3
"""research analysis for neutral-block pair-binding learned pilots.

Reads split outputs from connectivity_probe.py run on the research
pair_binding_neutral_substrate.  Produces the readouts needed to decide whether
pair-binding alignment produced a reusable coordinate or only local fitting.
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
DEFAULT_INPUTS = [
    PROJECT / "data/pair_binding_probe_connected_seed28700",
    PROJECT / "data/pair_binding_probe_rewired_seed28700",
]
DEFAULT_OUT = PROJECT / "data/pair_binding_probe_analysis"

DIRECT_ANCHORS = {"h0_dax", "h2_norp"}
GRAPH_TRANSFER = {"h1_mep", "h3_ziv"}
CONDITIONS = ["pair_connected", "pair_rewired"]
ARMS = ["aligned_state_bridge", "inverted_state_bridge", "heldheld_only"]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


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
    return {"n": len(vals), "mean": mean(vals), "std": std(vals), "sem": sem(vals), "values": [float(v) for v in vals]}


def safe_key(*xs: Any) -> str:
    return "|".join("None" if x is None else str(x) for x in xs)


def group(rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> Dict[str, List[Dict[str, Any]]]:
    d: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        d[safe_key(*(r.get(f) for f in fields))].append(r)
    return dict(d)


def row_score(r: Dict[str, Any]) -> float:
    return float(r["margin1_minus_0"])


def state_choice_records(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r.get("task") != "state_query":
            continue
        key = (r.get("condition"), r.get("arm"), r.get("seed"), r.get("suite"), r.get("pair_id"), r.get("query_kind"))
        by[key].append(r)
    out: List[Dict[str, Any]] = []
    for _k, rr in by.items():
        tr = [r for r in rr if bool(r.get("label"))]
        fa = [r for r in rr if not bool(r.get("label"))]
        if len(tr) != 1 or len(fa) != 1:
            continue
        t, f = tr[0], fa[0]
        margin = row_score(t) - row_score(f)
        rel = t.get("relation") or t.get("cause_relation")
        if rel in DIRECT_ANCHORS:
            rel_family = "direct_anchor"
        elif rel in GRAPH_TRANSFER:
            rel_family = "graph_transfer"
        else:
            rel_family = "seen_or_other"
        out.append({
            "condition": t.get("condition"),
            "arm": t.get("arm"),
            "seed": t.get("seed"),
            "suite": t.get("suite"),
            "pair_id": t.get("pair_id"),
            "query_kind": t.get("query_kind"),
            "choice_correct": bool(margin > 0),
            "choice_tie": bool(margin == 0),
            "true_minus_false_margin": float(margin),
            "initial_pattern": t.get("initial_pattern"),
            "static_slot": t.get("static_slot"),
            "relation": rel,
            "relation_family": rel_family,
            "voice": t.get("voice"),
        })
    return out


def pair_both_records(choice: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by: Dict[Tuple[Any, ...], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for r in choice:
        by[(r.get("condition"), r.get("arm"), r.get("seed"), r.get("suite"), r.get("pair_id"))][str(r.get("query_kind"))] = r
    out: List[Dict[str, Any]] = []
    for _k, d in by.items():
        if "changed" not in d or "unchanged" not in d:
            continue
        c, u = d["changed"], d["unchanged"]
        out.append({
            "condition": c.get("condition"), "arm": c.get("arm"), "seed": c.get("seed"), "suite": c.get("suite"),
            "pair_id": c.get("pair_id"), "pair_both_correct": bool(c.get("choice_correct") and u.get("choice_correct")),
            "changed_correct": bool(c.get("choice_correct")), "unchanged_correct": bool(u.get("choice_correct")),
            "initial_pattern": c.get("initial_pattern"), "static_slot": c.get("static_slot"),
            "relation": c.get("relation"), "relation_family": c.get("relation_family"), "voice": c.get("voice"),
            "changed_margin": c.get("true_minus_false_margin"), "unchanged_margin": u.get("true_minus_false_margin"),
        })
    return out


def summarize_choice(rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for k, rr in sorted(group(rows, fields).items()):
        out[k] = {
            "n": len(rr),
            "acc": mean([float(r["choice_correct"]) for r in rr]),
            "margin_mean": mean([float(r["true_minus_false_margin"]) for r in rr]),
            "margin_std": std([float(r["true_minus_false_margin"]) for r in rr]),
            "tie_frac": mean([float(r["choice_tie"]) for r in rr]),
        }
    return out


def summarize_both(rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for k, rr in sorted(group(rows, fields).items()):
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
        m = row_score(r)
        signed = m if bool(r.get("label")) else -m
        dep = str(r.get("orientation_dependency"))
        row_id = str(r.get("row_id"))
        if dep.startswith("neutral") or "neutral" in dep:
            info_kind = "neutral"
        elif dep.startswith("heldheld_informative") or row_id.startswith("B_hh") or row_id.startswith("R_hh"):
            info_kind = "informative_or_eval"
        else:
            info_kind = dep
        out.append({
            "condition": r.get("condition"), "arm": r.get("arm"), "seed": r.get("seed"), "suite": r.get("suite"),
            "relation1": r.get("relation1"), "relation2": r.get("relation2"),
            "component1": r.get("component1"), "component2": r.get("component2"),
            "label": bool(r.get("label")), "pred": bool(r.get("pred")), "correct": bool(r.get("correct")),
            "margin": float(m), "signed_margin_true_coordinate": float(signed),
            "orientation_dependency": dep, "info_kind": info_kind,
            "true_row_accept": bool(r.get("pred")) if bool(r.get("label")) else None,
            "false_row_reject": (not bool(r.get("pred"))) if not bool(r.get("label")) else None,
        })
    return out


def summarize_comp(rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for k, rr in sorted(group(rows, fields).items()):
        out[k] = {
            "n": len(rr),
            "accuracy_vs_label": mean([float(r["correct"]) for r in rr]),
            "pred_true_frac": mean([float(r["pred"]) for r in rr]),
            "signed_margin_mean": mean([float(r["signed_margin_true_coordinate"]) for r in rr]),
            "signed_margin_std": std([float(r["signed_margin_true_coordinate"]) for r in rr]),
            "true_row_accept": mean([float(r["true_row_accept"]) for r in rr if r["true_row_accept"] is not None]),
            "false_row_reject": mean([float(r["false_row_reject"]) for r in rr if r["false_row_reject"] is not None]),
        }
    return out


def all_row_fit(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for k, rr in sorted(group(rows, ["condition", "arm", "seed"]).items()):
        out[k] = {"n": len(rr), "acc": mean([float(r.get("correct")) for r in rr]), "label_true_frac": mean([float(r.get("label")) for r in rr if "label" in r])}
    return out


def build_report(input_dirs: Sequence[Path]) -> Dict[str, Any]:
    eval_rows: List[Dict[str, Any]] = []
    train_rows: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []
    sources: Dict[str, Any] = {}
    for d in input_dirs:
        d = Path(d)
        er = load_jsonl(d / "per_row_eval_predictions.jsonl")
        tr = load_jsonl(d / "per_row_train_predictions.jsonl")
        ar = load_json(d / "all_results.json") or []
        eval_rows.extend(er); train_rows.extend(tr); results.extend(ar)
        sources[str(d)] = {"eval_rows": len(er), "train_rows": len(tr), "results": len(ar), "exists": d.exists()}

    train_choice = state_choice_records(train_rows)
    eval_choice = state_choice_records(eval_rows)
    train_both = pair_both_records(train_choice)
    eval_both = pair_both_records(eval_choice)
    train_comp = comparison_signed_rows(train_rows)
    eval_comp = comparison_signed_rows(eval_rows)
    mixed = [r for r in eval_comp if r.get("suite") == "mixed_held_seen_orientation"]

    mixed_by_seed = summarize_comp(mixed, ["condition", "arm", "seed"])
    mixed_by_arm = summarize_comp(mixed, ["condition", "arm"])
    mixed_by_rel = summarize_comp(mixed, ["condition", "arm", "relation1", "relation2"])

    interactions: Dict[str, Any] = {}
    for cond in sorted({r.get("condition") for r in mixed}):
        diffs = []
        diffs_acc = []
        for seed in sorted({r.get("seed") for r in mixed if r.get("condition") == cond}):
            ka = f"{cond}|aligned_state_bridge|{seed}"
            ki = f"{cond}|inverted_state_bridge|{seed}"
            if ka in mixed_by_seed and ki in mixed_by_seed:
                diffs.append(float(mixed_by_seed[ka]["signed_margin_mean"]) - float(mixed_by_seed[ki]["signed_margin_mean"]))
                diffs_acc.append(float(mixed_by_seed[ka]["accuracy_vs_label"]) - float(mixed_by_seed[ki]["accuracy_vs_label"]))
        interactions[str(cond)] = {"aligned_minus_inverted_signed_margin": summarize(diffs), "aligned_minus_inverted_accuracy": summarize(diffs_acc)}

    return {
        "status": "PAIR_BINDING_PROBE_ANALYSIS_COMPLETE",
        "sources": sources,
        "n_results": len([r for r in results if isinstance(r, dict) and "error" not in r]),
        "n_errors": len([r for r in results if isinstance(r, dict) and "error" in r]),
        "all_results": results,
        "n_train_rows": len(train_rows),
        "n_eval_rows": len(eval_rows),
        "train_fit_by_condition_arm_seed": all_row_fit(train_rows),
        "train_comparison_by_info_kind": summarize_comp(train_comp, ["condition", "arm", "seed", "info_kind"]),
        "eval_state_changed_by_condition_arm_seed_suite_pattern": summarize_choice([r for r in eval_choice if r.get("query_kind") == "changed"], ["condition", "arm", "seed", "suite", "initial_pattern"]),
        "eval_state_changed_by_condition_arm_suite_pattern": summarize_choice([r for r in eval_choice if r.get("query_kind") == "changed"], ["condition", "arm", "suite", "initial_pattern"]),
        "eval_state_changed_by_condition_arm_suite_relation_family_pattern": summarize_choice([r for r in eval_choice if r.get("query_kind") == "changed"], ["condition", "arm", "suite", "relation_family", "initial_pattern"]),
        "eval_pair_both_by_condition_arm_suite_pattern": summarize_both(eval_both, ["condition", "arm", "suite", "initial_pattern"]),
        "eval_pair_both_by_condition_arm_suite_relation_family_pattern": summarize_both(eval_both, ["condition", "arm", "suite", "relation_family", "initial_pattern"]),
        "mixed_signed_by_condition_arm_seed": mixed_by_seed,
        "mixed_signed_by_condition_arm": mixed_by_arm,
        "mixed_signed_by_relation_pair": mixed_by_rel,
        "mixed_aligned_minus_inverted_interactions": interactions,
    }


def fmt(x: Any, digits: int = 3) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def get(d: Dict[str, Any], key: str, field: str) -> Any:
    return d.get(key, {}).get(field)


def write_summary(path: Path, rep: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# research pair-binding learned pilot analysis")
    lines.append("")
    lines.append("This analysis reads split learned outputs from the neutral-block pair-binding substrate.  The decisive interpretation requires local train fit, same-initial changed exact choice, exact pair-both conservation, and signed aligned-vs-inverted mixed margins.")
    lines.append("")
    lines.append(f"- completed results: {rep['n_results']}; errors: {rep['n_errors']}")
    lines.append(f"- train rows: {rep['n_train_rows']}; eval rows: {rep['n_eval_rows']}")
    lines.append("")
    lines.append("## Train fit")
    lines.append("")
    lines.append("| condition | arm | seed | all train acc | informative comparison acc | neutral comparison acc |")
    lines.append("|---|---|---:|---:|---:|---:|")
    fit = rep["train_fit_by_condition_arm_seed"]
    comp = rep["train_comparison_by_info_kind"]
    combos = sorted(set(tuple(k.split("|")) for k in fit.keys()))
    for cond, arm, seed in combos:
        k = f"{cond}|{arm}|{seed}"
        ki = f"{cond}|{arm}|{seed}|informative_or_eval"
        kn = f"{cond}|{arm}|{seed}|neutral"
        lines.append(f"| {cond} | {arm} | {seed} | {fmt(get(fit, k, 'acc'))} | {fmt(get(comp, ki, 'accuracy_vs_label'))} | {fmt(get(comp, kn, 'accuracy_vs_label'))} |")
    lines.append("")
    lines.append("## Paired-state exact choice")
    lines.append("")
    lines.append("| condition | arm | suite | pattern | changed exact | pair-both | changed margin |")
    lines.append("|---|---|---|---|---:|---:|---:|")
    ch = rep["eval_state_changed_by_condition_arm_suite_pattern"]
    pb = rep["eval_pair_both_by_condition_arm_suite_pattern"]
    combos2 = sorted(set(tuple(k.split("|")[:4]) for k in ch.keys() if "state" in k or "conservation" in k or "readout" in k))
    for cond, arm, suite, pattern in combos2:
        if suite not in {"paired_state_conservation", "cross_template_state_readout"}:
            continue
        k = f"{cond}|{arm}|{suite}|{pattern}"
        lines.append(f"| {cond} | {arm} | {suite} | {pattern} | {fmt(get(ch, k, 'acc'))} | {fmt(get(pb, k, 'pair_both_acc'))} | {fmt(get(ch, k, 'margin_mean'))} |")
    lines.append("")
    lines.append("## Direct-anchor versus graph-transfer state readout")
    lines.append("")
    lines.append("| condition | arm | suite | relation family | pattern | changed exact | pair-both |")
    lines.append("|---|---|---|---|---|---:|---:|")
    chf = rep["eval_state_changed_by_condition_arm_suite_relation_family_pattern"]
    pbf = rep["eval_pair_both_by_condition_arm_suite_relation_family_pattern"]
    for k in sorted(chf):
        cond, arm, suite, fam, pattern = k.split("|")
        if suite != "paired_state_conservation":
            continue
        lines.append(f"| {cond} | {arm} | {suite} | {fam} | {pattern} | {fmt(get(chf, k, 'acc'))} | {fmt(get(pbf, k, 'pair_both_acc'))} |")
    lines.append("")
    lines.append("## Mixed held-seen signed margins")
    lines.append("")
    lines.append("| condition | arm | acc vs true labels | pred true frac | signed margin mean | true accept | false reject |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    mx = rep["mixed_signed_by_condition_arm"]
    for k in sorted(mx):
        cond, arm = k.split("|")
        r = mx[k]
        lines.append(f"| {cond} | {arm} | {fmt(r.get('accuracy_vs_label'))} | {fmt(r.get('pred_true_frac'))} | {fmt(r.get('signed_margin_mean'))} | {fmt(r.get('true_row_accept'))} | {fmt(r.get('false_row_reject'))} |")
    lines.append("")
    lines.append("| condition | aligned-inverted signed margin | aligned-inverted accuracy |")
    lines.append("|---|---:|---:|")
    for cond, rec in sorted(rep["mixed_aligned_minus_inverted_interactions"].items()):
        sm = rec["aligned_minus_inverted_signed_margin"]
        ac = rec["aligned_minus_inverted_accuracy"]
        lines.append(f"| {cond} | {fmt(sm.get('mean'))}±{fmt(sm.get('std'))} | {fmt(ac.get('mean'))}±{fmt(ac.get('std'))} |")
    lines.append("")
    lines.append("## Scientific reading")
    lines.append("")
    lines.append("A pair-binding alignment signal requires pair_connected to outperform pair_rewired specifically on same-initial changed choice and pair-both conservation while also showing an aligned-vs-inverted reversal of mixed held-seen signed margins.  High train fit without those eval readouts reproduces the research pattern: local fitting without reusable coordinate transport.  If train fit is poor, optimize the supervised head/epochs before interpreting the substrate.")
    lines.append("")
    lines.append("## Files")
    lines.append(f"- full JSON: `{path.parent / 'pair_binding_probe_analysis.json'}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dirs", nargs="+", type=Path, default=DEFAULT_INPUTS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    rep = build_report(args.input_dirs)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "pair_binding_probe_analysis.json", rep)
    write_summary(args.out / "pair_binding_probe_analysis_summary.md", rep)
    print(json.dumps({
        "status": rep["status"],
        "summary": str(args.out / "pair_binding_probe_analysis_summary.md"),
        "json": str(args.out / "pair_binding_probe_analysis.json"),
        "n_results": rep["n_results"],
        "n_errors": rep["n_errors"],
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
