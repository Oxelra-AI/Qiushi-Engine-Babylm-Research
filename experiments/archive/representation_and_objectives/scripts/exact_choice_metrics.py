#!/usr/bin/env python3
"""research: exact candidate-choice metrics for per-row NLI outputs.

Many earlier summaries report accuracy on the true candidate rows.  That is useful
for pseudolikelihood-style readouts, but it can be fooled by an always-true model.
This script evaluates binary candidate pairs by comparing the true-label margin
for the two candidate rows within each changed/unchanged query.  It can analyze
learned per-row outputs and CPU baseline per-row outputs.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

PROJECT = Path("experiments/archive/representation_and_objectives")
DEFAULT_LEARNED_PER_ROW = PROJECT / "training/runs/balanced_k00_k16_probe/probe_outputs/per_row_eval_predictions.jsonl"
DEFAULT_SURFACE_PER_ROW = PROJECT / "data/surface_linear_baselines/surface_baseline_per_row_predictions.jsonl"
DEFAULT_OUT = PROJECT / "data/exact_choice_metrics"


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
    return float(np.std(np.asarray(xs, dtype=float))) if xs else None


def safe_key(*xs: Any) -> str:
    return "|".join("None" if x is None else str(x) for x in xs)


def group(rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> Dict[str, List[Dict[str, Any]]]:
    d: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        d[safe_key(*(r.get(f) for f in fields))].append(r)
    return dict(d)


def row_score(r: Dict[str, Any]) -> float:
    """Score for selecting the candidate as true.

    Learned outputs contain `margin1_minus_0`.  CPU baselines contain only boolean
    predictions; use 1.0/0.0 as a coarse score so exact-choice ties are visible.
    """
    if "margin1_minus_0" in r:
        return float(r["margin1_minus_0"])
    return 1.0 if bool(r.get("pred")) else 0.0


def summarize(vals: Sequence[float]) -> Dict[str, Any]:
    return {"n": len(vals), "acc": mean(vals), "std": std(vals)}


def pair_choice(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    state = [r for r in rows if r.get("task") == "state_query"]
    by_query: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for r in state:
        key = (
            r.get("condition"), r.get("arm"), r.get("seed"), r.get("baseline_kind"), r.get("suite"),
            r.get("pair_id"), r.get("query_kind"),
        )
        by_query[key].append(r)
    out: List[Dict[str, Any]] = []
    for key, rr in by_query.items():
        # Need exactly one true and one false candidate to make a choice.
        trues = [r for r in rr if bool(r.get("label"))]
        falses = [r for r in rr if not bool(r.get("label"))]
        if len(trues) != 1 or len(falses) != 1:
            continue
        t, f = trues[0], falses[0]
        st, sf = row_score(t), row_score(f)
        if st > sf:
            correct = True
            tie = False
        elif st < sf:
            correct = False
            tie = False
        else:
            correct = None
            tie = True
        out.append({
            "condition": t.get("condition"),
            "arm": t.get("arm"),
            "seed": t.get("seed"),
            "baseline_kind": t.get("baseline_kind"),
            "suite": t.get("suite"),
            "pair_id": t.get("pair_id"),
            "query_kind": t.get("query_kind"),
            "choice_correct": correct,
            "choice_tie": tie,
            "true_score": st,
            "false_score": sf,
            "true_minus_false": st - sf,
            "initial_pattern": t.get("initial_pattern"),
            "static_slot": t.get("static_slot"),
            "relation": t.get("relation"),
            "voice": t.get("voice"),
        })
    return out


def exact_pair_both(choice_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_pair: Dict[Tuple[Any, ...], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for r in choice_rows:
        key = (r.get("condition"), r.get("arm"), r.get("seed"), r.get("baseline_kind"), r.get("suite"), r.get("pair_id"))
        by_pair[key][str(r.get("query_kind"))] = r
    out = []
    for key, d in by_pair.items():
        if "changed" not in d or "unchanged" not in d:
            continue
        c, u = d["changed"], d["unchanged"]
        both = (c.get("choice_correct") is True and u.get("choice_correct") is True)
        out.append({
            "condition": c.get("condition"),
            "arm": c.get("arm"),
            "seed": c.get("seed"),
            "baseline_kind": c.get("baseline_kind"),
            "suite": c.get("suite"),
            "pair_id": c.get("pair_id"),
            "exact_pair_both": both,
            "any_tie": bool(c.get("choice_tie") or u.get("choice_tie")),
            "changed_correct": c.get("choice_correct"),
            "unchanged_correct": u.get("choice_correct"),
            "initial_pattern": c.get("initial_pattern"),
            "static_slot": c.get("static_slot"),
            "relation": c.get("relation"),
            "voice": c.get("voice"),
        })
    return out


def acc_records(records: Sequence[Dict[str, Any]], field: str) -> Dict[str, Any]:
    vals = [float(r[field]) for r in records if r.get(field) is not None]
    ties = [float(r.get("choice_tie") or r.get("any_tie") or False) for r in records]
    return {"n": len(vals), "acc": mean(vals), "std": std(vals), "tie_frac": mean(ties) if ties else None}


def analyze(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    choice = pair_choice(rows)
    both = exact_pair_both(choice)
    report: Dict[str, Any] = {
        "n_input_rows": len(rows),
        "n_choice_queries": len(choice),
        "n_pair_both_records": len(both),
        "choice_by_group": {},
        "pair_both_by_group": {},
    }
    group_fields_list = [
        ["condition", "arm", "seed", "baseline_kind", "suite", "query_kind"],
        ["condition", "arm", "baseline_kind", "suite", "query_kind"],
        ["condition", "arm", "baseline_kind", "suite", "query_kind", "initial_pattern"],
        ["condition", "arm", "baseline_kind", "suite", "query_kind", "initial_pattern", "static_slot"],
        ["condition", "arm", "baseline_kind", "suite", "query_kind", "relation"],
    ]
    for fields in group_fields_list:
        name = "__".join(fields)
        report["choice_by_group"][name] = {k: acc_records(v, "choice_correct") for k, v in sorted(group(choice, fields).items())}
    both_group_fields = [
        ["condition", "arm", "seed", "baseline_kind", "suite"],
        ["condition", "arm", "baseline_kind", "suite"],
        ["condition", "arm", "baseline_kind", "suite", "initial_pattern"],
        ["condition", "arm", "baseline_kind", "suite", "initial_pattern", "static_slot"],
        ["condition", "arm", "baseline_kind", "suite", "relation"],
    ]
    for fields in both_group_fields:
        name = "__".join(fields)
        report["pair_both_by_group"][name] = {k: acc_records(v, "exact_pair_both") for k, v in sorted(group(both, fields).items())}
    return report


def fnum(x: Any) -> str:
    if x is None:
        return "n/a"
    return f"{float(x):.3f}"


def get_metric(rep: Dict[str, Any], group_name: str, key: str, pair: bool = False) -> Dict[str, Any]:
    base = rep["pair_both_by_group" if pair else "choice_by_group"]
    return base.get(group_name, {}).get(key, {"n": 0})


def summary_text(reports: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# research exact candidate-choice metrics")
    lines.append("")
    lines.append("Exact-choice metrics compare the true candidate row's score to the false candidate row's score within each query.  They prevent an always-true classifier from appearing perfect on true-row-only metrics.")
    lines.append("")
    for label, rep in reports.items():
        lines.append(f"## Source `{label}`")
        lines.append("")
        lines.append(f"- input rows: {rep['n_input_rows']}")
        lines.append(f"- choice queries: {rep['n_choice_queries']}")
        lines.append(f"- paired changed+unchanged records: {rep['n_pair_both_records']}")
        lines.append("")
        # Only print central baseline-style table for surface outputs; learned outputs will have baseline_kind None.
        keys = rep["choice_by_group"].get("condition__arm__baseline_kind__suite__query_kind__initial_pattern", {})
        if keys:
            lines.append("| condition | arm | baseline/model | changed same exact | changed opposite exact | pair-both same exact | pair-both opposite exact | tie same changed |")
            lines.append("|---|---|---|---:|---:|---:|---:|---:|")
            combos = sorted(set(tuple(k.split("|")[:3]) for k in keys if "|paired_state_conservation|changed|" in k))
            for condition, arm, bk in combos:
                k_same = f"{condition}|{arm}|{bk}|paired_state_conservation|changed|same"
                k_opp = f"{condition}|{arm}|{bk}|paired_state_conservation|changed|opposite"
                pb_same = f"{condition}|{arm}|{bk}|paired_state_conservation|same"
                pb_opp = f"{condition}|{arm}|{bk}|paired_state_conservation|opposite"
                ms = get_metric(rep, "condition__arm__baseline_kind__suite__query_kind__initial_pattern", k_same)
                mo = get_metric(rep, "condition__arm__baseline_kind__suite__query_kind__initial_pattern", k_opp)
                ps = get_metric(rep, "condition__arm__baseline_kind__suite__initial_pattern", pb_same, pair=True)
                po = get_metric(rep, "condition__arm__baseline_kind__suite__initial_pattern", pb_opp, pair=True)
                lines.append(f"| {condition} | {arm} | {bk} | {fnum(ms.get('acc'))} | {fnum(mo.get('acc'))} | {fnum(ps.get('acc'))} | {fnum(po.get('acc'))} | {fnum(ms.get('tie_frac'))} |")
            lines.append("")
    lines.append("## Scientific reading")
    lines.append("")
    lines.append("- True-row-only accuracy is pseudolikelihood-style evidence; exact-choice is the stricter behavioral readout for two-candidate state queries.")
    lines.append("- The research learned run saves logits, so exact-choice margins should be the primary state/conservation metric once that run finishes.")
    lines.append("- A high true-row score but low exact-choice score indicates over-acceptance or poor calibration, not selective state knowledge.")
    lines.append("")
    lines.append("## Files")
    lines.append(f"- full JSON: `{DEFAULT_OUT / 'exact_choice_metrics_report.json'}`")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--learned-per-row", type=Path, default=DEFAULT_LEARNED_PER_ROW)
    ap.add_argument("--surface-per-row", type=Path, default=DEFAULT_SURFACE_PER_ROW)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--include-learned", action="store_true")
    ap.add_argument("--check-only", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    sources: Dict[str, Path] = {"surface_linear_baselines": args.surface_per_row}
    if args.include_learned:
        sources["learned_balanced_probe"] = args.learned_per_row
    status = {name: {"path": str(path), "exists": path.exists()} for name, path in sources.items()}
    if args.check_only:
        print(json.dumps(status, indent=2, sort_keys=True), flush=True)
        return
    reports: Dict[str, Any] = {}
    for name, path in sources.items():
        if not path.exists():
            raise SystemExit(f"missing {name}: {path}")
        reports[name] = analyze(load_jsonl(path))
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "exact_choice_metrics_report.json", reports)
    (args.out / "exact_choice_metrics_summary.md").write_text(summary_text(reports), encoding="utf-8")
    print(json.dumps({
        "status": "EXACT_CHOICE_METRICS_COMPLETE",
        "summary": str(args.out / "exact_choice_metrics_summary.md"),
        "report": str(args.out / "exact_choice_metrics_report.json"),
        "sources": status,
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
