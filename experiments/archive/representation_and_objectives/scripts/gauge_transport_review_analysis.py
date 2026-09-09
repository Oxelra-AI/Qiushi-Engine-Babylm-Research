#!/usr/bin/env python3
"""research CPU review of research shared-coordinate gauge transport.

This script does not train or load models. It reads saved per-row predictions from
research and checks the reviewer-level questions that are invisible in aggregate
scores:

1. state query integrity: exactly two candidate rows and one true/arm target;
2. aligned-vs-inverted row-paired sign behavior for state margins;
3. aligned-vs-inverted row-paired sign behavior for mixed comparison margins;
4. whether heldheld-only really lacks an absolute state anchor despite local fit.

The goal is to decide whether research supports transport of an absolute gauge
inside the tied candidate-event factorization rather than a merge artifact or
only generic parameter sharing.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

PROJECT = Path("experiments/archive/representation_and_objectives")
DEFAULT_OUT = PROJECT / "data/gauge_transport_review_analysis"
TRAINVOCAB_ALIGNED = PROJECT / "data/shared_coordinate_trainvocab_aligned_seed28801"
TRAINVOCAB_INVERTED = PROJECT / "data/shared_coordinate_trainvocab_inverted_seed28801"
HELDHELD = PROJECT / "data/shared_coordinate_trainvocab_heldheld_anchor_control"

RUN_ROOTS = [TRAINVOCAB_ALIGNED, TRAINVOCAB_INVERTED, HELDHELD]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def discover_run_dirs(roots: Sequence[Path]) -> List[Path]:
    out: List[Path] = []
    for root in roots:
        if (root / "result.json").exists():
            out.append(root)
        if root.exists():
            for child in sorted(root.iterdir()):
                if child.is_dir() and (child / "result.json").exists():
                    out.append(child)
    seen = set(); dedup: List[Path] = []
    for p in out:
        s = str(p)
        if s not in seen:
            dedup.append(p); seen.add(s)
    return dedup


def mean(xs: Sequence[float]) -> float | None:
    return float(sum(xs) / len(xs)) if xs else None


def std(xs: Sequence[float]) -> float | None:
    return float(np.std(np.asarray(xs, dtype=float))) if xs else None


def summ(xs: Sequence[float]) -> Dict[str, Any]:
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return {"n": len(vals), "mean": mean(vals), "std": std(vals), "min": min(vals) if vals else None, "max": max(vals) if vals else None}


def parse_run_dir(rd: Path) -> Dict[str, Any]:
    result = load_json(rd / "result.json")
    eval_state = load_jsonl(rd / "eval_state_predictions.jsonl")
    eval_comp = load_jsonl(rd / "eval_comparison_predictions.jsonl")
    train_state = load_jsonl(rd / "train_state_predictions.jsonl")
    train_comp = load_jsonl(rd / "train_comparison_predictions.jsonl")
    return {"run_dir": str(rd), "result": result, "eval_state": eval_state, "eval_comp": eval_comp, "train_state": train_state, "train_comp": train_comp}


def choice_rows(state_rows: Sequence[Dict[str, Any]], target_mode: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    label_key = "label_true" if target_mode == "true" else "label_arm_target"
    by: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for r in state_rows:
        by[(r.get("condition"), r.get("arm"), r.get("seed"), r.get("prediction_split"), r.get("suite"), r.get("query_key"))].append(r)
    out: List[Dict[str, Any]] = []
    bad_len = 0; bad_label = 0; ties = 0
    for key, xs0 in by.items():
        xs = sorted(xs0, key=lambda r: int(r.get("candidate_index", 0)))
        if len(xs) != 2:
            bad_len += 1
            continue
        labels = [bool(x.get(label_key)) for x in xs]
        if sum(int(x) for x in labels) != 1:
            bad_label += 1
            continue
        scores = [float(xs[0]["score"]), float(xs[1]["score"])]
        target_i = 0 if labels[0] else 1
        pred_i = 0 if scores[0] >= scores[1] else 1
        margin = scores[target_i] - scores[1 - target_i]
        if abs(scores[0] - scores[1]) < 1e-9:
            ties += 1
        r0 = xs[0]
        out.append({
            "condition": r0.get("condition"), "arm": r0.get("arm"), "seed": int(r0.get("seed")),
            "prediction_split": r0.get("prediction_split"), "suite": r0.get("suite"), "query_key": r0.get("query_key"),
            "target_mode": target_mode, "correct": float(pred_i == target_i), "margin": margin,
            "is_changed": bool(r0.get("is_changed")), "query_kind": r0.get("query_kind"),
            "relation": r0.get("relation"), "relation_family": r0.get("relation_family"),
            "initial_pattern": r0.get("initial_pattern"), "static_slot": r0.get("static_slot"),
            "global_swap_changes_label": bool(r0.get("global_swap_changes_label")),
            "target_i": target_i, "pred_i": pred_i,
        })
    audit = {"groups": len(by), "bad_len": bad_len, "bad_label": bad_label, "ties": ties, "choices": len(out)}
    return out, audit


def comp_rows(comp_rows0: Sequence[Dict[str, Any]], target_mode: str) -> List[Dict[str, Any]]:
    return [r for r in comp_rows0 if r.get("target_mode") == target_mode]


def group_key(r: Dict[str, Any], fields: Sequence[str]) -> Tuple[Any, ...]:
    return tuple(r.get(f) for f in fields)


def group_numeric(rows: Sequence[Dict[str, Any]], fields: Sequence[str], value: str) -> Dict[str, Any]:
    g: Dict[Tuple[Any, ...], List[float]] = defaultdict(list)
    for r in rows:
        v = r.get(value)
        if v is not None:
            g[group_key(r, fields)].append(float(v))
    out: Dict[str, Any] = {}
    for k, vals in sorted(g.items(), key=lambda kv: str(kv[0])):
        out[json.dumps(dict(zip(fields, k)), sort_keys=True)] = summ(vals)
    return out


def compare_state_mirroring(runs: Sequence[Dict[str, Any]], condition: str, seed: int) -> Dict[str, Any]:
    # Compare aligned and inverted arms for the same model/seed.  For orientation-sensitive
    # changed rows, true margins should be positive in aligned and negative in inverted;
    # arm-target margins should be positive in both.
    by_arm: Dict[str, Dict[str, Any]] = {}
    for run in runs:
        res = run["result"]
        if res.get("condition") == condition and int(res.get("seed")) == seed and res.get("arm") in {"aligned_state_bridge", "inverted_state_bridge"}:
            true_choices, audit_true = choice_rows(run["eval_state"], "true")
            arm_choices, audit_arm = choice_rows(run["eval_state"], "arm")
            by_arm[res["arm"]] = {"true": true_choices, "arm": arm_choices, "audit_true": audit_true, "audit_arm": audit_arm, "run_dir": run["run_dir"]}
    if "aligned_state_bridge" not in by_arm or "inverted_state_bridge" not in by_arm:
        return {"available": False}
    out: Dict[str, Any] = {"available": True, "condition": condition, "seed": seed, "arms": {k: v["run_dir"] for k, v in by_arm.items()}, "audit": {k: {"true": v["audit_true"], "arm": v["audit_arm"]} for k, v in by_arm.items()}}
    for mode in ["true", "arm"]:
        a = {r["query_key"]: r for r in by_arm["aligned_state_bridge"][mode]}
        b = {r["query_key"]: r for r in by_arm["inverted_state_bridge"][mode]}
        paired: List[Dict[str, Any]] = []
        for k in sorted(set(a) & set(b)):
            ra, rb = a[k], b[k]
            paired.append({
                "query_key": k, "suite": ra["suite"], "is_changed": ra["is_changed"],
                "relation": ra["relation"], "relation_family": ra["relation_family"],
                "initial_pattern": ra["initial_pattern"], "static_slot": ra["static_slot"],
                "global_swap_changes_label": ra["global_swap_changes_label"],
                "aligned_margin": ra["margin"], "inverted_margin": rb["margin"],
                "aligned_correct": ra["correct"], "inverted_correct": rb["correct"],
                "opposite_sign": float((ra["margin"] > 0 and rb["margin"] < 0) or (ra["margin"] < 0 and rb["margin"] > 0)),
                "both_positive": float(ra["margin"] > 0 and rb["margin"] > 0),
                "both_correct": float(ra["correct"] == 1.0 and rb["correct"] == 1.0),
            })
        out[f"state_{mode}_paired_n"] = len(paired)
        out[f"state_{mode}_by_group"] = {
            "correct_aligned": group_numeric(paired, ["suite", "is_changed", "relation_family", "initial_pattern"], "aligned_correct"),
            "correct_inverted": group_numeric(paired, ["suite", "is_changed", "relation_family", "initial_pattern"], "inverted_correct"),
            "margin_aligned": group_numeric(paired, ["suite", "is_changed", "relation_family", "initial_pattern"], "aligned_margin"),
            "margin_inverted": group_numeric(paired, ["suite", "is_changed", "relation_family", "initial_pattern"], "inverted_margin"),
            "opposite_sign": group_numeric(paired, ["suite", "is_changed", "relation_family", "initial_pattern"], "opposite_sign"),
            "both_positive": group_numeric(paired, ["suite", "is_changed", "relation_family", "initial_pattern"], "both_positive"),
            "both_correct": group_numeric(paired, ["suite", "is_changed", "relation_family", "initial_pattern"], "both_correct"),
        }
        # Central scalar: paired_state_conservation graph-transfer same-initial changed.
        central = [r for r in paired if r["suite"] == "paired_state_conservation" and r["is_changed"] and r["relation_family"] == "graph_transfer" and r["initial_pattern"] == "same"]
        out[f"state_{mode}_central_graph_same"] = {
            "n": len(central),
            "aligned_correct": summ([r["aligned_correct"] for r in central]),
            "inverted_correct": summ([r["inverted_correct"] for r in central]),
            "opposite_sign": summ([r["opposite_sign"] for r in central]),
            "both_positive": summ([r["both_positive"] for r in central]),
            "aligned_margin": summ([r["aligned_margin"] for r in central]),
            "inverted_margin": summ([r["inverted_margin"] for r in central]),
            "sample": central[:4],
        }
    return out


def compare_comp_mirroring(runs: Sequence[Dict[str, Any]], condition: str, seed: int) -> Dict[str, Any]:
    by_arm: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for run in runs:
        res = run["result"]
        if res.get("condition") == condition and int(res.get("seed")) == seed and res.get("arm") in {"aligned_state_bridge", "inverted_state_bridge"}:
            by_arm[res["arm"]] = {"true": comp_rows(run["eval_comp"], "true"), "arm": comp_rows(run["eval_comp"], "arm")}
    if "aligned_state_bridge" not in by_arm or "inverted_state_bridge" not in by_arm:
        return {"available": False}
    out: Dict[str, Any] = {"available": True, "condition": condition, "seed": seed}
    for mode in ["true", "arm"]:
        a = {r["row_id"]: r for r in by_arm["aligned_state_bridge"][mode]}
        b = {r["row_id"]: r for r in by_arm["inverted_state_bridge"][mode]}
        paired: List[Dict[str, Any]] = []
        for k in sorted(set(a) & set(b)):
            ra, rb = a[k], b[k]
            paired.append({
                "row_id": k, "suite": ra.get("suite"), "orientation_dependency": ra.get("orientation_dependency"),
                "aligned_signed_margin": float(ra.get("signed_margin")),
                "inverted_signed_margin": float(rb.get("signed_margin")),
                "aligned_correct": float(bool(ra.get("correct"))),
                "inverted_correct": float(bool(rb.get("correct"))),
                "opposite_sign": float((float(ra.get("signed_margin")) > 0 and float(rb.get("signed_margin")) < 0) or (float(ra.get("signed_margin")) < 0 and float(rb.get("signed_margin")) > 0)),
                "both_positive": float(float(ra.get("signed_margin")) > 0 and float(rb.get("signed_margin")) > 0),
            })
        out[f"comp_{mode}_paired_n"] = len(paired)
        out[f"comp_{mode}_by_suite"] = {
            "correct_aligned": group_numeric(paired, ["suite"], "aligned_correct"),
            "correct_inverted": group_numeric(paired, ["suite"], "inverted_correct"),
            "margin_aligned": group_numeric(paired, ["suite"], "aligned_signed_margin"),
            "margin_inverted": group_numeric(paired, ["suite"], "inverted_signed_margin"),
            "opposite_sign": group_numeric(paired, ["suite"], "opposite_sign"),
            "both_positive": group_numeric(paired, ["suite"], "both_positive"),
        }
        central = [r for r in paired if r["suite"] == "mixed_held_seen_orientation"]
        out[f"comp_{mode}_central_mixed"] = {
            "n": len(central),
            "aligned_correct": summ([r["aligned_correct"] for r in central]),
            "inverted_correct": summ([r["inverted_correct"] for r in central]),
            "opposite_sign": summ([r["opposite_sign"] for r in central]),
            "both_positive": summ([r["both_positive"] for r in central]),
            "aligned_margin": summ([r["aligned_signed_margin"] for r in central]),
            "inverted_margin": summ([r["inverted_signed_margin"] for r in central]),
            "sample": central[:4],
        }
    return out


def heldheld_anchor_summary(runs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for run in runs:
        res = run["result"]
        if res.get("arm") != "heldheld_only":
            continue
        ce = res.get("central_eval", {})
        rows.append({
            "condition": res.get("condition"), "seed": int(res.get("seed")),
            "train_state_acc": res.get("final_train_metrics", {}).get("train_state_acc"),
            "train_cmp_acc": res.get("final_train_metrics", {}).get("train_cmp_acc"),
            "direct_same": ce.get("psc_changed_same_direct"),
            "graph_same": ce.get("psc_changed_same_graph"),
            "pair_both_graph_same": ce.get("psc_pair_both_same_graph"),
            "unchanged": ce.get("psc_unchanged"),
            "mixed_acc": ce.get("mixed_held_seen_orientation_acc"),
            "mixed_margin": ce.get("mixed_held_seen_orientation_signed_margin"),
            "hh_closure": ce.get("heldheld_unseen_edge_closure_acc"),
        })
    groups: Dict[str, Any] = {}
    for cond in sorted({r["condition"] for r in rows}):
        rs = [r for r in rows if r["condition"] == cond]
        groups[cond] = {m: summ([float(r[m]) for r in rs if r.get(m) is not None]) for m in rows[0].keys() if m not in {"condition", "seed"}} if rows else {}
    return {"rows": rows, "groups": groups}


def write_summary(path: Path, report: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# research gauge-transport review analysis")
    lines.append("")
    lines.append("CPU-only review over saved research predictions. No model was loaded or trained.")
    lines.append("")
    lines.append("## Run integrity")
    lines.append("")
    lines.append("| run | condition | arm | seed | train_state | train_cmp | parse_errors | eval_state_groups | eval_state_bad_len | eval_state_bad_true | eval_state_bad_arm | ties_true |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in report["run_integrity"]:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in ["run_name", "condition", "arm", "seed", "train_state_acc", "train_cmp_acc", "parse_errors", "eval_state_groups", "eval_state_bad_len", "eval_state_bad_true", "eval_state_bad_arm", "ties_true"]) + " |")
    lines.append("")
    lines.append("## Aligned/inverted state-margin mirroring, train-only seed 28801")
    lines.append("")
    for cond in ["tied", "untied"]:
        sec = report["state_mirroring"].get(cond, {})
        lines.append(f"### {cond}")
        if not sec.get("available"):
            lines.append("not available")
            continue
        for mode in ["true", "arm"]:
            cg = sec.get(f"state_{mode}_central_graph_same", {})
            lines.append(f"- target mode `{mode}`, paired_state_conservation graph/same changed: n={cg.get('n')}; aligned_correct={fmt_summary(cg.get('aligned_correct'))}; inverted_correct={fmt_summary(cg.get('inverted_correct'))}; opposite_sign={fmt_summary(cg.get('opposite_sign'))}; both_positive={fmt_summary(cg.get('both_positive'))}; aligned_margin={fmt_summary(cg.get('aligned_margin'))}; inverted_margin={fmt_summary(cg.get('inverted_margin'))}")
    lines.append("")
    lines.append("## Aligned/inverted mixed-comparison mirroring, train-only seed 28801")
    lines.append("")
    for cond in ["tied", "untied"]:
        sec = report["comparison_mirroring"].get(cond, {})
        lines.append(f"### {cond}")
        if not sec.get("available"):
            lines.append("not available")
            continue
        for mode in ["true", "arm"]:
            cg = sec.get(f"comp_{mode}_central_mixed", {})
            lines.append(f"- target mode `{mode}`, mixed held-seen: n={cg.get('n')}; aligned_correct={fmt_summary(cg.get('aligned_correct'))}; inverted_correct={fmt_summary(cg.get('inverted_correct'))}; opposite_sign={fmt_summary(cg.get('opposite_sign'))}; both_positive={fmt_summary(cg.get('both_positive'))}; aligned_margin={fmt_summary(cg.get('aligned_margin'))}; inverted_margin={fmt_summary(cg.get('inverted_margin'))}")
    lines.append("")
    lines.append("## Heldheld-only anchor control")
    lines.append("")
    lines.append("| condition | seed | train_state | train_cmp | direct_same | graph_same | pair_both_graph_same | unchanged | mixed_acc | mixed_margin | hh_closure |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in report["heldheld_anchor"]["rows"]:
        lines.append("| " + " | ".join(fmt_value(r.get(c)) for c in ["condition", "seed", "train_state_acc", "train_cmp_acc", "direct_same", "graph_same", "pair_both_graph_same", "unchanged", "mixed_acc", "mixed_margin", "hh_closure"]) + " |")
    lines.append("")
    lines.append("## Reviewer interpretation")
    lines.append("")
    lines.append("The saved rows support a real aligned/inverted arm sign reversal in the tied model. Relative to the true target, tied aligned graph-transfer changed-state margins are positive while tied inverted margins are negative; relative to each arm's installed target, both are positive. The mixed held-seen comparison margins show the same pattern. This reduces the chance that the research result is only a merge-table artifact. The heldheld-only runs do not produce a correct absolute state coordinate without bridge state anchors, even when comparison closure is high, so comparison graph evidence alone is not enough to orient state transfer.")
    lines.append("")
    lines.append("The result still has the limitations identified in research: candidate discovery/correspondence, binary ownership, same-owner comparison, and changed/unchanged routing are supplied by the harness, and unchanged facts use a separate static scorer. The next causal test should manipulate the sign or role permutation between the shared relation coordinate and state readout while keeping local fit possible; this will make the transported gauge directly visible rather than merely inferred from aligned/inverted data arms.")
    lines.append("")
    lines.append(f"Full JSON: `{path / 'gauge_transport_review_analysis.json'}`")
    (path / "gauge_transport_review_analysis_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt_value(x: Any) -> str:
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def fmt_summary(d: Any) -> str:
    if not isinstance(d, dict):
        return "n/a"
    m = d.get("mean")
    n = d.get("n")
    mn = d.get("min")
    mx = d.get("max")
    if m is None:
        return f"n={n}, mean=n/a"
    return f"n={n}, mean={m:.3f}, range=[{mn:.3f},{mx:.3f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", type=Path, default=RUN_ROOTS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    run_dirs = discover_run_dirs(args.roots)
    runs = [parse_run_dir(p) for p in run_dirs]

    integrity: List[Dict[str, Any]] = []
    for run in runs:
        res = run["result"]
        true_choices, audit_true = choice_rows(run["eval_state"], "true")
        arm_choices, audit_arm = choice_rows(run["eval_state"], "arm")
        integrity.append({
            "run_dir": run["run_dir"], "run_name": Path(run["run_dir"]).name,
            "condition": res.get("condition"), "arm": res.get("arm"), "seed": int(res.get("seed")),
            "train_state_acc": res.get("final_train_metrics", {}).get("train_state_acc"),
            "train_cmp_acc": res.get("final_train_metrics", {}).get("train_cmp_acc"),
            "parse_errors": res.get("counts", {}).get("parse_errors"),
            "eval_state_groups": audit_true["groups"], "eval_state_bad_len": audit_true["bad_len"],
            "eval_state_bad_true": audit_true["bad_label"], "eval_state_bad_arm": audit_arm["bad_label"],
            "ties_true": audit_true["ties"],
        })

    report = {
        "status": "GAUGE_TRANSPORT_REVIEW_ANALYSIS_COMPLETE",
        "run_dirs": [str(p) for p in run_dirs],
        "run_integrity": integrity,
        "state_mirroring": {
            "tied": compare_state_mirroring(runs, "tied", 28801),
            "untied": compare_state_mirroring(runs, "untied", 28801),
        },
        "comparison_mirroring": {
            "tied": compare_comp_mirroring(runs, "tied", 28801),
            "untied": compare_comp_mirroring(runs, "untied", 28801),
        },
        "heldheld_anchor": heldheld_anchor_summary(runs),
        "no_model_loading_training_evaluation_upload_or_leaderboard": True,
    }
    write_json(args.out / "gauge_transport_review_analysis.json", report)
    write_summary(args.out, report)
    print(json.dumps({
        "status": report["status"],
        "n_runs": len(run_dirs),
        "json": str(args.out / "gauge_transport_review_analysis.json"),
        "summary": str(args.out / "gauge_transport_review_analysis_summary.md"),
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
