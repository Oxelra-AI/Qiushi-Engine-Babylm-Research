#!/usr/bin/env python3
"""EWoK domain readout for research directional fork evaluations.

Consumes the eval root produced by `eval_directional_fork_cheap7.py` or
`eval_directional_fork_cheap7_parallel.py` and computes EWoK correctness by
domain for FF/FR/RR/RF. It must use the same full `ewok_filtered` gold directory
as the endpoint evaluator, not the old fast-EWoK subset. The mechanism-relevant value is
I_domain = 0.5*(FR+RF)-0.5*(FF+RR), especially for the research relational domains:
social-properties, physical-dynamics, spatial-relations, physical-relations.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any, Dict, List

ARMS = ["ff", "fr", "rr", "rf"]
RELATIONAL_DOMAINS = ["social-properties", "physical-dynamics", "spatial-relations", "physical-relations"]
ADJ_INDEPENDENT_DOMAINS = ["material-properties", "social-interactions"]
DEFAULT_GOLD = pathlib.Path("experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/ewok_filtered")


def load_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def find_prediction(summary: Dict[str, Any]) -> pathlib.Path:
    tasks = summary.get("tasks", {})
    rec = tasks.get("EWoK")
    if not rec:
        raise KeyError("EWoK task record missing from cheap7_summary.json")
    pred = rec.get("predictions")
    if not pred:
        raise KeyError("EWoK predictions path missing")
    return pathlib.Path(pred)


def score_ewok(pred_path: pathlib.Path, gold_dir: pathlib.Path) -> Dict[str, Any]:
    pred = load_json(pred_path)
    domains: Dict[str, Dict[str, float]] = {}
    subtasks: Dict[str, Dict[str, float]] = {}
    total = correct = 0
    examples = []
    for subtask, data in pred.items():
        gold_file = gold_dir / f"{subtask}.jsonl"
        if not gold_file.exists():
            raise FileNotFoundError(gold_file)
        gold_rows = [json.loads(l) for l in gold_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        preds = data.get("predictions", [])
        if len(preds) != len(gold_rows):
            raise ValueError(f"Length mismatch for {subtask}: pred={len(preds)} gold={len(gold_rows)}")
        for pi, (pr, gold) in enumerate(zip(preds, gold_rows)):
            target = " ".join([gold["Context1"], gold["Target1"]]).strip()
            ok = pr["pred"].strip() == target
            dom = gold.get("Domain", subtask)
            for table, key in [(domains, dom), (subtasks, subtask)]:
                d = table.setdefault(key, {"n": 0, "correct": 0})
                d["n"] += 1
                d["correct"] += int(ok)
            total += 1
            correct += int(ok)
            if len(examples) < 5 and not ok:
                examples.append({"subtask": subtask, "domain": dom, "index": pi, "pred": pr.get("pred"), "target": target})
    for table in [domains, subtasks]:
        for v in table.values():
            v["accuracy"] = 100.0 * v["correct"] / max(v["n"], 1)
    return {"prediction_path": str(pred_path), "total": total, "correct": correct, "accuracy": 100.0 * correct / max(total, 1), "domains": domains, "subtasks": subtasks, "wrong_examples": examples}


def interaction(vals: Dict[str, float]) -> Dict[str, float]:
    one = 0.5 * (vals["ff"] + vals["rr"])
    rec = 0.5 * (vals["fr"] + vals["rf"])
    return {
        "ff": vals["ff"],
        "rr": vals["rr"],
        "fr": vals["fr"],
        "rf": vals["rf"],
        "oneway_avg_ff_rr": one,
        "reciprocal_avg_fr_rf": rec,
        "reciprocal_minus_oneway": rec - one,
        "direction_asymmetry_ff_minus_rr": vals["ff"] - vals["rr"],
        "order_asymmetry_fr_minus_rf": vals["fr"] - vals["rf"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_root", required=True)
    ap.add_argument("--gold_dir", default=str(DEFAULT_GOLD))
    ap.add_argument("--output", default="")
    args = ap.parse_args()
    eval_root = pathlib.Path(args.eval_root)
    gold_dir = pathlib.Path(args.gold_dir)
    arm_scores: Dict[str, Any] = {}
    for arm in ARMS:
        summary_path = eval_root / arm / "cheap7_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)
        pred_path = find_prediction(load_json(summary_path))
        if not pred_path.exists():
            raise FileNotFoundError(pred_path)
        arm_scores[arm] = score_ewok(pred_path, gold_dir)

    domains = sorted(set().union(*(set(arm_scores[a]["domains"].keys()) for a in ARMS)))
    domain_interactions = {}
    for dom in domains:
        vals = {arm: arm_scores[arm]["domains"][dom]["accuracy"] for arm in ARMS if dom in arm_scores[arm]["domains"]}
        if len(vals) == 4:
            domain_interactions[dom] = interaction(vals)
            domain_interactions[dom]["n"] = arm_scores["ff"]["domains"][dom]["n"]
    overall = interaction({arm: arm_scores[arm]["accuracy"] for arm in ARMS})
    relational_vals = {}
    for arm in ARMS:
        n = sum(arm_scores[arm]["domains"].get(d, {}).get("n", 0) for d in RELATIONAL_DOMAINS)
        c = sum(arm_scores[arm]["domains"].get(d, {}).get("correct", 0) for d in RELATIONAL_DOMAINS)
        relational_vals[arm] = 100.0 * c / max(n, 1)
    independent_vals = {}
    for arm in ARMS:
        n = sum(arm_scores[arm]["domains"].get(d, {}).get("n", 0) for d in ADJ_INDEPENDENT_DOMAINS)
        c = sum(arm_scores[arm]["domains"].get(d, {}).get("correct", 0) for d in ADJ_INDEPENDENT_DOMAINS)
        independent_vals[arm] = 100.0 * c / max(n, 1)
    summary = {
        "status": "EWOK_DOMAIN_DIRECTIONAL_ANALYSIS",
        "eval_root": str(eval_root),
        "gold_dir": str(gold_dir),
        "arms": arm_scores,
        "overall_interaction": overall,
        "domain_interactions": domain_interactions,
        "relational_domains": RELATIONAL_DOMAINS,
        "relational_domain_interaction": interaction(relational_vals),
        "adjacency_independent_domains": ADJ_INDEPENDENT_DOMAINS,
        "adjacency_independent_domain_interaction": interaction(independent_vals),
    }
    out = pathlib.Path(args.output) if args.output else eval_root / "ewok_domain_directional_analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    md = ["# research EWoK domain directional analysis", "", f"Eval root: `{eval_root}`", "", f"Overall I: {overall['reciprocal_minus_oneway']}", f"Relational-domain I: {summary['relational_domain_interaction']['reciprocal_minus_oneway']}", f"Adjacency-independent-domain I: {summary['adjacency_independent_domain_interaction']['reciprocal_minus_oneway']}", "", "## Domain interactions", "", "| domain | n | I = reciprocal - one-way | FF | RR | FR | RF |", "|---|---:|---:|---:|---:|---:|---:|"]
    for dom, rec in sorted(domain_interactions.items(), key=lambda kv: kv[1]["reciprocal_minus_oneway"], reverse=True):
        md.append(f"| {dom} | {rec['n']} | {rec['reciprocal_minus_oneway']:.4f} | {rec['ff']:.2f} | {rec['rr']:.2f} | {rec['fr']:.2f} | {rec['rf']:.2f} |")
    md += ["", f"JSON: `{out}`"]
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "output": str(out), "overall_I": overall["reciprocal_minus_oneway"], "relational_I": summary["relational_domain_interaction"]["reciprocal_minus_oneway"], "independent_I": summary["adjacency_independent_domain_interaction"]["reciprocal_minus_oneway"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
