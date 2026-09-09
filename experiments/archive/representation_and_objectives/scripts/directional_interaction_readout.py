#!/usr/bin/env python3
"""Readout for research forked directional causal screens.

The scientific estimand is not FR-FF alone. For each metric m, the directional
interaction is

  I_m(data) = 0.5 * (m_FR + m_RF) - 0.5 * (m_FF + m_RR)

where FF/RR are one-direction baselines and FR/RF are reciprocal two-direction
schedules. This removes the trivial possibility that reverse prediction alone is
better than forward prediction. Across data types, the compact-specific reciprocal
signal is I_m(compact) - I_m(copy_matched_control).

This script first verifies exact fork-prefix sharing from branch summaries. It can
then combine (a) training epoch summaries and (b) optional cheap7 eval summaries
for the final checkpoints. Missing evals are reported as pending rather than
imputed.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
from typing import Any, Dict, List

ARMS = ["ff", "fr", "rr", "rf"]
BRANCH_ARM = {"ff": "forward_branch", "fr": "forward_branch", "rr": "reverse_branch", "rf": "reverse_branch"}


def load_json(p: pathlib.Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def load_jsonl(p: pathlib.Path) -> List[Dict[str, Any]]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def find_arm_dir(root: pathlib.Path, arm: str) -> pathlib.Path:
    return root / BRANCH_ARM[arm] / arm


def find_branch_summary(root: pathlib.Path, branch: str) -> Dict[str, Any] | None:
    p = root / branch / "branch_summary.json"
    return load_json(p) if p.exists() else None


def read_training(root: pathlib.Path) -> Dict[str, Any]:
    branches = {b: find_branch_summary(root, b) for b in ["forward_branch", "reverse_branch"]}
    manifests: Dict[str, Any] = {}
    logs: Dict[str, Any] = {}
    for arm in ARMS:
        arm_dir = find_arm_dir(root, arm)
        mp = arm_dir / "training_manifest.json"
        lp = arm_dir / "training_log.jsonl"
        manifests[arm] = load_json(mp) if mp.exists() else None
        logs[arm] = load_jsonl(lp)
    prefixes = {}
    for b, pname in [("forward_branch", "prefix_forward"), ("reverse_branch", "prefix_reverse")]:
        mp = root / b / pname / "training_manifest.json"
        lp = root / b / pname / "training_log.jsonl"
        prefixes[pname] = {"manifest": load_json(mp) if mp.exists() else None, "log": load_jsonl(lp)}

    integrity = {
        "forward_branch_summary_present": branches["forward_branch"] is not None,
        "reverse_branch_summary_present": branches["reverse_branch"] is not None,
        "all_arm_manifests_present": all(manifests[a] is not None for a in ARMS),
    }
    if branches["forward_branch"]:
        integrity["forward_branch_status"] = branches["forward_branch"].get("status")
        integrity["forward_prefix_shared"] = branches["forward_branch"].get("checks", {}).get("shared_prefix")
        integrity["forward_finals_diverge"] = branches["forward_branch"].get("checks", {}).get("finals_diverge")
    if branches["reverse_branch"]:
        integrity["reverse_branch_status"] = branches["reverse_branch"].get("status")
        integrity["reverse_prefix_shared"] = branches["reverse_branch"].get("checks", {}).get("shared_prefix")
        integrity["reverse_finals_diverge"] = branches["reverse_branch"].get("checks", {}).get("finals_diverge")

    epoch2 = {}
    for arm, m in manifests.items():
        if not m:
            epoch2[arm] = None
            continue
        es = m.get("epoch_summary", {})
        epoch2[arm] = {
            "direction": m.get("direction"),
            "global_step": m.get("global_step"),
            "model_hash": m.get("model_hash"),
            "loss": es.get("loss"),
            "copied_loss": es.get("copied_loss"),
            "noncopied_loss": es.get("noncopied_loss"),
            "n_targets": es.get("n_targets"),
            "n_copied": es.get("n_copied"),
            "n_noncopied": es.get("n_noncopied"),
        }

    def avg(vals):
        vals = [v for v in vals if v is not None]
        return statistics.mean(vals) if vals else None
    # For losses lower is better; report raw reciprocal-minus-oneway (negative means reciprocal lower loss).
    train_interactions = {}
    for key in ["loss", "copied_loss", "noncopied_loss"]:
        train_interactions[key] = {
            "reciprocal_avg_fr_rf": avg([epoch2[a][key] for a in ["fr", "rf"] if epoch2[a]]),
            "oneway_avg_ff_rr": avg([epoch2[a][key] for a in ["ff", "rr"] if epoch2[a]]),
        }
        if train_interactions[key]["reciprocal_avg_fr_rf"] is not None and train_interactions[key]["oneway_avg_ff_rr"] is not None:
            train_interactions[key]["reciprocal_minus_oneway"] = train_interactions[key]["reciprocal_avg_fr_rf"] - train_interactions[key]["oneway_avg_ff_rr"]

    return {"branches": branches, "prefixes": prefixes, "manifests": manifests, "epoch2": epoch2, "integrity": integrity, "training_log_records": {a: len(logs[a]) for a in ARMS}, "training_interactions": train_interactions}


def read_evals(eval_root: pathlib.Path | None) -> Dict[str, Any]:
    if eval_root is None:
        return {"status": "not_requested"}
    out: Dict[str, Any] = {"status": "pending_or_partial", "eval_root": str(eval_root), "arms": {}}
    for arm in ARMS:
        p = eval_root / arm / "cheap7_summary.json"
        if not p.exists():
            out["arms"][arm] = {"present": False, "path": str(p)}
        else:
            s = load_json(p)
            out["arms"][arm] = {"present": True, "path": str(p), "scores": s.get("scores"), "cheap7": s.get("cheap7"), "columns": s.get("cheap7_columns")}
    if all(out["arms"][a].get("present") for a in ARMS):
        out["status"] = "complete"
        metrics = set(["cheap7"])
        for arm in ARMS:
            cols = out["arms"][arm].get("columns") or {}
            metrics.update(cols.keys())
            scores = out["arms"][arm].get("scores") or {}
            metrics.update(scores.keys())
        interactions = {}
        for m in sorted(metrics):
            vals = {}
            for arm in ARMS:
                if m == "cheap7":
                    vals[arm] = out["arms"][arm].get("cheap7")
                else:
                    vals[arm] = (out["arms"][arm].get("columns") or {}).get(m)
                    if vals[arm] is None:
                        vals[arm] = (out["arms"][arm].get("scores") or {}).get(m)
            if all(vals[a] is not None for a in ARMS):
                one = 0.5 * (vals["ff"] + vals["rr"])
                rec = 0.5 * (vals["fr"] + vals["rf"])
                interactions[m] = {"values": vals, "oneway_avg_ff_rr": one, "reciprocal_avg_fr_rf": rec, "reciprocal_minus_oneway": rec - one, "direction_asymmetry_ff_minus_rr": vals["ff"] - vals["rr"], "order_asymmetry_fr_minus_rf": vals["fr"] - vals["rf"]}
        out["interactions"] = interactions
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root", required=True)
    ap.add_argument("--data_type", default="compact")
    ap.add_argument("--eval_root", default="")
    ap.add_argument("--output", default="")
    args = ap.parse_args()
    run_root = pathlib.Path(args.run_root)
    eval_root = pathlib.Path(args.eval_root) if args.eval_root else None
    training = read_training(run_root)
    evals = read_evals(eval_root)
    summary = {
        "status": "DIRECTIONAL_INTERACTION_READOUT",
        "data_type": args.data_type,
        "run_root": str(run_root),
        "eval_root": str(eval_root) if eval_root else None,
        "training": training,
        "eval": evals,
        "scientific_estimand": "I(data)=0.5*(FR+RF)-0.5*(FF+RR); compact-specific signal requires subtracting copy-matched extractive I(data), not just FR-FF.",
    }
    out = pathlib.Path(args.output) if args.output else run_root / "directional_interaction_readout.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    md = [f"# research directional interaction readout: {args.data_type}", "", f"Run root: `{run_root}`", f"Eval root: `{eval_root}`" if eval_root else "Eval root: pending", "", "## Integrity", ""]
    for k, v in training["integrity"].items():
        md.append(f"- {k}: {v}")
    md += ["", "## Training loss interactions (reciprocal minus one-way; lower loss is better)", ""]
    for k, v in training["training_interactions"].items():
        md.append(f"- {k}: {v}")
    if evals.get("status") == "complete":
        md += ["", "## Eval interactions (higher score is better)", ""]
        for k, v in evals.get("interactions", {}).items():
            md.append(f"- {k}: reciprocal_minus_oneway={v['reciprocal_minus_oneway']}; values={v['values']}")
    md += ["", f"JSON: `{out}`"]
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "output": str(out), "eval_status": evals.get("status"), "integrity": training["integrity"], "training_interactions": training["training_interactions"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
