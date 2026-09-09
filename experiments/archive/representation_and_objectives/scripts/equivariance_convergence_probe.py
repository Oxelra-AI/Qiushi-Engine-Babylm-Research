#!/usr/bin/env python3
"""research convergence/reliability probe for the repaired equivariance task.

The first research diagnostic proved that the truth table and gradient path are
live and showed an equivariant advantage at 12 epochs, but seed-to-seed behavior
was bimodal. This CPU-only probe reruns the same repaired benchmark with more
training budget and more seeds in a separate output directory, to distinguish a
stable representation-learning benefit from non-converged optimization luck.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

STUDY = Path("experiments/archive/representation_and_objectives")
MODULE_PATH = STUDY / "scripts/equivariance_truth_gradient_and_composition.py"
DEFAULT_OUT = STUDY / "data/equivariance_convergence_e30_s6"


def load_module():
    spec = importlib.util.spec_from_file_location("base", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {MODULE_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def learned_rate(per_seed: list[dict[str, Any]], surface: str, threshold: float = 0.95) -> float:
    vals = [r["eval"][surface]["acc"] for r in per_seed]
    return float(np.mean([v >= threshold for v in vals])) if vals else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--train-fams", type=int, default=24)
    ap.add_argument("--held-fams", type=int, default=12)
    ap.add_argument("--held-domain-fams", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lam", type=float, default=0.35)
    ap.add_argument("--seeds", type=str, default="254,255,256,257,258,259")
    ap.add_argument("--max-eq-pairs", type=int, default=40000)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mod = load_module()
    comp = mod.run_composition_experiment(args, out_dir)

    surfaces = ["train_pair_held_family", "held_pair_seen_family", "held_pair_held_family", "held_pair_held_domain", "semantic_support_held_family"]
    rates = {}
    for mode in ["standard", "equivariant"]:
        rates[mode] = {s: learned_rate(comp["per_seed"][mode], s) for s in surfaces}

    summary = {
        "status": "EQUIVARIANCE_CONVERGENCE_PROBE",
        "created_utc": now(),
        "purpose": "Separate stable equivariance benefit from 12-epoch seed/optimization nonconvergence in the repaired predicate-supported composition task.",
        "config": {"epochs": args.epochs, "lr": args.lr, "lambda": args.lam, "seeds": [int(x) for x in args.seeds.split(',') if x.strip()], "train_fams": args.train_fams, "held_fams": args.held_fams, "held_domain_fams": args.held_domain_fams},
        "composition_experiment": comp,
        "learned_rate_acc_ge_0p95": rates,
        "interpretation_boundary": "If standard reaches the same held-composition surfaces once optimized, the earlier advantage is an optimization-speed/reliability signal under limited budget, not proof of a different asymptotic role operation. If equivariant remains higher among converged seeds, it is a stronger candidate mechanism.",
    }
    write_json(out_dir / "convergence_probe_summary.json", summary)

    agg = comp["aggregates"]
    md = ["# research convergence probe", "", "Same repaired task as the first research diagnostic, but with more epochs/seeds to distinguish nonconverged optimization from a stable objective advantage.", ""]
    md.append("| Evaluation surface | Standard acc | Equivariant acc | Δ equiv-std | standard learned-rate | equiv learned-rate |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for s in surfaces:
        st = agg[s]["standard"]; eq = agg[s]["equivariant"]; d = agg[s]["delta_equiv_minus_standard"]
        md.append(f"| {s} | {st['mean']:.3f}±{st['std']:.3f} | {eq['mean']:.3f}±{eq['std']:.3f} | {d:+.3f} | {rates['standard'][s]:.2f} | {rates['equivariant'][s]:.2f} |")
    md.append("")
    md.append("## Seed-level held-pair-held-family accuracies")
    md.append("")
    md.append("| seed | standard | equivariant |")
    md.append("|---:|---:|---:|")
    seeds = summary["config"]["seeds"]
    for i, seed in enumerate(seeds):
        st = comp["per_seed"]["standard"][i]["eval"]["held_pair_held_family"]["acc"]
        eq = comp["per_seed"]["equivariant"][i]["eval"]["held_pair_held_family"]["acc"]
        md.append(f"| {seed} | {st:.3f} | {eq:.3f} |")
    md.append("")
    md.append(f"Summary JSON: `{out_dir / 'convergence_probe_summary.json'}`")
    (out_dir / "convergence_probe_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": summary["status"],
        "summary_json": str(out_dir / "convergence_probe_summary.json"),
        "held_pair_held_family_standard": agg["held_pair_held_family"]["standard"]["mean"],
        "held_pair_held_family_equivariant": agg["held_pair_held_family"]["equivariant"]["mean"],
        "held_pair_held_domain_standard": agg["held_pair_held_domain"]["standard"]["mean"],
        "held_pair_held_domain_equivariant": agg["held_pair_held_domain"]["equivariant"]["mean"],
        "learned_rates": rates,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
