#!/usr/bin/env python3
"""Analyze research full3 secondary retention replication."""
import json, collections
from pathlib import Path

WS = Path("experiments/archive/representation_and_objectives")
summary_path = WS / "data/secondary_retention_full3/secondary_retention_summary.json"
out = WS / "data/secondary_retention_full3_analysis"
out.mkdir(parents=True, exist_ok=True)

data = json.loads(summary_path.read_text())

# Extract per-seed per-arm results for familiar surface
arms = ["exposure", "Etrue_Rtrue", "Eflip_Rtrue", "Etrue_Rflip", "Eflip_Rflip",
        "Etrue_only", "Eflip_only", "Rtrue_only", "Rflip_only"]

# Collect base and arm data from per_seed
results = data.get("per_seed", [])
if not results:
    results = data.get("results", [])

# Parse from results structure
base_data = []
arm_data = collections.defaultdict(list)

for entry in results:
    seed = entry["seed"]
    base = entry.get("base", {})
    if base:
        base_data.append({"seed": seed, 
                          "train_acc": base.get("base_train_acc"),
                          "secondary_margin": base.get("before_trainTrain_secondary_margin")})
    
    for arm_entry in entry.get("arms", []):
        arm_name = arm_entry["arm"]
        arm_data[arm_name].append({
            "seed": seed,
            "update_train_acc": arm_entry.get("update_train_acc"),
            "base_anchor_after": arm_entry.get("base_anchor_acc_after"),
            "event": arm_entry.get("trainTrain_event"),
            "focal": arm_entry.get("trainTrain_focal"),
            "secondary": arm_entry.get("trainTrain_secondary"),
            "secondary_margin": arm_entry.get("trainTrain_secondary_margin"),
        })

# If per_seed format is different, try flat
if not base_data:
    # Try reading from stdout log
    log_path = WS / "tasks/s264_t19_tool1/stdout.log"
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            if not line.strip(): continue
            try:
                j = json.loads(line)
            except: continue
            if j.get("event") == "base_done":
                base_data.append({
                    "seed": j["seed"],
                    "train_acc": j.get("base_train_acc"),
                    "secondary_margin": j.get("before_trainTrain_secondary_margin")
                })
            elif j.get("event") == "arm_done":
                arm_data[j["arm"]].append({
                    "seed": j["seed"],
                    "update_train_acc": j.get("update_train_acc"),
                    "base_anchor_after": j.get("base_anchor_acc_after"),
                    "event": j.get("trainTrain_event"),
                    "focal": j.get("trainTrain_focal"),
                    "secondary": j.get("trainTrain_secondary"),
                    "secondary_margin": j.get("trainTrain_secondary_margin"),
                })

import numpy as np

md = ["# research full3 secondary retention analysis\n"]

md.append("## Base (before sparse update)\n")
for b in base_data:
    md.append(f"- seed {b['seed']}: train_acc={b['train_acc']}, secondary_margin={b['secondary_margin']:.3f}")

md.append("\n## Per-arm familiar-surface results (trainTrain_trainHyp)\n")
md.append("| Arm | Seed | Update fit | Base anchor | Event | Focal | Secondary | Sec margin |")
md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
for arm in arms:
    for d in sorted(arm_data[arm], key=lambda x: x["seed"]):
        md.append(f"| {arm} | {d['seed']} | {d['update_train_acc']:.3f} | "
                  f"{d['base_anchor_after']:.3f} | {d['event']:.3f} | {d['focal']:.3f} | "
                  f"{d['secondary']:.3f} | {d['secondary_margin']:.2f} |")

# Fit-filtered means (require update_train_acc >= 0.95)
md.append("\n## Fit-filtered means (update_train_acc ≥ 0.95)\n")
md.append("| Arm | N fit | Event mean±std | Focal mean±std | Secondary mean±std | Sec margin mean±std |")
md.append("|---|---:|---|---|---|---|")
for arm in arms:
    fit = [d for d in arm_data[arm] if d["update_train_acc"] >= 0.95]
    if not fit:
        md.append(f"| {arm} | 0 | — | — | — | — |")
        continue
    ev = [d["event"] for d in fit]
    fo = [d["focal"] for d in fit]
    se = [d["secondary"] for d in fit]
    sm = [d["secondary_margin"] for d in fit]
    md.append(f"| {arm} | {len(fit)} | {np.mean(ev):.3f}±{np.std(ev):.3f} | "
              f"{np.mean(fo):.3f}±{np.std(fo):.3f} | {np.mean(se):.3f}±{np.std(se):.3f} | "
              f"{np.mean(sm):.2f}±{np.std(sm):.2f} |")

# Key comparisons
md.append("\n## Key comparisons\n")
for arm in ["Eflip_Rtrue", "Etrue_Rflip", "Rflip_only", "Eflip_only"]:
    fit = [d for d in arm_data[arm] if d["update_train_acc"] >= 0.95]
    if fit:
        se = [d["secondary"] for d in fit]
        sm = [d["secondary_margin"] for d in fit]
        md.append(f"**{arm}** (N={len(fit)}): secondary acc {np.mean(se):.3f}±{np.std(se):.3f}, margin {np.mean(sm):.2f}±{np.std(sm):.2f}")

md.append("\n**Interpretation**: If Rflip_only inverts secondary → shared ranking coordinate.")
md.append("If Eflip_only collapses secondary → staged update interference without correct anchoring.\n")

text = "\n".join(md)
((out.parents[4] / 'research/documents/representation_and_objectives/data/secondary_retention_full3_analysis/full3_retention_analysis.md')).write_text(text, encoding="utf-8")
print(json.dumps({"status": "done", "md": str((out.parents[4] / 'research/documents/representation_and_objectives/data/secondary_retention_full3_analysis/full3_retention_analysis.md'))}))
