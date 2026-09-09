#!/usr/bin/env python3
"""Compile cross-seed entity memory miniscreen results."""
import json
from pathlib import Path

results = {}
for seed in [43022, 43123]:
    for arm in ["vanilla", "shared_key", "independent_key"]:
        p = Path(f"experiments/archive/representation_and_objectives/training/data/{arm}_seed{seed}/result.json")
        results[(arm, seed)] = json.load(p.open())

print("=== CROSS-SEED ENTITY MEMORY MINISCREEN ===\n")
print(f"{'Metric':40s} | {'Vanilla':>12s} | {'Shared-Key':>12s} | {'Indep-Key':>12s}")
print("-" * 85)

for metric_name, split in [
    ("Held recomb acc (decisive)", "eval_held_recomb_template"),
    ("Train binding acc", "train_binding"),
    ("Held event order acc", "eval_held_order"),
]:
    for seed in [43022, 43123]:
        vals = []
        for arm in ["vanilla", "shared_key", "independent_key"]:
            r = results[(arm, seed)]
            b = r["baseline"].get(split, {})
            vals.append(f"{b.get('accuracy',0)*100:.1f}%" if b else "---")
        print(f"  seed{seed} {metric_name[:30]:>30s} | {vals[0]:>12s} | {vals[1]:>12s} | {vals[2]:>12s}")
    avgs = []
    for arm in ["vanilla", "shared_key", "independent_key"]:
        vs = [results[(arm, s)]["baseline"].get(split, {}).get("accuracy", 0) for s in [43022, 43123]]
        avgs.append(f"{sum(vs)/len(vs)*100:.1f}%")
    print(f"  {'MEAN':>37s} | {avgs[0]:>12s} | {avgs[1]:>12s} | {avgs[2]:>12s}")
    print()

print("\n=== WRITE-KEY PERMUTATION (causal intervention) ===\n")
for seed in [43022, 43123]:
    for split in ["eval_held_recomb_template", "train_binding"]:
        vals = []
        for arm in ["shared_key", "independent_key"]:
            r = results[(arm, seed)]
            base = r["baseline"].get(split, {}).get("accuracy", 0)
            perm = r["write_permutation"].get(split, {}).get("accuracy", 0)
            vals.append(f"{perm*100:.1f}% (d{(perm-base)*100:+.1f})")
        short = split.replace("eval_held_recomb_template", "held_recomb").replace("train_binding", "train_bind")
        print(f"  seed{seed} {short:>15s} |  {vals[0]:>22s} | {vals[1]:>22s}")

print("\n\n=== TRANSPORT (read/write attention) ===\n")
for seed in [43022, 43123]:
    for arm in ["shared_key", "independent_key"]:
        r = results[(arm, seed)]
        t = r["transport"]
        print(f"  seed{seed} {arm:>16s}: read_slot0={t['read_slot0_mean']:.3f}  read_H={t['read_entropy_mean']:.3f}  write_H={t['write_entropy_mean']:.3f}")

print("\n\n=== PARAMETERS ===")
for arm in ["vanilla", "shared_key", "independent_key"]:
    print(f"  {arm}: {results[(arm, 43022)]['parameters']:,}")

# Save JSON synthesis
synthesis = {}
for arm in ["vanilla", "shared_key", "independent_key"]:
    synthesis[arm] = {}
    for metric, split in [("held_recomb", "eval_held_recomb_template"),
                          ("train_binding", "train_binding"),
                          ("held_order", "eval_held_order")]:
        accs = [results[(arm, s)]["baseline"].get(split, {}).get("accuracy", 0) for s in [43022, 43123]]
        synthesis[arm][metric] = {"seed43022": accs[0], "seed43123": accs[1], "mean": sum(accs)/2}
    if arm != "vanilla":
        for metric, split in [("perm_held_recomb", "eval_held_recomb_template"),
                              ("perm_train_binding", "train_binding")]:
            accs = [results[(arm, s)]["write_permutation"].get(split, {}).get("accuracy", 0) for s in [43022, 43123]]
            base = [results[(arm, s)]["baseline"].get(split, {}).get("accuracy", 0) for s in [43022, 43123]]
            synthesis[arm][metric] = {
                "seed43022": accs[0], "seed43123": accs[1], "mean": sum(accs)/2,
                "delta_from_base_seed43022": accs[0]-base[0], "delta_from_base_seed43123": accs[1]-base[1],
            }
out_path = Path("experiments/archive/representation_and_objectives/data/miniscreen_synthesis.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(synthesis, indent=2) + "\n")
print(f"\nSaved synthesis to {out_path}")
