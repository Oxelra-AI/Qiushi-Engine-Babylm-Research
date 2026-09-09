#!/usr/bin/env python3
"""research analysis: learning curves and source-decomposition figure."""
import json, numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

od = Path("data/corr_acquisition")
with open(od / "summary.json") as f:
    data = json.load(f)

seeds = data["summary"]["config"]["seeds"]
arms = ["all_corr","ident_full","ident_masked","neutral","wrong"]
colors = {"all_corr":"#2ca02c","ident_full":"#1f77b4","ident_masked":"#1f77b4",
          "neutral":"#ff7f0e","wrong":"#d62728"}
styles = {"all_corr":"-","ident_full":"-","ident_masked":"--","neutral":"-","wrong":"-"}
labels = {"all_corr":"500 corr (upper bound)","ident_full":"100 corr + 400 identity (full attn)",
          "ident_masked":"100 corr + 400 identity (masked attn)","neutral":"100 corr + 400 neutral",
          "wrong":"100 corr + 400 wrong"}

# Extract per-arm per-seed learning curves
arm_curves = {}
for arm in arms:
    arm_curves[arm] = {}
    for s in seeds:
        curves = data["all_results"][f"seed{s}"][arm]["curves"]
        arm_curves[arm][s] = curves

# Compute mean + std for key metrics at each eval epoch
epochs = [r["e"] for r in arm_curves[arms[0]][seeds[0]]]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Held-out correspondence gain (hcg) learning curves
ax = axes[0, 0]
for arm in arms:
    means, stds = [], []
    for ei, ep in enumerate(epochs):
        vals = [arm_curves[arm][s][ei]["hcg"] for s in seeds]
        means.append(np.mean(vals)); stds.append(np.std(vals))
    means, stds = np.array(means), np.array(stds)
    ax.plot(epochs, means, color=colors[arm], linestyle=styles[arm], label=labels[arm], linewidth=2)
    ax.fill_between(epochs, means-stds, means+stds, alpha=0.15, color=colors[arm])
ax.set_xlabel("Epoch"); ax.set_ylabel("Held-out corr gain (nats)")
ax.set_title("Correspondence acquisition: source benefit for held-out entities")
ax.legend(fontsize=7, loc='upper left')
ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
ax.grid(True, alpha=0.3)

# Panel 2: Held-out corr NLL with source (hc_s) - absolute performance
ax = axes[0, 1]
for arm in arms:
    means = [np.mean([arm_curves[arm][s][ei]["hc_s"] for s in seeds]) for ei in range(len(epochs))]
    ax.plot(epochs, means, color=colors[arm], linestyle=styles[arm], label=labels[arm], linewidth=2)
ax.set_xlabel("Epoch"); ax.set_ylabel("Held-out corr NLL (source present)")
ax.set_title("Absolute correspondence NLL (lower = better)")
ax.legend(fontsize=7, loc='upper right')
ax.grid(True, alpha=0.3)

# Panel 3: Held-out copy NLL with source (hcp_s) - copy tendency
ax = axes[1, 0]
for arm in arms:
    means = [np.mean([arm_curves[arm][s][ei]["hcp_s"] for s in seeds]) for ei in range(len(epochs))]
    ax.plot(epochs, means, color=colors[arm], linestyle=styles[arm], label=labels[arm], linewidth=2)
ax.set_xlabel("Epoch"); ax.set_ylabel("Held-out copy NLL (source present)")
ax.set_title("Copy tendency (lower = stronger copy)")
ax.legend(fontsize=7, loc='lower right')
ax.grid(True, alpha=0.3)

# Panel 4: Train corr NLL (tc_s) - training correspondence learning
ax = axes[1, 1]
for arm in arms:
    means = [np.mean([arm_curves[arm][s][ei]["tc_s"] for s in seeds]) for ei in range(len(epochs))]
    ax.plot(epochs, means, color=colors[arm], linestyle=styles[arm], label=labels[arm], linewidth=2)
ax.set_xlabel("Epoch"); ax.set_ylabel("Train corr NLL (source present)")
ax.set_title("Training entity correspondence learning")
ax.legend(fontsize=7, loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig("figures/corr_acquisition.png", dpi=150, bbox_inches='tight')
plt.close()

# Print key epoch-by-epoch contrasts
print("=== KEY CONTRASTS AT SELECTED EPOCHS ===")
for ep_idx, ep in enumerate(epochs):
    if ep in [10, 50, 100, 150, 200, 300]:
        hcg_id = np.mean([arm_curves["ident_full"][s][ep_idx]["hcg"] for s in seeds])
        hcg_im = np.mean([arm_curves["ident_masked"][s][ep_idx]["hcg"] for s in seeds])
        hcg_ne = np.mean([arm_curves["neutral"][s][ep_idx]["hcg"] for s in seeds])
        hcg_wr = np.mean([arm_curves["wrong"][s][ep_idx]["hcg"] for s in seeds])
        hcg_ac = np.mean([arm_curves["all_corr"][s][ep_idx]["hcg"] for s in seeds])
        print(f"  e{ep:3d}: all_corr={hcg_ac:+.3f}  ident_full={hcg_id:+.3f}  "
              f"ident_masked={hcg_im:+.3f}  neutral={hcg_ne:+.3f}  wrong={hcg_wr:+.3f}  "
              f"id-neut={hcg_id-hcg_ne:+.3f}  full-masked={hcg_id-hcg_im:+.3f}")

# Absolute hc_s comparison
print("\n=== ABSOLUTE CORRESPONDENCE NLL (hc_s) AT FINAL EPOCH ===")
for arm in arms:
    vals = [arm_curves[arm][s][-1]["hc_s"] for s in seeds]
    print(f"  {arm:15s}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")

print("\n=== COPY NLL (hcp_s) AT FINAL EPOCH ===")
for arm in arms:
    vals = [arm_curves[arm][s][-1]["hcp_s"] for s in seeds]
    print(f"  {arm:15s}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")

# Nosrc NLL for completeness
print("\n=== NOSRC CORRESPONDENCE NLL (hc_n) AT FINAL EPOCH ===")
for arm in arms:
    vals = [arm_curves[arm][s][-1]["hc_n"] for s in seeds]
    print(f"  {arm:15s}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")

print(f"\n✓ Figure saved: figures/corr_acquisition.png")
print(json.dumps({"status":"ANALYSIS_DONE"}))
