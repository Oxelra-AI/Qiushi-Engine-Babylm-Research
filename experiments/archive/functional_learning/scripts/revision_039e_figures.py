#!/usr/bin/env python3
"""Step039e: Figures for relation-first scoring results."""

import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = pathlib.Path("experiments/archive/functional_learning/figures")
OUT.mkdir(parents=True, exist_ok=True)

metrics = [json.loads(l) for l in open(
    "experiments/archive/functional_learning/data/relation_first_scored/pair_metrics.jsonl")]
valid = [m for m in metrics if not m.get("has_nan")]

# ─── Figure 1: alpha-beta scatter ───
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

betas = [m["beta"] for m in valid]
alphas = [m["alpha"] for m in valid]
splits = [m["split"] for m in valid]
relations = [m["relation"] for m in valid]

# Color by split
train = [(a, b) for a, b, s in zip(alphas, betas, splits) if s == "train"]
held = [(a, b) for a, b, s in zip(alphas, betas, splits) if s == "held"]

ax.scatter([x[0] for x in train], [x[1] for x in train], 
           alpha=0.5, s=30, c="steelblue", label=f"train (n={len(train)})")
ax.scatter([x[0] for x in held], [x[1] for x in held], 
           alpha=0.7, s=50, c="firebrick", marker="^", label=f"held (n={len(held)})")

# Joint-correct region: beta > |alpha|
a_range = np.linspace(-3, 12, 200)
ax.fill_between(a_range, np.abs(a_range), 5, alpha=0.08, color="green")
ax.plot(a_range, np.abs(a_range), "g--", alpha=0.3, linewidth=1, label="γ=0 (joint correct boundary)")

ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")

ax.set_xlabel("α = (U−R)/2  (shared new-value preference)", fontsize=11)
ax.set_ylabel("β = (U+R)/2  (recipient dependence)", fontsize=11)
ax.set_title("Coherent86 baseline on 120 relation-first pairs\n"
             "β ≈ 0: zero entity-conditioned tracking", fontsize=12)
ax.legend(fontsize=9, loc="upper left")
ax.set_xlim(-4, 12)
ax.set_ylim(-2, 2)

fig.tight_layout()
fig.savefig(OUT / "relation_first_alpha_beta.png", dpi=150)
plt.close()

# ─── Figure 2: U/R distribution ───
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

Us = [m["U"] for m in valid]
Rs = [m["R"] for m in valid]

axes[0].hist(Us, bins=30, alpha=0.7, color="steelblue", edgecolor="white")
axes[0].axvline(0, color="red", linewidth=1, linestyle="--")
axes[0].set_xlabel("U (UPDATE margin: new > source)", fontsize=10)
axes[0].set_ylabel("Count", fontsize=10)
axes[0].set_title(f"UPDATE margins\nPositive (correct): {sum(1 for u in Us if u>0)}/120")

axes[1].hist(Rs, bins=30, alpha=0.7, color="firebrick", edgecolor="white")
axes[1].axvline(0, color="red", linewidth=1, linestyle="--")
axes[1].set_xlabel("R (RETAIN margin: source > new)", fontsize=10)
axes[1].set_ylabel("Count", fontsize=10)
axes[1].set_title(f"RETAIN margins\nPositive (correct): {sum(1 for r in Rs if r>0)}/120")

fig.suptitle("Coherent86 baseline: shared new-value preference\n"
             "U and R are nearly equal/opposite → β≈0, large |α|", fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "relation_first_U_R_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

# Check sizes
for name in ["relation_first_alpha_beta.png", "relation_first_U_R_distribution.png"]:
    p = OUT / name
    print(f"{p}: {p.stat().st_size} bytes")

print("\nKey statistics:")
print(f"  mean beta = {np.mean(betas):.4f}")
print(f"  mean |alpha| = {np.mean(np.abs(alphas)):.4f}")
print(f"  mean gamma = {np.mean([b - abs(a) for b, a in zip(betas, alphas)]):.4f}")
print(f"  Joint correct: {sum(1 for m in valid if m['joint'])}/120")
