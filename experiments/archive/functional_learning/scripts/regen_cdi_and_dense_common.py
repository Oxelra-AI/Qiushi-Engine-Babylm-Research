"""Regenerate fig_cdi_selectivity.pdf and fig_dense_common.pdf from canonical data.

Fixes:
 (1) fig_cdi_selectivity: axis said 'vs Stage I' while data were referenced to a
     historical checkpoint (chck82). Now plots the whole-bank scale control and the
     preserved endpoint against Stage I explicitly, so the visual conclusion matches
     the text: preservation halves the vocabulary cost but does NOT return to Stage I.
 (2) fig_dense_common: single 'Parent KL Distance' panel conflated two different
     supports. Now shows ordinary-rendering KL and dense-common-support KL as
     separate grouped bars with explicit support names.

Canonical sources:
  experiments/archive/functional_learning/data/mechanism_trade_revision/mechanism_trade_revision_data.json
  experiments/archive/relation_learning/data/shrinkage_source_cdi_probe/combined_compact.csv
"""
import json, csv, pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(".")
OUT = ROOT / "data/external/figures"
MECH = ROOT / "experiments/archive/functional_learning/data/mechanism_trade_revision/mechanism_trade_revision_data.json"
SHRINK = ROOT / "experiments/archive/relation_learning/data/shrinkage_source_cdi_probe/combined_compact.csv"

C_ORD = "#7f8c8d"; C_MS = "#2980b9"; C_PRES = "#e67e22"; C_DC = "#c0392b"; C_SCALE = "#8e44ad"

mech = json.loads(MECH.read_text())
panel_a = {r["endpoint"]: r for r in mech["panel_a_joint_acquisition_retention"]}

rows = {}
with open(SHRINK) as f:
    for r in csv.DictReader(f):
        rows[r["model"]] = r

# Stage I (coherent86) position on the same chck82-anchored 96-word CDI bank.
# Established value: coherent86 is -0.0766447487 NLL relative to chck82.
STAGE_I_CDI_VS_CHCK82 = -0.0766447487
STAGE_I_RANK_VS_CHCK82 = -68.1096142

def cdi_vs_stage_i(model):
    return float(rows[model]["cdi_delta_nll_Tfit"]) - STAGE_I_CDI_VS_CHCK82

def spec_vs_chck82(model):
    return float(rows[model]["qwen_delta_spec_Tfit"])

pts = [
    ("Dense acquisition (M,M)\nscale 0.75", "dense64_scale0p75", C_MS),
    ("Whole-bank scale-down\nto 0.60 (null control)", "dense64_scale0p60", C_SCALE),
    ("Ordinary-state\npreservation", "clean_pres64", C_PRES),
]

# ---------------- Figure: CDI selectivity ----------------
fig, ax = plt.subplots(figsize=(7.2, 4.6))
for label, key, col in pts:
    x = spec_vs_chck82(key)
    y = cdi_vs_stage_i(key)
    ax.scatter([x], [y], s=150, color=col, edgecolor="white", zorder=5, linewidth=1.5)
    ax.annotate(f"{label}\n({x:+.3f}, {y:+.3f})", xy=(x, y), xytext=(0, -34),
                textcoords="offset points", ha="center", fontsize=7.6, color="#222")

ax.axhline(0, color="black", lw=1.0, ls="-")
ax.text(0.155, 0.004, "Stage I model (zero on this axis)", fontsize=8,
        ha="right", va="bottom", color="black")

ax.set_xlabel("Source-specificity change (nats, vs historical reference checkpoint)")
ax.set_ylabel("CDI vocabulary cost (NLL nats, vs Stage I model)\nhigher = worse than Stage I")
ax.set_title("Preservation is distribution-selective, not overall shrinkage\n"
             "(fixed 96-word CDI bank; single seed 62064)", fontweight="bold", fontsize=10)
ax.set_ylim(-0.02, 0.175)
ax.set_xlim(0.085, 0.185)
ax.grid(alpha=0.25, ls=":")

# annotate the comparison the text makes
ax.annotate("", xy=(spec_vs_chck82("clean_pres64"), cdi_vs_stage_i("clean_pres64")),
            xytext=(spec_vs_chck82("dense64_scale0p60"), cdi_vs_stage_i("dense64_scale0p60")),
            arrowprops=dict(arrowstyle="<->", color="#555", lw=1.2, ls="--"))
ax.text(0.105, 0.087, "matched source movement,\nbut ~0.049 nats better vocabulary",
        fontsize=7.8, color="#333", style="italic")

fig.tight_layout()
fig.savefig(OUT / "fig_cdi_selectivity.pdf", bbox_inches="tight")
plt.close(fig)
print("fig_cdi_selectivity.pdf regenerated (Stage I reference on y-axis)")

# ---------------- Figure: dense common support ----------------
order = [
    ("Ordinary\ncontinuation", "Ordinary", C_ORD),
    ("Exact\n(M,S)", "Acq. (M,S)", C_MS),
    ("Ordinary-state\npreservation", "Ordinary-state anchor s64", C_PRES),
    ("Dense-state\nanchoring control", "Dense-state anchor", C_DC),
]
labels = [o[0] for o in order]
colors = [o[2] for o in order]
ce = [panel_a[k]["dense_common_ce_gain_vs_parent"] for _, k, _ in order]
rk = [panel_a[k]["dense_common_rank_gain_vs_parent"] for _, k, _ in order]
kl_ord = [panel_a[k]["ordinary_rendering_kl_teacher_to_student"] for _, k, _ in order]
kl_dense = [panel_a[k]["dense_common_kl_parent_to_endpoint"] for _, k, _ in order]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4.3))

ax1.bar(range(4), ce, color=colors, edgecolor="white", width=0.62)
ax1.set_xticks(range(4)); ax1.set_xticklabels(labels, fontsize=7.5)
ax1.set_ylabel("True-label CE gain (nats)")
ax1.set_title("(a) Acquisition on the common support\n(familiar packed-pair rows)",
              fontweight="bold", fontsize=9.5)
ax1.axhline(0, color="black", lw=0.6)
for i, v in enumerate(ce):
    ax1.text(i, v + 0.02, f"{v:+.3f}", ha="center", fontsize=7.5)
ax1.set_ylim(-0.06, 0.88)

ax2.bar(range(4), rk, color=colors, edgecolor="white", width=0.62)
ax2.set_xticks(range(4)); ax2.set_xticklabels(labels, fontsize=7.5)
ax2.set_ylabel("True-label rank gain (positions)")
ax2.set_title("(b) Rank gain on the same fixed positions",
              fontweight="bold", fontsize=9.5)
ax2.axhline(0, color="black", lw=0.6)
for i, v in enumerate(rk):
    ax2.text(i, v + 3, f"{v:+.1f}", ha="center", fontsize=7.5)
ax2.set_ylim(-6, 152)

# Grouped bars: two DIFFERENT supports, log scale so both are readable
w = 0.36
xs = np.arange(4)
b1 = ax3.bar(xs - w/2, kl_ord, width=w, color="#34495e", edgecolor="white",
             label="ordinary-WWM rendering")
b2 = ax3.bar(xs + w/2, kl_dense, width=w, color="#95a5a6", edgecolor="white",
             label="dense common support")
ax3.set_yscale("log")
ax3.set_xticks(xs); ax3.set_xticklabels(labels, fontsize=7.5)
ax3.set_ylabel("Parent KL (nats, log scale)")
ax3.set_title("(c) Parent distance is support-specific\n(two different measurement supports)",
              fontweight="bold", fontsize=9.5)
for i in range(4):
    ax3.text(xs[i] - w/2, kl_ord[i]*1.35, f"{kl_ord[i]:.4f}", ha="center", fontsize=6.6, rotation=90)
    ax3.text(xs[i] + w/2, kl_dense[i]*1.35, f"{kl_dense[i]:.4f}", ha="center", fontsize=6.6, rotation=90)
ax3.legend(fontsize=7.5, loc="upper left")
ax3.set_ylim(2e-4, 8.0)

fig.suptitle("Common-support probe on 169 familiar packed-pair rows / 1,003 positions "
             "(state-specific fit, not held-out transfer)", fontweight="bold", y=1.02, fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "fig_dense_common.pdf", bbox_inches="tight")
plt.close(fig)
print("fig_dense_common.pdf regenerated (two KL supports separated)")

print("\nKey values for report text:")
print(f"  dense0.75 CDI vs Stage I : {cdi_vs_stage_i('dense64_scale0p75'):+.4f}")
print(f"  scale0.60 CDI vs Stage I : {cdi_vs_stage_i('dense64_scale0p60'):+.4f}")
print(f"  clean64   CDI vs Stage I : {cdi_vs_stage_i('clean_pres64'):+.4f}")
print(f"  scale0.60 - clean64 CDI  : {cdi_vs_stage_i('dense64_scale0p60')-cdi_vs_stage_i('clean_pres64'):+.4f}")
for lbl, k, _ in order:
    print(f"  {k:32s} ordKL={panel_a[k]['ordinary_rendering_kl_teacher_to_student']:.6f} "
          f"denseKL={panel_a[k]['dense_common_kl_parent_to_endpoint']:.6f}")
