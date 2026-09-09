#!/usr/bin/env python3
"""research: Explanatory mechanism figure.

Shows acquired dense-state CE improvement against ordinary-function prediction cost
for all key endpoints, with changed-source response shown separately.

This figure illustrates why low KL alone is insufficient
and why clean represents a partial trade-off repair rather than universal preservation.
"""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json, pathlib, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
OUT = _public_path('experiments/archive/functional_learning/figures')
OUT.mkdir(parents=True, exist_ok=True)

# ---- Load data from completed probes ----
# Common-support endpoint probe
cs_path = _public_path('experiments/archive/functional_learning/data/common_support_endpoint_probe_with_densecorr/common_support_endpoint_probe.json')
cs = json.loads(cs_path.read_text())

# Evidence availability readout
ea_path = _public_path('experiments/archive/functional_learning/data/evidence_availability_readout_densecorr/evidence_availability_readout.json')
ea = json.loads(ea_path.read_text())

# Preservation drift
drift_path = _public_path('experiments/archive/functional_learning/data/preservation_drift_profile_densecorr/preservation_drift_profile.json')
drift = json.loads(drift_path.read_text())

# Temperature/source
ts_path = _public_path('experiments/archive/functional_learning/data/temperature_source_readout_densecorr_full/temperature_source_readout.json')
ts = json.loads(ts_path.read_text())

# ---- Extract numbers ----
endpoint_order = [
    ("coherent86_parent", "Parent\n(coherent86)", "#666666"),
    ("ordinary_inherited_wwm_seed62064", "Ordinary\ncontinuation", "#2196F3"),
    ("exact_ms_seed62064", "Acquisition-only\n(M,S)", "#FF5722"),
    ("clean_ms_kl_seed62064", "Clean\n(M,S)+KL s64", "#4CAF50"),
    ("clean_ms_kl_seed62065", "Clean\n(M,S)+KL s65", "#81C784"),
    ("densecorr_ms_kl_seed62064", "Dense-corr\npreservation", "#9C27B0"),
]

# Panel A: Dense-state CE gain vs parent KL on common support
dense_ce_gains = []
dense_kls = []
labels_a = []
colors_a = []
for name, label, color in endpoint_order:
    scores = cs["endpoint_scores"].get(name, {}).get("dense_common", {})
    if scores:
        dense_ce_gains.append(scores.get("ce_gain_vs_parent", 0))
        dense_kls.append(scores.get("kl_parent_to_model", 0))
        labels_a.append(label)
        colors_a.append(color)

# Panel B: Evidence-availability trade — correct-source vs wrong-source NLL delta
ea_correct = []
ea_wrong = []
ea_viewonly = []
labels_b = []
colors_b = []

# Map evidence names to model summary keys
ea_model_map = {
    "coherent86_parent": "coherent86",
    "ordinary_inherited_wwm_seed62064": "ordinary_inherited_wwm_seed62064",
    "exact_ms_seed62064": "densemask_sparselabel_seed62064",
    "clean_ms_kl_seed62064": "clean_pres_lambda1_eval_seed62064",
    "clean_ms_kl_seed62065": "clean_pres_lambda1_eval_seed62065",
    "densecorr_ms_kl_seed62064": "densecorr_pres_lambda1_seed62064",
}

for name, label, color in endpoint_order:
    ea_key = ea_model_map.get(name)
    if ea_key and ea_key in ea["model_summaries"]:
        ms = ea["model_summaries"][ea_key]["by_condition"]
        pc = ms["pair_correct_source"]["mean_delta_nll_vs_coherent86"]
        pw = ms["pair_wrong_source"]["mean_delta_nll_vs_coherent86"]
        vo = ms["view_only"]["mean_delta_nll_vs_coherent86"]
        ea_correct.append(pc)
        ea_wrong.append(pw)
        ea_viewonly.append(vo)
        labels_b.append(label)
        colors_b.append(color)

# Panel C: Ordinary-function drift (KL from parent on clean rendering)
drift_kls = []
drift_ces = []
labels_c = []
colors_c = []
for name, label, color in endpoint_order:
    # Map to drift model keys
    drift_map = {
        "coherent86_parent": "coherent86",
        "ordinary_inherited_wwm_seed62064": "ordinary_inherited_wwm_seed62064",
        "exact_ms_seed62064": "densemask_sparselabel_seed62064",
        "clean_ms_kl_seed62064": "clean_pres_lambda1_eval_seed62064",
        "clean_ms_kl_seed62065": "clean_pres_lambda1_eval_seed62065",
        "densecorr_ms_kl_seed62064": "densecorr_pres_lambda1_seed62064",
    }
    dk = drift_map.get(name)
    if dk and dk in drift["results"]:
        ms = drift["results"][dk]
        drift_kls.append(ms["kl_teacher_to_student_mean"])
        drift_ces.append(ms["delta_ce_student_minus_teacher"])
        labels_c.append(label)
        colors_c.append(color)

# ---- Create figure ----
fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
fig.suptitle("Stage III Mechanism Trade-offs: Acquisition, Retention, and Evidence Dependence",
             fontsize=14, fontweight="bold", y=1.02)

# Panel A: Dense-state CE gain vs Parent KL
ax = axes[0]
for i, (ce, kl, lab, col) in enumerate(zip(dense_ce_gains, dense_kls, labels_a, colors_a)):
    ax.scatter(kl, ce, s=180, c=col, zorder=5, edgecolors="black", linewidths=0.8)
    offset_x = 0.02 if kl < 0.3 else -0.02
    ha = "left" if kl < 0.3 else "right"
    ax.annotate(lab, (kl, ce), textcoords="offset points",
                xytext=(8 if kl < 0.3 else -8, 8), fontsize=7.5, ha=ha, va="bottom")
ax.set_xlabel("Parent KL on dense-common support", fontsize=10)
ax.set_ylabel("Dense-common CE gain vs parent", fontsize=10)
ax.set_title("A. Dense-state fit vs parent distance", fontsize=11, fontweight="bold")
ax.axhline(0, color="grey", linestyle="--", alpha=0.4, linewidth=0.8)
ax.axvline(0, color="grey", linestyle="--", alpha=0.4, linewidth=0.8)

# Panel B: Evidence-availability trade
ax = axes[1]
x_pos = np.arange(len(labels_b))
w = 0.25
bars_c = ax.bar(x_pos - w, ea_correct, w, color=[c for c in colors_b], alpha=0.7,
                label="Correct source", edgecolor="black", linewidth=0.5)
bars_w = ax.bar(x_pos, ea_wrong, w, color=[c for c in colors_b], alpha=0.4,
                label="Wrong source", edgecolor="black", linewidth=0.5, hatch="//")
bars_v = ax.bar(x_pos + w, ea_viewonly, w, color=[c for c in colors_b], alpha=0.3,
                label="View only", edgecolor="black", linewidth=0.5, hatch="\\\\")
ax.set_xticks(x_pos)
ax.set_xticklabels(labels_b, fontsize=7, rotation=30, ha="right")
ax.set_ylabel("ΔNLL vs coherent86 (negative = better)", fontsize=10)
ax.set_title("B. Evidence-availability trade", fontsize=11, fontweight="bold")
ax.axhline(0, color="grey", linestyle="--", alpha=0.4, linewidth=0.8)
# Custom legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="grey", alpha=0.7, label="Correct source"),
    Patch(facecolor="grey", alpha=0.4, hatch="//", label="Wrong source"),
    Patch(facecolor="grey", alpha=0.3, hatch="\\\\", label="View only"),
]
ax.legend(handles=legend_elements, fontsize=8, loc="upper right")

# Panel C: Ordinary-function drift
ax = axes[2]
for i, (kl, ce, lab, col) in enumerate(zip(drift_kls, drift_ces, labels_c, colors_c)):
    ax.scatter(kl, ce, s=180, c=col, zorder=5, edgecolors="black", linewidths=0.8)
    offset_x = 0.001 if kl < 0.015 else -0.001
    ha = "left" if kl < 0.015 else "right"
    ax.annotate(lab, (kl, ce), textcoords="offset points",
                xytext=(8 if kl < 0.015 else -8, 8), fontsize=7.5, ha=ha, va="bottom")
ax.set_xlabel("KL(parent || endpoint) on ordinary rendering", fontsize=10)
ax.set_ylabel("ΔCE vs parent on ordinary rendering", fontsize=10)
ax.set_title("C. Ordinary-function preservation", fontsize=11, fontweight="bold")
ax.axhline(0, color="grey", linestyle="--", alpha=0.4, linewidth=0.8)
ax.axvline(0, color="grey", linestyle="--", alpha=0.4, linewidth=0.8)

plt.tight_layout()
out_path = _public_path('experiments/archive/functional_learning/figures/mechanism_trade_figure.png')
fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
plt.close()

print(json.dumps({"figure": str(out_path.relative_to(ROOT)), "panels": ["A_dense_state_fit", "B_evidence_availability", "C_ordinary_function_drift"]}, indent=2), flush=True)
