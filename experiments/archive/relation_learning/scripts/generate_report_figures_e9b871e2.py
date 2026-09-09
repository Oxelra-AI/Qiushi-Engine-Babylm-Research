#!/usr/bin/env python3
"""research: Generate comprehensive high-quality figures for the Chinese research report.

All data sourced from verified JSON/CSV files. Outputs to figures/report/.
"""
import json, pathlib, sys
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────
ROOT = pathlib.Path("experiments/archive/relation_learning")
OUT = ROOT / "figures/report"
OUT.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

# Suppress font warnings
import warnings; warnings.filterwarnings("ignore", category=UserWarning)

# Use default sans-serif; Chinese captions go in LaTeX
plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
})

# ── Color palette ──────────────────────────────────────────────────────
C_COH = "#6C757D"   # coherent86 (grey)
C_ORD = "#0D6EFD"   # ordinary (blue)
C_MS  = "#FD7E14"   # (M,S) (orange)
C_CLN = "#198754"   # clean (green)
C_DC  = "#DC3545"   # dense-corruption control (red)
C_POS = "#198754"   # positive
C_NEG = "#DC3545"   # negative

# ══════════════════════════════════════════════════════════════════════
# Data from research / research authoritative ladder
# ══════════════════════════════════════════════════════════════════════
components = ["BLiMP","Supplement","EWoK","Entity","COMPS","SuperGLUE","GlobalPIQA","Reading","AoA"]

scores = {
    "coherent86": [68.51,63.64,50.02,28.32,52.05,68.946,38.565,8.165,0.0],
    "O s64":      [68.47,63.63,49.83,28.16,52.00,68.988,39.535,8.22,0.0],
    "O s65":      [68.52,63.62,49.81,28.09,52.00,69.273,39.535,8.195,0.0],
    "(M,S) s64":  [68.12,63.08,49.95,29.39,52.15,68.888,40.05,8.195,0.0],
    "(M,S) s65":  [68.09,63.09,49.82,29.29,52.14,68.935,40.05,8.195,0.0],
    "clean s64":  [68.26,63.28,49.82,29.40,52.16,69.048,40.05,8.20,0.0],
    "clean s65":  [68.22,63.29,49.72,29.45,52.14,69.021,40.05,8.195,0.0],
}
overalls = {
    "coherent86": 42.024, "O s64": 42.093, "O s65": 42.116,
    "(M,S) s64": 42.203, "(M,S) s65": 42.179,
    "clean s64": 42.246, "clean s65": 42.232,
}

# ══════════════════════════════════════════════════════════════════════
# Figure 1: Two-seed strategy ladder — Overall with increments
# ══════════════════════════════════════════════════════════════════════
def fig1_strategy_ladder():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    
    for ax, seed, sfx in zip(axes, ["s64","s65"], ["seed 62064","seed 62065"]):
        rungs = [f"coherent86", f"O {sfx}", f"(M,S) {sfx}", f"clean {sfx}"]
        labels = ["coherent86\n(v4 start)", f"Ordinary\ncontinuation", f"Dense mask +\nsparse label", f"+ Preservation\n(v5)"]
        colors = [C_COH, C_ORD, C_MS, C_CLN]
        vals = [overalls[f"coherent86"], overalls[f"O {seed}"], 
                overalls[f"(M,S) {seed}"], overalls[f"clean {seed}"]]
        
        bars = ax.barh(range(4), vals, color=colors, edgecolor="white", height=0.6, zorder=3)
        ax.set_xlim(41.95, 42.30)
        ax.set_yticks(range(4))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Overall Score")
        ax.set_title(sfx, fontweight="bold")
        
        # Add value labels and increments
        for i, (bar, v) in enumerate(zip(bars, vals)):
            ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=8, fontweight="bold")
            if i > 0:
                delta = v - vals[i-1]
                sign = "+" if delta > 0 else ""
                ax.annotate(f"{sign}{delta:.3f}", 
                    xy=(min(v, vals[i-1]) + abs(delta)/2, i - 0.35),
                    fontsize=7, ha="center", color="#333",
                    bbox=dict(boxstyle="round,pad=0.15", fc="lightyellow", ec="gray", lw=0.5))
    
    fig.suptitle("Strategy Ladder: Overall Score by Policy Stage (Two Seeds)", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_strategy_ladder.pdf")
    plt.close(fig)
    print(f"  Saved: fig_strategy_ladder.pdf")

# ══════════════════════════════════════════════════════════════════════
# Figure 2: Per-component deltas (O → clean) for both seeds
# ══════════════════════════════════════════════════════════════════════
def fig2_component_deltas():
    # Compute O→clean deltas for each component
    comps_show = ["BLiMP","Supplement","EWoK","Entity","COMPS","SuperGLUE","GlobalPIQA","Reading"]
    idx_map = {c: i for i, c in enumerate(components)}
    
    delta_s64 = [scores["clean s64"][idx_map[c]] - scores["O s64"][idx_map[c]] for c in comps_show]
    delta_s65 = [scores["clean s65"][idx_map[c]] - scores["O s65"][idx_map[c]] for c in comps_show]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(comps_show))
    w = 0.35
    
    bars1 = ax.bar(x - w/2, delta_s64, w, label="seed 62064", color=C_CLN, alpha=0.8, edgecolor="white")
    bars2 = ax.bar(x + w/2, delta_s65, w, label="seed 62065", color=C_CLN, alpha=0.5, edgecolor="white", hatch="//")
    
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(comps_show, rotation=30, ha="right")
    ax.set_ylabel("Score change (clean − matched ordinary)")
    ax.set_title("Per-Component Gains and Costs: Principle-Guided v5 vs Matched Ordinary Continuation", fontweight="bold")
    ax.legend()
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if abs(h) > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2, h + (0.02 if h >= 0 else -0.06),
                    f"{h:+.2f}", ha="center", fontsize=7, fontweight="bold")
    
    # Add noise floor annotation
    ax.annotate("O-pair noise floor: max deterministic\ncolumn delta = 0.07 (Entity)", 
        xy=(0.02, 0.02), xycoords="axes fraction", fontsize=7.5, color="#666",
        bbox=dict(boxstyle="round", fc="lightyellow", ec="gray", lw=0.5))
    
    fig.tight_layout()
    fig.savefig(OUT / "fig_component_deltas.pdf")
    plt.close(fig)
    print(f"  Saved: fig_component_deltas.pdf")

# ══════════════════════════════════════════════════════════════════════
# Figure 3: Three-rung increment decomposition (both seeds)
# ══════════════════════════════════════════════════════════════════════
def fig3_rung_increments():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    
    comps_det = ["BLiMP","Supplement","EWoK","Entity","COMPS","GlobalPIQA"]
    idx = {c: i for i, c in enumerate(components)}
    
    for ax, seed, title in zip(axes, ["s64","s65"], ["Seed 62064","Seed 62065"]):
        coh = scores["coherent86"]
        o = scores[f"O {seed}"]
        ms = scores[f"(M,S) {seed}"]
        cl = scores[f"clean {seed}"]
        
        # Three increments
        d_ord = [o[idx[c]] - coh[idx[c]] for c in comps_det]
        d_acq = [ms[idx[c]] - o[idx[c]] for c in comps_det]
        d_pres = [cl[idx[c]] - ms[idx[c]] for c in comps_det]
        
        x = np.arange(len(comps_det))
        w = 0.25
        
        ax.bar(x - w, d_ord, w, label="Ordinary continuation", color=C_ORD, edgecolor="white")
        ax.bar(x, d_acq, w, label="Dense acq. (M,S)", color=C_MS, edgecolor="white")
        ax.bar(x + w, d_pres, w, label="+ Preservation", color=C_CLN, edgecolor="white")
        
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(comps_det, rotation=35, ha="right")
        ax.set_title(title, fontweight="bold")
        if ax == axes[0]:
            ax.set_ylabel("Score increment")
            ax.legend(loc="upper left", fontsize=7.5)
    
    fig.suptitle("Component-Level Increments at Each Policy Stage", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_rung_increments.pdf")
    plt.close(fig)
    print(f"  Saved: fig_rung_increments.pdf")

# ══════════════════════════════════════════════════════════════════════
# Figure 4: Source specificity and dense-corruption prediction
# ══════════════════════════════════════════════════════════════════════
def fig4_source_specificity():
    # Data from verified source readout JSONs
    endpoints = ["Ordinary", "Exact (M,S)", "Clean pres.", "Dense-corr.\npres."]
    tfit = [0.000075, 0.146789, 0.100746, 0.003914]
    rank = [-1.24, 66.97, 42.70, 1.58]
    colors = [C_ORD, C_MS, C_CLN, C_DC]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    
    bars1 = ax1.bar(range(4), tfit, color=colors, edgecolor="white", width=0.6)
    ax1.set_xticks(range(4))
    ax1.set_xticklabels(endpoints, fontsize=8)
    ax1.set_ylabel("Source-specific Tfit (nats)")
    ax1.set_title("Qwen Source Specificity", fontweight="bold")
    for i, v in enumerate(tfit):
        ax1.text(i, v + 0.003, f"{v:.4f}", ha="center", fontsize=7.5, fontweight="bold")
    
    bars2 = ax2.bar(range(4), rank, color=colors, edgecolor="white", width=0.6)
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(endpoints, fontsize=8)
    ax2.set_ylabel("Rank improvement (positions)")
    ax2.set_title("Rank-Based Source Specificity", fontweight="bold")
    ax2.axhline(0, color="black", lw=0.5)
    for i, v in enumerate(rank):
        offset = 1.5 if v >= 0 else -3
        ax2.text(i, v + offset, f"{v:+.1f}", ha="center", fontsize=7.5, fontweight="bold")
    
    # Add prospective prediction annotation
    ax1.annotate("Prospective prediction confirmed:\ndense-corruption preservation\nerases source-responsive behavior",
        xy=(3, 0.003914), xytext=(2.2, 0.09),
        arrowprops=dict(arrowstyle="->", color=C_DC, lw=1.5),
        fontsize=7.5, color=C_DC, ha="center",
        bbox=dict(boxstyle="round", fc="mistyrose", ec=C_DC, lw=0.8))
    
    fig.suptitle("Source Specificity by Preservation Strategy (vs coherent86 parent)", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_source_specificity.pdf")
    plt.close(fig)
    print(f"  Saved: fig_source_specificity.pdf")

# ══════════════════════════════════════════════════════════════════════
# Figure 5: Dense common support probe
# ══════════════════════════════════════════════════════════════════════
def fig5_dense_common():
    endpoints = ["Ordinary", "Exact (M,S)", "Clean pres.", "Dense-corr.\npres."]
    ce_gain = [-0.005, 0.757, 0.682, 0.159]
    rank_gain = [0.7, 132.8, 128.3, 34.8]
    kl = [0.000551, 0.024486, 0.009411, 0.008036]
    colors = [C_ORD, C_MS, C_CLN, C_DC]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    
    ax1.bar(range(4), ce_gain, color=colors, edgecolor="white", width=0.6)
    ax1.set_xticks(range(4)); ax1.set_xticklabels(endpoints, fontsize=7.5)
    ax1.set_ylabel("CE gain (nats)"); ax1.set_title("Dense Common CE Gain", fontweight="bold")
    ax1.axhline(0, color="black", lw=0.5)
    for i, v in enumerate(ce_gain):
        ax1.text(i, v + 0.015, f"{v:+.3f}", ha="center", fontsize=7)
    
    ax2.bar(range(4), rank_gain, color=colors, edgecolor="white", width=0.6)
    ax2.set_xticks(range(4)); ax2.set_xticklabels(endpoints, fontsize=7.5)
    ax2.set_ylabel("Rank gain"); ax2.set_title("Dense Common Rank Gain", fontweight="bold")
    for i, v in enumerate(rank_gain):
        ax2.text(i, v + 2, f"{v:.1f}", ha="center", fontsize=7)
    
    ax3.bar(range(4), kl, color=colors, edgecolor="white", width=0.6)
    ax3.set_xticks(range(4)); ax3.set_xticklabels(endpoints, fontsize=7.5)
    ax3.set_ylabel("KL divergence (nats)"); ax3.set_title("Parent KL Distance", fontweight="bold")
    for i, v in enumerate(kl):
        ax3.text(i, v + 0.0005, f"{v:.4f}", ha="center", fontsize=7)
    
    fig.suptitle("Dense Common Support Probe: Mechanism Signatures", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_dense_common.pdf")
    plt.close(fig)
    print(f"  Saved: fig_dense_common.pdf")

# ══════════════════════════════════════════════════════════════════════
# Figure 6: CDI selectivity — dense vs clean vs scale null
# ══════════════════════════════════════════════════════════════════════
def fig6_cdi_selectivity():
    # Data from research CDI probe
    endpoints = ["coherent86\n(baseline)", "Dense s64", "Dense s65", "Clean s64", "Clean s65", "Scale 0.60\n(null)"]
    cdi_delta = [0, +0.052032, +0.055456, -0.014410, -0.009818, None]  # relative to coherent86
    # Scale null: from research, matched KL but worse CDI — specific value not available in simple form
    # Use relative: worse than clean, comparable to dense
    
    # Simplify: show coherent86, dense mean, clean mean
    labels = ["Dense acq.\n(mean 2 seeds)", "Clean pres.\n(mean 2 seeds)"]
    vals = [np.mean([0.052032, 0.055456]), np.mean([-0.014410, -0.009818])]
    colors_bar = [C_MS, C_CLN]
    
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar([0, 1], vals, color=colors_bar, edgecolor="white", width=0.5)
    ax.axhline(0, color="black", lw=0.8, label="coherent86 baseline")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_ylabel("CDI NLL change (nats, vs coherent86)")
    ax.set_title("Vocabulary Acquisition (CDI) Selectivity", fontweight="bold")
    
    for bar, v in zip(bars, vals):
        label = f"{v:+.4f}"
        y = v + 0.002 if v > 0 else v - 0.004
        ax.text(bar.get_x() + bar.get_width()/2, y, label, ha="center", fontsize=9, fontweight="bold")
    
    ax.annotate("Dense acquisition worsens\nvocabulary prediction", xy=(0, vals[0]),
        xytext=(0.5, vals[0]+0.02), fontsize=8, ha="center",
        arrowprops=dict(arrowstyle="->", color=C_MS))
    ax.annotate("Clean preservation recovers\nvocabulary prediction", xy=(1, vals[1]),
        xytext=(1.3, vals[1]-0.025), fontsize=8, ha="center",
        arrowprops=dict(arrowstyle="->", color=C_CLN))
    
    fig.tight_layout()
    fig.savefig(OUT / "fig_cdi_selectivity.pdf")
    plt.close(fig)
    print(f"  Saved: fig_cdi_selectivity.pdf")

# ══════════════════════════════════════════════════════════════════════
# Figure 7: Full nine-component heatmap comparison
# ══════════════════════════════════════════════════════════════════════
def fig7_full_heatmap():
    # Show all endpoints, all components as a heatmap of deltas from coherent86
    endpoints_show = ["O s64","O s65","(M,S) s64","(M,S) s65","clean s64","clean s65"]
    labels_show = ["O s64","O s65","(M,S) s64","(M,S) s65","v5 clean s64","v5 clean s65"]
    comps_8 = ["BLiMP","Supplement","EWoK","Entity","COMPS","SuperGLUE","GlobalPIQA","Reading"]
    idx = {c: i for i, c in enumerate(components)}
    
    coh = scores["coherent86"]
    data = np.array([[scores[ep][idx[c]] - coh[idx[c]] for c in comps_8] for ep in endpoints_show])
    
    fig, ax = plt.subplots(figsize=(10, 4))
    vmax = max(abs(data.min()), abs(data.max()))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)
    
    ax.set_xticks(range(len(comps_8)))
    ax.set_xticklabels(comps_8, rotation=35, ha="right")
    ax.set_yticks(range(len(labels_show)))
    ax.set_yticklabels(labels_show)
    ax.set_title("Component Changes Relative to coherent86 (v4)", fontweight="bold")
    
    # Add value text
    for i in range(len(endpoints_show)):
        for j in range(len(comps_8)):
            v = data[i, j]
            color = "white" if abs(v) > vmax * 0.6 else "black"
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7, color=color)
    
    plt.colorbar(im, ax=ax, label="Score change", shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_component_heatmap.pdf")
    plt.close(fig)
    print(f"  Saved: fig_component_heatmap.pdf")

# ══════════════════════════════════════════════════════════════════════
# Figure 8: Training data structure schematic
# ══════════════════════════════════════════════════════════════════════
def fig8_data_structure():
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    
    # Title
    ax.text(6, 3.7, "Packed Qwen-Pair Row: Training Data Structure", 
        ha="center", fontsize=12, fontweight="bold")
    
    # View 1 (source/original) - visible
    v1_box = FancyBboxPatch((0.3, 2.2), 4.8, 1.0, boxstyle="round,pad=0.1",
        facecolor="#D4EDDA", edgecolor="#198754", linewidth=2)
    ax.add_patch(v1_box)
    ax.text(2.7, 2.9, "View 1: Original text (fully visible)", 
        ha="center", fontsize=9, fontweight="bold", color="#155724")
    ax.text(2.7, 2.45, '"The cat sat on the warm mat by the fire"', 
        ha="center", fontsize=8, style="italic", color="#333")
    
    # View 2 (rewrite) - densely masked
    v2_box = FancyBboxPatch((5.5, 2.2), 6.2, 1.0, boxstyle="round,pad=0.1",
        facecolor="#FFF3CD", edgecolor="#FD7E14", linewidth=2)
    ax.add_patch(v2_box)
    ax.text(8.6, 2.9, "View 2: Rewrite (densely masked input)", 
        ha="center", fontsize=9, fontweight="bold", color="#856404")
    ax.text(8.6, 2.45, '"A [MASK] [MASK] on a [MASK] [MASK] near the [MASK]"', 
        ha="center", fontsize=8, style="italic", color="#333")
    
    # Arrow showing dense masking
    ax.annotate("Dense masking removes\nwithin-view completion clues",
        xy=(8.6, 2.15), xytext=(8.6, 1.3),
        fontsize=8, ha="center", color="#DC3545",
        arrowprops=dict(arrowstyle="->", color="#DC3545", lw=1.5))
    
    # Sparse labels indicator
    for x_pos, word, is_label in [(6.2, "cat", True), (7.0, "rested", True), 
                                    (8.2, "warm", False), (8.8, "rug", False), 
                                    (10.2, "fireplace", False)]:
        if is_label:
            ax.plot(x_pos, 1.7, "v", markersize=8, color="#198754")
            ax.text(x_pos, 1.45, f'"{word}"', ha="center", fontsize=7, color="#198754", fontweight="bold")
    
    ax.text(3.5, 1.5, "Sparse focus labels:", fontsize=8, fontweight="bold", color="#198754")
    ax.text(3.5, 1.15, "Only a small subset of masked\npositions receive prediction loss", 
        fontsize=7.5, color="#333")
    
    # Key insight box
    insight = FancyBboxPatch((0.5, 0.1), 11, 0.7, boxstyle="round,pad=0.1",
        facecolor="#E8F4FD", edgecolor="#0D6EFD", linewidth=1.5)
    ax.add_patch(insight)
    ax.text(6, 0.5, "Key: Because the rewrite's own context is hidden, the model can only predict the sparse\n"
        "target words by using the visible original text — learning the source-to-rewrite correspondence.",
        ha="center", fontsize=8.5, color="#0D6EFD")
    
    fig.savefig(OUT / "fig_data_structure.pdf")
    plt.close(fig)
    print(f"  Saved: fig_data_structure.pdf")

# ══════════════════════════════════════════════════════════════════════
# Figure 9: Three-stage knowledge flow diagram
# ══════════════════════════════════════════════════════════════════════
def fig9_three_stage_flow():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5)
    ax.axis("off")
    
    # Stage boxes
    stages = [
        (0.2, 3.0, 3.4, 1.5, "Stage I: Frontier Advancement",
         "• Aligned restatement pairing\n• Compact view + budget reinvestment\n• coherent86 endpoint (v4)\n• Exposure audit methodology",
         "#1F3A5F", "#D6E4F0"),
        (4.3, 3.0, 3.4, 1.5, "Stage II: Principle Discovery", 
         "• Relation-typed source readout\n• Same-window causal locality\n• Target-variable credit principle\n• Reach asymmetry",
         "#8A5A2B", "#FAE5D3"),
        (8.4, 3.0, 3.4, 1.5, "Stage III: Principle-Guided\nFrontier Advancement",
         "• Dense mask + sparse label\n• Ordinary-mask preservation\n• Prospective prediction test\n• v5 model (+0.222)",
         "#198754", "#D4EDDA"),
    ]
    
    for x, y, w, h, title, content, tc, bc in stages:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
            facecolor=bc, edgecolor=tc, linewidth=2)
        ax.add_patch(box)
        ax.text(x + w/2, y + h - 0.2, title, ha="center", fontsize=9, fontweight="bold", color=tc)
        ax.text(x + 0.2, y + h - 0.55, content, fontsize=7, va="top", color="#333")
    
    # Arrows between stages with labels
    ax.annotate("", xy=(4.2, 3.75), xytext=(3.7, 3.75),
        arrowprops=dict(arrowstyle="-|>", color="#333", lw=2))
    ax.text(3.95, 4.15, "Successes &\nfailures as\nevidence", ha="center", fontsize=7, color="#666")
    
    ax.annotate("", xy=(8.3, 3.75), xytext=(7.8, 3.75),
        arrowprops=dict(arrowstyle="-|>", color="#333", lw=2))
    ax.text(8.05, 4.15, "Principles\nguide new\ndesign", ha="center", fontsize=7, color="#666")
    
    # Key failures feeding into principles
    failures_y = 1.5
    failures = [
        (1.0, "Loader\naudit", "#DC3545"),
        (2.5, "Exposure\naudit", "#DC3545"),
        (5.0, "State-update\nfailure", "#DC3545"),
        (6.5, "Descriptor\nfailure", "#DC3545"),
        (8.0, "Format\nfailure", "#DC3545"),
    ]
    for fx, label, fc in failures:
        box = FancyBboxPatch((fx-0.5, failures_y-0.35), 1.2, 0.7, boxstyle="round,pad=0.05",
            facecolor="#FADBD8", edgecolor=fc, linewidth=1)
        ax.add_patch(box)
        ax.text(fx+0.1, failures_y, label, ha="center", fontsize=6.5, color=fc)
    
    ax.text(6, 0.8, "Failures narrow design space and sharpen principles", 
        ha="center", fontsize=9, style="italic", color="#666")
    
    # Return arrow
    ax.annotate("", xy=(1.9, 2.9), xytext=(9.8, 2.9),
        arrowprops=dict(arrowstyle="-|>", color="#198754", lw=1.5, 
                       connectionstyle="arc3,rad=0.4", linestyle="dashed"))
    ax.text(6, 2.5, "Closed loop: practice → principles → principle-guided practice",
        ha="center", fontsize=8, fontweight="bold", color="#198754")
    
    fig.savefig(OUT / "fig_three_stage_flow.pdf")
    plt.close(fig)
    print(f"  Saved: fig_three_stage_flow.pdf")

# ══════════════════════════════════════════════════════════════════════
# Figure 10: AoA timing result (30M)
# ══════════════════════════════════════════════════════════════════════
def fig10_aoa_timing():
    fig, ax = plt.subplots(figsize=(6, 4))
    
    arms = ["Control\n(random order)", "Schedule\n(child-AoA order)", "Enrichment\n(mask priority)"]
    r_vals = [-0.04955, 0.20713, 0.02390]
    p_vals = [0.407, 0.009, 0.695]
    colors_bar = [C_ORD, C_CLN, C_MS]
    
    bars = ax.bar(range(3), r_vals, color=colors_bar, edgecolor="white", width=0.5)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(3))
    ax.set_xticklabels(arms)
    ax.set_ylabel("Pearson r (model AoA vs child AoA)")
    ax.set_title("30M-Word Acquisition-Order Experiment", fontweight="bold")
    
    for i, (bar, r, p) in enumerate(zip(bars, r_vals, p_vals)):
        sig = "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        ax.text(bar.get_x() + bar.get_width()/2, r + (0.01 if r >= 0 else -0.03),
            f"r={r:.3f}\np={p:.3f} {sig}", ha="center", fontsize=7.5, fontweight="bold")
    
    ax.set_ylim(-0.15, 0.30)
    ax.annotate("Common-word shift r = +0.330 (p = 5.4×10⁻⁵)",
        xy=(0.5, 0.92), xycoords="axes fraction", fontsize=8, ha="center",
        bbox=dict(boxstyle="round", fc="lightyellow", ec="gray", lw=0.5))
    
    fig.tight_layout()
    fig.savefig(OUT / "fig_aoa_timing.pdf")
    plt.close(fig)
    print(f"  Saved: fig_aoa_timing.pdf")

# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating report figures...")
    fig1_strategy_ladder()
    fig2_component_deltas()
    fig3_rung_increments()
    fig4_source_specificity()
    fig5_dense_common()
    fig6_cdi_selectivity()
    fig7_full_heatmap()
    fig8_data_structure()
    fig9_three_stage_flow()
    fig10_aoa_timing()
    
    # Copy existing Stage II figures to the same directory
    import shutil
    fig_dir = ROOT / "deliverables/report/figures"
    stage2_figs = [
        "fig_target_class_crossing.pdf",
        "fig_entity_relevant_updates.pdf", 
        "fig_reach_asymmetry.pdf",
        "fig_split_attenuation.pdf",
        "fig_composition_dose.pdf",
    ]
    for fn in stage2_figs:
        src = fig_dir / fn
        if src.exists():
            shutil.copy2(src, OUT / fn)
            print(f"  Copied: {fn}")
    
    # Also copy mechanism_trade if present
    mt = fig_dir / "mechanism_trade.pdf"
    if mt.exists():
        shutil.copy2(mt, OUT / "mechanism_trade.pdf")
        print(f"  Copied: mechanism_trade.pdf")
    
    print(f"\nAll figures saved to {OUT}/")
    print(f"Total: {len(list(OUT.glob('*.pdf')))} PDF figures")
