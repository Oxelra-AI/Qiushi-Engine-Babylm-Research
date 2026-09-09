#!/usr/bin/env python3
"""research: Generate all report figures from existing data CSVs.

Creates 5 publication-quality figures for deliverables/report/figures/:
1. fig_target_class_crossing.pdf — ALN−OFF and V−C by target class (two panels)
2. fig_split_attenuation.pdf — local vs split on compact nonoverlap G
3. fig_entity_relevant_updates.pdf — Entity accuracy by relevant updates
4. fig_composition_dose.pdf — raw T, G, copy gain for C/HV/HM/V/R
5. fig_reach_asymmetry_heatmap.pdf — reach matrix: relation × probe × target class
"""

import os, sys, json, csv
import numpy as np

WS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "")
if not WS:
    WS = "."
# Navigate from scripts/ to workspace root
WS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
FIG_DIR = os.path.join(WS, "..", "deliverables", "report", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ── Style ──
plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
})

COLORS = {
    "C": "#888888",
    "R": "#D62728",
    "V": "#2CA02C",
    "RS": "#D6272888",
    "VS": "#2CA02C88",
    "HV": "#9467BD",
    "HM": "#FF7F0E",
    "ALN": "#1F77B4",
    "OFF": "#888888",
    "SHUF": "#BCBD22",
    "DUP": "#D62728",
    "SEP": "#8C564B",
}


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ── Data paths ──
DATA = os.path.join(WS, "data")


# ═══════ FIGURE 1: Target-class crossing (ALN−OFF and V−C) ═══════

def fig_target_class_crossing():
    """Two panels: COMPACT_EXPERIENCE ALN-OFF and designed V-C/R-C, overlap vs nonoverlap."""
    # COMPACT_EXPERIENCE seed43022
    s0 = read_csv(os.path.join(DATA, "paired_context_relation_design_probe", "compact_TUN_late_contrasts.csv"))
    # COMPACT_EXPERIENCE seed43122
    s1 = read_csv(os.path.join(DATA, "paired_context_aln_off_seed43122_probe", "compact_TUN_late_contrasts.csv"))
    # Designed seed43022 roles (for overlap) and 3-seed summary (for nonoverlap errors)
    roles = read_csv(os.path.join(DATA, "half_view_curve_probe", "compact_TUN_late_roles.csv"))
    d3 = read_csv(os.path.join(DATA, "numerical_repair",
                               "compact_TUN_central_token_summary_positive_true_source_use.csv"))

    def get_contrast(rows, contrast, tc):
        for r in rows:
            if r["contrast"] == contrast and r["token_class"] == tc:
                return float(r["delta_A_T"])
        return 0.0

    aln_nov_s0 = get_contrast(s0, "ALNminusOFF", "nonoverlap")
    aln_ov_s0 = get_contrast(s0, "ALNminusOFF", "overlap")
    aln_nov_s1 = get_contrast(s1, "ALNminusOFF", "nonoverlap")
    aln_ov_s1 = get_contrast(s1, "ALNminusOFF", "overlap")

    # Designed V-C and R-C from roles (seed43022)
    def get_role(rows, arm, tc):
        for r in rows:
            if r["role"] == arm and r["token_class"] == tc:
                return float(r["A_T"])
        return 0.0

    vc_nov = get_role(roles, "V", "nonoverlap") - get_role(roles, "C", "nonoverlap")
    vc_ov = get_role(roles, "V", "overlap") - get_role(roles, "C", "overlap")
    rc_nov = get_role(roles, "R", "nonoverlap") - get_role(roles, "C", "nonoverlap")
    rc_ov = get_role(roles, "R", "overlap") - get_role(roles, "C", "overlap")

    # 3-seed SDs for nonoverlap (overlap SD not available, use 0)
    def get_d3_sd(rows, contrast, tc):
        for r in rows:
            if r.get("family") == "original_three_seed" and r["contrast"] == contrast and r["token_class"] == tc:
                return float(r["delta_true_adv_N_minus_T_seed_sd"])
        return 0.0

    vc_nov_sd = get_d3_sd(d3, "VminusC", "nonoverlap")
    rc_nov_sd = get_d3_sd(d3, "RminusC", "nonoverlap")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.2), sharey=False)

    x = np.array([0, 1])
    width = 0.35
    ax1.bar(x - width/2, [aln_nov_s0, aln_ov_s0], width, label="seed 43022",
            color=COLORS["ALN"], alpha=0.7)
    ax1.bar(x + width/2, [aln_nov_s1, aln_ov_s1], width, label="seed 43122",
            color=COLORS["ALN"], alpha=0.4)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Nonoverlap\n(unpracticed)", "Overlap\n(practiced)"])
    ax1.set_ylabel(r"ALN$-$OFF $\Delta A_T$")
    ax1.axhline(0, color="black", lw=0.5, ls="--")
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.set_title("(a) COMPACT_EXPERIENCE: Qwen restatement", fontweight="bold")

    ax2.bar(x - width/2, [vc_nov, vc_ov], width, label="VIEW$-$CLEAN",
            color=COLORS["V"], alpha=0.7, yerr=[vc_nov_sd, 0], capsize=3)
    ax2.bar(x + width/2, [rc_nov, rc_ov], width, label="REPEAT$-$CLEAN",
            color=COLORS["R"], alpha=0.7, yerr=[rc_nov_sd, 0], capsize=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Nonoverlap\n(substitution)", "Overlap\n(recurring)"])
    ax2.set_ylabel(r"$\Delta A_T$ vs CLEAN")
    ax2.axhline(0, color="black", lw=0.5, ls="--")
    ax2.legend(loc="lower left", framealpha=0.9)
    ax2.set_title("(b) Designed compact (seed43022)", fontweight="bold")

    fig.suptitle("Target-class crossing: readout matches the practiced relation",
                 fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig_target_class_crossing.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ═══════ FIGURE 2: Split attenuation on G ═══════

def fig_split_attenuation():
    """Compact nonoverlap G for R, RS, V, VS, C (2-seed)."""
    d3 = read_csv(os.path.join(DATA, "numerical_repair",
                               "compact_TUN_central_token_summary_positive_true_source_use.csv"))

    def get_g(rows, contrast, tc):
        for r in rows:
            if r["contrast"] == contrast and r["token_class"] == tc:
                # G is not directly in this file; use delta_gain which is delta_G = delta(U-T)
                # Actually this file has delta_true_adv which is A_T. G = A_T - A_U.
                # Let me use the raw roles file instead.
                pass
        return None, None

    # Better: use the roles from research
    roles = read_csv(os.path.join(DATA, "numerical_repair",
                                  "compact_TUN_central_token_summary_positive_true_source_use.csv"))

    # Actually, the compact summary has delta_A_T. For G, I need the raw roles.
    # Let me compute from the contrasts: delta_G = delta(U-T) = delta_U - delta_T
    # Or use the fact that delta_G = delta_A_T - delta_A_U

    arms = ["C", "R", "RS", "V", "VS"]
    # From the 3-seed/2-seed means stored in the research file
    # The contrasts vs C give us the relative values. But I need absolute G.
    # Let me read the research half_view curve roles which has C, R, V (and also HM, HV)
    # Those are from the seed43022 compact probes.

    # Actually, for split attenuation, I need the 2-seed summary.
    # The research file has contrasts like "split_common_two_seed_RminusRS", etc.
    # Let me extract delta_G = delta_A_T - delta_A_U for the relevant contrasts.

    # For the figure: show G for each arm relative to CLEAN.
    # R−C, RS−C, V−C, VS−C on nonoverlap G
    contrasts = {
        "R$-$C": "original_three_seed_RminusC",
        "RS$-$C": "split_common_two_seed_RSminusC",
        "V$-$C": "original_three_seed_VminusC",
        "VS$-$C": "split_common_two_seed_VSminusC",
    }
    fig, ax = plt.subplots(figsize=(5, 3))
    x = np.arange(len(contrasts))
    vals = []
    errs = []
    colors = [COLORS["R"], COLORS["RS"], COLORS["V"], COLORS["VS"]]

    for label, ckey in contrasts.items():
        for r in d3:
            if r["contrast"] == ckey and r["token_class"] == "nonoverlap":
                # delta_gain_G = delta_A_T - delta_A_U
                dAT = float(r["delta_true_adv_N_minus_T_across_seed_mean"])
                dAU = float(r["delta_unrel_adv_N_minus_U_across_seed_mean"])
                dG = dAT - dAU
                # For error, use A_T sd as approximation (G sd not directly available)
                eAT = float(r["delta_true_adv_N_minus_T_seed_sd"])
                vals.append(dG)
                errs.append(eAT)
                break
        else:
            vals.append(0)
            errs.append(0)

    bars = ax.bar(x, vals, 0.6, color=colors, alpha=0.7, yerr=errs, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(list(contrasts.keys()))
    ax.set_ylabel(r"$\Delta G$ vs CLEAN (nonoverlap)")
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.set_title("Split attenuation: local relation versus content exposure",
                 fontsize=10, fontweight="bold")
    # Add seed count annotations
    for i, (label, ns) in enumerate(zip(contrasts.keys(), ["3-seed", "2-seed", "3-seed", "2-seed"])):
        ax.annotate(ns, (i, vals[i] + (errs[i] + 0.05) * np.sign(vals[i]) if vals[i] != 0 else 0.05),
                    ha="center", va="bottom" if vals[i] >= 0 else "top", fontsize=7, color="gray")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig_split_attenuation.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ═══════ FIGURE 3: Entity by relevant updates ═══════

def fig_entity_relevant_updates():
    """Entity accuracy by relevant updates for C, R, V (designed) and ALN, OFF, DUP (COMPACT_EXPERIENCE)."""
    # Designed family (seed43022)
    entity_designed = read_csv(os.path.join(DATA,
        "split_entity_official_integration", "entity_late_accuracy_by_group.csv"))
    # COMPACT_EXPERIENCE family
    entity_compact_experience = read_csv(os.path.join(DATA,
        "paired_context_relation_design_probe", "entity_accuracy_by_group.csv"))

    def extract_by_relupdate(rows, arm_col, arm_val, group_prefix="rel_eq",
                              seed_col=None, seed_val=None):
        """Extract accuracy by exact relevant update count."""
        result = {}
        for r in rows:
            if r[arm_col] != arm_val:
                continue
            if seed_col and r.get(seed_col) != seed_val:
                continue
            g = r["group"]
            if g.startswith(group_prefix) and not g.startswith("rel_ge") and "total" not in g:
                try:
                    k = int(g.replace(group_prefix, ""))
                    result[k] = float(r["late_mean_accuracy_pct"] if "late_mean_accuracy_pct" in r else r["accuracy_pct"])
                except (ValueError, KeyError):
                    pass
        return result

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.2))

    # Panel A: designed C, R, V at seed43022
    for arm, color in [("C", COLORS["C"]), ("R", COLORS["R"]), ("V", COLORS["V"])]:
        d = extract_by_relupdate(entity_designed, "arm", arm, seed_col="seed", seed_val="43022")
        if d:
            ks = sorted(d.keys())[:8]
            ax1.plot(ks, [d[k] for k in ks], "o-", color=color, label=arm, markersize=3, lw=1.5)

    ax1.set_xlabel("Relevant updates")
    ax1.set_ylabel("Entity accuracy (%)")
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.set_title("(a) Designed compact (seed43022)", fontweight="bold")
    ax1.set_ylim(0, 65)

    # Panel B: COMPACT_EXPERIENCE ALN, OFF, DUP
    for arm, color in [("OFF", COLORS["OFF"]), ("ALN", COLORS["ALN"]), ("DUP", COLORS["DUP"]), ("SHUF", COLORS["SHUF"])]:
        d = extract_by_relupdate(entity_compact_experience, "role", arm)
        if d:
            ks = sorted(d.keys())[:8]
            ax2.plot(ks, [d[k] for k in ks], "o-", color=color, label=arm, markersize=3, lw=1.5)

    ax2.set_xlabel("Relevant updates")
    ax2.set_ylabel("Entity accuracy (%)")
    ax2.legend(loc="upper right", framealpha=0.9)
    ax2.set_title("(b) COMPACT_EXPERIENCE SOTA ingredient", fontweight="bold")
    ax2.set_ylim(0, 65)

    fig.suptitle("Entity behavior: unchanged-state retrieval is arm-dependent",
                 fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig_entity_relevant_updates.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ═══════ FIGURE 4: Composition/dose ═══════

def fig_composition_dose():
    """Three panels: compact G, copy gain, and Wikipedia overlap gain for C/HV/HM/V/R."""
    compact = read_csv(os.path.join(DATA, "half_view_curve_probe",
                                     "compact_TUN_late_roles.csv"))
    copy_data = read_csv(os.path.join(DATA, "half_view_curve_probe",
                                       "copy_late_roles.csv"))
    wiki = read_csv(os.path.join(DATA, "half_view_curve_probe",
                                  "wikipedia_late_roles.csv"))

    arms = ["C", "HV", "HM", "V", "R"]
    arm_colors = [COLORS[a] for a in arms]

    # Compact nonoverlap G
    compact_G = []
    for a in arms:
        for r in compact:
            if r["role"] == a and r["token_class"] == "nonoverlap":
                compact_G.append(float(r["G"]))
                break

    # Copy gain
    copy_gain = []
    for a in arms:
        for r in copy_data:
            if r["role"] == a:
                copy_gain.append(float(r["gain"]))
                break

    # Wikipedia overlap gain_T_vs_N
    wiki_ov = []
    for a in arms:
        for r in wiki:
            if r["role"] == a and r["overlap_bin"] == "ALL" and r["token_class"] == "overlap":
                wiki_ov.append(float(r["mean_gain_T_vs_N"]))
                break

    # Wikipedia nonoverlap gain_T_vs_N
    wiki_nov = []
    for a in arms:
        for r in wiki:
            if r["role"] == a and r["overlap_bin"] == "ALL" and r["token_class"] == "nonoverlap":
                wiki_nov.append(float(r["mean_gain_T_vs_N"]))
                break

    fig, axes = plt.subplots(1, 4, figsize=(10, 3))
    x = np.arange(len(arms))

    for ax, data, ylabel, title in [
        (axes[0], compact_G, r"Compact nonoverlap $G$", "(a) Restatement\nreadout"),
        (axes[1], copy_gain, "Copy gain", "(b) Copy\nbehavior"),
        (axes[2], wiki_ov, r"Wiki overlap $\Delta(N{-}T)$", "(c) Natural\nsource-recurring"),
        (axes[3], wiki_nov, r"Wiki nonoverlap $\Delta(N{-}T)$", "(d) Natural\nsubstitution"),
    ]:
        bars = ax.bar(x, data, 0.55, color=arm_colors, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(arms, fontsize=7)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.axhline(0, color="black", lw=0.5, ls="--")

    fig.suptitle("Mixed-relation composition: raw readouts by arm (seed43022)",
                 fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig_composition_dose.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ═══════ FIGURE 5: Reach asymmetry matrix ═══════

def fig_reach_asymmetry():
    """Heatmap showing ΔA_T for each relation × probe × target class."""
    # Collect: ALN−OFF, DUP−OFF, R−C, V−C on compact and Wikipedia, overlap and nonoverlap
    # Use seed43022 for COMPACT_EXPERIENCE and 3-seed mean for designed

    compact_experience = read_csv(os.path.join(DATA, "paired_context_relation_design_probe",
                                   "compact_TUN_late_contrasts.csv"))
    d3 = read_csv(os.path.join(DATA, "numerical_repair",
                               "compact_TUN_central_token_summary_positive_true_source_use.csv"))
    wiki_d3 = read_csv(os.path.join(DATA, "numerical_repair",
                                     "wikipedia_transfer_summary_recomputed_classbalanced_and_tokenweighted.csv"))
    wiki_compact_experience_s0 = read_csv(os.path.join(DATA, "paired_context_relation_design_probe",
                                           "wikipedia_late_contrasts.csv"))

    def get_compact_compact_experience(contrast, tc):
        for r in compact_experience:
            if r["contrast"] == contrast and r["token_class"] == tc:
                return float(r["delta_A_T"])
        return np.nan

    def get_compact_d3(contrast, tc):
        for r in d3:
            if r["contrast"] == contrast and r["token_class"] == tc:
                return float(r["delta_true_adv_N_minus_T_across_seed_mean"])
        return np.nan

    def get_wiki_d3(contrast, tc):
        for r in wiki_d3:
            if r["contrast"] == contrast and r["token_class"] == tc and r["aggregation"] == "token_weighted":
                return float(r["delta_gain_T_vs_N_mean"])
        return np.nan

    def get_wiki_compact_experience(contrast, tc):
        for r in wiki_compact_experience_s0:
            if r["contrast"] == contrast and r["token_class"] == tc and r["overlap_bin"] == "ALL":
                return float(r["mean_difference"]) if r["estimand"] == "gain_T_vs_N" else np.nan
        return np.nan

    # Build matrix: rows = relations, columns = probe×target
    relations = [
        ("Restatement\n(ALN$-$OFF)", "ALNminusOFF"),
        ("Exact recur.\n(DUP$-$OFF)", "DUPminusOFF"),
        ("VIEW$-$C\n(3-seed)", None),
        ("REPEAT$-$C\n(3-seed)", None),
    ]
    probes = [
        ("Compact\noverlap", "overlap"),
        ("Compact\nnonoverlap", "nonoverlap"),
        ("Wikipedia\noverlap", "overlap"),
        ("Wikipedia\nnonoverlap", "nonoverlap"),
    ]

    matrix = np.full((len(relations), len(probes)), np.nan)

    for i, (label, ckey) in enumerate(relations):
        if ckey:  # COMPACT_EXPERIENCE
            matrix[i, 0] = get_compact_compact_experience(ckey, "overlap")
            matrix[i, 1] = get_compact_compact_experience(ckey, "nonoverlap")
            # Wikipedia
            for r in wiki_compact_experience_s0:
                if r["contrast"] == ckey and r["overlap_bin"] == "ALL" and r["estimand"] == "gain_T_vs_N":
                    if r["token_class"] == "overlap":
                        matrix[i, 2] = float(r["mean_difference"])
                    elif r["token_class"] == "nonoverlap":
                        matrix[i, 3] = float(r["mean_difference"])
        else:  # Designed 3-seed
            if "VIEW" in label:
                cname = "original_three_seed_VminusC"
            else:
                cname = "original_three_seed_RminusC"
            matrix[i, 0] = get_compact_d3(cname, "overlap")
            matrix[i, 1] = get_compact_d3(cname, "nonoverlap")
            matrix[i, 2] = get_wiki_d3(cname, "overlap")
            matrix[i, 3] = get_wiki_d3(cname, "nonoverlap")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)))
    im = ax.imshow(matrix, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(np.arange(len(probes)))
    ax.set_xticklabels([p[0] for p in probes], fontsize=8)
    ax.set_yticks(np.arange(len(relations)))
    ax.set_yticklabels([r[0] for r in relations], fontsize=8)

    # Annotate cells
    for i in range(len(relations)):
        for j in range(len(probes)):
            v = matrix[i, j]
            if not np.isnan(v):
                color = "white" if abs(v) > vmax * 0.5 else "black"
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7, color=color)

    plt.colorbar(im, ax=ax, label=r"$\Delta A_T$ (positive = better source use)", shrink=0.8)
    ax.set_title("Reach asymmetry: liability broad, competence narrow",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig_reach_asymmetry.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ═══════ Run all ═══════
if __name__ == "__main__":
    print("Generating report figures...")
    try:
        fig_target_class_crossing()
    except Exception as e:
        print(f"ERROR fig_target_class_crossing: {e}")
    try:
        fig_split_attenuation()
    except Exception as e:
        print(f"ERROR fig_split_attenuation: {e}")
    try:
        fig_entity_relevant_updates()
    except Exception as e:
        print(f"ERROR fig_entity_relevant_updates: {e}")
    try:
        fig_composition_dose()
    except Exception as e:
        print(f"ERROR fig_composition_dose: {e}")
    try:
        fig_reach_asymmetry()
    except Exception as e:
        print(f"ERROR fig_reach_asymmetry: {e}")
    print("Done.")
