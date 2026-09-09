#!/usr/bin/env python3
"""research: Corrected report figures from existing data CSVs.

Fixes from independent review:
1. fig_entity_relevant_updates: use rel_updates_N groups (was using rel_eq only→1 point)
2. fig_split_attenuation: fix contrast key lookup (family+contrast separate columns)
3. fig_reach_asymmetry: fix designed-family key lookup (same bug)
"""

import os, sys, csv
import numpy as np

WS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
FIG_DIR = os.path.join(WS, "figures", "report")
os.makedirs(FIG_DIR, exist_ok=True)
DATA = os.path.join(WS, "data")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 11,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "serif",
})

COLORS = {
    "C": "#888888", "R": "#D62728", "V": "#2CA02C",
    "RS": "#D6272888", "VS": "#2CA02C88",
    "HV": "#9467BD", "HM": "#FF7F0E",
    "ALN": "#1F77B4", "OFF": "#888888", "SHUF": "#BCBD22",
    "DUP": "#D62728", "SEP": "#8C564B",
}

def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ═══════ FIGURE 1: Target-class crossing (ALN−OFF and V−C) ═══════

def fig_target_class_crossing():
    """Two panels: COMPACT_EXPERIENCE ALN-OFF and designed V-C/R-C, overlap vs nonoverlap."""
    s0 = read_csv(os.path.join(DATA, "paired_context_relation_design_probe", "compact_TUN_late_contrasts.csv"))
    s1 = read_csv(os.path.join(DATA, "paired_context_aln_off_seed43122_probe", "compact_TUN_late_contrasts.csv"))
    roles = read_csv(os.path.join(DATA, "half_view_curve_probe", "compact_TUN_late_roles.csv"))
    # 3-seed summary for error bars
    d3 = read_csv(os.path.join(DATA, "numerical_repair",
                               "compact_TUN_central_token_summary_positive_true_source_use.csv"))

    def get_contrast(rows, contrast, tc):
        for r in rows:
            if r.get("contrast") == contrast and r.get("token_class") == tc:
                return float(r.get("delta_A_T", 0))
        return 0.0

    aln_nov_s0 = get_contrast(s0, "ALNminusOFF", "nonoverlap")
    aln_ov_s0 = get_contrast(s0, "ALNminusOFF", "overlap")
    aln_nov_s1 = get_contrast(s1, "ALNminusOFF", "nonoverlap")
    aln_ov_s1 = get_contrast(s1, "ALNminusOFF", "overlap")

    def get_role(rows, arm, tc):
        for r in rows:
            if r.get("role") == arm and r.get("token_class") == tc:
                return float(r.get("A_T", 0))
        return 0.0

    vc_nov = get_role(roles, "V", "nonoverlap") - get_role(roles, "C", "nonoverlap")
    vc_ov = get_role(roles, "V", "overlap") - get_role(roles, "C", "overlap")
    rc_nov = get_role(roles, "R", "nonoverlap") - get_role(roles, "C", "nonoverlap")
    rc_ov = get_role(roles, "R", "overlap") - get_role(roles, "C", "overlap")

    # 3-seed SDs: match on BOTH family and contrast columns
    def get_d3_sd(rows, family, contrast, tc):
        for r in rows:
            if (r.get("family") == family and r.get("contrast") == contrast
                    and r.get("token_class") == tc):
                return float(r.get("delta_true_adv_N_minus_T_seed_sd", 0))
        return 0.0

    vc_nov_sd = get_d3_sd(d3, "original_three_seed", "VminusC", "nonoverlap")
    rc_nov_sd = get_d3_sd(d3, "original_three_seed", "RminusC", "nonoverlap")

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
    ax1.set_title("(a) Aligned restatement\n(2 seeds)", fontweight="bold")

    ax2.bar(x - width/2, [vc_nov, vc_ov], width, label="VIEW$-$CLEAN",
            color=COLORS["V"], alpha=0.7, yerr=[vc_nov_sd, 0], capsize=3)
    ax2.bar(x + width/2, [rc_nov, rc_ov], width, label="REPEAT$-$CLEAN",
            color=COLORS["R"], alpha=0.7, yerr=[rc_nov_sd, 0], capsize=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Nonoverlap\n(substitution)", "Overlap\n(recurring)"])
    ax2.set_ylabel(r"$\Delta A_T$ vs CLEAN (seed SD)")
    ax2.axhline(0, color="black", lw=0.5, ls="--")
    ax2.legend(loc="lower left", framealpha=0.9)
    ax2.set_title("(b) Designed compact\n(seed43022; bars = 3-seed SD)", fontweight="bold")

    fig.suptitle("Target-class crossing: readout matches the practiced relation",
                 fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig_target_class_crossing.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ═══════ FIGURE 2: Entity by relevant updates ═══════

def fig_entity_relevant_updates():
    """Entity accuracy by relevant-update bins for designed and COMPACT_EXPERIENCE arms."""
    # Designed family (seeds 43022)
    entity_designed = read_csv(os.path.join(DATA,
        "split_entity_official_integration", "entity_late_accuracy_by_group.csv"))
    # COMPACT_EXPERIENCE family — extract raw accuracy from contrasts file
    entity_compact_experience = read_csv(os.path.join(DATA,
        "paired_context_relation_design_probe", "entity_contrasts_by_group.csv"))

    def extract_designed(rows, arm, seed="43022", bins=range(6)):
        """Extract accuracy from rel_updates_N groups."""
        result = {}
        for r in rows:
            if r.get("arm") != arm or r.get("seed") != seed:
                continue
            g = r.get("group", "")
            if g.startswith("rel_updates_"):
                try:
                    k = int(g.replace("rel_updates_", ""))
                    if k in bins:
                        result[k] = float(r["late_mean_accuracy_pct"])
                except (ValueError, KeyError):
                    pass
        return result

    def extract_compact_experience(rows, arm, bins=range(6)):
        """Extract accuracy from COMPACT_EXPERIENCE contrasts using acc_a/acc_b columns."""
        result = {}
        for r in rows:
            g = r.get("group", "")
            if not g.startswith("rel_updates_"):
                continue
            try:
                k = int(g.replace("rel_updates_", ""))
            except ValueError:
                continue
            if k not in bins:
                continue
            # Check if this arm appears as role_a or role_b
            if r.get("role_a") == arm:
                result[k] = float(r["acc_a"])
            elif r.get("role_b") == arm:
                result[k] = float(r["acc_b"])
        return result

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.2))

    # Panel A: designed C, R, V at seed43022
    for arm, color, label in [("C", COLORS["C"], "CLEAN"), ("R", COLORS["R"], "REPEAT"),
                               ("V", COLORS["V"], "VIEW")]:
        d = extract_designed(entity_designed, arm, "43022")
        if d:
            ks = sorted(d.keys())
            ax1.plot(ks, [d[k] for k in ks], "o-", color=color, label=label,
                     markersize=4, lw=1.5)

    ax1.set_xlabel("Relevant queried-entity updates")
    ax1.set_ylabel("Entity accuracy (%)")
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.set_title("(a) Designed compact\n(seed 43022, 1 seed)", fontweight="bold")
    ax1.set_ylim(0, 65)
    ax1.set_xticks(range(6))

    # Panel B: COMPACT_EXPERIENCE ALN, OFF, DUP, SHUF
    for arm, color in [("OFF", COLORS["OFF"]), ("ALN", COLORS["ALN"]),
                        ("DUP", COLORS["DUP"]), ("SHUF", COLORS["SHUF"])]:
        d = extract_compact_experience(entity_compact_experience, arm)
        if d:
            ks = sorted(d.keys())
            ax2.plot(ks, [d[k] for k in ks], "o-", color=color, label=arm,
                     markersize=4, lw=1.5)

    ax2.set_xlabel("Relevant queried-entity updates")
    ax2.set_ylabel("Entity accuracy (%)")
    ax2.legend(loc="upper right", framealpha=0.9)
    ax2.set_title("(b) Paired-text experiment\n(seed 43022, 1 seed)", fontweight="bold")
    ax2.set_ylim(0, 65)
    ax2.set_xticks(range(6))

    fig.suptitle("Entity behavior by relevant updates",
                 fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig_entity_relevant_updates.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ═══════ FIGURE 3: Split attenuation ═══════

def fig_split_attenuation():
    """Compact nonoverlap ΔG for R-C, RS-C, V-C, VS-C."""
    d3 = read_csv(os.path.join(DATA, "numerical_repair",
                               "compact_TUN_central_token_summary_positive_true_source_use.csv"))

    # Correct lookup: match family AND contrast columns separately
    contrasts = [
        ("R$-$C\n(3-seed)", "original_three_seed", "RminusC"),
        ("RS$-$C\n(2-seed)", "split_common_two_seed", "RSminusC"),
        ("V$-$C\n(3-seed)", "original_three_seed", "VminusC"),
        ("VS$-$C\n(2-seed)", "split_common_two_seed", "VSminusC"),
    ]

    vals, errs, colors = [], [], []
    color_map = {"R": COLORS["R"], "RS": COLORS["RS"], "V": COLORS["V"], "VS": COLORS["VS"]}

    for label, family, contrast in contrasts:
        found = False
        for r in d3:
            if (r.get("family") == family and r.get("contrast") == contrast
                    and r.get("token_class") == "nonoverlap"):
                dAT = float(r["delta_true_adv_N_minus_T_across_seed_mean"])
                dAU = float(r["delta_unrel_adv_N_minus_U_across_seed_mean"])
                dG = dAT - dAU  # ΔG = Δ(U-T) = ΔA_T - ΔA_U
                eAT = float(r["delta_true_adv_N_minus_T_seed_sd"])
                vals.append(dG)
                errs.append(eAT)
                found = True
                break
        if not found:
            print(f"WARNING: no data for {family}/{contrast}/nonoverlap")
            vals.append(0)
            errs.append(0)
        arm_key = contrast.replace("minusC", "").replace("minusVS", "VS").replace("minusRS", "RS")
        colors.append(color_map.get(arm_key, "#333333"))

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    x = np.arange(len(contrasts))
    bars = ax.bar(x, vals, 0.6, color=colors, alpha=0.7, yerr=errs, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in contrasts])
    ax.set_ylabel(r"$\Delta G$ vs CLEAN (nonoverlap, seed SD)")
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.set_title("Split attenuation: local relation vs content exposure",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig_split_attenuation.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ═══════ FIGURE 4: Composition/dose (raw arm values) ═══════

def fig_composition_dose():
    """Four panels: raw readouts for C/HV/HM/V/R arms (seed43022)."""
    compact = read_csv(os.path.join(DATA, "half_view_curve_probe",
                                     "compact_TUN_late_roles.csv"))
    copy_data = read_csv(os.path.join(DATA, "half_view_curve_probe",
                                       "copy_late_roles.csv"))
    wiki = read_csv(os.path.join(DATA, "half_view_curve_probe",
                                  "wikipedia_late_roles.csv"))

    arms = ["C", "HV", "HM", "V", "R"]
    arm_colors = [COLORS[a] for a in arms]

    def get_compact(arm, tc, field):
        for r in compact:
            if r["role"] == arm and r["token_class"] == tc:
                return float(r[field])
        return 0.0

    compact_G = [get_compact(a, "nonoverlap", "G") for a in arms]
    copy_gain = []
    for a in arms:
        for r in copy_data:
            if r["role"] == a:
                copy_gain.append(float(r["gain"]))
                break
        else:
            copy_gain.append(0)

    wiki_ov, wiki_nov = [], []
    for a in arms:
        for r in wiki:
            if r["role"] == a and r["overlap_bin"] == "ALL" and r["token_class"] == "overlap":
                wiki_ov.append(float(r["mean_gain_T_vs_N"]))
                break
        else:
            wiki_ov.append(0)
        for r in wiki:
            if r["role"] == a and r["overlap_bin"] == "ALL" and r["token_class"] == "nonoverlap":
                wiki_nov.append(float(r["mean_gain_T_vs_N"]))
                break
        else:
            wiki_nov.append(0)

    fig, axes = plt.subplots(1, 4, figsize=(10.5, 3))
    x = np.arange(len(arms))

    for ax, data, ylabel, title in [
        (axes[0], compact_G, r"Compact nonoverlap $G$", "(a) Restatement\nreadout"),
        (axes[1], copy_gain, "Copy gain (nats)", "(b) Copy\nbehavior"),
        (axes[2], wiki_ov, r"Wiki overlap $A_T$", "(c) Natural\nsource-recurring"),
        (axes[3], wiki_nov, r"Wiki nonoverlap $A_T$", "(d) Natural\nsubstitution"),
    ]:
        bars = ax.bar(x, data, 0.55, color=arm_colors, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(arms, fontsize=7)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.axhline(0, color="black", lw=0.5, ls="--")

    fig.suptitle("Mixed-relation composition: raw arm readouts (seed 43022, 1 seed)",
                 fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig_composition_dose.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ═══════ FIGURE 5: Reach asymmetry matrix ═══════

def fig_reach_asymmetry():
    """Heatmap: ΔA_T for each relation × probe × target class."""
    compact_experience = read_csv(os.path.join(DATA, "paired_context_relation_design_probe",
                                   "compact_TUN_late_contrasts.csv"))
    d3 = read_csv(os.path.join(DATA, "numerical_repair",
                               "compact_TUN_central_token_summary_positive_true_source_use.csv"))
    wiki_d3 = read_csv(os.path.join(DATA, "numerical_repair",
                                     "wikipedia_transfer_summary_recomputed_classbalanced_and_tokenweighted.csv"))
    wiki_compact_experience = read_csv(os.path.join(DATA, "paired_context_relation_design_probe",
                                        "wikipedia_late_contrasts.csv"))

    def get_compact_experience_compact(contrast, tc):
        for r in compact_experience:
            if r.get("contrast") == contrast and r.get("token_class") == tc:
                return float(r.get("delta_A_T", "nan"))
        return np.nan

    # FIXED: match family + contrast separately
    def get_d3_compact(family, contrast, tc):
        for r in d3:
            if (r.get("family") == family and r.get("contrast") == contrast
                    and r.get("token_class") == tc):
                return float(r["delta_true_adv_N_minus_T_across_seed_mean"])
        return np.nan

    # Designed compact overlap from roles file (seed43022 only)
    roles = read_csv(os.path.join(DATA, "half_view_curve_probe", "compact_TUN_late_roles.csv"))
    def get_roles_AT(arm, tc):
        for r in roles:
            if r.get("role") == arm and r.get("token_class") == tc:
                return float(r["A_T"])
        return np.nan
    c_ov_at = get_roles_AT("C", "overlap")

    # Wikipedia d3: uses pair_mean_by_target_class aggregation
    def get_wiki_d3(contrast, tc):
        for r in wiki_d3:
            if (r.get("contrast") == contrast and r.get("token_class") == tc
                    and r.get("aggregation") == "pair_mean_by_target_class"):
                return float(r["delta_gain_T_vs_N_mean"])
        return np.nan

    def get_wiki_compact_experience(contrast, tc):
        for r in wiki_compact_experience:
            if (r.get("contrast") == contrast and r.get("overlap_bin") == "ALL"
                    and r.get("token_class") == tc and r.get("estimand") == "gain_T_vs_N"):
                return float(r["mean_difference"])
        return np.nan

    relations = [
        ("Restatement\n(ALN$-$OFF, 1 seed)", "compact_experience", "ALNminusOFF"),
        ("Exact recur.\n(DUP$-$OFF, 1 seed)", "compact_experience", "DUPminusOFF"),
        ("VIEW$-$C\n(3 seeds)", "d3", "VminusC"),
        ("REPEAT$-$C\n(3 seeds)", "d3", "RminusC"),
    ]
    probes = [
        ("Compact\noverlap", "overlap"),
        ("Compact\nnonoverlap", "nonoverlap"),
        ("Wikipedia\noverlap", "overlap"),
        ("Wikipedia\nnonoverlap", "nonoverlap"),
    ]

    matrix = np.full((len(relations), len(probes)), np.nan)

    for i, (label, source, ckey) in enumerate(relations):
        if source == "compact_experience":
            matrix[i, 0] = get_compact_experience_compact(ckey, "overlap")
            matrix[i, 1] = get_compact_experience_compact(ckey, "nonoverlap")
            matrix[i, 2] = get_wiki_compact_experience(ckey, "overlap")
            matrix[i, 3] = get_wiki_compact_experience(ckey, "nonoverlap")
        else:  # d3: compact overlap from roles (1-seed), nonoverlap from d3 (3-seed)
            arm = "V" if "V" in ckey else "R"
            matrix[i, 0] = get_roles_AT(arm, "overlap") - c_ov_at  # 1-seed
            matrix[i, 1] = get_d3_compact("original_three_seed", ckey, "nonoverlap")  # 3-seed
            matrix[i, 2] = get_wiki_d3(ckey, "overlap")  # 3-seed
            matrix[i, 3] = get_wiki_d3(ckey, "nonoverlap")  # 3-seed

    print(f"Reach matrix:\n{matrix}")

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)))
    im = ax.imshow(matrix, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(np.arange(len(probes)))
    ax.set_xticklabels([p[0] for p in probes], fontsize=8)
    ax.set_yticks(np.arange(len(relations)))
    ax.set_yticklabels([r[0] for r in relations], fontsize=8)

    for i in range(len(relations)):
        for j in range(len(probes)):
            v = matrix[i, j]
            if not np.isnan(v):
                color = "white" if abs(v) > vmax * 0.5 else "black"
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7, color=color)

    plt.colorbar(im, ax=ax, label=r"$\Delta A_T$ (positive = better source use)", shrink=0.8)
    ax.set_title("Reach asymmetry: recurrence liability broad, restatement competence narrow",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig_reach_asymmetry.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    print("Generating corrected report figures...")
    for name, fn in [("target_class_crossing", fig_target_class_crossing),
                     ("entity_relevant_updates", fig_entity_relevant_updates),
                     ("split_attenuation", fig_split_attenuation),
                     ("composition_dose", fig_composition_dose),
                     ("reach_asymmetry", fig_reach_asymmetry)]:
        try:
            fn()
        except Exception as e:
            import traceback
            print(f"ERROR {name}: {e}")
            traceback.print_exc()
    print("Done.")
