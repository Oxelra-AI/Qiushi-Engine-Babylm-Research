#!/usr/bin/env python3
"""research research-facing figures for relation-typed composition evidence."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import math
import pathlib
import statistics
from collections import defaultdict
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = _public_path('experiments/archive/relation_learning/scripts/make_research_figures.py')
ROOT = _PUBLIC_ROOT
WS = _public_path('experiments/archive/relation_learning')
FIG = _public_path('experiments/archive/relation_learning/figures')
DATA = _public_path('experiments/archive/relation_learning/data')
OUT = _public_path('experiments/archive/relation_learning/figures/relation_typed_composition')
NOTE = _public_path('research/notes/relation_learning/research_figure_manifest.md')


def rel(p: pathlib.Path) -> str:
    try: return str(p.relative_to(ROOT))
    except Exception: return str(p)


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(x: Any) -> float:
    try: return float(x)
    except Exception: return float("nan")


def mean(xs):
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return statistics.mean(vals) if vals else float("nan")


def sd(xs):
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return statistics.stdev(vals) if len(vals) >= 2 else 0.0


def savefig(name: str) -> tuple[pathlib.Path, pathlib.Path]:
    png = OUT / f"{name}.png"
    pdf = OUT / f"{name}.pdf"
    plt.savefig(png, dpi=240, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.close()
    return png, pdf


def fig1_compact_tun():
    orig = read_csv(_public_path('experiments/archive/relation_learning/data/original_threeseed_neutral_anchor/rewrite_TUN_across_seed_contrasts.csv'))
    split430 = read_csv(_public_path('experiments/archive/relation_learning/data/neutral_anchor_rewrite_probe/rewrite_TUN_late_contrasts.csv'))
    split431 = read_csv(_public_path('experiments/archive/relation_learning/data/split_seed43122_neutral_anchor/rewrite_TUN_late_contrasts.csv'))
    values = []
    for label, contrast, kind in [("R−C", "RminusC", "orig"), ("V−C", "VminusC", "orig")]:
        row = next(r for r in orig if r["token_class"] == "nonoverlap" and r["contrast"] == contrast)
        values.append({"label": label, "kind": kind, "tn": f(row["delta_T_minus_N_mean"]), "tn_sd": f(row["delta_T_minus_N_sd"]), "un": f(row["delta_U_minus_N_mean"]), "un_sd": f(row["delta_U_minus_N_sd"])})
    split_vals = defaultdict(lambda: {"tn": [], "un": []})
    for rows in [split430, split431]:
        for contrast, label in [("RSminusC", "RS−C"), ("VSminusC", "VS−C")]:
            row = next(r for r in rows if r["token_class"] == "nonoverlap" and r["contrast"] == contrast)
            split_vals[label]["tn"].append(f(row["delta_T_minus_N"]))
            split_vals[label]["un"].append(f(row["delta_U_minus_N"]))
    for label in ["RS−C", "VS−C"]:
        values.append({"label": label, "kind": "split", "tn": mean(split_vals[label]["tn"]), "tn_sd": sd(split_vals[label]["tn"]), "un": mean(split_vals[label]["un"]), "un_sd": sd(split_vals[label]["un"])})
    labels = [v["label"] for v in values]
    x = np.arange(len(labels))
    width = 0.36
    colors = ["#B64A3A", "#3A7EB6", "#E3A199", "#9BC3E6"]
    plt.figure(figsize=(8.2, 4.8))
    plt.axhline(0, color="black", lw=0.8)
    plt.bar(x - width/2, [v["tn"] for v in values], width, yerr=[v["tn_sd"] for v in values], label="Δ(T−N): related true source", color=colors, edgecolor="black", linewidth=0.5, capsize=3)
    plt.bar(x + width/2, [v["un"] for v in values], width, yerr=[v["un_sd"] for v in values], label="Δ(U−N): unrelated source", color=["#F5D0C7", "#C7DBF2", "#F5D0C7", "#C7DBF2"], edgecolor="black", linewidth=0.5, capsize=3)
    plt.xticks(x, labels)
    plt.ylabel("Arm − CLEAN in NLL difference (nats)\npositive Δ(T−N) = weaker true-source use")
    plt.title("Compact rewrite T/U/N: relation type changes source-conditioned computation")
    plt.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    return savefig("fig1_compact_TUN_relation_locality")


def fig2_wikipedia():
    rows = read_csv(_public_path('experiments/archive/relation_learning/data/integrated_scope_and_mechanism/wikipedia_scope_key_across_seed.csv'))
    def get(bin_, tc, contrast):
        r = next(x for x in rows if x["overlap_bin"] == bin_ and x["token_class"] == tc and x["contrast"] == contrast)
        return f(r["gain_T_vs_N_mean"]), f(r["gain_T_vs_N_seed_sd"])
    groups = ["ALL", "low", "medium", "high"]
    labels = ["All", "Low", "Medium", "High"]
    r_vals, r_err, v_vals, v_err = [], [], [], []
    for g in groups:
        m, s = get(g, "nonoverlap", "RminusC"); r_vals.append(m); r_err.append(s)
        m, s = get(g, "nonoverlap", "VminusC"); v_vals.append(m); v_err.append(s)
    x = np.arange(len(groups)); width = 0.34
    plt.figure(figsize=(8.2, 4.6))
    plt.axhline(0, color="black", lw=0.8)
    plt.bar(x - width/2, r_vals, width, yerr=r_err, label="R−C", color="#B64A3A", edgecolor="black", linewidth=0.5, capsize=3)
    plt.bar(x + width/2, v_vals, width, yerr=v_err, label="V−C", color="#3A7EB6", edgecolor="black", linewidth=0.5, capsize=3)
    plt.xticks(x, labels)
    plt.ylabel("Δ true-source benefit N−T (nats)\npositive = more benefit than CLEAN")
    plt.xlabel("Source/rewrite overlap bin; nonoverlap targets")
    plt.title("Wikipedia/Simple-English restatement: recurrence cost transfers; VIEW benefit is small")
    plt.legend(frameon=False)
    plt.tight_layout()
    return savefig("fig2_wikipedia_natural_restatement_scope")


def fig3_entity():
    rows = read_csv(_public_path('experiments/archive/relation_learning/data/integrated_scope_and_mechanism/entity_twoseed_key_contrasts.csv'))
    groups = ["rel_eq0", "rel_updates_1", "rel_updates_2", "rel_updates_3", "rel_updates_4", "rel_updates_5"]
    xlabels = ["0", "1", "2", "3", "4", "5"]
    plt.figure(figsize=(8.0, 5.0))
    styles = {
        (43022, "RminusC"): ("#B64A3A", "o", "R−C 43022"),
        (43022, "RSminusC"): ("#E3A199", "s", "RS−C 43022"),
        (43122, "RminusC"): ("#7E2F24", "o", "R−C 43122"),
        (43122, "RSminusC"): ("#C77A70", "s", "RS−C 43122"),
    }
    for (seed, contrast), (color, marker, label) in styles.items():
        vals = []
        for g in groups:
            row = next(r for r in rows if int(r["seed"]) == seed and r["group"] == g)
            vals.append(f(row[contrast]))
        plt.plot(range(len(groups)), vals, color=color, marker=marker, lw=2, label=label)
    plt.axhline(0, color="black", lw=0.8)
    plt.xticks(range(len(groups)), xlabels)
    plt.xlabel("Relevant queried-entity updates")
    plt.ylabel("Official Entity accuracy delta vs CLEAN (points)")
    plt.title("Entity behavioral face: local exact recurrence benefit disappears when split")
    plt.legend(frameon=False, ncol=2, fontsize=9)
    plt.tight_layout()
    return savefig("fig3_entity_twoseed_repeat_localization")


def fig4_mechanism():
    spec = read_csv(_public_path('experiments/archive/relation_learning/data/source_specificity_misfire/specificity_late_contrasts.csv'))
    # Mean across seeds for T/U condition and R/V contrasts.
    records = []
    for contrast in ["RminusC", "VminusC"]:
        for condition in ["T", "U"]:
            vals = [r for r in spec if r["contrast"] == contrast and r["condition"] == condition]
            records.append({
                "contrast": contrast, "condition": condition,
                "src_mass": mean([f(r["true_src_content_mass_delta"]) for r in vals]),
                "src_mass_sd": sd([f(r["true_src_content_mass_delta"]) for r in vals]),
                "target_prob": mean([f(r["target_prob_delta"]) for r in vals]),
                "target_prob_sd": sd([f(r["target_prob_delta"]) for r in vals]),
            })
    target = read_csv(_public_path('experiments/archive/relation_learning/data/integrated_scope_and_mechanism/rewrite_target_probability_across_seed.csv'))
    # Plot source mass/target prob for nonoverlap plus target-prob direct R/V overlap/nonoverlap.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    labels = ["R−C T", "R−C U", "V−C T", "V−C U"]
    xs = np.arange(len(labels))
    rec_order = [("RminusC","T"),("RminusC","U"),("VminusC","T"),("VminusC","U")]
    recmap = {(r["contrast"], r["condition"]): r for r in records}
    width = 0.34
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].bar(xs - width/2, [recmap[k]["src_mass"] for k in rec_order], width, yerr=[recmap[k]["src_mass_sd"] for k in rec_order], color="#999999", edgecolor="black", capsize=3, label="source-content mass")
    axes[0].bar(xs + width/2, [recmap[k]["target_prob"] for k in rec_order], width, yerr=[recmap[k]["target_prob_sd"] for k in rec_order], color="#55A868", edgecolor="black", capsize=3, label="target probability")
    axes[0].set_xticks(xs); axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].set_ylabel("Probability change vs comparison arm")
    axes[0].set_title("True-source specificity on nonoverlap rewrite masks")
    axes[0].legend(frameon=False, fontsize=8)
    def trow(tc, contrast):
        return next(r for r in target if r["token_class"] == tc and r["contrast"] == contrast)
    labels2 = ["R−C overlap", "R−C nonoverlap", "V−C overlap", "V−C nonoverlap"]
    vals2 = [f(trow("overlap","RminusC")["delta_true_target_prob_mean"]), f(trow("nonoverlap","RminusC")["delta_true_target_prob_mean"]), f(trow("overlap","VminusC")["delta_true_target_prob_mean"]), f(trow("nonoverlap","VminusC")["delta_true_target_prob_mean"])]
    err2 = [f(trow("overlap","RminusC")["delta_true_target_prob_seed_sd"]), f(trow("nonoverlap","RminusC")["delta_true_target_prob_seed_sd"]), f(trow("overlap","VminusC")["delta_true_target_prob_seed_sd"]), f(trow("nonoverlap","VminusC")["delta_true_target_prob_seed_sd"])]
    colors = ["#B64A3A", "#B64A3A", "#3A7EB6", "#3A7EB6"]
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].bar(np.arange(4), vals2, yerr=err2, color=colors, alpha=0.9, edgecolor="black", capsize=3)
    axes[1].set_xticks(np.arange(4)); axes[1].set_xticklabels(labels2, rotation=20, ha="right")
    axes[1].set_ylabel("Δ true-source target probability")
    axes[1].set_title("Correct-token probability in rewrite context")
    plt.suptitle("Mechanism: source recognition transfers; precise answer use is relation-format dependent", y=1.02)
    plt.tight_layout()
    return savefig("fig4_source_output_and_target_probability")


def fig5_ordinary():
    rows = read_csv(_public_path('experiments/archive/relation_learning/data/integrated_scope_and_mechanism/ordinary_loss_decomposition.csv'))
    labels = [r["arm_family"] for r in rows]
    split = [f(r["split_exposure_vs_clean"]) for r in rows]
    local = [f(r["inwindow_share_local_minus_split"]) for r in rows]
    x = np.arange(len(labels))
    plt.figure(figsize=(6.4, 4.6))
    plt.bar(x, split, label="spaced exposure share (split−C)", color="#BBBBBB", edgecolor="black")
    plt.bar(x, local, bottom=split, label="same-window share (local−split)", color="#555555", edgecolor="black")
    plt.xticks(x, labels)
    plt.ylabel("Ordinary held-out MLM loss delta vs CLEAN (nats)")
    plt.title("Ordinary held-out cost is small and mostly not the same-window component")
    plt.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    return savefig("fig5_ordinary_loss_decomposition")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    made = []
    for maker in [fig1_compact_tun, fig2_wikipedia, fig3_entity, fig4_mechanism, fig5_ordinary]:
        png, pdf = maker()
        made.append((png, pdf))
    lines = ["# research research figure manifest", "", "These are research-facing evidence figures generated from existing tables. They are not final publication layouts.", ""]
    for png, pdf in made:
        lines.append(f"- `{rel(png)}`")
        lines.append(f"  - `{rel(pdf)}`")
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"status":"RESEARCH_FIGURES_DONE", "figures": [rel(p[0]) for p in made], "manifest": rel(NOTE)})

if __name__ == "__main__":
    main()
