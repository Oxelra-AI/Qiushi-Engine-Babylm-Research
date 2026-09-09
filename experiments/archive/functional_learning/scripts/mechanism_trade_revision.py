#!/usr/bin/env python3
"""research: revised research-facing mechanism figure.

This figure implements the corrected mechanism interpretation and evidence
correction. It is a scientific synthesis
artifact summarizing the mechanism evidence for scientific interpretation.

Panels:
A. Joint plane: ordinary-rendering parent KL (preservation drift surface) vs
   dense-common ground-truth CE gain (acquisition surface).
B. Evidence-availability NLL deltas from a single research JSON source.
C. Whole-bank inference-time private-scale attenuation: Qwen source movement
   can be approximately matched while the same chck82-anchored CDI bank remains worse than clean.
D. Matched rollback of the exact (M,S) update relative to coherent86: an
   attenuation hypothesis different from whole-bank private-scale rescaling.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import json
import pathlib
from typing import Dict, Any, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
PEER = _public_path('experiments/archive/relation_learning')
OUT_FIG = _public_path('experiments/archive/functional_learning/figures')
OUT_DATA = _public_path('experiments/archive/functional_learning/data/mechanism_trade_revision')
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_DATA.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "common_support": _public_path('experiments/archive/functional_learning/data/common_support_endpoint_probe_with_densecorr/common_support_endpoint_probe.json'),
    "evidence_availability": _public_path('experiments/archive/functional_learning/data/evidence_availability_readout_densecorr/evidence_availability_readout.json'),
    "ordinary_drift": _public_path('experiments/archive/functional_learning/data/preservation_drift_profile_densecorr/preservation_drift_profile.json'),
    "whole_bank_scale": _public_path('experiments/archive/relation_learning/data/shrinkage_source_cdi_probe/combined_compact.csv'),
    "matched_rollback": _public_path('experiments/archive/functional_learning/data/frozen_bank_rollback_source_control_full/compact_source_summary.csv'),
}

for name, path in SOURCES.items():
    if not path.exists():
        raise FileNotFoundError(f"missing source {name}: {path}")

cs: Dict[str, Any] = json.loads(SOURCES["common_support"].read_text())
ea: Dict[str, Any] = json.loads(SOURCES["evidence_availability"].read_text())
drift: Dict[str, Any] = json.loads(SOURCES["ordinary_drift"].read_text())


def read_csv(path: pathlib.Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))

scale_rows = read_csv(SOURCES["whole_bank_scale"])
rollback_rows = read_csv(SOURCES["matched_rollback"])

# Endpoint naming deliberately avoids internal shorthand in the figure labels.
endpoints = [
    {
        "key_cs": "coherent86_parent",
        "key_drift": "coherent86",
        "key_ea": "coherent86",
        "short": "Parent",
        "color": "#595959",
        "marker": "o",
        "include_ea": False,
    },
    {
        "key_cs": "ordinary_inherited_wwm_seed62064",
        "key_drift": "ordinary_inherited_wwm_seed62064",
        "key_ea": "ordinary_inherited_wwm_seed62064",
        "short": "Ordinary",
        "color": "#1f77b4",
        "marker": "s",
        "include_ea": True,
    },
    {
        "key_cs": "exact_ms_seed62064",
        "key_drift": "densemask_sparselabel_seed62064",
        "key_ea": "densemask_sparselabel_seed62064",
        "short": "Acq. (M,S)",
        "color": "#ff7f0e",
        "marker": "^",
        "include_ea": True,
    },
    {
        "key_cs": "clean_ms_kl_seed62064",
        "key_drift": "clean_pres_lambda1_eval_seed62064",
        "key_ea": "clean_pres_lambda1_eval_seed62064",
        "short": "Ordinary-state\nanchor s64",
        "color": "#2ca02c",
        "marker": "D",
        "include_ea": True,
    },
    {
        "key_cs": "clean_ms_kl_seed62065",
        "key_drift": "clean_pres_lambda1_eval_seed62065",
        "key_ea": "clean_pres_lambda1_eval_seed62065",
        "short": "Ordinary-state\nanchor s65",
        "color": "#98df8a",
        "marker": "D",
        "include_ea": False,
    },
    {
        "key_cs": "densecorr_ms_kl_seed62064",
        "key_drift": "densecorr_pres_lambda1_seed62064",
        "key_ea": "densecorr_pres_lambda1_seed62064",
        "short": "Dense-state\nanchor",
        "color": "#9467bd",
        "marker": "X",
        "include_ea": True,
    },
]

panel_a_rows = []
for ep in endpoints:
    dense_common = cs["endpoint_scores"][ep["key_cs"]]["dense_common"]
    drift_res = drift["results"][ep["key_drift"]]
    panel_a_rows.append({
        "endpoint": ep["short"].replace("\n", " "),
        "ordinary_rendering_kl_teacher_to_student": drift_res["kl_teacher_to_student_mean"],
        "ordinary_rendering_delta_ce_student_minus_teacher": drift_res["delta_ce_student_minus_teacher"],
        "dense_common_ce_gain_vs_parent": dense_common["ce_gain_vs_parent"],
        "dense_common_rank_gain_vs_parent": dense_common["rank_gain_vs_parent"],
        "dense_common_kl_parent_to_endpoint": dense_common["kl_parent_to_model"],
    })

# Evidence availability: same source and four conditions, excluding parent zero row.
conditions = [
    ("pair_correct_source", "correct\nsource"),
    ("pair_wrong_source", "wrong\nsource"),
    ("view_only", "view\nonly"),
    ("full_row_this_source_erased", "source\nerased"),
]
panel_b_rows = []
for ep in endpoints:
    if not ep["include_ea"]:
        continue
    by = ea["model_summaries"][ep["key_ea"]]["by_condition"]
    row = {"endpoint": ep["short"].replace("\n", " ")}
    for key, _ in conditions:
        row[key] = by[key]["mean_delta_nll_vs_coherent86"]
    panel_b_rows.append(row)

# Whole-bank scale rows used in Panel C.
scale_keys = ["dense64_scale0p75", "dense64_scale0p60", "clean_pres64"]
panel_c_rows = []
for r in scale_rows:
    if r["model"] in scale_keys:
        label = {
            "dense64_scale0p75": "Dense64 scale .75",
            "dense64_scale0p60": "Dense64 scale .60",
            "clean_pres64": "Clean64",
        }[r["model"]]
        panel_c_rows.append({
            "endpoint": label,
            "qwen_delta_spec_Tfit_vs_chck82": float(r["qwen_delta_spec_Tfit"]),
            "cdi_delta_nll_Tfit_vs_chck82": float(r["cdi_delta_nll_Tfit"]),
            "cdi_delta_rank_vs_chck82": float(r["cdi_delta_rank"]),
            "view_only_delta_nll_Tfit_vs_chck82": float(r["view_only_delta_nll_Tfit"]),
        })

# Matched rollback rows used in Panel D.
rollback_keys = [
    "coherent86",
    "densemask_sparselabel_seed62064",
    "clean_pres_lambda1_eval_full80",
    "rollback_alpha_0p7625",
    "rollback_alpha_0p775",
]
panel_d_rows = []
for r in rollback_rows:
    if r["model"] in rollback_keys:
        label = {
            "coherent86": "Parent",
            "densemask_sparselabel_seed62064": "Acq. (M,S)",
            "clean_pres_lambda1_eval_full80": "Clean64",
            "rollback_alpha_0p7625": "Rollback .7625",
            "rollback_alpha_0p775": "Rollback .775",
        }[r["model"]]
        panel_d_rows.append({
            "endpoint": label,
            "calibration_kl_vs_parent": float(r["calibration_kl_vs_parent"]),
            "qwen_delta_spec_Tfit_vs_parent": float(r["qwen_mean_delta_specific_advantage_Tfit_vs_parent"]),
            "common_delta_swing_Tfit_vs_parent": float(r["common_mean_delta_source_follow_swing_Tfit_vs_parent"]),
            "qwen_delta_rank_vs_parent": float(r["qwen_mean_delta_specific_rank_advantage_vs_parent"]),
        })

# Write machine-readable extracted data.
summary = {
    "status": "MECHANISM_TRADE_REVISION_DATA",
    "purpose": "Research-facing figure data: distinguish acquisition-retention trade, whole-bank scale control, matched rollback, and dense-state anchoring.",
    "source_paths": {k: str(v.relative_to(ROOT)) for k, v in SOURCES.items()},
    "panel_a_joint_acquisition_retention": panel_a_rows,
    "panel_b_evidence_availability_delta_nll": panel_b_rows,
    "panel_c_whole_bank_scale_vs_cdi": panel_c_rows,
    "panel_d_matched_update_rollback": panel_d_rows,
    "interpretation": {
        "panel_a": "Ordinary-state anchoring occupies the useful knee: lower ordinary-rendering drift than acquisition-only while retaining most dense-common ground-truth fit on familiar packed-Qwen training rows; dense-state anchoring is near-parent in KL but loses most of that fit.",
        "panel_b": "Acquisition-only and ordinary-state anchoring are evidence-dependent: small correct-source benefit comes with wrong/view/source-erased costs; ordinary continuation and dense-state anchoring are near-uniform or parent-like.",
        "panel_c": "Inference-time dense private-scale shrinkage is a whole-bank attenuation test on a chck82-anchored 96-word CDI bank; scale .60 can match Qwen source movement approximately but is worse than clean on that same bank. Clean's negative CDI delta in this panel is relative to chck82, not recovery to coherent86.",
        "panel_d": "Rollback attenuates only the exact (M,S) update relative to coherent86; its ordinary-KL-matched source movement shows bounded movement can explain much of clean's repair, without being the same hypothesis as whole-bank scale shrinkage.",
    },
}
(_public_path('experiments/archive/functional_learning/data/mechanism_trade_revision/mechanism_trade_revision_data.json')).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
for fname, rows in [
    ("panel_a_joint.csv", panel_a_rows),
    ("panel_b_evidence_availability.csv", panel_b_rows),
    ("panel_c_scale_cdi.csv", panel_c_rows),
    ("panel_d_matched_rollback.csv", panel_d_rows),
]:
    if rows:
        with (OUT_DATA / fname).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

# ---- Plot ----
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})
fig, axes = plt.subplots(2, 2, figsize=(13.8, 10.2))
fig.suptitle("State-selective anchoring: acquisition retained, ordinary drift reduced", fontsize=14, fontweight="bold", y=0.985)

# Panel A
ax = axes[0, 0]
for ep, row in zip(endpoints, panel_a_rows):
    ax.scatter(row["ordinary_rendering_kl_teacher_to_student"], row["dense_common_ce_gain_vs_parent"],
               s=110, color=ep["color"], marker=ep["marker"], edgecolor="black", linewidth=0.7, zorder=3)
    # Manual offsets reduce label collisions near origin and clean seeds.
    offsets = {
        "Parent": (5, -13),
        "Ordinary": (6, 7),
        "Acq. (M,S)": (-47, 7),
        "Ordinary-state\nanchor s64": (7, 7),
        "Ordinary-state\nanchor s65": (7, -20),
        "Dense-state\nanchor": (8, 7),
    }
    dx, dy = offsets.get(ep["short"], (5, 5))
    ax.annotate(ep["short"],
                (row["ordinary_rendering_kl_teacher_to_student"], row["dense_common_ce_gain_vs_parent"]),
                xytext=(dx, dy), textcoords="offset points", fontsize=7.5,
                arrowprops=dict(arrowstyle="-", color="#777777", lw=0.5, shrinkA=0, shrinkB=3))
ax.axhline(0, color="#777777", lw=0.8, ls="--")
ax.axvline(0, color="#777777", lw=0.8, ls="--")
ax.set_xlabel("ordinary-rendering KL teacher||endpoint")
ax.set_ylabel("dense-common CE gain on training rows (nats/token)")
ax.set_title("A. Joint acquisition--retention plane")
ax.set_xlim(-0.001, 0.0285)
ax.set_ylim(-0.06, 0.82)
ax.grid(alpha=0.18, linewidth=0.6)

# Panel B
ax = axes[0, 1]
x = np.arange(len(conditions))
for ep in endpoints:
    if not ep["include_ea"]:
        continue
    row = next(r for r in panel_b_rows if r["endpoint"] == ep["short"].replace("\n", " "))
    y = [row[key] for key, _ in conditions]
    ax.plot(x, y, marker=ep["marker"], color=ep["color"], lw=2, ms=6, label=ep["short"].replace("\n", " "))
ax.axhline(0, color="#777777", lw=0.8, ls="--")
ax.set_xticks(x)
ax.set_xticklabels([lab for _, lab in conditions])
ax.set_ylabel("ΔNLL vs coherent86 (negative is better)")
ax.set_title("B. Evidence availability from one fixed bank")
ax.grid(axis="y", alpha=0.18, linewidth=0.6)
ax.legend(loc="upper left", frameon=True, ncol=1)

# Panel C
ax = axes[1, 0]
panel_c_style = {
    "Dense64 scale .75": ("#ff7f0e", "^"),
    "Dense64 scale .60": ("#d62728", "v"),
    "Clean64": ("#2ca02c", "D"),
}
for row in panel_c_rows:
    col, mark = panel_c_style[row["endpoint"]]
    ax.scatter(row["qwen_delta_spec_Tfit_vs_chck82"], row["cdi_delta_nll_Tfit_vs_chck82"],
               s=105, color=col, marker=mark, edgecolor="black", linewidth=0.7, zorder=3)
    offsets = {"Dense64 scale .75": (-48, 7), "Dense64 scale .60": (8, 6), "Clean64": (8, -18)}
    ax.annotate(row["endpoint"], (row["qwen_delta_spec_Tfit_vs_chck82"], row["cdi_delta_nll_Tfit_vs_chck82"]),
                xytext=offsets[row["endpoint"]], textcoords="offset points", fontsize=8,
                arrowprops=dict(arrowstyle="-", color="#777777", lw=0.5, shrinkA=0, shrinkB=3))
# Visual connector between scale .60 and clean: source movement approximately matched, chck82-anchored CDI differs.
sc60 = next(r for r in panel_c_rows if r["endpoint"] == "Dense64 scale .60")
cl = next(r for r in panel_c_rows if r["endpoint"] == "Clean64")
ax.plot([sc60["qwen_delta_spec_Tfit_vs_chck82"], cl["qwen_delta_spec_Tfit_vs_chck82"]],
        [sc60["cdi_delta_nll_Tfit_vs_chck82"], cl["cdi_delta_nll_Tfit_vs_chck82"]],
        color="#444444", lw=1.0, ls=":")
ax.axhline(0, color="#777777", lw=0.8, ls="--")
ax.set_xlabel("Qwen source-specific movement (T-fit, vs chck82)")
ax.set_ylabel("96-word CDI ΔNLL vs chck82 (lower)")
ax.set_title("C. Whole-bank scale on chck82-CDI bank")
ax.grid(alpha=0.18, linewidth=0.6)

# Panel D
ax = axes[1, 1]
rollback_style = {
    "Parent": ("#595959", "o"),
    "Acq. (M,S)": ("#ff7f0e", "^"),
    "Clean64": ("#2ca02c", "D"),
    "Rollback .7625": ("#8c564b", "P"),
    "Rollback .775": ("#a0522d", "P"),
}
for row in panel_d_rows:
    col, mark = rollback_style[row["endpoint"]]
    ax.scatter(row["calibration_kl_vs_parent"], row["qwen_delta_spec_Tfit_vs_parent"],
               s=105, color=col, marker=mark, edgecolor="black", linewidth=0.7, zorder=3)
    offsets = {
        "Parent": (6, -15),
        "Acq. (M,S)": (-58, 7),
        "Clean64": (8, 7),
        "Rollback .7625": (-70, -15),
        "Rollback .775": (8, -17),
    }
    ax.annotate(row["endpoint"], (row["calibration_kl_vs_parent"], row["qwen_delta_spec_Tfit_vs_parent"]),
                xytext=offsets[row["endpoint"]], textcoords="offset points", fontsize=8,
                arrowprops=dict(arrowstyle="-", color="#777777", lw=0.5, shrinkA=0, shrinkB=3))
ax.axhline(0, color="#777777", lw=0.8, ls="--")
ax.axvline(0, color="#777777", lw=0.8, ls="--")
ax.set_xlabel("ordinary-text calibration KL vs parent")
ax.set_ylabel("Qwen source-specific movement (T-fit, vs parent)")
ax.set_title("D. Matched update rollback tests a different attenuation")
ax.grid(alpha=0.18, linewidth=0.6)

fig.text(0.5, 0.012,
         "Sources: research common-support/evidence/drift readouts; research whole-bank scale/CDI; research frozen-bank rollback. "
         "Panel A dense-common is familiar-row fit; Panel C CDI is chck82-anchored; Panels C/D test different hypotheses.",
         ha="center", va="bottom", fontsize=8, color="#444444")
plt.tight_layout(rect=[0, 0.035, 1, 0.965])

fig_path = _public_path('experiments/archive/functional_learning/figures/mechanism_trade_revision.png')
pdf_path = _public_path('experiments/archive/functional_learning/figures/mechanism_trade_revision.pdf')
fig.savefig(fig_path, dpi=220, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
plt.close(fig)

md_lines = [
    "# research mechanism trade revision", "",
    "This research-facing figure implements the independent_review repair of the research plot and the research distinction between attenuation controls.", "",
    f"PNG: `{fig_path.relative_to(ROOT)}`", f"PDF: `{pdf_path.relative_to(ROOT)}`", "",
    "## Scientific reading", "",
    "- Panel A puts ordinary-rendering drift and dense-state acquisition in one plane. Clean ordinary-state anchoring keeps most dense-common CE gain on familiar packed-Qwen training rows while reducing ordinary-rendering KL relative to acquisition-only; dense-state anchoring attains low KL while losing most acquired dense-state fit. This panel is a state-specific fit probe, not held-out transfer or official benchmark evidence.",
    "- Panel B is regenerated from a single research evidence-availability JSON and includes the source-erased condition, so the acquisition/clean asymmetry is not mixed from different banks.",
    "- Panel C preserves the scale-control result narrowly: whole-bank inference-time scale 0.60 can nearly match clean's Qwen source movement but remains worse than clean on the same chck82-anchored 96-word CDI bank. The clean64 CDI deltas in that source are improvements relative to chck82, not recovery to coherent86; clean still carries residual CDI damage relative to the Stage III teacher/parent.",
    "- Panel D preserves the rollback result separately: rollback attenuates only the exact (M,S) update relative to coherent86 and lands near the clean ordinary-KL/source-response region, so bounded movement explains much of clean's repair without proving a unique training-time teacher mechanism.",
    "", "## Source paths", "",
]
for name, path in SOURCES.items():
    md_lines.append(f"- {name}: `{path.relative_to(ROOT)}`")
(_public_path('research/documents/functional_learning/data/mechanism_trade_revision/mechanism_trade_revision.md')).write_text("\n".join(md_lines) + "\n")

print(json.dumps({
    "status": "MECHANISM_TRADE_REVISION_DONE",
    "figure_png": str(fig_path.relative_to(ROOT)),
    "figure_pdf": str(pdf_path.relative_to(ROOT)),
    "data_json": str((_public_path('experiments/archive/functional_learning/data/mechanism_trade_revision/mechanism_trade_revision_data.json')).relative_to(ROOT)),
    "summary_md": str((_public_path('research/documents/functional_learning/data/mechanism_trade_revision/mechanism_trade_revision.md')).relative_to(ROOT)),
}, indent=2), flush=True)
