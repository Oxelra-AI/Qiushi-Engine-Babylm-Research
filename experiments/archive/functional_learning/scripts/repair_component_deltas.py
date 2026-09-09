#!/usr/bin/env python3
"""Regenerate fig_component_deltas.pdf with the two contrasts separated.

The old figure/caption described clean-minus-ordinary as a preservation-only
improvement. That is wrong: clean - ordinary includes the acquisition policy and
preservation. Preservation recovery is clean - (M,S). This script reads the
canonical strict admission JSON and plots both contrasts side by side.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = _public_path('experiments/archive/functional_learning')
ADMISSION = _public_path('experiments/archive/functional_learning/data/strict_split_eval_admission_o_complete/strict_split_eval_admission.json')
OUT = _public_path('data/external/fig_component_deltas.pdf')
DATA_OUT = _public_path('experiments/archive/functional_learning/data/report_argument_repair/component_delta_contrasts.json')
_public_path('experiments/archive/functional_learning/data/report_argument_repair').mkdir(parents=True, exist_ok=True)

COMPONENTS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "SuperGLUE", "GlobalPIQA", "Reading"]
SHORT = ["BLiMP", "Suppl", "EWoK", "Entity", "COMPS", "SG", "GPIQA", "Read"]

with ADMISSION.open("r", encoding="utf-8") as f:
    adm = json.load(f)

seed64 = adm["seed62064_counterparts"]
cohort = {
    "seed 62064": {
        "ordinary": seed64["ordinary_inherited_wwm_seed62064"]["components"],
        "ms": seed64["densemask_sparselabel_seed62064"]["components"],
        "clean": seed64["clean_pres_lambda1_eval_seed62064"]["components"],
    },
    "seed 62065": {
        "ordinary": adm["endpoints"]["o62065"]["scores"],
        "ms": adm["endpoints"]["ms62065"]["scores"],
        "clean": seed64["clean_pres_lambda1_eval_seed62065"]["components"],
    },
}

records = {}
for seed, vals in cohort.items():
    clean_minus_ordinary = {c: vals["clean"][c] - vals["ordinary"][c] for c in COMPONENTS}
    clean_minus_ms = {c: vals["clean"][c] - vals["ms"][c] for c in COMPONENTS}
    records[seed] = {
        "complete_policy_gain_clean_minus_matched_ordinary": clean_minus_ordinary,
        "preservation_recovery_clean_minus_ms": clean_minus_ms,
    }

DATA_OUT.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.3), sharey=True)
x = np.arange(len(COMPONENTS))
w = 0.36
colors = {"seed 62064": "#2A9D8F", "seed 62065": "#7BC8A4"}
hatches = {"seed 62064": "", "seed 62065": "//"}

panels = [
    ("complete_policy_gain_clean_minus_matched_ordinary", "A. Complete policy vs matched ordinary\n(clean − O; acquisition + preservation)",
     "Not a preservation-only contrast"),
    ("preservation_recovery_clean_minus_ms", "B. Recovery relative to acquisition-only\n(clean − (M,S); preservation-associated)",
     "This is the BLiMP/Supp recovery contrast"),
]

for ax, (key, title, subtitle) in zip(axes, panels):
    for offset, seed in [(-w/2, "seed 62064"), (w/2, "seed 62065")]:
        vals = [records[seed][key][c] for c in COMPONENTS]
        bars = ax.bar(x + offset, vals, width=w, color=colors[seed], edgecolor="white",
                      linewidth=0.7, hatch=hatches[seed], label=seed)
        for bar, val in zip(bars, vals):
            if abs(val) >= 0.045:
                y = val + (0.045 if val >= 0 else -0.06)
                ax.text(bar.get_x() + bar.get_width()/2, y, f"{val:+.2f}",
                        ha="center", va="bottom" if val >= 0 else "top",
                        fontsize=6.7, rotation=90 if abs(val) > 0.9 else 0)
    ax.axhline(0, color="#222222", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT, rotation=35, ha="right")
    ax.set_title(title, fontsize=10.5, fontweight="bold")
    ax.text(0.5, 0.97, subtitle, transform=ax.transAxes, ha="center", va="top",
            fontsize=8, color="#555555")
    ax.grid(axis="y", alpha=0.22, linewidth=0.6)

axes[0].set_ylabel("Score difference on repaired coordinate")
axes[0].legend(frameon=False, loc="upper left", fontsize=8)
fig.suptitle("Per-component contrasts: complete policy gain is not the same as preservation recovery",
             fontsize=12.5, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print(f"Wrote {OUT}")
print(f"Wrote {DATA_OUT}")
for seed in records:
    print(seed)
    print("  clean-O BLiMP", records[seed]["complete_policy_gain_clean_minus_matched_ordinary"]["BLiMP"],
          "Supp", records[seed]["complete_policy_gain_clean_minus_matched_ordinary"]["Supplement"],
          "Entity", records[seed]["complete_policy_gain_clean_minus_matched_ordinary"]["Entity"])
    print("  clean-MS BLiMP", records[seed]["preservation_recovery_clean_minus_ms"]["BLiMP"],
          "Supp", records[seed]["preservation_recovery_clean_minus_ms"]["Supplement"],
          "Entity", records[seed]["preservation_recovery_clean_minus_ms"]["Entity"])
