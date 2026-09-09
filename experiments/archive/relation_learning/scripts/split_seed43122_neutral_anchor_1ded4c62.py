#!/usr/bin/env python3
"""research: neutral-source anchor for seed43122 C/R/V/RS/VS DeBERTa arms.

Repairs the research wrapper that crashed because it pointed T_U_ROWS at a summary
CSV instead of raw pair-level rows.  This script:
  1. Loads raw pair-level T/U rewrite rows for all five seed43122 arms from
     the actual scored outputs.
  2. Adds the role2 column expected by the research neutral-anchor integration.
  3. Runs the same N scoring + integration pipeline on seed43122 arms.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import pathlib
import sys

import pandas as pd

ROOT0 = _public_path('experiments/archive/relation_learning/scripts/split_seed43122_neutral_anchor_1ded4c62.py')
ROOT = _PUBLIC_ROOT

sys.path.insert(0, str(ROOT / "experiments/archive/relation_learning/scripts"))
import neutral_anchor_rewrite_probe as base  # noqa: E402

WS = ROOT / "experiments/archive/relation_learning"
REPRESENTATION_FRONTIER_STUDIES_RUNS = ROOT / "experiments/archive/frontier_consolidation/training/runs"
FUNCTIONAL_RELATION_STUDIES_RUNS = WS / "training/runs"

# Override output paths
base.OUT = WS / "data/split_seed43122_neutral_anchor"
base.NOTE = (_PUBLIC_ROOT / 'research/notes/relation_learning/split_seed43122_neutral_anchor.md')

# All five seed43122 arm configs
base.ARM_CONFIGS = {
    "D_C_43122": REPRESENTATION_FRONTIER_STUDIES_RUNS / "full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43122",
    "D_R_43122": REPRESENTATION_FRONTIER_STUDIES_RUNS / "full_p2c_c2p_abs_repeat_dose2p64x_matched_rowholdout_deberta100M_seed43122",
    "D_V_43122": REPRESENTATION_FRONTIER_STUDIES_RUNS / "full_p2c_c2p_abs_view_dose2p64x_matched_rowholdout_deberta100M_seed43122",
    "D_RS_43122": FUNCTIONAL_RELATION_STUDIES_RUNS / "full_p2c_c2p_abs_repeat_split_dose2p64x_rowholdout_deberta100M_seed43122",
    "D_VS_43122": FUNCTIONAL_RELATION_STUDIES_RUNS / "full_p2c_c2p_abs_view_split_dose2p64x_rowholdout_deberta100M_seed43122",
}
base.ROLE = {
    "D_C_43122": "C",
    "D_R_43122": "R",
    "D_V_43122": "V",
    "D_RS_43122": "RS",
    "D_VS_43122": "VS",
}

# Source pair-level rows: original C/R/V from research, split RS/VS from research
ORIG_PAIR_ROWS = WS / "data/heldout_copy_rewrite_entity_ablation/rewrite_pair_rows.csv"
RS_PAIR_ROWS = WS / "data/split_repeat_split_seed43122_probes/rewrite_pair_rows.csv"
VS_PAIR_ROWS = WS / "data/split_view_split_seed43122_probes/rewrite_pair_rows.csv"


def role_from_arm(arm: str) -> str:
    """Derive role from arm name: D_RS_43122 -> RS, D_C_43122 -> C."""
    bits = str(arm).split("_")
    if len(bits) >= 3:
        return bits[1]
    return str(arm)


def load_tu_rows_seed43122():
    """Load raw pair-level T/U rewrite rows for all five seed43122 arms."""
    frames = []

    # Original C/R/V from the master research scored output
    df_orig = pd.read_csv(ORIG_PAIR_ROWS)
    df_orig["role2"] = df_orig["arm"].apply(role_from_arm)
    seed43122_orig = df_orig[
        (df_orig["arm"].str.contains("43122")) &
        (df_orig["role2"].isin(["C", "R", "V"])) &
        (df_orig["checkpoint"].isin(base.CKPTS))
    ].copy()
    frames.append(seed43122_orig)

    # Split RS from research scorer
    df_rs = pd.read_csv(RS_PAIR_ROWS)
    df_rs["role2"] = df_rs["arm"].apply(role_from_arm)
    rs_only = df_rs[
        (df_rs["role2"] == "RS") &
        (df_rs["checkpoint"].isin(base.CKPTS))
    ].copy()
    frames.append(rs_only)

    # Split VS from research scorer
    df_vs = pd.read_csv(VS_PAIR_ROWS)
    df_vs["role2"] = df_vs["arm"].apply(role_from_arm)
    vs_only = df_vs[
        (df_vs["role2"] == "VS") &
        (df_vs["checkpoint"].isin(base.CKPTS))
    ].copy()
    frames.append(vs_only)

    combined = pd.concat(frames, ignore_index=True)

    # Select columns expected by the integration
    keep = ["arm", "checkpoint", "seed", "role2", "probe_id", "pair_id",
            "token_class", "gain", "true_source_nll", "unrelated_source_nll"]
    missing = [c for c in keep if c not in combined.columns]
    if missing:
        raise RuntimeError({
            "missing_columns": missing,
            "columns": list(combined.columns),
            "n_rows": len(combined),
            "arms_found": sorted(combined["arm"].unique().tolist()),
        })

    result = combined[keep].drop_duplicates(subset=["arm", "checkpoint", "probe_id"])
    arms_found = sorted(result["arm"].unique().tolist())
    print(f"[load_tu_rows_seed43122] {len(result)} rows, arms: {arms_found}", flush=True)
    return result


base.load_tu_rows = load_tu_rows_seed43122  # type: ignore[assignment]

if __name__ == "__main__":
    base.main()
