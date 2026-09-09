#!/usr/bin/env python3
"""research: neutral-source anchor for seed43122 split DeBERTa arms.

research measured T/U/N for seed43022 original and split arms; research measured
T/U for seed43122 split arms.  This script adds N for seed43122 C/R/V/RS/VS on
the same compact rewrite target records and joins it to the research two-seed T/U
integration rows.  The result tests whether the compact-family local-vs-split N
price repeats in the second split seed.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import pathlib
import sys

ROOT0 = _public_path('experiments/archive/relation_learning/scripts/split_seed43122_neutral_anchor.py')
ROOT = _PUBLIC_ROOT

sys.path.insert(0, str(ROOT / "experiments/archive/relation_learning/scripts"))
import neutral_anchor_rewrite_probe as base  # noqa: E402

WS = ROOT / "experiments/archive/relation_learning"
REPRESENTATION_FRONTIER_STUDIES_RUNS = ROOT / "experiments/archive/frontier_consolidation/training/runs"
FUNCTIONAL_RELATION_STUDIES_RUNS = WS / "training/runs"

base.OUT = WS / "data/split_seed43122_neutral_anchor"
base.NOTE = (_PUBLIC_ROOT / 'research/notes/relation_learning/022_split_seed43122_neutral_anchor.md')
base.T_U_ROWS = WS / "data/split_seed_replication/rewrite_by_checkpoint.csv"
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


def load_tu_rows_seed43122():
    import pandas as pd
    df = pd.read_csv(base.T_U_ROWS)
    df = df[(df["seed"] == 43122) & (df["checkpoint"].isin(base.CKPTS))].copy()
    # research integration normalizes split roles to RS/VS but keeps arm labels.
    keep = ["arm", "checkpoint", "seed", "role2", "probe_id", "pair_id", "token_class", "gain", "true_source_nll", "unrelated_source_nll"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise RuntimeError({"missing_columns": missing, "columns": list(df.columns), "tu_rows": str(base.T_U_ROWS)})
    return df[keep].drop_duplicates(subset=["arm", "checkpoint", "probe_id"])


base.load_tu_rows = load_tu_rows_seed43122  # type: ignore[assignment]

if __name__ == "__main__":
    base.main()
