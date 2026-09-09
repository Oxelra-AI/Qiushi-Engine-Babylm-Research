#!/usr/bin/env python3
"""research: recompute old COMPACT_EXPERIENCE ALN-OFF ordinary-loss contrasts by leakage subset.

Existing research/032 row-level MLM-loss tables scored the later 6,992-row
rowholdout set for COMPACT_EXPERIENCE OFF/ALN at seeds 43022 and 43122.  research established
that this set was not held out from the COMPACT_EXPERIENCE streams: all 6,992 rows appear in
OFF and ALN official text, and 2,354 rows additionally contain inherited ALN
source/rewrite pair text.  This script does not rescue the old readout as a true
generalization measure; it quantifies how much the old ALN-OFF heldout-loss
improvement is concentrated in rows with inherited-pair exposure.
"""
from __future__ import annotations

import csv
import json
import math
import pathlib
import statistics
import time
from typing import Any

import numpy as np
import pandas as pd

ROOT = pathlib.Path.cwd()
WS = ROOT / "experiments/archive/relation_learning"
OUT = WS / "data/paired_context_aln_off_heldout_subsets"
ROWS_43022 = WS / "data/paired_context_ordinary_heldout_loss/ordinary_heldout_rows.csv"
ROWS_43122 = WS / "data/paired_context_aln_off_seed43122_ordinary_heldout/ordinary_heldout_rows.csv"
FULL_HELDOUT = ROOT / "experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/heldout_cleanqwen_rows.jsonl"
PAIR_CLEAN = WS / "data/inherited_aln_leakage_audit/heldout6992_no_inherited_aln_pair_text_hits.jsonl"
PAIR_BLOCKED = WS / "data/inherited_aln_leakage_audit/heldout6992_inherited_aln_pair_text_hit_rows.jsonl"
STREAM_CLEAN = WS / "data/current_dose_stream_heldout_exposure/heldout6992_no_inherited_pair_or_base_dose_stream_hits.jsonl"


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def read_jsonl(path: pathlib.Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def row_key(obj: dict[str, Any]) -> tuple[str, int]:
    return (str(obj.get("source", "")), int(obj.get("example_id")))


def load_subset_ids() -> dict[str, set[tuple[str, int]]]:
    full = {row_key(o) for o in read_jsonl(FULL_HELDOUT)}
    pair_clean = {row_key(o) for o in read_jsonl(PAIR_CLEAN)}
    pair_blocked = {row_key(o) for o in read_jsonl(PAIR_BLOCKED)}
    stream_clean = {row_key(o) for o in read_jsonl(STREAM_CLEAN)} if STREAM_CLEAN.exists() else set()
    out = {
        "all6992": full,
        "no_inherited_aln_pair_text_hits4638": pair_clean,
        "inherited_aln_pair_text_hit_rows2354": pair_blocked,
    }
    if stream_clean:
        out["no_inherited_pair_or_base_dose_stream_hits1743"] = stream_clean
    return out


def load_rows(path: pathlib.Path, true_seed: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["role"].isin(["OFF", "ALN"])].copy()
    df["true_seed"] = true_seed
    df["key_source"] = df["source"].astype(str)
    df["key_example_id"] = df["example_id"].astype(int)
    return df


def summarize_seed(df: pd.DataFrame, subset_name: str, subset_ids: set[tuple[str, int]], seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    mask = [(s, int(e)) in subset_ids for s, e in zip(df["key_source"], df["key_example_id"])]
    g = df.loc[mask].copy()
    by_ck = []
    for (role, ck), x in g.groupby(["role", "checkpoint"]):
        by_ck.append({
            "subset": subset_name,
            "seed": seed,
            "role": role,
            "checkpoint": ck,
            "n": int(len(x)),
            "mean_loss": float(x["loss"].mean()),
            "sd_loss": float(x["loss"].std()),
            "mean_masked": float(x["n_masked"].mean()),
        })
    late = g.groupby(["role"], dropna=False).agg(n=("example_id", "count"), mean_loss=("loss", "mean"), mean_masked=("n_masked", "mean")).reset_index()
    late_rows = []
    for _, r in late.iterrows():
        late_rows.append({
            "subset": subset_name,
            "seed": seed,
            "role": r["role"],
            "n_score_rows": int(r["n"]),
            "n_examples_per_checkpoint": int(r["n"] // max(1, g["checkpoint"].nunique())),
            "mean_loss": float(r["mean_loss"]),
            "mean_masked": float(r["mean_masked"]),
        })
    con_rows = []
    piv = late.set_index("role")
    if "ALN" in piv.index and "OFF" in piv.index:
        a = piv.loc["ALN"]; b = piv.loc["OFF"]
        con_rows.append({
            "subset": subset_name,
            "seed": seed,
            "contrast": "ALN-OFF",
            "delta_loss_late_mean": float(a["mean_loss"] - b["mean_loss"]),
            "loss_ALN": float(a["mean_loss"]),
            "loss_OFF": float(b["mean_loss"]),
            "n_examples_per_checkpoint": int(min(a["n"], b["n"]) // max(1, g["checkpoint"].nunique())),
        })
    # paired row/checkpoint-average contrast
    wide = g.groupby(["key_source", "key_example_id", "role"], dropna=False)["loss"].mean().reset_index().pivot(index=["key_source", "key_example_id"], columns="role", values="loss").dropna()
    if "ALN" in wide.columns and "OFF" in wide.columns:
        vals = (wide["ALN"] - wide["OFF"]).to_numpy(dtype=float)
        con_rows.append({
            "subset": subset_name,
            "seed": seed,
            "contrast": "ALN-OFF_rowpaired",
            "delta_loss_late_mean": float(vals.mean()),
            "median_delta_loss": float(np.median(vals)),
            "se_delta_loss": float(vals.std(ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else float("nan"),
            "n_examples_per_checkpoint": int(len(vals)),
        })
    return by_ck, late_rows, con_rows


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    subsets = load_subset_ids()
    dfs = {43022: load_rows(ROWS_43022, 43022), 43122: load_rows(ROWS_43122, 43122)}
    by_ck_all: list[dict[str, Any]] = []
    late_all: list[dict[str, Any]] = []
    con_all: list[dict[str, Any]] = []
    for seed, df in dfs.items():
        for subset_name, ids in subsets.items():
            by_ck, late, con = summarize_seed(df, subset_name, ids, seed)
            by_ck_all.extend(by_ck); late_all.extend(late); con_all.extend(con)
    pd.DataFrame(by_ck_all).to_csv(OUT / "paired_context_aln_off_by_checkpoint_subsets.csv", index=False)
    pd.DataFrame(late_all).to_csv(OUT / "paired_context_aln_off_late_roles_subsets.csv", index=False)
    pd.DataFrame(con_all).to_csv(OUT / "paired_context_aln_off_contrasts_subsets.csv", index=False)

    con_df = pd.DataFrame(con_all)
    cross = []
    for (subset, contrast), g in con_df.groupby(["subset", "contrast"]):
        vals = g["delta_loss_late_mean"].to_numpy(dtype=float)
        cross.append({
            "subset": subset,
            "contrast": contrast,
            "n_seeds": int(len(vals)),
            "mean_delta_loss": float(vals.mean()),
            "min_delta_loss": float(vals.min()),
            "max_delta_loss": float(vals.max()),
            "seed_values": ";".join(f"{int(r.seed)}:{float(r.delta_loss_late_mean):+.6f}" for r in g.itertuples()),
        })
    pd.DataFrame(cross).to_csv(OUT / "paired_context_aln_off_crossseed_subset_summary.csv", index=False)
    result = {
        "status": "COMPACT_EXPERIENCE_ALN_OFF_HELDOUT_SUBSET_RECOMPUTED",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": round(time.time() - t0, 2),
        "subset_sizes": {k: len(v) for k, v in subsets.items()},
        "contrast_rows": con_all,
        "crossseed_summary": cross,
        "inputs": {
            "rows_43022": rel(ROWS_43022),
            "rows_43122": rel(ROWS_43122),
            "full_heldout": rel(FULL_HELDOUT),
            "pair_clean": rel(PAIR_CLEAN),
            "pair_blocked": rel(PAIR_BLOCKED),
            "stream_clean": rel(STREAM_CLEAN),
        },
        "outputs": {
            "contrasts": rel(OUT / "paired_context_aln_off_contrasts_subsets.csv"),
            "crossseed": rel(OUT / "paired_context_aln_off_crossseed_subset_summary.csv"),
        },
        "scientific_interpretation": "The later 6,992 rows are not true heldout for COMPACT_EXPERIENCE streams because both OFF and ALN contain them as official text. This recomputation asks only whether the old ALN-OFF improvement is concentrated in rows that ALN also sees through inherited pair sources/rewrites; it must not be read as a clean generalization effect.",
    }
    (OUT / "paired_context_aln_off_subset_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
