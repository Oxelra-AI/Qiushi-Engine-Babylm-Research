#!/usr/bin/env python3
"""Utilities for research split-arm robustness and seed-replicate scoring.

This script has two CPU-only analysis modes and one GPU scoring mode:

1. robustness: recompute pair-level and strict-word analyses for the already
   scored seed43022 REPEAT_SPLIT probe rows, comparing RS with C and original R.
2. score: score a completed split arm at a requested seed with the research
   held-out copy/rewrite/Entity probes, using the within-seed CLEAN baseline.

The scoring mode is intended for immediate use when seed43122 split training
finishes. It performs no training or final expression.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import math
import pathlib
import re
import sys
from typing import Any

import pandas as pd
from transformers import AutoTokenizer

ROOT = _public_path('experiments/archive/relation_learning/scripts/split_robustness_and_seed_scorer.py')
ROOT = _PUBLIC_ROOT
sys.path.insert(0, str(_public_path('experiments/archive/relation_learning/scripts')))

import heldout_copy_rewrite_entity_ablation as base  # noqa: E402

FUNCTIONAL_RELATION_STUDIES = _public_path('experiments/archive/relation_learning')
REPRESENTATION_FRONTIER_STUDIES = _public_path('experiments/archive/frontier_consolidation')
OUT_ROOT = _public_path('experiments/archive/relation_learning/data')
NOTE_DIR = _public_path('research/notes/relation_learning')

ORIG_REWRITE_ROWS = _public_path('experiments/archive/relation_learning/data/heldout_copy_rewrite_entity_ablation/rewrite_pair_rows.csv')
RS_REWRITE_ROWS = _public_path('experiments/archive/relation_learning/data/split_repeat_split_probes/rewrite_pair_rows.csv')
WORD_MAP = _public_path('experiments/archive/relation_learning/data/relation_decomposition/rewrite_probe_word_nonoverlap_map.csv')
TOKENIZER_DIR = _public_path('experiments/archive/frontier_consolidation/data/compliant_tokenizer')

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "than", "that", "this", "these", "those",
    "of", "in", "on", "at", "to", "for", "from", "with", "without", "by", "as", "is", "are", "was",
    "were", "be", "been", "being", "it", "its", "they", "them", "their", "he", "she", "his", "her",
    "we", "you", "i", "not", "no", "do", "does", "did", "can", "could", "would", "should", "will",
}

CLEAN_RUN = {
    43022: _public_path('experiments/archive/frontier_consolidation/training/runs/full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43022'),
    43122: _public_path('experiments/archive/frontier_consolidation/training/runs/full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43122'),
    43222: _public_path('experiments/archive/relation_learning/training/runs/full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43222_parallel'),
}
SPLIT_RUN = {
    (43022, "repeat_split"): _public_path('experiments/archive/relation_learning/training/runs/full_p2c_c2p_abs_repeat_split_dose2p64x_rowholdout_deberta100M_seed43022'),
    (43022, "view_split"): _public_path('experiments/archive/relation_learning/training/runs/full_p2c_c2p_abs_view_split_dose2p64x_rowholdout_deberta100M_seed43022'),
    (43122, "repeat_split"): _public_path('experiments/archive/relation_learning/training/runs/full_p2c_c2p_abs_repeat_split_dose2p64x_rowholdout_deberta100M_seed43122'),
    (43122, "view_split"): _public_path('experiments/archive/relation_learning/training/runs/full_p2c_c2p_abs_view_split_dose2p64x_rowholdout_deberta100M_seed43122'),
}


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def fmt(x: Any, nd: int = 4) -> str:
    try:
        xf = float(x)
    except Exception:
        return str(x)
    if math.isnan(xf):
        return "NA"
    return f"{xf:+.{nd}f}"


def role_from_arm(arm: str) -> str:
    s = str(arm)
    if "D_RS" in s:
        return "RS"
    if "D_VS" in s:
        return "VS"
    parts = s.split("_")
    return parts[1] if len(parts) > 1 else "?"


def load_rewrite_for_robustness() -> pd.DataFrame:
    orig = pd.read_csv(ORIG_REWRITE_ROWS)
    orig["source"] = "orig"
    rs = pd.read_csv(RS_REWRITE_ROWS)
    if "arch" in rs.columns:
        rs = rs.drop(columns=["arch"])
    rs["source"] = "split"
    df = pd.concat([orig, rs], ignore_index=True)
    df = df[df["seed"] == 43022].copy()
    df["role2"] = df["arm"].map(role_from_arm)
    # Retain original V/R/C and split RS, dropping duplicate split C rows.
    df = df[~((df["source"] == "split") & (df["role2"] == "C"))].copy()
    df = df[df["checkpoint"].isin(["chck_80M", "chck_90M", "chck_100M"])].copy()
    return df


def summarize_series(xs: pd.Series) -> dict[str, Any]:
    xs = xs.dropna().astype(float)
    n = int(len(xs))
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "mean": float(xs.mean()),
        "median": float(xs.median()),
        "sd": float(xs.std(ddof=1)) if n > 1 else float("nan"),
        "se": float(xs.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan"),
        "q10": float(xs.quantile(0.10)),
        "q25": float(xs.quantile(0.25)),
        "q75": float(xs.quantile(0.75)),
        "q90": float(xs.quantile(0.90)),
        "frac_positive": float((xs > 0).mean()),
        "frac_negative": float((xs < 0).mean()),
    }


def pair_level(df: pd.DataFrame, out: pathlib.Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = df[df["token_class"] == "nonoverlap"].copy()
    pair_arm = (
        d.groupby(["seed", "checkpoint", "pair_id", "role2"], dropna=False)
        .agg(
            n_tokens=("probe_id", "count"),
            gain=("gain", "mean"),
            true_source_nll=("true_source_nll", "mean"),
            unrelated_source_nll=("unrelated_source_nll", "mean"),
        )
        .reset_index()
    )
    contrasts = [("RS", "C"), ("RS", "R"), ("R", "C"), ("V", "C"), ("V", "R")]
    rows = []
    for (seed, ck), g in pair_arm.groupby(["seed", "checkpoint"], dropna=False):
        roles = set(g["role2"])
        for a, b in contrasts:
            if a not in roles or b not in roles:
                continue
            aa = g[g["role2"] == a].rename(columns={"gain": "gain_a", "true_source_nll": "true_a", "unrelated_source_nll": "unrel_a"})
            bb = g[g["role2"] == b].rename(columns={"gain": "gain_b", "true_source_nll": "true_b", "unrelated_source_nll": "unrel_b"})
            j = aa[["pair_id", "gain_a", "true_a", "unrel_a"]].merge(bb[["pair_id", "gain_b", "true_b", "unrel_b"]], on="pair_id", how="inner")
            j["seed"] = seed; j["checkpoint"] = ck; j["contrast"] = f"{a}minus{b}"
            j["gain_delta"] = j["gain_a"] - j["gain_b"]
            j["true_delta"] = j["true_a"] - j["true_b"]
            j["unrel_delta"] = j["unrel_a"] - j["unrel_b"]
            j["excess_true_cost"] = j["true_delta"] - j["unrel_delta"]
            rows.append(j)
    pc = pd.concat(rows, ignore_index=True)
    pc.to_csv(out / "repeat_split_pair_level_contrasts.csv", index=False)
    late_rows = []
    for (seed, contrast, pair_id), g in pc.groupby(["seed", "contrast", "pair_id"], dropna=False):
        if g["checkpoint"].nunique() != 3:
            continue
        late_rows.append({
            "seed": int(seed),
            "contrast": contrast,
            "pair_id": pair_id,
            "mean_gain_delta": float(g["gain_delta"].mean()),
            "mean_true_delta": float(g["true_delta"].mean()),
            "mean_unrel_delta": float(g["unrel_delta"].mean()),
            "mean_excess_true_cost": float(g["excess_true_cost"].mean()),
        })
    late_pair = pd.DataFrame(late_rows)
    summary_rows = []
    for (seed, contrast), g in late_pair.groupby(["seed", "contrast"], dropna=False):
        rec = {"seed": int(seed), "contrast": contrast}
        for col, pref in [("mean_gain_delta", "gain"), ("mean_true_delta", "true"), ("mean_unrel_delta", "unrel"), ("mean_excess_true_cost", "excess")]:
            rec.update({f"{pref}_{k}": v for k, v in summarize_series(g[col]).items()})
        summary_rows.append(rec)
    summ = pd.DataFrame(summary_rows)
    late_pair.to_csv(out / "repeat_split_pair_level_late_pairs.csv", index=False)
    summ.to_csv(out / "repeat_split_pair_level_late_summary.csv", index=False)
    return late_pair, summ


def strict_word(df: pd.DataFrame, out: pathlib.Path) -> pd.DataFrame:
    wm = pd.read_csv(WORD_MAP)
    d = df.merge(wm[["probe_id", "pair_id", "strict_word_nonoverlap"]], on=["probe_id", "pair_id"], how="inner")
    d = d[(d["token_class"] == "nonoverlap") & (d["strict_word_nonoverlap"] == True)].copy()
    rows = []
    contrasts = [("RS", "C"), ("RS", "R"), ("R", "C"), ("V", "C"), ("V", "R")]
    for (seed, ck), g in d.groupby(["seed", "checkpoint"], dropna=False):
        roles = set(g["role2"])
        for a, b in contrasts:
            if a not in roles or b not in roles:
                continue
            aa = g[g["role2"] == a][["probe_id", "gain", "true_source_nll", "unrelated_source_nll"]].rename(columns={"gain":"gain_a", "true_source_nll":"true_a", "unrelated_source_nll":"unrel_a"})
            bb = g[g["role2"] == b][["probe_id", "gain", "true_source_nll", "unrelated_source_nll"]].rename(columns={"gain":"gain_b", "true_source_nll":"true_b", "unrelated_source_nll":"unrel_b"})
            j = aa.merge(bb, on="probe_id", how="inner")
            gd = j["gain_a"] - j["gain_b"]
            td = j["true_a"] - j["true_b"]
            ud = j["unrel_a"] - j["unrel_b"]
            ex = td - ud
            rows.append({
                "seed": int(seed), "checkpoint": ck, "contrast": f"{a}minus{b}", "n_tokens": int(len(j)),
                "gain_delta": float(gd.mean()), "true_delta": float(td.mean()), "unrel_delta": float(ud.mean()),
                "excess_true_cost": float(ex.mean()), "frac_excess_positive_tokens": float((ex > 0).mean()),
            })
    byck = pd.DataFrame(rows)
    byck.to_csv(out / "repeat_split_strict_word_by_checkpoint.csv", index=False)
    late_rows = []
    for (seed, contrast), g in byck.groupby(["seed", "contrast"], dropna=False):
        late_rows.append({
            "seed": int(seed), "contrast": contrast, "n_checkpoints": int(g["checkpoint"].nunique()),
            "n_tokens_min": int(g["n_tokens"].min()),
            "mean_gain_delta": float(g["gain_delta"].mean()),
            "mean_true_delta": float(g["true_delta"].mean()),
            "mean_unrel_delta": float(g["unrel_delta"].mean()),
            "mean_excess_true_cost": float(g["excess_true_cost"].mean()),
            "mean_frac_excess_positive_tokens": float(g["frac_excess_positive_tokens"].mean()),
        })
    late = pd.DataFrame(late_rows)
    late.to_csv(out / "repeat_split_strict_word_late_summary.csv", index=False)
    return late


def write_robust_note(pair_summary: pd.DataFrame, strict_summary: pd.DataFrame, out: pathlib.Path) -> None:
    lines = []
    lines.append("# research REPEAT_SPLIT pair-level and strict-word robustness")
    lines.append("")
    lines.append("This CPU-only check repeats the research robustness logic on the research REPEAT_SPLIT scored rows. It compares RS with CLEAN and original REPEAT in seed43022 over 80M/90M/100M.")
    lines.append("")
    lines.append("## Pair-level tokenizer-nonoverlap distributions")
    lines.append("")
    lines.append("| contrast | n pairs | mean gain Δ | median gain Δ | mean true Δ | mean unrel Δ | mean excess true cost | median excess | frac excess>0 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in pair_summary.sort_values("contrast").iterrows():
        lines.append(f"| {r['contrast']} | {int(r['gain_n'])} | {fmt(r['gain_mean'])} | {fmt(r['gain_median'])} | {fmt(r['true_mean'])} | {fmt(r['unrel_mean'])} | {fmt(r['excess_mean'])} | {fmt(r['excess_median'])} | {float(r['excess_frac_positive']):.3f} |")
    lines.append("")
    lines.append("## Strict word-level nonoverlap")
    lines.append("")
    lines.append("| contrast | n tokens min | gain Δ | true Δ | unrel Δ | excess true cost | frac tokens excess>0 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for _, r in strict_summary.sort_values("contrast").iterrows():
        lines.append(f"| {r['contrast']} | {int(r['n_tokens_min'])} | {fmt(r['mean_gain_delta'])} | {fmt(r['mean_true_delta'])} | {fmt(r['mean_unrel_delta'])} | {fmt(r['mean_excess_true_cost'])} | {float(r['mean_frac_excess_positive_tokens']):.3f} |")
    lines.append("")
    lines.append("Reading: RS−C remains near zero on tokenizer-nonoverlap gain at pair level and its excess true-source cost is near zero rather than original-REPEAT-like. RS−R is strongly positive in gain because RS removes the true-source cost while retaining lower unrelated-source NLL. The strict word filter preserves this reading: removing in-window exact recurrence eliminates the excess true-source cost rather than merely moving it into lexical-overlap artifacts.")
    lines.append("")
    lines.append(f"Data outputs: `{rel(out)}`.")
    (_public_path('research/notes/relation_learning/repeat_split_robustness.md')).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_robustness() -> None:
    out = _public_path('experiments/archive/relation_learning/data/repeat_split_robustness')
    out.mkdir(parents=True, exist_ok=True)
    df = load_rewrite_for_robustness()
    df.to_csv(out / "repeat_split_robustness_input_rows.csv", index=False)
    pair_rows, pair_summary = pair_level(df, out)
    strict_summary = strict_word(df, out)
    write_robust_note(pair_summary, strict_summary, out)
    result = {
        "status": "REPEAT_SPLIT_ROBUSTNESS_DONE",
        "outputs": rel(out),
        "note": rel(_public_path('research/notes/relation_learning/repeat_split_robustness.md')),
        "pair_summary": rel(out / "repeat_split_pair_level_late_summary.csv"),
        "strict_summary": rel(out / "repeat_split_strict_word_late_summary.csv"),
        "key_pair_rows": pair_summary[pair_summary["contrast"].isin(["RSminusC", "RSminusR", "RminusC"])].to_dict(orient="records"),
        "key_strict_rows": strict_summary[strict_summary["contrast"].isin(["RSminusC", "RSminusR", "RminusC"])].to_dict(orient="records"),
    }
    (out / "repeat_split_robustness_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": result["status"], "note": result["note"], "outputs": result["outputs"]}, indent=2), flush=True)


def run_score(seed: int, arm: str, device: str, batch_size: int, plan_only: bool) -> None:
    run = SPLIT_RUN[(seed, arm)]
    clean = CLEAN_RUN[seed]
    label = "D_RS" if arm == "repeat_split" else "D_VS"
    split_arm = f"{label}_{seed}"
    clean_arm = f"D_C_{seed}"
    out_dir = OUT_ROOT / f"split_{arm}_seed{seed}_probes"
    ready = (run / "hf_model/chck_100M/model.safetensors").exists() or (run / "hf_model/chck_100M/pytorch_model.bin").exists()
    if not ready:
        raise SystemExit(f"split arm is not ready: {run}")
    if not ((clean / "hf_model/chck_100M/model.safetensors").exists() or (clean / "hf_model/chck_100M/pytorch_model.bin").exists()):
        raise SystemExit(f"clean baseline missing: {clean}")
    base.ARM_CONFIGS.clear()
    base.ARM_CONFIGS.update({split_arm: run, clean_arm: clean})
    base.OUT_DEFAULT = out_dir
    sys.argv = [
        "split_robustness_and_seed_scorer.py",
        "--arms", split_arm, clean_arm,
        "--checkpoints", "chck_80M", "chck_90M", "chck_100M",
        "--gpu", device.replace("cuda:", ""),
        "--batch-size", str(batch_size),
    ]
    if plan_only:
        sys.argv.append("--plan-only")
    base.main()
    print(json.dumps({"status": "SPLIT_SEED_SCORE_DONE", "seed": seed, "arm": arm, "out_dir": rel(out_dir)}, indent=2), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("robustness")
    sc = sub.add_parser("score")
    sc.add_argument("--seed", type=int, required=True, choices=[43022, 43122, 43222])
    sc.add_argument("--arm", required=True, choices=["repeat_split", "view_split"])
    sc.add_argument("--device", default="cuda:0")
    sc.add_argument("--batch-size", type=int, default=48)
    sc.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()
    if args.cmd == "robustness":
        run_robustness()
    elif args.cmd == "score":
        run_score(args.seed, args.arm, args.device, args.batch_size, args.plan_only)


if __name__ == "__main__":
    main()
