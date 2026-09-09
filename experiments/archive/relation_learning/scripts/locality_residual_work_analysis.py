#!/usr/bin/env python3
"""research locality x residual-work analysis from existing probe rows.

Compares original REPEAT, REPEAT_SPLIT, CLEAN, and VIEW at seed43022 using scored
held-out copy/rewrite probes. The aim is to separate:
  (i) in-window relation practice (R vs RS with identical selected tokens/budget),
  (ii) generic cross-row repeated-content quality (RS vs C), and
  (iii) copy-gain interpretation after normalizing by each arm's own control NLL.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path("experiments/archive/relation_learning")
OUT = ROOT / "data/locality_residual_work"
NOTE = (ROOT.parents[2] / 'research/notes/relation_learning/locality_residual_work.md')
ORIG_COPY = ROOT / "data/heldout_copy_rewrite_entity_ablation/copy_summary.csv"
ORIG_REWRITE = ROOT / "data/heldout_copy_rewrite_entity_ablation/rewrite_summary.csv"
RS_COPY = ROOT / "data/split_repeat_split_probes/copy_summary.csv"
RS_REWRITE = ROOT / "data/split_repeat_split_probes/rewrite_summary.csv"


def read_copy_late() -> pd.DataFrame:
    orig = pd.read_csv(ORIG_COPY)
    orig["source"] = "orig"
    rs = pd.read_csv(RS_COPY)
    if "arch" in rs.columns:
        rs = rs.drop(columns=["arch"])
    rs["source"] = "split"
    df = pd.concat([orig, rs], ignore_index=True)
    df = df[(df["seed"] == 43022) & (df["group"] == "ALL") & (df["checkpoint"].isin(["chck_80M", "chck_90M", "chck_100M"]))].copy()
    df["arm_role"] = df["arm"].map(lambda s: str(s).split("_")[1] if str(s).startswith("D_") else "?")
    df.loc[df["arm"].astype(str).str.contains("D_RS"), "arm_role"] = "RS"
    # The split scorer re-scores the same CLEAN seed43022 baseline. Keep a single
    # original-C row per checkpoint so contrasts are well-defined.
    df = df[~((df["source"] == "split") & (df["arm_role"] == "C"))].copy()
    df["norm_gain"] = df["mean_gain"] / df["mean_b_nll"]
    return df


def read_rewrite_late() -> pd.DataFrame:
    orig = pd.read_csv(ORIG_REWRITE)
    orig["source"] = "orig"
    rs = pd.read_csv(RS_REWRITE)
    if "arch" in rs.columns:
        rs = rs.drop(columns=["arch"])
    rs["source"] = "split"
    df = pd.concat([orig, rs], ignore_index=True)
    df = df[(df["seed"] == 43022) & (df["group"] == "token_nonoverlap") & (df["checkpoint"].isin(["chck_80M", "chck_90M", "chck_100M"]))].copy()
    df["arm_role"] = df["arm"].map(lambda s: str(s).split("_")[1] if str(s).startswith("D_") else "?")
    df.loc[df["arm"].astype(str).str.contains("D_RS"), "arm_role"] = "RS"
    # The split scorer re-scores the same CLEAN seed43022 baseline. Keep a single
    # original-C row per checkpoint so contrasts are well-defined.
    df = df[~((df["source"] == "split") & (df["arm_role"] == "C"))].copy()
    return df


def arm_mean(df: pd.DataFrame, role: str, cols: list[str]) -> dict[str, float]:
    g = df[df["arm_role"] == role]
    return {c: float(g[c].mean()) for c in cols}


def ckpt_contrasts(df: pd.DataFrame, roles: tuple[str, str], metrics: list[str]) -> pd.DataFrame:
    a, b = roles
    aa = df[df["arm_role"] == a].set_index("checkpoint")
    bb = df[df["arm_role"] == b].set_index("checkpoint")
    rows = []
    for ck in sorted(set(aa.index).intersection(set(bb.index))):
        rec: dict[str, Any] = {"checkpoint": ck, "contrast": f"{a}minus{b}"}
        for m in metrics:
            rec[f"{m}_delta"] = float(aa.loc[ck, m] - bb.loc[ck, m])
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize_contrasts(ck: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for c in ck.columns:
        if c in {"checkpoint", "contrast"}:
            continue
        vals = ck[c].astype(float)
        out[c] = float(vals.mean())
        out[c + "_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")
        out[c + "_values"] = ";".join(f"{r.checkpoint}:{getattr(r, c):.6g}" for r in ck.itertuples())
    return out


def fmt(x: Any, nd: int = 4) -> str:
    try:
        xf = float(x)
    except Exception:
        return str(x)
    if math.isnan(xf):
        return "NA"
    return f"{xf:+.{nd}f}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    copy_df = read_copy_late()
    rewrite_df = read_rewrite_late()
    copy_df.to_csv(OUT / "copy_late_arm_terms_seed43022_with_split.csv", index=False)
    rewrite_df.to_csv(OUT / "rewrite_nonoverlap_late_arm_terms_seed43022_with_split.csv", index=False)

    copy_metrics = ["mean_gain", "norm_gain", "mean_a_nll", "mean_b_nll"]
    rewrite_metrics = ["mean_gain", "mean_a_nll", "mean_b_nll"]
    copy_contrast_frames = []
    rewrite_contrast_frames = []
    for roles in [("R", "C"), ("RS", "C"), ("RS", "R"), ("V", "C"), ("V", "R")]:
        if set(roles).issubset(set(copy_df["arm_role"])):
            copy_contrast_frames.append(ckpt_contrasts(copy_df, roles, copy_metrics))
        if set(roles).issubset(set(rewrite_df["arm_role"])):
            rewrite_contrast_frames.append(ckpt_contrasts(rewrite_df, roles, rewrite_metrics))
    copy_ck = pd.concat(copy_contrast_frames, ignore_index=True)
    rewrite_ck = pd.concat(rewrite_contrast_frames, ignore_index=True)
    copy_ck.to_csv(OUT / "copy_late_checkpoint_contrasts_seed43022.csv", index=False)
    rewrite_ck.to_csv(OUT / "rewrite_nonoverlap_checkpoint_contrasts_seed43022.csv", index=False)

    copy_late_rows = []
    for contrast, g in copy_ck.groupby("contrast"):
        row = {"contrast": contrast, **summarize_contrasts(g)}
        copy_late_rows.append(row)
    rewrite_late_rows = []
    for contrast, g in rewrite_ck.groupby("contrast"):
        row = {"contrast": contrast, **summarize_contrasts(g)}
        rewrite_late_rows.append(row)
    copy_late = pd.DataFrame(copy_late_rows)
    rewrite_late = pd.DataFrame(rewrite_late_rows)
    copy_late.to_csv(OUT / "copy_late_contrast_summary_seed43022.csv", index=False)
    rewrite_late.to_csv(OUT / "rewrite_nonoverlap_contrast_summary_seed43022.csv", index=False)

    # Compact JSON with key quantities.
    arm_copy = {role: arm_mean(copy_df, role, copy_metrics) for role in sorted(set(copy_df["arm_role"]))}
    arm_rewrite = {role: arm_mean(rewrite_df, role, rewrite_metrics) for role in sorted(set(rewrite_df["arm_role"]))}
    key = {
        "status": "LOCALITY_RESIDUAL_WORK_ANALYSIS_DONE",
        "copy_arm_terms": arm_copy,
        "rewrite_nonoverlap_arm_terms": arm_rewrite,
        "copy_contrasts": copy_late.to_dict(orient="records"),
        "rewrite_nonoverlap_contrasts": rewrite_late.to_dict(orient="records"),
        "files": {
            "copy_terms": str(OUT / "copy_late_arm_terms_seed43022_with_split.csv"),
            "rewrite_terms": str(OUT / "rewrite_nonoverlap_late_arm_terms_seed43022_with_split.csv"),
            "copy_contrasts": str(OUT / "copy_late_contrast_summary_seed43022.csv"),
            "rewrite_contrasts": str(OUT / "rewrite_nonoverlap_contrast_summary_seed43022.csv"),
            "note": str(NOTE),
        },
    }
    (OUT / "locality_residual_work_summary.json").write_text(json.dumps(key, indent=2, ensure_ascii=False), encoding="utf-8")

    # Extract named rows for note.
    def row(df: pd.DataFrame, contrast: str) -> dict[str, Any]:
        recs = df[df["contrast"] == contrast].to_dict(orient="records")
        return recs[0] if recs else {}
    r_c_copy = row(copy_late, "RminusC")
    rs_c_copy = row(copy_late, "RSminusC")
    rs_r_copy = row(copy_late, "RSminusR")
    rs_r_rew = row(rewrite_late, "RSminusR")
    rs_c_rew = row(rewrite_late, "RSminusC")
    r_c_rew = row(rewrite_late, "RminusC")

    lines = []
    lines.append("# research locality × residual-prediction-work readout")
    lines.append("")
    lines.append("This analysis reuses the scored held-out copy and compact-rewrite rows from research and research. No model forward passes are run. It asks what REPEAT_SPLIT means relative to original REPEAT, not only relative to CLEAN.")
    lines.append("")
    lines.append("## 1. In-window recurrence reduces learning from the repeated content itself")
    lines.append("")
    lines.append("Seed43022 token-nonoverlap compact-rewrite terms, averaged over 80M/90M/100M:")
    lines.append("")
    lines.append("| contrast | gain Δ (U−T) | true-source NLL Δ | unrelated-source NLL Δ | reading |")
    lines.append("|---|---:|---:|---:|---|")
    lines.append(f"| original R−C | {fmt(r_c_rew.get('mean_gain_delta'))} | {fmt(r_c_rew.get('mean_a_nll_delta'))} | {fmt(r_c_rew.get('mean_b_nll_delta'))} | original in-window exact recurrence: true source becomes worse while unrelated source improves |")
    lines.append(f"| REPEAT_SPLIT−C | {fmt(rs_c_rew.get('mean_gain_delta'))} | {fmt(rs_c_rew.get('mean_a_nll_delta'))} | {fmt(rs_c_rew.get('mean_b_nll_delta'))} | same repeated tokens across rows: no source-specific cost, both terms improve |")
    lines.append(f"| REPEAT_SPLIT−original R | {fmt(rs_r_rew.get('mean_gain_delta'))} | {fmt(rs_r_rew.get('mean_a_nll_delta'))} | {fmt(rs_r_rew.get('mean_b_nll_delta'))} | same selected tokens/budget but no in-window copy: true-source NLL is about one nat lower and unrelated-source NLL is also lower |")
    lines.append("")
    lines.append("The RS−R true-source improvement is the new residual-work signal. The exact same selected source and companion tokens are more useful when they are spaced into separate rows than when an exact companion sits in the same masked-LM window. In-window identity supplies an easy copy route for masked tokens, so it exhausts residual prediction work in those rows and weakens ordinary content learning; cross-row repetition preserves exposure without creating the local shortcut.")
    lines.append("")
    lines.append("## 2. Copy gain should be normalized by each arm's own control level")
    lines.append("")
    lines.append("Held-out natural-copy terms over never-trained rows, averaged over 80M/90M/100M:")
    lines.append("")
    lines.append("| arm | raw copy gain | normalized gain = gain/control NLL | repeated-source NLL | unrepeated-control NLL |")
    lines.append("|---|---:|---:|---:|---:|")
    for role in ["C", "R", "RS", "V"]:
        vals = arm_copy.get(role, {})
        if vals:
            lines.append(f"| {role} | {fmt(vals.get('mean_gain'))} | {fmt(vals.get('norm_gain'))} | {fmt(vals.get('mean_a_nll'))} | {fmt(vals.get('mean_b_nll'))} |")
    lines.append("")
    lines.append("Contrasts:")
    lines.append("")
    lines.append("| contrast | raw gain Δ | normalized gain Δ | repeated NLL Δ | control NLL Δ |")
    lines.append("|---|---:|---:|---:|---:|")
    for rec in [r_c_copy, rs_c_copy, rs_r_copy]:
        if rec:
            lines.append(f"| {rec['contrast']} | {fmt(rec.get('mean_gain_delta'))} | {fmt(rec.get('norm_gain_delta'))} | {fmt(rec.get('mean_a_nll_delta'))} | {fmt(rec.get('mean_b_nll_delta'))} |")
    lines.append("")
    lines.append("The raw RS−C copy-gain contrast is negative, but this should not be read as lower copy ability alone: RS has a lower unrepeated-control NLL than C, leaving less room for a nat-scale gain. After normalization RS remains below original R by about 0.061 of its control level, while RS is only slightly below C (about −0.036). Thus the strong statement is not that split repetition destroys copy; it is that the original REPEAT copy gain requires in-window co-occurrence, while cross-row repetition primarily improves ordinary control fit rather than installing a large local copy advantage.")
    lines.append("")
    lines.append("## 3. Mechanistic implication")
    lines.append("")
    lines.append("Exact recurrence inside a window has two separable costs under a fixed budget: (i) it installs recognized-source identity routing, which creates the T/U sign reversal and source-token misfire on nonidentical targets; and (ii) it reduces residual prediction work available for ordinary content learning, because the same-window exact copy lets masked tokens be solved locally. Spaced cross-row repetition of the same tokens at this dose preserves the content-exposure benefit and removes the in-window shortcut. This reconnects the current BabyLM mechanism to the inherited frontier_consolidation residual-work principle in a concrete training-window form.")
    lines.append("")
    lines.append("Data outputs: `experiments/archive/relation_learning/data/locality_residual_work`.")
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": key["status"], "note": str(NOTE), "outputs": str(OUT)}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
