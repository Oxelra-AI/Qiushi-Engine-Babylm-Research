#!/usr/bin/env python3
"""research: overlap-gradient analysis for the relation-practice mixture hypothesis.

Uses already-scored held-out compact rewrite probe rows. Joins each pair to its
source/rewrite texts and computes word-level overlap. Then asks whether R/C/V/RS
source-conditioning contrasts vary across overlap bins, especially for token
nonoverlap targets where identity readout should misfire most when source and
rewrite are similar enough to trigger recognition.
"""
from __future__ import annotations

import json
import math
import pathlib
import re
from typing import Any

import pandas as pd

ROOT = pathlib.Path("experiments/archive/relation_learning")
ALL_ACCEPTED = pathlib.Path("experiments/archive/frontier_consolidation/data/expansion_analysis/combined_all_accepted_pairs.jsonl")
ORIG_ROWS = ROOT / "data/heldout_copy_rewrite_entity_ablation/rewrite_pair_rows.csv"
RS_ROWS = ROOT / "data/split_repeat_split_probes/rewrite_pair_rows.csv"
OUT = ROOT / "data/overlap_gradient"
NOTE = (ROOT.parents[2] / 'research/notes/relation_learning/overlap_gradient_analysis.md')
CKPTS = ["chck_80M", "chck_90M", "chck_100M"]
CONTRASTS = [("R", "C"), ("RS", "C"), ("RS", "R"), ("V", "C"), ("V", "R")]
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "than", "that", "this", "these", "those",
    "of", "in", "on", "at", "to", "for", "from", "with", "without", "by", "as", "is", "are", "was",
    "were", "be", "been", "being", "it", "its", "they", "them", "their", "he", "she", "his", "her",
    "we", "you", "i", "not", "no", "do", "does", "did", "can", "could", "would", "should", "will",
    "so", "also", "one", "two", "many", "more", "most", "such", "into", "over", "under", "up", "down",
}


def read_jsonl(path: pathlib.Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def canonical_pair_id(obj: dict[str, Any]) -> str:
    pid = str(obj.get("pair_id") or obj.get("prompt_id") or f"sid:{obj.get('sentence_id')}|doc:{obj.get('doc_id')}")
    return pid if pid.startswith("compact:") else f"compact:{pid}"


def norm_word(w: str) -> str:
    w = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", w).lower()
    if w.endswith("'s"):
        w = w[:-2]
    if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    return w


def words(text: str) -> list[str]:
    out: list[str] = []
    for m in re.finditer(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", str(text)):
        w = norm_word(m.group(0))
        if w and w not in STOPWORDS and len(w) >= 2:
            out.append(w)
    return out


def pair_map(needed: set[str]) -> pd.DataFrame:
    rows = []
    for obj in read_jsonl(ALL_ACCEPTED):
        pid = canonical_pair_id(obj)
        if pid not in needed:
            continue
        src = str(obj.get("source_text", ""))
        rew = str(obj.get("rewrite_text", ""))
        sw = words(src); rw = words(rew)
        ss = set(sw); rs = set(rw)
        inter = ss & rs
        union = ss | rs
        rows.append({
            "pair_id": pid,
            "source_content_word_types": len(ss),
            "rewrite_content_word_types": len(rs),
            "shared_content_word_types": len(inter),
            "word_jaccard": len(inter) / len(union) if union else float("nan"),
            "rewrite_overlap_frac": len(inter) / len(rs) if rs else float("nan"),
            "source_overlap_frac": len(inter) / len(ss) if ss else float("nan"),
            "rewrite_new_frac": 1.0 - (len(inter) / len(rs) if rs else float("nan")),
            "source_words_raw": len(str(src).split()),
            "rewrite_words_raw": len(str(rew).split()),
        })
        if len(rows) == len(needed):
            break
    return pd.DataFrame(rows)


def role_from_arm(arm: str) -> str:
    s = str(arm)
    if "D_RS" in s:
        return "RS"
    if "D_VS" in s:
        return "VS"
    parts = s.split("_")
    return parts[1] if len(parts) > 1 else "?"


def load_rows() -> pd.DataFrame:
    orig = pd.read_csv(ORIG_ROWS)
    orig["row_source"] = "orig"
    rs = pd.read_csv(RS_ROWS)
    if "arch" in rs.columns:
        rs = rs.drop(columns=["arch"])
    rs["row_source"] = "split"
    df = pd.concat([orig, rs], ignore_index=True)
    df = df[(df["seed"] == 43022) & (df["checkpoint"].isin(CKPTS))].copy()
    df["role2"] = df["arm"].map(role_from_arm)
    # split scorer contains duplicate C rows; keep the original C baseline.
    df = df[~((df["row_source"] == "split") & (df["role2"] == "C"))].copy()
    return df


def add_bins(meta: pd.DataFrame) -> pd.DataFrame:
    meta = meta.copy()
    # Fixed bins for intuitive reading, plus tertiles for counts.
    bins = [-0.001, 0.25, 0.50, 0.75, 1.001]
    labels = ["low_<25", "mid_25_50", "high_50_75", "very_high_75_100"]
    meta["overlap_fixed_bin"] = pd.cut(meta["rewrite_overlap_frac"], bins=bins, labels=labels)
    # Compact rewrites may cluster; tertiles preserve resolution.
    try:
        meta["overlap_tertile"] = pd.qcut(meta["rewrite_overlap_frac"], q=3, labels=["tertile_low", "tertile_mid", "tertile_high"], duplicates="drop")
    except Exception:
        meta["overlap_tertile"] = "all"
    return meta


def pair_arm_rows(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["seed", "checkpoint", "pair_id", "role2", "token_class"], dropna=False)
        .agg(
            n_tokens=("probe_id", "count"),
            gain=("gain", "mean"),
            true_source_nll=("true_source_nll", "mean"),
            unrelated_source_nll=("unrelated_source_nll", "mean"),
        )
        .reset_index()
    )


def contrast_rows(pa: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (seed, ck, token_class), g in pa.groupby(["seed", "checkpoint", "token_class"], dropna=False):
        roles = set(g["role2"])
        for a, b in CONTRASTS:
            if a not in roles or b not in roles:
                continue
            aa = g[g["role2"] == a].rename(columns={"gain": "gain_a", "true_source_nll": "true_a", "unrelated_source_nll": "unrel_a"})
            bb = g[g["role2"] == b].rename(columns={"gain": "gain_b", "true_source_nll": "true_b", "unrelated_source_nll": "unrel_b"})
            j = aa[["pair_id", "gain_a", "true_a", "unrel_a"]].merge(bb[["pair_id", "gain_b", "true_b", "unrel_b"]], on="pair_id", how="inner")
            if j.empty:
                continue
            j["seed"] = seed; j["checkpoint"] = ck; j["token_class"] = token_class; j["contrast"] = f"{a}minus{b}"
            j["gain_delta"] = j["gain_a"] - j["gain_b"]
            j["true_delta"] = j["true_a"] - j["true_b"]
            j["unrel_delta"] = j["unrel_a"] - j["unrel_b"]
            j["excess_true_cost"] = j["true_delta"] - j["unrel_delta"]
            rows.append(j)
    cr = pd.concat(rows, ignore_index=True)
    cr = cr.merge(meta, on="pair_id", how="left")
    return cr


def summarize(cr: pd.DataFrame, bin_col: str) -> pd.DataFrame:
    rows = []
    for (token_class, contrast, binval), g in cr.groupby(["token_class", "contrast", bin_col], dropna=False):
        if len(g) == 0:
            continue
        # Average each pair over checkpoints first.
        pg = (
            g.groupby("pair_id", dropna=False)
            .agg(
                n_checkpoints=("checkpoint", "nunique"),
                rewrite_overlap_frac=("rewrite_overlap_frac", "first"),
                mean_gain_delta=("gain_delta", "mean"),
                mean_true_delta=("true_delta", "mean"),
                mean_unrel_delta=("unrel_delta", "mean"),
                mean_excess_true_cost=("excess_true_cost", "mean"),
            )
            .reset_index()
        )
        pg = pg[pg["n_checkpoints"] == len(CKPTS)]
        if pg.empty:
            continue
        rows.append({
            "token_class": token_class,
            "contrast": contrast,
            "bin_col": bin_col,
            "bin": str(binval),
            "n_pairs": int(len(pg)),
            "overlap_mean": float(pg["rewrite_overlap_frac"].mean()),
            "gain_mean": float(pg["mean_gain_delta"].mean()),
            "gain_median": float(pg["mean_gain_delta"].median()),
            "true_mean": float(pg["mean_true_delta"].mean()),
            "unrel_mean": float(pg["mean_unrel_delta"].mean()),
            "excess_mean": float(pg["mean_excess_true_cost"].mean()),
            "excess_median": float(pg["mean_excess_true_cost"].median()),
            "frac_gain_negative": float((pg["mean_gain_delta"] < 0).mean()),
            "frac_excess_positive": float((pg["mean_excess_true_cost"] > 0).mean()),
        })
    return pd.DataFrame(rows)


def corr_rows(cr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # pair-averaged correlations by token class and contrast.
    for (token_class, contrast), g in cr.groupby(["token_class", "contrast"], dropna=False):
        pg = (
            g.groupby("pair_id", dropna=False)
            .agg(
                n_checkpoints=("checkpoint", "nunique"),
                rewrite_overlap_frac=("rewrite_overlap_frac", "first"),
                mean_gain_delta=("gain_delta", "mean"),
                mean_excess_true_cost=("excess_true_cost", "mean"),
            )
            .reset_index()
        )
        pg = pg[(pg["n_checkpoints"] == len(CKPTS)) & pg["rewrite_overlap_frac"].notna()]
        if len(pg) < 10:
            continue
        rows.append({
            "token_class": token_class,
            "contrast": contrast,
            "n_pairs": int(len(pg)),
            "corr_overlap_gain_delta": float(pg["rewrite_overlap_frac"].corr(pg["mean_gain_delta"])),
            "corr_overlap_excess_true_cost": float(pg["rewrite_overlap_frac"].corr(pg["mean_excess_true_cost"])),
        })
    return pd.DataFrame(rows)


def f(x: Any, nd: int = 3) -> str:
    try:
        xf = float(x)
    except Exception:
        return str(x)
    if math.isnan(xf):
        return "NA"
    return f"{xf:+.{nd}f}"


def note(meta: pd.DataFrame, fixed: pd.DataFrame, tertile: pd.DataFrame, corr: pd.DataFrame) -> None:
    lines = []
    lines.append("# research overlap-gradient analysis")
    lines.append("")
    lines.append("This analysis joins existing held-out compact-rewrite probe rows with source/rewrite word-overlap metrics. It tests the mixture-model prediction that the identity readout should matter most when a nonidentical target sits in a source-recognizable, high-overlap pair.")
    lines.append("")
    lines.append("## Pair overlap distribution")
    lines.append("")
    lines.append(f"Pairs with overlap metadata: {len(meta)}. Mean rewrite content-word overlap fraction: {meta['rewrite_overlap_frac'].mean():.3f}; median {meta['rewrite_overlap_frac'].median():.3f}; Jaccard mean {meta['word_jaccard'].mean():.3f}.")
    lines.append("")
    lines.append("## Fixed overlap bins, nonoverlap targets")
    lines.append("")
    lines.append("| contrast | bin | n pairs | overlap mean | gain Δ | true Δ | unrel Δ | excess true cost | frac excess>0 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    show = fixed[(fixed["token_class"] == "nonoverlap") & (fixed["contrast"].isin(["RminusC", "RSminusC", "RSminusR", "VminusC", "VminusR"]))].copy()
    order = {"low_<25":0, "mid_25_50":1, "high_50_75":2, "very_high_75_100":3}
    show["_ord"] = show["bin"].map(order).fillna(9)
    for _, r in show.sort_values(["contrast", "_ord"]).iterrows():
        lines.append(f"| {r['contrast']} | {r['bin']} | {int(r['n_pairs'])} | {float(r['overlap_mean']):.3f} | {f(r['gain_mean'])} | {f(r['true_mean'])} | {f(r['unrel_mean'])} | {f(r['excess_mean'])} | {float(r['frac_excess_positive']):.3f} |")
    lines.append("")
    lines.append("## Correlations")
    lines.append("")
    lines.append("| token class | contrast | n pairs | corr(overlap, gain Δ) | corr(overlap, excess cost) |")
    lines.append("|---|---|---:|---:|---:|")
    for _, r in corr.sort_values(["token_class", "contrast"]).iterrows():
        lines.append(f"| {r['token_class']} | {r['contrast']} | {int(r['n_pairs'])} | {f(r['corr_overlap_gain_delta'])} | {f(r['corr_overlap_excess_true_cost'])} |")
    lines.append("")
    lines.append("## Reading")
    lines.append("")
    lines.append("If the mixture picture is right, overlap should not be treated as a simple monotone scalar good. At high enough overlap, the source is recognizable; if the target token is absent from the source, identity readout can hurt and content readout can help. RS should stay near CLEAN across bins because it has the same content exposure without local source/rewrite co-occurrence. The empirical bin table should be used to refine the natural variation-set bins: the strongest REPEAT deficit is expected where overlap is high enough for recognition but the target is nonidentical.")
    lines.append("")
    lines.append("Data outputs: `experiments/archive/relation_learning/data/overlap_gradient`.")
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_rows()
    needed = set(df["pair_id"].unique())
    meta = add_bins(pair_map(needed))
    meta.to_csv(OUT / "rewrite_pair_overlap_metrics.csv", index=False)
    pa = pair_arm_rows(df)
    pa = pa.merge(meta, on="pair_id", how="left")
    pa.to_csv(OUT / "rewrite_pair_arm_terms_with_overlap.csv", index=False)
    cr = contrast_rows(pair_arm_rows(df), meta)
    cr.to_csv(OUT / "rewrite_pair_contrasts_with_overlap.csv", index=False)
    fixed = summarize(cr, "overlap_fixed_bin")
    tertile = summarize(cr, "overlap_tertile")
    corr = corr_rows(cr)
    fixed.to_csv(OUT / "overlap_fixed_bin_summary.csv", index=False)
    tertile.to_csv(OUT / "overlap_tertile_summary.csv", index=False)
    corr.to_csv(OUT / "overlap_correlation_summary.csv", index=False)
    result = {
        "status": "OVERLAP_GRADIENT_DONE",
        "pairs": int(len(meta)),
        "mean_rewrite_overlap_frac": float(meta["rewrite_overlap_frac"].mean()),
        "median_rewrite_overlap_frac": float(meta["rewrite_overlap_frac"].median()),
        "files": {
            "metadata": str(OUT / "rewrite_pair_overlap_metrics.csv"),
            "fixed_bins": str(OUT / "overlap_fixed_bin_summary.csv"),
            "tertiles": str(OUT / "overlap_tertile_summary.csv"),
            "corr": str(OUT / "overlap_correlation_summary.csv"),
            "note": str(NOTE),
        },
    }
    (OUT / "overlap_gradient_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    note(meta, fixed, tertile, corr)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
