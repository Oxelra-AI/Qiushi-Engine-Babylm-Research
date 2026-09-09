#!/usr/bin/env python3
"""research: component-level checks requested by independent_review.

1. Decompose natural-copy raw gains into source-present NLL and no-source/control
   NLL for DeBERTa seeds 43022/43122/43222. This tests whether VIEW's raw copy
   gain is real source-present improvement or only worse no-source baseline.
2. Compute within-probe token-overlap vs token-nonoverlap interactions for compact
   rewrite targets. Identity readout predicts REPEAT should be relatively better
   on overlap targets but worse on nonoverlap targets.

Uses existing scored rows only; no model execution.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path("experiments/archive/relation_learning")
OUT = ROOT / "data/component_targetclass_checks"
NOTE = (ROOT.parents[2] / 'research/notes/relation_learning/component_targetclass_checks.md')
CKPTS = ["chck_80M", "chck_90M", "chck_100M"]

COPY_SOURCES = [
    ROOT / "data/heldout_copy_rewrite_entity_ablation/copy_pair_rows.csv",  # 43022,43122 V/C/R
    ROOT / "data/seed43222_probes/copy_pair_rows.csv",  # 43222 V/R
    ROOT / "data/seed43222_parallel_clean_probes/copy_pair_rows.csv",  # 43222 C
    ROOT / "data/split_repeat_split_probes/copy_pair_rows.csv",  # 43022 RS/C duplicate
]
REWRITE_SOURCES = [
    ROOT / "data/heldout_copy_rewrite_entity_ablation/rewrite_pair_rows.csv",
    ROOT / "data/seed43222_probes/rewrite_pair_rows.csv",
    ROOT / "data/seed43222_parallel_clean_probes/rewrite_pair_rows.csv",
    ROOT / "data/split_repeat_split_probes/rewrite_pair_rows.csv",
]


def role_from_arm(arm: str) -> str:
    s = str(arm)
    if "D_RS" in s:
        return "RS"
    if "D_VS" in s:
        return "VS"
    parts = s.split("_")
    return parts[1] if len(parts) > 1 else "?"


def read_concat(paths: list[Path]) -> pd.DataFrame:
    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        df["source_file"] = str(p)
        if "arch" in df.columns:
            df = df.drop(columns=["arch"])
        dfs.append(df)
    out = pd.concat(dfs, ignore_index=True)
    out["role2"] = out["arm"].map(role_from_arm)
    out = out[out["checkpoint"].isin(CKPTS)].copy()
    # Drop duplicate CLEAN rows introduced by split scorer for seed43022; keep research original C.
    out = out[~((out["source_file"].str.contains("split")) & (out["role2"] == "C"))].copy()
    # Keep one row if the same arm/checkpoint/probe was read from duplicate source; first source is canonical.
    keys = [c for c in ["arm", "checkpoint", "probe_id", "pair_id", "token_class", "span_len"] if c in out.columns]
    out = out.drop_duplicates(subset=keys, keep="first")
    return out


def mean_sd(vals: pd.Series) -> tuple[float, float]:
    vals = vals.astype(float)
    return float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")


def summarize_copy(copy: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    group = (
        copy.groupby(["seed", "role2", "checkpoint"], dropna=False)
        .agg(
            n=("probe_id", "count"),
            mean_gain=("gain", "mean"),
            repeated_nll=("repeated_nll", "mean"),
            control_nll=("unrepeated_nll", "mean"),
        )
        .reset_index()
    )
    group["norm_gain"] = group["mean_gain"] / group["control_nll"]
    group.to_csv(OUT / "copy_component_by_checkpoint.csv", index=False)
    late_rows = []
    for (seed, role), g in group.groupby(["seed", "role2"], dropna=False):
        if g["checkpoint"].nunique() != len(CKPTS):
            continue
        late_rows.append({
            "seed": int(seed), "role": role, "n_min": int(g["n"].min()),
            "mean_gain": float(g["mean_gain"].mean()),
            "norm_gain": float(g["norm_gain"].mean()),
            "repeated_nll": float(g["repeated_nll"].mean()),
            "control_nll": float(g["control_nll"].mean()),
        })
    late = pd.DataFrame(late_rows)
    late.to_csv(OUT / "copy_component_late_summary.csv", index=False)

    contrasts = []
    for seed, g in group.groupby("seed", dropna=False):
        roles = set(g["role2"])
        for a, b in [("R", "C"), ("V", "C"), ("V", "R"), ("RS", "C"), ("RS", "R")]:
            if a not in roles or b not in roles:
                continue
            aa = g[g["role2"] == a].set_index("checkpoint")
            bb = g[g["role2"] == b].set_index("checkpoint")
            vals = []
            for ck in CKPTS:
                if ck in aa.index and ck in bb.index:
                    rec = {"seed": int(seed), "contrast": f"{a}minus{b}", "checkpoint": ck}
                    for m in ["mean_gain", "norm_gain", "repeated_nll", "control_nll"]:
                        rec[f"{m}_delta"] = float(aa.loc[ck, m] - bb.loc[ck, m])
                    vals.append(rec)
            if len(vals) == len(CKPTS):
                dfv = pd.DataFrame(vals)
                rec2: dict[str, Any] = {"seed": int(seed), "contrast": f"{a}minus{b}"}
                for c in [x for x in dfv.columns if x.endswith("_delta")]:
                    rec2[c] = float(dfv[c].mean())
                    rec2[c + "_sd"] = float(dfv[c].std(ddof=1))
                contrasts.append(rec2)
    con = pd.DataFrame(contrasts)
    con.to_csv(OUT / "copy_component_late_contrasts.csv", index=False)
    return late, con


def summarize_rewrite_interaction(rewrite: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Token rows -> pair-level arm means by token class, then checkpoint means.
    pair_arm = (
        rewrite.groupby(["seed", "checkpoint", "pair_id", "role2", "token_class"], dropna=False)
        .agg(
            n_tokens=("probe_id", "count"),
            gain=("gain", "mean"),
            true_source_nll=("true_source_nll", "mean"),
            unrelated_source_nll=("unrelated_source_nll", "mean"),
        )
        .reset_index()
    )
    rows = []
    for (seed, ck, token_class), g in pair_arm.groupby(["seed", "checkpoint", "token_class"], dropna=False):
        roles = set(g["role2"])
        for a, b in [("R", "C"), ("V", "C"), ("V", "R"), ("RS", "C"), ("RS", "R")]:
            if a not in roles or b not in roles:
                continue
            aa = g[g["role2"] == a].rename(columns={"gain":"gain_a", "true_source_nll":"true_a", "unrelated_source_nll":"unrel_a"})
            bb = g[g["role2"] == b].rename(columns={"gain":"gain_b", "true_source_nll":"true_b", "unrelated_source_nll":"unrel_b"})
            j = aa[["pair_id", "gain_a", "true_a", "unrel_a"]].merge(bb[["pair_id", "gain_b", "true_b", "unrel_b"]], on="pair_id", how="inner")
            if j.empty:
                continue
            gd = j["gain_a"] - j["gain_b"]
            td = j["true_a"] - j["true_b"]
            ud = j["unrel_a"] - j["unrel_b"]
            rows.append({
                "seed": int(seed), "checkpoint": ck, "token_class": token_class, "contrast": f"{a}minus{b}", "n_pairs": int(len(j)),
                "gain_delta": float(gd.mean()), "true_delta": float(td.mean()), "unrel_delta": float(ud.mean()),
                "excess_true_cost": float((td - ud).mean()),
            })
    byck = pd.DataFrame(rows)
    byck.to_csv(OUT / "rewrite_targetclass_by_checkpoint.csv", index=False)
    late_rows = []
    for (seed, contrast, token_class), g in byck.groupby(["seed", "contrast", "token_class"], dropna=False):
        if g["checkpoint"].nunique() != len(CKPTS):
            continue
        late_rows.append({
            "seed": int(seed), "contrast": contrast, "token_class": token_class, "n_pairs_min": int(g["n_pairs"].min()),
            "gain_delta": float(g["gain_delta"].mean()),
            "true_delta": float(g["true_delta"].mean()),
            "unrel_delta": float(g["unrel_delta"].mean()),
            "excess_true_cost": float(g["excess_true_cost"].mean()),
        })
    late = pd.DataFrame(late_rows)
    late.to_csv(OUT / "rewrite_targetclass_late_summary.csv", index=False)

    # overlap-vs-nonoverlap difference of gain_delta within same contrast/seed.
    ints = []
    for (seed, contrast), g in late.groupby(["seed", "contrast"], dropna=False):
        d = {r["token_class"]: r for _, r in g.iterrows()}
        if "overlap" in d and "nonoverlap" in d:
            ov = d["overlap"]; no = d["nonoverlap"]
            ints.append({
                "seed": int(seed), "contrast": contrast,
                "overlap_gain_delta": float(ov["gain_delta"]),
                "nonoverlap_gain_delta": float(no["gain_delta"]),
                "overlap_minus_nonoverlap_gain_delta": float(ov["gain_delta"] - no["gain_delta"]),
                "overlap_true_delta": float(ov["true_delta"]),
                "nonoverlap_true_delta": float(no["true_delta"]),
                "overlap_minus_nonoverlap_true_delta": float(ov["true_delta"] - no["true_delta"]),
                "overlap_excess": float(ov["excess_true_cost"]),
                "nonoverlap_excess": float(no["excess_true_cost"]),
            })
    inter = pd.DataFrame(ints)
    inter.to_csv(OUT / "rewrite_overlap_nonoverlap_interaction.csv", index=False)
    return late, inter


def f(x: Any, nd: int = 4) -> str:
    try:
        xf = float(x)
    except Exception:
        return str(x)
    if math.isnan(xf):
        return "NA"
    return f"{xf:+.{nd}f}"


def write_note(copy_late: pd.DataFrame, copy_con: pd.DataFrame, rw_late: pd.DataFrame, inter: pd.DataFrame) -> None:
    lines = []
    lines.append("# research component-level copy and target-class checks")
    lines.append("")
    lines.append("These are existing-data checks prompted by independent_review. They prevent two overstatements: treating raw copy gain as source-present copy ability, and treating the mixture model as established without the clean overlap/nonoverlap target interaction.")
    lines.append("")
    lines.append("## Natural-copy components across DeBERTa seeds")
    lines.append("")
    lines.append("Copy gain = no-source/control NLL minus source-present repeated NLL. Source-present NLL and control NLL are reported separately.")
    lines.append("")
    lines.append("| seed | role | raw gain | normalized gain | source-present NLL | no-source/control NLL |")
    lines.append("|---:|---|---:|---:|---:|---:|")
    for _, r in copy_late.sort_values(["seed", "role"]).iterrows():
        lines.append(f"| {int(r['seed'])} | {r['role']} | {f(r['mean_gain'])} | {f(r['norm_gain'])} | {f(r['repeated_nll'])} | {f(r['control_nll'])} |")
    lines.append("")
    lines.append("## Copy component contrasts")
    lines.append("")
    lines.append("| seed | contrast | raw gain Δ | norm gain Δ | source-present NLL Δ | control NLL Δ |")
    lines.append("|---:|---|---:|---:|---:|---:|")
    for _, r in copy_con.sort_values(["seed", "contrast"]).iterrows():
        lines.append(f"| {int(r['seed'])} | {r['contrast']} | {f(r['mean_gain_delta'])} | {f(r['norm_gain_delta'])} | {f(r['repeated_nll_delta'])} | {f(r['control_nll_delta'])} |")
    lines.append("")
    lines.append("Reading: REPEAT's raw copy advantage over CLEAN is supported by source-present NLL improvement in seeds43022/43122 and by raw gain in seed43222. VIEW's raw copy advantage over CLEAN is not a clean source-present copy improvement in the first two seeds; its component terms are close to or worse than CLEAN while the no-source control is also worse. Therefore VIEW's positive copy statement should be weak and component-qualified.")
    lines.append("")
    lines.append("## Compact-rewrite target-class interaction")
    lines.append("")
    lines.append("| seed | contrast | overlap gain Δ | nonoverlap gain Δ | overlap−nonoverlap gain Δ | overlap true Δ | nonoverlap true Δ |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|")
    for _, r in inter.sort_values(["seed", "contrast"]).iterrows():
        lines.append(f"| {int(r['seed'])} | {r['contrast']} | {f(r['overlap_gain_delta'])} | {f(r['nonoverlap_gain_delta'])} | {f(r['overlap_minus_nonoverlap_gain_delta'])} | {f(r['overlap_true_delta'])} | {f(r['nonoverlap_true_delta'])} |")
    lines.append("")
    lines.append("Reading: The clean within-probe interaction is strong for original R−C in DeBERTa seeds: REPEAT is less harmful or even source-helpful on overlap targets but strongly harmful on nonoverlap targets. This is direct support for the identity-readout interpretation. V−C is positive on both token classes and usually larger on overlap targets, so VIEW is not a pure nonoverlap-only transformation arm; it improves source-conditioned use broadly on compact rewrites. RS−C lacks the original nonoverlap sign reversal and is source-helpful on overlap tokens, consistent with cross-row content exposure without local identity misfire.")
    lines.append("")
    lines.append("Data outputs: `experiments/archive/relation_learning/data/component_targetclass_checks`.")
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    copy = read_concat(COPY_SOURCES)
    rewrite = read_concat(REWRITE_SOURCES)
    copy.to_csv(OUT / "copy_component_input_rows.csv", index=False)
    rewrite.to_csv(OUT / "rewrite_targetclass_input_rows.csv", index=False)
    copy_late, copy_con = summarize_copy(copy)
    rw_late, inter = summarize_rewrite_interaction(rewrite)
    write_note(copy_late, copy_con, rw_late, inter)
    result = {
        "status": "COMPONENT_TARGETCLASS_CHECKS_DONE",
        "note": str(NOTE),
        "outputs": str(OUT),
        "copy_components": str(OUT / "copy_component_late_summary.csv"),
        "copy_contrasts": str(OUT / "copy_component_late_contrasts.csv"),
        "targetclass_interaction": str(OUT / "rewrite_overlap_nonoverlap_interaction.csv"),
    }
    (OUT / "component_targetclass_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
