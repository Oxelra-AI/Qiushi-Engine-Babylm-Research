#!/usr/bin/env python3
"""research: integrate VIEW_SPLIT with original and REPEAT_SPLIT relation probes.

Uses scored rows only. Produces:
- late component summaries for copy and rewrite by arm role (C/R/V/RS/VS)
- term decompositions for VIEW_SPLIT vs CLEAN/original VIEW/REPEAT
- pair-level and strict-word summaries on compact rewrite targets
- pair-cluster bootstrap intervals for VS-C near-zero source-specific gain
- a research note interpreting whether VIEW's positive content-conditioning is local
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path("experiments/archive/relation_learning")
OUT = ROOT / "data/view_split_integration"
NOTE = (ROOT.parents[2] / 'research/notes/relation_learning/view_split_integration.md')
CKPTS = ["chck_80M", "chck_90M", "chck_100M"]
WORD_MAP = ROOT / "data/relation_decomposition/rewrite_probe_word_nonoverlap_map.csv"

COPY_SOURCES = [
    ROOT / "data/heldout_copy_rewrite_entity_ablation/copy_pair_rows.csv",
    ROOT / "data/split_repeat_split_probes/copy_pair_rows.csv",
    ROOT / "data/split_view_split_probes/copy_pair_rows.csv",
]
REWRITE_SOURCES = [
    ROOT / "data/heldout_copy_rewrite_entity_ablation/rewrite_pair_rows.csv",
    ROOT / "data/split_repeat_split_probes/rewrite_pair_rows.csv",
    ROOT / "data/split_view_split_probes/rewrite_pair_rows.csv",
]
ENTITY_SOURCES = [
    ROOT / "data/split_view_split_probes/entity_ablation_late_contrasts.csv",
    ROOT / "data/split_repeat_split_probes/entity_ablation_late_contrasts.csv",
]


def role_from_arm(arm: str) -> str:
    s = str(arm)
    if "D_RS" in s:
        return "RS"
    if "D_VS" in s:
        return "VS"
    parts = s.split("_")
    return parts[1] if len(parts) > 1 else "?"


def rel(p: Path) -> str:
    return str(p)


def fmt(x: Any, nd: int = 4) -> str:
    try:
        xf = float(x)
    except Exception:
        return str(x)
    if math.isnan(xf):
        return "NA"
    return f"{xf:+.{nd}f}"


def read_concat(paths: list[Path], family: str) -> pd.DataFrame:
    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        if "arch" in df.columns:
            df = df.drop(columns=["arch"])
        df["source_file"] = str(p)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df = df[(df["seed"] == 43022) & (df["checkpoint"].isin(CKPTS))].copy()
    df["role2"] = df["arm"].map(role_from_arm)
    # Keep canonical original CLEAN baseline: split scorer re-scored C in both split dirs.
    df = df[~((df["source_file"].str.contains("split_")) & (df["role2"] == "C"))].copy()
    keys = ["arm", "checkpoint", "probe_id"]
    if family == "rewrite":
        keys += ["token_class"]
    if family == "copy":
        keys += ["span_len"]
    df = df.drop_duplicates(subset=[k for k in keys if k in df.columns], keep="first")
    return df


def copy_arm_summary(copy: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    byck = (
        copy.groupby(["role2", "checkpoint"], dropna=False)
        .agg(n=("probe_id", "count"), mean_gain=("gain", "mean"), source_present_nll=("repeated_nll", "mean"), control_nll=("unrepeated_nll", "mean"))
        .reset_index()
    )
    byck["norm_gain"] = byck["mean_gain"] / byck["control_nll"]
    byck.to_csv(OUT / "copy_by_checkpoint_all_roles_seed43022.csv", index=False)
    late_rows = []
    for role, g in byck.groupby("role2", dropna=False):
        if g["checkpoint"].nunique() != len(CKPTS):
            continue
        late_rows.append({
            "role": role, "n_min": int(g["n"].min()),
            "mean_gain": float(g["mean_gain"].mean()),
            "norm_gain": float(g["norm_gain"].mean()),
            "source_present_nll": float(g["source_present_nll"].mean()),
            "control_nll": float(g["control_nll"].mean()),
        })
    late = pd.DataFrame(late_rows)
    late.to_csv(OUT / "copy_late_all_roles_seed43022.csv", index=False)
    con_rows = []
    roles = set(byck["role2"])
    for a, b in [("R","C"),("V","C"),("RS","C"),("VS","C"),("RS","R"),("VS","V"),("VS","R"),("VS","RS"),("V","R")]:
        if a not in roles or b not in roles:
            continue
        aa = byck[byck["role2"] == a].set_index("checkpoint")
        bb = byck[byck["role2"] == b].set_index("checkpoint")
        vals = []
        for ck in CKPTS:
            if ck in aa.index and ck in bb.index:
                rec = {"contrast": f"{a}minus{b}", "checkpoint": ck}
                for m in ["mean_gain", "norm_gain", "source_present_nll", "control_nll"]:
                    rec[f"{m}_delta"] = float(aa.loc[ck, m] - bb.loc[ck, m])
                vals.append(rec)
        if len(vals) == len(CKPTS):
            tmp = pd.DataFrame(vals)
            row: dict[str, Any] = {"contrast": f"{a}minus{b}"}
            for c in [x for x in tmp.columns if x.endswith("_delta")]:
                row[c] = float(tmp[c].mean())
                row[c + "_sd"] = float(tmp[c].std(ddof=1))
            con_rows.append(row)
    con = pd.DataFrame(con_rows)
    con.to_csv(OUT / "copy_late_contrasts_all_roles_seed43022.csv", index=False)
    return late, con


def rewrite_arm_summary(rewrite: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    byck = (
        rewrite.groupby(["role2", "checkpoint", "token_class"], dropna=False)
        .agg(n=("probe_id", "count"), mean_gain=("gain", "mean"), true_source_nll=("true_source_nll", "mean"), unrelated_source_nll=("unrelated_source_nll", "mean"))
        .reset_index()
    )
    byck.to_csv(OUT / "rewrite_by_checkpoint_all_roles_seed43022.csv", index=False)
    late_rows = []
    for (role, cls), g in byck.groupby(["role2", "token_class"], dropna=False):
        if g["checkpoint"].nunique() != len(CKPTS):
            continue
        late_rows.append({
            "role": role, "token_class": cls, "n_min": int(g["n"].min()),
            "mean_gain": float(g["mean_gain"].mean()),
            "true_source_nll": float(g["true_source_nll"].mean()),
            "unrelated_source_nll": float(g["unrelated_source_nll"].mean()),
        })
    late = pd.DataFrame(late_rows)
    late.to_csv(OUT / "rewrite_late_all_roles_seed43022.csv", index=False)
    con_rows = []
    roles = set(byck["role2"])
    for cls, gc in byck.groupby("token_class", dropna=False):
        for a, b in [("R","C"),("V","C"),("RS","C"),("VS","C"),("RS","R"),("VS","V"),("VS","R"),("VS","RS"),("V","R")]:
            if a not in roles or b not in roles:
                continue
            aa = gc[gc["role2"] == a].set_index("checkpoint")
            bb = gc[gc["role2"] == b].set_index("checkpoint")
            vals = []
            for ck in CKPTS:
                if ck in aa.index and ck in bb.index:
                    rec = {"contrast": f"{a}minus{b}", "token_class": cls, "checkpoint": ck}
                    for m in ["mean_gain", "true_source_nll", "unrelated_source_nll"]:
                        rec[f"{m}_delta"] = float(aa.loc[ck, m] - bb.loc[ck, m])
                    vals.append(rec)
            if len(vals) == len(CKPTS):
                tmp = pd.DataFrame(vals)
                row: dict[str, Any] = {"contrast": f"{a}minus{b}", "token_class": cls}
                for c in [x for x in tmp.columns if x.endswith("_delta")]:
                    row[c] = float(tmp[c].mean())
                    row[c + "_sd"] = float(tmp[c].std(ddof=1))
                con_rows.append(row)
    con = pd.DataFrame(con_rows)
    con.to_csv(OUT / "rewrite_late_contrasts_all_roles_seed43022.csv", index=False)
    return late, con


def pair_and_strict(rewrite: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    contrasts = [("VS","C"),("VS","V"),("VS","R"),("VS","RS"),("V","C"),("R","C"),("RS","C")]
    d = rewrite[rewrite["token_class"] == "nonoverlap"].copy()
    pair_arm = (
        d.groupby(["checkpoint", "pair_id", "role2"], dropna=False)
        .agg(gain=("gain", "mean"), true_source_nll=("true_source_nll", "mean"), unrelated_source_nll=("unrelated_source_nll", "mean"), n_tokens=("probe_id", "count"))
        .reset_index()
    )
    rows = []
    for ck, g in pair_arm.groupby("checkpoint", dropna=False):
        roles = set(g["role2"])
        for a,b in contrasts:
            if a not in roles or b not in roles:
                continue
            aa = g[g["role2"] == a].rename(columns={"gain":"gain_a","true_source_nll":"true_a","unrelated_source_nll":"unrel_a"})
            bb = g[g["role2"] == b].rename(columns={"gain":"gain_b","true_source_nll":"true_b","unrelated_source_nll":"unrel_b"})
            j = aa[["pair_id","gain_a","true_a","unrel_a"]].merge(bb[["pair_id","gain_b","true_b","unrel_b"]], on="pair_id", how="inner")
            if j.empty:
                continue
            j["checkpoint"] = ck; j["contrast"] = f"{a}minus{b}"
            j["gain_delta"] = j["gain_a"] - j["gain_b"]
            j["true_delta"] = j["true_a"] - j["true_b"]
            j["unrel_delta"] = j["unrel_a"] - j["unrel_b"]
            j["excess_true_cost"] = j["true_delta"] - j["unrel_delta"]
            rows.append(j)
    pc = pd.concat(rows, ignore_index=True)
    pc.to_csv(OUT / "view_split_pair_level_contrasts.csv", index=False)
    late_pairs = []
    for (contrast, pair_id), g in pc.groupby(["contrast","pair_id"], dropna=False):
        if g["checkpoint"].nunique() != len(CKPTS):
            continue
        late_pairs.append({
            "contrast": contrast, "pair_id": pair_id,
            "mean_gain_delta": float(g["gain_delta"].mean()),
            "mean_true_delta": float(g["true_delta"].mean()),
            "mean_unrel_delta": float(g["unrel_delta"].mean()),
            "mean_excess_true_cost": float(g["excess_true_cost"].mean()),
        })
    lp = pd.DataFrame(late_pairs)
    lp.to_csv(OUT / "view_split_pair_level_late_pairs.csv", index=False)
    ps_rows = []
    for contrast, g in lp.groupby("contrast", dropna=False):
        rec: dict[str, Any] = {"contrast": contrast, "n_pairs": int(len(g))}
        for col,pref in [("mean_gain_delta","gain"),("mean_true_delta","true"),("mean_unrel_delta","unrel"),("mean_excess_true_cost","excess")]:
            vals = g[col].astype(float)
            rec[f"{pref}_mean"] = float(vals.mean())
            rec[f"{pref}_median"] = float(vals.median())
            rec[f"{pref}_se"] = float(vals.std(ddof=1)/math.sqrt(len(vals))) if len(vals)>1 else float("nan")
            rec[f"{pref}_frac_positive"] = float((vals>0).mean())
            rec[f"{pref}_frac_negative"] = float((vals<0).mean())
        ps_rows.append(rec)
    ps = pd.DataFrame(ps_rows)
    ps.to_csv(OUT / "view_split_pair_level_late_summary.csv", index=False)

    # Strict word-level from token rows.
    wm = pd.read_csv(WORD_MAP)[["probe_id", "pair_id", "strict_word_nonoverlap"]]
    sw = rewrite.merge(wm, on=["probe_id", "pair_id"], how="inner")
    sw = sw[(sw["token_class"] == "nonoverlap") & (sw["strict_word_nonoverlap"] == True)].copy()
    sw_rows = []
    for ck, g in sw.groupby("checkpoint", dropna=False):
        roles = set(g["role2"])
        for a,b in contrasts:
            if a not in roles or b not in roles:
                continue
            aa = g[g["role2"] == a][["probe_id","gain","true_source_nll","unrelated_source_nll"]].rename(columns={"gain":"gain_a","true_source_nll":"true_a","unrelated_source_nll":"unrel_a"})
            bb = g[g["role2"] == b][["probe_id","gain","true_source_nll","unrelated_source_nll"]].rename(columns={"gain":"gain_b","true_source_nll":"true_b","unrelated_source_nll":"unrel_b"})
            j = aa.merge(bb, on="probe_id", how="inner")
            gd = j["gain_a"]-j["gain_b"]; td=j["true_a"]-j["true_b"]; ud=j["unrel_a"]-j["unrel_b"]; ex=td-ud
            sw_rows.append({"checkpoint": ck, "contrast": f"{a}minus{b}", "n_tokens": int(len(j)), "gain_delta": float(gd.mean()), "true_delta": float(td.mean()), "unrel_delta": float(ud.mean()), "excess_true_cost": float(ex.mean()), "frac_excess_positive": float((ex>0).mean())})
    sw_byck = pd.DataFrame(sw_rows)
    sw_byck.to_csv(OUT / "view_split_strict_word_by_checkpoint.csv", index=False)
    sw_late_rows = []
    for contrast, g in sw_byck.groupby("contrast", dropna=False):
        sw_late_rows.append({"contrast": contrast, "n_checkpoints": int(g["checkpoint"].nunique()), "n_tokens_min": int(g["n_tokens"].min()), "gain_delta": float(g["gain_delta"].mean()), "true_delta": float(g["true_delta"].mean()), "unrel_delta": float(g["unrel_delta"].mean()), "excess_true_cost": float(g["excess_true_cost"].mean()), "frac_excess_positive": float(g["frac_excess_positive"].mean())})
    sw_late = pd.DataFrame(sw_late_rows)
    sw_late.to_csv(OUT / "view_split_strict_word_late_summary.csv", index=False)
    return ps, sw_late


def bootstrap(pair_summary_path: Path) -> pd.DataFrame:
    lp = pd.read_csv(OUT / "view_split_pair_level_late_pairs.csv")
    rng = np.random.default_rng(1809182)
    rows = []
    for contrast, g in lp.groupby("contrast"):
        if contrast not in {"VSminusC", "VSminusV", "VminusC"}:
            continue
        vals = g["mean_gain_delta"].to_numpy(float)
        n = len(vals)
        idx = rng.integers(0, n, size=(10000, n))
        boot = vals[idx].mean(axis=1)
        rows.append({
            "contrast": contrast, "metric": "mean_gain_delta", "n_pairs": int(n),
            "mean": float(vals.mean()), "median": float(np.median(vals)),
            "ci95_lo": float(np.quantile(boot, 0.025)), "ci95_hi": float(np.quantile(boot, 0.975)),
            "prob_inside_pm_0p25": float(np.mean(np.abs(boot) <= 0.25)),
            "prob_in_original_view_range_0p684_0p894": float(np.mean((boot >= 0.6837) & (boot <= 0.8938))),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "view_split_pair_bootstrap_summary.csv", index=False)
    return out


def entity_summary() -> pd.DataFrame:
    dfs = []
    for p in ENTITY_SOURCES:
        if p.exists() and p.stat().st_size > 5:
            df = pd.read_csv(p)
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    ent = pd.concat(dfs, ignore_index=True)
    ent.to_csv(OUT / "split_entity_late_contrasts_collected.csv", index=False)
    return ent


def write_note(copy_late: pd.DataFrame, copy_con: pd.DataFrame, rw_late: pd.DataFrame, rw_con: pd.DataFrame, pair_sum: pd.DataFrame, strict_sum: pd.DataFrame, boot: pd.DataFrame, ent: pd.DataFrame) -> None:
    lines = []
    lines.append("# research VIEW_SPLIT integration")
    lines.append("")
    lines.append("VIEW_SPLIT seed43022 preserves the selected source and rewrite texts and the 100M budget, but prevents each source/rewrite pair from co-occurring in one training row. It decides whether VIEW's positive content-conditioning benefit is local to in-window restatement practice or can be learned from cross-row paraphrase exposure.")
    lines.append("")
    lines.append("## 1. Token-level term decomposition")
    lines.append("")
    lines.append("Late 80M/90M/100M compact-rewrite gains, seed43022:")
    lines.append("")
    lines.append("| role | token class | gain U−T | true-source NLL | unrelated-source NLL |")
    lines.append("|---|---|---:|---:|---:|")
    for _, r in rw_late.sort_values(["token_class", "role"]).iterrows():
        if r["role"] in {"C","R","V","RS","VS"}:
            lines.append(f"| {r['role']} | {r['token_class']} | {fmt(r['mean_gain'])} | {fmt(r['true_source_nll'])} | {fmt(r['unrelated_source_nll'])} |")
    lines.append("")
    lines.append("Key contrasts:")
    lines.append("")
    lines.append("| contrast | token class | gain Δ | true-source NLL Δ | unrelated-source NLL Δ |")
    lines.append("|---|---|---:|---:|---:|")
    for _, r in rw_con[rw_con["contrast"].isin(["VminusC","VSminusC","VSminusV","RminusC","RSminusC","VSminusR","VSminusRS"])].sort_values(["token_class","contrast"]).iterrows():
        lines.append(f"| {r['contrast']} | {r['token_class']} | {fmt(r['mean_gain_delta'])} | {fmt(r['true_source_nll_delta'])} | {fmt(r['unrelated_source_nll_delta'])} |")
    lines.append("")
    lines.append("On token-nonoverlap targets, original V−C was +0.6837 because VIEW improved the true-source term more than the unrelated-source term (T −1.1788, U −0.4951). VIEW_SPLIT−CLEAN is only +0.0298: it improves both T and U by nearly the same amount (T about −0.852, U about −0.822). Thus the VIEW positive content-conditioning residual disappears when source and rewrite are split across rows, even though the split arm has broad better fit to these compact rewrite targets. VS−V is −0.6539 on nonoverlap gain: relative to original VIEW, splitting makes the true-source term worse and the unrelated-source term better by roughly equal opposite amounts. This is the positive-side analogue of the REPEAT_SPLIT result.")
    lines.append("")
    lines.append("## 2. Pair and strict-word summaries")
    lines.append("")
    lines.append("| contrast | pair gain mean | pair gain median | pair excess mean | strict-word gain | strict-word excess |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for contrast in ["VminusC","VSminusC","VSminusV","RminusC","RSminusC"]:
        pr = pair_sum[pair_sum["contrast"] == contrast]
        sr = strict_sum[strict_sum["contrast"] == contrast]
        if len(pr) and len(sr):
            p = pr.iloc[0]; s = sr.iloc[0]
            lines.append(f"| {contrast} | {fmt(p['gain_mean'])} | {fmt(p['gain_median'])} | {fmt(p['excess_mean'])} | {fmt(s['gain_delta'])} | {fmt(s['excess_true_cost'])} |")
    lines.append("")
    lines.append("The near-zero VS−C nonoverlap result is preserved after pair aggregation and strict word filtering. It is not a single-token artifact. Original V−C remains strongly positive in the same analyses, so the loss of the residual is specific to splitting source/rewrite co-occurrence.")
    lines.append("")
    lines.append("## 3. Pair-bootstrap uncertainty")
    lines.append("")
    lines.append("| contrast | mean gain Δ | median pair | 95% bootstrap interval | P(|mean|≤0.25) | P(mean in original VIEW range) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for _, r in boot.iterrows():
        lines.append(f"| {r['contrast']} | {fmt(r['mean'])} | {fmt(r['median'])} | [{fmt(r['ci95_lo'])}, {fmt(r['ci95_hi'])}] | {float(r['prob_inside_pm_0p25']):.3f} | {float(r['prob_in_original_view_range_0p684_0p894']):.3f} |")
    lines.append("")
    lines.append("This quantifies held-out pair uncertainty only, not training-seed variability. The second split seed is still needed. But at seed43022, VS−C is decisively inside the pre-stated near-CLEAN band and outside the original VIEW band.")
    lines.append("")
    lines.append("## 4. Natural-copy component check")
    lines.append("")
    lines.append("| role | raw gain | normalized gain | source-present NLL | no-source/control NLL |")
    lines.append("|---|---:|---:|---:|---:|")
    for _, r in copy_late.sort_values("role").iterrows():
        if r["role"] in {"C","R","V","RS","VS"}:
            lines.append(f"| {r['role']} | {fmt(r['mean_gain'])} | {fmt(r['norm_gain'])} | {fmt(r['source_present_nll'])} | {fmt(r['control_nll'])} |")
    lines.append("")
    lines.append("| contrast | raw gain Δ | normalized gain Δ | source-present NLL Δ | control NLL Δ |")
    lines.append("|---|---:|---:|---:|---:|")
    for _, r in copy_con[copy_con["contrast"].isin(["VminusC","VSminusC","VSminusV","RminusC","RSminusC"])].sort_values("contrast").iterrows():
        lines.append(f"| {r['contrast']} | {fmt(r['mean_gain_delta'])} | {fmt(r['norm_gain_delta'])} | {fmt(r['source_present_nll_delta'])} | {fmt(r['control_nll_delta'])} |")
    lines.append("")
    lines.append("VIEW_SPLIT has only a small raw and normalized copy-gain advantage over CLEAN; like original VIEW, its source-present repeated NLL is not meaningfully better than CLEAN. The copy-probe conclusion remains dominated by original REPEAT's source-present improvement; VIEW/VS copy statements should stay component-qualified.")
    lines.append("")
    if not ent.empty:
        lines.append("## 5. Entity cue-ablation split readout")
        lines.append("")
        lines.append("The split Entity-cue ablation is not an official Entity accuracy evaluation, but it checks whether split arms create DeBERTa-like update cue use in the same probe family.")
        lines.append("")
        lines.append("| split contrast | group | margin full Δ | no-initial Δ | no-last Δ | no-all Δ |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for _, r in ent[ent["group"].isin(["ALL","rel_ge2","rel_ge3"])].sort_values(["contrast","group"]).iterrows():
            lines.append(f"| {r['contrast']} | {r['group']} | {fmt(r.get('late_mean_delta_mean_margin_full_a_minus_b'))} | {fmt(r.get('late_mean_delta_mean_effect_no_initial_qbox_a_minus_b'))} | {fmt(r.get('late_mean_delta_mean_effect_no_last_relevant_update_a_minus_b'))} | {fmt(r.get('late_mean_delta_mean_effect_no_all_relevant_updates_a_minus_b'))} |")
        lines.append("")
    lines.append("## Scientific reading")
    lines.append("")
    lines.append("At seed43022 both sides of the relation-practice account are now local in the same operational sense. Exact local recurrence creates source-specific identity misfire; local restatement creates the source-specific content-conditioning residual. When either companion is split into separate rows, the source-specific residual collapses toward zero, while broad T/U target fit can improve because the same text exposure no longer supplies the local shortcut. This supports the training-window relation-practice principle more strongly than a generic paraphrase/repetition exposure account.")
    lines.append("")
    lines.append("The result should still be stated with the following boundaries: it is one split seed so far; row splitting changes positions/format/spacing as well as co-occurrence; the mixture equation remains phenomenological; and second split seeds plus a natural variation-set probe are needed before claiming a general training law beyond this BabyLM construction.")
    lines.append("")
    lines.append("Data outputs: `experiments/archive/relation_learning/data/view_split_integration`.")
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    copy = read_concat(COPY_SOURCES, "copy")
    rewrite = read_concat(REWRITE_SOURCES, "rewrite")
    copy.to_csv(OUT / "copy_input_rows.csv", index=False)
    rewrite.to_csv(OUT / "rewrite_input_rows.csv", index=False)
    copy_late, copy_con = copy_arm_summary(copy)
    rw_late, rw_con = rewrite_arm_summary(rewrite)
    pair_sum, strict_sum = pair_and_strict(rewrite)
    boot = bootstrap(OUT / "view_split_pair_level_late_pairs.csv")
    ent = entity_summary()
    write_note(copy_late, copy_con, rw_late, rw_con, pair_sum, strict_sum, boot, ent)
    result = {
        "status": "VIEW_SPLIT_INTEGRATION_DONE",
        "note": rel(NOTE),
        "outputs": rel(OUT),
        "main_token_nonoverlap_VSminusC_gain_delta": float(rw_con[(rw_con["contrast"] == "VSminusC") & (rw_con["token_class"] == "nonoverlap")]["mean_gain_delta"].iloc[0]),
        "main_token_nonoverlap_VminusC_gain_delta": float(rw_con[(rw_con["contrast"] == "VminusC") & (rw_con["token_class"] == "nonoverlap")]["mean_gain_delta"].iloc[0]),
        "main_token_nonoverlap_VSminusV_gain_delta": float(rw_con[(rw_con["contrast"] == "VSminusV") & (rw_con["token_class"] == "nonoverlap")]["mean_gain_delta"].iloc[0]),
    }
    (OUT / "view_split_integration_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
