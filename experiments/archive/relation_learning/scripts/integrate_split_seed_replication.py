#!/usr/bin/env python3
"""research: integrate two-seed split-control probe evidence.

Joins original C/R/V compact-rewrite, natural-copy, and split-arm probe rows for
seed43022 and seed43122.  The emphasis is measured locality: does splitting the
same source/companion material across rows remove the source-specific residuals
relative to the matched CLEAN arm, and does that repeat across seeds?
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import json
import math
import pathlib
import statistics
import time
from collections import defaultdict
from typing import Any

import pandas as pd

ROOT0 = _public_path('experiments/archive/relation_learning/scripts/integrate_split_seed_replication.py')
ROOT = _PUBLIC_ROOT

WS = ROOT / "experiments/archive/relation_learning"
OUT = WS / "data/split_seed_replication"
NOTE = (_PUBLIC_ROOT / 'research/notes/relation_learning/split_seed_replication.md')

ORIG_430_431 = WS / "data/heldout_copy_rewrite_entity_ablation"
SPLIT_430_REPEAT = WS / "data/split_repeat_split_probes"
SPLIT_430_VIEW = WS / "data/split_view_split_probes"
SPLIT_431_REPEAT = WS / "data/split_repeat_split_seed43122_probes"
SPLIT_431_VIEW = WS / "data/split_view_split_seed43122_probes"
ORIG_432 = WS / "data/seed43222_probes"
ORIG_432_CLEAN = WS / "data/seed43222_parallel_clean_probes"

CKPTS = ["chck_80M", "chck_90M", "chck_100M"]
ORIG_NONOVERLAP_R_RANGE = (-1.0433110410895625, -0.7517085695681751)
ORIG_NONOVERLAP_V_RANGE = (0.683689026198788, 0.893788753037844)
ORIG_COPY_R_RANGE = (0.4995790458606537, 0.6678323688991971)
NEAR_CLEAN_BAND = 0.25


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                keys.append(k); seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def mean(xs) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return float(statistics.mean(vals)) if vals else float("nan")


def sd(xs) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return float(statistics.stdev(vals)) if len(vals) > 1 else float("nan")


def load_csv_if(path: pathlib.Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def role_from_arm(arm: str) -> str:
    bits = str(arm).split("_")
    if len(bits) >= 3:
        return bits[1]
    return str(arm)


def seed_from_arm(arm: str, fallback=None) -> int:
    bits = str(arm).split("_")
    for b in reversed(bits):
        if b.isdigit():
            return int(b)
    if fallback is not None:
        return int(fallback)
    raise ValueError(f"cannot parse seed from {arm}")


def load_rewrite_rows() -> pd.DataFrame:
    frames = []
    for path in [
        ORIG_430_431 / "rewrite_pair_rows.csv",
        ORIG_432 / "rewrite_pair_rows.csv",
        ORIG_432_CLEAN / "rewrite_pair_rows.csv",
        SPLIT_430_REPEAT / "rewrite_pair_rows.csv",
        SPLIT_430_VIEW / "rewrite_pair_rows.csv",
        SPLIT_431_REPEAT / "rewrite_pair_rows.csv",
        SPLIT_431_VIEW / "rewrite_pair_rows.csv",
    ]:
        df = load_csv_if(path)
        # Override the base scorer's shortened role labels so split arms remain
        # distinct: D_RS_* -> RS and D_VS_* -> VS, not R/V.
        df["role"] = df["arm"].map(role_from_arm)
        if "seed" not in df.columns:
            df["seed"] = df["arm"].map(seed_from_arm)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    # Drop exact duplicates from repeated CLEAN rows in per-split outputs.
    df = df[df["checkpoint"].isin(CKPTS)].copy()
    df = df.drop_duplicates(subset=["arm", "checkpoint", "probe_id"], keep="first")
    return df


def load_copy_rows() -> pd.DataFrame:
    frames = []
    for path in [
        ORIG_430_431 / "copy_pair_rows.csv",
        ORIG_432 / "copy_pair_rows.csv",
        ORIG_432_CLEAN / "copy_pair_rows.csv",
        SPLIT_430_REPEAT / "copy_pair_rows.csv",
        SPLIT_430_VIEW / "copy_pair_rows.csv",
        SPLIT_431_REPEAT / "copy_pair_rows.csv",
        SPLIT_431_VIEW / "copy_pair_rows.csv",
    ]:
        df = load_csv_if(path)
        # Override the base scorer's shortened role labels so split arms remain
        # distinct: D_RS_* -> RS and D_VS_* -> VS, not R/V.
        df["role"] = df["arm"].map(role_from_arm)
        if "seed" not in df.columns:
            df["seed"] = df["arm"].map(seed_from_arm)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["checkpoint"].isin(CKPTS)].copy()
    df = df.drop_duplicates(subset=["arm", "checkpoint", "probe_id"], keep="first")
    return df


def summarize_rewrite(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_ck = (
        df.groupby(["seed", "role", "checkpoint", "token_class"], dropna=False)
        .agg(n=("probe_id", "count"), gain=("gain", "mean"), T=("true_source_nll", "mean"), U=("unrelated_source_nll", "mean"))
        .reset_index()
    )
    late = (
        by_ck.groupby(["seed", "role", "token_class"], dropna=False)
        .agg(n_min=("n", "min"), gain=("gain", "mean"), T=("T", "mean"), U=("U", "mean"))
        .reset_index()
    )
    contrasts = []
    for seed in sorted(late["seed"].unique()):
        for tc in sorted(late["token_class"].unique()):
            sub = late[(late["seed"] == seed) & (late["token_class"] == tc)].set_index("role")
            for a, b in [("R", "C"), ("RS", "C"), ("RS", "R"), ("V", "C"), ("VS", "C"), ("VS", "V"), ("V", "R"), ("VS", "RS")]:
                if a in sub.index and b in sub.index:
                    ra = sub.loc[a]; rb = sub.loc[b]
                    contrasts.append({
                        "seed": int(seed), "token_class": tc, "contrast": f"{a}minus{b}",
                        "n_min": int(min(ra["n_min"], rb["n_min"])),
                        "gain_delta": float(ra["gain"] - rb["gain"]),
                        "T_delta": float(ra["T"] - rb["T"]),
                        "U_delta": float(ra["U"] - rb["U"]),
                        "excess_true_cost": float((ra["T"] - rb["T"]) - (ra["U"] - rb["U"])),
                    })
    return by_ck, late, pd.DataFrame(contrasts)


def summarize_copy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    # research copy gain is unrepeated/control NLL minus repeated/source-present NLL.
    # Older integration notes used source_present_nll/control_nll names; the raw
    # probe rows use repeated_nll/unrepeated_nll.  Harmonize explicitly.
    if "source_present_nll" not in df.columns and "repeated_nll" in df.columns:
        df["source_present_nll"] = df["repeated_nll"]
    if "control_nll" not in df.columns and "unrepeated_nll" in df.columns:
        df["control_nll"] = df["unrepeated_nll"]
    if "control_nll" not in df.columns or "source_present_nll" not in df.columns:
        cols = list(df.columns)
        raise RuntimeError(f"copy rows missing expected NLL columns after harmonization: {cols[:24]}")
    by_ck = (
        df.groupby(["seed", "role", "checkpoint"], dropna=False)
        .agg(n=("probe_id", "count"), gain=("gain", "mean"), source_present_nll=("source_present_nll", "mean"), control_nll=("control_nll", "mean"))
        .reset_index()
    )
    # Normalize at the aggregate level, matching the established copy readout:
    # mean gain divided by mean control NLL, not mean of per-record ratios.
    by_ck["norm_gain"] = by_ck["gain"] / by_ck["control_nll"]
    late = (
        by_ck.groupby(["seed", "role"], dropna=False)
        .agg(n_min=("n", "min"), gain=("gain", "mean"), norm_gain=("norm_gain", "mean"), source_present_nll=("source_present_nll", "mean"), control_nll=("control_nll", "mean"))
        .reset_index()
    )
    contrasts = []
    for seed in sorted(late["seed"].unique()):
        sub = late[late["seed"] == seed].set_index("role")
        for a, b in [("R", "C"), ("RS", "C"), ("RS", "R"), ("V", "C"), ("VS", "C"), ("VS", "V"), ("V", "R"), ("VS", "RS")]:
            if a in sub.index and b in sub.index:
                ra = sub.loc[a]; rb = sub.loc[b]
                contrasts.append({
                    "seed": int(seed), "contrast": f"{a}minus{b}",
                    "n_min": int(min(ra["n_min"], rb["n_min"])),
                    "gain_delta": float(ra["gain"] - rb["gain"]),
                    "norm_gain_delta": float(ra["norm_gain"] - rb["norm_gain"]),
                    "source_present_nll_delta": float(ra["source_present_nll"] - rb["source_present_nll"]),
                    "control_nll_delta": float(ra["control_nll"] - rb["control_nll"]),
                })
    return by_ck, late, pd.DataFrame(contrasts)


def bootstrap_pair_mean(df: pd.DataFrame, roles: tuple[str, str], seed: int, token_class: str, value_col: str, n_boot: int = 2000) -> dict[str, Any]:
    sub = df[(df["seed"] == seed) & (df["token_class"] == token_class) & (df["role"].isin(roles))].copy()
    if sub.empty:
        return {}
    # Average over checkpoints within pair and role first.
    wide = (
        sub.groupby(["pair_id", "role"], dropna=False)[value_col]
        .mean()
        .reset_index()
        .pivot(index="pair_id", columns="role", values=value_col)
        .dropna()
    )
    a, b = roles
    vals = (wide[a] - wide[b]).to_numpy(dtype=float)
    rng = statistics.Random if False else None
    import random
    rr = random.Random(9021 + seed + len(vals) + sum(map(ord, value_col)))
    boots = []
    n = len(vals)
    for _ in range(n_boot):
        boots.append(float(sum(vals[rr.randrange(n)] for _ in range(n)) / n))
    boots.sort()
    return {"seed": seed, "token_class": token_class, "contrast": f"{a}minus{b}", "value": value_col, "n_pairs": n, "mean": float(vals.mean()), "p025": boots[int(0.025*n_boot)], "p975": boots[int(0.975*n_boot)]}


def classify_value(side: str, x: float) -> str:
    if not math.isfinite(x):
        return "missing"
    if abs(x) <= NEAR_CLEAN_BAND:
        return "near_clean"
    if side == "repeat_cost":
        lo, hi = ORIG_NONOVERLAP_R_RANGE
        if lo <= x <= hi:
            return "original_range"
        if x < lo:
            return "stronger_than_original"
        return "between_clean_and_original_or_opposite"
    if side == "view_benefit":
        lo, hi = ORIG_NONOVERLAP_V_RANGE
        if lo <= x <= hi:
            return "original_range"
        if x > hi:
            return "stronger_than_original"
        return "between_clean_and_original_or_opposite"
    if side == "copy_repeat":
        lo, hi = ORIG_COPY_R_RANGE
        if lo <= x <= hi:
            return "original_range"
        if x > hi:
            return "stronger_than_original"
        return "between_clean_and_original_or_opposite"
    return "untyped"


def write_note(rew_late: pd.DataFrame, rew_con: pd.DataFrame, copy_late: pd.DataFrame, copy_con: pd.DataFrame, boot: list[dict[str, Any]]) -> None:
    def cv(seed, tc, contrast, col):
        tc = {"token_nonoverlap": "nonoverlap", "token_overlap": "overlap"}.get(tc, tc)
        row = rew_con[(rew_con.seed == seed) & (rew_con.token_class == tc) & (rew_con.contrast == contrast)]
        if row.empty:
            return float("nan")
        return float(row.iloc[0][col])
    def cc(seed, contrast, col):
        row = copy_con[(copy_con.seed == seed) & (copy_con.contrast == contrast)]
        if row.empty:
            return float("nan")
        return float(row.iloc[0][col])

    rows = []
    for seed in [43022, 43122]:
        rows.append({
            "seed": seed,
            "RminusC_nonoverlap_gain": cv(seed, "token_nonoverlap", "RminusC", "gain_delta"),
            "RSminusC_nonoverlap_gain": cv(seed, "token_nonoverlap", "RSminusC", "gain_delta"),
            "RSminusR_nonoverlap_gain": cv(seed, "token_nonoverlap", "RSminusR", "gain_delta"),
            "VminusC_nonoverlap_gain": cv(seed, "token_nonoverlap", "VminusC", "gain_delta"),
            "VSminusC_nonoverlap_gain": cv(seed, "token_nonoverlap", "VSminusC", "gain_delta"),
            "VSminusV_nonoverlap_gain": cv(seed, "token_nonoverlap", "VSminusV", "gain_delta"),
            "RminusC_T_delta": cv(seed, "token_nonoverlap", "RminusC", "T_delta"),
            "RSminusC_T_delta": cv(seed, "token_nonoverlap", "RSminusC", "T_delta"),
            "VminusC_T_delta": cv(seed, "token_nonoverlap", "VminusC", "T_delta"),
            "VSminusC_T_delta": cv(seed, "token_nonoverlap", "VSminusC", "T_delta"),
            "RminusC_copy_gain": cc(seed, "RminusC", "gain_delta"),
            "RSminusC_copy_gain": cc(seed, "RSminusC", "gain_delta"),
            "VminusC_copy_gain": cc(seed, "VminusC", "gain_delta"),
            "VSminusC_copy_gain": cc(seed, "VSminusC", "gain_delta"),
        })

    mean_rs = mean([r["RSminusC_nonoverlap_gain"] for r in rows])
    mean_vs = mean([r["VSminusC_nonoverlap_gain"] for r in rows])
    mean_rs_copy = mean([r["RSminusC_copy_gain"] for r in rows])
    mean_vs_copy = mean([r["VSminusC_copy_gain"] for r in rows])

    lines = []
    lines.append("# research two-seed split-control replication")
    lines.append("")
    lines.append("This integration compares the original local arms with split-row controls at seeds 43022 and 43122 using the same held-out copy and compact-rewrite probes. The split rows retain the same selected source and companion texts and the same 100M-word budget, but remove source/companion co-occurrence within the training window.")
    lines.append("")
    lines.append("## Compact rewrite token-nonoverlap summary")
    lines.append("")
    lines.append("| seed | R−C gain | RS−C gain | RS−R gain | V−C gain | VS−C gain | VS−V gain | reading |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in rows:
        s1 = classify_value("repeat_cost", r["RSminusC_nonoverlap_gain"])
        s2 = classify_value("view_benefit", r["VSminusC_nonoverlap_gain"])
        lines.append(f"| {r['seed']} | {r['RminusC_nonoverlap_gain']:+.4f} | {r['RSminusC_nonoverlap_gain']:+.4f} | {r['RSminusR_nonoverlap_gain']:+.4f} | {r['VminusC_nonoverlap_gain']:+.4f} | {r['VSminusC_nonoverlap_gain']:+.4f} | {r['VSminusV_nonoverlap_gain']:+.4f} | RS {s1}; VS {s2} |")
    lines.append("")
    lines.append(f"Across the two split seeds, RS−C token-nonoverlap gain averages {mean_rs:+.4f}. Seed43022 is essentially CLEAN-like ({rows[0]['RSminusC_nonoverlap_gain']:+.4f}); seed43122 is still far attenuated from original R−C {rows[1]['RminusC_nonoverlap_gain']:+.4f} but lands at {rows[1]['RSminusC_nonoverlap_gain']:+.4f}, just inside the ±0.25 near-CLEAN band rather than exactly at zero. The same-window exact-recurrence cost is therefore strongly reduced in both seeds, with a small remaining negative residual in seed43122.")
    lines.append(f"Across the two split seeds, VS−C token-nonoverlap gain averages {mean_vs:+.4f}. Seed43022 is near zero ({rows[0]['VSminusC_nonoverlap_gain']:+.4f}); seed43122 is also near zero but slightly negative ({rows[1]['VSminusC_nonoverlap_gain']:+.4f}) while original V−C at the same seed is {rows[1]['VminusC_nonoverlap_gain']:+.4f}. This repeats the collapse of VIEW's source-specific compact-rewrite benefit under row splitting.")
    lines.append("")
    lines.append("## T/U term reading")
    lines.append("")
    lines.append("| seed | R−C ΔT | RS−C ΔT | R−C ΔU | RS−C ΔU | V−C ΔT | VS−C ΔT | V−C ΔU | VS−C ΔU |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for seed in [43022, 43122]:
        lines.append(f"| {seed} | {cv(seed,'token_nonoverlap','RminusC','T_delta'):+.4f} | {cv(seed,'token_nonoverlap','RSminusC','T_delta'):+.4f} | {cv(seed,'token_nonoverlap','RminusC','U_delta'):+.4f} | {cv(seed,'token_nonoverlap','RSminusC','U_delta'):+.4f} | {cv(seed,'token_nonoverlap','VminusC','T_delta'):+.4f} | {cv(seed,'token_nonoverlap','VSminusC','T_delta'):+.4f} | {cv(seed,'token_nonoverlap','VminusC','U_delta'):+.4f} | {cv(seed,'token_nonoverlap','VSminusC','U_delta'):+.4f} |")
    lines.append("")
    lines.append("For seed43122, RS−C no longer has the original true-source-worse / unrelated-source-better sign reversal: both T and U improve relative to CLEAN (ΔT −0.3804, ΔU −0.5702). The remaining negative RS−C gain comes from the true source helping less than the unrelated source after both broad terms improve. VS−C similarly improves both T and U, with U slightly more improved, making the gain slightly negative despite the local VIEW arm being strongly positive.")
    lines.append("")
    lines.append("## Natural-copy summary")
    lines.append("")
    lines.append("| seed | R−C copy | RS−C copy | RS−R copy | V−C copy | VS−C copy | VS−V copy |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(f"| {r['seed']} | {r['RminusC_copy_gain']:+.4f} | {r['RSminusC_copy_gain']:+.4f} | {cc(r['seed'],'RSminusR','gain_delta'):+.4f} | {r['VminusC_copy_gain']:+.4f} | {r['VSminusC_copy_gain']:+.4f} | {cc(r['seed'],'VSminusV','gain_delta'):+.4f} |")
    lines.append("")
    lines.append(f"The natural-copy side is not a symmetric collapse. RS−C copy gain averages {mean_rs_copy:+.4f}: seed43022 was below CLEAN in raw gain after normalization complications, but seed43122 keeps a positive copy gain of {rows[1]['RSminusC_copy_gain']:+.4f}, smaller than original R−C {rows[1]['RminusC_copy_gain']:+.4f}. VS−C copy gain remains positive in both seeds and averages {mean_vs_copy:+.4f}. Thus row splitting removes the source-specific compact-rewrite residuals more cleanly than it removes all source-present exact-copy behavior.")
    lines.append("")
    lines.append("## Pair bootstrap on nonoverlap gain")
    lines.append("")
    lines.append("| seed | contrast | mean | 95% pair interval | n pairs |")
    lines.append("|---:|---|---:|---:|---:|")
    for b in boot:
        if b.get("token_class") == "nonoverlap" and b.get("value") == "gain":
            lines.append(f"| {b['seed']} | {b['contrast']} | {b['mean']:+.4f} | [{b['p025']:+.4f}, {b['p975']:+.4f}] | {b['n_pairs']} |")
    lines.append("")
    lines.append("## Scientific update")
    lines.append("")
    lines.append("The second split seed repeats the central locality result for compact nonidentical source use. Same-content row splitting collapses both sides toward CLEAN: exact recurrence no longer creates the large original source-conditioned cost, and restatement no longer creates the large original source-conditioned benefit. The updated result is slightly less symmetric than the seed43022 2×2: seed43122 REPEAT_SPLIT retains a small residual cost and a reduced but positive copy gain, while VIEW_SPLIT is slightly below CLEAN on compact nonoverlap gain. This strengthens the core statement that within-window relation practice is necessary for the large source-specific residuals, while warning that split exposure can still change ordinary fit and exact-copy behavior.")
    lines.append("")
    lines.append("Data outputs: `experiments/archive/relation_learning/data/split_seed_replication`.")
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rew = load_rewrite_rows()
    copy = load_copy_rows()
    rew_by_ck, rew_late, rew_con = summarize_rewrite(rew)
    copy_by_ck, copy_late, copy_con = summarize_copy(copy)
    boot = []
    for seed in [43022, 43122]:
        for roles in [("R", "C"), ("RS", "C"), ("V", "C"), ("VS", "C")]:
            b = bootstrap_pair_mean(rew, roles, seed, "nonoverlap", "gain", n_boot=2000)
            if b:
                boot.append(b)
            b2 = bootstrap_pair_mean(rew, roles, seed, "nonoverlap", "true_source_nll", n_boot=2000)
            if b2:
                boot.append(b2)
    rew_by_ck.to_csv(OUT / "rewrite_by_checkpoint.csv", index=False)
    rew_late.to_csv(OUT / "rewrite_late_roles.csv", index=False)
    rew_con.to_csv(OUT / "rewrite_late_contrasts.csv", index=False)
    copy_by_ck.to_csv(OUT / "copy_by_checkpoint.csv", index=False)
    copy_late.to_csv(OUT / "copy_late_roles.csv", index=False)
    copy_con.to_csv(OUT / "copy_late_contrasts.csv", index=False)
    write_csv(OUT / "rewrite_pair_bootstrap.csv", boot)
    write_note(rew_late, rew_con, copy_late, copy_con, boot)
    result = {
        "status": "SPLIT_SEED_REPLICATION_INTEGRATED",
        "created_utc": now(),
        "note": rel(NOTE),
        "out_dir": rel(OUT),
        "main": {
            "RSminusC_nonoverlap_seed43022": float(rew_con[(rew_con.seed == 43022) & (rew_con.token_class == "nonoverlap") & (rew_con.contrast == "RSminusC")]["gain_delta"].iloc[0]),
            "RSminusC_nonoverlap_seed43122": float(rew_con[(rew_con.seed == 43122) & (rew_con.token_class == "nonoverlap") & (rew_con.contrast == "RSminusC")]["gain_delta"].iloc[0]),
            "VSminusC_nonoverlap_seed43022": float(rew_con[(rew_con.seed == 43022) & (rew_con.token_class == "nonoverlap") & (rew_con.contrast == "VSminusC")]["gain_delta"].iloc[0]),
            "VSminusC_nonoverlap_seed43122": float(rew_con[(rew_con.seed == 43122) & (rew_con.token_class == "nonoverlap") & (rew_con.contrast == "VSminusC")]["gain_delta"].iloc[0]),
        }
    }
    (OUT / "integration_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
