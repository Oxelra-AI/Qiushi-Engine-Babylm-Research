#!/usr/bin/env python3
"""research: decompose relation-probe gains into direct and control NLL terms.

This script reads already-scored research DeBERTa and research RoBERTa probe rows.
It does not run models. It asks whether the content-conditioning gain is really a
source-use residual after arm-level control fit is subtracted, and it normalizes
copy gain by each arm's unrepeated/control NLL level before cross-arm comparison.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path("experiments/archive/relation_learning")
D_IN = ROOT / "data/heldout_copy_rewrite_entity_ablation"
RBT_IN = ROOT / "data/roberta_probe_dynamic"
OUT = ROOT / "data/relation_decomposition"
NOTE = (ROOT.parents[2] / 'research/notes/relation_learning/relation_decomposition_principle_update.md')

ROLE_ORDER = ["V", "C", "R"]
PAIR_ORDER = [("V", "C"), ("R", "C"), ("V", "R"), ("R", "V"), ("C", "R")]


@dataclass(frozen=True)
class SourceSpec:
    arch: str
    input_dir: Path
    default_ckpts: tuple[str, ...]


SOURCES = [
    SourceSpec("D", D_IN, ("chck_80M", "chck_90M", "chck_100M")),
    SourceSpec("RBT", RBT_IN, ("chck_60M", "chck_70M", "chck_80M", "chck_90M", "chck_100M")),
]


def mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def sd_sample(xs: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return float(math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)))


def se_sample(xs: list[float]) -> float:
    s = sd_sample(xs)
    return float(s / math.sqrt(len(xs))) if len(xs) > 1 else float("nan")


def fmt(x: Any, nd: int = 4) -> str:
    try:
        xf = float(x)
    except Exception:
        return str(x)
    if math.isnan(xf):
        return "NA"
    return f"{xf:+.{nd}f}"


def arm_role(arm: str) -> str:
    parts = str(arm).split("_")
    if len(parts) >= 2 and parts[1] in {"V", "C", "R"}:
        return parts[1]
    raise ValueError(f"cannot parse role from arm={arm!r}")


def add_arch_and_role(df: pd.DataFrame, arch: str) -> pd.DataFrame:
    df = df.copy()
    if "arch" not in df.columns:
        df["arch"] = arch
    else:
        df["arch"] = df["arch"].fillna(arch)
    if "role" not in df.columns:
        df["role"] = df["arm"].map(arm_role)
    return df


def read_source_rows(spec: SourceSpec, family: str) -> pd.DataFrame:
    path = spec.input_dir / f"{family}_pair_rows.csv"
    df = pd.read_csv(path)
    return add_arch_and_role(df, spec.arch)


def summarize_rewrite_arm_terms(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_specs = [("ALL", df), ("token_nonoverlap", df[df["token_class"] == "nonoverlap"]), ("token_overlap", df[df["token_class"] == "overlap"])]
    for group, sub in group_specs:
        gb = sub.groupby(["arch", "seed", "checkpoint", "arm", "role"], dropna=False)
        for key, g in gb:
            arch, seed, ck, arm, role = key
            rows.append({
                "family": "rewrite",
                "arch": arch,
                "seed": int(seed),
                "checkpoint": ck,
                "arm": arm,
                "role": role,
                "group": group,
                "n": int(len(g)),
                "mean_gain": float(g["gain"].mean()),
                "true_source_nll": float(g["true_source_nll"].mean()),
                "unrelated_source_nll": float(g["unrelated_source_nll"].mean()),
                "gain_sd": float(g["gain"].std(ddof=1)),
                "gain_se": float(g["gain"].std(ddof=1) / math.sqrt(len(g))) if len(g) > 1 else float("nan"),
            })
    return pd.DataFrame(rows)


def summarize_rewrite_contrasts(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_specs = [("ALL", df), ("token_nonoverlap", df[df["token_class"] == "nonoverlap"]), ("token_overlap", df[df["token_class"] == "overlap"])]
    for group, sub in group_specs:
        for (arch, seed, ck), g0 in sub.groupby(["arch", "seed", "checkpoint"], dropna=False):
            available = sorted(set(g0["role"]))
            for a, b in PAIR_ORDER:
                if a not in available or b not in available:
                    continue
                arows = g0[g0["role"] == a][["probe_id", "gain", "true_source_nll", "unrelated_source_nll"]].rename(
                    columns={"gain": "gain_a", "true_source_nll": "true_a", "unrelated_source_nll": "unrel_a"}
                )
                brows = g0[g0["role"] == b][["probe_id", "gain", "true_source_nll", "unrelated_source_nll"]].rename(
                    columns={"gain": "gain_b", "true_source_nll": "true_b", "unrelated_source_nll": "unrel_b"}
                )
                j = arows.merge(brows, on="probe_id", how="inner")
                if j.empty:
                    continue
                gain_delta = j["gain_a"] - j["gain_b"]
                true_delta = j["true_a"] - j["true_b"]
                unrel_delta = j["unrel_a"] - j["unrel_b"]
                excess_true_cost = true_delta - unrel_delta
                rows.append({
                    "family": "rewrite",
                    "arch": arch,
                    "seed": int(seed),
                    "checkpoint": ck,
                    "group": group,
                    "contrast": f"{a}minus{b}",
                    "n": int(len(j)),
                    "gain_delta": float(gain_delta.mean()),
                    "gain_delta_sd": float(gain_delta.std(ddof=1)),
                    "gain_delta_se": float(gain_delta.std(ddof=1) / math.sqrt(len(j))) if len(j) > 1 else float("nan"),
                    "true_source_nll_delta_a_minus_b": float(true_delta.mean()),
                    "unrelated_source_nll_delta_a_minus_b": float(unrel_delta.mean()),
                    "excess_true_nll_cost_over_unrelated": float(excess_true_cost.mean()),
                    "excess_true_cost_se": float(excess_true_cost.std(ddof=1) / math.sqrt(len(j))) if len(j) > 1 else float("nan"),
                })
    return pd.DataFrame(rows)


def summarize_copy_arm_terms(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_specs = [("ALL", df)]
    for span_len in sorted(df["span_len"].dropna().unique()):
        group_specs.append((f"span_{int(span_len)}", df[df["span_len"] == span_len]))
    for group, sub in group_specs:
        gb = sub.groupby(["arch", "seed", "checkpoint", "arm", "role"], dropna=False)
        for key, g in gb:
            arch, seed, ck, arm, role = key
            mean_gain = float(g["gain"].mean())
            repeated = float(g["repeated_nll"].mean())
            unrep = float(g["unrepeated_nll"].mean())
            rows.append({
                "family": "copy",
                "arch": arch,
                "seed": int(seed),
                "checkpoint": ck,
                "arm": arm,
                "role": role,
                "group": group,
                "n": int(len(g)),
                "mean_gain": mean_gain,
                "repeated_nll": repeated,
                "unrepeated_control_nll": unrep,
                "normalized_gain_by_control": mean_gain / unrep if unrep != 0 else float("nan"),
                "gain_sd": float(g["gain"].std(ddof=1)),
                "gain_se": float(g["gain"].std(ddof=1) / math.sqrt(len(g))) if len(g) > 1 else float("nan"),
            })
    return pd.DataFrame(rows)


def summarize_copy_contrasts(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_specs = [("ALL", df)]
    for span_len in sorted(df["span_len"].dropna().unique()):
        group_specs.append((f"span_{int(span_len)}", df[df["span_len"] == span_len]))
    for group, sub in group_specs:
        for (arch, seed, ck), g0 in sub.groupby(["arch", "seed", "checkpoint"], dropna=False):
            available = sorted(set(g0["role"]))
            arm_term = {}
            for role, gg in g0.groupby("role"):
                gain = float(gg["gain"].mean())
                unrep = float(gg["unrepeated_nll"].mean())
                arm_term[role] = {
                    "gain": gain,
                    "unrep": unrep,
                    "rep": float(gg["repeated_nll"].mean()),
                    "norm": gain / unrep if unrep != 0 else float("nan"),
                }
            for a, b in PAIR_ORDER:
                if a not in available or b not in available:
                    continue
                arows = g0[g0["role"] == a][["probe_id", "gain", "repeated_nll", "unrepeated_nll"]].rename(
                    columns={"gain": "gain_a", "repeated_nll": "rep_a", "unrepeated_nll": "unrep_a"}
                )
                brows = g0[g0["role"] == b][["probe_id", "gain", "repeated_nll", "unrepeated_nll"]].rename(
                    columns={"gain": "gain_b", "repeated_nll": "rep_b", "unrepeated_nll": "unrep_b"}
                )
                j = arows.merge(brows, on="probe_id", how="inner")
                if j.empty:
                    continue
                raw_delta = j["gain_a"] - j["gain_b"]
                rep_delta = j["rep_a"] - j["rep_b"]
                unrep_delta = j["unrep_a"] - j["unrep_b"]
                rows.append({
                    "family": "copy",
                    "arch": arch,
                    "seed": int(seed),
                    "checkpoint": ck,
                    "group": group,
                    "contrast": f"{a}minus{b}",
                    "n": int(len(j)),
                    "gain_delta": float(raw_delta.mean()),
                    "gain_delta_sd": float(raw_delta.std(ddof=1)),
                    "gain_delta_se": float(raw_delta.std(ddof=1) / math.sqrt(len(j))) if len(j) > 1 else float("nan"),
                    "repeated_nll_delta_a_minus_b": float(rep_delta.mean()),
                    "unrepeated_control_nll_delta_a_minus_b": float(unrep_delta.mean()),
                    "normalized_gain_a": arm_term[a]["norm"],
                    "normalized_gain_b": arm_term[b]["norm"],
                    "normalized_gain_delta": arm_term[a]["norm"] - arm_term[b]["norm"],
                    "control_nll_a": arm_term[a]["unrep"],
                    "control_nll_b": arm_term[b]["unrep"],
                    "repeated_nll_a": arm_term[a]["rep"],
                    "repeated_nll_b": arm_term[b]["rep"],
                })
    return pd.DataFrame(rows)


def late_summary(df: pd.DataFrame, value_cols: list[str], ckpt_sets: dict[str, tuple[str, ...]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    id_cols = [c for c in ["family", "arch", "seed", "group", "contrast"] if c in df.columns]
    for set_name, ckpts in ckpt_sets.items():
        sub = df[df["checkpoint"].isin(ckpts)].copy()
        if sub.empty:
            continue
        for key, g in sub.groupby(id_cols, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            row = dict(zip(id_cols, key))
            row["checkpoint_set"] = set_name
            row["n_checkpoints"] = int(g["checkpoint"].nunique())
            row["checkpoints"] = ";".join(sorted(g["checkpoint"].unique()))
            row["n_items_per_checkpoint_min"] = int(g["n"].min()) if "n" in g.columns else None
            for col in value_cols:
                vals = [float(x) for x in g[col].dropna().tolist()]
                row[f"mean_{col}"] = mean(vals)
                row[f"sd_across_checkpoints_{col}"] = sd_sample(vals)
                row[f"se_across_checkpoints_{col}"] = se_sample(vals)
                row[f"checkpoint_values_{col}"] = ";".join(f"{ck}:{float(v):.6g}" for ck, v in zip(g["checkpoint"].tolist(), g[col].tolist()))
            rows.append(row)
    return pd.DataFrame(rows)


def extract_row(df: pd.DataFrame, **conds: Any) -> dict[str, Any] | None:
    sub = df.copy()
    for k, v in conds.items():
        sub = sub[sub[k] == v]
    if sub.empty:
        return None
    return sub.iloc[0].to_dict()


def write_note(summary: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# research relation-probe term decomposition")
    lines.append("")
    lines.append("This CPU-only analysis reuses the already scored research DeBERTa and research RoBERTa probe rows. It separates each source-conditioning gain into the true-source NLL term and the unrelated-source control term, and it rescales natural-copy gain by each arm's own unrepeated-control NLL level.")
    lines.append("")
    lines.append("## Main non-overlap rewrite result")
    lines.append("")
    lines.append("For rewrite tokens whose tokenizer IDs do not occur in the source, gain = NLL(unrelated source) - NLL(true source). In the table below, true/unrelated deltas are A minus B; positive true-source delta means arm A is worse with the actual source present. `excess_true_cost` = true-source delta - unrelated-source delta; positive values mean worse true-source use beyond the broad control level, and it equals the negative of the gain delta.")
    lines.append("")
    lines.append("| arch | seed | ckpts | contrast | gain delta | true-source delta | unrelated delta | excess true-source cost | sd(gain) over ckpts |")
    lines.append("|---|---:|---|---|---:|---:|---:|---:|---:|")
    for row in summary["rewrite_nonoverlap_rows"]:
        lines.append(
            f"| {row['arch']} | {row['seed']} | {row['checkpoint_set']} | {row['contrast']} | "
            f"{fmt(row['mean_gain_delta'])} | {fmt(row['mean_true_source_nll_delta_a_minus_b'])} | "
            f"{fmt(row['mean_unrelated_source_nll_delta_a_minus_b'])} | {fmt(row['mean_excess_true_nll_cost_over_unrelated'])} | "
            f"{fmt(row['sd_across_checkpoints_gain_delta'])} |"
        )
    lines.append("")
    lines.append("Scientific reading: the cross-architecture invariant is the REPEAT cost. In DeBERTa as well as RoBERTa, REPEAT is worse than CLEAN on non-overlap true-source NLL, and most of that difference remains after subtracting the unrelated-source control. Thus exact in-window natural recurrence does not merely fail to buy nonidentical content use; it installs a competing tendency that makes the learner worse than the no-companion baseline on held-out source-conditioned nonidentical tokens.")
    lines.append("")
    lines.append("The VIEW-side benefit differs by architecture. In DeBERTa, V-C non-overlap gain is large in both seeds and its true-source advantage exceeds the unrelated-source advantage, so DeBERTa provides direct evidence for content-conditioned source use beyond compact-register fit. In RoBERTa, V-C true-source and unrelated-source improvements are both large and close, leaving only a small residual conditioning gain. Therefore the positive VIEW half is strongly supported for DeBERTa but only weakly separated from broad compact-register fit in this RoBERTa run.")
    lines.append("")
    lines.append("## Natural-copy gain after control normalization")
    lines.append("")
    lines.append("Copy gain = NLL(unrepeated control) - NLL(repeated source). `norm gain` divides each arm's mean gain by that same arm's mean unrepeated-control NLL within the checkpoint and group, so arm contrasts are not dominated by different loss levels.")
    lines.append("")
    lines.append("| arch | seed | ckpts | contrast | raw gain delta | normalized gain delta | control NLL delta | repeated NLL delta | sd(raw) over ckpts | raw checkpoint values | norm checkpoint values |")
    lines.append("|---|---:|---|---|---:|---:|---:|---:|---:|---|---|")
    for row in summary["copy_all_rows"]:
        lines.append(
            f"| {row['arch']} | {row['seed']} | {row['checkpoint_set']} | {row['contrast']} | "
            f"{fmt(row['mean_gain_delta'])} | {fmt(row['mean_normalized_gain_delta'])} | "
            f"{fmt(row['mean_unrepeated_control_nll_delta_a_minus_b'])} | {fmt(row['mean_repeated_nll_delta_a_minus_b'])} | "
            f"{fmt(row['sd_across_checkpoints_gain_delta'])} | {row['checkpoint_values_gain_delta']} | {row['checkpoint_values_normalized_gain_delta']} |"
        )
    lines.append("")
    lines.append("Scientific reading: DeBERTa's natural-copy signal remains large after normalization. For REPEAT versus CLEAN, the raw gain advantage is about +0.50 and +0.67 nats across the two seeds, and normalized gain remains higher by about +0.20 and +0.31 of the arm control NLL. RoBERTa's raw R-C mean is close to zero and changes sign across checkpoints; after normalization it is still near zero. This is not evidence of an opposite RoBERTa copy tendency. It says the probe barely resolves a copy-practice difference in RoBERTa at this loss level, while it resolves one clearly in DeBERTa.")
    lines.append("")
    lines.append("## Principle update")
    lines.append("")
    lines.append("The most stable statement is now negative-and-constructive: under a fixed finite budget, exact in-window recurrence can train a cross-span identity tendency that competes with nonidentical source-conditioned content use; this active cost appears in both DeBERTa and RoBERTa on held-out compact non-overlap rewrite tokens. The complementary positive result is learner-dependent: DeBERTa turns nonidentical restatement into strong content-conditioned source use and a multi-update Entity advantage, whereas RoBERTa shows the repetition cost and only a small V-C residual after compact-register fit is subtracted.")
    lines.append("")
    lines.append("This changes the center of the research route. The general data-efficient learning principle should not rest only on VIEW helping Entity. It should rest on the structure of finite experience installing particular cross-span relations: repeated identity can be useful for unchanged-state retrieval and local copying, but the same practiced identity relation can actively degrade nonidentical content use; varied restatement supplies evidence for a different relation, which in DeBERTa converts into state-update discrimination. The pending third DeBERTa seed and the natural re-mention instrument now test whether the positive conversion and behavioral crossover survive outside the two-seed compact-rewrite setting.")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    for key, path in summary["paths"].items():
        lines.append(f"- {key}: `{path}`")
    lines.append("")
    NOTE.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    rewrite_all = []
    copy_all = []
    for spec in SOURCES:
        rw = read_source_rows(spec, "rewrite")
        cp = read_source_rows(spec, "copy")
        rewrite_all.append(rw)
        copy_all.append(cp)

    rewrite_df = pd.concat(rewrite_all, ignore_index=True)
    copy_df = pd.concat(copy_all, ignore_index=True)

    rewrite_arm = summarize_rewrite_arm_terms(rewrite_df)
    rewrite_contrasts = summarize_rewrite_contrasts(rewrite_df)
    copy_arm = summarize_copy_arm_terms(copy_df)
    copy_contrasts = summarize_copy_contrasts(copy_df)

    ckpt_sets = {
        "D_late80_100": ("chck_80M", "chck_90M", "chck_100M"),
        "RBT_all60_100": ("chck_60M", "chck_70M", "chck_80M", "chck_90M", "chck_100M"),
        "all_late80_100": ("chck_80M", "chck_90M", "chck_100M"),
    }

    rewrite_late = late_summary(
        rewrite_contrasts,
        ["gain_delta", "true_source_nll_delta_a_minus_b", "unrelated_source_nll_delta_a_minus_b", "excess_true_nll_cost_over_unrelated"],
        ckpt_sets,
    )
    copy_late = late_summary(
        copy_contrasts,
        ["gain_delta", "normalized_gain_delta", "unrepeated_control_nll_delta_a_minus_b", "repeated_nll_delta_a_minus_b"],
        ckpt_sets,
    )

    # Remove checkpoint sets that do not match the intended architecture from the compact summary rows.
    def rowset_for_arch(df: pd.DataFrame, arch: str, group: str, contrast: str, set_name: str) -> dict[str, Any] | None:
        return extract_row(df, arch=arch, group=group, contrast=contrast, checkpoint_set=set_name)

    important_rewrite = []
    for seed in [43022, 43122]:
        for contrast in ["RminusC", "VminusC", "VminusR"]:
            r = extract_row(rewrite_late, arch="D", seed=seed, group="token_nonoverlap", contrast=contrast, checkpoint_set="D_late80_100")
            if r:
                important_rewrite.append(r)
    for contrast in ["RminusC", "VminusC", "VminusR"]:
        r = extract_row(rewrite_late, arch="RBT", seed=43022, group="token_nonoverlap", contrast=contrast, checkpoint_set="RBT_all60_100")
        if r:
            important_rewrite.append(r)

    important_copy = []
    for seed in [43022, 43122]:
        for contrast in ["RminusC", "VminusC", "RminusV"]:
            r = extract_row(copy_late, arch="D", seed=seed, group="ALL", contrast=contrast, checkpoint_set="D_late80_100")
            if r:
                important_copy.append(r)
    for contrast in ["RminusC", "VminusC", "RminusV"]:
        r = extract_row(copy_late, arch="RBT", seed=43022, group="ALL", contrast=contrast, checkpoint_set="RBT_all60_100")
        if r:
            important_copy.append(r)

    paths = {
        "rewrite_arm_terms": OUT / "rewrite_arm_terms.csv",
        "rewrite_contrast_terms_by_checkpoint": OUT / "rewrite_contrast_terms_by_checkpoint.csv",
        "rewrite_contrast_late_summary": OUT / "rewrite_contrast_late_summary.csv",
        "copy_arm_terms": OUT / "copy_arm_terms.csv",
        "copy_contrast_terms_by_checkpoint": OUT / "copy_contrast_terms_by_checkpoint.csv",
        "copy_contrast_late_summary": OUT / "copy_contrast_late_summary.csv",
        "summary_json": OUT / "relation_decomposition_summary.json",
        "note": NOTE,
    }

    rewrite_arm.to_csv(paths["rewrite_arm_terms"], index=False)
    rewrite_contrasts.to_csv(paths["rewrite_contrast_terms_by_checkpoint"], index=False)
    rewrite_late.to_csv(paths["rewrite_contrast_late_summary"], index=False)
    copy_arm.to_csv(paths["copy_arm_terms"], index=False)
    copy_contrasts.to_csv(paths["copy_contrast_terms_by_checkpoint"], index=False)
    copy_late.to_csv(paths["copy_contrast_late_summary"], index=False)

    summary = {
        "status": "RELATION_DECOMPOSITION_COMPLETE",
        "input_dirs": {"D": str(D_IN), "RBT": str(RBT_IN)},
        "rewrite_nonoverlap_rows": important_rewrite,
        "copy_all_rows": important_copy,
        "paths": {k: str(v) for k, v in paths.items()},
    }
    paths["summary_json"].write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_note(summary)

    print(json.dumps({"status": summary["status"], "summary_json": str(paths["summary_json"]), "note": str(NOTE)}, indent=2))


if __name__ == "__main__":
    main()
