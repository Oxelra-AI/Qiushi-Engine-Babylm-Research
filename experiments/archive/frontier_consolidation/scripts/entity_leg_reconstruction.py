#!/usr/bin/env python3
"""research: reconstruct the Entity dose leg and chance-level EWoK context.

This script uses only already-produced stable-family score files. It does not
run model inference. Its purpose is to freeze the quantitative reading of the
first-basin Entity leg before second-basin and breadth score tables arrive.
"""
from __future__ import annotations

import csv
import json
import math
import pathlib
import statistics as stats
import time
from collections import defaultdict
from typing import Any

ROOT = pathlib.Path.cwd()
STUDY = ROOT / "experiments/archive" / 'frontier_consolidation'
OUT_DIR = STUDY / "data" / "entity_leg_reconstruction"
FAMILY_DIR = STUDY / "data" / "family_lfo_readout"
HARVEST_DIR = STUDY / "data" / "score_harvest_after_max90"
SEED_DIR = STUDY / "data" / "full_deberta_seed_ladder_stable_eval"
MAX_REPEAT_100 = STUDY / "data" / "dose_ladder_stable_eval" / "eval" / "per_target" / "dose_max_repeat_chck_100M.json"
MAX_VIEW_100 = STUDY / "data" / "dose_ladder_stable_eval" / "eval" / "per_target" / "dose_max_view_chck_100M.json"

DOSE_META = {
    "dose1": {"dose": 1.0, "rho": 0.042352, "label": "1x"},
    "dose1p82": {"dose": 1.8209293539856442, "rho": 0.07712, "label": "1.82x"},
    "dose2p64": {"dose": 2.641480921798262, "rho": 0.111872, "label": "2.64x/MAX"},
}
FAMILIES = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "Reading"]
CHECKPOINTS = [f"chck_{m}M" for m in range(10, 101, 10)]


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def f(x: Any, default: float | None = None) -> float | None:
    if x is None or x == "":
        return default
    try:
        y = float(x)
    except Exception:
        return default
    return y if math.isfinite(y) else default


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        seen: list[str] = []
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.append(k)
        fieldnames = seen
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def score_from_per_target(path: pathlib.Path, col: str) -> float | None:
    if not path.exists():
        return None
    obj = json.loads(path.read_text(encoding="utf-8"))
    stable = obj.get("stable_family_scores") or {}
    if finite(stable.get(col)):
        return float(stable[col])
    official = (obj.get("official_overall") or {}).get("scores") or {}
    if finite(official.get(col)):
        return float(official[col])
    task = (obj.get("tasks") or {}).get(col) or {}
    if finite(task.get("score")):
        return float(task["score"])
    scores = task.get("scores") or {}
    if finite(scores.get(col)):
        return float(scores[col])
    if col == "Reading" and finite(scores.get("Reading")):
        return float(scores["Reading"])
    return None


def summarize_values(vals: list[float]) -> dict[str, Any]:
    vals = [float(v) for v in vals if math.isfinite(float(v))]
    if not vals:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None, "stdev": None}
    return {
        "n": len(vals),
        "mean": sum(vals) / len(vals),
        "median": stats.median(vals),
        "min": min(vals),
        "max": max(vals),
        "stdev": stats.pstdev(vals) if len(vals) > 1 else 0.0,
    }


def linear_fit(xs: list[float], ys: list[float]) -> dict[str, Any]:
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx else float("nan")
    intercept = my - slope * mx
    preds = [intercept + slope * x for x in xs]
    ss_res = sum((y - p) ** 2 for y, p in zip(ys, preds))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 1.0
    slope0 = sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs)
    preds0 = [slope0 * x for x in xs]
    ss_res0 = sum((y - p) ** 2 for y, p in zip(ys, preds0))
    return {
        "ordinary_least_squares": {"intercept": intercept, "slope_per_rho": slope, "r2_three_points": r2, "predictions": preds, "residuals": [y - p for y, p in zip(ys, preds)]},
        "through_origin": {"slope_per_rho": slope0, "predictions": preds0, "residuals": [y - p for y, p in zip(ys, preds0)], "sum_squared_residual": ss_res0},
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Family-level first-basin V-R means from the frozen research readout.
    family_rows_raw = read_csv(FAMILY_DIR / "semantic_VR_family_means.csv")
    family_rows: list[dict[str, Any]] = []
    for r in family_rows_raw:
        if r.get("profile") != "semantic_VR" or r.get("window") != "common_10M_80M":
            continue
        g = r.get("group")
        meta = DOSE_META.get(g)
        if not meta:
            continue
        mean = f(r.get("mean"))
        if mean is None:
            continue
        family_rows.append({
            "dose_name": g,
            "dose_label": meta["label"],
            "dose": meta["dose"],
            "rho": meta["rho"],
            "family": r.get("family"),
            "window": r.get("window"),
            "n_checkpoints": int(float(r.get("n") or 0)),
            "V_minus_R_mean": mean,
            "median": f(r.get("median")),
            "min": f(r.get("min")),
            "max": f(r.get("max")),
            "stdev": f(r.get("stdev")),
            "positive_rows": int(float(r.get("positive_rows") or 0)),
            "negative_rows": int(float(r.get("negative_rows") or 0)),
        })
    write_csv(OUT_DIR / "family_vr_common10_80_by_dose.csv", family_rows)

    entity_curve = [r for r in family_rows if r["family"] == "Entity"]
    entity_curve.sort(key=lambda r: r["rho"])
    base = entity_curve[0]["V_minus_R_mean"]
    entity_rows: list[dict[str, Any]] = []
    for r in entity_curve:
        y = float(r["V_minus_R_mean"])
        entity_rows.append({
            **r,
            "entity_contribution_to_cheap6": y / 6.0,
            "entity_contribution_to_cheap5": y / 5.0,
            "effect_per_rho": y / float(r["rho"]),
            "effect_per_dose_multiple": y / float(r["dose"]),
            "ratio_vs_1x_entity_effect": y / base if base else None,
            "dose_ratio_vs_1x": float(r["dose"]),
            "rho_ratio_vs_1x": float(r["rho"]) / float(entity_curve[0]["rho"]),
        })
    xs = [float(r["rho"]) for r in entity_curve]
    ys = [float(r["V_minus_R_mean"]) for r in entity_curve]
    fit = linear_fit(xs, ys)
    finite_slopes = []
    for a, b in zip(entity_curve[:-1], entity_curve[1:]):
        finite_slopes.append({
            "from": a["dose_label"],
            "to": b["dose_label"],
            "delta_rho": b["rho"] - a["rho"],
            "delta_entity_V_minus_R": b["V_minus_R_mean"] - a["V_minus_R_mean"],
            "slope_per_rho": (b["V_minus_R_mean"] - a["V_minus_R_mean"]) / (b["rho"] - a["rho"]),
        })
    write_csv(OUT_DIR / "entity_vr_dose_curve_common10_80.csv", entity_rows)

    # 2) Seed-spread scale from the low-dose seed replication.
    seed_rows = read_csv(SEED_DIR / "full_deberta_seed_ladder_seed_spread.csv")
    seed_scale: dict[str, Any] = {}
    for metric in ["cheap6_no_GlobalPIQA", "cheap5_no_GlobalPIQA_Reading", "EWoK_plus_Entity_sum", "Entity", "EWoK"]:
        col = f"treatment_delta_seed43122_minus_seed43022_{metric}"
        vals = [f(r.get(col)) for r in seed_rows]
        vals = [v for v in vals if v is not None]
        if vals:
            seed_scale[metric] = {
                "trajectory_mean_signed_difference": sum(vals) / len(vals),
                "pointwise_mean_abs_difference": sum(abs(v) for v in vals) / len(vals),
                "pointwise_max_abs_difference": max(abs(v) for v in vals),
                "n": len(vals),
            }

    # 3) Absolute EWoK levels across the already-scored view/repeat ladders,
    # including the CPU-safe MAX-repeat 100M EWoK row that arrived after the
    # research harvest.
    ladder_rows = read_csv(HARVEST_DIR / "harvested_deberta_stable_rows.csv")
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for r in ladder_rows:
        arm = r.get("arm") or ""
        ck = r.get("checkpoint") or ""
        if arm and ck:
            by_key[(arm, ck)] = dict(r)
    # add partial max_repeat 100M EWoK row if absent
    if ("max_repeat", "chck_100M") not in by_key:
        ew = score_from_per_target(MAX_REPEAT_100, "EWoK")
        bl = score_from_per_target(MAX_REPEAT_100, "BLiMP")
        sp = score_from_per_target(MAX_REPEAT_100, "Supplement")
        by_key[("max_repeat", "chck_100M")] = {
            "arm": "max_repeat", "dose": DOSE_META["dose2p64"]["dose"], "rho": DOSE_META["dose2p64"]["rho"],
            "dose_name": "dose2p64", "data_arm": "repeat", "checkpoint": "chck_100M", "words": "100000000",
            "BLiMP": bl, "Supplement": sp, "EWoK": ew, "Entity": "", "COMPS": "", "Reading": "",
            "source_type": "cpu_safe_partial_json", "per_target": rel(MAX_REPEAT_100),
        }
    # ensure MAX view 100M is available even if harvest changes later
    if ("max_view", "chck_100M") not in by_key:
        ew = score_from_per_target(MAX_VIEW_100, "EWoK")
        en = score_from_per_target(MAX_VIEW_100, "Entity")
        by_key[("max_view", "chck_100M")] = {
            "arm": "max_view", "dose": DOSE_META["dose2p64"]["dose"], "rho": DOSE_META["dose2p64"]["rho"],
            "dose_name": "dose2p64", "data_arm": "view", "checkpoint": "chck_100M", "words": "100000000",
            "EWoK": ew, "Entity": en, "source_type": "json", "per_target": rel(MAX_VIEW_100),
        }

    ewok_rows: list[dict[str, Any]] = []
    for (arm, ck), r in sorted(by_key.items(), key=lambda kv: (kv[0][0], int(kv[0][1].split("_")[1].replace("M", "")))):
        if r.get("data_arm") not in {"view", "repeat"}:
            continue
        dose_name = r.get("dose_name")
        if dose_name not in DOSE_META:
            continue
        ew = f(r.get("EWoK"))
        if ew is None:
            continue
        ewok_rows.append({
            "dose_name": dose_name,
            "dose_label": DOSE_META[dose_name]["label"],
            "dose": DOSE_META[dose_name]["dose"],
            "rho": DOSE_META[dose_name]["rho"],
            "arm": arm,
            "data_arm": r.get("data_arm"),
            "checkpoint": ck,
            "words": int(float(r.get("words") or ck.split("_")[1].replace("M", "")) * (1 if str(r.get("words") or "").isdigit() else 1)),
            "EWoK": ew,
            "EWoK_minus_50": ew - 50.0,
            "abs_EWoK_minus_50": abs(ew - 50.0),
            "source_type": r.get("source_type"),
            "per_target": r.get("per_target"),
        })
    write_csv(OUT_DIR / "ewok_absolute_ladder_view_repeat.csv", ewok_rows)

    ewok_abs_summary_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in ewok_rows:
        grouped[(r["dose_name"], r["data_arm"])].append(r)
    for (dose_name, data_arm), rs in sorted(grouped.items(), key=lambda kv: (DOSE_META[kv[0][0]]["rho"], kv[0][1])):
        vals = [float(r["EWoK"]) for r in rs]
        devs = [float(r["abs_EWoK_minus_50"]) for r in rs]
        endpoints = {r["checkpoint"]: r["EWoK"] for r in rs if r["checkpoint"] in {"chck_80M", "chck_90M", "chck_100M"}}
        s = summarize_values(vals)
        ewok_abs_summary_rows.append({
            "dose_name": dose_name,
            "dose_label": DOSE_META[dose_name]["label"],
            "data_arm": data_arm,
            "n": s["n"],
            "mean_EWoK": s["mean"],
            "min_EWoK": s["min"],
            "max_EWoK": s["max"],
            "mean_abs_deviation_from_50": sum(devs) / len(devs) if devs else None,
            "chck_80M": endpoints.get("chck_80M"),
            "chck_90M": endpoints.get("chck_90M"),
            "chck_100M": endpoints.get("chck_100M"),
        })
    write_csv(OUT_DIR / "ewok_absolute_summary_by_dose_arm.csv", ewok_abs_summary_rows)

    ewok_vr_rows: list[dict[str, Any]] = []
    for dose_name in ["dose1", "dose1p82", "dose2p64"]:
        meta = DOSE_META[dose_name]
        for ck in CHECKPOINTS:
            view = None
            repeat = None
            for r in ewok_rows:
                if r["dose_name"] == dose_name and r["checkpoint"] == ck:
                    if r["data_arm"] == "view":
                        view = float(r["EWoK"])
                    elif r["data_arm"] == "repeat":
                        repeat = float(r["EWoK"])
            if view is None or repeat is None:
                continue
            ewok_vr_rows.append({
                "dose_name": dose_name,
                "dose_label": meta["label"],
                "dose": meta["dose"],
                "rho": meta["rho"],
                "checkpoint": ck,
                "view_EWoK": view,
                "repeat_EWoK": repeat,
                "V_minus_R_EWoK": view - repeat,
                "both_scores_within_49_51": (49.0 <= view <= 51.0) and (49.0 <= repeat <= 51.0),
            })
    write_csv(OUT_DIR / "ewok_vr_by_checkpoint.csv", ewok_vr_rows)

    ewok_vr_summary: list[dict[str, Any]] = []
    for dose_name in ["dose1", "dose1p82", "dose2p64"]:
        rs = [r for r in ewok_vr_rows if r["dose_name"] == dose_name]
        for window_name, pred in [("common_10M_80M", lambda ck: int(ck.split("_")[1].replace("M", "")) <= 80), ("available_10M_100M", lambda ck: True)]:
            wrs = [r for r in rs if pred(r["checkpoint"])]
            vals = [float(r["V_minus_R_EWoK"]) for r in wrs]
            view_vals = [float(r["view_EWoK"]) for r in wrs]
            repeat_vals = [float(r["repeat_EWoK"]) for r in wrs]
            if not vals:
                continue
            ewok_vr_summary.append({
                "dose_name": dose_name,
                "dose_label": DOSE_META[dose_name]["label"],
                "window": window_name,
                "n": len(vals),
                "mean_V_minus_R_EWoK": sum(vals) / len(vals),
                "median_V_minus_R_EWoK": stats.median(vals),
                "min_V_minus_R_EWoK": min(vals),
                "max_V_minus_R_EWoK": max(vals),
                "mean_view_EWoK": sum(view_vals) / len(view_vals),
                "mean_repeat_EWoK": sum(repeat_vals) / len(repeat_vals),
                "mean_abs_view_deviation_from_50": sum(abs(v - 50.0) for v in view_vals) / len(view_vals),
                "mean_abs_repeat_deviation_from_50": sum(abs(v - 50.0) for v in repeat_vals) / len(repeat_vals),
                "fraction_pairs_both_within_49_51": sum(1 for r in wrs if r["both_scores_within_49_51"]) / len(wrs),
            })
    write_csv(OUT_DIR / "ewok_vr_summary_by_dose.csv", ewok_vr_summary)

    # 4) Quantitative prediction written before second-basin/breadth score tables.
    max_entity = next(r for r in entity_rows if r["dose_name"] == "dose2p64")
    one_entity = next(r for r in entity_rows if r["dose_name"] == "dose1")
    mid_entity = next(r for r in entity_rows if r["dose_name"] == "dose1p82")
    prediction = {
        "second_basin_MAX_view_minus_repeat_Entity_common_window": {
            "expected_center_score_points": max_entity["V_minus_R_mean"],
            "plain_language": "near +2 Entity score points for MAX, much closer to the first-basin MAX value than to the 1x value",
            "comparison_values": {"first_basin_1x": one_entity["V_minus_R_mean"], "first_basin_1p82": mid_entity["V_minus_R_mean"], "first_basin_MAX": max_entity["V_minus_R_mean"]},
        },
        "MAX_breadth_relation": {
            "source_conditioned_reexpression_prediction": "MAX breadth should fall well short of the MAX compact-view Entity lift if the carrier is source-conditioned re-expression rather than merely avoiding exact duplication.",
            "generic_nonduplicate_prediction": "If any non-duplicate same-population companion experience is sufficient, MAX breadth should approach the compact-view Entity level rather than staying near repeat.",
        },
        "EWoK_handling": "Read EWoK as an absolute near-chance family and relative allocation signal, not as a settled cost carrier unless a future full-ladder table shows a stable movement away from chance.",
    }

    # 5) Optional compact figure.
    fig_path = OUT_DIR / "entity_ewok_dose_reconstruction.png"
    fig_status: dict[str, Any] = {"path": rel(fig_path), "created": False}
    try:
        import matplotlib.pyplot as plt  # type: ignore
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), dpi=160)
        ax = axes[0]
        ax.plot([r["rho"] for r in entity_rows], [r["V_minus_R_mean"] for r in entity_rows], marker="o", color="#8b1a1a", label="Entity V-R")
        for r in entity_rows:
            ax.annotate(r["dose_label"], (r["rho"], r["V_minus_R_mean"]), textcoords="offset points", xytext=(5, 5), fontsize=8)
        ax.axhline(0, color="0.3", lw=0.8)
        ax.set_xlabel("restructured fraction rho")
        ax.set_ylabel("Entity view − repeat (score points)")
        ax.set_title("First-basin Entity leg is dose-scaled")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)

        ax = axes[1]
        colors = {"dose1": "#4c78a8", "dose1p82": "#f58518", "dose2p64": "#54a24b"}
        for dose_name in ["dose1", "dose1p82", "dose2p64"]:
            rs = [r for r in ewok_vr_rows if r["dose_name"] == dose_name]
            xs2 = [int(r["checkpoint"].split("_")[1].replace("M", "")) for r in rs]
            ys2 = [r["V_minus_R_EWoK"] for r in rs]
            ax.plot(xs2, ys2, marker="o", ms=3, color=colors[dose_name], label=DOSE_META[dose_name]["label"])
        ax.axhline(0, color="0.3", lw=0.8)
        ax.set_xlabel("checkpoint words (M)")
        ax.set_ylabel("EWoK view − repeat (score points)")
        ax.set_title("EWoK moves around chance; MAX recovers late")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_path)
        plt.close(fig)
        fig_status["created"] = True
    except Exception as exc:
        fig_status["error"] = repr(exc)

    summary = {
        "status": "ENTITY_LEG_RECONSTRUCTION_COMPLETE",
        "created_utc": now_utc(),
        "inputs": {
            "semantic_family_means": rel(FAMILY_DIR / "semantic_VR_family_means.csv"),
            "stable_row_harvest": rel(HARVEST_DIR / "harvested_deberta_stable_rows.csv"),
            "seed_spread": rel(SEED_DIR / "full_deberta_seed_ladder_seed_spread.csv"),
            "max_repeat_100_partial": rel(MAX_REPEAT_100),
        },
        "outputs": {
            "family_vr_common10_80_by_dose_csv": rel(OUT_DIR / "family_vr_common10_80_by_dose.csv"),
            "entity_vr_dose_curve_csv": rel(OUT_DIR / "entity_vr_dose_curve_common10_80.csv"),
            "ewok_absolute_ladder_csv": rel(OUT_DIR / "ewok_absolute_ladder_view_repeat.csv"),
            "ewok_absolute_summary_csv": rel(OUT_DIR / "ewok_absolute_summary_by_dose_arm.csv"),
            "ewok_vr_by_checkpoint_csv": rel(OUT_DIR / "ewok_vr_by_checkpoint.csv"),
            "ewok_vr_summary_csv": rel(OUT_DIR / "ewok_vr_summary_by_dose.csv"),
            "figure": fig_status,
        },
        "entity_curve_common10_80": entity_rows,
        "entity_fit_vs_rho": fit,
        "entity_finite_slopes": finite_slopes,
        "seed_scale_from_1x_replication": seed_scale,
        "ewok_vr_summary": ewok_vr_summary,
        "ewok_absolute_summary": ewok_abs_summary_rows,
        "quantitative_prediction_before_new_tables": prediction,
        "scientific_reading": {
            "primary": "The first-basin semantic leg is most cleanly read as a dose-scaled Entity/state-tracking gain: Entity V-R rises from +0.55875 to +1.06625 to +2.0625 over rho 0.042352, 0.07712, 0.111872.",
            "noise_floor_reconciliation": "At 1x this contributes only +0.0931 to cheap6, essentially the same scale as the 0.1133 trajectory-mean cheap6 seed difference, so minimum-dose six-family comparisons were underpowered for this specific family carrier.",
            "ewok": "EWoK relative V-R is negative over the common 10M-80M window but MAX turns positive at 90M/100M and all absolute EWoK levels remain near chance; EWoK should not be hardened as the cost side before full late rows and margin tables arrive.",
        },
        "no_model_inference_no_globalpiqa_superglue_aoa_upload_or_leaderboard": True,
    }
    (OUT_DIR / "entity_leg_reconstruction_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines: list[str] = []
    lines.append("# research Entity-leg reconstruction before fresh tables")
    lines.append("")
    lines.append("This readout uses only existing stable-family score files; it runs no model inference.")
    lines.append("")
    lines.append("## Entity V-R dose curve on common 10M-80M window")
    lines.append("")
    lines.append("| dose | rho | Entity V-R | contribution to cheap6 | contribution to cheap5 | effect/rho | ratio vs 1x |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for r in entity_rows:
        lines.append(f"| {r['dose_label']} | {r['rho']:.6f} | {r['V_minus_R_mean']:+.4f} | {r['entity_contribution_to_cheap6']:+.4f} | {r['entity_contribution_to_cheap5']:+.4f} | {r['effect_per_rho']:.2f} | {r['ratio_vs_1x_entity_effect']:.2f} |")
    lines.append("")
    lines.append(f"The finite slopes are {finite_slopes[0]['slope_per_rho']:.2f} score-points/rho from 1x to 1.82x and {finite_slopes[1]['slope_per_rho']:.2f} from 1.82x to MAX. A through-origin fit gives {fit['through_origin']['slope_per_rho']:.2f} score-points/rho; ordinary least squares gives slope {fit['ordinary_least_squares']['slope_per_rho']:.2f}, intercept {fit['ordinary_least_squares']['intercept']:+.3f}, R^2 {fit['ordinary_least_squares']['r2_three_points']:.3f} on the three points.")
    lines.append("")
    lines.append("## Noise-scale reconciliation")
    lines.append("")
    ch6 = seed_scale.get("cheap6_no_GlobalPIQA", {})
    ch5 = seed_scale.get("cheap5_no_GlobalPIQA_Reading", {})
    ent = seed_scale.get("Entity", {})
    lines.append(f"At 1x, the Entity leg contributes {one_entity['V_minus_R_mean']/6:+.4f} to cheap6. The 1x two-seed trajectory-mean cheap6 treatment difference is {ch6.get('trajectory_mean_signed_difference'):+.4f}, with pointwise mean absolute difference {ch6.get('pointwise_mean_abs_difference'):.4f}. This explains why minimum-dose six-family comparisons could miss the carrier even when Entity itself is moving.")
    lines.append(f"For cheap5 the 1x Entity contribution is {one_entity['V_minus_R_mean']/5:+.4f}, while the trajectory-mean cheap5 seed difference is {ch5.get('trajectory_mean_signed_difference'):+.4f}; at MAX the Entity contribution grows to {max_entity['V_minus_R_mean']/5:+.4f}.")
    if ent:
        lines.append(f"The 1x Entity family itself has a two-seed signed trajectory difference {ent.get('trajectory_mean_signed_difference'):+.4f}, pointwise mean absolute difference {ent.get('pointwise_mean_abs_difference'):.4f}, and pointwise max absolute difference {ent.get('pointwise_max_abs_difference'):.4f}; this is why the second-basin MAX measurement must be read directly, not inferred from one basin.")
    lines.append("")
    lines.append("## EWoK absolute level and relative movement")
    lines.append("")
    lines.append("| dose | window | n | mean EWoK V-R | view mean | repeat mean | fraction of paired checkpoints with both scores in [49,51] |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|")
    for r in ewok_vr_summary:
        lines.append(f"| {r['dose_label']} | {r['window']} | {r['n']} | {r['mean_V_minus_R_EWoK']:+.4f} | {r['mean_view_EWoK']:.3f} | {r['mean_repeat_EWoK']:.3f} | {r['fraction_pairs_both_within_49_51']:.2f} |")
    lines.append("")
    lines.append("EWoK is negative over the common 10M-80M relative leg, but MAX recovers at 90M/100M (+0.47 and +0.27). Across view/repeat ladders its absolute scores hover near chance, so the current first-basin science is safer as a dose-scaled Entity/state-tracking gain than as a settled Entity-versus-EWoK trade.")
    lines.append("")
    lines.append("## Frozen quantitative prediction")
    lines.append("")
    lines.append("The fresh second-basin MAX view-minus-repeat Entity score should be near +2 score points on the same stable-family readout, much closer to the first-basin MAX value (+2.0625) than to the first-basin 1x value (+0.5588). MAX breadth should fall well short of that Entity lift if source-conditioned re-expression is the carrier; if generic non-duplicate same-population companion text is sufficient, breadth should approach the compact-view Entity level. EWoK should be read as near-chance context unless the coming full-ladder or margin output shows stable movement away from 50.")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    for k, v in summary["outputs"].items():
        lines.append(f"- {k}: `{v}`")
    (OUT_DIR / "entity_leg_reconstruction_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": summary["status"],
        "summary_json": rel(OUT_DIR / "entity_leg_reconstruction_summary.json"),
        "summary_md": rel(OUT_DIR / "entity_leg_reconstruction_summary.md"),
        "entity_MAX_common10_80_V_minus_R": max_entity["V_minus_R_mean"],
        "entity_1x_common10_80_V_minus_R": one_entity["V_minus_R_mean"],
        "max_EWoK_available_10M_100M_mean_V_minus_R": [r for r in ewok_vr_summary if r["dose_name"] == "dose2p64" and r["window"] == "available_10M_100M"][0]["mean_V_minus_R_EWoK"],
        "figure_created": fig_status.get("created", False),
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
