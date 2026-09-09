#!/usr/bin/env python3
"""research: common-window budget-identity decomposition for compact dose.

Reads harvested stable-family rows and computes the fixed-budget identity

    V - C = (V - R) + (R - C)

on comparable checkpoint windows.  The scientific purpose is to separate true
view-arm improvement from deterioration of the exact-duplicate repeat
counterfactual as more of the fixed 10M-word budget is converted into
source-companion packets.

Stable selected-family scores only.  No model evaluation, GlobalPIQA,
SuperGLUE, AoA, upload, or leaderboard action.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import json
import math
import pathlib
import statistics
import time
from typing import Any

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
IN_ROWS = WS / "data/score_harvest_readout/harvested_deberta_stable_rows.csv"
OUT = WS / "data/common_window_budget_decomposition"

STABLE = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "Reading"]
FIVE = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]
METRICS = STABLE + ["cheap6_no_GlobalPIQA", "cheap5_no_GlobalPIQA_Reading", "EWoK_plus_Entity_sum"]
PRIMARY = ["cheap6_no_GlobalPIQA", "cheap5_no_GlobalPIQA_Reading"]
SECONDARY = ["EWoK_plus_Entity_sum"]
CKS_10_80 = [f"chck_{i}M" for i in range(10, 81, 10)]
CKS_10_100 = [f"chck_{i}M" for i in range(10, 101, 10)]
DOSES = ["dose1", "dose1p82", "dose2p64"]
ARM = {
    ("dose1", "view"): "dose1_view",
    ("dose1", "repeat"): "dose1_repeat",
    ("dose1p82", "view"): "dose1p82_view",
    ("dose1p82", "repeat"): "dose1p82_repeat",
    ("dose2p64", "view"): "max_view",
    ("dose2p64", "repeat"): "max_repeat",
}
DOSE_VALUE = {"dose1": 1.0, "dose1p82": 1.8209293539856442, "dose2p64": 2.641480921798262}
RHO = {"dose1": 0.042352, "dose1p82": 0.07712, "dose2p64": 0.111872}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def f(x: Any) -> float | None:
    if x is None or x == "" or x == "None":
        return None
    try:
        y = float(x)
        return y if math.isfinite(y) else None
    except Exception:
        return None


def words_from_ck(ck: str) -> int:
    return int(ck.split("_")[1].rstrip("M")) * 1_000_000


def read_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in ["dose", "rho", "words"] + METRICS:
            if k in r:
                y = f(r[k])
                if y is not None:
                    r[k] = y
    return rows


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    preferred = [
        "window", "checkpoint", "words", "dose_name", "dose", "rho", "metric",
        "clean", "view", "repeat", "V_minus_C", "R_minus_C", "V_minus_R",
        "identity_residual", "view_growth_vs_1x", "repeat_growth_vs_1x",
        "VR_growth_vs_1x", "repeat_deterioration_component", "view_growth_component",
        "view_share_of_VR_growth", "repeat_deterioration_share_of_VR_growth",
        "family", "mean", "median", "min", "max", "stdev", "n", "positive", "negative",
    ]
    for p in preferred:
        if p not in fields:
            fields.append(p)
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def stat(vals: list[float]) -> dict[str, Any]:
    return {
        "n": len(vals),
        "mean": statistics.mean(vals) if vals else None,
        "median": statistics.median(vals) if vals else None,
        "min": min(vals) if vals else None,
        "max": max(vals) if vals else None,
        "stdev": statistics.stdev(vals) if len(vals) > 1 else (0.0 if vals else None),
        "positive": sum(v > 0 for v in vals),
        "negative": sum(v < 0 for v in vals),
    }


def lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(r.get("arm")), str(r.get("checkpoint"))): r for r in rows}


def complete_window(by: dict[tuple[str, str], dict[str, Any]], checkpoints: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    missing: dict[str, list[str]] = {}
    for arm in ["clean0", "dose1_view", "dose1_repeat", "dose1p82_view", "dose1p82_repeat", "max_view", "max_repeat"]:
        # clean0 is only expected through 80M.
        cks = [ck for ck in checkpoints if arm != "clean0" or words_from_ck(ck) <= 80_000_000]
        miss = [ck for ck in cks if (arm, ck) not in by]
        if miss:
            missing[arm] = miss
    available = [ck for ck in checkpoints if all(((arm, ck) in by) for arm in ["clean0", "dose1_view", "dose1_repeat", "dose1p82_view", "dose1p82_repeat", "max_view", "max_repeat"] if not (arm == "clean0" and words_from_ck(ck) > 80_000_000))]
    return available, missing


def point_rows(rows: list[dict[str, Any]], checkpoints: list[str], window_name: str) -> list[dict[str, Any]]:
    by = lookup(rows)
    out: list[dict[str, Any]] = []
    for ck in checkpoints:
        c = by.get(("clean0", ck))
        if c is None:
            continue
        for dose in DOSES:
            v = by.get((ARM[(dose, "view")], ck))
            r = by.get((ARM[(dose, "repeat")], ck))
            if v is None or r is None:
                continue
            for m in METRICS:
                cv, vv, rv = f(c.get(m)), f(v.get(m)), f(r.get(m))
                if cv is None or vv is None or rv is None:
                    continue
                rec = {
                    "window": window_name,
                    "checkpoint": ck,
                    "words": words_from_ck(ck),
                    "dose_name": dose,
                    "dose": DOSE_VALUE[dose],
                    "rho": RHO[dose],
                    "metric": m,
                    "clean": cv,
                    "view": vv,
                    "repeat": rv,
                    "V_minus_C": vv - cv,
                    "R_minus_C": rv - cv,
                    "V_minus_R": vv - rv,
                    "identity_residual": (vv - cv) - ((vv - rv) + (rv - cv)),
                }
                out.append(rec)
    # Add growth terms versus 1x at same checkpoint/metric.
    base: dict[tuple[str, str], dict[str, Any]] = {(r["checkpoint"], r["metric"]): r for r in out if r["dose_name"] == "dose1"}
    for r in out:
        b = base.get((r["checkpoint"], r["metric"]))
        if b is None:
            continue
        r["view_growth_vs_1x"] = r["view"] - b["view"]
        r["repeat_growth_vs_1x"] = r["repeat"] - b["repeat"]
        r["VR_growth_vs_1x"] = r["V_minus_R"] - b["V_minus_R"]
        r["view_growth_component"] = r["view_growth_vs_1x"]
        r["repeat_deterioration_component"] = -r["repeat_growth_vs_1x"]
        den = r["VR_growth_vs_1x"]
        if abs(float(den)) > 1e-12:
            r["view_share_of_VR_growth"] = r["view_growth_component"] / den
            r["repeat_deterioration_share_of_VR_growth"] = r["repeat_deterioration_component"] / den
    return out


def summarize(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for window in sorted({str(r["window"]) for r in points}):
        for dose in DOSES:
            for m in METRICS:
                rs = [r for r in points if r["window"] == window and r["dose_name"] == dose and r["metric"] == m]
                if not rs:
                    continue
                row = {"window": window, "dose_name": dose, "dose": DOSE_VALUE[dose], "rho": RHO[dose], "metric": m}
                for key in ["V_minus_C", "R_minus_C", "V_minus_R", "identity_residual", "view_growth_vs_1x", "repeat_growth_vs_1x", "VR_growth_vs_1x", "view_growth_component", "repeat_deterioration_component", "view_share_of_VR_growth", "repeat_deterioration_share_of_VR_growth"]:
                    vals = [float(r[key]) for r in rs if f(r.get(key)) is not None]
                    s = stat(vals)
                    row[f"{key}_mean"] = s["mean"]
                    row[f"{key}_median"] = s["median"]
                    row[f"{key}_min"] = s["min"]
                    row[f"{key}_max"] = s["max"]
                    row[f"{key}_stdev"] = s["stdev"]
                    row[f"{key}_positive"] = s["positive"]
                    row[f"{key}_negative"] = s["negative"]
                row["n"] = len(rs)
                out.append(row)
    return out


def family_summary(points: list[dict[str, Any]], window: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for dose in DOSES:
        for family in STABLE:
            rs = [r for r in points if r["window"] == window and r["dose_name"] == dose and r["metric"] == family]
            for key in ["V_minus_C", "R_minus_C", "V_minus_R", "view_growth_vs_1x", "repeat_growth_vs_1x", "VR_growth_vs_1x"]:
                vals = [float(r[key]) for r in rs if f(r.get(key)) is not None]
                if not vals:
                    continue
                out.append({"window": window, "dose_name": dose, "dose": DOSE_VALUE[dose], "rho": RHO[dose], "family": family, "component": key, **stat(vals)})
    return out


def lfo_summary(points: list[dict[str, Any]], window: str) -> list[dict[str, Any]]:
    # Leave-one-family-out recomputed from family means per checkpoint, for each dose/component.
    out: list[dict[str, Any]] = []
    for dose in DOSES:
        cks = sorted({str(r["checkpoint"]) for r in points if r["window"] == window and r["dose_name"] == dose}, key=words_from_ck)
        for excluded in STABLE:
            keep6 = [x for x in STABLE if x != excluded]
            for component in ["V_minus_C", "R_minus_C", "V_minus_R", "view_growth_vs_1x", "repeat_growth_vs_1x", "VR_growth_vs_1x"]:
                vals6 = []
                vals5 = []
                for ck in cks:
                    rr = {(r["metric"]): r for r in points if r["window"] == window and r["dose_name"] == dose and r["checkpoint"] == ck}
                    xs = [f(rr[fam].get(component)) for fam in keep6 if fam in rr]
                    if len(xs) == len(keep6) and all(x is not None for x in xs):
                        vals6.append(sum(float(x) for x in xs) / len(xs))
                    keep5 = [x for x in FIVE if x != excluded]
                    if len(keep5) == 5:  # excluded Reading, keep all five cheap5 families
                        keep5 = FIVE[:]
                    ys = [f(rr[fam].get(component)) for fam in keep5 if fam in rr]
                    if len(ys) == len(keep5) and all(y is not None for y in ys):
                        vals5.append(sum(float(y) for y in ys) / len(ys))
                s6 = stat(vals6); s5 = stat(vals5)
                out.append({
                    "window": window,
                    "dose_name": dose,
                    "dose": DOSE_VALUE[dose],
                    "excluded_family": excluded,
                    "component": component,
                    "cheap6_lfo_mean": s6["mean"],
                    "cheap5_lfo_mean": s5["mean"],
                    "n": s6["n"],
                    "positive": s6["positive"],
                    "negative": s6["negative"],
                })
    return out


def make_plot(summary_rows: list[dict[str, Any]], out_path: pathlib.Path) -> str | None:
    if plt is None:
        return None
    rows = [r for r in summary_rows if r["window"] == "common_10M_80M" and r["metric"] in PRIMARY and r["dose_name"] in ["dose1p82", "dose2p64"]]
    if not rows:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)
    for ax, metric in zip(axes, PRIMARY):
        rr = [r for r in rows if r["metric"] == metric]
        labels = [r["dose_name"] for r in rr]
        view = [r.get("view_growth_component_mean") or 0.0 for r in rr]
        repdet = [r.get("repeat_deterioration_component_mean") or 0.0 for r in rr]
        vr = [r.get("VR_growth_vs_1x_mean") or 0.0 for r in rr]
        x = list(range(len(labels)))
        width = 0.24
        ax.axhline(0, color="#333333", lw=0.8)
        ax.bar([i - width for i in x], view, width, label="view growth Vd-V1", color="#4c78a8")
        ax.bar(x, repdet, width, label="repeat deterioration -(Rd-R1)", color="#f58518")
        ax.bar([i + width for i in x], vr, width, label="V-R growth", color="#54a24b")
        ax.set_xticks(x, labels)
        ax.set_title(metric)
        ax.set_ylabel("stable-family points")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("research common 10M-80M budget decomposition: why V-R rises with dose")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return rel(out_path)


def fmt(x: Any) -> str:
    y = f(x)
    return "NA" if y is None else f"{y:.4f}"


def direct_reading(summary_rows: list[dict[str, Any]], full_possible: bool) -> str:
    def row(window: str, dose: str, metric: str) -> dict[str, Any] | None:
        return next((r for r in summary_rows if r["window"] == window and r["dose_name"] == dose and r["metric"] == metric), None)
    parts: list[str] = []
    for metric in PRIMARY:
        d1 = row("common_10M_80M", "dose1", metric)
        d182 = row("common_10M_80M", "dose1p82", metric)
        d264 = row("common_10M_80M", "dose2p64", metric)
        if not (d1 and d182 and d264):
            continue
        parts.append(
            f"On the common 10M-80M window for {metric}, V-R is {fmt(d1.get('V_minus_R_mean'))} at 1x, "
            f"{fmt(d182.get('V_minus_R_mean'))} at 1.82x, and {fmt(d264.get('V_minus_R_mean'))} at 2.64x. "
            f"But view growth relative to 1x is {fmt(d182.get('view_growth_vs_1x_mean'))}/{fmt(d264.get('view_growth_vs_1x_mean'))}, "
            f"while repeat growth is {fmt(d182.get('repeat_growth_vs_1x_mean'))}/{fmt(d264.get('repeat_growth_vs_1x_mean'))}; "
            f"therefore the V-R increase decomposes into view component {fmt(d182.get('view_growth_component_mean'))}/{fmt(d264.get('view_growth_component_mean'))} "
            f"and duplicate-counterfactual deterioration component {fmt(d182.get('repeat_deterioration_component_mean'))}/{fmt(d264.get('repeat_deterioration_component_mean'))}."
        )
    parts.append(
        "This changes the first-basin interpretation: the increasing V-R curve cannot be named as compact-view scaling alone. "
        "It is a budget-allocation identity in which higher-dose exact duplication loses value as it displaces independent experience, while the view arm avoids part of that loss and may add some structured re-expression value."
    )
    if not full_possible:
        parts.append(
            "The full 10M-100M MAX decomposition is still incomplete in the input rows because max_repeat 90M/100M are missing or partial; this is exactly the late-exposure boundary where MAX view growth turns negative, so the late CPU finisher must be read before finalizing the turnover interpretation."
        )
    return " ".join(parts)


def render_md(summary: dict[str, Any]) -> str:
    lines = ["# research common-window budget decomposition", ""]
    lines.append("Stable selected families only. Scores are read through the identity `V-C = (V-R) + (R-C)` and growth decomposition `Δ(V-R)=ΔV-ΔR`.")
    lines.append("")
    lines.append("## Coverage")
    for k, v in summary["coverage"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Primary common 10M-80M means")
    for metric in PRIMARY:
        lines.append(f"### {metric}")
        lines.append("| dose | V-C | R-C | V-R | view growth vs 1x | repeat growth vs 1x | duplicate deterioration | V-R growth vs 1x |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for r in summary["primary_common_rows"]:
            if r["metric"] != metric:
                continue
            lines.append(f"| {r['dose_name']} | {fmt(r.get('V_minus_C_mean'))} | {fmt(r.get('R_minus_C_mean'))} | {fmt(r.get('V_minus_R_mean'))} | {fmt(r.get('view_growth_vs_1x_mean'))} | {fmt(r.get('repeat_growth_vs_1x_mean'))} | {fmt(r.get('repeat_deterioration_component_mean'))} | {fmt(r.get('VR_growth_vs_1x_mean'))} |")
        lines.append("")
    lines.append("## Reading")
    lines.append(summary["scientific_reading"])
    lines.append("")
    if summary.get("plot"):
        lines.append(f"Figure: `{summary['plot']}`")
    lines.append("")
    lines.append("## Files")
    for k, v in summary["files"].items():
        lines.append(f"- {k}: `{v}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-rows", default=str(IN_ROWS))
    ap.add_argument("--out-dir", default=str(OUT))
    args = ap.parse_args()
    in_rows = pathlib.Path(args.input_rows)
    if not in_rows.is_absolute():
        in_rows = ROOT / in_rows
    out_dir = pathlib.Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(in_rows)
    by = lookup(rows)
    common, missing_common = complete_window(by, CKS_10_80)
    full_all_arms, missing_full = complete_window(by, CKS_10_100)
    # Full 10-100 is only meaningful for V/R growth without clean; keep coverage
    # explicit and do not pretend clean exists beyond 80M.
    points_common = point_rows(rows, common, "common_10M_80M")
    summary_rows = summarize(points_common)
    fam = family_summary(points_common, "common_10M_80M")
    lfo = lfo_summary(points_common, "common_10M_80M")
    plot = make_plot(summary_rows, out_dir / "common10_80_budget_decomposition.png")

    write_csv(out_dir / "common10_80_point_decomposition.csv", points_common)
    write_csv(out_dir / "common10_80_summary_by_dose_metric.csv", summary_rows)
    write_csv(out_dir / "common10_80_family_components.csv", fam)
    write_csv(out_dir / "common10_80_leave_one_family_out.csv", lfo)

    primary_common = [r for r in summary_rows if r["metric"] in PRIMARY]
    full_possible = not any(k in missing_full for k in ["max_repeat"])
    summary = {
        "status": "COMMON_WINDOW_BUDGET_DECOMPOSITION_COMPLETE",
        "created_utc": now(),
        "input_rows": rel(in_rows),
        "coverage": {
            "common_10M_80M_checkpoints": common,
            "missing_common_10M_80M": missing_common,
            "full_10M_100M_all_available_checkpoints": full_all_arms,
            "missing_full_10M_100M": missing_full,
            "full_MAX_late_complete": full_possible,
        },
        "primary_common_rows": primary_common,
        "scientific_reading": direct_reading(summary_rows, full_possible),
        "plot": plot,
        "boundary": "Stable selected families only. No model evaluation, GlobalPIQA, SuperGLUE, AoA, upload, or leaderboard action.",
        "files": {
            "point_csv": rel(out_dir / "common10_80_point_decomposition.csv"),
            "summary_csv": rel(out_dir / "common10_80_summary_by_dose_metric.csv"),
            "family_csv": rel(out_dir / "common10_80_family_components.csv"),
            "lfo_csv": rel(out_dir / "common10_80_leave_one_family_out.csv"),
            "summary_json": rel(out_dir / "common_window_budget_decomposition_summary.json"),
            "summary_md": rel(out_dir / "common_window_budget_decomposition_summary.md"),
        },
    }
    (out_dir / "common_window_budget_decomposition_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "common_window_budget_decomposition_summary.md").write_text(render_md(summary), encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "coverage": summary["coverage"],
        "scientific_reading": summary["scientific_reading"],
        "plot": summary["plot"],
        "summary_json": summary["files"]["summary_json"],
        "summary_md": summary["files"]["summary_md"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
