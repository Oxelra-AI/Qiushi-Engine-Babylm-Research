#!/usr/bin/env python3
"""research: familywise arm-growth decomposition for the MAX-dose mechanism decision.

This file-only analysis separates, for each stable family, the dose growth of
view and repeat arms relative to the inherited 1x packet dose:

    view_growth = V_d - V_1
    repeat_growth = R_d - R_1
    V-R growth = (V_d-R_d) - (V_1-R_1) = view_growth - repeat_growth

The scientific purpose is to decide whether the apparent MAX Entity carrier is
mostly the view arm rising, the repeat arm collapsing under increased duplicate
recurrence, or both.  It uses only already harvested stable-family scores on the
common 10M-80M window and performs no model inference, training, upload, or
leaderboard action.
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
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
WS = ROOT / "experiments/archive" / 'frontier_consolidation'
IN_POINTS = WS / "data" / "common_window_budget_decomposition" / "common10_80_point_decomposition.csv"
OUT = WS / "data" / "family_arm_growth_decomposition"

STABLE = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "Reading"]
FIVE = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]
DOSES = ["dose1", "dose1p82", "dose2p64"]
PRIMARY_COMPONENTS = [
    "V_minus_C",
    "R_minus_C",
    "V_minus_R",
    "view_growth_vs_1x",
    "repeat_growth_vs_1x",
    "VR_growth_vs_1x",
]


def rel(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except Exception:
        return str(path)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def f(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, float) and math.isfinite(x):
        return x
    s = str(x).strip()
    if not s:
        return None
    try:
        y = float(s)
        return y if math.isfinite(y) else None
    except Exception:
        return None


def mean(xs: list[float]) -> float | None:
    return statistics.mean(xs) if xs else None


def stdev(xs: list[float]) -> float | None:
    return statistics.stdev(xs) if len(xs) > 1 else (0.0 if xs else None)


def read_points(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in ["words", "dose", "rho"] + PRIMARY_COMPONENTS:
            if k in r:
                val = f(r[k])
                if val is not None:
                    r[k] = val
    return rows


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    preferred = ["window", "dose_name", "metric", "family_set", "n", "checkpoints"] + [f"{c}_{s}" for c in PRIMARY_COMPONENTS for s in ["mean", "median", "min", "max", "stdev", "positive", "negative"]] + ["view_fraction_of_positive_vr_growth", "repeat_collapse_fraction_of_positive_vr_growth"]
    for p in preferred:
        if p not in fields:
            fields.append(p)
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def summarize_values(vals: list[float]) -> dict[str, Any]:
    if not vals:
        return {"mean": None, "median": None, "min": None, "max": None, "stdev": None, "positive": 0, "negative": 0}
    return {
        "mean": mean(vals),
        "median": statistics.median(vals),
        "min": min(vals),
        "max": max(vals),
        "stdev": stdev(vals),
        "positive": sum(1 for v in vals if v > 0),
        "negative": sum(1 for v in vals if v < 0),
    }


def summarize_groups(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    metrics = STABLE + ["cheap6_no_GlobalPIQA", "cheap5_no_GlobalPIQA_Reading", "EWoK_plus_Entity_sum"]
    for dose in DOSES:
        for metric in metrics:
            group = [r for r in points if r.get("dose_name") == dose and r.get("metric") == metric]
            if not group:
                continue
            row: dict[str, Any] = {
                "window": "common_10M_80M",
                "dose_name": dose,
                "metric": metric,
                "family_set": "native" if metric in STABLE else "composite",
                "n": len(group),
                "checkpoints": ";".join(str(r.get("checkpoint")) for r in sorted(group, key=lambda x: int(f(x.get("words")) or 0))),
            }
            for comp in PRIMARY_COMPONENTS:
                vals = [f(r.get(comp)) for r in group]
                vals = [float(v) for v in vals if v is not None]
                s = summarize_values(vals)
                for k, v in s.items():
                    row[f"{comp}_{k}"] = v
            vr = f(row.get("VR_growth_vs_1x_mean"))
            vg = f(row.get("view_growth_vs_1x_mean"))
            rg = f(row.get("repeat_growth_vs_1x_mean"))
            if vr is not None and abs(vr) > 1e-12:
                row["view_fraction_of_positive_vr_growth"] = (vg or 0.0) / vr
                row["repeat_collapse_fraction_of_positive_vr_growth"] = (-(rg or 0.0)) / vr
            out.append(row)
    return out


def subset_composite(points: list[dict[str, Any]], dose: str, family_subset: list[str], label: str) -> dict[str, Any]:
    # Average per checkpoint across the requested family subset, then summarize
    # across checkpoints.  This avoids overweighting checkpoints with any missing
    # component; research point table is complete for the common window.
    cks = sorted({str(r.get("checkpoint")) for r in points if r.get("dose_name") == dose}, key=lambda ck: int(ck.split("_")[1].rstrip("M")))
    row: dict[str, Any] = {"window": "common_10M_80M", "dose_name": dose, "metric": label, "family_set": "+".join(family_subset), "n": 0, "checkpoints": ";".join(cks)}
    for comp in PRIMARY_COMPONENTS:
        vals: list[float] = []
        for ck in cks:
            rr = {(str(r.get("metric"))): r for r in points if r.get("dose_name") == dose and r.get("checkpoint") == ck and r.get("metric") in family_subset}
            xs = [f(rr[fam].get(comp)) for fam in family_subset if fam in rr]
            if len(xs) == len(family_subset) and all(x is not None for x in xs):
                vals.append(sum(float(x) for x in xs) / len(xs))
        row["n"] = max(int(row.get("n") or 0), len(vals))
        s = summarize_values(vals)
        for k, v in s.items():
            row[f"{comp}_{k}"] = v
    vr = f(row.get("VR_growth_vs_1x_mean"))
    vg = f(row.get("view_growth_vs_1x_mean"))
    rg = f(row.get("repeat_growth_vs_1x_mean"))
    if vr is not None and abs(vr) > 1e-12:
        row["view_fraction_of_positive_vr_growth"] = (vg or 0.0) / vr
        row["repeat_collapse_fraction_of_positive_vr_growth"] = (-(rg or 0.0)) / vr
    return row


def fmt(x: Any) -> str:
    y = f(x)
    return "NA" if y is None else f"{y:+.4f}"


def make_interpretation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def find(metric: str, dose: str = "dose2p64") -> dict[str, Any]:
        return next(r for r in rows if r.get("metric") == metric and r.get("dose_name") == dose)

    ent = find("Entity")
    ex6 = find("cheap6_exEntity")
    ex5 = find("cheap5_exEntity")
    c6 = find("cheap6_no_GlobalPIQA")
    c5 = find("cheap5_no_GlobalPIQA_Reading")
    return {
        "max_entity_common_10_80": {
            "V_minus_R_mean": f(ent.get("V_minus_R_mean")),
            "view_growth_vs_1x_mean": f(ent.get("view_growth_vs_1x_mean")),
            "repeat_growth_vs_1x_mean": f(ent.get("repeat_growth_vs_1x_mean")),
            "VR_growth_vs_1x_mean": f(ent.get("VR_growth_vs_1x_mean")),
            "view_fraction_of_VR_growth": f(ent.get("view_fraction_of_positive_vr_growth")),
            "repeat_collapse_fraction_of_VR_growth": f(ent.get("repeat_collapse_fraction_of_positive_vr_growth")),
            "reading": "At MAX on the common 10M-80M first-basin window, Entity V-R is not explained by repeat collapse alone: relative to 1x, the view arm rises by about +0.99 pp while the repeat arm falls by about -0.52 pp, giving about +1.50 pp V-R growth. This still does not prove record addressability, because research showed the view-arm movement can be operation-propensity shaped.",
        },
        "max_primary_composites_common_10_80": {
            "cheap6_VC": f(c6.get("V_minus_C_mean")),
            "cheap6_RC": f(c6.get("R_minus_C_mean")),
            "cheap6_VR": f(c6.get("V_minus_R_mean")),
            "cheap5_VC": f(c5.get("V_minus_C_mean")),
            "cheap5_RC": f(c5.get("R_minus_C_mean")),
            "cheap5_VR": f(c5.get("V_minus_R_mean")),
        },
        "max_ex_entity_common_10_80": {
            "cheap6_exEntity_VR": f(ex6.get("V_minus_R_mean")),
            "cheap6_exEntity_VC": f(ex6.get("V_minus_C_mean")),
            "cheap6_exEntity_RC": f(ex6.get("R_minus_C_mean")),
            "cheap5_exEntity_VR": f(ex5.get("V_minus_R_mean")),
            "cheap5_exEntity_VC": f(ex5.get("V_minus_C_mean")),
            "cheap5_exEntity_RC": f(ex5.get("R_minus_C_mean")),
            "reading": "Removing Entity leaves MAX V-R near the earlier seed-noise scale, even though both MAX view and MAX repeat remain above the 1x-geometry clean arm. If Entity does not reproduce in the second basin, the surviving result is a fixed-budget allocation contrast, not a source-correspondence mechanism.",
        },
    }


def render_md(payload: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# research familywise arm-growth decomposition")
    lines.append("")
    lines.append("This readout separates `V_d−V_1` from `R_d−R_1` on the common 10M-80M window, using existing research scores only.")
    lines.append("")
    lines.append("## MAX dose: stable-family means")
    lines.append("")
    lines.append("| family/composite | V-C | R-C | V-R | Vd-V1 | Rd-R1 | Δ(V-R) vs 1x |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for metric in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "Reading", "cheap6_no_GlobalPIQA", "cheap6_exEntity", "cheap5_no_GlobalPIQA_Reading", "cheap5_exEntity"]:
        r = next((x for x in rows if x.get("dose_name") == "dose2p64" and x.get("metric") == metric), None)
        if not r:
            continue
        lines.append(f"| {metric} | {fmt(r.get('V_minus_C_mean'))} | {fmt(r.get('R_minus_C_mean'))} | {fmt(r.get('V_minus_R_mean'))} | {fmt(r.get('view_growth_vs_1x_mean'))} | {fmt(r.get('repeat_growth_vs_1x_mean'))} | {fmt(r.get('VR_growth_vs_1x_mean'))} |")
    lines.append("")
    lines.append("## Mechanism reading")
    interp = payload["interpretation"]
    lines.append(f"- Entity: {interp['max_entity_common_10_80']['reading']}")
    lines.append(f"- Ex-Entity: {interp['max_ex_entity_common_10_80']['reading']}")
    lines.append("- Route consequence: the prepared permuted-companion arm should remain unlaunched unless second-basin official Entity and breadth results show a reproducible aligned carrier that is not mostly an operation-propensity redistribution.")
    lines.append("")
    lines.append("## Files")
    for k, v in payload["files"].items():
        lines.append(f"- {k}: `{v}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    points = read_points(IN_POINTS)
    rows = summarize_groups(points)
    for dose in DOSES:
        rows.append(subset_composite(points, dose, [x for x in STABLE if x != "Entity"], "cheap6_exEntity"))
        rows.append(subset_composite(points, dose, [x for x in FIVE if x != "Entity"], "cheap5_exEntity"))
        rows.append(subset_composite(points, dose, ["EWoK", "Entity"], "EWoK_plus_Entity_mean"))
    rows = sorted(rows, key=lambda r: (str(r.get("dose_name")), str(r.get("family_set")), str(r.get("metric"))))
    csv_path = OUT / "family_arm_growth_common10_80.csv"
    write_csv(csv_path, rows)
    payload = {
        "status": "FAMILY_ARM_GROWTH_DECOMPOSITION_DONE",
        "created_utc": now(),
        "input_points": rel(IN_POINTS),
        "row_count": len(rows),
        "interpretation": make_interpretation(rows),
        "boundary": "File-only stable selected-family scores from research common 10M-80M window; no model inference, training, GlobalPIQA, SuperGLUE, AoA, upload, or leaderboard action.",
        "files": {
            "csv": rel(csv_path),
            "summary_json": rel(OUT / "family_arm_growth_decomposition_summary.json"),
            "summary_md": rel(OUT / "family_arm_growth_decomposition_summary.md"),
        },
    }
    (OUT / "family_arm_growth_decomposition_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "family_arm_growth_decomposition_summary.md").write_text(render_md(payload, rows), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "interpretation": payload["interpretation"], "files": payload["files"]}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
