#!/usr/bin/env python3
"""research: read second-basin MAX EWoK/Entity stable-family outputs.

This script is intended to run after the CPU-safe arm ladder jobs have produced
per-target JSONs for both second-basin MAX view and repeat.  It does not score
models.  It merges the two arms, computes V-R over common checkpoints, and
compares Entity against the frozen research first-basin prediction.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import statistics as stats
import time
from typing import Any

ROOT = pathlib.Path.cwd()
WS = ROOT / "experiments/archive" / 'frontier_consolidation'
DEFAULT_VIEW_ROOT = WS / "data/second_basin_entity_ewok_eval/view_eval"
DEFAULT_REPEAT_ROOT = WS / "data/second_basin_entity_ewok_eval/repeat_eval"
DEFAULT_OUT = WS / "data/second_basin_entity_ewok_readout"
PREDICTION_JSON = WS / "data/entity_leg_reconstruction/entity_leg_reconstruction_summary.json"
CHECKPOINTS = [f"chck_{i}M" for i in range(10, 101, 10)]
FAMILIES = ["EWoK", "Entity"]


def now() -> str:
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


def read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: pathlib.Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    if not fieldnames:
        fieldnames = ["empty"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def target(prefix: str, ck: str) -> str:
    return f"{prefix}_{ck}"


def load_score(root: pathlib.Path, prefix: str, ck: str, family: str) -> tuple[float | None, dict[str, Any] | None, pathlib.Path]:
    path = root / "per_target" / f"{target(prefix, ck)}.json"
    if not path.exists():
        return None, None, path
    obj = read_json(path)
    stable = obj.get("stable_family_scores") or {}
    if finite(stable.get(family)):
        return float(stable[family]), obj, path
    rec = (obj.get("tasks") or {}).get(family) or {}
    val = rec.get("score")
    if finite(val):
        return float(val), obj, path
    return None, obj, path


def summarize(vals: list[float]) -> dict[str, Any]:
    vals = [float(v) for v in vals if finite(v)]
    if not vals:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None, "stdev": None, "positive": 0, "negative": 0}
    return {
        "n": len(vals),
        "mean": sum(vals) / len(vals),
        "median": stats.median(vals),
        "min": min(vals),
        "max": max(vals),
        "stdev": stats.pstdev(vals) if len(vals) > 1 else 0.0,
        "positive": sum(1 for v in vals if v > 0),
        "negative": sum(1 for v in vals if v < 0),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--view-root", default=str(DEFAULT_VIEW_ROOT))
    ap.add_argument("--repeat-root", default=str(DEFAULT_REPEAT_ROOT))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--require-complete", action="store_true")
    args = ap.parse_args()

    view_root = pathlib.Path(args.view_root)
    repeat_root = pathlib.Path(args.repeat_root)
    out = pathlib.Path(args.out_dir)
    if not view_root.is_absolute():
        view_root = ROOT / view_root
    if not repeat_root.is_absolute():
        repeat_root = ROOT / repeat_root
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    pred = read_json(PREDICTION_JSON) if PREDICTION_JSON.exists() else {}
    pred_entity = (((pred.get("quantitative_prediction_before_new_tables") or {}).get("second_basin_MAX_view_minus_repeat_Entity_common_window") or {}).get("expected_center_score_points"))
    first_entity_curve = {r.get("dose_name"): r for r in pred.get("entity_curve_common10_80", [])} if pred else {}

    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for ck in CHECKPOINTS:
        words_m = int(ck.split("_")[1].replace("M", ""))
        for fam in FAMILIES:
            v, vobj, vp = load_score(view_root, "second_basin_max_view_seed43122", ck, fam)
            r, robj, rp = load_score(repeat_root, "second_basin_max_repeat_seed43122", ck, fam)
            if v is None or r is None:
                missing.append({"checkpoint": ck, "family": fam, "view_score_present": v is not None, "repeat_score_present": r is not None, "view_per_target": rel(vp), "repeat_per_target": rel(rp)})
                continue
            rows.append({
                "checkpoint": ck,
                "words_m": words_m,
                "family": fam,
                "view_score": v,
                "repeat_score": r,
                "V_minus_R": v - r,
                "view_minus_50": v - 50.0 if fam == "EWoK" else "",
                "repeat_minus_50": r - 50.0 if fam == "EWoK" else "",
                "both_EWoK_within_49_51": (49.0 <= v <= 51.0 and 49.0 <= r <= 51.0) if fam == "EWoK" else "",
                "view_per_target": rel(vp),
                "repeat_per_target": rel(rp),
            })
    write_csv(out / "second_basin_entity_ewok_by_checkpoint.csv", rows)
    write_csv(out / "second_basin_entity_ewok_missing.csv", missing)

    summary_rows: list[dict[str, Any]] = []
    for fam in FAMILIES:
        for window_name, pred_ck in [("common_10M_80M", lambda m: m <= 80), ("full_10M_100M", lambda m: True), ("late_90M_100M", lambda m: m >= 90)]:
            subset = [r for r in rows if r["family"] == fam and pred_ck(int(r["words_m"]))]
            vals = [float(r["V_minus_R"]) for r in subset]
            rec = {"family": fam, "window": window_name, **summarize(vals)}
            if fam == "EWoK" and subset:
                rec["mean_view_score"] = sum(float(r["view_score"]) for r in subset) / len(subset)
                rec["mean_repeat_score"] = sum(float(r["repeat_score"]) for r in subset) / len(subset)
                rec["fraction_both_scores_49_51"] = sum(1 for r in subset if r["both_EWoK_within_49_51"] is True) / len(subset)
            if fam == "Entity" and window_name == "common_10M_80M" and rec.get("mean") is not None and pred_entity is not None:
                rec["first_basin_predicted_MAX_entity_VR"] = pred_entity
                rec["difference_from_first_basin_MAX_entity"] = float(rec["mean"]) - float(pred_entity)
                rec["ratio_to_first_basin_MAX_entity"] = float(rec["mean"]) / float(pred_entity) if float(pred_entity) != 0 else None
                rec["ratio_to_first_basin_1x_entity"] = float(rec["mean"]) / float(first_entity_curve.get("dose1", {}).get("V_minus_R_mean", float("nan")))
            summary_rows.append(rec)
    write_csv(out / "second_basin_entity_ewok_summary.csv", summary_rows)

    complete = len(missing) == 0
    status = "SECOND_BASIN_ENTITY_EWOK_READOUT_COMPLETE" if complete else "SECOND_BASIN_ENTITY_EWOK_READOUT_INCOMPLETE"
    if args.require_complete and not complete:
        status = "SECOND_BASIN_ENTITY_EWOK_READOUT_MISSING_REQUIRED"

    interpretation = "pending complete view/repeat score rows"
    ent_common = next((r for r in summary_rows if r["family"] == "Entity" and r["window"] == "common_10M_80M"), None)
    ewok_common = next((r for r in summary_rows if r["family"] == "EWoK" and r["window"] == "common_10M_80M"), None)
    if ent_common and ent_common.get("mean") is not None:
        ent_mean = float(ent_common["mean"])
        if pred_entity is not None and ent_mean >= 1.5:
            interpretation = "second-basin Entity V-R is in the predeclared MAX-scale range; compare breadth next before attributing carrier specificity"
        elif ent_mean > 0.5:
            interpretation = "second-basin Entity V-R remains positive but is below the first-basin MAX-scale prediction; inspect checkpoint/family breadth before strengthening the principle"
        else:
            interpretation = "second-basin Entity V-R does not reproduce the first-basin MAX-scale lift; the dose-scaled Entity mechanism must be revised or demoted"
    if ewok_common and ewok_common.get("mean") is not None:
        interpretation += f"; common-window EWoK V-R={float(ewok_common['mean']):+.3f} and should be read together with absolute EWoK levels."

    summary = {
        "status": status,
        "created_utc": now(),
        "view_root": rel(view_root),
        "repeat_root": rel(repeat_root),
        "prediction_source": rel(PREDICTION_JSON),
        "missing_count": len(missing),
        "row_count": len(rows),
        "summary_rows": summary_rows,
        "missing": missing,
        "interpretation": interpretation,
        "files": {
            "by_checkpoint_csv": rel(out / "second_basin_entity_ewok_by_checkpoint.csv"),
            "summary_csv": rel(out / "second_basin_entity_ewok_summary.csv"),
            "missing_csv": rel(out / "second_basin_entity_ewok_missing.csv"),
            "summary_json": rel(out / "second_basin_entity_ewok_readout_summary.json"),
            "summary_md": rel(out / "second_basin_entity_ewok_readout_summary.md"),
        },
        "no_model_inference_no_globalpiqa_superglue_aoa_upload_or_leaderboard": True,
    }
    write_json(out / "second_basin_entity_ewok_readout_summary.json", summary)

    lines = ["# research second-basin Entity/EWoK readout", "", f"Status: `{status}`", "", interpretation, "", "## Summary", "", "| family | window | n | mean V-R | median | min | max | positive | negative |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary_rows:
        mean = r.get("mean")
        lines.append(f"| {r['family']} | {r['window']} | {r.get('n')} | {float(mean):+.4f} | {float(r['median']):+.4f} | {float(r['min']):+.4f} | {float(r['max']):+.4f} | {r.get('positive')} | {r.get('negative')} |" if mean is not None else f"| {r['family']} | {r['window']} | 0 |  |  |  |  |  |  |")
    if missing:
        lines += ["", "## Missing rows", ""]
        for m in missing[:20]:
            lines.append(f"- {m['checkpoint']} {m['family']}: view_present={m['view_score_present']} repeat_present={m['repeat_score_present']}")
    lines += ["", "## Files", ""]
    for k, v in summary["files"].items():
        lines.append(f"- {k}: `{v}`")
    (out / "second_basin_entity_ewok_readout_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"status": status, "missing_count": len(missing), "row_count": len(rows), "summary_json": summary["files"]["summary_json"], "interpretation": interpretation}, indent=2, ensure_ascii=False), flush=True)
    if args.require_complete and not complete:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
