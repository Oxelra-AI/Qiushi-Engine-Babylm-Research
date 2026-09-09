#!/usr/bin/env python3
"""research: RoBERTa low-dose and MAX-dose transfer readout.

This is a file-only readout.  It merges three RoBERTa evidence layers:
  1. the existing 1x compact-vs-repeat 100M endpoint from research;
  2. the existing MAX view-clean stable-family rows from research/265 harvests;
  3. the future MAX view-repeat rows produced by the research CPU-safe evaluator
     after the MAX repeat training finishes.

The script intentionally reports missing MAX repeat scores as missing rather
than imputing them.  It performs no model inference and no upload/leaderboard
work.
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
OUT_DEFAULT = WS / "data" / "roberta_transfer_readout"
LOW1_SUMMARY = WS / "data/roberta_minimal_selected_eval/minimal_selected_eval_summary.json"
MAX_VIEWCLEAN_CSV = WS / "data/score_harvest_after_max90/harvested_roberta_stable_rows.csv"
MAX_REPEAT_EVAL_ROOT = WS / "data/roberta_maxdose_repeat_eval/eval"
PREDICTION_JSON = WS / "data/entity_leg_reconstruction/entity_leg_reconstruction_summary.json"
CHECKPOINTS = [f"chck_{i}M" for i in range(10, 101, 10)]
FAMILIES = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "Reading"]
PRIMARY = ["cheap6_no_GlobalPIQA", "cheap5_no_GlobalPIQA_Reading"]


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
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    if not fields:
        fields = ["empty"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


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


def load_low1_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not LOW1_SUMMARY.exists():
        return [], {"exists": False, "path": rel(LOW1_SUMMARY)}
    s = read_json(LOW1_SUMMARY)
    rows: list[dict[str, Any]] = []
    for item in s.get("per_checkpoint", []):
        ck = item.get("ck")
        cmp = item.get("compact_minus_repeat") or {}
        compact = item.get("compact") or {}
        repeat = item.get("repeat") or {}
        for k, v in cmp.items():
            if finite(v):
                rows.append({
                    "coordinate": "roberta_1x_step221_endpoint",
                    "contrast": "view_minus_repeat",
                    "dose_name": "dose1",
                    "rho": 0.042352,
                    "checkpoint": ck,
                    "window": "endpoint_100M",
                    "metric": k,
                    "delta": float(v),
                    "view_score": (compact.get("scores") or {}).get(k) if k in FAMILIES else "",
                    "repeat_score": (repeat.get("scores") or {}).get(k) if k in FAMILIES else "",
                    "source": rel(LOW1_SUMMARY),
                })
    return rows, {"exists": True, "path": rel(LOW1_SUMMARY), "status": s.get("status"), "row_count": len(rows), "mean_compact_minus_repeat": s.get("mean_compact_minus_repeat")}


def read_harvest_csv(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def arm_key_from_harvest(row: dict[str, Any]) -> str:
    arm = row.get("arm") or ""
    data = row.get("data_arm") or ""
    if "roberta_view" in arm or data == "view":
        return "view"
    if "roberta_clean" in arm or data == "clean":
        return "clean"
    if "roberta_repeat" in arm or data == "repeat":
        return "repeat"
    return data or arm


def score_from_row(row: dict[str, Any], metric: str) -> float | None:
    if metric == "cheap6_no_GlobalPIQA":
        vals = [score_from_row(row, fam) for fam in FAMILIES]
        vals2 = [v for v in vals if v is not None]
        return sum(vals2) / len(vals2) if len(vals2) == len(FAMILIES) else None
    if metric == "cheap5_no_GlobalPIQA_Reading":
        vals = [score_from_row(row, fam) for fam in FAMILIES if fam != "Reading"]
        vals2 = [v for v in vals if v is not None]
        return sum(vals2) / len(vals2) if len(vals2) == len(FAMILIES) - 1 else None
    val = row.get(metric)
    return float(val) if finite(val) else None


def load_max_viewclean_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw = read_harvest_csv(MAX_VIEWCLEAN_CSV)
    by: dict[tuple[str, str], dict[str, Any]] = {}
    for r in raw:
        arm = arm_key_from_harvest(r)
        ck = r.get("checkpoint") or ""
        if arm in {"view", "clean"} and ck:
            by[(arm, ck)] = r
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for ck in CHECKPOINTS:
        v = by.get(("view", ck))
        c = by.get(("clean", ck))
        if not v or not c:
            missing.append({"checkpoint": ck, "view_present": bool(v), "clean_present": bool(c)})
            continue
        for metric in FAMILIES + PRIMARY:
            vs = score_from_row(v, metric)
            cs = score_from_row(c, metric)
            if vs is not None and cs is not None:
                rows.append({
                    "coordinate": "roberta_MAX_step260_viewclean",
                    "contrast": "view_minus_clean",
                    "dose_name": "dose2p64",
                    "rho": 0.111872,
                    "checkpoint": ck,
                    "window": "ladder",
                    "metric": metric,
                    "delta": vs - cs,
                    "view_score": vs,
                    "clean_score": cs,
                    "source": rel(MAX_VIEWCLEAN_CSV),
                })
    return rows, {"exists": MAX_VIEWCLEAN_CSV.exists(), "path": rel(MAX_VIEWCLEAN_CSV), "raw_row_count": len(raw), "contrast_row_count": len(rows), "missing_checkpoint_pairs": missing}


def stable_scores_from_payload(path: pathlib.Path) -> dict[str, float]:
    obj = read_json(path)
    out: dict[str, float] = {}
    stable = obj.get("stable_family_scores") or {}
    for fam in FAMILIES:
        if finite(stable.get(fam)):
            out[fam] = float(stable[fam])
            continue
        rec = (obj.get("tasks") or {}).get(fam) or {}
        if fam == "Reading":
            val = (rec.get("scores") or {}).get("Reading", rec.get("score"))
        else:
            val = rec.get("score")
        if finite(val):
            out[fam] = float(val)
    if len(out) == len(FAMILIES):
        out["cheap6_no_GlobalPIQA"] = sum(out[f] for f in FAMILIES) / len(FAMILIES)
        out["cheap5_no_GlobalPIQA_Reading"] = sum(out[f] for f in FAMILIES if f != "Reading") / (len(FAMILIES) - 1)
    return out


def existing_max_view_payload(ck: str) -> pathlib.Path:
    return WS / "data/roberta_viewclean_total_stable_eval/eval/per_target" / f"roberta_viewclean_total_view_{ck}.json"


def future_max_repeat_payload(ck: str) -> pathlib.Path:
    return MAX_REPEAT_EVAL_ROOT / "per_target" / f"roberta_max_repeat_seed43022_{ck}.json"


def load_max_viewrepeat_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for ck in CHECKPOINTS:
        vp = existing_max_view_payload(ck)
        rp = future_max_repeat_payload(ck)
        if not vp.exists() or not rp.exists():
            missing.append({"checkpoint": ck, "view_present": vp.exists(), "repeat_present": rp.exists(), "view_per_target": rel(vp), "repeat_per_target": rel(rp)})
            continue
        v = stable_scores_from_payload(vp)
        r = stable_scores_from_payload(rp)
        for metric in FAMILIES + PRIMARY:
            if metric in v and metric in r:
                rows.append({
                    "coordinate": "roberta_MAX_step269_viewrepeat",
                    "contrast": "view_minus_repeat",
                    "dose_name": "dose2p64",
                    "rho": 0.111872,
                    "checkpoint": ck,
                    "window": "ladder",
                    "metric": metric,
                    "delta": v[metric] - r[metric],
                    "view_score": v[metric],
                    "repeat_score": r[metric],
                    "view_source": rel(vp),
                    "repeat_source": rel(rp),
                })
    return rows, {"view_source_root": rel(WS / "data/roberta_viewclean_total_stable_eval/eval/per_target"), "repeat_eval_root": rel(MAX_REPEAT_EVAL_ROOT), "contrast_row_count": len(rows), "missing_checkpoint_pairs": missing}


def summarize_by(rows: list[dict[str, Any]], contrast: str, metrics: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for metric in metrics:
        for window, keep in [
            ("available_all", lambda m: True),
            ("common_10M_80M", lambda m: m <= 80),
            ("endpoint_100M", lambda m: m == 100),
            ("late_90M_100M", lambda m: m >= 90),
        ]:
            vals = []
            for r in rows:
                if r.get("contrast") != contrast or r.get("metric") != metric:
                    continue
                ck = str(r.get("checkpoint") or "")
                if ck.startswith("chck_") and ck.endswith("M"):
                    m = int(ck.split("_")[1].replace("M", ""))
                else:
                    m = 100
                if keep(m):
                    vals.append(float(r["delta"]))
            rec = {"contrast": contrast, "metric": metric, "window": window, **summarize(vals)}
            out.append(rec)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--require-max-repeat", action="store_true")
    args = ap.parse_args()
    out = pathlib.Path(args.out_dir)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    low1_rows, low1_meta = load_low1_rows()
    max_vc_rows, max_vc_meta = load_max_viewclean_rows()
    max_vr_rows, max_vr_meta = load_max_viewrepeat_rows()
    all_rows = low1_rows + max_vc_rows + max_vr_rows

    summary_rows = []
    summary_rows += summarize_by(low1_rows, "view_minus_repeat", ["Entity", "EWoK", "cheap6_no_GlobalPIQA", "cheap5_no_GlobalPIQA_Reading"])
    # keep only meaningful endpoint low-dose rows
    summary_rows = [r for r in summary_rows if not (r["contrast"] == "view_minus_repeat" and r["window"] not in {"available_all", "endpoint_100M"} and r.get("n") == 1)]
    summary_rows += summarize_by(max_vc_rows, "view_minus_clean", ["Entity", "EWoK", "cheap6_no_GlobalPIQA", "cheap5_no_GlobalPIQA_Reading"])
    summary_rows += summarize_by(max_vr_rows, "view_minus_repeat", ["Entity", "EWoK", "cheap6_no_GlobalPIQA", "cheap5_no_GlobalPIQA_Reading"])

    # Add explicit dose comparison rows for RoBERTa semantic leg when both are present.
    low_entity = next((r for r in low1_rows if r.get("metric") == "Entity"), None)
    max_entity_common_vals = [float(r["delta"]) for r in max_vr_rows if r.get("metric") == "Entity" and str(r.get("checkpoint", "")) in CHECKPOINTS[:8]]
    dose_comparison = {
        "low_dose_1x_entity_endpoint_delta": low_entity.get("delta") if low_entity else None,
        "max_dose_entity_common10_80_delta": sum(max_entity_common_vals) / len(max_entity_common_vals) if max_entity_common_vals else None,
        "max_minus_low_entity_delta": (sum(max_entity_common_vals) / len(max_entity_common_vals) - float(low_entity["delta"])) if (max_entity_common_vals and low_entity) else None,
        "expected_if_deberta_like": "+2-ish common-window Entity V-R at MAX, much larger than the 1x RoBERTa endpoint -0.41",
    }

    pred = read_json(PREDICTION_JSON) if PREDICTION_JSON.exists() else {}
    status = "ROBERTA_TRANSFER_READOUT_COMPLETE" if max_vr_meta["contrast_row_count"] > 0 else "ROBERTA_TRANSFER_READOUT_WAITING_FOR_MAX_REPEAT"
    if args.require_max_repeat and max_vr_meta["contrast_row_count"] == 0:
        status = "ROBERTA_TRANSFER_READOUT_MISSING_REQUIRED_MAX_REPEAT"

    write_csv(out / "roberta_transfer_contrast_rows.csv", all_rows)
    write_csv(out / "roberta_transfer_summary_rows.csv", summary_rows)

    interpretation = []
    if low_entity:
        interpretation.append(f"The inherited 1x RoBERTa endpoint has Entity V-R {float(low_entity['delta']):+.3f}, too small/negative to resolve the high-dose DeBERTa carrier.")
    vc_entity_vals = [float(r["delta"]) for r in max_vc_rows if r.get("metric") == "Entity"]
    if vc_entity_vals:
        interpretation.append(f"Existing MAX RoBERTa view-clean Entity deltas are available for {len(vc_entity_vals)} checkpoints with mean {sum(vc_entity_vals)/len(vc_entity_vals):+.3f}; this is total fixed-budget transfer, not semantic isolation.")
    if max_vr_meta["contrast_row_count"] == 0:
        interpretation.append("MAX RoBERTa view-repeat is intentionally missing until the research repeat training and CPU-safe scoring complete; no semantic-leg conclusion is drawn from view-clean alone.")
    else:
        me = dose_comparison["max_dose_entity_common10_80_delta"]
        interpretation.append(f"MAX RoBERTa view-repeat Entity common-window delta is now {float(me):+.3f}; compare this with DeBERTa MAX +2.0625 and RoBERTa 1x -0.410.")

    summary = {
        "status": status,
        "created_utc": now(),
        "low1_meta": low1_meta,
        "max_viewclean_meta": max_vc_meta,
        "max_viewrepeat_meta": max_vr_meta,
        "prediction_source": rel(PREDICTION_JSON),
        "deberta_reference": {
            "first_basin_MAX_entity_common10_80_V_minus_R": 2.0625,
            "first_basin_1x_entity_common10_80_V_minus_R": 0.55875,
            "entity_slope_r2": (((pred.get("entity_curve_fit") or {}).get("ordinary_least_squares") or {}).get("r2")),
        },
        "dose_comparison": dose_comparison,
        "summary_rows": summary_rows,
        "interpretation": " ".join(interpretation),
        "files": {
            "contrast_rows_csv": rel(out / "roberta_transfer_contrast_rows.csv"),
            "summary_rows_csv": rel(out / "roberta_transfer_summary_rows.csv"),
            "summary_json": rel(out / "roberta_transfer_readout_summary.json"),
            "summary_md": rel(out / "roberta_transfer_readout_summary.md"),
        },
        "no_model_inference_no_globalpiqa_superglue_aoa_upload_or_leaderboard": True,
    }
    write_json(out / "roberta_transfer_readout_summary.json", summary)

    lines = ["# research RoBERTa transfer readout", "", f"Status: `{status}`", "", summary["interpretation"], "", "## Key quantities", ""]
    lines.append(f"- Low-dose RoBERTa 1x endpoint Entity V-R: `{dose_comparison['low_dose_1x_entity_endpoint_delta']}`")
    lines.append(f"- MAX RoBERTa semantic Entity common-window V-R: `{dose_comparison['max_dose_entity_common10_80_delta']}`")
    lines.append(f"- MAX minus low-dose Entity delta: `{dose_comparison['max_minus_low_entity_delta']}`")
    lines.append("- DeBERTa reference: first-basin MAX Entity common-window V-R `+2.0625`; 1x `+0.55875`.")
    lines += ["", "## Missing MAX view-repeat rows", ""]
    for m in max_vr_meta.get("missing_checkpoint_pairs", [])[:12]:
        lines.append(f"- {m['checkpoint']}: view_present={m['view_present']} repeat_present={m['repeat_present']}")
    lines += ["", "## Files", ""]
    for k, v in summary["files"].items():
        lines.append(f"- {k}: `{v}`")
    (out / "roberta_transfer_readout_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"status": status, "summary_json": summary["files"]["summary_json"], "max_vr_rows": max_vr_meta["contrast_row_count"], "interpretation": summary["interpretation"]}, indent=2, ensure_ascii=False), flush=True)
    if args.require_max_repeat and max_vr_meta["contrast_row_count"] == 0:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
