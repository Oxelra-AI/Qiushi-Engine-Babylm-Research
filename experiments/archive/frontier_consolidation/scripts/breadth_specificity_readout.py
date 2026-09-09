#!/usr/bin/env python3
"""research: DeBERTa MAX view-versus-breadth EWoK/Entity readout.

File-only merger for the specificity leg of the fixed-budget mechanism test.
It compares the existing first-basin MAX compact-view scores with future MAX
source-breadth scores produced by the CPU-safe evaluator.  Missing breadth rows
are reported explicitly.  No model inference is performed here.
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
OUT_DEFAULT = WS / "data/breadth_specificity_readout"
MAX_VIEW_PER_TARGET = WS / "data/dose_ladder_stable_eval/eval/per_target"
BREADTH_EVAL_ROOT = WS / "data/breadth_entity_ewok_eval/eval"
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


def score_from_payload(path: pathlib.Path, family: str) -> float | None:
    if not path.exists():
        return None
    obj = read_json(path)
    stable = obj.get("stable_family_scores") or {}
    if finite(stable.get(family)):
        return float(stable[family])
    rec = (obj.get("tasks") or {}).get(family) or {}
    val = rec.get("score")
    if finite(val):
        return float(val)
    return None


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
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--breadth-root", default=str(BREADTH_EVAL_ROOT))
    ap.add_argument("--require-breadth", action="store_true")
    args = ap.parse_args()
    out = pathlib.Path(args.out_dir)
    breadth_root = pathlib.Path(args.breadth_root)
    if not out.is_absolute():
        out = ROOT / out
    if not breadth_root.is_absolute():
        breadth_root = ROOT / breadth_root
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for ck in CHECKPOINTS:
        words = int(ck.split("_")[1].replace("M", ""))
        vp = MAX_VIEW_PER_TARGET / f"dose_max_view_{ck}.json"
        bp = breadth_root / "per_target" / f"max_breadth_seed43022_{ck}.json"
        for fam in FAMILIES:
            v = score_from_payload(vp, fam)
            b = score_from_payload(bp, fam)
            if v is None or b is None:
                missing.append({"checkpoint": ck, "family": fam, "view_present": v is not None, "breadth_present": b is not None, "view_per_target": rel(vp), "breadth_per_target": rel(bp)})
                continue
            rows.append({
                "checkpoint": ck,
                "words_m": words,
                "family": fam,
                "view_score": v,
                "breadth_score": b,
                "view_minus_breadth": v - b,
                "view_per_target": rel(vp),
                "breadth_per_target": rel(bp),
            })
    write_csv(out / "breadth_specificity_by_checkpoint.csv", rows)
    write_csv(out / "breadth_specificity_missing.csv", missing)

    summary_rows: list[dict[str, Any]] = []
    for fam in FAMILIES:
        for window, keep in [("common_10M_80M", lambda m: m <= 80), ("full_10M_100M", lambda m: True), ("late_90M_100M", lambda m: m >= 90)]:
            vals = [float(r["view_minus_breadth"]) for r in rows if r["family"] == fam and keep(int(r["words_m"]))]
            rec = {"family": fam, "window": window, **summarize(vals)}
            if fam == "EWoK" and vals:
                subset = [r for r in rows if r["family"] == fam and keep(int(r["words_m"]))]
                rec["mean_view_score"] = sum(float(r["view_score"]) for r in subset) / len(subset)
                rec["mean_breadth_score"] = sum(float(r["breadth_score"]) for r in subset) / len(subset)
            summary_rows.append(rec)
    write_csv(out / "breadth_specificity_summary.csv", summary_rows)

    pred = read_json(PREDICTION_JSON) if PREDICTION_JSON.exists() else {}
    first_max_entity = 2.0625
    status = "BREADTH_SPECIFICITY_READOUT_COMPLETE" if len(missing) == 0 else "BREADTH_SPECIFICITY_READOUT_WAITING_FOR_BREADTH"
    if args.require_breadth and missing:
        status = "BREADTH_SPECIFICITY_READOUT_MISSING_REQUIRED_BREADTH"
    ent_common = next((r for r in summary_rows if r["family"] == "Entity" and r["window"] == "common_10M_80M"), None)
    interpretation = "Breadth EWoK/Entity scores are not yet complete; no view-minus-breadth specificity conclusion is drawn."
    if ent_common and ent_common.get("mean") is not None:
        ev = float(ent_common["mean"])
        interpretation = f"MAX view-minus-breadth Entity common-window delta is {ev:+.3f}; compare with DeBERTa view-minus-repeat Entity {first_max_entity:+.4f}. A small or negative value favors generic anti-duplication/content-breadth, while a large positive value supports source-conditioned re-expression specificity."

    summary = {
        "status": status,
        "created_utc": now(),
        "max_view_source_root": rel(MAX_VIEW_PER_TARGET),
        "breadth_eval_root": rel(breadth_root),
        "prediction_source": rel(PREDICTION_JSON),
        "deberta_reference": {
            "MAX_view_minus_repeat_entity_common10_80": first_max_entity,
            "entity_slope_r2": (((pred.get("entity_curve_fit") or {}).get("ordinary_least_squares") or {}).get("r2")),
        },
        "row_count": len(rows),
        "missing_count": len(missing),
        "summary_rows": summary_rows,
        "missing": missing,
        "interpretation": interpretation,
        "files": {
            "by_checkpoint_csv": rel(out / "breadth_specificity_by_checkpoint.csv"),
            "summary_csv": rel(out / "breadth_specificity_summary.csv"),
            "missing_csv": rel(out / "breadth_specificity_missing.csv"),
            "summary_json": rel(out / "breadth_specificity_readout_summary.json"),
            "summary_md": rel(out / "breadth_specificity_readout_summary.md"),
        },
        "no_model_inference_no_globalpiqa_superglue_aoa_upload_or_leaderboard": True,
    }
    write_json(out / "breadth_specificity_readout_summary.json", summary)
    lines = ["# research breadth specificity readout", "", f"Status: `{status}`", "", interpretation, "", "## Missing rows", ""]
    for m in missing[:20]:
        lines.append(f"- {m['checkpoint']} {m['family']}: view_present={m['view_present']} breadth_present={m['breadth_present']}")
    lines += ["", "## Files", ""]
    for k, v in summary["files"].items():
        lines.append(f"- {k}: `{v}`")
    (out / "breadth_specificity_readout_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "row_count": len(rows), "missing_count": len(missing), "summary_json": summary["files"]["summary_json"], "interpretation": interpretation}, indent=2, ensure_ascii=False), flush=True)
    if args.require_breadth and missing:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
