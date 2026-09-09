#!/usr/bin/env python3
"""research: read dose-ladder exposure profiles against the research seed-spread reference.

This script runs after dose_ladder_stable_eval.py has produced its CSVs.
It summarizes the scientific statistic that motivated the experiment: not a
single checkpoint, but the exposure profile of treatment-vs-clean and
compact-minus-repeat as dose increases from 1x to matched MAX.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import statistics
from typing import Any

ROOT = pathlib.Path("experiments/archive/frontier_consolidation")
DOSE_DIR_DEFAULT = ROOT / "data/dose_ladder_stable_eval"
OUT_DIR_DEFAULT = ROOT / "data/dose_ladder_profile_readout"
SEED_SPREAD_DEFAULT = ROOT / "data/full_deberta_seed_ladder_stable_eval/full_deberta_seed_ladder_seed_spread.csv"
METRICS = ["cheap6_no_GlobalPIQA", "cheap5_no_GlobalPIQA_Reading", "EWoK_plus_Entity_sum", "BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "Reading"]
PRIMARY = ["cheap6_no_GlobalPIQA", "cheap5_no_GlobalPIQA_Reading", "EWoK_plus_Entity_sum"]


def finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def read_csv(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def fval(row: dict[str, Any], key: str) -> float | None:
    x = row.get(key)
    return float(x) if finite(x) else None


def series_stats(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    vals = [float(r[key]) for r in rows if finite(r.get(key))]
    if not vals:
        return {"n": 0, "mean": None, "min": None, "max": None, "last": None, "auc_per_checkpoint": None}
    words = [int(float(r.get("words", 0))) for r in rows if finite(r.get(key))]
    pairs = sorted(zip(words, vals))
    # Equal-checkpoint mean is intentional: the ladder is sampled uniformly every 10M.
    return {
        "n": len(vals),
        "mean": statistics.fmean(vals),
        "median": statistics.median(vals),
        "min": min(vals),
        "max": max(vals),
        "first": pairs[0][1],
        "last": pairs[-1][1],
        "positive_checkpoints": sum(1 for v in vals if v > 0),
        "negative_checkpoints": sum(1 for v in vals if v < 0),
        "auc_per_checkpoint": sum(vals) / len(vals),
    }


def slope(rows: list[dict[str, Any]], key: str) -> float | None:
    pts = [(float(r["words"])/1_000_000.0, float(r[key])) for r in rows if finite(r.get(key))]
    if len(pts) < 2:
        return None
    xs, ys = zip(*pts)
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    den = sum((x-mx)**2 for x in xs)
    return sum((x-mx)*(y-my) for x, y in pts) / den if den else None


def seed_spread_by_checkpoint(path: pathlib.Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    out: dict[str, dict[str, float]] = {}
    for r in read_csv(path):
        ck = str(r.get("checkpoint"))
        out[ck] = {}
        for m in METRICS:
            key = f"treatment_delta_seed43122_minus_seed43022_{m}"
            if finite(r.get(key)):
                out[ck][m] = abs(float(r[key]))
    return out


def contrast_stats(rows: list[dict[str, Any]], contrast: str) -> dict[str, Any]:
    rr = sorted([r for r in rows if r.get("contrast") == contrast], key=lambda r: int(float(r["words"])))
    return {m: {**series_stats(rr, m), "slope_per_10M": (slope(rr, m) * 10.0 if slope(rr, m) is not None else None)} for m in PRIMARY}


def checkpoint_ratios(amp_rows: list[dict[str, Any]], spread: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    out = []
    for r in amp_rows:
        if r.get("contrast") != "max_cmr_minus_1x_cmr":
            continue
        ck = str(r.get("checkpoint"))
        rec: dict[str, Any] = {"checkpoint": ck, "words": r.get("words")}
        for m in PRIMARY:
            a = fval(r, m)
            s = spread.get(ck, {}).get(m)
            rec[f"amplification_{m}"] = a
            rec[f"seed_spread_abs_{m}"] = s
            rec[f"abs_amplification_over_seed_spread_{m}"] = abs(a) / s if a is not None and s and s > 0 else None
        out.append(rec)
    return out


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dose-dir", default=str(DOSE_DIR_DEFAULT))
    ap.add_argument("--seed-spread", default=str(SEED_SPREAD_DEFAULT))
    ap.add_argument("--out-dir", default=str(OUT_DIR_DEFAULT))
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    dose_dir = pathlib.Path(args.dose_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "rows": dose_dir / "dose_ladder_stable_rows.csv",
        "treatment": dose_dir / "dose_ladder_treatment_vs_clean.csv",
        "cmr": dose_dir / "dose_ladder_compact_minus_repeat.csv",
        "amp": dose_dir / "dose_ladder_max_minus_1x.csv",
        "summary": dose_dir / "dose_ladder_stable_summary.json",
    }
    plan = {"status": "DOSE_LADDER_PROFILE_READOUT_PLAN", "inputs": {k: str(v) for k, v in paths.items()}, "exists": {k: v.exists() for k, v in paths.items()}, "out_dir": str(out_dir)}
    (out_dir / "dose_ladder_profile_readout_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if args.plan_only:
        return
    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"dose ladder files missing: {missing}")

    treatment = read_csv(paths["treatment"])
    cmr = read_csv(paths["cmr"])
    amp = read_csv(paths["amp"])
    spread = seed_spread_by_checkpoint(pathlib.Path(args.seed_spread))
    ratios = checkpoint_ratios(amp, spread)
    write_csv(out_dir / "max_cmr_minus_1x_seed_spread_ratios.csv", ratios)

    summary = {
        "status": "DOSE_LADDER_PROFILE_READOUT_DONE",
        "dose_ladder_summary": str(paths["summary"]),
        "profile_stats": {
            "treatment_vs_clean": {c: contrast_stats(treatment, c) for c in sorted({str(r.get("contrast")) for r in treatment})},
            "compact_minus_repeat": {c: contrast_stats(cmr, c) for c in sorted({str(r.get("contrast")) for r in cmr})},
            "max_minus_1x": {c: contrast_stats(amp, c) for c in sorted({str(r.get("contrast")) for r in amp})},
        },
        "seed_spread_ratio_rows": ratios,
        "files": {
            "seed_spread_ratios_csv": str(out_dir / "max_cmr_minus_1x_seed_spread_ratios.csv"),
        },
        "interpretation_targets": [
            "A positive and exposure-persistent max_cmr_minus_1x_cmr above same-coordinate seed-spread scale supports dose-scaling semantic re-expression.",
            "Positive view treatment-vs-clean growth with flat max_cmr_minus_1x_cmr supports redundancy reduction and reinvested source diversity rather than semantic re-expression as the operative term.",
            "Decline at MAX identifies a boundary where displacing independent clean-Qwen source experience outweighs the structured packet benefit.",
        ],
    }
    out_json = out_dir / "dose_ladder_profile_readout.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research dose ladder exposure-profile readout", ""]
    for family, stats in summary["profile_stats"].items():
        lines.append(f"## {family}")
        for contrast, d in stats.items():
            pieces = []
            for m in PRIMARY:
                s = d[m]
                pieces.append(f"{m}: mean {s['mean']}, first {s['first']}, last {s['last']}, slope/10M {s['slope_per_10M']}")
            lines.append(f"- {contrast}: " + "; ".join(pieces))
    lines += ["", f"JSON: `{out_json}`", f"Seed-spread ratios: `{out_dir / 'max_cmr_minus_1x_seed_spread_ratios.csv'}`"]
    (out_dir / "dose_ladder_profile_readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "out_json": str(out_json), "ratio_rows": len(ratios)}, indent=2, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
