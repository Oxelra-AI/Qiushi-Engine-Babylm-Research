#!/usr/bin/env python3
"""research: AoA common fitted-word subset refit.

Join the corrected research control and schedule fitted-word tables and recompute raw
Pearson model-child AoA correlations on the identical word subset.  This answers
whether the schedule arm's positive 30M AoA relation persists when the comparison
uses the same fitted words as the control, rather than different word coverage.
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
import time
from typing import Any

import numpy as np
from scipy.stats import pearsonr

ROOT = _public_path('.')
IN_DIR = _public_path('experiments/archive/relation_learning/data/aoa_calibration_extract_fit')
OUT_DIR = _public_path('experiments/archive/relation_learning/data/aoa_common_subset_refit')


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def read_fits(path: pathlib.Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            word = str(row.get("word", "")).strip().lower()
            if not word:
                continue
            rec: dict[str, Any] = dict(row)
            for key in ["child_aoa", "model_aoa_log10_step", "n_subword_tokens", "n_points", "first_step", "last_step"]:
                if key in rec and rec[key] not in (None, ""):
                    try:
                        if key in {"n_subword_tokens", "n_points"}:
                            rec[key] = int(float(rec[key]))
                        else:
                            rec[key] = float(rec[key])
                    except Exception:
                        pass
            if math.isfinite(float(rec.get("child_aoa", float("nan")))) and math.isfinite(float(rec.get("model_aoa_log10_step", float("nan")))):
                out[word] = rec
    return out


def corr(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    xs = np.array([float(r["child_aoa"]) for r in rows], dtype=float)
    ys = np.array([float(r[key]) for r in rows], dtype=float)
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[mask]
    ys = ys[mask]
    if len(xs) < 3 or float(np.std(xs)) == 0.0 or float(np.std(ys)) == 0.0:
        return {"pearson_r": None, "pearson_p": None, "n": int(len(xs))}
    r, p = pearsonr(xs, ys)
    return {"pearson_r": float(r), "pearson_p": float(p), "n": int(len(xs))}


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    control = read_fits(_public_path('experiments/archive/relation_learning/data/aoa_calibration_extract_fit/control_word_fits.csv'))
    schedule = read_fits(_public_path('experiments/archive/relation_learning/data/aoa_calibration_extract_fit/schedule_word_fits.csv'))
    enrichment = read_fits(_public_path('experiments/archive/relation_learning/data/aoa_calibration_extract_fit/enrichment_word_fits.csv')) if (_public_path('experiments/archive/relation_learning/data/aoa_calibration_extract_fit/enrichment_word_fits.csv')).is_file() else {}
    common = sorted(set(control) & set(schedule))
    rows: list[dict[str, Any]] = []
    for w in common:
        c = control[w]
        s = schedule[w]
        row = {
            "word": w,
            "child_aoa": float(c["child_aoa"]),
            "control_model_aoa_log10_step": float(c["model_aoa_log10_step"]),
            "schedule_model_aoa_log10_step": float(s["model_aoa_log10_step"]),
            "schedule_minus_control_model_aoa": float(s["model_aoa_log10_step"]) - float(c["model_aoa_log10_step"]),
            "control_n_points": int(c.get("n_points", 0)),
            "schedule_n_points": int(s.get("n_points", 0)),
        }
        if w in enrichment:
            row["enrichment_model_aoa_log10_step"] = float(enrichment[w]["model_aoa_log10_step"])
            row["enrichment_minus_control_model_aoa"] = float(enrichment[w]["model_aoa_log10_step"]) - float(c["model_aoa_log10_step"])
        rows.append(row)
    out_csv = _public_path('experiments/archive/relation_learning/data/aoa_common_subset_refit/common_control_schedule_word_fits.csv')
    write_csv(out_csv, rows)

    control_rows = [{"child_aoa": r["child_aoa"], "model": r["control_model_aoa_log10_step"]} for r in rows]
    schedule_rows = [{"child_aoa": r["child_aoa"], "model": r["schedule_model_aoa_log10_step"]} for r in rows]
    # corr() expects a configurable model key.
    control_corr = corr([{"child_aoa": r["child_aoa"], "model": r["control_model_aoa_log10_step"]} for r in rows], "model")
    schedule_corr = corr([{"child_aoa": r["child_aoa"], "model": r["schedule_model_aoa_log10_step"]} for r in rows], "model")
    delta_corr = corr([{"child_aoa": r["child_aoa"], "delta": r["schedule_minus_control_model_aoa"]} for r in rows], "delta")
    enrichment_common_corr = None
    enrichment_delta_corr = None
    if any("enrichment_model_aoa_log10_step" in r for r in rows):
        erows = [r for r in rows if "enrichment_model_aoa_log10_step" in r]
        enrichment_common_corr = corr([{"child_aoa": r["child_aoa"], "model": r["enrichment_model_aoa_log10_step"]} for r in erows], "model")
        enrichment_delta_corr = corr([{"child_aoa": r["child_aoa"], "delta": r["enrichment_minus_control_model_aoa"]} for r in erows], "delta")

    summary = {
        "status": "AOA_COMMON_SUBSET_REFIT_DONE",
        "created_utc": now(),
        "purpose": "raw model-child AoA magnitude for schedule and control on the identical fitted-word subset, preventing fitted-word coverage from carrying the comparison",
        "input_dir": rel(IN_DIR),
        "n_control_all": len(control),
        "n_schedule_all": len(schedule),
        "n_common_control_schedule": len(common),
        "common_subset_raw": {
            "control": control_corr,
            "schedule": schedule_corr,
            "schedule_minus_control_vs_child_aoa": delta_corr,
            "enrichment_on_control_schedule_common": enrichment_common_corr,
            "enrichment_minus_control_vs_child_aoa_on_common": enrichment_delta_corr,
        },
        "interpretation": "If schedule remains positive on the common subset, the 30M AoA signal is not only a fitted-word selection artifact; cheap7 competence cost still determines whether to ask for a 100M trunk.",
        "outputs": {"common_csv": rel(out_csv), "summary_json": rel(_public_path('experiments/archive/relation_learning/data/aoa_common_subset_refit/summary.json')), "summary_md": rel(_public_path('research/documents/relation_learning/data/aoa_common_subset_refit/summary.md'))},
    }
    (_public_path('experiments/archive/relation_learning/data/aoa_common_subset_refit/summary.json')).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research AoA common fitted-word subset refit",
        "",
        summary["purpose"],
        "",
        f"Common control/schedule words: {len(common)} (control all {len(control)}, schedule all {len(schedule)}).",
        "",
        "| arm/statistic | r | p | n |",
        "|---|---:|---:|---:|",
    ]
    def fmt(x: Any) -> str:
        if x is None:
            return "NA"
        if isinstance(x, float):
            return f"{x:.6g}"
        return str(x)
    for label, c in [("control raw", control_corr), ("schedule raw", schedule_corr), ("schedule-control delta vs child AoA", delta_corr)]:
        lines.append(f"| {label} | {fmt(c.get('pearson_r'))} | {fmt(c.get('pearson_p'))} | {c.get('n')} |")
    if enrichment_common_corr is not None:
        lines.append(f"| enrichment raw on same common words | {fmt(enrichment_common_corr.get('pearson_r'))} | {fmt(enrichment_common_corr.get('pearson_p'))} | {enrichment_common_corr.get('n')} |")
        lines.append(f"| enrichment-control delta vs child AoA | {fmt(enrichment_delta_corr.get('pearson_r'))} | {fmt(enrichment_delta_corr.get('pearson_p'))} | {enrichment_delta_corr.get('n')} |")
    lines += ["", f"Full JSON: `{rel(_public_path('experiments/archive/relation_learning/data/aoa_common_subset_refit/summary.json'))}`"]
    (_public_path('research/documents/relation_learning/data/aoa_common_subset_refit/summary.md')).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "common": len(common), "control": control_corr, "schedule": schedule_corr, "delta": delta_corr, "summary_json": summary["outputs"]["summary_json"]}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
