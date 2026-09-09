#!/usr/bin/env python3
"""research: interpret future in-corpus admitted-block rate using measured spread.

This script is file-only. It reads:
  - research repeat/breadth/view resampled 60M->100M rate bands
  - research pre-score in-corpus rate commitment
  - research post-training in-corpus rate ladder, if present
  - research static-profile prediction
and writes a compact interpretation record. It can be run now (pending future
rate) and again after the post-training in-corpus rate ladder is available.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import itertools
import json
import math
import pathlib
import statistics
import time
from typing import Any


def find_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_root()
WS = ROOT / "experiments/archive" / 'frontier_consolidation'
SPREAD = WS / "data" / "register_removal_rate_and_reference_spread" / "register_removal_rate_and_reference_spread.json"
PRE = WS / "data" / "incorpus_rate_prediction" / "prediction_commitment.json"
STATIC = WS / "data" / "incorpus_static_profile_prediction" / "prediction_commitment.json"
FUTURE_RATE = WS / "data" / "incorpus_rate_ladder_after_training" / "incorpus_rate_prediction.json"
OUT = WS / "data" / "incorpus_rate_spread_interpretation"
LATE_EXENTITY5 = {"repeat": 0.0343, "breadth": 0.3350, "view": 0.3853}
ORDER = ["repeat", "breadth", "view"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: pathlib.Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def finite(x: Any) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except Exception:
        return False


def summarize(vals: list[float]) -> dict[str, Any]:
    vals = [float(v) for v in vals if finite(v)]
    if not vals:
        return {"n": 0, "mean": None, "sd": None, "min": None, "max": None, "values": []}
    return {
        "n": len(vals),
        "mean": round(statistics.mean(vals), 6),
        "sd": round(statistics.stdev(vals), 6) if len(vals) >= 2 else 0.0,
        "min": round(min(vals), 6),
        "max": round(max(vals), 6),
        "values": [round(v, 6) for v in vals],
    }


def linfit(xs: list[float], ys: list[float]) -> tuple[float, float] | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    xm, ym = statistics.mean(xs), statistics.mean(ys)
    den = sum((x - xm) ** 2 for x in xs)
    if den == 0:
        return None
    slope = sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / den
    return slope, ym - slope * xm


def extract_spread(spread_payload: dict[str, Any]) -> dict[str, Any]:
    ref = (((spread_payload.get("rate_summary") or {}).get("reference_rate_spread") or {}).get("resampled_rates_60_to_100") or {})
    vals: dict[str, list[float]] = {}
    for arm in ORDER:
        vals[arm] = [float(v) for v in ((ref.get(arm) or {}).get("values") or []) if finite(v)]
    mid_rb = (((spread_payload.get("rate_summary") or {}).get("reference_rate_spread") or {}).get("repeat_breadth_midpoint_spread") or {})
    mid_rv = (((spread_payload.get("rate_summary") or {}).get("reference_rate_spread") or {}).get("repeat_view_midpoint_spread") or {})
    return {"arm_values": vals, "repeat_breadth_midpoint": mid_rb, "repeat_view_midpoint": mid_rv}


def extract_future_rate(future_payload: dict[str, Any]) -> dict[str, Any]:
    owner = future_payload.get("incorpus_owner_loss_on_admitted") or {}
    rate_metrics = (((future_payload.get("prediction_commitment") or {}).get("incorpus_owner_rate_metrics")) or {})
    rate = rate_metrics.get("loss_reduction_60_to_100")
    if not finite(rate) and finite(owner.get("chck_60M")) and finite(owner.get("chck_100M")):
        rate = float(owner["chck_60M"]) - float(owner["chck_100M"])
    return {
        "owner_loss_on_admitted": owner,
        "rate_60_to_100": round(float(rate), 6) if finite(rate) else None,
        "rate_metrics": rate_metrics,
        "future_status": future_payload.get("status"),
    }


def classify_rate(rate: float | None, spread: dict[str, Any]) -> dict[str, Any]:
    arm_vals = spread["arm_values"]
    rep = arm_vals.get("repeat", [])
    breadth = arm_vals.get("breadth", [])
    view = arm_vals.get("view", [])
    out: dict[str, Any] = {
        "rate_60_to_100": rate,
        "reference_rates": {a: summarize(v) for a, v in arm_vals.items()},
        "transition_bands": {
            "repeat_breadth_midpoint": spread.get("repeat_breadth_midpoint"),
            "repeat_view_midpoint": spread.get("repeat_view_midpoint"),
            "repeat_range": {"min": min(rep) if rep else None, "max": max(rep) if rep else None},
            "breadth_range": {"min": min(breadth) if breadth else None, "max": max(breadth) if breadth else None},
            "view_range": {"min": min(view) if view else None, "max": max(view) if view else None},
        },
        "classification": "pending",
    }
    if rate is None:
        out["classification"] = "pending_future_incorpus_60M_100M_rate"
        out["interpretation"] = "The post-training in-corpus rate ladder remains pending; repeat the analysis when it is available."
        return out
    rb_mid = spread.get("repeat_breadth_midpoint") or {}
    rv_mid = spread.get("repeat_view_midpoint") or {}
    rb_min, rb_max = rb_mid.get("min"), rb_mid.get("max")
    rv_min, rv_max = rv_mid.get("min"), rv_mid.get("max")
    if finite(rb_min) and rate < float(rb_min):
        cls = "repeat_like_or_exhausted"
    elif finite(rb_max) and finite(rv_min) and float(rb_max) <= rate < float(rv_min):
        cls = "between_breadth_and_view_transition"
    elif finite(rv_min) and rate >= float(rv_min):
        cls = "view_like_active_late_error"
    else:
        cls = "overlap_transition_region"
    out["classification"] = cls

    # Reference-fit spread: all combinations of one replicate per reference arm.
    preds = []
    combos = []
    for r, b, v in itertools.product(rep, breadth, view):
        fit = linfit([r, b, v], [LATE_EXENTITY5["repeat"], LATE_EXENTITY5["breadth"], LATE_EXENTITY5["view"]])
        if fit is None:
            continue
        slope, intercept = fit
        y = intercept + slope * float(rate)
        preds.append(y)
        combos.append({"repeat": r, "breadth": b, "view": v, "slope": slope, "intercept": intercept, "pred": y})
    out["rate_to_late_exEntity5_fit_spread"] = summarize(preds)
    out["fit_warning"] = "Fits use only three historical arm-level points and sampled-rate variability; they are a quantitative lens, not a causal proof."
    if cls == "repeat_like_or_exhausted":
        interp = "The in-corpus adult-prose block has late loss reduction below the resampled repeat/breadth midpoint band; under the rate account, broad late official gain should be small even if static profile proximity is positive."
    elif cls == "view_like_active_late_error":
        interp = "The in-corpus adult-prose block remains within or above the repeat/view transition band; under the rate account, persistent broad official gain is expected."
    else:
        interp = "The in-corpus adult-prose block falls in the measured transition region; official scores should be interpreted as mixed evidence and not as a sharp threshold result."
    out["interpretation"] = interp
    return out


def write_md(payload: dict[str, Any], path: pathlib.Path) -> None:
    lines: list[str] = []
    lines.append("# research in-corpus rate spread interpretation")
    lines.append("")
    lines.append("This file-only readout places the future in-corpus admitted-block late rate against the research resampled repeat/breadth/view rate scale. It intentionally does not rerun model inference.")
    lines.append("")
    lines.append("## Status")
    lines.append(f"- In-corpus post-training rate available: {payload['future_rate_available']}")
    lines.append(f"- Classification: {payload['rate_classification'].get('classification')}")
    lines.append(f"- Interpretation: {payload['rate_classification'].get('interpretation')}")
    lines.append("")
    lines.append("## In-corpus rate")
    fr = payload.get("future_rate", {})
    lines.append(f"- Owner admitted-block loss by checkpoint: {fr.get('owner_loss_on_admitted')}")
    lines.append(f"- 60M→100M loss reduction: {fr.get('rate_60_to_100')}")
    lines.append("")
    lines.append("## Reference spread")
    cls = payload.get("rate_classification", {})
    lines.append(json.dumps(cls.get("reference_rates"), indent=2, ensure_ascii=False))
    lines.append("")
    lines.append(f"- Transition bands: {cls.get('transition_bands')}")
    lines.append(f"- Rate→late exEntity5 fit spread: {cls.get('rate_to_late_exEntity5_fit_spread')}")
    lines.append("")
    lines.append("## Static profile comparator")
    lines.append(json.dumps(payload.get("static_profile_prediction"), indent=2, ensure_ascii=False))
    lines.append("")
    lines.append("## Files")
    for k, v in payload.get("files", {}).items():
        lines.append(f"- {k}: `{v}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(OUT))
    args = ap.parse_args()
    out_dir = pathlib.Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    missing = [rel(p) for p in [SPREAD, PRE, STATIC] if not p.exists()]
    if missing:
        raise SystemExit("Missing required fixed inputs: " + "; ".join(missing))
    spread_payload = read_json(SPREAD)
    spread = extract_spread(spread_payload)
    pre = read_json(PRE)
    static = read_json(STATIC)
    future_available = FUTURE_RATE.exists()
    future = extract_future_rate(read_json(FUTURE_RATE)) if future_available else {"owner_loss_on_admitted": {}, "rate_60_to_100": None, "future_status": "pending"}
    classification = classify_rate(future.get("rate_60_to_100"), spread)

    files = {
        "summary_json": rel(out_dir / "incorpus_rate_spread_interpretation.json"),
        "summary_md": rel(out_dir / "incorpus_rate_spread_interpretation.md"),
        "spread_source": rel(SPREAD),
        "future_rate_source": rel(FUTURE_RATE),
    }
    payload = {
        "status": "INCORPUS_RATE_SPREAD_INTERPRETATION",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "future_rate_available": future_available,
        "future_rate": future,
        "rate_classification": classification,
        "pre_score_incorpus_rate_commitment": pre,
        "static_profile_prediction": {
            "primary_profile_predictions_mass_scaled": static.get("primary_profile_predictions_mass_scaled"),
            "word_js_control_predictions_mass_scaled": static.get("word_js_control_predictions_mass_scaled"),
            "interpretation_rule": static.get("interpretation_rule"),
        },
        "files": files,
        "no_model_loading_training_official_eval_gpu_upload_or_leaderboard": True,
    }
    write_json(out_dir / "incorpus_rate_spread_interpretation.json", payload)
    write_md(payload, out_dir / "incorpus_rate_spread_interpretation.md")
    print(json.dumps({
        "status": payload["status"],
        "future_rate_available": future_available,
        "classification": classification.get("classification"),
        "rate_60_to_100": future.get("rate_60_to_100"),
        "summary_md": files["summary_md"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
