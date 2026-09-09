#!/usr/bin/env python3
"""research: compare prior single-seed BabyLM readings to the measured full-DeBERTa seed spread.

The current research question is whether compact semantic second views produce a
reproducible data-efficient learning effect or mainly a small, basin-specific
competence reallocation. The full-DeBERTa seed ladder will measure the scale of
same-architecture, same-data, different-seed movement. This script consumes that
future ladder and expresses earlier single-seed readings on the same scale.

It performs no training, no model loading, no evaluation, no SuperGLUE, no AoA,
no upload, and no leaderboard action.
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


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
WS = USER_ROOT / "experiments/archive/frontier_consolidation"
DEFAULT_LADDER_SUMMARY = WS / "data/full_deberta_seed_ladder_stable_eval/full_deberta_seed_ladder_stable_summary.json"
DEFAULT_OUT = WS / "data/prior_reading_reprice_against_seed_spread"

# Curated prior readings. Values are percentage-point score deltas on the same
# cheap-family scale whenever available. Positive/negative signs are preserved;
# pricing uses both signed and absolute scale.
PRIOR_READINGS: list[dict[str, Any]] = [
    {
        "name": "GPT2 causal compact-minus-repeat mean over six endpoints",
        "source": "data/causal_compact_repeat_selected_contrast/causal_compact_repeat_selected_contrast.json",
        "kind": "cross-architecture transfer",
        "values": {"cheap7": 0.077738, "cheap6_no_GlobalPIQA": -0.144583, "cheap5_no_GlobalPIQA_Reading": -0.151333, "EWoK_plus_Entity_sum": -0.319167},
        "reading_before_seed_spread": "Broad-transfer evidence was negative after removing volatile GlobalPIQA/Reading movement, but magnitudes were a few tenths.",
    },
    {
        "name": "RoBERTa compact-minus-repeat 100M",
        "source": "notes/roberta_selected_transfer_resolution.md",
        "kind": "cross-architecture transfer",
        "values": {"cheap7": 0.042143, "cheap6_no_GlobalPIQA": -0.115000, "cheap5_no_GlobalPIQA_Reading": -0.194000, "EWoK_plus_Entity_sum": 0.640000, "Supplement": -2.210000, "Entity": -0.410000, "COMPS": 0.110000},
        "reading_before_seed_spread": "Tiny cheap7 positive with stable-family damage; local source-absent channel did not become broad selected competence.",
    },
    {
        "name": "RoBERTa compact-minus-repeat late 60M/70M stable mean",
        "source": "notes/roberta_selected_transfer_resolution.md",
        "kind": "cross-architecture transfer",
        "values": {"cheap7": -0.5125, "cheap6_no_GlobalPIQA": -0.2659, "cheap5_no_GlobalPIQA_Reading": -0.3040, "EWoK_plus_Entity_sum": -0.1450},
        "reading_before_seed_spread": "Separated late checkpoints pointed negative or mixed, not a mature stable transfer signal.",
    },
    {
        "name": "Extractive balanced minus natural compact mean 80M/100M",
        "source": "notes/extractive_selected_readout_result.md",
        "kind": "data-factor contrast",
        "values": {"cheap7": 0.246, "cheap6_no_GlobalPIQA": -0.257, "cheap5_no_GlobalPIQA_Reading": -0.287, "EWoK_plus_Entity_sum": -1.580, "Supplement": 0.620, "Entity": -0.760, "COMPS": -0.690},
        "reading_before_seed_spread": "Compact-like source-word density without fluent semantic re-expression failed stable families; cheap7 was GlobalPIQA-carried.",
    },
    {
        "name": "Extractive wide minus natural compact mean 80M/100M",
        "source": "notes/extractive_selected_readout_result.md",
        "kind": "data-factor contrast",
        "values": {"cheap7": -0.098, "cheap6_no_GlobalPIQA": -0.662, "cheap5_no_GlobalPIQA_Reading": -0.746, "EWoK_plus_Entity_sum": -2.820, "Supplement": -2.065, "Entity": -2.355, "COMPS": -0.035},
        "reading_before_seed_spread": "Maximum source span did worse on stable selected families and relation/state tasks.",
    },
    {
        "name": "Plain no-disentangle minus full interaction mean 80M/100M before common-copy",
        "source": "data/architecture_interaction_selected_panel_final/architecture_interaction_selected_panel_rows.csv",
        "kind": "architecture-coordinate confounded interaction",
        "values": {"cheap6_no_GlobalPIQA": -0.6908, "cheap5_no_GlobalPIQA_Reading": -0.6080, "EWoK_plus_Entity_sum": -3.1200},
        "reading_before_seed_spread": "Large collapse later traced mainly to initialization/optimization-basin artifact by common-copy retraining.",
    },
    {
        "name": "Common-copy no-disentangle minus full interaction mean 80M/100M",
        "source": "notes/commoncopy_architecture_interaction_result.md",
        "kind": "architecture-coordinate causal interaction",
        "values": {"cheap6_no_GlobalPIQA": 0.0679, "cheap5_no_GlobalPIQA_Reading": 0.1550, "EWoK_plus_Entity_sum": -0.3850},
        "reading_before_seed_spread": "Compact benefit survived removal of DeBERTa positional-score terms when common initialization was controlled.",
    },
    {
        "name": "Scale1.25 residual-adapter trajectory minus scale1.75 reference mean",
        "source": "notes/lead_cross_seed_decision_framework.md",
        "kind": "residual-capacity route",
        "values": {"cheap7": -0.410982, "cheap6_no_GlobalPIQA": -0.815208, "cheap5_no_GlobalPIQA_Reading": -0.949125, "EWoK_plus_Entity_sum": -0.425313},
        "reading_before_seed_spread": "Lower adapter energy did not reproduce the late high phase and was broadly weaker.",
    },
    {
        "name": "Reference ordinary late chck84-over-chck82 selected cheap7 shift",
        "source": "Research memory Steps 185-195",
        "kind": "late-training allocation shift",
        "values": {"cheap7": 0.164122, "cheap6_no_GlobalPIQA": None, "cheap5_no_GlobalPIQA_Reading": None},
        "reading_before_seed_spread": "Chck84 was the clean ordinary peak but with broad item churn and a subtask-resampling interval crossing zero.",
    },
]

KEY_ALIASES = {
    "cheap7": ["cheap7"],
    "cheap6_no_GlobalPIQA": ["cheap6_no_GlobalPIQA"],
    "cheap5_no_GlobalPIQA_Reading": ["cheap5_no_GlobalPIQA_Reading"],
    "EWoK_plus_Entity_sum": ["EWoK_plus_Entity_sum"],
    "Supplement": ["Supplement"],
    "Entity": ["Entity"],
    "COMPS": ["COMPS"],
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def resolve(path: str | pathlib.Path) -> pathlib.Path:
    p = pathlib.Path(path)
    return p if p.is_absolute() else USER_ROOT / p


def finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def quantile(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    pos = p * (len(ys) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ys[lo]
    return ys[lo] * (hi - pos) + ys[hi] * (pos - lo)


def series_stats(xs: list[float]) -> dict[str, Any]:
    vals = [float(x) for x in xs if finite(x)]
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "mean": statistics.mean(vals),
        "median": statistics.median(vals),
        "mean_abs": statistics.mean(abs(v) for v in vals),
        "median_abs": statistics.median(abs(v) for v in vals),
        "p75_abs": quantile([abs(v) for v in vals], 0.75),
        "p90_abs": quantile([abs(v) for v in vals], 0.90),
        "max_abs": max(abs(v) for v in vals),
        "min": min(vals),
        "max": max(vals),
        "frac_positive": sum(v > 0 for v in vals) / len(vals),
    }


def wait_for_file(path: pathlib.Path, timeout_sec: int, poll_sec: int, log_path: pathlib.Path) -> dict[str, Any]:
    start = time.time()
    while True:
        exists = path.exists()
        rec = {"event": "file_wait_sample", "utc": now(), "path": str(path), "exists": exists}
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(json.dumps(rec), flush=True)
        if exists:
            return {"status": "file_ready", "waited_sec": round(time.time() - start, 1), "path": str(path)}
        if time.time() - start > timeout_sec:
            return {"status": "file_wait_timeout", "waited_sec": round(time.time() - start, 1), "path": str(path)}
        time.sleep(poll_sec)


def load_ladder(ladder_summary: pathlib.Path) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]]]:
    summary = json.loads(ladder_summary.read_text(encoding="utf-8"))
    spread_csv = resolve(summary["spread_csv"])
    delta_csv = resolve(summary["delta_csv"])
    return summary, read_csv(spread_csv), read_csv(delta_csv)


def seed_spread_stats(spread_rows: list[dict[str, str]]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for canonical, aliases in KEY_ALIASES.items():
        vals: list[float] = []
        for row in spread_rows:
            for alias in aliases:
                field = f"treatment_delta_seed43122_minus_seed43022_{alias}"
                if field in row and finite(row[field]):
                    vals.append(float(row[field]))
                    break
        if vals:
            stats[canonical] = series_stats(vals)
    return stats


def treatment_series_stats(delta_rows: list[dict[str, str]]) -> dict[str, Any]:
    by_seed: dict[str, dict[str, Any]] = {}
    for seed in sorted({row.get("seed", "") for row in delta_rows}):
        rs = [row for row in delta_rows if row.get("seed") == seed]
        by_seed[seed] = {}
        for canonical in KEY_ALIASES:
            vals = [float(r[canonical]) for r in rs if canonical in r and finite(r[canonical])]
            if vals:
                by_seed[seed][canonical] = series_stats(vals)
    return by_seed


def price_value(value: Any, spread_stats: dict[str, Any]) -> dict[str, Any] | None:
    if not finite(value) or not spread_stats or spread_stats.get("n", 0) == 0:
        return None
    v = float(value)
    med = spread_stats.get("median_abs")
    p75 = spread_stats.get("p75_abs")
    p90 = spread_stats.get("p90_abs")
    mean_abs = spread_stats.get("mean_abs")
    return {
        "value": v,
        "abs_value": abs(v),
        "signed_over_mean_abs_seed_spread": v / mean_abs if finite(mean_abs) and float(mean_abs) != 0 else None,
        "abs_over_median_abs_seed_spread": abs(v) / float(med) if finite(med) and float(med) != 0 else None,
        "abs_over_p75_abs_seed_spread": abs(v) / float(p75) if finite(p75) and float(p75) != 0 else None,
        "abs_over_p90_abs_seed_spread": abs(v) / float(p90) if finite(p90) and float(p90) != 0 else None,
        "same_scale_as_measured_seed_spread": (abs(v) <= float(p90)) if finite(p90) else None,
    }


def build_pricing(spread_stats: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rec in PRIOR_READINGS:
        priced: dict[str, Any] = {}
        for key, val in rec["values"].items():
            if key in spread_stats:
                pv = price_value(val, spread_stats[key])
                if pv is not None:
                    priced[key] = pv
        out.append({**rec, "priced_against_seed_spread": priced})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder-summary", default=str(DEFAULT_LADDER_SUMMARY))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--wait-timeout-sec", type=int, default=57_600)
    ap.add_argument("--poll-sec", type=int, default=180)
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "reprice_wait_log.jsonl"
    ladder_summary = resolve(args.ladder_summary)

    plan = {
        "status": "REPRICE_PRIOR_SINGLE_SEED_READINGS_PLAN",
        "created_utc": now(),
        "decision_role": "After the full-DeBERTa seed ladder lands, express earlier single-seed transfer/data-factor/late-peak readings in units of measured same-architecture seed spread.",
        "ladder_summary": str(ladder_summary),
        "ladder_summary_exists": ladder_summary.exists(),
        "prior_reading_count": len(PRIOR_READINGS),
        "out_dir": str(out_dir),
        "no_training_model_loading_evaluation_superglue_aoa_upload_or_leaderboard": True,
    }
    (out_dir / "reprice_prior_readings_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if args.plan_only:
        return

    if args.wait:
        wr = wait_for_file(ladder_summary, args.wait_timeout_sec, args.poll_sec, log_path)
        (out_dir / "wait_record.json").write_text(json.dumps(wr, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if wr.get("status") != "file_ready":
            raise SystemExit(2)
    if not ladder_summary.exists():
        raise FileNotFoundError(ladder_summary)

    ladder, spread_rows, delta_rows = load_ladder(ladder_summary)
    spreads = seed_spread_stats(spread_rows)
    treatment_stats = treatment_series_stats(delta_rows)
    priced = build_pricing(spreads)
    result = {
        "status": "PRIOR_SINGLE_SEED_READINGS_REPRICED",
        "created_utc": now(),
        "ladder_summary": str(ladder_summary),
        "ladder_rows_csv": ladder.get("rows_csv"),
        "ladder_delta_csv": ladder.get("delta_csv"),
        "ladder_spread_csv": ladder.get("spread_csv"),
        "same_architecture_treatment_delta_seed_spread": spreads,
        "treatment_delta_series_by_seed": treatment_stats,
        "prior_readings_priced": priced,
        "interpretation_boundary": "This compares magnitudes; it does not retroactively erase mechanically matched negative results. It tells which prior few-tenth readings require seed-spread-aware wording and which remain large relative to the measured full-DeBERTa treatment-spread scale.",
        "no_training_model_loading_evaluation_superglue_aoa_upload_or_leaderboard": True,
    }
    out_json = out_dir / "prior_single_seed_readings_repriced.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["# research prior readings priced against full-DeBERTa seed spread", "", result["interpretation_boundary"], "", "## Measured treatment-delta seed spread"]
    for k, v in spreads.items():
        lines.append(f"- {k}: median_abs={v.get('median_abs')}, p75_abs={v.get('p75_abs')}, p90_abs={v.get('p90_abs')}, max_abs={v.get('max_abs')}, n={v.get('n')}")
    lines += ["", "## Prior readings"]
    for rec in priced:
        lines.append(f"- {rec['name']} ({rec['kind']}): source `{rec['source']}`")
        for key, pv in rec.get("priced_against_seed_spread", {}).items():
            lines.append(f"  - {key}: value={pv['value']:+.4f}, abs/median_spread={pv.get('abs_over_median_abs_seed_spread')}, abs/p90_spread={pv.get('abs_over_p90_abs_seed_spread')}, same_scale={pv.get('same_scale_as_measured_seed_spread')}")
    lines += ["", f"JSON: `{out_json}`"]
    out_md = out_dir / "prior_single_seed_readings_repriced.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "out_json": str(out_json), "out_md": str(out_md)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
