#!/usr/bin/env python3
"""research: budget-decomposition readout for the fixed-budget compact-dose ladder.

Scientific purpose
------------------
The dose experiment should be interpreted as a decomposition of one fixed-budget
substitution, not as two unrelated contrasts. For each dose and checkpoint where
a clean reference exists,

    view - clean = (view - repeat) + (repeat - clean)

where:
  * view-repeat is the value of semantic re-expression / compact rewriting on
    the same admitted source packet budget;
  * repeat-clean is the value of admitting the MAX/1x FineWeb source material at
    all, with source repetition replacing clean-Qwen filler;
  * view-clean is the total compact-view treatment relative to the clean-Qwen
    fixed-budget reference.

This script reads the eventual research stable-family ladder outputs and writes
checkpointwise legs, MAX-minus-1x leg growth, exposure-profile summaries, and
ratios to the research same-coordinate seed-spread scale. It deliberately uses
stable selected families only and performs no training or evaluation.
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
DOSE_DIR_DEFAULT = WS / "data/dose_ladder_stable_eval"
OUT_DIR_DEFAULT = WS / "data/dose_budget_decomposition_readout"
SEED_SPREAD_DEFAULT = WS / "data/full_deberta_seed_ladder_stable_eval/full_deberta_seed_ladder_seed_spread.csv"

STABLE_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "Reading"]
DERIVED_COLUMNS = ["cheap6_no_GlobalPIQA", "cheap5_no_GlobalPIQA_Reading", "EWoK_plus_Entity_sum"]
METRICS = [*STABLE_COLUMNS, *DERIVED_COLUMNS]
PRIMARY = ["cheap6_no_GlobalPIQA", "cheap5_no_GlobalPIQA_Reading", "EWoK_plus_Entity_sum"]
CKS_10_80 = [f"chck_{i}M" for i in range(10, 81, 10)]
CKS_10_100 = [f"chck_{i}M" for i in range(10, 101, 10)]
DOSE_PAIRS = {
    "dose1": {"dose": 1.0, "view": "dose1_view", "repeat": "dose1_repeat"},
    "max": {"dose": 2.641480921798262, "view": "max_view", "repeat": "max_repeat"},
}
LEGS = ["total_view_minus_clean", "semantic_view_minus_repeat", "source_repeat_minus_clean"]
GROWTH_LEGS = ["total_growth", "semantic_growth", "source_growth"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def fval(row: dict[str, Any] | None, key: str) -> float | None:
    if row is None:
        return None
    x = row.get(key)
    return float(x) if finite(x) else None


def ck_words(ck: str) -> int:
    return int(str(ck)[5:-1]) * 1_000_000


def read_csv(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
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


def seed_spread_by_checkpoint(path: pathlib.Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    if not path.exists():
        return out
    for r in read_csv(path):
        ck = str(r.get("checkpoint"))
        out[ck] = {}
        for m in METRICS:
            key = f"treatment_delta_seed43122_minus_seed43022_{m}"
            if finite(r.get(key)):
                out[ck][m] = abs(float(r[key]))
    return out


def slope_per_10m(rows: list[dict[str, Any]], value_key: str) -> float | None:
    pts = [(float(r["words"]) / 1_000_000.0, float(r[value_key])) for r in rows if finite(r.get(value_key))]
    if len(pts) < 2:
        return None
    xs, ys = zip(*pts)
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return None
    return 10.0 * sum((x - mx) * (y - my) for x, y in pts) / den


def series_stats(vals_rows: list[dict[str, Any]], key: str = "value") -> dict[str, Any]:
    vals = [float(r[key]) for r in vals_rows if finite(r.get(key))]
    if not vals:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None, "first": None, "last": None, "positive_checkpoints": 0, "negative_checkpoints": 0, "slope_per_10M": None}
    ordered = sorted([r for r in vals_rows if finite(r.get(key))], key=lambda r: int(float(r["words"])))
    return {
        "n": len(vals),
        "mean": statistics.fmean(vals),
        "median": statistics.median(vals),
        "min": min(vals),
        "max": max(vals),
        "first": float(ordered[0][key]),
        "last": float(ordered[-1][key]),
        "positive_checkpoints": sum(1 for v in vals if v > 0),
        "negative_checkpoints": sum(1 for v in vals if v < 0),
        "slope_per_10M": slope_per_10m(ordered, key),
    }


def build_checkpoint_decomposition(rows: list[dict[str, Any]], spread: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    by = {(str(r.get("arm")), str(r.get("checkpoint"))): r for r in rows}
    out: list[dict[str, Any]] = []
    for dose_name, spec in DOSE_PAIRS.items():
        for ck in CKS_10_80:
            clean = by.get(("clean0", ck))
            view = by.get((spec["view"], ck))
            repeat = by.get((spec["repeat"], ck))
            if clean is None or view is None or repeat is None:
                continue
            for m in METRICS:
                v, r, c = fval(view, m), fval(repeat, m), fval(clean, m)
                if v is None or r is None or c is None:
                    total = semantic = source = residual = None
                else:
                    total = v - c
                    semantic = v - r
                    source = r - c
                    residual = total - (semantic + source)
                ss = spread.get(ck, {}).get(m)
                rec = {
                    "dose_name": dose_name,
                    "dose": spec["dose"],
                    "checkpoint": ck,
                    "words": ck_words(ck),
                    "metric": m,
                    "clean_score": c,
                    "view_score": v,
                    "repeat_score": r,
                    "total_view_minus_clean": total,
                    "semantic_view_minus_repeat": semantic,
                    "source_repeat_minus_clean": source,
                    "decomposition_residual": residual,
                    "seed_spread_abs_same_ck": ss,
                    "abs_total_over_seed_spread": abs(total) / ss if total is not None and ss and ss > 0 else None,
                    "abs_semantic_over_seed_spread": abs(semantic) / ss if semantic is not None and ss and ss > 0 else None,
                    "abs_source_over_seed_spread": abs(source) / ss if source is not None and ss and ss > 0 else None,
                    "semantic_fraction_of_total": semantic / total if finite(total) and abs(float(total)) > 1e-12 and semantic is not None else None,
                    "source_fraction_of_total": source / total if finite(total) and abs(float(total)) > 1e-12 and source is not None else None,
                }
                out.append(rec)
    return out


def build_cmr_10_100(rows: list[dict[str, Any]], spread: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    by = {(str(r.get("arm")), str(r.get("checkpoint"))): r for r in rows}
    out: list[dict[str, Any]] = []
    for dose_name, spec in DOSE_PAIRS.items():
        for ck in CKS_10_100:
            view = by.get((spec["view"], ck))
            repeat = by.get((spec["repeat"], ck))
            if view is None or repeat is None:
                continue
            for m in METRICS:
                v, r = fval(view, m), fval(repeat, m)
                semantic = v - r if v is not None and r is not None else None
                ss = spread.get(ck, {}).get(m)
                out.append({
                    "dose_name": dose_name,
                    "dose": spec["dose"],
                    "checkpoint": ck,
                    "words": ck_words(ck),
                    "metric": m,
                    "view_minus_repeat": semantic,
                    "seed_spread_abs_same_ck": ss,
                    "abs_semantic_over_seed_spread": abs(semantic) / ss if semantic is not None and ss and ss > 0 else None,
                })
    return out


def build_growth_rows(decomp: list[dict[str, Any]], spread: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    by = {(str(r["dose_name"]), str(r["checkpoint"]), str(r["metric"])): r for r in decomp}
    out: list[dict[str, Any]] = []
    for ck in CKS_10_80:
        for m in METRICS:
            one = by.get(("dose1", ck, m))
            mx = by.get(("max", ck, m))
            if one is None or mx is None:
                continue
            total = fval(mx, "total_view_minus_clean") - fval(one, "total_view_minus_clean") if fval(mx, "total_view_minus_clean") is not None and fval(one, "total_view_minus_clean") is not None else None
            semantic = fval(mx, "semantic_view_minus_repeat") - fval(one, "semantic_view_minus_repeat") if fval(mx, "semantic_view_minus_repeat") is not None and fval(one, "semantic_view_minus_repeat") is not None else None
            source = fval(mx, "source_repeat_minus_clean") - fval(one, "source_repeat_minus_clean") if fval(mx, "source_repeat_minus_clean") is not None and fval(one, "source_repeat_minus_clean") is not None else None
            residual = total - (semantic + source) if total is not None and semantic is not None and source is not None else None
            ss = spread.get(ck, {}).get(m)
            out.append({
                "checkpoint": ck,
                "words": ck_words(ck),
                "metric": m,
                "total_growth": total,
                "semantic_growth": semantic,
                "source_growth": source,
                "growth_decomposition_residual": residual,
                "seed_spread_abs_same_ck": ss,
                "abs_total_growth_over_seed_spread": abs(total) / ss if total is not None and ss and ss > 0 else None,
                "abs_semantic_growth_over_seed_spread": abs(semantic) / ss if semantic is not None and ss and ss > 0 else None,
                "abs_source_growth_over_seed_spread": abs(source) / ss if source is not None and ss and ss > 0 else None,
                "semantic_growth_fraction_of_total_growth": semantic / total if finite(total) and abs(float(total)) > 1e-12 and semantic is not None else None,
                "source_growth_fraction_of_total_growth": source / total if finite(total) and abs(float(total)) > 1e-12 and source is not None else None,
            })
    return out


def summarize_decomposition(decomp: list[dict[str, Any]], cmr: list[dict[str, Any]], growth: list[dict[str, Any]]) -> dict[str, Any]:
    by_dose_leg: dict[str, Any] = {}
    for dose in sorted({str(r["dose_name"]) for r in decomp}):
        by_dose_leg[dose] = {}
        for leg in LEGS:
            by_dose_leg[dose][leg] = {}
            for m in PRIMARY:
                rr = [{"words": r["words"], "value": r.get(leg)} for r in decomp if r["dose_name"] == dose and r["metric"] == m]
                by_dose_leg[dose][leg][m] = series_stats(rr)

    cmr_10_100: dict[str, Any] = {}
    for dose in sorted({str(r["dose_name"]) for r in cmr}):
        cmr_10_100[dose] = {}
        for m in PRIMARY:
            rr = [{"words": r["words"], "value": r.get("view_minus_repeat")} for r in cmr if r["dose_name"] == dose and r["metric"] == m]
            cmr_10_100[dose][m] = series_stats(rr)

    growth_stats: dict[str, Any] = {}
    for leg in GROWTH_LEGS:
        growth_stats[leg] = {}
        for m in PRIMARY:
            rr = [{"words": r["words"], "value": r.get(leg)} for r in growth if r["metric"] == m]
            growth_stats[leg][m] = series_stats(rr)

    load_readout: dict[str, Any] = {}
    for m in PRIMARY:
        means = {leg: growth_stats[leg][m]["mean"] for leg in GROWTH_LEGS}
        finite_means = {k: v for k, v in means.items() if finite(v)}
        if not finite_means:
            largest = None
        else:
            largest = max(["semantic_growth", "source_growth"], key=lambda k: abs(float(finite_means.get(k, 0.0))))
        load_readout[m] = {
            "mean_leg_growth_10M_to_80M": means,
            "largest_abs_component_between_semantic_and_source": largest,
            "reading": (
                "semantic re-expression carries more of the MAX-minus-1x change" if largest == "semantic_growth" else
                "source admission/repeat-minus-clean carries more of the MAX-minus-1x change" if largest == "source_growth" else
                "not enough finite rows"
            ),
        }
    return {
        "by_dose_leg_10M_to_80M": by_dose_leg,
        "compact_minus_repeat_10M_to_100M": cmr_10_100,
        "max_minus_1x_leg_growth_10M_to_80M": growth_stats,
        "primary_load_readout": load_readout,
    }


def max_abs_residual(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [abs(float(r[key])) for r in rows if finite(r.get(key))]
    return max(vals) if vals else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dose-dir", default=str(DOSE_DIR_DEFAULT))
    ap.add_argument("--seed-spread", default=str(SEED_SPREAD_DEFAULT))
    ap.add_argument("--out-dir", default=str(OUT_DIR_DEFAULT))
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    dose_dir = pathlib.Path(args.dose_dir)
    if not dose_dir.is_absolute():
        dose_dir = USER_ROOT / dose_dir
    out_dir = pathlib.Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = USER_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = dose_dir / "dose_ladder_stable_rows.csv"
    spread_path = pathlib.Path(args.seed_spread)
    if not spread_path.is_absolute():
        spread_path = USER_ROOT / spread_path
    plan = {
        "status": "DOSE_BUDGET_DECOMPOSITION_PLAN",
        "created_utc": now(),
        "inputs": {"dose_rows": str(rows_path), "seed_spread": str(spread_path)},
        "exists": {"dose_rows": rows_path.exists(), "seed_spread": spread_path.exists()},
        "out_dir": str(out_dir),
        "identity": "view-clean = (view-repeat) + (repeat-clean) at each dose/checkpoint/metric where all three arms are available",
        "stable_only": True,
        "no_training_or_evaluation": True,
    }
    (out_dir / "budget_decomposition_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if args.plan_only:
        return
    if not rows_path.exists():
        raise FileNotFoundError(rows_path)

    rows = read_csv(rows_path)
    spread = seed_spread_by_checkpoint(spread_path)
    decomp = build_checkpoint_decomposition(rows, spread)
    cmr = build_cmr_10_100(rows, spread)
    growth = build_growth_rows(decomp, spread)
    write_csv(out_dir / "budget_decomposition_by_checkpoint.csv", decomp)
    write_csv(out_dir / "compact_minus_repeat_10M_to_100M.csv", cmr)
    write_csv(out_dir / "max_minus_1x_budget_leg_growth.csv", growth)
    summary = {
        "status": "DOSE_BUDGET_DECOMPOSITION_DONE",
        "created_utc": now(),
        "input_rows": str(rows_path),
        "row_counts": {"decomposition_rows": len(decomp), "cmr_10_100_rows": len(cmr), "growth_rows": len(growth)},
        "max_abs_decomposition_residual": max_abs_residual(decomp, "decomposition_residual"),
        "max_abs_growth_decomposition_residual": max_abs_residual(growth, "growth_decomposition_residual"),
        "summary": summarize_decomposition(decomp, cmr, growth),
        "files": {
            "decomposition_csv": str(out_dir / "budget_decomposition_by_checkpoint.csv"),
            "cmr_10_100_csv": str(out_dir / "compact_minus_repeat_10M_to_100M.csv"),
            "growth_csv": str(out_dir / "max_minus_1x_budget_leg_growth.csv"),
        },
        "interpretation_use": [
            "If MAX-minus-1x growth is mainly semantic_view_minus_repeat and exceeds same-coordinate seed spread over the exposure profile, retest the view/repeat pair in a second bidirectional architecture.",
            "If MAX-minus-1x growth is mainly source_repeat_minus_clean while semantic_view_minus_repeat stays flat, retest view-versus-clean in a second bidirectional architecture and treat freed-budget/source admission as the load-bearing term.",
            "If total growth declines at MAX, use the decomposition to locate whether the decline comes from semantic rewriting or the cost of replacing independent clean-Qwen rows.",
        ],
    }
    out_json = out_dir / "budget_decomposition_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["# research dose budget decomposition readout", "", "Identity checked: `view-clean = (view-repeat) + (repeat-clean)`.", ""]
    lines.append(f"Rows: decomposition {len(decomp)}, compact-minus-repeat 10M-100M {len(cmr)}, growth {len(growth)}.")
    lines.append(f"Max residuals: decomposition {summary['max_abs_decomposition_residual']}, growth {summary['max_abs_growth_decomposition_residual']}.")
    lines.append("")
    for m, rec in summary["summary"]["primary_load_readout"].items():
        lines.append(f"- {m}: {rec['reading']}; means {rec['mean_leg_growth_10M_to_80M']}")
    lines += ["", f"JSON: `{out_json}`", f"Decomposition CSV: `{out_dir / 'budget_decomposition_by_checkpoint.csv'}`", f"Growth CSV: `{out_dir / 'max_minus_1x_budget_leg_growth.csv'}`"]
    (out_dir / "budget_decomposition_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "out_json": str(out_json), "decomposition_rows": len(decomp), "growth_rows": len(growth)}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
