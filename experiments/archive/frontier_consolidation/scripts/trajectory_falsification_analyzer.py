#!/usr/bin/env python3
"""research: falsifiable analysis of dense cheap7 checkpoint trajectories.

This script is intentionally separate from the evaluator.  It reads one or more
trajectory JSON/CSV files produced by batch_trajectory_eval.py plus an
optional built-in reference trajectory, and summarizes what the resulting paths
actually say about two hypotheses:

1. seed robustness of the scale1.75 82M peak;
2. adapter-scale control of the timing and width of the late competence peak.

It never treats any pattern as automatically favorable.  Irregular timing,
scattered column maxima, or a peak carried by one small/unstable column are
reported as evidence against the simple account.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import re
from statistics import mean, pstdev
from typing import Any

CHEAP_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]

REFERENCE_SCALE1P75_SEED43022 = [
    {"endpoint": "chck_77M", "words": 77_000_000, "cheap7": 43.28214285714286},
    {"endpoint": "chck_78M", "words": 78_000_000, "cheap7": 43.70214285714286},
    {"endpoint": "chck_79M", "words": 79_000_000, "cheap7": 43.57857142857143},
    {"endpoint": "chck_80M", "words": 80_000_000, "cheap7": 43.81214285714286},
    {"endpoint": "chck_81M", "words": 81_000_000, "cheap7": 43.64928571428572},
    {"endpoint": "chck_82M", "words": 82_000_000, "cheap7": 43.95944987645173},
    {"endpoint": "chck_83M", "words": 83_000_000, "cheap7": 43.807857142857145},
    {"endpoint": "chck_100M", "words": 100_000_000, "cheap7": 43.543159919261925},
]

# Hardened full-score vector for the protected seed43022 reference where available.
REFERENCE_FULL_ROWS = {
    "chck_80M": {"SuperGLUE": 69.260, "Overall": 41.77163494053074},
    "chck_82M": {"SuperGLUE": 69.7661813713118, "Overall": 41.942481167385985},
    "chck_100M": {"SuperGLUE": 69.3346, "Overall": 41.57074653643003},
}


def parse_words(endpoint: str, fallback: Any = None) -> int | None:
    if fallback is not None:
        try:
            return int(float(fallback))
        except Exception:
            pass
    m = re.search(r"chck_(\d+)M", str(endpoint))
    if m:
        return int(m.group(1)) * 1_000_000
    return None


def finite_float(x: Any) -> float | None:
    try:
        y = float(x)
    except Exception:
        return None
    if math.isfinite(y):
        return y
    return None


def read_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("trajectory"), list):
            rows = data["trajectory"]
        elif isinstance(data, list):
            rows = data
        else:
            raise ValueError(f"Unrecognized trajectory payload in {path}")
    out = []
    for r in rows:
        endpoint = str(r.get("endpoint") or r.get("checkpoint") or "")
        c7 = finite_float(r.get("cheap7"))
        if c7 is None:
            continue
        words = parse_words(endpoint, r.get("words"))
        if words is None:
            continue
        rr: dict[str, Any] = {"endpoint": endpoint, "words": words, "cheap7": c7}
        for c in CHEAP_COLUMNS:
            v = finite_float(r.get(c))
            if v is not None:
                rr[c] = v
        out.append(rr)
    return sorted(out, key=lambda r: (r["words"], r["endpoint"]))


def summarize(label: str, rows: list[dict[str, Any]], width_drop: float) -> dict[str, Any]:
    if not rows:
        return {"label": label, "status": "no_valid_rows"}
    best = max(rows, key=lambda r: r["cheap7"])
    peak = best["cheap7"]
    band = [r for r in rows if r["cheap7"] >= peak - width_drop]
    late = max(rows, key=lambda r: r["words"])
    early = min(rows, key=lambda r: r["words"])
    sorted_by_words = sorted(rows, key=lambda r: r["words"])
    adjacent_steps = []
    for a, b in zip(sorted_by_words, sorted_by_words[1:]):
        adjacent_steps.append({
            "from": a["endpoint"], "to": b["endpoint"],
            "delta_words_m": (b["words"] - a["words"]) / 1e6,
            "delta_cheap7": b["cheap7"] - a["cheap7"],
        })
    neighbor_mean = None
    local_peak_excess = None
    for i, r in enumerate(sorted_by_words):
        if r is best and 0 < i < len(sorted_by_words) - 1:
            neighbor_mean = (sorted_by_words[i - 1]["cheap7"] + sorted_by_words[i + 1]["cheap7"]) / 2.0
            local_peak_excess = peak - neighbor_mean
            break
    col_maxima = {}
    available_col_count = 0
    for c in CHEAP_COLUMNS:
        vals = [r for r in rows if c in r]
        if vals:
            available_col_count += 1
            br = max(vals, key=lambda r: r[c])
            col_maxima[c] = {"endpoint": br["endpoint"], "words": br["words"], "score": br[c]}
    col_peak_words = [v["words"] for v in col_maxima.values()]
    col_peak_spread_m = (max(col_peak_words) - min(col_peak_words)) / 1e6 if col_peak_words else None
    return {
        "label": label,
        "status": "ok",
        "n_rows": len(rows),
        "first_words": early["words"],
        "last_words": late["words"],
        "best_endpoint": best["endpoint"],
        "best_words": best["words"],
        "best_cheap7": peak,
        "last_endpoint": late["endpoint"],
        "last_cheap7": late["cheap7"],
        "late_drop_from_peak": late["cheap7"] - peak,
        "width_drop": width_drop,
        "width_endpoints": [r["endpoint"] for r in band],
        "width_words_m_span": (max(r["words"] for r in band) - min(r["words"] for r in band)) / 1e6 if band else 0.0,
        "mean_cheap7": mean(r["cheap7"] for r in rows),
        "std_cheap7": pstdev(r["cheap7"] for r in rows) if len(rows) > 1 else 0.0,
        "local_peak_neighbor_mean": neighbor_mean,
        "local_peak_excess": local_peak_excess,
        "adjacent_steps": adjacent_steps,
        "available_column_count": available_col_count,
        "column_maxima": col_maxima,
        "column_peak_spread_m": col_peak_spread_m,
        "rows": rows,
    }


def compare_to_reference(label: str, summary: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    if summary.get("status") != "ok":
        return {"label": label, "status": "missing"}
    peak_words_m = summary["best_words"] / 1e6
    ref_peak_words_m = ref["best_words"] / 1e6
    peak_shift_m = peak_words_m - ref_peak_words_m
    width_shift_m = summary.get("width_words_m_span") - ref.get("width_words_m_span")
    late_drop = summary.get("late_drop_from_peak")
    col_spread = summary.get("column_peak_spread_m")
    return {
        "label": label,
        "reference_label": ref.get("label"),
        "peak_shift_m_words": peak_shift_m,
        "width_shift_m_words": width_shift_m,
        "late_drop_from_peak": late_drop,
        "column_peak_spread_m": col_spread,
        "reading": {
            "seed_robustness_support": "stronger if scale1.75 seed43122 peaks within roughly 4M words of seed43022 and also has a real late decline; weakened if the best point is far away, flat, or carried by one column",
            "scale_interference_support": "stronger if scale1.25 shifts later and/or widens the near-peak band while preserving similar family profile; weakened if timing is irregular or family maxima scatter",
            "semantic_overlap_surface_diversity_not_tested": True,
        }
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory", action="append", nargs=2, metavar=("LABEL", "PATH"), default=[],
                    help="trajectory JSON/CSV produced by batch_trajectory_eval.py")
    ap.add_argument("--include-reference", action="store_true")
    ap.add_argument("--width-drop", type=float, default=0.2)
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    trajectories: dict[str, list[dict[str, Any]]] = {}
    if args.include_reference:
        rows = []
        for r in REFERENCE_SCALE1P75_SEED43022:
            rr = dict(r)
            rr.update(REFERENCE_FULL_ROWS.get(r["endpoint"], {}))
            rows.append(rr)
        trajectories["scale1p75_seed43022_reference"] = rows
    for label, p in args.trajectory:
        trajectories[label] = read_rows(pathlib.Path(p))

    summaries = {label: summarize(label, rows, args.width_drop) for label, rows in trajectories.items()}
    ref = summaries.get("scale1p75_seed43022_reference")
    comparisons = {}
    if ref and ref.get("status") == "ok":
        for label, s in summaries.items():
            if label != "scale1p75_seed43022_reference":
                comparisons[label] = compare_to_reference(label, s, ref)

    result = {
        "status": "TRAJECTORY_FALSIFICATION_ANALYSIS",
        "purpose": "Summarize dense cheap7 trajectories as falsifiable local contrasts, not as a transferable law by themselves.",
        "width_drop": args.width_drop,
        "summaries": summaries,
        "comparisons_to_reference": comparisons,
        "interpretation_rules_fixed_before_step168_results": {
            "scale1p75_seed43122": [
                "Similar best time and a late decline would support seed robustness of the seed43022 peak.",
                "A far-away best time, flat/noisy path, or peak driven by one small column weakens the seed-robust peak account."
            ],
            "scale1p25_seed43022": [
                "A later or broader high-score band with similar family profile supports adapter-amplitude control of interference speed.",
                "Earlier peak, irregular timing, or capability-specific tradeoffs weakens the amplitude-control account."
            ],
            "not_tested_by_these_runs": "The context-mediated invariance theory predicts semantic-overlap by surface-diversity interaction; seed/scale runs do not vary that factor."
        }
    }
    out_json = args.out_dir / "trajectory_falsification_analysis.json"
    out_md = args.out_dir / "trajectory_falsification_analysis.md"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["# research trajectory falsification analysis\n\n"]
    lines.append("This file treats the dense runs as local contrasts. They can support or weaken specific seed/scale accounts, but they do not by themselves prove a transferable data-efficient learning law.\n\n")
    for label, s in summaries.items():
        lines.append(f"## {label}\n")
        if s.get("status") != "ok":
            lines.append("No valid rows.\n\n")
            continue
        lines.append(f"- best: {s['best_endpoint']} at {s['best_words']/1e6:.0f}M, cheap7={s['best_cheap7']:.4f}\n")
        lines.append(f"- last: {s['last_endpoint']} cheap7={s['last_cheap7']:.4f}; last-minus-best={s['late_drop_from_peak']:+.4f}\n")
        lines.append(f"- band within {args.width_drop:.3f} of best spans {s['width_words_m_span']:.1f}M words: {', '.join(s['width_endpoints'])}\n")
        if s.get("local_peak_excess") is not None:
            lines.append(f"- local peak excess over adjacent-neighbor mean: {s['local_peak_excess']:+.4f}\n")
        if s.get("column_peak_spread_m") is not None:
            lines.append(f"- column peak time spread: {s['column_peak_spread_m']:.1f}M words across {s['available_column_count']} columns\n")
        lines.append("\n")
    if comparisons:
        lines.append("## Comparisons to scale1.75 seed43022 reference\n\n")
        for label, c in comparisons.items():
            lines.append(f"- {label}: peak shift {c['peak_shift_m_words']:+.1f}M, width shift {c['width_shift_m_words']:+.1f}M, late drop {c['late_drop_from_peak']:+.4f}, column peak spread {c.get('column_peak_spread_m')}M.\n")
        lines.append("\n")
    lines.append("## Fixed interpretation\n\n")
    lines.append("- Similar scale1.75 peak timing supports seed robustness only if the late decline and family profile also resemble the reference.\n")
    lines.append("- Lower scale supports amplitude-controlled interference only if it shifts later and/or broadens the high-score band without becoming a different family tradeoff.\n")
    lines.append("- Semantic-overlap by surface-diversity interaction remains untested here and needs a matched semantic-view contrast under a different architecture or scale.\n\n")
    lines.append(f"JSON: `{out_json}`\n")
    out_md.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": result["status"], "out_json": str(out_json), "out_md": str(out_md), "labels": list(summaries)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
