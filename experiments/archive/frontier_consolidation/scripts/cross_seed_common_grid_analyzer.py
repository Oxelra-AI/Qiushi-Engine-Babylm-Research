#!/usr/bin/env python3
"""research: analyze cross-seed DeBERTa common-grid reproduction structure.

This script is deliberately written before the seed43122 result is read.  Its main
readout is family/column peak-vector structure, not only the aggregate cheap7 peak.
It compares any number of selected_trajectory.json files on the common 70M--100M
2M grid and reports:

- aggregate peak time, near-best connected band, local peak excess;
- per-column argmax vector and column-peak spread;
- Spearman/Pearson correlation of column peak times against a reference;
- family-aggregate peak times for non-volatile, relation/state, syntax/COMPS, and
  volatile columns;
- pairwise deltas against the reference, including volatile-gain share.

It does not perform model scoring and does not submit anything.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
from typing import Any, Dict, Iterable, List, Tuple

CHEAP_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
DERIVED = {
    "cheap6_no_GlobalPIQA": ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "Reading"],
    "cheap5_no_GlobalPIQA_Reading": ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"],
    "relation_state_mean": ["EWoK", "Entity"],
    "syntax_comp_mean": ["BLiMP", "Supplement", "COMPS"],
    "volatile_mean": ["GlobalPIQA", "Reading"],
}
EXPECTED_ENDPOINTS = [f"chck_{m}M" for m in range(70, 102, 2)]


def load_rows(path: pathlib.Path) -> List[Dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError(f"{path} is not a list trajectory")
    out = []
    for r in rows:
        if not isinstance(r, dict):
            raise TypeError(f"bad row in {path}: {r!r}")
        rr = dict(r)
        if rr.get("cheap7") is None and all(rr.get(c) is not None for c in CHEAP_COLUMNS):
            rr["cheap7"] = mean([float(rr[c]) for c in CHEAP_COLUMNS])
        for name, cols in DERIVED.items():
            if all(rr.get(c) is not None for c in cols):
                rr[name] = mean([float(rr[c]) for c in cols])
        out.append(rr)
    return sorted(out, key=lambda x: int(x.get("words", endpoint_to_words(str(x.get("endpoint", "chck_0M"))))))


def endpoint_to_words(endpoint: str) -> int:
    if endpoint.startswith("chck_") and endpoint.endswith("M"):
        return int(endpoint[5:-1]) * 1_000_000
    raise ValueError(f"cannot parse endpoint {endpoint!r}")


def mean(vals: Iterable[float]) -> float:
    vals = list(vals)
    return float(sum(vals) / len(vals))


def pearson(xs: List[float], ys: List[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx, my = mean(xs), mean(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    vx = sum(x*x for x in dx)
    vy = sum(y*y for y in dy)
    if vx <= 0 or vy <= 0:
        return None
    return float(sum(x*y for x, y in zip(dx, dy)) / math.sqrt(vx * vy))


def rankdata(vals: List[float]) -> List[float]:
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and vals[order[j]] == vals[order[i]]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = rank
        i = j
    return ranks


def spearman(xs: List[float], ys: List[float]) -> float | None:
    return pearson(rankdata(xs), rankdata(ys))


def contiguous_band(rows: List[Dict[str, Any]], metric: str, drop: float) -> Dict[str, Any]:
    valid = [r for r in rows if r.get(metric) is not None]
    if not valid:
        return {"endpoints": [], "span_m": None}
    best_val = max(float(r[metric]) for r in valid)
    best_idxs = [i for i, r in enumerate(rows) if r.get(metric) is not None and float(r[metric]) >= best_val - drop]
    if not best_idxs:
        return {"endpoints": [], "span_m": None}
    # Return the contiguous segment containing the first global max.
    best_index = max(range(len(rows)), key=lambda i: (float(rows[i][metric]) if rows[i].get(metric) is not None else -1e99))
    s = best_index
    while s - 1 >= 0 and (s - 1) in best_idxs:
        s -= 1
    e = best_index
    while e + 1 < len(rows) and (e + 1) in best_idxs:
        e += 1
    eps = [rows[i]["endpoint"] for i in range(s, e + 1)]
    span = (endpoint_to_words(eps[-1]) - endpoint_to_words(eps[0])) / 1_000_000 if eps else None
    return {"endpoints": eps, "span_m": span, "best_value": best_val, "drop": drop}


def local_peak_excess(rows: List[Dict[str, Any]], metric: str, endpoint: str) -> float | None:
    idx = next((i for i, r in enumerate(rows) if r.get("endpoint") == endpoint), None)
    if idx is None or rows[idx].get(metric) is None:
        return None
    neigh = []
    for j in [idx - 1, idx + 1]:
        if 0 <= j < len(rows) and rows[j].get(metric) is not None:
            neigh.append(float(rows[j][metric]))
    if not neigh:
        return None
    return float(float(rows[idx][metric]) - mean(neigh))


def summarize(label: str, rows: List[Dict[str, Any]], high_band_drop: float) -> Dict[str, Any]:
    endpoints = [r.get("endpoint") for r in rows]
    missing = [e for e in EXPECTED_ENDPOINTS if e not in endpoints]
    valid = [r for r in rows if r.get("cheap7") is not None]
    best = max(valid, key=lambda r: float(r["cheap7"])) if valid else None
    col_peaks = {}
    for col in CHEAP_COLUMNS + list(DERIVED.keys()):
        col_valid = [r for r in rows if r.get(col) is not None]
        if col_valid:
            b = max(col_valid, key=lambda r: float(r[col]))
            col_peaks[col] = {"endpoint": b["endpoint"], "words": int(b.get("words", endpoint_to_words(b["endpoint"]))), "score": float(b[col])}
    cheap_col_words = [v["words"] for k, v in col_peaks.items() if k in CHEAP_COLUMNS]
    cheap_col_spread = (max(cheap_col_words) - min(cheap_col_words)) / 1_000_000 if cheap_col_words else None
    rows_by_endpoint = {r["endpoint"]: r for r in rows if "endpoint" in r}
    return {
        "status": "ok" if not missing and len(valid) == len(EXPECTED_ENDPOINTS) else "incomplete",
        "label": label,
        "n_rows": len(rows),
        "n_valid": len(valid),
        "missing_expected_endpoints": missing,
        "best_endpoint": best.get("endpoint") if best else None,
        "best_words": int(best.get("words", endpoint_to_words(best["endpoint"]))) if best else None,
        "best_cheap7": float(best["cheap7"]) if best else None,
        "final_endpoint": rows[-1].get("endpoint") if rows else None,
        "final_cheap7": float(rows[-1]["cheap7"]) if rows and rows[-1].get("cheap7") is not None else None,
        "final_minus_best_cheap7": (float(rows[-1]["cheap7"]) - float(best["cheap7"])) if rows and best and rows[-1].get("cheap7") is not None else None,
        "chck_82M_cheap7": float(rows_by_endpoint["chck_82M"]["cheap7"]) if "chck_82M" in rows_by_endpoint and rows_by_endpoint["chck_82M"].get("cheap7") is not None else None,
        "chck_84M_cheap7": float(rows_by_endpoint["chck_84M"]["cheap7"]) if "chck_84M" in rows_by_endpoint and rows_by_endpoint["chck_84M"].get("cheap7") is not None else None,
        "chck_100M_cheap7": float(rows_by_endpoint["chck_100M"]["cheap7"]) if "chck_100M" in rows_by_endpoint and rows_by_endpoint["chck_100M"].get("cheap7") is not None else None,
        "chck_82M_to_100M_delta_cheap7": (float(rows_by_endpoint["chck_100M"]["cheap7"]) - float(rows_by_endpoint["chck_82M"]["cheap7"])) if all(e in rows_by_endpoint and rows_by_endpoint[e].get("cheap7") is not None for e in ["chck_82M", "chck_100M"]) else None,
        "local_peak_excess_cheap7": local_peak_excess(rows, "cheap7", best["endpoint"]) if best else None,
        "connected_near_best_band": contiguous_band(rows, "cheap7", high_band_drop),
        "column_peaks": col_peaks,
        "cheap_column_peak_spread_m": cheap_col_spread,
        "rows": rows,
    }


def compare(reference: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
    ref_peaks = reference.get("column_peaks", {})
    oth_peaks = other.get("column_peaks", {})
    common_cols = [c for c in CHEAP_COLUMNS if c in ref_peaks and c in oth_peaks]
    ref_words = [ref_peaks[c]["words"] / 1_000_000 for c in common_cols]
    oth_words = [oth_peaks[c]["words"] / 1_000_000 for c in common_cols]
    diffs = [o - r for r, o in zip(ref_words, oth_words)]
    rows_ref = {r["endpoint"]: r for r in reference.get("rows", []) if r.get("cheap7") is not None}
    rows_oth = {r["endpoint"]: r for r in other.get("rows", []) if r.get("cheap7") is not None}
    common_endpoints = [e for e in EXPECTED_ENDPOINTS if e in rows_ref and e in rows_oth]
    endpoint_deltas = []
    score_metrics = ["cheap7"] + CHEAP_COLUMNS + list(DERIVED.keys())
    for e in common_endpoints:
        rr, oo = rows_ref[e], rows_oth[e]
        d = {"endpoint": e, "words": endpoint_to_words(e)}
        for col in score_metrics:
            if rr.get(col) is not None and oo.get(col) is not None:
                d[f"delta_{col}"] = float(oo[col]) - float(rr[col])
        pos_delta = max(0.0, d.get("delta_cheap7", 0.0))
        vol_delta = 0.0
        # Contribution to cheap7 from GlobalPIQA and Reading among positive total gain.
        for c in ["GlobalPIQA", "Reading"]:
            val = d.get(f"delta_{c}")
            if val is not None:
                vol_delta += float(val) / 7.0
        d["volatile_gain_share_if_positive_cheap7"] = (vol_delta / pos_delta) if pos_delta > 0 else None
        endpoint_deltas.append(d)
    mean_deltas = {}
    for col in ["cheap7"] + list(DERIVED.keys()) + CHEAP_COLUMNS:
        vals = [d.get(f"delta_{col}") for d in endpoint_deltas if d.get(f"delta_{col}") is not None]
        if vals:
            mean_deltas[f"mean_delta_{col}"] = mean([float(v) for v in vals])
    return {
        "reference_label": reference.get("label"),
        "other_label": other.get("label"),
        "common_peak_columns": common_cols,
        "column_peak_words_reference_m": dict(zip(common_cols, ref_words)),
        "column_peak_words_other_m": dict(zip(common_cols, oth_words)),
        "column_peak_word_diffs_other_minus_reference_m": dict(zip(common_cols, diffs)),
        "column_peak_pearson": pearson(ref_words, oth_words),
        "column_peak_spearman": spearman(ref_words, oth_words),
        "mean_abs_column_peak_shift_m": mean([abs(x) for x in diffs]) if diffs else None,
        "aggregate_peak_shift_m": ((other.get("best_words") - reference.get("best_words")) / 1_000_000) if other.get("best_words") and reference.get("best_words") else None,
        "reference_peak_spread_m": reference.get("cheap_column_peak_spread_m"),
        "other_peak_spread_m": other.get("cheap_column_peak_spread_m"),
        "endpoint_delta_summary": mean_deltas,
        "positive_cheap7_rows": sum(1 for d in endpoint_deltas if d.get("delta_cheap7", -1e9) > 0),
        "broad_positive_rows": sum(1 for d in endpoint_deltas if d.get("delta_cheap7", -1e9) > 0 and d.get("delta_cheap6_no_GlobalPIQA", -1e9) > 0 and d.get("delta_cheap5_no_GlobalPIQA_Reading", -1e9) > 0 and d.get("delta_relation_state_mean", -1e9) >= 0),
        "endpoint_deltas": endpoint_deltas,
    }


def write_md(summary: Dict[str, Any], path: pathlib.Path) -> None:
    lines = []
    lines.append("# research cross-seed/common-grid structural analyzer\n\n")
    lines.append("This readout was defined before reading the seed43122 grid. It treats per-column peak-vector structure as the primary evidence for whether the late phase is structural or stochastic.\n\n")
    lines.append("## Trajectories\n\n")
    lines.append("| label | status | best | best cheap7 | 82M | 84M | 100M | 82→100 Δ | connected near-best span M | local peak excess | cheap column peak spread M |\n")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for label, s in summary["summaries"].items():
        band = s.get("connected_near_best_band", {})
        lines.append(f"| {label} | {s.get('status')} | {s.get('best_endpoint')} | {fmt(s.get('best_cheap7'))} | {fmt(s.get('chck_82M_cheap7'))} | {fmt(s.get('chck_84M_cheap7'))} | {fmt(s.get('chck_100M_cheap7'))} | {fmt(s.get('chck_82M_to_100M_delta_cheap7'))} | {fmt(band.get('span_m'))} | {fmt(s.get('local_peak_excess_cheap7'))} | {fmt(s.get('cheap_column_peak_spread_m'))} |\n")
    lines.append("\n## Column peak vectors\n\n")
    lines.append("| label | " + " | ".join(CHEAP_COLUMNS) + " |\n")
    lines.append("|---|" + "|".join(["---:"] * len(CHEAP_COLUMNS)) + "|\n")
    for label, s in summary["summaries"].items():
        vals = []
        for c in CHEAP_COLUMNS:
            rec = s.get("column_peaks", {}).get(c, {})
            vals.append(str(int(rec.get("words", 0) / 1_000_000)) if rec else "")
        lines.append(f"| {label} | " + " | ".join(vals) + " |\n")
    if summary.get("comparisons_to_reference"):
        lines.append("\n## Comparisons to reference\n\n")
        lines.append("| label | aggregate peak shift M | mean abs column peak shift M | peak Pearson | peak Spearman | other spread M | mean Δcheap7 | mean Δcheap6(no GP) | mean Δcheap5(no GP/Reading) | mean ΔEWoK/Entity | positive cheap7 rows | broad positive rows |\n")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for label, c in summary["comparisons_to_reference"].items():
            md = c.get("endpoint_delta_summary", {})
            lines.append(f"| {label} | {fmt(c.get('aggregate_peak_shift_m'))} | {fmt(c.get('mean_abs_column_peak_shift_m'))} | {fmt(c.get('column_peak_pearson'))} | {fmt(c.get('column_peak_spearman'))} | {fmt(c.get('other_peak_spread_m'))} | {fmt(md.get('mean_delta_cheap7'))} | {fmt(md.get('mean_delta_cheap6_no_GlobalPIQA'))} | {fmt(md.get('mean_delta_cheap5_no_GlobalPIQA_Reading'))} | {fmt(md.get('mean_delta_relation_state_mean'))} | {c.get('positive_cheap7_rows')} | {c.get('broad_positive_rows')} |\n")
    lines.append("\n## Predeclared reading\n\n")
    lines.append("- Structural aligned: column peak vectors correlate and aggregate peak also aligns; study residual-capacity allocation dynamics.\n")
    lines.append("- Structural but misaligned: column dispersion resembles reference but aggregate peak shifts; study allocation dynamics plus stochastic stabilization.\n")
    lines.append("- Unstructured: column peak vectors do not correlate and no similar late broad phase appears; treat seed43022 chck84 as fragile endpoint and study stochastic competence stabilization before new training.\n")
    lines.append("- Any stabilization or averaging probe must improve cheap7, cheap6 without GlobalPIQA, cheap5 without GlobalPIQA/Reading, and nonnegative EWoK/Entity; volatile-column-only gains are not enough.\n")
    lines.append(f"\nJSON: `{summary['out_json']}`\n")
    path.write_text("".join(lines), encoding="utf-8")


def fmt(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.6f}"
    return str(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory", action="append", required=True, help="LABEL=PATH selected_trajectory.json")
    ap.add_argument("--reference-label", required=True)
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    ap.add_argument("--high-band-drop", type=float, default=0.2)
    ap.add_argument("--strict-complete", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    labels_paths: List[Tuple[str, pathlib.Path]] = []
    for spec in args.trajectory:
        if "=" not in spec:
            raise SystemExit(f"--trajectory must be LABEL=PATH, got {spec}")
        label, path = spec.split("=", 1)
        labels_paths.append((label, pathlib.Path(path)))

    summaries = {label: summarize(label, load_rows(path), args.high_band_drop) for label, path in labels_paths}
    if args.reference_label not in summaries:
        raise SystemExit(f"reference label {args.reference_label} absent from trajectories")
    comparisons = {}
    ref = summaries[args.reference_label]
    for label, s in summaries.items():
        if label != args.reference_label:
            comparisons[label] = compare(ref, s)
    complete = all(s.get("status") == "ok" for s in summaries.values())
    out_json = args.out_dir / "cross_seed_common_grid_analysis.json"
    out_md = args.out_dir / "cross_seed_common_grid_analysis.md"
    result = {
        "status": "CROSS_SEED_COMMON_GRID_ANALYSIS",
        "complete": complete,
        "reference_label": args.reference_label,
        "expected_endpoints": EXPECTED_ENDPOINTS,
        "cheap_columns": CHEAP_COLUMNS,
        "derived_columns": DERIVED,
        "summaries": summaries,
        "comparisons_to_reference": comparisons,
        "predeclared_reading": {
            "primary_readout": "per-column peak-vector correlation/shift, then aggregate cheap7 peak location",
            "structural_aligned": "column peak vectors correlate and aggregate peak aligns; residual-capacity allocation dynamics remain strong",
            "structural_misaligned": "column peak vectors share dispersion but aggregate peak shifts; combine allocation dynamics with stochastic stabilization",
            "unstructured": "column peak vectors uncorrelated or no late broad phase; seed43022 chck84 is fragile endpoint and stabilization/variance becomes scientific object",
            "stabilization_minimum_signal": "positive cheap7, cheap6-no-GlobalPIQA, cheap5-no-GlobalPIQA/Reading, and nonnegative EWoK/Entity; volatile-only gain fails",
        },
        "out_json": str(out_json),
        "out_md": str(out_md),
    }
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_md(result, out_md)
    print(json.dumps({"status": result["status"], "complete": complete, "out_json": str(out_json), "out_md": str(out_md)}, indent=2), flush=True)
    if args.strict_complete and not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
