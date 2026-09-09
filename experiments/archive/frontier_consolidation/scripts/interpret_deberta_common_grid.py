#!/usr/bin/env python3
"""research: interpret selected DeBERTa MLM common-grid trajectories.

This script is intended for use after the common 70M--100M selected scoring files
exist and have passed the research integrity script. It keeps the scientific readout
fixed before seeing the pending score files: peak timing, high-score width, late
falloff, family balance, volatile-column contribution, and the adapter-energy
context from research.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
from statistics import mean, pstdev
from typing import Any

EXPECTED_ENDPOINTS = [
    "chck_70M", "chck_72M", "chck_74M", "chck_76M", "chck_78M", "chck_80M",
    "chck_82M", "chck_84M", "chck_86M", "chck_88M", "chck_90M", "chck_92M",
    "chck_94M", "chck_96M", "chck_98M", "chck_100M",
]
CHEAP_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
NO_GLOBAL_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "Reading"]
CORE_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]
REL_STATE_COLUMNS = ["EWoK", "Entity"]
SYNTAX_COMP_COLUMNS = ["BLiMP", "Supplement", "COMPS"]
VOLATILE_COLUMNS = ["GlobalPIQA", "Reading"]
WINDOWS = {
    "70_76M": (70_000_000, 76_000_000),
    "78_86M": (78_000_000, 86_000_000),
    "90_100M": (90_000_000, 100_000_000),
    "70_100M": (70_000_000, 100_000_000),
}
DERIVED_NAMES = [
    "cheap7",
    "cheap6_no_GlobalPIQA",
    "cheap5_no_GlobalPIQA_Reading",
    "relation_state_mean",
    "syntax_comp_mean",
    "volatile_mean",
]


def fnum(x: Any) -> float | None:
    if x is None or x == "":
        return None
    try:
        y = float(x)
    except Exception:
        return None
    if math.isnan(y):
        return None
    return y


def mean_cols(row: dict[str, Any], cols: list[str]) -> float | None:
    vals = [fnum(row.get(c)) for c in cols]
    if any(v is None for v in vals):
        return None
    return float(mean(float(v) for v in vals))


def endpoint_words(endpoint: str) -> int | None:
    if not isinstance(endpoint, str) or not endpoint.startswith("chck_") or not endpoint.endswith("M"):
        return None
    try:
        return int(endpoint[len("chck_"):-1]) * 1_000_000
    except Exception:
        return None


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_labeled_path(spec: str) -> tuple[str, pathlib.Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("Use LABEL=PATH")
    label, path = spec.split("=", 1)
    if not label:
        raise argparse.ArgumentTypeError("empty label")
    return label, pathlib.Path(path)


def load_trajectory(path: pathlib.Path, expected_endpoints: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    data = read_json(path)
    if isinstance(data, dict) and isinstance(data.get("rows"), list):
        data = data["rows"]
    if not isinstance(data, list):
        return [], [f"not_list:{path}"]

    rows: list[dict[str, Any]] = []
    issues: list[str] = []
    seen: set[str] = set()
    for raw in data:
        if not isinstance(raw, dict):
            issues.append("non_dict_row")
            continue
        endpoint = raw.get("endpoint")
        words = fnum(raw.get("words"))
        if words is None and endpoint:
            ew = endpoint_words(str(endpoint))
            words = float(ew) if ew is not None else None
        if endpoint is None or words is None:
            issues.append(f"missing_endpoint_or_words:{raw}")
            continue
        row = dict(raw)
        row["endpoint"] = str(endpoint)
        row["words"] = int(words)
        seen.add(row["endpoint"])
        if row.get("error"):
            issues.append(f"{row['endpoint']}:error:{str(row.get('error'))[:160]}")
            rows.append(row)
            continue
        for col in CHEAP_COLUMNS:
            row[col] = fnum(row.get(col))
            if row[col] is None:
                issues.append(f"{row['endpoint']}:missing_score:{col}")
        recomputed = mean_cols(row, CHEAP_COLUMNS)
        c7 = fnum(row.get("cheap7"))
        if c7 is None:
            issues.append(f"{row['endpoint']}:missing_cheap7")
            row["cheap7"] = recomputed
        elif recomputed is not None and abs(c7 - recomputed) > 1e-9:
            issues.append(f"{row['endpoint']}:cheap7_mismatch:{c7}:{recomputed}")
            row["cheap7"] = c7
        else:
            row["cheap7"] = c7
        row["cheap6_no_GlobalPIQA"] = mean_cols(row, NO_GLOBAL_COLUMNS)
        row["cheap5_no_GlobalPIQA_Reading"] = mean_cols(row, CORE_COLUMNS)
        row["relation_state_mean"] = mean_cols(row, REL_STATE_COLUMNS)
        row["syntax_comp_mean"] = mean_cols(row, SYNTAX_COMP_COLUMNS)
        row["volatile_mean"] = mean_cols(row, VOLATILE_COLUMNS)
        rows.append(row)

    expected_set = set(expected_endpoints)
    missing = [e for e in expected_endpoints if e not in seen]
    extra = sorted(e for e in seen if e not in expected_set)
    for e in missing:
        issues.append(f"missing_expected_endpoint:{e}")
    for e in extra:
        issues.append(f"extra_endpoint:{e}")
    rows = sorted(rows, key=lambda r: int(r.get("words", 0)))
    return rows, issues


def summarize_adapter_energy(adapter_json: pathlib.Path | None, labels: list[str], reference_label: str) -> dict[str, Any]:
    if adapter_json is None:
        return {"status": "not_provided"}
    if not adapter_json.exists():
        return {"status": "missing", "path": str(adapter_json)}
    data = read_json(adapter_json)
    runs = data.get("runs", {}) if isinstance(data, dict) else {}
    out: dict[str, Any] = {"status": "ok", "path": str(adapter_json), "runs": {}}
    ref_mean = None
    ref_by_endpoint: dict[str, float] = {}
    tmp: dict[str, Any] = {}
    for label in labels:
        block = runs.get(label)
        if not isinstance(block, dict):
            tmp[label] = {"status": "missing"}
            continue
        rows = block.get("rows", [])
        vals = [fnum(r.get("effective_up_to_stock_ratio")) for r in rows if isinstance(r, dict)]
        vals = [v for v in vals if v is not None]
        scales = [fnum(r.get("adapter_scale")) for r in rows if isinstance(r, dict)]
        scales = [v for v in scales if v is not None]
        by_ep = {}
        for r in rows:
            if isinstance(r, dict) and r.get("endpoint") is not None and fnum(r.get("effective_up_to_stock_ratio")) is not None:
                by_ep[str(r["endpoint"])] = float(r["effective_up_to_stock_ratio"])
        tmp[label] = {
            "status": "ok" if vals else "no_values",
            "n": len(vals),
            "adapter_scale": float(scales[0]) if scales else None,
            "mean_effective_up_to_stock_ratio": float(mean(vals)) if vals else None,
            "min_effective_up_to_stock_ratio": float(min(vals)) if vals else None,
            "max_effective_up_to_stock_ratio": float(max(vals)) if vals else None,
            "endpoint_effective_up_to_stock_ratio": by_ep,
            "late_70_to_100_slope_per_2M": (by_ep.get("chck_100M") - by_ep.get("chck_70M")) / 15.0 if by_ep.get("chck_100M") is not None and by_ep.get("chck_70M") is not None else None,
            "delta_82_to_100": (by_ep.get("chck_100M") - by_ep.get("chck_82M")) if by_ep.get("chck_100M") is not None and by_ep.get("chck_82M") is not None else None,
        }
        if label == reference_label and vals:
            ref_mean = float(mean(vals))
            ref_by_endpoint = by_ep
    for label, block in tmp.items():
        if block.get("mean_effective_up_to_stock_ratio") is not None and ref_mean is not None and ref_mean != 0:
            block["ratio_to_reference_mean"] = float(block["mean_effective_up_to_stock_ratio"] / ref_mean)
        by_ep = block.get("endpoint_effective_up_to_stock_ratio") if isinstance(block, dict) else None
        if isinstance(by_ep, dict) and ref_by_endpoint:
            block["endpoint_ratio_to_reference"] = {ep: (float(v) / ref_by_endpoint[ep]) for ep, v in by_ep.items() if ep in ref_by_endpoint and ref_by_endpoint[ep] != 0}
    out["runs"] = tmp
    return out


def contiguous_high_band(rows: list[dict[str, Any]], best: dict[str, Any], drop: float, key: str = "cheap7") -> dict[str, Any]:
    """Return the connected high-score band around the first maximum row."""
    if not rows or fnum(best.get(key)) is None:
        return {"status": "unavailable"}
    best_idx = next((i for i, r in enumerate(rows) if r.get("endpoint") == best.get("endpoint")), None)
    if best_idx is None:
        return {"status": "best_not_found"}
    threshold = float(best[key]) - drop
    lo = best_idx
    hi = best_idx
    while lo > 0 and fnum(rows[lo - 1].get(key)) is not None and float(rows[lo - 1][key]) >= threshold:
        lo -= 1
    while hi < len(rows) - 1 and fnum(rows[hi + 1].get(key)) is not None and float(rows[hi + 1][key]) >= threshold:
        hi += 1
    band = rows[lo:hi + 1]
    return {
        "status": "ok",
        "threshold": threshold,
        "endpoints": [r["endpoint"] for r in band],
        "span_m": (int(band[-1]["words"]) - int(band[0]["words"])) / 1_000_000.0,
        "n": len(band),
        "left_endpoint": band[0]["endpoint"],
        "right_endpoint": band[-1]["endpoint"],
    }

def local_peak_excess(rows: list[dict[str, Any]], key: str = "cheap7") -> dict[str, Any] | None:
    vals = [(i, fnum(r.get(key))) for i, r in enumerate(rows)]
    best_rec = None
    for i, val in vals:
        if val is None or i == 0 or i == len(rows) - 1:
            continue
        prev = vals[i - 1][1]
        nxt = vals[i + 1][1]
        if prev is None or nxt is None:
            continue
        excess = float(val - (prev + nxt) / 2.0)
        rec = {"endpoint": rows[i]["endpoint"], "words": rows[i]["words"], "excess": excess}
        if best_rec is None or excess > best_rec["excess"]:
            best_rec = rec
    return best_rec


def summarize_trajectory(label: str, rows: list[dict[str, Any]], issues: list[str], high_band_drop: float) -> dict[str, Any]:
    valid = [r for r in rows if fnum(r.get("cheap7")) is not None]
    if not valid:
        return {"status": "no_valid_rows", "issues": issues, "n_rows": len(rows)}
    best = max(valid, key=lambda r: float(r["cheap7"]))
    final = max(valid, key=lambda r: int(r["words"]))
    by_endpoint = {r["endpoint"]: r for r in valid}
    near = [r for r in valid if float(r["cheap7"]) >= float(best["cheap7"]) - high_band_drop]
    connected_near = contiguous_high_band(valid, best, high_band_drop, "cheap7")
    column_peaks: dict[str, Any] = {}
    peak_words: list[int] = []
    for col in CHEAP_COLUMNS:
        cvalid = [r for r in valid if fnum(r.get(col)) is not None]
        if not cvalid:
            continue
        cbest = max(cvalid, key=lambda r: float(r[col]))
        column_peaks[col] = {"endpoint": cbest["endpoint"], "words": int(cbest["words"]), "score": float(cbest[col])}
        peak_words.append(int(cbest["words"]))

    windows: dict[str, Any] = {}
    for wname, (lo, hi) in WINDOWS.items():
        inside = [r for r in valid if lo <= int(r["words"]) <= hi]
        w: dict[str, Any] = {"n": len(inside)}
        for key in DERIVED_NAMES:
            vals = [fnum(r.get(key)) for r in inside]
            vals = [v for v in vals if v is not None]
            w[f"mean_{key}"] = float(mean(vals)) if vals else None
            w[f"std_{key}"] = float(pstdev(vals)) if len(vals) > 1 else (0.0 if vals else None)
        windows[wname] = w

    row82 = by_endpoint.get("chck_82M")
    row100 = by_endpoint.get("chck_100M")
    row80 = by_endpoint.get("chck_80M")
    out = {
        "status": "ok" if not issues else "has_issues",
        "label": label,
        "issues": issues,
        "n_rows": len(rows),
        "n_valid": len(valid),
        "first_endpoint": valid[0]["endpoint"],
        "last_endpoint": valid[-1]["endpoint"],
        "best_endpoint": best["endpoint"],
        "best_words": int(best["words"]),
        "best_cheap7": float(best["cheap7"]),
        "best_cheap6_no_GlobalPIQA": fnum(best.get("cheap6_no_GlobalPIQA")),
        "best_cheap5_no_GlobalPIQA_Reading": fnum(best.get("cheap5_no_GlobalPIQA_Reading")),
        "best_relation_state_mean": fnum(best.get("relation_state_mean")),
        "best_syntax_comp_mean": fnum(best.get("syntax_comp_mean")),
        "best_volatile_mean": fnum(best.get("volatile_mean")),
        "final_endpoint": final["endpoint"],
        "final_cheap7": fnum(final.get("cheap7")),
        "final_minus_best_cheap7": float(float(final["cheap7"]) - float(best["cheap7"])),
        "near_best_drop": high_band_drop,
        "near_best_endpoints": [r["endpoint"] for r in near],
        "near_best_span_m": (max(int(r["words"]) for r in near) - min(int(r["words"]) for r in near)) / 1_000_000.0 if near else None,
        "contiguous_near_best_band": connected_near,
        "contiguous_near_best_span_m": connected_near.get("span_m") if isinstance(connected_near, dict) else None,
        "mean_cheap7": float(mean(float(r["cheap7"]) for r in valid)),
        "std_cheap7": float(pstdev([float(r["cheap7"]) for r in valid])) if len(valid) > 1 else 0.0,
        "chck_80M_cheap7": fnum(row80.get("cheap7")) if row80 else None,
        "chck_82M_cheap7": fnum(row82.get("cheap7")) if row82 else None,
        "chck_100M_cheap7": fnum(row100.get("cheap7")) if row100 else None,
        "chck_82M_to_100M_delta_cheap7": (float(row100["cheap7"]) - float(row82["cheap7"])) if row82 and row100 and fnum(row82.get("cheap7")) is not None and fnum(row100.get("cheap7")) is not None else None,
        "local_peak_excess_cheap7": local_peak_excess(valid, "cheap7"),
        "column_peaks": column_peaks,
        "column_peak_spread_m": (max(peak_words) - min(peak_words)) / 1_000_000.0 if peak_words else None,
        "windows": windows,
        "rows": valid,
    }
    return out


def delta_row(row: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"endpoint": row["endpoint"], "words": int(row["words"]), "reference_endpoint": ref["endpoint"]}
    keys = CHEAP_COLUMNS + ["cheap7", "cheap6_no_GlobalPIQA", "cheap5_no_GlobalPIQA_Reading", "relation_state_mean", "syntax_comp_mean", "volatile_mean"]
    for key in keys:
        a = fnum(row.get(key)); b = fnum(ref.get(key))
        if a is not None and b is not None:
            out[f"delta_{key}"] = float(a - b)
    column_deltas = [out.get(f"delta_{c}") for c in CHEAP_COLUMNS if out.get(f"delta_{c}") is not None]
    out["n_positive_columns"] = sum(1 for d in column_deltas if float(d) > 0)
    out["n_negative_columns"] = sum(1 for d in column_deltas if float(d) < 0)
    pos_total = sum(max(float(d), 0.0) for d in column_deltas)
    volatile_pos = sum(max(float(out.get(f"delta_{c}", 0.0)), 0.0) for c in VOLATILE_COLUMNS)
    out["positive_gain_share_volatile"] = float(volatile_pos / pos_total) if pos_total > 0 else None
    out["positive_gain_share_GlobalPIQA"] = float(max(float(out.get("delta_GlobalPIQA", 0.0)), 0.0) / pos_total) if pos_total > 0 else None
    out["wide_family_positive"] = bool(
        out["n_positive_columns"] >= 4
        and (fnum(out.get("delta_cheap6_no_GlobalPIQA")) or -999.0) > 0.0
        and (fnum(out.get("delta_cheap5_no_GlobalPIQA_Reading")) or -999.0) > 0.0
        and (fnum(out.get("positive_gain_share_volatile")) or 1.0) <= 0.5
    )
    return out


def summarize_comparison(label: str, rows: list[dict[str, Any]], ref_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_ref = {r["endpoint"]: r for r in ref_rows if fnum(r.get("cheap7")) is not None}
    paired = [delta_row(r, by_ref[r["endpoint"]]) for r in rows if r["endpoint"] in by_ref and fnum(r.get("cheap7")) is not None]
    if not paired:
        return {"status": "no_common_rows", "label": label}

    def vals(key: str, subset: list[dict[str, Any]] | None = None) -> list[float]:
        src = paired if subset is None else subset
        out = [fnum(r.get(key)) for r in src]
        return [float(v) for v in out if v is not None]

    def mean_key(key: str, subset: list[dict[str, Any]] | None = None) -> float | None:
        v = vals(key, subset)
        return float(mean(v)) if v else None

    def max_key_row(key: str) -> dict[str, Any]:
        eligible = [r for r in paired if fnum(r.get(key)) is not None]
        return max(eligible, key=lambda r: float(r[key])) if eligible else {}

    windows: dict[str, Any] = {}
    for wname, (lo, hi) in WINDOWS.items():
        inside = [r for r in paired if lo <= int(r["words"]) <= hi]
        windows[wname] = {
            "n": len(inside),
            "mean_delta_cheap7": mean_key("delta_cheap7", inside),
            "mean_delta_cheap6_no_GlobalPIQA": mean_key("delta_cheap6_no_GlobalPIQA", inside),
            "mean_delta_cheap5_no_GlobalPIQA_Reading": mean_key("delta_cheap5_no_GlobalPIQA_Reading", inside),
            "mean_delta_relation_state_mean": mean_key("delta_relation_state_mean", inside),
            "mean_delta_syntax_comp_mean": mean_key("delta_syntax_comp_mean", inside),
            "mean_delta_volatile_mean": mean_key("delta_volatile_mean", inside),
            "n_positive_delta_cheap7": sum(1 for r in inside if (fnum(r.get("delta_cheap7")) or 0.0) > 0),
            "n_wide_family_positive": sum(1 for r in inside if r.get("wide_family_positive")),
        }

    best_delta = max_key_row("delta_cheap7")
    best_no_global = max_key_row("delta_cheap6_no_GlobalPIQA")
    positive = [r for r in paired if (fnum(r.get("delta_cheap7")) or 0.0) > 0]
    wide = [r for r in paired if r.get("wide_family_positive")]
    return {
        "status": "ok",
        "label": label,
        "n_common": len(paired),
        "mean_delta_cheap7": mean_key("delta_cheap7"),
        "mean_delta_cheap6_no_GlobalPIQA": mean_key("delta_cheap6_no_GlobalPIQA"),
        "mean_delta_cheap5_no_GlobalPIQA_Reading": mean_key("delta_cheap5_no_GlobalPIQA_Reading"),
        "mean_delta_relation_state_mean": mean_key("delta_relation_state_mean"),
        "mean_delta_syntax_comp_mean": mean_key("delta_syntax_comp_mean"),
        "mean_delta_volatile_mean": mean_key("delta_volatile_mean"),
        "n_positive_delta_cheap7": len(positive),
        "n_wide_family_positive": len(wide),
        "best_delta_cheap7_row": best_delta,
        "best_delta_no_GlobalPIQA_row": best_no_global,
        "windows": windows,
        "deltas": paired,
    }


def build_cross_interpretation(summaries: dict[str, Any], comparisons: dict[str, Any], reference_label: str, adapter_energy: dict[str, Any]) -> dict[str, Any]:
    ref = summaries.get(reference_label, {})
    notes: list[str] = []
    ref_best = ref.get("best_words")
    ref_width = ref.get("near_best_span_m")
    ref_drop = ref.get("chck_82M_to_100M_delta_cheap7")
    for label, s in summaries.items():
        if label == reference_label or s.get("status") not in ("ok", "has_issues"):
            continue
        c = comparisons.get(label, {})
        shift = None
        if ref_best is not None and s.get("best_words") is not None:
            shift = (int(s["best_words"]) - int(ref_best)) / 1_000_000.0
        width_delta = None
        if ref_width is not None and s.get("near_best_span_m") is not None:
            width_delta = float(s["near_best_span_m"] - ref_width)
        notes.append(
            f"{label}: best shift vs reference = {shift:+.1f}M" if shift is not None else f"{label}: best shift unavailable"
        )
        notes.append(
            f"{label}: high-band span change vs reference = {width_delta:+.1f}M" if width_delta is not None else f"{label}: high-band span change unavailable"
        )
        ref_contig = ref.get("contiguous_near_best_span_m")
        contig = s.get("contiguous_near_best_span_m")
        if ref_contig is not None and contig is not None:
            notes.append(f"{label}: connected high-band span change vs reference = {float(contig - ref_contig):+.1f}M.")
        if c.get("status") == "ok":
            notes.append(
                f"{label}: mean deltas cheap7={c.get('mean_delta_cheap7'):+.6f}, "
                f"cheap6(no GlobalPIQA)={c.get('mean_delta_cheap6_no_GlobalPIQA'):+.6f}, "
                f"EWoK/Entity={c.get('mean_delta_relation_state_mean'):+.6f}."
            )
        if label.lower().find("1p25") >= 0 or label.lower().find("scale1p25") >= 0:
            aruns = adapter_energy.get("runs", {}) if isinstance(adapter_energy, dict) else {}
            ablock = aruns.get(label, {}) if isinstance(aruns, dict) else {}
            ratio = ablock.get("ratio_to_reference_mean")
            if ratio is not None:
                notes.append(f"{label}: mean effective adapter-up/stock ratio is {ratio:.3f} of the reference mean.")
            notes.append("Read this as the seed/mask-matched lower-residual-energy contrast, not as a new data intervention.")
        if label.lower().find("43122") >= 0:
            notes.append("Read this as joint initialization plus mask-stream robustness, not as an isolated seed-only contrast.")
    if ref_drop is not None:
        notes.append(f"Reference 82M-to-100M cheap7 movement in the common grid: {ref_drop:+.6f}.")
    notes.append("A useful amplitude/timing result should move the high-score band or reduce late falloff while preserving broad family support; volatile-column-only movement remains weak evidence.")
    return {"status": "measurements_only", "notes": notes}


def fmt(x: Any, nd: int = 6) -> str:
    y = fnum(x)
    if y is None:
        return ""
    return f"{y:.{nd}f}"


def write_markdown(path: pathlib.Path, result: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# research DeBERTa common-grid trajectory interpreter\n\n")
    lines.append("This file is a fixed score-readout tool for the 70M--100M selected DeBERTa grid. It records measurements only; it does not promote any route by itself.\n\n")
    lines.append("## Trajectory summaries\n\n")
    lines.append("| label | status | best | best cheap7 | 82M cheap7 | 100M cheap7 | 82→100 Δ | final-best Δ | high-band span M | connected span M | local peak excess | column peak spread M |\n")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for label, s in result["summaries"].items():
        lpe = s.get("local_peak_excess_cheap7") if isinstance(s, dict) else None
        lines.append(
            f"| {label} | {s.get('status')} | {s.get('best_endpoint','')} | {fmt(s.get('best_cheap7'))} | "
            f"{fmt(s.get('chck_82M_cheap7'))} | {fmt(s.get('chck_100M_cheap7'))} | {fmt(s.get('chck_82M_to_100M_delta_cheap7'))} | "
            f"{fmt(s.get('final_minus_best_cheap7'))} | {fmt(s.get('near_best_span_m'), 1)} | {fmt(s.get('contiguous_near_best_span_m'), 1)} | "
            f"{fmt(lpe.get('excess') if isinstance(lpe, dict) else None)} | {fmt(s.get('column_peak_spread_m'), 1)} |\n"
        )
    lines.append("\n## Adapter energy context\n\n")
    ae = result.get("adapter_energy", {})
    if ae.get("status") == "ok":
        lines.append("| label | scale | mean effective up/stock | ratio to reference | 82M energy | 100M energy | 82→100 energy Δ | slope per 2M | n |\n")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for label, block in ae.get("runs", {}).items():
            lines.append(
                f"| {label} | {fmt(block.get('adapter_scale'), 2)} | {fmt(block.get('mean_effective_up_to_stock_ratio'))} | "
                f"{fmt(block.get('ratio_to_reference_mean'), 3)} | "
                f"{fmt((block.get('endpoint_effective_up_to_stock_ratio') or {}).get('chck_82M'))} | "
                f"{fmt((block.get('endpoint_effective_up_to_stock_ratio') or {}).get('chck_100M'))} | "
                f"{fmt(block.get('delta_82_to_100'))} | {fmt(block.get('late_70_to_100_slope_per_2M'))} | {block.get('n','')} |\n"
            )
    else:
        lines.append(f"Adapter-energy file status: `{ae.get('status')}`.\n")
    lines.append("\n## Comparisons against reference\n\n")
    lines.append("| label | common | mean Δcheap7 | mean Δcheap6(no GlobalPIQA) | mean Δcheap5(no GP/Reading) | mean ΔEWoK/Entity | mean Δvolatile | positive Δcheap7 rows | wide-family positive rows | best Δ endpoint | best Δcheap7 | volatile gain share at best |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|\n")
    for label, c in result.get("comparisons_to_reference", {}).items():
        if c.get("status") != "ok":
            lines.append(f"| {label} | 0 | | | | | | | | | | |\n")
            continue
        b = c.get("best_delta_cheap7_row", {})
        lines.append(
            f"| {label} | {c.get('n_common','')} | {fmt(c.get('mean_delta_cheap7'))} | {fmt(c.get('mean_delta_cheap6_no_GlobalPIQA'))} | "
            f"{fmt(c.get('mean_delta_cheap5_no_GlobalPIQA_Reading'))} | {fmt(c.get('mean_delta_relation_state_mean'))} | {fmt(c.get('mean_delta_volatile_mean'))} | "
            f"{c.get('n_positive_delta_cheap7','')} | {c.get('n_wide_family_positive','')} | {b.get('endpoint','')} | {fmt(b.get('delta_cheap7'))} | {fmt(b.get('positive_gain_share_volatile'), 3)} |\n"
        )
    lines.append("\n## Reading the measurements\n\n")
    for note in result.get("cross_interpretation", {}).get("notes", []):
        lines.append(f"- {note}\n")
    lines.append("\nImportant: run `selected_mlm_integrity_check.py` on each selected-evaluation output before treating these measurements as evidence. No leaderboard submission is part of this workflow.\n\n")
    lines.append(f"JSON: `{result['out_json']}`\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory", action="append", type=parse_labeled_path, required=True, help="LABEL=selected_trajectory.json")
    ap.add_argument("--reference-label", required=True)
    ap.add_argument("--adapter-json", type=pathlib.Path, default=None)
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    ap.add_argument("--expected-endpoints", nargs="+", default=EXPECTED_ENDPOINTS)
    ap.add_argument("--high-band-drop", type=float, default=0.2)
    ap.add_argument("--strict-complete", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    trajectories: dict[str, list[dict[str, Any]]] = {}
    trajectory_issues: dict[str, list[str]] = {}
    for label, path in args.trajectory:
        rows, issues = load_trajectory(path, args.expected_endpoints)
        trajectories[label] = rows
        trajectory_issues[label] = issues
    if args.reference_label not in trajectories:
        raise SystemExit(f"reference label {args.reference_label!r} not among {list(trajectories)}")

    labels = list(trajectories)
    adapter_energy = summarize_adapter_energy(args.adapter_json, labels, args.reference_label)
    summaries = {label: summarize_trajectory(label, trajectories[label], trajectory_issues[label], args.high_band_drop) for label in labels}
    ref_rows = trajectories[args.reference_label]
    comparisons = {label: summarize_comparison(label, trajectories[label], ref_rows) for label in labels if label != args.reference_label}
    cross = build_cross_interpretation(summaries, comparisons, args.reference_label, adapter_energy)
    ok_complete = all(s.get("status") == "ok" for s in summaries.values())

    out_json = args.out_dir / "deberta_common_grid_interpretation.json"
    out_md = args.out_dir / "deberta_common_grid_interpretation.md"
    result = {
        "status": "DEBERTA_COMMON_GRID_INTERPRETATION",
        "complete_and_issue_free": ok_complete,
        "reference_label": args.reference_label,
        "expected_endpoints": args.expected_endpoints,
        "high_band_drop": args.high_band_drop,
        "derived_columns": {
            "cheap6_no_GlobalPIQA": NO_GLOBAL_COLUMNS,
            "cheap5_no_GlobalPIQA_Reading": CORE_COLUMNS,
            "relation_state_mean": REL_STATE_COLUMNS,
            "syntax_comp_mean": SYNTAX_COMP_COLUMNS,
            "volatile_mean": VOLATILE_COLUMNS,
        },
        "summaries": summaries,
        "adapter_energy": adapter_energy,
        "comparisons_to_reference": comparisons,
        "cross_interpretation": cross,
        "out_json": str(out_json),
        "out_md": str(out_md),
    }
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(out_md, result)
    print(json.dumps({"status": result["status"], "complete_and_issue_free": ok_complete, "out_json": str(out_json), "out_md": str(out_md)}, indent=2), flush=True)
    if args.strict_complete and not ok_complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
