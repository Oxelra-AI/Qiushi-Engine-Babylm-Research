#!/usr/bin/env python3
"""research: compare 100M seed-spread fingerprints in reinvest vs clean-Qwen.

Inputs:
- Reinvest seed43022/43122 fast localization JSON from research.
- Clean-Qwen seed43022/43122 full report txt files from COMPACT_EXPERIENCE trajectory screen.

This CPU-only comparison does not replace the matched sparse temporal probe, but it
separates inherited base-recipe spread from reinvest-specific amplification for
subtasks with shared labels.
"""
from __future__ import annotations

import json
import pathlib
import re
import statistics
from typing import Any

ROOT = pathlib.Path.cwd()
STUDY = ROOT / "experiments/archive" / 'frontier_consolidation'
OUT_DIR = STUDY / "data" / "reinvest_vs_clean_seed_fingerprint"
OUT_JSON = OUT_DIR / "reinvest_vs_clean_seed_fingerprint.json"
OUT_MD = STUDY / "notes" / "reinvest_vs_clean_seed_fingerprint.md"

REINVEST_LOC = STUDY / "data" / "seed43122_fast_localization" / "seed43122_fast_localization.json"
CLEAN_BASE = ROOT / "experiments/archive" / 'compact_experience' / "data" / "trajectory_screen" / "official_outputs"
CLEAN_REPORTS = {
    "43022": {
        "BLiMP": CLEAN_BASE / "clean_qwen_seed43022" / "chck_100M" / "BLiMP" / "chck_100M" / "clean_qwen_seed43022_chck_100M_BLiMP" / "zero_shot" / "mlm" / "blimp" / "blimp_filtered" / "best_temperature_report.txt",
        "Supplement": CLEAN_BASE / "clean_qwen_seed43022" / "chck_100M" / "Supplement" / "chck_100M" / "clean_qwen_seed43022_chck_100M_Supplement" / "zero_shot" / "mlm" / "blimp" / "supplement_filtered" / "best_temperature_report.txt",
        "EWoK": CLEAN_BASE / "clean_qwen_seed43022" / "chck_100M" / "EWoK" / "chck_100M" / "clean_qwen_seed43022_chck_100M_EWoK" / "zero_shot" / "mlm" / "ewok" / "ewok_filtered" / "best_temperature_report.txt",
        "Entity": CLEAN_BASE / "clean_qwen_seed43022" / "chck_100M" / "Entity" / "chck_100M" / "clean_qwen_seed43022_chck_100M_Entity" / "zero_shot" / "mlm" / "entity_tracking" / "entity_tracking" / "best_temperature_report.txt",
    },
    "43122": {
        "BLiMP": CLEAN_BASE / "clean_qwen_seed43122" / "chck_100M" / "BLiMP" / "chck_100M" / "clean_qwen_seed43122_chck_100M_BLiMP" / "zero_shot" / "mlm" / "blimp" / "blimp_filtered" / "best_temperature_report.txt",
        "Supplement": CLEAN_BASE / "clean_qwen_seed43122" / "chck_100M" / "Supplement" / "chck_100M" / "clean_qwen_seed43122_chck_100M_Supplement" / "zero_shot" / "mlm" / "blimp" / "supplement_filtered" / "best_temperature_report.txt",
        "EWoK": CLEAN_BASE / "clean_qwen_seed43122" / "chck_100M" / "EWoK" / "chck_100M" / "clean_qwen_seed43122_chck_100M_EWoK" / "zero_shot" / "mlm" / "ewok" / "ewok_filtered" / "best_temperature_report.txt",
        "Entity": CLEAN_BASE / "clean_qwen_seed43122" / "chck_100M" / "Entity" / "chck_100M" / "clean_qwen_seed43122_chck_100M_Entity" / "zero_shot" / "mlm" / "entity_tracking" / "entity_tracking" / "best_temperature_report.txt",
    },
}


def parse_uid_report(path: pathlib.Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8")
    out: dict[str, float] = {}
    in_uid = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("### UID ACCURACY"):
            in_uid = True
            continue
        if in_uid and line.startswith("### "):
            break
        if not in_uid or not line or ":" not in line:
            continue
        key, val = line.split(":", 1)
        key = key.strip()
        val = val.strip()
        try:
            out[key] = float(val)
        except ValueError:
            continue
    return out


def extract_reinvest_deltas(loc: dict[str, Any]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for col, cdata in loc["columns"].items():
        rows = cdata.get("subtasks_sorted_by_delta", [])
        if not rows:
            continue
        out[col] = {str(r["subtask"]): float(r["delta"]) for r in rows if "subtask" in r and "delta" in r}
    return out


def clean_deltas() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for col in CLEAN_REPORTS["43022"]:
        a = parse_uid_report(CLEAN_REPORTS["43022"][col])
        b = parse_uid_report(CLEAN_REPORTS["43122"][col])
        common = sorted(set(a) & set(b))
        out[col] = {k: b[k] - a[k] for k in common}
    return out


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    mx = statistics.mean(xs); my = statistics.mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (vx ** 0.5 * vy ** 0.5)


def sign(x: float) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


def summarize_shared(reinv: dict[str, dict[str, float]], clean: dict[str, dict[str, float]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col in sorted(set(reinv) & set(clean)):
        common = sorted(set(reinv[col]) & set(clean[col]))
        pairs = []
        for k in common:
            rv = reinv[col][k]
            cv = clean[col][k]
            pairs.append({
                "subtask": k,
                "reinvest_delta_43122_minus_43022": rv,
                "clean_delta_43122_minus_43022": cv,
                "excess_reinvest_minus_clean": rv - cv,
                "same_sign": sign(rv) == sign(cv),
            })
        xs = [p["clean_delta_43122_minus_43022"] for p in pairs]
        ys = [p["reinvest_delta_43122_minus_43022"] for p in pairs]
        excess = [p["excess_reinvest_minus_clean"] for p in pairs]
        out[col] = {
            "n_shared_subtasks": len(pairs),
            "mean_clean_delta": statistics.mean(xs) if xs else None,
            "mean_reinvest_delta": statistics.mean(ys) if ys else None,
            "mean_excess_reinvest_minus_clean": statistics.mean(excess) if excess else None,
            "median_excess_reinvest_minus_clean": statistics.median(excess) if excess else None,
            "pearson_clean_vs_reinvest_deltas": pearson(xs, ys),
            "same_sign_fraction": sum(1 for p in pairs if p["same_sign"]) / len(pairs) if pairs else None,
            "top_reinvest_lower_than_clean": sorted(pairs, key=lambda p: p["excess_reinvest_minus_clean"])[:12],
            "top_reinvest_higher_than_clean": sorted(pairs, key=lambda p: p["excess_reinvest_minus_clean"], reverse=True)[:12],
            "pairs_sorted_by_reinvest_delta": sorted(pairs, key=lambda p: p["reinvest_delta_43122_minus_43022"]),
        }
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    loc = json.loads(REINVEST_LOC.read_text(encoding="utf-8"))
    reinv = extract_reinvest_deltas(loc)
    clean = clean_deltas()
    shared = summarize_shared(reinv, clean)
    result = {
        "status": "REINVEST_VS_CLEAN_SEED_FINGERPRINT",
        "purpose": "CPU comparison of 100M seed-spread subtask fingerprints using existing reinvest fast localization and clean-Qwen full reports.",
        "reinvest_source": str(REINVEST_LOC),
        "clean_report_sources": {seed: {col: str(path) for col, path in cols.items()} for seed, cols in CLEAN_REPORTS.items()},
        "shared_subtask_comparison": shared,
        "interpretation": {
            "not_a_temporal_match": "This compares 100M subtask fingerprints only; the running matched sparse temporal probe is needed for the direct trajectory control.",
            "use": "Large negative excess values identify subtasks where seed43122 falls further behind seed43022 under reinvest than under clean-Qwen; similar signs and magnitudes identify inherited base seed spread.",
        },
    }
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines: list[str] = []
    lines.append("# research reinvest vs clean 100M seed-spread fingerprint\n\n")
    lines.append("CPU-only comparison from existing reports. It is a 100M fingerprint, not the matched sparse temporal control.\n\n")
    for col, rec in shared.items():
        lines.append(f"## {col}\n")
        lines.append(f"- shared subtasks: {rec['n_shared_subtasks']}\n")
        lines.append(f"- mean clean delta: {rec['mean_clean_delta']:.3f}\n")
        lines.append(f"- mean reinvest delta: {rec['mean_reinvest_delta']:.3f}\n")
        lines.append(f"- mean excess reinvest-clean: {rec['mean_excess_reinvest_minus_clean']:.3f}\n")
        corr = rec['pearson_clean_vs_reinvest_deltas']
        lines.append(f"- delta correlation clean vs reinvest: {corr:.3f}\n" if corr is not None else "- delta correlation clean vs reinvest: None\n")
        lines.append(f"- same-sign fraction: {rec['same_sign_fraction']:.3f}\n")
        lines.append("- largest negative excess: " + "; ".join(f"{p['subtask']} r={p['reinvest_delta_43122_minus_43022']:.1f} c={p['clean_delta_43122_minus_43022']:.1f} ex={p['excess_reinvest_minus_clean']:.1f}" for p in rec['top_reinvest_lower_than_clean'][:5]) + "\n")
        lines.append("- largest positive excess: " + "; ".join(f"{p['subtask']} r={p['reinvest_delta_43122_minus_43022']:.1f} c={p['clean_delta_43122_minus_43022']:.1f} ex={p['excess_reinvest_minus_clean']:.1f}" for p in rec['top_reinvest_higher_than_clean'][:5]) + "\n\n")
    lines.append("Interpretation: where signs/magnitudes match clean-Qwen, the seed spread is likely inherited from the base recipe. Where reinvest has much more negative excess, the compact-view reinvestment route may be amplifying or changing instability. The running matched sparse trajectories are still needed for a time-resolved answer.\n\n")
    lines.append(f"Machine-readable output: `{OUT_JSON}`\n")
    OUT_MD.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": result["status"], "out_json": str(OUT_JSON), "out_md": str(OUT_MD)}, indent=2))


if __name__ == "__main__":
    main()
