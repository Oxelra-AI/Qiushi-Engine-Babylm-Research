#!/usr/bin/env python3
"""research: paired delta anatomy for strict-innovation target probe outputs."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import math
import pathlib
import random
import statistics
import time
from collections import defaultdict
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT

ROOT = find_user_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
IN_DIR = WS / "data/strict_innovation_target_probe"
OUT_DIR = WS / "data/strict_innovation_target_delta_anatomy"
OUT_JSON = OUT_DIR / "strict_innovation_target_delta_anatomy.json"
OUT_MD = OUT_DIR / "strict_innovation_target_delta_anatomy.md"
METRICS = ["true_loss", "source_help", "same_decoy_advantage", "cross_decoy_advantage", "masked_minus_same_decoy"]


def read_rows(label: str) -> dict[int, dict[str, Any]] | None:
    path = IN_DIR / f"per_target_{label}.jsonl"
    if not path.exists():
        return None
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            out[int(obj["target_id"])] = obj
    return out


def mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if math.isfinite(x)]
    return statistics.mean(xs) if xs else None


def paired_delta(a: str, b: str, seed: int = 77017, boot: int = 500) -> dict[str, Any] | None:
    ra = read_rows(a); rb = read_rows(b)
    if ra is None or rb is None:
        return None
    ids = sorted(set(ra) & set(rb))
    if not ids:
        return None
    deltas = []
    for tid in ids:
        row = {"target_id": tid, "global_row_1based": int(ra[tid]["global_row_1based"]), "norm": ra[tid]["norm"]}
        for m in METRICS:
            row[f"delta_{m}"] = float(ra[tid][m]) - float(rb[tid][m])
        deltas.append(row)
    by_row = defaultdict(list)
    for r in deltas:
        by_row[int(r["global_row_1based"])].append(r)
    keys = list(by_row)
    rng = random.Random(seed)
    out = {"a": a, "b": b, "n_targets": len(deltas), "n_rows": len(keys), "metrics": {}}
    for m in METRICS:
        vals = [float(r[f"delta_{m}"]) for r in deltas]
        boots = []
        for _ in range(boot):
            sample_keys = [rng.choice(keys) for _ in keys]
            xs = [float(r[f"delta_{m}"]) for k in sample_keys for r in by_row[k]]
            boots.append(statistics.mean(xs))
        boots.sort()
        out["metrics"][m] = {
            "mean_delta": statistics.mean(vals),
            "median_delta": statistics.median(vals),
            "ci05": boots[int(0.05 * (len(boots)-1))],
            "ci95": boots[int(0.95 * (len(boots)-1))],
            "positive_fraction": sum(1 for x in vals if x > 0) / len(vals),
        }
    return out


def fmt(x: Any) -> str:
    try:
        y = float(x)
    except Exception:
        return "NA"
    return f"{y:.4f}" if math.isfinite(y) else "NA"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = [
        ("tokenmean_80M", "tokenmean_70M"),
        ("tokenmean_100M", "tokenmean_80M"),
        ("tokenmean_80M", "clean_80M"),
        ("reference_70M", "tokenmean_70M"),
        ("reference_80M", "tokenmean_80M"),
    ]
    comparisons = {}
    for a, b in pairs:
        res = paired_delta(a, b)
        if res is not None:
            comparisons[f"{a}_minus_{b}"] = res
    result = {
        "status": "STRICT_INNOVATION_TARGET_DELTA_ANATOMY",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Paired target-level deltas on the fixed strict-innovation sample, with row-bootstrap intervals. Negative true_loss delta means easier target prediction; positive source_help/same_decoy deltas mean stronger source-conditioned dependence.",
        "input_dir": str(IN_DIR),
        "comparisons": comparisons,
        "missing_expected_comparisons": [f"{a}_minus_{b}" for a, b in pairs if f"{a}_minus_{b}" not in comparisons],
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research strict-innovation target paired deltas", "", result["purpose"], "", "| comparison | n | rows | Δ true loss | Δ source help | Δ same decoy adv | Δ cross decoy adv | Δ masked-same |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, comp in comparisons.items():
        ms = comp["metrics"]
        lines.append(f"| {name} | {comp['n_targets']} | {comp['n_rows']} | {fmt(ms['true_loss']['mean_delta'])} [{fmt(ms['true_loss']['ci05'])},{fmt(ms['true_loss']['ci95'])}] | {fmt(ms['source_help']['mean_delta'])} [{fmt(ms['source_help']['ci05'])},{fmt(ms['source_help']['ci95'])}] | {fmt(ms['same_decoy_advantage']['mean_delta'])} [{fmt(ms['same_decoy_advantage']['ci05'])},{fmt(ms['same_decoy_advantage']['ci95'])}] | {fmt(ms['cross_decoy_advantage']['mean_delta'])} [{fmt(ms['cross_decoy_advantage']['ci05'])},{fmt(ms['cross_decoy_advantage']['ci95'])}] | {fmt(ms['masked_minus_same_decoy']['mean_delta'])} |")
    if result["missing_expected_comparisons"]:
        lines += ["", "## Missing expected comparisons"]
        lines.extend(f"- `{x}`" for x in result["missing_expected_comparisons"])
    lines += ["", "## Reading", "- Tokenmean 80M versus clean 80M quantifies how much the compact-view reinvest substrate already improves this exact target object.", "- Tokenmean 80M->100M quantifies natural late-exposure movement on the same target sample, useful as a reference for research target deltas.", "- research deltas are absent until managed checkpoints are delivered and the target probe is rerun including research labels.", "", f"Full JSON: `{OUT_JSON}`"]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "out_json": str(OUT_JSON), "out_md": str(OUT_MD), "comparisons": list(comparisons), "missing": result["missing_expected_comparisons"]}, indent=2), flush=True)

if __name__ == "__main__":
    main()
