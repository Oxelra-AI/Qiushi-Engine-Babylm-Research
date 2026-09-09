#!/usr/bin/env python3
"""research: reconstruct a selected DeBERTa cheap7 trajectory from per-target payloads.

Use this after a selected-eval bg task terminates or times out. The selected wrapper
writes each endpoint payload under OUT_DIR/eval/per_target/<label>_<endpoint>.json
before it writes the final selected_trajectory files. This script recovers those
already completed official-compatible evaluations, reports missing endpoints, and
writes selected_trajectory.{json,csv} plus a summary. It performs no model eval.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
from statistics import mean
from typing import Any

CHEAP_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
ZERO_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]
GP_COLS = ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]
EXPECTED_COMMON2M = [
    "chck_70M", "chck_72M", "chck_74M", "chck_76M", "chck_78M", "chck_80M",
    "chck_82M", "chck_84M", "chck_86M", "chck_88M", "chck_90M", "chck_92M",
    "chck_94M", "chck_96M", "chck_98M", "chck_100M",
]


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def endpoint_words(endpoint: str) -> int:
    if not (endpoint.startswith("chck_") and endpoint.endswith("M")):
        raise ValueError(endpoint)
    return int(endpoint[len("chck_"):-1]) * 1_000_000


def extract_scores(payload: dict[str, Any]) -> dict[str, float | None]:
    tasks = payload.get("tasks", {}) if isinstance(payload, dict) else {}
    out: dict[str, float | None] = {}
    for c in ZERO_COLUMNS:
        rec = tasks.get(c, {}) if isinstance(tasks, dict) else {}
        out[c] = fnum(rec.get("score")) if isinstance(rec, dict) else None
    gp_vals: list[float] = []
    for c in GP_COLS:
        rec = tasks.get(c, {}) if isinstance(tasks, dict) else {}
        val = fnum(rec.get("score")) if isinstance(rec, dict) else None
        if val is not None:
            gp_vals.append(val)
    out["GlobalPIQA"] = float(mean(gp_vals)) if len(gp_vals) == 2 else None
    rd = tasks.get("Reading", {}) if isinstance(tasks, dict) else {}
    if isinstance(rd, dict) and isinstance(rd.get("scores"), dict) and rd["scores"].get("Reading") is not None:
        out["Reading"] = fnum(rd["scores"].get("Reading"))
    elif isinstance(rd, dict):
        out["Reading"] = fnum(rd.get("score"))
    else:
        out["Reading"] = None
    return out


def cheap7(scores: dict[str, float | None]) -> float | None:
    vals = [scores.get(c) for c in CHEAP_COLUMNS]
    if any(v is None for v in vals):
        return None
    return float(mean(float(v) for v in vals))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=pathlib.Path, required=True, help="Selected evaluator output directory")
    ap.add_argument("--label", required=True)
    ap.add_argument("--endpoints", nargs="+", default=EXPECTED_COMMON2M)
    ap.add_argument("--write", action="store_true", help="Write/overwrite selected_trajectory files")
    args = ap.parse_args()

    per_target = args.out_dir / "eval" / "per_target"
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    incomplete: list[dict[str, Any]] = []
    for ep in args.endpoints:
        target = f"{args.label}_{ep}"
        path = per_target / f"{target}.json"
        if not path.exists():
            missing.append(ep)
            rows.append({"words": endpoint_words(ep), "endpoint": ep, "target": target, "error": "missing_per_target", "payload_path": str(path)})
            continue
        payload = read_json(path)
        scores = extract_scores(payload)
        c7 = cheap7(scores)
        row = {"words": endpoint_words(ep), "endpoint": ep, "target": target, **scores, "cheap7": c7, "payload_path": str(path)}
        if c7 is None:
            row["error"] = "missing_one_or_more_scores"
            incomplete.append({"endpoint": ep, "scores": scores, "payload_path": str(path)})
        rows.append(row)

    valid = [r for r in rows if r.get("cheap7") is not None]
    best = max(valid, key=lambda r: float(r["cheap7"])) if valid else None
    summary = {
        "status": "RECONSTRUCT_SELECTED_MLM_TRAJECTORY",
        "out_dir": str(args.out_dir),
        "label": args.label,
        "endpoints_requested": args.endpoints,
        "n_requested": len(args.endpoints),
        "n_valid": len(valid),
        "n_missing": len(missing),
        "n_incomplete": len(incomplete),
        "missing_endpoints": missing,
        "incomplete_endpoints": incomplete,
        "complete": len(valid) == len(args.endpoints) and not incomplete,
        "best_endpoint": best.get("endpoint") if best else None,
        "best_words": best.get("words") if best else None,
        "best_cheap7": best.get("cheap7") if best else None,
        "best_scores": {c: best.get(c) for c in CHEAP_COLUMNS} if best else None,
        "trajectory_json": str(args.out_dir / "selected_trajectory.json"),
        "trajectory_csv": str(args.out_dir / "selected_trajectory.csv"),
        "summary_json": str(args.out_dir / "selected_trajectory_summary.json"),
    }

    if args.write:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "selected_trajectory.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        with (args.out_dir / "selected_trajectory.csv").open("w", encoding="utf-8", newline="") as f:
            cols = ["words", "endpoint"] + CHEAP_COLUMNS + ["cheap7", "target", "payload_path", "error"]
            wr = csv.DictWriter(f, fieldnames=cols)
            wr.writeheader()
            for r in rows:
                wr.writerow({c: r.get(c, "") for c in cols})
        (args.out_dir / "selected_trajectory_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        md = [f"# Reconstructed selected MLM trajectory: {args.label}\n\n"]
        md.append(f"Valid {len(valid)}/{len(args.endpoints)}; missing {len(missing)}; incomplete {len(incomplete)}.\n\n")
        if best:
            md.append(f"Best recovered endpoint: **{best['endpoint']}** cheap7={best['cheap7']:.6f}.\n\n")
        md.append("| endpoint | cheap7 | BLiMP | Supplement | EWoK | Entity | COMPS | GlobalPIQA | Reading |\n")
        md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            md.append("| " + str(r.get("endpoint")) + " | " + " | ".join(
                f"{float(r[c]):.6f}" if r.get(c) is not None else "" for c in ["cheap7"] + CHEAP_COLUMNS
            ) + " |\n")
        md.append(f"\nJSON: `{summary['summary_json']}`\n")
        (args.out_dir / "selected_trajectory_summary.md").write_text("".join(md), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if not summary["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
