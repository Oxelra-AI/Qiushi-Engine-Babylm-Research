#!/usr/bin/env python3
"""Compare legal-tokenizer reinvest vs clean-Qwen cheap-column trajectories.

This is a CPU-only summarizer.  It expects per-target JSON files produced by
evaluate_compliant_endpoint.py for the research-tokenizer reinvest and
clean-Qwen controls at matched endpoints.  It writes a compact JSON/Markdown table
of per-column treatment deltas and a seven/eight-column average so later agents
can decide whether compact-view reinvestment survives the legal tokenizer
coordinate without relying on endpoint score deficits alone.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import time
from typing import Any


STUDY = pathlib.Path("experiments/archive/frontier_consolidation")
DEFAULT_OUT = STUDY / "data/legal_treatment_trajectory"
DEFAULT_FILES = {
    "reinvest_20M": STUDY / "data/legal20m_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_20M.json",
    "clean_20M": STUDY / "data/legal_mature_clean_control_eval/per_target/complianttok_cleanqwen_seed43022_20M.json",
    "reinvest_70M": STUDY / "data/legal_mature_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_70M.json",
    "clean_70M": STUDY / "data/legal_mature_clean_control_eval/per_target/complianttok_cleanqwen_seed43022_70M.json",
    "reinvest_80M": STUDY / "data/legal_mature_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_80M.json",
    "clean_80M": STUDY / "data/legal_mature_clean_control_eval/per_target/complianttok_cleanqwen_seed43022_80M.json",
}
COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
RAW_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"]


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_payload(path: pathlib.Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def task_score(payload: dict[str, Any], column: str) -> float | None:
    tasks = payload.get("tasks", {})
    if column == "GlobalPIQA":
        vals = []
        for c in ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]:
            rec = tasks.get(c)
            if not isinstance(rec, dict) or rec.get("score") is None:
                return None
            vals.append(float(rec["score"]))
        return sum(vals) / len(vals)
    if column == "Reading":
        rec = tasks.get("Reading")
        if not isinstance(rec, dict):
            return None
        scores = rec.get("scores")
        if isinstance(scores, dict) and scores.get("Reading") is not None:
            return float(scores["Reading"])
        return float(rec["score"]) if rec.get("score") is not None else None
    rec = tasks.get(column)
    if not isinstance(rec, dict):
        return None
    return float(rec["score"]) if rec.get("score") is not None else None


def raw_score(payload: dict[str, Any], column: str) -> float | None:
    if column == "Reading":
        return task_score(payload, "Reading")
    rec = payload.get("tasks", {}).get(column)
    if not isinstance(rec, dict):
        return None
    return float(rec["score"]) if rec.get("score") is not None else None


def finite_mean(vals: list[float | None]) -> float | None:
    xs = [v for v in vals if v is not None and math.isfinite(v)]
    if len(xs) != len(vals) or not xs:
        return None
    return sum(xs) / len(xs)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    for key, path in DEFAULT_FILES.items():
        ap.add_argument(f"--{key.replace('_', '-')}", default=str(path))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {key: pathlib.Path(getattr(args, key)) for key in DEFAULT_FILES}
    payloads = {key: load_payload(path) for key, path in paths.items()}

    exposures = [20, 70, 80]
    rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    for m in exposures:
        r = payloads.get(f"reinvest_{m}M")
        c = payloads.get(f"clean_{m}M")
        row: dict[str, Any] = {"exposure_m": m, "reinvest_path": str(paths[f"reinvest_{m}M"]), "clean_path": str(paths[f"clean_{m}M"]), "complete_pair": r is not None and c is not None}
        for col in COLUMNS:
            rv = task_score(r, col) if r else None
            cv = task_score(c, col) if c else None
            row[f"reinvest_{col}"] = rv
            row[f"clean_{col}"] = cv
            row[f"delta_{col}"] = (rv - cv) if rv is not None and cv is not None else None
        row["reinvest_mean7"] = finite_mean([row[f"reinvest_{col}"] for col in COLUMNS])
        row["clean_mean7"] = finite_mean([row[f"clean_{col}"] for col in COLUMNS])
        row["delta_mean7"] = (row["reinvest_mean7"] - row["clean_mean7"]) if row["reinvest_mean7"] is not None and row["clean_mean7"] is not None else None
        rows.append(row)

        raw_row: dict[str, Any] = {"exposure_m": m, "complete_pair": r is not None and c is not None}
        for col in RAW_COLUMNS:
            rv = raw_score(r, col) if r else None
            cv = raw_score(c, col) if c else None
            raw_row[f"reinvest_{col}"] = rv
            raw_row[f"clean_{col}"] = cv
            raw_row[f"delta_{col}"] = (rv - cv) if rv is not None and cv is not None else None
        raw_row["delta_mean8_raw"] = finite_mean([raw_row[f"delta_{col}"] for col in RAW_COLUMNS])
        raw_rows.append(raw_row)

    summary = {
        "status": "LEGAL_TREATMENT_TRAJECTORY_COMPARISON",
        "created_utc": now_utc(),
        "interpretation_scope": "cheap official-compatible zero-shot/reading columns only; no SuperGLUE or AoA; clean_qwen is a fixed-reinvest-tokenizer scientific control and not an independently budget-valid submission endpoint",
        "paths": {k: str(v) for k, v in paths.items()},
        "rows": rows,
        "raw_rows": raw_rows,
        "missing": [k for k, v in payloads.items() if v is None],
    }
    out_json = out_dir / "legal_treatment_trajectory_comparison.json"
    out_md = out_dir / "legal_treatment_trajectory_comparison.md"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["# research legal-tokenizer treatment trajectory", "", summary["interpretation_scope"], "", "## Matched mean and column deltas", "", "| exposure | pair complete | reinvest mean7 | clean mean7 | delta mean7 | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | Reading |", "|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    def fmt(x: Any) -> str:
        return "NA" if x is None else f"{float(x):.4f}"
    for row in rows:
        lines.append("| {exposure_m} | {complete_pair} | {rm} | {cm} | {dm} | {b} | {s} | {e} | {ent} | {comp} | {g} | {r} |".format(
            exposure_m=row["exposure_m"],
            complete_pair="yes" if row["complete_pair"] else "no",
            rm=fmt(row["reinvest_mean7"]),
            cm=fmt(row["clean_mean7"]),
            dm=fmt(row["delta_mean7"]),
            b=fmt(row["delta_BLiMP"]),
            s=fmt(row["delta_Supplement"]),
            e=fmt(row["delta_EWoK"]),
            ent=fmt(row["delta_Entity"]),
            comp=fmt(row["delta_COMPS"]),
            g=fmt(row["delta_GlobalPIQA"]),
            r=fmt(row["delta_Reading"]),
        ))
    lines += ["", "## Input files", ""]
    for key, path in paths.items():
        lines.append(f"- `{key}`: `{path}` exists={path.exists()}")
    lines += ["", f"Full JSON: `{out_json}`"]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "out_json": str(out_json), "out_md": str(out_md), "missing": summary["missing"]}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
