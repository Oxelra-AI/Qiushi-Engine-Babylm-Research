#!/usr/bin/env python3
"""research: compare init-matched minfreq50 screen against research token-mean reinvest.

Cheap official-compatible columns only at 70M/80M.  This comparison answers
whether the support-floored vocabulary/segmentation point improves over the fully
legal research 16k compact-view reinvest trajectory while preserving GlobalPIQA and
Entity better than the legal40k package.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import time
from typing import Any

STUDY = pathlib.Path("experiments/archive/frontier_consolidation")
DEFAULT_OUT = STUDY / "data/minfreq50_initmatched_trajectory"
MINFREQ_OUT = STUDY / "data/minfreq50_initmatched_70_80M_eval/per_target"
TOKENMEAN_REFS = STUDY / "data/legal_mature_treatment_effect_eval/per_target"
CLEAN_REFS = STUDY / "data/legal_mature_clean_control_eval/per_target"
FACTOR_AUDIT = (STUDY.parents[2] / 'research/documents/frontier_consolidation/data/supportfloor_factor_audit/supportfloor_factor_audit.md')
COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
RAW_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"]


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_payload(path: pathlib.Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def score(payload: dict[str, Any] | None, column: str) -> float | None:
    if not isinstance(payload, dict):
        return None
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


def finite_mean(vals: list[float | None]) -> float | None:
    xs = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if len(xs) != len(vals) or not xs:
        return None
    return sum(xs) / len(xs)


def fmt(x: Any) -> str:
    return "NA" if x is None else f"{float(x):.4f}"


def paths_for(minfreq_dir: pathlib.Path, token_dir: pathlib.Path, clean_dir: pathlib.Path, m: int) -> dict[str, pathlib.Path]:
    return {
        "minfreq50": minfreq_dir / f"minfreq50_initmatched_seed43022_{m}M.json",
        "reinvest": token_dir / f"complianttok_reinvest_seed43022_{m}M.json",
        "clean_control": clean_dir / f"complianttok_cleanqwen_seed43022_{m}M.json",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minfreq-per-target", default=str(MINFREQ_OUT))
    ap.add_argument("--tokenmean-ref-per-target", default=str(TOKENMEAN_REFS))
    ap.add_argument("--clean-ref-per-target", default=str(CLEAN_REFS))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--exposures", type=int, nargs="*", default=[70, 80])
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    min_dir = pathlib.Path(args.minfreq_per_target)
    tok_dir = pathlib.Path(args.tokenmean_ref_per_target)
    clean_dir = pathlib.Path(args.clean_ref_per_target)

    rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for m in args.exposures:
        ps = paths_for(min_dir, tok_dir, clean_dir, int(m))
        payloads = {k: load_payload(v) for k, v in ps.items()}
        missing.extend([f"{m}M:{k}:{v}" for k, v in ps.items() if payloads[k] is None])
        row: dict[str, Any] = {"exposure_m": int(m), "paths": {k: str(v) for k, v in ps.items()}, "complete": all(payloads.values())}
        for col in COLUMNS:
            a = score(payloads["minfreq50"], col)
            b = score(payloads["reinvest"], col)
            c = score(payloads["clean_control"], col)
            row[f"minfreq50_{col}"] = a
            row[f"reinvest_{col}"] = b
            row[f"clean_{col}"] = c
            row[f"delta_minfreq50_minus_step35_{col}"] = (a - b) if a is not None and b is not None else None
            row[f"delta_minfreq50_minus_clean_{col}"] = (a - c) if a is not None and c is not None else None
        row["minfreq50_mean7"] = finite_mean([row[f"minfreq50_{c}"] for c in COLUMNS])
        row["reinvest_mean7"] = finite_mean([row[f"reinvest_{c}"] for c in COLUMNS])
        row["clean_mean7"] = finite_mean([row[f"clean_{c}"] for c in COLUMNS])
        row["delta_minfreq50_minus_step35_mean7"] = (row["minfreq50_mean7"] - row["reinvest_mean7"]) if row["minfreq50_mean7"] is not None and row["reinvest_mean7"] is not None else None
        row["delta_minfreq50_minus_clean_mean7"] = (row["minfreq50_mean7"] - row["clean_mean7"]) if row["minfreq50_mean7"] is not None and row["clean_mean7"] is not None else None
        rows.append(row)
        raw: dict[str, Any] = {"exposure_m": int(m), "complete": all(payloads.values())}
        for col in RAW_COLUMNS:
            a = score(payloads["minfreq50"], col)
            b = score(payloads["reinvest"], col)
            raw[f"delta_minfreq50_minus_step35_{col}"] = (a - b) if a is not None and b is not None else None
        raw["delta_minfreq50_minus_step35_mean8_raw"] = finite_mean([raw[f"delta_minfreq50_minus_step35_{c}"] for c in RAW_COLUMNS])
        raw_rows.append(raw)

    summary = {
        "status": "MINFREQ50_INITMATCHED_TRAJECTORY_COMPARISON",
        "created_utc": now_utc(),
        "scope": "cheap official-compatible zero-shot/reading columns only at 70M/80M; no SuperGLUE or AoA; not a complete endpoint judgment",
        "scientific_question": "Does a support-floored minfreq50 legal tokenizer, with same-shape contextual initialization matched to research, improve compact-view reinvestment enough to justify continuation?",
        "factor_audit_md": str(FACTOR_AUDIT),
        "decision_reading": "Continue toward 100M/full only if the mature cheap surface is broadly positive vs research reinvest, improves Supplement/EWoK/BLiMP without the legal40k-style GlobalPIQA/Entity collapse, and is plausibly large enough to close the SOTA gap after SuperGLUE/AoA.",
        "rows": rows,
        "raw_rows": raw_rows,
        "missing": missing,
    }
    out_json = out_dir / "minfreq50_vs_trajectory.json"
    out_md = out_dir / "minfreq50_vs_trajectory.md"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research init-matched minfreq50 trajectory comparison",
        "",
        summary["scope"],
        "",
        summary["scientific_question"],
        "",
        f"Factor audit: `{FACTOR_AUDIT}`",
        "",
        "| exposure | complete | minfreq50 mean7 | research reinvest mean7 | clean mean7 | Δ minfreq-research | Δ minfreq-clean | BLiMP Δ | Supp | EWoK | Entity | COMPS | GPIQA | Reading |",
        "|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {m} | {complete} | {a} | {b} | {c} | {dab} | {dac} | {bl} | {su} | {ew} | {en} | {co} | {gp} | {rd} |".format(
                m=row["exposure_m"], complete="yes" if row["complete"] else "no",
                a=fmt(row["minfreq50_mean7"]), b=fmt(row["reinvest_mean7"]), c=fmt(row["clean_mean7"]),
                dab=fmt(row["delta_minfreq50_minus_step35_mean7"]), dac=fmt(row["delta_minfreq50_minus_clean_mean7"]),
                bl=fmt(row["delta_minfreq50_minus_step35_BLiMP"]), su=fmt(row["delta_minfreq50_minus_step35_Supplement"]),
                ew=fmt(row["delta_minfreq50_minus_step35_EWoK"]), en=fmt(row["delta_minfreq50_minus_step35_Entity"]),
                co=fmt(row["delta_minfreq50_minus_step35_COMPS"]), gp=fmt(row["delta_minfreq50_minus_step35_GlobalPIQA"]),
                rd=fmt(row["delta_minfreq50_minus_step35_Reading"]),
            )
        )
    lines += ["", "## Missing inputs", ""]
    lines += [f"- `{m}`" for m in missing] if missing else ["- none"]
    lines += ["", f"Full JSON: `{out_json}`"]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "out_json": str(out_json), "out_md": str(out_md), "missing": missing}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
