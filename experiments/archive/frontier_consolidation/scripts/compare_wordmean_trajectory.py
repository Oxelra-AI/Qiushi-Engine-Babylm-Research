#!/usr/bin/env python3
"""research: compare word-mean MLM screen against token-mean legal reinvest references.

The comparison is intentionally cheap-column only: BLiMP, Supplement, EWoK,
Entity, COMPS, GlobalPIQA_parallel/nonparallel averaged as GlobalPIQA, and
Reading at chck_70M and chck_80M.  It is used to decide whether the research
whole-word-unit loss-normalization screen should continue to a full 100M endpoint
or stop.  It does not infer SuperGLUE/AoA and it does not interpret an endpoint as
complete SOTA evidence.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import time
from typing import Any


STUDY = pathlib.Path("experiments/archive/frontier_consolidation")
DEFAULT_OUT = STUDY / "data/wordmean_trajectory"
WORDMEAN_OUT = STUDY / "data/wordmean_70_80M_eval/per_target"
TOKENMEAN_REFS = STUDY / "data/legal_mature_treatment_effect_eval/per_target"
CLEAN_REFS = STUDY / "data/legal_mature_clean_control_eval/per_target"
SCALE_NOTE = (STUDY.parents[2] / 'research/documents/frontier_consolidation/data/wordmean_credit_scale_analysis/wordmean_credit_scale_analysis.md')
COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
RAW_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"]


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_payload(path: pathlib.Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def task_score(payload: dict[str, Any] | None, column: str) -> float | None:
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


def raw_score(payload: dict[str, Any] | None, column: str) -> float | None:
    if column == "Reading":
        return task_score(payload, "Reading")
    if not isinstance(payload, dict):
        return None
    rec = payload.get("tasks", {}).get(column)
    if not isinstance(rec, dict):
        return None
    return float(rec["score"]) if rec.get("score") is not None else None


def finite_mean(vals: list[float | None]) -> float | None:
    xs = [v for v in vals if v is not None and math.isfinite(float(v))]
    if len(xs) != len(vals) or not xs:
        return None
    return sum(float(v) for v in xs) / len(xs)


def fmt(x: Any) -> str:
    return "NA" if x is None else f"{float(x):.4f}"


def payload_paths(wordmean_dir: pathlib.Path, tokenmean_dir: pathlib.Path, clean_dir: pathlib.Path, m: int) -> dict[str, pathlib.Path]:
    return {
        "wordmean": wordmean_dir / f"wordmean_mlm_seed43022_{m}M.json",
        "tokenmean_reinvest": tokenmean_dir / f"complianttok_reinvest_seed43022_{m}M.json",
        "clean_control": clean_dir / f"complianttok_cleanqwen_seed43022_{m}M.json",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wordmean-per-target", default=str(WORDMEAN_OUT))
    ap.add_argument("--tokenmean-ref-per-target", default=str(TOKENMEAN_REFS))
    ap.add_argument("--clean-ref-per-target", default=str(CLEAN_REFS))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--exposures", type=int, nargs="*", default=[70, 80])
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wordmean_dir = pathlib.Path(args.wordmean_per_target)
    tokenmean_dir = pathlib.Path(args.tokenmean_ref_per_target)
    clean_dir = pathlib.Path(args.clean_ref_per_target)

    rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for m in args.exposures:
        paths = payload_paths(wordmean_dir, tokenmean_dir, clean_dir, int(m))
        payloads = {k: load_payload(v) for k, v in paths.items()}
        missing.extend([f"{m}M:{k}:{v}" for k, v in paths.items() if payloads[k] is None])
        row: dict[str, Any] = {"exposure_m": int(m), "paths": {k: str(v) for k, v in paths.items()}, "complete": all(payloads.values())}
        for col in COLUMNS:
            w = task_score(payloads["wordmean"], col)
            t = task_score(payloads["tokenmean_reinvest"], col)
            c = task_score(payloads["clean_control"], col)
            row[f"wordmean_{col}"] = w
            row[f"tokenmean_{col}"] = t
            row[f"clean_{col}"] = c
            row[f"delta_wordmean_minus_tokenmean_{col}"] = (w - t) if w is not None and t is not None else None
            row[f"delta_wordmean_minus_clean_{col}"] = (w - c) if w is not None and c is not None else None
            row[f"delta_tokenmean_minus_clean_{col}"] = (t - c) if t is not None and c is not None else None
        row["wordmean_mean7"] = finite_mean([row[f"wordmean_{c}"] for c in COLUMNS])
        row["tokenmean_mean7"] = finite_mean([row[f"tokenmean_{c}"] for c in COLUMNS])
        row["clean_mean7"] = finite_mean([row[f"clean_{c}"] for c in COLUMNS])
        row["delta_wordmean_minus_tokenmean_mean7"] = (row["wordmean_mean7"] - row["tokenmean_mean7"]) if row["wordmean_mean7"] is not None and row["tokenmean_mean7"] is not None else None
        row["delta_wordmean_minus_clean_mean7"] = (row["wordmean_mean7"] - row["clean_mean7"]) if row["wordmean_mean7"] is not None and row["clean_mean7"] is not None else None
        row["delta_tokenmean_minus_clean_mean7"] = (row["tokenmean_mean7"] - row["clean_mean7"]) if row["tokenmean_mean7"] is not None and row["clean_mean7"] is not None else None
        rows.append(row)

        raw: dict[str, Any] = {"exposure_m": int(m), "complete": all(payloads.values())}
        for col in RAW_COLUMNS:
            w = raw_score(payloads["wordmean"], col)
            t = raw_score(payloads["tokenmean_reinvest"], col)
            c = raw_score(payloads["clean_control"], col)
            raw[f"delta_wordmean_minus_tokenmean_{col}"] = (w - t) if w is not None and t is not None else None
            raw[f"delta_wordmean_minus_clean_{col}"] = (w - c) if w is not None and c is not None else None
        raw["delta_wordmean_minus_tokenmean_mean8_raw"] = finite_mean([raw[f"delta_wordmean_minus_tokenmean_{c}"] for c in RAW_COLUMNS])
        raw["delta_wordmean_minus_clean_mean8_raw"] = finite_mean([raw[f"delta_wordmean_minus_clean_{c}"] for c in RAW_COLUMNS])
        raw_rows.append(raw)

    summary = {
        "status": "WORDMEAN_TRAJECTORY_COMPARISON",
        "created_utc": now_utc(),
        "scope": "cheap official-compatible zero-shot/reading columns only at 70M/80M; no SuperGLUE or AoA; not a complete endpoint judgment",
        "scientific_question": "Does selected-whole-word-group mean MLM strengthen the broad late-emerging compact-view reinvestment benefit relative to the fixed-tokenizer token-mean legal reinvest trajectory?",
        "scale_caution": "research credit-scale analysis shows word-mean also changes logit-level gradient weight RMS; unchanged LR does not isolate relative credit allocation from update-scale effects.",
        "scale_analysis_md": str(SCALE_NOTE),
        "rows": rows,
        "raw_rows": raw_rows,
        "missing": missing,
    }
    out_json = out_dir / "wordmean_vs_tokenmean_trajectory.json"
    out_md = out_dir / "wordmean_vs_tokenmean_trajectory.md"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# research word-mean MLM trajectory comparison",
        "",
        summary["scope"],
        "",
        summary["scientific_question"],
        "",
        f"Scale caution: {summary['scale_caution']} See `{SCALE_NOTE}`.",
        "",
        "| exposure | complete | wordmean mean7 | tokenmean reinvest mean7 | clean mean7 | Δ wordmean-tokenmean | Δ wordmean-clean | BLiMP Δwm-tm | Supp | EWoK | Entity | COMPS | GPIQA | Reading |",
        "|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {m} | {complete} | {wm} | {tm} | {clean} | {dmt} | {dmc} | {b} | {s} | {e} | {ent} | {comp} | {g} | {r} |".format(
                m=row["exposure_m"],
                complete="yes" if row["complete"] else "no",
                wm=fmt(row["wordmean_mean7"]),
                tm=fmt(row["tokenmean_mean7"]),
                clean=fmt(row["clean_mean7"]),
                dmt=fmt(row["delta_wordmean_minus_tokenmean_mean7"]),
                dmc=fmt(row["delta_wordmean_minus_clean_mean7"]),
                b=fmt(row["delta_wordmean_minus_tokenmean_BLiMP"]),
                s=fmt(row["delta_wordmean_minus_tokenmean_Supplement"]),
                e=fmt(row["delta_wordmean_minus_tokenmean_EWoK"]),
                ent=fmt(row["delta_wordmean_minus_tokenmean_Entity"]),
                comp=fmt(row["delta_wordmean_minus_tokenmean_COMPS"]),
                g=fmt(row["delta_wordmean_minus_tokenmean_GlobalPIQA"]),
                r=fmt(row["delta_wordmean_minus_tokenmean_Reading"]),
            )
        )
    lines += ["", "## Missing inputs", ""]
    if missing:
        lines += [f"- `{m}`" for m in missing]
    else:
        lines.append("- none")
    lines += ["", f"Full JSON: `{out_json}`"]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "out_json": str(out_json), "out_md": str(out_md), "missing": missing}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
