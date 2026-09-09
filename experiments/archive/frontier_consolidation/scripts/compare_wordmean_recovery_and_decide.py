#!/usr/bin/env python3
"""research: compare recovered word-mean cheap-column evaluation and decide route.

This script reads the minimum research rerun outputs for the completed research
word-mean 70M/80M checkpoints, compares them to the research fixed-tokenizer
standard token-mean reinvest references and clean controls, and writes a durable
route decision.  It does not evaluate or train.
"""
from __future__ import annotations

import json
import math
import pathlib
import time
from typing import Any

STUDY = pathlib.Path("experiments/archive/frontier_consolidation")
OUT_DIR = STUDY / "data/wordmean_recovery_decision"
WM = STUDY / "data/wordmean_70_80M_eval/per_target"
TOKENMEAN_REFS = STUDY / "data/legal_mature_treatment_effect_eval/per_target"
CLEAN_REFS = STUDY / "data/legal_mature_clean_control_eval/per_target"
SCALE_NOTE = (STUDY.parents[2] / 'research/notes/frontier_consolidation/wordmean_screen_and_substrate_constraints.md')
OUT_ROOT = STUDY / "data/wordmean_70_80M_eval"
COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
RAW_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"]


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_json(path: pathlib.Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def task_score(payload: dict[str, Any], column: str) -> float:
    tasks = payload.get("tasks", {})
    if column == "GlobalPIQA":
        vals = [float(tasks[c]["score"]) for c in ("GlobalPIQA_parallel", "GlobalPIQA_nonparallel")]
        return sum(vals) / len(vals)
    if column == "Reading":
        rec = tasks["Reading"]
        scores = rec.get("scores")
        if isinstance(scores, dict) and scores.get("Reading") is not None:
            return float(scores["Reading"])
        return float(rec["score"])
    return float(tasks[column]["score"])


def raw_score(payload: dict[str, Any], column: str) -> float:
    if column == "Reading":
        return task_score(payload, "Reading")
    return float(payload["tasks"][column]["score"])


def mean(xs: list[float]) -> float:
    assert xs and all(math.isfinite(x) for x in xs)
    return sum(xs) / len(xs)


def payload_paths(m: int) -> dict[str, pathlib.Path]:
    return {
        "wordmean": WM / f"wordmean_mlm_seed43022_{m}M.json",
        "tokenmean_reinvest": TOKENMEAN_REFS / f"complianttok_reinvest_seed43022_{m}M.json",
        "clean_control": CLEAN_REFS / f"complianttok_cleanqwen_seed43022_{m}M.json",
    }


def collect_row(m: int) -> dict[str, Any]:
    paths = payload_paths(m)
    payloads = {k: load_json(v) for k, v in paths.items()}
    row: dict[str, Any] = {
        "exposure_m": m,
        "paths": {k: str(v) for k, v in paths.items()},
        "wordmean_target": payloads["wordmean"].get("target"),
        "tokenmean_target": payloads["tokenmean_reinvest"].get("target"),
        "clean_target": payloads["clean_control"].get("target"),
        "wordmean_run_summary": payloads["wordmean"].get("run_summary", {}),
    }
    for col in COLUMNS:
        w = task_score(payloads["wordmean"], col)
        t = task_score(payloads["tokenmean_reinvest"], col)
        c = task_score(payloads["clean_control"], col)
        row[f"wordmean_{col}"] = w
        row[f"tokenmean_{col}"] = t
        row[f"clean_{col}"] = c
        row[f"delta_wordmean_minus_tokenmean_{col}"] = w - t
        row[f"delta_wordmean_minus_clean_{col}"] = w - c
        row[f"delta_tokenmean_minus_clean_{col}"] = t - c
    row["wordmean_mean7"] = mean([row[f"wordmean_{c}"] for c in COLUMNS])
    row["tokenmean_mean7"] = mean([row[f"tokenmean_{c}"] for c in COLUMNS])
    row["clean_mean7"] = mean([row[f"clean_{c}"] for c in COLUMNS])
    row["delta_wordmean_minus_tokenmean_mean7"] = row["wordmean_mean7"] - row["tokenmean_mean7"]
    row["delta_wordmean_minus_clean_mean7"] = row["wordmean_mean7"] - row["clean_mean7"]
    row["delta_tokenmean_minus_clean_mean7"] = row["tokenmean_mean7"] - row["clean_mean7"]
    raw = {}
    for col in RAW_COLUMNS:
        raw[f"wordmean_{col}"] = raw_score(payloads["wordmean"], col)
        raw[f"tokenmean_{col}"] = raw_score(payloads["tokenmean_reinvest"], col)
        raw[f"clean_{col}"] = raw_score(payloads["clean_control"], col)
        raw[f"delta_wordmean_minus_tokenmean_{col}"] = raw[f"wordmean_{col}"] - raw[f"tokenmean_{col}"]
        raw[f"delta_wordmean_minus_clean_{col}"] = raw[f"wordmean_{col}"] - raw[f"clean_{col}"]
    raw["delta_wordmean_minus_tokenmean_mean8_raw"] = mean([raw[f"delta_wordmean_minus_tokenmean_{c}"] for c in RAW_COLUMNS])
    raw["delta_wordmean_minus_clean_mean8_raw"] = mean([raw[f"delta_wordmean_minus_clean_{c}"] for c in RAW_COLUMNS])
    row["raw_column_deltas"] = raw
    return row


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [collect_row(m) for m in (70, 80)]
    final80 = next(r for r in rows if r["exposure_m"] == 80)
    final70 = next(r for r in rows if r["exposure_m"] == 70)
    harmful_core_cols = ["BLiMP", "Supplement", "EWoK", "Entity", "Reading"]
    helpful_cols_80 = [c for c in COLUMNS if final80[f"delta_wordmean_minus_tokenmean_{c}"] > 0]
    harmful_cols_80 = [c for c in COLUMNS if final80[f"delta_wordmean_minus_tokenmean_{c}"] < 0]
    core_delta80 = mean([final80[f"delta_wordmean_minus_tokenmean_{c}"] for c in harmful_core_cols])
    core_delta70 = mean([final70[f"delta_wordmean_minus_tokenmean_{c}"] for c in harmful_core_cols])
    decision = {
        "continue_wordmean": False,
        "stop_wordmean_without_lr_retune": True,
        "next_expensive_action": "launch exactly one init-matched minfreq50 support-floor 80M screen, then evaluate 70M/80M cheap columns",
        "reason": (
            "Word-mean does not broadly strengthen the legal compact-view reinvestment trajectory. "
            "At 80M it improves only COMPS and GlobalPIQA while damaging BLiMP, Supplement, EWoK, Entity, and Reading; "
            "at 70M mean7 is flat only because a large GlobalPIQA gain masks losses in EWoK, Entity, BLiMP/Reading and near-flat Supplement."
        ),
        "decision_rule_match": "fails the predeclared continuation rule: no broad improvement and no repair of BLiMP/Supplement/EWoK; behavior is redistributive toward GlobalPIQA/COMPS",
        "broad_surface": {
            "70M_delta_mean7": final70["delta_wordmean_minus_tokenmean_mean7"],
            "80M_delta_mean7": final80["delta_wordmean_minus_tokenmean_mean7"],
            "70M_core_mean_delta_BLiMP_Supp_EWoK_Entity_Reading": core_delta70,
            "80M_core_mean_delta_BLiMP_Supp_EWoK_Entity_Reading": core_delta80,
            "80M_positive_columns": helpful_cols_80,
            "80M_negative_columns": harmful_cols_80,
        },
    }
    summary = {
        "status": "WORDMEAN_RECOVERY_DECISION",
        "created_utc": now_utc(),
        "scope": "Recovered minimum official-compatible cheap-column evaluation for the completed 70M/80M word-mean checkpoints; no SuperGLUE or AoA; not a complete endpoint.",
        "why_recovered": "The managed research evaluator waited indefinitely for 60000 MiB free on GPU1 while GPU0 was free; research reran only the two required cheap-column evaluations into an isolated output root and cancelled the obsolete stuck task.",
        "scientific_question": "Should selected-whole-word-group mean MLM continue to 100M/full evaluation, or stop and yield the H100 to the prepared support-floored tokenizer screen?",
        "scale_note": str(SCALE_NOTE),
        "output_root": str(OUT_ROOT),
        "rows": rows,
        "decision": decision,
    }
    out_json = OUT_DIR / "wordmean_recovery_decision.json"
    out_md = (OUT_DIR.parents[4] / 'research/documents/frontier_consolidation/data/wordmean_recovery_decision/wordmean_recovery_decision.md')
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# research word-mean recovery decision",
        "",
        summary["scope"],
        "",
        f"Recovered because: {summary['why_recovered']}",
        "",
        "Question: " + summary["scientific_question"],
        "",
        "| exposure | wordmean mean7 | tokenmean mean7 | clean mean7 | Δ wm-token | Δ wm-clean | Δ BLiMP | Δ Supp | Δ EWoK | Δ Entity | Δ COMPS | Δ GPIQA | Δ Reading |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['exposure_m']} | {r['wordmean_mean7']:.4f} | {r['tokenmean_mean7']:.4f} | {r['clean_mean7']:.4f} | "
            f"{r['delta_wordmean_minus_tokenmean_mean7']:+.4f} | {r['delta_wordmean_minus_clean_mean7']:+.4f} | "
            f"{r['delta_wordmean_minus_tokenmean_BLiMP']:+.4f} | {r['delta_wordmean_minus_tokenmean_Supplement']:+.4f} | "
            f"{r['delta_wordmean_minus_tokenmean_EWoK']:+.4f} | {r['delta_wordmean_minus_tokenmean_Entity']:+.4f} | "
            f"{r['delta_wordmean_minus_tokenmean_COMPS']:+.4f} | {r['delta_wordmean_minus_tokenmean_GlobalPIQA']:+.4f} | "
            f"{r['delta_wordmean_minus_tokenmean_Reading']:+.4f} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        f"- continue word-mean: `{decision['continue_wordmean']}`",
        f"- stop word-mean without LR retune: `{decision['stop_wordmean_without_lr_retune']}`",
        f"- next expensive action: {decision['next_expensive_action']}",
        f"- reason: {decision['reason']}",
        "",
        "This is a route decision from cheap columns only. It does not change the best legal endpoint or create a submission candidate.",
        "",
        f"Full JSON: `{out_json}`",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "out_json": str(out_json), "out_md": str(out_md), "decision": decision}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
