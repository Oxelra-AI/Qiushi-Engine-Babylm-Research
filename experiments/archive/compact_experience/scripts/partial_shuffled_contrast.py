#!/usr/bin/env python3
"""research: partial (pre-SuperGLUE/AoA) contrast between the completed clean-Qwen
treatment full result and the shuffled-pair control's already-completed zero-shot +
Reading columns.

This is an interim mechanism read from measurements already on disk while the full
control/replication evaluation runs in the background. It does NOT replace the complete
nine-column contrast; SuperGLUE and AoA for the shuffled control are still pending.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json
import pathlib

ROOT = _public_path('experiments/archive/compact_experience')
PER = _public_path('experiments/archive/compact_experience/data/full_eval/per_target')
OUT = _public_path('experiments/archive/compact_experience/data/partial_shuffled_contrast.json')
NOTE = _public_path('research/notes/compact_experience/partial_shuffled_contrast.md')

ZERO_COLS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]


def load(target: str) -> dict:
    return json.loads((PER / f"{target}.json").read_text(encoding="utf-8"))


def zero_score(payload: dict, col: str):
    rec = payload.get("tasks", {}).get(col)
    return rec.get("score") if isinstance(rec, dict) else None


def gpiqa_mean(payload: dict):
    vals = []
    for col in ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]:
        rec = payload.get("tasks", {}).get(col)
        if isinstance(rec, dict) and rec.get("score") is not None:
            vals.append(float(rec["score"]))
    return sum(vals) / len(vals) if vals else None


def reading(payload: dict):
    rec = payload.get("tasks", {}).get("Reading")
    return rec.get("scores", {}).get("Reading") if isinstance(rec, dict) else None


def superglue(payload: dict):
    rec = payload.get("tasks", {}).get("SuperGLUE")
    if isinstance(rec, dict):
        return rec.get("superglue_mean"), len(rec.get("tasks", []) or [])
    return None, 0


def main() -> None:
    treat = load("qwen_clean_aligned")
    shuf = load("qwen_shuffled_control")
    cols = {}
    for col in ZERO_COLS:
        t = zero_score(treat, col)
        s = zero_score(shuf, col)
        cols[col] = {"treatment": t, "shuffled": s, "delta": None if t is None or s is None else round(t - s, 4)}
    for name, fn in [("GlobalPIQA_mean", gpiqa_mean), ("Reading", reading)]:
        t = fn(treat)
        s = fn(shuf)
        cols[name] = {"treatment": t, "shuffled": s, "delta": None if t is None or s is None else round(t - s, 4)}
    sg_t, sg_t_n = superglue(treat)
    sg_s, sg_s_n = superglue(shuf)
    cols["SuperGLUE_partial"] = {
        "treatment": sg_t, "treatment_tasks": sg_t_n,
        "shuffled": sg_s, "shuffled_tasks": sg_s_n,
        "note": "SuperGLUE means over DIFFERENT numbers of finetuning tasks are not directly comparable until the shuffled control completes all seven.",
    }
    # Equal-weight partial screen over the seven columns that are complete for BOTH targets
    complete_common = [k for k in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_mean", "Reading"] if cols[k]["delta"] is not None]
    partial_mean = round(sum(cols[k]["delta"] for k in complete_common) / len(complete_common), 4) if complete_common else None
    payload = {
        "status": "PARTIAL_SHUFFLED_CONTRAST",
        "purpose": "Interim mechanism read from measurements already on disk while full control eval runs; not a complete nine-column result.",
        "columns": cols,
        "partial_equal_weight_mean_over_common_complete_columns": partial_mean,
        "common_complete_columns": complete_common,
        "caveats": [
            "Shuffled control SuperGLUE/AoA still pending; do not compute Overall from this file.",
            "Positive partial mean would indicate original--rewrite correspondence contributes beyond the fixed generated-text multiset on zero-shot/Reading columns.",
            "Negative or near-zero partial mean would indicate the first-seed effect is driven mainly by the generated multiset / register / source mixture rather than pair correspondence.",
        ],
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research partial shuffled-pair contrast (interim, zero-shot + Reading only)", "",
             f"Summary JSON: `{OUT}`", "",
             "Treatment = `qwen_clean_aligned`; control = `qwen_shuffled_control` (same selected originals + Qwen rewrite multiset, correspondence broken).", "",
             "| column | treatment | shuffled | Δ(treat−shuf) |", "|---|---:|---:|---:|"]
    for k, v in cols.items():
        if k == "SuperGLUE_partial":
            continue
        lines.append(f"| {k} | {v['treatment']} | {v['shuffled']} | {v['delta']} |")
    lines += ["",
              f"Partial equal-weight mean over complete common columns {complete_common}: **{partial_mean}**", "",
              "SuperGLUE partial: treatment mean={} over {} tasks; shuffled mean={} over {} tasks (not comparable yet).".format(
                  cols['SuperGLUE_partial']['treatment'], cols['SuperGLUE_partial']['treatment_tasks'],
                  cols['SuperGLUE_partial']['shuffled'], cols['SuperGLUE_partial']['shuffled_tasks']), "",
              "This is interim evidence only. The decisive contrast requires the complete nine-column shuffled-control result from the running evaluation."]
    NOTE.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "note": str(NOTE), "partial_mean": partial_mean, "common_complete_columns": complete_common, "columns": cols}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
