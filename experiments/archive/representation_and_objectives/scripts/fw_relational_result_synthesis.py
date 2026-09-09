#!/usr/bin/env python3
"""research: synthesize FW official scores with relational readouts.

After the FineWeb compact/breadth endpoints mature and are evaluated, this script
combines three evidence layers:
  1. official-compatible nine-column collation from research,
  2. EWoK four-cell conditional-reversal reader from research,
  3. GlobalPIQA all-option hard-row margin reader from research.

The scientific question is:
does compact recurrence improve context-conditioned world relations, visible both
as fewer EWoK conditional-reversal failures and better GlobalPIQA_parallel hard-row
ranks/margins, or is corpus allocation alone failing the dominant frontier gap?
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

A01_WS = Path("experiments/archive/representation_and_objectives")
OUT_DIR = A01_WS / "data/fw_relational_result_synthesis"
NOTE = (A01_WS.parents[2] / 'research/notes/representation_and_objectives/fw_relational_result_synthesis.md')

TARGETS = {
    "compact_anchor": {
        "target": "fw_compact_fullbatch_seed43022",
        "role": "same-proposition compact recurrence anchor",
        "official": A01_WS / "data/fw_shared_anchor_pristine_collate/fw_compact_fullbatch_seed43022/pristine_collate_a02_fw_compact_fullbatch_seed43022_summary.json",
        "ewok": A01_WS / "data/fw_ewok_interaction_reader/fw_compact_fullbatch_seed43022/ewok_interaction_summary.json",
        "gpiqa": A01_WS / "data/fw_globalpiqa_margin_reader/a02_fw_compact_fullbatch_seed43022_margins.json",
    },
    "rowblock_breadth": {
        "target": "fw_breadth_rowblock_fullbatch_seed43022",
        "role": "whole-sentence independent breadth in row-block layout",
        "official": A01_WS / "data/fw_shared_anchor_pristine_collate/fw_breadth_rowblock_fullbatch_seed43022/pristine_collate_a02_fw_breadth_rowblock_fullbatch_seed43022_summary.json",
        "ewok": A01_WS / "data/fw_ewok_interaction_reader/fw_breadth_rowblock_fullbatch_seed43022/ewok_interaction_summary.json",
        "gpiqa": A01_WS / "data/fw_globalpiqa_margin_reader/a02_fw_breadth_rowblock_fullbatch_seed43022_margins.json",
    },
    "interleaved_breadth": {
        "target": "fw_breadth_interleaved_fullbatch_seed43022",
        "role": "whole-sentence independent breadth interleaved with common source segments",
        "official": A01_WS / "data/fw_shared_anchor_pristine_collate/fw_breadth_interleaved_fullbatch_seed43022/pristine_collate_a01_fw_breadth_interleaved_fullbatch_seed43022_summary.json",
        "ewok": A01_WS / "data/fw_ewok_interaction_reader/fw_breadth_interleaved_fullbatch_seed43022/ewok_interaction_summary.json",
        "gpiqa": A01_WS / "data/fw_globalpiqa_margin_reader/a01_fw_breadth_interleaved_fullbatch_seed43022_margins.json",
    },
}

BASELINES = {
    "best_compliant_legal40k8x480": {
        "Overall": 41.141,
        "EWoK": 51.47,
        "GlobalPIQA": 34.67,
        "GlobalPIQA_parallel": 22.33009708737864,
        "ewok_stable_failure_frac_wrong": 0.684931506849315,
        "gpiqa_hard_mean_top_minus_correct": 1.953398905689513,
    },
    "visible_leader": {
        "Overall": 41.80,
        "EWoK": 56.07,
        "GlobalPIQA": 39.67,
    },
}
COLUMN_KEYS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "SuperGLUE", "GlobalPIQA", "Reading", "AoA", "Overall"]


def load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def official_scores(path: Path) -> dict[str, Any] | None:
    rec = load_json(path)
    if rec is None:
        return None
    score = rec.get("score_summary") if isinstance(rec, dict) else None
    if not isinstance(score, dict):
        return {"present": True, "score_missing": True, "path": str(path), "status": rec.get("status") if isinstance(rec, dict) else None}
    scores = dict(score.get("scores", {}))
    scores["Overall"] = score.get("Overall")
    return {
        "present": True,
        "score_missing": False,
        "path": str(path),
        "scores": scores,
        "GlobalPIQA_parallel": score.get("GlobalPIQA_parallel"),
        "GlobalPIQA_nonparallel": score.get("GlobalPIQA_nonparallel"),
        "collated_sha256": rec.get("collated_sha256"),
        "validation_errors": rec.get("validation_errors"),
    }


def ewok_summary(path: Path) -> dict[str, Any] | None:
    rec = load_json(path)
    if rec is None:
        return None
    s = rec.get("summary") if isinstance(rec, dict) else None
    if not isinstance(s, dict):
        return {"present": True, "summary_missing": True, "path": str(path)}
    return {
        "present": True,
        "path": str(path),
        "accuracy": s.get("accuracy"),
        "saved_wrong": s.get("saved_wrong"),
        "stable_failure": s.get("stable_failure"),
        "stable_failure_frac_all": s.get("stable_failure_frac_all"),
        "stable_failure_frac_wrong": s.get("stable_failure_frac_wrong"),
        "interaction_sum_wrong_median": (s.get("interaction_sum_wrong") or {}).get("median"),
        "interaction_sum_wrong_mean": (s.get("interaction_sum_wrong") or {}).get("mean"),
        "within_both_positive_wrong_frac": s.get("within_both_positive_wrong_frac"),
    }


def gpiqa_summary(path: Path) -> dict[str, Any] | None:
    rec = load_json(path)
    if rec is None:
        return None
    modes = rec.get("modes") if isinstance(rec, dict) else None
    if not isinstance(modes, dict) or "parallel" not in modes:
        return {"present": True, "summary_missing": True, "path": str(path)}
    s = modes["parallel"].get("summary", {})
    aw = s.get("always_wrong_subset") or {}
    allm = s.get("all_rows_margin_summary") or {}
    return {
        "present": True,
        "path": str(path),
        "parallel_accuracy": s.get("accuracy"),
        "rank_counts": s.get("correct_rank_counts"),
        "hard_n": aw.get("n"),
        "hard_rank_counts": aw.get("correct_rank_counts"),
        "hard_mean_top_minus_correct": aw.get("mean_top_minus_correct"),
        "hard_median_top_minus_correct": aw.get("median_top_minus_correct"),
        "hard_small_margin_le_0p50": aw.get("small_wrong_margin_le_0p50_nats"),
        "all_mean_top_minus_correct": allm.get("mean_top_minus_correct"),
        "all_median_top_minus_correct": allm.get("median_top_minus_correct"),
    }


def fnum(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def delta_val(a: Any, b: Any) -> float | None:
    fa = fnum(a); fb = fnum(b)
    if fa is None or fb is None:
        return None
    return fa - fb


def pair_delta(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    oa = a.get("official") or {}; ob = b.get("official") or {}
    ea = a.get("ewok") or {}; eb = b.get("ewok") or {}
    ga = a.get("gpiqa") or {}; gb = b.get("gpiqa") or {}
    out = {"official": {}, "ewok": {}, "gpiqa": {}}
    sa = oa.get("scores") or {}; sb = ob.get("scores") or {}
    for col in COLUMN_KEYS:
        out["official"][col] = delta_val(sa.get(col), sb.get(col))
    out["official"]["GlobalPIQA_parallel"] = delta_val(oa.get("GlobalPIQA_parallel"), ob.get("GlobalPIQA_parallel"))
    out["official"]["GlobalPIQA_nonparallel"] = delta_val(oa.get("GlobalPIQA_nonparallel"), ob.get("GlobalPIQA_nonparallel"))
    for key in ["accuracy", "saved_wrong", "stable_failure", "stable_failure_frac_all", "stable_failure_frac_wrong", "interaction_sum_wrong_median", "interaction_sum_wrong_mean", "within_both_positive_wrong_frac"]:
        out["ewok"][key] = delta_val(ea.get(key), eb.get(key))
    for key in ["parallel_accuracy", "hard_mean_top_minus_correct", "hard_median_top_minus_correct", "hard_small_margin_le_0p50", "all_mean_top_minus_correct", "all_median_top_minus_correct"]:
        out["gpiqa"][key] = delta_val(ga.get(key), gb.get(key))
    return out


def relation_movement(record: dict[str, Any]) -> dict[str, Any]:
    official = record.get("official") or {}
    ewok = record.get("ewok") or {}
    gpiqa = record.get("gpiqa") or {}
    scores = official.get("scores") or {}
    base = BASELINES["best_compliant_legal40k8x480"]
    return {
        "overall_minus_best_compliant": delta_val(scores.get("Overall"), base["Overall"]),
        "overall_minus_visible_leader": delta_val(scores.get("Overall"), BASELINES["visible_leader"]["Overall"]),
        "ewok_score_minus_best_compliant": delta_val(scores.get("EWoK"), base["EWoK"]),
        "globalpiqa_score_minus_best_compliant": delta_val(scores.get("GlobalPIQA"), base["GlobalPIQA"]),
        "globalpiqa_parallel_minus_best_compliant": delta_val(official.get("GlobalPIQA_parallel") or gpiqa.get("parallel_accuracy"), base["GlobalPIQA_parallel"]),
        "ewok_stable_failure_frac_wrong_minus_best_compliant": delta_val(ewok.get("stable_failure_frac_wrong"), base["ewok_stable_failure_frac_wrong"]),
        "gpiqa_hard_margin_minus_best_compliant": delta_val(gpiqa.get("hard_mean_top_minus_correct"), base["gpiqa_hard_mean_top_minus_correct"]),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records: dict[str, Any] = {}
    missing: dict[str, list[str]] = {}
    for key, spec in TARGETS.items():
        off = official_scores(spec["official"])
        ew = ewok_summary(spec["ewok"])
        gp = gpiqa_summary(spec["gpiqa"])
        rec = {"target": spec["target"], "role": spec["role"], "paths": {"official": str(spec["official"]), "ewok": str(spec["ewok"]), "gpiqa": str(spec["gpiqa"])}, "official": off, "ewok": ew, "gpiqa": gp}
        rec["movement_vs_best_compliant"] = relation_movement(rec)
        records[key] = rec
        miss = []
        if off is None or off.get("score_missing"):
            miss.append("official")
        if ew is None or ew.get("summary_missing"):
            miss.append("ewok_interaction")
        if gp is None or gp.get("summary_missing"):
            miss.append("globalpiqa_margins")
        if miss:
            missing[key] = miss

    comparisons = {}
    for name, akey, bkey in [
        ("compact_minus_rowblock_breadth", "compact_anchor", "rowblock_breadth"),
        ("compact_minus_interleaved_breadth", "compact_anchor", "interleaved_breadth"),
        ("interleaved_minus_rowblock_breadth", "interleaved_breadth", "rowblock_breadth"),
    ]:
        if akey in records and bkey in records:
            comparisons[name] = pair_delta(records[akey], records[bkey])

    reading: list[str] = []
    if missing:
        reading.append("Post-endpoint relational synthesis is not ready because some evidence layers are absent: " + json.dumps(missing, ensure_ascii=False) + ".")
    else:
        ci = comparisons.get("compact_minus_interleaved_breadth", {})
        cr = comparisons.get("compact_minus_rowblock_breadth", {})
        ci_overall = (((ci.get("official") or {}).get("Overall")))
        cr_overall = (((cr.get("official") or {}).get("Overall")))
        ci_ewok_fail = (((ci.get("ewok") or {}).get("stable_failure_frac_wrong")))
        ci_gpiqa_margin = (((ci.get("gpiqa") or {}).get("hard_mean_top_minus_correct")))
        cr_ewok_fail = (((cr.get("ewok") or {}).get("stable_failure_frac_wrong")))
        cr_gpiqa_margin = (((cr.get("gpiqa") or {}).get("hard_mean_top_minus_correct")))
        reading.append(f"Compact Overall deltas: vs interleaved {ci_overall}, vs row-block {cr_overall}.")
        reading.append(f"Compact relational deltas: EWoK stable-failure-frac-wrong vs interleaved {ci_ewok_fail}, vs row-block {cr_ewok_fail}; GlobalPIQA hard-margin vs interleaved {ci_gpiqa_margin}, vs row-block {cr_gpiqa_margin} (negative hard-margin movement is improvement).")
        # We do not use this to decide automatically, but the text keeps the scientific interpretation explicit.
        if ci_overall is not None and cr_overall is not None and ci_ewok_fail is not None and cr_ewok_fail is not None and ci_gpiqa_margin is not None and cr_gpiqa_margin is not None:
            if ci_overall > 0 and cr_overall > 0 and ci_ewok_fail < 0 and cr_ewok_fail < 0 and ci_gpiqa_margin < 0 and cr_gpiqa_margin < 0:
                reading.append("Compact recurrence improves official score and both relational readouts relative to both breadth layouts; this would strengthen aligned recurrence as a real relation-learning substrate.")
            elif ci_ewok_fail >= 0 and cr_ewok_fail >= 0 and ci_gpiqa_margin >= 0 and cr_gpiqa_margin >= 0:
                reading.append("Compact recurrence does not reduce the measured relation failures relative to breadth; if official Overall is also not above frontier, corpus allocation alone is probably not the crossing mechanism.")
            else:
                reading.append("Official and relational movements do not line up cleanly; inspect rows and columns before choosing compact scaling, breadth scaling, source-repeat attribution, or a new transition-objective route.")

    payload = {
        "status": "PENDING_INPUTS" if missing else "FW_RELATIONAL_RESULTS_SYNTHESIZED",
        "baselines": BASELINES,
        "targets": records,
        "missing": missing,
        "comparisons": comparisons,
        "scientific_reading": reading,
        "files": {
            "summary_json": str(OUT_DIR / "fw_relational_result_synthesis.json"),
            "score_table_csv": str(OUT_DIR / "fw_relational_score_table.csv"),
            "note": str(NOTE),
        },
    }
    (OUT_DIR / "fw_relational_result_synthesis.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    csv_rows = []
    for key, rec in records.items():
        scores = ((rec.get("official") or {}).get("scores") or {})
        ew = rec.get("ewok") or {}
        gp = rec.get("gpiqa") or {}
        mv = rec.get("movement_vs_best_compliant") or {}
        csv_rows.append({
            "arm": key,
            "target": rec.get("target"),
            "role": rec.get("role"),
            **{col: scores.get(col) for col in COLUMN_KEYS},
            "GlobalPIQA_parallel": (rec.get("official") or {}).get("GlobalPIQA_parallel") or gp.get("parallel_accuracy"),
            "EWoK_stable_failure_frac_wrong": ew.get("stable_failure_frac_wrong"),
            "EWoK_stable_failure": ew.get("stable_failure"),
            "GlobalPIQA_hard_mean_top_minus_correct": gp.get("hard_mean_top_minus_correct"),
            "GlobalPIQA_hard_rank_counts": json.dumps(gp.get("hard_rank_counts"), ensure_ascii=False),
            "Overall_minus_best_compliant": mv.get("overall_minus_best_compliant"),
            "EWoK_score_minus_best_compliant": mv.get("ewok_score_minus_best_compliant"),
            "GlobalPIQA_score_minus_best_compliant": mv.get("globalpiqa_score_minus_best_compliant"),
            "GlobalPIQA_parallel_minus_best_compliant": mv.get("globalpiqa_parallel_minus_best_compliant"),
        })
    csv_path = OUT_DIR / "fw_relational_score_table.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        if csv_rows:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader(); w.writerows(csv_rows)

    lines = [
        "# research — FW relational result synthesis\n\n",
        "This combines official-compatible scores with EWoK four-cell conditional-reversal measurements and GlobalPIQA_parallel hard-row margins for the FW compact/breadth endpoints.\n\n",
        f"Status: `{payload['status']}`.\n\n",
    ]
    if missing:
        lines.append("Missing evidence layers:\n")
        for k, v in missing.items():
            lines.append(f"- `{k}`: {', '.join(v)}\n")
        lines.append("\n")
    lines.append("Scientific reading:\n")
    for item in reading:
        lines.append(f"- {item}\n")
    lines.append("\nFiles:\n")
    for k, p in payload["files"].items():
        lines.append(f"- {k}: `{p}`\n")
    NOTE.write_text("".join(lines), encoding="utf-8")

    print(json.dumps({"status": payload["status"], "missing": missing, "json": payload["files"]["summary_json"], "csv": str(csv_path), "note": str(NOTE)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
