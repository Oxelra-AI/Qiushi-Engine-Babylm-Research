#!/usr/bin/env python3
"""research result synthesis for the FW compact anchor and breadth comparators.

Reads official-compatible collated summaries when available and reports the column
movements that determine whether the compact same-proposition recurrence, the
row-block breadth arm, or the interleaved breadth arm is the better data substrate.
Before evaluations exist, it records exactly which summaries are still absent.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

A01_WS = Path("experiments/archive/representation_and_objectives")
OUT_DIR = A01_WS / "data/fw_anchor_result_synthesis"
NOTE = (A01_WS.parents[2] / 'research/notes/representation_and_objectives/fw_anchor_result_synthesis.md')

TARGETS = {
    "compact_anchor": {
        "target": "fw_compact_fullbatch_seed43022",
        "summary": A01_WS / "data/fw_shared_anchor_pristine_collate/fw_compact_fullbatch_seed43022/pristine_collate_a02_fw_compact_fullbatch_seed43022_summary.json",
        "role": "same-proposition compact recurrence",
    },
    "rowblock_breadth": {
        "target": "fw_breadth_rowblock_fullbatch_seed43022",
        "summary": A01_WS / "data/fw_shared_anchor_pristine_collate/fw_breadth_rowblock_fullbatch_seed43022/pristine_collate_a02_fw_breadth_rowblock_fullbatch_seed43022_summary.json",
        "role": "whole-sentence independent breadth in row-block layout",
    },
    "interleaved_breadth": {
        "target": "fw_breadth_interleaved_fullbatch_seed43022",
        "summary": A01_WS / "data/fw_shared_anchor_pristine_collate/fw_breadth_interleaved_fullbatch_seed43022/pristine_collate_a01_fw_breadth_interleaved_fullbatch_seed43022_summary.json",
        "role": "whole-sentence independent breadth interleaved with common source segments",
    },
}

COLUMN_KEYS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "SuperGLUE", "GlobalPIQA", "Reading", "AoA", "Overall"]


def load_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    rec = json.loads(path.read_text(encoding="utf-8"))
    score = rec.get("score_summary")
    if not isinstance(score, dict):
        return {"path": str(path), "present": True, "score_missing": True, "raw_status": rec.get("status")}
    scores = dict(score.get("scores", {}))
    scores["Overall"] = score.get("Overall")
    return {
        "path": str(path),
        "present": True,
        "score_missing": False,
        "status": rec.get("status"),
        "validation_errors": rec.get("validation_errors"),
        "collated_sha256": rec.get("collated_sha256"),
        "scores": scores,
        "globalpiqa_parallel": score.get("GlobalPIQA_parallel"),
        "globalpiqa_nonparallel": score.get("GlobalPIQA_nonparallel"),
        "nlp_average": score.get("NLP_average"),
        "human_like_average": score.get("Human_like_average"),
        "margin_over_visible_leader_41p8": score.get("margin_over_visible_leader_41p8"),
    }


def delta(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    sa = a.get("scores") or {}
    sb = b.get("scores") or {}
    for k in COLUMN_KEYS:
        va = sa.get(k)
        vb = sb.get(k)
        out[k] = None if va is None or vb is None else float(va) - float(vb)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records: dict[str, Any] = {}
    missing: list[str] = []
    for key, spec in TARGETS.items():
        rec = load_summary(spec["summary"])
        if rec is None:
            missing.append(key)
            records[key] = {"present": False, "path": str(spec["summary"]), "role": spec["role"], "target": spec["target"]}
        else:
            rec["role"] = spec["role"]
            rec["target"] = spec["target"]
            records[key] = rec

    comparisons: dict[str, Any] = {}
    if all(records[k].get("present") and not records[k].get("score_missing") for k in ["compact_anchor", "rowblock_breadth"]):
        comparisons["compact_minus_rowblock_breadth"] = delta(records["compact_anchor"], records["rowblock_breadth"])
    if all(records[k].get("present") and not records[k].get("score_missing") for k in ["compact_anchor", "interleaved_breadth"]):
        comparisons["compact_minus_interleaved_breadth"] = delta(records["compact_anchor"], records["interleaved_breadth"])
    if all(records[k].get("present") and not records[k].get("score_missing") for k in ["rowblock_breadth", "interleaved_breadth"]):
        comparisons["interleaved_minus_rowblock_breadth"] = delta(records["interleaved_breadth"], records["rowblock_breadth"])

    interpretation = []
    if missing:
        interpretation.append("No result movement can be read yet; missing summaries: " + ", ".join(missing) + ".")
    else:
        ci = comparisons.get("compact_minus_interleaved_breadth", {})
        cr = comparisons.get("compact_minus_rowblock_breadth", {})
        if ci.get("Overall") is not None and cr.get("Overall") is not None:
            interpretation.append(f"Compact minus interleaved Overall {ci['Overall']:+.4f}; compact minus row-block Overall {cr['Overall']:+.4f}.")
            if ci["Overall"] >= 0.3 and cr["Overall"] >= 0.3:
                interpretation.append("Compact is better than both breadth layouts; same-proposition recurrence is the stronger data substrate to scale or refine.")
            elif ci["Overall"] <= -0.3 and cr["Overall"] <= -0.3:
                interpretation.append("Independent FineWeb breadth is better than compact recurrence in both layouts; future data construction should favor broader source coverage.")
            else:
                interpretation.append("The two breadth layouts or columns disagree; inspect task-family movements before choosing between compact scaling, breadth scaling, or the literal source-repeat attribution arm.")

    payload = {
        "status": "PENDING_INPUTS" if missing else "FW_ANCHOR_RESULTS_SYNTHESIZED",
        "targets": records,
        "missing": missing,
        "comparisons": comparisons,
        "interpretation": interpretation,
    }
    out_json = OUT_DIR / "fw_anchor_result_synthesis.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    csv_path = OUT_DIR / "fw_anchor_score_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["arm", "target", "role", *COLUMN_KEYS, "collated_sha256"])
        for key, rec in records.items():
            scores = rec.get("scores") or {}
            writer.writerow([key, rec.get("target"), rec.get("role"), *[scores.get(k) for k in COLUMN_KEYS], rec.get("collated_sha256")])

    lines = [
        "# research — FW shared-anchor result synthesis",
        "",
        f"Status: `{payload['status']}`.",
        "",
        "This file will compare A02 compact, A02 row-block breadth, and A01 interleaved breadth once the official-compatible collation summaries exist.",
        "",
        "Missing summaries now: " + (", ".join(missing) if missing else "none") + ".",
        "",
        "Interpretation notes:",
    ]
    for item in interpretation:
        lines.append(f"- {item}")
    lines += ["", f"JSON: `{out_json}`", f"CSV: `{csv_path}`"]
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"status": payload["status"], "missing": missing, "json": str(out_json), "csv": str(csv_path), "note": str(NOTE)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
