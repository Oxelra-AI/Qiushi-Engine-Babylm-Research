#!/usr/bin/env python3
"""Summarize research full official-style Overall evaluation outputs."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json
import pathlib
from typing import Any, Dict

ROOT = _public_path('experiments/archive/compact_experience')
OUT = _public_path('experiments/archive/compact_experience/data/full_eval')
PER = _public_path('experiments/archive/compact_experience/data/full_eval/per_target')
SUMMARY = _public_path('experiments/archive/compact_experience/data/full_eval/full_eval_summary.json')
NOTE = _public_path('research/notes/compact_experience/clean_qwen_full_eval.md')
LEADER = {"Overall": 41.80, "name": "visible 2026 Strict-Small leader wwm_curriculum_simplification_40k"}
INHERITED = {"Overall": 40.7027874108262, "name": "INITIAL_MODEL_STUDIES inherited internal coordinate"}
KEYS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "SuperGLUE", "Reading", "AoA"]

def r(x: Any, nd: int = 4):
    if x is None:
        return None
    try:
        return round(float(x), nd)
    except Exception:
        return x

def load_rows() -> Dict[str, Dict[str, Any]]:
    rows = {}
    if PER.exists():
        for p in sorted(PER.glob("*.json")):
            d = json.loads(p.read_text(encoding="utf-8"))
            oo = d.get("official_overall") or {}
            scores = oo.get("scores") or {}
            row = {k: scores.get(k) for k in KEYS}
            row.update({
                "Overall": oo.get("Overall"),
                "NLP_average": oo.get("NLP_average"),
                "Human_like_average": oo.get("Human_like_average"),
                "submit_ready_overall": oo.get("submit_ready_overall"),
                "submit_ready_aoa": oo.get("submit_ready_aoa"),
                  "aoa_status": oo.get("aoa_status"),
                  "aoa_raw_correlation": oo.get("aoa_raw_correlation"),
                  "aoa_leaderboard_score": oo.get("aoa_leaderboard_score"),
                "per_target_json": str(p),
                "model_path": d.get("model_path"),
            })
            rows[p.stem] = row
    return rows

def main() -> None:
    rows = load_rows()
    delta = None
    if "official_lengthmatched" in rows and "qwen_clean_aligned" in rows:
        delta = {}
        for k in KEYS + ["Overall", "NLP_average", "Human_like_average"]:
            a, b = rows["qwen_clean_aligned"].get(k), rows["official_lengthmatched"].get(k)
            delta[k] = None if a is None or b is None else round(float(a) - float(b), 6)
    payload = {
        "status": "FULL_EVAL_SUMMARY",
        "targets": rows,
        "qwen_clean_aligned_minus_official_lengthmatched": delta,
        "leader_reference": LEADER,
        "inherited_reference": INHERITED,
          "decision_rule": "Continue Qwen-aligned route only if complete nine-column ΔOverall >= +0.25 over matched official control, using AoA in leaderboard units (100*raw correlation); otherwise terminate this route.",
    }
    if delta and delta.get("Overall") is not None:
        payload["decision"] = "continue_to_second_seed_and_shuffled_control" if delta["Overall"] >= 0.25 else "terminate_clean_qwen_route"
    OUT.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# research clean-Qwen full evaluation", "",
        f"Summary JSON: `{SUMMARY}`", "",
          "| target | Overall | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | SuperGLUE | Reading | AoA leaderboard | AoA raw | submit-ready AoA |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, row in sorted(rows.items()):
          lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
              name, r(row.get("Overall")), r(row.get("BLiMP"),2), r(row.get("Supplement"),2), r(row.get("EWoK"),2),
              r(row.get("Entity"),2), r(row.get("COMPS"),2), r(row.get("GlobalPIQA"),2), r(row.get("SuperGLUE"),4),
              r(row.get("Reading"),2), r(row.get("AoA"),4), r(row.get("aoa_raw_correlation"),6), row.get("submit_ready_aoa"),
          ))
    if delta:
        lines += ["", "## qwen_clean_aligned minus official_lengthmatched", "", "```json", json.dumps(delta, indent=2), "```"]
    lines += ["", f"Decision: `{payload.get('decision', 'pending')}`"]
    _public_path('research/notes/compact_experience').mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(SUMMARY), "note": str(NOTE), "decision": payload.get("decision"), "delta": delta}, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
