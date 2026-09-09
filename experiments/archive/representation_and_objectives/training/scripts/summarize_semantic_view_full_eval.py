#!/usr/bin/env python3
"""Summarize selected semantic-view full official-style evaluations."""
from __future__ import annotations

import json
import pathlib
from typing import Any

COMPACT_EXPERIENCE_SCORING = pathlib.Path("experiments/archive/compact_experience/scripts/babylm_official_scoring.py")
OUT = pathlib.Path("experiments/archive/representation_and_objectives/data/semantic_view_full_eval")
PER = OUT / "per_target"
NOTE = pathlib.Path("research/notes/representation_and_objectives/semantic_view_selected_full_eval_summary.md")
SUMMARY = OUT / "semantic_view_selected_full_eval_summary.json"
KEYS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "SuperGLUE", "Reading", "AoA"]
LEADER = {"name": "wwm_curriculum_simplification_40k", "Overall": 41.80}


def r(x: Any, nd: int = 4):
    if x is None:
        return None
    try:
        return round(float(x), nd)
    except Exception:
        return x


def load_rows() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not PER.exists():
        return rows
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
            "endpoint": d.get("endpoint"),
            "run_dir": d.get("run_dir"),
            "model_path": d.get("model_path"),
            "per_target_json": str(p),
            "prefill_source": d.get("prefill_source"),
        })
        rows[p.stem] = row
    return rows


def pair_deltas(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    deltas: dict[str, Any] = {}
    for t_name, t in rows.items():
        if not t_name.startswith("semantic_view_treatment__"):
            continue
        endpoint = t.get("endpoint")
        c_name = f"original_packet_local__{endpoint}"
        if c_name not in rows:
            # If the best endpoints differ, the launcher may still have a packet row
            # at another endpoint; pairwise same-endpoint delta remains unavailable.
            continue
        c = rows[c_name]
        d = {}
        for k in KEYS + ["Overall", "NLP_average", "Human_like_average"]:
            tv, cv = t.get(k), c.get(k)
            d[k] = None if tv is None or cv is None else round(float(tv) - float(cv), 6)
        deltas[f"{t_name}_minus_{c_name}"] = d
    return deltas


def main() -> None:
    rows = load_rows()
    deltas = pair_deltas(rows)
    best = None
    complete = {k: v for k, v in rows.items() if v.get("Overall") is not None}
    if complete:
        best = max(complete.items(), key=lambda kv: float(kv[1]["Overall"]))
    payload = {
        "status": "SEMANTIC_VIEW_SELECTED_FULL_EVAL_SUMMARY",
        "leader_reference": LEADER,
        "targets": rows,
        "same_endpoint_treatment_minus_packet_local": deltas,
        "best_by_overall": {"target": best[0], "row": best[1]} if best else None,
        "decision_note": "A same-endpoint full Overall treatment-control gain is evidence about generated paraphrastic variation vs packet-local same-source repetition; SOTA comparison still requires submit-ready full official artifacts and comparison to the public 41.8 leader.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["# research semantic-view selected full-evaluation summary\n\n", f"Summary JSON: `{SUMMARY}`\n\n"]
    lines.append("| target | endpoint | Overall | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | SuperGLUE | Reading | AoA leaderboard | AoA raw | submit-ready |\n")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
    for name, row in sorted(rows.items()):
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n".format(
            name, row.get("endpoint"), r(row.get("Overall")), r(row.get("BLiMP"),2), r(row.get("Supplement"),2),
            r(row.get("EWoK"),2), r(row.get("Entity"),2), r(row.get("COMPS"),2), r(row.get("GlobalPIQA"),2),
            r(row.get("SuperGLUE"),4), r(row.get("Reading"),2), r(row.get("AoA"),4), r(row.get("aoa_raw_correlation"),6),
            row.get("submit_ready_overall"),
        ))
    if deltas:
        lines.append("\n## Same-endpoint treatment minus packet-local deltas\n\n")
        lines.append("```json\n" + json.dumps(deltas, indent=2, ensure_ascii=False) + "\n```\n")
    if best:
        lines.append(f"\nBest full-eval target: `{best[0]}` Overall {best[1].get('Overall')}. Public reference: {LEADER['Overall']}.\n")
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"summary": str(SUMMARY), "note": str(NOTE), "best_by_overall": payload["best_by_overall"], "deltas": deltas}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
