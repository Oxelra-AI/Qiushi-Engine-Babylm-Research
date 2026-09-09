#!/usr/bin/env python3
"""Summarize research mechanism-control full-eval results together with research reference rows."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json
import pathlib
from typing import Any

ROOT = _public_path('experiments/archive/compact_experience')
research = _public_path('experiments/archive/compact_experience/data/control_eval_summary.json')
PER = _public_path('experiments/archive/compact_experience/data/mechanism_eval/per_target')
OUT = _public_path('experiments/archive/compact_experience/data/mechanism_eval_summary.json')
NOTE = _public_path('research/notes/compact_experience/mechanism_control_eval_summary.md')
TASK_KEYS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "SuperGLUE", "Reading", "AoA"]
KEYS = TASK_KEYS + ["Overall", "NLP_average", "Human_like_average"]
NEW_TARGETS = ["selected_original_dup_all", "qwen_separated_pair"]
REF_TARGETS = ["official_lengthmatched", "qwen_clean_aligned", "qwen_shuffled_control", "official_original_dup"]


def r(x: Any, nd: int = 6):
    if x is None:
        return None
    try:
        return round(float(x), nd)
    except Exception:
        return x


def load_new(name: str) -> dict[str, Any] | None:
    p = PER / f"{name}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    oo = d.get("official_overall") or {}
    scores = oo.get("scores") or {}
    row = {k: scores.get(k) for k in TASK_KEYS}
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
    return row


def delta(rows: dict[str, dict[str, Any] | None], a: str, b: str) -> dict[str, Any] | None:
    if not rows.get(a) or not rows.get(b):
        return None
    out = {}
    for k in KEYS:
        av = rows[a].get(k)  # type: ignore[union-attr]
        bv = rows[b].get(k)  # type: ignore[union-attr]
        out[k] = None if av is None or bv is None else round(float(av) - float(bv), 6)
    return out


def md_table(rows: dict[str, dict[str, Any] | None], order: list[str]) -> list[str]:
    lines = [
        "| target | Overall | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | SuperGLUE | Reading | AoA leaderboard | AoA raw | ready |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name in order:
        row = rows.get(name)
        if not row:
            lines.append(f"| {name} | pending | | | | | | | | | | | |")
        else:
            lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                name, r(row.get("Overall"), 4), r(row.get("BLiMP"), 2), r(row.get("Supplement"), 2),
                r(row.get("EWoK"), 2), r(row.get("Entity"), 2), r(row.get("COMPS"), 2),
                r(row.get("GlobalPIQA"), 2), r(row.get("SuperGLUE"), 4), r(row.get("Reading"), 2),
                r(row.get("AoA"), 4), r(row.get("aoa_raw_correlation"), 6), row.get("submit_ready_overall"),
            ))
    return lines


def main():
    rows: dict[str, dict[str, Any] | None] = {}
    if research.exists():
        old = json.loads(research.read_text(encoding="utf-8"))
        for k in REF_TARGETS:
            rows[k] = old.get("targets", {}).get(k)
    for k in NEW_TARGETS:
        rows[k] = load_new(k)
    contrast_defs = [
        ("aligned_minus_selected_original_dup_all", "qwen_clean_aligned", "selected_original_dup_all", "generated rewrite second-view plus same-window correspondence beyond duplicating all selected originals"),
        ("aligned_minus_separated_pair", "qwen_clean_aligned", "qwen_separated_pair", "same-window adjacency/correspondence beyond same selected originals and rewrites in separate windows"),
        ("separated_minus_shuffled", "qwen_separated_pair", "qwen_shuffled_control", "same rewrite paired with same original in corpus but not same window, versus wrong rewrite in same window"),
        ("selected_original_dup_all_minus_official", "selected_original_dup_all", "official_lengthmatched", "effect of selected-original duplication/redundancy/source matching over row-length official control"),
        ("separated_pair_minus_official", "qwen_separated_pair", "official_lengthmatched", "effect of selected originals plus Qwen rewrites without same-window adjacency over official control"),
        ("selected_original_dup_all_minus_old_originaldup", "selected_original_dup_all", "official_original_dup", "impact of keeping all selected originals and matched source totals compared with earlier imperfect original-dup control"),
    ]
    contrasts = {name: {"a": a, "b": b, "description": desc, "delta": delta(rows, a, b)} for name, a, b, desc in contrast_defs}
    interp = {"complete_new_targets": [k for k in NEW_TARGETS if rows.get(k) and rows[k].get("submit_ready_overall")], "mechanism_read": "pending"}
    d_dup = contrasts["aligned_minus_selected_original_dup_all"]["delta"]
    d_sep = contrasts["aligned_minus_separated_pair"]["delta"]
    if d_dup is not None and d_sep is not None:
        dup_over = d_dup.get("Overall")
        sep_over = d_sep.get("Overall")
        interp["aligned_beats_selected_original_dup_all"] = dup_over is not None and dup_over > 0.10
        interp["aligned_beats_separated_pair"] = sep_over is not None and sep_over > 0.10
        if interp["aligned_beats_selected_original_dup_all"] and interp["aligned_beats_separated_pair"]:
            interp["mechanism_read"] = "aligned treatment retains a positive component beyond selected-original duplication and beyond separated coexistence, supporting same-window generated second-view correspondence as an active component under corrected AoA leaderboard units"
        elif not interp["aligned_beats_selected_original_dup_all"]:
            interp["mechanism_read"] = "selected-original duplication/exposure explains most aligned benefit; rewrite/correspondence mechanism is weaker than research suggested"
        elif not interp["aligned_beats_separated_pair"]:
            interp["mechanism_read"] = "coexistence of originals and rewrites explains most benefit; same-window adjacency is not necessary"
    payload = {"status": "MECHANISM_EVAL_SUMMARY", "targets": rows, "contrasts": contrasts, "interpretation": interp, "comparison_standard": "local official-style nine-column eval with AoA in leaderboard units (100*raw correlation); mechanism controls are not SOTA claims"}
    _public_path('experiments/archive/compact_experience/data').mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research mechanism-control evaluation", "", f"Summary JSON: `{OUT}`", "", "## Targets"]
    lines += md_table(rows, REF_TARGETS + NEW_TARGETS)
    lines += ["", "## Contrasts", ""]
    for name, rec in contrasts.items():
        d = rec["delta"]
        if d is None:
            lines.append(f"- `{name}` pending: {rec['description']}")
        else:
            lines.append(f"- `{name}` Overall Δ={r(d.get('Overall'),4)}; task Δs: " + ", ".join(f"{k}={r(d.get(k),3)}" for k in TASK_KEYS))
    lines += ["", "## Current interpretation", "", "```json", json.dumps(interp, indent=2), "```", ""]
    _public_path('research/notes/compact_experience').mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"summary": str(OUT), "note": str(NOTE), "complete_new_targets": interp["complete_new_targets"], "interpretation": interp}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
