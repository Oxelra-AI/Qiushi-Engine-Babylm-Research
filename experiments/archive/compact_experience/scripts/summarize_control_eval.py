#!/usr/bin/env python3
"""Summarize clean-Qwen research control/replication full-eval results.

Reads the same per-target JSONs produced by full_eval_runner.py and writes a
mechanism-facing comparison summary.  It is safe to run before all targets exist; missing
rows are recorded as pending rather than silently ignored.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json
import pathlib
from typing import Any

ROOT = _public_path('experiments/archive/compact_experience')
PER = _public_path('experiments/archive/compact_experience/data/full_eval/per_target')
OUT = _public_path('experiments/archive/compact_experience/data/control_eval_summary.json')
NOTE = _public_path('research/notes/compact_experience/clean_qwen_control_eval_summary.md')
KEYS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "SuperGLUE", "Reading", "AoA", "Overall", "NLP_average", "Human_like_average"]
TASK_KEYS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "SuperGLUE", "Reading", "AoA"]
EXPECTED = [
    "official_lengthmatched",
    "qwen_clean_aligned",
    "qwen_shuffled_control",
    "official_sourcematched",
    "official_original_dup",
    "official_lengthmatched_seed43122",
    "qwen_clean_aligned_seed43122",
]
CONTRASTS = [
    ("seed43022_qwen_minus_lengthmatched", "qwen_clean_aligned", "official_lengthmatched", "first-seed filtered-Qwen pipeline effect over row-length-matched official control"),
    ("seed43022_qwen_minus_shuffled", "qwen_clean_aligned", "qwen_shuffled_control", "effect of preserving original--rewrite correspondence after holding selected originals and rewrite multiset fixed"),
    ("seed43022_qwen_minus_sourcematched", "qwen_clean_aligned", "official_sourcematched", "effect beyond official-only source/domain reweighting and row-length matching"),
    ("seed43022_qwen_minus_originaldup", "qwen_clean_aligned", "official_original_dup", "rewrite/generated second-view effect beyond official original+original redundancy, not an exact source/length control"),
    ("seed43122_qwen_minus_lengthmatched", "qwen_clean_aligned_seed43122", "official_lengthmatched_seed43122", "second-seed replication of filtered-Qwen pipeline effect"),
    ("qwen_seed43122_minus_seed43022", "qwen_clean_aligned_seed43122", "qwen_clean_aligned", "treatment seed sensitivity"),
    ("official_seed43122_minus_seed43022", "official_lengthmatched_seed43122", "official_lengthmatched", "official-control seed sensitivity"),
]


def r(x: Any, nd: int = 6):
    if x is None:
        return None
    try:
        return round(float(x), nd)
    except Exception:
        return x


def load_target(name: str) -> dict[str, Any] | None:
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
          "complete_for_provisional_overall": oo.get("complete_for_provisional_overall"),
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


def interpretation(contrasts: dict[str, Any]) -> dict[str, Any]:
    first = contrasts.get("seed43022_qwen_minus_lengthmatched", {}).get("delta") if contrasts.get("seed43022_qwen_minus_lengthmatched") else None
    second = contrasts.get("seed43122_qwen_minus_lengthmatched", {}).get("delta") if contrasts.get("seed43122_qwen_minus_lengthmatched") else None
    shuf = contrasts.get("seed43022_qwen_minus_shuffled", {}).get("delta") if contrasts.get("seed43022_qwen_minus_shuffled") else None
    src = contrasts.get("seed43022_qwen_minus_sourcematched", {}).get("delta") if contrasts.get("seed43022_qwen_minus_sourcematched") else None
    dup = contrasts.get("seed43022_qwen_minus_originaldup", {}).get("delta") if contrasts.get("seed43022_qwen_minus_originaldup") else None
    interp = {
        "first_seed_positive_over_0p25": None if first is None or first.get("Overall") is None else first["Overall"] >= 0.25,
        "second_seed_positive_over_0p25": None if second is None or second.get("Overall") is None else second["Overall"] >= 0.25,
        "pair_correspondence_survives_shuffled_control": None if shuf is None or shuf.get("Overall") is None else shuf["Overall"] > 0.10,
        "beyond_source_matching": None if src is None or src.get("Overall") is None else src["Overall"] > 0.10,
        "beyond_original_duplication": None if dup is None or dup.get("Overall") is None else dup["Overall"] > 0.10,
        "mechanism_statement": "pending until shuffled, source-matched, original-duplication, and second-seed rows all have complete official-style evaluation",
    }
    if interp["second_seed_positive_over_0p25"] is False:
        interp["mechanism_statement"] = "second seed does not reproduce the +0.25 Overall effect; treat first positive as seed/context-specific and redirect before mechanism claims"
    elif interp["pair_correspondence_survives_shuffled_control"] is False:
        interp["mechanism_statement"] = "shuffled generated-text control matches or exceeds paired treatment; effect is not attributable to original--rewrite correspondence"
    elif all(interp[k] is True for k in ["first_seed_positive_over_0p25", "second_seed_positive_over_0p25", "pair_correspondence_survives_shuffled_control", "beyond_source_matching", "beyond_original_duplication"]):
        interp["mechanism_statement"] = "replicated positive effect with controls favors a semantic-pair/cross-view training signal component, now under corrected AoA leaderboard units; official-rule confirmation and leaderboard-level score remain separate requirements"
    return interp


def md_table(rows: dict[str, dict[str, Any] | None]) -> list[str]:
    lines = [
        "| target | Overall | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | SuperGLUE | Reading | AoA leaderboard | AoA raw | ready |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name in EXPECTED:
        row = rows.get(name)
        if not row:
            lines.append(f"| {name} | pending | | | | | | | | | | | |")
            continue
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            name, r(row.get("Overall"),4), r(row.get("BLiMP"),2), r(row.get("Supplement"),2), r(row.get("EWoK"),2),
            r(row.get("Entity"),2), r(row.get("COMPS"),2), r(row.get("GlobalPIQA"),2), r(row.get("SuperGLUE"),4),
            r(row.get("Reading"),2), r(row.get("AoA"),4), r(row.get("aoa_raw_correlation"),6), row.get("submit_ready_overall"),
        ))
    return lines


def main() -> None:
    rows = {name: load_target(name) for name in EXPECTED}
    contrasts = {}
    for cname, a, b, desc in CONTRASTS:
        contrasts[cname] = {"a": a, "b": b, "description": desc, "delta": delta(rows, a, b)}
    interp = interpretation(contrasts)
    payload = {
        "status": "CONTROL_EVAL_SUMMARY",
        "targets": rows,
        "contrasts": contrasts,
        "interpretation": interp,
          "comparison_standard": "Use complete nine-column official-style local evaluation with AoA in leaderboard units (100*raw correlation); controls determine mechanism, not final SOTA. Visible leaderboard reference remains 41.8.",
    }
    _public_path('experiments/archive/compact_experience/data').mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research clean-Qwen control and replication evaluation", "", f"Summary JSON: `{OUT}`", "", "## Targets"]
    lines += md_table(rows)
    lines += ["", "## Contrasts", ""]
    for cname, rec in contrasts.items():
        d = rec["delta"]
        if d is None:
            lines.append(f"- `{cname}` pending: {rec['description']}")
        else:
            lines.append(f"- `{cname}` Overall Δ={r(d.get('Overall'),4)}; task Δs: " + ", ".join(f"{k}={r(d.get(k),3)}" for k in TASK_KEYS))
    lines += ["", "## Current interpretation", "", "```json", json.dumps(interp, indent=2), "```", ""]
    _public_path('research/notes/compact_experience').mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"summary": str(OUT), "note": str(NOTE), "complete_targets": [k for k,v in rows.items() if v and v.get("submit_ready_overall")], "interpretation": interp}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
