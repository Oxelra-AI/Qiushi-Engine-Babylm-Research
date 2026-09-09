#!/usr/bin/env python3
"""Summarize research developmental first-pass evaluation against same-seed clean-Qwen baselines."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json
import pathlib
from typing import Any, Dict

ROOT = _public_path('experiments/archive/compact_experience')
PER = _public_path('experiments/archive/compact_experience/data/devcurr_eval/per_target')
BASE_PER = _public_path('experiments/archive/compact_experience/data/full_eval/per_target')
OUT = _public_path('experiments/archive/compact_experience/data/devcurr_eval/devcurr_eval_summary.json')
NOTE = _public_path('research/notes/compact_experience/devcurr_eval_summary.md')
META = _public_path('experiments/archive/compact_experience/data/aoa_developmental_order/devcurr_materialization_metadata.json')
KEYS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "SuperGLUE", "Reading", "AoA", "Overall", "NLP_average", "Human_like_average"]
TASK_KEYS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "SuperGLUE", "Reading", "AoA"]
TARGETS = {
    "qwen_devcurr_firstpass_seed43022": _public_path('experiments/archive/compact_experience/data/devcurr_eval/per_target/qwen_devcurr_firstpass_seed43022.json'),
    "qwen_devcurr_firstpass_seed43122": _public_path('experiments/archive/compact_experience/data/devcurr_eval/per_target/qwen_devcurr_firstpass_seed43122.json'),
    "qwen_clean_aligned_seed43022": _public_path('experiments/archive/compact_experience/data/full_eval/per_target/qwen_clean_aligned.json'),
    "qwen_clean_aligned_seed43122": _public_path('experiments/archive/compact_experience/data/full_eval/per_target/qwen_clean_aligned_seed43122.json'),
}
PAIRS = [
    ("devcurr_seed43022_minus_clean", "qwen_devcurr_firstpass_seed43022", "qwen_clean_aligned_seed43022"),
    ("devcurr_seed43122_minus_clean", "qwen_devcurr_firstpass_seed43122", "qwen_clean_aligned_seed43122"),
]
LEADER_OVERALL = 41.8


def load_row(name: str) -> Dict[str, Any] | None:
    p = TARGETS[name]
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    oo = d.get("official_overall", {})
    scores = oo.get("scores", {})
    row = {k: scores.get(k) for k in TASK_KEYS}
    for k in ["Overall", "NLP_average", "Human_like_average"]:
        row[k] = oo.get(k)
    row.update({
        "aoa_status": oo.get("aoa_status"),
        "aoa_raw_correlation": oo.get("aoa_raw_correlation"),
        "aoa_leaderboard_score": oo.get("aoa_leaderboard_score"),
        "submit_ready_overall": oo.get("submit_ready_overall"),
        "per_target_json": str(p),
        "model_path": d.get("model_path"),
    })
    return row


def delta(a: Dict[str, Any] | None, b: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if not a or not b:
        return None
    out = {}
    for k in KEYS:
        av, bv = a.get(k), b.get(k)
        out[k] = None if av is None or bv is None else round(float(av) - float(bv), 6)
    # Contributions to official Overall from task groups
    if out.get("Overall") is not None:
        nlp_contrib = sum(out[k] for k in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "SuperGLUE"] if out.get(k) is not None) / 9.0
        reading_contrib = (out.get("Reading") or 0.0) / 9.0
        aoa_contrib = (out.get("AoA") or 0.0) / 9.0
        out["Overall_contrib_from_7NLP"] = round(nlp_contrib, 6)
        out["Overall_contrib_from_Reading"] = round(reading_contrib, 6)
        out["Overall_contrib_from_AoA"] = round(aoa_contrib, 6)
    return out


def r(x: Any, nd: int = 4) -> str:
    if x is None:
        return "NA"
    try:
        return str(round(float(x), nd))
    except Exception:
        return str(x)


def main() -> None:
    rows = {name: load_row(name) for name in TARGETS}
    contrasts = {}
    for cname, a, b in PAIRS:
        contrasts[cname] = {"a": a, "b": b, "delta": delta(rows.get(a), rows.get(b))}
    for cname, rec in contrasts.items():
        d = rec.get("delta") or {}
        rec["interpretation"] = {
            "beats_clean_same_seed": (d.get("Overall") is not None and d.get("Overall") > 0),
            "beats_visible_leader_absolute": (rows.get(rec["a"]) is not None and rows[rec["a"]].get("Overall") is not None and rows[rec["a"]]["Overall"] > LEADER_OVERALL),
            "aoa_positive_raw": (rows.get(rec["a"]) is not None and (rows[rec["a"]].get("aoa_raw_correlation") or 0.0) > 0),
            "nlp_change": d.get("NLP_average"),
            "aoa_overall_contribution": d.get("Overall_contrib_from_AoA"),
        }
    payload = {
        "status": "DEVCURR_EVAL_SUMMARY",
        "comparison_standard": "corrected local official-style nine-column evaluation; AoA in leaderboard units (100*raw correlation)",
        "visible_leader_reference": {"Overall": LEADER_OVERALL, "AoA": 0.0},
        "materialization_metadata": str(META),
        "targets": rows,
        "contrasts": contrasts,
    }
    _public_path('experiments/archive/compact_experience/data/devcurr_eval').mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research developmental first-pass clean-Qwen evaluation",
        "",
        f"Summary JSON: `{OUT}`",
        "",
        "AoA below is leaderboard units (`100 * raw correlation`). Same-seed contrasts isolate the first-pass order effect relative to the already evaluated clean-Qwen arms.",
        "",
        "| target | Overall | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | SuperGLUE | Reading | AoA lb | AoA raw | ready |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name in TARGETS:
        row = rows.get(name)
        if row is None:
            lines.append(f"| {name} | missing | | | | | | | | | | | |")
        else:
            lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                name, r(row.get("Overall")), r(row.get("BLiMP"),2), r(row.get("Supplement"),2), r(row.get("EWoK"),2),
                r(row.get("Entity"),2), r(row.get("COMPS"),2), r(row.get("GlobalPIQA"),2), r(row.get("SuperGLUE"),4),
                r(row.get("Reading"),2), r(row.get("AoA"),4), r(row.get("aoa_raw_correlation"),6), row.get("submit_ready_overall"),
            ))
    lines += ["", "## Same-seed contrasts", ""]
    for cname, rec in contrasts.items():
        d = rec.get("delta")
        if d is None:
            lines.append(f"- `{cname}`: missing")
        else:
            parts = ", ".join(f"{k}={r(d.get(k),3)}" for k in TASK_KEYS)
            lines.append(f"- `{cname}` Overall Δ={r(d.get('Overall'),4)}; task Δs: {parts}; contributions: NLP7={r(d.get('Overall_contrib_from_7NLP'),4)}, Reading={r(d.get('Overall_contrib_from_Reading'),4)}, AoA={r(d.get('Overall_contrib_from_AoA'),4)}")
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(OUT), "note": str(NOTE), "contrasts": contrasts}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
