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
OUT = _public_path('experiments/archive/compact_experience/data/full_overall_eval')
PER = _public_path('experiments/archive/compact_experience/data/full_overall_eval/per_target')
SUMMARY = _public_path('experiments/archive/compact_experience/data/full_overall_eval/full_overall_summary.json')
NOTE = _public_path('research/notes/compact_experience/full_overall_eval.md')
LEADER = {
    "BLiMP": 67.20,
    "Supplement": 56.01,
    "EWoK": 56.07,
    "Entity": 28.45,
    "COMPS": 53.57,
    "GlobalPIQA": 39.67,
    "SuperGLUE": 69.79,
    "Reading": 5.42,
    "AoA": 0.00,
    "Overall": 41.80,
}
INHERITED = {
    "source": "experiments/archive/initial_model_studies/data/current_best_internal_coordinate.json",
    "model": "wwm_seed43 chck_80M",
    "BLiMP": 66.54,
    "Supplement": 61.00,
    "EWoK": 50.44,
    "Entity": 22.20,
    "COMPS": 53.00,
    "GlobalPIQA": 37.59,
    "SuperGLUE": 68.2550866974358,
    "Reading": 7.30,
    "AoA": 0.00,
    "Overall": 40.7027874108262,
}
KEYS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "SuperGLUE", "Reading", "AoA"]


def round_or_none(x: Any, nd: int = 4):
    if x is None:
        return None
    try:
        return round(float(x), nd)
    except Exception:
        return x


def main() -> None:
    rows: Dict[str, Dict[str, Any]] = {}
    if PER.exists():
        for p in sorted(PER.glob("*.json")):
            data = json.loads(p.read_text(encoding="utf-8"))
            name = p.stem
            oo = data.get("official_overall") or {}
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
                "model_path": data.get("model_path"),
            })
            rows[name] = row
    best = None
    completed = {k: v for k, v in rows.items() if v.get("Overall") is not None}
    if completed:
        best = max(completed.items(), key=lambda kv: float(kv[1]["Overall"]))
    payload = {
        "status": "FULL_OVERALL_SUMMARY",
          "note": "Full official-style local evaluation for selected candidates. AoA is in leaderboard units (100*raw correlation). If AoA is recorded as not_official_missing_checkpoints, Overall uses AoA=0.0 only as a provisional leaderboard-style arithmetic value and the artifact is not submit-ready for official AoA.",
        "leader_reference": LEADER,
        "inherited_reference": INHERITED,
        "targets": rows,
        "best_by_provisional_overall": {"target": best[0], "row": best[1]} if best else None,
    }
    if best:
        payload["best_vs_leader"] = round(float(best[1]["Overall"]) - LEADER["Overall"], 6)
        payload["best_vs_inherited"] = round(float(best[1]["Overall"]) - INHERITED["Overall"], 6)
    OUT.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _public_path('research/notes/compact_experience').mkdir(parents=True, exist_ok=True)
    lines = [
        "# research — Full official-style Overall evaluation", "",
        f"Summary JSON: `{SUMMARY}`", "",
          "Important: AoA column below is the leaderboard score (`100 * raw correlation`). Rows with `aoa_status = not_official_missing_checkpoints` use AoA=0.0 only for immediate arithmetic; they are not submit-ready official AoA artifacts.", "",
          "| target | Overall | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | SuperGLUE | Reading | AoA leaderboard | AoA raw | submit-ready AoA |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, row in sorted(rows.items()):
          lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
              name,
              round_or_none(row.get("Overall")), round_or_none(row.get("BLiMP"), 2), round_or_none(row.get("Supplement"), 2),
              round_or_none(row.get("EWoK"), 2), round_or_none(row.get("Entity"), 2), round_or_none(row.get("COMPS"), 2),
              round_or_none(row.get("GlobalPIQA"), 2), round_or_none(row.get("SuperGLUE"), 4), round_or_none(row.get("Reading"), 2),
              round_or_none(row.get("AoA"), 4), round_or_none(row.get("aoa_raw_correlation"), 6), row.get("submit_ready_aoa"),
          ))
    lines.extend(["", "## References", "", f"Leader: Overall {LEADER['Overall']}", f"Inherited internal coordinate: Overall {INHERITED['Overall']:.4f}"])
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(SUMMARY), "note": str(NOTE), "targets": list(rows), "best": payload.get("best_by_provisional_overall")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
