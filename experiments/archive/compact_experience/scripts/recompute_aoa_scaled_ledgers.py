#!/usr/bin/env python3
"""research AoA-unit correction for saved BabyLM local full-eval ledgers.

The official BabyLM leaderboard stores AoA in leaderboard units (100 * raw
curve-fitness/correlation). Earlier evaluation scripts used the raw AoA number in
Overall arithmetic. This script patches saved per-target JSONs in place after
backing them up, and writes a before/after audit.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import pathlib
import shutil
from typing import Any

from babylm_official_scoring import patch_payload_official_overall, self_test, leaderboard_overall

ROOT = _public_path('experiments/archive/compact_experience')
OUT = _public_path('experiments/archive/compact_experience/data/aoa_unit_correction')
BACKUP = _public_path('experiments/archive/compact_experience/data/aoa_unit_correction/per_target_backups')
AUDIT = _public_path('experiments/archive/compact_experience/data/aoa_unit_correction/aoa_unit_correction_audit.json')
NOTE = _public_path('research/notes/compact_experience/aoa_unit_correction.md')

PER_TARGET_DIRS = [
    _public_path('experiments/archive/compact_experience/data/full_overall_eval/per_target'),
    _public_path('experiments/archive/compact_experience/data/full_eval/per_target'),
    _public_path('experiments/archive/compact_experience/data/mechanism_eval/per_target'),
]
LEADERBOARD_TOP20 = _public_path('experiments/archive/compact_experience/data/babylm2026_surface/strict_small_top20.json')


def r(x: Any, nd: int = 6):
    if x is None:
        return None
    try:
        return round(float(x), nd)
    except Exception:
        return x


def row_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    oo = payload.get("official_overall") or {}
    scores = oo.get("scores") or {}
    aoa_task = (payload.get("tasks") or {}).get("AoA") or {}
    return {
        "AoA_score_used": scores.get("AoA"),
        "Overall": oo.get("Overall"),
        "Human_like_average": oo.get("Human_like_average"),
        "NLP_average": oo.get("NLP_average"),
        "aoa_status": oo.get("aoa_status") or aoa_task.get("status"),
        "aoa_official_legacy": aoa_task.get("aoa_official"),
        "aoa_raw_correlation": aoa_task.get("aoa_raw_correlation"),
        "aoa_leaderboard_score": aoa_task.get("aoa_leaderboard_score"),
    }


def patch_file(path: pathlib.Path) -> dict[str, Any]:
    rel = path.relative_to(ROOT)
    backup_path = BACKUP / rel
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    if not backup_path.exists():
        shutil.copy2(path, backup_path)
    original = json.loads(path.read_text(encoding="utf-8"))
    before = row_snapshot(original)
    patched = patch_payload_official_overall(original)
    after = row_snapshot(patched)
    path.write_text(json.dumps(patched, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"path": str(path), "backup_path": str(backup_path), "before": before, "after": after, "delta": {
        "AoA_score_used": None if before.get("AoA_score_used") is None or after.get("AoA_score_used") is None else r(float(after["AoA_score_used"]) - float(before["AoA_score_used"])),
        "Overall": None if before.get("Overall") is None or after.get("Overall") is None else r(float(after["Overall"]) - float(before["Overall"])),
        "Human_like_average": None if before.get("Human_like_average") is None or after.get("Human_like_average") is None else r(float(after["Human_like_average"]) - float(before["Human_like_average"])),
    }}


def leaderboard_regressions() -> dict[str, Any]:
    out = {"self_test": self_test(), "top20_checks": []}
    if LEADERBOARD_TOP20.exists():
        rows = json.loads(LEADERBOARD_TOP20.read_text(encoding="utf-8"))
        for row in rows:
            if row.get("AoA") not in (None, 0, 0.0):
                calc = leaderboard_overall(row)
                out["top20_checks"].append({
                    "Model_plain": row.get("Model_plain"),
                    "reported_overall": row.get("Overall Average"),
                    "calculated_from_leaderboard_units": calc,
                    "AoA_leaderboard_units": row.get("AoA"),
                    "abs_error": None if calc is None else abs(float(calc) - float(row.get("Overall Average"))),
                })
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    patched = []
    for d in PER_TARGET_DIRS:
        if not d.exists():
            continue
        for p in sorted(d.glob("*.json")):
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data.get("tasks"), dict) and "AoA" in data.get("tasks", {}):
                patched.append(patch_file(p))
    regress = leaderboard_regressions()
    # Route leverage calculation.
    qwen_path = _public_path('experiments/archive/compact_experience/data/full_eval/per_target/qwen_clean_aligned.json')
    qwen_leverage = None
    if qwen_path.exists():
        q = json.loads(qwen_path.read_text(encoding="utf-8"))
        oo = q.get("official_overall") or {}
        scores = dict(oo.get("scores") or {})
        if all(scores.get(k) is not None for k in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "SuperGLUE", "GlobalPIQA", "Reading", "AoA"]):
            current = float(oo["Overall"])
            scores["AoA"] = 22.9
            what_if = sum(float(scores[k]) for k in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "SuperGLUE", "GlobalPIQA", "Reading", "AoA"]) / 9.0
            qwen_leverage = {"current_overall": current, "what_if_raw_aoa_0p229_leaderboard_22p9": what_if, "delta": what_if - current}
    payload = {
        "status": "AOA_UNIT_CORRECTION_DONE",
        "regression_tests": regress,
        "patched_files": patched,
        "qwen_aoa_leverage_example": qwen_leverage,
        "rule": "AoA evaluator raw curve_fitness/correlation is saved as aoa_raw_correlation; leaderboard score and Overall use 100*raw. Missing AoA remains leaderboard 0.0 and raw None.",
    }
    AUDIT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# research AoA unit correction",
        "",
        f"Audit JSON: `{AUDIT}`",
        "",
        "Official leaderboard arithmetic uses AoA in leaderboard units (`100 * raw correlation`). The previous local ledgers used raw AoA directly in Overall; this note records the corrected state.",
        "",
        "## Regression",
        "",
        "```json",
        json.dumps(regress, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Patched rows",
        "",
        "| file | AoA before | AoA after | Overall before | Overall after | ΔOverall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for rec in patched:
        before, after, delta = rec["before"], rec["after"], rec["delta"]
        lines.append(f"| `{rec['path']}` | {r(before.get('AoA_score_used'),4)} | {r(after.get('AoA_score_used'),4)} | {r(before.get('Overall'),4)} | {r(after.get('Overall'),4)} | {r(delta.get('Overall'),4)} |")
    if qwen_leverage:
        lines += ["", "## AoA leverage for current clean-Qwen endpoint", "", "```json", json.dumps(qwen_leverage, indent=2), "```"]
    _public_path('research/notes/compact_experience').mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"audit": str(AUDIT), "note": str(NOTE), "patched": len(patched), "qwen_leverage": qwen_leverage, "regression_top20_checks": regress.get("top20_checks")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
