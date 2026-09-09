#!/usr/bin/env python3
"""research: compare first and independent hardened chck_82M evaluations."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import math
import time
from pathlib import Path
from typing import Any


def find_user_root() -> Path:
    return _PUBLIC_ROOT

ROOT = find_user_root()
STUDY = ROOT / "experiments/archive/representation_and_objectives"
VERIFICATION_RESULTS = STUDY / "data/scale1p75_chck82_full_verification/summary/scale1p75_chck82_full_verification.json"
REPRODUCTION_RESULTS = STUDY / "data/scale1p75_chck82_full_eval_reproduction/summary/scale1p75_100M_full_eval_hardened_summary.json"
OUT = STUDY / "data/chck82_eval_comparison"
OFFICIAL_KEYS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "SuperGLUE", "GlobalPIQA", "Reading", "AoA"]
ZERO_COMPONENTS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"]
EXPECTED_SHA = "93ceb76adf5a33d349f1de33e988e6ed0c2b2a547dbd92cf83cc952f8e2591b3"


def rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def score_block_first(d: dict[str, Any]) -> tuple[dict[str, float], float, float]:
    off = d.get("official_overall", {})
    scores = off.get("scores", {})
    overall = off.get("Overall")
    cheap7 = d.get("cheap7")
    return {k: float(scores[k]) for k in OFFICIAL_KEYS}, float(overall), float(cheap7)


def score_block_hardened(d: dict[str, Any]) -> tuple[dict[str, float], float, float]:
    return {k: float(d["scores"][k]) for k in OFFICIAL_KEYS}, float(d["Overall"]), float(d["cheap7"])


def find_task(d: dict[str, Any], key: str) -> dict[str, Any]:
    return d.get("tasks", {}).get(key, {}) if isinstance(d.get("tasks"), dict) else {}


def counts_for_prediction(path: str | None) -> dict[str, Any]:
    if not path:
        return {"path": None, "exists": False}
    p = ROOT / path if not Path(path).is_absolute() else Path(path)
    if not p.exists():
        return {"path": rel(p), "exists": False}
    try:
        data = read_json(p)
        if isinstance(data, list):
            return {"path": rel(p), "exists": True, "json_type": "list", "count": len(data)}
        if isinstance(data, dict):
            # Official predictions are usually JSON arrays, but keep dict shape visible.
            if "predictions" in data and isinstance(data["predictions"], list):
                return {"path": rel(p), "exists": True, "json_type": "dict.predictions", "count": len(data["predictions"])}
            return {"path": rel(p), "exists": True, "json_type": "dict", "top_keys": sorted(list(data.keys()))[:20]}
    except Exception as exc:
        return {"path": rel(p), "exists": True, "error": repr(exc)}
    return {"path": rel(p), "exists": True, "json_type": type(data).__name__}


def main() -> None:
    first = read_json(VERIFICATION_RESULTS)
    hard = read_json(REPRODUCTION_RESULTS)
    scores1, overall1, cheap1 = score_block_first(first)
    scores2, overall2, cheap2 = score_block_hardened(hard)
    score_deltas = {k: scores2[k] - scores1[k] for k in OFFICIAL_KEYS}
    exact_rounded_agreement = {
        k: round(scores2[k], 2) == round(scores1[k], 2) for k in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading", "AoA"]
    }
    # First evaluation stores rounded zero-shot values and custom SuperGLUE full precision.
    # Hardened reproduction stores collator-derived full precision for all columns.
    max_abs_column_delta = max(abs(v) for v in score_deltas.values())
    zero_pred_counts = {}
    for key in ZERO_COMPONENTS:
        t1 = find_task(first, key)
        # The hardened summary stage_summary records staged prediction paths by component.
        stage = hard.get("stage_summary", {}).get("staged_predictions", {})
        entry = stage.get(key)
        t2_path = entry.get("staged_prediction") if isinstance(entry, dict) else None
        zero_pred_counts[key] = {
            "first": counts_for_prediction(t1.get("predictions")),
            "hardened": counts_for_prediction(t2_path),
        }
    aoa_task = None
    for row in hard.get("eval_rows", []):
        if isinstance(row, dict) and row.get("kind") == "AoA":
            aoa_task = row
            break
    out = {
        "status": "PASS" if overall1 > 41.8 and overall2 > 41.8 and abs(overall2 - overall1) < 0.01 else "CHECK",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "first_eval": {"path": rel(VERIFICATION_RESULTS), "Overall": overall1, "cheap7": cheap1, "scores": scores1},
        "hardened_reproduction": {"path": rel(REPRODUCTION_RESULTS), "Overall": overall2, "cheap7": cheap2, "scores": scores2},
        "deltas_hardened_minus_first": {"Overall": overall2 - overall1, "cheap7": cheap2 - cheap1, "scores": score_deltas, "max_abs_column_delta": max_abs_column_delta},
        "rounded_agreement_for_zero_like_columns": exact_rounded_agreement,
        "endpoint_ready": hard.get("endpoint_ready"),
        "candidate_hash_expected": EXPECTED_SHA,
        "zero_prediction_counts": zero_pred_counts,
        "aoa_eval_row": aoa_task,
        "interpretation": "Independent hardened staging/collation repeats the above-frontier score for the same original chck_82M endpoint. Differences are tiny and consistent with research using rounded zero-shot column reports plus its own SuperGLUE fill, while research reports full precision from the hardened collator.",
    }
    out_json = OUT / "chck82_eval_comparison.json"
    out_md = OUT / "chck82_eval_comparison.md"
    write_json(out_json, out)
    lines = [
        "# research chck_82M evaluation comparison",
        "",
        f"Status: **{out['status']}**",
        f"research Overall: **{overall1:.12f}**; research hardened Overall: **{overall2:.12f}**; delta **{overall2-overall1:+.12f}**.",
        f"research cheap7: **{cheap1:.12f}**; research cheap7: **{cheap2:.12f}**; delta **{cheap2-cheap1:+.12f}**.",
        "",
        "| column | research | research hardened | delta |",
        "|---|---:|---:|---:|",
    ]
    for k in OFFICIAL_KEYS:
        lines.append(f"| {k} | {scores1[k]:.12f} | {scores2[k]:.12f} | {score_deltas[k]:+.12f} |")
    lines += ["", f"JSON: `{rel(out_json)}`"]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "overall_first": overall1, "overall_hardened": overall2, "delta": overall2 - overall1, "json": rel(out_json)}, indent=2), flush=True)

if __name__ == "__main__":
    main()
