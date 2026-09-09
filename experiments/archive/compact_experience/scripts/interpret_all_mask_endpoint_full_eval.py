#!/usr/bin/env python3
"""Interpret all available research/048 mask endpoint full-eval payloads.

Keeps official Overall separate from no-AoA equal7. It summarizes whether each
endpoint's failure or success comes from SuperGLUE/AoA and computes the AoA-zero
counterfactual using only aggregate task scores. It never reads AoA/CDI item text
or curve internals.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import pathlib
import re
from typing import Any

ROOT = _public_path('experiments/archive/compact_experience')
SUMMARY = _public_path('experiments/archive/compact_experience/data/mask_endpoint_full_eval/mask_endpoint_full_eval_summary.json')
PER = _public_path('experiments/archive/compact_experience/data/mask_endpoint_full_eval/per_target')
OUT = _public_path('experiments/archive/compact_experience/data/all_mask_endpoint_interpretation/all_mask_endpoint_interpretation.json')
NOTE = _public_path('research/notes/compact_experience/all_mask_endpoint_interpretation.md')
VISIBLE = 41.8
CLEAN = {
    "BLiMP": 66.84,
    "Supplement": 62.84,
    "EWoK": 50.19,
    "Entity": 25.76,
    "COMPS": 51.78,
    "GlobalPIQA": 36.62,
    "SuperGLUE": 70.30861598316157,
    "Reading": 7.76,
    "AoA": 0.0,
    "Overall": 41.34429066479573,
}
ORDER = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "SuperGLUE", "GlobalPIQA", "Reading", "AoA"]
NOAOA7 = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]


def parse_arm(target: str | None) -> str:
    if not target:
        return "unknown"
    m = re.match(r"mask_(.*)_\d+M$", target)
    return m.group(1) if m else target


def endpoint_m(endpoint: str | None) -> int | None:
    if endpoint and endpoint.startswith("chck_") and endpoint.endswith("M"):
        try:
            return int(endpoint[5:-1])
        except Exception:
            return None
    return None


def mean(vals: list[float]) -> float:
    return sum(vals) / len(vals)


def recompute_from_payload(path: pathlib.Path) -> dict[str, Any] | None:
    obj = json.loads(path.read_text(encoding="utf-8"))
    tasks = obj.get("tasks", {})
    if "SuperGLUE" not in tasks or "AoA" not in tasks:
        return None
    scores: dict[str, float] = {}
    for c in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]:
        rec = tasks.get(c, {})
        if rec.get("score") is None:
            return None
        scores[c] = float(rec["score"])
    if tasks.get("GlobalPIQA", {}).get("score") is not None:
        scores["GlobalPIQA"] = float(tasks["GlobalPIQA"]["score"])
    else:
        gp = []
        for sub in ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]:
            if tasks.get(sub, {}).get("score") is None:
                return None
            gp.append(float(tasks[sub]["score"]))
        scores["GlobalPIQA"] = mean(gp)
    reading = tasks.get("Reading", {}).get("scores", {}).get("Reading")
    if reading is None:
        return None
    scores["Reading"] = float(reading)
    sg = tasks.get("SuperGLUE", {})
    scores["SuperGLUE"] = float(sg.get("superglue_mean"))
    aoa = tasks.get("AoA", {})
    scores["AoA"] = float(aoa.get("aoa_leaderboard_score", 100 * float(aoa.get("aoa_official", 0.0))))
    overall = mean([scores[c] for c in ORDER])
    endpoint = obj.get("endpoint")
    epm = endpoint_m(endpoint)
    return {
        "target": obj.get("target"),
        "arm": parse_arm(obj.get("target")),
        "endpoint": endpoint,
        "endpoint_m": epm,
        "endpoint_frozen": epm is not None and epm < 100,
        "submit_ready_overall": obj.get("official_overall", {}).get("submit_ready_overall"),
        "scores": scores,
        "overall": overall,
        "overall_if_aoa_zero": mean([scores[c] if c != "AoA" else 0.0 for c in ORDER]),
        "aoa_penalty_vs_zero": overall - mean([scores[c] if c != "AoA" else 0.0 for c in ORDER]),
        "aoa_raw": scores["AoA"] / 100.0,
        "noaoa_equal7": mean([scores[c] for c in NOAOA7]),
        "nlp8_excluding_aoa": mean([scores[c] for c in ORDER if c != "AoA"]),
        "needed_aoa_lb_for_41p8": 9 * VISIBLE - sum(scores[c] for c in ORDER if c != "AoA"),
        "delta_vs_visible": overall - VISIBLE,
        "delta_vs_clean_overall": overall - CLEAN["Overall"],
        "delta_scores_vs_clean": {c: scores[c] - CLEAN[c] for c in ORDER},
        "payload_path": str(path),
    }


def load_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if PER.exists():
        for p in sorted(PER.glob("*.json")):
            r = recompute_from_payload(p)
            if r is not None:
                rows.append(r)
    # If scorer summary exists, prefer its submit-ready flags when present.
    if SUMMARY.exists():
        summ = json.loads(SUMMARY.read_text(encoding="utf-8"))
        by_target = {r.get("target"): r for r in rows}
        for s in summ.get("candidates", []):
            t = s.get("target")
            if t in by_target:
                by_target[t]["submit_ready_overall"] = s.get("submit_ready_overall")
                by_target[t]["endpoint_frozen"] = s.get("endpoint_frozen")
        rows = list(by_target.values())
    rows.sort(key=lambda r: r["overall"], reverse=True)
    return rows


def main() -> None:
    rows = load_rows()
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_arm.setdefault(r["arm"], []).append(r)
    best_submit = [r for r in rows if r.get("submit_ready_overall")]
    best_true100 = [r for r in rows if r.get("endpoint_m") == 100]
    interpretation = {
        "best_available": rows[0] if rows else None,
        "best_submit_ready": best_submit[0] if best_submit else None,
        "best_true100": best_true100[0] if best_true100 else None,
        "arms_present": sorted(by_arm),
        "active_target": "Find a target distribution or continuation dynamic that preserves the no-AoA gains without producing strongly negative AoA; do not replicate a route whose complete Overall is far below clean-Qwen.",
    }
    payload = {
        "status": "ALL_MASK_ENDPOINT_FULL_EVAL_INTERPRETATION",
        "visible_leader": VISIBLE,
        "clean_reference": CLEAN,
        "rows": rows,
        "by_arm_targets": {arm: [r["target"] for r in rs] for arm, rs in by_arm.items()},
        "interpretation": interpretation,
        "non_leakage_statement": "Uses aggregate completed task scores only; no AoA/CDI item words, child curves, or per-word curve internals are used for training design.",
    }
    _public_path('experiments/archive/compact_experience/data/all_mask_endpoint_interpretation').mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research all mask endpoint full-eval interpretation", "",
        f"Visible leader: {VISIBLE:.3f}; clean-Qwen reference: {CLEAN['Overall']:.6f}.", "",
        "Official Overall includes AoA in leaderboard units. no-AoA equal7 is shown only to interpret the source of changes.", "",
        "| target | endpoint | ready | Overall | Overall if AoA=0 | AoA lb | AoA raw | SuperGLUE | no-AoA equal7 | needed AoA lb | Δleader |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(f"| {r['target']} | {r['endpoint']} | {r.get('submit_ready_overall')} | {r['overall']:.6f} | {r['overall_if_aoa_zero']:.6f} | {r['scores']['AoA']:.3f} | {r['aoa_raw']:.4f} | {r['scores']['SuperGLUE']:.3f} | {r['noaoa_equal7']:.4f} | {r['needed_aoa_lb_for_41p8']:+.3f} | {r['delta_vs_visible']:+.4f} |")
    lines.extend(["", "## Interpretation", ""])
    if rows:
        lines.append(f"- Best available endpoint: `{rows[0]['target']}` Overall {rows[0]['overall']:.6f}.")
    if best_submit:
        lines.append(f"- Best submit-ready endpoint in this family: `{best_submit[0]['target']}` Overall {best_submit[0]['overall']:.6f}.")
    if best_true100:
        lines.append(f"- Best true-100M endpoint: `{best_true100[0]['target']}` Overall {best_true100[0]['overall']:.6f}.")
    lines.append("- If no endpoint has non-negative AoA with strong no-AoA columns, the masking-tail family is not a current SOTA route; use it as a mechanism clue for later AoA-safe continuation design.")
    lines.append("")
    _public_path('research/notes/compact_experience').mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "note": str(NOTE), "n_rows": len(rows), "best": {k: rows[0].get(k) for k in ["target", "overall", "scores"]} if rows else None}, indent=2), flush=True)


if __name__ == "__main__":
    main()
