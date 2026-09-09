#!/usr/bin/env python3
"""Compare seed43122 official-coordinate staged score against seed43022 and the leader.

Run this only after:
  - full seed43122 evaluation is delivered,
  - official EWoK/AoA coordinate repairs are delivered,
  - stage_pristine_collate_seed43122.py has produced a staged official-coordinate summary.

If the staged summary is still missing, this script writes an explicit waiting record
without reading partial managed outputs.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import hashlib
import json
from pathlib import Path
import time
from typing import Any


def find_user_root() -> Path:
    return _PUBLIC_ROOT

ROOT = find_user_root()
WORKSPACE = ROOT / "experiments/archive/representation_and_objectives"
OUT_DIR = WORKSPACE / "data/seed43122_official_comparison_after_delivery"
NOTE = WORKSPACE / "notes/seed43122_official_comparison_after_delivery.md"
SEED430_SUMMARY = WORKSPACE / "data/pristine_collate_seed43022/pristine_collate_seed43022_summary.json"
SEED431_STAGED = WORKSPACE / "data/pristine_collate_seed43122/pristine_collate_seed43122_summary.json"
PREFLIGHT = WORKSPACE / "data/seed43122_robustness_decision_preflight/seed43122_robustness_decision_preflight.json"
REPAIRS = WORKSPACE / "data/seed43122_official_coordinate_repairs/seed43122_official_coordinate_repairs_summary.json"
VARIANCE = WORKSPACE / "data/seed_variance_fast_and_dynamics_repaired/seed_variance_fast_and_dynamics.json"

OVERALL_KEYS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "SuperGLUE", "GlobalPIQA", "Reading", "AoA"]
NLP_KEYS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "SuperGLUE"]


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else None, "sha256": sha256_file(path)}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_get(obj: Any, dotted: str, default: Any = None) -> Any:
    cur = obj
    for p in dotted.split("."):
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default
    return cur


def official_score_block(summary: dict[str, Any]) -> dict[str, Any]:
    # Both research and research staging scripts use score_summary directly; allow
    # minor historical variants to keep this comparison stable.
    if isinstance(summary.get("score_summary"), dict):
        ss = summary["score_summary"]
        if "official_overall" in ss and isinstance(ss["official_overall"], dict):
            return ss["official_overall"]
        if "scores" in ss and "Overall" in ss:
            return ss
    raise KeyError("Cannot locate official score block in summary")


def compare(seed430: dict[str, Any], seed431: dict[str, Any], variance: dict[str, Any]) -> dict[str, Any]:
    s430 = official_score_block(seed430)
    s431 = official_score_block(seed431)
    scores430 = s430["scores"]
    scores431 = s431["scores"]
    deltas = {k: float(scores431[k]) - float(scores430[k]) for k in OVERALL_KEYS}
    overall430 = float(s430["Overall"])
    overall431 = float(s431["Overall"])
    nlp430 = sum(float(scores430[k]) for k in NLP_KEYS) / len(NLP_KEYS)
    nlp431 = sum(float(scores431[k]) for k in NLP_KEYS) / len(NLP_KEYS)
    return {
        "seed43022_official_overall": overall430,
        "seed43122_official_overall": overall431,
        "seed43122_margin_over_visible_leader_41p8": overall431 - 41.8,
        "seed43122_minus_seed43022_overall": overall431 - overall430,
        "seed43122_minus_seed43022_scores": deltas,
        "seed43022_nlp_average": nlp430,
        "seed43122_nlp_average": nlp431,
        "seed43122_minus_seed43022_nlp_average": nlp431 - nlp430,
        "seed43122_clears_visible_leader_41p8": overall431 >= 41.8,
        "strongest_negative_columns": sorted(deltas.items(), key=lambda kv: kv[1])[:5],
        "strongest_positive_columns": sorted(deltas.items(), key=lambda kv: kv[1], reverse=True)[:5],
        "fast_surface_context": safe_get(variance, "fast_endpoint_delta_431_minus_430", {}),
        "interpretation": (
            "If seed43122 clears 41.8 on this staged official coordinate, compact_view_reinvest has two independent-seed "
            "coordinates above the visible leader and can be frozen before mechanism testing. If it falls below 41.8, "
            "the full component vector should guide seed-sensitivity analysis before any new expensive corpus/recipe branch."
        ),
    }


def write_note(payload: dict[str, Any]) -> None:
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    if payload["status"].endswith("AWAITING_STAGED_SEED43122"):
        lines = [
            "# research — seed43122 official comparison placeholder",
            "",
            "The staged seed43122 official-coordinate summary is not present yet. This file is a non-final waiting record, not a robustness result.",
            "",
            "Required sequence after runtime delivery:",
            "1. Read the completed seed43122 full-evaluation result.",
            "2. Read the completed official EWoK/AoA coordinate-repair result.",
            "3. Run `stage_pristine_collate_seed43122.py` with the repair EWoK predictions path and AoA directory.",
            "4. Rerun this comparison script.",
            "",
            f"JSON: `{payload['out_json']}`",
        ]
    else:
        c = payload["comparison"]
        lines = [
            "# research — seed43122 official-coordinate comparison",
            "",
            f"Seed43022 official Overall: `{c['seed43022_official_overall']}`.",
            f"Seed43122 official Overall: `{c['seed43122_official_overall']}`.",
            f"Seed43122 margin over visible 41.8 leader: `{c['seed43122_margin_over_visible_leader_41p8']}`.",
            f"Seed43122 minus seed43022 Overall: `{c['seed43122_minus_seed43022_overall']}`.",
            f"Clears visible leader: `{c['seed43122_clears_visible_leader_41p8']}`.",
            "",
            "Column deltas seed43122 minus seed43022:",
        ]
        for k, v in c["seed43122_minus_seed43022_scores"].items():
            lines.append(f"- {k}: `{v}`")
        lines += ["", f"JSON: `{payload['out_json']}`"]
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_inputs = {
        "seed43022_summary": file_record(SEED430_SUMMARY),
        "seed43122_staged_summary": file_record(SEED431_STAGED),
        "preflight": file_record(PREFLIGHT),
        "repairs": file_record(REPAIRS),
        "variance": file_record(VARIANCE),
    }
    out_json = OUT_DIR / "seed43122_official_comparison_after_delivery.json"
    if not SEED431_STAGED.exists():
        payload = {
            "status": "SEED43122_OFFICIAL_COMPARISON_AWAITING_STAGED_SEED43122",
            "created_utc": now_utc(),
            "out_json": str(out_json),
            "scientific_purpose": "Avoid using partial managed outputs; compare seed43122 only after staged official coordinate exists.",
            "input_artifacts": base_inputs,
            "missing": [str(SEED431_STAGED)],
        }
        out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        write_note(payload)
        print(json.dumps({"status": payload["status"], "out_json": str(out_json), "note": str(NOTE), "missing": payload["missing"]}, indent=2), flush=True)
        return
    seed430 = load_json(SEED430_SUMMARY)
    seed431 = load_json(SEED431_STAGED)
    variance = load_json(VARIANCE) if VARIANCE.exists() else {}
    payload = {
        "status": "SEED43122_OFFICIAL_COMPARISON_DONE",
        "created_utc": now_utc(),
        "out_json": str(out_json),
        "input_artifacts": base_inputs,
        "comparison": compare(seed430, seed431, variance),
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_note(payload)
    print(json.dumps({
        "status": payload["status"],
        "out_json": str(out_json),
        "note": str(NOTE),
        "overall431": payload["comparison"]["seed43122_official_overall"],
        "margin431": payload["comparison"]["seed43122_margin_over_visible_leader_41p8"],
        "clears_visible_leader": payload["comparison"]["seed43122_clears_visible_leader_41p8"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
