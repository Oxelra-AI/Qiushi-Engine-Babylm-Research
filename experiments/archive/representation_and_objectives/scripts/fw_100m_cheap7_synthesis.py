#!/usr/bin/env python3
"""research: complete 100M cheap7 synthesis for the three FW shared-anchor arms.

This combines the just-measured 100M non-EWoK/non-AoA surface columns
(BLiMP, Supplement, Entity, COMPS, GlobalPIQA, Reading) with the current-coordinate
official 7,618-row EWoK to form cheap7 for compact, row-block breadth, and
interleaved breadth. It does NOT run any model; it reads saved eval artifacts.

Scientific boundary: this is a cheap decision readout. Full official-compatible
Overall requires SuperGLUE (primary metrics) and min-context-zero AoA, which are
NOT run here. cheap7 is used only to decide whether any single FW arm merits full
evaluation, following the standing SOTA-plausibility arithmetic.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

WS = Path("experiments/archive/representation_and_objectives")
FULL_EVAL_ROOT = WS / "data/fw_shared_anchor_full_eval/per_target"
EWOK_ROOT = WS / "data/fw_shared_anchor_official_ewok"
OUT = WS / "data/fw_100m_cheap7_synthesis"
NOTE = (WS.parents[2] / 'research/notes/representation_and_objectives/fw_100m_cheap7_synthesis.md')

CHEAP7_COLS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]

# Non-EWoK/non-AoA surface columns measured this step (research), read verbatim from run stdout.
# Cross-checked against the per_target JSON below; if the JSON exists it overrides these.
SURFACE_FALLBACK = {
    "fw_compact_fullbatch_seed43022": {
        "BLiMP": 66.87, "Supplement": 58.86, "Entity": 28.36, "COMPS": 51.84,
        "GlobalPIQA_parallel": 24.27, "GlobalPIQA_nonparallel": 53.0, "GlobalPIQA": 38.635, "Reading": 7.455,
    },
    "fw_breadth_rowblock_fullbatch_seed43022": {
        "BLiMP": 67.9, "Supplement": 59.69, "Entity": 23.88, "COMPS": 51.33,
        "GlobalPIQA_parallel": 29.13, "GlobalPIQA_nonparallel": 45.0, "GlobalPIQA": 37.065, "Reading": 8.105,
    },
    "fw_breadth_interleaved_fullbatch_seed43022": {
        "BLiMP": 66.89, "Supplement": 56.42, "Entity": 25.6, "COMPS": 51.75,
        "GlobalPIQA_parallel": 26.21, "GlobalPIQA_nonparallel": 56.0, "GlobalPIQA": 41.105, "Reading": 7.94,
    },
}

ARMS = {
    "fw_compact_fullbatch_seed43022": "compact (same-proposition recurrence anchor)",
    "fw_breadth_rowblock_fullbatch_seed43022": "row-block whole-sentence breadth",
    "fw_breadth_interleaved_fullbatch_seed43022": "interleaved whole-sentence breadth",
}

VISIBLE_LEADER = {"Overall": 41.80, "cheap7": 43.77, "SuperGLUE": 69.79, "AoA": 0.0}


def load_surface(target: str) -> dict[str, Any]:
    # Prefer the per_target JSON if present; else use the recorded fallback.
    cands = list(FULL_EVAL_ROOT.glob(f"*{target}*.json")) if FULL_EVAL_ROOT.exists() else []
    scores = dict(SURFACE_FALLBACK[target])
    src = "recorded_stdout_fallback"
    for p in cands:
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        s = None
        if isinstance(d, dict):
            if isinstance(d.get("scores"), dict):
                s = d.get("scores")
            elif isinstance(d.get("official_overall"), dict) and isinstance(d["official_overall"].get("scores"), dict):
                s = d["official_overall"].get("scores")
        if isinstance(s, dict) and s.get("BLiMP") is not None:
            for k in ["BLiMP", "Supplement", "Entity", "COMPS", "GlobalPIQA", "Reading"]:
                if s.get(k) is not None:
                    scores[k] = s[k]
            src = str(p)
            break
    return {"scores": scores, "source": src}


def load_ewok(target: str) -> float | None:
    p = EWOK_ROOT / target / f"official_ewok_reeval_{target}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return d.get("official_ewok_score")


def cheap7(scores: dict[str, Any]) -> float | None:
    vals = [scores.get(c) for c in CHEAP7_COLS]
    if any(v is None for v in vals):
        return None
    return sum(vals) / 7.0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = {}
    for target, label in ARMS.items():
        surf = load_surface(target)
        ewok = load_ewok(target)
        scores = dict(surf["scores"])
        scores["EWoK"] = ewok
        c7 = cheap7(scores)
        # SOTA plausibility: to reach Overall 41.80 with SuperGLUE+AoA at leader 69.79,
        # need cheap7 = (9*41.80 - 69.79)/7.
        need_c7_leader_sg = (9 * VISIBLE_LEADER["Overall"] - VISIBLE_LEADER["SuperGLUE"]) / 7.0
        # Alternatively, given this arm's cheap7, what SuperGLUE+AoA would be needed?
        need_sg_plus_aoa = (9 * VISIBLE_LEADER["Overall"] - 7 * c7) if c7 is not None else None
        rows[target] = {
            "label": label,
            "scores": scores,
            "cheap7": c7,
            "gap_to_leader_cheap7": (c7 - VISIBLE_LEADER["cheap7"]) if c7 is not None else None,
            "needed_cheap7_if_leader_superglue": need_c7_leader_sg,
            "needed_superglue_plus_aoa_for_41p80_given_this_cheap7": need_sg_plus_aoa,
            "surface_source": surf["source"],
        }

    # Column-level deltas breadth arms minus compact.
    compact = rows["fw_compact_fullbatch_seed43022"]["scores"]
    deltas = {}
    for target in ["fw_breadth_rowblock_fullbatch_seed43022", "fw_breadth_interleaved_fullbatch_seed43022"]:
        s = rows[target]["scores"]
        deltas[target] = {c: (s.get(c) - compact.get(c)) if (s.get(c) is not None and compact.get(c) is not None) else None
                          for c in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"]}
        c7b = rows[target]["cheap7"]; c7c = rows["fw_compact_fullbatch_seed43022"]["cheap7"]
        deltas[target]["cheap7"] = (c7b - c7c) if (c7b is not None and c7c is not None) else None

    payload = {
        "status": "FW_100M_CHEAP7_SYNTHESIS",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "boundary": "cheap7 = mean of BLiMP/Supplement/EWoK/Entity/COMPS/GlobalPIQA/Reading; SuperGLUE and AoA not run; not a submission Overall.",
        "visible_leader": VISIBLE_LEADER,
        "arms": rows,
        "breadth_minus_compact": deltas,
    }
    (OUT / "fw_100m_cheap7_synthesis.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["# research — FW 100M cheap7 synthesis (compact vs row-block vs interleaved)\n"]
    lines.append("\ncheap7 = mean(BLiMP, Supplement, EWoK, Entity, COMPS, GlobalPIQA, Reading). SuperGLUE and AoA are NOT run; this is a cheap decision readout, not a submission Overall.\n")
    lines.append("\n| arm | BLiMP | Supplement | EWoK | Entity | COMPS | GlobalPIQA | Reading | cheap7 | gap_to_leader_cheap7 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for target, r in rows.items():
        s = r["scores"]
        lines.append("| {label} | {BLiMP} | {Supplement} | {EWoK} | {Entity} | {COMPS} | {GlobalPIQA} | {Reading} | {c7:.4f} | {gap:+.4f} |\n".format(
            label=r["label"], BLiMP=s["BLiMP"], Supplement=s["Supplement"], EWoK=s["EWoK"], Entity=s["Entity"],
            COMPS=s["COMPS"], GlobalPIQA=s["GlobalPIQA"], Reading=s["Reading"], c7=r["cheap7"], gap=r["gap_to_leader_cheap7"]))
    lines.append("\n## Breadth minus compact (column deltas)\n\n")
    for target, d in deltas.items():
        lines.append(f"- **{ARMS[target]}**: cheap7 {d['cheap7']:+.4f}; "
                     f"BLiMP {d['BLiMP']:+.2f}, Supplement {d['Supplement']:+.2f}, EWoK {d['EWoK']:+.2f}, "
                     f"Entity {d['Entity']:+.2f}, COMPS {d['COMPS']:+.2f}, GlobalPIQA {d['GlobalPIQA']:+.3f} "
                     f"(parallel {d['GlobalPIQA_parallel']:+.2f}, nonparallel {d['GlobalPIQA_nonparallel']:+.2f}), Reading {d['Reading']:+.2f}\n")
    lines.append("\n## SOTA plausibility\n\n")
    lines.append(f"- Leader cheap7 = {VISIBLE_LEADER['cheap7']}; needed cheap7 for Overall 41.80 at leader SuperGLUE 69.79 = {(9*41.80-69.79)/7.0:.4f}.\n")
    for target, r in rows.items():
        lines.append(f"- {r['label']}: cheap7 {r['cheap7']:.4f}; to reach Overall 41.80 would need SuperGLUE+AoA = {r['needed_superglue_plus_aoa_for_41p80_given_this_cheap7']:.2f} (leader SuperGLUE+AoA ≈ 69.79).\n")
    lines.append(f"\nFiles: `{OUT / 'fw_100m_cheap7_synthesis.json'}`\n")
    NOTE.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": payload["status"],
                      "cheap7": {t: rows[t]["cheap7"] for t in rows},
                      "gap_to_leader": {t: rows[t]["gap_to_leader_cheap7"] for t in rows},
                      "note": str(NOTE)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
