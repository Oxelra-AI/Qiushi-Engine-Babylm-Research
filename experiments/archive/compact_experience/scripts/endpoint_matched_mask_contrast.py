#!/usr/bin/env python3
"""Endpoint-matched (chck_100M) inverse-vs-uniform masking contrast analysis.

The relevant scientific contrast is
NOT delta-vs-clean at per-arm-selected best checkpoints, but the exposure-matched
arm-minus-uniform contrast at a single common endpoint, and that most of the
apparent effect may live in the high-variance GlobalPIQA column. This script uses
the already-computed no-AoA trajectory tables for all four research masking arms to
compute, without new training:

  1. equal7 (7 no-AoA columns) at every checkpoint for each arm.
  2. equal6 = equal7 with GlobalPIQA removed, to test whether any mask-target
     signal survives outside the noisiest column.
  3. arm-minus-uniform contrasts at the exposure-matched chck_100M endpoint,
     both equal7 and equal6, plus per-column deltas.
  4. per-arm best-checkpoint-selection inflation: best_equal7 - chck100_equal7.

All inputs are post-training no-AoA measurements; no downstream labels/items or
AoA/CDI material are used. equal7/equal6 are internal no-AoA screening metrics on
a different scale than the official SuperGLUE+AoA Overall; they are never mixed
with the official 41.x numbers.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import pathlib

ROOT = _public_path('experiments/archive/compact_experience')
TRAJ_DIR = _public_path('experiments/archive/compact_experience/../staging/processing/mask_noaoa_eval')
TRAJ_DIR = _public_path('experiments/archive/compact_experience/../staging/processing/mask_noaoa_eval')
OUT = _public_path('experiments/archive/compact_experience/data/endpoint_matched_mask_contrast/endpoint_matched_mask_contrast.json')
NOTE = _public_path('research/notes/compact_experience/endpoint_matched_mask_contrast.md')

COLUMNS7 = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
COLUMNS6 = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "Reading"]
CKPTS = ["chck_85M", "chck_90M", "chck_95M", "chck_100M"]
ARMS = {
    "uniform_control": "mask_uniform_control_trajectory_summary.json",
    "inverse_priority": "mask_inverse_priority_trajectory_summary.json",
    "evidence_visible": "mask_evidence_visible_trajectory_summary.json",
    "random_priority": "mask_random_priority_trajectory_summary.json",
}
# Clean-Qwen seed43022 chck_100M no-AoA reference columns (from research mask summary reference).
CLEAN = {
    "BLiMP": 66.84, "Supplement": 62.84, "EWoK": 50.19, "Entity": 25.76,
    "COMPS": 51.78, "GlobalPIQA": 36.62, "Reading": 7.76,
}


def eq(cols: list[str], row: dict) -> float:
    return sum(float(row[c]) for c in cols) / len(cols)


def load_arm(fn: str) -> dict:
    obj = json.loads((TRAJ_DIR / fn).read_text(encoding="utf-8"))
    table = obj["table"]
    out = {}
    for ck in CKPTS:
        if ck not in table:
            continue
        row = table[ck]
        out[ck] = {
            "row": {c: float(row[c]) for c in COLUMNS7},
            "equal7": eq(COLUMNS7, row),
            "equal6": eq(COLUMNS6, row),
        }
    return out


def main() -> None:
    arms = {arm: load_arm(fn) for arm, fn in ARMS.items()}
    clean_eq7 = eq(COLUMNS7, CLEAN)
    clean_eq6 = eq(COLUMNS6, CLEAN)

    # Best-checkpoint selection inflation per arm.
    selection = {}
    for arm, ck_data in arms.items():
        best_ck = max(ck_data, key=lambda c: ck_data[c]["equal7"])
        c100 = ck_data.get("chck_100M")
        selection[arm] = {
            "best_checkpoint_by_equal7": best_ck,
            "best_equal7": ck_data[best_ck]["equal7"],
            "chck_100M_equal7": c100["equal7"] if c100 else None,
            "selection_inflation_equal7": (ck_data[best_ck]["equal7"] - c100["equal7"]) if c100 else None,
        }

    # Exposure-matched chck_100M contrasts vs uniform and vs clean.
    uni100 = arms["uniform_control"]["chck_100M"]
    endpoint_matched = {}
    for arm, ck_data in arms.items():
        a100 = ck_data.get("chck_100M")
        if not a100:
            continue
        per_col_vs_uni = {c: a100["row"][c] - uni100["row"][c] for c in COLUMNS7}
        per_col_vs_clean = {c: a100["row"][c] - CLEAN[c] for c in COLUMNS7}
        endpoint_matched[arm] = {
            "equal7": a100["equal7"],
            "equal6": a100["equal6"],
            "equal7_minus_uniform": a100["equal7"] - uni100["equal7"],
            "equal6_minus_uniform": a100["equal6"] - uni100["equal6"],
            "equal7_minus_clean": a100["equal7"] - clean_eq7,
            "equal6_minus_clean": a100["equal6"] - clean_eq6,
            "per_column_minus_uniform": per_col_vs_uni,
            "per_column_minus_clean": per_col_vs_clean,
        }

    # Same contrasts at each arm's own best checkpoint (for comparison with prior reporting).
    best_matched = {}
    for arm, ck_data in arms.items():
        best_ck = selection[arm]["best_checkpoint_by_equal7"]
        a = ck_data[best_ck]
        best_matched[arm] = {
            "checkpoint": best_ck,
            "equal7": a["equal7"],
            "equal6": a["equal6"],
            "equal7_minus_uniform_best": a["equal7"] - arms["uniform_control"][selection["uniform_control"]["best_checkpoint_by_equal7"]]["equal7"],
        }

    payload = {
        "status": "ENDPOINT_MATCHED_MASK_CONTRAST",
        "note": "equal7/equal6 are internal no-AoA screening metrics, not official SuperGLUE+AoA Overall.",
        "clean_reference": {"columns": CLEAN, "equal7": clean_eq7, "equal6": clean_eq6},
        "per_checkpoint": {arm: {ck: {"equal7": d["equal7"], "equal6": d["equal6"]} for ck, d in ck_data.items()} for arm, ck_data in arms.items()},
        "best_checkpoint_selection": selection,
        "endpoint_matched_chck_100M": endpoint_matched,
        "best_checkpoint_matched": best_matched,
        "key_findings": {
            "uniform_restart_gain_equal7_vs_clean_at_100M": endpoint_matched["uniform_control"]["equal7_minus_clean"],
            "inverse_minus_uniform_equal7_at_100M": endpoint_matched["inverse_priority"]["equal7_minus_uniform"],
            "inverse_minus_uniform_equal6_at_100M": endpoint_matched["inverse_priority"]["equal6_minus_uniform"],
            "inverse_minus_uniform_globalpiqa_at_100M": endpoint_matched["inverse_priority"]["per_column_minus_uniform"]["GlobalPIQA"],
            "evidence_minus_uniform_equal7_at_100M": endpoint_matched["evidence_visible"]["equal7_minus_uniform"],
            "evidence_minus_uniform_equal6_at_100M": endpoint_matched["evidence_visible"]["equal6_minus_uniform"],
        },
        "non_leakage_statement": "Uses only post-training no-AoA official-style column measurements; no downstream labels/items and no AoA/CDI material.",
    }
    _public_path('experiments/archive/compact_experience/data/endpoint_matched_mask_contrast').mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# research endpoint-matched (chck_100M) masking contrast", "",
        "equal7/equal6 are internal no-AoA screening metrics (different scale from official Overall).", "",
        f"Clean-Qwen chck_100M reference: equal7 {clean_eq7:.4f}, equal6 {clean_eq6:.4f}.", "",
        "## Per-checkpoint equal7 (all arms)", "",
        "| arm | chck_85M | chck_90M | chck_95M | chck_100M | best ck | selection inflation |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for arm, ck_data in arms.items():
        vals = [f"{ck_data[c]['equal7']:.4f}" if c in ck_data else "-" for c in CKPTS]
        s = selection[arm]
        lines.append(f"| {arm} | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} | {s['best_checkpoint_by_equal7']} | {s['selection_inflation_equal7']:+.4f} |")
    lines.extend(["", "## Exposure-matched chck_100M contrasts vs uniform_control", "",
                  "| arm | equal7 | Δeq7 vs uni | equal6 (no GPIQA) | Δeq6 vs uni | ΔGPIQA vs uni |",
                  "|---|---:|---:|---:|---:|---:|"])
    for arm, d in endpoint_matched.items():
        lines.append(f"| {arm} | {d['equal7']:.4f} | {d['equal7_minus_uniform']:+.4f} | {d['equal6']:.4f} | {d['equal6_minus_uniform']:+.4f} | {d['per_column_minus_uniform']['GlobalPIQA']:+.3f} |")
    lines.extend(["", "## Inverse chck_100M per-column deltas vs uniform_control", ""])
    for c in COLUMNS7:
        lines.append(f"- {c}: {endpoint_matched['inverse_priority']['per_column_minus_uniform'][c]:+.3f}")
    lines.extend(["", "## Interpretation", "",
                  f"- Uniform restart alone gains equal7 {endpoint_matched['uniform_control']['equal7_minus_clean']:+.4f} vs clean at 100M.",
                  f"- Inverse minus uniform at matched 100M: equal7 {endpoint_matched['inverse_priority']['equal7_minus_uniform']:+.4f}, equal6 {endpoint_matched['inverse_priority']['equal6_minus_uniform']:+.4f}.",
                  f"- Inverse minus uniform GlobalPIQA at 100M: {endpoint_matched['inverse_priority']['per_column_minus_uniform']['GlobalPIQA']:+.3f}.",
                  "- If equal6 (GlobalPIQA-excluded) contrast is near zero, the endpoint-matched mask-target effect is carried mainly by the high-variance GlobalPIQA column.",
                  ""])
    _public_path('research/notes/compact_experience').mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload["key_findings"], indent=2))
    print("OUT", str(OUT))
    print("NOTE", str(NOTE))


if __name__ == "__main__":
    main()
