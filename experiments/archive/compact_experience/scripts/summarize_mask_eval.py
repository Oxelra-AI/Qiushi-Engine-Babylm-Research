#!/usr/bin/env python3
"""Summarize research no-AoA masking-continuation trajectories."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import pathlib
import time
from typing import Any

ROOT = _public_path('experiments/archive/compact_experience')
DEFAULT_OUT = _public_path('experiments/archive/compact_experience/data/mask_eval')
DEFAULT_NOTE = _public_path('research/notes/compact_experience/mask_noaoa_result.md')
ARMS = ["uniform_control", "evidence_visible", "random_priority", "inverse_priority"]
TARGETS = ["EWoK", "COMPS", "GlobalPIQA"]
PRESERVE = ["BLiMP", "Supplement", "Entity", "Reading"]
COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
CLEAN_QWEN = {
    "BLiMP": 66.84,
    "Supplement": 62.84,
    "EWoK": 50.19,
    "Entity": 25.76,
    "COMPS": 51.78,
    "GlobalPIQA": 36.62,
    "Reading": 7.76,
    "equal7_full_eval": 43.112857142857145,
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_arm(out_root: pathlib.Path, arm: str) -> dict[str, Any] | None:
    target = f"mask_{arm}"
    path = out_root / f"{target}_trajectory_summary.json"
    if not path.exists():
        nested = out_root / target / f"{target}_trajectory_summary.json"
        if not nested.exists():
            return None
        path = nested
    obj = json.loads(path.read_text(encoding="utf-8"))
    best = obj.get("best_by_equal7_full_eval") or obj.get("best", {})
    row = best.get("row", {})
    if not row:
        return None
    return {
        "arm": arm,
        "best_checkpoint": best.get("checkpoint"),
        "equal7": float(row.get("equal7_full_eval", 0.0)),
        "delta_vs_clean_equal7": float(row.get("equal7_full_eval", 0.0)) - CLEAN_QWEN["equal7_full_eval"],
        "target_recovery_sum_delta_vs_clean": sum(float(row.get(c, 0.0)) - CLEAN_QWEN[c] for c in TARGETS),
        "preserve_sum_delta_vs_clean": sum(float(row.get(c, 0.0)) - CLEAN_QWEN[c] for c in PRESERVE),
        "scores": {c: float(row.get(c, 0.0)) for c in COLUMNS},
        "summary_path": str(path),
    }


def load_metrics(arm: str) -> dict[str, Any] | None:
    path = _public_path('experiments/archive/compact_experience/training/runs') / f"mask_{arm}" / "scientific_metrics.json"
    if not path.exists():
        return None
    m = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": str(path),
        "mask_mode": m.get("mask_mode"),
        "masked_token_budget_ratio": m.get("masked_token_budget_ratio"),
        "high_priority_fraction_among_selected_words": m.get("high_priority_fraction_among_selected_words"),
        "actual_total_word_exposure": m.get("actual_total_word_exposure"),
        "continuation_word_exposure": m.get("continuation_word_exposure", m.get("word_exposure")),
        "loss_last": m.get("loss_last"),
        "saved_checkpoint_names": [x.get("name") for x in m.get("saved_checkpoints", [])],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default=str(DEFAULT_OUT))
    ap.add_argument("--note", default=str(DEFAULT_NOTE))
    args = ap.parse_args()
    out_root = pathlib.Path(args.out_root)
    note_path = pathlib.Path(args.note)
    rows = []
    missing = []
    for arm in ARMS:
        r = read_arm(out_root, arm)
        if r is None:
            missing.append(arm)
        else:
            r["metrics"] = load_metrics(arm)
            rows.append(r)
    rows.sort(key=lambda r: r["equal7"], reverse=True)
    by_arm = {r["arm"]: r for r in rows}
    contrasts = {}
    if "evidence_visible" in by_arm:
        ev = by_arm["evidence_visible"]
        for ctrl in ["uniform_control", "random_priority", "inverse_priority"]:
            if ctrl in by_arm:
                c = by_arm[ctrl]
                contrasts[f"evidence_visible_minus_{ctrl}"] = {
                    "equal7": ev["equal7"] - c["equal7"],
                    "target_recovery_sum": ev["target_recovery_sum_delta_vs_clean"] - c["target_recovery_sum_delta_vs_clean"],
                    "preserve_sum": ev["preserve_sum_delta_vs_clean"] - c["preserve_sum_delta_vs_clean"],
                    "per_column": {col: ev["scores"][col] - c["scores"][col] for col in COLUMNS},
                }
    restart_signal = None
    if "uniform_control" in by_arm:
        restart_signal = {
            "uniform_best_equal7": by_arm["uniform_control"]["equal7"],
            "uniform_delta_vs_clean_equal7": by_arm["uniform_control"]["delta_vs_clean_equal7"],
            "interpretation": "uniform_control isolates common clean-parent optimizer/LR restart with unchanged mask-target distribution",
        }
    decision_signal = "missing_outputs"
    if not missing and "evidence_visible" in by_arm and "uniform_control" in by_arm:
        ev = by_arm["evidence_visible"]
        ev_best_equal7 = all(ev["equal7"] > by_arm[ctrl]["equal7"] for ctrl in ["uniform_control", "random_priority", "inverse_priority"])
        ev_beats_uniform_target = ev["target_recovery_sum_delta_vs_clean"] > by_arm["uniform_control"]["target_recovery_sum_delta_vs_clean"]
        decision_signal = "evidence_visible_adds_to_restart" if ev_best_equal7 and ev_beats_uniform_target else "mask_targeting_not_additive_on_noaoa"
    payload = {
        "status": "MASK_NOAOA_SUMMARY",
        "created_utc": now(),
        "out_root": str(out_root),
        "clean_qwen_reference": CLEAN_QWEN,
        "missing_arms": missing,
        "ranked_results": rows,
        "contrasts": contrasts,
        "restart_signal_from_uniform_control": restart_signal,
        "decision_signal": decision_signal,
        "non_leakage_statement": "Post-training no-AoA measurement only; no downstream labels/items or AoA/CDI material used for training.",
    }
    out_root.mkdir(parents=True, exist_ok=True)
    out_json = out_root / "mask_noaoa_summary.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research masking continuation no-AoA result",
        "",
        f"Created UTC: {payload['created_utc']}",
        "",
        "| arm | best | equal7 | Δequal7 vs clean | Δ(EWoK+COMPS+GPIQA) vs clean | Δpreserve vs clean | mask budget | high-priority selected | loss_last | EWoK | COMPS | GPIQA | Supp | Entity | Read | BLiMP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        s = r["scores"]
        m = r.get("metrics") or {}
        lines.append(
            f"| {r['arm']} | {r['best_checkpoint']} | {r['equal7']:.4f} | {r['delta_vs_clean_equal7']:+.4f} | {r['target_recovery_sum_delta_vs_clean']:+.4f} | {r['preserve_sum_delta_vs_clean']:+.4f} | {m.get('masked_token_budget_ratio')} | {m.get('high_priority_fraction_among_selected_words')} | {m.get('loss_last')} | {s['EWoK']:.2f} | {s['COMPS']:.2f} | {s['GlobalPIQA']:.2f} | {s['Supplement']:.2f} | {s['Entity']:.2f} | {s['Reading']:.3f} | {s['BLiMP']:.2f} |"
        )
    lines.extend(["", "## Evidence-visible contrasts", ""])
    for name, vals in contrasts.items():
        lines.append(f"- `{name}`: equal7 {vals['equal7']:+.4f}; target-recovery-sum {vals['target_recovery_sum']:+.4f}; preserve-sum {vals['preserve_sum']:+.4f}; per-column {json.dumps(vals['per_column'], ensure_ascii=False)}")
    lines.extend(["", f"Scientific signal: `{decision_signal}`.", ""])
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out_json), "note": str(note_path), "decision_signal": decision_signal, "missing_arms": missing}, indent=2), flush=True)


if __name__ == "__main__":
    main()
