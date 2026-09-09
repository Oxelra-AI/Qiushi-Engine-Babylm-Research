#!/usr/bin/env python3
"""Summarize second-seed inverse-priority vs uniform no-AoA replication.

The comparison isolates whether the research/046 inverse-priority mask-target
allocation adds beyond common clean-parent optimizer/LR rephasing on the second
clean-Qwen seed. It uses only post-training no-AoA columns.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import pathlib
import time
from typing import Any

ROOT = _public_path('experiments/archive/compact_experience')
OUT_ROOT = _public_path('experiments/archive/compact_experience/data/seed43122_mask_noaoa_eval')
OUT = _public_path('experiments/archive/compact_experience/data/seed43122_mask_noaoa_eval/seed43122_mask_noaoa_summary.json')
NOTE = _public_path('research/notes/compact_experience/seed43122_inverse_uniform_noaoa_summary.md')
ARMS = {
    "uniform_control": {
        "target": "seed43122_uniform_control",
        "run_dir": _public_path('experiments/archive/compact_experience/training/runs/mask_uniform_control_seed43144'),
    },
    "inverse_priority": {
        "target": "seed43122_inverse_priority",
        "run_dir": _public_path('experiments/archive/compact_experience/training/runs/mask_inverse_priority_seed43144'),
    },
}
COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
TARGET_RECOVERY = ["EWoK", "COMPS", "GlobalPIQA"]
PRESERVE = ["BLiMP", "Supplement", "Entity", "Reading"]
# Same-seed clean-Qwen 100M complete evaluation, from data/full_eval/full_eval_summary.json.
CLEAN43122 = {
    "BLiMP": 66.02,
    "Supplement": 61.51,
    "EWoK": 50.43,
    "Entity": 25.26,
    "COMPS": 52.06,
    "GlobalPIQA": 34.62,
    "Reading": 7.205,
}
CLEAN43122["equal7_full_eval"] = sum(CLEAN43122[c] for c in COLUMNS) / len(COLUMNS)
# Seed43022 research reference, for cross-seed shape comparison only.
SEED43022_INVERSE = {
    "chck_95M": {"equal7": 43.86571428571428, "BLiMP": 67.33, "Supplement": 63.74, "EWoK": 51.05, "Entity": 28.66, "COMPS": 51.71, "GlobalPIQA": 36.635, "Reading": 7.935},
    "chck_100M": {"equal7": 43.705, "BLiMP": 67.39, "Supplement": 63.86, "EWoK": 50.88, "Entity": 28.55, "COMPS": 51.64, "GlobalPIQA": 35.635, "Reading": 7.98},
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def trajectory_path(target: str) -> pathlib.Path:
    direct = OUT_ROOT / f"{target}_trajectory_summary.json"
    nested = OUT_ROOT / target / f"{target}_trajectory_summary.json"
    if direct.exists():
        return direct
    if nested.exists():
        return nested
    raise FileNotFoundError(direct)


def load_metrics(run_dir: pathlib.Path) -> dict[str, Any] | None:
    p = run_dir / "scientific_metrics.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text(encoding="utf-8"))
    return {k: m.get(k) for k in [
        "variant", "mask_mode", "mask_budget", "masked_token_budget_ratio",
        "high_priority_fraction_among_selected_words", "parent_start_word_exposure",
        "continuation_word_exposure", "actual_total_word_exposure", "loss_first", "loss_last",
        "actual_training_steps", "train_rng_seed", "train_file_sha256",
    ]} | {"path": str(p), "saved_checkpoint_names": [x.get("name") for x in m.get("saved_checkpoints", [])]}


def summarize_arm(arm: str, spec: dict[str, Any]) -> dict[str, Any]:
    path = trajectory_path(spec["target"])
    obj = json.loads(path.read_text(encoding="utf-8"))
    table = obj.get("table") or {}
    rows = []
    for ckpt, row in table.items():
        if not isinstance(row, dict) or row.get("equal7_full_eval") is None:
            continue
        scores = {c: float(row.get(c, 0.0)) for c in COLUMNS}
        rows.append({
            "checkpoint": ckpt,
            "equal7": float(row["equal7_full_eval"]),
            "delta_vs_clean43122_equal7": float(row["equal7_full_eval"]) - CLEAN43122["equal7_full_eval"],
            "target_recovery_sum_delta_vs_clean43122": sum(scores[c] - CLEAN43122[c] for c in TARGET_RECOVERY),
            "preserve_sum_delta_vs_clean43122": sum(scores[c] - CLEAN43122[c] for c in PRESERVE),
            "scores": scores,
        })
    rows.sort(key=lambda r: r["equal7"], reverse=True)
    if not rows:
        raise RuntimeError(f"No evaluated rows for {arm} in {path}")
    by_ckpt = {r["checkpoint"]: r for r in rows}
    return {
        "arm": arm,
        "target": spec["target"],
        "trajectory_summary": str(path),
        "metrics": load_metrics(spec["run_dir"]),
        "best_checkpoint": rows[0]["checkpoint"],
        "best_equal7": rows[0]["equal7"],
        "ranked_results": rows,
        "by_checkpoint": by_ckpt,
    }


def diff_rows(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = {"equal7": a["equal7"] - b["equal7"]}
    out["target_recovery_sum"] = a["target_recovery_sum_delta_vs_clean43122"] - b["target_recovery_sum_delta_vs_clean43122"]
    out["preserve_sum"] = a["preserve_sum_delta_vs_clean43122"] - b["preserve_sum_delta_vs_clean43122"]
    out["per_column"] = {c: a["scores"][c] - b["scores"][c] for c in COLUMNS}
    return out


def main() -> None:
    arms = {arm: summarize_arm(arm, spec) for arm, spec in ARMS.items()}
    inv = arms["inverse_priority"]
    uni = arms["uniform_control"]
    inv_best = inv["ranked_results"][0]
    uni_best = uni["ranked_results"][0]
    contrasts = {
        "inverse_best_minus_uniform_best": diff_rows(inv_best, uni_best),
    }
    for ckpt in ["chck_85M", "chck_90M", "chck_95M", "chck_100M"]:
        if ckpt in inv["by_checkpoint"] and ckpt in uni["by_checkpoint"]:
            contrasts[f"inverse_minus_uniform_at_{ckpt}"] = diff_rows(inv["by_checkpoint"][ckpt], uni["by_checkpoint"][ckpt])
    signal = "inverse_replication_pending_interpretation"
    if inv_best["equal7"] > uni_best["equal7"] + 0.15 and ("chck_100M" in inv["by_checkpoint"] and inv["by_checkpoint"]["chck_100M"]["equal7"] > uni["by_checkpoint"].get("chck_100M", {"equal7": -1e9})["equal7"] + 0.15):
        signal = "inverse_priority_adds_beyond_uniform_on_second_seed_noaoa"
    elif inv_best["equal7"] <= uni_best["equal7"]:
        signal = "inverse_priority_does_not_replicate_beyond_uniform_on_second_seed_noaoa"
    payload = {
        "status": "SEED43122_MASK_NOAOA_REPLICATION_SUMMARY",
        "created_utc": now(),
        "clean_seed43122_reference_100M_noaoa_columns": CLEAN43122,
        "clean_seed43122_reference_equal7": CLEAN43122["equal7_full_eval"],
        "arms": arms,
        "contrasts": contrasts,
        "seed43022_inverse_reference_for_shape_comparison": SEED43022_INVERSE,
        "signal": signal,
        "non_leakage_statement": "Training used only the clean Qwen-aligned 10M pool and legal surface mask-priority rules; this summary uses post-training no-AoA official-style measurements only.",
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research seed43122 inverse-priority vs uniform no-AoA replication", "",
        f"Created UTC: {payload['created_utc']}", "",
        f"Same-seed clean-Qwen 100M equal7 reference: {CLEAN43122['equal7_full_eval']:.6f}.", "",
        "| arm | checkpoint | equal7 | Δclean43122 | Δtarget recovery | Δpreserve | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | Reading |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ["inverse_priority", "uniform_control"]:
        for r in arms[arm]["ranked_results"]:
            s = r["scores"]
            lines.append(f"| {arm} | {r['checkpoint']} | {r['equal7']:.4f} | {r['delta_vs_clean43122_equal7']:+.4f} | {r['target_recovery_sum_delta_vs_clean43122']:+.4f} | {r['preserve_sum_delta_vs_clean43122']:+.4f} | {s['BLiMP']:.2f} | {s['Supplement']:.2f} | {s['EWoK']:.2f} | {s['Entity']:.2f} | {s['COMPS']:.2f} | {s['GlobalPIQA']:.2f} | {s['Reading']:.3f} |")
    lines.extend(["", "## Inverse minus uniform contrasts", ""])
    for k, v in contrasts.items():
        lines.append(f"- `{k}`: equal7 {v['equal7']:+.4f}; target-recovery-sum {v['target_recovery_sum']:+.4f}; preserve-sum {v['preserve_sum']:+.4f}; per-column {json.dumps(v['per_column'], ensure_ascii=False)}")
    lines.append("")
    lines.append(f"Signal: `{signal}`.")
    _public_path('research/notes/compact_experience').mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT), "note": str(NOTE), "signal": signal, "inverse_best": {"checkpoint": inv_best["checkpoint"], "equal7": inv_best["equal7"]}, "uniform_best": {"checkpoint": uni_best["checkpoint"], "equal7": uni_best["equal7"]}, "inverse_minus_uniform_best_equal7": contrasts["inverse_best_minus_uniform_best"]["equal7"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
