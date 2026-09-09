#!/usr/bin/env python3
"""Summarize decoupled-RNG mask controls on no-AoA columns."""
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
COLUMNS7 = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
COLUMNS6 = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "Reading"]
ARMS = ["uniform", "inverse", "length_matched_random"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def eq(cols: list[str], row: dict[str, Any]) -> float:
    return sum(float(row[c]) for c in cols) / len(cols)


def find_traj(out_root: pathlib.Path, target: str) -> pathlib.Path:
    direct = out_root / f"{target}_trajectory_summary.json"
    nested = out_root / target / f"{target}_trajectory_summary.json"
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
        "trainer_variant", "mask_mode", "mask_budget", "masked_token_budget_ratio",
        "base_high_fraction_among_selected_words", "selected_wordpiece_len_mean",
        "selected_base_priority_mean", "length_template_match_failures",
        "parent_start_word_exposure", "continuation_word_exposure", "actual_total_word_exposure",
        "actual_training_steps", "train_rng_seed", "loss_last", "train_file_sha256",
    ]} | {"path": str(p), "saved_checkpoint_names": [x.get("name") for x in m.get("saved_checkpoints", [])]}


def load_arm(case: str, arm: str, out_root: pathlib.Path) -> dict[str, Any]:
    target = f"decoupled_{arm}_{case}"
    if arm == "length_matched_random":
        target = f"decoupled_length_matched_random_{case}"
    run_name = target.replace("step048_", "step048_")
    run_dir = _public_path('experiments/archive/compact_experience/training/runs') / target.replace("decoupled_", "decoupled_")
    traj = find_traj(out_root, target)
    obj = json.loads(traj.read_text(encoding="utf-8"))
    table = obj.get("table") or {}
    rows = []
    for ckpt, row in table.items():
        if not isinstance(row, dict) or row.get("equal7_full_eval") is None:
            continue
        scores = {c: float(row[c]) for c in COLUMNS7}
        rows.append({
            "checkpoint": ckpt,
            "equal7": eq(COLUMNS7, scores),
            "equal6_without_GlobalPIQA": eq(COLUMNS6, scores),
            "scores": scores,
        })
    rows.sort(key=lambda r: r["equal7"], reverse=True)
    if not rows:
        raise RuntimeError(f"No scored rows in {traj}")
    return {
        "arm": arm,
        "target": target,
        "trajectory_summary": str(traj),
        "run_dir": str(run_dir),
        "metrics": load_metrics(run_dir),
        "best_checkpoint": rows[0]["checkpoint"],
        "best_equal7": rows[0]["equal7"],
        "best_equal6_without_GlobalPIQA": rows[0]["equal6_without_GlobalPIQA"],
        "ranked_results": rows,
        "by_checkpoint": {r["checkpoint"]: r for r in rows},
    }


def diff(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    return {
        "equal7": a["equal7"] - b["equal7"],
        "equal6_without_GlobalPIQA": a["equal6_without_GlobalPIQA"] - b["equal6_without_GlobalPIQA"],
        "GlobalPIQA": a["scores"]["GlobalPIQA"] - b["scores"]["GlobalPIQA"],
        "per_column": {c: a["scores"][c] - b["scores"][c] for c in COLUMNS7},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=["seed43022", "seed43122"], default="seed43022")
    args = ap.parse_args()
    out_root = _public_path('experiments/archive/compact_experience/data') / f"decoupled_mask_noaoa_{args.case}"
    out = out_root / f"decoupled_mask_noaoa_summary_{args.case}.json"
    note = _public_path('research/notes/compact_experience') / f"48_decoupled_mask_noaoa_summary_{args.case}.md"
    arms = {arm: load_arm(args.case, arm, out_root) for arm in ARMS}
    uni = arms["uniform"]
    inv = arms["inverse"]
    lm = arms["length_matched_random"]
    contrasts: dict[str, Any] = {
        "inverse_best_minus_uniform_best": diff(inv["ranked_results"][0], uni["ranked_results"][0]),
        "length_best_minus_uniform_best": diff(lm["ranked_results"][0], uni["ranked_results"][0]),
        "inverse_best_minus_length_best": diff(inv["ranked_results"][0], lm["ranked_results"][0]),
    }
    for ck in ["chck_85M", "chck_90M", "chck_95M", "chck_100M"]:
        if ck in inv["by_checkpoint"] and ck in uni["by_checkpoint"] and ck in lm["by_checkpoint"]:
            contrasts[f"inverse_minus_uniform_at_{ck}"] = diff(inv["by_checkpoint"][ck], uni["by_checkpoint"][ck])
            contrasts[f"length_minus_uniform_at_{ck}"] = diff(lm["by_checkpoint"][ck], uni["by_checkpoint"][ck])
            contrasts[f"inverse_minus_length_at_{ck}"] = diff(inv["by_checkpoint"][ck], lm["by_checkpoint"][ck])
    signal = "decoupled_controls_need_interpretation"
    c100_iu = contrasts.get("inverse_minus_uniform_at_chck_100M")
    c100_lu = contrasts.get("length_minus_uniform_at_chck_100M")
    c100_il = contrasts.get("inverse_minus_length_at_chck_100M")
    if c100_iu:
        if c100_iu["equal6_without_GlobalPIQA"] > 0.25 and c100_il and c100_il["equal6_without_GlobalPIQA"] > 0.15:
            signal = "identity_signal_survives_length_and_rng_controls"
        elif c100_lu and c100_lu["equal6_without_GlobalPIQA"] > 0.25 and c100_il and abs(c100_il["equal6_without_GlobalPIQA"]) <= 0.15:
            signal = "length_count_geometry_explains_most_inverse_signal"
        elif c100_iu["equal6_without_GlobalPIQA"] <= 0.1:
            signal = "inverse_signal_largely_removed_by_rng_decoupling"
    payload = {
        "status": "DECOUPLED_MASK_CONTROLS_NOAOA_SUMMARY",
        "case": args.case,
        "created_utc": now(),
        "arms": arms,
        "contrasts": contrasts,
        "signal": signal,
        "non_leakage_statement": "Post-training no-AoA measurements of legal continuation runs; no downstream labels/items or AoA/CDI material used for training or control construction.",
    }
    out_root.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        f"# research decoupled mask controls no-AoA summary ({args.case})", "",
        "equal7/equal6 are no-AoA screening metrics, not official Overall.", "",
        "| arm | best ck | best equal7 | best equal6 no GPIQA | chck_100M equal7 | chck_100M equal6 no GPIQA | base-high frac | wp len mean |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        a = arms[arm]
        m = a.get("metrics") or {}
        c100 = a["by_checkpoint"].get("chck_100M", {})
        lines.append(f"| {arm} | {a['best_checkpoint']} | {a['best_equal7']:.4f} | {a['best_equal6_without_GlobalPIQA']:.4f} | {c100.get('equal7')} | {c100.get('equal6_without_GlobalPIQA')} | {m.get('base_high_fraction_among_selected_words')} | {m.get('selected_wordpiece_len_mean')} |")
    lines.extend(["", "## Key contrasts", ""])
    for k in ["inverse_minus_uniform_at_chck_100M", "length_minus_uniform_at_chck_100M", "inverse_minus_length_at_chck_100M", "inverse_best_minus_uniform_best", "length_best_minus_uniform_best", "inverse_best_minus_length_best"]:
        if k in contrasts:
            v = contrasts[k]
            lines.append(f"- `{k}`: equal7 {v['equal7']:+.4f}; equal6(no GPIQA) {v['equal6_without_GlobalPIQA']:+.4f}; GPIQA {v['GlobalPIQA']:+.3f}; per-column {json.dumps(v['per_column'], ensure_ascii=False)}")
    lines.extend(["", f"Signal: `{signal}`.", ""])
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(out), "note": str(note), "signal": signal, "best": {arm: {"ckpt": arms[arm]["best_checkpoint"], "equal7": arms[arm]["best_equal7"]} for arm in ARMS}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
