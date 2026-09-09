#!/usr/bin/env python3
"""Compare the fixed-data WWM-to-token masking-recipe run to clean-Qwen WWM.

The run evaluated by `eval_wwm_to_token_recipe_isolation.sh` keeps the
COMPACT_EXPERIENCE clean-Qwen data, architecture, tokenizer, optimizer, seeds, and 100M word
exposure fixed while changing only the masking schedule from fixed whole-word
masking to WWM followed by token masking.  This script reads the official-
compatible zero-shot+Reading trajectory summaries and writes a checkpoint-matched
scientific readout of the recipe effect.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading", "equal7_full_eval"]


def ck_key(name: str) -> int:
    if name.startswith("chck_") and name.endswith("M"):
        try:
            return int(name[len("chck_"):-1])
        except Exception:
            return 10**9
    return 10**9


def as_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def load_summary(path: pathlib.Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "table" not in payload:
        raise ValueError(f"summary has no table: {path}")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", default="experiments/archive/representation_and_objectives/data/wwm_to_token_fullzeroshot_reading")
    ap.add_argument("--target", default="qwen_wwm_to_token")
    ap.add_argument("--baseline-summary", default="experiments/archive/compact_experience/data/trajectory_screen/clean_qwen_seed43022_trajectory_summary.json")
    ap.add_argument("--out-json", default="")
    ap.add_argument("--out-md", default="")
    args = ap.parse_args()

    eval_root = pathlib.Path(args.eval_root)
    target_summary_path = eval_root / f"{args.target}_trajectory_summary.json"
    target = load_summary(target_summary_path)
    baseline = load_summary(pathlib.Path(args.baseline_summary))
    ttab = target.get("table", {})
    btab = baseline.get("table", {})
    checkpoints = sorted(set(ttab) & set(btab), key=ck_key)

    rows: dict[str, dict[str, Any]] = {}
    for ck in checkpoints:
        tr = ttab.get(ck, {})
        br = btab.get(ck, {})
        row: dict[str, Any] = {"wwm_to_token": {}, "clean_qwen_wwm": {}, "delta_wwm_to_token_minus_clean": {}}
        for col in COLUMNS:
            tv = as_float(tr.get(col))
            bv = as_float(br.get(col))
            row["wwm_to_token"][col] = tv
            row["clean_qwen_wwm"][col] = bv
            row["delta_wwm_to_token_minus_clean"][col] = None if tv is None or bv is None else round(tv - bv, 6)
        rows[ck] = row

    valid_target = {ck: r for ck, r in rows.items() if r["wwm_to_token"].get("equal7_full_eval") is not None}
    valid_delta = {ck: r for ck, r in rows.items() if r["delta_wwm_to_token_minus_clean"].get("equal7_full_eval") is not None}
    best_target = max(valid_target.items(), key=lambda kv: kv[1]["wwm_to_token"]["equal7_full_eval"]) if valid_target else None
    best_delta = max(valid_delta.items(), key=lambda kv: kv[1]["delta_wwm_to_token_minus_clean"]["equal7_full_eval"]) if valid_delta else None
    baseline_best = baseline.get("best_by_equal7_full_eval")

    payload = {
        "status": "WWM_TO_TOKEN_RECIPE_DELTA_SUMMARY",
        "target_summary": str(target_summary_path),
        "baseline_summary": str(args.baseline_summary),
        "target": args.target,
        "baseline_target": baseline.get("target"),
        "columns": COLUMNS,
        "checkpoints": checkpoints,
        "matched_rows": rows,
        "best_wwm_to_token_equal7": {"checkpoint": best_target[0], "row": best_target[1]} if best_target else None,
        "best_delta_equal7": {"checkpoint": best_delta[0], "row": best_delta[1]} if best_delta else None,
        "baseline_best_by_equal7": baseline_best,
        "scientific_readout": {
            "broad_positive_signal": "A positive matched equal7 delta with no concentrated loss in Supplement, Entity, GlobalPIQA, or Reading would support carrying the WWM-to-token schedule into the next combined run.",
            "weak_or_negative_signal": "A near-zero or negative matched delta would keep fixed WWM as the protected recipe while data mechanisms continue to be tested.",
            "scope": "This is a no-AoA zero-shot+Reading trajectory comparison, not a complete leaderboard score.",
        },
    }

    out_json = pathlib.Path(args.out_json) if args.out_json else eval_root / "wwm_to_token_vs_clean_qwen_delta_summary.json"
    out_md = pathlib.Path(args.out_md) if args.out_md else eval_root / "wwm_to_token_vs_clean_qwen_delta_summary.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def fmt(x: Any) -> str:
        v = as_float(x)
        return "" if v is None else f"{v:.4f}"

    lines = ["# research WWM-to-token recipe effect vs clean-Qwen fixed WWM\n\n"]
    lines.append(f"Target summary: `{target_summary_path}`\n\n")
    lines.append(f"Baseline summary: `{args.baseline_summary}`\n\n")
    lines.append("This compares official-compatible zero-shot columns plus Reading at matched checkpoints. It does not include SuperGLUE or AoA.\n\n")
    lines.append("| checkpoint | WWM->token equal7 | clean-Qwen WWM equal7 | delta | BLiMP | Supplement | EWoK | Entity | COMPS | GlobalPIQA | Reading |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for ck in checkpoints:
        row = rows[ck]
        d = row["delta_wwm_to_token_minus_clean"]
        lines.append(
            f"| {ck} | {fmt(row['wwm_to_token']['equal7_full_eval'])} | {fmt(row['clean_qwen_wwm']['equal7_full_eval'])} | {fmt(d['equal7_full_eval'])} | {fmt(d['BLiMP'])} | {fmt(d['Supplement'])} | {fmt(d['EWoK'])} | {fmt(d['Entity'])} | {fmt(d['COMPS'])} | {fmt(d['GlobalPIQA'])} | {fmt(d['Reading'])} |\n"
        )
    if best_target:
        lines.append(f"\nBest WWM->token checkpoint by equal7: `{best_target[0]}` = {best_target[1]['wwm_to_token']['equal7_full_eval']:.4f}.\n")
    if best_delta:
        lines.append(f"Best matched delta checkpoint: `{best_delta[0]}` = {best_delta[1]['delta_wwm_to_token_minus_clean']['equal7_full_eval']:.4f}.\n")
    if baseline_best:
        lines.append(f"Clean-Qwen WWM best in baseline summary: `{baseline_best.get('checkpoint')}` = {baseline_best.get('row', {}).get('equal7_full_eval')}.\n")
    lines.append(f"\nJSON: `{out_json}`\n")
    out_md.write_text("".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": payload["status"],
        "checkpoints": checkpoints,
        "best_wwm_to_token_equal7": payload["best_wwm_to_token_equal7"],
        "best_delta_equal7": payload["best_delta_equal7"],
        "out_json": str(out_json),
        "out_md": str(out_md),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
