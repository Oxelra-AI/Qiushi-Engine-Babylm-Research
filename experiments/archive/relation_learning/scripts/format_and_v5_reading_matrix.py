#!/usr/bin/env python3
"""research matrix for reading the pending format and faithful-v4 results.

This is CPU-only bookkeeping that fixes the numeric comparison frame before the
pending endpoint scores are delivered.  It quantifies the two coherent private-seed
column band already in the record and writes the exact payload locations the next
step should read.
"""
from __future__ import annotations

import json
import pathlib
from typing import Any

ROOT = pathlib.Path.cwd()
OUT = ROOT / "experiments/archive/relation_learning/data/format_and_v5_reading_matrix"
PAIR = ROOT / "experiments/archive/relation_learning/data/pairwise_item_flips/pairwise_item_flips.json"
SG_SPREAD = ROOT / "experiments/archive/relation_learning/data/superglue_old_path_spread/superglue_old_path_spread_and_lever_rules.json"
FORMAT_BASE = ROOT / "experiments/archive/relation_learning/data/eval_format_replay_corrected"
TRAIN_BASE = ROOT / "experiments/archive/relation_learning/data/format_replay_corrected"
ARMS = ["coherent_unsplit_special", "isolated_all", "half_coherent_half_isolated"]
SEEDS = [98097, 98098]
CHEAP_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def read_json(p: pathlib.Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def format_paths() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in ARMS:
        out[arm] = {}
        for seed in SEEDS:
            label = f"step098_{arm}_seed{seed}_alpha0p75"
            train_dir = TRAIN_BASE / arm / f"seed{seed}"
            payload = FORMAT_BASE / arm / "eval" / label / "per_target" / f"{label}.json"
            summary = FORMAT_BASE / arm / "summary" / f"{label}_summary.json"
            out[arm][str(seed)] = {
                "label": label,
                "train_dir": rel(train_dir),
                "train_summary_exists": (train_dir / "summary.json").exists(),
                "training_log_exists": (train_dir / "training_log.jsonl").exists(),
                "payload": rel(payload),
                "payload_exists": payload.exists(),
                "cheap7_summary": rel(summary),
                "cheap7_summary_exists": summary.exists(),
            }
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pair = read_json(PAIR)
    sg = read_json(SG_SPREAD)
    deltas = pair["deltas_vs_anchor"]
    coherent_labels = ["coherent86_s43022", "coherent_s43122"]
    band = {}
    for col in CHEAP_COLUMNS + ["cheap7"]:
        xs = [float(deltas[label][col]) for label in coherent_labels if col in deltas[label] and deltas[label][col] is not None]
        band[col] = {"min_delta_vs_chck82": min(xs), "max_delta_vs_chck82": max(xs), "width": max(xs) - min(xs), "values": {label: deltas[label][col] for label in coherent_labels}}
    old_v4_overall = sg["old_coherent86_reference"]["old_overall_reproduced"]
    matrix = {
        "status": "FORMAT_AND_V5_READING_MATRIX",
        "purpose": "Fix the quantitative reading frame before pending endpoint results return.",
        "coherent_private_two_seed_band_vs_chck82": band,
        "old_stripped_superglue_macro_spread": sg["macro_superglue_spread"],
        "old_coherent86_overall_stripped_path": old_v4_overall,
        "faithful_v4_formula": "mean(BLiMP, Supplement, EWoK, Entity, COMPS, GlobalPIQA, Reading, repaired coherent86 SuperGLUE, measured coherent86 AoA)",
        "format_paths": format_paths(),
        "reading_order": [
            "1. Read training summaries/logs for all six format endpoints and reject only runs with broken masking, wrong word/update count, missing alpha endpoint, or failed initial function equality.",
            "2. Read coherent_unsplit_special first as special-token exposure on the exact coherent86 suffix words.",
            "3. Read isolated_all next as isolation beyond special-token exposure.",
            "4. Read half_coherent_half_isolated last as context-presence conditioning.",
            "5. Compare columns to coherent86/v4 and chck82, then read item localization for replication and mechanism rather than as a row-count-weighted score.",
            "6. A composed training candidate may use only levers with same-direction two-seed column movement and no important column falling outside the coherent private two-seed band; after selection it must be trained and evaluated directly over two private seeds.",
        ],
    }
    out_json = OUT / "format_and_v5_reading_matrix.json"
    out_md = OUT / "format_and_v5_reading_matrix.md"
    out_json.write_text(json.dumps(matrix, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research format and v5 reading matrix\n\n"]
    lines.append("This fixes the quantitative comparison frame before the pending endpoint outputs are read.\n\n")
    lines.append("## Coherent private two-seed band versus chck82\n\n")
    lines.append("| column | seed43022 Δ | seed43122 Δ | min | max | width |\n|---|---:|---:|---:|---:|---:|\n")
    for col in CHEAP_COLUMNS + ["cheap7"]:
        b = band[col]
        vals = b["values"]
        lines.append(f"| {col} | {vals.get('coherent86_s43022')} | {vals.get('coherent_s43122')} | {b['min_delta_vs_chck82']} | {b['max_delta_vs_chck82']} | {b['width']} |\n")
    lines.append("\n")
    lines.append(f"Old stripped-path coherent86 Overall: `{old_v4_overall}`. Faithful v4 must replace SuperGLUE with the repaired AutoModel result before a trained v5 is judged.\n\n")
    lines.append("## Pending format payloads\n\n")
    lines.append("| arm | seed | train summary? | payload? | payload path |\n|---|---:|---:|---:|---|\n")
    for arm in ARMS:
        for seed in SEEDS:
            rec = matrix["format_paths"][arm][str(seed)]
            lines.append(f"| {arm} | {seed} | {rec['train_summary_exists']} | {rec['payload_exists']} | `{rec['payload']}` |\n")
    lines.append("\n## Reading order\n\n")
    for x in matrix["reading_order"]:
        lines.append(f"- {x}\n")
    lines.append(f"\nJSON: `{rel(out_json)}`\n")
    out_md.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": matrix["status"], "out_json": rel(out_json), "out_md": rel(out_md)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
