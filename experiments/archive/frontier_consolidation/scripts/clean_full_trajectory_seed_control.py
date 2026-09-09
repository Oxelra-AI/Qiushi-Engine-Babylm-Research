#!/usr/bin/env python3
"""research: CPU summary of inherited clean-Qwen seed trajectories.

Uses already-saved compact_experience full zero-shot/reading checkpoint summaries for the two
clean-Qwen seeds. This is a cheap context control for the compact_view_reinvest
seed-spread question while the exact matched sparse fast-slice probe runs.
"""
from __future__ import annotations

import json
import math
import pathlib
from typing import Any

ROOT = pathlib.Path.cwd()
STUDY = ROOT / "experiments/archive" / 'frontier_consolidation'
OUT_DIR = STUDY / "data" / "clean_full_trajectory_seed_control"
OUT_JSON = OUT_DIR / "clean_full_trajectory_seed_control.json"
OUT_MD = STUDY / "notes" / "clean_full_trajectory_seed_control.md"
COMPACT_EXPERIENCE = ROOT / "experiments/archive" / 'compact_experience' / "data" / "trajectory_screen"
SUMMARIES = {
    "43022": COMPACT_EXPERIENCE / "clean_qwen_seed43022_trajectory_summary.json",
    "43122": COMPACT_EXPERIENCE / "clean_qwen_seed43122_trajectory_summary.json",
}
FULL_EVAL = ROOT / "experiments/archive" / 'compact_experience' / "data" / "control_eval_summary.json"
EXPOSURES = [10, 40, 100]
COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading", "equal7_full_eval"]


def load(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def chck(exp: int) -> str:
    return f"chck_{exp}M"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = {seed: load(path) for seed, path in SUMMARIES.items()}
    table = {}
    for exp in EXPOSURES:
        c = chck(exp)
        r0 = rows["43022"]["table"][c]
        r1 = rows["43122"]["table"][c]
        diff = {col: r1[col] - r0[col] for col in COLUMNS if col in r0 and col in r1}
        table[str(exp)] = {
            "seed43022": {col: r0[col] for col in COLUMNS if col in r0},
            "seed43122": {col: r1[col] for col in COLUMNS if col in r1},
            "seed43122_minus_43022": diff,
            "mean_zero_shot_delta_excluding_reading_equal7": sum(diff[col] for col in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA"]) / 6.0,
        }
    # Add complete 100M official surface from research for context, including SuperGLUE and AoA.
    full = load(FULL_EVAL)["targets"]
    full_430 = full["qwen_clean_aligned"]
    full_431 = full["qwen_clean_aligned_seed43122"]
    official_cols = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "SuperGLUE", "Reading", "AoA", "Overall", "NLP_average"]
    full_diff = {col: full_431[col] - full_430[col] for col in official_cols if col in full_430 and col in full_431}
    # Compact summary statistics over the selected trajectory exposures.
    per_col = {}
    for col in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]:
        vals = [table[str(exp)]["seed43122_minus_43022"].get(col) for exp in EXPOSURES]
        vals = [v for v in vals if v is not None]
        per_col[col] = {
            "deltas_10_40_100M": vals,
            "mean": sum(vals) / len(vals),
            "max_abs": max(abs(v) for v in vals),
            "final_100M": table["100"]["seed43122_minus_43022"].get(col),
        }
    result = {
        "status": "CLEAN_FULL_TRAJECTORY_SEED_CONTROL",
        "purpose": "Cheap CPU context control from existing clean-Qwen full trajectory summaries while matched sparse slices run.",
        "source_summaries": {seed: str(path) for seed, path in SUMMARIES.items()},
        "selected_exposures_m": EXPOSURES,
        "table": table,
        "per_column_delta_summary": per_col,
        "official_100M_surface_from_step030": {
            "seed43022": {col: full_430[col] for col in official_cols if col in full_430},
            "seed43122": {col: full_431[col] for col in official_cols if col in full_431},
            "seed43122_minus_43022": full_diff,
        },
        "interpretation": {
            "full_trajectory_not_exact_sparse_match": True,
            "seed43122_lower_at_100M_on_clean_qwen": full_diff.get("Overall"),
            "clean_has_broad_seed_spread": "At 100M, clean-Qwen seed43122 is lower on BLiMP, Supplement, Entity, GlobalPIQA, SuperGLUE, and Reading, but slightly higher on EWoK and COMPS. This means some broad seed spread is inherited from the base recipe before compact-view reinvestment.",
            "why_matched_sparse_still_needed": "The COMPACT_EXPERIENCE full trajectory uses full official data and lacks the exact selected fast BLiMP/EWoK slices and 1M exposure, so the research matched sparse run remains the direct control for reinvest trajectories.",
        },
    }
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = []
    lines.append("# research clean-Qwen full-trajectory seed-control context\n\n")
    lines.append("CPU-only summary from existing COMPACT_EXPERIENCE clean-Qwen checkpoint trajectory files. This is context for the reinvest seed-stability question, not a substitute for the matched fast-slice probe.\n\n")
    lines.append("## Seed43122 minus seed43022 on selected full-trajectory exposures\n")
    for exp in EXPOSURES:
        diffs = table[str(exp)]["seed43122_minus_43022"]
        lines.append(f"- {exp}M: " + ", ".join(f"{k}={v:.3f}" for k, v in diffs.items()) + "\n")
    lines.append("\n## Official 100M full surface from COMPACT_EXPERIENCE research\n")
    lines.append(", ".join(f"{k}={v:.3f}" for k, v in full_diff.items()) + "\n\n")
    lines.append("At 100M, clean-Qwen seed43122 is not an equal replicate of seed43022: it loses Overall -0.694, NLP_average -0.813, SuperGLUE -1.563, Supplement -1.330, BLiMP -0.820, Entity -0.500, GlobalPIQA -2.000, and Reading -0.555, while EWoK +0.240 and COMPS +0.280 rise slightly. This makes base-recipe seed spread a real alternative explanation for part of the reinvest seed gap.\n\n")
    lines.append("The exact matched answer requires the research clean sparse temporal run because it uses the same fast Supplement/EWoK slices, selected BLiMP files, Entity rows, and 1/10/40/100M exposures as the reinvest sparse probe.\n\n")
    lines.append(f"Machine-readable output: `{OUT_JSON}`\n")
    OUT_MD.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": result["status"], "out_json": str(OUT_JSON), "out_md": str(OUT_MD)}, indent=2))


if __name__ == "__main__":
    main()
