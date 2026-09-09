#!/usr/bin/env python3
"""research: write local correction note for the current official coordinate."""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path.cwd()
STUDY = ROOT / "experiments/archive" / 'frontier_consolidation'
SEED43022_SUMMARY = ROOT / "experiments/archive" / 'representation_and_objectives' / "data" / "pristine_collate_seed43022" / "pristine_collate_seed43022_summary.json"
AOA_SUMMARY = STUDY / "data" / "official_rowcount_aoa_reinvest_seeds" / "official_rowcount_aoa_run_summary.json"
OUT_DIR = STUDY / "data" / "official_coordinate_correction"
OUT_JSON = OUT_DIR / "official_coordinate_correction.json"
OUT_MD = STUDY / "notes" / "official_coordinate_correction.md"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    d = json.loads(SEED43022_SUMMARY.read_text(encoding="utf-8"))
    official = d["score_summary"]["official_overall"]
    scores = official["scores"]
    a = json.loads(AOA_SUMMARY.read_text(encoding="utf-8"))
    aoa_records = a["records"]
    result = {
        "status": "OFFICIAL_COORDINATE_CORRECTION",
        "seed43022_pristine_source": str(SEED43022_SUMMARY),
        "seed43022_official_scores": scores,
        "seed43022_official_overall": official["Overall"],
        "seed43022_margin_over_visible_41p8": official["margin_over_visible_leader"],
        "seed43022_nlp_average": official["NLP_average"],
        "seed43022_human_like_average": official["Human_like_average"],
        "old_step025_overall": d["input_corrections"]["overall"],
        "old_minus_official": d["input_corrections"]["overall"] - official["Overall"],
        "corrections": d["input_corrections"],
        "frontier_consolidation_aoa_summary_source": str(AOA_SUMMARY),
        "frontier_consolidation_independent_aoa_records": aoa_records,
        "superseded_local_note": "research/notes/frontier_consolidation/aoa_overall_update.md",
        "interpretation": {
            "current_coordinate": "Use the A01 research pristine-collator coordinate for seed43022: Overall 42.0331347900748, not the earlier local 42.086785719138156.",
            "why": "The earlier number used stale local EWoK and an all-accuracy SuperGLUE convention. A01 research restages predictions through a clean upstream collator coordinate with official 7618-row EWoK, primary-metric SuperGLUE, and official min_context=0 AoA.",
            "a02_independent_aoa_confirmation": "A02 research/26 independently confirms min_context=0 AoA=0.0 for both reinvest seed43022 and seed43122 with 8005 rows per checkpoint and 19 checkpoint steps.",
        },
    }
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = []
    lines.append("# research official-coordinate correction for compact_view_reinvest\n\n")
    lines.append("Use this local correction instead of the research `26_aoa_overall_update` Overall arithmetic.\n\n")
    lines.append("## Current seed43022 official coordinate\n")
    for key in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "SuperGLUE", "GlobalPIQA", "Reading", "AoA"]:
        lines.append(f"- {key}: {scores[key]}\n")
    lines.append(f"- Overall: {official['Overall']}\n")
    lines.append(f"- Margin over visible 41.8 leader: {official['margin_over_visible_leader']}\n")
    lines.append(f"- NLP average: {official['NLP_average']}\n")
    lines.append(f"- Human-like average: {official['Human_like_average']}\n\n")
    lines.append("## Correction\n")
    lines.append(f"Earlier research/research arithmetic used {d['input_corrections']['overall']}. The current official-coordinate value is lower by {d['input_corrections']['overall'] - official['Overall']}. The coordinate changed because EWoK was re-scored on the pristine 7618-row official data and SuperGLUE was aggregated by the official primary metrics.\n\n")
    lines.append("## Independent A02 AoA confirmation\n")
    for r in aoa_records:
        lines.append(f"- {r['target']}: AoA={r['aoa']}, row_count_values={r['row_count_values']}, num_rows={r['num_rows']}, n_words={r['curve_fitness_record']['n_words']}\n")
    lines.append("\nSeed43022 remains above the visible leader as a single official-coordinate endpoint, but seed43122 robustness and the mechanism/source of seed spread are still active research questions.\n\n")
    lines.append(f"Machine-readable output: `{OUT_JSON}`\n")
    OUT_MD.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": result["status"], "out_json": str(OUT_JSON), "out_md": str(OUT_MD), "overall": official["Overall"]}, indent=2))


if __name__ == "__main__":
    main()
