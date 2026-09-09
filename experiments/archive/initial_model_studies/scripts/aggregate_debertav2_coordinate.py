#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
STRICT = ROOT / "repos/babylm-eval/strict"
GLUE_DATA = STRICT / "evaluation_data/full_eval/glue_filtered"
AVAIL = ROOT / "data/debertav2_b256_available_coordinate.json"
AOA = ROOT / "data/debertav2_b256_aoa_result.json"
SUPER = ROOT / "data/debertav2_b256_superglue_result.json"
BASELINE = ROOT / "data/wwm100m_8of9_coordinate.json"
OUT = ROOT / "data/debertav2_b256_coordinate.json"
NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/debertav2_b256_coordinate.md')

# 2026 leader reference
LEADER = {"blimp": 67.20, "supplement": 56.01, "ewok": 56.07, "entity_tracking": 28.45, "comps": 53.57, "superglue": 69.79, "global_piqa": 39.67, "reading": 5.42, "aoa": 0.0, "overall": 41.80}


def read_json(p):
    return json.loads(p.read_text(encoding="utf-8"))


def calc_glue(super_data):
    rows = []
    for tr in super_data["tasks"]:
        task = tr["task"]
        pred_path = pathlib.Path(tr["predictions"])
        pred = read_json(pred_path)[task]["predictions"]
        labels = [json.loads(l)["label"] for l in (GLUE_DATA / f"{task}.valid.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        if len(pred) != len(labels):
            raise RuntimeError(f"{task} len mismatch {len(pred)} vs {len(labels)}")
        correct = sum(1 for p, y in zip(pred, labels) if int(p["pred"]) == int(y))
        rows.append({"task": task, "n": len(labels), "correct": correct, "accuracy": correct / len(labels) * 100.0})
    glue = sum(r["accuracy"] for r in rows) / len(rows)
    return glue, rows


def main():
    avail = read_json(AVAIL)["scores"]
    aoa = read_json(AOA)
    glue, glue_rows = calc_glue(read_json(SUPER))
    scores = {
        "blimp": avail["blimp"],
        "supplement": avail["supplement"],
        "entity_tracking": avail["entity_tracking"],
        "comps": avail["comps"],
        "superglue": glue,
        "global_piqa": (avail["global_piqa_parallel"] + avail["global_piqa_nonparallel"]) / 2.0,
        "reading": (avail["reading_eye_tracking"] + avail["reading_self_paced"]) / 2.0,
        "aoa": float(aoa["aoa"]),
    }
    subcols = {
        "global_piqa_parallel": avail["global_piqa_parallel"],
        "global_piqa_nonparallel": avail["global_piqa_nonparallel"],
        "reading_eye_tracking": avail["reading_eye_tracking"],
        "reading_self_paced": avail["reading_self_paced"],
    }
    # Overall sensitivity: 8 present NLP+humanlike columns; EWoK still missing.
    present_nlp = [scores["blimp"], scores["supplement"], scores["entity_tracking"], scores["comps"], scores["superglue"], scores["global_piqa"]]
    humanlike = [scores["reading"], scores["aoa"]]
    overall_8col_sensitivity = (sum(present_nlp) + sum(humanlike)) / 8.0
    # If EWoK treated as fast-interim 46.18 for a NON-OFFICIAL sensitivity only:
    ewok_fast_interim = 46.18
    nlp_with_fast = present_nlp + [ewok_fast_interim]
    overall_9col_with_fast_interim = (sum(nlp_with_fast) + sum(humanlike)) / 9.0
    payload = {
        "run_name": "babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256",
        "model_path": read_json(AVAIL)["model_path"],
        "revision": "chck_100M",
        "scores_official_columns": scores,
        "subcolumns": subcols,
        "superglue_subtasks": glue_rows,
        "aoa_state": {"value": scores["aoa"], "degenerate": scores["aoa"] == 0.0, "note": "official AoA runner; 0 valid words for correlation, same degenerate path as baseline"},
        "ewok_state": {"official_full": None, "reason": "gated ewok-core/ewok-core-1.0", "fast_interim_non_official": ewok_fast_interim},
        "overall_sensitivity_not_official": {
            "overall_8col_ewok_missing": overall_8col_sensitivity,
            "overall_9col_with_ewok_fast_interim_NONOFFICIAL": overall_9col_with_fast_interim,
            "warning": "Neither value is an official Overall: EWoK full is gated and AoA is a degenerate 0. Provided only to gauge distance to the 41.80 leader."
        },
        "leader_reference_2026": LEADER,
        "gap_to_leader_per_column": {k: (scores[k] - LEADER[k]) for k in ["blimp", "supplement", "entity_tracking", "comps", "superglue", "global_piqa", "reading", "aoa"]},
        "input_files": {"available": str(AVAIL), "aoa": str(AOA), "superglue": str(SUPER)},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research — DeBERTa-v2 b256 near-complete coordinate", "",
        f"Evidence JSON: `{OUT}`", "",
        "EWoK full is gated (missing); AoA is a degenerate official 0. No true official Overall yet.", "",
        "| column | DeBERTa-v2 b256 | 2026 leader | gap |", "|---|---:|---:|---:|",
        f"| BLiMP | {scores['blimp']:.2f} | {LEADER['blimp']:.2f} | {scores['blimp']-LEADER['blimp']:+.2f} |",
        f"| BLiMP Supplement | {scores['supplement']:.2f} | {LEADER['supplement']:.2f} | {scores['supplement']-LEADER['supplement']:+.2f} |",
        f"| EWoK | missing (fast-interim {ewok_fast_interim}) | {LEADER['ewok']:.2f} | n/a |",
        f"| Entity Tracking | {scores['entity_tracking']:.2f} | {LEADER['entity_tracking']:.2f} | {scores['entity_tracking']-LEADER['entity_tracking']:+.2f} |",
        f"| COMPS | {scores['comps']:.2f} | {LEADER['comps']:.2f} | {scores['comps']-LEADER['comps']:+.2f} |",
        f"| (Super)GLUE | {scores['superglue']:.2f} | {LEADER['superglue']:.2f} | {scores['superglue']-LEADER['superglue']:+.2f} |",
        f"| GlobalPIQA | {scores['global_piqa']:.2f} | {LEADER['global_piqa']:.2f} | {scores['global_piqa']-LEADER['global_piqa']:+.2f} |",
        f"| Reading | {scores['reading']:.2f} | {LEADER['reading']:.2f} | {scores['reading']-LEADER['reading']:+.2f} |",
        f"| AoA | {scores['aoa']:.2f} (degenerate) | {LEADER['aoa']:.2f} | {scores['aoa']-LEADER['aoa']:+.2f} |",
        "",
        "## (Super)GLUE subtasks", "", "| task | accuracy | correct/n |", "|---|---:|---:|",
    ]
    for r in glue_rows:
        lines.append(f"| {r['task']} | {r['accuracy']:.2f} | {r['correct']}/{r['n']} |")
    lines += [
        "", f"Mean (Super)GLUE = {glue:.2f}",
        "",
        f"Non-official sensitivity Overall (8 cols, EWoK missing) = {overall_8col_sensitivity:.2f}",
        f"Non-official sensitivity Overall (9 cols using fast-interim EWoK {ewok_fast_interim}) = {overall_9col_with_fast_interim:.2f}",
        f"2026 leader Overall = {LEADER['overall']:.2f}",
        "",
        "These sensitivity Overalls are NOT official leaderboard numbers.",
    ]
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "DEBERTA_COORDINATE_DONE", "out": str(OUT), "superglue": glue, "scores": scores, "overall_8col_sensitivity": overall_8col_sensitivity, "overall_9col_with_fast_interim": overall_9col_with_fast_interim, "leader_overall": LEADER["overall"]}, indent=2))

if __name__ == "__main__":
    main()
