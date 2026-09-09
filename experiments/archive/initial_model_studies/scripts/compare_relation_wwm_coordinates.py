#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
FILES = {
    "relation_10M": ROOT / "data/relation_wwm_chck_10M_available_coordinate.json",
    "relation_20M": ROOT / "data/relation_wwm_chck_20M_available_coordinate.json",
    "shuffled_10M": ROOT / "data/relation_wwm_shuffled_chck_10M_available_coordinate.json",
    "shuffled_20M": ROOT / "data/relation_wwm_shuffled_chck_20M_available_coordinate.json",
}
BASE_TRAJ = ROOT / "data/debertav2_b256_coordinate.json"
TRAIN_VAL = ROOT / "data/relation_wwm_20m_training_validation.json"
OUT = ROOT / "data/relation_wwm_available_coordinate_comparison.json"
NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/relation_wwm_available_coordinate_comparison.md')
COLS = [
    ("blimp", "BLiMP"),
    ("supplement", "Supplement"),
    ("entity_tracking", "Entity"),
    ("comps", "COMPS"),
    ("global_piqa_parallel", "GlobalPIQA parallel"),
    ("global_piqa_nonparallel", "GlobalPIQA nonparallel"),
    ("global_piqa_mean", "GlobalPIQA mean"),
    ("reading_eye_tracking", "Reading eye"),
    ("reading_self_paced", "Reading self-paced"),
    ("reading_mean", "Reading mean"),
]


def load_coordinate(path: pathlib.Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    s = dict(d["scores"])
    der = d.get("derived_columns", {})
    s["global_piqa_mean"] = der.get("GlobalPIQA_mean_parallel_nonparallel", (s["global_piqa_parallel"] + s["global_piqa_nonparallel"]) / 2.0)
    s["reading_mean"] = der.get("Reading_mean_eye_self_paced", (s["reading_eye_tracking"] + s["reading_self_paced"]) / 2.0)
    return {"raw": d, "scores": s}


def row_delta(a: dict, b: dict) -> dict:
    return {k: a["scores"][k] - b["scores"][k] for k, _ in COLS if k in a["scores"] and k in b["scores"]}


def sum_selected(delta: dict, keys: list[str]) -> float:
    return sum(delta[k] for k in keys if k in delta)


def main():
    coords = {k: load_coordinate(v) for k, v in FILES.items()}
    train = json.loads(TRAIN_VAL.read_text(encoding="utf-8")) if TRAIN_VAL.exists() else None
    d10 = row_delta(coords["relation_10M"], coords["shuffled_10M"])
    d20 = row_delta(coords["relation_20M"], coords["shuffled_20M"])
    progression_relation = row_delta(coords["relation_20M"], coords["relation_10M"])
    progression_shuffled = row_delta(coords["shuffled_20M"], coords["shuffled_10M"])
    scoreboard_keys = ["blimp", "supplement", "entity_tracking", "comps", "global_piqa_mean", "reading_mean"]
    target_keys = ["entity_tracking", "global_piqa_mean"]
    payload = {
        "status": "RELATION_WWM_AVAILABLE_COORDINATE_COMPARISON",
        "files": {k: str(v) for k, v in FILES.items()},
        "scores": {k: coords[k]["scores"] for k in coords},
        "relation_minus_shuffled_10M": d10,
        "relation_minus_shuffled_20M": d20,
        "progression_relation_20M_minus_10M": progression_relation,
        "progression_shuffled_20M_minus_10M": progression_shuffled,
        "summary": {
            "known_six_sum_delta_relation_minus_shuffled_10M": sum_selected(d10, scoreboard_keys),
            "known_six_sum_delta_relation_minus_shuffled_20M": sum_selected(d20, scoreboard_keys),
            "target_entity_globalpiqa_sum_delta_10M": sum_selected(d10, target_keys),
            "target_entity_globalpiqa_sum_delta_20M": sum_selected(d20, target_keys),
            "reading_mean_delta_10M": d10.get("reading_mean"),
            "reading_mean_delta_20M": d20.get("reading_mean"),
        },
        "training_validation": train,
        "interpretation": "Higher is better for all listed leaderboard columns including Reading. Relation-vs-shuffled deltas test whether targeted lexical relation/entity masking, not just exact-K nonuniform masking, improves task scores.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research — relation-WWM available-coordinate comparison", "", f"Evidence JSON: `{OUT}`", "", "Higher is better for all columns shown, including Reading.", ""]
    for exposure, rel_key, shuf_key, delta in [("10M", "relation_10M", "shuffled_10M", d10), ("20M", "relation_20M", "shuffled_20M", d20)]:
        lines += [f"## {exposure}: relation_wwm vs shuffled", "", "| column | relation | shuffled | delta |", "|---|---:|---:|---:|"]
        for key, label in COLS:
            if key in coords[rel_key]["scores"] and key in coords[shuf_key]["scores"]:
                lines.append(f"| {label} | {coords[rel_key]['scores'][key]:.2f} | {coords[shuf_key]['scores'][key]:.2f} | {delta[key]:+.2f} |")
        lines.append("")
    lines += ["## Summary", ""]
    for k, v in payload["summary"].items():
        lines.append(f"- {k}: {v}")
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out": str(OUT), "summary": payload["summary"]}, indent=2))

if __name__ == "__main__":
    main()
