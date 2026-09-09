#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
STRICT = ROOT / "repos/babylm-eval/strict"
GLUE_DATA = STRICT / "evaluation_data/full_eval/glue_filtered"
WWM_REFERENCE_PATH = ROOT / "data/wwm100m_available_official_coordinate.json"
AOA = ROOT / "data/wwm100m_aoa_result.json"
SUPER = ROOT / "data/wwm100m_superglue_result.json"
OUT = ROOT / "data/wwm100m_8of9_coordinate.json"
NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/wwm100m_8of9_coordinate.md')


def read_json(p: pathlib.Path):
    return json.loads(p.read_text(encoding="utf-8"))


def calc_glue(super_data):
    rows = []
    for task_row in super_data["tasks"]:
        task = task_row["task"]
        pred_path = pathlib.Path(task_row["predictions"])
        if not pred_path.exists():
            # tolerate absolute paths from tool output if stored root-relative in JSON
            pred_path = ROOT.parent.parent.parent.parent / pred_path if not pred_path.is_absolute() else pred_path
        pred = read_json(pred_path)[task]["predictions"]
        data_path = GLUE_DATA / f"{task}.valid.jsonl"
        labels = [json.loads(line)["label"] for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(pred) != len(labels):
            raise RuntimeError(f"{task} length mismatch predictions={len(pred)} labels={len(labels)}")
        correct = sum(1 for p, y in zip(pred, labels) if int(p["pred"]) == int(y))
        acc = correct / len(labels) * 100.0
        rows.append({"task": task, "predictions": str(pred_path), "valid_data": str(data_path), "num_examples": len(labels), "correct": correct, "accuracy": acc})
    glue = sum(r["accuracy"] for r in rows) / len(rows)
    return glue, rows


def main():
    wwm_reference = read_json(WWM_REFERENCE_PATH)
    aoa = read_json(AOA)
    super_data = read_json(SUPER)
    glue, glue_rows = calc_glue(super_data)
    scores = dict(wwm_reference["scores"])
    scores["superglue"] = glue
    scores["aoa"] = float(aoa["aoa"])
    derived = dict(wwm_reference.get("derived_columns", {}))
    derived["GlobalPIQA_mean_parallel_nonparallel"] = (scores["global_piqa_parallel"] + scores["global_piqa_nonparallel"]) / 2.0
    # Reading aggregation remains not asserted as official unless leaderboard code confirms it. Store both raw components and a simple mean for sensitivity.
    derived["Reading_mean_eye_self_paced"] = (scores["reading_eye_tracking"] + scores["reading_self_paced"]) / 2.0
    # 8/9 direct columns if using GlobalPIQA mean and Reading simple mean; EWoK absent, so no true Overall.
    direct8_for_sensitivity = [
        scores["blimp"], scores["supplement"], scores["entity_tracking"], scores["comps"],
        derived["GlobalPIQA_mean_parallel_nonparallel"], scores["superglue"], derived["Reading_mean_eye_self_paced"], scores["aoa"],
    ]
    payload = {
        "model_path": wwm_reference["model_path"],
        "revision": wwm_reference["revision"],
        "backend": wwm_reference["backend"],
        "status": "8_of_9_columns_available_EWoK_missing",
        "scores": scores,
        "derived_columns": derived,
        "superglue_subtasks": glue_rows,
        "aoa_source": aoa,
        "reports": {**wwm_reference.get("reports", {}), "aoa": aoa.get("score_path"), "superglue_summary": str(SUPER)},
        "missing_columns": ["EWoK"],
        "ewok_state": {
            "full_eval_dir": "experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/ewok_filtered",
            "local_full_eval_files": 0,
            "known_blocker": "ewok-core/ewok-core-1.0 is gated/unauthenticated; official download_evals.py did not populate full_eval/ewok_filtered",
            "fast_ewok_available_but_not_full_substitute": "experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/fast_eval/evaluation_data/fast_eval/ewok_fast",
        },
        "no_true_overall_reason": "EWoK full-eval column is missing; complete nine-column Overall cannot be computed.",
        "sensitivity_not_official_overall": {
            "mean_of_available_8_using_globalpiqa_mean_and_reading_mean": sum(direct8_for_sensitivity)/len(direct8_for_sensitivity),
            "columns_used": ["BLiMP", "Supplement", "Entity", "COMPS", "GlobalPIQA_mean", "SuperGLUE", "Reading_mean", "AoA"],
            "warning": "This is not the official Overall because EWoK is missing and Reading/GlobalPIQA aggregation should be confirmed from leaderboard/collation rules."
        },
        "input_files": {"research": str(WWM_REFERENCE_PATH), "aoa": str(AOA), "superglue": str(SUPER)},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research — WWM100M coordinate after AoA and (Super)GLUE", "",
        f"Evidence JSON: `{OUT}`", "",
        "Full official Overall is still not computable because EWoK full-eval data is missing.", "",
        "| column/task | score | status |", "|---|---:|---|",
        f"| BLiMP | {scores['blimp']:.2f} | full eval |",
        f"| BLiMP Supplement | {scores['supplement']:.2f} | full eval |",
        "| EWoK | — | missing: local full_eval/ewok_filtered empty; upstream gated |",
        f"| Entity Tracking | {scores['entity_tracking']:.2f} | full eval |",
        f"| COMPS | {scores['comps']:.2f} | full eval |",
        f"| (Super)GLUE | {scores['superglue']:.2f} | official finetune predictions aggregated by validation accuracy |",
        f"| GlobalPIQA parallel | {scores['global_piqa_parallel']:.2f} | full eval subcomponent |",
        f"| GlobalPIQA nonparallel | {scores['global_piqa_nonparallel']:.2f} | full eval subcomponent |",
        f"| GlobalPIQA mean | {derived['GlobalPIQA_mean_parallel_nonparallel']:.2f} | derived |",
        f"| Reading eye | {scores['reading_eye_tracking']:.2f} | full eval subcomponent |",
        f"| Reading self-paced | {scores['reading_self_paced']:.2f} | full eval subcomponent |",
        f"| Reading simple mean | {derived['Reading_mean_eye_self_paced']:.2f} | derived sensitivity |",
        f"| AoA | {scores['aoa']:.2f} | official AoA runner; warning: 0 valid words for correlation |",
        "", "## (Super)GLUE subtasks", "", "| task | accuracy | correct / total |", "|---|---:|---:|",
    ]
    for r in glue_rows:
        lines.append(f"| {r['task']} | {r['accuracy']:.2f} | {r['correct']} / {r['num_examples']} |")
    lines += ["", f"Mean (Super)GLUE: {glue:.2f}", "", "The sensitivity mean of the eight currently available columns is not an official Overall and should not be used as a leaderboard claim."]
    NOTE.write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"status":"WWM100M_8OF9_COORDINATE_DONE", "out": str(OUT), "superglue": glue, "aoa": scores["aoa"], "missing": payload["missing_columns"], "sensitivity_available8": payload["sensitivity_not_official_overall"]["mean_of_available_8_using_globalpiqa_mean_and_reading_mean"]}, indent=2))

if __name__ == "__main__":
    main()
