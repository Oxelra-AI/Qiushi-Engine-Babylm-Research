#!/usr/bin/env python3
"""Write a machine-readable research row-chunked vs pairfit comparison.

This script only consolidates already-produced training/evaluation artifacts into one
small JSON file for later agents. It does not rerun evaluation.
"""
from __future__ import annotations

import json
import pathlib
import time

ROOT = pathlib.Path("experiments/archive/frontier_consolidation")
SUMMARY_PATH = ROOT / "data/repaired_pilot_eval/repaired_pilot_eval_summary.json"
OUT_PATH = ROOT / "data/rowchunk_pairfit_comparison.json"
ROW = "stagewise_qwen10_12x384_40k_seed44011"
PAIR = "pairfit_qwen10_12x384_40k_seed44011"


def selected_train_fields(payload: dict) -> dict:
    keep = [
        "parameter_count",
        "word_exposure",
        "actual_training_steps",
        "loss_first",
        "loss_last",
        "elapsed_sec",
        "optimizer",
        "learning_rate",
        "mask_switch_words",
        "amp_dtype",
        "saved_checkpoints",
    ]
    return {k: payload.get(k) for k in keep if k in payload}


def main() -> None:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    table = summary["table"]
    keys = [
        "BLiMP",
        "Supplement",
        "EWoK",
        "Entity",
        "Entity_full",
        "COMPS",
        "GlobalPIQA_parallel",
        "GlobalPIQA_nonparallel",
        "GlobalPIQA_mean",
        "Reading",
        "Reading_eye",
        "Reading_self_paced",
        "equal7_mean",
        "equal7_full_entity",
    ]
    delta = {
        k: round(table[PAIR][k] - table[ROW][k], 6)
        for k in keys
        if table[ROW].get(k) is not None and table[PAIR].get(k) is not None
    }

    row_train_path = ROOT / "training/runs/qwen10_stagewise_rowchunk_40k_12x384_lamb_bf16_seed44011/scientific_metrics.json"
    pair_train_path = ROOT / "training/runs/qwen10_stagewise_pairfit_40k_12x384_lamb_bf16_seed44011/scientific_metrics.json"
    row_train = json.loads(row_train_path.read_text(encoding="utf-8"))
    pair_train = json.loads(pair_train_path.read_text(encoding="utf-8"))

    contract_path = ROOT / "data/pairfit_contract/pairfit_joint_visibility_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))

    out = {
        "status": "ROWCHUNK_PAIRFIT_MATCHED_SCREEN_DONE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "question": "Does restoring original+rewrite joint visibility inside the same 12x384/40k/LAMB stagewise curriculum produce a large official-task gain over row-chunked stagewise clean-Qwen?",
        "result_summary_path": str(SUMMARY_PATH),
        "rowchunk_target": ROW,
        "pairfit_target": PAIR,
        "fast_official_compatible_columns_note": "BLiMP/Supp/EWoK/Entity fast/full/COMPS/GPIQA fast/Reading fast; SuperGLUE and AoA not included.",
        "scores": {ROW: table[ROW], PAIR: table[PAIR]},
        "pairfit_minus_rowchunk": delta,
        "training": {
            ROW: selected_train_fields(row_train),
            PAIR: selected_train_fields(pair_train),
        },
        "mechanism": {
            "rowchunk_complete_pair_joint_visibility_rate": 0.572006,
            "pairfit_complete_pair_joint_visibility_rate": contract.get("complete_pair_joint_visibility_rate"),
            "pairfit_pairs_present_once": contract.get("ok") and contract.get("selected_pair_count") == contract.get("seen_pair_count") and not contract.get("duplicate_or_missing_pair_ids", {}).get("missing") and not contract.get("duplicate_or_missing_pair_ids", {}).get("duplicates"),
            "pairfit_qwen_pair_words_preserved": contract.get("total_qwen_pair_words"),
            "pairfit_truncating_pair_examples": contract.get("n_truncating_pair_examples"),
            "visibility_audit_path": str(ROOT / "data/pair_joint_visibility/pair_joint_visibility_audit.json"),
            "pairfit_contract_path": str(contract_path),
        },
        "interpretation": "Pairfit is mechanistically cleaner and modestly improves several syntactic/semantic columns, but the matched 10M official-task gain is too small (+0.042 equal7 fast, +0.195 equal7 full-Entity) and includes a GlobalPIQA mean decline (-1.91). This weakens further investment in this secondary curriculum arm and supports returning to denser faithful second-view construction.",
        "note_path": str((ROOT.parents[2] / 'research/notes/frontier_consolidation/rowchunk_pairfit_screen_and_route_decision.md')),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "wrote": str(OUT_PATH),
        "delta_equal7": delta.get("equal7_mean"),
        "delta_equal7_full_entity": delta.get("equal7_full_entity"),
        "delta_gpiqa_mean": delta.get("GlobalPIQA_mean"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
