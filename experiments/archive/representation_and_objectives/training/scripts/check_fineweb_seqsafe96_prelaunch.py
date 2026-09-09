#!/usr/bin/env python3
"""Check that the research seqsafe96 FineWeb fallback is ready to launch.

This performs the same invariant checks as the shell launcher without starting any
training job.
"""
from __future__ import annotations
import ast
import json
import pathlib

SCRIPT_DIR = pathlib.Path("experiments/archive/representation_and_objectives/training/scripts")
VERIFY = pathlib.Path("experiments/archive/representation_and_objectives/training/data/cached_fineweb_seqsafe96_candidate/materialization_verification.json")
VIS = pathlib.Path("experiments/archive/representation_and_objectives/training/data/cached_fineweb_seqsafe96_candidate/seq256_visibility_audit.json")

PY_SCRIPTS = [
    "summarize_fineweb_seqsafe96_noaoa_delta.py",
    "audit_seqsafe96_candidate_visibility.py",
    "materialize_cached_fineweb_seqsafe_candidate.py",
    "audit_cached_fineweb_quality.py",
    "probe_fineweb_seqsafe_chunk_lengths.py",
]


def main() -> None:
    syntax = {}
    for name in PY_SCRIPTS:
        p = SCRIPT_DIR / name
        ast.parse(p.read_text(encoding="utf-8"))
        syntax[name] = True
    v = json.loads(VERIFY.read_text(encoding="utf-8"))
    s = json.loads(VIS.read_text(encoding="utf-8"))
    fw = s["results"]["treatment"]["focus_by_source_prefix"]["fineweb_edu_random_quality_single_doc_seqsafe_cached_initial_model_studies"]
    ct = s["results"]["control"]["focus_by_source_prefix"]["official_lengthmatched_to_seqsafe_fineweb"]
    need = {
        "materialization_status": v.get("status") == "CACHED_FINEWEB_SEQSAFE_CANDIDATE_VERIFIED",
        "write_training": bool(v.get("write_training")),
        "training_exposure_words": v.get("training_exposure_words") == 100000000,
        "all_required_checks_pass": bool(v.get("all_required_checks_pass")),
        "chunk_words_96": v.get("chunk_words") == 96,
        "row_length_sequence_matched": v.get("required_checks", {}).get("row_length_sequence_identical"),
        "qwen_pair_identical": v.get("required_checks", {}).get("qwen_pair_block_identical"),
        "tail_identical": v.get("required_checks", {}).get("tail_filler_identical"),
        "fineweb_words_expected": v.get("fineweb_words") == 1753280,
        "visibility_status": s.get("status") == "SEQSAFE96_CANDIDATE_SEQ256_VISIBILITY",
        "fineweb_visible_fraction_gt_999": fw.get("visible_word_fraction", 0) > 0.999,
        "control_visible_fraction_gt_999": ct.get("visible_word_fraction", 0) > 0.999,
        "replacement_visibility_delta_small": abs(fw.get("visible_word_fraction", 0) - ct.get("visible_word_fraction", 0)) < 0.001,
    }
    payload = {
        "status": "FINEWEB_SEQSAFE96_PRELAUNCH_CHECK",
        "syntax_ast_ok": syntax,
        "prelaunch_requirements": need,
        "all_prelaunch_checks_pass": all(need.values()),
        "fineweb_visible_word_fraction": fw.get("visible_word_fraction"),
        "control_visible_word_fraction": ct.get("visible_word_fraction"),
        "fineweb_hidden_words": fw.get("hidden_words"),
        "control_hidden_words": ct.get("hidden_words"),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if not payload["all_prelaunch_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
