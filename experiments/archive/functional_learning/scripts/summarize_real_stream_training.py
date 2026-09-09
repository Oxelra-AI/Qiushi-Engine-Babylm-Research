#!/usr/bin/env python3
"""research: summarize real-stream training trajectories and credit allocation.

This consumes the completed research appended-tail 80-update comparison and the
available research credit audit. It does not run evaluation; it prepares the exact
training-side evidence needed to interpret the pending common-target/Cheap7 readout.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
TRAIN_ROOT = _public_path('experiments/archive/functional_learning/data/real_stream_policy_comparison')
CREDIT_SUMMARY = _public_path('experiments/archive/functional_learning/data/real_stream_credit_audit/available_credit_audit_summary.json')
OUT_DIR = _public_path('experiments/archive/functional_learning/data/real_stream_training_synthesis')


def rel(p: Path | str) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def stats(vals: List[float]) -> Dict[str, Any]:
    vals = [float(v) for v in vals if v is not None]
    if not vals:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(vals),
        "mean": sum(vals) / len(vals),
        "median": statistics.median(vals),
        "min": min(vals),
        "max": max(vals),
    }


def summarize_objective(objective: str) -> Dict[str, Any]:
    root = TRAIN_ROOT / objective
    summary = load_json(root / "train_summary.json")
    logs = load_jsonl(root / "update_log.jsonl")
    kind_counts = Counter()
    for r in logs:
        kind_counts.update(r.get("focus_selected_candidate_kind_counts") or {})
    ckpt_paths = [Path(ROOT / c["path"]) for c in summary.get("checkpoints", [])]
    return {
        "objective": objective,
        "train_summary_path": rel(root / "train_summary.json"),
        "update_log_path": rel(root / "update_log.jsonl"),
        "completed_updates": int(summary.get("completed_updates", len(logs))),
        "total_words_consumed": int(summary.get("total_words_consumed", 0)),
        "total_targets": int(summary.get("total_targets", 0)),
        "total_ordinary_targets": int(summary.get("total_ordinary_targets", 0)),
        "total_focus_targets": int(summary.get("total_focus_targets", 0)),
        "mean_focus_target_fraction": float(summary.get("mean_focus_target_fraction", 0.0)),
        "loss_stats": stats([float(r.get("loss", 0.0)) for r in logs]),
        "grad_norm_preclip_stats": stats([float(r.get("grad_norm_preclip", 0.0)) for r in logs]),
        "focus_fraction_stats": stats([float(r.get("focus_target_fraction", 0.0)) for r in logs]),
        "qwen_rows_total": sum(int(r.get("qwen_rows", 0)) for r in logs),
        "compact_modified_rows_total": sum(int(r.get("compact_modified_rows", 0)) for r in logs),
        "focus_selected_groups_total": sum(int(r.get("focus_selected_groups", 0)) for r in logs),
        "focus_candidate_groups_total": sum(int(r.get("focus_candidate_groups", 0)) for r in logs),
        "focus_selected_pair_refs_total": sum(int(r.get("focus_selected_pair_count", 0)) for r in logs),
        "focus_selected_candidate_kind_counts_total": dict(kind_counts),
        "first_update": logs[0] if logs else None,
        "update_40": next((r for r in logs if int(r.get("update", -1)) == 40), None),
        "final_update": logs[-1] if logs else None,
        "checkpoints": [{"path": rel(p), "exists": p.exists(), "bytes_model": (p / "model.safetensors").stat().st_size if (p / "model.safetensors").exists() else None} for p in ckpt_paths],
        "model_identity": summary.get("model_identity", {}),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison = load_json(_public_path('experiments/archive/functional_learning/data/real_stream_policy_comparison/comparison_train_summary.json'))
    inherited = summarize_objective("inherited_wwm")
    focus = summarize_objective("correspondence_focus")
    credit = load_json(CREDIT_SUMMARY) if CREDIT_SUMMARY.exists() else {}
    appended_credit = None
    for ts in credit.get("tail_summaries", []):
        if ts.get("tail_name") == "structural_appended":
            appended_credit = ts
            break
    # Derived comparisons useful for interpreting evaluation.
    total_ratio = focus["total_targets"] / inherited["total_targets"] if inherited["total_targets"] else None
    qwen_target_fraction_focus = focus["total_focus_targets"] / focus["total_targets"] if focus["total_targets"] else None
    focus_kind = focus["focus_selected_candidate_kind_counts_total"]
    focus_kind_total = sum(int(v) for v in focus_kind.values())
    focus_kind_fracs = {k: (int(v) / focus_kind_total if focus_kind_total else 0.0) for k, v in focus_kind.items()}
    final = {
        "status": "REAL_STREAM_TRAINING_SYNTHESIS",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "train_root": rel(TRAIN_ROOT),
        "comparison_train_summary": rel(_public_path('experiments/archive/functional_learning/data/real_stream_policy_comparison/comparison_train_summary.json')),
        "plan": comparison.get("plan"),
        "prefix_scope": comparison.get("summaries", [{}])[0].get("prefix_info", {}),
        "objective_summaries": {
            "inherited_wwm": inherited,
            "correspondence_focus": focus,
        },
        "derived_comparison": {
            "focus_total_targets_over_inherited_total_targets": total_ratio,
            "focus_qwen_target_fraction_of_its_total_targets": qwen_target_fraction_focus,
            "focus_target_groups_selected_over_candidate": (focus["focus_selected_groups_total"] / focus["focus_candidate_groups_total"]) if focus["focus_candidate_groups_total"] else None,
            "focus_selected_candidate_kind_fractions": focus_kind_fracs,
            "focus_unverified_structural_fraction_of_selected_groups": focus_kind_fracs.get("unverified_structural_shortening_candidate"),
            "focus_inherited_or_not_in_policy_fraction_of_selected_groups": focus_kind_fracs.get("not_in_policy", 0.0) + focus_kind_fracs.get("not_structural_candidate", 0.0) + focus_kind_fracs.get("keep_current", 0.0) + focus_kind_fracs.get("exact_source_return_repair_or_recurrence", 0.0),
            "training_time_sec_inherited": inherited.get("final_update", {}).get("elapsed_sec"),
            "training_time_sec_focus": focus.get("final_update", {}).get("elapsed_sec"),
        },
        "credit_audit_structural_appended": appended_credit,
        "scientific_interpretation": (
            "The two models consumed identical appended structural-policy rows for 80 updates. "
            "The correspondence-focus arm kept source spans visible on qwen rows and targeted second-view content words, "
            "but it used fewer total targets and only about 4.6% qwen target mass versus about 14.9% qwen target mass under inherited WWM. "
            "Most selected focus groups were not admitted structural compactions; only the unverified_structural_shortening_candidate subset directly tests generated shortening. "
            "Therefore common-target/Cheap7 evaluation must distinguish better correspondence use from reduced qwen pressure and broad-preservation effects."
        ),
    }
    (_public_path('experiments/archive/functional_learning/data/real_stream_training_synthesis/training_synthesis.json')).write_text(json.dumps(final, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(final, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
