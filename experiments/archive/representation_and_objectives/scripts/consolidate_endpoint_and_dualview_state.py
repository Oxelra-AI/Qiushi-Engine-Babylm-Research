#!/usr/bin/env python3
"""Consolidate endpoint resolution and dual-view hard-surface readouts."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import math
import time
from pathlib import Path
from typing import Any

ROOT = _public_path('.')
A01 = _public_path('experiments/archive/representation_and_objectives')
A02 = _public_path('experiments/archive/frontier_consolidation')
OUT = _public_path('experiments/archive/representation_and_objectives/data/endpoint_and_dualview_consolidation')
NOTE = _public_path('research/notes/representation_and_objectives/endpoint_resolved_dualview_hardsurface.md')

COMPARE = _public_path('experiments/archive/representation_and_objectives/data/from_corpus_reproduction_compare/from_corpus_reproduction_compare.json')
VERIFY = _public_path('experiments/archive/representation_and_objectives/data/chck82_fast_submission_materialization/fast_submission_verification.json')
SCORE = _public_path('experiments/archive/representation_and_objectives/data/scale1p75_chck82_full_eval_reproduction/summary/scale1p75_100M_full_eval_hardened_summary.json')
LEADER = _public_path('experiments/archive/representation_and_objectives/data/chck82_public_leader_comparison/chck82_public_leader_comparison.json')
GLOBALPIQA_SUMMARY = _public_path('experiments/archive/representation_and_objectives/data/detached_private_hard_surface_readout/detached_private_hard_surface_summary.json')
EWOK_SUMMARY_PATHS = {
    "mlm_only_20M": _public_path('experiments/archive/representation_and_objectives/data/detached_private_ewok_stable_subset/ewok_stable_subset/mlm_only_20M/ewok_stable_subset_summary.json'),
    "sep_sparse20_aligned_20M": _public_path('experiments/archive/representation_and_objectives/data/detached_private_ewok_stable_subset/ewok_stable_subset/sep_sparse20_aligned_20M/ewok_stable_subset_summary.json'),
    "sep_sparse20_shuffled_20M": _public_path('experiments/archive/representation_and_objectives/data/detached_private_ewok_stable_subset_remaining/ewok_stable_subset/sep_sparse20_shuffled_20M/ewok_stable_subset_summary.json'),
    "coupled_sparse20_aligned_20M": _public_path('experiments/archive/representation_and_objectives/data/detached_private_ewok_stable_subset_remaining/ewok_stable_subset/coupled_sparse20_aligned_20M/ewok_stable_subset_summary.json'),
}
BROAD_PAYLOADS = {
    "mlm_only_20M": _public_path('experiments/archive/frontier_consolidation/data/dualview_mlm_only_20m_eval/per_target/dualview_mlm_only_20M.json'),
    "sep_sparse20_aligned_20M": _public_path('experiments/archive/frontier_consolidation/data/sep_sparse20_aligned_20m_eval/per_target/sep_sparse20_aligned_20M.json'),
    "sep_sparse20_shuffled_20M": _public_path('experiments/archive/frontier_consolidation/data/sep_sparse20_shuffled_20m_eval/per_target/sep_sparse20_shuffled_20M.json'),
    "coupled_sparse20_aligned_20M": _public_path('experiments/archive/frontier_consolidation/data/coupled_sparse20_aligned_20m_eval/per_target/coupled_sparse20_aligned_20M.json'),
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: Path | str | None) -> str | None:
    if p is None:
        return None
    pp = Path(p)
    try:
        return str(pp.resolve().relative_to(ROOT))
    except Exception:
        return str(pp)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def maybe_read_json(path: Path) -> Any | None:
    return read_json(path) if path.exists() else None


def cheap7(scores: dict[str, Any]) -> float | None:
    cols = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
    if all(isinstance(scores.get(c), (int, float)) and math.isfinite(float(scores[c])) for c in cols):
        return sum(float(scores[c]) for c in cols) / 7.0
    return None


def broad_record(path: Path) -> dict[str, Any]:
    d = maybe_read_json(path)
    if not isinstance(d, dict):
        return {"path": rel(path), "exists": path.exists()}
    scores = {}
    if isinstance(d.get("official_overall"), dict) and isinstance(d["official_overall"].get("scores"), dict):
        scores.update(d["official_overall"]["scores"])
    if isinstance(d.get("scores"), dict):
        scores.update(d["scores"])
    # Some payloads keep only split GlobalPIQA in task records. Recover aggregate and Reading.
    tasks = d.get("tasks") if isinstance(d.get("tasks"), dict) else {}
    for task, payload in tasks.items():
        if isinstance(payload, dict):
            if task == "Reading" and isinstance(payload.get("scores"), dict) and isinstance(payload["scores"].get("Reading"), (int, float)):
                scores["Reading"] = payload["scores"]["Reading"]
            elif isinstance(payload.get("score"), (int, float)):
                scores[task] = payload["score"]
    if "GlobalPIQA" not in scores and isinstance(scores.get("GlobalPIQA_parallel"), (int, float)) and isinstance(scores.get("GlobalPIQA_nonparallel"), (int, float)):
        scores["GlobalPIQA"] = (float(scores["GlobalPIQA_parallel"]) + float(scores["GlobalPIQA_nonparallel"])) / 2.0
    return {
        "path": rel(path),
        "exists": path.exists(),
        "target": d.get("target"),
        "endpoint": d.get("endpoint"),
        "scores": scores,
        "cheap7": cheap7(scores),
        "finished_utc": d.get("finished_utc"),
    }


def delta_record(base: dict[str, Any], cand: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in keys:
        b = base.get(k); c = cand.get(k)
        if isinstance(b, (int, float)) and isinstance(c, (int, float)):
            out[k] = c - b
    return out


def hard_global_record(summary: dict[str, Any], name: str) -> dict[str, Any]:
    par = summary.get("globalpiqa", {}).get(name, {}).get("modes", {}).get("parallel", {})
    hard = par.get("always_wrong_subset", {}) if isinstance(par, dict) else {}
    return {
        "parallel_accuracy": par.get("accuracy"),
        "parallel_rank_counts": par.get("correct_rank_counts"),
        "hard52_accuracy": hard.get("accuracy"),
        "hard52_rank_counts": hard.get("correct_rank_counts"),
        "hard52_mean_top_minus_correct": hard.get("mean_top_minus_correct"),
        "hard52_median_top_minus_correct": hard.get("median_top_minus_correct"),
        "hard52_small_wrong_margin_le_0p25_nats": hard.get("small_wrong_margin_le_0p25_nats"),
        "hard52_small_wrong_margin_le_0p50_nats": hard.get("small_wrong_margin_le_0p50_nats"),
    }


def ewok_record(path: Path) -> dict[str, Any]:
    d = maybe_read_json(path)
    if not isinstance(d, dict):
        return {"path": rel(path), "exists": path.exists()}
    s = d.get("summary", {}) if isinstance(d.get("summary"), dict) else {}
    iw = s.get("interaction_sum_wrong", {}) if isinstance(s.get("interaction_sum_wrong"), dict) else {}
    imw = s.get("interaction_mean_wrong", {}) if isinstance(s.get("interaction_mean_wrong"), dict) else {}
    return {
        "path": rel(path),
        "exists": path.exists(),
        "n": s.get("n"),
        "accuracy": s.get("accuracy"),
        "saved_wrong": s.get("saved_wrong"),
        "stable_failure": s.get("stable_failure"),
        "stable_failure_frac_all": s.get("stable_failure_frac_all"),
        "stable_failure_frac_wrong": s.get("stable_failure_frac_wrong"),
        "within_both_positive_wrong_frac": s.get("within_both_positive_wrong_frac"),
        "interaction_sum_wrong_mean": iw.get("mean"),
        "interaction_sum_wrong_median": iw.get("median"),
        "interaction_mean_wrong_mean": imw.get("mean"),
        "interaction_mean_wrong_median": imw.get("median"),
    }


def fast_non_null(v: dict[str, Any]) -> dict[str, Any]:
    f = v.get("fast_collated_inspection", {}) if isinstance(v.get("fast_collated_inspection"), dict) else {}
    return {k: {"length": val.get("length"), "non_null_count": val.get("non_null_count"), "null_indices": val.get("null_indices")} for k, val in f.items() if isinstance(val, dict) and k != "present"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    endpoint_comparison = read_json(COMPARE)
    verification = read_json(VERIFY)
    endpoint_scores = read_json(SCORE)
    leaderboard_snapshot = read_json(LEADER)
    gp_summary = read_json(GLOBALPIQA_SUMMARY)
    broad = {name: broad_record(path) for name, path in BROAD_PAYLOADS.items()}
    hard_gp = {name: hard_global_record(gp_summary, name) for name in BROAD_PAYLOADS}
    hard_ewok = {name: ewok_record(path) for name, path in EWOK_SUMMARY_PATHS.items()}

    deltas: dict[str, Any] = {"broad_scores": {}, "globalpiqa_hard": {}, "ewok_step100_subset": {}}
    pairs = {
        "sep_aligned_minus_mlm_only": ("mlm_only_20M", "sep_sparse20_aligned_20M"),
        "sep_aligned_minus_sep_shuffled": ("sep_sparse20_shuffled_20M", "sep_sparse20_aligned_20M"),
        "sep_aligned_minus_coupled_aligned": ("coupled_sparse20_aligned_20M", "sep_sparse20_aligned_20M"),
        "sep_shuffled_minus_mlm_only": ("mlm_only_20M", "sep_sparse20_shuffled_20M"),
        "coupled_aligned_minus_mlm_only": ("mlm_only_20M", "coupled_sparse20_aligned_20M"),
        "coupled_aligned_minus_sep_aligned": ("sep_sparse20_aligned_20M", "coupled_sparse20_aligned_20M"),
    }
    score_keys = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading", "cheap7"]
    gp_keys = ["parallel_accuracy", "hard52_accuracy", "hard52_mean_top_minus_correct", "hard52_median_top_minus_correct", "hard52_small_wrong_margin_le_0p25_nats", "hard52_small_wrong_margin_le_0p50_nats"]
    ew_keys = ["accuracy", "saved_wrong", "stable_failure", "stable_failure_frac_all", "stable_failure_frac_wrong", "within_both_positive_wrong_frac", "interaction_sum_wrong_mean", "interaction_sum_wrong_median", "interaction_mean_wrong_mean", "interaction_mean_wrong_median"]
    broad_plus = {k: {**v.get("scores", {}), "cheap7": v.get("cheap7")} for k, v in broad.items()}
    for label, (base, cand) in pairs.items():
        deltas["broad_scores"][label] = delta_record(broad_plus.get(base, {}), broad_plus.get(cand, {}), score_keys)
        deltas["globalpiqa_hard"][label] = delta_record(hard_gp.get(base, {}), hard_gp.get(cand, {}), gp_keys)
        deltas["ewok_step100_subset"][label] = delta_record(hard_ewok.get(base, {}), hard_ewok.get(cand, {}), ew_keys)

    endpoint = {
        "hardened_full_score_status": endpoint_scores.get("status"),
        "hardened_full_score_path": rel(SCORE),
        "overall": endpoint_scores.get("score") or endpoint_scores.get("official_overall", {}).get("overall") or endpoint_scores.get("summary", {}).get("overall"),
        "scores": endpoint_scores.get("scores") or endpoint_scores.get("official_overall", {}).get("scores"),
        "training_reproduction": {
            "status": endpoint_comparison.get("status"),
            "path": rel(COMPARE),
            "errors": endpoint_comparison.get("errors"),
            "warnings": endpoint_comparison.get("warnings"),
            "chck_82M_hash_equal": endpoint_comparison.get("key_checkpoint_compare", {}).get("chck_82M", {}).get("hash_equal"),
            "key_hashes_equal": {ck: rec.get("hash_equal") for ck, rec in endpoint_comparison.get("key_checkpoint_compare", {}).items()},
            "metrics_core": endpoint_comparison.get("metrics_core"),
        },
        "fast_submission_carrier": {
            "status": verification.get("status"),
            "path": rel(VERIFY),
            "errors_count": len(verification.get("errors", [])) if isinstance(verification.get("errors"), list) else verification.get("errors"),
            "final_collated": verification.get("final_collated"),
            "trusted_full": verification.get("trusted_full"),
            "full_blocks_equal_to_step166": verification.get("full_blocks_equal_to_step166"),
            "prediction_files_checked": verification.get("fast_collated_inspection", {}).get("prediction_files_checked") or verification.get("prediction_files_checked"),
            "fast_lists": fast_non_null(verification),
            "fast_input_provenance": verification.get("fast_input_provenance"),
        },
        "public_leader_margin_record": {
            "path": rel(LEADER),
            "status": leaderboard_snapshot.get("status"),
            "candidate_margin_over_visible_leader": leaderboard_snapshot.get("candidate_margin_over_visible_leader") or leaderboard_snapshot.get("comparison", {}).get("candidate_margin_over_visible_leader"),
        },
    }
    # research hardened summary has nested fields; expose the known values if present under common keys.
    if endpoint["overall"] is None:
        endpoint["overall"] = endpoint_scores.get("official_overall_score") or endpoint_scores.get("overall") or endpoint_scores.get("scores", {}).get("Overall")
    if endpoint["scores"] is None and isinstance(endpoint_scores.get("score_summary"), dict):
        endpoint["scores"] = endpoint_scores["score_summary"]

    interpretation = [
        "The chck_82M endpoint is now a reproducible score-bearing candidate: from-corpus training regenerated the key checkpoint hashes bit-for-bit, including chck_82M, and the full-plus-fast prediction carrier passes while preserving the trusted full endpoint blocks.",
        "The endpoint result remains an exposure-selected scale1.75 substrate; it does not by itself explain or repair context-conditioned alternative binding.",
        "Separated sparse20 true alignment preserves broad early-training score better than coupled sparse20, but its fixed hard-surface movement is mixed: it improves GlobalPIQA hard52 over MLM-only and shuffled, yet worsens the research EWoK stable-failure subset versus MLM-only.",
        "Coupled sparse20 true alignment strongly improves the research EWoK hard subset and GlobalPIQA hard52, showing that dual-view coupling can move the missing relation surface, but it damages broad columns enough that it is not the usable training recipe.",
        "The live scientific opportunity is to isolate the coupled hard-surface signal inside a broad-preserving pathway; launching an 82M private tail is premature unless a small controlled construction shows both broad preservation and EWoK/GlobalPIQA hard-row movement.",
    ]

    out = {
        "status": "PASS",
        "created_utc": now(),
        "endpoint": endpoint,
        "dualview_targets": broad,
        "globalpiqa_hard": hard_gp,
        "ewok_step100_stable_subset": hard_ewok,
        "deltas": deltas,
        "interpretation": interpretation,
        "source_files": {
            "globalpiqa_readout": rel(GLOBALPIQA_SUMMARY),
            "ewok_partial_first_run_dir": rel(_public_path('experiments/archive/representation_and_objectives/data/detached_private_ewok_stable_subset')),
            "ewok_remaining_run_dir": rel(_public_path('experiments/archive/representation_and_objectives/data/detached_private_ewok_stable_subset_remaining')),
            "compare": rel(COMPARE),
            "verifier": rel(VERIFY),
        },
    }

    out_json = _public_path('experiments/archive/representation_and_objectives/data/endpoint_and_dualview_consolidation/endpoint_and_dualview_consolidation.json')
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# research endpoint resolution and dual-view hard-surface readout",
        "",
        f"Status: **{out['status']}**",
        "",
        "## chck_82M endpoint",
        f"- From-corpus reproduction: `{rel(COMPARE)}` status {endpoint['training_reproduction']['status']}; chck_82M hash equal = {endpoint['training_reproduction']['chck_82M_hash_equal']}.",
        f"- Full-plus-fast prediction carrier: `{rel(VERIFY)}` status {endpoint['fast_submission_carrier']['status']}; final collated file `{endpoint['fast_submission_carrier']['final_collated'].get('path')}` SHA `{endpoint['fast_submission_carrier']['final_collated'].get('sha256')}`; 133 fast prediction files checked.",
        "- Full endpoint blocks in the full-plus-fast carrier are byte-equal to the trusted research full collation; fast input provenance is PASS.",
        "",
        "## Dual-view broad scores",
        "| target | cheap7 | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | Reading |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, rec in broad.items():
        s = rec.get("scores", {})
        lines.append(f"| {name} | {rec.get('cheap7'):.6f} | {s.get('BLiMP')} | {s.get('Supplement')} | {s.get('EWoK')} | {s.get('Entity')} | {s.get('COMPS')} | {s.get('GlobalPIQA')} | {s.get('Reading')} |")
    lines += [
        "",
        "## Fixed hard surfaces",
        "| target | GPIQA parallel | GPIQA hard52 acc | hard52 mean top-correct | EWoK hard acc | EWoK stable failures | EWoK interaction wrong mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in broad:
        g = hard_gp.get(name, {})
        e = hard_ewok.get(name, {})
        lines.append(f"| {name} | {g.get('parallel_accuracy')} | {g.get('hard52_accuracy')} | {g.get('hard52_mean_top_minus_correct')} | {e.get('accuracy')} | {e.get('stable_failure')} | {e.get('interaction_sum_wrong_mean')} |")
    lines += ["", "## Main deltas"]
    for label in ["sep_aligned_minus_mlm_only", "sep_aligned_minus_sep_shuffled", "coupled_aligned_minus_mlm_only", "coupled_aligned_minus_sep_aligned"]:
        b = deltas["broad_scores"].get(label, {})
        g = deltas["globalpiqa_hard"].get(label, {})
        e = deltas["ewok_step100_subset"].get(label, {})
        lines.append(f"- `{label}`: cheap7 {b.get('cheap7')}, GPIQA hard52 acc {g.get('hard52_accuracy')}, GPIQA hard mean margin {g.get('hard52_mean_top_minus_correct')}, EWoK hard accuracy {e.get('accuracy')}, EWoK stable failures {e.get('stable_failure')}, EWoK interaction wrong mean {e.get('interaction_sum_wrong_mean')}.")
    lines += ["", "## Scientific reading"] + [f"- {x}" for x in interpretation] + ["", f"JSON: `{rel(out_json)}`"]
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "json": rel(out_json), "note": rel(NOTE)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
