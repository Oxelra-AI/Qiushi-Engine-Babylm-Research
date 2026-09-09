#!/usr/bin/env python3
"""research repair for compact-view GC reference implementation equivalence.

research's queued 1M GC reference used the shortened run's default scheduler
(lr_total_steps=0 -> 26 total steps).  Its row/mask sequence matched the
historical compact_view reference but its learning-rate schedule did not, so the
reported divergence was a schedule-control artifact rather than evidence that
activation checkpointing changes the implementation.

This script compares the corrected research GC reference run, launched with
--lr_total_steps 2529, against the historical research compact_view run through
the same 26-update / 1,027,470-word horizon.  It writes a repaired summary and
also replaces the guard summary consumed by downstream triangle scripts, while
saving the flawed research file under an explicit superseded filename.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import shutil
import time
from pathlib import Path
from typing import Any

ROOT = _public_path('.')
A01 = _public_path('experiments/archive/representation_and_objectives')
A02 = _public_path('experiments/archive/frontier_consolidation')
CORRECTED_RUN = _public_path('experiments/archive/representation_and_objectives/training/runs/gc_compact_view_reinvest_1M_lr2529_seed43022')
REFERENCE_RUN = _public_path('experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022')
GUARD_DIR = _public_path('experiments/archive/representation_and_objectives/data/gc_reference_equivalence')
MAIN_GUARD_SUMMARY = _public_path('experiments/archive/representation_and_objectives/data/gc_reference_equivalence/gc_reference_equivalence_summary.json')
SUPERSEDED_STEP203 = _public_path('experiments/archive/representation_and_objectives/data/gc_reference_equivalence/gc_reference_equivalence_summary_flawed_lr26_superseded.json')
REPAIR_DIR = _public_path('experiments/archive/representation_and_objectives/data/gc_reference_equivalence_repair')
NUMERIC_KEYS = ["loss", "lr", "batch_words", "cumulative_word_exposure", "masked_tokens", "effective_mask_rate"]
METRIC_KEYS = ["word_exposure", "actual_training_steps", "loss_first", "loss_last", "parameter_count", "vocab_size", "tokenizer_label", "seed", "extra_init_seed", "train_rng_seed"]


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def metrics_summary(run: Path) -> dict[str, Any]:
    p = run / "scientific_metrics.json"
    if not p.exists():
        return {"metrics_exists": False, "metrics_path": rel(p)}
    m = read_json(p)
    cps = m.get("saved_checkpoints", [])
    out = {"metrics_exists": True, "metrics_path": rel(p)}
    for k in METRIC_KEYS:
        if k in m:
            out[k] = m[k]
    out["saved_checkpoint_count"] = len(cps)
    out["first_checkpoint"] = cps[0] if cps else None
    cfg = run / "hf_model/chck_1M/config.json"
    if cfg.exists():
        out["chck_1M_activation_checkpointing_flag"] = read_json(cfg).get("activation_checkpointing")
    out["chck_1M_model_exists"] = (run / "hf_model/chck_1M/model.safetensors").exists()
    return out


def compare_logs(corrected_rows: list[dict[str, Any]], reference_rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = min(len(corrected_rows), len(reference_rows))
    max_abs = {k: 0.0 for k in NUMERIC_KEYS}
    mismatch_count = 0
    selected = []
    for i in range(n):
        c = corrected_rows[i]
        r = reference_rows[i]
        rec: dict[str, Any] = {"index": i, "step_corrected": c.get("step"), "step_reference": r.get("step")}
        exact = True
        for k in NUMERIC_KEYS:
            cv = c.get(k)
            rv = r.get(k)
            rec[k + "_corrected"] = cv
            rec[k + "_reference"] = rv
            if isinstance(cv, (int, float)) and isinstance(rv, (int, float)):
                delta = float(cv) - float(rv)
                rec[k + "_delta"] = delta
                max_abs[k] = max(max_abs[k], abs(delta))
                if abs(delta) > 1e-12:
                    exact = False
            elif cv != rv:
                exact = False
        rec["numeric_exact_for_checked_keys"] = exact
        if not exact:
            mismatch_count += 1
        if i < 5 or i == n - 1:
            selected.append(rec)
    return {
        "n_compared_steps": n,
        "corrected_steps_available": len(corrected_rows),
        "reference_steps_available": len(reference_rows),
        "numeric_mismatch_count_at_tol_1e-12": mismatch_count,
        "max_abs_delta": max_abs,
        "selected_step_records": selected,
    }


def main() -> None:
    REPAIR_DIR.mkdir(parents=True, exist_ok=True)
    GUARD_DIR.mkdir(parents=True, exist_ok=True)
    if not CORRECTED_RUN.exists():
        raise FileNotFoundError(CORRECTED_RUN)
    if not REFERENCE_RUN.exists():
        raise FileNotFoundError(REFERENCE_RUN)
    if MAIN_GUARD_SUMMARY.exists() and not SUPERSEDED_STEP203.exists():
        shutil.copy2(MAIN_GUARD_SUMMARY, SUPERSEDED_STEP203)

    corrected_rows = read_jsonl(_public_path('experiments/archive/representation_and_objectives/training/runs/gc_compact_view_reinvest_1M_lr2529_seed43022/training_log.jsonl'))
    reference_rows = read_jsonl(_public_path('experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/training_log.jsonl'))[: len(corrected_rows)]
    comp = compare_logs(corrected_rows, reference_rows)
    corrected_metrics = metrics_summary(CORRECTED_RUN)
    reference_metrics = metrics_summary(REFERENCE_RUN)
    equivalent = (
        corrected_metrics.get("metrics_exists")
        and corrected_metrics.get("word_exposure") == 1_027_470
        and corrected_metrics.get("actual_training_steps") == 26
        and comp["n_compared_steps"] == 26
        and comp["numeric_mismatch_count_at_tol_1e-12"] == 0
    )
    interpretation = "trajectory_equivalent_to_checked_horizon" if equivalent else "trajectory_not_equivalent_after_lr2529_repair"
    payload = {
        "status": "GC_REFERENCE_EQUIVALENCE_REPAIRED_COMPARISON",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "question": "Does activation-checkpointed compact_view_reinvest reproduce the historical compact_view trajectory when the same 2529-step scheduler is used?",
        "repair_reason": "research comparison used lr_total_steps=0 in the 1M shortened run, producing a 26-step schedule; the corrected run uses lr_total_steps=2529, matching historical research.",
        "corrected_run": rel(CORRECTED_RUN),
        "superseded_step203_summary": rel(SUPERSEDED_STEP203),
        "reference_run": rel(REFERENCE_RUN),
        "max_word_exposure_requested": 1_027_470,
        "corrected_metrics": corrected_metrics,
        "reference_metrics_summary": {k: reference_metrics.get(k) for k in ["metrics_exists", "word_exposure", "actual_training_steps", "loss_first", "loss_last", "parameter_count", "vocab_size", "tokenizer_label", "saved_checkpoint_count", "first_checkpoint"]},
        "log_comparison": comp,
        "interpretation": interpretation,
        "allows_historical_reference_for_causal_triangle": bool(equivalent),
        "triangle_use_rule": "The historical compact_view reference can be used against the research activation-checkpointed repeat/adjbreak controls only because the corrected same-data 26-update GC reference is exactly trajectory-equivalent through the chck_1M row-boundary horizon. Keep the superseded research lr26 comparison as an implementation lesson, not as a data-mechanism result.",
    }
    repair_json = _public_path('experiments/archive/representation_and_objectives/data/gc_reference_equivalence_repair/gc_reference_equivalence_repair.json')
    repair_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    MAIN_GUARD_SUMMARY.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    note = [
        "# research repaired compact-view GC reference equivalence",
        "",
        f"Status: `{payload['status']}`.",
        f"Interpretation: `{interpretation}`.",
        "",
        "The research guard summary was superseded because the 1M GC reference run used a 26-step cosine schedule. The corrected research run uses `--lr_total_steps 2529`, matching historical research.",
        f"Compared steps: {comp['n_compared_steps']}; mismatch count at 1e-12: {comp['numeric_mismatch_count_at_tol_1e-12']}; max deltas: `{json.dumps(comp['max_abs_delta'])}`.",
        f"Corrected run: `{rel(CORRECTED_RUN)}`.",
        f"Repaired JSON: `{rel(repair_json)}`. Guard summary overwritten for downstream scripts: `{rel(MAIN_GUARD_SUMMARY)}`.",
        f"Superseded research file: `{rel(SUPERSEDED_STEP203)}`.",
    ]
    (_public_path('research/documents/representation_and_objectives/data/gc_reference_equivalence_repair/gc_reference_equivalence_repair.md')).write_text("\n".join(note) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "interpretation": interpretation, "allows": equivalent, "repair_json": rel(repair_json), "main_guard_summary": rel(MAIN_GUARD_SUMMARY), "max_abs_delta": comp["max_abs_delta"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
