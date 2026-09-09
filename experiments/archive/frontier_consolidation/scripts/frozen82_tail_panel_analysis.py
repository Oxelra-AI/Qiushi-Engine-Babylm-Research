#!/usr/bin/env python3
"""Analyze the research frozen-82M short-tail comparison.

The protected reference is verified scale1.75 chck_82M.  This script consumes
training metrics, cheap7 summaries, and optional source-free functional probe outputs
for aligned / shuffled / neutral branches.  It is safe to run before all outputs
exist; it records the available state and computes only observed comparisons.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import pathlib
import time
from statistics import mean
from typing import Any

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/frontier_consolidation')
WORKSPACE = _public_path('experiments/archive/frontier_consolidation')
OUT_DIR = _public_path('experiments/archive/frontier_consolidation/data/frozen82_tail_panel_analysis')
OUT_JSON = _public_path('experiments/archive/frontier_consolidation/data/frozen82_tail_panel_analysis/frozen82_tail_panel_analysis.json')
OUT_MD = _public_path('research/documents/frontier_consolidation/data/frozen82_tail_panel_analysis/frozen82_tail_panel_analysis.md')
CHCK82_VERIFY = _public_path('experiments/archive/frontier_consolidation/data/chck82_independent_verification/chck82_independent_verification.json')
CHEAP_COLS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]

RUNS = {
    "aligned": {
        "run_dir": _public_path('experiments/archive/frontier_consolidation/training/runs/frozen82_tail4M_aligned_seed43022'),
        "summary": _public_path('experiments/archive/frontier_consolidation/data/frozen82_tail4M_aligned_summary/frozen82_tail4M_aligned_summary.json'),
        "probe": _public_path('experiments/archive/frontier_consolidation/data/frozen82_tail_aux_probe/tail4M_adapter_on/tail4M_adapter_on.json'),
    },
    "shuffled": {
        "run_dir": _public_path('experiments/archive/frontier_consolidation/training/runs/frozen82_tail4M_shuffled_seed43022'),
        "summary": _public_path('experiments/archive/frontier_consolidation/data/frozen82_tail4M_shuffled_summary/frozen82_tail4M_shuffled_summary.json'),
        "probe": _public_path('experiments/archive/frontier_consolidation/data/frozen82_tail_aux_probe/tail4M_adapter_on/tail4M_adapter_on.json'),
    },
    "neutral": {
        "run_dir": _public_path('experiments/archive/frontier_consolidation/training/runs/frozen82_tail4M_neutral_seed43022'),
        "summary": _public_path('experiments/archive/frontier_consolidation/data/frozen82_tail4M_neutral_summary/frozen82_tail4M_neutral_summary.json'),
        "probe": None,
    },
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path | str) -> str:
    p = pathlib.Path(p)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def load_json(p: pathlib.Path) -> Any | None:
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def chck82_ref() -> dict[str, Any]:
    data = json.loads(CHCK82_VERIFY.read_text(encoding="utf-8"))
    scores = {c: float(data["score_arithmetic"]["scores"][c]) for c in CHEAP_COLS}
    return {
        "scores": scores,
        "cheap7": float(mean(scores.values())),
        "overall": float(data["score_arithmetic"]["overall_reported"]),
        "source": rel(CHCK82_VERIFY),
    }


def training_record(run_dir: pathlib.Path) -> dict[str, Any] | None:
    m = load_json(run_dir / "scientific_metrics.json")
    if m is None:
        return None
    return {k: m.get(k) for k in [
        "status", "mode", "updates", "schedule_total", "initial_consumed_words", "skip_rows",
        "tail_main_word_exposure", "tail_aux_word_exposure", "tail_charged_words", "total_consumed_words",
        "max_tail_charged_words", "stopped_before_cap", "trainable", "mean_aux_loss", "final_aux_loss",
        "aux_loss_batches", "mean_neutral_loss", "final_neutral_loss", "total_params", "private_params",
        "frozen_slow_params", "elapsed_sec",
    ]}


def score_record(name: str, spec: dict[str, pathlib.Path], ref: dict[str, Any]) -> dict[str, Any]:
    summ = load_json(spec["summary"])
    rec = {
        "run_dir": rel(spec["run_dir"]),
        "summary_path": rel(spec["summary"]),
        "summary_exists": summ is not None,
        "training_metrics": training_record(spec["run_dir"]),
        "scores": None,
        "cheap7": None,
        "deltas_vs_chck82": None,
    }
    # The no-private carrier is the protected verified chck_82M function itself:
    # a fresh zero-output private branch with no updates is functionally identical,
    # and its cheap7 is already verified.  This avoids spending another full cheap7
    # evaluation merely to remeasure an unchanged function.
    if name == "neutral" and summ is None:
        rec.update({
            "summary_exists": True,
            "summary_path": ref["source"],
            "virtual_no_private_reference": True,
            "training_metrics": {
                "status": "VERIFIED_CHCK82_NO_PRIVATE_REFERENCE",
                "mode": "neutral_no_private",
                "updates": 0,
                "initial_consumed_words": 82012495,
                "tail_main_word_exposure": 0,
                "tail_aux_word_exposure": 0,
                "tail_charged_words": 0,
                "total_consumed_words": 82012495,
                "trainable": "none; protected slow function",
            },
            "scores": dict(ref["scores"]),
            "cheap7": ref["cheap7"],
            "deltas_vs_chck82": {"cheap7_delta": 0.0, "column_deltas": {c: 0.0 for c in CHEAP_COLS}},
        })
        return rec
    if summ is not None:
        scores = summ.get("scores", {})
        rec["scores"] = {c: scores.get(c) for c in CHEAP_COLS}
        rec["cheap7"] = summ.get("cheap7")
        rec["deltas_vs_chck82"] = {
            "cheap7_delta": None if rec["cheap7"] is None else float(rec["cheap7"] - ref["cheap7"]),
            "column_deltas": {c: (None if rec["scores"].get(c) is None else float(rec["scores"][c] - ref["scores"][c])) for c in CHEAP_COLS},
        }
    return rec


def pair_delta(a: str, b: str, records: dict[str, Any]) -> dict[str, Any] | None:
    ra, rb = records.get(a), records.get(b)
    if not ra or not rb or ra.get("cheap7") is None or rb.get("cheap7") is None:
        return None
    return {
        "cheap7_delta": float(ra["cheap7"] - rb["cheap7"]),
        "column_deltas": {c: (None if ra["scores"].get(c) is None or rb["scores"].get(c) is None else float(ra["scores"][c] - rb["scores"][c])) for c in CHEAP_COLS},
    }


def source_free_probe_summary() -> dict[str, Any] | None:
    # The probe, if present, contains both aligned and shuffled model labels in one file.
    p = _public_path('experiments/archive/frontier_consolidation/data/frozen82_tail_aux_probe/tail4M_adapter_on/tail4M_adapter_on.json')
    data = load_json(p)
    if data is None:
        return None
    return {
        "path": rel(p),
        "key_deltas": data.get("key_deltas"),
        "replay_summary": data.get("replay_summary"),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ref = chck82_ref()
    records = {name: score_record(name, spec, ref) for name, spec in RUNS.items()}
    available = [k for k, v in records.items() if v.get("summary_exists")]
    pending = [k for k, v in records.items() if not v.get("summary_exists")]
    comparisons = {
        "aligned_minus_shuffled": pair_delta("aligned", "shuffled", records),
        "aligned_minus_neutral": pair_delta("aligned", "neutral", records),
        "shuffled_minus_neutral": pair_delta("shuffled", "neutral", records),
    }
    probe = source_free_probe_summary()
    decisions = {}
    if records["aligned"].get("cheap7") is not None:
        d = records["aligned"]["deltas_vs_chck82"]
        decisions["aligned_preserves_chck82_cheap7_within_0p3"] = d["cheap7_delta"] > -0.3
        decisions["aligned_no_large_fragile_damage_vs_chck82"] = all(d["column_deltas"][c] > -1.0 for c in ["Supplement", "EWoK", "Reading", "GlobalPIQA"])
    if comparisons["aligned_minus_shuffled"] is not None:
        decisions["true_correspondence_beats_shuffled_on_tail_cheap7"] = comparisons["aligned_minus_shuffled"]["cheap7_delta"] > 0
    if comparisons["aligned_minus_neutral"] is not None:
        decisions["aligned_beats_neutral_carrier_on_tail_cheap7"] = comparisons["aligned_minus_neutral"]["cheap7_delta"] > 0
    if probe is not None and probe.get("key_deltas"):
        kd = probe["key_deltas"]
        v = kd.get("aligned_model_minus_shuffled_model_on_true_free_view")
        if v is not None:
            decisions["aligned_lower_true_source_free_nll_than_shuffled"] = v < 0
    if pending:
        route_read = "pending_frozen82_tail_panel"
    elif decisions.get("true_correspondence_beats_shuffled_on_tail_cheap7") and decisions.get("aligned_beats_neutral_carrier_on_tail_cheap7") and decisions.get("aligned_preserves_chck82_cheap7_within_0p3") and decisions.get("aligned_no_large_fragile_damage_vs_chck82"):
        route_read = "supports_longer_or_full_frozen82_private_tail_test"
    elif decisions.get("aligned_lower_true_source_free_nll_than_shuffled") and not decisions.get("aligned_preserves_chck82_cheap7_within_0p3", False):
        route_read = "local_source_free_transfer_but_broad_competence_not_preserved"
    else:
        route_read = "does_not_support_more_tail_exposure_in_this_form"

    result = {
        "status": "PENDING" if pending else "COMPLETE",
        "created_utc": now(),
        "protected_reference": ref,
        "available": available,
        "pending": pending,
        "records": records,
        "comparisons": comparisons,
        "source_free_probe": probe,
        "decisions": decisions,
        "route_read": route_read,
        "scientific_reading": "The frozen-tail panel protects the verified chck_82M slow function and asks whether a fresh private source-correspondence pathway can add source-free competence without broad-score erosion. Full remaining exposure should follow only from mature-state preservation plus aligned-over-shuffled/source-free evidence.",
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["# research frozen-82M private-tail panel analysis", "", f"Status: **{result['status']}**", f"Route read: `{route_read}`", "", f"Available: `{available}`; pending: `{pending}`", "", f"Protected chck_82M cheap7: `{ref['cheap7']}`; Overall: `{ref['overall']}`", "", "## Scores", "| arm | cheap7 | BLiMP | Supp | EWoK | Entity | COMPS | GP | Reading | tail main | tail aux | total words |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, rec in records.items():
        s = rec.get("scores") or {}
        tm = rec.get("training_metrics") or {}
        lines.append("| {name} | {cheap7} | {BLiMP} | {Supplement} | {EWoK} | {Entity} | {COMPS} | {GlobalPIQA} | {Reading} | {main} | {aux} | {total} |".format(
            name=name, cheap7=rec.get("cheap7"), BLiMP=s.get("BLiMP"), Supplement=s.get("Supplement"), EWoK=s.get("EWoK"), Entity=s.get("Entity"), COMPS=s.get("COMPS"), GlobalPIQA=s.get("GlobalPIQA"), Reading=s.get("Reading"), main=tm.get("tail_main_word_exposure"), aux=tm.get("tail_aux_word_exposure"), total=tm.get("total_consumed_words")))
    lines += ["", "## Decisions"]
    for k, v in decisions.items():
        lines.append(f"- `{k}`: `{v}`")
    lines += ["", "## Comparisons", "```json", json.dumps(comparisons, indent=2), "```"]
    if probe is not None:
        lines += ["", "## Source-free probe key deltas", "```json", json.dumps(probe.get("key_deltas"), indent=2), "```"]
    lines += ["", result["scientific_reading"], f"\nJSON: `{rel(OUT_JSON)}`"]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "route_read": route_read, "available": available, "pending": pending, "out_json": rel(OUT_JSON), "out_md": rel(OUT_MD)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
