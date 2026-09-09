#!/usr/bin/env python3
"""research: summarize complete legal-tokenizer endpoint results.

Reads the two complete pristine collation summaries produced in research and writes
a compact result table. This is post-result evidence, not a
pre-result readiness artifact.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import pathlib
import time


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
W = USER_ROOT / "experiments/archive/frontier_consolidation"
OUT = W / "data/compliant_endpoint_results"
ENDPOINTS = {
    "same_pool_tokenizer": {
        "summary": W / "data/compliant_pristine_collate/complianttok_reinvest_seed43022/pristine_collate_complianttok_reinvest_seed43022_summary.json",
        "driver": W / "data/compliant_postdelivery_driver/complianttok_reinvest_seed43022_postdelivery_driver.json",
        "inspection": W / "data/compliant_retrain_inspection/compliant_retrain_inspection.json",
        "training_metrics": W / "training/runs/complianttok_reinvest_seed43022_r2/scientific_metrics.json",
        "tokenizer_sha": "91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9",
        "tokenizer_note": "same-pool legal tokenizer; lacks explicit ByteLevel alphabet but is budget-compliant",
    },
    "bytealphabet_tokenizer": {
        "summary": W / "data/bytealphatok_pristine_collate/bytealphatok_reinvest_seed43022/pristine_collate_bytealphatok_reinvest_seed43022_summary.json",
        "driver": W / "data/bytealphatok_postdelivery_driver/bytealphatok_reinvest_seed43022_postdelivery_driver.json",
        "inspection": W / "data/bytealphatok_retrain_inspection/compliant_retrain_inspection.json",
        "training_metrics": W / "training/runs/bytealphatok_reinvest_seed43022/scientific_metrics.json",
        "tokenizer_sha": "b2b317e655a96f2c14eb559ce9fa7573f180904d84819cacfb5f61174cf355cf",
        "tokenizer_note": "same-pool legal tokenizer with standard full ByteLevel alphabet",
    },
}
OLD_NONSUBMITTABLE = {
    "Overall": 42.0331347900748,
    "scores": {
        "BLiMP": 66.87232315173485,
        "Supplement": 63.27576417952158,
        "EWoK": 53.536575594886855,
        "Entity": 27.745741097952372,
        "COMPS": 51.968828052457084,
        "SuperGLUE": 71.03604952825312,
        "GlobalPIQA": 35.62135922330097,
        "Reading": 8.241572282566393,
        "AoA": 0.0,
    },
}
LIVE_LEADER_OVERALL = 41.8


def load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: pathlib.Path) -> str:
    try:
        return str(path.resolve().relative_to(USER_ROOT))
    except Exception:
        return str(path)


def endpoint_record(name: str, spec: dict) -> dict:
    summary = load(spec["summary"])
    score_summary = summary["score_summary"]
    scores = score_summary["scores"]
    metrics = load(spec["training_metrics"])
    inspection = load(spec["inspection"])
    inspection_record = inspection["records"][0]
    validation_errors = summary.get("validation_errors") or summary.get("validation", {}).get("errors", [])
    return {
        "name": name,
        "tokenizer_sha": spec["tokenizer_sha"],
        "tokenizer_note": spec["tokenizer_note"],
        "summary_path": rel(spec["summary"]),
        "driver_path": rel(spec["driver"]),
        "inspection_path": rel(spec["inspection"]),
        "training_metrics_path": rel(spec["training_metrics"]),
        "inspection_complete": bool(inspection_record.get("complete_100m_compliant_retrain")),
        "inspection_failed_required": inspection_record.get("failed_required", []),
        "validation_error_count": len(validation_errors),
        "overall": score_summary["Overall"],
        "margin_over_live_leader_41p8": score_summary["Overall"] - LIVE_LEADER_OVERALL,
        "delta_vs_old_nonsubmittable_overall": score_summary["Overall"] - OLD_NONSUBMITTABLE["Overall"],
        "scores": scores,
        "deltas_vs_old_nonsubmittable": {k: scores[k] - OLD_NONSUBMITTABLE["scores"][k] for k in OLD_NONSUBMITTABLE["scores"]},
        "nlp_average": score_summary["NLP_average"],
        "human_like_average": score_summary["Human_like_average"],
        "aoa_row_count_values": summary.get("collated_summary", {}).get("aoa_surprisal_row_count_values"),
        "ewok_total": summary.get("collated_summary", {}).get("ewok_lengths", {}),
        "training": {
            "word_exposure": metrics.get("word_exposure"),
            "actual_training_steps": metrics.get("actual_training_steps"),
            "loss_first": metrics.get("loss_first"),
            "loss_last": metrics.get("loss_last"),
            "parameter_count": metrics.get("parameter_count"),
            "vocab_size": metrics.get("vocab_size"),
            "tokenizer_label": metrics.get("tokenizer_label"),
            "saved_checkpoint_count": len(metrics.get("saved_checkpoints", [])),
        },
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = [endpoint_record(k, v) for k, v in ENDPOINTS.items()]
    best = max(records, key=lambda r: r["overall"])
    pair_delta = {
        k: records[1]["scores"][k] - records[0]["scores"][k]
        for k in OLD_NONSUBMITTABLE["scores"]
    }
    payload = {
        "status": "COMPLIANT_ENDPOINT_RESULTS_SUMMARY",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "live_leader_overall": LIVE_LEADER_OVERALL,
        "old_nonsubmittable_reference": OLD_NONSUBMITTABLE,
        "records": records,
        "best_legal_endpoint": best["name"],
        "best_legal_overall": best["overall"],
        "best_legal_margin_over_live_leader_41p8": best["margin_over_live_leader_41p8"],
        "best_legal_shortfall_to_old_nonsubmittable": OLD_NONSUBMITTABLE["Overall"] - best["overall"],
        "bytealphabet_minus_step35_same_pool": {
            "Overall": records[1]["overall"] - records[0]["overall"],
            "scores": pair_delta,
        },
        "scientific_interpretation": [
            "Both legal-tokenizer retrains are complete and official-compatible; neither clears the live 41.8 Strict-Small target.",
            "The same-pool research tokenizer endpoint is the better legal endpoint despite the newline <unk> weakness; byte-alphabet coverage repair worsened the complete Overall.",
            "The largest old-to-legal losses are EWoK and Supplement for the research endpoint, and EWoK/Supplement/Entity/SuperGLUE for byte-alphabet; GlobalPIQA slightly improves under both legal tokenizers.",
            "The old non-submittable 42.033 result remains mechanism evidence for compact-view reinvestment but cannot be used as a submission endpoint. The current bottleneck is now legal tokenizer/learning-coordinate recovery, not post-delivery evaluation readiness.",
        ],
    }
    out_json = OUT / "compliant_endpoint_results_summary.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def fmt(x: float) -> str:
        return f"{x:.6f}"

    cols = list(OLD_NONSUBMITTABLE["scores"].keys())
    lines = [
        "# research compliant endpoint result summary",
        "",
        f"Live Strict-Small target Overall: **{LIVE_LEADER_OVERALL:.6f}**.",
        f"Old non-submittable reference Overall: **{OLD_NONSUBMITTABLE['Overall']:.6f}**.",
        "",
        "| endpoint | Overall | margin vs 41.8 | delta vs old 42.033 | BLiMP | Supplement | EWoK | Entity | COMPS | SuperGLUE | GlobalPIQA | Reading | AoA |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in records:
        lines.append(
            "| " + r["name"] + " | "
            + " | ".join([
                fmt(r["overall"]),
                fmt(r["margin_over_live_leader_41p8"]),
                fmt(r["delta_vs_old_nonsubmittable_overall"]),
            ] + [fmt(r["scores"][c]) for c in cols])
            + " |"
        )
    lines += [
        "",
        "## Deltas vs old non-submittable 42.033 endpoint",
        "",
        "| endpoint | " + " | ".join(cols) + " |",
        "|---|" + "---:|" * len(cols),
    ]
    for r in records:
        lines.append("| " + r["name"] + " | " + " | ".join(fmt(r["deltas_vs_old_nonsubmittable"][c]) for c in cols) + " |")
    lines += [
        "",
        "## Byte-alphabet minus research same-pool legal endpoint",
        "",
        f"Overall delta: **{fmt(payload['bytealphabet_minus_step35_same_pool']['Overall'])}**.",
        "",
        "| column | bytealphabet - research |",
        "|---|---:|",
    ]
    for c in cols:
        lines.append(f"| {c} | {fmt(pair_delta[c])} |")
    lines += [
        "",
        "## Evidence files",
        "",
    ]
    for r in records:
        lines += [
            f"- `{r['name']}` summary: `{r['summary_path']}`",
            f"  - inspection: `{r['inspection_path']}` complete={r['inspection_complete']} failed_required={r['inspection_failed_required']}",
            f"  - training metrics: `{r['training_metrics_path']}` loss_last={r['training']['loss_last']} checkpoints={r['training']['saved_checkpoint_count']}",
        ]
    lines += ["", f"Full JSON: `{rel(out_json)}`", ""]
    (OUT / "compliant_endpoint_results_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "best_legal_endpoint": payload["best_legal_endpoint"],
        "best_legal_overall": payload["best_legal_overall"],
        "best_legal_margin_over_live_leader_41p8": payload["best_legal_margin_over_live_leader_41p8"],
        "bytealphabet_overall": records[1]["overall"],
        "overall": records[0]["overall"],
        "out_json": rel(out_json),
        "out_md": rel(OUT / "compliant_endpoint_results_summary.md"),
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
