#!/usr/bin/env python3
"""research: static profile/novelty prediction for the in-corpus adult-prose cell.

This CPU-only script is run before reading any new official scores for the
in-corpus cell. It complements the rate prediction by quantifying what a static
surface/profile proximity account would predict from the text blocks alone.

For an arm A vs clean, the profile model uses the same distance form as research:

  predicted_delta = beta * [JS(eval_family, removed_clean_block)
                            - JS(eval_family, admitted_block)]

The beta is imported from the research strict-evaluation-text profile calibration
on the already-scored MAX V/B/R-minus-clean arms. Because the in-corpus and
full-1x cells are ~0.38-0.40 of MAX substitution mass, the script reports both
raw and mass-scaled predictions. Word-JS is retained as the failed lexical
control.

No model loading, no training, no official evaluation, no GPU, no upload.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import importlib.util
import json
import math
import pathlib
import statistics
import time
from typing import Any


def find_root() -> pathlib.Path:
    return _PUBLIC_ROOT

ROOT = find_root()
WS = ROOT / "experiments/archive" / 'frontier_consolidation'
PRED288 = WS / "scripts" / "distribution_proximity_prediction.py"
STRICT288 = WS / "scripts" / "strict_eval_text_prediction.py"
STRICT288_SUMMARY = WS / "data" / "strict_eval_text_prediction" / "strict_eval_text_prediction_summary.json"
CLEAN_POOL = WS / "data" / "dose_2p64x_rowholdout_pools" / "cleanqwen_lengthmatched_dose2p64x_10M.jsonl"
INC_POOL = WS / "data" / "incorpus_adultprose_arm" / "incorpus_adultprose_rho0p042_10M.jsonl"
FULL1X_POOL = WS / "data" / "subdose_ladder_maxgeom_pools" / "subdose_full_1x_view_10M.jsonl"
OUT = WS / "data" / "incorpus_static_profile_prediction"

FAMILIES = ["BLiMP", "Supplement", "EWoK", "COMPS", "Reading", "Entity"]
EXE4 = ["BLiMP", "Supplement", "EWoK", "COMPS"]
EXE5 = ["BLiMP", "Supplement", "EWoK", "COMPS", "Reading"]
CHEAP5_NOREADING = ["BLiMP", "Supplement", "EWoK", "COMPS", "Entity"]
CHEAP6 = FAMILIES
MAX_SUBSTITUTION_WORDS = 1_118_587  # research/284 MAX admitted FineWeb block words, excluding 133w topup convention.


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def import_module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def iter_jsonl(path: pathlib.Path):
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def text(row: dict[str, Any]) -> str:
    return " ".join(str(row.get("text", "")).split())


def words(row: dict[str, Any]) -> int:
    try:
        return int(row.get("words", len(text(row).split())))
    except Exception:
        return len(text(row).split())


def changed_blocks(clean_rows: list[dict[str, Any]], arm_rows: list[dict[str, Any]], arm_name: str) -> dict[str, Any]:
    changed_idx: list[int] = []
    unchanged_idx: list[int] = []
    for i, (c, a) in enumerate(zip(clean_rows, arm_rows)):
        if text(c) != text(a):
            changed_idx.append(i)
        else:
            unchanged_idx.append(i)
    admitted = [text(arm_rows[i]) for i in changed_idx if text(arm_rows[i])]
    removed = [text(clean_rows[i]) for i in changed_idx if text(clean_rows[i])]
    admitted_words = sum(words(arm_rows[i]) for i in changed_idx)
    removed_words = sum(words(clean_rows[i]) for i in changed_idx)
    source_words: dict[str, int] = {}
    for i in changed_idx:
        src = str(arm_rows[i].get("source", "unknown")).split("::")[-1]
        source_words[src] = source_words.get(src, 0) + words(arm_rows[i])
    removed_source_words: dict[str, int] = {}
    for i in changed_idx:
        src = str(clean_rows[i].get("source", "unknown")).split("::")[-1]
        removed_source_words[src] = removed_source_words.get(src, 0) + words(clean_rows[i])
    return {
        "arm": arm_name,
        "changed_indices": changed_idx,
        "unchanged_indices": unchanged_idx,
        "admitted_texts": admitted,
        "removed_texts": removed,
        "changed_rows": len(changed_idx),
        "admitted_words": admitted_words,
        "removed_words": removed_words,
        "source_words": source_words,
        "removed_source_words": removed_source_words,
        "mass_ratio_to_MAX": admitted_words / MAX_SUBSTITUTION_WORDS,
    }


def load_betas() -> dict[str, float]:
    summary = json.loads(STRICT288_SUMMARY.read_text(encoding="utf-8"))
    betas: dict[str, float] = {}
    for row in summary.get("fit_summary_rows", []):
        metric = row.get("metric")
        if metric in {"word_js", "profile_js"}:
            betas[metric] = float(row.get("beta"))
    if "word_js" not in betas or "profile_js" not in betas:
        raise RuntimeError("Could not recover research strict betas")
    return betas


def js(mod, metric: str, texts1: list[str], texts2: list[str]) -> float:
    if metric == "word_js":
        return float(mod.js_divergence(mod.word_counter_from_texts(texts1), mod.word_counter_from_texts(texts2)))
    if metric == "profile_js":
        return float(mod.js_divergence(mod.profile_counter_from_texts(texts1), mod.profile_counter_from_texts(texts2)))
    raise ValueError(metric)


def mean(vals: list[float]) -> float:
    return statistics.mean(vals) if vals else float("nan")


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    mod = import_module(PRED288, "distribution_proximity_prediction_for_step294")
    strict = import_module(STRICT288, "strict_eval_text_prediction_for_step294")
    betas = load_betas()

    clean = load_rows(CLEAN_POOL)
    inc = load_rows(INC_POOL)
    full = load_rows(FULL1X_POOL)
    if len(clean) != len(inc) or len(clean) != len(full):
        raise RuntimeError(f"row count mismatch clean={len(clean)} inc={len(inc)} full={len(full)}")

    blocks = {
        "incorpus_adultprose": changed_blocks(clean, inc, "incorpus_adultprose"),
        "full1x_view": changed_blocks(clean, full, "full1x_view"),
    }
    eval_texts: dict[str, list[str]] = {}
    eval_meta: dict[str, Any] = {}
    for fam in FAMILIES:
        ts, meta = strict.strict_eval_family_texts(fam)
        eval_texts[fam] = ts
        eval_meta[fam] = meta

    distance_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for arm, blk in blocks.items():
        for metric in ["word_js", "profile_js"]:
            beta = betas[metric]
            for fam in FAMILIES:
                d_removed = js(mod, metric, eval_texts[fam], blk["removed_texts"])
                d_admit = js(mod, metric, eval_texts[fam], blk["admitted_texts"])
                x = d_removed - d_admit
                raw_pred = beta * x
                scaled_pred = raw_pred * blk["mass_ratio_to_MAX"]
                distance_rows.append({
                    "arm": arm,
                    "metric": metric,
                    "family": fam,
                    "js_eval_removed": d_removed,
                    "js_eval_admitted": d_admit,
                    "distance_advantage_removed_minus_admitted": x,
                    "beta_from_step288_strict": beta,
                    "raw_prediction_points": raw_pred,
                    "mass_ratio_to_MAX": blk["mass_ratio_to_MAX"],
                    "mass_scaled_prediction_points": scaled_pred,
                })
                prediction_rows.append({
                    "arm": arm,
                    "metric": metric,
                    "family": fam,
                    "prediction_points_raw": raw_pred,
                    "prediction_points_mass_scaled": scaled_pred,
                    "sign_scaled": "positive" if scaled_pred > 0 else ("negative" if scaled_pred < 0 else "zero"),
                })

    aggregate_rows: list[dict[str, Any]] = []
    for arm in blocks:
        for metric in ["word_js", "profile_js"]:
            fam_to_pred = {r["family"]: r for r in prediction_rows if r["arm"] == arm and r["metric"] == metric}
            for qty, fams in [("exEntity4_noReading", EXE4), ("exEntity5_withReading", EXE5), ("cheap5_noReading", CHEAP5_NOREADING), ("cheap6_withReading", CHEAP6)]:
                vals_raw = [float(fam_to_pred[f]["prediction_points_raw"]) for f in fams]
                vals_scaled = [float(fam_to_pred[f]["prediction_points_mass_scaled"]) for f in fams]
                aggregate_rows.append({
                    "arm": arm,
                    "metric": metric,
                    "quantity": qty,
                    "raw_prediction_points": mean(vals_raw),
                    "mass_scaled_prediction_points": mean(vals_scaled),
                    "min_scaled_family": min(vals_scaled),
                    "max_scaled_family": max(vals_scaled),
                    "positive_scaled_family_count": sum(1 for v in vals_scaled if v > 0),
                    "negative_scaled_family_count": sum(1 for v in vals_scaled if v < 0),
                    "families": ";".join(fams),
                })
    # Direct prediction for the matched 70M in-corpus minus full1x contrast by static profile.
    for metric in ["word_js", "profile_js"]:
        for qty in ["exEntity4_noReading", "exEntity5_withReading", "cheap5_noReading", "cheap6_withReading"]:
            ai = next(r for r in aggregate_rows if r["arm"] == "incorpus_adultprose" and r["metric"] == metric and r["quantity"] == qty)
            af = next(r for r in aggregate_rows if r["arm"] == "full1x_view" and r["metric"] == metric and r["quantity"] == qty)
            aggregate_rows.append({
                "arm": "incorpus_minus_full1x_static_prediction",
                "metric": metric,
                "quantity": qty,
                "raw_prediction_points": float(ai["raw_prediction_points"]) - float(af["raw_prediction_points"]),
                "mass_scaled_prediction_points": float(ai["mass_scaled_prediction_points"]) - float(af["mass_scaled_prediction_points"]),
                "min_scaled_family": None,
                "max_scaled_family": None,
                "positive_scaled_family_count": None,
                "negative_scaled_family_count": None,
                "families": ai["families"],
            })

    def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
        fields: list[str] = []
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(rows)

    write_csv(OUT / "distance_rows.csv", distance_rows)
    write_csv(OUT / "prediction_rows.csv", prediction_rows)
    write_csv(OUT / "aggregate_prediction_rows.csv", aggregate_rows)

    def agg(arm: str, metric: str, qty: str) -> dict[str, Any]:
        return next(r for r in aggregate_rows if r["arm"] == arm and r["metric"] == metric and r["quantity"] == qty)

    commitment = {
        "status": "INCORPUS_STATIC_PROFILE_PRE_SCORE_COMMITMENT",
        "created_utc": now(),
        "made_without_reading_step294_official_scores": True,
        "model": "research strict-eval-text beta applied to in-corpus and full1x changed/removed text blocks; word-JS retained as failed lexical control; mass-scaled values reflect ~0.38-0.40x MAX word substitution.",
        "primary_profile_predictions_mass_scaled": {
            "incorpus_minus_clean_exEntity4_noReading": agg("incorpus_adultprose", "profile_js", "exEntity4_noReading")["mass_scaled_prediction_points"],
            "incorpus_minus_clean_exEntity5_withReading": agg("incorpus_adultprose", "profile_js", "exEntity5_withReading")["mass_scaled_prediction_points"],
            "incorpus_minus_clean_cheap5_noReading": agg("incorpus_adultprose", "profile_js", "cheap5_noReading")["mass_scaled_prediction_points"],
            "incorpus_minus_full1x_exEntity4_noReading": agg("incorpus_minus_full1x_static_prediction", "profile_js", "exEntity4_noReading")["mass_scaled_prediction_points"],
            "incorpus_minus_full1x_cheap5_noReading": agg("incorpus_minus_full1x_static_prediction", "profile_js", "cheap5_noReading")["mass_scaled_prediction_points"],
        },
        "word_js_control_predictions_mass_scaled": {
            "incorpus_minus_clean_exEntity4_noReading": agg("incorpus_adultprose", "word_js", "exEntity4_noReading")["mass_scaled_prediction_points"],
            "incorpus_minus_full1x_exEntity4_noReading": agg("incorpus_minus_full1x_static_prediction", "word_js", "exEntity4_noReading")["mass_scaled_prediction_points"],
        },
        "interpretation_rule": {
            "static_profile_proximity": "supported if in-corpus official ex-Entity movement is positive and closer to the mass-scaled profile prediction than to zero even if the late admitted-block loss rate is small",
            "out_of_corpus_novelty": "supported if in-corpus remains near clean while full1x/FineWeb stays positive despite similar or positive profile prediction for in-corpus",
            "rate_account": "supported if the official in-corpus movement follows the future 60M->100M admitted-block loss reduction rather than the static profile prediction alone",
        },
    }
    (OUT / "prediction_commitment.json").write_text(json.dumps(commitment, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = {
        "status": "INCORPUS_STATIC_PROFILE_PREDICTION_DONE",
        "created_utc": now(),
        "inputs": {
            "clean_pool": rel(CLEAN_POOL),
            "incorpus_pool": rel(INC_POOL),
            "full1x_pool": rel(FULL1X_POOL),
            "strict_summary": rel(STRICT288_SUMMARY),
        },
        "blocks": {k: {kk: vv for kk, vv in v.items() if kk not in {"admitted_texts", "removed_texts", "changed_indices", "unchanged_indices"}} for k, v in blocks.items()},
        "betas_from_step288_strict": betas,
        "eval_meta": eval_meta,
        "aggregate_prediction_rows": aggregate_rows,
        "commitment": commitment,
        "files": {
            "summary_json": rel(OUT / "incorpus_static_profile_prediction.json"),
            "summary_md": rel(OUT / "incorpus_static_profile_prediction.md"),
            "commitment_json": rel(OUT / "prediction_commitment.json"),
            "distance_rows_csv": rel(OUT / "distance_rows.csv"),
            "prediction_rows_csv": rel(OUT / "prediction_rows.csv"),
            "aggregate_prediction_rows_csv": rel(OUT / "aggregate_prediction_rows.csv"),
        },
        "no_model_loading_training_official_evaluation_gpu_upload_or_leaderboard": True,
        "elapsed_sec": round(time.time() - t0, 3),
    }
    (OUT / "incorpus_static_profile_prediction.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def fmt(x: Any) -> str:
        return "NA" if x is None else f"{float(x):+.4f}"
    lines = [
        "# research in-corpus static profile prediction",
        "",
        "This CPU-only prediction was made without reading new official scorer outputs. It quantifies the static adult-prose profile/proximity account for the in-corpus final cell and keeps word-JS as a failed lexical control.",
        "",
        "## Blocks",
        "",
    ]
    for name, blk in blocks.items():
        lines.append(f"- {name}: changed_rows={blk['changed_rows']}, admitted_words={blk['admitted_words']}, removed_words={blk['removed_words']}, mass_ratio_to_MAX={blk['mass_ratio_to_MAX']:.4f}, admitted_sources={blk['source_words']}, removed_sources={blk['removed_source_words']}")
    lines += ["", "## Aggregate predictions", "", "| arm/model | metric | quantity | raw Δ points | mass-scaled Δ points |", "|---|---|---|---:|---:|"]
    for r in aggregate_rows:
        if r["quantity"] in {"exEntity4_noReading", "cheap5_noReading"}:
            lines.append(f"| {r['arm']} | {r['metric']} | {r['quantity']} | {fmt(r['raw_prediction_points'])} | {fmt(r['mass_scaled_prediction_points'])} |")
    lines += ["", "## Frozen commitment", "", json.dumps(commitment, indent=2, ensure_ascii=False), "", "## Files"]
    for k, v in summary["files"].items():
        lines.append(f"- {k}: `{v}`")
    (OUT / "incorpus_static_profile_prediction.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": summary["status"],
        "commitment": commitment,
        "summary_md": summary["files"]["summary_md"],
        "elapsed_sec": summary["elapsed_sec"],
        "no_model_loading_training_official_evaluation_gpu_upload_or_leaderboard": True,
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
