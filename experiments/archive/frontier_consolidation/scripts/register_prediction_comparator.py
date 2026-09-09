#!/usr/bin/env python3
"""research: compare future MAX-register scores with pre-score predictions.

File-only comparator. It can run before MAX-register scores exist and will then
report predictions only. Once the queued scorer writes register results, rerun
`register_max_readout.py` first, then this script. It binds the observed
`child_minus_adult` trajectory to predictions that were saved before scores:

* committed full-profile prediction from `distribution_proximity_prediction`;
* strict task-aware extraction profile prediction;
* word-unigram control that already failed calibration.

No model loading, training, official evaluation, GPU work, GlobalPIQA,
SuperGLUE, AoA, upload, or leaderboard action.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import json
import math
import pathlib
import statistics
import time
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
PRED = WS / "data/distribution_proximity_prediction"
STRICT = WS / "data/strict_eval_text_prediction"
REG_READOUT = WS / "data/register_max_readout"
OUT = WS / "data/register_prediction_comparison"
STABLE = ["BLiMP", "Supplement", "EWoK", "COMPS", "Reading", "Entity"]
EX_ENTITY = ["BLiMP", "Supplement", "EWoK", "COMPS", "Reading"]
PRIMARY_METRIC = "profile_js"
PRIMARY_SPEC = "common10_80_all3_exEntity"
LEXICAL_CONTROL = ("word_js", "common10_80_all3_exEntity")
STRICT_SPEC = "strict_eval_common10_80_all3_exEntity"
SPREAD_REF = {"exEntity5_mature": 0.1765, "cheap6_pointwise": 0.4199}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def finite(x: Any) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except Exception:
        return False


def fnum(x: Any) -> float | None:
    if not finite(x):
        return None
    return float(x)


def read_csv(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def ranks(vals: list[float]) -> list[float]:
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    out = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and vals[order[j]] == vals[order[i]]:
            j += 1
        r = (i + 1 + j) / 2.0
        for k in range(i, j):
            out[order[k]] = r
        i = j
    return out


def spearman(xs: list[float], ys: list[float]) -> float | None:
    return pearson(ranks(xs), ranks(ys)) if len(xs) >= 2 and len(xs) == len(ys) else None


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def load_predictions() -> dict[tuple[str, str, str], float]:
    out: dict[tuple[str, str, str], float] = {}
    for r in read_csv(PRED / "prediction_rows.csv"):
        val = fnum(r.get("pred_childspeech_removed_minus_adultprose_removed_points"))
        if val is not None:
            out[(r.get("metric", ""), r.get("fit_spec", ""), r.get("family", ""))] = val
    # Strict task-aware extraction variant uses different column names.
    for r in read_csv(STRICT / "prediction_rows.csv"):
        val = fnum(r.get("pred_points"))
        if val is not None:
            out[(r.get("metric", ""), STRICT_SPEC, r.get("family", ""))] = val
    return out


def load_aggregate_predictions() -> dict[tuple[str, str, str], float]:
    out: dict[tuple[str, str, str], float] = {}
    for r in read_csv(PRED / "aggregate_prediction_rows.csv"):
        val = fnum(r.get("pred_childspeech_removed_minus_adultprose_removed_points"))
        if val is not None:
            out[(r.get("metric", ""), r.get("fit_spec", ""), r.get("quantity", ""))] = val
    for r in read_csv(STRICT / "aggregate_prediction_rows.csv"):
        val = fnum(r.get("pred_points"))
        if val is not None:
            out[(r.get("metric", ""), STRICT_SPEC, r.get("quantity", ""))] = val
    return out


def load_observed_trajectory() -> dict[str, dict[str, Any]]:
    rows = read_csv(REG_READOUT / "trajectory_contrasts.csv")
    return {r.get("contrast", ""): r for r in rows}


def obs_mean_for_family(child_row: dict[str, Any], fam: str) -> float | None:
    return fnum(child_row.get(f"{fam}_mean"))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    preds = load_predictions()
    apreds = load_aggregate_predictions()
    obs = load_observed_trajectory()
    child = obs.get("child_minus_adult", {})
    available = int(child.get("complete_checkpoint_count") or 0) if child else 0

    family_rows: list[dict[str, Any]] = []
    obs_vals: list[float] = []
    primary_vals: list[float] = []
    strict_vals: list[float] = []
    word_vals: list[float] = []
    for fam in STABLE:
        observed = obs_mean_for_family(child, fam)
        primary = preds.get((PRIMARY_METRIC, PRIMARY_SPEC, fam))
        strict = preds.get((PRIMARY_METRIC, STRICT_SPEC, fam))
        word = preds.get((LEXICAL_CONTROL[0], LEXICAL_CONTROL[1], fam))
        row = {
            "family": fam,
            "observed_child_minus_adult_points": observed,
            "pred_profile_original_points": primary,
            "pred_profile_strict_eval_text_points": strict,
            "pred_word_control_points": word,
            "observed_minus_profile_original": (observed - primary) if observed is not None and primary is not None else None,
            "observed_minus_profile_strict": (observed - strict) if observed is not None and strict is not None else None,
            "observed_minus_word_control": (observed - word) if observed is not None and word is not None else None,
            "profile_original_sign_match": ((observed > 0) == (primary > 0)) if observed is not None and primary is not None and observed != 0 and primary != 0 else None,
            "profile_strict_sign_match": ((observed > 0) == (strict > 0)) if observed is not None and strict is not None and observed != 0 and strict != 0 else None,
            "word_control_sign_match": ((observed > 0) == (word > 0)) if observed is not None and word is not None and observed != 0 and word != 0 else None,
        }
        family_rows.append(row)
        if fam in EX_ENTITY and observed is not None and primary is not None and strict is not None and word is not None:
            obs_vals.append(observed)
            primary_vals.append(primary)
            strict_vals.append(strict)
            word_vals.append(word)

    aggregate_rows: list[dict[str, Any]] = []
    for quantity, fams, spread_key in [("exEntity5", EX_ENTITY, "exEntity5_mature"), ("cheap6", STABLE, "cheap6_pointwise")]:
        observed_fam_vals = [obs_mean_for_family(child, fam) for fam in fams]
        complete_vals = [v for v in observed_fam_vals if v is not None]
        observed = statistics.mean(complete_vals) if len(complete_vals) == len(fams) else None
        profile_original = apreds.get((PRIMARY_METRIC, PRIMARY_SPEC, quantity))
        profile_strict = apreds.get((PRIMARY_METRIC, STRICT_SPEC, quantity))
        word = apreds.get((LEXICAL_CONTROL[0], LEXICAL_CONTROL[1], quantity))
        spread = SPREAD_REF[spread_key]
        aggregate_rows.append({
            "quantity": quantity,
            "observed_child_minus_adult_points": observed,
            "pred_profile_original_points": profile_original,
            "pred_profile_strict_eval_text_points": profile_strict,
            "pred_word_control_points": word,
            "observed_minus_profile_original": (observed - profile_original) if observed is not None and profile_original is not None else None,
            "observed_minus_profile_strict": (observed - profile_strict) if observed is not None and profile_strict is not None else None,
            "observed_minus_word_control": (observed - word) if observed is not None and word is not None else None,
            "observed_over_spread_ref": (observed / spread) if observed is not None else None,
            "profile_original_over_spread_ref": (profile_original / spread) if profile_original is not None else None,
            "profile_strict_over_spread_ref": (profile_strict / spread) if profile_strict is not None else None,
            "word_control_over_spread_ref": (word / spread) if word is not None else None,
            "families_complete": len(complete_vals),
            "families_expected": len(fams),
        })

    if obs_vals:
        order_agreement = {
            "n_exEntity": len(obs_vals),
            "pearson_original_profile_vs_observed": pearson(primary_vals, obs_vals),
            "spearman_original_profile_vs_observed": spearman(primary_vals, obs_vals),
            "original_profile_sign_match_count": sum((p > 0) == (o > 0) for p, o in zip(primary_vals, obs_vals)),
            "pearson_strict_profile_vs_observed": pearson(strict_vals, obs_vals),
            "spearman_strict_profile_vs_observed": spearman(strict_vals, obs_vals),
            "strict_profile_sign_match_count": sum((p > 0) == (o > 0) for p, o in zip(strict_vals, obs_vals)),
            "pearson_word_control_vs_observed": pearson(word_vals, obs_vals),
            "spearman_word_control_vs_observed": spearman(word_vals, obs_vals),
            "word_control_sign_match_count": sum((p > 0) == (o > 0) for p, o in zip(word_vals, obs_vals)),
        }
    else:
        order_agreement = {
            "n_exEntity": 0,
            "pearson_original_profile_vs_observed": None,
            "spearman_original_profile_vs_observed": None,
            "original_profile_sign_match_count": 0,
            "pearson_strict_profile_vs_observed": None,
            "spearman_strict_profile_vs_observed": None,
            "strict_profile_sign_match_count": 0,
            "pearson_word_control_vs_observed": None,
            "spearman_word_control_vs_observed": None,
            "word_control_sign_match_count": 0,
        }

    interpretation = {
        "data_state": "complete_or_partial_observed" if available else "prediction_only_no_register_scores_yet",
        "complete_child_minus_adult_checkpoints": available,
        "main_reading": "The main pre-score expectation is positive child_minus_adult on exEntity5, around +0.75 to +0.77 over chck_10M..80M. Positive means adult-prose removal was costlier. The profile expectation is strengthened if the family pattern is Supplement/Reading/BLiMP/EWoK positive with COMPS small. Near-zero or negative exEntity weakens the surface/register-profile account.",
        "strict_variant_role": "The strict task-aware extraction reproduces the original profile magnitude and should be reported beside it to avoid over-reliance on heuristic evaluation-text parsing.",
        "word_control_role": "The word-unigram control had poor calibration before scores arrived. Matching its tiny negative value would contradict the profile account but would not by itself validate lexical proximity.",
        "persistence_requirement": "Any resulting principle must retain distinct-content persistence: repeated MAX FineWeb recurrence loses broad ex-Entity value late while distinct V/B content persists.",
    }
    summary = {
        "status": "REGISTER_PREDICTION_COMPARISON_COMPLETE",
        "created_utc": now(),
        "register_readout": rel(REG_READOUT / "trajectory_contrasts.csv"),
        "prediction_sources": {
            "original_profile_and_word": rel(PRED / "prediction_rows.csv"),
            "strict_eval_text": rel(STRICT / "prediction_rows.csv"),
            "commitment": rel(PRED / "prediction_commitment.json"),
        },
        "data_state": interpretation["data_state"],
        "complete_child_minus_adult_checkpoints": available,
        "family_comparison_rows": family_rows,
        "aggregate_comparison_rows": aggregate_rows,
        "exEntity_order_agreement": order_agreement,
        "interpretation": interpretation,
        "files": {
            "summary_json": rel(OUT / "register_prediction_comparison_summary.json"),
            "summary_md": rel(OUT / "register_prediction_comparison_summary.md"),
            "family_csv": rel(OUT / "family_prediction_vs_observed.csv"),
            "aggregate_csv": rel(OUT / "aggregate_prediction_vs_observed.csv"),
        },
        "no_model_loading_training_official_evaluation_gpu_globalpiqa_superglue_aoa_upload_or_leaderboard": True,
    }
    write_csv(OUT / "family_prediction_vs_observed.csv", family_rows)
    write_csv(OUT / "aggregate_prediction_vs_observed.csv", aggregate_rows)
    (OUT / "register_prediction_comparison_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def fmt(x: Any) -> str:
        return "NA" if x is None or x == "" else (f"{float(x):.4f}" if finite(x) else str(x))

    lines = [
        "# research register prediction comparison",
        "",
        f"Data state: `{summary['data_state']}`; complete child-minus-adult checkpoints: {available}.",
        "",
        "This file binds future MAX-register scores to pre-score predictions. It is valid to run before scores exist; rows then contain predictions only.",
        "",
        "## Aggregate comparison",
        "",
        "| quantity | observed | profile original | profile strict eval-text | word control | obs-profile original | obs-profile strict | obs-word |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in aggregate_rows:
        lines.append(
            f"| {r['quantity']} | {fmt(r['observed_child_minus_adult_points'])} | {fmt(r['pred_profile_original_points'])} | {fmt(r['pred_profile_strict_eval_text_points'])} | {fmt(r['pred_word_control_points'])} | {fmt(r['observed_minus_profile_original'])} | {fmt(r['observed_minus_profile_strict'])} | {fmt(r['observed_minus_word_control'])} |"
        )
    lines += [
        "",
        "## Family comparison",
        "",
        "| family | observed | profile original | profile strict eval-text | word control |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in family_rows:
        lines.append(f"| {r['family']} | {fmt(r['observed_child_minus_adult_points'])} | {fmt(r['pred_profile_original_points'])} | {fmt(r['pred_profile_strict_eval_text_points'])} | {fmt(r['pred_word_control_points'])} |")
    lines += [
        "",
        "## Reading",
        "",
        interpretation["main_reading"],
        "",
        interpretation["strict_variant_role"],
        "",
        interpretation["word_control_role"],
        "",
        interpretation["persistence_requirement"],
        "",
        f"JSON: `{rel(OUT / 'register_prediction_comparison_summary.json')}`",
    ]
    (OUT / "register_prediction_comparison_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "data_state": summary["data_state"],
        "complete_child_minus_adult_checkpoints": available,
        "aggregate_comparison_rows": aggregate_rows,
        "summary_md": summary["files"]["summary_md"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
