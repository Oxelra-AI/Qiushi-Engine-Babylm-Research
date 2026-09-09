#!/usr/bin/env python3
"""research: ablated structural-profile distance predictions.

Companion to the research pre-score prediction.  Recomputes profile JS under
feature-class ablations before MAX-register scores are read, then fits the same
one-parameter calibration against matched-geometry V/B/R-minus-clean family
movements.  This tests whether the profile prediction depends entirely on
transcript/markup artifacts.

No model loading, training, official evaluation, GPU work, GlobalPIQA,
SuperGLUE, AoA, upload, or leaderboard action.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import collections
import csv
import importlib.util
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
PRED_SCRIPT = WS / "scripts/distribution_proximity_prediction.py"
OUT = WS / "data/profile_ablation_prediction"
CONTRAST_CSV = WS / "data/reference_decomposition_readout/contrast_window_summaries.csv"
PAIR_ROWS = 7923
FAMILIES = ["BLiMP", "Supplement", "EWoK", "COMPS", "Reading", "Entity"]
EX_ENTITY = ["BLiMP", "Supplement", "EWoK", "COMPS", "Reading"]
ARM_BLOCK = {"V": "admitted_view", "B": "admitted_breadth", "R": "admitted_repeat"}
ARM_CONTRAST = {"V": "D1_VminusCmax", "B": "D1_BminusCmax", "R": "D1_RminusCmax"}
SPREAD = {"exEntity5": 0.1765, "cheap6": 0.4199}

ABLATIONS = {
    "profile_all": {"drop_groups": [], "description": "full non-open-lexical profile from research"},
    "no_transcript_format": {"drop_groups": ["transcript_or_markup_format"], "description": "drop speaker/tier/bracket/url-like format counters"},
    "no_transcript_no_punct": {"drop_groups": ["transcript_or_markup_format", "punctuation"], "description": "drop transcript/markup and punctuation"},
    "function_only": {"keep_groups": ["function_word_identity", "function_category"], "description": "closed-class function words/categories only"},
    "shape_punct_func_no_format": {"drop_groups": ["transcript_or_markup_format", "sentence_length"], "description": "drop explicit transcript markers and sentence-length bins; keep function, punctuation, shapes, suffixes"},
    "abstract_shape_only": {"keep_groups": ["open_class_abstract_shape", "token_length", "number_token"], "description": "open-class abstract shapes/lengths/numbers only; no function identities or punctuation"},
}


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_pred_module():
    spec = importlib.util.spec_from_file_location("distribution_proximity_prediction", PRED_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {PRED_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def group(k: str) -> str:
    if ":" in k:
        p = k.split(":", 1)[0]
        if p == "func_word":
            return "function_word_identity"
        if p == "func_cat":
            return "function_category"
        if p == "punct":
            return "punctuation"
        if p == "format":
            return "transcript_or_markup_format"
        if p == "sent_len":
            return "sentence_length"
        if p in {"open_shape", "open_len", "open_suffix"}:
            return "open_class_abstract_shape"
        if p == "tok_len":
            return "token_length"
    if k == "tok:number":
        return "number_token"
    return "other"


def filter_counter(c: collections.Counter[str], spec: dict[str, Any]) -> collections.Counter[str]:
    keep = set(spec.get("keep_groups") or [])
    drop = set(spec.get("drop_groups") or [])
    out: collections.Counter[str] = collections.Counter()
    for k, v in c.items():
        g = group(k)
        if keep and g not in keep:
            continue
        if g in drop:
            continue
        out[k] = v
    return out


def js(c1: collections.Counter[str], c2: collections.Counter[str]) -> float:
    n1, n2 = float(sum(c1.values())), float(sum(c2.values()))
    if n1 <= 0 or n2 <= 0:
        return float("nan")
    s = 0.0
    for k in set(c1) | set(c2):
        p = c1.get(k, 0) / n1
        q = c2.get(k, 0) / n2
        m = 0.5 * (p + q)
        if p > 0:
            s += 0.5 * p * math.log(p / m, 2)
        if q > 0:
            s += 0.5 * q * math.log(q / m, 2)
    return s


def read_scores() -> dict[tuple[str, str, str], float]:
    out: dict[tuple[str, str, str], float] = {}
    with CONTRAST_CSV.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            try:
                out[(r["contrast"], r["quantity"], r["window"])] = float(r["mean"])
            except Exception:
                pass
    return out


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


def fit_beta(rows: list[dict[str, Any]]) -> dict[str, Any]:
    xs = [r["x"] for r in rows]
    ys = [r["y"] for r in rows]
    denom = sum(x*x for x in xs)
    beta = sum(x*y for x, y in zip(xs, ys)) / denom if denom else float("nan")
    pred = [beta*x for x in xs]
    sse = sum((y-p)**2 for y, p in zip(ys, pred))
    syy0 = sum(y*y for y in ys)
    syyc = sum((y-statistics.mean(ys))**2 for y in ys) if len(ys) > 1 else float("nan")
    return {
        "beta": beta,
        "n": len(rows),
        "pearson": pearson(xs, ys),
        "spearman": spearman(xs, ys),
        "rmse": math.sqrt(sse/len(rows)) if rows else None,
        "r2_zero": 1 - sse/syy0 if syy0 else None,
        "r2_centered": 1 - sse/syyc if syyc and syyc > 0 else None,
        "x_mean": statistics.mean(xs) if xs else None,
        "x_min": min(xs) if xs else None,
        "x_max": max(xs) if xs else None,
        "y_mean": statistics.mean(ys) if ys else None,
    }


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    mod = load_pred_module()
    # Reconstruct all text sets using the already validated helper functions.
    texts: dict[str, list[str]] = {}
    for name, path in [("admitted_view", mod.VIEW10), ("admitted_repeat", mod.REPEAT10), ("admitted_breadth", mod.BREADTH10)]:
        texts[name], _ = mod.first_n_texts(path, PAIR_ROWS)
    texts["removed_proportional"], _ = mod.all_jsonl_texts(mod.HELDOUT_PROP)
    child_idx, adult_idx, _sel = mod.load_selection_indices()
    selected, _ = mod.selected_base_texts(child_idx, adult_idx)
    texts["removed_childspeech"] = selected["removed_childspeech"]
    texts["removed_adultprose"] = selected["removed_adultprose"]
    for fam in FAMILIES:
        texts[f"eval_{fam}"], _ = mod.eval_family_texts(fam)

    base_profiles = {name: mod.profile_counter_from_texts(ts) for name, ts in texts.items()}
    scores = read_scores()
    fit_rows_all: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    for abl_name, abl in ABLATIONS.items():
        counters = {name: filter_counter(c, abl) for name, c in base_profiles.items()}
        distances: dict[str, dict[str, float]] = {fam: {} for fam in FAMILIES}
        for fam in FAMILIES:
            ev = counters[f"eval_{fam}"]
            for block in ["admitted_view", "admitted_breadth", "admitted_repeat", "removed_proportional", "removed_childspeech", "removed_adultprose"]:
                distances[fam][block] = js(ev, counters[block])
        cal: list[dict[str, Any]] = []
        for arm in ["V", "B", "R"]:
            for fam in EX_ENTITY:
                y = scores[(ARM_CONTRAST[arm], fam, "common10_80")]
                x = distances[fam]["removed_proportional"] - distances[fam][ARM_BLOCK[arm]]
                row = {"ablation": abl_name, "arm": arm, "family": fam, "x": x, "y": y}
                cal.append(row)
                fit_rows_all.append(row)
        fit = fit_beta(cal)
        fit_rows.append({"ablation": abl_name, "description": abl["description"], **fit})
        beta = fit["beta"]
        fam_pred: dict[str, float] = {}
        for fam in FAMILIES:
            xreg = distances[fam]["removed_childspeech"] - distances[fam]["removed_adultprose"]
            pred = beta * xreg
            fam_pred[fam] = pred
            prediction_rows.append({"ablation": abl_name, "family": fam, "distance_child_minus_adult": xreg, "pred_points": pred, "beta": beta})
        for qty, fams in [("exEntity5", EX_ENTITY), ("cheap6", FAMILIES)]:
            val = statistics.mean([fam_pred[f] for f in fams])
            aggregate_rows.append({
                "ablation": abl_name,
                "quantity": qty,
                "pred_points": val,
                "min_family_pred": min(fam_pred[f] for f in fams),
                "max_family_pred": max(fam_pred[f] for f in fams),
                "positive_family_count": sum(1 for f in fams if fam_pred[f] > 0),
                "negative_family_count": sum(1 for f in fams if fam_pred[f] < 0),
                "over_spread_ref": val / SPREAD[qty],
            })
    write_csv(OUT / "calibration_rows.csv", fit_rows_all)
    write_csv(OUT / "fit_summary_rows.csv", fit_rows)
    write_csv(OUT / "prediction_rows.csv", prediction_rows)
    write_csv(OUT / "aggregate_prediction_rows.csv", aggregate_rows)
    summary = {
        "status": "PROFILE_ABLATION_PREDICTION_COMPLETE",
        "created_utc": now(),
        "purpose": "Robustness of profile-distance prediction to feature-class ablation before register scores.",
        "fit_summary_rows": fit_rows,
        "aggregate_prediction_rows": aggregate_rows,
        "reading": "Predictions that remain positive after removing transcript/markup are evidence that the structural profile account is not only CHILDES transcript notation. If only the full profile is positive, future score support would be less general.",
        "files": {"summary_json": rel(OUT / "profile_ablation_prediction_summary.json"), "summary_md": rel(OUT / "profile_ablation_prediction_summary.md"), "fit_summary_rows": rel(OUT / "fit_summary_rows.csv"), "aggregate_prediction_rows": rel(OUT / "aggregate_prediction_rows.csv"), "prediction_rows": rel(OUT / "prediction_rows.csv")},
        "no_model_loading_training_official_evaluation_gpu_globalpiqa_superglue_aoa_upload_or_leaderboard": True,
        "elapsed_sec": round(time.time() - t0, 3),
    }
    (OUT / "profile_ablation_prediction_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    def fmt(x: Any) -> str:
        if x is None:
            return "NA"
        try:
            return f"{float(x):.4f}"
        except Exception:
            return str(x)
    lines = ["# research profile ablation prediction", "", "CPU-only ablation of the committed profile-distance prediction before MAX-register scores.", "", "## Fits", "", "| ablation | beta | Pearson | Spearman | RMSE | exEntity pred | cheap6 pred |", "|---|---:|---:|---:|---:|---:|---:|"]
    agg_by = {(r["ablation"], r["quantity"]): r for r in aggregate_rows}
    for r in fit_rows:
        lines.append(f"| {r['ablation']} | {fmt(r['beta'])} | {fmt(r['pearson'])} | {fmt(r['spearman'])} | {fmt(r['rmse'])} | {fmt(agg_by[(r['ablation'],'exEntity5')]['pred_points'])} | {fmt(agg_by[(r['ablation'],'cheap6')]['pred_points'])} |")
    lines += ["", "The no_transcript_format row is the key robustness check: if it remains positive, the predicted register effect is not just a CHILDES transcript-marker artifact. The no_transcript_no_punct and function_only rows show how much of the calibration survives under still stricter structural abstractions.", "", f"JSON: `{rel(OUT / 'profile_ablation_prediction_summary.json')}`"]
    (OUT / "profile_ablation_prediction_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "fit_summary_rows": fit_rows, "aggregate_prediction_rows": aggregate_rows, "summary_md": summary["files"]["summary_md"], "elapsed_sec": summary["elapsed_sec"]}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
