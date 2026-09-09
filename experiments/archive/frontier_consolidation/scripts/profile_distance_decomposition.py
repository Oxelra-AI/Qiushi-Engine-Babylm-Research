#!/usr/bin/env python3
"""research: decompose the non-open-lexical profile JS prediction.

This is a CPU-only companion to `distribution_proximity_prediction.py`.
It recomputes the same profile counters before MAX-register scores are read and
breaks JS(eval, removed_childspeech)-JS(eval, removed_adultprose) into feature
classes and top feature contributions.  The goal is to see whether the committed
profile prediction is carried only by transcript markers or by broader
function/shape/punctuation structure.

No model loading, training, official evaluation, GPU work, GlobalPIQA,
SuperGLUE, AoA, upload, or leaderboard action.
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


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
PRED_SCRIPT = WS / "scripts/distribution_proximity_prediction.py"
OUT = WS / "data/distribution_proximity_prediction"
DECOMP_OUT = WS / "data/profile_distance_decomposition"
FAMILIES = ["BLiMP", "Supplement", "EWoK", "COMPS", "Reading", "Entity"]
EX_ENTITY = ["BLiMP", "Supplement", "EWoK", "COMPS", "Reading"]
PAIR_ROWS = 7923


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


def js_contribs(p_counts: dict[str, int], q_counts: dict[str, int]) -> dict[str, float]:
    np = float(sum(p_counts.values()))
    nq = float(sum(q_counts.values()))
    out: dict[str, float] = {}
    for k in set(p_counts) | set(q_counts):
        p = p_counts.get(k, 0) / np if np > 0 else 0.0
        q = q_counts.get(k, 0) / nq if nq > 0 else 0.0
        m = 0.5 * (p + q)
        v = 0.0
        if p > 0 and m > 0:
            v += 0.5 * p * math.log(p / m, 2)
        if q > 0 and m > 0:
            v += 0.5 * q * math.log(q / m, 2)
        out[k] = v
    return out


def feature_group(k: str) -> str:
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


def sign(x: float) -> str:
    return "positive_child_block_farther" if x > 0 else ("negative_adult_block_farther" if x < 0 else "zero")


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


def main() -> None:
    t0 = time.time()
    DECOMP_OUT.mkdir(parents=True, exist_ok=True)
    mod = load_pred_module()

    child_idx, adult_idx, selection_meta = mod.load_selection_indices()
    selected_texts, _ = mod.selected_base_texts(child_idx, adult_idx)
    block_profile = {
        "removed_childspeech": mod.profile_counter_from_texts(selected_texts["removed_childspeech"]),
        "removed_adultprose": mod.profile_counter_from_texts(selected_texts["removed_adultprose"]),
    }
    eval_profile = {}
    for fam in FAMILIES:
        texts, _meta = mod.eval_family_texts(fam)
        eval_profile[fam] = mod.profile_counter_from_texts(texts)

    component_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    for fam in FAMILIES:
        child_contrib = js_contribs(eval_profile[fam], block_profile["removed_childspeech"])
        adult_contrib = js_contribs(eval_profile[fam], block_profile["removed_adultprose"])
        diffs = {k: child_contrib.get(k, 0.0) - adult_contrib.get(k, 0.0) for k in set(child_contrib) | set(adult_contrib)}
        by_group: dict[str, float] = {}
        abs_by_group: dict[str, float] = {}
        for k, v in diffs.items():
            g = feature_group(k)
            by_group[g] = by_group.get(g, 0.0) + v
            abs_by_group[g] = abs_by_group.get(g, 0.0) + abs(v)
        total = sum(diffs.values())
        for g in sorted(by_group):
            component_rows.append({
                "family": fam,
                "feature_group": g,
                "contribution_to_profile_js_child_minus_adult": by_group[g],
                "abs_feature_contribution_mass": abs_by_group[g],
                "fraction_of_signed_total": (by_group[g] / total) if total else None,
                "signed_total": total,
                "sign": sign(by_group[g]),
            })
        for rank, (k, v) in enumerate(sorted(diffs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:40], start=1):
            top_rows.append({
                "family": fam,
                "rank_abs": rank,
                "feature": k,
                "feature_group": feature_group(k),
                "contribution_to_profile_js_child_minus_adult": v,
                "sign": sign(v),
                "child_js_contribution": child_contrib.get(k, 0.0),
                "adult_js_contribution": adult_contrib.get(k, 0.0),
            })

    # Relate feature-group family vectors to the committed profile prediction.
    prediction_rows = []
    pred_path = OUT / "prediction_rows.csv"
    with pred_path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("metric") == "profile_js" and r.get("fit_spec") == "common10_80_all3_exEntity" and r.get("family") in FAMILIES:
                prediction_rows.append({"family": r["family"], "pred": float(r["pred_childspeech_removed_minus_adultprose_removed_points"])})
    pred_by = {r["family"]: r["pred"] for r in prediction_rows}
    group_summary_rows: list[dict[str, Any]] = []
    for g in sorted({r["feature_group"] for r in component_rows}):
        fams = [f for f in EX_ENTITY if f in pred_by]
        xs = [next(r for r in component_rows if r["family"] == f and r["feature_group"] == g)["contribution_to_profile_js_child_minus_adult"] for f in fams]
        ys = [pred_by[f] for f in fams]
        group_summary_rows.append({
            "feature_group": g,
            "exEntity_mean_contribution": statistics.mean(xs),
            "exEntity_min_contribution": min(xs),
            "exEntity_max_contribution": max(xs),
            "exEntity_positive_family_count": sum(1 for x in xs if x > 0),
            "exEntity_negative_family_count": sum(1 for x in xs if x < 0),
            "pearson_with_committed_profile_prediction": pearson(xs, ys),
            "spearman_with_committed_profile_prediction": spearman(xs, ys),
            "families": ";".join(fams),
        })

    write_csv(DECOMP_OUT / "profile_component_rows.csv", component_rows)
    write_csv(DECOMP_OUT / "profile_top_feature_contributions.csv", top_rows)
    write_csv(DECOMP_OUT / "profile_group_summary_rows.csv", group_summary_rows)

    # Compact interpretation.
    total_ex = [sum(r["contribution_to_profile_js_child_minus_adult"] for r in component_rows if r["family"] == f) for f in EX_ENTITY]
    transcript_group = next((r for r in group_summary_rows if r["feature_group"] == "transcript_or_markup_format"), None)
    sent_group = next((r for r in group_summary_rows if r["feature_group"] == "sentence_length"), None)
    func_group = next((r for r in group_summary_rows if r["feature_group"] == "function_word_identity"), None)
    punct_group = next((r for r in group_summary_rows if r["feature_group"] == "punctuation"), None)
    summary = {
        "status": "PROFILE_DISTANCE_DECOMPOSITION_COMPLETE",
        "created_utc": now(),
        "purpose": "Decompose profile JS child-minus-adult distance before register scores arrive.",
        "selection_meta": selection_meta,
        "exEntity_total_profile_child_minus_adult_mean": statistics.mean(total_ex),
        "group_summary_rows": group_summary_rows,
        "interpretation": {
            "transcript_marker_component": transcript_group,
            "sentence_length_component": sent_group,
            "function_word_component": func_group,
            "punctuation_component": punct_group,
            "reading": "Positive signed components make child/speech removed blocks farther from evaluation families than adult-prose removed blocks under that profile slice. Large transcript/format contributions would mean the prediction partly tests register-format mismatch rather than deep semantic proximity.",
        },
        "files": {
            "component_rows": rel(DECOMP_OUT / "profile_component_rows.csv"),
            "top_feature_contributions": rel(DECOMP_OUT / "profile_top_feature_contributions.csv"),
            "group_summary_rows": rel(DECOMP_OUT / "profile_group_summary_rows.csv"),
            "summary_json": rel(DECOMP_OUT / "profile_distance_decomposition_summary.json"),
            "summary_md": rel(DECOMP_OUT / "profile_distance_decomposition_summary.md"),
        },
        "no_model_loading_training_official_evaluation_gpu_globalpiqa_superglue_aoa_upload_or_leaderboard": True,
        "elapsed_sec": round(time.time() - t0, 3),
    }
    (DECOMP_OUT / "profile_distance_decomposition_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def fmt(x: Any) -> str:
        if x is None:
            return "NA"
        try:
            return f"{float(x):.5f}"
        except Exception:
            return str(x)
    lines = [
        "# research profile-distance decomposition",
        "",
        "This CPU-only decomposition was produced before reading any MAX-register score. It decomposes `profile_js(eval, removed_childspeech) - profile_js(eval, removed_adultprose)` into feature classes.",
        "",
        "## Ex-Entity feature-group summary",
        "",
        "| feature group | mean contribution | min | max | positive families | negative families | Spearman vs committed prediction |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(group_summary_rows, key=lambda r: abs(r["exEntity_mean_contribution"]), reverse=True):
        lines.append(f"| {r['feature_group']} | {fmt(r['exEntity_mean_contribution'])} | {fmt(r['exEntity_min_contribution'])} | {fmt(r['exEntity_max_contribution'])} | {r['exEntity_positive_family_count']} | {r['exEntity_negative_family_count']} | {fmt(r['spearman_with_committed_profile_prediction'])} |")
    lines += [
        "",
        "## Scientific reading",
        "",
        "The committed profile prediction is not a pure lexical-frequency account, but its scientific interpretation depends on which profile classes carry the distance. If transcript/markup features dominate, the register experiment is partly a format mismatch test. If function words, punctuation, sentence length and abstract open-class shape all contribute with the same sign, the result is closer to a structural-register proximity account.",
        "",
        f"JSON: `{rel(DECOMP_OUT / 'profile_distance_decomposition_summary.json')}`",
        f"Top feature table: `{rel(DECOMP_OUT / 'profile_top_feature_contributions.csv')}`",
    ]
    (DECOMP_OUT / "profile_distance_decomposition_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "exEntity_total_profile_child_minus_adult_mean": summary["exEntity_total_profile_child_minus_adult_mean"],
        "group_summary_rows": group_summary_rows,
        "summary_md": summary["files"]["summary_md"],
        "elapsed_sec": summary["elapsed_sec"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
