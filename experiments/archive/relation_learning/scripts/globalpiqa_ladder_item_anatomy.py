#!/usr/bin/env python3
"""research GlobalPIQA ladder item anatomy.

This CPU-only analyzer extends the research candidate GlobalPIQA item table to the
matched ordinary-continuation and dense-mask/sparse-label ladder.  It reads the
actual prediction JSON files, compares exact prediction strings against the
BabyLM GlobalPIQA gold strings, and records which items move relative to the
faithful coherent86 parent.

The goal is not to amplify GlobalPIQA into broad evidence.  The goal is to check
whether the few-item contribution in the v4->v5 margin is also carried by the
matched ordinary continuation rung and whether item identities are deterministic
across seeds/policies.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import json
import pathlib
from collections import defaultdict
from typing import Any

ROOT = _public_path('.')
OUT = _public_path('experiments/archive/relation_learning/data/globalpiqa_ladder_item_anatomy')
EVAL_DATA = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval')
GOLD_PATHS = {
    "parallel": _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/global_piqa_parallel/eng_latn.jsonl'),
    "nonparallel": _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/global_piqa_nonparallel/eng_latn.jsonl'),
}

PRED_PATHS: dict[str, dict[str, pathlib.Path]] = {
    "coherent86": {
        "parallel": _public_path('experiments/archive/frontier_consolidation/data/private_scale_sentinel_eval/coherent86_private_scale_0p75/official_outputs/coherent86_private_scale_0p75/GlobalPIQA_parallel/final/full_coherent86_private_scale_0p75_GlobalPIQA_parallel/zero_shot/mlm/global_piqa_parallel/global_piqa_parallel/predictions.json'),
        "nonparallel": _public_path('experiments/archive/frontier_consolidation/data/private_scale_sentinel_eval/coherent86_private_scale_0p75/official_outputs/coherent86_private_scale_0p75/GlobalPIQA_nonparallel/final/full_coherent86_private_scale_0p75_GlobalPIQA_nonparallel/zero_shot/mlm/global_piqa_nonparallel/global_piqa_nonparallel/predictions.json'),
    },
    "ordinary64": {
        "parallel": _public_path('experiments/archive/functional_learning/data/repaired_ordinary_zero_reading/official_outputs/ordinary_inherited_wwm_seed62064_u0080_zero_reading/GlobalPIQA_parallel/repaired_ordinary_inherited_wwm_seed62064_u0080/full_ordinary_inherited_wwm_seed62064_u0080_zero_reading_GlobalPIQA_parallel/zero_shot/mlm/global_piqa_parallel/global_piqa_parallel/predictions.json'),
        "nonparallel": _public_path('experiments/archive/functional_learning/data/repaired_ordinary_zero_reading/official_outputs/ordinary_inherited_wwm_seed62064_u0080_zero_reading/GlobalPIQA_nonparallel/repaired_ordinary_inherited_wwm_seed62064_u0080/full_ordinary_inherited_wwm_seed62064_u0080_zero_reading_GlobalPIQA_nonparallel/zero_shot/mlm/global_piqa_nonparallel/global_piqa_nonparallel/predictions.json'),
    },
    "ordinary65": {
        "parallel": _public_path('experiments/archive/relation_learning/data/o62065_GlobalPIQA_parallel/o62065/with_special/GlobalPIQA_parallel/predictions.json'),
        "nonparallel": _public_path('experiments/archive/relation_learning/data/o62065_GlobalPIQA_nonparallel/o62065/with_special/GlobalPIQA_nonparallel/predictions.json'),
    },
    "ms64": {
        "parallel": _public_path('experiments/archive/functional_learning/data/repaired_densemask_sparselabel_zero_reading/official_outputs/densemask_sparselabel_seed62064_u0080_zero_reading/GlobalPIQA_parallel/repaired_densemask_sparselabel_seed62064_u0080/full_densemask_sparselabel_seed62064_u0080_zero_reading_GlobalPIQA_parallel/zero_shot/mlm/global_piqa_parallel/global_piqa_parallel/predictions.json'),
        "nonparallel": _public_path('experiments/archive/functional_learning/data/repaired_densemask_sparselabel_zero_reading/official_outputs/densemask_sparselabel_seed62064_u0080_zero_reading/GlobalPIQA_nonparallel/repaired_densemask_sparselabel_seed62064_u0080/full_densemask_sparselabel_seed62064_u0080_zero_reading_GlobalPIQA_nonparallel/zero_shot/mlm/global_piqa_nonparallel/global_piqa_nonparallel/predictions.json'),
    },
    "ms65": {
        "parallel": _public_path('experiments/archive/functional_learning/data/parallel_zero_reading/ms62065_GlobalPIQA_parallel/official_outputs/ms_acquisition_seed62065_GlobalPIQA_parallel/GlobalPIQA_parallel/repaired_ms62065_u0080/full_ms_acquisition_seed62065_GlobalPIQA_parallel_GlobalPIQA_parallel/zero_shot/mlm/global_piqa_parallel/global_piqa_parallel/predictions.json'),
        "nonparallel": _public_path('experiments/archive/functional_learning/data/parallel_zero_reading/ms62065_GlobalPIQA_nonparallel/official_outputs/ms_acquisition_seed62065_GlobalPIQA_nonparallel/GlobalPIQA_nonparallel/repaired_ms62065_u0080/full_ms_acquisition_seed62065_GlobalPIQA_nonparallel_GlobalPIQA_nonparallel/zero_shot/mlm/global_piqa_nonparallel/global_piqa_nonparallel/predictions.json'),
    },
    "clean64": {
        "parallel": _public_path('experiments/archive/functional_learning/data/repaired_clean_eval_zero_reading/official_outputs/clean_pres_lambda1_eval_seed62064_u0080_zero_reading/GlobalPIQA_parallel/repaired_clean_pres_lambda1_eval_seed62064_u0080/full_clean_pres_lambda1_eval_seed62064_u0080_zero_reading_GlobalPIQA_parallel/zero_shot/mlm/global_piqa_parallel/global_piqa_parallel/predictions.json'),
        "nonparallel": _public_path('experiments/archive/functional_learning/data/repaired_clean_eval_zero_reading/official_outputs/clean_pres_lambda1_eval_seed62064_u0080_zero_reading/GlobalPIQA_nonparallel/repaired_clean_pres_lambda1_eval_seed62064_u0080/full_clean_pres_lambda1_eval_seed62064_u0080_zero_reading_GlobalPIQA_nonparallel/zero_shot/mlm/global_piqa_nonparallel/global_piqa_nonparallel/predictions.json'),
    },
    "clean65": {
        "parallel": _public_path('experiments/archive/functional_learning/data/repaired_clean_seed62065_zero_reading/official_outputs/clean_pres_lambda1_eval_seed62065_u0080_zero_reading/GlobalPIQA_parallel/repaired_clean_pres_lambda1_eval_seed62065_u0080/full_clean_pres_lambda1_eval_seed62065_u0080_zero_reading_GlobalPIQA_parallel/zero_shot/mlm/global_piqa_parallel/global_piqa_parallel/predictions.json'),
        "nonparallel": _public_path('experiments/archive/functional_learning/data/repaired_clean_seed62065_zero_reading/official_outputs/clean_pres_lambda1_eval_seed62065_u0080_zero_reading_GlobalPIQA_nonparallel/repaired_clean_pres_lambda1_eval_seed62065_u0080/full_clean_pres_lambda1_eval_seed62065_u0080_zero_reading_GlobalPIQA_nonparallel/zero_shot/mlm/global_piqa_nonparallel/global_piqa_nonparallel/predictions.json'),
    },
}
# Fix clean65 nonparallel historical path if the common typo path above does not exist.
PRED_PATHS["clean65"]["nonparallel"] = _public_path('experiments/archive/functional_learning/data/repaired_clean_seed62065_zero_reading/official_outputs/clean_pres_lambda1_eval_seed62065_u0080_zero_reading/GlobalPIQA_nonparallel/repaired_clean_pres_lambda1_eval_seed62065_u0080/full_clean_pres_lambda1_eval_seed62065_u0080_zero_reading_GlobalPIQA_nonparallel/zero_shot/mlm/global_piqa_nonparallel/global_piqa_nonparallel/predictions.json')


def rel(path: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def gold_id(row: dict[str, Any], i: int) -> str:
    for key in ("example_id", "id", "idx", "uid"):
        if row.get(key) is not None:
            return str(row[key])
    return f"ex{i:06d}"


def gold_solution(row: dict[str, Any]) -> tuple[str, Any]:
    label = row.get("label", row.get("answer", row.get("gold", row.get("target", row.get("answer_idx", row.get("answer_label"))))))
    for key in ("answer_text", "gold_text", "target_text", "correct", "solution"):
        if row.get(key) is not None:
            return str(row[key]), label
    if label is not None:
        try:
            return str(row[f"solution{int(str(label).strip())}"]), label
        except Exception:
            pass
    raise RuntimeError(f"cannot recover GlobalPIQA gold from keys={sorted(row)} label={label}")


def load_gold() -> dict[tuple[str, str], dict[str, Any]]:
    out = {}
    for split, path in GOLD_PATHS.items():
        for i, row in enumerate(read_jsonl(path)):
            eid = gold_id(row, i)
            gold, label = gold_solution(row)
            out[(split, eid)] = {
                "split": split,
                "example_id": eid,
                "row_index": i,
                "gold": gold,
                "label": label,
                "prompt": str(row.get("goal", row.get("question", row.get("prompt", row.get("context", ""))))),
                "solution0": row.get("solution0"),
                "solution1": row.get("solution1"),
            }
    return out


def normalize_pred(v: Any) -> str:
    if isinstance(v, dict):
        if isinstance(v.get("predictions"), list) and v["predictions"]:
            first = v["predictions"][0]
            if isinstance(first, dict):
                return str(first.get("pred", first.get("prediction", first.get("text", ""))))
            return str(first)
        for key in ("pred", "prediction", "text", "answer"):
            if key in v:
                return str(v[key])
    return str(v)


def load_preds(path: pathlib.Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and isinstance(obj.get("predictions"), dict):
        obj = obj["predictions"]
    if not isinstance(obj, dict):
        raise RuntimeError(f"unexpected predictions object at {path}")
    return {str(k): normalize_pred(v) for k, v in obj.items()}


def exact_correct(pred: str, gold: str) -> bool:
    return pred.strip() == gold.strip()


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    preferred = [
        "endpoint", "split", "example_id", "row_index", "correct", "delta_correct_vs_coherent86",
        "coherent86_correct", "endpoint_correct", "gold", "prediction", "coherent86_prediction",
        "prompt", "solution0", "solution1",
    ]
    keys = []
    for k in preferred + sorted(set().union(*(r.keys() for r in rows))):
        if k in rows[0] or any(k in r for r in rows):
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def endpoint_group(endpoint: str) -> str:
    if endpoint == "coherent86": return "parent"
    if endpoint.startswith("ordinary"): return "ordinary"
    if endpoint.startswith("ms"): return "ms"
    if endpoint.startswith("clean"): return "clean"
    return endpoint


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gold = load_gold()
    item_rows: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    preds_by_ep: dict[str, dict[tuple[str, str], str]] = defaultdict(dict)
    for ep, by_split in PRED_PATHS.items():
        for split, path in by_split.items():
            if not path.exists():
                missing.append({"endpoint": ep, "split": split, "path": rel(path)})
                continue
            preds = load_preds(path)
            for key, grow in gold.items():
                sp, eid = key
                if sp != split:
                    continue
                pred = preds.get(eid)
                if pred is None:
                    hits = [v for k, v in preds.items() if k == eid or k.startswith(eid) or eid in k]
                    pred = hits[0] if hits else ""
                preds_by_ep[ep][key] = pred
                item_rows.append({
                    "endpoint": ep,
                    "group": endpoint_group(ep),
                    **grow,
                    "prediction": pred,
                    "correct": int(exact_correct(pred, grow["gold"])),
                    "prediction_path": rel(path),
                })
    by_key_ep: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in item_rows:
        by_key_ep[(r["split"], r["example_id"])][r["endpoint"]] = r

    counts: list[dict[str, Any]] = []
    for ep in PRED_PATHS:
        for split in ("parallel", "nonparallel"):
            vals = [r for r in item_rows if r["endpoint"] == ep and r["split"] == split]
            if not vals:
                continue
            corr = sum(int(r["correct"]) for r in vals)
            coh = [r for r in item_rows if r["endpoint"] == "coherent86" and r["split"] == split]
            coh_corr = sum(int(r["correct"]) for r in coh) if coh else None
            counts.append({
                "endpoint": ep,
                "group": endpoint_group(ep),
                "split": split,
                "correct": corr,
                "total": len(vals),
                "accuracy": 100.0 * corr / len(vals),
                "delta_correct_vs_coherent86": None if coh_corr is None else corr - coh_corr,
            })

    overall: list[dict[str, Any]] = []
    by_ep = defaultdict(dict)
    for r in counts:
        by_ep[r["endpoint"]][r["split"]] = r
    for ep, d in by_ep.items():
        if "parallel" in d and "nonparallel" in d:
            par, non = d["parallel"], d["nonparallel"]
            overall.append({
                "endpoint": ep,
                "group": endpoint_group(ep),
                "parallel_correct": par["correct"],
                "parallel_total": par["total"],
                "parallel_delta": par["delta_correct_vs_coherent86"],
                "nonparallel_correct": non["correct"],
                "nonparallel_total": non["total"],
                "nonparallel_delta": non["delta_correct_vs_coherent86"],
                "globalpiqa_mean": (par["accuracy"] + non["accuracy"]) / 2.0,
                "globalpiqa_delta_vs_coherent86": ((par["accuracy"] + non["accuracy"]) / 2.0) - next((x["globalpiqa_mean"] for x in overall if x["endpoint"] == "coherent86"), 0.0) if ep != "coherent86" and any(x["endpoint"] == "coherent86" for x in overall) else 0.0,
            })
    coh_mean = next((r["globalpiqa_mean"] for r in overall if r["endpoint"] == "coherent86"), None)
    if coh_mean is not None:
        for r in overall:
            r["globalpiqa_delta_vs_coherent86"] = float(r["globalpiqa_mean"] - coh_mean)
            r["overall_delta_contribution"] = r["globalpiqa_delta_vs_coherent86"] / 9.0

    flips: list[dict[str, Any]] = []
    for key, eps in sorted(by_key_ep.items()):
        if "coherent86" not in eps:
            continue
        coh = eps["coherent86"]
        for ep, r in sorted(eps.items()):
            if ep == "coherent86":
                continue
            delta = int(r["correct"]) - int(coh["correct"])
            if delta != 0:
                flips.append({
                    "endpoint": ep,
                    "group": endpoint_group(ep),
                    "split": key[0],
                    "example_id": key[1],
                    "row_index": r["row_index"],
                    "coherent86_correct": int(coh["correct"]),
                    "endpoint_correct": int(r["correct"]),
                    "delta_correct_vs_coherent86": delta,
                    "gold": r["gold"],
                    "coherent86_prediction": coh["prediction"],
                    "prediction": r["prediction"],
                    "prompt": r["prompt"],
                    "solution0": r["solution0"],
                    "solution1": r["solution1"],
                })

    pattern_rows: list[dict[str, Any]] = []
    endpoints = [ep for ep in PRED_PATHS if ep != "coherent86" and any(r["endpoint"] == ep for r in item_rows)]
    for key, eps in sorted(by_key_ep.items()):
        if "coherent86" not in eps:
            continue
        deltas = {ep: int(eps[ep]["correct"]) - int(eps["coherent86"]["correct"]) for ep in endpoints if ep in eps}
        if not any(v != 0 for v in deltas.values()):
            continue
        grow = eps["coherent86"]
        rec: dict[str, Any] = {
            "split": key[0], "example_id": key[1], "row_index": grow["row_index"],
            "coherent86_correct": int(grow["correct"]), "gold": grow["gold"],
            "prompt": grow["prompt"], "solution0": grow["solution0"], "solution1": grow["solution1"],
            "coherent86_prediction": grow["prediction"],
        }
        for ep in endpoints:
            if ep in eps:
                rec[f"delta_{ep}"] = deltas[ep]
                rec[f"pred_{ep}"] = eps[ep]["prediction"]
            else:
                rec[f"delta_{ep}"] = None
        pattern_rows.append(rec)

    # Compact deterministic pattern tests.
    ordinary_eps = [ep for ep in ("ordinary64", "ordinary65") if ep in endpoints]
    recipe_eps = [ep for ep in ("ms64", "ms65", "clean64", "clean65") if ep in endpoints]
    ordinary_same_items = []
    recipe_same_items = []
    for r in pattern_rows:
        ord_vals = [r.get(f"delta_{ep}") for ep in ordinary_eps if r.get(f"delta_{ep}") not in (None, 0)]
        if len(ord_vals) == len(ordinary_eps) and len(set(ord_vals)) == 1:
            ordinary_same_items.append({k: r.get(k) for k in ["split", "example_id", "row_index", "coherent86_correct", "gold", "prompt", "solution0", "solution1"] + [f"delta_{ep}" for ep in ordinary_eps] + [f"pred_{ep}" for ep in ordinary_eps]})
        rec_vals = [r.get(f"delta_{ep}") for ep in recipe_eps if r.get(f"delta_{ep}") not in (None, 0)]
        if len(rec_vals) == len(recipe_eps) and len(set(rec_vals)) == 1:
            recipe_same_items.append({k: r.get(k) for k in ["split", "example_id", "row_index", "coherent86_correct", "gold", "prompt", "solution0", "solution1"] + [f"delta_{ep}" for ep in recipe_eps] + [f"pred_{ep}" for ep in recipe_eps]})

    write_csv(_public_path('experiments/archive/relation_learning/data/globalpiqa_ladder_item_anatomy/item_correctness.csv'), item_rows)
    write_csv(_public_path('experiments/archive/relation_learning/data/globalpiqa_ladder_item_anatomy/flips_vs_coherent86.csv'), flips)
    write_csv(_public_path('experiments/archive/relation_learning/data/globalpiqa_ladder_item_anatomy/pattern_rows_vs_coherent86.csv'), pattern_rows)
    write_csv(_public_path('experiments/archive/relation_learning/data/globalpiqa_ladder_item_anatomy/ordinary_seed_shared_changed_items.csv'), ordinary_same_items)
    write_csv(_public_path('experiments/archive/relation_learning/data/globalpiqa_ladder_item_anatomy/ms_clean_seed_shared_changed_items.csv'), recipe_same_items)
    write_csv(_public_path('experiments/archive/relation_learning/data/globalpiqa_ladder_item_anatomy/counts.csv'), counts)
    write_csv(_public_path('experiments/archive/relation_learning/data/globalpiqa_ladder_item_anatomy/overall.csv'), overall)

    summary = {
        "status": "GLOBALPIQA_LADDER_ITEM_ANATOMY",
        "missing_prediction_files": missing,
        "counts": counts,
        "overall": overall,
        "n_flips_vs_coherent86_rows": len(flips),
        "n_pattern_rows": len(pattern_rows),
        "n_ordinary_seed_shared_changed_items": len(ordinary_same_items),
        "n_ms_clean_seed_shared_changed_items": len(recipe_same_items),
        "ordinary_seed_shared_changed_items": ordinary_same_items,
        "ms_clean_seed_shared_changed_items": recipe_same_items,
        "outputs": {
            "item_correctness": rel(_public_path('experiments/archive/relation_learning/data/globalpiqa_ladder_item_anatomy/item_correctness.csv')),
            "flips": rel(_public_path('experiments/archive/relation_learning/data/globalpiqa_ladder_item_anatomy/flips_vs_coherent86.csv')),
            "patterns": rel(_public_path('experiments/archive/relation_learning/data/globalpiqa_ladder_item_anatomy/pattern_rows_vs_coherent86.csv')),
            "ordinary_seed_shared": rel(_public_path('experiments/archive/relation_learning/data/globalpiqa_ladder_item_anatomy/ordinary_seed_shared_changed_items.csv')),
            "ms_clean_seed_shared": rel(_public_path('experiments/archive/relation_learning/data/globalpiqa_ladder_item_anatomy/ms_clean_seed_shared_changed_items.csv')),
            "counts": rel(_public_path('experiments/archive/relation_learning/data/globalpiqa_ladder_item_anatomy/counts.csv')),
            "overall": rel(_public_path('experiments/archive/relation_learning/data/globalpiqa_ladder_item_anatomy/overall.csv')),
        },
    }
    (_public_path('experiments/archive/relation_learning/data/globalpiqa_ladder_item_anatomy/summary.json')).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research GlobalPIQA ladder item anatomy", "",
        "This table checks exact GlobalPIQA item identities for coherent86, matched ordinary continuation, exact dense-mask/sparse-label acquisition, and clean preservation.", "",
        "## Overall split counts", "",
        "| endpoint | group | parallel | nonparallel | mean | delta vs coherent86 | contribution to Overall |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in overall:
        lines.append(f"| {r['endpoint']} | {r['group']} | {r['parallel_correct']}/{r['parallel_total']} (Δ{r['parallel_delta']}) | {r['nonparallel_correct']}/{r['nonparallel_total']} (Δ{r['nonparallel_delta']}) | {r['globalpiqa_mean']:.6f} | {r['globalpiqa_delta_vs_coherent86']:.6f} | {r.get('overall_delta_contribution', 0.0):.6f} |")
    lines += ["", "## Deterministic item sets", "", f"Ordinary seed-shared changed items: {len(ordinary_same_items)}", f"MS/clean seed-shared changed items: {len(recipe_same_items)}", ""]
    if ordinary_same_items:
        lines += ["### Ordinary shared items", ""]
        for r in ordinary_same_items:
            lines.append(f"- {r['split']} {r['example_id']}: delta {r.get('delta_ordinary64')} / {r.get('delta_ordinary65')}; gold={r['gold']!r}; prompt={r['prompt']!r}")
    if recipe_same_items:
        lines += ["", "### MS/clean shared items", ""]
        for r in recipe_same_items:
            vals = [r.get(f"delta_{ep}") for ep in recipe_eps]
            lines.append(f"- {r['split']} {r['example_id']}: deltas={vals}; gold={r['gold']!r}; prompt={r['prompt']!r}")
    if missing:
        lines += ["", "## Missing prediction files", "", json.dumps(missing, indent=2)]
    lines += ["", "Outputs:", json.dumps(summary["outputs"], indent=2)]
    (_public_path('research/documents/relation_learning/data/globalpiqa_ladder_item_anatomy/summary.md')).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "out": rel(OUT), "overall": overall, "missing": missing}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
