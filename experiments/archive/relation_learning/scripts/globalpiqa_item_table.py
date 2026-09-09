#!/usr/bin/env python3
"""research: item-level GlobalPIQA support for dense and preservation endpoints.

Reads prediction paths from known official-eval trees, compares exact stripped text
against the official GlobalPIQA gold files, and records which items flip relative
to coherent86.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import pandas as pd

ROOT = _public_path('.')
OUT = _public_path('experiments/archive/relation_learning/data/globalpiqa_item_table')
EVAL_DATA = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval')
GOLD_PATHS = {
    "parallel": _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/global_piqa_parallel/eng_latn.jsonl'),
    "nonparallel": _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/global_piqa_nonparallel/eng_latn.jsonl'),
}

PRED_PATHS = {
    "coherent86": {
        "parallel": _public_path('experiments/archive/frontier_consolidation/data/private_scale_sentinel_eval/coherent86_private_scale_0p75/official_outputs/coherent86_private_scale_0p75/GlobalPIQA_parallel/final/full_coherent86_private_scale_0p75_GlobalPIQA_parallel/zero_shot/mlm/global_piqa_parallel/global_piqa_parallel/predictions.json'),
        "nonparallel": _public_path('experiments/archive/frontier_consolidation/data/private_scale_sentinel_eval/coherent86_private_scale_0p75/official_outputs/coherent86_private_scale_0p75/GlobalPIQA_nonparallel/final/full_coherent86_private_scale_0p75_GlobalPIQA_nonparallel/zero_shot/mlm/global_piqa_nonparallel/global_piqa_nonparallel/predictions.json'),
    },
    "dense_seed62064": {
        "parallel": _public_path('experiments/archive/functional_learning/data/dense_focus_official_eval_seed62064/official_outputs/dense_focus_seed62064_u0080/GlobalPIQA_parallel/update_0080/full_dense_focus_seed62064_u0080_GlobalPIQA_parallel/zero_shot/mlm/global_piqa_parallel/global_piqa_parallel/predictions.json'),
        "nonparallel": _public_path('experiments/archive/functional_learning/data/dense_focus_official_eval_seed62064/official_outputs/dense_focus_seed62064_u0080/GlobalPIQA_nonparallel/update_0080/full_dense_focus_seed62064_u0080_GlobalPIQA_nonparallel/zero_shot/mlm/global_piqa_nonparallel/global_piqa_nonparallel/predictions.json'),
    },
    "dense_seed62065": {
        "parallel": _public_path('experiments/archive/functional_learning/data/dense_focus_official_eval_seed62065/official_outputs/dense_focus_seed62065_u0080/GlobalPIQA_parallel/update_0080/full_dense_focus_seed62065_u0080_GlobalPIQA_parallel/zero_shot/mlm/global_piqa_parallel/global_piqa_parallel/predictions.json'),
        "nonparallel": _public_path('experiments/archive/functional_learning/data/dense_focus_official_eval_seed62065/official_outputs/dense_focus_seed62065_u0080/GlobalPIQA_nonparallel/update_0080/full_dense_focus_seed62065_u0080_GlobalPIQA_nonparallel/zero_shot/mlm/global_piqa_nonparallel/global_piqa_nonparallel/predictions.json'),
    },
    "clean_pres_seed62064": {
        "parallel": _public_path('experiments/archive/functional_learning/data/repaired_clean_eval_zero_reading/official_outputs/clean_pres_lambda1_eval_seed62064_u0080_zero_reading/GlobalPIQA_parallel/repaired_clean_pres_lambda1_eval_seed62064_u0080/full_clean_pres_lambda1_eval_seed62064_u0080_zero_reading_GlobalPIQA_parallel/zero_shot/mlm/global_piqa_parallel/global_piqa_parallel/predictions.json'),
        "nonparallel": _public_path('experiments/archive/functional_learning/data/repaired_clean_eval_zero_reading/official_outputs/clean_pres_lambda1_eval_seed62064_u0080_zero_reading/GlobalPIQA_nonparallel/repaired_clean_pres_lambda1_eval_seed62064_u0080/full_clean_pres_lambda1_eval_seed62064_u0080_zero_reading_GlobalPIQA_nonparallel/zero_shot/mlm/global_piqa_nonparallel/global_piqa_nonparallel/predictions.json'),
    },
    "clean_pres_seed62065": {
        "parallel": _public_path('experiments/archive/functional_learning/data/repaired_clean_seed62065_zero_reading/official_outputs/clean_pres_lambda1_eval_seed62065_u0080_zero_reading/GlobalPIQA_parallel/repaired_clean_pres_lambda1_eval_seed62065_u0080/full_clean_pres_lambda1_eval_seed62065_u0080_zero_reading_GlobalPIQA_parallel/zero_shot/mlm/global_piqa_parallel/global_piqa_parallel/predictions.json'),
        "nonparallel": _public_path('experiments/archive/functional_learning/data/repaired_clean_seed62065_zero_reading/official_outputs/clean_pres_lambda1_eval_seed62065_u0080_zero_reading/GlobalPIQA_nonparallel/repaired_clean_pres_lambda1_eval_seed62065_u0080/full_clean_pres_lambda1_eval_seed62065_u0080_zero_reading_GlobalPIQA_nonparallel/zero_shot/mlm/global_piqa_nonparallel/global_piqa_nonparallel/predictions.json'),
    },
}


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_gold_id(row: dict, fallback_i: int) -> str:
    for k in ["example_id", "id", "idx", "uid"]:
        if k in row:
            return str(row[k])
    # Match BabyLM GlobalPIQA id conventions if ids are absent.
    return f"ex{fallback_i:06d}"


def label_to_solution(row: dict) -> Tuple[str, str, Any]:
    label = row.get("label", row.get("answer", row.get("gold", row.get("target"))))
    if label is None:
        # Some PIQA formats use answer_idx.
        label = row.get("answer_idx", row.get("answer_label"))
    candidates = []
    if label is not None:
        label_s = str(label).strip()
        candidates += [f"solution{label_s}", f"sol{label_s}", f"ending{label_s}", f"option{label_s}"]
        try:
            li = int(label_s)
            candidates += [f"solution{li}", f"sol{li}", f"ending{li}", f"option{li}", f"solution{li+1}", f"sol{li+1}"]
        except Exception:
            pass
    # If there is an explicit answer string, prefer it.
    for k in ["answer_text", "gold_text", "target_text", "correct", "solution"]:
        if k in row and row[k] is not None:
            return str(row[k]), k, label
    for k in candidates:
        if k in row and row[k] is not None:
            return str(row[k]), k, label
    # Fallback for two-solution PIQA rows with label 0/1.
    if "solution0" in row and "solution1" in row and label is not None:
        try:
            return str(row[f"solution{int(label)}"]), f"solution{int(label)}", label
        except Exception:
            pass
    raise RuntimeError(f"Could not identify gold solution in row keys={list(row.keys())}, label={label}")


def load_gold(split: str) -> Dict[str, dict]:
    rows = load_jsonl(GOLD_PATHS[split])
    out = {}
    for i, row in enumerate(rows):
        eid = get_gold_id(row, i)
        gold, key, label = label_to_solution(row)
        out[eid] = {"id": eid, "gold": gold, "gold_key": key, "label": label, "row": row, "row_index": i}
    return out


def normalize_prediction_object(pred_obj: Any) -> str:
    if isinstance(pred_obj, dict):
        if "predictions" in pred_obj and pred_obj["predictions"]:
            first = pred_obj["predictions"][0]
            if isinstance(first, dict):
                return str(first.get("pred", first.get("prediction", first.get("text", ""))))
            return str(first)
        for k in ["pred", "prediction", "text", "answer"]:
            if k in pred_obj:
                return str(pred_obj[k])
    return str(pred_obj)


def load_preds(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "predictions" in obj and isinstance(obj["predictions"], dict):
        obj = obj["predictions"]
    if not isinstance(obj, dict):
        raise RuntimeError(f"Prediction file {path} is not a dict")
    return {str(k): normalize_prediction_object(v) for k, v in obj.items()}


def exact_correct(pred: str, gold: str) -> bool:
    return pred.strip() == gold.strip()


def prompt_text(row: dict) -> str:
    for k in ["goal", "question", "prompt", "context", "activity", "obs1"]:
        if k in row and row[k] is not None:
            return str(row[k])
    return ""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gold = {split: load_gold(split) for split in GOLD_PATHS}
    item_rows = []
    missing = []
    for endpoint, by_split in PRED_PATHS.items():
        for split, path in by_split.items():
            if not path.exists():
                missing.append({"endpoint": endpoint, "split": split, "path": str(path.relative_to(ROOT))})
                continue
            preds = load_preds(path)
            for eid, grow in gold[split].items():
                pred = preds.get(eid)
                if pred is None:
                    # Nonparallel ids can be verbose but exact keys are expected. Try a suffix/prefix fallback.
                    hits = [v for k, v in preds.items() if k == eid or k.startswith(eid) or eid in k]
                    pred = hits[0] if hits else ""
                item_rows.append({
                    "endpoint": endpoint,
                    "split": split,
                    "example_id": eid,
                    "row_index": grow["row_index"],
                    "label": grow["label"],
                    "gold_key": grow["gold_key"],
                    "gold": grow["gold"],
                    "prediction": pred,
                    "correct": exact_correct(pred, grow["gold"]),
                    "prompt": prompt_text(grow["row"]),
                    "solution0": grow["row"].get("solution0"),
                    "solution1": grow["row"].get("solution1"),
                    "prediction_path": str(path.relative_to(ROOT)),
                })
    df = pd.DataFrame(item_rows)
    if df.empty:
        raise RuntimeError("No item rows were assembled")
    counts = df.groupby(["endpoint", "split"], as_index=False)["correct"].agg(correct="sum", total="count")
    counts["accuracy"] = counts["correct"] / counts["total"] * 100.0
    pivot = df.pivot_table(index=["split", "example_id"], columns="endpoint", values="correct", aggfunc="first")
    pred_pivot = df.pivot_table(index=["split", "example_id"], columns="endpoint", values="prediction", aggfunc="first")
    flip_rows = []
    endpoints = [e for e in PRED_PATHS.keys() if e != "coherent86"]
    for idx, row in pivot.iterrows():
        split, eid = idx
        coh = bool(row.get("coherent86", False))
        grow = df[(df["split"] == split) & (df["example_id"] == eid)].iloc[0]
        for ep in endpoints:
            if ep not in row or pd.isna(row[ep]):
                continue
            epc = bool(row[ep])
            if epc != coh:
                flip_rows.append({
                    "endpoint": ep,
                    "split": split,
                    "example_id": eid,
                    "coherent86_correct": coh,
                    "endpoint_correct": epc,
                    "delta_correct": int(epc) - int(coh),
                    "coherent86_prediction": pred_pivot.loc[idx].get("coherent86"),
                    "endpoint_prediction": pred_pivot.loc[idx].get(ep),
                    "gold": grow["gold"],
                    "prompt": grow["prompt"],
                    "solution0": grow["solution0"],
                    "solution1": grow["solution1"],
                })
    flips = pd.DataFrame(flip_rows)
    # Shared item sets: compare candidate correctness patterns against coherent86.
    shared_rows = []
    for split in ["parallel", "nonparallel"]:
        split_piv = pivot.loc[split]
        candidate_eps = ["dense_seed62064", "dense_seed62065", "clean_pres_seed62064", "clean_pres_seed62065"]
        for eid, row in split_piv.iterrows():
            if "coherent86" not in row:
                continue
            coh = bool(row["coherent86"])
            deltas = {ep: (None if ep not in row or pd.isna(row[ep]) else int(bool(row[ep])) - int(coh)) for ep in candidate_eps}
            all_same_delta = len(set(v for v in deltas.values() if v is not None)) == 1
            if any(v not in (None, 0) for v in deltas.values()):
                grow = df[(df["split"] == split) & (df["example_id"] == eid)].iloc[0]
                rec = {"split": split, "example_id": eid, "coherent86_correct": coh, **{f"delta_{k}": v for k, v in deltas.items()}, "all_candidates_same_nonzero_delta": all_same_delta and next(v for v in deltas.values() if v is not None) != 0, "gold": grow["gold"], "prompt": grow["prompt"], "solution0": grow["solution0"], "solution1": grow["solution1"]}
                shared_rows.append(rec)
    shared = pd.DataFrame(shared_rows)
    # Net deltas by endpoint/split.
    coh_counts = counts[counts["endpoint"] == "coherent86"].set_index("split")
    net_rows = []
    for _, r in counts.iterrows():
        split = r["split"]
        coh_correct = int(coh_counts.loc[split, "correct"]) if split in coh_counts.index else None
        net_rows.append({"endpoint": r["endpoint"], "split": split, "correct": int(r["correct"]), "total": int(r["total"]), "accuracy": float(r["accuracy"]), "delta_correct_vs_coherent86": None if coh_correct is None else int(r["correct"]) - coh_correct})
    net = pd.DataFrame(net_rows)
    # Overall GlobalPIQA mean in BabyLM coordinate: average split accuracies.
    overall_rows = []
    for ep, g in net.groupby("endpoint"):
        if set(g["split"]) >= {"parallel", "nonparallel"}:
            par = g[g["split"] == "parallel"].iloc[0]
            non = g[g["split"] == "nonparallel"].iloc[0]
            overall_rows.append({
                "endpoint": ep,
                "parallel_correct": int(par["correct"]), "parallel_total": int(par["total"]), "parallel_acc": float(par["accuracy"]), "parallel_delta": int(par["delta_correct_vs_coherent86"]) if par["delta_correct_vs_coherent86"] is not None else None,
                "nonparallel_correct": int(non["correct"]), "nonparallel_total": int(non["total"]), "nonparallel_acc": float(non["accuracy"]), "nonparallel_delta": int(non["delta_correct_vs_coherent86"]) if non["delta_correct_vs_coherent86"] is not None else None,
                "globalpiqa_mean": (float(par["accuracy"]) + float(non["accuracy"])) / 2.0,
            })
    overall = pd.DataFrame(overall_rows)
    item_csv = _public_path('experiments/archive/relation_learning/data/globalpiqa_item_table/globalpiqa_item_correctness.csv')
    flips_csv = _public_path('experiments/archive/relation_learning/data/globalpiqa_item_table/globalpiqa_flips_vs_coherent86.csv')
    shared_csv = _public_path('experiments/archive/relation_learning/data/globalpiqa_item_table/globalpiqa_candidate_changed_items.csv')
    net_csv = _public_path('experiments/archive/relation_learning/data/globalpiqa_item_table/globalpiqa_counts.csv')
    overall_csv = _public_path('experiments/archive/relation_learning/data/globalpiqa_item_table/globalpiqa_overall.csv')
    df.to_csv(item_csv, index=False)
    flips.to_csv(flips_csv, index=False)
    shared.to_csv(shared_csv, index=False)
    net.to_csv(net_csv, index=False)
    overall.to_csv(overall_csv, index=False)
    out_json = {
        "status": "GLOBALPIQA_ITEM_TABLE",
        "gold_paths": {k: str(v.relative_to(ROOT)) for k, v in GOLD_PATHS.items()},
        "missing_prediction_files": missing,
        "counts": net.to_dict(orient="records"),
        "overall": overall.to_dict(orient="records"),
        "n_flips_vs_coherent86": int(len(flips)),
        "n_candidate_changed_items": int(len(shared)),
        "outputs": {"item_csv": str(item_csv.relative_to(ROOT)), "flips_csv": str(flips_csv.relative_to(ROOT)), "shared_csv": str(shared_csv.relative_to(ROOT)), "counts_csv": str(net_csv.relative_to(ROOT)), "overall_csv": str(overall_csv.relative_to(ROOT))},
    }
    (_public_path('experiments/archive/relation_learning/data/globalpiqa_item_table/summary.json')).write_text(json.dumps(out_json, indent=2, sort_keys=True), encoding="utf-8")
    lines = ["# research GlobalPIQA item-level comparison", "", "## Split counts", "", net.to_markdown(index=False), "", "## BabyLM GlobalPIQA mean", "", overall.to_markdown(index=False), "", f"Flip rows vs coherent86: {len(flips)}", f"Candidate changed item rows: {len(shared)}", "", "Outputs:", json.dumps(out_json["outputs"], indent=2)]
    if missing:
        lines += ["", "Missing prediction files:", json.dumps(missing, indent=2)]
    (_public_path('research/documents/relation_learning/data/globalpiqa_item_table/summary.md')).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "done", "out": str(OUT.relative_to(ROOT)), "overall": out_json["overall"]}), flush=True)


if __name__ == "__main__":
    main()
