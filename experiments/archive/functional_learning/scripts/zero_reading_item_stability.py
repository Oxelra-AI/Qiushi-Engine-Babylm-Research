#!/usr/bin/env python3
"""research: item/family stability analysis for official zero-shot/Reading payloads.

Scientific purpose
------------------
The clean eval-mode preservation candidate now has one complete official endpoint
(seed62064) and one replicate with official zero-shot/Reading plus measured AoA but
pending SuperGLUE.  Aggregate zero-shot scores show the same mixed profile, but the
research question is whether this is a stable item/family displacement or a fragile
set of unrelated flips.  This script parses the actual official prediction files and
full_eval gold files to compare coherent86, dense seeds, and clean seeds at the item
and family level.  It deliberately avoids using any incomplete SuperGLUE payload.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import json
import math
import pathlib
import statistics
import time
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = _public_path('.')
STRICT_ROOT = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict')
DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/zero_reading_item_stability')

MODEL_PAYLOADS: Dict[str, str] = {
    "coherent86": "experiments/archive/frontier_consolidation/data/private_scale_sentinel_eval/coherent86_private_scale_0p75/per_target/coherent86_private_scale_0p75.json",
    "dense_seed62064": "experiments/archive/functional_learning/data/dense_focus_official_eval_seed62064/per_target/dense_focus_seed62064_u0080.json",
    "dense_seed62065": "experiments/archive/functional_learning/data/dense_focus_official_eval_seed62065/per_target/dense_focus_seed62065_u0080.json",
    "clean_seed62064": "experiments/archive/functional_learning/data/repaired_clean_eval_zero_reading/per_target/clean_pres_lambda1_eval_seed62064_u0080_zero_reading.json",
    "clean_seed62065": "experiments/archive/functional_learning/data/repaired_clean_seed62065_zero_reading/per_target/clean_pres_lambda1_eval_seed62065_u0080_zero_reading.json",
}
ZERO_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]
READING_COLUMN = "Reading"
COMPS_FILE_BY_SUBTASK = {
    "base": "comps_base",
    "wugs_dist_before": "comps_wugs_dist-before",
    "wugs_dist_in_between": "comps_wugs_dist-in-between",
    "wugs": "comps_wugs",
}


def rel(path: pathlib.Path | str | None) -> Optional[str]:
    if path is None:
        return None
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def resolve_path(path_s: str | pathlib.Path) -> pathlib.Path:
    p = pathlib.Path(path_s)
    if p.is_absolute():
        return p
    cand = ROOT / p
    if cand.exists():
        return cand
    cand2 = STRICT_ROOT / p
    if cand2.exists():
        return cand2
    return cand


def load_json_file(path_s: str | pathlib.Path) -> Any:
    return json.loads(resolve_path(path_s).read_text(encoding="utf-8"))


def iter_jsonl(path: pathlib.Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                yield i, json.loads(line)


def strip_pred(x: Any) -> Any:
    return x.strip() if isinstance(x, str) else x


def pred_list_for(preds: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
    if key not in preds:
        raise KeyError(f"missing prediction key {key}")
    vals = preds[key].get("predictions")
    if not isinstance(vals, list):
        raise TypeError(f"prediction key {key} has no list predictions")
    return vals


def parse_blimp_like(col: str, pred_path: pathlib.Path, data_dir: pathlib.Path, task_kind: str) -> Dict[str, Dict[str, Any]]:
    preds = json.loads(pred_path.read_text(encoding="utf-8"))
    out: Dict[str, Dict[str, Any]] = {}
    for subtask, pobj in preds.items():
        pred_rows = pobj.get("predictions") or []
        data_file = data_dir / f"{subtask}.jsonl"
        for idx, (pred_row, (_, data)) in enumerate(zip(pred_rows, iter_jsonl(data_file))):
            pred = strip_pred(pred_row.get("pred"))
            if task_kind == "ewok":
                gold = " ".join([data["Context1"], data["Target1"]]).strip()
                family = data.get("Domain", subtask)
                meta = {
                    "domain": data.get("Domain"),
                    "context_type": data.get("ContextType"),
                    "context_contrast": data.get("ContextDiff"),
                    "target_contrast": data.get("TargetDiff"),
                }
            else:
                gold = strip_pred(data.get("sentence_good"))
                family = data.get("UID", subtask)
                meta = {
                    "field": data.get("field"),
                    "linguistics_term": data.get("linguistics_term"),
                    "pair_id": data.get("pair_id"),
                }
            item_key = f"{col}/{subtask}/{idx}"
            out[item_key] = {
                "column": col,
                "subtask": subtask,
                "family": family,
                "idx": idx,
                "pred": pred,
                "gold": gold,
                "correct": bool(pred == gold),
                "meta": meta,
            }
        if len(pred_rows) != sum(1 for _ in iter_jsonl(data_file)):
            # The full BLiMP/EWoK/COMPS paths should match exactly.  Record by a synthetic key rather than raising,
            # so the report can preserve evidence if an old payload has an unusual but official format.
            out[f"{col}/{subtask}/__count_warning__"] = {
                "column": col,
                "subtask": subtask,
                "family": subtask,
                "idx": -1,
                "pred": None,
                "gold": None,
                "correct": False,
                "meta": {"warning": "prediction_count_mismatch", "n_pred": len(pred_rows), "n_gold": sum(1 for _ in iter_jsonl(data_file))},
            }
    return out


def parse_entity(col: str, pred_path: pathlib.Path, data_dir: pathlib.Path) -> Dict[str, Dict[str, Any]]:
    preds = json.loads(pred_path.read_text(encoding="utf-8"))
    out: Dict[str, Dict[str, Any]] = {}
    used = defaultdict(int)
    for file_path in sorted(data_dir.glob("*.jsonl")):
        split = file_path.stem
        for raw_idx, data in iter_jsonl(file_path):
            if any("nothing" in str(option) for option in data.get("options", [])):
                continue
            numops = int(data["numops"])
            subtask = f"{split}_{numops}_ops"
            rows = pred_list_for(preds, subtask)
            pos = used[subtask]
            if pos >= len(rows):
                raise IndexError(f"too few predictions for {subtask}: need {pos+1}, have {len(rows)}")
            pred = strip_pred(rows[pos].get("pred"))
            gold = strip_pred(data["options"][0])
            item_id = data.get("example_id", raw_idx)
            item_key = f"{col}/{subtask}/{item_id}"
            out[item_key] = {
                "column": col,
                "subtask": subtask,
                "family": split,
                "idx": int(item_id) if isinstance(item_id, int) else raw_idx,
                "pred": pred,
                "gold": gold,
                "correct": bool(pred == gold),
                "meta": {"split": split, "numops": numops, "sample_id": data.get("sample_id"), "raw_idx": raw_idx},
            }
            used[subtask] += 1
    for subtask, pobj in preds.items():
        n_pred = len(pobj.get("predictions") or [])
        if used[subtask] != n_pred:
            out[f"{col}/{subtask}/__count_warning__"] = {
                "column": col,
                "subtask": subtask,
                "family": subtask.split("_")[0],
                "idx": -1,
                "pred": None,
                "gold": None,
                "correct": False,
                "meta": {"warning": "prediction_count_mismatch", "n_pred": n_pred, "n_gold_filtered": used[subtask]},
            }
    return out


def parse_comps(col: str, pred_path: pathlib.Path, data_dir: pathlib.Path) -> Dict[str, Dict[str, Any]]:
    preds = json.loads(pred_path.read_text(encoding="utf-8"))
    out: Dict[str, Dict[str, Any]] = {}
    for subtask, pobj in preds.items():
        data_stem = COMPS_FILE_BY_SUBTASK[subtask]
        data_file = data_dir / f"{data_stem}.jsonl"
        pred_rows = pobj.get("predictions") or []
        for idx, (pred_row, (_, data)) in enumerate(zip(pred_rows, iter_jsonl(data_file))):
            pred = strip_pred(pred_row.get("pred"))
            gold = " ".join([data["prefix_acceptable"], data["property_phrase"]]).strip()
            item_key = f"{col}/{subtask}/{data.get('id', idx)}"
            out[item_key] = {
                "column": col,
                "subtask": subtask,
                "family": subtask,
                "idx": int(data.get("id", idx)),
                "pred": pred,
                "gold": gold,
                "correct": bool(pred == gold),
                "meta": {
                    "negative_sample_type": data.get("negative_sample_type"),
                    "property": data.get("property"),
                    "acceptable_concept": data.get("acceptable_concept"),
                    "unacceptable_concept": data.get("unacceptable_concept"),
                },
            }
        n_gold = sum(1 for _ in iter_jsonl(data_file))
        if len(pred_rows) != n_gold:
            out[f"{col}/{subtask}/__count_warning__"] = {
                "column": col,
                "subtask": subtask,
                "family": subtask,
                "idx": -1,
                "pred": None,
                "gold": None,
                "correct": False,
                "meta": {"warning": "prediction_count_mismatch", "n_pred": len(pred_rows), "n_gold": n_gold},
            }
    return out


def parse_global_piqa(col: str, pred_path: pathlib.Path, data_file: pathlib.Path) -> Dict[str, Dict[str, Any]]:
    preds = json.loads(pred_path.read_text(encoding="utf-8"))
    out: Dict[str, Dict[str, Any]] = {}
    for idx, data in iter_jsonl(data_file):
        ex_id = data["example_id"]
        pred = strip_pred(preds[ex_id]["predictions"][0]["pred"])
        gold = strip_pred(data[f"solution{data['label']}"])
        out[f"{col}/{ex_id}"] = {
            "column": col,
            "subtask": col,
            "family": data.get("categories", col),
            "idx": idx,
            "pred": pred,
            "gold": gold,
            "correct": bool(pred == gold),
            "meta": {"example_id": ex_id, "label": data.get("label"), "categories": data.get("categories")},
        }
    return out


def parse_zero_column(col: str, task: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    pred_path = resolve_path(task["predictions"])
    data_path = resolve_path(task["data_path"])
    if col in {"BLiMP", "Supplement"}:
        return parse_blimp_like(col, pred_path, data_path, "blimp")
    if col == "EWoK":
        return parse_blimp_like(col, pred_path, data_path, "ewok")
    if col == "Entity":
        return parse_entity(col, pred_path, data_path)
    if col == "COMPS":
        return parse_comps(col, pred_path, data_path)
    if col in {"GlobalPIQA_parallel", "GlobalPIQA_nonparallel"}:
        return parse_global_piqa(col, pred_path, data_path / "eng_latn.jsonl")
    raise ValueError(col)


def parse_reading(task: Dict[str, Any]) -> Dict[str, Any]:
    preds = load_json_file(task["predictions"])
    rows = preds["reading"]["predictions"]
    vals = [float(r["pred"]) for r in rows]
    prev_vals = [float(r["prev_pred"]) for r in rows]
    return {
        "n": len(rows),
        "pred": vals,
        "prev_pred": prev_vals,
        "scores": task.get("scores") or {},
    }


def official_like_score(items: Dict[str, Dict[str, Any]], col: str) -> float:
    by_sub: Dict[str, List[bool]] = defaultdict(list)
    for rec in items.values():
        if rec.get("idx") == -1:
            continue
        by_sub[rec["subtask"]].append(bool(rec["correct"]))
    rates = [100.0 * sum(v) / len(v) for _, v in sorted(by_sub.items()) if v]
    return sum(rates) / len(rates)


def summarize_model_column(items: Dict[str, Dict[str, Any]], col: str, payload_score: Optional[float]) -> Dict[str, Any]:
    vals = [bool(r["correct"]) for r in items.values() if r.get("idx") != -1]
    by_sub: Dict[str, List[bool]] = defaultdict(list)
    for rec in items.values():
        if rec.get("idx") == -1:
            continue
        by_sub[rec["subtask"]].append(bool(rec["correct"]))
    official_like = official_like_score(items, col) if vals else None
    item_weighted = 100.0 * sum(vals) / len(vals) if vals else None
    return {
        "n_items": len(vals),
        "n_subtasks": len(by_sub),
        "official_like_score_from_predictions": official_like,
        "item_weighted_accuracy": item_weighted,
        "payload_score": payload_score,
        "payload_minus_computed_official_like": (payload_score - official_like) if (payload_score is not None and official_like is not None) else None,
    }


def rates_by_group(items: Dict[str, Dict[str, Any]], group_field: str = "subtask") -> Dict[str, Dict[str, Any]]:
    d: Dict[str, List[bool]] = defaultdict(list)
    for rec in items.values():
        if rec.get("idx") == -1:
            continue
        key = str(rec.get(group_field) or rec.get("subtask"))
        d[key].append(bool(rec["correct"]))
    return {k: {"n": len(v), "acc": 100.0 * sum(v) / len(v)} for k, v in sorted(d.items()) if v}


def finite_list(vals: Iterable[float]) -> List[float]:
    out: List[float] = []
    for x in vals:
        try:
            y = float(x)
        except Exception:
            continue
        if math.isfinite(y):
            out.append(y)
    return out


def mean_float(vals: Iterable[float]) -> Optional[float]:
    finite = finite_list(vals)
    if not finite:
        return None
    return sum(finite) / len(finite)


def pstdev_float(vals: Iterable[float]) -> Optional[float]:
    finite = finite_list(vals)
    if not finite:
        return None
    mu = sum(finite) / len(finite)
    return math.sqrt(sum((x - mu) ** 2 for x in finite) / len(finite))


def pearson(a: List[float], b: List[float]) -> Optional[float]:
    pairs: List[Tuple[float, float]] = []
    for x, y in zip(a, b):
        try:
            xf, yf = float(x), float(y)
        except Exception:
            continue
        if math.isfinite(xf) and math.isfinite(yf):
            pairs.append((xf, yf))
    if len(pairs) < 2:
        return None
    av = [x for x, _ in pairs]
    bv = [y for _, y in pairs]
    ma = sum(av) / len(av)
    mb = sum(bv) / len(bv)
    da = [x - ma for x in av]
    db = [y - mb for y in bv]
    va = sum(x * x for x in da)
    vb = sum(y * y for y in db)
    if va <= 0 or vb <= 0:
        return None
    return sum(x * y for x, y in zip(da, db)) / math.sqrt(va * vb)


def correctness_vector(records: Dict[str, Dict[str, Any]], keys: List[str]) -> List[bool]:
    return [bool(records[k]["correct"]) for k in keys]


def compare_two(records_a: Dict[str, Dict[str, Any]], records_b: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    keys = sorted(set(records_a) & set(records_b))
    keys = [k for k in keys if records_a[k].get("idx") != -1 and records_b[k].get("idx") != -1]
    n = len(keys)
    if n == 0:
        return {"n": 0}
    same_correctness = sum(bool(records_a[k]["correct"]) == bool(records_b[k]["correct"]) for k in keys)
    same_pred = sum(strip_pred(records_a[k].get("pred")) == strip_pred(records_b[k].get("pred")) for k in keys)
    a_correct_b_wrong = sum(bool(records_a[k]["correct"]) and not bool(records_b[k]["correct"]) for k in keys)
    b_correct_a_wrong = sum(bool(records_b[k]["correct"]) and not bool(records_a[k]["correct"]) for k in keys)
    both_correct = sum(bool(records_a[k]["correct"]) and bool(records_b[k]["correct"]) for k in keys)
    both_wrong = sum((not bool(records_a[k]["correct"])) and (not bool(records_b[k]["correct"])) for k in keys)
    return {
        "n": n,
        "correctness_agreement_fraction": same_correctness / n,
        "prediction_string_agreement_fraction": same_pred / n,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "a_correct_b_wrong": a_correct_b_wrong,
        "b_correct_a_wrong": b_correct_a_wrong,
        "net_item_accuracy_points_a_minus_b": 100.0 * (a_correct_b_wrong - b_correct_a_wrong) / n,
    }


def change_vs_parent(parent: Dict[str, Dict[str, Any]], model: Dict[str, Dict[str, Any]], key: str) -> int:
    pc = bool(parent[key]["correct"])
    mc = bool(model[key]["correct"])
    if (not pc) and mc:
        return 1
    if pc and (not mc):
        return -1
    return 0


def compare_changes(parent: Dict[str, Dict[str, Any]], a: Dict[str, Dict[str, Any]], b: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    keys = sorted(set(parent) & set(a) & set(b))
    keys = [k for k in keys if parent[k].get("idx") != -1 and a[k].get("idx") != -1 and b[k].get("idx") != -1]
    n = len(keys)
    if n == 0:
        return {"n": 0}
    ca = {k: change_vs_parent(parent, a, k) for k in keys}
    cb = {k: change_vs_parent(parent, b, k) for k in keys}
    gains_a = {k for k, v in ca.items() if v == 1}
    gains_b = {k for k, v in cb.items() if v == 1}
    losses_a = {k for k, v in ca.items() if v == -1}
    losses_b = {k for k, v in cb.items() if v == -1}
    any_change = {k for k in keys if ca[k] != 0 or cb[k] != 0}
    same_nonzero = {k for k in any_change if ca[k] == cb[k] and ca[k] != 0}
    opposite = {k for k in any_change if ca[k] * cb[k] == -1}
    return {
        "n": n,
        "n_any_changed_vs_parent": len(any_change),
        "a_gain_parent_wrong_to_correct": len(gains_a),
        "b_gain_parent_wrong_to_correct": len(gains_b),
        "shared_gains": len(gains_a & gains_b),
        "gain_jaccard": (len(gains_a & gains_b) / len(gains_a | gains_b)) if (gains_a | gains_b) else None,
        "a_loss_parent_correct_to_wrong": len(losses_a),
        "b_loss_parent_correct_to_wrong": len(losses_b),
        "shared_losses": len(losses_a & losses_b),
        "loss_jaccard": (len(losses_a & losses_b) / len(losses_a | losses_b)) if (losses_a | losses_b) else None,
        "same_nonzero_change": len(same_nonzero),
        "opposite_change": len(opposite),
        "same_nonzero_fraction_among_any_changed": (len(same_nonzero) / len(any_change)) if any_change else None,
        "opposite_fraction_among_any_changed": (len(opposite) / len(any_change)) if any_change else None,
        "net_item_accuracy_points_a_vs_parent": 100.0 * (len(gains_a) - len(losses_a)) / n,
        "net_item_accuracy_points_b_vs_parent": 100.0 * (len(gains_b) - len(losses_b)) / n,
    }


def dense_clean_parent_decomposition(parent: Dict[str, Dict[str, Any]], dense: Dict[str, Dict[str, Any]], clean: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    keys = sorted(set(parent) & set(dense) & set(clean))
    keys = [k for k in keys if parent[k].get("idx") != -1 and dense[k].get("idx") != -1 and clean[k].get("idx") != -1]
    out = defaultdict(int)
    for k in keys:
        p = bool(parent[k]["correct"])
        d = bool(dense[k]["correct"])
        c = bool(clean[k]["correct"])
        if p and (not d) and c:
            out["clean_recovers_dense_parent_loss"] += 1
        if p and d and (not c):
            out["clean_loses_parent_item_dense_kept"] += 1
        if (not p) and d and c:
            out["clean_keeps_dense_gain"] += 1
        if (not p) and d and (not c):
            out["clean_drops_dense_gain"] += 1
        if (not p) and (not d) and c:
            out["clean_new_gain_beyond_dense"] += 1
        if p and (not d) and (not c):
            out["shared_loss_vs_parent"] += 1
    out["n"] = len(keys)
    out["net_clean_minus_dense_item_accuracy_points"] = 100.0 * ((out["clean_recovers_dense_parent_loss"] + out["clean_new_gain_beyond_dense"]) - (out["clean_loses_parent_item_dense_kept"] + out["clean_drops_dense_gain"])) / len(keys) if keys else None
    return dict(out)


def sample_items(parent: Dict[str, Dict[str, Any]], a: Dict[str, Dict[str, Any]], b: Optional[Dict[str, Dict[str, Any]]] = None, limit: int = 12) -> Dict[str, List[Dict[str, Any]]]:
    keys = sorted(set(parent) & set(a) & (set(b) if b is not None else set(a)))
    keys = [k for k in keys if parent[k].get("idx") != -1 and a[k].get("idx") != -1 and (b is None or b[k].get("idx") != -1)]
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for k in keys:
        p = bool(parent[k]["correct"])
        ac = bool(a[k]["correct"])
        bc = bool(b[k]["correct"]) if b is not None else None
        if b is None:
            label = "gain" if ((not p) and ac) else "loss" if (p and not ac) else None
        else:
            label = None
            if (not p) and ac and bc:
                label = "shared_gain"
            elif p and (not ac) and (not bc):
                label = "shared_loss"
            elif (not p) and ac and not bc:
                label = "a_only_gain"
            elif (not p) and bc and not ac:
                label = "b_only_gain"
            elif p and (not ac) and bc:
                label = "a_only_loss"
            elif p and (not bc) and ac:
                label = "b_only_loss"
        if label and len(buckets[label]) < limit:
            rec = parent[k]
            entry = {
                "key": k,
                "subtask": rec.get("subtask"),
                "family": rec.get("family"),
                "gold": rec.get("gold"),
                "parent_pred": rec.get("pred"),
                "a_pred": a[k].get("pred"),
                "a_correct": ac,
                "parent_correct": p,
                "meta": rec.get("meta"),
            }
            if b is not None:
                entry["b_pred"] = b[k].get("pred")
                entry["b_correct"] = bc
            buckets[label].append(entry)
    return dict(buckets)


def build_analysis() -> Dict[str, Any]:
    payloads = {name: load_json_file(path) for name, path in MODEL_PAYLOADS.items()}
    records: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = defaultdict(dict)
    model_summaries: Dict[str, Any] = {}
    for model, payload in payloads.items():
        tasks = payload.get("tasks") or {}
        model_summaries[model] = {"payload": rel(MODEL_PAYLOADS[model]), "scores": {}, "computed": {}}
        for col in ZERO_COLUMNS:
            task = tasks[col]
            items = parse_zero_column(col, task)
            records[model][col] = items
            payload_score = task.get("score")
            model_summaries[model]["scores"][col] = payload_score
            model_summaries[model]["computed"][col] = summarize_model_column(items, col, float(payload_score) if payload_score is not None else None)
        rtask = tasks.get(READING_COLUMN)
        if rtask:
            reading = parse_reading(rtask)
            records[model][READING_COLUMN] = {"__reading__": reading}  # type: ignore[assignment]
            model_summaries[model]["scores"][READING_COLUMN] = reading["scores"]
            model_summaries[model]["computed"][READING_COLUMN] = {"n_items": reading["n"], "scores": reading["scores"]}

    pairwise_seed: Dict[str, Any] = {}
    change_seed: Dict[str, Any] = {}
    dense_clean: Dict[str, Any] = {}
    family_rows: List[Dict[str, Any]] = []
    flip_samples: Dict[str, Any] = {}

    for col in ZERO_COLUMNS:
        clean64 = records["clean_seed62064"][col]
        clean65 = records["clean_seed62065"][col]
        parent = records["coherent86"][col]
        dense64 = records["dense_seed62064"][col]
        dense65 = records["dense_seed62065"][col]
        pairwise_seed[col] = compare_two(clean64, clean65)
        change_seed[col] = compare_changes(parent, clean64, clean65)
        dense_clean[col] = {
            "dense64_vs_clean64": compare_two(dense64, clean64),
            "dense65_vs_clean65": compare_two(dense65, clean65),
            "parent_dense64_clean64_decomposition": dense_clean_parent_decomposition(parent, dense64, clean64),
            "parent_dense65_clean65_decomposition": dense_clean_parent_decomposition(parent, dense65, clean65),
        }
        flip_samples[col] = {
            "clean64_clean65_vs_parent": sample_items(parent, clean64, clean65, limit=10),
            "clean64_vs_parent": sample_items(parent, clean64, None, limit=8),
        }

        # family/subtask rates for all five models, with deltas vs parent and seed differences.
        group_rates = {m: rates_by_group(records[m][col], "subtask") for m in MODEL_PAYLOADS}
        all_groups = sorted(set().union(*[set(v.keys()) for v in group_rates.values()]))
        for g in all_groups:
            row: Dict[str, Any] = {"column": col, "subtask": g}
            for m in MODEL_PAYLOADS:
                row[f"{m}_n"] = group_rates[m].get(g, {}).get("n")
                row[f"{m}_acc"] = group_rates[m].get(g, {}).get("acc")
            if row.get("coherent86_acc") is not None:
                for m in ["dense_seed62064", "dense_seed62065", "clean_seed62064", "clean_seed62065"]:
                    row[f"{m}_delta_vs_coherent86"] = row.get(f"{m}_acc") - row.get("coherent86_acc") if row.get(f"{m}_acc") is not None else None
            if row.get("clean_seed62064_acc") is not None and row.get("clean_seed62065_acc") is not None:
                row["clean_seed65_minus_seed64"] = row["clean_seed62065_acc"] - row["clean_seed62064_acc"]
            family_rows.append(row)

    reading_summary: Dict[str, Any] = {"scores": {}, "pairwise_pred_vector": {}}
    for m in MODEL_PAYLOADS:
        r = records[m][READING_COLUMN]["__reading__"]
        vals = r["pred"]
        reading_summary["scores"][m] = r["scores"]
        reading_summary[m] = {
            "n": r["n"],
            "pred_mean": mean_float(vals),
            "pred_std": pstdev_float(vals),
            "prev_pred_mean": mean_float(r["prev_pred"]),
            "prev_pred_std": pstdev_float(r["prev_pred"]),
        }
    for a in MODEL_PAYLOADS:
        for b in MODEL_PAYLOADS:
            if a >= b:
                continue
            ra = records[a][READING_COLUMN]["__reading__"]
            rb = records[b][READING_COLUMN]["__reading__"]
            reading_summary["pairwise_pred_vector"][f"{a}__{b}"] = {
                "n": min(ra["n"], rb["n"]),
                "pred_pearson": pearson(ra["pred"], rb["pred"]),
                "prev_pred_pearson": pearson(ra["prev_pred"], rb["prev_pred"]),
                "pred_mean_signed_diff_a_minus_b": mean_float([x - y for x, y in zip(ra["pred"], rb["pred"]) if math.isfinite(float(x)) and math.isfinite(float(y))]),
                "pred_mean_abs_diff": mean_float([abs(x - y) for x, y in zip(ra["pred"], rb["pred"]) if math.isfinite(float(x)) and math.isfinite(float(y))]),
            }

    # Compact top family movements.
    def top_rows(col: str, metric: str, n: int = 12, reverse: bool = True) -> List[Dict[str, Any]]:
        rows = [r for r in family_rows if r["column"] == col and r.get(metric) is not None]
        rows.sort(key=lambda r: r[metric], reverse=reverse)
        keep = []
        for r in rows[:n]:
            keep.append({k: r.get(k) for k in ["column", "subtask", "coherent86_n", "coherent86_acc", "dense_seed62064_acc", "clean_seed62064_acc", "clean_seed62065_acc", metric, "clean_seed65_minus_seed64"]})
        return keep

    family_extremes: Dict[str, Any] = {}
    for col in ZERO_COLUMNS:
        family_extremes[col] = {
            "clean64_top_positive_delta_vs_parent": top_rows(col, "clean_seed62064_delta_vs_coherent86", 8, True),
            "clean64_top_negative_delta_vs_parent": top_rows(col, "clean_seed62064_delta_vs_coherent86", 8, False),
            "clean65_minus_clean64_largest": top_rows(col, "clean_seed65_minus_seed64", 8, True),
            "clean65_minus_clean64_smallest": top_rows(col, "clean_seed65_minus_seed64", 8, False),
        }

    return {
        "status": "ZERO_READING_ITEM_STABILITY",
        "created_utc": now(),
        "purpose": "Determine whether clean seed62064/seed62065 official zero-shot and Reading profiles reflect stable item/family movement while SuperGLUE jobs remain pending.",
        "models": model_summaries,
        "clean_seed_item_agreement": pairwise_seed,
        "clean_seed_change_overlap_vs_coherent86": change_seed,
        "dense_to_clean_decompositions": dense_clean,
        "reading_vector_summary": reading_summary,
        "family_extremes": family_extremes,
        "notes": {
            "official_weighting": "BLiMP/Supplement/EWoK/Entity/COMPS official-style zero-shot scores are unweighted means over UID/subtask accuracies, so item-weighted counts are mechanistic evidence, not replacement leaderboard arithmetic.",
            "superglue_not_used": "This analysis deliberately uses only official zero-shot/Reading payloads and does not infer either running SuperGLUE job.",
        },
        "paths": {"payloads": MODEL_PAYLOADS},
        "family_table_rows": family_rows,
        "flip_samples": flip_samples,
    }


def write_csv(path: pathlib.Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return "None"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def write_markdown(path: pathlib.Path, result: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# research zero-shot/Reading item stability\n\n")
    lines.append("This parses actual official prediction files for coherent86, dense seeds, and clean preservation seeds. It does not use pending SuperGLUE values.\n\n")
    lines.append("## Payload score validation\n\n")
    lines.append("| model | column | n items | subtasks | payload | computed official-like | item-weighted | payload-computed |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for model, ms in result["models"].items():
        for col, comp in ms["computed"].items():
            if col == "Reading":
                lines.append(f"| {model} | Reading | {comp.get('n_items')} | - | {ms['scores'].get('Reading')} | - | - | - |\n")
            else:
                lines.append(f"| {model} | {col} | {comp.get('n_items')} | {comp.get('n_subtasks')} | {fmt(comp.get('payload_score'), 4)} | {fmt(comp.get('official_like_score_from_predictions'), 4)} | {fmt(comp.get('item_weighted_accuracy'), 4)} | {fmt(comp.get('payload_minus_computed_official_like'), 6)} |\n")
    lines.append("\n## Clean seed62064/seed62065 item replication\n\n")
    lines.append("| column | n | correct agree | pred-string agree | shared gains | shared losses | gain Jaccard | loss Jaccard | opposite changed | net item pp seed64 | net item pp seed65 |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for col in ZERO_COLUMNS:
        p = result["clean_seed_item_agreement"][col]
        c = result["clean_seed_change_overlap_vs_coherent86"][col]
        lines.append(f"| {col} | {p.get('n')} | {fmt(p.get('correctness_agreement_fraction'),4)} | {fmt(p.get('prediction_string_agreement_fraction'),4)} | {c.get('shared_gains')} | {c.get('shared_losses')} | {fmt(c.get('gain_jaccard'),4)} | {fmt(c.get('loss_jaccard'),4)} | {c.get('opposite_change')} | {fmt(c.get('net_item_accuracy_points_a_vs_parent'),4)} | {fmt(c.get('net_item_accuracy_points_b_vs_parent'),4)} |\n")
    lines.append("\n## Dense-to-clean seed-matched decomposition\n\n")
    lines.append("Counts are item-weighted: clean recovery of dense parent losses, clean loss of items dense kept, kept dense gains, dropped dense gains, and new clean gains beyond dense.\n\n")
    lines.append("| column | seed pair | n | recover dense loss | lose dense-kept parent item | keep dense gain | drop dense gain | new clean gain | shared parent loss | net clean-dense item pp |\n")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for col in ZERO_COLUMNS:
        for key, label in [("parent_dense64_clean64_decomposition", "62064"), ("parent_dense65_clean65_decomposition", "62065")]:
            d = result["dense_to_clean_decompositions"][col][key]
            lines.append(f"| {col} | {label} | {d.get('n')} | {d.get('clean_recovers_dense_parent_loss',0)} | {d.get('clean_loses_parent_item_dense_kept',0)} | {d.get('clean_keeps_dense_gain',0)} | {d.get('clean_drops_dense_gain',0)} | {d.get('clean_new_gain_beyond_dense',0)} | {d.get('shared_loss_vs_parent',0)} | {fmt(d.get('net_clean_minus_dense_item_accuracy_points'),4)} |\n")
    lines.append("\n## Reading vector stability\n\n")
    rs = result["reading_vector_summary"]
    lines.append("| pair | n | pred Pearson | prev_pred Pearson | mean signed diff a-b | mean abs diff |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    for pair, vals in rs["pairwise_pred_vector"].items():
        if "clean_seed62064" in pair or "clean_seed62065" in pair:
            lines.append(f"| {pair} | {vals.get('n')} | {fmt(vals.get('pred_pearson'),5)} | {fmt(vals.get('prev_pred_pearson'),5)} | {fmt(vals.get('pred_mean_signed_diff_a_minus_b'),5)} | {fmt(vals.get('pred_mean_abs_diff'),5)} |\n")
    lines.append("\n## Family movements most relevant to interpretation\n\n")
    for col in ZERO_COLUMNS:
        lines.append(f"### {col}\n\n")
        lines.append("Clean64 largest positive subtask deltas vs coherent86:\n\n")
        for r in result["family_extremes"][col]["clean64_top_positive_delta_vs_parent"][:6]:
            lines.append(f"- {r['subtask']}: coherent {fmt(r.get('coherent86_acc'),2)}, dense64 {fmt(r.get('dense_seed62064_acc'),2)}, clean64 {fmt(r.get('clean_seed62064_acc'),2)}, clean65 {fmt(r.get('clean_seed62065_acc'),2)}, clean64-parent {fmt(r.get('clean_seed62064_delta_vs_coherent86'),2)}\n")
        lines.append("Clean64 largest negative subtask deltas vs coherent86:\n\n")
        for r in result["family_extremes"][col]["clean64_top_negative_delta_vs_parent"][:6]:
            lines.append(f"- {r['subtask']}: coherent {fmt(r.get('coherent86_acc'),2)}, dense64 {fmt(r.get('dense_seed62064_acc'),2)}, clean64 {fmt(r.get('clean_seed62064_acc'),2)}, clean65 {fmt(r.get('clean_seed62065_acc'),2)}, clean64-parent {fmt(r.get('clean_seed62064_delta_vs_coherent86'),2)}\n")
        lines.append("\n")
    lines.append("## Scientific reading\n\n")
    lines.append("The clean seeds should be read as a repeated displacement only where item agreement and shared parent-relative gain/loss sets are high. GlobalPIQA is a small fixed item set and its equality across clean seeds should not be mistaken for broad robustness by itself. Entity gains are meaningful only if they concentrate coherently by operation/split and repeat across seeds. BLiMP/Supplement/EWoK losses remain real where both clean seeds share losses against coherent86. The dense-to-clean decomposition indicates whether preservation truly recovers dense-induced parent losses on zero-shot items or whether the score difference is mostly unrelated item churn; this complements, but does not replace, the pending SuperGLUE and exact `(M,S)` official comparison.\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = build_analysis()
    family_rows = result.pop("family_table_rows")
    flip_samples = result.pop("flip_samples")
    out_json = args.out_dir / "zero_reading_item_stability_summary.json"
    out_md = args.out_dir / "zero_reading_item_stability.md"
    out_csv = args.out_dir / "zero_shot_subtask_family_table.csv"
    out_samples = args.out_dir / "zero_shot_flip_samples.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(out_csv, family_rows)
    out_samples.write_text(json.dumps(flip_samples, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(out_md, result)
    print(json.dumps({
        "status": result["status"],
        "out_json": rel(out_json),
        "out_md": rel(out_md),
        "out_csv": rel(out_csv),
        "out_samples": rel(out_samples),
        "clean_seed_item_agreement": result["clean_seed_item_agreement"],
        "reading_clean64_clean65": result["reading_vector_summary"]["pairwise_pred_vector"].get("clean_seed62064__clean_seed62065"),
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
