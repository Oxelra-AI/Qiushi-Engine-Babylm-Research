#!/usr/bin/env python3
"""research: CPU-only seed-variance analysis for compact_view_reinvest.

This script uses already-produced artifacts only:
  * seed43022 fast no-AoA predictions,
  * seed43122 fast no-AoA predictions,
  * seed43022 official coordinate summary,
  * both training logs/metrics/order manifests.

It does not launch training or evaluation. Its purpose is to separate what is already
known as deterministic pretraining-seed endpoint variation from the pending full
SuperGLUE/AoA/downstream result. The pending seed43122 full evaluation remains the
authoritative source for seed43122 full official-coordinate scoring.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import collections
import csv
import hashlib
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Callable

def find_user_root() -> Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
STUDY = ROOT / "experiments/archive/representation_and_objectives"
WORKSPACE = STUDY
PRISTINE_FULL = WORKSPACE / "data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval"
LOCAL_STRICT_FULL = ROOT / "experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval"

SEED430_FAST_ROOT = ROOT / "experiments/archive/frontier_consolidation/data/density_noaoa_eval_reinvest/eval_outputs/density_reinvest_noaoa/compact_view_reinvest"
SEED431_FAST_ROOT = WORKSPACE / "data/compact_reinvest_seed43122_fast/eval_outputs/compact_view_reinvest_seed43122/compact_view_reinvest_seed43122"
SEED430_METRICS = ROOT / "experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/scientific_metrics.json"
SEED431_METRICS = WORKSPACE / "training/runs/repl_compact_view_reinvest_seed43122/scientific_metrics.json"
SEED430_LOG = ROOT / "experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/training_log.jsonl"
SEED431_LOG = WORKSPACE / "training/runs/repl_compact_view_reinvest_seed43122/training_log.jsonl"
SEED430_ORDER = ROOT / "experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/example_order_manifest.json"
SEED431_ORDER = WORKSPACE / "training/runs/repl_compact_view_reinvest_seed43122/example_order_manifest.json"
SEED430_TRAIN_CMD = ROOT / "experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/train_command.json"
SEED431_TRAIN_CMD = WORKSPACE / "training/runs/repl_compact_view_reinvest_seed43122/train_command.json"
SEED430_OFFICIAL = WORKSPACE / "data/pristine_collate_seed43022/pristine_collate_seed43022_summary.json"
SEED431_FAST_SUMMARY = WORKSPACE / "data/compact_reinvest_seed43122_fast/compact_view_reinvest_seed43122_fast_summary.json"
OUT_DIR = WORKSPACE / "data/seed_variance_fast_and_dynamics"
NOTE = WORKSPACE / "notes/seed_variance_fast_and_dynamics.md"

FAST_PRED_PATHS = {
    "BLiMP": (
        SEED430_FAST_ROOT / "BLiMP/chck_100M/compact_view_reinvest_BLiMP/zero_shot/mlm/blimp/blimp_fast/predictions.json",
        SEED431_FAST_ROOT / "BLiMP/chck_100M/compact_view_reinvest_seed43122_BLiMP/zero_shot/mlm/blimp/blimp_fast/predictions.json",
    ),
    "Supplement": (
        SEED430_FAST_ROOT / "Supplement/chck_100M/compact_view_reinvest_Supplement/zero_shot/mlm/blimp/supplement_fast/predictions.json",
        SEED431_FAST_ROOT / "Supplement/chck_100M/compact_view_reinvest_seed43122_Supplement/zero_shot/mlm/blimp/supplement_fast/predictions.json",
    ),
    "EWoK": (
        SEED430_FAST_ROOT / "EWoK/chck_100M/compact_view_reinvest_EWoK/zero_shot/mlm/ewok/ewok_fast/predictions.json",
        SEED431_FAST_ROOT / "EWoK/chck_100M/compact_view_reinvest_seed43122_EWoK/zero_shot/mlm/ewok/ewok_fast/predictions.json",
    ),
    "Entity": (
        SEED430_FAST_ROOT / "Entity/chck_100M/compact_view_reinvest_Entity/zero_shot/mlm/entity_tracking/entity_tracking_fast/predictions.json",
        SEED431_FAST_ROOT / "Entity/chck_100M/compact_view_reinvest_seed43122_Entity/zero_shot/mlm/entity_tracking/entity_tracking_fast/predictions.json",
    ),
    "Entity_full": (
        SEED430_FAST_ROOT / "Entity_full/chck_100M/compact_view_reinvest_Entity_full/zero_shot/mlm/entity_tracking/entity_tracking/predictions.json",
        SEED431_FAST_ROOT / "Entity_full/chck_100M/compact_view_reinvest_seed43122_Entity_full/zero_shot/mlm/entity_tracking/entity_tracking/predictions.json",
    ),
    "COMPS": (
        SEED430_FAST_ROOT / "COMPS/chck_100M/compact_view_reinvest_COMPS/zero_shot/mlm/comps/comps/predictions.json",
        SEED431_FAST_ROOT / "COMPS/chck_100M/compact_view_reinvest_seed43122_COMPS/zero_shot/mlm/comps/comps/predictions.json",
    ),
    "GlobalPIQA_parallel": (
        SEED430_FAST_ROOT / "GlobalPIQA_parallel/chck_100M/compact_view_reinvest_GlobalPIQA_parallel/zero_shot/mlm/global_piqa_parallel/global_piqa_parallel/predictions.json",
        SEED431_FAST_ROOT / "GlobalPIQA_parallel/chck_100M/compact_view_reinvest_seed43122_GlobalPIQA_parallel/zero_shot/mlm/global_piqa_parallel/global_piqa_parallel/predictions.json",
    ),
    "GlobalPIQA_nonparallel": (
        SEED430_FAST_ROOT / "GlobalPIQA_nonparallel/chck_100M/compact_view_reinvest_GlobalPIQA_nonparallel/zero_shot/mlm/global_piqa_nonparallel/global_piqa_nonparallel/predictions.json",
        SEED431_FAST_ROOT / "GlobalPIQA_nonparallel/chck_100M/compact_view_reinvest_seed43122_GlobalPIQA_nonparallel/zero_shot/mlm/global_piqa_nonparallel/global_piqa_nonparallel/predictions.json",
    ),
    "Reading": (
        SEED430_FAST_ROOT / "Reading/chck_100M/compact_view_reinvest_Reading/zero_shot/mlm/reading/predictions.json",
        SEED431_FAST_ROOT / "Reading/chck_100M/compact_view_reinvest_seed43122_Reading/zero_shot/mlm/reading/predictions.json",
    ),
}

FAST_DATA = {
    "BLiMP": ROOT / "experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/fast_eval/blimp_fast",
    "Supplement": ROOT / "experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/fast_eval/supplement_fast",
    "EWoK": ROOT / "experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/fast_eval/evaluation_data/fast_eval/ewok_fast",
    "Entity": ROOT / "experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/fast_eval/entity_tracking_fast",
    "Entity_full": LOCAL_STRICT_FULL / "entity_tracking",
    "COMPS": LOCAL_STRICT_FULL / "comps",
    "GlobalPIQA_parallel": LOCAL_STRICT_FULL / "global_piqa_parallel/eng_latn.jsonl",
    "GlobalPIQA_nonparallel": LOCAL_STRICT_FULL / "global_piqa_nonparallel/eng_latn.jsonl",
    "Reading": LOCAL_STRICT_FULL / "reading/reading_data.csv",
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_jsonl(p: Path) -> list[dict[str, Any]]:
    out = []
    with p.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def flatten_preds(preds: dict[str, Any]) -> list[tuple[str, int, str, dict[str, Any]]]:
    out = []
    for subtask, block in preds.items():
        for i, rec in enumerate(block.get("predictions", [])):
            out.append((subtask, i, str(rec.get("pred", "")).strip(), rec))
    return out


def mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def target_blimp(subtask: str, i: int, data_root: Path) -> str:
    rows = read_jsonl((data_root / subtask).with_suffix(".jsonl"))
    return rows[i]["sentence_good"].strip()


def target_ewok(subtask: str, i: int, data_root: Path) -> str:
    rows = read_jsonl((data_root / subtask).with_suffix(".jsonl"))
    r = rows[i]
    return " ".join([r["Context1"], r["Target1"]]).strip()


def iter_entity_targets(data_root: Path, filtered_nothing: bool) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for p in sorted(data_root.glob("*.jsonl")):
        rows = read_jsonl(p)
        buckets: dict[int, list[str]] = collections.defaultdict(list)
        for ex in rows:
            if filtered_nothing and any("nothing" in str(option) for option in ex["options"]):
                continue
            buckets[int(ex["numops"])].append(str(ex["options"][0]).strip())
        for numops, vals in buckets.items():
            out[f"{p.stem}_{numops}_ops"] = vals
    return out


def iter_comps_targets(data_root: Path) -> dict[str, list[str]]:
    subtask_to_file = {
        "base": "comps_base",
        "wugs_dist_before": "comps_wugs_dist-before",
        "wugs_dist_in_between": "comps_wugs_dist-in-between",
        "wugs": "comps_wugs",
    }
    out = {}
    for subtask, fname in subtask_to_file.items():
        vals = []
        for ex in read_jsonl((data_root / fname).with_suffix(".jsonl")):
            vals.append(" ".join([ex["prefix_acceptable"], ex["property_phrase"]]).strip())
        out[subtask] = vals
    return out


def iter_globalpiqa_targets(path: Path) -> dict[str, str]:
    out = {}
    for ex in read_jsonl(path):
        idx = ex.get("example_id")
        gold = ex.get(f"solution{ex.get('label')}")
        out[str(idx)] = (" " + str(gold)).strip()
    return out


def analyze_choice_task(column: str, target_func: Callable[[str, int], str]) -> dict[str, Any]:
    p430, p431 = FAST_PRED_PATHS[column]
    j430, j431 = load_json(p430), load_json(p431)
    subtasks = sorted(set(j430) & set(j431))
    total = agree = both_correct = both_wrong = only430 = only431 = 0
    sub = {}
    flips = []
    for st in subtasks:
        arr430 = j430[st]["predictions"]
        arr431 = j431[st]["predictions"]
        n = min(len(arr430), len(arr431))
        s = collections.Counter()
        for i in range(n):
            pred430 = str(arr430[i].get("pred", "")).strip()
            pred431 = str(arr431[i].get("pred", "")).strip()
            tgt = target_func(st, i)
            c430 = pred430 == tgt
            c431 = pred431 == tgt
            s["total"] += 1
            total += 1
            if pred430 == pred431:
                s["agree"] += 1; agree += 1
            if c430 and c431:
                s["both_correct"] += 1; both_correct += 1
            elif (not c430) and (not c431):
                s["both_wrong"] += 1; both_wrong += 1
            elif c430 and not c431:
                s["only430_correct"] += 1; only430 += 1
                if len(flips) < 80:
                    flips.append({"column": column, "subtask": st, "index": i, "direction": "430_correct_431_wrong", "target": tgt, "pred430": pred430, "pred431": pred431})
            elif c431 and not c430:
                s["only431_correct"] += 1; only431 += 1
                if len(flips) < 80:
                    flips.append({"column": column, "subtask": st, "index": i, "direction": "431_correct_430_wrong", "target": tgt, "pred430": pred430, "pred431": pred431})
        if s["total"]:
            sub[st] = {
                **dict(s),
                "agreement_rate": s["agree"] / s["total"],
                "seed430_accuracy": (s["both_correct"] + s["only430_correct"]) / s["total"],
                "seed431_accuracy": (s["both_correct"] + s["only431_correct"]) / s["total"],
                "delta_431_minus_430_pctpt": 100 * ((s["both_correct"] + s["only431_correct"]) - (s["both_correct"] + s["only430_correct"])) / s["total"],
                "net_items_431_minus_430": s["only431_correct"] - s["only430_correct"],
            }
    worst = sorted(sub.items(), key=lambda kv: kv[1]["delta_431_minus_430_pctpt"])[:20]
    best = sorted(sub.items(), key=lambda kv: kv[1]["delta_431_minus_430_pctpt"], reverse=True)[:20]
    return {
        "column": column,
        "paths": {"seed430": str(p430), "seed431": str(p431)},
        "total": total,
        "agreement_rate": agree / total if total else None,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "only_seed430_correct": only430,
        "only_seed431_correct": only431,
        "net_items_431_minus_430": only431 - only430,
        "seed430_accuracy": (both_correct + only430) / total if total else None,
        "seed431_accuracy": (both_correct + only431) / total if total else None,
        "delta_431_minus_430_pctpt_micro": 100 * ((only431 - only430) / total) if total else None,
        "subtasks": sub,
        "worst_subtasks_for_seed431": [{"subtask": k, **v} for k, v in worst],
        "best_subtasks_for_seed431": [{"subtask": k, **v} for k, v in best],
        "example_flips_sample": flips,
    }


def analyze_entity(column: str, data_root: Path, filtered_nothing: bool) -> dict[str, Any]:
    targets = iter_entity_targets(data_root, filtered_nothing)
    return analyze_choice_task(column, lambda st, i: targets[st][i])


def analyze_comps() -> dict[str, Any]:
    targets = iter_comps_targets(FAST_DATA["COMPS"])
    return analyze_choice_task("COMPS", lambda st, i: targets[st][i])


def analyze_global(column: str) -> dict[str, Any]:
    targets = iter_globalpiqa_targets(FAST_DATA[column])
    p430, p431 = FAST_PRED_PATHS[column]
    j430, j431 = load_json(p430), load_json(p431)
    # GlobalPIQA prediction files are keyed directly by official example_id, e.g.
    # parallel_ex000000_eng_latn or group..._v1. Do not parse the prediction record
    # id suffix because it ends in _0 for all examples; use the outer key.
    keys = [k for k in j430.keys() if k in j431]
    missing_targets = [k for k in keys if k not in targets]
    if missing_targets:
        raise KeyError(f"{column}: {len(missing_targets)} prediction keys missing from target data; first={missing_targets[:3]}")
    cnt = collections.Counter(); flips=[]
    for i, exid in enumerate(keys):
        arr430 = j430[exid].get("predictions", [])
        arr431 = j431[exid].get("predictions", [])
        if not arr430 or not arr431:
            continue
        r430, r431 = arr430[0], arr431[0]
        tgt = targets[exid]
        pred430 = str(r430.get("pred", "")).strip()
        pred431 = str(r431.get("pred", "")).strip()
        c430, c431 = pred430 == tgt, pred431 == tgt
        cnt["total"] += 1
        if pred430 == pred431: cnt["agree"] += 1
        if c430 and c431: cnt["both_correct"] += 1
        elif (not c430) and (not c431): cnt["both_wrong"] += 1
        elif c430 and not c431:
            cnt["only430_correct"] += 1
            if len(flips) < 80: flips.append({"index": i, "id": exid, "direction": "430_correct_431_wrong", "target": tgt, "pred430": pred430, "pred431": pred431})
        else:
            cnt["only431_correct"] += 1
            if len(flips) < 80: flips.append({"index": i, "id": exid, "direction": "431_correct_430_wrong", "target": tgt, "pred430": pred430, "pred431": pred431})
    total = cnt["total"]
    return {
        "column": column,
        "paths": {"seed430": str(p430), "seed431": str(p431)},
        "num_prediction_keys_seed430": len(j430),
        "num_prediction_keys_seed431": len(j431),
        **dict(cnt),
        "agreement_rate": cnt["agree"] / total if total else None,
        "seed430_accuracy": (cnt["both_correct"] + cnt["only430_correct"]) / total if total else None,
        "seed431_accuracy": (cnt["both_correct"] + cnt["only431_correct"]) / total if total else None,
        "delta_431_minus_430_pctpt_micro": 100 * ((cnt["only431_correct"] - cnt["only430_correct"]) / total) if total else None,
        "net_items_431_minus_430": cnt["only431_correct"] - cnt["only430_correct"],
        "example_flips_sample": flips,
    }


def analyze_reading() -> dict[str, Any]:
    p430, p431 = FAST_PRED_PATHS["Reading"]
    j430, j431 = load_json(p430), load_json(p431)
    arr430 = j430["reading"]["predictions"]
    arr431 = j431["reading"]["predictions"]
    n = min(len(arr430), len(arr431))
    diffs_pred = []
    diffs_prev = []
    exact_same = 0
    finite = 0
    for i in range(n):
        a, b = arr430[i], arr431[i]
        pa, pb = float(a.get("pred", float("nan"))), float(b.get("pred", float("nan")))
        qa, qb = float(a.get("prev_pred", float("nan"))), float(b.get("prev_pred", float("nan")))
        if math.isfinite(pa) and math.isfinite(pb):
            diffs_pred.append(pb - pa); finite += 1
            if abs(pb - pa) < 1e-12: exact_same += 1
        if math.isfinite(qa) and math.isfinite(qb):
            diffs_prev.append(qb - qa)
    def stats(xs: list[float]) -> dict[str, Any]:
        if not xs:
            return {}
        s = sorted(xs)
        return {
            "n": len(xs),
            "mean_431_minus_430": mean(xs),
            "median": statistics.median(xs),
            "mean_abs": mean([abs(x) for x in xs]),
            "p05": s[int(0.05*(len(s)-1))],
            "p95": s[int(0.95*(len(s)-1))],
            "min": s[0],
            "max": s[-1],
        }
    return {
        "column": "Reading",
        "paths": {"seed430": str(p430), "seed431": str(p431)},
        "num_predictions_compared": n,
        "finite_pred_pairs": finite,
        "exact_same_pred_count": exact_same,
        "pred_diff_stats": stats(diffs_pred),
        "prev_pred_diff_stats": stats(diffs_prev),
        "interpretation_note": "Reading scores are regression correlations over surprisal predictors; item-level correctness does not apply. Diff stats describe raw surprisal shifts only.",
    }


def load_training_log(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def window_stats(log: list[dict[str, Any]], window: int = 100) -> dict[str, Any]:
    losses = [float(r["loss"]) for r in log]
    masks = [float(r.get("effective_mask_rate", float("nan"))) for r in log if math.isfinite(float(r.get("effective_mask_rate", float("nan"))))]
    return {
        "num_steps": len(log),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_mean_first100": mean(losses[:window]),
        "loss_mean_last100": mean(losses[-window:]),
        "loss_mean_all": mean(losses),
        "loss_min": min(losses),
        "loss_max": max(losses),
        "effective_mask_rate_mean": mean(masks),
        "effective_mask_rate_std": statistics.pstdev(masks) if masks else None,
    }


def compare_training() -> dict[str, Any]:
    l430 = load_training_log(SEED430_LOG)
    l431 = load_training_log(SEED431_LOG)
    by_step430 = {int(r["step"]): r for r in l430}
    by_step431 = {int(r["step"]): r for r in l431}
    common = sorted(set(by_step430) & set(by_step431))
    diffs = [float(by_step431[s]["loss"]) - float(by_step430[s]["loss"]) for s in common]
    checkpoints = [1, 10, 50, 100, 200, 500, 1000, 1500, 2000, 2529]
    loss_at = {}
    for s in checkpoints:
        if s in by_step430 and s in by_step431:
            loss_at[str(s)] = {
                "seed430": float(by_step430[s]["loss"]),
                "seed431": float(by_step431[s]["loss"]),
                "delta_431_minus_430": float(by_step431[s]["loss"]) - float(by_step430[s]["loss"]),
                "cum_words_430": int(by_step430[s]["cumulative_word_exposure"]),
                "cum_words_431": int(by_step431[s]["cumulative_word_exposure"]),
            }
    # 100-step block deltas, to see whether the gap is stable late.
    block_deltas = []
    for start in range(1, max(common)+1, 100):
        ss = [s for s in common if start <= s < start + 100]
        if not ss:
            continue
        block_deltas.append({
            "step_start": start,
            "step_end": ss[-1],
            "mean_delta_loss_431_minus_430": mean([float(by_step431[s]["loss"]) - float(by_step430[s]["loss"]) for s in ss]),
            "mean_loss_430": mean([float(by_step430[s]["loss"]) for s in ss]),
            "mean_loss_431": mean([float(by_step431[s]["loss"]) for s in ss]),
        })
    return {
        "paths": {"seed430": str(SEED430_LOG), "seed431": str(SEED431_LOG)},
        "seed430": window_stats(l430),
        "seed431": window_stats(l431),
        "common_steps": len(common),
        "loss_delta_431_minus_430": {
            "mean_all_common": mean(diffs),
            "mean_last100": mean(diffs[-100:]),
            "median_all": statistics.median(diffs),
            "min": min(diffs),
            "max": max(diffs),
        },
        "loss_at_steps": loss_at,
        "block100_deltas_tail": block_deltas[-8:],
    }


def compare_recipe_and_order() -> dict[str, Any]:
    cmd430, cmd431 = load_json(SEED430_TRAIN_CMD), load_json(SEED431_TRAIN_CMD)
    order430, order431 = load_json(SEED430_ORDER), load_json(SEED431_ORDER)
    metrics430, metrics431 = load_json(SEED430_METRICS), load_json(SEED431_METRICS)
    recipe430 = cmd430.get("recipe", cmd430.get("fixed_recipe", {}))
    recipe431 = cmd431.get("recipe", cmd431.get("fixed_recipe", {}))
    ignore = {"extra_init_seed", "train_rng_seed"}
    recipe_diffs = {}
    for k in sorted(set(recipe430) | set(recipe431)):
        if k in ignore:
            continue
        if recipe430.get(k) != recipe431.get(k):
            recipe_diffs[k] = {"seed430": recipe430.get(k), "seed431": recipe431.get(k)}
    order_diff = {}
    for k in sorted(set(order430) | set(order431)):
        if order430.get(k) != order431.get(k):
            order_diff[k] = {"seed430": order430.get(k), "seed431": order431.get(k)}
    ckpt430 = [x["name"] for x in metrics430.get("saved_checkpoints", [])]
    ckpt431 = [x["name"] for x in metrics431.get("saved_checkpoints", [])]
    return {
        "train_file_hashes": {
            "seed430_actual_sha256": cmd430.get("actual_sha256"),
            "seed431_actual_sha256": cmd431.get("actual_sha256"),
            "same_train_stream_hash": cmd430.get("actual_sha256") == cmd431.get("actual_sha256"),
        },
        "different_seeds": {
            "seed430_extra_init_seed": recipe430.get("extra_init_seed"),
            "seed431_extra_init_seed": recipe431.get("extra_init_seed"),
            "seed430_train_rng_seed": recipe430.get("train_rng_seed"),
            "seed431_train_rng_seed": recipe431.get("train_rng_seed"),
        },
        "recipe_diffs_except_seed_fields": recipe_diffs,
        "order_manifest_diffs": order_diff,
        "checkpoint_names_identical": ckpt430 == ckpt431,
        "num_checkpoints": {"seed430": len(ckpt430), "seed431": len(ckpt431)},
        "word_exposure": {"seed430": metrics430.get("word_exposure"), "seed431": metrics431.get("word_exposure")},
        "actual_training_steps": {"seed430": metrics430.get("actual_training_steps"), "seed431": metrics431.get("actual_training_steps")},
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fast_summary = load_json(SEED431_FAST_SUMMARY)
    seed430_official = load_json(SEED430_OFFICIAL)
    analyses = {}
    analyses["BLiMP"] = analyze_choice_task("BLiMP", lambda st, i: target_blimp(st, i, FAST_DATA["BLiMP"]))
    analyses["Supplement"] = analyze_choice_task("Supplement", lambda st, i: target_blimp(st, i, FAST_DATA["Supplement"]))
    analyses["EWoK"] = analyze_choice_task("EWoK", lambda st, i: target_ewok(st, i, FAST_DATA["EWoK"]))
    analyses["Entity"] = analyze_entity("Entity", FAST_DATA["Entity"], filtered_nothing=False)
    analyses["Entity_full"] = analyze_entity("Entity_full", FAST_DATA["Entity_full"], filtered_nothing=True)
    analyses["COMPS"] = analyze_comps()
    analyses["GlobalPIQA_parallel"] = analyze_global("GlobalPIQA_parallel")
    analyses["GlobalPIQA_nonparallel"] = analyze_global("GlobalPIQA_nonparallel")
    analyses["Reading"] = analyze_reading()

    fast_table = fast_summary["table"]
    seed431_fast = fast_table["compact_view_reinvest_seed43122"]
    seed430_fast = fast_table["a02_seed43022_compact_view_reinvest"]
    fast_columns = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_mean", "Reading"]
    fast_delta = {k: seed431_fast[k] - seed430_fast[k] for k in fast_columns}
    equal7_delta = seed431_fast["equal7_mean"] - seed430_fast["equal7_mean"]
    contribution_to_equal7 = {k: fast_delta[k] / 7.0 for k in fast_columns}

    training = compare_training()
    recipe = compare_recipe_and_order()

    # A provisional lower-bound estimate for seed43122 full coordinate if official SuperGLUE/AoA matched seed43022 exactly.
    # This is not used as evidence of the final result; it shows how much zero-shot gap alone would consume.
    seed430_official_overall = seed430_official.get("official_path_scores") or seed430_official.get("score_summary", {}).get("official_overall")
    if not seed430_official_overall:
        raise KeyError("Cannot locate seed43022 official score block in research summary")
    seed430_scores = seed430_official_overall["scores"]
    official_replaced_by_fast_deltas = {
        "BLiMP": seed430_scores["BLiMP"] + fast_delta["BLiMP"],
        "Supplement": seed430_scores["Supplement"] + fast_delta["Supplement"],
        "EWoK": seed430_scores["EWoK"] + fast_delta["EWoK"],
        "Entity": seed430_scores["Entity"] + fast_delta["Entity"],
        "COMPS": seed430_scores["COMPS"] + fast_delta["COMPS"],
        "GlobalPIQA": seed430_scores["GlobalPIQA"] + fast_delta["GlobalPIQA_mean"],
        "Reading": seed430_scores["Reading"] + fast_delta["Reading"],
        "SuperGLUE": seed430_scores["SuperGLUE"],
        "AoA": seed430_scores["AoA"],
    }
    provisional_same_sg_aoa = sum(official_replaced_by_fast_deltas.values()) / 9.0

    summary = {
        "status": "SEED_VARIANCE_FAST_AND_DYNAMICS",
        "created_utc": now(),
        "purpose": "CPU-only quantification of known seed43022 vs seed43122 endpoint variation while full seed43122 official-compatible evaluation remains managed asynchronously.",
        "recipe_and_order": recipe,
        "fast_endpoint_delta_431_minus_430": {
            "columns": fast_delta,
            "equal7_delta": equal7_delta,
            "contribution_to_equal7": contribution_to_equal7,
            "seed430_fast_equal7": seed430_fast["equal7_mean"],
            "seed431_fast_equal7": seed431_fast["equal7_mean"],
        },
        "fast_item_analyses": analyses,
        "training_dynamics": training,
        "seed430_official_coordinate_reference": {
            "summary": str(SEED430_OFFICIAL),
            "official_overall": seed430_official_overall["Overall"],
            "official_scores": seed430_scores,
        },
        "provisional_same_superglue_aoa_estimate_not_final": {
            "description": "Apply fast seed43122-minus-seed43022 endpoint deltas to seed43022 official full columns while holding SuperGLUE and AoA equal. This is only a sensitivity calculation; the real full vector remains pending.",
            "columns": official_replaced_by_fast_deltas,
            "overall": provisional_same_sg_aoa,
            "margin_over_visible_leader_41p8": provisional_same_sg_aoa - 41.8,
        },
        "interpretation": {
            "known_changed_variables": "train stream and example order are byte-identical; fixed recipe is identical except extra_init_seed and train_rng_seed. Observed endpoint differences therefore include pretraining initialization and training RNG, not data or tokenizer changes.",
            "seed43122_fast_weakness": "At 100M, seed43122 is lower by -1.3557 equal7. The largest fast losses are EWoK (-3.73), Supplement (-3.20), Entity (-1.39), then BLiMP (-0.88); Reading is higher (+0.625).",
            "not_yet_settled": "Downstream SuperGLUE fine-tuning randomness and official min_context=0 AoA for seed43122 are not yet known here; do not use this CPU analysis to accept or reject the full SOTA endpoint. Use it to interpret the completed seed43122 full evaluation when available.",
        },
    }

    out_json = OUT_DIR / "seed_variance_fast_and_dynamics.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Compact CSV: worst subtask deltas across choice tasks.
    csv_path = OUT_DIR / "seed_variance_worst_fast_subtasks.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["column", "subtask", "total", "agreement_rate", "seed430_accuracy", "seed431_accuracy", "delta_431_minus_430_pctpt", "net_items_431_minus_430"])
        w.writeheader()
        for col, a in analyses.items():
            if col == "Reading":
                continue
            for rec in a.get("worst_subtasks_for_seed431", [])[:10]:
                w.writerow({k: rec.get(k) for k in w.fieldnames})

    note_lines = []
    note_lines.append("# research — seed variance before full seed43122 delivery\n")
    note_lines.append(f"Machine-readable JSON: `{out_json}`")
    note_lines.append(f"Worst-subtask CSV: `{csv_path}`\n")
    note_lines.append("## Controlled comparison")
    note_lines.append(f"- Same 100M stream hash: `{recipe['train_file_hashes']['seed430_actual_sha256']}` vs `{recipe['train_file_hashes']['seed431_actual_sha256']}`; same={recipe['train_file_hashes']['same_train_stream_hash']}.")
    note_lines.append(f"- Recipe differences other than init/RNG seeds: `{recipe['recipe_diffs_except_seed_fields']}`.")
    note_lines.append(f"- Order manifest differences: `{recipe['order_manifest_diffs']}`.")
    note_lines.append(f"- Changed seeds: seed43022 extra_init/train_rng={recipe['different_seeds']['seed430_extra_init_seed']}/{recipe['different_seeds']['seed430_train_rng_seed']}; seed43122 extra_init/train_rng={recipe['different_seeds']['seed431_extra_init_seed']}/{recipe['different_seeds']['seed431_train_rng_seed']}.\n")
    note_lines.append("## Known fast endpoint movement (seed43122 − seed43022)")
    for k in fast_columns:
        note_lines.append(f"- {k}: {fast_delta[k]:+.4f} (equal7 contribution {contribution_to_equal7[k]:+.4f})")
    note_lines.append(f"- Equal7: {equal7_delta:+.4f} ({seed431_fast['equal7_mean']:.6f} vs {seed430_fast['equal7_mean']:.6f}).\n")
    note_lines.append("## Training dynamics")
    note_lines.append(f"- Final training loss: seed43022 {training['seed430']['loss_last']:.6f}, seed43122 {training['seed431']['loss_last']:.6f}, delta {training['loss_delta_431_minus_430']['mean_last100']:+.6f} mean over the last 100 matched steps.")
    note_lines.append(f"- Mean loss over all matched steps delta seed43122−seed43022: {training['loss_delta_431_minus_430']['mean_all_common']:+.6f}.")
    note_lines.append("- The training-loss disadvantage is broad and late, so the fast endpoint gap is not merely a reporting artifact; however, loss is not sufficient to predict the full official vector.\n")
    note_lines.append("## Item/subfamily concentration")
    for col in ["EWoK", "Supplement", "Entity", "BLiMP", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]:
        a = analyses[col]
        note_lines.append(f"- {col}: agreement {a.get('agreement_rate'):.4f}, only430_correct={a.get('only_seed430_correct', a.get('only430_correct'))}, only431_correct={a.get('only_seed431_correct', a.get('only431_correct'))}, net {a.get('net_items_431_minus_430')}; worst subtasks: " + "; ".join([f"{r['subtask']} {r['delta_431_minus_430_pctpt']:+.2f}" for r in a.get('worst_subtasks_for_seed431', [])[:5]]))
    note_lines.append("\n## Sensitivity only, not final")
    note_lines.append(f"If fast deltas were applied to seed43022's official full columns while holding SuperGLUE and AoA equal, the score would be {provisional_same_sg_aoa:.6f} (margin {provisional_same_sg_aoa - 41.8:+.6f}). This is not a result; the managed full evaluation decides the actual seed43122 coordinate.")
    note_lines.append("\n## Scientific interpretation")
    note_lines.append("The known seed43122 weakness is already visible in deterministic zero-shot/Reading outputs under identical data, tokenizer, architecture, and schedule, and is concentrated most strongly in EWoK and Supplement, with Entity/BLiMP secondary. Because SuperGLUE fine-tuning and official AoA are still pending, this analysis should guide interpretation rather than trigger ad hoc repair. If seed43122 falls below the leader, the next work should compare full components and possibly repeated downstream fine-tuning/eval seeds before changing the pretraining recipe. If it clears the leader, both endpoint coordinates should be frozen and the prepared adjacency-broken control becomes the key mechanism test.")
    NOTE.write_text("\n".join(note_lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": summary["status"],
        "out_json": str(out_json),
        "note": str(NOTE),
        "equal7_delta_431_minus_430": equal7_delta,
        "largest_fast_losses": {k: fast_delta[k] for k in sorted(fast_delta, key=lambda x: fast_delta[x])[:4]},
        "same_train_stream_hash": recipe["train_file_hashes"]["same_train_stream_hash"],
        "recipe_diffs_except_seed_fields": recipe["recipe_diffs_except_seed_fields"],
        "loss_delta_last100": training["loss_delta_431_minus_430"]["mean_last100"],
        "sensitivity_overall_same_sg_aoa": provisional_same_sg_aoa,
        "sensitivity_margin": provisional_same_sg_aoa - 41.8,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
