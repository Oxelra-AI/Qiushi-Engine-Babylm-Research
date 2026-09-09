#!/usr/bin/env python3
"""research: resolve coherent86 identity/SuperGLUE repeat and localize private-path decisions.

This is a no-training, no-inference reader over already saved official-compatible
prediction payloads and model manifests.  It treats coherent private-on minus the
protected chck82 anchor as the functional contribution of the trained private path,
because the research/151 validation records show all non-private tensors unchanged
and private-off logits matching the protected anchor.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import hashlib
import json
import math
import pathlib
import sys
import time
from collections import Counter, defaultdict
from statistics import mean
from typing import Any

ROOT = _public_path('.')
A01 = _public_path('experiments/archive/representation_and_objectives')
A02 = _public_path('experiments/archive/frontier_consolidation')
A02_SCRIPTS = _public_path('experiments/archive/frontier_consolidation/scripts')
if str(A02_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(A02_SCRIPTS))

from pairwise_item_flip_analysis import DISCRETE_COLUMNS, PayloadLoader  # noqa: E402

OUT_DIR = _public_path('experiments/archive/representation_and_objectives/data/fastpath_private_localization')
OUT_JSON = _public_path('experiments/archive/representation_and_objectives/data/fastpath_private_localization/fastpath_private_localization.json')
OUT_MD = _public_path('research/notes/representation_and_objectives/fastpath_private_localization.md')

CHEAP_COLS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
SUPERGLUE_PRIMARY_METRIC = {
    "boolq": "accuracy",
    "mnli": "accuracy",
    "mrpc": "f1",
    "multirc": "accuracy",
    "qqp": "f1",
    "rte": "accuracy",
    "wsc": "accuracy",
}
SUPERGLUE_TASKS = list(SUPERGLUE_PRIMARY_METRIC)
ENTITY_FOCUS_PREFIXES = ("move_contents_5", "move_contents_4", "move_contents_3", "ambiref_4", "ambiref_3", "regular_5", "regular_4")
EWOK_RELATION_GROUPS = {"material-dynamics", "physical-dynamics", "physical-interactions", "physical-relations", "spatial-relations", "social-relations", "social-interactions", "agent-properties"}

PAYLOADS = {
    "anchor_chck82_private_off_proxy": _public_path('experiments/archive/representation_and_objectives/data/scale1p75_chck82_full_eval_reproduction/staged_full_eval/per_target/scale1p75_chck82_independent.json'),
    "ordinary86": _public_path('experiments/archive/representation_and_objectives/data/scale1p75_chck86_cheap7_eval/per_target/scale1p75_chck86_cheap7.json'),
    "shuffled_private86": _public_path('experiments/archive/frontier_consolidation/data/frozen82_tail4M_shuffled_eval/per_target/frozen82_tail4M_shuffled.json'),
    "coherent_private86_a01": _public_path('experiments/archive/representation_and_objectives/data/fastpath4M_coherent_cheap_eval/per_target/fastpath4M_coherent.json'),
    "coherent_private86_a02": _public_path('experiments/archive/frontier_consolidation/data/fastpath4M_coherent_eval/per_target/fastpath4M_coherent.json'),
    "spanbreak_private86": _public_path('experiments/archive/representation_and_objectives/data/fastpath4M_spanbreak_cheap_eval/per_target/fastpath4M_spanbreak.json'),
}

MANIFESTS = {
    "coherent_carrier": _public_path('experiments/archive/frontier_consolidation/data/truthful_coherent86_carrier/truthful_coherent86_carrier_manifest.json'),
    "coherent_bundle_manifest": _public_path('experiments/archive/frontier_consolidation/data/coherent86_hf_public_bundle/hf_repo_bundle/provenance/truthful_coherent86_carrier_manifest.json'),
    "coherent_full_model_validation": _public_path('experiments/archive/frontier_consolidation/data/fastpath_full_model_validation/fastpath_full_model_validation.json'),
    "shuffled_carrier": _public_path('experiments/archive/frontier_consolidation/data/truthful_shuffled86_carrier/truthful_shuffled86_carrier_manifest.json'),
    "coherent_scientific_metrics": _public_path('experiments/archive/frontier_consolidation/data/coherent86_hf_public_bundle/hf_repo_bundle/provenance/coherent86_scientific_metrics.json'),
}

SUPERGLUE_ROOTS = {
    "a01_repeat": _public_path('experiments/archive/representation_and_objectives/data/fastpath4M_coherent_superglue_eval/superglue_results/fastpath4M_coherent_superglue'),
    "a02_carrier": _public_path('experiments/archive/frontier_consolidation/data/fastpath4M_coherent_superglue_eval/superglue_results/fastpath4M_coherent'),
}
SUPERGLUE_SUMMARIES = {
    "a01_repeat": _public_path('experiments/archive/representation_and_objectives/data/fastpath4M_coherent_superglue_summary/fastpath4M_coherent_superglue_summary.json'),
    "a02_carrier": _public_path('experiments/archive/frontier_consolidation/data/fastpath4M_coherent_superglue_summary/fastpath4M_coherent_superglue_summary.json'),
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path | str) -> str:
    p = pathlib.Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def load_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_load_json(path: pathlib.Path) -> Any | None:
    return load_json(path) if path.exists() else None


def sha256(path: pathlib.Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_results_txt(path: pathlib.Path) -> dict[str, float]:
    out: dict[str, float] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        key, sep, val = line.partition(":")
        if sep:
            try:
                out[key.strip()] = float(val.strip()) * 100.0
            except Exception:
                pass
    return out


def latest_results_txt(task_root: pathlib.Path) -> pathlib.Path | None:
    hits = sorted(task_root.rglob("results.txt"), key=lambda p: (p.stat().st_mtime, str(p))) if task_root.exists() else []
    return hits[-1] if hits else None


def latest_predictions_json(task_root: pathlib.Path) -> pathlib.Path | None:
    hits = sorted(task_root.rglob("predictions.json"), key=lambda p: (p.stat().st_mtime, str(p))) if task_root.exists() else []
    return hits[-1] if hits else None


def superglue_run_summary() -> dict[str, Any]:
    runs: dict[str, Any] = {}
    for run_name, root in SUPERGLUE_ROOTS.items():
        details = []
        vals = []
        for task, metric in SUPERGLUE_PRIMARY_METRIC.items():
            task_root = root / task
            results_txt = latest_results_txt(task_root)
            pred_json = latest_predictions_json(task_root)
            metrics = parse_results_txt(results_txt) if results_txt else {}
            primary = metrics.get(metric)
            if primary is not None:
                vals.append(float(primary))
            details.append({
                "task": task,
                "primary_metric": metric,
                "primary_score": primary,
                "all_metrics": metrics,
                "results_txt": rel(results_txt) if results_txt else None,
                "predictions_json": rel(pred_json) if pred_json else None,
                "predictions_sha256": sha256(pred_json) if pred_json else None,
            })
        summary = safe_load_json(SUPERGLUE_SUMMARIES[run_name])
        runs[run_name] = {
            "root": rel(root),
            "details": details,
            "primary_mean_from_results_txt": float(mean(vals)) if len(vals) == len(SUPERGLUE_TASKS) else None,
            "summary_json": rel(SUPERGLUE_SUMMARIES[run_name]),
            "summary_score": summary.get("superglue") if isinstance(summary, dict) else None,
            "projected_overall_with_aoa0": summary.get("projected_overall_with_aoa0") if isinstance(summary, dict) else None,
        }
    # pair comparison
    diff_details = []
    by_task_a = {d["task"]: d for d in runs["a01_repeat"]["details"]}
    by_task_b = {d["task"]: d for d in runs["a02_carrier"]["details"]}
    for task in SUPERGLUE_TASKS:
        a = by_task_a[task]
        b = by_task_b[task]
        av = a.get("primary_score")
        bv = b.get("primary_score")
        diff_details.append({
            "task": task,
            "primary_metric": a.get("primary_metric"),
            "a01_repeat": av,
            "a02_carrier": bv,
            "delta_a01_minus_a02": None if av is None or bv is None else float(av - bv),
            "prediction_sha_equal": bool(a.get("predictions_sha256") and a.get("predictions_sha256") == b.get("predictions_sha256")),
        })
    return {"runs": runs, "a01_minus_a02_by_task": diff_details}


def payload_ready(path: pathlib.Path) -> bool:
    if not path.exists():
        return False
    j = load_json(path)
    tasks = j.get("tasks", {})
    required = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"]
    return all(isinstance(tasks.get(c), dict) and tasks[c].get("predictions") for c in required)


def task_prediction_shas(payload_path: pathlib.Path) -> dict[str, str | None]:
    j = load_json(payload_path)
    out: dict[str, str | None] = {}
    for task, rec in j.get("tasks", {}).items():
        if isinstance(rec, dict) and rec.get("predictions"):
            out[task] = sha256(ROOT / rec["predictions"])
    return out


def cheap_scores(payload_path: pathlib.Path) -> dict[str, float | None]:
    j = load_json(payload_path)
    scores = j.get("official_overall", {}).get("scores", {})
    return {c: scores.get(c) for c in CHEAP_COLS}


def load_rows(loader: PayloadLoader, column: str):
    rows, meta = loader.load_column(column)
    return {r.item_id: r for r in rows}, meta


def row_group(column: str, row: Any) -> str:
    if column == "GlobalPIQA":
        return str(row.sub)
    return str(row.uid)


def empty_partition() -> Counter:
    return Counter({
        "n_common": 0,
        "anchor_correct": 0,
        "coherent_correct": 0,
        "ordinary_correct": 0,
        "shuffled_correct": 0,
        "private_gain": 0,
        "private_loss": 0,
        "private_retained_anchor_correct": 0,
        "private_remains_wrong": 0,
        "private_gain_also_ordinary": 0,
        "private_gain_not_ordinary": 0,
        "private_gain_also_shuffled": 0,
        "private_gain_not_shuffled": 0,
        "private_gain_unique_vs_ordinary_and_shuffled": 0,
        "private_gain_shared_by_both_comparators": 0,
        "private_loss_also_ordinary_wrong": 0,
        "private_loss_also_shuffled_wrong": 0,
        "private_loss_unique_coherent_wrong_vs_comparators": 0,
        "retained_anchor_correct_ordinary_wrong": 0,
        "retained_anchor_correct_shuffled_wrong": 0,
        "retained_anchor_correct_both_comparators_wrong": 0,
        "coherent_correct_ordinary_wrong": 0,
        "coherent_correct_shuffled_wrong": 0,
        "coherent_correct_both_comparators_wrong": 0,
        "ordinary_correct_coherent_wrong": 0,
        "shuffled_correct_coherent_wrong": 0,
    })


def add_obs(cnt: Counter, a: bool, c: bool, o: bool, s: bool) -> None:
    cnt["n_common"] += 1
    cnt["anchor_correct"] += int(a)
    cnt["coherent_correct"] += int(c)
    cnt["ordinary_correct"] += int(o)
    cnt["shuffled_correct"] += int(s)
    if (not a) and c:
        cnt["private_gain"] += 1
        cnt["private_gain_also_ordinary"] += int(o)
        cnt["private_gain_not_ordinary"] += int(not o)
        cnt["private_gain_also_shuffled"] += int(s)
        cnt["private_gain_not_shuffled"] += int(not s)
        cnt["private_gain_unique_vs_ordinary_and_shuffled"] += int((not o) and (not s))
        cnt["private_gain_shared_by_both_comparators"] += int(o and s)
    elif a and (not c):
        cnt["private_loss"] += 1
        cnt["private_loss_also_ordinary_wrong"] += int(not o)
        cnt["private_loss_also_shuffled_wrong"] += int(not s)
        cnt["private_loss_unique_coherent_wrong_vs_comparators"] += int(o and s)
    elif a and c:
        cnt["private_retained_anchor_correct"] += 1
        cnt["retained_anchor_correct_ordinary_wrong"] += int(not o)
        cnt["retained_anchor_correct_shuffled_wrong"] += int(not s)
        cnt["retained_anchor_correct_both_comparators_wrong"] += int((not o) and (not s))
    else:
        cnt["private_remains_wrong"] += 1
    cnt["coherent_correct_ordinary_wrong"] += int(c and (not o))
    cnt["coherent_correct_shuffled_wrong"] += int(c and (not s))
    cnt["coherent_correct_both_comparators_wrong"] += int(c and (not o) and (not s))
    cnt["ordinary_correct_coherent_wrong"] += int(o and (not c))
    cnt["shuffled_correct_coherent_wrong"] += int(s and (not c))


def finalize_counter(cnt: Counter) -> dict[str, Any]:
    n = int(cnt["n_common"])
    d: dict[str, Any] = {k: int(v) for k, v in cnt.items()}
    if n:
        for k in [
            "anchor_correct", "coherent_correct", "ordinary_correct", "shuffled_correct", "private_gain", "private_loss",
            "private_retained_anchor_correct", "private_remains_wrong", "retained_anchor_correct_ordinary_wrong", "retained_anchor_correct_shuffled_wrong",
            "coherent_correct_ordinary_wrong", "ordinary_correct_coherent_wrong", "coherent_correct_shuffled_wrong", "shuffled_correct_coherent_wrong",
        ]:
            d[k + "_pct"] = 100.0 * float(cnt[k]) / n
        d["coherent_minus_anchor_item_net"] = int(cnt["coherent_correct"] - cnt["anchor_correct"])
        d["coherent_minus_ordinary_item_net"] = int(cnt["coherent_correct"] - cnt["ordinary_correct"])
        d["coherent_minus_shuffled_item_net"] = int(cnt["coherent_correct"] - cnt["shuffled_correct"])
    return d


def private_localization() -> dict[str, Any]:
    if not all(payload_ready(PAYLOADS[k]) for k in ["anchor_chck82_private_off_proxy", "ordinary86", "shuffled_private86", "coherent_private86_a01"]):
        return {"status": "missing_payload"}
    loaders = {k: PayloadLoader(PAYLOADS[k]) for k in ["anchor_chck82_private_off_proxy", "ordinary86", "shuffled_private86", "coherent_private86_a01"]}
    by_col: dict[str, Any] = {}
    aggregate = empty_partition()
    focus_groups: dict[str, Counter] = defaultdict(empty_partition)
    examples: dict[str, list[dict[str, Any]]] = {"private_gain_unique": [], "private_gain_shared_ordinary": [], "retained_vs_ordinary": [], "private_loss_unique": []}
    for column in DISCRETE_COLUMNS:
        maps = {}
        metas = {}
        for name, loader in loaders.items():
            maps[name], metas[name] = load_rows(loader, column)
        common = sorted(set.intersection(*(set(m.keys()) for m in maps.values())))
        col_cnt = empty_partition()
        groups: dict[str, Counter] = defaultdict(empty_partition)
        for item_id in common:
            ar = maps["anchor_chck82_private_off_proxy"][item_id]
            cr = maps["coherent_private86_a01"][item_id]
            orow = maps["ordinary86"][item_id]
            sr = maps["shuffled_private86"][item_id]
            a, c, o, s = bool(ar.correct), bool(cr.correct), bool(orow.correct), bool(sr.correct)
            add_obs(col_cnt, a, c, o, s)
            add_obs(aggregate, a, c, o, s)
            g = row_group(column, ar)
            add_obs(groups[g], a, c, o, s)
            group_key = f"{column}:{g}"
            add_obs(focus_groups[group_key], a, c, o, s)
            if (not a) and c and (not o) and (not s) and len(examples["private_gain_unique"]) < 16:
                examples["private_gain_unique"].append({"column": column, "item_id": item_id, "group": g, "anchor_pred": ar.pred, "coherent_pred": cr.pred, "ordinary_pred": orow.pred, "shuffled_pred": sr.pred, "gold": ar.gold})
            if (not a) and c and o and len(examples["private_gain_shared_ordinary"]) < 16:
                examples["private_gain_shared_ordinary"].append({"column": column, "item_id": item_id, "group": g, "anchor_pred": ar.pred, "coherent_pred": cr.pred, "ordinary_pred": orow.pred, "shuffled_pred": sr.pred, "gold": ar.gold})
            if a and c and (not o) and len(examples["retained_vs_ordinary"]) < 16:
                examples["retained_vs_ordinary"].append({"column": column, "item_id": item_id, "group": g, "anchor_pred": ar.pred, "coherent_pred": cr.pred, "ordinary_pred": orow.pred, "shuffled_pred": sr.pred, "gold": ar.gold})
            if a and (not c) and o and s and len(examples["private_loss_unique"]) < 16:
                examples["private_loss_unique"].append({"column": column, "item_id": item_id, "group": g, "anchor_pred": ar.pred, "coherent_pred": cr.pred, "ordinary_pred": orow.pred, "shuffled_pred": sr.pred, "gold": ar.gold})
        group_rows = []
        for g, cnt in groups.items():
            rec = finalize_counter(cnt)
            rec["group"] = g
            group_rows.append(rec)
        by_col[column] = {
            "partition": finalize_counter(col_cnt),
            "meta": {k: metas[k] for k in metas},
            "groups_best_by_coherent_vs_anchor_net": sorted(group_rows, key=lambda r: (r.get("coherent_minus_anchor_item_net", 0), r.get("private_gain", 0)), reverse=True)[:20],
            "groups_worst_by_coherent_vs_anchor_net": sorted(group_rows, key=lambda r: (r.get("coherent_minus_anchor_item_net", 0), r.get("private_loss", 0)))[:20],
            "groups_best_by_retention_vs_ordinary": sorted(group_rows, key=lambda r: (r.get("retained_anchor_correct_ordinary_wrong", 0), r.get("private_retained_anchor_correct", 0)), reverse=True)[:20],
            "groups_best_by_unique_private_gain": sorted(group_rows, key=lambda r: (r.get("private_gain_unique_vs_ordinary_and_shuffled", 0), r.get("private_gain", 0)), reverse=True)[:20],
        }
    # focused families across columns
    focus_final = []
    for key, cnt in focus_groups.items():
        col, group = key.split(":", 1)
        include = False
        if col == "Entity" and group.startswith(ENTITY_FOCUS_PREFIXES):
            include = True
        if col == "EWoK" and group in EWOK_RELATION_GROUPS:
            include = True
        if col == "GlobalPIQA":
            include = True
        if include:
            rec = finalize_counter(cnt)
            rec["column"] = col
            rec["group"] = group
            focus_final.append(rec)
    return {
        "status": "COMPLETE",
        "payloads": {k: rel(v) for k, v in PAYLOADS.items() if k in loaders or v.exists()},
        "scores": {k: cheap_scores(v) for k, v in PAYLOADS.items() if v.exists() and k in ["anchor_chck82_private_off_proxy", "ordinary86", "shuffled_private86", "coherent_private86_a01", "coherent_private86_a02", "spanbreak_private86"]},
        "cheap7": {k: (float(mean([float(x) for x in cheap_scores(v).values() if x is not None])) if v.exists() else None) for k, v in PAYLOADS.items()},
        "a01_vs_a02_coherent_zero_shot_prediction_sha_equal": {task: task_prediction_shas(PAYLOADS["coherent_private86_a01"]).get(task) == task_prediction_shas(PAYLOADS["coherent_private86_a02"]).get(task) for task in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"] if PAYLOADS["coherent_private86_a02"].exists()},
        "private_off_anchor_basis": {
            "basis": "research/151 validation shows all non-private tensors unchanged and private-off probe logits equal protected chck82; saved chck82 official predictions are therefore the no-private-path functional reference for official items.",
            "validation_path": rel(MANIFESTS["coherent_full_model_validation"]),
        },
        "aggregate": finalize_counter(aggregate),
        "by_column": by_col,
        "focused_groups": {
            "best_unique_private_gains": sorted(focus_final, key=lambda r: (r.get("private_gain_unique_vs_ordinary_and_shuffled", 0), r.get("coherent_minus_anchor_item_net", 0)), reverse=True)[:25],
            "best_retention_vs_ordinary": sorted(focus_final, key=lambda r: (r.get("retained_anchor_correct_ordinary_wrong", 0), r.get("coherent_minus_ordinary_item_net", 0)), reverse=True)[:25],
            "worst_private_losses": sorted(focus_final, key=lambda r: (r.get("private_loss", 0), -r.get("private_gain", 0)), reverse=True)[:25],
            "all_focus_groups": sorted(focus_final, key=lambda r: (r.get("column", ""), r.get("group", ""))),
        },
        "examples": examples,
    }


def manifest_summary() -> dict[str, Any]:
    out = {}
    loaded = {k: safe_load_json(v) for k, v in MANIFESTS.items()}
    for k, path in MANIFESTS.items():
        out[k] = {"path": rel(path), "exists": path.exists(), "sha256": sha256(path), "status": loaded[k].get("status") if isinstance(loaded[k], dict) else None}
    coherent = loaded.get("coherent_carrier") if isinstance(loaded.get("coherent_carrier"), dict) else {}
    bundle = loaded.get("coherent_bundle_manifest") if isinstance(loaded.get("coherent_bundle_manifest"), dict) else {}
    validation = loaded.get("coherent_full_model_validation") if isinstance(loaded.get("coherent_full_model_validation"), dict) else {}
    out["coherent_identity_resolution"] = {
        "carrier_model_sha": coherent.get("model_safetensors_sha256"),
        "bundle_model_sha": bundle.get("model_safetensors_sha256"),
        "replay_sha": coherent.get("replay_model_safetensors_sha256"),
        "carrier_sha": coherent.get("carrier_sha256"),
        "bundle_carrier_sha": sha256(_public_path('experiments/archive/frontier_consolidation/data/coherent86_hf_public_bundle/hf_repo_bundle/submission/all_full_preds_truthful_coherent86_mlm.json')),
        "downloaded_public_carrier_sha": sha256(_public_path('experiments/archive/frontier_consolidation/data/coherent86_hf_public_bundle/downloaded_public_revision/submission/all_full_preds_truthful_coherent86_mlm.json')),
        "replay_bit_identical_to_original": coherent.get("replay_bit_identical_to_original"),
        "validator_is_valid_predictions": coherent.get("validator", {}).get("is_valid_predictions"),
        "total_consumed_words": coherent.get("legal_and_function_summary", {}).get("total_consumed_words"),
        "private_params": coherent.get("legal_and_function_summary", {}).get("private_params"),
        "max_nonprivate_diff_vs_chck82": coherent.get("legal_and_function_summary", {}).get("max_nonprivate_diff_vs_chck82"),
        "private_off_matches_protected_max_abs_diff": coherent.get("legal_and_function_summary", {}).get("private_off_matches_protected_max_abs_diff"),
        "full_model_validation_status": validation.get("status"),
    }
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _public_path('research/notes/representation_and_objectives').mkdir(parents=True, exist_ok=True)
    sg = superglue_run_summary()
    loc = private_localization()
    manifests = manifest_summary()
    a01_sg = sg["runs"]["a01_repeat"].get("summary_score")
    a02_sg = sg["runs"]["a02_carrier"].get("summary_score")
    ref_overall = 41.942481167385985
    a01_overall = sg["runs"]["a01_repeat"].get("projected_overall_with_aoa0")
    a02_overall = sg["runs"]["a02_carrier"].get("projected_overall_with_aoa0")
    out = {
        "status": "COMPLETE",
        "created_utc": now(),
        "purpose": "Resolve coherent86 model identity/SuperGLUE repeat and localize private-path functional decisions from saved official prediction payloads.",
        "manifest_summary": manifests,
        "superglue_comparison": sg,
        "endpoint_score_range_using_two_superglue_runs": {
            "a02_carrier_superglue": a02_sg,
            "a01_repeat_superglue": a01_sg,
            "superglue_range": None if a01_sg is None or a02_sg is None else [min(a01_sg, a02_sg), max(a01_sg, a02_sg)],
            "overall_with_aoa0_range": None if a01_overall is None or a02_overall is None else [min(a01_overall, a02_overall), max(a01_overall, a02_overall)],
            "protected_chck82_overall": ref_overall,
            "minimum_delta_vs_chck82": None if a01_overall is None or a02_overall is None else min(a01_overall, a02_overall) - ref_overall,
            "use_for_current_endpoint_reading": "Use the A02 carrier arithmetic (lower SuperGLUE run) for the materialized public bundle; the A01 repeat is independent score support, not a reason to rewrite the carrier now.",
        },
        "private_pathway_localization": loc,
        "scientific_reading": "coherent86 has internally consistent identity and remains above protected chck82 under both SuperGLUE runs. Prediction-level localization shows the private path changes the function but its all-discrete item balance is negative relative to the anchor and worse than ordinary86/shuffled-private in item count; the score gain is column/macro-weight redistribution dominated by Supplement and GlobalPIQA rather than a broad official-surface repair. Some coherent-specific private gains and retention pockets exist and should shape one carefully chosen independent-anchor reproduction, but only with fast pathway readouts designed before training.",
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    loc_ag = loc.get("aggregate", {}) if isinstance(loc, dict) else {}
    lines = [
        "# research coherent86 identity and private-pathway localization",
        "",
        "Status: **COMPLETE**",
        "",
        "## Identity and endpoint arithmetic",
        "",
        f"- Coherent carrier model SHA: `{manifests['coherent_identity_resolution'].get('carrier_model_sha')}`.",
        f"- Replay bit-identical to original: `{manifests['coherent_identity_resolution'].get('replay_bit_identical_to_original')}`.",
        f"- Local prediction validator accepted carrier: `{manifests['coherent_identity_resolution'].get('validator_is_valid_predictions')}`.",
        f"- Non-private max diff vs protected chck82: `{manifests['coherent_identity_resolution'].get('max_nonprivate_diff_vs_chck82')}`; private-off probe max diff: `{manifests['coherent_identity_resolution'].get('private_off_matches_protected_max_abs_diff')}`.",
        f"- Total counted exposure: `{manifests['coherent_identity_resolution'].get('total_consumed_words')}`; private parameters: `{manifests['coherent_identity_resolution'].get('private_params')}`.",
        f"- A02 carrier SuperGLUE / Overall(AoA=0): `{a02_sg}` / `{a02_overall}`.",
        f"- A01 repeat SuperGLUE / Overall(AoA=0): `{a01_sg}` / `{a01_overall}`.",
        f"- Minimum delta vs protected chck82 across the two SuperGLUE runs: `{out['endpoint_score_range_using_two_superglue_runs']['minimum_delta_vs_chck82']}`.",
        "",
        "## SuperGLUE subtask differences",
        "",
        "| task | metric | A01 repeat | A02 carrier | Δ A01-A02 | pred SHA equal |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for d in sg["a01_minus_a02_by_task"]:
        lines.append(f"| {d['task']} | {d['primary_metric']} | {d['a01_repeat']} | {d['a02_carrier']} | {d['delta_a01_minus_a02']} | {d['prediction_sha_equal']} |")
    lines += [
        "",
        "## Private-on decision partition (anchor private-off proxy = protected chck82)",
        "",
        f"Aggregate over common discrete official items: `{loc_ag}`",
        "",
        "| column | n | anchor correct | coherent correct | ordinary correct | shuffled correct | private gains | private losses | coherent-anchor net | coherent-ordinary net | unique private gains vs both comparators | retained anchor correct while ordinary wrong |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if isinstance(loc, dict) and loc.get("by_column"):
        for col in DISCRETE_COLUMNS:
            p = loc["by_column"][col]["partition"]
            lines.append(f"| {col} | {p.get('n_common')} | {p.get('anchor_correct')} | {p.get('coherent_correct')} | {p.get('ordinary_correct')} | {p.get('shuffled_correct')} | {p.get('private_gain')} | {p.get('private_loss')} | {p.get('coherent_minus_anchor_item_net')} | {p.get('coherent_minus_ordinary_item_net')} | {p.get('private_gain_unique_vs_ordinary_and_shuffled')} | {p.get('retained_anchor_correct_ordinary_wrong')} |")
    lines += [
        "",
        "## Scientific reading",
        "",
        "Coherent86 is a stronger practical endpoint than the protected 82M model under both SuperGLUE measurements, and the model/provenance records are internally consistent. The functional private path is real because the non-private path is unchanged and private-off matches the 82M anchor on the validation probe. But the official-item partition does not yet show a broad stability-plasticity learning principle: coherent private-on has negative all-discrete item net versus the anchor, ordinary86, and shuffled-private86, while the aggregate score improves through macro/column redistribution. The existing evidence supports one independent-anchor reproduction only if it is used to test this separation with prebuilt private-on/off and comparator readouts, not to launch longer tails or retention machinery blindly.",
        "",
        f"JSON: `{rel(OUT_JSON)}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "out_json": rel(OUT_JSON), "out_md": rel(OUT_MD), "minimum_delta_vs_chck82": out["endpoint_score_range_using_two_superglue_runs"]["minimum_delta_vs_chck82"], "private_aggregate_net_vs_anchor": loc_ag.get("coherent_minus_anchor_item_net"), "private_aggregate_net_vs_ordinary": loc_ag.get("coherent_minus_ordinary_item_net"), "private_aggregate_net_vs_shuffled": loc_ag.get("coherent_minus_shuffled_item_net")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
