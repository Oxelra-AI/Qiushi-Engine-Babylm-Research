#!/usr/bin/env python3
"""research: collate scale1.75 80M/100M score coordinates from completed parts.

This script does no evaluation. It reads zero-shot/Reading payloads, individual
SuperGLUE finetune outputs, and AoA records, then computes the exact local
leaderboard-style nine-column Overall using the corrected scoring helper.
MRPC and QQP use F1; all other SuperGLUE subtasks use accuracy, matching the
research/official-coordinate patch.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import pathlib
import sys
import time
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
COMPACT_EXPERIENCE_SCRIPTS = USER_ROOT / "experiments/archive/compact_experience/scripts"
if str(COMPACT_EXPERIENCE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(COMPACT_EXPERIENCE_SCRIPTS))
import full_overall_eval_runner as base  # noqa: E402
from babylm_official_scoring import patch_payload_official_overall, compute_overall_from_scores  # noqa: E402

OUT_DIR = USER_ROOT / "experiments/archive/representation_and_objectives/data/scale1p75_score_collation"
NOTE = USER_ROOT / "research/notes/representation_and_objectives/scale1p75_score_establishment.md"
PRIMARY = {"boolq": "accuracy", "mnli": "accuracy", "mrpc": "f1", "multirc": "accuracy", "qqp": "f1", "rte": "accuracy", "wsc": "accuracy"}
SPEC = {s["task"]: s for s in base.SUPERGLUE_TASKS}
ORDER = [s["task"] for s in base.SUPERGLUE_TASKS]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(p: pathlib.Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def latest_file(root: pathlib.Path, name: str) -> pathlib.Path | None:
    hits = sorted(root.rglob(name), key=lambda p: (p.stat().st_mtime, str(p))) if root.exists() else []
    return hits[-1] if hits else None


def parse_results_txt(path: pathlib.Path) -> dict[str, float]:
    d: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        k, sep, v = line.partition(":")
        if sep:
            try:
                d[k.strip()] = float(v.strip()) * 100.0
            except Exception:
                pass
    return d


def score_task(task: str, root: pathlib.Path, save_root: pathlib.Path | None = None, log_root: pathlib.Path | None = None) -> dict[str, Any] | None:
    pred = latest_file(root, "predictions.json")
    res = latest_file(root, "results.txt")
    if pred is None or res is None:
        return None
    acc_rec = base.score_superglue_predictions(task, pred)
    metrics = parse_results_txt(res)
    metric = PRIMARY[task]
    if metric not in metrics:
        raise RuntimeError({"missing_metric": metric, "task": task, "results_txt": str(res), "metrics": metrics})
    spec = SPEC[task]
    rec: dict[str, Any] = {
        "task": task,
        "num_labels": spec["num_labels"],
        "batch_size": spec["batch_size"],
        "epochs": spec["epochs"],
        "metric_for_valid": spec["metric_for_valid"],
        "results_dir": str(root),
        "save_dir": str(save_root) if save_root is not None else None,
        "returncode": 0,
        "predictions": str(pred),
        "results_txt": str(res),
        "num_examples": acc_rec["num_examples"],
        "correct": acc_rec["correct"],
        "accuracy": acc_rec["accuracy"],
        "pred_counts": acc_rec["pred_counts"],
        "metrics_percent": metrics,
        "primary_metric": metric,
        "primary_metric_score": metrics[metric],
    }
    if log_root is not None:
        rec["log_root"] = str(log_root)
    return rec


def build_superglue(target: str, result_roots: dict[str, pathlib.Path], save_roots: dict[str, pathlib.Path] | None = None) -> tuple[dict[str, Any] | None, list[str]]:
    recs = []
    missing = []
    for task in ORDER:
        tr = result_roots[task]
        sr = save_roots.get(task) if save_roots else None
        r = score_task(task, tr, sr)
        if r is None:
            missing.append(task)
        else:
            recs.append(r)
    if missing:
        return None, missing
    vals = [float(r["primary_metric_score"]) for r in recs]
    acc_vals = [float(r["accuracy"]) for r in recs]
    rec = {
        "column": "SuperGLUE",
        "tasks": recs,
        "started_utc": None,
        "finished_utc": now(),
        "superglue_mean_accuracy_only_legacy": sum(acc_vals) / len(acc_vals),
        "superglue_primary_metric_details": [
            {"task": r["task"], "metric": r["primary_metric"], "score": r["primary_metric_score"], "results_txt": r["results_txt"]}
            for r in recs
        ],
        "superglue_mean": sum(vals) / len(vals),
        "superglue_coordinate": "current official primary metrics: f1 for MRPC/QQP, accuracy otherwise",
    }
    return rec, []


def aoa_missing_80m(model_root: str) -> dict[str, Any]:
    return {
        "column": "AoA",
        "required_strict_small_steps": [f"chck_{i}M" for i in range(1, 10)] + [f"chck_{i*10}M" for i in range(1, 11)],
        "model_root": model_root,
        "status": "not_official_missing_checkpoints",
        "aoa_official": None,
        "aoa_raw_correlation": None,
        "aoa_leaderboard_score": 0.0,
        "aoa_for_provisional_overall": 0.0,
        "returncode": 0,
        "interpretation": "80M artifact lacks at least one required strict-small AoA checkpoint (chck_1M..chck_9M, chck_90M, chck_100M); leaderboard arithmetic uses AoA=0.0 for immediate exposure comparison, not a submit-ready AoA.",
    }


def aoa_from_repair(path: pathlib.Path) -> dict[str, Any]:
    d = read_json(path)
    return {
        "column": "AoA",
        "status": "official_aoa_done" if d.get("status") == "AOA_LOCAL_CKPTS_MINCTX_DONE" else d.get("status"),
        "helper_status": d.get("status"),
        "aoa_helper_runner": "experiments/archive/frontier_consolidation/scripts/aoa_local_ckpts_minctx.py",
        "aoa_official": d.get("aoa"),
        "aoa_raw_correlation": d.get("aoa"),
        "aoa_leaderboard_score": (float(d.get("aoa")) * 100.0 if d.get("aoa") is not None else None),
        "aoa_for_provisional_overall": (float(d.get("aoa")) * 100.0 if d.get("aoa") is not None else None),
        "out_json": str(path),
        "surprisal_path": d.get("surprisal_path"),
        "score_path": d.get("score_path"),
        "score_tokenizer_path": d.get("score_tokenizer_path"),
        "num_rows": d.get("num_rows"),
        "num_steps": d.get("num_steps"),
        "step_counts": d.get("step_counts"),
        "row_count_values": d.get("row_count_values"),
        "finite_surprisals": d.get("finite_surprisals"),
        "returncode": 0,
    }


def zero_scores_from_payload(payload: dict[str, Any]) -> dict[str, float]:
    tasks = payload["tasks"]
    scores: dict[str, float] = {}
    for c in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]:
        scores[c] = float(tasks[c]["score"])
    scores["GlobalPIQA"] = (float(tasks["GlobalPIQA_parallel"]["score"]) + float(tasks["GlobalPIQA_nonparallel"]["score"])) / 2.0
    scores["Reading"] = float(tasks["Reading"]["scores"]["Reading"])
    return scores


def endpoint_80m() -> dict[str, Any]:
    merged80 = read_json(USER_ROOT / "experiments/archive/frontier_consolidation/data/scale1p75_mature_merged/adapter128_scale1p75_chck_80M.json")
    base80 = read_json(USER_ROOT / "experiments/archive/representation_and_objectives/data/scale1p75_80M_full_eval/per_target/scale1p75_80M.json")
    target = "scale1p75_80M"
    roots = {t: USER_ROOT / f"experiments/archive/representation_and_objectives/data/scale1p75_80M_full_eval/superglue_results/{target}/{t}" for t in ORDER}
    saves = {t: USER_ROOT / f"experiments/archive/representation_and_objectives/data/scale1p75_80M_full_eval/superglue_models/{target}/{t}" for t in ORDER}
    sg, missing = build_superglue(target, roots, saves)
    payload = {
        "target": target,
        "description": "A02 adapter128 scale1.75 at 80M exposure; zero-shot/Reading from research split eval and SuperGLUE completed/filled under research/research.",
        "family": "scale1p75_residual_adapter_80M",
        "run_dir": merged80["run_dir"],
        "model_root": merged80["model_root"],
        "model_path": merged80["model_path"],
        "endpoint": "chck_80M",
        "created_utc": now(),
        "tasks": {k: v for k, v in merged80["tasks"].items()},
        "source_payloads": {
            "zero_reading": "experiments/archive/frontier_consolidation/data/scale1p75_mature_merged/adapter128_scale1p75_chck_80M.json",
            "partial_superglue": "experiments/archive/representation_and_objectives/data/scale1p75_80M_full_eval/per_target/scale1p75_80M.json",
        },
    }
    if sg is not None:
        payload["tasks"]["SuperGLUE"] = sg
    else:
        partial = base80.get("tasks", {}).get("SuperGLUE", {"column": "SuperGLUE", "tasks": []})
        partial = dict(partial)
        partial["superglue_mean_partial_only"] = partial.pop("superglue_mean", None)
        partial["superglue_mean"] = None
        partial["status"] = "incomplete_superglue_not_for_overall"
        payload["tasks"]["SuperGLUE"] = partial
    payload["tasks"]["AoA"] = aoa_missing_80m(merged80["model_root"])
    payload = patch_payload_official_overall(payload)
    payload["superglue_missing_tasks"] = missing
    payload["sota_threshold"] = 41.80
    payload["over_threshold"] = bool(payload.get("official_overall", {}).get("Overall", -999) > 41.80) if not missing else None
    return payload


def endpoint_100m() -> dict[str, Any]:
    # Choose zero/Reading scores from hardened split payloads.
    base_root = USER_ROOT / "experiments/archive/frontier_consolidation/data/scale1p75_100M_full_eval_hardened"
    endpoint_payload = read_json(base_root / "summary/endpoint_ready.json")
    target = "scale1p75_100M_seed43022"
    tasks: dict[str, Any] = {}
    for kind in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"]:
        p = base_root / "parts" / kind.replace("GlobalPIQA_", "GP_") / "eval" / "per_target" / f"{target}__{kind.replace('GlobalPIQA_', 'GP_')}.json"
        if not p.exists():
            # safe_col in the evaluator maps GlobalPIQA_parallel -> GP_parallel.
            p = base_root / "parts" / kind / "eval" / "per_target" / f"{target}__{kind}.json"
        d = read_json(p)
        # Each part has a single task under d['tasks'] keyed by original column.
        tasks[kind] = d["tasks"][kind]
    roots = {t: base_root / f"parts/SuperGLUE/eval/superglue_results/{target}__SuperGLUE/{t}" for t in ORDER}
    saves = {t: base_root / f"parts/SuperGLUE/eval/superglue_models/{target}__SuperGLUE/{t}" for t in ORDER}
    sg, missing = build_superglue(target, roots, saves)
    if sg is not None:
        tasks["SuperGLUE"] = sg
    else:
        part_p = base_root / f"parts/SuperGLUE/eval/per_target/{target}__SuperGLUE.json"
        partial = read_json(part_p).get("tasks", {}).get("SuperGLUE", {"column": "SuperGLUE", "tasks": []}) if part_p.exists() else {"column": "SuperGLUE", "tasks": []}
        partial = dict(partial)
        partial["superglue_mean_partial_only"] = partial.pop("superglue_mean", None)
        partial["superglue_mean"] = None
        partial["status"] = "incomplete_superglue_not_for_overall"
        tasks["SuperGLUE"] = partial
    tasks["AoA"] = aoa_from_repair(USER_ROOT / "experiments/archive/representation_and_objectives/data/scale1p75_100m_aoa_repair/aoa_local_ckpts_minctx0.json")
    run_dir = endpoint_payload["run_dir"]
    model_root = str(pathlib.Path(run_dir) / "hf_model")
    model_path = str(pathlib.Path(model_root) / endpoint_payload["endpoint"])
    payload = {
        "target": target,
        "description": "A02 adapter128 scale1.75 at exact 100M exposure; zero-shot/Reading from hardened split eval, SuperGLUE completed/filled under research/research, AoA from research writable-cache repair.",
        "family": "scale1p75_residual_adapter_100M_official_ladder",
        "run_dir": run_dir,
        "model_root": model_root,
        "model_path": model_path,
        "endpoint": endpoint_payload["endpoint"],
        "run_summary": endpoint_payload,
        "created_utc": now(),
        "tasks": tasks,
        "source_payloads": {
            "hardened_eval_root": str(base_root),
            "aoa_repair": "experiments/archive/representation_and_objectives/data/scale1p75_100m_aoa_repair/aoa_local_ckpts_minctx0.json",
        },
    }
    payload = patch_payload_official_overall(payload)
    payload["superglue_missing_tasks"] = missing
    payload["sota_threshold"] = 41.80
    payload["over_threshold"] = bool(payload.get("official_overall", {}).get("Overall", -999) > 41.80) if not missing else None
    return payload


def threshold_requirements(payload: dict[str, Any]) -> dict[str, Any]:
    oo = payload.get("official_overall", {})
    scores = oo.get("scores", {})
    fixed = [v for k, v in scores.items() if k != "SuperGLUE" and v is not None]
    required_sg = 9 * 41.80 - sum(float(x) for x in fixed) if len(fixed) == 8 else None
    return {"required_SuperGLUE_to_exceed_41p80_given_other_columns": required_sg,
            "current_SuperGLUE": scores.get("SuperGLUE"),
            "missing_columns": [k for k, v in scores.items() if v is None]}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    p80 = endpoint_80m()
    p100 = endpoint_100m()
    p80["threshold_arithmetic"] = threshold_requirements(p80)
    p100["threshold_arithmetic"] = threshold_requirements(p100)
    write80 = OUT_DIR / "scale1p75_80M_collated_score.json"
    write100 = OUT_DIR / "scale1p75_100M_collated_score.json"
    summary = OUT_DIR / "scale1p75_score_summary.json"
    write80.write_text(json.dumps(p80, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write100.write_text(json.dumps(p100, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    s = {
        "status": "SCALE1P75_SCORE_COLLATION",
        "created_utc": now(),
        "endpoints": {
            "80M": {"path": str(write80), "official_overall": p80.get("official_overall"), "superglue_missing_tasks": p80.get("superglue_missing_tasks"), "threshold_arithmetic": p80["threshold_arithmetic"]},
            "100M": {"path": str(write100), "official_overall": p100.get("official_overall"), "superglue_missing_tasks": p100.get("superglue_missing_tasks"), "threshold_arithmetic": p100["threshold_arithmetic"]},
        },
        "superglue_primary_metric": PRIMARY,
    }
    summary.write_text(json.dumps(s, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research scale1.75 score establishment\n",
        "This note is a research score ledger, not a final deliverable. SuperGLUE uses accuracy except MRPC/QQP use F1.\n",
        f"Summary JSON: `{summary}`\n",
        "\n## Current endpoint arithmetic\n",
    ]
    for name, p in [("80M", p80), ("100M", p100)]:
        oo = p.get("official_overall", {})
        lines.append(f"- {name}: missing SuperGLUE tasks = {p.get('superglue_missing_tasks')}; Overall = {oo.get('Overall')}; scores = {oo.get('scores')}; required SuperGLUE for 41.80 = {p['threshold_arithmetic']['required_SuperGLUE_to_exceed_41p80_given_other_columns']}\n")
    NOTE.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": s["status"], "summary": str(summary), "80M_missing": p80.get("superglue_missing_tasks"), "100M_missing": p100.get("superglue_missing_tasks"), "80M_overall": p80.get("official_overall", {}).get("Overall"), "100M_overall": p100.get("official_overall", {}).get("Overall")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
