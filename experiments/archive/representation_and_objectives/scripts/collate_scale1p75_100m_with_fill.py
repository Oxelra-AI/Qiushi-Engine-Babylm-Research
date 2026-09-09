#!/usr/bin/env python3
"""research: collate scale1.75 100M score using split SuperGLUE task roots.

This is a read-only collation over completed evaluation outputs. It merges the
already-present hardened SuperGLUE subtasks with local MNLI/QQP
fills. It refuses to compute an Overall if any SuperGLUE subtask is missing or
if an MRPC/QQP F1 metric is absent.
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
from babylm_official_scoring import patch_payload_official_overall  # noqa: E402

OUT_DIR = USER_ROOT / "experiments/archive/representation_and_objectives/data/scale1p75_100m_score_collation"
A02_ROOT = USER_ROOT / "experiments/archive/frontier_consolidation/data/scale1p75_100M_full_eval_hardened"
FILL_ROOT = USER_ROOT / "experiments/archive/representation_and_objectives/data/100M_superglue_fill"
AOA_REPAIR = USER_ROOT / "experiments/archive/representation_and_objectives/data/scale1p75_100m_aoa_repair/aoa_local_ckpts_minctx0.json"
TARGET = "scale1p75_100M_seed43022"
SG_TARGET = TARGET + "__SuperGLUE"
PRIMARY = {"boolq": "accuracy", "mnli": "accuracy", "mrpc": "f1", "multirc": "accuracy", "qqp": "f1", "rte": "accuracy", "wsc": "accuracy"}
ORDER = [s["task"] for s in base.SUPERGLUE_TASKS]
SPEC = {s["task"]: s for s in base.SUPERGLUE_TASKS}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.resolve().relative_to(USER_ROOT))
    except Exception:
        return str(p)


def read_json(p: pathlib.Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def latest_file(root: pathlib.Path, name: str) -> pathlib.Path | None:
    hits = sorted(root.rglob(name), key=lambda p: (p.stat().st_mtime, str(p))) if root.exists() else []
    return hits[-1] if hits else None


def parse_results_txt(path: pathlib.Path) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        k, sep, v = line.partition(":")
        if not sep:
            continue
        try:
            out[k.strip()] = float(v.strip()) * 100.0
        except Exception:
            pass
    return out


def candidate_roots(task: str) -> list[pathlib.Path]:
    roots: list[pathlib.Path] = []
    # Prefer repaired task outputs; use the hardened tree for already completed tasks.
    roots.append(FILL_ROOT / "superglue_results" / SG_TARGET / task)
    roots.append(A02_ROOT / "parts/SuperGLUE/eval/superglue_results" / SG_TARGET / task)
    return roots


def score_task(task: str) -> dict[str, Any] | None:
    selected_root = None
    pred = None
    res = None
    for root in candidate_roots(task):
        p = latest_file(root, "predictions.json")
        r = latest_file(root, "results.txt")
        if p is not None and r is not None:
            selected_root, pred, res = root, p, r
            break
    if pred is None or res is None or selected_root is None:
        return None
    acc_rec = base.score_superglue_predictions(task, pred)
    metrics = parse_results_txt(res)
    metric = PRIMARY[task]
    if metric not in metrics:
        raise RuntimeError({"task": task, "missing_primary_metric": metric, "results_txt": rel(res), "metrics": metrics})
    spec = SPEC[task]
    return {
        "task": task,
        "selected_root": rel(selected_root),
        "predictions": rel(pred),
        "results_txt": rel(res),
        "num_labels": spec["num_labels"],
        "batch_size": spec["batch_size"],
        "epochs": spec["epochs"],
        "metric_for_valid": spec["metric_for_valid"],
        "num_examples": acc_rec["num_examples"],
        "correct": acc_rec["correct"],
        "accuracy": acc_rec["accuracy"],
        "pred_counts": acc_rec["pred_counts"],
        "metrics_percent": metrics,
        "primary_metric": metric,
        "primary_metric_score": metrics[metric],
    }


def build_superglue() -> tuple[dict[str, Any] | None, list[str]]:
    recs = []
    missing = []
    for task in ORDER:
        rec = score_task(task)
        if rec is None:
            missing.append(task)
        else:
            recs.append(rec)
    if missing:
        return None, missing
    primary_vals = [float(r["primary_metric_score"]) for r in recs]
    acc_vals = [float(r["accuracy"]) for r in recs]
    return {
        "column": "SuperGLUE",
        "tasks": recs,
        "finished_utc": now(),
        "superglue_mean_accuracy_only_legacy": sum(acc_vals) / len(acc_vals),
        "superglue_primary_metric_details": [
            {"task": r["task"], "metric": r["primary_metric"], "score": r["primary_metric_score"], "results_txt": r["results_txt"]}
            for r in recs
        ],
        "superglue_mean": sum(primary_vals) / len(primary_vals),
        "superglue_coordinate": "current official primary metrics: f1 for MRPC/QQP, accuracy otherwise",
        "source_policy": "A01 research fill roots preferred where complete, otherwise A02 hardened roots",
    }, []


def aoa_task() -> dict[str, Any]:
    d = read_json(AOA_REPAIR)
    return {
        "column": "AoA",
        "status": "official_aoa_done" if d.get("status") == "AOA_LOCAL_CKPTS_MINCTX_DONE" else d.get("status"),
        "helper_status": d.get("status"),
        "aoa_official": d.get("aoa"),
        "aoa_raw_correlation": d.get("aoa"),
        "aoa_leaderboard_score": float(d.get("aoa")) * 100.0 if d.get("aoa") is not None else None,
        "aoa_for_overall": float(d.get("aoa")) * 100.0 if d.get("aoa") is not None else None,
        "out_json": rel(AOA_REPAIR),
        "num_rows": d.get("num_rows"),
        "num_steps": d.get("num_steps"),
        "row_count_values": d.get("row_count_values"),
        "finite_surprisals": d.get("finite_surprisals"),
    }


def zero_task(kind: str) -> dict[str, Any]:
    safe = kind.replace("GlobalPIQA_", "GP_")
    candidates = [
        A02_ROOT / "parts" / safe / "eval/per_target" / f"{TARGET}__{safe}.json",
        A02_ROOT / "parts" / kind / "eval/per_target" / f"{TARGET}__{kind}.json",
    ]
    for p in candidates:
        if p.exists():
            d = read_json(p)
            return d["tasks"][kind]
    raise FileNotFoundError({"kind": kind, "candidates": [rel(p) for p in candidates]})


def threshold(scores: dict[str, Any]) -> dict[str, Any]:
    fixed = [float(v) for k, v in scores.items() if k != "SuperGLUE" and v is not None]
    req = 9 * 41.80 - sum(fixed) if len(fixed) == 8 else None
    return {
        "required_SuperGLUE_to_reach_41p80_given_other_columns": req,
        "current_SuperGLUE": scores.get("SuperGLUE"),
        "missing_columns": [k for k, v in scores.items() if v is None],
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sg, missing = build_superglue()
    endpoint = read_json(A02_ROOT / "summary/endpoint_ready.json")
    tasks: dict[str, Any] = {}
    for kind in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"]:
        tasks[kind] = zero_task(kind)
    if sg is not None:
        tasks["SuperGLUE"] = sg
    else:
        tasks["SuperGLUE"] = {"column": "SuperGLUE", "status": "incomplete_superglue_not_for_overall", "tasks": [score_task(t) for t in ORDER if score_task(t) is not None], "superglue_mean": None}
    tasks["AoA"] = aoa_task()
    run_dir = pathlib.Path(endpoint["run_dir"])
    payload = {
        "target": TARGET,
        "created_utc": now(),
        "description": "scale1.75 100M exact endpoint; zero-shot/Reading from A02 hardened split eval, AoA from research repair, SuperGLUE merged from A02 and research fills.",
        "family": "scale1p75_residual_adapter_100M_official_ladder",
        "run_dir": str(run_dir),
        "model_root": str(run_dir / "hf_model"),
        "model_path": str(run_dir / "hf_model" / endpoint["endpoint"]),
        "endpoint": endpoint["endpoint"],
        "run_summary": endpoint,
        "tasks": tasks,
        "source_payloads": {
            "a02_hardened_eval_root": rel(A02_ROOT),
            "superglue_fill_root": rel(FILL_ROOT),
            "aoa_repair": rel(AOA_REPAIR),
        },
        "superglue_missing_tasks": missing,
        "sota_threshold": 41.80,
    }
    if not missing:
        payload = patch_payload_official_overall(payload)
        payload["threshold_arithmetic"] = threshold(payload["official_overall"]["scores"])
        payload["over_threshold"] = bool(payload["official_overall"].get("Overall", -999) > 41.80)
    else:
        payload["official_overall"] = None
        # Compute fixed-column required SuperGLUE from non-SuperGLUE scores already known.
        payload = patch_payload_official_overall(payload)
        payload["threshold_arithmetic"] = threshold(payload.get("official_overall", {}).get("scores", {}))
        payload["official_overall"] = None
        payload["over_threshold"] = None
    out = OUT_DIR / "scale1p75_100M_collated_score_with_fill.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {
        "status": "SCALE1P75_100M_COLLATION",
        "path": rel(out),
        "missing_superglue_tasks": missing,
        "official_overall": payload.get("official_overall"),
        "threshold_arithmetic": payload.get("threshold_arithmetic"),
        "over_threshold": payload.get("over_threshold"),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
