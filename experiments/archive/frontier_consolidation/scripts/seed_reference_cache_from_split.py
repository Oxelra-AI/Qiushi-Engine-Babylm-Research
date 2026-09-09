#!/usr/bin/env python3
"""research: seed protected-reference selected-eval cache from exact-hash split payloads.

Only chck_70M and chck_80M have prior split-column official-compatible evaluations.
This script merges those split payloads into the cache format expected by
selected_mlm_checkpoint_eval.py, but only if every source payload's model_path
SHA256 exactly matches the final protected scale1.75 reference ladder checkpoint.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
from statistics import mean
from typing import Any

STUDY = pathlib.Path("experiments/archive/frontier_consolidation")
DATA = STUDY / "data"
REF_RUN = STUDY / "training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder"
SPLIT_ROOT = DATA / "scale1p75_split_eval"
CACHE_ROOT = DATA / "selected_trajectory_eval_scale1p75_seed43022_reference_common2M/eval/per_target"
OUT = DATA / "reference_split_cache_seed"
ENDPOINTS = ["chck_70M", "chck_80M"]
SPLIT_TASKS = {
    "BLiMP": ("BLiMP", "scale1p75_{ep}_BLiMP.json"),
    "Supplement": ("Supplement", "scale1p75_{ep}_Supplement.json"),
    "EWoK": ("EWoK", "scale1p75_{ep}_EWoK.json"),
    "Entity": ("Entity", "scale1p75_{ep}_Entity.json"),
    "COMPS": ("COMPS", "scale1p75_{ep}_COMPS.json"),
    "GlobalPIQA_parallel": ("GlobalPIQA_parallel", "scale1p75_{ep}_GP_parallel.json"),
    "GlobalPIQA_nonparallel": ("GlobalPIQA_nonparallel", "scale1p75_{ep}_GP_nonparallel.json"),
    "Reading": ("Reading", "scale1p75_{ep}_Reading.json"),
}
CHEAP_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]


def sha_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def split_path(ep: str, task_name: str) -> pathlib.Path:
    folder, filename = SPLIT_TASKS[task_name]
    return SPLIT_ROOT / ep / folder / "eval/per_target" / filename.format(ep=ep)


def read_payload(path: pathlib.Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def score_from_task(task_name: str, rec: dict[str, Any]) -> float | None:
    if task_name == "Reading":
        scores = rec.get("scores")
        if isinstance(scores, dict) and scores.get("Reading") is not None:
            return float(scores["Reading"])
    if rec.get("score") is not None:
        return float(rec["score"])
    return None


def merge_endpoint(ep: str) -> dict[str, Any]:
    ref_model_path = REF_RUN / "hf_model" / ep / "model.safetensors"
    if not ref_model_path.exists():
        raise FileNotFoundError(ref_model_path)
    ref_sha = sha_file(ref_model_path)
    tasks: dict[str, Any] = {}
    source_records: dict[str, Any] = {}
    failures: list[str] = []
    for task_name in SPLIT_TASKS:
        p = split_path(ep, task_name)
        try:
            payload = read_payload(p)
        except Exception as e:
            failures.append(f"{task_name}: cannot read {p}: {e}")
            continue
        payload_model_path = pathlib.Path(payload.get("model_path", ""))
        payload_weight_path = payload_model_path / "model.safetensors" if payload_model_path.is_dir() else payload_model_path
        payload_sha = sha_file(payload_weight_path) if payload_weight_path.exists() else None
        task_rec = payload.get("tasks", {}).get(task_name)
        if not isinstance(task_rec, dict):
            failures.append(f"{task_name}: task record missing in {p}")
            continue
        score = score_from_task(task_name, task_rec)
        ok = (task_rec.get("returncode") == 0 and score is not None and payload_sha == ref_sha)
        if not ok:
            failures.append(f"{task_name}: returncode={task_rec.get('returncode')} score={score} payload_sha={payload_sha} ref_sha={ref_sha}")
        tasks[task_name] = task_rec
        source_records[task_name] = {
            "payload": str(p),
            "payload_model_path": str(payload_model_path),
            "payload_weight_path": str(payload_weight_path),
            "payload_model_sha256": payload_sha,
            "reference_model_sha256": ref_sha,
            "hash_matches_reference_ladder": payload_sha == ref_sha,
            "returncode": task_rec.get("returncode"),
            "score_used": score,
        }
    scores: dict[str, float | None] = {
        "BLiMP": score_from_task("BLiMP", tasks.get("BLiMP", {})),
        "Supplement": score_from_task("Supplement", tasks.get("Supplement", {})),
        "EWoK": score_from_task("EWoK", tasks.get("EWoK", {})),
        "Entity": score_from_task("Entity", tasks.get("Entity", {})),
        "COMPS": score_from_task("COMPS", tasks.get("COMPS", {})),
        "Reading": score_from_task("Reading", tasks.get("Reading", {})),
    }
    gp_vals = [score_from_task("GlobalPIQA_parallel", tasks.get("GlobalPIQA_parallel", {})), score_from_task("GlobalPIQA_nonparallel", tasks.get("GlobalPIQA_nonparallel", {}))]
    scores["GlobalPIQA"] = mean([v for v in gp_vals if v is not None]) if all(v is not None for v in gp_vals) else None
    complete = (not failures) and all(scores.get(c) is not None for c in CHEAP_COLUMNS)
    target = f"scale1p75_seed43022_reference_{ep}"
    merged = {
        "target": target,
        "description": "cache-seeded protected-reference selected cheap7 payload merged from prior exact-hash split official-compatible evaluations",
        "family": "scale1p75_reference_common2M_cache_seed",
        "run_dir": str(REF_RUN),
        "model_root": str(REF_RUN / "hf_model"),
        "model_path": str(REF_RUN / "hf_model" / ep),
        "endpoint": ep,
        "tasks": tasks,
        "cache_seeded_from": {
            "created_by": "seed_reference_cache_from_split.py",
            "reference_model_sha256": ref_sha,
            "source_records": source_records,
            "scores": scores,
            "cheap7": mean([float(scores[c]) for c in CHEAP_COLUMNS]) if complete else None,
            "safety_condition": "Every source task returncode=0, score present, and source payload model_path/model.safetensors SHA256 equals final protected reference ladder checkpoint SHA256.",
        },
        "official_overall": {
            "scores": {**scores, "SuperGLUE": None, "AoA": None},
            "complete_for_provisional_overall": False,
            "aoa_status": None,
            "submit_ready_aoa": False,
            "submit_ready_overall": False,
        },
    }
    return {"endpoint": ep, "target": target, "complete": complete, "failures": failures, "scores": scores, "cheap7": merged["cache_seeded_from"]["cheap7"], "payload": merged}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    records = []
    written = []
    for ep in ENDPOINTS:
        rec = merge_endpoint(ep)
        records.append({k: v for k, v in rec.items() if k != "payload"})
        if rec["complete"]:
            out_path = CACHE_ROOT / f"scale1p75_seed43022_reference_{ep}.json"
            out_path.write_text(json.dumps(rec["payload"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            written.append(str(out_path))
    summary = {
        "status": "REFERENCE_SPLIT_CACHE_SEED",
        "cache_root": str(CACHE_ROOT),
        "records": records,
        "written_cache_payloads": written,
        "n_written": len(written),
    }
    out_json = OUT / "reference_split_cache_seed.json"
    out_md = (OUT.parents[4] / 'research/documents/frontier_consolidation/data/reference_split_cache_seed/reference_split_cache_seed.md')
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md = ["# research reference split cache seed\n\n"]
    for rec in records:
        md.append(f"- {rec['endpoint']}: complete={rec['complete']}, cheap7={rec['cheap7']}, written={rec['complete']}\n")
        md.append(f"  scores={rec['scores']}\n")
        if rec["failures"]:
            md.append(f"  failures={rec['failures']}\n")
    md.append(f"\nWritten cache payloads: {len(written)}\n\n")
    md.append(f"JSON: `{out_json}`\n")
    out_md.write_text("".join(md), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "n_written": len(written), "written": written, "out_json": str(out_json), "out_md": str(out_md)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
