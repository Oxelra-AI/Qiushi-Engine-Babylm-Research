#!/usr/bin/env python3
"""research: exact-hash audit and optional merge plan for old 70M/80M reference eval payloads.

The selected DeBERTa common-grid evaluator can reuse a cached per_target payload if the
payload is a complete official-compatible cheap7 record for the exact final-ladder
checkpoint. This script checks only the old 70M/80M scale1.75 reference evaluations
against final-ladder model SHA256. It does not copy anything.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
from statistics import mean
from typing import Any

STUDY = pathlib.Path("experiments/archive/frontier_consolidation")
DATA = STUDY / "data"
REF = STUDY / "training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder"
OLD = STUDY / "training/runs/adapter128_scale1p75_h100M80M_seed43022"
OUT = DATA / "reference_cache_manual_audit"
CHEAP_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
ZERO_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]
SPLIT_ROOT = DATA / "scale1p75_split_eval"
MERGED_70 = DATA / "scale1p75_70_80_eval/chck_70M/eval/per_target/adapter128_scale1p75_chck_70M.json"


def sha_file(path: pathlib.Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: pathlib.Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"_error": repr(e)}


def split_paths(ep: str) -> dict[str, pathlib.Path]:
    return {
        "BLiMP": SPLIT_ROOT / ep / "BLiMP/eval/per_target" / f"scale1p75_{ep}_BLiMP.json",
        "Supplement": SPLIT_ROOT / ep / "Supplement/eval/per_target" / f"scale1p75_{ep}_Supplement.json",
        "EWoK": SPLIT_ROOT / ep / "EWoK/eval/per_target" / f"scale1p75_{ep}_EWoK.json",
        "Entity": SPLIT_ROOT / ep / "Entity/eval/per_target" / f"scale1p75_{ep}_Entity.json",
        "COMPS": SPLIT_ROOT / ep / "COMPS/eval/per_target" / f"scale1p75_{ep}_COMPS.json",
        "GlobalPIQA_parallel": SPLIT_ROOT / ep / "GlobalPIQA_parallel/eval/per_target" / f"scale1p75_{ep}_GP_parallel.json",
        "GlobalPIQA_nonparallel": SPLIT_ROOT / ep / "GlobalPIQA_nonparallel/eval/per_target" / f"scale1p75_{ep}_GP_nonparallel.json",
        "Reading": SPLIT_ROOT / ep / "Reading/eval/per_target" / f"scale1p75_{ep}_Reading.json",
    }


def collect_tasks_from_payloads(ep: str) -> tuple[dict[str, Any], dict[str, Any]]:
    tasks: dict[str, Any] = {}
    sources: dict[str, Any] = {}
    # Prefer the old merged 70M payload if complete; split paths still override missing/duplicate safely.
    candidates: dict[str, pathlib.Path] = {}
    if ep == "chck_70M" and MERGED_70.exists():
        candidates["merged70"] = MERGED_70
    candidates.update(split_paths(ep))
    for label, path in candidates.items():
        payload = read_json(path)
        if not payload:
            sources[label] = {"path": str(path), "exists": False}
            continue
        if payload.get("_error"):
            sources[label] = {"path": str(path), "exists": True, "error": payload.get("_error")}
            continue
        for task_name, rec in payload.get("tasks", {}).items():
            tasks[task_name] = rec
            sources[task_name] = {"path": str(path), "returncode": rec.get("returncode"), "score": rec.get("score"), "model_path": payload.get("model_path"), "run_dir": payload.get("run_dir")}
    return tasks, sources


def extract_scores(tasks: dict[str, Any]) -> dict[str, float | None]:
    scores: dict[str, float | None] = {}
    for c in ZERO_COLUMNS:
        rec = tasks.get(c, {})
        scores[c] = float(rec.get("score")) if rec.get("score") is not None else None
    gp = []
    for c in ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]:
        rec = tasks.get(c, {})
        if rec.get("score") is not None:
            gp.append(float(rec["score"]))
    scores["GlobalPIQA"] = mean(gp) if len(gp) == 2 else None
    rd = tasks.get("Reading", {})
    scores["Reading"] = float(rd.get("score")) if rd.get("score") is not None else None
    return scores


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    endpoints = ["chck_70M", "chck_80M"]
    result: dict[str, Any] = {"status": "REFERENCE_CACHE_MANUAL_AUDIT", "endpoints": {}, "selected_wrapper_cache_destination_root": str(DATA / "selected_trajectory_eval_scale1p75_seed43022_reference_common2M/eval/per_target")}
    for ep in endpoints:
        ref_model = REF / "hf_model" / ep / "model.safetensors"
        old_model = OLD / "hf_model" / ep / "model.safetensors"
        ref_sha = sha_file(ref_model)
        old_sha = sha_file(old_model)
        tasks, sources = collect_tasks_from_payloads(ep)
        scores = extract_scores(tasks)
        complete_tasks = all(k in tasks and tasks[k].get("returncode") == 0 and tasks[k].get("score") is not None for k in ZERO_COLUMNS + ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"])
        complete_scores = all(scores.get(c) is not None for c in CHEAP_COLUMNS)
        cheap7 = mean([float(scores[c]) for c in CHEAP_COLUMNS]) if complete_scores else None
        safe = bool(ref_sha and old_sha and ref_sha == old_sha and complete_tasks and complete_scores)
        result["endpoints"][ep] = {
            "reference_model_path": str(ref_model),
            "old_model_path": str(old_model),
            "reference_sha256": ref_sha,
            "old_sha256": old_sha,
            "hash_match": ref_sha == old_sha if ref_sha and old_sha else False,
            "tasks_present": sorted(tasks),
            "complete_tasks_returncode0": complete_tasks,
            "scores": scores,
            "cheap7": cheap7,
            "sources": sources,
            "safe_to_seed_selected_cache": safe,
            "target_cache_path": str(DATA / "selected_trajectory_eval_scale1p75_seed43022_reference_common2M/eval/per_target" / f"scale1p75_seed43022_reference_{ep}.json"),
        }
    out_json = OUT / "reference_cache_manual_audit.json"
    out_md = (OUT.parents[4] / 'research/documents/frontier_consolidation/data/reference_cache_manual_audit/reference_cache_manual_audit.md')
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md = ["# research reference cache manual audit\n\n"]
    for ep, rec in result["endpoints"].items():
        md.append(f"## {ep}\n\n")
        md.append(f"Hash match old standalone vs final ladder: {rec['hash_match']}\n\n")
        md.append(f"Complete official-compatible split tasks: {rec['complete_tasks_returncode0']}; safe to seed: {rec['safe_to_seed_selected_cache']}\n\n")
        md.append(f"Cheap7 if merged: {rec['cheap7']} with scores {rec['scores']}\n\n")
    md.append("No cache files were written by this audit. Seed only if safe_to_seed_selected_cache is true.\n\n")
    md.append(f"JSON: `{out_json}`\n")
    out_md.write_text("".join(md), encoding="utf-8")
    print(json.dumps({"status": result["status"], "out_json": str(out_json), "out_md": str(out_md)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
