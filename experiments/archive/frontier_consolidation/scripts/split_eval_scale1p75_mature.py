#!/usr/bin/env python3
"""research split parallel official-compatible cheap-column evaluation for scale1.75 70/80M.

Each endpoint/column is an independent job with a unique target and output root, so
multiple columns can run concurrently across both H100s without per-target JSON or
HF-cache races. The merger reads the per-column payloads and builds endpoint-level
payloads compatible with the research item-flip comparator.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any

USER_ROOT = _public_path('.')
WORKSPACE = _public_path('experiments/archive/frontier_consolidation')
EVAL = _public_path('experiments/archive/frontier_consolidation/scripts/evaluate_compliant_endpoint.py')
RUN_DIR = "experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M80M_seed43022"
OUT_BASE = _public_path('experiments/archive/frontier_consolidation/data/scale1p75_split_eval')
MERGED_DIR = _public_path('experiments/archive/frontier_consolidation/data/scale1p75_mature_merged')
COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"]
CHEAP = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
SERIAL = _public_path('experiments/archive/frontier_consolidation/data/scale1p75_70_80_eval')
REFS = _public_path('experiments/archive/frontier_consolidation/data/legal_mature_treatment_effect_eval/per_target')


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_scores(payload: dict[str, Any]) -> dict[str, float | None]:
    tasks = payload.get("tasks", {})
    out: dict[str, float | None] = {}
    for c in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]:
        r = tasks.get(c, {})
        out[c] = float(r["score"]) if isinstance(r, dict) and r.get("score") is not None else None
    gp = []
    for c in ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]:
        r = tasks.get(c, {})
        if isinstance(r, dict) and r.get("score") is not None:
            gp.append(float(r["score"]))
    out["GlobalPIQA"] = float(mean(gp)) if len(gp) == 2 else None
    r = tasks.get("Reading", {})
    if isinstance(r, dict) and isinstance(r.get("scores"), dict) and r["scores"].get("Reading") is not None:
        out["Reading"] = float(r["scores"]["Reading"])
    elif isinstance(r, dict) and r.get("score") is not None:
        out["Reading"] = float(r["score"])
    else:
        out["Reading"] = None
    return out


def cheap7(scores: dict[str, float | None]) -> float | None:
    vals = [scores.get(c) for c in CHEAP]
    if any(v is None for v in vals):
        return None
    return float(mean(float(v) for v in vals))


def endpoint_of(s: str) -> str:
    if s.startswith("chck_"):
        return s
    return f"chck_{s}"


def single_target(ep: str, col: str) -> str:
    safe = col.replace("GlobalPIQA_", "GP_")
    return f"scale1p75_{ep}_{safe}"


def output_for(ep: str, col: str) -> Path:
    return OUT_BASE / ep / col


def part_payload_path(ep: str, col: str) -> Path:
    target = single_target(ep, col)
    return output_for(ep, col) / "eval" / "per_target" / f"{target}.json"


def task_complete(ep: str, col: str) -> bool:
    # A column may already exist either as a split one-column payload or inside the
    # research/research endpoint-level serial payload. Reuse both to avoid wasting GPU.
    if part_task_source(ep, col) is not None:
        return True
    if serial_task_source(ep, col) is not None:
        return True
    return False


def run_job(ep: str, col: str, gpu: int, force: bool) -> dict[str, Any]:
    out_base = output_for(ep, col)
    target = single_target(ep, col)
    out_base.mkdir(parents=True, exist_ok=True)
    if (not force) and task_complete(ep, col):
        return {"endpoint": ep, "column": col, "target": target, "gpu": gpu, "status": "skip_existing", "payload": str(part_payload_path(ep, col).relative_to(USER_ROOT))}
    cmd = [
        sys.executable, "-B", str(EVAL),
        "--arm", "reinvest",
        "--run-dir", RUN_DIR,
        "--target", target,
        "--endpoint", ep,
        "--out-root", str(out_base / "eval"),
        "--collate-root", str(out_base / "collate"),
        "--gpu", str(gpu),
        "--columns", col,
        "--force",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    log_dir = out_base / "driver_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(json.dumps({"event": "split_eval_start", "endpoint": ep, "column": col, "gpu": gpu, "target": target, "utc": now()}), flush=True)
    proc = subprocess.run(cmd, cwd=str(USER_ROOT), env=env, capture_output=True, text=True, timeout=7200)
    elapsed = time.time() - t0
    (log_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
    (log_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8")
    rec: dict[str, Any] = {"endpoint": ep, "column": col, "target": target, "gpu": gpu, "returncode": proc.returncode, "elapsed_sec": round(elapsed, 3), "stdout_tail": proc.stdout[-2000:], "stderr_tail": proc.stderr[-3000:]}
    if proc.returncode == 0:
        rec["payload"] = str(part_payload_path(ep, col).relative_to(USER_ROOT))
    print(json.dumps({"event": "split_eval_done", **rec}, ensure_ascii=False), flush=True)
    if proc.returncode != 0:
        raise RuntimeError(json.dumps(rec, ensure_ascii=False))
    return rec


def serial_task_source(ep: str, col: str) -> dict[str, Any] | None:
    p = SERIAL / ep / "eval" / "per_target" / f"adapter128_scale1p75_{ep}.json"
    if not p.exists():
        return None
    d = load_json(p)
    r = d.get("tasks", {}).get(col)
    if isinstance(r, dict) and r.get("returncode") == 0:
        return {"task": r, "source_payload": str(p.relative_to(USER_ROOT)), "source": "serial_or_repaired_payload"}
    return None


def part_task_source(ep: str, col: str) -> dict[str, Any] | None:
    p = part_payload_path(ep, col)
    if not p.exists():
        return None
    d = load_json(p)
    r = d.get("tasks", {}).get(col)
    if isinstance(r, dict) and r.get("returncode") == 0:
        return {"task": r, "source_payload": str(p.relative_to(USER_ROOT)), "source": "split_payload"}
    return None


def merge_endpoint(ep: str) -> dict[str, Any]:
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    target = f"adapter128_scale1p75_{ep}"
    tasks: dict[str, Any] = {}
    sources: dict[str, Any] = {}
    for col in COLUMNS:
        src = part_task_source(ep, col) or serial_task_source(ep, col)
        if src:
            tasks[col] = src["task"]
            sources[col] = {k: v for k, v in src.items() if k != "task"}
    payload = {
        "target": target,
        "description": "adapter128 scale1.75 fixed-scale residual adapter on legal research compact-view reinvest substrate; endpoint-level payload merged from independent per-column official-compatible jobs",
        "family": "scale1p75_residual_adapter_mature_split_eval",
        "run_dir": RUN_DIR,
        "model_root": f"{RUN_DIR}/hf_model",
        "model_path": f"{RUN_DIR}/hf_model/{ep}",
        "endpoint": ep,
        "started_utc": now(),
        "tasks": tasks,
        "merge_sources": sources,
    }
    scores = extract_scores(payload)
    payload["official_overall"] = {"scores": {"BLiMP": scores.get("BLiMP"), "Supplement": scores.get("Supplement"), "EWoK": scores.get("EWoK"), "Entity": scores.get("Entity"), "COMPS": scores.get("COMPS"), "SuperGLUE": None, "GlobalPIQA": scores.get("GlobalPIQA"), "Reading": scores.get("Reading"), "AoA": None}, "complete_for_provisional_overall": False}
    payload["cheap7"] = cheap7(scores)
    payload["cheap7_scores"] = scores
    payload["missing_columns"] = [c for c in COLUMNS if c not in tasks]
    out = MERGED_DIR / f"{target}.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"endpoint": ep, "payload": str(out.relative_to(USER_ROOT)), "cheap7": payload["cheap7"], "scores": scores, "missing_columns": payload["missing_columns"]}


def ref_scores(exposure: str) -> dict[str, Any] | None:
    p = REFS / f"complianttok_reinvest_seed43022_{exposure}.json"
    if not p.exists():
        return None
    d = load_json(p)
    scores = d.get("official_overall", {}).get("scores", {})
    out = {c: scores.get(c) for c in CHEAP}
    if any(v is None for v in out.values()):
        return None
    return {"payload": str(p.relative_to(USER_ROOT)), "scores": out, "cheap7": float(mean(float(out[c]) for c in CHEAP))}


def write_summary(merge_rows: list[dict[str, Any]], job_rows: list[dict[str, Any]], endpoints: list[str]) -> dict[str, Any]:
    comparisons = {}
    for m in merge_rows:
        exp = m["endpoint"].replace("chck_", "")
        ref = ref_scores(exp)
        if ref and m.get("cheap7") is not None:
            deltas = {c: float(m["scores"][c]) - float(ref["scores"][c]) for c in CHEAP if m["scores"].get(c) is not None and ref["scores"].get(c) is not None}
            comparisons[exp] = {"ref": ref, "adapter": m, "cheap7_delta": float(m["cheap7"] - ref["cheap7"]), "column_deltas": deltas}
    # Route criterion: do not continue on aggregate gain if the known losing families remain traded.
    route_signal = "pending_item_family_readout"
    interpretation = []
    if comparisons:
        for exp, comp in comparisons.items():
            interpretation.append(f"{exp}: cheap7 Δ {comp['cheap7_delta']:+.4f}, columns {comp['column_deltas']}")
    out = {"status": "SCALE1P75_SPLIT_EVAL", "created_utc": now(), "endpoints": endpoints, "jobs": job_rows, "merged": merge_rows, "comparisons_vs_step35": comparisons, "route_signal": route_signal, "interpretation": interpretation}
    out_json = _public_path('experiments/archive/frontier_consolidation/data/scale1p75_mature_merged/scale1p75_split_eval_summary.json')
    out_md = _public_path('research/documents/frontier_consolidation/data/scale1p75_mature_merged/scale1p75_split_eval_summary.md')
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research scale1.75 split mature evaluation", "", "## Scores", "", "| exposure | arm | cheap7 | BLiMP | Supp | EWoK | Entity | COMPS | GP | Reading |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for ep in endpoints:
        exp = ep.replace("chck_", "")
        ref = ref_scores(exp)
        if ref:
            sc = ref["scores"]
            lines.append(f"| {exp} | research | {ref['cheap7']:.4f} | {sc['BLiMP']:.3f} | {sc['Supplement']:.3f} | {sc['EWoK']:.3f} | {sc['Entity']:.3f} | {sc['COMPS']:.3f} | {sc['GlobalPIQA']:.3f} | {sc['Reading']:.3f} |")
        mr = next((m for m in merge_rows if m["endpoint"] == ep), None)
        if mr and mr.get("cheap7") is not None:
            sc = mr["scores"]
            lines.append(f"| {exp} | scale1.75 | {mr['cheap7']:.4f} | {sc['BLiMP']:.3f} | {sc['Supplement']:.3f} | {sc['EWoK']:.3f} | {sc['Entity']:.3f} | {sc['COMPS']:.3f} | {sc['GlobalPIQA']:.3f} | {sc['Reading']:.3f} |")
    lines += ["", "## Deltas", "", "| exposure | cheap7 Δ | BLiMP | Supp | EWoK | Entity | COMPS | GP | Reading |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for exp, comp in comparisons.items():
        cd = comp["column_deltas"]
        lines.append(f"| {exp} | {comp['cheap7_delta']:+.4f} | {cd.get('BLiMP', float('nan')):+.3f} | {cd.get('Supplement', float('nan')):+.3f} | {cd.get('EWoK', float('nan')):+.3f} | {cd.get('Entity', float('nan')):+.3f} | {cd.get('COMPS', float('nan')):+.3f} | {cd.get('GlobalPIQA', float('nan')):+.3f} | {cd.get('Reading', float('nan')):+.3f} |")
    lines += ["", "## Interpretation", ""]
    for s in interpretation:
        lines.append(f"- {s}")
    lines.append("- Next required readout: pairwise item-flip family analysis at 70M/80M. Only launch 100M if losing families recover while gains persist; otherwise fixed scale1.75 is localized redistribution.")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "out_json": str(out_json.relative_to(USER_ROOT)), "out_md": str(out_md.relative_to(USER_ROOT)), "comparisons": comparisons}, indent=2, ensure_ascii=False), flush=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--endpoints", nargs="*", default=["chck_70M", "chck_80M"])
    p.add_argument("--columns", nargs="*", default=COLUMNS)
    p.add_argument("--gpus", nargs="*", type=int, default=[0, 1])
    p.add_argument("--slots-per-gpu", type=int, default=3)
    p.add_argument("--force", action="store_true")
    p.add_argument("--merge-only", action="store_true")
    args = p.parse_args()
    endpoints = [endpoint_of(x) for x in args.endpoints]
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(ep, col) for ep in endpoints for col in args.columns if not ((not args.force) and task_complete(ep, col))]
    job_rows: list[dict[str, Any]] = []
    if args.merge_only:
        jobs = []
    print(json.dumps({"event": "split_eval_plan", "endpoints": endpoints, "columns": args.columns, "jobs_to_run": len(jobs), "gpus": args.gpus, "slots_per_gpu": args.slots_per_gpu, "utc": now()}), flush=True)
    if jobs:
        gpu_slots = []
        for gpu in args.gpus:
            gpu_slots.extend([gpu] * args.slots_per_gpu)
        with ThreadPoolExecutor(max_workers=len(gpu_slots)) as ex:
            futs = []
            for i, (ep, col) in enumerate(jobs):
                gpu = gpu_slots[i % len(gpu_slots)]
                futs.append(ex.submit(run_job, ep, col, gpu, args.force))
            for fut in as_completed(futs):
                job_rows.append(fut.result())
    # Include already-existing split jobs in the job record.
    for ep in endpoints:
        for col in args.columns:
            if task_complete(ep, col) and not any(r.get("endpoint") == ep and r.get("column") == col for r in job_rows):
                job_rows.append({"endpoint": ep, "column": col, "target": single_target(ep, col), "status": "existing_after_run", "payload": str(part_payload_path(ep, col).relative_to(USER_ROOT))})
    merge_rows = [merge_endpoint(ep) for ep in endpoints]
    write_summary(merge_rows, job_rows, endpoints)


if __name__ == "__main__":
    main()
