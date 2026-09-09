#!/usr/bin/env python3
"""research: stable-family ladder evaluator for the prepared 1.82x dose.

This script is dormant until the research intermediate DeBERTa view/repeat runs
exist.  It mirrors the research stable-family evaluator for the new midpoint arm:
BLiMP, Supplement, EWoK, Entity, COMPS, and Reading across chck_10M..chck_100M.
It writes a compact-minus-repeat semantic leg for 10M..100M and, where the old
clean0 reference exists, view-clean/repeat-clean decompositions for 10M..80M.

No GlobalPIQA, SuperGLUE, AoA, upload, or leaderboard action is performed.
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
import os
import pathlib
import subprocess
import sys
import time
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
WS = USER_ROOT / "experiments/archive/frontier_consolidation"
EVALUATOR = WS / "scripts/evaluate_compliant_endpoint.py"
OUT_DEFAULT = WS / "data/intermediate_dose_ladder_stable_eval"
SEED_LADDER_ROWS = WS / "data/full_deberta_seed_ladder_stable_eval/full_deberta_seed_ladder_rows.csv"
SEED_SPREAD_CSV = WS / "data/full_deberta_seed_ladder_stable_eval/full_deberta_seed_ladder_seed_spread.csv"
STABLE_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "Reading"]
DERIVED_COLUMNS = ["cheap6_no_GlobalPIQA", "cheap5_no_GlobalPIQA_Reading", "EWoK_plus_Entity_sum"]
ALL_SCORE_COLUMNS = [*STABLE_COLUMNS, *DERIVED_COLUMNS]
CKS_10_80 = [f"chck_{i}M" for i in range(10, 81, 10)]
CKS_10_100 = [f"chck_{i}M" for i in range(10, 101, 10)]

ARMS = {
    "dose1p82_view": {
        "dose": 1.8209293539856442,
        "rho": 0.07712,
        "data_arm": "view",
        "arm_arg": "reinvest",
        "run_dir": WS / "training/runs/full_p2c_c2p_abs_view_dose1p82x_matched_rowholdout_deberta100M_seed43022",
        "checkpoints": CKS_10_100,
        "max_words": 100_000_000,
        "expected_actual_updates": 2541,
        "role": "prepared intermediate 1.82x compact-view arm",
    },
    "dose1p82_repeat": {
        "dose": 1.8209293539856442,
        "rho": 0.07712,
        "data_arm": "repeat",
        "arm_arg": "reinvest",
        "run_dir": WS / "training/runs/full_p2c_c2p_abs_repeat_dose1p82x_matched_rowholdout_deberta100M_seed43022",
        "checkpoints": CKS_10_100,
        "max_words": 100_000_000,
        "expected_actual_updates": 2541,
        "role": "prepared intermediate 1.82x hash-rotated repeat arm",
    },
}

CLEAN_EXTERNAL = {
    "chck_20M": WS / "data/legal_mature_clean_control_eval/per_target/complianttok_cleanqwen_seed43022_20M.json",
    "chck_70M": WS / "data/legal_mature_clean_control_eval/per_target/complianttok_cleanqwen_seed43022_70M.json",
    "chck_80M": WS / "data/legal_mature_clean_control_eval/per_target/complianttok_cleanqwen_seed43022_80M.json",
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ck_words(ck: str) -> int:
    return int(ck[5:-1]) * 1_000_000


def finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def smi_rows() -> list[dict[str, int]]:
    p = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
    if p.returncode != 0:
        return []
    rows = []
    for line in p.stdout.splitlines():
        if not line.strip():
            continue
        idx, total, used, free, util = [int(x.strip()) for x in line.split(",")]
        rows.append({"index": idx, "total_mib": total, "used_mib": used, "free_mib": free, "util_gpu_pct": util})
    return rows


def metric_ready(run_dir: pathlib.Path, max_words: int) -> bool:
    p = run_dir / "scientific_metrics.json"
    if not p.exists():
        return False
    try:
        m = read_json(p)
    except Exception:
        return False
    return int(m.get("word_exposure", 0) or 0) >= max_words


def checkpoint_exists(run_dir: pathlib.Path, ck: str) -> bool:
    d = run_dir / "hf_model" / ck
    return (d / "model.safetensors").exists() or (d / "pytorch_model.bin").exists()


def load_external_clean_and_dose1() -> dict[tuple[str, str], pathlib.Path]:
    out: dict[tuple[str, str], pathlib.Path] = {}
    if SEED_LADDER_ROWS.exists():
        with SEED_LADDER_ROWS.open("r", encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                old_arm = r.get("arm")
                ck = str(r.get("checkpoint"))
                p = pathlib.Path(str(r.get("per_target")))
                if not p.is_absolute():
                    p = USER_ROOT / p
                if old_arm == "seed43022_compact":
                    out[("dose1_view", ck)] = p
                elif old_arm == "seed43022_repeat":
                    out[("dose1_repeat", ck)] = p
    for ck, p in CLEAN_EXTERNAL.items():
        out[("clean0", ck)] = p if p.is_absolute() else USER_ROOT / p
    return out


def extract_scores(payload: dict[str, Any]) -> dict[str, float | None]:
    tasks = payload.get("tasks") or {}
    out: dict[str, float | None] = {}
    for col in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]:
        rec = tasks.get(col) or {}
        out[col] = float(rec["score"]) if rec.get("score") is not None else None
    rec = tasks.get("Reading") or {}
    if isinstance(rec.get("scores"), dict) and rec["scores"].get("Reading") is not None:
        out["Reading"] = float(rec["scores"]["Reading"])
    elif rec.get("score") is not None:
        out["Reading"] = float(rec["score"])
    else:
        out["Reading"] = None
    return out


def derived(scores: dict[str, float | None]) -> dict[str, float | None]:
    vals6 = [scores.get(c) for c in STABLE_COLUMNS]
    vals5 = [scores.get(c) for c in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]]
    return {
        "cheap6_no_GlobalPIQA": mean([float(x) for x in vals6]) if all(finite(x) for x in vals6) else None,
        "cheap5_no_GlobalPIQA_Reading": mean([float(x) for x in vals5]) if all(finite(x) for x in vals5) else None,
        "EWoK_plus_Entity_sum": float(scores["EWoK"]) + float(scores["Entity"]) if finite(scores.get("EWoK")) and finite(scores.get("Entity")) else None,
    }


def per_target_ok(path: pathlib.Path) -> tuple[bool, list[str]]:
    problems: list[str] = []
    if not path.exists():
        return False, [f"missing {path}"]
    try:
        payload = read_json(path)
    except Exception as exc:
        return False, [f"json read failed {exc!r}"]
    tasks = payload.get("tasks") or {}
    for col in STABLE_COLUMNS:
        rec = tasks.get(col)
        if not isinstance(rec, dict):
            problems.append(f"missing task {col}")
            continue
        if rec.get("returncode") != 0:
            problems.append(f"{col} returncode {rec.get('returncode')}")
        val = (rec.get("scores") or {}).get("Reading", rec.get("score")) if col == "Reading" else rec.get("score")
        if not finite(val):
            problems.append(f"{col} score nonfinite {val}")
    return not problems, problems


def summarize_payload(path: pathlib.Path, arm_name: str, ck: str, source_type: str) -> dict[str, Any]:
    payload = read_json(path)
    sc = extract_scores(payload)
    de = derived(sc)
    if arm_name in ARMS:
        arm = ARMS[arm_name]
        dose = arm["dose"]
        data_arm = arm["data_arm"]
        run_dir = str(arm["run_dir"])
    elif arm_name == "clean0":
        dose = 0.0; data_arm = "clean"; run_dir = "external_clean0"
    else:
        dose = 1.0; data_arm = "view" if "view" in arm_name else "repeat"; run_dir = "external_dose1"
    return {"arm": arm_name, "dose": dose, "data_arm": data_arm, "checkpoint": ck, "words": ck_words(ck), **sc, **de, "source_type": source_type, "per_target": str(path), "run_dir": run_dir}


def out_per_target(out_dir: pathlib.Path, arm_name: str, ck: str) -> pathlib.Path:
    return out_dir / "eval/per_target" / f"intermediate_{arm_name}_{ck}.json"


def task_out_base(out_dir: pathlib.Path, arm_name: str, ck: str) -> pathlib.Path:
    return out_dir / "runs" / arm_name / ck


def eval_command(out_dir: pathlib.Path, arm_name: str, ck: str, gpu: int, force: bool) -> list[str]:
    arm = ARMS[arm_name]
    cmd = [
        sys.executable, "-B", str(EVALUATOR),
        "--arm", str(arm["arm_arg"]),
        "--target", f"intermediate_{arm_name}_{ck}",
        "--run-dir", str(arm["run_dir"]),
        "--endpoint", ck,
        "--out-root", str(out_dir / "eval"),
        "--collate-root", str(out_dir / "collate"),
        "--gpu", str(gpu),
        "--columns", *STABLE_COLUMNS,
    ]
    if force:
        cmd.append("--force")
    return cmd


def choose_free_gpu(min_free_mib: int, busy: set[int]) -> int | None:
    rows = smi_rows()
    candidates = [r for r in rows if r["index"] not in busy and r["free_mib"] >= min_free_mib]
    if not candidates:
        return None
    candidates.sort(key=lambda r: r["free_mib"], reverse=True)
    return int(candidates[0]["index"])


def launch_eval(out_dir: pathlib.Path, arm_name: str, ck: str, gpu: int, force: bool) -> tuple[subprocess.Popen[str], Any]:
    base = task_out_base(out_dir, arm_name, ck)
    base.mkdir(parents=True, exist_ok=True)
    log = (base / "stable_eval_stdout.log").open("w", encoding="utf-8")
    cmd = eval_command(out_dir, arm_name, ck, gpu, force)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    log.write(json.dumps({"event": "launch_eval", "utc": now(), "arm": arm_name, "checkpoint": ck, "gpu": gpu, "cmd": cmd}) + "\n")
    log.flush()
    p = subprocess.Popen(cmd, cwd=str(USER_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    return p, log


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def build_intermediate_contrasts(rows: list[dict[str, Any]], external: dict[tuple[str, str], pathlib.Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by = {(r["arm"], r["checkpoint"]): r for r in rows}
    # Add clean0 rows from external for decomposition only.
    for (arm, ck), p in external.items():
        if arm == "clean0" and p.exists() and (arm, ck) not in by:
            ok, _ = per_target_ok(p)
            if ok:
                row = summarize_payload(p, arm, ck, "external_clean")
                by[(arm, ck)] = row
    cmr: list[dict[str, Any]] = []
    decomp: list[dict[str, Any]] = []
    for ck in CKS_10_100:
        v = by.get(("dose1p82_view", ck)); r = by.get(("dose1p82_repeat", ck))
        if v is None or r is None:
            continue
        rec = {"contrast": "dose1p82_view_minus_repeat", "dose": ARMS["dose1p82_view"]["dose"], "rho": ARMS["dose1p82_view"]["rho"], "checkpoint": ck, "words": ck_words(ck)}
        for key in ALL_SCORE_COLUMNS:
            rec[key] = float(v[key]) - float(r[key]) if finite(v.get(key)) and finite(r.get(key)) else None
        cmr.append(rec)
    for ck in CKS_10_80:
        c = by.get(("clean0", ck)); v = by.get(("dose1p82_view", ck)); r = by.get(("dose1p82_repeat", ck))
        if c is None or v is None or r is None:
            continue
        for key in ALL_SCORE_COLUMNS:
            val_v = float(v[key]) if finite(v.get(key)) else None
            val_r = float(r[key]) if finite(r.get(key)) else None
            val_c = float(c[key]) if finite(c.get(key)) else None
            if val_v is None or val_r is None or val_c is None:
                total = semantic = source = resid = None
            else:
                total = val_v - val_c
                semantic = val_v - val_r
                source = val_r - val_c
                resid = total - (semantic + source)
            decomp.append({
                "dose_name": "dose1p82", "dose": ARMS["dose1p82_view"]["dose"], "rho": ARMS["dose1p82_view"]["rho"],
                "checkpoint": ck, "words": ck_words(ck), "metric": key,
                "total_view_minus_clean": total, "semantic_view_minus_repeat": semantic, "source_repeat_minus_clean": source, "decomposition_residual": resid,
            })
    return cmr, decomp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--min-free-mib", type=int, default=24_000)
    ap.add_argument("--max-parallel", type=int, default=2)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    if not EVALUATOR.exists():
        raise FileNotFoundError(EVALUATOR)
    out_dir = pathlib.Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = USER_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "intermediate_ladder_driver_log.jsonl"
    external = load_external_clean_and_dose1()
    plan = {
        "status": "INTERMEDIATE_DOSE_LADDER_STABLE_PLAN",
        "created_utc": now(),
        "scientific_purpose": "Evaluate the prepared 1.82x dose only after its DeBERTa view/repeat trainings exist, so the curve can locate whether fixed-budget restructuring turns over between 1x and MAX.",
        "out_dir": str(out_dir),
        "columns": STABLE_COLUMNS,
        "arms": {name: {**{k: (str(v) if isinstance(v, pathlib.Path) else v) for k, v in arm.items()}, "metric_ready": metric_ready(pathlib.Path(arm["run_dir"]), int(arm["max_words"])), "checkpoint_exists": {ck: checkpoint_exists(pathlib.Path(arm["run_dir"]), ck) for ck in arm["checkpoints"]}} for name, arm in ARMS.items()},
        "external_reuse_count": sum(1 for p in external.values() if p.exists()),
        "gpu_rows_at_plan": smi_rows(),
        "seed_spread_reference": str(SEED_SPREAD_CSV),
        "no_globalpiqa_superglue_aoa_upload_or_leaderboard": True,
    }
    (out_dir / "intermediate_dose_ladder_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": plan["status"], "out_dir": str(out_dir), "external_reuse_count": plan["external_reuse_count"], "plan_json": str(out_dir / "intermediate_dose_ladder_plan.json")}, indent=2), flush=True)
    if args.plan_only:
        return

    rows: list[dict[str, Any]] = []
    tasks: list[tuple[str, str]] = []
    problems: list[str] = []
    for arm_name, arm in ARMS.items():
        rd = pathlib.Path(arm["run_dir"])
        for ck in arm["checkpoints"]:
            local = out_per_target(out_dir, arm_name, ck)
            if local.exists() and not args.force:
                ok, ps = per_target_ok(local)
                if ok:
                    rows.append(summarize_payload(local, arm_name, ck, "local_reuse")); continue
                problems.extend([f"local {arm_name} {ck}: {p}" for p in ps])
            if not checkpoint_exists(rd, ck):
                problems.append(f"missing checkpoint {arm_name} {ck}: {rd / 'hf_model' / ck}")
                continue
            if not metric_ready(rd, int(arm["max_words"])):
                problems.append(f"metrics not ready for {arm_name}: {rd / 'scientific_metrics.json'}")
                continue
            tasks.append((arm_name, ck))
    (out_dir / "pre_eval_task_plan.json").write_text(json.dumps({"tasks": tasks, "reused_rows": len(rows), "problems": problems[:80]}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "intermediate_eval_task_count", "tasks": len(tasks), "reused_rows": len(rows), "pre_eval_problem_count": len(problems)}), flush=True)

    active: dict[int, tuple[str, str, subprocess.Popen[str], Any]] = {}
    pending = list(tasks)
    while pending or active:
        finished_gpus: list[int] = []
        for gpu, (arm_name, ck, proc, log) in list(active.items()):
            rc = proc.poll()
            if rc is not None:
                log.close(); finished_gpus.append(gpu)
                rec = {"event": "eval_finished", "utc": now(), "gpu": gpu, "arm": arm_name, "checkpoint": ck, "returncode": rc}
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(json.dumps(rec), flush=True)
                if rc != 0:
                    problems.append(f"eval failed {arm_name} {ck} rc={rc}; see {task_out_base(out_dir, arm_name, ck)}")
                else:
                    pt = out_per_target(out_dir, arm_name, ck)
                    ok, ps = per_target_ok(pt)
                    if ok:
                        rows.append(summarize_payload(pt, arm_name, ck, "new_eval"))
                    else:
                        problems.extend([f"new {arm_name} {ck}: {p}" for p in ps])
        for gpu in finished_gpus:
            active.pop(gpu, None)
        while pending and len(active) < max(1, args.max_parallel):
            gpu = choose_free_gpu(args.min_free_mib, set(active))
            if gpu is None:
                break
            arm_name, ck = pending.pop(0)
            proc, log = launch_eval(out_dir, arm_name, ck, gpu, args.force)
            active[gpu] = (arm_name, ck, proc, log)
            rec = {"event": "eval_launched", "utc": now(), "gpu": gpu, "arm": arm_name, "checkpoint": ck, "remaining": len(pending)}
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(json.dumps(rec), flush=True)
            time.sleep(10)
        if pending or active:
            time.sleep(30)

    rows = sorted(rows, key=lambda r: (str(r["data_arm"]), int(r["words"])))
    rows_csv = out_dir / "intermediate_dose_ladder_rows.csv"
    write_csv(rows_csv, rows)
    cmr, decomp = build_intermediate_contrasts(rows, external)
    cmr_csv = out_dir / "intermediate_dose_compact_minus_repeat.csv"
    decomp_csv = out_dir / "intermediate_dose_budget_decomposition.csv"
    write_csv(cmr_csv, cmr); write_csv(decomp_csv, decomp)
    summary = {
        "status": "INTERMEDIATE_DOSE_LADDER_STABLE_EVAL_DONE" if not problems else "INTERMEDIATE_DOSE_LADDER_STABLE_EVAL_DONE_WITH_PROBLEMS",
        "created_utc": now(), "plan": plan, "row_count": len(rows), "cmr_rows": len(cmr), "decomposition_rows": len(decomp),
        "problem_count": len(problems), "problems": problems[:50],
        "files": {"rows_csv": str(rows_csv), "compact_minus_repeat_csv": str(cmr_csv), "budget_decomposition_csv": str(decomp_csv), "seed_spread_csv": str(SEED_SPREAD_CSV)},
        "scope_note": "Stable selected families only; no GlobalPIQA, SuperGLUE, AoA, upload, or leaderboard action.",
    }
    (out_dir / "intermediate_dose_ladder_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research intermediate dose stable-family summary", "", summary["scope_note"], "", f"Rows: {len(rows)}; compact-minus-repeat rows: {len(cmr)}; decomposition rows: {len(decomp)}; problems: {len(problems)}.", "", f"Rows CSV: `{rows_csv}`", f"Compact-minus-repeat CSV: `{cmr_csv}`", f"Budget decomposition CSV: `{decomp_csv}`"]
    (out_dir / "intermediate_dose_ladder_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = {"status": summary["status"], "summary": str(out_dir / "intermediate_dose_ladder_summary.json"), "row_count": len(rows), "problem_count": len(problems), "problems": problems[:30]}
    (out_dir / "intermediate_dose_ladder_driver_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    if problems:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
