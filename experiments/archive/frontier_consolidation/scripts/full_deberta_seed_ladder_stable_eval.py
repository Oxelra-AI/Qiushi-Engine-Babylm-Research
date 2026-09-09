#!/usr/bin/env python3
"""research: stable-family selected-evaluation ladder for full-DeBERTa seed reference.

Purpose
-------
The seed43122 compact/repeat pair is not only a two-point replicate at 80M/100M.
Together with the original seed43022 full-DeBERTa compact/repeat cells it defines
four same-architecture cells over a common 10M checkpoint ladder. This script
scores the stable selected families on that ladder and summarizes:

  * compact-minus-repeat as a function of exposure for each seed,
  * compact/repeat same-data cross-seed spread,
  * treatment-delta seed spread.

It deliberately skips SuperGLUE, AoA, upload, and leaderboard actions. By default
it also skips GlobalPIQA to focus on the stable families that governed the recent
mechanism arguments and to reduce compute.
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
from statistics import mean
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
STUDY = USER_ROOT / "experiments/archive/frontier_consolidation"
WS = STUDY
EVALUATOR = WS / "scripts/evaluate_compliant_endpoint.py"
DEFAULT_OUT = WS / "data/full_deberta_seed_ladder_stable_eval"

STABLE_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "Reading"]
SCORE_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "Reading"]
CHECKPOINTS = [f"chck_{i}M" for i in range(10, 101, 10)]

DEFAULT_ARMS = {
    "seed43022_compact": {
        "seed": 43022,
        "data_arm": "compact",
        "run_dir": WS / "training/runs/complianttok_reinvest_seed43022_r2",
        "role": "original full-DeBERTa compact semantic-view cell; 1M checkpoint cadence, reused endpoint asset",
    },
    "seed43022_repeat": {
        "seed": 43022,
        "data_arm": "repeat",
        "run_dir": WS / "training/runs/full_p2c_c2p_abs_repeat_deberta100M_seed43022",
        "role": "original full-DeBERTa repeat control from research positional-interaction grid",
    },
    "seed43122_compact": {
        "seed": 43122,
        "data_arm": "compact",
        "run_dir": WS / "training/runs/full_p2c_c2p_abs_compact_deberta100M_seed43122",
        "role": "second-seed full-DeBERTa compact replicate launched in research",
    },
    "seed43122_repeat": {
        "seed": 43122,
        "data_arm": "repeat",
        "run_dir": WS / "training/runs/full_p2c_c2p_abs_repeat_deberta100M_seed43122_retry2",
        "role": "second-seed full-DeBERTa repeat retry launched in research after research OOM",
    },
}

EXISTING_PER_TARGETS: dict[tuple[str, str], pathlib.Path] = {
    ("seed43022_compact", "chck_20M"): WS / "data/legal20m_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_20M.json",
    ("seed43022_compact", "chck_70M"): WS / "data/legal_mature_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_70M.json",
    ("seed43022_compact", "chck_80M"): WS / "data/legal_mature_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_80M.json",
    ("seed43022_compact", "chck_100M"): WS / "data/compliant_full_eval/per_target/complianttok_reinvest_seed43022.json",
    ("seed43022_repeat", "chck_80M"): WS / "data/architecture_interaction_selected_panel_final/full_repeat/chck_80M/eval/per_target/arch_full_repeat_chck_80M.json",
    ("seed43022_repeat", "chck_100M"): WS / "data/architecture_interaction_selected_panel_final/full_repeat/chck_100M/eval/per_target/arch_full_repeat_chck_100M.json",
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def resolve_path(p: str | pathlib.Path) -> pathlib.Path:
    q = pathlib.Path(p)
    return q if q.is_absolute() else USER_ROOT / q


def ck_words(ck: str) -> int:
    if not (ck.startswith("chck_") and ck.endswith("M")):
        raise ValueError(ck)
    return int(ck[5:-1]) * 1_000_000


def finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def smi_rows() -> list[dict[str, int]]:
    p = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    if p.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {p.stderr}")
    rows: list[dict[str, int]] = []
    for line in p.stdout.splitlines():
        if not line.strip():
            continue
        idx, total, used, free, util = [int(x.strip()) for x in line.split(",")]
        rows.append({"index": idx, "total_mib": total, "used_mib": used, "free_mib": free, "util_gpu_pct": util})
    return rows


def metric_ready(run_dir: pathlib.Path) -> bool:
    p = run_dir / "scientific_metrics.json"
    if not p.exists():
        return False
    try:
        m = read_json(p)
    except Exception:
        return False
    return int(m.get("word_exposure", 0) or 0) >= 100_000_000 and int(m.get("actual_training_steps", 0) or 0) >= 2529


def checkpoint_exists(run_dir: pathlib.Path, ck: str) -> bool:
    d = run_dir / "hf_model" / ck
    return (d / "model.safetensors").exists() or (d / "pytorch_model.bin").exists()


def wait_for_runs(arms: dict[str, dict[str, Any]], checkpoints: list[str], timeout_sec: int, poll_sec: int, log_path: pathlib.Path) -> dict[str, Any]:
    start = time.time()
    samples = 0
    while True:
        status: dict[str, Any] = {}
        all_ready = True
        for name, arm in arms.items():
            rd = pathlib.Path(arm["run_dir"])
            missing = [ck for ck in checkpoints if not checkpoint_exists(rd, ck)]
            ready = metric_ready(rd) and not missing
            status[name] = {"run_dir": str(rd), "metric_ready": metric_ready(rd), "missing_checkpoints": missing, "ready": ready}
            all_ready = all_ready and ready
        rec = {"event": "run_wait_sample", "utc": now(), "all_ready": all_ready, "status": status}
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(json.dumps({"event": "run_wait_sample", "utc": rec["utc"], "all_ready": all_ready}), flush=True)
        samples += 1
        if all_ready:
            return {"status": "runs_ready", "waited_sec": round(time.time() - start, 1), "sample_count": samples, "last_status": status}
        if time.time() - start > timeout_sec:
            return {"status": "run_wait_timeout", "waited_sec": round(time.time() - start, 1), "sample_count": samples, "last_status": status}
        time.sleep(poll_sec)


def out_per_target(out_dir: pathlib.Path, arm_name: str, ck: str) -> pathlib.Path:
    target = f"ladder_{arm_name}_{ck}"
    return out_dir / "eval" / "per_target" / f"{target}.json"


def task_out_base(out_dir: pathlib.Path, arm_name: str, ck: str) -> pathlib.Path:
    return out_dir / "runs" / arm_name / ck


def external_per_target(arm_name: str, ck: str) -> pathlib.Path | None:
    p = EXISTING_PER_TARGETS.get((arm_name, ck))
    return p if p is not None and p.exists() else None


def extract_scores(payload: dict[str, Any]) -> dict[str, float | None]:
    tasks = payload.get("tasks") or {}
    out: dict[str, float | None] = {}
    for col in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]:
        rec = tasks.get(col) or {}
        out[col] = float(rec["score"]) if rec.get("score") is not None else None
    rd = tasks.get("Reading") or {}
    if isinstance(rd.get("scores"), dict) and rd["scores"].get("Reading") is not None:
        out["Reading"] = float(rd["scores"]["Reading"])
    elif rd.get("score") is not None:
        out["Reading"] = float(rd["score"])
    else:
        out["Reading"] = None
    return out


def derived(scores: dict[str, float | None]) -> dict[str, float | None]:
    vals6 = [scores.get(c) for c in SCORE_COLUMNS]
    vals5 = [scores.get(c) for c in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]]
    return {
        "cheap6_no_GlobalPIQA": float(mean([float(x) for x in vals6])) if all(finite(x) for x in vals6) else None,
        "cheap5_no_GlobalPIQA_Reading": float(mean([float(x) for x in vals5])) if all(finite(x) for x in vals5) else None,
        "EWoK_plus_Entity_sum": float(scores["EWoK"]) + float(scores["Entity"]) if finite(scores.get("EWoK")) and finite(scores.get("Entity")) else None,
    }


def per_target_ok(path: pathlib.Path) -> tuple[bool, list[str]]:
    problems: list[str] = []
    if not path.exists():
        return False, [f"missing {path}"]
    try:
        payload = read_json(path)
    except Exception as exc:
        return False, [f"json read failed: {exc!r}"]
    tasks = payload.get("tasks") or {}
    for col in STABLE_COLUMNS:
        rec = tasks.get(col)
        if not isinstance(rec, dict):
            problems.append(f"missing task {col}")
            continue
        if rec.get("returncode") != 0:
            problems.append(f"{col} returncode {rec.get('returncode')}")
        if col == "Reading":
            val = (rec.get("scores") or {}).get("Reading", rec.get("score"))
        else:
            val = rec.get("score")
        if not finite(val):
            problems.append(f"{col} score nonfinite {val}")
    return not problems, problems


def summarize_payload(path: pathlib.Path, arm_name: str, ck: str, source_type: str, arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    payload = read_json(path)
    scores = extract_scores(payload)
    der = derived(scores)
    return {
        "arm": arm_name,
        "seed": arms[arm_name]["seed"],
        "data_arm": arms[arm_name]["data_arm"],
        "checkpoint": ck,
        "words": ck_words(ck),
        **scores,
        **der,
        "source_type": source_type,
        "per_target": str(path),
    }


def eval_command(out_dir: pathlib.Path, arm_name: str, ck: str, gpu: int, arms: dict[str, dict[str, Any]], force: bool) -> list[str]:
    target = f"ladder_{arm_name}_{ck}"
    cmd = [
        sys.executable,
        "-B",
        str(EVALUATOR),
        "--arm",
        "reinvest",
        "--target",
        target,
        "--run-dir",
        str(arms[arm_name]["run_dir"]),
        "--endpoint",
        ck,
        "--out-root",
        str(out_dir / "eval"),
        "--collate-root",
        str(out_dir / "collate"),
        "--gpu",
        str(gpu),
        "--columns",
        *STABLE_COLUMNS,
    ]
    if force:
        cmd.append("--force")
    return cmd


def launch_eval(out_dir: pathlib.Path, arm_name: str, ck: str, gpu: int, arms: dict[str, dict[str, Any]], force: bool) -> tuple[subprocess.Popen[str], pathlib.TextIOWrapper]:
    base = task_out_base(out_dir, arm_name, ck)
    base.mkdir(parents=True, exist_ok=True)
    log = (base / "stable_eval_stdout.log").open("w", encoding="utf-8")
    cmd = eval_command(out_dir, arm_name, ck, gpu, arms, force)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    log.write(json.dumps({"event": "launch_eval", "utc": now(), "arm": arm_name, "checkpoint": ck, "gpu": gpu, "cmd": cmd}, ensure_ascii=False) + "\n")
    log.flush()
    p = subprocess.Popen(cmd, cwd=str(USER_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    return p, log


def choose_free_gpu(min_free_mib: int, busy_gpus: set[int]) -> int | None:
    rows = smi_rows()
    candidates = [r for r in rows if r["index"] not in busy_gpus and r["free_mib"] >= min_free_mib]
    if not candidates:
        return None
    candidates.sort(key=lambda r: r["free_mib"], reverse=True)
    return candidates[0]["index"]


def write_summary(out_dir: pathlib.Path, rows: list[dict[str, Any]], plan: dict[str, Any]) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: (int(r["seed"]), str(r["data_arm"]), int(r["words"])))
    csv_path = out_dir / "full_deberta_seed_ladder_stable_rows.csv"
    fieldnames = ["arm", "seed", "data_arm", "checkpoint", "words", *SCORE_COLUMNS, "cheap6_no_GlobalPIQA", "cheap5_no_GlobalPIQA_Reading", "EWoK_plus_Entity_sum", "source_type", "per_target"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    by = {(int(r["seed"]), str(r["data_arm"]), str(r["checkpoint"])): r for r in rows}
    delta_rows: list[dict[str, Any]] = []
    for seed in sorted({int(r["seed"]) for r in rows}):
        for ck in CHECKPOINTS:
            c = by.get((seed, "compact", ck))
            rep = by.get((seed, "repeat", ck))
            if c is None or rep is None:
                continue
            d: dict[str, Any] = {"seed": seed, "checkpoint": ck, "words": ck_words(ck)}
            for key in [*SCORE_COLUMNS, "cheap6_no_GlobalPIQA", "cheap5_no_GlobalPIQA_Reading", "EWoK_plus_Entity_sum"]:
                d[key] = float(c[key]) - float(rep[key]) if finite(c.get(key)) and finite(rep.get(key)) else None
            delta_rows.append(d)
    delta_csv = out_dir / "full_deberta_seed_ladder_treatment_deltas.csv"
    delta_fields = ["seed", "checkpoint", "words", *SCORE_COLUMNS, "cheap6_no_GlobalPIQA", "cheap5_no_GlobalPIQA_Reading", "EWoK_plus_Entity_sum"]
    with delta_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=delta_fields, extrasaction="ignore")
        w.writeheader()
        for r in delta_rows:
            w.writerow(r)

    delta_by = {(int(r["seed"]), str(r["checkpoint"])): r for r in delta_rows}
    spread_rows: list[dict[str, Any]] = []
    for ck in CHECKPOINTS:
        r0 = delta_by.get((43022, ck))
        r1 = delta_by.get((43122, ck))
        if r0 is None or r1 is None:
            continue
        sp: dict[str, Any] = {"checkpoint": ck, "words": ck_words(ck)}
        for key in [*SCORE_COLUMNS, "cheap6_no_GlobalPIQA", "cheap5_no_GlobalPIQA_Reading", "EWoK_plus_Entity_sum"]:
            sp[f"delta_seed43022_{key}"] = r0.get(key)
            sp[f"delta_seed43122_{key}"] = r1.get(key)
            sp[f"treatment_delta_seed43122_minus_seed43022_{key}"] = float(r1[key]) - float(r0[key]) if finite(r0.get(key)) and finite(r1.get(key)) else None
        spread_rows.append(sp)
    spread_csv = out_dir / "full_deberta_seed_ladder_seed_spread.csv"
    if spread_rows:
        spread_fields = list(spread_rows[0].keys())
        with spread_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=spread_fields, extrasaction="ignore")
            w.writeheader()
            for r in spread_rows:
                w.writerow(r)

    def series_stats(rs: list[dict[str, Any]], key: str) -> dict[str, Any]:
        vals = [float(r[key]) for r in rs if finite(r.get(key))]
        return {"n": len(vals), "mean": mean(vals) if vals else None, "min": min(vals) if vals else None, "max": max(vals) if vals else None}

    summary = {
        "status": "FULL_DEBERTA_SEED_LADDER_STABLE_SUMMARY",
        "created_utc": now(),
        "plan": plan,
        "row_count": len(rows),
        "delta_row_count": len(delta_rows),
        "spread_row_count": len(spread_rows),
        "rows_csv": str(csv_path),
        "delta_csv": str(delta_csv),
        "spread_csv": str(spread_csv) if spread_rows else None,
        "delta_stats": {
            str(seed): {
                "cheap6_no_GlobalPIQA": series_stats([r for r in delta_rows if int(r["seed"]) == seed], "cheap6_no_GlobalPIQA"),
                "cheap5_no_GlobalPIQA_Reading": series_stats([r for r in delta_rows if int(r["seed"]) == seed], "cheap5_no_GlobalPIQA_Reading"),
                "EWoK_plus_Entity_sum": series_stats([r for r in delta_rows if int(r["seed"]) == seed], "EWoK_plus_Entity_sum"),
            }
            for seed in sorted({int(r["seed"]) for r in delta_rows})
        },
        "boundary": "Stable selected-family evaluation only: BLiMP, Supplement, EWoK, Entity, COMPS, Reading. No GlobalPIQA, SuperGLUE, AoA, upload, or leaderboard action.",
    }
    (out_dir / "full_deberta_seed_ladder_stable_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["# research full-DeBERTa seed ladder stable-family summary", "", summary["boundary"], "", f"Rows: {len(rows)}; treatment delta rows: {len(delta_rows)}; spread rows: {len(spread_rows)}", "", "## Treatment delta stats across 10M ladder"]
    for seed, stats in summary["delta_stats"].items():
        lines.append(f"- seed {seed}: cheap6 {stats['cheap6_no_GlobalPIQA']}; cheap5 {stats['cheap5_no_GlobalPIQA_Reading']}; EWoK+Entity {stats['EWoK_plus_Entity_sum']}")
    lines += ["", f"Rows CSV: `{csv_path}`", f"Delta CSV: `{delta_csv}`", f"Spread CSV: `{spread_csv}`"]
    (out_dir / "full_deberta_seed_ladder_stable_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--repeat43122-run-dir", default=str(DEFAULT_ARMS["seed43122_repeat"]["run_dir"]))
    ap.add_argument("--compact43122-run-dir", default=str(DEFAULT_ARMS["seed43122_compact"]["run_dir"]))
    ap.add_argument("--checkpoints", nargs="*", default=CHECKPOINTS)
    ap.add_argument("--wait-for-training", action="store_true")
    ap.add_argument("--training-wait-timeout-sec", type=int, default=36_000)
    ap.add_argument("--poll-sec", type=int, default=120)
    ap.add_argument("--min-free-mib", type=int, default=24_000)
    ap.add_argument("--max-parallel", type=int, default=2)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    if not EVALUATOR.exists():
        raise FileNotFoundError(EVALUATOR)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "ladder_eval_driver_log.jsonl"
    arms = {k: dict(v) for k, v in DEFAULT_ARMS.items()}
    arms["seed43122_compact"]["run_dir"] = resolve_path(args.compact43122_run_dir)
    arms["seed43122_repeat"]["run_dir"] = resolve_path(args.repeat43122_run_dir)
    checkpoints = args.checkpoints

    plan = {
        "status": "FULL_DEBERTA_SEED_LADDER_STABLE_PLAN",
        "created_utc": now(),
        "decision_role": "Quantify exposure profile and same-architecture seed spread of compact-minus-repeat stable-family movement over the common 10M ladder, so earlier single-seed readings can be priced against measured run-to-run spread.",
        "out_dir": str(out_dir),
        "checkpoints": checkpoints,
        "columns": STABLE_COLUMNS,
        "arms": {k: {**{kk: (str(vv) if isinstance(vv, pathlib.Path) else vv) for kk, vv in v.items()}, "metric_ready": metric_ready(pathlib.Path(v["run_dir"])), "checkpoint_exists": {ck: checkpoint_exists(pathlib.Path(v["run_dir"]), ck) for ck in checkpoints}} for k, v in arms.items()},
        "existing_reuse": {f"{a}:{ck}": str(p) for (a, ck), p in EXISTING_PER_TARGETS.items() if ck in checkpoints and p.exists()},
        "gpu_rows_at_plan": smi_rows(),
        "no_globalpiqa_superglue_aoa_upload_or_leaderboard": True,
    }
    (out_dir / "full_deberta_seed_ladder_stable_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": plan["status"], "out_dir": str(out_dir), "arms": list(arms), "checkpoints": checkpoints, "plan_json": str(out_dir / "full_deberta_seed_ladder_stable_plan.json")}, indent=2), flush=True)
    if args.plan_only:
        return

    if args.wait_for_training:
        wait = wait_for_runs(arms, checkpoints, args.training_wait_timeout_sec, args.poll_sec, log_path)
        (out_dir / "training_wait_record.json").write_text(json.dumps(wait, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if wait.get("status") != "runs_ready":
            print(json.dumps({"status": "LADDER_TRAINING_WAIT_TIMEOUT_NO_EVAL", "wait_record": wait}, indent=2, ensure_ascii=False), flush=True)
            raise SystemExit(2)

    rows: list[dict[str, Any]] = []
    tasks: list[tuple[str, str]] = []
    problems: list[str] = []
    for arm_name, arm in arms.items():
        rd = pathlib.Path(arm["run_dir"])
        for ck in checkpoints:
            ext = external_per_target(arm_name, ck)
            local = out_per_target(out_dir, arm_name, ck)
            if ext is not None and not args.force:
                ok, ps = per_target_ok(ext)
                if ok:
                    rows.append(summarize_payload(ext, arm_name, ck, "external_reuse", arms))
                    continue
                problems.extend([f"external {arm_name} {ck}: {p}" for p in ps])
            if local.exists() and not args.force:
                ok, ps = per_target_ok(local)
                if ok:
                    rows.append(summarize_payload(local, arm_name, ck, "local_reuse", arms))
                    continue
                problems.extend([f"local {arm_name} {ck}: {p}" for p in ps])
            if not checkpoint_exists(rd, ck):
                problems.append(f"missing checkpoint {arm_name} {ck}: {rd / 'hf_model' / ck}")
                continue
            if not metric_ready(rd):
                problems.append(f"metrics not terminal for {arm_name}: {rd / 'scientific_metrics.json'}")
                continue
            tasks.append((arm_name, ck))

    (out_dir / "pre_eval_task_plan.json").write_text(json.dumps({"tasks": tasks, "initial_rows": len(rows), "problems": problems}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if problems:
        print(json.dumps({"event": "pre_eval_problems", "count": len(problems), "first": problems[:8]}, indent=2), flush=True)
        # Missing not-yet-trained new arms is normally handled by --wait-for-training; after waiting, any problem is real.
        if not tasks:
            write_summary(out_dir, rows, plan)
            raise SystemExit(2)
    print(json.dumps({"event": "eval_task_count", "tasks": len(tasks), "reused_rows": len(rows)}), flush=True)

    active: dict[int, tuple[str, str, subprocess.Popen[str], pathlib.TextIOWrapper]] = {}
    pending = list(tasks)
    while pending or active:
        # Harvest finished evals.
        finished_gpus: list[int] = []
        for gpu, (arm_name, ck, proc, log) in list(active.items()):
            rc = proc.poll()
            if rc is not None:
                log.close()
                finished_gpus.append(gpu)
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
                        rows.append(summarize_payload(pt, arm_name, ck, "new_eval", arms))
                    else:
                        problems.extend([f"new {arm_name} {ck}: {p}" for p in ps])
        for gpu in finished_gpus:
            active.pop(gpu, None)

        # Launch on free GPUs.
        while pending and len(active) < max(1, args.max_parallel):
            gpu = choose_free_gpu(args.min_free_mib, set(active))
            if gpu is None:
                break
            arm_name, ck = pending.pop(0)
            proc, log = launch_eval(out_dir, arm_name, ck, gpu, arms, args.force)
            active[gpu] = (arm_name, ck, proc, log)
            rec = {"event": "eval_launched", "utc": now(), "gpu": gpu, "arm": arm_name, "checkpoint": ck, "remaining": len(pending)}
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(json.dumps(rec), flush=True)
            time.sleep(10)

        if pending or active:
            time.sleep(30)

    summary = write_summary(out_dir, rows, plan)
    final = {"status": "LADDER_STABLE_EVAL_DONE" if not problems else "LADDER_STABLE_EVAL_DONE_WITH_PROBLEMS", "summary": str(out_dir / "full_deberta_seed_ladder_stable_summary.json"), "row_count": len(rows), "problem_count": len(problems), "problems": problems[:30]}
    (out_dir / "ladder_eval_driver_result.json").write_text(json.dumps(final, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(final, indent=2, ensure_ascii=False), flush=True)
    if problems:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
