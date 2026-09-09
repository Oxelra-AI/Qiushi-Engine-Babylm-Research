#!/usr/bin/env python3
"""research: stable-family exposure-ladder evaluation for the fixed-budget dose family.

The dose experiment should not be read at a single checkpoint. This script scores
or reuses official-compatible selected-family outputs over the common exposure
ladder:
  * treatment-vs-clean: clean0 vs 1x/MAX view (and repeat) at 10M..80M;
  * compact-minus-repeat: 1x and MAX at 10M..100M;
  * dose amplification: MAX minus 1x exposure profiles.

It deliberately skips GlobalPIQA, SuperGLUE, AoA, upload, and leaderboard actions.
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
WS = USER_ROOT / "experiments/archive/frontier_consolidation"
EVALUATOR = WS / "scripts/evaluate_compliant_endpoint.py"
OUT_DEFAULT = WS / "data/dose_ladder_stable_eval"
SEED_LADDER_ROWS = WS / "data/full_deberta_seed_ladder_stable_eval/full_deberta_seed_ladder_stable_rows.csv"
SEED_SPREAD_CSV = WS / "data/full_deberta_seed_ladder_stable_eval/full_deberta_seed_ladder_seed_spread.csv"

STABLE_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "Reading"]
DERIVED_COLUMNS = ["cheap6_no_GlobalPIQA", "cheap5_no_GlobalPIQA_Reading", "EWoK_plus_Entity_sum"]
ALL_SCORE_COLUMNS = [*STABLE_COLUMNS, *DERIVED_COLUMNS]
CKS_10_80 = [f"chck_{i}M" for i in range(10, 81, 10)]
CKS_10_100 = [f"chck_{i}M" for i in range(10, 101, 10)]

ARMS = {
    "clean0": {
        "dose": 0.0,
        "data_arm": "clean",
        "arm_arg": "clean_qwen",
        "run_dir": WS / "training/runs/complianttok_cleanqwen_seed43022_80M",
        "checkpoints": CKS_10_80,
        "max_words": 80_000_000,
        "role": "0x fixed-tokenizer clean-Qwen reference, trained through 80M",
    },
    "dose1_view": {
        "dose": 1.0,
        "data_arm": "view",
        "arm_arg": "reinvest",
        "run_dir": WS / "training/runs/complianttok_reinvest_seed43022_r2",
        "checkpoints": CKS_10_100,
        "max_words": 100_000_000,
        "role": "1x compact-view reinvest anchor",
    },
    "dose1_repeat": {
        "dose": 1.0,
        "data_arm": "repeat",
        "arm_arg": "reinvest",
        "run_dir": WS / "training/runs/full_p2c_c2p_abs_repeat_deberta100M_seed43022",
        "checkpoints": CKS_10_100,
        "max_words": 100_000_000,
        "role": "1x hash-rotated repeat anchor",
    },
    "max_view": {
        "dose": 2.641480921798262,
        "data_arm": "view",
        "arm_arg": "reinvest",
        "run_dir": WS / "training/runs/full_p2c_c2p_abs_view_dose2p64x_matched_rowholdout_deberta100M_seed43022",
        "checkpoints": CKS_10_100,
        "max_words": 100_000_000,
        "role": "matched MAX 2.64x compact-view arm",
    },
    "max_repeat": {
        "dose": 2.641480921798262,
        "data_arm": "repeat",
        "arm_arg": "reinvest",
        "run_dir": WS / "training/runs/full_p2c_c2p_abs_repeat_dose2p64x_matched_rowholdout_deberta100M_seed43022",
        "checkpoints": CKS_10_100,
        "max_words": 100_000_000,
        "role": "matched MAX 2.64x hash-rotated repeat arm",
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


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def smi_rows() -> list[dict[str, int]]:
    p = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
    if p.returncode != 0:
        raise RuntimeError(p.stderr)
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


def wait_for_max(timeout_sec: int, poll_sec: int, log_path: pathlib.Path) -> dict[str, Any]:
    start = time.time()
    samples = 0
    while True:
        status = {}
        all_ready = True
        for arm_name in ["max_view", "max_repeat"]:
            arm = ARMS[arm_name]
            rd = pathlib.Path(arm["run_dir"])
            missing = [ck for ck in arm["checkpoints"] if not checkpoint_exists(rd, ck)]
            ready = metric_ready(rd, int(arm["max_words"])) and not missing
            status[arm_name] = {"run_dir": str(rd), "metric_ready": metric_ready(rd, int(arm["max_words"])), "missing_checkpoints": missing, "ready": ready}
            all_ready = all_ready and ready
        rec = {"event": "max_training_wait_sample", "utc": now(), "all_ready": all_ready, "status": status}
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(json.dumps({"event": "max_training_wait_sample", "utc": rec["utc"], "all_ready": all_ready}), flush=True)
        samples += 1
        if all_ready:
            return {"status": "max_runs_ready", "waited_sec": round(time.time()-start, 1), "sample_count": samples, "last_status": status}
        if time.time() - start > timeout_sec:
            return {"status": "max_wait_timeout", "waited_sec": round(time.time()-start, 1), "sample_count": samples, "last_status": status}
        time.sleep(poll_sec)


def load_external_from_seed_ladder() -> dict[tuple[str, str], pathlib.Path]:
    out: dict[tuple[str, str], pathlib.Path] = {}
    if not SEED_LADDER_ROWS.exists():
        return out
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
    arm = ARMS[arm_name]
    return {
        "arm": arm_name,
        "dose": arm["dose"],
        "data_arm": arm["data_arm"],
        "checkpoint": ck,
        "words": ck_words(ck),
        **sc,
        **de,
        "source_type": source_type,
        "per_target": str(path),
        "run_dir": str(arm["run_dir"]),
    }


def out_per_target(out_dir: pathlib.Path, arm_name: str, ck: str) -> pathlib.Path:
    return out_dir / "eval/per_target" / f"dose_{arm_name}_{ck}.json"


def task_out_base(out_dir: pathlib.Path, arm_name: str, ck: str) -> pathlib.Path:
    return out_dir / "runs" / arm_name / ck


def eval_command(out_dir: pathlib.Path, arm_name: str, ck: str, gpu: int, force: bool) -> list[str]:
    arm = ARMS[arm_name]
    cmd = [
        sys.executable, "-B", str(EVALUATOR),
        "--arm", str(arm["arm_arg"]),
        "--target", f"dose_{arm_name}_{ck}",
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


def series_stats(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    vals = [float(r[key]) for r in rows if finite(r.get(key))]
    return {"n": len(vals), "mean": mean(vals) if vals else None, "min": min(vals) if vals else None, "max": max(vals) if vals else None}


def build_deltas(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by = {(r["arm"], r["checkpoint"]): r for r in rows}
    treatment: list[dict[str, Any]] = []
    for dose_arm in ["dose1_view", "dose1_repeat", "max_view", "max_repeat"]:
        for ck in CKS_10_80:
            a = by.get((dose_arm, ck)); c = by.get(("clean0", ck))
            if a is None or c is None:
                continue
            rec = {"contrast": f"{dose_arm}_minus_clean0", "arm": dose_arm, "dose": ARMS[dose_arm]["dose"], "data_arm": ARMS[dose_arm]["data_arm"], "checkpoint": ck, "words": ck_words(ck)}
            for key in ALL_SCORE_COLUMNS:
                rec[key] = float(a[key]) - float(c[key]) if finite(a.get(key)) and finite(c.get(key)) else None
            treatment.append(rec)
    cmr: list[dict[str, Any]] = []
    for prefix, view_arm, repeat_arm in [("dose1", "dose1_view", "dose1_repeat"), ("max", "max_view", "max_repeat")]:
        for ck in CKS_10_100:
            v = by.get((view_arm, ck)); r = by.get((repeat_arm, ck))
            if v is None or r is None:
                continue
            rec = {"contrast": f"{prefix}_view_minus_repeat", "dose": ARMS[view_arm]["dose"], "checkpoint": ck, "words": ck_words(ck)}
            for key in ALL_SCORE_COLUMNS:
                rec[key] = float(v[key]) - float(r[key]) if finite(v.get(key)) and finite(r.get(key)) else None
            cmr.append(rec)
    dose_amp: list[dict[str, Any]] = []
    for kind, max_arm, one_arm in [("view", "max_view", "dose1_view"), ("repeat", "max_repeat", "dose1_repeat")]:
        for ck in CKS_10_100:
            m = by.get((max_arm, ck)); o = by.get((one_arm, ck))
            if m is None or o is None:
                continue
            rec = {"contrast": f"max_minus_1x_{kind}", "data_arm": kind, "checkpoint": ck, "words": ck_words(ck)}
            for key in ALL_SCORE_COLUMNS:
                rec[key] = float(m[key]) - float(o[key]) if finite(m.get(key)) and finite(o.get(key)) else None
            dose_amp.append(rec)
    # Difference in compact-minus-repeat between MAX and 1x.
    cmr_by = {(r["contrast"], r["checkpoint"]): r for r in cmr}
    for ck in CKS_10_100:
        m = cmr_by.get(("max_view_minus_repeat", ck)); o = cmr_by.get(("dose1_view_minus_repeat", ck))
        if m is None or o is None:
            continue
        rec = {"contrast": "max_cmr_minus_1x_cmr", "data_arm": "view_minus_repeat", "checkpoint": ck, "words": ck_words(ck)}
        for key in ALL_SCORE_COLUMNS:
            rec[key] = float(m[key]) - float(o[key]) if finite(m.get(key)) and finite(o.get(key)) else None
        dose_amp.append(rec)
    return treatment, cmr, dose_amp


def read_seed_spread() -> list[dict[str, Any]]:
    if not SEED_SPREAD_CSV.exists():
        return []
    with SEED_SPREAD_CSV.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_summary(out_dir: pathlib.Path, rows: list[dict[str, Any]], plan: dict[str, Any], problems: list[str]) -> dict[str, Any]:
    rows = sorted(rows, key=lambda r: (float(r["dose"]), str(r["data_arm"]), int(r["words"])))
    row_fields = ["arm", "dose", "data_arm", "checkpoint", "words", *ALL_SCORE_COLUMNS, "source_type", "per_target", "run_dir"]
    rows_csv = out_dir / "dose_ladder_stable_rows.csv"
    write_csv(rows_csv, rows, row_fields)
    treatment, cmr, dose_amp = build_deltas(rows)
    contrast_fields = ["contrast", "arm", "dose", "data_arm", "checkpoint", "words", *ALL_SCORE_COLUMNS]
    treatment_csv = out_dir / "dose_ladder_treatment_vs_clean.csv"
    cmr_csv = out_dir / "dose_ladder_compact_minus_repeat.csv"
    amp_csv = out_dir / "dose_ladder_max_minus_1x.csv"
    write_csv(treatment_csv, treatment, contrast_fields)
    write_csv(cmr_csv, cmr, contrast_fields)
    write_csv(amp_csv, dose_amp, contrast_fields)

    def stats_by_contrast(rs: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for c in sorted({str(r["contrast"]) for r in rs}):
            rr = [r for r in rs if r["contrast"] == c]
            out[c] = {k: series_stats(rr, k) for k in ["cheap6_no_GlobalPIQA", "cheap5_no_GlobalPIQA_Reading", "EWoK_plus_Entity_sum"]}
        return out

    seed_spread = read_seed_spread()
    summary = {
        "status": "DOSE_LADDER_STABLE_EVAL_DONE" if not problems else "DOSE_LADDER_STABLE_EVAL_DONE_WITH_PROBLEMS",
        "created_utc": now(),
        "plan": plan,
        "row_count": len(rows),
        "treatment_vs_clean_rows": len(treatment),
        "compact_minus_repeat_rows": len(cmr),
        "max_minus_1x_rows": len(dose_amp),
        "problem_count": len(problems),
        "problems": problems[:50],
        "files": {
            "rows_csv": str(rows_csv),
            "treatment_vs_clean_csv": str(treatment_csv),
            "compact_minus_repeat_csv": str(cmr_csv),
            "max_minus_1x_csv": str(amp_csv),
            "seed_spread_csv": str(SEED_SPREAD_CSV),
        },
        "contrast_stats": {
            "treatment_vs_clean_10M_to_80M": stats_by_contrast(treatment),
            "compact_minus_repeat_10M_to_100M": stats_by_contrast(cmr),
            "max_minus_1x_10M_to_100M": stats_by_contrast(dose_amp),
        },
        "seed_spread_reference": {
            "path": str(SEED_SPREAD_CSV),
            "rows": len(seed_spread),
            "meaning": "research same-coordinate seed43022-vs-seed43122 treatment-delta spread; use to price dose-profile differences, not as a per-item replication guarantee.",
        },
        "boundary": "Stable selected families only: BLiMP, Supplement, EWoK, Entity, COMPS, Reading. No GlobalPIQA, SuperGLUE, AoA, upload, or leaderboard action.",
    }
    (out_dir / "dose_ladder_stable_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research dose ladder stable-family summary",
        "",
        summary["boundary"],
        "",
        f"Rows: {len(rows)}; treatment-vs-clean rows: {len(treatment)}; compact-minus-repeat rows: {len(cmr)}; problems: {len(problems)}.",
        "",
        "## Main exposure-profile summaries",
    ]
    for family, stats in summary["contrast_stats"].items():
        lines.append(f"### {family}")
        for contrast, d in stats.items():
            lines.append(f"- {contrast}: cheap6 {d['cheap6_no_GlobalPIQA']}; cheap5 {d['cheap5_no_GlobalPIQA_Reading']}; EWoK+Entity {d['EWoK_plus_Entity_sum']}")
    lines += ["", f"Rows CSV: `{rows_csv}`", f"Treatment CSV: `{treatment_csv}`", f"Compact-minus-repeat CSV: `{cmr_csv}`", f"MAX-minus-1x CSV: `{amp_csv}`"]
    (out_dir / "dose_ladder_stable_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--wait-for-max-training", action="store_true")
    ap.add_argument("--training-wait-timeout-sec", type=int, default=14_400)
    ap.add_argument("--poll-sec", type=int, default=120)
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
    log_path = out_dir / "dose_ladder_driver_log.jsonl"
    external = load_external_from_seed_ladder()

    plan = {
        "status": "DOSE_LADDER_STABLE_PLAN",
        "created_utc": now(),
        "scientific_purpose": "Read dose response as exposure profiles, not a single 80M draw: treatment-vs-clean over 10M-80M and compact-minus-repeat over 10M-100M for 1x and MAX.",
        "out_dir": str(out_dir),
        "columns": STABLE_COLUMNS,
        "arms": {name: {**{k: (str(v) if isinstance(v, pathlib.Path) else v) for k, v in arm.items()}, "metric_ready": metric_ready(pathlib.Path(arm["run_dir"]), int(arm["max_words"])), "checkpoint_exists": {ck: checkpoint_exists(pathlib.Path(arm["run_dir"]), ck) for ck in arm["checkpoints"]}} for name, arm in ARMS.items()},
        "external_reuse_count": sum(1 for p in external.values() if p.exists()),
        "gpu_rows_at_plan": smi_rows(),
        "seed_spread_reference": str(SEED_SPREAD_CSV),
        "no_globalpiqa_superglue_aoa_upload_or_leaderboard": True,
    }
    (out_dir / "dose_ladder_stable_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": plan["status"], "out_dir": str(out_dir), "external_reuse_count": plan["external_reuse_count"], "plan_json": str(out_dir / "dose_ladder_stable_plan.json")}, indent=2), flush=True)
    if args.plan_only:
        return

    if args.wait_for_max_training:
        wait = wait_for_max(args.training_wait_timeout_sec, args.poll_sec, log_path)
        (out_dir / "max_training_wait_record.json").write_text(json.dumps(wait, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if wait.get("status") != "max_runs_ready":
            print(json.dumps({"status": "DOSE_LADDER_MAX_WAIT_TIMEOUT_NO_EVAL", "wait_record": wait}, indent=2, ensure_ascii=False), flush=True)
            raise SystemExit(2)

    rows: list[dict[str, Any]] = []
    tasks: list[tuple[str, str]] = []
    problems: list[str] = []
    for arm_name, arm in ARMS.items():
        rd = pathlib.Path(arm["run_dir"])
        for ck in arm["checkpoints"]:
            ext = external.get((arm_name, ck))
            if ext is not None and ext.exists() and not args.force:
                ok, ps = per_target_ok(ext)
                if ok:
                    rows.append(summarize_payload(ext, arm_name, ck, "external_reuse"))
                    continue
                problems.extend([f"external {arm_name} {ck}: {p}" for p in ps])
            local = out_per_target(out_dir, arm_name, ck)
            if local.exists() and not args.force:
                ok, ps = per_target_ok(local)
                if ok:
                    rows.append(summarize_payload(local, arm_name, ck, "local_reuse"))
                    continue
                problems.extend([f"local {arm_name} {ck}: {p}" for p in ps])
            if not checkpoint_exists(rd, ck):
                problems.append(f"missing checkpoint {arm_name} {ck}: {rd / 'hf_model' / ck}")
                continue
            if not metric_ready(rd, int(arm["max_words"])):
                problems.append(f"metrics not ready for {arm_name}: {rd / 'scientific_metrics.json'}")
                continue
            tasks.append((arm_name, ck))
    (out_dir / "pre_eval_task_plan.json").write_text(json.dumps({"tasks": tasks, "reused_rows": len(rows), "problems": problems[:80]}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "eval_task_count", "tasks": len(tasks), "reused_rows": len(rows), "pre_eval_problem_count": len(problems)}), flush=True)

    active: dict[int, tuple[str, str, subprocess.Popen[str], Any]] = {}
    pending = list(tasks)
    while pending or active:
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

    summary = write_summary(out_dir, rows, plan, problems)
    result = {"status": summary["status"], "summary": str(out_dir / "dose_ladder_stable_summary.json"), "row_count": len(rows), "problem_count": len(problems), "problems": problems[:30]}
    (out_dir / "dose_ladder_driver_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    if problems:
        raise SystemExit(3)

if __name__ == "__main__":
    main()
