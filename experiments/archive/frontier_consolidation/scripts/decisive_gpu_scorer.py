#!/usr/bin/env python3
"""research: corrected decisive GPU scorer for final fixed-budget substitution readouts.

Repairs the research GPU scorer:
  * Supplement uses official task `blimp` with `supplement_filtered` data.
  * Per-family timeouts are long enough for BLiMP/COMPS/Entity on H100.
  * The job set is restricted to decisive checkpoints/families, not a broad 125-job sweep.
  * A lightweight wait loop allows this task to start before childspeech/incorpus training
    finishes and then score the required checkpoints as they appear.

Scientific purpose:
  - Register contrast: childspeech_removed - adultprose_removed with identical admitted FineWeb.
  - In-corpus contrast: official-corpus adult prose versus FineWeb full_1x/clean at matched geometry.
  - Sub-dose trajectory: use completed quarter/half and partial full checkpoints only as available.

No training, no GlobalPIQA, no SuperGLUE, no AoA, no upload, no leaderboard action.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import math
import os
import pathlib
import re
import statistics
import subprocess
import sys
import time
from typing import Any


def find_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_root()
WS = ROOT / "experiments/archive" / 'frontier_consolidation'
STRICT = ROOT / "experiments/archive" / 'representation_and_objectives' / "data" / "pristine_official_coordinate" / "babylm-eval" / "strict"
EVAL_DATA = STRICT / "evaluation_data" / "full_eval"
NLP_DATA = ROOT / "experiments/archive" / 'initial_model_studies' / "data" / "nltk_data"
OUT_ROOT = WS / "data" / "decisive_gpu_eval"

FAMILIES: dict[str, dict[str, Any]] = {
    "BLiMP": {"task": "blimp", "data": EVAL_DATA / "blimp_filtered", "batch": 128, "timeout": 1800},
    "Supplement": {"task": "blimp", "data": EVAL_DATA / "supplement_filtered", "batch": 128, "timeout": 900},
    "EWoK": {"task": "ewok", "data": EVAL_DATA / "ewok_filtered", "batch": 64, "timeout": 900},
    "COMPS": {"task": "comps", "data": EVAL_DATA / "comps", "batch": 128, "timeout": 1800},
    "Entity": {"task": "entity_tracking", "data": EVAL_DATA / "entity_tracking", "batch": 128, "timeout": 1800},
}

ARMS: dict[str, dict[str, Any]] = {
    "regmax_adultprose": {
        "run_dir": WS / "training" / "runs" / "regmax_adultprose_samefw_deberta100M_seed43022",
        "role": "Same FineWeb admission, adult-prose clean rows sacrificed.",
    },
    "regmax_childspeech": {
        "run_dir": WS / "training" / "runs" / "regmax_childspeech_samefw_deberta100M_seed43022",
        "role": "Same FineWeb admission, developmental/speech clean rows sacrificed.",
    },
    "incorpus_adultprose": {
        "run_dir": WS / "training" / "runs" / "incorpus_adultprose_deberta100M_seed43022",
        "role": "No FineWeb: unused official Gutenberg/SimpleWiki admitted while developmental/speech rows sacrificed.",
    },
    "subdose_quarter": {
        "run_dir": WS / "training" / "runs" / "subdose_quarter_1x_deberta100M_seed43022",
        "role": "Quarter full-1x dose, same MAX geometry.",
    },
    "subdose_half": {
        "run_dir": WS / "training" / "runs" / "subdose_half_1x_deberta100M_seed43022",
        "role": "Half full-1x dose, same MAX geometry.",
    },
    "subdose_full": {
        "run_dir": WS / "training" / "runs" / "subdose_full_1x_deberta100M_seed43022",
        "role": "Full-1x dose; training killed at chck_70M, so late checkpoints absent.",
    },
}

CHECKPOINTS_DECISIVE = ["chck_80M", "chck_100M"]
CHECKPOINTS_SUBDOSE = ["chck_40M", "chck_50M", "chck_60M", "chck_70M", "chck_80M", "chck_100M"]
FAMILY_ORDER = ["BLiMP", "Supplement", "EWoK", "COMPS", "Entity"]
EX_ENTITY = ["BLiMP", "Supplement", "EWoK", "COMPS"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def write_json(path: pathlib.Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def parse_score(text: str) -> float | None:
    for pat in [
        r"### AVERAGE [A-Z_ '\\-]*\s*\n\s*([+-]?[0-9]+(?:\.[0-9]+)?)",
        r"AVERAGE ACCURACY\s*\n\s*([+-]?[0-9]+(?:\.[0-9]+)?)",
        r"\n\s*1\.0\s+([0-9]+(?:\.[0-9]+)?)\s*\n\s*TEMPERATURE",
    ]:
        m = re.search(pat, text)
        if m:
            v = float(m.group(1))
            if math.isfinite(v) and -5 <= v <= 105:
                return v
    return None


def latest_file(root: pathlib.Path, pattern: str) -> pathlib.Path | None:
    if not root.exists():
        return None
    hits = sorted(root.rglob(pattern), key=lambda p: (p.stat().st_mtime, str(p)))
    return hits[-1] if hits else None


def read_report_score(task_out: pathlib.Path) -> float | None:
    p = latest_file(task_out, "best_temperature_report.txt")
    if p is None:
        return None
    return parse_score(p.read_text(encoding="utf-8", errors="replace"))


def task_done(payload: dict[str, Any], fam: str) -> bool:
    rec = (payload.get("tasks") or {}).get(fam)
    return isinstance(rec, dict) and rec.get("returncode") == 0 and finite(rec.get("score"))


def aggregate(tasks: dict[str, Any]) -> dict[str, Any]:
    vals: dict[str, float] = {}
    for fam in FAMILY_ORDER:
        rec = tasks.get(fam, {}) if isinstance(tasks, dict) else {}
        s = rec.get("score") if isinstance(rec, dict) else None
        if finite(s):
            vals[fam] = float(s)
    out: dict[str, Any] = {"scored_families": sorted(vals)}
    if all(f in vals for f in EX_ENTITY):
        out["exEntity4"] = round(statistics.mean(vals[f] for f in EX_ENTITY), 6)
    else:
        out["exEntity4"] = None
    if all(f in vals for f in FAMILY_ORDER):
        out["cheap5_noReading"] = round(statistics.mean(vals[f] for f in FAMILY_ORDER), 6)
    else:
        out["cheap5_noReading"] = None
    out["Entity"] = vals.get("Entity")
    return out


def payload_path(arm: str, ck: str) -> pathlib.Path:
    return OUT_ROOT / "per_target" / f"{arm}_{ck}.json"


def load_payload(arm: str, ck: str) -> dict[str, Any]:
    p = payload_path(arm, ck)
    if p.exists():
        data = read_json(p)
        data.setdefault("tasks", {})
        return data
    cfg = ARMS[arm]
    return {
        "target": f"{arm}_{ck}",
        "arm": arm,
        "checkpoint": ck,
        "run_dir": rel(cfg["run_dir"]),
        "model_path": rel(cfg["run_dir"] / "hf_model" / ck),
        "role": cfg["role"],
        "started_utc": now(),
        "tasks": {},
        "no_globalpiqa_superglue_aoa_upload_or_leaderboard": True,
    }


def save_payload(payload: dict[str, Any]) -> None:
    payload["updated_utc"] = now()
    payload["aggregates"] = aggregate(payload.get("tasks", {}))
    write_json(payload_path(payload["arm"], payload["checkpoint"]), payload)


def model_exists(arm: str, ck: str, require_metrics: bool, partial_no_metrics: set[str] | None = None) -> bool:
    partial_no_metrics = partial_no_metrics or set()
    run = ARMS[arm]["run_dir"]
    if not (run / "hf_model" / ck).exists():
        return False
    if require_metrics and arm not in partial_no_metrics and not (run / "scientific_metrics.json").exists():
        return False
    return True


def all_requested_models_ready(jobs: list[tuple[str, str, str]], require_metrics: bool, partial_no_metrics: set[str] | None = None) -> bool:
    return all(model_exists(a, c, require_metrics, partial_no_metrics) for a, c, _ in jobs)


def gpu_state(gpu: int) -> dict[str, Any]:
    try:
        proc = subprocess.run([
            "nvidia-smi", f"--id={gpu}",
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15)
        if proc.returncode != 0:
            return {"available": False, "error": proc.stderr.strip()[:200]}
        line = proc.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        return {"available": True, "memory_used_mib": int(float(parts[0])), "util_pct": int(float(parts[1]))}
    except Exception as exc:
        return {"available": False, "error": repr(exc)}


def gpu_ready(gpu: int, max_used_mib: int, max_util_pct: int) -> bool:
    st = gpu_state(gpu)
    if not st.get("available"):
        # Do not fail closed if nvidia-smi is hidden; the evaluator will surface true CUDA failures.
        return True
    return int(st.get("memory_used_mib", 10**9)) <= max_used_mib and int(st.get("util_pct", 100)) <= max_util_pct


def gpu_env(target: str, gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["NLTK_DATA"] = str(NLP_DATA.resolve())
    cache = OUT_ROOT / "hf_cache" / target
    tmp = OUT_ROOT / "tmp" / target
    for p in [cache, cache / "hub", cache / "transformers", cache / "modules", cache / "datasets", tmp]:
        p.mkdir(parents=True, exist_ok=True)
    env["HF_HOME"] = str(cache.resolve())
    env["HF_HUB_CACHE"] = str((cache / "hub").resolve())
    env["TRANSFORMERS_CACHE"] = str((cache / "transformers").resolve())
    env["HF_MODULES_CACHE"] = str((cache / "modules").resolve())
    env["HF_DATASETS_CACHE"] = str((cache / "datasets").resolve())
    env["TMPDIR"] = str(tmp.resolve())
    return env


def run_family(arm: str, ck: str, fam: str, gpu: int, force: bool) -> dict[str, Any]:
    payload = load_payload(arm, ck)
    if task_done(payload, fam) and not force:
        rec = payload["tasks"][fam]
        return {"status": "skip_existing", "arm": arm, "checkpoint": ck, "family": fam, "score": rec.get("score"), "per_target": rel(payload_path(arm, ck))}

    spec = FAMILIES[fam]
    run_dir = ARMS[arm]["run_dir"]
    model_path = run_dir / "hf_model" / ck
    target = f"{arm}_{ck}"
    task_out = OUT_ROOT / "official_outputs" / target / fam
    log = OUT_ROOT / "logs" / target / f"gpu_{fam}.log"
    task_out.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    revision = f"step293_{target}_{fam}"
    argv = [
        sys.executable, "-B", "-m", "evaluation_pipeline.sentence_zero_shot.run",
        "--model_path_or_name", str(model_path.resolve()),
        "--backend", "mlm",
        "--task", str(spec["task"]),
        "--data_path", str(pathlib.Path(spec["data"]).resolve()),
        "--revision_name", revision,
        "--save_predictions",
        "--batch_size", str(spec["batch"]),
        "--non_causal_batch_size", "64",
        "--output_dir", str(task_out.resolve()),
    ]
    header = {
        "event": "gpu_family_start", "utc": now(), "arm": arm, "checkpoint": ck,
        "family": fam, "gpu": gpu, "cmd": argv,
        "no_globalpiqa_superglue_aoa_upload_or_leaderboard": True,
    }
    with log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(header, ensure_ascii=False) + "\n")
        fh.flush()
        t0 = time.time()
        rc = -999
        err: str | None = None
        try:
            proc = subprocess.run(argv, cwd=str(STRICT.resolve()), env=gpu_env(target, gpu), stdout=fh, stderr=subprocess.STDOUT, text=True, timeout=int(spec["timeout"]))
            rc = proc.returncode
        except subprocess.TimeoutExpired as exc:
            rc = -124
            err = f"timeout after {spec['timeout']}s: {exc}"
            fh.write(json.dumps({"event": "timeout", "error": err}, ensure_ascii=False) + "\n")
        elapsed = round(time.time() - t0, 3)
        fh.write(json.dumps({"event": "gpu_family_finished", "utc": now(), "returncode": rc, "elapsed_sec": elapsed}, ensure_ascii=False) + "\n")

    score = read_report_score(task_out)
    pred = latest_file(task_out, "predictions.json")
    report = latest_file(task_out, "best_temperature_report.txt")
    rec: dict[str, Any] = {
        "family": fam,
        "task": spec["task"],
        "data_path": rel(pathlib.Path(spec["data"])),
        "revision_name": revision,
        "output_dir": rel(task_out),
        "returncode": rc,
        "elapsed_sec": elapsed,
        "log": rel(log),
        "gpu": gpu,
        "score": score,
    }
    if pred:
        rec["predictions"] = rel(pred)
    if report:
        rec["report"] = rel(report)
    if err:
        rec["error"] = err
    elif rc != 0:
        rec["error"] = f"official zero-shot failed rc={rc}"
    elif score is None:
        rec["error"] = "official zero-shot produced no parseable score"
    payload.setdefault("tasks", {})[fam] = rec
    save_payload(payload)
    result = {
        "status": "chunk_done" if not rec.get("error") else "chunk_failed",
        "arm": arm,
        "checkpoint": ck,
        "family": fam,
        "score": score,
        "returncode": rc,
        "elapsed_sec": elapsed,
        "per_target": rel(payload_path(arm, ck)),
        "predictions": rec.get("predictions"),
        "report": rec.get("report"),
        "log": rel(log),
    }
    write_json(OUT_ROOT / "chunk_results" / f"{arm}_{ck}_{fam}.json", result)
    return result


def default_jobs(include_subdose: bool, include_register: bool, include_incorpus: bool, include_early_subdose: bool) -> list[tuple[str, str, str]]:
    jobs: list[tuple[str, str, str]] = []
    if include_register:
        for arm in ["regmax_adultprose", "regmax_childspeech"]:
            for ck in CHECKPOINTS_DECISIVE:
                for fam in FAMILY_ORDER:
                    jobs.append((arm, ck, fam))
    if include_incorpus:
        for arm in ["incorpus_adultprose"]:
            for ck in CHECKPOINTS_DECISIVE:
                for fam in FAMILY_ORDER:
                    jobs.append((arm, ck, fam))
    if include_subdose:
        cks = CHECKPOINTS_SUBDOSE if include_early_subdose else CHECKPOINTS_DECISIVE
        for arm in ["subdose_quarter", "subdose_half", "subdose_full"]:
            for ck in cks:
                for fam in FAMILY_ORDER:
                    jobs.append((arm, ck, fam))
    # de-duplicate preserving order
    out: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for j in jobs:
        if j not in seen:
            out.append(j); seen.add(j)
    return out


def parse_jobs(raw: list[str] | None) -> list[tuple[str, str, str]]:
    if not raw:
        return []
    out = []
    for s in raw:
        parts = s.split(":")
        if len(parts) != 3:
            raise ValueError(f"bad job spec {s!r}; expected arm:chck_80M:Entity")
        out.append((parts[0], parts[1], parts[2]))
    return out


def readiness(arms: list[str], checkpoints: list[str], require_metrics: bool, partial_no_metrics: set[str] | None = None) -> dict[str, Any]:
    partial_no_metrics = partial_no_metrics or set()
    stat: dict[str, Any] = {}
    for arm in arms:
        run = ARMS[arm]["run_dir"]
        stat[arm] = {
            "run_dir": rel(run),
            "metrics_exists": (run / "scientific_metrics.json").exists(),
            "metrics_required": require_metrics and arm not in partial_no_metrics,
            "checkpoints_present": [ck for ck in checkpoints if (run / "hf_model" / ck).exists()],
        }
        if (run / "scientific_metrics.json").exists():
            try:
                m = read_json(run / "scientific_metrics.json")
                stat[arm].update({k: m.get(k) for k in ["word_exposure", "actual_training_steps", "loss_last", "parameter_count"] if k in m})
            except Exception as exc:
                stat[arm]["metrics_error"] = repr(exc)
    return stat


def main() -> None:
    global OUT_ROOT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out-root", default=str(OUT_ROOT), help="Output root for per-target payloads and official outputs")
    ap.add_argument("--jobs", nargs="*", default=None, help="Explicit arm:checkpoint:family jobs")
    ap.add_argument("--include-subdose", action="store_true")
    ap.add_argument("--include-early-subdose", action="store_true")
    ap.add_argument("--no-register", action="store_true")
    ap.add_argument("--no-incorpus", action="store_true")
    ap.add_argument("--require-metrics", action="store_true", help="Wait for scientific_metrics.json before scoring an arm/checkpoint")
    ap.add_argument("--partial-no-metrics-arms", nargs="*", default=[], help="Arms whose existing partial checkpoints may be scored without scientific_metrics.json")
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--wait-all-before-score", action="store_true", help="Do not score any ready arm until all requested model checkpoints are present")
    ap.add_argument("--wait-gpu-free", action="store_true", help="After model readiness, wait for the selected GPU to be mostly free before scoring")
    ap.add_argument("--max-gpu-used-mib", type=int, default=20000)
    ap.add_argument("--max-gpu-util-pct", type=int, default=30)
    ap.add_argument("--wait-timeout-sec", type=int, default=7200)
    ap.add_argument("--wait-interval-sec", type=int, default=60)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    OUT_ROOT = pathlib.Path(args.out_root)
    if not OUT_ROOT.is_absolute():
        OUT_ROOT = ROOT / OUT_ROOT
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    jobs = parse_jobs(args.jobs) if args.jobs else default_jobs(
        include_subdose=args.include_subdose,
        include_register=not args.no_register,
        include_incorpus=not args.no_incorpus,
        include_early_subdose=args.include_early_subdose,
    )
    for arm, ck, fam in jobs:
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm}")
        if fam not in FAMILIES:
            raise ValueError(f"unknown family {fam}")

    plan_arms = sorted({a for a, _, _ in jobs})
    plan_cks = sorted({c for _, c, _ in jobs})
    plan = {
        "status": "DECISIVE_GPU_PLAN",
        "created_utc": now(),
        "out_root": rel(OUT_ROOT),
        "gpu": args.gpu,
        "n_jobs_requested": len(jobs),
        "jobs": [{"arm": a, "checkpoint": c, "family": f} for a, c, f in jobs],
        "readiness": readiness(plan_arms, plan_cks, args.require_metrics, set(args.partial_no_metrics_arms)),
        "require_metrics": args.require_metrics,
        "partial_no_metrics_arms": args.partial_no_metrics_arms,
        "wait": args.wait,
        "wait_all_before_score": args.wait_all_before_score,
        "wait_gpu_free": args.wait_gpu_free,
        "gpu_state_now": gpu_state(args.gpu),
        "no_training_no_globalpiqa_superglue_aoa_upload_or_leaderboard": True,
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    write_json(OUT_ROOT / "last_plan.json", plan)
    if args.plan_only:
        return

    wait_start = time.time()
    partial_no_metrics = set(args.partial_no_metrics_arms)
    if args.wait_all_before_score:
        while not all_requested_models_ready(jobs, args.require_metrics, partial_no_metrics):
            if not args.wait or time.time() - wait_start > args.wait_timeout_sec:
                print(json.dumps({
                    "status": "DECISIVE_WAIT_ALL_NOT_READY",
                    "utc": now(),
                    "readiness": readiness(sorted({a for a,_,_ in jobs}), sorted({c for _,c,_ in jobs}), args.require_metrics, partial_no_metrics),
                }, indent=2, ensure_ascii=False), flush=True)
                return
            print(json.dumps({
                "event": "wait_all_models",
                "utc": now(),
                "readiness": readiness(sorted({a for a,_,_ in jobs}), sorted({c for _,c,_ in jobs}), args.require_metrics, partial_no_metrics),
            }, indent=2, ensure_ascii=False), flush=True)
            time.sleep(args.wait_interval_sec)
        print(json.dumps({"event": "all_requested_models_ready", "utc": now()}, indent=2), flush=True)

    if args.wait_gpu_free:
        while not gpu_ready(args.gpu, args.max_gpu_used_mib, args.max_gpu_util_pct):
            if not args.wait or time.time() - wait_start > args.wait_timeout_sec:
                print(json.dumps({"status": "DECISIVE_GPU_NOT_FREE", "utc": now(), "gpu_state": gpu_state(args.gpu)}, indent=2), flush=True)
                return
            print(json.dumps({"event": "wait_gpu_free", "utc": now(), "gpu_state": gpu_state(args.gpu)}, indent=2), flush=True)
            time.sleep(args.wait_interval_sec)
        print(json.dumps({"event": "gpu_ready_for_scoring", "utc": now(), "gpu_state": gpu_state(args.gpu)}, indent=2), flush=True)

    pending = list(jobs)
    done: list[dict[str, Any]] = []
    iteration = 0
    while pending:
        iteration += 1
        progressed = False
        next_pending: list[tuple[str, str, str]] = []
        for arm, ck, fam in pending:
            if not model_exists(arm, ck, args.require_metrics, partial_no_metrics):
                next_pending.append((arm, ck, fam))
                continue
            print(f"\n[{len(done)+1}/{len(jobs)}] {arm} {ck} {fam} on gpu={args.gpu}", flush=True)
            try:
                res = run_family(arm, ck, fam, args.gpu, args.force)
            except Exception as exc:
                res = {"status": "exception", "arm": arm, "checkpoint": ck, "family": fam, "error": repr(exc)}
                write_json(OUT_ROOT / "chunk_results" / f"{arm}_{ck}_{fam}_exception.json", res)
            done.append(res)
            progressed = True
            print(json.dumps(res, ensure_ascii=False), flush=True)

        pending = next_pending
        summary = {
            "status": "DECISIVE_GPU_RUNNING" if pending else "DECISIVE_GPU_DONE",
            "updated_utc": now(),
            "n_requested": len(jobs),
            "n_done": len(done),
            "n_pending": len(pending),
            "pending": [{"arm": a, "checkpoint": c, "family": f} for a, c, f in pending[:50]],
            "done_status_counts": {s: sum(1 for r in done if r.get("status") == s) for s in sorted({r.get("status") for r in done})},
            "no_training_no_globalpiqa_superglue_aoa_upload_or_leaderboard": True,
        }
        write_json(OUT_ROOT / "scoring_summary.json", summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
        if not pending:
            break
        if not args.wait:
            break
        if time.time() - wait_start > args.wait_timeout_sec:
            print(json.dumps({"status": "DECISIVE_GPU_WAIT_TIMEOUT", "pending": summary["pending"]}, indent=2), flush=True)
            break
        if not progressed:
            print(json.dumps({"event": "wait_for_models", "utc": now(), "iteration": iteration, "n_pending": len(pending), "readiness": readiness(sorted({a for a,_,_ in pending}), sorted({c for _,c,_ in pending}), args.require_metrics, partial_no_metrics)}, indent=2), flush=True)
            time.sleep(args.wait_interval_sec)


if __name__ == "__main__":
    main()
