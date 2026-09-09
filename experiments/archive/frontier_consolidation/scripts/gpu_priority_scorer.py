#!/usr/bin/env python3
"""research: Priority GPU co-resident scorer for unscored register and sub-dose arms.

Uses the proven research GPU scoring pattern. Runs official BabyLM evaluator
on GPU co-resident with training processes. Small batch sizes minimize memory.

Families: BLiMP, Supplement, EWoK, COMPS, Entity (Reading excluded).
Arms: adultprose register, quarter_1x, half_1x sub-dose.
Checkpoints: 40M, 60M, 80M, 100M (early checkpoints excluded).

No GlobalPIQA, SuperGLUE, AoA, upload, or leaderboard action.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import argparse, json, math, os, pathlib, re, subprocess, sys, time
from typing import Any


def find_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_root()
WS = ROOT / "experiments/archive" / 'frontier_consolidation'
STRICT = ROOT / "experiments/archive" / 'representation_and_objectives' / "data" / "pristine_official_coordinate" / "babylm-eval" / "strict"
EVAL_DATA = STRICT / "evaluation_data" / "full_eval"
NLP_DATA = ROOT / "experiments/archive" / 'initial_model_studies' / "data" / "nltk_data"
TOKENIZER = WS / "data" / "compliant_tokenizer"
OUT_ROOT = WS / "data" / "gpu_priority_eval"

FAMILIES: dict[str, dict[str, Any]] = {
    "BLiMP":      {"task": "blimp",           "data": "blimp_filtered",      "batch": 32},
    "Supplement": {"task": "blimp_supplement", "data": "supplement_filtered", "batch": 32},
    "EWoK":       {"task": "ewok",            "data": "ewok_filtered",       "batch": 32},
    "COMPS":      {"task": "comps",           "data": "comps",               "batch": 32},
    "Entity":     {"task": "entity_tracking", "data": "entity_tracking",     "batch": 64},
}

ARMS: dict[str, str] = {
    "regmax_adultprose": "regmax_adultprose_samefw_deberta100M_seed43022",
    "subdose_quarter":   "subdose_quarter_1x_deberta100M_seed43022",
    "subdose_half":      "subdose_half_1x_deberta100M_seed43022",
    "subdose_full":      "subdose_full_1x_deberta100M_seed43022",
}

DEFAULT_CKS = ["chck_40M", "chck_50M", "chck_60M", "chck_70M", "chck_80M", "chck_90M", "chck_100M"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def parse_score(text: str) -> float | None:
    """Parse accuracy from official best_temperature_report.txt."""
    for pat in [
        r"###\s*AVERAGE[^\n]*\n\s*([0-9]+(?:\.[0-9]+)?)",
        r"AVERAGE\s+ACCURACY\s*\n\s*([0-9]+(?:\.[0-9]+)?)",
    ]:
        m = re.search(pat, text)
        if m:
            v = float(m.group(1))
            if math.isfinite(v) and -5 <= v <= 105:
                return v
    return None


def make_env(gpu: int, cache_dir: pathlib.Path) -> dict[str, str]:
    """Environment for co-resident GPU scoring."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["NLTK_DATA"] = str(NLP_DATA.resolve())
    cache_dir.mkdir(parents=True, exist_ok=True)
    env["HF_HOME"] = str(cache_dir.resolve())
    env["HF_HUB_CACHE"] = str((cache_dir / "hub").resolve())
    env["TRANSFORMERS_CACHE"] = str((cache_dir / "transformers").resolve())
    env["HF_MODULES_CACHE"] = str((cache_dir / "modules").resolve())
    env["HF_DATASETS_CACHE"] = str((cache_dir / "datasets").resolve())
    tmp = cache_dir / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(tmp.resolve())
    return env


def run_one_family(model_path: pathlib.Path, family: str, out_dir: pathlib.Path,
                   gpu: int, timeout: int = 600) -> dict[str, Any]:
    """Run official evaluator for one (model, family) and return result dict."""
    spec = FAMILIES[family]
    data_path = EVAL_DATA / spec["data"]
    task_out = out_dir / "official_outputs" / family
    task_out.mkdir(parents=True, exist_ok=True)
    log = out_dir / "logs" / f"{family}.log"
    log.parent.mkdir(parents=True, exist_ok=True)

    cache_dir = out_dir / "hf_cache"
    env = make_env(gpu, cache_dir)

    argv = [
        sys.executable, "-B", "-m",
        "evaluation_pipeline.sentence_zero_shot.run",
        "--model_path_or_name", str(model_path.resolve()),
        "--backend", "mlm",
        "--task", spec["task"],
        "--data_path", str(data_path.resolve()),
        "--revision_name", f"step292_{family}",
        "--save_predictions",
        "--batch_size", str(spec["batch"]),
        "--non_causal_batch_size", "32",
        "--output_dir", str(task_out.resolve()),
    ]

    t0 = time.time()
    with log.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"event": "start", "utc": now(), "family": family,
                              "model": str(model_path), "gpu": gpu}) + "\n")
        fh.flush()
        proc = subprocess.run(argv, cwd=str(STRICT.resolve()), env=env,
                              stdout=fh, stderr=subprocess.STDOUT,
                              text=True, timeout=timeout)
    elapsed = round(time.time() - t0, 1)

    # Parse score from report
    score = None
    reports = sorted(task_out.rglob("best_temperature_report.txt"))
    if reports:
        score = parse_score(reports[-1].read_text(encoding="utf-8", errors="replace"))

    return {
        "family": family, "score": score, "returncode": proc.returncode,
        "elapsed_sec": elapsed, "log": rel(log),
    }


def compute_aggregates(tasks: dict[str, dict]) -> dict[str, Any]:
    """Compute cheap5, exEntity4, and Entity from scored tasks."""
    scores = {f: t["score"] for f, t in tasks.items()
              if isinstance(t.get("score"), (int, float)) and math.isfinite(t["score"])}
    agg: dict[str, Any] = {"scored_families": list(scores.keys())}
    if scores:
        all_vals = list(scores.values())
        agg["cheap5_noReading"] = round(sum(all_vals) / len(all_vals), 6) if len(all_vals) == 5 else None
        ex_entity = [v for f, v in scores.items() if f != "Entity"]
        agg["exEntity4"] = round(sum(ex_entity) / len(ex_entity), 6) if len(ex_entity) == 4 else None
        agg["Entity"] = scores.get("Entity")
    return agg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--checkpoints", nargs="*", default=DEFAULT_CKS)
    ap.add_argument("--families", nargs="*", default=list(FAMILIES))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=600, help="Per-family timeout in seconds")
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Build job list
    jobs: list[dict[str, Any]] = []
    for arm in args.arms:
        run_name = ARMS.get(arm)
        if not run_name:
            print(f"SKIP unknown arm {arm}", flush=True)
            continue
        run_dir = WS / "training" / "runs" / run_name
        for ck in args.checkpoints:
            model_path = run_dir / "hf_model" / ck
            if not model_path.exists():
                print(f"SKIP {arm} {ck}: model not found", flush=True)
                continue
            for fam in args.families:
                jobs.append({"arm": arm, "checkpoint": ck, "family": fam,
                             "model_path": str(model_path)})

    plan = {
        "status": "GPU_PRIORITY_PLAN",
        "created_utc": now(),
        "total_jobs": len(jobs),
        "arms": args.arms,
        "checkpoints": args.checkpoints,
        "families": args.families,
        "gpu": args.gpu,
        "out_root": rel(OUT_ROOT),
        "no_globalpiqa_superglue_aoa_upload_or_leaderboard": True,
    }
    print(json.dumps(plan, indent=2), flush=True)
    if args.plan_only:
        return

    # Run sequentially
    all_targets: dict[str, dict[str, Any]] = {}
    ok, fail = 0, 0

    for i, job in enumerate(jobs):
        arm, ck, fam = job["arm"], job["checkpoint"], job["family"]
        target_key = f"{arm}_{ck}"
        job_out = OUT_ROOT / target_key

        print(f"\n[{i+1}/{len(jobs)}] {arm} {ck} {fam} (gpu={args.gpu})", flush=True)
        try:
            res = run_one_family(pathlib.Path(job["model_path"]), fam, job_out,
                                args.gpu, args.timeout)
        except subprocess.TimeoutExpired:
            res = {"family": fam, "score": None, "returncode": -1,
                   "elapsed_sec": args.timeout, "error": "timeout"}
        except Exception as exc:
            res = {"family": fam, "score": None, "returncode": -1,
                   "elapsed_sec": 0, "error": repr(exc)}

        if res.get("score") is not None:
            ok += 1
            print(f"  OK: {fam} = {res['score']:.4f} ({res['elapsed_sec']}s)", flush=True)
        else:
            fail += 1
            print(f"  FAIL: {fam} ({res.get('error', 'no score')}) ({res.get('elapsed_sec', 0)}s)",
                  flush=True)

        # Accumulate
        if target_key not in all_targets:
            all_targets[target_key] = {
                "target": target_key, "arm": arm, "checkpoint": ck,
                "run_dir": rel(WS / "training" / "runs" / ARMS[arm]),
                "tasks": {},
            }
        all_targets[target_key]["tasks"][fam] = res
        all_targets[target_key]["aggregates"] = compute_aggregates(all_targets[target_key]["tasks"])

        # Save per-target incrementally
        pt_dir = OUT_ROOT / "per_target"
        pt_dir.mkdir(parents=True, exist_ok=True)
        pt_file = pt_dir / f"{target_key}.json"
        pt_file.write_text(json.dumps(all_targets[target_key], indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")

    # Final summary
    summary = {
        "status": "GPU_PRIORITY_DONE",
        "finished_utc": now(),
        "total_jobs": len(jobs),
        "ok": ok,
        "fail": fail,
        "targets": {k: v.get("aggregates", {}) for k, v in all_targets.items()},
        "no_globalpiqa_superglue_aoa_upload_or_leaderboard": True,
    }
    (OUT_ROOT / "scoring_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
