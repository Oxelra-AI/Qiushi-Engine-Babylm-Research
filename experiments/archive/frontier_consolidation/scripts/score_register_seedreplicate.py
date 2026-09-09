#!/usr/bin/env python3
"""research: official-compatible narrow scoring for the register seed replicate.

This wrapper reuses the corrected research GPU scorer, but registers the research
seed43122 register-replication run directories and restricts scoring to the four
broad families needed for the register reversal: BLiMP, Supplement, EWoK, and
COMPS at chck_80M and chck_100M.  It waits for the requested trained arm when
asked.  It does not score Entity, GlobalPIQA, SuperGLUE, or AoA and never uploads
or touches the leaderboard.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import importlib.util
import json
import pathlib
import sys
import time
from typing import Any


def find_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
BASE_SCORER = WS / "scripts/decisive_gpu_scorer.py"
RUNS = WS / "training/runs"
DEFAULT_OUT = WS / "data/register_seed43122_decisive_eval"
FAMILIES = ["BLiMP", "Supplement", "EWoK", "COMPS"]
CHECKPOINTS = ["chck_80M", "chck_100M"]
ARM_KEYS = {
    "childspeech": "regmax_childspeech_seed43122",
    "adultprose": "regmax_adultprose_seed43122",
}
RUN_DIRS = {
    "regmax_childspeech_seed43122": RUNS / "regmax_childspeech_samefw_deberta100M_seed43122",
    "regmax_adultprose_seed43122": RUNS / "regmax_adultprose_samefw_deberta100M_seed43122",
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def write_json(path: pathlib.Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_base():
    spec = importlib.util.spec_from_file_location("decisive_gpu_scorer_for_step299", BASE_SCORER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {BASE_SCORER}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def install_arms(mod, out_root: pathlib.Path) -> None:
    mod.OUT_ROOT = out_root
    mod.ARMS = {
        "regmax_childspeech_seed43122": {
            "run_dir": RUN_DIRS["regmax_childspeech_seed43122"],
            "role": "research seed43122 replicate: identical FineWeb admission, developmental/speech clean rows sacrificed.",
        },
        "regmax_adultprose_seed43122": {
            "run_dir": RUN_DIRS["regmax_adultprose_seed43122"],
            "role": "research seed43122 replicate: identical FineWeb admission, adult-prose clean rows sacrificed.",
        },
    }


def readiness(mod, arm_key: str, require_metrics: bool) -> dict[str, Any]:
    run = RUN_DIRS[arm_key]
    out: dict[str, Any] = {
        "arm_key": arm_key,
        "run_dir": rel(run),
        "metrics_exists": (run / "scientific_metrics.json").exists(),
        "checkpoints_present": [ck for ck in CHECKPOINTS if (run / "hf_model" / ck).exists()],
    }
    if (run / "scientific_metrics.json").exists():
        try:
            m = json.loads((run / "scientific_metrics.json").read_text(encoding="utf-8"))
            out.update({k: m.get(k) for k in ["word_exposure", "actual_training_steps", "loss_last", "parameter_count"]})
        except Exception as exc:
            out["metrics_error"] = repr(exc)
    out["ready"] = all(mod.model_exists(arm_key, ck, require_metrics) for ck in CHECKPOINTS)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, choices=["childspeech", "adultprose"])
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--out-root", default=str(DEFAULT_OUT))
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--wait-timeout-sec", type=int, default=14400)
    ap.add_argument("--wait-interval-sec", type=int, default=120)
    ap.add_argument("--require-metrics", action="store_true", default=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    out_root = pathlib.Path(args.out_root)
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    out_root = out_root / args.arm if out_root.name != args.arm else out_root
    out_root.mkdir(parents=True, exist_ok=True)

    mod = load_base()
    install_arms(mod, out_root)
    arm_key = ARM_KEYS[args.arm]
    jobs = [(arm_key, ck, fam) for ck in CHECKPOINTS for fam in FAMILIES]
    plan = {
        "status": "REGISTER_SEED_REPLICATE_SCORE_PLAN",
        "created_utc": now(),
        "arm": args.arm,
        "arm_key": arm_key,
        "gpu": args.gpu,
        "out_root": rel(out_root),
        "jobs": [{"arm": a, "checkpoint": ck, "family": fam} for a, ck, fam in jobs],
        "readiness": readiness(mod, arm_key, args.require_metrics),
        "scientific_role": "Narrow official-compatible readout of the independent-basin MAX-register pair; this decides whether the seed43022 register reversal reproduces on broad ex-Entity families.",
        "no_entity_globalpiqa_superglue_aoa_upload_or_leaderboard": True,
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    write_json(out_root / "last_plan.json", plan)
    if args.plan_only:
        return

    start = time.time()
    while not all(mod.model_exists(arm_key, ck, args.require_metrics) for ck in CHECKPOINTS):
        if not args.wait:
            print(json.dumps({"status": "REGISTER_SEED_REPLICATE_SCORE_NOT_READY", "utc": now(), "readiness": readiness(mod, arm_key, args.require_metrics)}, indent=2, ensure_ascii=False), flush=True)
            return
        if time.time() - start > args.wait_timeout_sec:
            print(json.dumps({"status": "REGISTER_SEED_REPLICATE_SCORE_WAIT_TIMEOUT", "utc": now(), "readiness": readiness(mod, arm_key, args.require_metrics)}, indent=2, ensure_ascii=False), flush=True)
            return
        rec = {"event": "wait_for_step299_register_training", "utc": now(), "readiness": readiness(mod, arm_key, args.require_metrics)}
        print(json.dumps(rec, indent=2, ensure_ascii=False), flush=True)
        with (out_root / "training_wait.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        time.sleep(args.wait_interval_sec)

    done: list[dict[str, Any]] = []
    for a, ck, fam in jobs:
        print(f"[{len(done)+1}/{len(jobs)}] {a} {ck} {fam} on gpu={args.gpu}", flush=True)
        try:
            res = mod.run_family(a, ck, fam, args.gpu, args.force)
        except Exception as exc:
            res = {"status": "exception", "arm": a, "checkpoint": ck, "family": fam, "error": repr(exc)}
            write_json(out_root / "chunk_results" / f"{a}_{ck}_{fam}_exception.json", res)
        done.append(res)
        print(json.dumps(res, ensure_ascii=False), flush=True)
    summary = {
        "status": "REGISTER_SEED_REPLICATE_SCORE_DONE",
        "finished_utc": now(),
        "arm": args.arm,
        "arm_key": arm_key,
        "n_jobs": len(jobs),
        "done_status_counts": {s: sum(1 for r in done if r.get("status") == s) for s in sorted({str(r.get("status")) for r in done})},
        "results": done,
        "no_entity_globalpiqa_superglue_aoa_upload_or_leaderboard": True,
    }
    write_json(out_root / "scoring_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
