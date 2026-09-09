#!/usr/bin/env python3
"""research: GPU-score trained sub-dose DeBERTa arms on stable common-window families.

This is a thin wrapper around the validated research stable-family GPU chunk scorer.
It registers the three research sub-dose DeBERTa runs as arms and scores only the
stable BabyLM families over a common checkpoint window, default chck_10M..chck_80M.

It never trains and never runs GlobalPIQA, SuperGLUE, AoA, packaging, upload, or
leaderboard code.  Run --plan-only before scoring; if models are not trained yet,
the plan reports missing checkpoint paths.
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


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
MODPATH = WS / "scripts/gpu_stable_backlog_eval.py"
OUT_ROOT = WS / "data/subdose_commonwindow_eval/eval"


def load_scoring_module():
    spec = importlib.util.spec_from_file_location("gpu_stable_backlog_eval", MODPATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module spec for {MODPATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except Exception:
        return str(path)


def write_json(path: pathlib.Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def registered_arms() -> dict[str, dict[str, Any]]:
    base = WS / "training/runs"
    return {
        "subdose_quarter_1x": {
            "run_dir": base / "subdose_quarter_1x_deberta100M_seed43022",
            "target_prefix": "subdose_quarter_1x_seed43022",
            "out_root": OUT_ROOT,
            "description": "research quarter_1x MAX-geometry view arm; tests whether fixed-budget admission effect appears by rho≈0.0106.",
            "family": "deberta_subdose_quarter_1x_seed43022",
            "architecture": "deberta",
            "data_arm": "view",
            "seed": 43022,
        },
        "subdose_half_1x": {
            "run_dir": base / "subdose_half_1x_deberta100M_seed43022",
            "target_prefix": "subdose_half_1x_seed43022",
            "out_root": OUT_ROOT,
            "description": "research half_1x MAX-geometry view arm; tests admission curve shape by rho≈0.0212.",
            "family": "deberta_subdose_half_1x_seed43022",
            "architecture": "deberta",
            "data_arm": "view",
            "seed": 43022,
        },
        "subdose_full_1x": {
            "run_dir": base / "subdose_full_1x_deberta100M_seed43022",
            "target_prefix": "subdose_full_1x_seed43022",
            "out_root": OUT_ROOT,
            "description": "research full_1x MAX-geometry view arm; strict row-matched version of the old rho≈0.042 point.",
            "family": "deberta_subdose_full_1x_seed43022",
            "architecture": "deberta",
            "data_arm": "view",
            "seed": 43022,
        },
    }


def parse_chunks(arms: list[str], checkpoints: list[str], columns: list[str], extras: list[str] | None) -> list[tuple[str, str, str]]:
    chunks = [(a, ck, col) for a in arms for ck in checkpoints for col in columns]
    if extras:
        for v in extras:
            parts = v.split(":")
            if len(parts) != 3:
                raise ValueError(f"bad chunk spec {v!r}; expected arm:chck_80M:Entity")
            chunks.append((parts[0], parts[1], parts[2]))
    out: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for c in chunks:
        if c not in seen:
            out.append(c); seen.add(c)
    return out


def main() -> None:
    mod = load_scoring_module()
    mod.ARM_CONFIGS = registered_arms()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="*", default=["subdose_quarter_1x", "subdose_half_1x", "subdose_full_1x"])
    ap.add_argument("--checkpoints", nargs="*", default=[f"chck_{i}M" for i in range(10, 81, 10)])
    ap.add_argument("--columns", nargs="*", default=mod.STABLE_COLUMNS)
    ap.add_argument("--chunks", nargs="*", default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--timeout-sec", type=int, default=2400)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--max-chunks", type=int, default=0)
    ap.add_argument("--continue-on-error", action="store_true")
    args = ap.parse_args()

    chunks = parse_chunks(args.arms, args.checkpoints, args.columns, args.chunks)
    rows = mod.plan_rows(chunks, args.force)
    will = [r for r in rows if r["will_run"]]
    if args.max_chunks and args.max_chunks > 0:
        allowed = {(r["arm"], r["checkpoint"], r["column"]) for r in will[:args.max_chunks]}
        chunks = [c for c in chunks if c in allowed or not next((r for r in rows if (r["arm"], r["checkpoint"], r["column"]) == c), {"will_run": False})["will_run"]]
        rows = mod.plan_rows(chunks, args.force)
        will = [r for r in rows if r["will_run"]]
    plan = {
        "status": "SUBDOSE_COMMONWINDOW_PLAN" if args.plan_only else "SUBDOSE_COMMONWINDOW_START",
        "created_utc": now(),
        "gpu": args.gpu,
        "timeout_sec": args.timeout_sec,
        "requested_chunk_count": len(rows),
        "will_run_count": len(will),
        "chunks": rows,
        "scientific_purpose": "Score trained sub-dose view arms over a common trajectory window so V-C onset can be compared with admitted/displaced content structure under MAX geometry.",
        "no_training_globalpiqa_superglue_aoa_upload_or_leaderboard": True,
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if args.plan_only:
        return
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for arm, ck, col in chunks:
        try:
            res = mod.run_chunk(arm, ck, col, args.gpu, args.force, args.timeout_sec)
            results.append(res)
            if res.get("status") == "chunk_failed":
                failures.append(res)
                if not args.continue_on_error:
                    break
            print(json.dumps(res, indent=2, ensure_ascii=False), flush=True)
        except Exception as exc:
            err = {"status": "chunk_exception", "arm": arm, "checkpoint": ck, "column": col, "error": repr(exc)}
            results.append(err); failures.append(err)
            print(json.dumps(err, indent=2, ensure_ascii=False), flush=True)
            if not args.continue_on_error:
                break
    final = {**plan, "status": "SUBDOSE_COMMONWINDOW_DONE" if not failures else "SUBDOSE_COMMONWINDOW_FINISHED_WITH_FAILURES", "finished_utc": now(), "result_count": len(results), "failed_count": len(failures), "results": results}
    out = OUT_ROOT.parent / f"subdose_commonwindow_result_{int(time.time())}.json"
    write_json(out, final)
    print(json.dumps({"status": final["status"], "result_count": len(results), "failed_count": len(failures), "result_path": rel(out)}, indent=2, ensure_ascii=False), flush=True)
    if failures and not args.continue_on_error:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
