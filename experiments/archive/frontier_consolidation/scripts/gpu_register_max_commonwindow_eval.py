#!/usr/bin/env python3
"""research: GPU-score trained MAX-register DeBERTa arms on stable families.

Thin wrapper around the validated research stable-family GPU scorer.  It registers
only the two research MAX-rho register-displacement arms and scores BLiMP,
Supplement, EWoK, Entity, COMPS, and Reading over chck_10M..chck_80M by default.
It never trains and never runs GlobalPIQA, SuperGLUE, AoA, packaging, upload, or
leaderboard code.
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
OUT_ROOT = WS / "data/register_max_commonwindow_eval/eval"


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
        "regmax_childspeech": {
            "run_dir": base / "regmax_childspeech_samefw_deberta100M_seed43022",
            "target_prefix": "regmax_childspeech_samefw_seed43022",
            "out_root": OUT_ROOT,
            "description": "research MAX-rho same-FineWeb arm removing developmental/speech clean rows; tests opportunity cost of child/speech material under a fixed small-data budget.",
            "family": "deberta_regmax_childspeech_seed43022",
            "architecture": "deberta",
            "data_arm": "view_register_max_childspeech_removed",
            "seed": 43022,
        },
        "regmax_adultprose": {
            "run_dir": base / "regmax_adultprose_samefw_deberta100M_seed43022",
            "target_prefix": "regmax_adultprose_samefw_seed43022",
            "out_root": OUT_ROOT,
            "description": "research MAX-rho same-FineWeb arm removing adult-prose clean rows; tests opportunity cost of adult prose under a fixed small-data budget.",
            "family": "deberta_regmax_adultprose_seed43022",
            "architecture": "deberta",
            "data_arm": "view_register_max_adultprose_removed",
            "seed": 43022,
        },
    }


def parse_chunks(arms: list[str], checkpoints: list[str], columns: list[str], extras: list[str] | None) -> list[tuple[str, str, str]]:
    chunks = [(a, ck, c) for a in arms for ck in checkpoints for c in columns]
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
    ap.add_argument("--arms", nargs="*", default=["regmax_childspeech", "regmax_adultprose"])
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
        "status": "REGISTER_MAX_COMMONWINDOW_PLAN" if args.plan_only else "REGISTER_MAX_COMMONWINDOW_START",
        "created_utc": now(),
        "gpu": args.gpu,
        "timeout_sec": args.timeout_sec,
        "requested_chunk_count": len(rows),
        "will_run_count": len(will),
        "chunks": rows,
        "scientific_purpose": "Score trained MAX-register arms over a common trajectory window. The primary contrast is childspeech-removed minus adultprose-removed at identical admitted FineWeb content and seed43022; clean anchors and proportional MAX view are read downstream.",
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
    final = {**plan, "status": "REGISTER_MAX_COMMONWINDOW_DONE" if not failures else "REGISTER_MAX_COMMONWINDOW_FINISHED_WITH_FAILURES", "finished_utc": now(), "result_count": len(results), "failed_count": len(failures), "results": results}
    out = OUT_ROOT.parent / f"register_max_commonwindow_result_{int(time.time())}.json"
    write_json(out, final)
    print(json.dumps({"status": final["status"], "result_count": len(results), "failed_count": len(failures), "result_path": rel(out)}, indent=2, ensure_ascii=False), flush=True)
    if failures and not args.continue_on_error:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
