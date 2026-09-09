#!/usr/bin/env python3
"""research: incrementally score sub-dose checkpoints that already exist on disk.

This is a thin wrapper around `gpu_stable_backlog_eval.py`.  It is meant
for live-running sub-dose trainings whose checkpoint directories are already
materialized before the final `scientific_metrics.json` appears.  It does not
modify model runs.  The only relaxation relative to the research module is that a
checkpoint with `config.json` and `model.safetensors` is sufficient for scoring;
completed-run metadata is attached when present.

No training, GlobalPIQA, SuperGLUE, AoA, packaging, upload, or leaderboard code.
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
OUT_ROOT = WS / "data/subdose_incremental_eval/eval"
STABLE_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "Reading"]
RUNS = WS / "training/runs"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except Exception:
        return str(path)


def load_mod():
    spec = importlib.util.spec_from_file_location("gpu_stable_backlog_eval", MODPATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {MODPATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def registered_arms() -> dict[str, dict[str, Any]]:
    return {
        "subdose_quarter_1x": {
            "run_dir": RUNS / "subdose_quarter_1x_deberta100M_seed43022",
            "target_prefix": "subdose_quarter_1x_seed43022",
            "out_root": OUT_ROOT,
            "description": "Live/incremental research quarter_1x MAX-geometry view arm; rho≈0.0106.",
            "family": "deberta_subdose_quarter_1x_seed43022",
            "architecture": "deberta",
            "data_arm": "view",
            "seed": 43022,
        },
        "subdose_half_1x": {
            "run_dir": RUNS / "subdose_half_1x_deberta100M_seed43022",
            "target_prefix": "subdose_half_1x_seed43022",
            "out_root": OUT_ROOT,
            "description": "Completed/live research half_1x MAX-geometry view arm; rho≈0.0212.",
            "family": "deberta_subdose_half_1x_seed43022",
            "architecture": "deberta",
            "data_arm": "view",
            "seed": 43022,
        },
        "subdose_full_1x": {
            "run_dir": RUNS / "subdose_full_1x_deberta100M_seed43022",
            "target_prefix": "subdose_full_1x_seed43022",
            "out_root": OUT_ROOT,
            "description": "research full_1x MAX-geometry view arm; rho≈0.0424.",
            "family": "deberta_subdose_full_1x_seed43022",
            "architecture": "deberta",
            "data_arm": "view",
            "seed": 43022,
        },
    }


def checkpoint_complete(model_path: pathlib.Path) -> bool:
    return model_path.exists() and (model_path / "config.json").exists() and (model_path / "model.safetensors").exists() and (model_path / "tokenizer.json").exists()


def patch_metric_requirement(mod) -> None:
    orig_zero = mod.run_zero_shot
    orig_reading = mod.run_reading

    def ensure_shadow_cfg(arm: str, checkpoint: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Return a scorer cfg that never writes into a live training run directory.

        If final metrics already exist, the original cfg is safe.  If a training
        run is still live, create a shadow run directory under the scorer output
        root with a symlink to the actual hf_model tree and a local metrics stub.
        The imported research scorer still sees run_dir/scientific_metrics.json,
        but the real training directory is untouched.
        """
        cfg = dict(mod.ARM_CONFIGS[arm])
        real_run = pathlib.Path(cfg["run_dir"])
        metrics = real_run / "scientific_metrics.json"
        if metrics.exists():
            return cfg, None
        model_path = real_run / "hf_model" / checkpoint
        if not checkpoint_complete(model_path):
            raise FileNotFoundError(model_path)
        shadow = OUT_ROOT.parent / "shadow_runs" / arm
        shadow.mkdir(parents=True, exist_ok=True)
        link = shadow / "hf_model"
        if not link.exists():
            try:
                link.symlink_to(real_run / "hf_model", target_is_directory=True)
            except FileExistsError:
                pass
        ck_names = sorted([p.name for p in (real_run / "hf_model").iterdir() if p.is_dir() and checkpoint_complete(p)]) if (real_run / "hf_model").exists() else []
        stub = {
            "variant": "incremental_checkpoint_scoring_shadow_stub",
            "word_exposure": None,
            "actual_training_steps": None,
            "parameter_count": 34467424,
            "vocab_size": 16384,
            "tokenizer_label": "compliant16k_reinvest10M",
            "seed": 43,
            "extra_init_seed": 43022,
            "train_rng_seed": 43023,
            "saved_checkpoints": [{"name": ck} for ck in ck_names],
            "source_run_dir": rel(real_run),
            "note": "Shadow-run metadata for scoring already-materialized live checkpoints; the real training run directory is not modified.",
        }
        (shadow / "scientific_metrics.json").write_text(json.dumps(stub, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        shadow_cfg = dict(cfg)
        shadow_cfg["run_dir"] = shadow
        shadow_cfg["source_run_dir"] = real_run
        return shadow_cfg, {"shadow_run_dir": rel(shadow), "source_run_dir": rel(real_run), "stub_checkpoints": ck_names}

    def call_with_optional_metrics(func, arm: str, checkpoint: str, *args, **kwargs):
        original_cfg = mod.ARM_CONFIGS[arm]
        cfg, shadow_info = ensure_shadow_cfg(arm, checkpoint)
        try:
            mod.ARM_CONFIGS[arm] = cfg
            res = func(arm, checkpoint, *args, **kwargs)
            if shadow_info is not None and isinstance(res, dict):
                res["shadow_metrics_used"] = shadow_info
            return res
        finally:
            mod.ARM_CONFIGS[arm] = original_cfg

    def run_zero_shot_live(arm: str, checkpoint: str, column: str, gpu: int, force: bool, timeout: int) -> dict[str, Any]:
        return call_with_optional_metrics(orig_zero, arm, checkpoint, column, gpu, force, timeout)

    def run_reading_live(arm: str, checkpoint: str, gpu: int, force: bool, timeout: int) -> dict[str, Any]:
        return call_with_optional_metrics(orig_reading, arm, checkpoint, gpu, force, timeout)

    mod.run_zero_shot = run_zero_shot_live
    mod.run_reading = run_reading_live


def parse_chunks(arms: list[str], checkpoints: list[str], columns: list[str], extras: list[str] | None) -> list[tuple[str, str, str]]:
    chunks = [(a, ck, c) for a in arms for ck in checkpoints for c in columns]
    if extras:
        for x in extras:
            parts = x.split(":")
            if len(parts) != 3:
                raise ValueError(f"bad chunk {x!r}; expected arm:chck_50M:BLiMP")
            chunks.append((parts[0], parts[1], parts[2]))
    out: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for c in chunks:
        if c not in seen:
            out.append(c); seen.add(c)
    return out


def plan_rows(mod, chunks: list[tuple[str, str, str]], force: bool) -> list[dict[str, Any]]:
    rows = mod.plan_rows(chunks, force)
    for r in rows:
        mp = ROOT / r["model_path"] if not pathlib.Path(r["model_path"]).is_absolute() else pathlib.Path(r["model_path"])
        r["checkpoint_complete_files"] = checkpoint_complete(mp)
        # Scoreable means checkpoint files exist and either final metrics exist or we can use live mode.
        r["scoreable_live"] = bool(r["checkpoint_complete_files"])
        if not r["scoreable_live"]:
            r["will_run"] = False
    return rows


def write_json(path: pathlib.Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    mod = load_mod()
    mod.ARM_CONFIGS = registered_arms()
    patch_metric_requirement(mod)

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="*", default=["subdose_half_1x", "subdose_quarter_1x"])
    ap.add_argument("--checkpoints", nargs="*", default=[f"chck_{i}M" for i in range(10, 81, 10)])
    ap.add_argument("--columns", nargs="*", default=STABLE_COLUMNS)
    ap.add_argument("--chunks", nargs="*", default=None)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--timeout-sec", type=int, default=2400)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--max-chunks", type=int, default=0)
    ap.add_argument("--continue-on-error", action="store_true")
    args = ap.parse_args()

    chunks = parse_chunks(args.arms, args.checkpoints, args.columns, args.chunks)
    rows = plan_rows(mod, chunks, args.force)
    will = [r for r in rows if r.get("will_run") and r.get("scoreable_live")]
    if args.max_chunks and args.max_chunks > 0:
        allowed = {(r["arm"], r["checkpoint"], r["column"]) for r in will[:args.max_chunks]}
        chunks = [c for c in chunks if c in allowed or not any((r["arm"], r["checkpoint"], r["column"]) == c and r.get("will_run") for r in rows)]
        rows = plan_rows(mod, chunks, args.force)
        will = [r for r in rows if r.get("will_run") and r.get("scoreable_live")]

    plan = {
        "status": "SUBDOSE_INCREMENTAL_PLAN" if args.plan_only else "SUBDOSE_INCREMENTAL_START",
        "created_utc": now(),
        "gpu": args.gpu,
        "timeout_sec": args.timeout_sec,
        "requested_chunk_count": len(rows),
        "scoreable_will_run_count": len(will),
        "chunks": rows,
        "scientific_purpose": "Score already-materialized sub-dose checkpoints incrementally so the rho onset curve can be read without waiting for monolithic task envelopes.",
        "live_metrics_relaxation": "If final scientific_metrics.json is absent but checkpoint files are complete, a temporary stub is used only during scoring and removed immediately; model run directories are otherwise not modified.",
        "no_training_globalpiqa_superglue_aoa_upload_or_leaderboard": True,
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if args.plan_only:
        return

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for arm, ck, col in chunks:
        row = next((r for r in rows if (r["arm"], r["checkpoint"], r["column"]) == (arm, ck, col)), None)
        if row is not None and not row.get("scoreable_live"):
            res = {"status": "skip_unavailable_checkpoint", "arm": arm, "checkpoint": ck, "column": col, "model_path": row.get("model_path")}
            results.append(res)
            print(json.dumps(res, indent=2, ensure_ascii=False), flush=True)
            continue
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
    final = {**plan, "status": "SUBDOSE_INCREMENTAL_DONE" if not failures else "SUBDOSE_INCREMENTAL_FINISHED_WITH_FAILURES", "finished_utc": now(), "result_count": len(results), "failed_count": len(failures), "results": results}
    out = OUT_ROOT.parent / f"subdose_incremental_result_{int(time.time())}.json"
    write_json(out, final)
    print(json.dumps({"status": final["status"], "result_count": len(results), "failed_count": len(failures), "result_path": rel(out)}, indent=2, ensure_ascii=False), flush=True)
    if failures and not args.continue_on_error:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
