#!/usr/bin/env python3
"""Alpha0.75 truthful Fast/AoA preflight v2.

CPU-only, safe staging under the new output directory. This avoids modifying the
research staged tree and avoids the failed pattern of replacing runtime-created write
mount directories with symlinks.

What it does:
  1. Build `base_out/hf_model_truthful_alpha075` with symlinks for all 19 official
     Strict-Small fast revisions: chck_1M..80M from the research/slow truthful
     ancestors, chck_90M/chck_100M from the research exact alpha0.75 replay.
  2. Optionally link reusable pre-82M fast prediction files from the chck82 fast
     materializer into `base_out/collate_fast/results/hf_model/<rev>/...`.
  3. Enumerate exactly the alpha-specific fast predictions still needing GPU work
     (normally 14 files: 2 revisions × 7 tasks).
  4. Record the full alpha0.75 carrier path/hash and AoA runner/data paths.

It does not evaluate, collate, or impute AoA.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import time
from typing import Any, Dict

USER_ROOT = pathlib.Path(".").resolve()
WORK = USER_ROOT / "experiments/archive/representation_and_objectives"
FAST_REVISIONS = [f"chck_{i}M" for i in range(1, 10)] + [f"chck_{i * 10}M" for i in range(1, 11)]
FAST_TASKS = ["BLiMP", "Supplement", "EWoK", "Entity", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"]
SOURCE_STAGE = WORK / "data/alpha075_developmental_staging/hf_model"
REPLAY_ALPHA = WORK / "training/runs/alpha075_exact_replay_from82M_seed43022/hf_model_alpha0p75"
CARRIER_MANIFEST = USER_ROOT / "experiments/archive/frontier_consolidation/data/truthful_private_scale_carriers/coherent86_alpha0p75/truthful_coherent86_alpha0p75_carrier_manifest.json"
CHCK82_FAST = WORK / "data/chck82_fast_submission_materialization/collate_fast/results/hf_model"
EVAL_REPO = USER_ROOT / "experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_pipeline"
AOA_RUN = EVAL_REPO / "AoA_word/run.py"
AOA_DATA = USER_ROOT / "experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/aoa"
DEFAULT_OUT = WORK / "data/alpha075_fast_preflight_v2"

FAST_REL = {
    "BLiMP": "blimp/blimp_fast/predictions.json",
    "Supplement": "blimp/supplement_fast/predictions.json",
    "EWoK": "ewok/ewok_fast/predictions.json",
    "Entity": "entity_tracking/entity_tracking_fast/predictions.json",
    "GlobalPIQA_parallel": "global_piqa_parallel/global_piqa_parallel/predictions.json",
    "GlobalPIQA_nonparallel": "global_piqa_nonparallel/global_piqa_nonparallel/predictions.json",
    "Reading": "reading/predictions.json",
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.resolve().relative_to(USER_ROOT))
    except Exception:
        return str(p)


def sha256_file(p: pathlib.Path) -> str | None:
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def clean_link(src: pathlib.Path, dst: pathlib.Path) -> Dict[str, Any]:
    if not src.exists():
        raise FileNotFoundError(src)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(src.resolve(), dst, target_is_directory=src.is_dir())
    return {"mode": "symlink", "src": rel(src), "dst": rel(dst), "is_dir": src.is_dir(), "sha256": sha256_file(src) if src.is_file() else sha256_file(src / "model.safetensors")}


def source_for_revision(rev: str) -> pathlib.Path:
    if rev in {"chck_90M", "chck_100M"}:
        return REPLAY_ALPHA / rev
    return SOURCE_STAGE / rev


def model_identity(model_dir: pathlib.Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {"path": rel(model_dir), "exists": model_dir.exists(), "is_symlink": model_dir.is_symlink(), "model_sha256": sha256_file(model_dir / "model.safetensors"), "config_sha256": sha256_file(model_dir / "config.json")}
    if (model_dir / "config.json").exists():
        try:
            cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
            out.update({"architectures": cfg.get("architectures"), "auto_map": cfg.get("auto_map"), "adapter_scale": cfg.get("adapter_scale"), "private_adapter_scale": cfg.get("private_adapter_scale"), "private_adapter_enabled": cfg.get("private_adapter_enabled"), "vocab_size": cfg.get("vocab_size")})
        except Exception as exc:
            out["config_error"] = repr(exc)
    return out


def chck82_pred(rev: str, task: str) -> pathlib.Path:
    return CHCK82_FAST / rev / "zero_shot" / "mlm" / FAST_REL[task]


def alpha_pred(base_out: pathlib.Path, rev: str, task: str) -> pathlib.Path:
    return base_out / "collate_fast" / "results" / "hf_model" / rev / "zero_shot" / "mlm" / FAST_REL[task]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-out", default=str(DEFAULT_OUT))
    ap.add_argument("--link-reusable-preds", action="store_true")
    args = ap.parse_args()
    base_out = pathlib.Path(args.base_out)
    model_tree = base_out / "hf_model_truthful_alpha075"
    base_out.mkdir(parents=True, exist_ok=True)

    model_links = {}
    model_identities = {}
    missing_models = []
    for rev in FAST_REVISIONS:
        src = source_for_revision(rev)
        dst = model_tree / rev
        if src.exists() and (src / "model.safetensors").exists():
            model_links[rev] = clean_link(src, dst)
        else:
            missing_models.append({"revision": rev, "source": rel(src)})
        model_identities[rev] = model_identity(dst)

    reusable = []
    linked = []
    missing_reusable = []
    needs_generation = []
    existing_alpha = []
    for rev in FAST_REVISIONS:
        for task in FAST_TASKS:
            dst = alpha_pred(base_out, rev, task)
            if dst.exists() and not dst.is_symlink():
                existing_alpha.append({"revision": rev, "task": task, "path": rel(dst), "sha256": sha256_file(dst), "size_bytes": dst.stat().st_size})
                continue
            if rev not in {"chck_90M", "chck_100M"}:
                src = chck82_pred(rev, task)
                rec = {"revision": rev, "task": task, "source": rel(src), "target": rel(dst), "source_exists": src.exists(), "sha256": sha256_file(src) if src.exists() else None, "size_bytes": src.stat().st_size if src.exists() else None}
                if src.exists():
                    reusable.append(rec)
                    if args.link_reusable_preds:
                        linked.append(clean_link(src, dst))
                else:
                    missing_reusable.append(rec)
            else:
                needs_generation.append({"revision": rev, "task": task, "model": rel(model_tree / rev), "expected_prediction": rel(dst)})

    carrier = json.loads(CARRIER_MANIFEST.read_text(encoding="utf-8"))
    errors = []
    if missing_models:
        errors.append("missing_model_revisions")
    if missing_reusable:
        errors.append("missing_pre82_reusable_predictions")
    if not AOA_RUN.exists() or not AOA_DATA.exists():
        errors.append("missing_aoa_runner_or_data")
    status = "ALPHA075_FAST_PREFLIGHT_V2_PASS" if not errors else "ALPHA075_FAST_PREFLIGHT_V2_NEEDS_REPAIR"
    summary = {
        "status": status,
        "created_utc": now(),
        "purpose": "Prepare truthful alpha0.75 fast/AoA evaluation without model substitution; CPU-only preflight, no scoring launched.",
        "base_out": rel(base_out),
        "model_tree": rel(model_tree),
        "model_links": model_links,
        "model_identities": model_identities,
        "missing_models": missing_models,
        "carrier_manifest": {"path": rel(CARRIER_MANIFEST), "status": carrier.get("status"), "carrier_path": carrier.get("carrier_path"), "carrier_sha256": carrier.get("carrier_sha256"), "carrier_size_bytes": carrier.get("carrier_size_bytes"), "overall_with_aoa0": carrier.get("score_arithmetic_candidate_native", {}).get("overall_with_aoa0")},
        "aoa_runner": {"path": rel(AOA_RUN), "exists": AOA_RUN.exists(), "aoa_data": rel(AOA_DATA), "aoa_data_exists": AOA_DATA.exists(), "note": "AoA still must be run with min_context=0 over official 8005 contexts per revision; this preflight does not score it."},
        "fast_revisions": FAST_REVISIONS,
        "fast_tasks": FAST_TASKS,
        "total_fast_prediction_files_required": len(FAST_REVISIONS) * len(FAST_TASKS),
        "pre82_reusable_prediction_files": reusable,
        "linked_reusable_prediction_files": linked,
        "missing_pre82_reusable_prediction_files": missing_reusable,
        "existing_alpha_prediction_files": existing_alpha,
        "needs_alpha_generation": needs_generation,
        "generation_count_if_reuse_pre82": len(needs_generation),
        "errors": errors,
        "next_expensive_action": "After compact directional tasks finish or a GPU is free, generate the 14 chck_90M/chck_100M fast prediction files from model_tree, run official min_context=0 AoA for the truthful 19-revision tree, then collate with the existing alpha0.75 full carrier. Do not use alpha1.0, ordinary states, or scalar AoA zero.",
    }
    (base_out / "alpha075_fast_preflight_v2_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    md = ["# Alpha0.75 Fast/AoA preflight v2", "", f"Status: `{status}`", f"Model tree: `{rel(model_tree)}`", f"Missing models: {len(missing_models)}", f"Reusable pre-82 fast prediction files: {len(reusable)}", f"Linked reusable prediction files: {len(linked)}", f"Missing reusable prediction files: {len(missing_reusable)}", f"Alpha-specific fast prediction files to generate: {len(needs_generation)}", f"AoA runner exists: {AOA_RUN.exists()}; AoA data exists: {AOA_DATA.exists()}", "", f"JSON: `{rel(base_out / 'alpha075_fast_preflight_v2_summary.json')}`"]
    (base_out / "alpha075_fast_preflight_v2_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "base_out": rel(base_out), "model_tree": rel(model_tree), "missing_models": len(missing_models), "reusable_pre82": len(reusable), "linked_reusable": len(linked), "missing_reusable": len(missing_reusable), "needs_alpha_generation": len(needs_generation), "errors": errors}, indent=2), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
