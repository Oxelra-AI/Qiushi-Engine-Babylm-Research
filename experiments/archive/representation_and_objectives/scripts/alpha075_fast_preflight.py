#!/usr/bin/env python3
"""Alpha0.75 truthful Fast/AoA preflight and staging repair.

This CPU script does not evaluate the model. It repairs the research staged model tree
by adding truthful chck_90M/chck_100M links from the research exact replay, checks all
19 official Strict-Small fast revisions, records existing reusable chck82 fast
prediction files for pre-82M ancestors, and enumerates alpha-specific predictions
that still need GPU generation.

It intentionally does not hand-fill AoA and does not borrow ordinary or alpha1.0
post-82M states.
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
A01 = USER_ROOT / "experiments/archive/representation_and_objectives"
WORK = A01
FAST_REVISIONS = [f"chck_{i}M" for i in range(1, 10)] + [f"chck_{i * 10}M" for i in range(1, 11)]
FAST_TASKS = ["BLiMP", "Supplement", "EWoK", "Entity", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"]
STAGED_ROOT = WORK / "data/alpha075_developmental_staging/hf_model"
REPLAY_ALPHA_ROOT = WORK / "training/runs/alpha075_exact_replay_from82M_seed43022/hf_model_alpha0p75"
CARRIER_MANIFEST = USER_ROOT / "experiments/archive/frontier_consolidation/data/truthful_private_scale_carriers/coherent86_alpha0p75/truthful_coherent86_alpha0p75_carrier_manifest.json"
CHCK82_FAST_ROOT = WORK / "data/chck82_fast_submission_materialization/collate_fast/results/hf_model"
DEFAULT_OUT = WORK / "data/alpha075_fast_preflight"

# These are relative to collate_fast/results/hf_model/<revision>/zero_shot/mlm/...
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


def sha256_file(p: pathlib.Path) -> str | None:
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.resolve().relative_to(USER_ROOT))
    except Exception:
        return str(p)


def link_dir(src: pathlib.Path, dst: pathlib.Path) -> Dict[str, Any]:
    if not src.exists():
        raise FileNotFoundError(src)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(src.resolve(), dst, target_is_directory=True)
    return {"mode": "symlink", "src": rel(src), "dst": rel(dst), "model_sha256": sha256_file(src / "model.safetensors"), "config_sha256": sha256_file(src / "config.json")}


def model_identity(model_dir: pathlib.Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {"path": rel(model_dir), "exists": model_dir.exists(), "is_symlink": model_dir.is_symlink()}
    if model_dir.exists():
        out["model_sha256"] = sha256_file(model_dir / "model.safetensors")
        out["config_sha256"] = sha256_file(model_dir / "config.json")
        try:
            cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
            out["architectures"] = cfg.get("architectures")
            out["auto_map"] = cfg.get("auto_map")
            out["adapter_scale"] = cfg.get("adapter_scale")
            out["private_adapter_scale"] = cfg.get("private_adapter_scale")
            out["private_adapter_enabled"] = cfg.get("private_adapter_enabled")
            out["vocab_size"] = cfg.get("vocab_size")
        except Exception as exc:
            out["config_error"] = repr(exc)
    return out


def existing_chck82_pred(rev: str, task: str) -> pathlib.Path:
    return CHCK82_FAST_ROOT / rev / "zero_shot" / "mlm" / FAST_REL[task]


def expected_alpha_pred(base_out: pathlib.Path, rev: str, task: str) -> pathlib.Path:
    return base_out / "collate_fast" / "results" / "hf_model" / rev / "zero_shot" / "mlm" / FAST_REL[task]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-out", default=str(DEFAULT_OUT))
    ap.add_argument("--repair-stage", action="store_true", help="Symlink chck_90M/chck_100M from research into research staged model tree")
    args = ap.parse_args()
    out = pathlib.Path(args.base_out)
    out.mkdir(parents=True, exist_ok=True)

    links = {}
    if args.repair_stage:
        for rev in ["chck_90M", "chck_100M"]:
            links[rev] = link_dir(REPLAY_ALPHA_ROOT / rev, STAGED_ROOT / rev)

    carrier = json.loads(CARRIER_MANIFEST.read_text(encoding="utf-8"))
    revisions = {rev: model_identity(STAGED_ROOT / rev) for rev in FAST_REVISIONS}
    missing_revisions = [rev for rev, rec in revisions.items() if not rec["exists"] or rec.get("model_sha256") is None]

    reusable_pre82 = []
    missing_reusable_pre82 = []
    needs_alpha_generation = []
    existing_alpha_preds = []
    for rev in FAST_REVISIONS:
        for task in FAST_TASKS:
            alpha_p = expected_alpha_pred(out, rev, task)
            if alpha_p.exists():
                existing_alpha_preds.append({"revision": rev, "task": task, "path": rel(alpha_p), "sha256": sha256_file(alpha_p), "size_bytes": alpha_p.stat().st_size})
                continue
            # pre-private official fast revisions are chck_1M..chck_80M and identical to chck82/scale1.75 ladder.
            if rev not in ["chck_90M", "chck_100M"]:
                src = existing_chck82_pred(rev, task)
                rec = {"revision": rev, "task": task, "source": rel(src), "target": rel(alpha_p), "source_exists": src.exists(), "sha256": sha256_file(src) if src.exists() else None, "size_bytes": src.stat().st_size if src.exists() else None}
                if src.exists():
                    reusable_pre82.append(rec)
                else:
                    missing_reusable_pre82.append(rec)
            else:
                needs_alpha_generation.append({"revision": rev, "task": task, "model": rel(STAGED_ROOT / rev), "expected_prediction": rel(alpha_p)})

    status = "ALPHA075_FAST_PREFLIGHT_PASS" if not missing_revisions and not missing_reusable_pre82 else "ALPHA075_FAST_PREFLIGHT_NEEDS_REPAIR"
    summary = {
        "status": status,
        "created_utc": now(),
        "purpose": "CPU preflight for truthful alpha0.75 Fast/AoA materialization; no evaluation launched and no missing AoA is imputed.",
        "base_out": rel(out),
        "staged_root": rel(STAGED_ROOT),
        "replay_alpha_root": rel(REPLAY_ALPHA_ROOT),
        "repair_stage_links": links,
        "carrier_manifest": {"path": rel(CARRIER_MANIFEST), "status": carrier.get("status"), "carrier_path": carrier.get("carrier_path"), "carrier_sha256": carrier.get("carrier_sha256"), "overall_with_aoa0": carrier.get("score_arithmetic_candidate_native", {}).get("overall_with_aoa0")},
        "fast_revisions": FAST_REVISIONS,
        "fast_tasks": FAST_TASKS,
        "revision_identities": revisions,
        "missing_revisions": missing_revisions,
        "pre82_reusable_prediction_files": reusable_pre82,
        "missing_pre82_reusable_prediction_files": missing_reusable_pre82,
        "existing_alpha_prediction_files": existing_alpha_preds,
        "needs_alpha_generation": needs_alpha_generation,
        "generation_count_if_reuse_pre82": len(needs_alpha_generation),
        "total_prediction_files_required": len(FAST_REVISIONS) * len(FAST_TASKS),
        "next_expensive_action": "If GPU is free and scientific branch is not delayed, link/copy pre-82M prediction files into base_out/collate_fast/results/hf_model, generate only chck_90M/chck_100M fast predictions on the truthful alpha0.75 staged models, run official min_context=0 AoA over all 19 revisions, then collate fail-closed with the existing alpha0.75 full carrier.",
    }
    (out / "alpha075_fast_preflight_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    md = ["# Alpha0.75 Fast/AoA preflight", "", f"Status: `{status}`", f"Missing revisions: {missing_revisions}", f"Reusable pre-82 prediction files: {len(reusable_pre82)}", f"Missing reusable pre-82 prediction files: {len(missing_reusable_pre82)}", f"Alpha-specific fast prediction files to generate: {len(needs_alpha_generation)}", "", f"JSON: `{out / 'alpha075_fast_preflight_summary.json'}`"]
    (out / "alpha075_fast_preflight_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "base_out": rel(out), "missing_revisions": missing_revisions, "reusable_pre82": len(reusable_pre82), "missing_reusable_pre82": len(missing_reusable_pre82), "needs_alpha_generation": len(needs_alpha_generation)}, indent=2), flush=True)
    if status.endswith("NEEDS_REPAIR"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
