#!/usr/bin/env python3
"""research: assemble O62065 measured AoA from the completed endpoint extraction.

The O62065 endpoint AoA extraction was completed at
experiments/archive/relation_learning/data/o62065_aoa_endpoint_extract  This script
performs the missing assembly/scoring step without
re-running endpoint extraction.  It reuses the established research shared ancestry
and platform AoA scorer and writes a manifest suitable for strict split-evaluation
admission.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import importlib.util
import json
import pathlib
import shutil
import sys
import time
from typing import Any

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
research = _public_path('experiments/archive/functional_learning/scripts/batched_aoa_assemble_score.py')
SHARED = _public_path('experiments/archive/functional_learning/data/batched_aoa_shared/full_retry')
A02_ENDPOINT = _public_path('experiments/archive/relation_learning/data/o62065_aoa_endpoint_extract')
CKPT = _public_path('experiments/archive/relation_learning/data/repair_ordinary62065_bundle/repaired_ordinary62065_u0080')
ENDPOINT_ROOT = _public_path('experiments/archive/functional_learning/data/o62065_aoa_endpoint_from_extract')
OUT_ROOT = _public_path('experiments/archive/functional_learning/data/o62065_aoa_measured_from_extract')
TARGET = "o62065"
WORDS = 89_168_037


def rel(path: pathlib.Path | str | None) -> str | None:
    if path is None:
        return None
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def import_step085():
    spec = importlib.util.spec_from_file_location("for_step113_o62065", research)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {research}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["for_step113_o62065"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def stage_endpoint_raw() -> pathlib.Path:
    if not (_public_path('experiments/archive/relation_learning/data/o62065_aoa_endpoint_extract/manifest.json')).is_file():
        raise FileNotFoundError(_public_path('experiments/archive/relation_learning/data/o62065_aoa_endpoint_extract/manifest.json'))
    if not (_public_path('experiments/archive/relation_learning/data/o62065_aoa_endpoint_extract/surprisal.json')).is_file():
        raise FileNotFoundError(_public_path('experiments/archive/relation_learning/data/o62065_aoa_endpoint_extract/surprisal.json'))
    dest = _public_path('experiments/archive/functional_learning/data/o62065_aoa_endpoint_from_extract/o62065/full')
    dest.mkdir(parents=True, exist_ok=True)
    for name in ["manifest.json", "surprisal.json", "manifest.md", "physical_gpu_snapshot_pre.json", "physical_gpu_snapshot_post.json"]:
        src = A02_ENDPOINT / name
        if src.is_file():
            shutil.copy2(src, dest / name)
    source_manifest = {
        "status": "O62065_AOA_RAW_STAGED_FROM_A02",
        "created_utc": now(),
        "a02_endpoint_extract_dir": rel(A02_ENDPOINT),
        "staged_endpoint_dir": rel(dest),
        "checkpoint": rel(CKPT),
        "words": WORDS,
        "interpretation": "Raw endpoint surprisal copied for assembly only; endpoint extraction was not re-run.",
    }
    (dest / "source_manifest.json").write_text(json.dumps(source_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return dest


def main() -> None:
    staged = stage_endpoint_raw()
    mod = import_step085()
    mod.ENDPOINTS[TARGET] = {"path": CKPT, "words": WORDS, "description": "repaired ordinary-continuation seed62065 endpoint"}
    if hasattr(mod, "S81") and hasattr(mod.S81, "ENDPOINTS"):
        mod.S81.ENDPOINTS[TARGET] = mod.ENDPOINTS[TARGET]
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = mod.assemble_target(SHARED, ENDPOINT_ROOT, "full", OUT_ROOT, TARGET, allow_incomplete=False)
    manifest_path = _public_path('experiments/archive/functional_learning/data/o62065_aoa_measured_from_extract/o62065/full/aoa_manifest.json')
    report = {
        "status": "O62065_AOA_ASSEMBLED_FROM_A02_EXTRACT",
        "created_utc": now(),
        "staged_raw_endpoint": rel(staged),
        "a02_endpoint_extract": rel(A02_ENDPOINT),
        "checkpoint": rel(CKPT),
        "shared_ancestry": rel(SHARED),
        "manifest": rel(manifest_path),
        "complete_measured_evidence": manifest.get("complete_measured_evidence"),
        "aoa": (manifest.get("score") or {}).get("aoa_leaderboard_score"),
        "raw_correlation": (manifest.get("score") or {}).get("aoa_raw_correlation"),
        "n_results": (manifest.get("assembled") or {}).get("summary", {}).get("n_results"),
        "n_finite": (manifest.get("assembled") or {}).get("summary", {}).get("n_finite"),
        "n_steps": len((manifest.get("assembled") or {}).get("steps") or []),
    }
    out_report = _public_path('experiments/archive/functional_learning/data/o62065_aoa_measured_from_extract/o62065_aoa_from_extract_report.json')
    out_report.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str), flush=True)
    if not manifest.get("complete_measured_evidence"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
