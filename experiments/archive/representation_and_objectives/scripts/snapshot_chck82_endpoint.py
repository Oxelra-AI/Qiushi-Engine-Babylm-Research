#!/usr/bin/env python3
"""research: preserve the first above-frontier scale1.75 chck_82M endpoint.

This is a file/provenance preservation script, not a new measurement.  It copies
checkpoint files and the score/integrity evidence into a local immutable
workspace snapshot and records SHA256/size for every preserved file.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

def find_user_root() -> Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
SRC_RUN = USER_ROOT / "experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder"
SRC_CKPT = SRC_RUN / "hf_model/chck_82M"
SNAP_ROOT = USER_ROOT / "experiments/archive/representation_and_objectives/data/chck82_endpoint_snapshot"
SNAP_CKPT = SNAP_ROOT / "checkpoint/chck_82M"
EVIDENCE_ROOT = SNAP_ROOT / "evidence"

EVIDENCE_FILES = [
    USER_ROOT / "experiments/archive/representation_and_objectives/data/scale1p75_chck82_full_verification/summary/scale1p75_chck82_full_verification.json",
    USER_ROOT / "research/documents/representation_and_objectives/data/scale1p75_chck82_full_verification/summary/scale1p75_chck82_full_verification.md",
    USER_ROOT / "experiments/archive/representation_and_objectives/data/scale1p75_chck82_integrity/chck82_candidate_integrity.json",
    USER_ROOT / "experiments/archive/representation_and_objectives/data/scale1p75_checkpoint_sweep/summary/scale1p75_checkpoint_sweep_summary.json",
    USER_ROOT / "research/notes/frontier_consolidation/scale1p75_endpoint_readiness.md",
]
RUN_META_FILES = [
    SRC_RUN / "scientific_metrics.json",
    SRC_RUN / "example_order_manifest.json",
    SRC_RUN / "dynamics_traces.jsonl",
]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(USER_ROOT))
    except Exception:
        return str(p)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def record(path: Path) -> dict[str, Any]:
    return {"path": rel(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def copy_file(src: Path, dst: Path) -> dict[str, Any]:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {"src": rel(src), "dst": rel(dst), **record(dst)}


def main() -> None:
    if not SRC_CKPT.exists():
        raise FileNotFoundError(SRC_CKPT)
    SNAP_ROOT.mkdir(parents=True, exist_ok=True)

    copied_ckpt = []
    for src in sorted(SRC_CKPT.iterdir(), key=lambda p: p.name):
        if src.is_file():
            copied_ckpt.append(copy_file(src, SNAP_CKPT / src.name))

    copied_evidence = []
    for src in EVIDENCE_FILES:
        copied_evidence.append(copy_file(src, EVIDENCE_ROOT / src.name))
    for src in RUN_META_FILES:
        copied_evidence.append(copy_file(src, EVIDENCE_ROOT / "run_meta" / src.name))

    # Extract key score and training facts into the manifest for quick recovery.
    score_path = USER_ROOT / "experiments/archive/representation_and_objectives/data/scale1p75_chck82_full_verification/summary/scale1p75_chck82_full_verification.json"
    score = json.loads(score_path.read_text(encoding="utf-8"))
    integ_path = USER_ROOT / "experiments/archive/representation_and_objectives/data/scale1p75_chck82_integrity/chck82_candidate_integrity.json"
    integrity = json.loads(integ_path.read_text(encoding="utf-8"))
    chck82_rec = integrity.get("saved_checkpoint_records", {}).get("chck_82M", {})

    manifest = {
        "status": "CHCK82_ENDPOINT_SNAPSHOT",
        "created_utc": now(),
        "purpose": "Preserve the first complete above-41.80 scale1.75 chck_82M endpoint and its score evidence before independent reproduction and from-corpus rerun.",
        "source_checkpoint": rel(SRC_CKPT),
        "snapshot_checkpoint": rel(SNAP_CKPT),
        "candidate_model_sha256": next((r["sha256"] for r in copied_ckpt if r["dst"].endswith("model.safetensors")), None),
        "expected_candidate_sha256_step165": "93ceb76adf5a33d349f1de33e988e6ed0c2b2a547dbd92cf83cc952f8e2591b3",
        "score_from_step165": {
            "Overall": score.get("official_overall", {}).get("Overall"),
            "scores": score.get("official_overall", {}).get("scores"),
            "cheap7": score.get("cheap7"),
            "overall_margin_vs_41p8": score.get("overall_margin_vs_41p8"),
            "submit_ready_overall": score.get("official_overall", {}).get("submit_ready_overall"),
        },
        "checkpoint_exposure": {
            "target_word_exposure": chck82_rec.get("target_word_exposure"),
            "actual_cumulative_word_exposure": chck82_rec.get("actual_cumulative_word_exposure"),
        },
        "copied_checkpoint_files": copied_ckpt,
        "copied_evidence_files": copied_evidence,
    }
    if manifest["candidate_model_sha256"] != manifest["expected_candidate_sha256_step165"]:
        raise RuntimeError({"candidate_hash_mismatch": manifest["candidate_model_sha256"], "expected": manifest["expected_candidate_sha256_step165"]})
    (SNAP_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research chck_82M endpoint snapshot",
        "",
        f"- Source checkpoint: `{rel(SRC_CKPT)}`",
        f"- Snapshot checkpoint: `{rel(SNAP_CKPT)}`",
        f"- Model SHA256: `{manifest['candidate_model_sha256']}`",
        f"- research Overall: `{manifest['score_from_step165']['Overall']}`; margin vs 41.80: `{manifest['score_from_step165']['overall_margin_vs_41p8']}`",
        f"- chck_82M actual cumulative word exposure: `{manifest['checkpoint_exposure']['actual_cumulative_word_exposure']}`",
        "",
        "This snapshot protects the endpoint artifact; it does not by itself reproduce the training result or complete the requested scientific principle.",
        "",
        f"Manifest: `{rel(SNAP_ROOT / 'manifest.json')}`",
    ]
    (SNAP_ROOT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "manifest": rel(SNAP_ROOT / "manifest.json"), "snapshot_checkpoint": rel(SNAP_CKPT), "model_sha256": manifest["candidate_model_sha256"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
