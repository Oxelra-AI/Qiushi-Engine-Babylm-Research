#!/usr/bin/env python3
"""Seed the selected reference trajectory cache from a prior exact-hash payload.

This saves re-evaluating the protected reference chck_100M in the research selected
common-grid wrapper. The copied payload retains all task paths and adds an explicit
cache provenance block; it is not a new evaluation.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import time

SRC = pathlib.Path("experiments/archive/frontier_consolidation/data/scale1p75_100M_repaired_merge/staged_full_eval/per_target/scale1p75_100M_seed43022.json")
DST = pathlib.Path("experiments/archive/frontier_consolidation/data/selected_trajectory_eval_scale1p75_seed43022_reference_common2M/eval/per_target/scale1p75_seed43022_reference_chck_100M.json")
REFERENCE_MODEL = pathlib.Path("experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_100M/model.safetensors")
EXPECTED_SHA = "7349475846ef2a193df850cc23876b2a86f99e82b9f051816064e4561b0f6f52"
OUT_DIR = pathlib.Path("experiments/archive/frontier_consolidation/data/reference_cache_seed")


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not SRC.exists():
        raise FileNotFoundError(SRC)
    if not REFERENCE_MODEL.exists():
        raise FileNotFoundError(REFERENCE_MODEL)
    model_sha = sha256_file(REFERENCE_MODEL)
    if model_sha != EXPECTED_SHA:
        raise RuntimeError({"model_sha": model_sha, "expected": EXPECTED_SHA})
    src_payload_sha = sha256_file(SRC)
    payload = json.loads(SRC.read_text(encoding="utf-8"))
    copied = dict(payload)
    copied["target"] = "scale1p75_seed43022_reference_chck_100M"
    copied["cache_seeded_from"] = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_payload": str(SRC),
        "source_payload_sha256": src_payload_sha,
        "source_target": payload.get("target"),
        "reason": "Prior staged full-eval payload is exact-hash matched to the protected reference chck_100M; seeded to avoid duplicate selected-grid GPU evaluation.",
        "reference_model": str(REFERENCE_MODEL),
        "reference_model_sha256": model_sha,
    }
    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text(json.dumps(copied, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    dst_sha = sha256_file(DST)
    manifest = {
        "status": "REFERENCE_CACHE_SEEDED",
        "source_payload": str(SRC),
        "source_payload_sha256": src_payload_sha,
        "destination_payload": str(DST),
        "destination_payload_sha256": dst_sha,
        "reference_model": str(REFERENCE_MODEL),
        "reference_model_sha256": model_sha,
        "expected_selected_target": "scale1p75_seed43022_reference_chck_100M",
        "note": "The selected evaluator will mark this endpoint cached; provenance remains in cache_seeded_from. Scores use the same rounded per-target coordinate as the selected evaluator.",
    }
    (OUT_DIR / "reference_cache_seed_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    ((OUT_DIR.parents[4] / 'research/documents/frontier_consolidation/data/reference_cache_seed/reference_cache_seed_manifest.md')).write_text(
        "# research reference cache seed\n\n"
        f"Seeded `{DST}` from exact-hash source `{SRC}`.\n\n"
        f"Reference model SHA: `{model_sha}`.\n\n"
        "This is a cache seed, not a new evaluation.\n\n"
        f"JSON: `{OUT_DIR / 'reference_cache_seed_manifest.json'}`\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
