#!/usr/bin/env python3
"""research: score seed43122 split-control DeBERTa arms with existing probes.

This wrapper reuses the research held-out copy / compact-rewrite T-U / Entity-cue
ablation framework, but points it at seed43122 split runs and the matched
seed43122 CLEAN baseline.  It writes one output directory per split arm so the
results can be joined with original seed43122 C/R/V rows.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import pathlib
import sys

ROOT0 = _public_path('experiments/archive/relation_learning/scripts/score_split_seed43122.py')
ROOT = _PUBLIC_ROOT

sys.path.insert(0, str(ROOT / "experiments/archive/relation_learning/scripts"))
import heldout_copy_rewrite_entity_ablation as base  # noqa: E402

REPRESENTATION_FRONTIER_STUDIES_RUNS = ROOT / "experiments/archive/frontier_consolidation/training/runs"
FUNCTIONAL_RELATION_STUDIES_RUNS = ROOT / "experiments/archive/relation_learning/training/runs"
OUT_BASE = ROOT / "experiments/archive/relation_learning/data"

CLEAN_43122 = REPRESENTATION_FRONTIER_STUDIES_RUNS / "full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43122"

SPLIT_ARMS = {
    "repeat_split": {
        "arm": "D_RS_43122",
        "path": FUNCTIONAL_RELATION_STUDIES_RUNS / "full_p2c_c2p_abs_repeat_split_dose2p64x_rowholdout_deberta100M_seed43122",
        "out": OUT_BASE / "split_repeat_split_seed43122_probes",
    },
    "view_split": {
        "arm": "D_VS_43122",
        "path": FUNCTIONAL_RELATION_STUDIES_RUNS / "full_p2c_c2p_abs_view_split_dose2p64x_rowholdout_deberta100M_seed43122",
        "out": OUT_BASE / "split_view_split_seed43122_probes",
    },
}


def ck_ready(run_dir: pathlib.Path, ck: str) -> bool:
    d = run_dir / "hf_model" / ck
    return (d / "model.safetensors").exists() or (d / "pytorch_model.bin").exists()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, choices=["repeat_split", "view_split", "both"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=48)
    args = ap.parse_args()

    selected = list(SPLIT_ARMS) if args.arm == "both" else [args.arm]
    done = []
    skipped = []
    for arm_key in selected:
        cfg = SPLIT_ARMS[arm_key]
        missing = [ck for ck in ["chck_80M", "chck_90M", "chck_100M"] if not ck_ready(cfg["path"], ck)]
        if missing:
            skipped.append({"arm": arm_key, "run_dir": str(cfg["path"]), "missing": missing})
            print(f"[SKIP] {arm_key}: missing {missing} at {cfg['path']}", flush=True)
            continue
        if not CLEAN_43122.exists():
            raise FileNotFoundError(CLEAN_43122)

        print(f"\n=== Scoring seed43122 {arm_key} ({cfg['arm']}) ===", flush=True)
        base.ARM_CONFIGS.clear()
        base.ARM_CONFIGS.update({
            cfg["arm"]: cfg["path"],
            "D_C_43122": CLEAN_43122,
        })
        base.OUT_DEFAULT = cfg["out"]
        sys.argv = [
            "score_split_seed43122.py",
            "--arms", cfg["arm"], "D_C_43122",
            "--checkpoints", "chck_80M", "chck_90M", "chck_100M",
            "--gpu", args.device.replace("cuda:", ""),
            "--batch-size", str(args.batch_size),
        ]
        base.main()
        done.append({"arm": arm_key, "output": str(cfg["out"]), "run_dir": str(cfg["path"])})
        print(f"[DONE] {arm_key} -> {cfg['out']}", flush=True)

    print(json.dumps({"status": "SPLIT_SEED43122_SCORING_WRAPPER_DONE", "done": done, "skipped": skipped}, indent=2), flush=True)


if __name__ == "__main__":
    main()
