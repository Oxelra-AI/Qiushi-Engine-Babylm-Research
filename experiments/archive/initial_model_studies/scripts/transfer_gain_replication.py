#!/usr/bin/env python3
"""Independent replication of the research relation-specific transfer signal.

The sample is disjoint by WikiAuto pair id from research.  The primary contrast
is fixed in advance: true rewrite transfer gain minus the exact-word-multiset,
block-reordered target control, at the same update size in both WWM seeds.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


STUDY = _public_path('experiments/archive/initial_model_studies')
SCRIPT = _public_path('experiments/archive/initial_model_studies/scripts/transfer_gain_per_word_gate.py')
RESULT = _public_path('experiments/archive/initial_model_studies/data/transfer_gain_per_word_gate.json')
OUT = _public_path('experiments/archive/initial_model_studies/data/transfer_gain_replication.json')
NOTE = _public_path('research/notes/initial_model_studies/transfer_gain_replication.md')


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    started = time.time()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    torch.set_num_threads(4)
    research = load_module(SCRIPT, "for_replication")
    h3 = research.load_module(research.H3_SCRIPT, "for_step309")

    prior = json.loads(RESULT.read_text(encoding="utf-8"))
    excluded = {row["source_pair_id"] for row in prior["pair_manifest"]}
    candidates = research.load_pairs(160, 30901)
    pairs = [row for row in candidates if row["source_pair_id"] not in excluded][:72]
    if len(pairs) != 72:
        raise RuntimeError(f"only {len(pairs)}/72 disjoint pairs")
    targets = [row["target"] for row in pairs]
    shift = len(pairs) // 2
    for index, row in enumerate(pairs):
        row["cross_pair_control"] = targets[(index + shift) % len(pairs)]
        row["cross_pair_jaccard"] = research.jaccard(
            row["source"], row["cross_pair_control"]
        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    results = {}
    for run_name, checkpoint in research.RUNS.items():
        print(json.dumps({"event": "run_start", "run": run_name}), flush=True)
        results[run_name] = research.run_model(
            run_name, checkpoint, pairs, h3, device, learning_rate=0.01
        )
        print(json.dumps({"event": "run_complete", "run": run_name}), flush=True)

    per_run_pass = {}
    for run_name, result in results.items():
        contrast = result["contrasts"]["aligned_minus_same_bag"]
        aligned = result["statistics"]["aligned"]
        raw_rho = result["predictors"]["raw_source_loss_spearman_with_aligned_transfer"]
        per_run_pass[run_name] = (
            contrast["bootstrap_ci95"][0] > 0
            and contrast["fraction_positive"] >= 0.55
            and aligned["fraction_positive"] >= 0.90
            and abs(raw_rho) < 0.30
        )
    if all(per_run_pass.values()):
        decision = "RELATION_SPECIFIC_TRANSFER_GAIN_REPLICATED"
        next_action = (
            "Promote the measurement principle, not a model claim: build a scalable, "
            "coverage-constrained selector and compare true utility with shuffled utility "
            "on a natural-text 60M two-seed screen."
        )
    else:
        decision = "REJECT_TRANSFER_GAIN_PER_WORD_AFTER_REPLICATION"
        next_action = (
            "Close this estimator. Do not select a corpus with raw loss, self gain, or the "
            "current cross-view transfer measurement."
        )

    payload = {
        "status": "TRANSFER_GAIN_REPLICATION_COMPLETE",
        "design": {
            "new_pair_count": len(pairs),
            "disjoint_from_step308": True,
            "excluded_pair_count": len(excluded),
            "primary_contrast": "aligned rewrite minus same-word-multiset block-reordered target",
            "checkpoints": {
                name: str(path.resolve()) for name, path in research.RUNS.items()
            },
            "learning_rate": 0.01,
            "gradient_clip_norm": 1.0,
            "device": str(device),
            "success_rule_each_seed": {
                "primary_ci95_lower": "> 0",
                "primary_fraction_positive": ">= 0.55",
                "aligned_fraction_positive": ">= 0.90",
                "abs_raw_loss_rho": "< 0.30",
            },
            "no_babylm_eval_data": True,
        },
        "provenance": {
            "script": str(_public_path('experiments/archive/initial_model_studies/scripts/transfer_gain_replication.py')),
            "script_sha256": sha256_file(_public_path('experiments/archive/initial_model_studies/scripts/transfer_gain_replication.py')),
            "script": str(_public_path('experiments/archive/initial_model_studies/scripts/transfer_gain_per_word_gate.py')),
            "script_sha256": sha256_file(SCRIPT),
            "result_sha256": sha256_file(RESULT),
            "pair_data_sha256": sha256_file(research.PAIR_DATA),
        },
        "pair_ids": [row["source_pair_id"] for row in pairs],
        "results": results,
        "per_run_pass": per_run_pass,
        "decision": decision,
        "next_action": next_action,
        "elapsed_sec": time.time() - started,
    }
    atomic_json(OUT, payload)

    lines = [
        "# research transfer gain replication",
        "",
        f"Decision: **{decision}**",
        "",
        "The 72 source/rewrite pairs are disjoint by pair id from research.",
        "",
        "| checkpoint | aligned gain | aligned-same-bag mean (CI95) | contrast positive | raw-loss rho | pass |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for run_name, result in results.items():
        aligned = result["statistics"]["aligned"]
        contrast = result["contrasts"]["aligned_minus_same_bag"]
        raw_rho = result["predictors"]["raw_source_loss_spearman_with_aligned_transfer"]
        ci = contrast["bootstrap_ci95"]
        lines.append(
            f"| {run_name} | {aligned['mean']:+.8f} | {contrast['mean']:+.8f} "
            f"([{ci[0]:+.8f}, {ci[1]:+.8f}]) | "
            f"{contrast['fraction_positive']:.3f} | {raw_rho:+.3f} | "
            f"{per_run_pass[run_name]} |"
        )
    lines.extend(
        [
            "",
            f"Next action: {next_action}",
            "",
            f"Evidence JSON: `{OUT}`",
        ]
    )
    _public_path('research/notes/initial_model_studies').mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "decision": decision,
                "per_run_pass": per_run_pass,
                "summary": {
                    run: {
                        "aligned": result["statistics"]["aligned"],
                        "aligned_minus_same_bag": result["contrasts"][
                            "aligned_minus_same_bag"
                        ],
                        "predictors": result["predictors"],
                    }
                    for run, result in results.items()
                },
                "out": str(OUT),
                "elapsed_sec": payload["elapsed_sec"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
