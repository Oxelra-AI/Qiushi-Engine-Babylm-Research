#!/usr/bin/env python3
"""research: multi-seed three-way + reassignment replication.

Runs 3 seeds of the exact same training (answer-only, private-adapter, 80 epochs)
and evaluates three-way cross-source discrimination and source-reassignment on each.
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
import gc
import time

import torch

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
sys.path.insert(0, str(_public_path('experiments/archive/functional_learning/scripts')))

import threeway_and_reassignment_probe as probe  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seeds", type=str, default="40040,40041,40042")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Construct once (shared data)
    v1_pairs = probe.read_jsonl(probe.V1_PAIRS)
    pairs_seed0 = probe.construct_repaired_pairs(v1_pairs, seeds[0])
    tokenizer = probe.AutoTokenizer.from_pretrained(str(probe.MODEL_PATH), local_files_only=True, use_fast=True)

    # Entity familiarity classification
    train_entities = set()
    for p in pairs_seed0:
        if p.get("split") == "train":
            train_entities.add(p["entity_a"])
            train_entities.add(p["entity_b"])

    # Build reassignment contexts (shared)
    reassignment_ctxs = []
    for p in pairs_seed0:
        ctx, valid, _ = probe.construct_reassignment_contexts(p)
        reassignment_ctxs.append(ctx if valid else None)

    all_results = {}
    for seed in seeds:
        print(f"\n{'='*60}\nSEED {seed}\n{'='*60}", flush=True)
        pairs = probe.construct_repaired_pairs(v1_pairs, seed)
        held_pairs = [p for p in pairs if p.get("split") == "held"]
        scoring_rows = probe.build_scoring_rows(pairs)
        train_rows = [r for r in scoring_rows if r.get("split") == "train" and r.get("role") != "NEUTRAL"]

        model, load_info = probe.load_trusted_model(device)
        ident = probe.model_identity(model, load_info)

        # Train
        trainable_info, train_time = probe.train_model(
            model, tokenizer, train_rows, args.epochs, 16, 5e-5, 0.01, 1.0, seed, device, 512
        )

        # Three-way on held
        model.eval()
        held_tw = []
        for p in held_pairs:
            tw = probe.threeway_score_pair(model, tokenizer, p, device, 512)
            held_tw.append(tw)
        tw_summary, tw_records = probe.analyze_threeway(held_tw, held_pairs, f"held_seed{seed}")

        # Decompose by familiarity
        for rec in tw_records:
            ea = next((p["entity_a"] for p in held_pairs if p["pair_id"] == rec["pair_id"]), "")
            eb = next((p["entity_b"] for p in held_pairs if p["pair_id"] == rec["pair_id"]), "")
            if ea not in train_entities and eb not in train_entities:
                rec["familiarity"] = "both_novel"
            elif ea in train_entities and eb in train_entities:
                rec["familiarity"] = "both_familiar"
            else:
                rec["familiarity"] = "one_familiar"

        # Reassignment on held
        valid_held = [(p, c) for p, c in zip(pairs, reassignment_ctxs) if p.get("split") == "held" and c is not None]
        reass_results = []
        for p, ctx in valid_held:
            rr = probe.reassignment_score_pair(model, tokenizer, p, ctx, device, 512)
            reass_results.append(rr)
        reass_pairs = [p for p, c in valid_held]
        reass_summary, reass_records = probe.analyze_reassignment(
            reass_results, reass_pairs,
            [tw for tw, p in zip(held_tw, held_pairs) if any(p["pair_id"] == rp["pair_id"] for rp, _ in valid_held)],
            f"reassignment_seed{seed}"
        )

        # Novel-entity subset
        novel_tw = [r for r in tw_records if r.get("familiarity") == "both_novel"]
        novel_cross = sum(1 for r in novel_tw if r["retain_qa_entity_retrieval"] and r["retain_qb_entity_retrieval"])

        seed_result = {
            "seed": seed,
            "threeway_held": tw_summary,
            "reassignment_held": reass_summary,
            "novel_n": len(novel_tw),
            "novel_cross_source_both": novel_cross,
            "novel_mean_cross": float(probe.finite_mean(
                [r["retain_qa_cross_source_margin"] for r in novel_tw] +
                [r["retain_qb_cross_source_margin"] for r in novel_tw]
            )) if novel_tw else None,
        }
        all_results[f"seed_{seed}"] = seed_result
        print(f"\nSeed {seed}: held cross-source both={tw_summary['n_retain_both_entity_retrieval']}/{tw_summary['n_pairs']}, "
              f"reassignment both={reass_summary['n_swap_both_follow']}/{reass_summary['n_pairs']}, "
              f"novel cross={novel_cross}/{len(novel_tw)}", flush=True)

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Aggregate
    agg = {
        "status": "MULTI_SEED_REPLICATION",
        "seeds": seeds,
        "epochs": args.epochs,
        "per_seed": all_results,
    }
    (out_dir / "multi_seed_summary.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")
    print("\n" + "=" * 60)
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
