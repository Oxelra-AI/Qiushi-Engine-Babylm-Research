#!/usr/bin/env python3
"""Step277b: diagnostic multi-seed pilot WITHOUT common_seen.

Scientific question: does removing common_seen reveal the Z2 ambiguity?
Without common_seen, the only orientation information for held relations
comes from bridge rows. Aligned and inverted should then show opposite
mixed_held_seen accuracy across seeds. If both consistently show high
accuracy, the pretrained encoder already resolves the Z2 ambiguity
(i.e., has a prior about nonce-word transfer direction).

Also tests: with no common_seen, does the model need pretraining to know
seen relations? If mixed accuracy drops to ~0.5 for all arms, the model
relies on common_seen for seen orientation.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

# Import from the pilot script
import sys
sys.path.insert(0, str(_public_path('experiments/archive/representation_and_objectives/training/scripts')))
from symmetry_pilot import (
    AI_LAB_DIR, WORKSPACE, SUBSTRATE_DIR, MODEL_PATH,
    ARMS, EVAL_SUITES,
    load_jsonl, SubstrateDataset, build_model, eval_suite, train_and_eval,
    format_input,
)

OUT_DEFAULT = WORKSPACE / "data" / "symmetry_pilot_noseen"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrate-dir", type=Path, default=SUBSTRATE_DIR)
    ap.add_argument("--model-path", type=Path, default=MODEL_PATH)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--seeds", nargs="*", type=int, default=[27700, 27701, 27702])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path))
    print(f"Device: {device}, tokenizer vocab={tokenizer.vocab_size}", flush=True)

    # Load eval suites (NO common_seen!)
    eval_suites: Dict[str, List[Dict]] = {}
    for suite in EVAL_SUITES:
        p = args.substrate_dir / "eval" / f"{suite}.jsonl"
        if p.exists():
            eval_suites[suite] = load_jsonl(p)

    all_results = []
    for seed in args.seeds:
        for arm in ARMS:
            print(f"\n{'='*60}\nArm: {arm}, seed: {seed}\n{'='*60}", flush=True)
            sup_path = args.substrate_dir / "arms" / arm / "train_supervised.jsonl"
            sup_rows = load_jsonl(sup_path) if sup_path.exists() else []
            # NO common_seen - only arm-specific supervised data
            train_rows = sup_rows

            result = train_and_eval(
                arm_name=arm,
                train_rows=train_rows,
                eval_suites=eval_suites,
                tokenizer=tokenizer,
                model_path=args.model_path,
                device=device,
                seed=seed,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                max_len=args.max_len,
            )
            all_results.append(result)
            # Print key metrics
            hh = result.get("heldheld_unseen_edge_closure", {}).get("accuracy")
            mx = result.get("mixed_held_seen_orientation", {}).get("accuracy")
            sc = result.get("paired_state_conservation", {}).get("accuracy")
            sc_c = result.get("paired_state_conservation", {}).get("acc_changed")
            print(f"  train_acc={result.get('train_acc_last')}, hh={hh}, mixed={mx}, state_cons={sc} (chg={sc_c})", flush=True)

    # Save
    rp = out / "noseen_pilot_results.json"
    with open(rp, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Summary table: arm × seed × key metrics
    sp = out / "noseen_pilot_summary.md"
    lines = ["# Step277b: no-common-seen multi-seed pilot", ""]
    lines.append("| arm | seed | train_acc | hh_closure | mixed_orient | state_chg | state_unchg |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in all_results:
        def g(s, k="accuracy"):
            v = r.get(s, {}).get(k)
            return f"{v:.3f}" if v is not None else "nan"
        ta = r.get("train_acc_last")
        ta_s = f"{ta:.3f}" if ta is not None else "nan"
        lines.append(f"| {r['arm']} | {r['seed']} | {ta_s} | {g('heldheld_unseen_edge_closure')} | {g('mixed_held_seen_orientation')} | {g('paired_state_conservation', 'acc_changed')} | {g('paired_state_conservation', 'acc_unchanged')} |")

    # Cross-seed summary
    lines.append("")
    lines.append("## Cross-seed means")
    lines.append("")
    lines.append("| arm | mean_hh | mean_mixed | std_mixed | mean_state_chg |")
    lines.append("|---|---:|---:|---:|---:|")
    for arm in ARMS:
        arm_results = [r for r in all_results if r["arm"] == arm]
        hhs = [r.get("heldheld_unseen_edge_closure", {}).get("accuracy", 0.5) for r in arm_results]
        mxs = [r.get("mixed_held_seen_orientation", {}).get("accuracy", 0.5) for r in arm_results]
        scs = [r.get("paired_state_conservation", {}).get("acc_changed", 0.5) for r in arm_results]
        lines.append(f"| {arm} | {np.mean(hhs):.3f} | {np.mean(mxs):.3f} | {np.std(mxs):.3f} | {np.mean(scs):.3f} |")

    lines.append("")
    with open(sp, "w") as f:
        f.write("\n".join(lines))

    print(json.dumps({
        "status": "STEP277B_NOSEEN_PILOT_COMPLETE",
        "results": str(rp),
        "summary": str(sp),
        "n_total": len(all_results),
    }, indent=2), flush=True)

if __name__ == "__main__":
    main()
