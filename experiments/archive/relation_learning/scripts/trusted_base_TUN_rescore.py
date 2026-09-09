#!/usr/bin/env python3
"""research: Trusted-loader T/U/N rescore of base43022 on 400 balanced held-out packets.

Reopens the deployed-model base state description suspended since research
(research/052 used unsafe stock loading on adapter-scaled checkpoints).
The T/U/N template instrument measures entity-conditioned state retrieval:
    T: source + true update → does the model predict the correct current state?
    U: source + swapped compatible update → does recency from a foreign entity shift?
    N: source only → what does the model predict without any update context?

Candidate phrases: source_state, original_new_state, swapped_new_state.
Phrase scores: one-token-at-a-time pseudo-log-likelihood normalized by token count.

This script imports key functions from research (build_examples, encode_examples,
score_encoded, pivot_margins, summarize_margins) but replaces the model loading
with trusted adapter-aware loading and identity preambles.

Usage:
    python trusted_base_TUN_rescore.py [--device cpu] [--plan-only]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import sys
import time
from typing import Any

# ---------- HF cache setup BEFORE transformers import ----------
ROOT = pathlib.Path.cwd()
OUT_DEFAULT = ROOT / "experiments/archive/relation_learning/data/trusted_base_TUN_rescore"
CACHE_BASE = OUT_DEFAULT / "hf_cache"
for _sub in ["hf_home", "hf_home/hub", "transformers", "modules", "datasets", "tmp"]:
    (CACHE_BASE / _sub).mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(CACHE_BASE / "hf_home")
os.environ["HF_HUB_CACHE"] = str(CACHE_BASE / "hf_home/hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(CACHE_BASE / "hf_home/hub")
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_BASE / "transformers")
os.environ["HF_MODULES_CACHE"] = str(CACHE_BASE / "modules")
os.environ["HF_DATASETS_CACHE"] = str(CACHE_BASE / "datasets")
os.environ["TMPDIR"] = str(CACHE_BASE / "tmp")

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

# ---------- Import research reusable functions ----------
SCRIPTS_DIR = ROOT / "experiments/archive/relation_learning/scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
from score_state_margin_probe import (
    build_examples, encode_examples, score_encoded,
    pivot_margins, summarize_margins,
    read_jsonl, write_csv, write_json, now_utc, rel,
)

# ---------- Paths ----------
PROBE_BALANCED = ROOT / "experiments/archive/relation_learning/data/state_margin_probe/state_margin_probe_balanced_heldout.jsonl"
BASE_43022_CHCK = ROOT / "experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_100M"


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def trusted_load_model(model_path: pathlib.Path, device: torch.device) -> tuple[Any, dict]:
    """Load adapter-scaled model with trust_remote_code=True and record identity."""
    model = AutoModelForMaskedLM.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float32,
    )
    # Identity preamble
    total_params = sum(p.numel() for p in model.parameters())
    adapter_params = 0
    for name, p in model.named_parameters():
        if "adapter" in name.lower():
            adapter_params += p.numel()
    config = model.config
    identity = {
        "checkpoint_path": rel(model_path),
        "loaded_class": type(model).__module__ + "." + type(model).__qualname__,
        "total_params_loaded": total_params,
        "adapter_params_loaded": adapter_params,
        "architectures": getattr(config, "architectures", []),
        "model_type": getattr(config, "model_type", "unknown"),
        "auto_map_present": hasattr(config, "auto_map") and bool(config.auto_map),
        "adapter_enabled_config": getattr(config, "adapter_enabled", False),
        "adapter_scale_config": getattr(config, "adapter_scale", None),
        "adapter_bottleneck_config": getattr(config, "adapter_bottleneck", None),
        "trust_remote_code": True,
        "local_files_only": True,
    }
    model.eval().to(device)
    return model, identity


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--max-items", type=int, default=0, help="Smoke test: use first N packets")
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    out_dir = OUT_DEFAULT
    out_dir.mkdir(parents=True, exist_ok=True)

    # Probe
    if not PROBE_BALANCED.exists():
        raise FileNotFoundError(f"Probe not found: {PROBE_BALANCED}")
    
    plan = {
        "status": "TRUSTED_BASE_TUN_RESCORE_PLAN",
        "created_utc": now_utc(),
        "probe_path": rel(PROBE_BALANCED),
        "probe_exists": PROBE_BALANCED.exists(),
        "probe_sha256": sha256_file(PROBE_BALANCED),
        "model_path": rel(BASE_43022_CHCK),
        "model_exists": BASE_43022_CHCK.exists(),
        "max_items": args.max_items,
        "slot_modes": ["template"],
        "device": args.device,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "measurement": "T/U/N template slot; trusted adapter-aware loading; candidates source_state/original_new_state/swapped_new_state; one-token-at-a-time PLL per candidate token.",
        "purpose": "Reopen the base43022 state-margin characterization under trusted loading (research/052 used unsafe stock loader). This is both the deployed-model base description and the before-readout for the natural-packet answer-slot/private-adapter cell.",
    }
    write_json(out_dir / "plan.json", plan)
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if args.plan_only:
        return

    # Load probe
    rows = read_jsonl(PROBE_BALANCED)
    if args.max_items > 0:
        rows = rows[:args.max_items]
    n_packets = len(rows)
    probe_set = "balanced_heldout"

    # Build T/U/N examples
    examples, swap_meta = build_examples(rows, probe_set, ["template"])
    
    # Load tokenizer from model path
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_43022_CHCK), use_fast=True)
    if tokenizer.mask_token_id is None:
        raise RuntimeError("Tokenizer has no mask token")
    
    encoded, encoding_stats = encode_examples(tokenizer, examples, args.max_length)
    write_json(out_dir / "encoding_metadata.json", {
        "status": "ENCODING_READY",
        "created_utc": now_utc(),
        "n_packets": n_packets,
        "encoded_candidate_records": len(encoded),
        "encoding_stats": encoding_stats,
    })
    print(f"Encoded {len(encoded)} candidate records from {n_packets} packets", flush=True)

    # Load model with trusted loader
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    print(f"Loading base43022 with trusted loader on {device}...", flush=True)
    model, identity = trusted_load_model(BASE_43022_CHCK, device)
    write_json(out_dir / "model_identity.json", identity)
    print(json.dumps(identity, indent=2), flush=True)

    # Verify this is the adapter model
    if identity["adapter_params_loaded"] == 0:
        raise RuntimeError(f"Expected adapter params > 0, got 0. Loaded class: {identity['loaded_class']}")
    if identity["total_params_loaded"] != 35463008:
        print(f"WARNING: unexpected total params {identity['total_params_loaded']} (expected 35463008)", flush=True)

    # Score
    started = time.time()
    score_rows = score_encoded(model, encoded, tokenizer, device, args.batch_size)
    scoring_elapsed = time.time() - started
    print(f"Scoring complete in {scoring_elapsed:.1f}s, {len(score_rows)} rows", flush=True)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Attach packet metadata
    packets_by_id = {str(r.get("pair_id")): r for r in rows}
    swap_by_id = {}
    for sr in swap_meta.get("swap_rows", []):
        swap_by_id[str(sr["pair_id"])] = {
            "pair_id": sr.get("swap_pair_id"),
            "new_state": sr.get("swap_new_state"),
            "state_type_new": sr.get("swap_state_type_new"),
        }
    
    for r in score_rows:
        pkt = packets_by_id.get(str(r["packet_id"]), {})
        sw = swap_by_id.get(str(r["packet_id"]), {})
        r.update({
            "model_name": "base43022",
            "seed": "43022",
            "arm": "base",
            "checkpoint": "chck_100M",
            "model_path": rel(BASE_43022_CHCK),
            "gold_state_type": pkt.get("gold_state_type"),
            "competitor_state_type": pkt.get("competitor_state_type"),
            "state_type_source": pkt.get("state_type_source"),
            "state_type_new": pkt.get("state_type_new"),
            "updated_conflict_heuristic": pkt.get("updated_conflict_heuristic"),
            "source_state": pkt.get("source_state"),
            "original_new_state": pkt.get("new_state"),
            "target_entity": pkt.get("target_entity"),
            "swapped_new_state": sw.get("new_state"),
            "swap_pair_id": sw.get("pair_id"),
            "swap_state_type_new": sw.get("state_type_new"),
        })

    score_path = out_dir / "candidate_phrase_score_rows.csv"
    write_csv(score_path, score_rows)

    # Pivot margins
    margin_rows = pivot_margins(score_rows)
    margin_path = out_dir / "raw_TUN_margin_rows.csv"
    write_csv(margin_path, margin_rows)

    # Summarize
    raw_summary = summarize_margins(margin_rows, out_dir)
    write_csv(out_dir / "raw_TUN_margin_summary.csv", raw_summary)

    # Compute gold accuracy and key aggregate margins
    # For UPDATED_USE: gold is new_state, competitor is source_state
    # For DISTRACTOR: gold is source_state, competitor is new_state
    # Under T (true update context): the correct margin direction
    agg = {}
    for ptype in ["UPDATED_USE", "UNCHANGED_DISTRACTOR_USE"]:
        for cond in ["T", "U", "N"]:
            subset = [r for r in margin_rows if r.get("packet_type") == ptype and r.get("condition") == cond]
            if not subset:
                continue
            key = f"{ptype}_{cond}"
            gold_margins = [float(r.get("full_gold_margin_vs_original_competitor", 0)) for r in subset]
            n_correct = sum(1 for m in gold_margins if m > 0)
            agg[key] = {
                "n": len(subset),
                "gold_correct": n_correct,
                "gold_accuracy": n_correct / len(subset),
                "mean_gold_margin": sum(gold_margins) / len(gold_margins),
                "se_gold_margin": (sum((m - sum(gold_margins)/len(gold_margins))**2 for m in gold_margins) / max(1, len(gold_margins) - 1))**0.5 / max(1, len(gold_margins))**0.5,
            }
    write_json(out_dir / "gold_accuracy_by_type_condition.json", agg)

    # Summary
    meta = {
        "status": "TRUSTED_BASE_TUN_RESCORE_DONE",
        "created_utc": now_utc(),
        "scoring_elapsed_sec": round(scoring_elapsed, 2),
        "probe_path": rel(PROBE_BALANCED),
        "n_packets": n_packets,
        "encoded_candidate_records": len(encoded),
        "score_rows": len(score_rows),
        "margin_rows": len(margin_rows),
        "model_identity": identity,
        "gold_accuracy_aggregate": agg,
        "outputs": {
            "plan": rel(out_dir / "plan.json"),
            "model_identity": rel(out_dir / "model_identity.json"),
            "candidate_phrase_score_rows": rel(score_path),
            "raw_TUN_margin_rows": rel(margin_path),
            "raw_TUN_margin_summary": rel(out_dir / "raw_TUN_margin_summary.csv"),
            "gold_accuracy": rel(out_dir / "gold_accuracy_by_type_condition.json"),
        },
    }
    write_json(out_dir / "summary.json", meta)
    print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
