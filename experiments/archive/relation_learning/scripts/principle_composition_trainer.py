#!/usr/bin/env python3
"""research: Principle-guided private-phase composition trainer.

Scientific design:
  The data composition IS the principle. Instead of designing special objectives
  (which have failed four times in the binding route), this trainer uses the
  measured relation→column map to compose the private phase from two relation types:
  
  1. Coherent replay (ordinary text): broad benchmark preservation
  2. Correct-correspondence pairs (aligned restatement): context-supported columns
  
  The objective is standard CE + neutral KL, identical to coherent86's training.
  The only change is the data composition: ~21% of words come from probe-clean
  aligned-restatement pairs, interleaved with ordinary coherent replay text.
  
  The mixed JSONL is pre-materialized by materialize_principle_composition.py.

Uses the identical FrozenSlowPrivateDebertaV2ForMaskedLM architecture and
hyperparameters as coherent86 (research), differing only in data composition.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import os
import pathlib
import sys
import time

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/relation_learning')
WS = _public_path('experiments/archive/relation_learning')
REPRESENTATION_FRONTIER_STUDIES_SCRIPTS = _public_path('experiments/archive/frontier_consolidation/scripts')
sys.path.insert(0, str(REPRESENTATION_FRONTIER_STUDIES_SCRIPTS))

import frozen82_fastpath_replay_trainer as S150  # noqa: E402

# ── defaults matching coherent86 ──
CHCK_82M = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M')
MIXED_JSONL = _public_path('experiments/archive/relation_learning/data/principle_composition/principle_composition_mixed.jsonl')
METADATA = _public_path('experiments/archive/relation_learning/data/principle_composition/metadata.json')

def main():
    t0 = time.time()
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True, help="Output directory for checkpoints and metrics")
    ap.add_argument("--endpoint", default=str(CHCK_82M), help="Path to chck_82M endpoint")
    ap.add_argument("--mixed-jsonl", default=str(MIXED_JSONL), help="Pre-mixed data JSONL")
    ap.add_argument("--max-tail-charged-words", type=int, default=3_992_918,
                    help="Max words to charge (matches coherent86)")
    ap.add_argument("--private-adapter-scale", type=float, default=1.0,
                    help="Training-time adapter scale (evaluate at 0.75)")
    ap.add_argument("--smoke", action="store_true", help="CPU smoke test with small data")
    args = ap.parse_args()
    
    # Read composition metadata for provenance
    meta = json.loads(pathlib.Path(args.mixed_jsonl).parent.joinpath("metadata.json").read_text())
    total_pair_words = meta["pair_words"]
    total_ordinary_words = meta["ordinary_words"]
    pair_fraction = meta["pair_word_fraction"]
    
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    # Configure writable caches
    for k, p in {
        "HF_HOME": out / "hf_cache",
        "TRANSFORMERS_CACHE": out / "hf_cache",
        "HF_MODULES_CACHE": out / "hf_cache" / "modules",
        "TMPDIR": out / "tmp",
    }.items():
        p.mkdir(parents=True, exist_ok=True)
        os.environ[k] = str(p.resolve())
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    
    # Build args matching research's argparse expectations
    class S150Args:
        pass
    
    s150_args = S150Args()
    s150_args.endpoint = args.endpoint
    s150_args.example_jsonl = args.mixed_jsonl
    s150_args.output_dir = str(out)
    s150_args.replay_mode = "coherent_replay"  # standard, no span breaking
    
    # Copy all defaults from research
    for k, v in S150.DEFAULTS.items():
        setattr(s150_args, k, v)
    
    # Override for composition
    s150_args.skip_rows = 0  # mixed JSONL already starts at the right position
    s150_args.max_tail_charged_words = args.max_tail_charged_words
    s150_args.private_adapter_scale = args.private_adapter_scale
    
    if args.smoke:
        s150_args.max_tail_charged_words = 2000
        s150_args.batch_size = 4
        s150_args.log_every = 1
        s150_args.checkpoint_words = 1000
        s150_args.neutral_subsample = 2
    
    # Save composition-specific config
    config = {
        "status": "PRINCIPLE_COMPOSITION_CONFIG",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scientific_rationale": "Relation→column map composition: coherent replay + aligned restatement pairs "
                                "under standard CE+KL. Data composition IS the principle.",
        "data_source": str(args.mixed_jsonl),
        "data_sha256": meta["output_sha256"],
        "pair_words": total_pair_words,
        "pair_fraction": pair_fraction,
        "ordinary_words": total_ordinary_words,
        "endpoint": args.endpoint,
        "initial_consumed_words": s150_args.initial_consumed_words,
        "max_tail_charged_words": s150_args.max_tail_charged_words,
        "total_consumed_after": s150_args.initial_consumed_words + s150_args.max_tail_charged_words,
        "replay_mode": "coherent_replay",
        "training_scale": s150_args.private_adapter_scale,
        "batch_size": s150_args.batch_size,
        "learning_rate": s150_args.learning_rate,
        "lr_total_steps": s150_args.lr_total_steps,
        "warmup_fraction": s150_args.warmup_fraction,
        "neutral_lambda": s150_args.neutral_lambda,
        "smoke": args.smoke,
    }
    (out / "composition_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    
    print(json.dumps({
        "event": "composition_start",
        "pair_fraction": pair_fraction,
        "pair_words": total_pair_words,
        "ordinary_words": total_ordinary_words,
        "max_tail_charged_words": s150_args.max_tail_charged_words,
        "smoke": args.smoke,
    }, indent=2), flush=True)
    
    # Run training using research's proven machinery
    S150.train(s150_args)
    
    # After training, add composition provenance to scientific_metrics
    metrics_path = out / "scientific_metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        metrics["composition_provenance"] = {
            "config": "PRINCIPLE_COMPOSITION_CONFIG",
            "pair_words": total_pair_words,
            "pair_fraction": pair_fraction,
            "ordinary_words": total_ordinary_words,
            "data_sha256": meta["output_sha256"],
        }
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    
    print(json.dumps({
        "event": "composition_complete",
        "elapsed_sec": round(time.time() - t0, 2),
    }), flush=True)

if __name__ == "__main__":
    main()
