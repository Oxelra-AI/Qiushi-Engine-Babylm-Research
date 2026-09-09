#!/usr/bin/env python3
"""Step046c: evaluate bridge checkpoints on relation behavior and broad competence.

For each saved checkpoint:
1. Held three-way discrimination (neutral/retain full source retrieval)
2. Source reassignment
3. Fast Cheap7 screen (Entity + other columns)

Uses trusted remote-code loading throughout.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import gc
import json
import os
import pathlib
import sys
import time
from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
sys.path.insert(0, str(_public_path('experiments/archive/functional_learning/scripts')))

import threeway_and_reassignment_probe as probe  # noqa: E402
import saved_state_replicate_neutral_threeentity as repl  # noqa: E402


def rel(path: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def load_checkpoint_model(ckpt_path: pathlib.Path, device: torch.device,
                          private_scale: float = 0.75):
    """Load a saved bridge checkpoint with trusted remote-code loading."""
    # Use the same trusted loader as probe but from the checkpoint path
    from transformers import DebertaV2Config
    from frozen82_private_modeling import FrozenSlowPrivateDebertaV2ForMaskedLM
    from safetensors.torch import load_file

    cfg = DebertaV2Config.from_pretrained(str(ckpt_path), local_files_only=True)
    cfg.private_adapter_bottleneck = 128
    cfg.private_adapter_scale = float(private_scale)
    cfg.private_adapter_enabled = True
    model = FrozenSlowPrivateDebertaV2ForMaskedLM(cfg)
    sd = load_file(str(ckpt_path / "model.safetensors"), device="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    tied_missing = {"cls.predictions.decoder.weight", "cls.predictions.decoder.bias"}
    bad_missing = [x for x in missing if "private_adapter" not in x and x not in tied_missing]
    if bad_missing or unexpected:
        raise RuntimeError(f"load mismatch: bad_missing={bad_missing[:5]} unexpected={unexpected[:5]}")
    model.tie_weights()
    if hasattr(model, "set_private_enabled"):
        model.set_private_enabled(True)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.to(device)
    model.eval()
    private = [(n, p) for n, p in model.named_parameters() if ".private_adapter." in n]
    return model, {
        "class": type(model).__name__,
        "private_tensors": len(private),
        "private_params": sum(p.numel() for _, p in private),
    }


def eval_relation_held(model, tokenizer, held_pairs, device, seq_length):
    """Score on held three-way + reassignment."""
    held_tw = []
    for p in held_pairs:
        held_tw.append(probe.threeway_score_pair(model, tokenizer, p, device, seq_length))
    tw_summary, tw_records = repl.analyze_threeway_extended(held_tw, held_pairs, "ckpt")
    reassign_summary, reassign_records, reassign_n = repl.score_reassignment_held(
        model, tokenizer, held_pairs, held_tw, device, seq_length)
    return {
        "threeway": tw_summary,
        "reassignment": reassign_summary,
        "reassignment_valid_n": reassign_n,
    }


def eval_cheap7_entity(model, tokenizer, ckpt_path, device):
    """Run the fast Cheap7 screen using the saved checkpoint directory."""
    # Import the evaluator
    sys.path.insert(0, str(_public_path('experiments/archive/functional_learning/scripts')))
    try:
        from eval_saved_state_cheap7_entity import (
            EVAL_COLUMNS, evaluate_column, eval_preflight
        )
        scores = {}
        for col in EVAL_COLUMNS:
            score, rc = evaluate_column(str(ckpt_path), col)
            scores[col] = score
        # Compute equal_valid_mean
        valid = [v for k, v in scores.items()
                 if v is not None and k not in ("Reading_eye", "Reading_self_paced",
                                                 "GlobalPIQA_mean")]
        scores["equal_valid_mean"] = sum(valid) / len(valid) if valid else None
        return scores
    except ImportError:
        return {"error": "research evaluator not importable"}


def main():
    ap = argparse.ArgumentParser(description="Evaluate bridge checkpoints")
    ap.add_argument("--bridge-dir", required=True,
                    help="Bridge training output directory")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seq-length", type=int, default=512)
    ap.add_argument("--private-scale", type=float, default=0.75)
    ap.add_argument("--construction-seed", type=int, default=40040)
    ap.add_argument("--eval-relation", action="store_true", default=True)
    ap.add_argument("--eval-cheap7", action="store_true", default=False)
    ap.add_argument("--checkpoints", default="all",
                    help="Comma-separated checkpoint names or 'all'")
    args = ap.parse_args()

    bridge_dir = pathlib.Path(args.bridge_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Find checkpoints
    ckpt_base = bridge_dir / "checkpoints"
    if not ckpt_base.exists():
        raise FileNotFoundError(f"No checkpoints at {ckpt_base}")
    if args.checkpoints == "all":
        ckpt_names = sorted([d.name for d in ckpt_base.iterdir() if d.is_dir()])
    else:
        ckpt_names = [x.strip() for x in args.checkpoints.split(",")]
    print(f"evaluating {len(ckpt_names)} checkpoints: {ckpt_names}", flush=True)

    # Load held pairs for relation evaluation
    tokenizer = AutoTokenizer.from_pretrained(
        str(probe.MODEL_PATH), local_files_only=True, use_fast=True)
    pairs, train_pairs, held_pairs, _, _ = repl.fixed_repaired_pairs(
        args.construction_seed)

    results = []
    for ckpt_name in ckpt_names:
        ckpt_path = ckpt_base / ckpt_name
        if not (ckpt_path / "model.safetensors").exists():
            print(f"skipping {ckpt_name}: no model.safetensors", flush=True)
            continue
        print(f"\n=== {ckpt_name} ===", flush=True)
        t0 = time.time()

        # Load bridge metadata
        meta_path = ckpt_path / "bridge_metadata.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

        model, ident = load_checkpoint_model(ckpt_path, device, args.private_scale)
        print(json.dumps({"event": "loaded", "checkpoint": ckpt_name, **ident}), flush=True)

        result: Dict[str, Any] = {
            "checkpoint": ckpt_name,
            "path": rel(ckpt_path),
            "bridge_metadata": meta,
            "model_identity": ident,
        }

        if args.eval_relation:
            rel_result = eval_relation_held(model, tokenizer, held_pairs, device, args.seq_length)
            result["relation"] = rel_result
            nf = rel_result["threeway"].get("neutral_both_full_source", "?")
            rf = rel_result["threeway"].get("retain_both_full_source", "?")
            ra = rel_result["reassignment"].get("n_swap_both_follow", "?")
            print(json.dumps({
                "event": "relation_done", "checkpoint": ckpt_name,
                "neutral_full": nf, "retain_full": rf, "reassign_both": ra,
            }), flush=True)

        if args.eval_cheap7:
            scores = eval_cheap7_entity(model, tokenizer, ckpt_path, device)
            result["cheap7"] = scores
            print(json.dumps({
                "event": "cheap7_done", "checkpoint": ckpt_name,
                "Entity": scores.get("Entity"),
                "equal_valid_mean": scores.get("equal_valid_mean"),
            }), flush=True)

        result["elapsed_sec"] = round(time.time() - t0, 1)
        results.append(result)

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save
    summary = {
        "status": "BRIDGE_EVAL_DONE",
        "bridge_dir": rel(bridge_dir),
        "n_checkpoints": len(results),
        "results": results,
    }
    (out_dir / "bridge_eval_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
