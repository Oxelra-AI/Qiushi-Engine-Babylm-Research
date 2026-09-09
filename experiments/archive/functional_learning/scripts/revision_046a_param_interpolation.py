#!/usr/bin/env python3
"""Step046a: private-adapter parameter interpolation diagnostic.

Tests whether the specialist answer-only training overshot a useful regime.
θ(t) = θ_parent + t * (θ_acquired - θ_parent) for private-adapter tensors only.
Base DeBERTa weights are identical, so only private-adapter tensors are interpolated.

Scores at each interpolation point on held three-way (neutral/retain/update
cross-source, full retrieval) and source reassignment.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import gc
import json
import math
import pathlib
import sys
import time
from typing import Any, Dict, List, Sequence, Tuple

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
sys.path.insert(0, str(_public_path('experiments/archive/functional_learning/scripts')))

import threeway_and_reassignment_probe as probe  # noqa: E402
import saved_state_replicate_neutral_threeentity as repl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Private-adapter parameter interpolation diagnostic")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--acquired-path", default=str(
        _public_path('experiments/archive/functional_learning/data/saved_state_replicate_neutral_threeentity/seed_40040/hf_model/final')))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--t-values", default="0.0,0.1,0.2,0.3,0.4,0.5,0.7,1.0")
    ap.add_argument("--seq-length", type=int, default=512)
    ap.add_argument("--private-scale", type=float, default=0.75)
    ap.add_argument("--construction-seed", type=int, default=40040)
    args = ap.parse_args()

    acquired_path = pathlib.Path(args.acquired_path)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    t_values = [float(x) for x in args.t_values.split(",")]

    # ---- Load parent with trusted loader ----
    model, load_info = probe.load_trusted_model(device, private_scale=args.private_scale)
    private_tensors = [(n, p) for n, p in model.named_parameters()
                       if ".private_adapter." in n]
    n_keys = len(private_tensors)
    n_params = sum(p.numel() for _, p in private_tensors)
    print(json.dumps({"event": "loaded_parent",
                       "class": type(model).__name__,
                       "private_tensors": n_keys,
                       "private_params": n_params}), flush=True)

    # ---- Save parent private-adapter tensors ----
    parent_priv: Dict[str, torch.Tensor] = {}
    for name, param in private_tensors:
        parent_priv[name] = param.data.clone().cpu()

    # ---- Load acquired checkpoint ----
    acq_sf = acquired_path / "model.safetensors"
    if not acq_sf.exists():
        raise FileNotFoundError(f"No safetensors at {acq_sf}")
    acq_sd = load_file(str(acq_sf), device="cpu")
    acq_priv: Dict[str, torch.Tensor] = {}
    for k in parent_priv:
        if k in acq_sd:
            acq_priv[k] = acq_sd[k].clone()
        else:
            raise ValueError(f"Private key {k} not in acquired checkpoint")
    del acq_sd
    gc.collect()

    # ---- Displacement ----
    total_disp = math.sqrt(sum(
        float((acq_priv[k] - parent_priv[k]).norm().item() ** 2)
        for k in parent_priv))
    per_tensor = [{
        "key": k,
        "L2": round(float((acq_priv[k] - parent_priv[k]).norm().item()), 6),
        "numel": int(parent_priv[k].numel()),
    } for k in sorted(parent_priv)]
    print(f"total displacement L2: {total_disp:.6f}", flush=True)

    # ---- Held pairs (same fixed construction as research) ----
    tokenizer = AutoTokenizer.from_pretrained(
        str(probe.MODEL_PATH), local_files_only=True, use_fast=True)
    pairs, train_pairs, held_pairs, _, _ = repl.fixed_repaired_pairs(
        args.construction_seed)
    print(f"loaded {len(pairs)} pairs, {len(held_pairs)} held", flush=True)

    # ---- Score at each t ----
    results: List[Dict[str, Any]] = []
    for t in t_values:
        t0 = time.time()
        print(f"\n=== t = {t:.2f} ===", flush=True)

        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in parent_priv:
                    interp = parent_priv[name] + t * (
                        acq_priv[name] - parent_priv[name])
                    param.copy_(interp.to(param.device))
        model.eval()

        held_tw = []
        for i, p in enumerate(held_pairs):
            held_tw.append(probe.threeway_score_pair(
                model, tokenizer, p, device, args.seq_length))
            if (i + 1) % 10 == 0:
                print(f"  held tw {i + 1}/{len(held_pairs)}", flush=True)

        tw_summary, tw_records = repl.analyze_threeway_extended(
            held_tw, held_pairs, f"t{t:.2f}")
        reassign_summary, reassign_records, reassign_n = \
            repl.score_reassignment_held(
                model, tokenizer, held_pairs, held_tw, device, args.seq_length)

        elapsed = time.time() - t0
        result: Dict[str, Any] = {
            "t": t,
            "effective_L2": round(t * total_disp, 6),
            "threeway": tw_summary,
            "reassignment": reassign_summary,
            "reassignment_valid_n": reassign_n,
            "elapsed_sec": round(elapsed, 1),
        }
        results.append(result)
        repl.write_jsonl(out_dir / f"t{t:.2f}_threeway.jsonl", tw_records)
        repl.write_jsonl(out_dir / f"t{t:.2f}_reassignment.jsonl",
                         reassign_records)
        print(json.dumps({
            "event": "t_done", "t": t,
            "neutral_cross": tw_summary.get("neutral_cross_source_correct"),
            "neutral_full": tw_summary.get("neutral_both_full_source"),
            "retain_cross": tw_summary.get("retain_cross_source_correct"),
            "retain_full": tw_summary.get("retain_both_full_source"),
            "reassign_both": reassign_summary.get("n_swap_both_follow"),
            "elapsed_sec": round(elapsed, 1),
        }), flush=True)

    # ---- Save summary ----
    final = {
        "status": "PARAM_INTERPOLATION_DONE",
        "parent_path": str(probe.MODEL_PATH),
        "acquired_path": str(acquired_path),
        "total_displacement_L2": round(total_disp, 6),
        "private_adapter_keys": n_keys,
        "private_adapter_params": n_params,
        "per_tensor_displacement": per_tensor,
        "t_values": t_values,
        "results": results,
    }
    (out_dir / "interpolation_summary.json").write_text(
        json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + json.dumps(final, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
