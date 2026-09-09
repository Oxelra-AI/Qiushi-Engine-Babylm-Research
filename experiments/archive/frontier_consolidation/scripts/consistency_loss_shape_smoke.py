#!/usr/bin/env python3
"""research: CPU shape smoke for a positive-only source-view consistency loss.

This uses the research consistency-collate interface and random hidden states to
exercise the tensor operations a future trainer would need: pooling source and
rewrite token ranges, computing a low-weight positive-only stop-gradient symmetric
agreement loss, and normalizing by eligible rows/pairs. It performs no model
training or evaluation and does not choose the source-view consistency route.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import importlib.util
import json
import math
import pathlib
import statistics
import time
from collections import defaultdict
from typing import Any

import torch
from torch.utils.data import DataLoader


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
WORKSPACE = USER_ROOT / "experiments/archive/frontier_consolidation"
COLLATE_SCRIPT = WORKSPACE / "scripts/consistency_collate_smoke.py"
OUT_DIR = WORKSPACE / "data/consistency_loss_shape_smoke"


def load_collate_module():
    spec = importlib.util.spec_from_file_location("consistency_collate_smoke", COLLATE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {COLLATE_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def pool_ranges(hidden: torch.Tensor, batch_row: int, ranges: list[list[int]]) -> torch.Tensor:
    pieces = []
    for a, b in ranges:
        a = int(a); b = int(b)
        if a < 0 or b <= a or b > hidden.shape[1]:
            raise RuntimeError(f"invalid range {(a, b)} for hidden shape {tuple(hidden.shape)}")
        pieces.append(hidden[batch_row, a:b, :])
    if not pieces:
        raise RuntimeError("empty ranges")
    toks = torch.cat(pieces, dim=0)
    return toks.mean(dim=0)


def finite_float(x: torch.Tensor) -> float:
    v = float(x.detach().cpu().item())
    if not math.isfinite(v):
        raise RuntimeError(f"nonfinite scalar {v}")
    return v


def summarize(vals: list[float]) -> dict[str, float | int]:
    if not vals:
        return {"n": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
    return {
        "n": len(vals),
        "mean": float(statistics.mean(vals)),
        "median": float(statistics.median(vals)),
        "min": float(min(vals)),
        "max": float(max(vals)),
        "std": float(statistics.pstdev(vals)),
    }


def run_one_sample(mod, name: str, sample_mode: str, sample_rows: int, batch_size: int, hidden_size: int, max_batches: int, seed: int) -> dict[str, Any]:
    examples = mod.read_stream_sample(mod.TRAIN_100M, sample_mode, sample_rows)
    span_by_ex = mod.load_span_map(mod.SPAN_JSONL)
    tokenizer = mod.AutoTokenizer.from_pretrained(str(mod.TOKENIZER_DIR), use_fast=True)
    ds = mod.ConsistencySpanDataset(examples, tokenizer, span_by_ex, mod.SEQ_LEN)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=mod.consistency_collate)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    batches_seen = 0
    eligible_batches = 0
    total_aux_records = 0
    total_aux_rows = 0
    pair_mean_losses: list[float] = []
    row_mean_losses: list[float] = []
    symmetric_pair_losses: list[float] = []
    pair_source_norms: list[float] = []
    pair_rewrite_norms: list[float] = []
    per_batch_payload: list[dict[str, Any]] = []

    for batch in loader:
        batches_seen += 1
        input_ids = batch["input_ids"]
        bsz, seq = input_ids.shape
        # Random states stand in for last_hidden_state.  requires_grad checks that
        # the proposed loss is differentiable w.r.t. a future model output, without
        # doing any model computation here.
        hidden = torch.randn((bsz, seq, hidden_size), generator=gen, dtype=torch.float32, requires_grad=True)
        aux = batch["aux_records"]
        if not aux:
            per_batch_payload.append({"batch_index_1based": batches_seen, "aux_records": 0, "aux_rows": 0})
            if batches_seen >= max_batches:
                break
            continue
        eligible_batches += 1
        total_aux_records += len(aux)
        row_keys = {(int(r["batch_row"]), int(r["example_id"])) for r in aux}
        total_aux_rows += len(row_keys)
        losses = []
        sym_losses = []
        row_losses: dict[tuple[int, int], list[torch.Tensor]] = defaultdict(list)
        for r in aux:
            br = int(r["batch_row"])
            src = pool_ranges(hidden, br, r["source_ranges"])
            rew = pool_ranges(hidden, br, r["rewrite_ranges"])
            src_n = torch.nn.functional.normalize(src, dim=0, eps=1e-6)
            rew_n = torch.nn.functional.normalize(rew, dim=0, eps=1e-6)
            # Positive-only, no negatives.  Symmetric stop-gradient version gives
            # each side a gradient through one term while the counterpart is fixed.
            loss_sr = torch.mean((src_n - rew_n.detach()) ** 2)
            loss_rs = torch.mean((rew_n - src_n.detach()) ** 2)
            loss_sym = 0.5 * (loss_sr + loss_rs)
            losses.append(loss_sr)
            sym_losses.append(loss_sym)
            row_losses[(br, int(r["example_id"]))].append(loss_sym)
            pair_source_norms.append(finite_float(src.norm()))
            pair_rewrite_norms.append(finite_float(rew.norm()))
        pair_mean = torch.stack(losses).mean()
        sym_pair_mean = torch.stack(sym_losses).mean()
        row_mean = torch.stack([torch.stack(v).mean() for v in row_losses.values()]).mean()
        # A tiny backward pass validates gradient plumbing through the student-side
        # terms; it is immediately discarded and never updates parameters.
        row_mean.backward()
        grad_nonzero = bool(hidden.grad is not None and torch.isfinite(hidden.grad).all() and hidden.grad.abs().sum().item() > 0)
        if not grad_nonzero:
            raise RuntimeError("consistency shape smoke produced no finite gradient")
        pair_mean_losses.append(finite_float(pair_mean))
        symmetric_pair_losses.append(finite_float(sym_pair_mean))
        row_mean_losses.append(finite_float(row_mean))
        per_batch_payload.append({
            "batch_index_1based": batches_seen,
            "aux_records": len(aux),
            "aux_rows": len(row_keys),
            "pair_mean_loss": pair_mean_losses[-1],
            "symmetric_pair_mean_loss": symmetric_pair_losses[-1],
            "row_mean_loss": row_mean_losses[-1],
            "grad_nonzero": grad_nonzero,
        })
        if batches_seen >= max_batches:
            break

    return {
        "name": name,
        "sample_mode": sample_mode,
        "sample_rows": sample_rows,
        "batch_size": batch_size,
        "hidden_size": hidden_size,
        "max_batches": max_batches,
        "batches_seen": batches_seen,
        "eligible_batches": eligible_batches,
        "total_aux_records": total_aux_records,
        "total_aux_rows": total_aux_rows,
        "pair_mean_loss_by_batch": summarize(pair_mean_losses),
        "symmetric_pair_mean_loss_by_batch": summarize(symmetric_pair_losses),
        "row_mean_loss_by_batch": summarize(row_mean_losses),
        "pooled_source_norms": summarize(pair_source_norms),
        "pooled_rewrite_norms": summarize(pair_rewrite_norms),
        "per_batch_payload": per_batch_payload,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--sample-rows", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--hidden-size", type=int, default=480)
    ap.add_argument("--max-batches", type=int, default=4)
    ap.add_argument("--seed", type=int, default=5401)
    args = ap.parse_args()
    t0 = time.time()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mod = load_collate_module()

    train_sha = mod.sha256_file(mod.TRAIN_100M)
    tok_sha = mod.sha256_file(mod.TOKENIZER_DIR / "tokenizer.json")
    if train_sha != mod.EXPECTED_TRAIN_SHA:
        raise RuntimeError(f"train SHA mismatch: {train_sha}")
    if tok_sha != mod.EXPECTED_TOKENIZER_SHA:
        raise RuntimeError(f"tokenizer SHA mismatch: {tok_sha}")

    sample_results = [
        run_one_sample(mod, "front_batches", "front", args.sample_rows, args.batch_size, args.hidden_size, args.max_batches, args.seed),
        run_one_sample(mod, "stride_batches", "stride", args.sample_rows, args.batch_size, args.hidden_size, args.max_batches, args.seed + 1),
    ]
    summary = {
        "status": "CONSISTENCY_LOSS_SHAPE_SMOKE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Exercise tensor pooling and normalization for a possible positive-only source-view consistency auxiliary loss, without model training/evaluation.",
        "inputs": {
            "collate_script": str(COLLATE_SCRIPT),
            "train_sha256": train_sha,
            "tokenizer_sha256": tok_sha,
            "hidden_size": args.hidden_size,
            "batch_size": args.batch_size,
        },
        "sample_results": sample_results,
        "loss_contract_for_future_trainer": [
            "Use model outputs with hidden states enabled, pool token ranges for both-visible source/rewrite pairs, and compute a low-weight positive-only agreement loss.",
            "Prefer row-normalized symmetric stop-gradient MSE/cosine agreement for a first screen so rows with more packed pairs do not dominate and both sides receive a gradient through one term.",
            "Keep MLM as the primary objective and do not add negatives, static-prior masking, corpus changes, tokenizer changes, or optimizer changes in the first source-view consistency screen.",
        ],
        "scientific_interpretation": [
            "The shape smoke removes a tensor-construction risk for the broad-disappearance branch, but it is not behavioral evidence and does not justify GPU use without the mature 70M/80M pattern.",
            "The auxiliary payload is large enough in ordinary batches to support a row-normalized loss if that branch is selected.",
        ],
        "elapsed_sec": round(time.time() - t0, 3),
    }
    out_json = out_dir / "consistency_loss_shape_smoke.json"
    out_md = out_dir / "consistency_loss_shape_smoke.md"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research consistency-loss shape smoke",
        "",
        summary["purpose"],
        "",
        "CPU-only; no model training/evaluation and no route choice.",
        "",
        "## Samples",
    ]
    for s in sample_results:
        lines.append(
            f"- {s['name']}: batches_seen={s['batches_seen']}, eligible_batches={s['eligible_batches']}, "
            f"aux_records={s['total_aux_records']}, aux_rows={s['total_aux_rows']}, "
            f"row_mean_loss_mean={s['row_mean_loss_by_batch']['mean']:.6f}, "
            f"source_norm_mean={s['pooled_source_norms']['mean']:.4f}, rewrite_norm_mean={s['pooled_rewrite_norms']['mean']:.4f}"
        )
    lines += ["", "## Future trainer loss contract"]
    for item in summary["loss_contract_for_future_trainer"]:
        lines.append(f"- {item}")
    lines += ["", "## Interpretation"]
    for item in summary["scientific_interpretation"]:
        lines.append(f"- {item}")
    lines += ["", f"Full JSON: `{out_json}`"]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "out_json": str(out_json),
        "out_md": str(out_md),
        "sample_results": [{"name": s["name"], "eligible_batches": s["eligible_batches"], "aux_records": s["total_aux_records"], "aux_rows": s["total_aux_rows"]} for s in sample_results],
        "elapsed_sec": summary["elapsed_sec"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
