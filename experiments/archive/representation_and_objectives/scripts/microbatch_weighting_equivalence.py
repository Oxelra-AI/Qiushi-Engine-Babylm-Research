#!/usr/bin/env python3
"""research masked-token weighted microbatch equivalence test.

CPU-only.  No model training for BabyLM, no official evaluation text, no GPU use.

This validates one load-bearing implementation property for future chunk-stream
runs: if WWM labels are sampled once on the full effective batch, then splitting
forward/backward into microbatches and scaling each microbatch loss by its share
of masked targets gives the same deterministic objective gradient as a full
batch pass.  It also records how much a naive average of microbatch means would
move the gradient on the same synthetic shapes.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

USER_ROOT = Path(".").resolve()
STUDY = USER_ROOT / "experiments/archive" / 'representation_and_objectives'
WORKSPACE = STUDY
PREFLIGHT_CSV = WORKSPACE / "data" / "chunk_stream_preflight" / "chunk_stream_by_length.csv"
OUT_DIR = WORKSPACE / "data" / "microbatch_weighting_equivalence"
NOTE = (USER_ROOT / 'research/notes/representation_and_objectives/microbatch_weighting_equivalence.md')
MICRO = 64
VOCAB = 127
HIDDEN = 16
MASK_PROB = 0.15


class TinyMLM(nn.Module):
    def __init__(self, vocab: int = VOCAB, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden)
        self.proj = nn.Linear(hidden, vocab, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.tanh(self.emb(x)))


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(USER_ROOT))
    except Exception:
        return str(p)


def load_shapes() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with PREFLIGHT_CSV.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "tokenizer": r["tokenizer"],
                "length": int(r["length"]),
                "chunks": int(r["first_step_chunks"]),
                "active_tokens": int(r["first_step_active_tokens"]),
                "masked_tokens_reference": int(r["first_step_masked_tokens"]),
                "accumulation_microbatches": int(r["first_step_accumulation_microbatches"]),
            })
    if not rows:
        raise RuntimeError(f"no rows in {PREFLIGHT_CSV}")
    return rows


def make_synthetic_batch(shape: dict[str, Any], seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B = int(shape["chunks"])
    L = int(shape["length"])
    active = int(shape["active_tokens"])
    if active > B * L:
        raise RuntimeError("active token count exceeds nominal slots")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    input_ids = torch.randint(0, VOCAB, (B, L), generator=gen, dtype=torch.long)
    # Packed attention pattern: each row gets floor/ceil active slots.  This
    # mimics padded chunk tensors without using real BabyLM tokens.
    attention = torch.zeros((B, L), dtype=torch.bool)
    remaining = active
    for i in range(B):
        rows_left = B - i
        take = min(L, (remaining + rows_left - 1) // rows_left)
        attention[i, :take] = True
        remaining -= take
    if int(attention.sum().item()) != active:
        raise RuntimeError("failed to build requested active-token pattern")
    labels = torch.full((B, L), -100, dtype=torch.long)
    probs = torch.rand((B, L), generator=gen)
    select = attention & (probs < MASK_PROB)
    if int(select.sum().item()) == 0:
        first = attention.view(-1).nonzero(as_tuple=False)[0, 0]
        select.view(-1)[first] = True
    labels[select] = torch.randint(0, VOCAB, (int(select.sum().item()),), generator=gen, dtype=torch.long)
    return input_ids, attention, labels


def clone_model(model: nn.Module) -> nn.Module:
    other = TinyMLM()
    other.load_state_dict(model.state_dict())
    return other


def grad_vector(model: nn.Module) -> torch.Tensor:
    parts = []
    for p in model.parameters():
        if p.grad is None:
            parts.append(torch.zeros_like(p).reshape(-1))
        else:
            parts.append(p.grad.detach().reshape(-1).clone())
    return torch.cat(parts)


def full_pass(model: nn.Module, input_ids: torch.Tensor, labels: torch.Tensor) -> tuple[float, torch.Tensor]:
    model.zero_grad(set_to_none=True)
    logits = model(input_ids)
    loss = F.cross_entropy(logits.view(-1, VOCAB), labels.view(-1), ignore_index=-100, reduction="mean")
    loss.backward()
    return float(loss.detach()), grad_vector(model)


def weighted_micro_pass(model: nn.Module, input_ids: torch.Tensor, labels: torch.Tensor) -> tuple[float, torch.Tensor, list[int]]:
    model.zero_grad(set_to_none=True)
    total_targets = int((labels != -100).sum().item())
    if total_targets <= 0:
        raise RuntimeError("no targets")
    weighted_loss = 0.0
    target_counts: list[int] = []
    B = input_ids.shape[0]
    for start in range(0, B, MICRO):
        end = min(start + MICRO, B)
        y = labels[start:end]
        n = int((y != -100).sum().item())
        target_counts.append(n)
        if n <= 0:
            continue
        logits = model(input_ids[start:end])
        loss_i = F.cross_entropy(logits.view(-1, VOCAB), y.reshape(-1), ignore_index=-100, reduction="mean")
        scale = n / total_targets
        (loss_i * scale).backward()
        weighted_loss += float(loss_i.detach()) * scale
    return weighted_loss, grad_vector(model), target_counts


def naive_micro_pass(model: nn.Module, input_ids: torch.Tensor, labels: torch.Tensor) -> tuple[float, torch.Tensor]:
    model.zero_grad(set_to_none=True)
    losses = []
    B = input_ids.shape[0]
    active_micro = 0
    for start in range(0, B, MICRO):
        end = min(start + MICRO, B)
        y = labels[start:end]
        n = int((y != -100).sum().item())
        if n <= 0:
            continue
        logits = model(input_ids[start:end])
        loss_i = F.cross_entropy(logits.view(-1, VOCAB), y.reshape(-1), ignore_index=-100, reduction="mean")
        losses.append(loss_i)
        active_micro += 1
    loss = sum(losses) / max(1, active_micro)
    loss.backward()
    return float(loss.detach()), grad_vector(model)


def run_case(shape: dict[str, Any], seed: int) -> dict[str, Any]:
    x, _attention, labels = make_synthetic_batch(shape, seed)
    base = TinyMLM()
    torch.manual_seed(seed + 1000)
    base = TinyMLM()
    full_model = clone_model(base)
    weighted_model = clone_model(base)
    naive_model = clone_model(base)
    loss_full, g_full = full_pass(full_model, x, labels)
    loss_weighted, g_weighted, target_counts = weighted_micro_pass(weighted_model, x, labels)
    loss_naive, g_naive = naive_micro_pass(naive_model, x, labels)
    diff = (g_full - g_weighted).abs()
    naive_diff = (g_full - g_naive).abs()
    total_targets = int((labels != -100).sum().item())
    return {
        "tokenizer": shape["tokenizer"],
        "length": shape["length"],
        "chunks": shape["chunks"],
        "active_tokens": shape["active_tokens"],
        "nominal_slots": shape["chunks"] * shape["length"],
        "microbatch_size": MICRO,
        "microbatches": (shape["chunks"] + MICRO - 1) // MICRO,
        "synthetic_masked_targets": total_targets,
        "reference_first_step_masked_tokens_from_chunk_preflight": shape["masked_tokens_reference"],
        "loss_full": loss_full,
        "loss_weighted_micro": loss_weighted,
        "loss_naive_micro_mean": loss_naive,
        "abs_loss_diff_weighted": abs(loss_full - loss_weighted),
        "max_abs_grad_diff_weighted": float(diff.max().item()),
        "mean_abs_grad_diff_weighted": float(diff.mean().item()),
        "max_abs_grad_diff_naive": float(naive_diff.max().item()),
        "mean_abs_grad_diff_naive": float(naive_diff.mean().item()),
        "micro_target_count_min": min(target_counts),
        "micro_target_count_max": max(target_counts),
        "micro_target_count_mean": sum(target_counts) / len(target_counts),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    shapes = load_shapes()
    cases = []
    for i, shape in enumerate(shapes):
        cases.append(run_case(shape, 771000 + i))
    result = {
        "status": "MICROBATCH_WEIGHTING_EQUIVALENCE",
        "purpose": "Validate masked-token weighted microbatch accumulation on synthetic tensors shaped like the research chunk-stream first updates.",
        "preflight_csv": rel(PREFLIGHT_CSV),
        "microbatch_size": MICRO,
        "tiny_model": {"vocab": VOCAB, "hidden": HIDDEN, "dropout": 0.0},
        "cases": cases,
        "all_weighted_gradients_match_tolerance_1e_6": all(c["max_abs_grad_diff_weighted"] < 1e-6 for c in cases),
        "naive_micro_mean_changes_gradient": any(c["max_abs_grad_diff_naive"] > 1e-6 for c in cases),
        "scope": "This proves the loss-weighting algebra for deterministic forward paths. Real DeBERTa training still has dropout and floating-point order differences between full-batch and microbatched execution, so run logs must record actual target counts and losses.",
    }
    out_json = OUT_DIR / "microbatch_weighting_equivalence.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    out_csv = OUT_DIR / "microbatch_weighting_equivalence.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "tokenizer", "length", "chunks", "active_tokens", "microbatches", "synthetic_masked_targets",
            "loss_full", "loss_weighted_micro", "abs_loss_diff_weighted", "max_abs_grad_diff_weighted",
            "max_abs_grad_diff_naive", "micro_target_count_min", "micro_target_count_max",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for c in cases:
            w.writerow({k: c[k] for k in fields})

    lines = [
        "# research — Microbatch weighting equivalence",
        "",
        "CPU-only synthetic tensor test using first-update shapes from the chunk-stream preflight.",
        "",
        f"Weighted microbatch gradients matched full-batch gradients below 1e-6 for all cases: {result['all_weighted_gradients_match_tolerance_1e_6']}.",
        f"Naive averaging of microbatch means changed gradients in at least one case: {result['naive_micro_mean_changes_gradient']}.",
        "",
        "| tokenizer | L | chunks | active tokens | microbatches | targets | max grad diff weighted | max grad diff naive |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in cases:
        lines.append(
            f"| {c['tokenizer']} | {c['length']} | {c['chunks']} | {c['active_tokens']} | {c['microbatches']} | {c['synthetic_masked_targets']} | {c['max_abs_grad_diff_weighted']:.3e} | {c['max_abs_grad_diff_naive']:.3e} |"
        )
    lines.extend([
        "",
        "Scope: this validates masked-target weighting for deterministic forward paths. Real DeBERTa microbatch training still differs from a hypothetical single full-batch pass through dropout and numerical order, so a future run must record realized targets and accumulation depth per update.",
        "",
        f"JSON: `{rel(out_json)}`",
        f"CSV: `{rel(out_csv)}`",
    ])
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "out_json": rel(out_json),
        "csv": rel(out_csv),
        "note": rel(NOTE),
        "all_weighted_match": result["all_weighted_gradients_match_tolerance_1e_6"],
        "max_weighted_grad_diff": max(c["max_abs_grad_diff_weighted"] for c in cases),
        "max_naive_grad_diff": max(c["max_abs_grad_diff_naive"] for c in cases),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
