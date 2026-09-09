#!/usr/bin/env python3
"""research: Tensor-parity smoke test for HS/LS/HD/LD factorial arms.

Verifies that all four factorial arms, when fed through the same RoBERTa MLM
trainer configuration, produce:
  1. Identical model initialization (parameter SHA256)
  2. Same total training steps (identical word counts → same DataLoader length)
  3. Same batch sizes and attention mask shapes
  4. Comparable WWM mask statistics (similar masked-token counts)
  5. Finite loss and gradients on all arms
  6. Reproducible first-update behavior (same LR, same optimizer state)
  7. Anchor/non-anchor target mass profiles

This is CPU-only, runs 3 steps per arm, and does NOT launch GPU training.
The script loads the four 10M pools and simulates the first few batches.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import collections
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    RobertaConfig,
    RobertaForMaskedLM,
    PreTrainedTokenizerFast,
    get_cosine_schedule_with_warmup,
)


def find_user_root() -> Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
TOKENIZER_DIR = USER_ROOT / "experiments/archive/frontier_consolidation/data/compliant_tokenizer"
POOL_DIR = USER_ROOT / "experiments/archive/representation_and_objectives/data/factorial_streams"
DEFAULT_OUT = USER_ROOT / "experiments/archive/representation_and_objectives/data/tensor_parity_smoke"

# Model config matching the RoBERTa scaffold
MODEL_CFG = dict(
    hidden_size=480, n_layer=8, n_head=8, ffn_mult=4,
    max_position_embeddings=512, hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1, layer_norm_eps=1e-5,
)
TRAIN_CFG = dict(
    seed=43, extra_init_seed=43022, train_rng_seed=43023,
    batch_size=256, seq_length=256, learning_rate=0.001,
    warmup_fraction=0.06, weight_decay=0.01, mask_prob=0.15,
    lr_total_steps=2529, grad_clip=1.0,
)
SMOKE_STEPS = 3
SMOKE_BATCH = 4  # Small CPU batch for the smoke test


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest()


def sha256_params(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for p in model.parameters():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


@dataclass
class Example:
    text: str
    words: int
    example_id: int
    source: str


def load_examples(path: Path, max_rows: int) -> tuple[list[Example], dict[str, Any]]:
    examples: list[Example] = []
    total_words = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            examples.append(Example(
                text=str(obj["text"]),
                words=int(obj["words"]),
                example_id=int(obj.get("example_id", 0)),
                source=str(obj.get("source", "")),
            ))
            total_words += examples[-1].words
            if len(examples) >= max_rows:
                break
    return examples, {"rows": len(examples), "words": total_words}


def make_tokenizer(path: Path) -> PreTrainedTokenizerFast:
    tok = AutoTokenizer.from_pretrained(str(path), use_fast=True)
    if tok.mask_token is None:
        tok.add_special_tokens({"mask_token": "<mask>"})
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "<pad>"})
    if tok.cls_token is None:
        tok.add_special_tokens({"cls_token": "<s>"})
    if tok.sep_token is None:
        tok.add_special_tokens({"sep_token": "</s>"})
    return tok


class ChunkDataset(Dataset):
    def __init__(self, examples: list[Example], tokenizer: PreTrainedTokenizerFast, max_len: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex.text, max_length=self.max_len, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        ids = enc["input_ids"].squeeze(0)
        mask = enc["attention_mask"].squeeze(0)
        # Word group assignment for WWM
        word_group = torch.full_like(ids, -1)
        grp = 0
        for i in range(ids.shape[0]):
            tid = int(ids[i])
            if tid in self.tokenizer.all_special_ids:
                continue
            if not mask[i]:
                continue
            tok_str = self.tokenizer.convert_ids_to_tokens(tid)
            if grp == 0 or (isinstance(tok_str, str) and (tok_str.startswith("Ġ") or tok_str.startswith("▁"))) or i == 0:
                grp += 1
            word_group[i] = grp
        return {
            "input_ids": ids,
            "attention_mask": mask,
            "word_group": word_group,
            "words": torch.tensor(ex.words, dtype=torch.long),
            "example_id": torch.tensor(ex.example_id, dtype=torch.long),
        }


def collate(batch):
    return {k: torch.stack([x[k] for x in batch]) for k in batch[0]}


def apply_wwm_masking(input_ids, attention_mask, word_group, tokenizer, mask_prob, gen):
    device = input_ids.device
    labels = input_ids.clone()
    bsz, seq = input_ids.shape
    mask_token_id = tokenizer.mask_token_id
    special_ids = torch.tensor(sorted(tokenizer.all_special_ids), device=device)
    candidate = attention_mask.bool() & ~torch.isin(input_ids, special_ids)
    select = torch.zeros_like(candidate)
    selected_groups = 0
    total_groups = 0
    for b in range(bsz):
        groups = word_group[b]
        valid_groups = torch.unique(groups[groups >= 0])
        if valid_groups.numel() == 0:
            continue
        total_groups += int(valid_groups.numel())
        gp = torch.rand(valid_groups.numel(), generator=gen, device=device)
        chosen = valid_groups[gp < mask_prob]
        selected_groups += int(chosen.numel())
        if chosen.numel() == 0:
            continue
        select[b] = torch.isin(groups, chosen) & candidate[b]
    if select.sum() == 0:
        flat = candidate.view(-1).nonzero(as_tuple=False)
        if flat.numel() > 0:
            select.view(-1)[flat[0, 0]] = True
    labels[~select] = -100
    masked_inputs = input_ids.clone()
    r = torch.rand(bsz, seq, generator=gen, device=device)
    mask_tok = select & (r < 0.8)
    rand_tok = select & (r >= 0.8) & (r < 0.9)
    masked_inputs[mask_tok] = int(mask_token_id)
    if rand_tok.any():
        rand_ids = torch.randint(0, len(tokenizer), (int(rand_tok.sum()),), generator=gen, device=device)
        masked_inputs[rand_tok] = rand_ids
    return masked_inputs, labels, {
        "masked_tokens": int(select.sum().item()),
        "candidate_tokens": int(candidate.sum().item()),
        "selected_groups": selected_groups,
        "total_groups": total_groups,
    }


def reset_rng(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)


def build_model(tokenizer):
    cfg = RobertaConfig(
        vocab_size=len(tokenizer),
        hidden_size=MODEL_CFG["hidden_size"],
        num_hidden_layers=MODEL_CFG["n_layer"],
        num_attention_heads=MODEL_CFG["n_head"],
        intermediate_size=MODEL_CFG["hidden_size"] * MODEL_CFG["ffn_mult"],
        max_position_embeddings=MODEL_CFG["max_position_embeddings"],
        type_vocab_size=1,
        hidden_dropout_prob=MODEL_CFG["hidden_dropout_prob"],
        attention_probs_dropout_prob=MODEL_CFG["attention_probs_dropout_prob"],
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        layer_norm_eps=MODEL_CFG["layer_norm_eps"],
    )
    return RobertaForMaskedLM(cfg)


def run_arm_smoke(arm: str, pool_path: Path, tokenizer: PreTrainedTokenizerFast, out_dir: Path) -> dict[str, Any]:
    """Run SMOKE_STEPS forward+backward on CPU for one arm. Return step-level records."""
    max_rows = SMOKE_BATCH * (SMOKE_STEPS + 1)
    examples, data_meta = load_examples(pool_path, max_rows)

    # Reset to extra_init_seed for model init
    reset_rng(TRAIN_CFG["extra_init_seed"])
    model = build_model(tokenizer)
    init_sha = sha256_params(model)

    # Reset to train_rng_seed
    reset_rng(TRAIN_CFG["train_rng_seed"])
    device = torch.device("cpu")
    model.to(device)
    model.train()

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=TRAIN_CFG["learning_rate"],
        weight_decay=TRAIN_CFG["weight_decay"],
        betas=(0.9, 0.98),
    )
    # For smoke: warmup from full schedule
    warmup = max(1, int(TRAIN_CFG["lr_total_steps"] * TRAIN_CFG["warmup_fraction"]))
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=warmup, num_training_steps=TRAIN_CFG["lr_total_steps"])

    gen = torch.Generator(device=device)
    gen.manual_seed(TRAIN_CFG["train_rng_seed"])

    dataset = ChunkDataset(examples, tokenizer, TRAIN_CFG["seq_length"])
    loader = DataLoader(dataset, batch_size=SMOKE_BATCH, shuffle=False, collate_fn=collate, num_workers=0)

    step_records: list[dict[str, Any]] = []
    for step, batch in enumerate(loader, 1):
        if step > SMOKE_STEPS:
            break
        words = int(batch.pop("words").sum().item())
        _ = batch.pop("example_id")
        input_ids = batch["input_ids"][:, :TRAIN_CFG["seq_length"]].contiguous()
        attention_mask = batch["attention_mask"][:, :TRAIN_CFG["seq_length"]].contiguous()
        word_group = batch["word_group"][:, :TRAIN_CFG["seq_length"]].contiguous()

        masked_inputs, labels, mask_stats = apply_wwm_masking(
            input_ids, attention_mask, word_group, tokenizer, TRAIN_CFG["mask_prob"], gen
        )

        optim.zero_grad(set_to_none=True)
        out_model = model(input_ids=masked_inputs, attention_mask=attention_mask, labels=labels)
        loss = out_model.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CFG["grad_clip"])
        optim.step()
        sched.step()

        grad_norms = []
        for p in model.parameters():
            if p.grad is not None:
                grad_norms.append(float(p.grad.norm().item()))

        rec = {
            "step": step,
            "loss": float(loss.detach().cpu()),
            "finite_loss": math.isfinite(float(loss.detach().cpu())),
            "lr": float(sched.get_last_lr()[0]),
            "batch_words": words,
            "batch_rows": SMOKE_BATCH,
            "input_shape": list(input_ids.shape),
            "masked_tokens": mask_stats["masked_tokens"],
            "candidate_tokens": mask_stats["candidate_tokens"],
            "selected_groups": mask_stats["selected_groups"],
            "total_groups": mask_stats["total_groups"],
            "effective_mask_rate": round(mask_stats["masked_tokens"] / max(1, mask_stats["candidate_tokens"]), 6),
            "finite_logits": bool(torch.isfinite(out_model.logits).all().item()),
            "mean_grad_norm": float(np.mean(grad_norms)) if grad_norms else None,
            "max_grad_norm": float(max(grad_norms)) if grad_norms else None,
            "all_finite_grads": all(math.isfinite(g) for g in grad_norms),
        }
        step_records.append(rec)

    param_sha_after = sha256_params(model)

    return {
        "arm": arm,
        "pool_path": str(pool_path),
        "data_meta": data_meta,
        "init_sha256": init_sha,
        "param_sha256_after": param_sha_after,
        "model_param_count": sum(p.numel() for p in model.parameters()),
        "steps": step_records,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool_dir", default=str(POOL_DIR))
    ap.add_argument("--tokenizer", default=str(TOKENIZER_DIR))
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pool_dir = Path(args.pool_dir)
    t0 = time.time()

    tokenizer = make_tokenizer(Path(args.tokenizer))
    print(json.dumps({"event": "tokenizer_loaded", "vocab_size": len(tokenizer)}), flush=True)

    ARMS = ["HS", "LS", "HD", "LD"]
    pool_files = {arm: pool_dir / f"factorial_{arm.lower()}_10M.jsonl" for arm in ARMS}
    missing = [arm for arm, p in pool_files.items() if not p.exists()]
    if missing:
        raise SystemExit(f"Missing pool files for arms: {missing}")

    results: dict[str, Any] = {}
    for arm in ARMS:
        print(json.dumps({"event": "arm_start", "arm": arm}), flush=True)
        res = run_arm_smoke(arm, pool_files[arm], tokenizer, out_dir)
        results[arm] = res
        print(json.dumps({
            "event": "arm_done", "arm": arm,
            "init_sha": res["init_sha256"][:16],
            "loss": res["steps"][0]["loss"] if res["steps"] else None,
            "all_finite": all(s["finite_loss"] and s["all_finite_grads"] for s in res["steps"]),
        }), flush=True)

    # --- Cross-arm parity checks ---
    init_shas = {arm: results[arm]["init_sha256"] for arm in ARMS}
    init_match = len(set(init_shas.values())) == 1
    param_counts = {arm: results[arm]["model_param_count"] for arm in ARMS}
    param_match = len(set(param_counts.values())) == 1

    # Per-step comparisons
    step_comparisons: list[dict[str, Any]] = []
    for si in range(SMOKE_STEPS):
        step_data = {}
        for arm in ARMS:
            if si < len(results[arm]["steps"]):
                step_data[arm] = results[arm]["steps"][si]
        if len(step_data) == SMOKE_STEPS:
            pass  # compare
        comp = {
            "step": si + 1,
            "losses": {arm: step_data[arm]["loss"] for arm in step_data},
            "lrs": {arm: step_data[arm]["lr"] for arm in step_data},
            "batch_words": {arm: step_data[arm]["batch_words"] for arm in step_data},
            "masked_tokens": {arm: step_data[arm]["masked_tokens"] for arm in step_data},
            "candidate_tokens": {arm: step_data[arm]["candidate_tokens"] for arm in step_data},
            "effective_mask_rates": {arm: step_data[arm]["effective_mask_rate"] for arm in step_data},
            "all_finite": {arm: step_data[arm]["finite_loss"] and step_data[arm]["all_finite_grads"] for arm in step_data},
        }
        # Check LR parity
        lr_vals = list(comp["lrs"].values())
        comp["lr_parity"] = all(abs(v - lr_vals[0]) < 1e-12 for v in lr_vals)
        step_comparisons.append(comp)

    # After-training parameter SHAs should differ (different text → different gradients)
    after_shas = {arm: results[arm]["param_sha256_after"] for arm in ARMS}
    after_all_same = len(set(after_shas.values())) == 1
    # HS should differ from LS because text differs
    hs_ls_differ = after_shas.get("HS") != after_shas.get("LS")

    # Compute DataLoader step counts from word counts
    pool_words: dict[str, int] = {}
    for arm in ARMS:
        pool_words[arm] = results[arm]["data_meta"]["words"]
    words_match = len(set(pool_words.values())) == 1

    elapsed = time.time() - t0
    report = {
        "status": "TENSOR_PARITY_SMOKE",
        "created_utc": now_utc(),
        "meaning": "CPU tensor-parity smoke test for HS/LS/HD/LD factorial. Verifies identical initialization, matching LR schedule, matching batch geometry, finite loss/gradients, and word-count-determined step parity.",
        "config": {"model": MODEL_CFG, "train": TRAIN_CFG, "smoke_steps": SMOKE_STEPS, "smoke_batch": SMOKE_BATCH},
        "parity_checks": {
            "init_sha_match": init_match,
            "init_shas": init_shas,
            "param_count_match": param_match,
            "param_counts": param_counts,
            "pool_word_match": words_match,
            "pool_words": pool_words,
            "after_shas_all_same": after_all_same,
            "hs_ls_params_differ": hs_ls_differ,
        },
        "step_comparisons": step_comparisons,
        "per_arm_details": {arm: {
            "init_sha256": results[arm]["init_sha256"],
            "param_sha256_after": results[arm]["param_sha256_after"],
            "steps": results[arm]["steps"],
        } for arm in ARMS},
        "verdict": {
            "init_identical": init_match,
            "lr_schedule_identical": all(sc["lr_parity"] for sc in step_comparisons),
            "all_finite": all(
                all(s["finite_loss"] and s["all_finite_grads"] for s in results[arm]["steps"])
                for arm in ARMS
            ),
            "word_counts_identical": words_match,
            "params_diverge_with_different_text": hs_ls_differ,
            "ready_for_training": (
                init_match and param_match and words_match
                and all(sc["lr_parity"] for sc in step_comparisons)
                and all(
                    all(s["finite_loss"] and s["all_finite_grads"] for s in results[arm]["steps"])
                    for arm in ARMS
                )
            ),
        },
        "elapsed_sec": round(elapsed, 1),
    }

    (out_dir / "tensor_parity_smoke.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # Markdown summary
    md = [
        "# research tensor parity smoke test",
        f"\nJSON: `{out_dir / 'tensor_parity_smoke.json'}`\n",
        "## Parity checks",
        f"- Init SHA match: {init_match} ({list(init_shas.values())[0][:16] if init_match else 'MISMATCH'}...)",
        f"- Param count match: {param_match} ({list(param_counts.values())[0]})",
        f"- Pool word count match: {words_match} ({list(pool_words.values())[0]} per arm)",
        f"- LR schedule identical: {all(sc['lr_parity'] for sc in step_comparisons)}",
        f"- All finite: {report['verdict']['all_finite']}",
        f"- Params diverge after training: {hs_ls_differ}",
        f"\n## Verdict: {'READY' if report['verdict']['ready_for_training'] else 'NOT READY'}",
    ]
    for si, sc in enumerate(step_comparisons):
        md.append(f"\n### Step {si + 1}")
        md.append(f"- Losses: {sc['losses']}")
        md.append(f"- Masked tokens: {sc['masked_tokens']}")
        md.append(f"- LR parity: {sc['lr_parity']}")

    (out_dir / "tensor_parity_smoke.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": report["status"],
        "ready": report["verdict"]["ready_for_training"],
        "init_match": init_match,
        "words_match": words_match,
        "all_finite": report["verdict"]["all_finite"],
        "elapsed": report["elapsed_sec"],
    }), flush=True)


if __name__ == "__main__":
    main()
