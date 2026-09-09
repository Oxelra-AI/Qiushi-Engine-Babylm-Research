#!/usr/bin/env python3
"""research: Auxiliary Contrastive Softmax (ACS) paired mechanism screen.

Scientific purpose
------------------
Standard MLM distributes the masked-token gradient across the entire vocabulary
(40K entries), giving each wrong token a gradient proportional to its current
probability.  For context-conditioned alternatives — plausible words that must
bind differently under different contexts — this gradient is diluted: the
model can reduce loss by adjusting global compatibility without forming context-
discriminative representations.

ACS (Auxiliary Contrastive Softmax) adds a focused contrastive loss that
compares the correct token against only the K hardest competitors at each
masked position.  The gradient for each top-K wrong token is amplified by
roughly V/(K+1) ≈ 4400× relative to standard CE, making the decision boundary
between correct and contextually-confused alternatives load-bearing.

Design: paired 80M→100M tail continuations from the legal40k DeBERTa-v2 8×480
baseline, sharing weights, stream, tokenizer, masking seeds, and dropout seeds.
  • treatment: L = L_CE + α·L_ACS (K negatives)
  • control:   L = L_CE only (α = 0)
Both arms use fresh AdamW moments with a reheated cosine LR schedule over the
tail, matching the research reheat pattern.  The only intended difference is the
ACS auxiliary loss.

Success criterion (before endpoint-scale training):
  Treatment must show treatment-specific EWoK four-cell improvement (reduced
  stable reversals, improved conditional interaction) AND GlobalPIQA hard52
  improvement (more correct rows, lower mean top-minus-correct margin) relative
  to the matched control, without damaging held-text MLM loss or broad columns.

Substrate: legal40k_accum_compact_view_reinvest_seed43022, chck_80M.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import DebertaV2ForMaskedLM

USER_ROOT = Path(".").resolve()
COMPACT_EXPERIENCE_SCRIPTS = USER_ROOT / "experiments/archive" / 'compact_experience' / "scripts"
if str(COMPACT_EXPERIENCE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(COMPACT_EXPERIENCE_SCRIPTS))

import masking_curriculum_trainer as base  # noqa: E402


# ── ACS loss ─────────────────────────────────────────────────────────────────

def compute_acs_loss(
    logits_masked: torch.Tensor,
    labels_masked: torch.Tensor,
    topk: int = 8,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Auxiliary Contrastive Softmax over the K hardest negatives.

    For each masked position, build a (K+1)-way classification problem:
    correct token vs. the K strongest wrong tokens by logit value.

    The top-K selection is done on *detached* logits so there is no
    second-order gradient through the selection.  The loss itself is
    computed from the *live* logits, so gradient flows to the model
    through the correct and negative logit values.

    Returns (acs_loss, stats_dict).
    """
    n, V = logits_masked.shape
    stats: dict[str, float] = {}
    if n == 0:
        return torch.tensor(0.0, device=logits_masked.device, requires_grad=True), stats

    # Gather correct logit (with gradient)
    correct_logits = logits_masked.gather(1, labels_masked.unsqueeze(1))  # [n, 1]

    # Select top-K negative indices (no gradient through selection)
    with torch.no_grad():
        sel = logits_masked.detach().clone()
        sel.scatter_(1, labels_masked.unsqueeze(1), float("-inf"))
        _, neg_idx = sel.topk(topk, dim=1)  # [n, K]

    # Gather negative logits (with gradient) at the selected positions
    neg_logits = logits_masked.gather(1, neg_idx)  # [n, K]

    # Contrastive logits: position 0 = correct, positions 1..K = negatives
    contrastive = torch.cat([correct_logits, neg_logits], dim=1)  # [n, K+1]
    targets = torch.zeros(n, dtype=torch.long, device=logits_masked.device)

    loss = F.cross_entropy(contrastive, targets)

    # Diagnostics (detached)
    with torch.no_grad():
        # How often is the correct token already top-1?
        correct_is_top1 = (correct_logits.squeeze(1) >= neg_logits.max(dim=1).values).float()
        stats["acs_correct_top1_frac"] = float(correct_is_top1.mean().item())
        stats["acs_loss_raw"] = float(loss.item())
        # Mean margin: correct logit minus best negative logit
        margin = correct_logits.squeeze(1) - neg_logits.max(dim=1).values
        stats["acs_mean_margin"] = float(margin.mean().item())
        stats["acs_margin_negative_frac"] = float((margin < 0).float().mean().item())

    return loss, stats


# ── Infrastructure ────────────────────────────────────────────────────────────

def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="research ACS paired mechanism screen")
    p.add_argument("--example_jsonl", required=True)
    p.add_argument("--tokenizer_path", required=True)
    p.add_argument("--source_model_path", required=True)
    p.add_argument("--source_metrics_path", required=True)
    p.add_argument("--source_training_log", required=True)
    p.add_argument("--source_checkpoint_name", default="chck_80M")
    p.add_argument("--output_dir", required=True)

    # ACS parameters
    p.add_argument("--acs_alpha", type=float, default=0.0,
                   help="ACS loss weight.  0 = pure control (standard MLM).")
    p.add_argument("--acs_topk", type=int, default=8,
                   help="Number of hard negatives for ACS.")

    # Training parameters (matched across arms)
    p.add_argument("--max_total_word_exposure", type=int, default=100_000_000)
    p.add_argument("--max_seq_length", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--micro_batch_size", type=int, default=64)
    p.add_argument("--mask_prob", type=float, default=0.15)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.98)

    # LR schedule (reheat cosine, matching research reheat arm)
    p.add_argument("--reheat_peak_lr", type=float, default=3e-4)
    p.add_argument("--reheat_warmup_fraction", type=float, default=0.06)

    p.add_argument("--checkpoint_interval_words", type=int, default=10_000_000)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--train_rng_seed", type=int, default=53023)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def reset_all_rng(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_sha(t: torch.Tensor) -> str:
    arr = t.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(arr).hexdigest()


def read_source_checkpoint(metrics_path: Path, checkpoint_name: str) -> dict[str, Any]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    for rec in metrics.get("saved_checkpoints", []):
        if rec.get("name") == checkpoint_name:
            return dict(rec)
    raise RuntimeError(f"checkpoint {checkpoint_name!r} not found in {metrics_path}")


def read_training_log(log_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"empty training log: {log_path}")
    return rows


def find_source_step(training_rows: list[dict[str, Any]], consumed_words: int) -> dict[str, Any]:
    exact = [r for r in training_rows if int(r.get("cumulative_word_exposure", -1)) == consumed_words]
    if not exact:
        near = min(training_rows, key=lambda r: abs(int(r.get("cumulative_word_exposure", 0)) - consumed_words))
        raise RuntimeError(
            f"source checkpoint at {consumed_words} not exact logged boundary; "
            f"nearest step={near.get('step')}, words={near.get('cumulative_word_exposure')}"
        )
    return dict(exact[0])


def select_tail_examples(examples: list, consumed_words: int, total_target: int):
    cumulative = 0
    start_idx = None
    for i, ex in enumerate(examples):
        cumulative += int(ex.words)
        if cumulative == consumed_words:
            start_idx = i + 1
            break
        if cumulative > consumed_words:
            raise RuntimeError(f"consumed_words={consumed_words} inside example at row={i}, cumulative={cumulative}")
    if start_idx is None:
        raise RuntimeError(f"could not match consumed_words={consumed_words}")
    tail = []
    running = consumed_words
    for ex in examples[start_idx:]:
        w = int(ex.words)
        if running + w > total_target:
            raise RuntimeError(f"partial example: running={running}, next={w}, target={total_target}")
        tail.append(ex)
        running += w
        if running == total_target:
            break
    if running != total_target:
        raise RuntimeError(f"tail ended at {running}, target {total_target}")
    return tail, start_idx, running - consumed_words


def combine_microbatches(micro_batches):
    return {
        "input_ids": torch.cat([b["input_ids"] for b in micro_batches], dim=0),
        "attention_mask": torch.cat([b["attention_mask"] for b in micro_batches], dim=0),
        "word_group": torch.cat([b["word_group"] for b in micro_batches], dim=0),
        "words": torch.cat([b["words"] for b in micro_batches], dim=0),
    }


def effective_batch_iterator(loader, accum_steps):
    buf = []
    for batch in loader:
        buf.append(batch)
        if len(buf) == accum_steps:
            yield combine_microbatches(buf)
            buf = []
    if buf:
        yield combine_microbatches(buf)


def hf_cosine_lr(base_lr, warmup_steps, total_steps, completed_steps):
    s = max(0, int(completed_steps))
    if s < warmup_steps:
        return base_lr * (float(s) / float(max(1, warmup_steps)))
    progress = float(s - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    return base_lr * max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def set_optimizer_lr(optim, lr):
    for group in optim.param_groups:
        group["lr"] = lr


def main() -> None:
    args = build_args()
    if args.batch_size % args.micro_batch_size != 0:
        raise ValueError("batch_size must be divisible by micro_batch_size")
    accum_steps = args.batch_size // args.micro_batch_size

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    start_time = time.time()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    mode_label = f"acs_alpha{args.acs_alpha}_topk{args.acs_topk}" if args.acs_alpha > 0 else "standard_control"

    # ── Source checkpoint ──
    source_metrics_path = Path(args.source_metrics_path)
    source_training_log_path = Path(args.source_training_log)
    source_model_path = Path(args.source_model_path)
    source_rec = read_source_checkpoint(source_metrics_path, args.source_checkpoint_name)
    consumed_words = int(source_rec["actual_cumulative_word_exposure"])
    if consumed_words >= args.max_total_word_exposure:
        raise RuntimeError(f"source checkpoint already at {consumed_words}")

    training_rows = read_training_log(source_training_log_path)
    source_step_rec = find_source_step(training_rows, consumed_words)
    source_step = int(source_step_rec["step"])

    # ── Tokenizer and data ──
    tokenizer = base.make_portable_tokenizer(args.tokenizer_path)
    all_examples, jsonl_total_words, jsonl_total_rows, sample_rows = base.load_examples_jsonl(
        Path(args.example_jsonl), args.max_total_word_exposure
    )
    tail_examples, tail_start_idx, continuation_words = select_tail_examples(
        all_examples, consumed_words, args.max_total_word_exposure
    )

    dataset = base.MaskedChunkDataset(tail_examples, tokenizer, args.max_seq_length)
    total_steps = math.ceil(len(dataset) / args.batch_size)
    if total_steps <= 0:
        raise RuntimeError("empty tail dataset")
    final_global_step = source_step + total_steps

    # LR schedule
    reheat_warmup_steps = max(1, int(total_steps * args.reheat_warmup_fraction))

    def lr_for_update(local_step):
        return hf_cosine_lr(args.reheat_peak_lr, reheat_warmup_steps, total_steps, local_step - 1)

    def lr_after_update(local_step):
        return hf_cosine_lr(args.reheat_peak_lr, reheat_warmup_steps, total_steps, local_step)

    # ── Manifest ──
    manifest = {
        "status": "ACS_SCREEN_READY" if args.dry_run else "ACS_SCREEN_RUNNING",
        "created_utc": now_utc(),
        "purpose": (
            "Paired ACS mechanism screen: treatment uses Auxiliary Contrastive Softmax "
            "focusing MLM gradient on the K hardest competitors; control uses standard "
            "MLM only.  Both arms share weights, stream, tokenizer, masking/dropout seeds."
        ),
        "mode_label": mode_label,
        "mechanism": {
            "name": "Auxiliary Contrastive Softmax (ACS)",
            "acs_alpha": args.acs_alpha,
            "acs_topk": args.acs_topk,
            "loss_formula": "L = L_CE + alpha * L_ACS"
                if args.acs_alpha > 0
                else "L = L_CE (standard MLM)",
            "acs_description": (
                "For each masked position, compute a (K+1)-way contrastive loss "
                "between the correct token and the K tokens the model currently "
                "ranks highest (excluding the correct one).  Selection uses "
                "detached logits; loss flows through live logits.  This concentrates "
                "gradient on the context-dependent decision boundary."
            ),
        },
        "pairing_contract": {
            "same_source_checkpoint": True,
            "same_tail_rows": True,
            "same_mask_rng_seed": args.train_rng_seed,
            "same_dropout_seed": args.train_rng_seed,
            "fresh_adamw_moments": True,
            "only_intended_difference": f"ACS alpha={args.acs_alpha} topk={args.acs_topk} vs standard MLM",
        },
        "source_checkpoint": {
            "name": args.source_checkpoint_name,
            "path": str(source_model_path),
            "consumed_words": consumed_words,
            "source_step": source_step,
        },
        "stream": {
            "example_jsonl": str(args.example_jsonl),
            "tail_start_row": tail_start_idx,
            "tail_examples": len(tail_examples),
            "continuation_words": continuation_words,
        },
        "training": {
            "total_tail_steps": total_steps,
            "final_global_step": final_global_step,
            "batch_size": args.batch_size,
            "micro_batch_size": args.micro_batch_size,
            "max_seq_length": args.max_seq_length,
            "mask_prob": args.mask_prob,
            "optimizer": "AdamW_fresh_state",
            "reheat_peak_lr": args.reheat_peak_lr,
            "reheat_warmup_fraction": args.reheat_warmup_fraction,
            "reheat_warmup_steps": reheat_warmup_steps,
            "first_update_lr": lr_for_update(1),
            "last_update_lr": lr_for_update(total_steps),
            "seed": args.seed,
            "train_rng_seed": args.train_rng_seed,
        },
    }
    manifest_path = out / "acs_screen_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"event": "manifest", "mode": mode_label, **manifest["stream"],
                       "total_steps": total_steps}), flush=True)

    if args.dry_run:
        print(json.dumps({"event": "dry_run_done", "mode": mode_label}), flush=True)
        return

    # ── Training ──
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    reset_all_rng(args.seed)
    model = DebertaV2ForMaskedLM.from_pretrained(source_model_path, local_files_only=True)
    reset_all_rng(args.train_rng_seed)
    model.to(device)
    param_count = sum(p.numel() for p in model.parameters())

    loader = DataLoader(dataset, batch_size=args.micro_batch_size, shuffle=False,
                        collate_fn=base.collate, num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"))
    curriculum_state = base.MaskingCurriculumState(
        curriculum="wwm_fixed", mask_prob_start=args.mask_prob, mask_prob_end=args.mask_prob, switch_frac=1.0
    )
    curriculum_state.initialize(vocab_size=len(tokenizer), total_steps=total_steps)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.train_rng_seed)
    optim = torch.optim.AdamW(
        model.parameters(), lr=0.0, weight_decay=args.weight_decay,
        betas=(args.adam_beta1, args.adam_beta2)
    )

    print(json.dumps({"event": "training_start", "mode": mode_label, "device": str(device),
                       "params": param_count, "source_step": source_step,
                       "tail_steps": total_steps, "acs_alpha": args.acs_alpha,
                       "acs_topk": args.acs_topk}), flush=True)

    log_path = out / "training_log.jsonl"
    saved_checkpoints: list[dict[str, Any]] = []
    cumulative_tail_words = 0
    cumulative_total_words = consumed_words
    if args.checkpoint_interval_words > 0:
        next_ckpt_total = ((consumed_words // args.checkpoint_interval_words) + 1) * args.checkpoint_interval_words
    else:
        next_ckpt_total = None

    model.train()
    with log_path.open("w", encoding="utf-8") as logf:
        for local_step, batch in enumerate(effective_batch_iterator(loader, accum_steps), 1):
            words = int(batch.pop("words").sum().item())
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            word_group = batch["word_group"].to(device, non_blocking=True)

            curriculum_state.current_step = local_step - 1
            masked_inputs, labels = base.apply_masking_curriculum(
                input_ids, attention_mask, word_group, tokenizer, curriculum_state, gen
            )
            n_pred_total_t = (labels != -100).sum()
            n_pred_total = int(n_pred_total_t.item())
            if n_pred_total <= 0:
                raise RuntimeError(f"no masked tokens at step {local_step}")

            update_lr = lr_for_update(local_step)
            set_optimizer_lr(optim, update_lr)
            optim.zero_grad(set_to_none=True)

            weighted_ce_sum = 0.0
            weighted_acs_sum = 0.0
            step_acs_stats: dict[str, list[float]] = {}
            active_microbatches = 0
            batch_rows = input_ids.shape[0]

            for mb_start in range(0, batch_rows, args.micro_batch_size):
                mb_end = min(mb_start + args.micro_batch_size, batch_rows)
                sl_labels = labels[mb_start:mb_end]
                n_pred_i = int((sl_labels != -100).sum().item())
                if n_pred_i <= 0:
                    continue

                out_model = model(
                    input_ids=masked_inputs[mb_start:mb_end],
                    attention_mask=attention_mask[mb_start:mb_end],
                    labels=sl_labels,
                )
                ce_loss = out_model.loss
                if ce_loss is None:
                    raise RuntimeError("model returned no loss")

                scale = n_pred_i / n_pred_total

                # ── ACS auxiliary ──
                if args.acs_alpha > 0:
                    # Extract logits at masked positions
                    logits = out_model.logits  # [mb, seq, V]
                    mask_pos = sl_labels != -100
                    logits_masked = logits[mask_pos]  # [n_pred_i, V]
                    labels_masked = sl_labels[mask_pos]  # [n_pred_i]

                    acs_loss, acs_st = compute_acs_loss(
                        logits_masked, labels_masked, topk=args.acs_topk
                    )
                    total_loss = ce_loss + args.acs_alpha * acs_loss
                    weighted_acs_sum += float(acs_loss.detach().cpu()) * scale
                    for k, v in acs_st.items():
                        step_acs_stats.setdefault(k, []).append(v)
                else:
                    total_loss = ce_loss

                (total_loss * scale).backward()
                weighted_ce_sum += float(ce_loss.detach().cpu()) * scale
                active_microbatches += 1
                del out_model, total_loss
                if args.acs_alpha > 0:
                    del logits, logits_masked, acs_loss

            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
            optim.step()

            cumulative_tail_words += words
            cumulative_total_words += words
            global_step = source_step + local_step

            rec: dict[str, Any] = {
                "local_step": local_step,
                "global_step": global_step,
                "ce_loss": round(weighted_ce_sum, 6),
                "total_word_exposure": cumulative_total_words,
                "tail_word_exposure": cumulative_tail_words,
                "masked_tokens": n_pred_total,
                "update_lr": update_lr,
                "grad_norm": round(grad_norm, 6),
                "elapsed_sec": round(time.time() - start_time, 1),
            }
            if args.acs_alpha > 0:
                rec["acs_loss"] = round(weighted_acs_sum, 6)
                rec["total_loss"] = round(weighted_ce_sum + args.acs_alpha * weighted_acs_sum, 6)
                for k, vals in step_acs_stats.items():
                    rec[k] = round(sum(vals) / len(vals), 6) if vals else 0.0
            else:
                rec["total_loss"] = rec["ce_loss"]

            logf.write(json.dumps(rec) + "\n")
            logf.flush()

            if local_step == 1 or local_step % args.log_every == 0 or local_step == total_steps:
                wps = cumulative_tail_words / max(1, time.time() - start_time)
                print(json.dumps({"event": "train", "mode": mode_label, **rec,
                                   "wps": round(wps, 0)}), flush=True)

            # Checkpointing
            while (next_ckpt_total is not None
                   and cumulative_total_words >= next_ckpt_total
                   and next_ckpt_total <= args.max_total_word_exposure):
                name = (f"chck_{next_ckpt_total // 1_000_000}M"
                        if next_ckpt_total % 1_000_000 == 0
                        else f"chck_{next_ckpt_total}w")
                cp = out / "hf_model" / name
                base.save_hf_checkpoint(model, tokenizer, cp)
                saved_checkpoints.append({
                    "name": name,
                    "total_word_exposure": cumulative_total_words,
                    "tail_word_exposure": cumulative_tail_words,
                    "local_step": local_step,
                    "global_step": global_step,
                    "path": str(cp),
                })
                print(json.dumps({"event": "checkpoint", "mode": mode_label, "name": name,
                                   "total_words": cumulative_total_words}), flush=True)
                next_ckpt_total += args.checkpoint_interval_words

            del input_ids, attention_mask, word_group, masked_inputs, labels

    # Final checkpoint
    final_dir = out / "hf_model" / "chck_100M"
    if not any(c["name"] == "chck_100M" for c in saved_checkpoints):
        base.save_hf_checkpoint(model, tokenizer, final_dir)
        saved_checkpoints.append({
            "name": "chck_100M",
            "total_word_exposure": cumulative_total_words,
            "tail_word_exposure": cumulative_tail_words,
            "local_step": total_steps,
            "global_step": final_global_step,
            "path": str(final_dir),
        })

    metrics = {
        "status": "ACS_SCREEN_DONE",
        "variant": mode_label,
        "model_family": "DebertaV2ForMaskedLM",
        "parameter_count": param_count,
        "vocab_size": len(tokenizer),
        "mechanism": manifest["mechanism"],
        "source_checkpoint": manifest["source_checkpoint"],
        "pairing_contract": manifest["pairing_contract"],
        "word_exposure": cumulative_total_words,
        "tail_word_exposure": cumulative_tail_words,
        "actual_training_steps": total_steps,
        "final_global_step": final_global_step,
        "saved_checkpoints": saved_checkpoints,
        "training_log": str(log_path),
    }
    (out / "scientific_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "status": metrics["status"],
        "mode": mode_label,
        "word_exposure": cumulative_total_words,
        "steps": total_steps,
        "checkpoints": [c["name"] for c in saved_checkpoints],
        "metrics": str(out / "scientific_metrics.json"),
    }), flush=True)


if __name__ == "__main__":
    main()
