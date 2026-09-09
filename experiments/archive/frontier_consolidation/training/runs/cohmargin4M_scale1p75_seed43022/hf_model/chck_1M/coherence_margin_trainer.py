#!/usr/bin/env python3
"""research coherent-context margin MLM trainer.

A new training-time mechanism after the private-scale family closed: keep the
validated compact-view/source-diversity legal stream and the scale1.75
function-preserving adapter architecture, but add a small target-shared
coherence-margin loss.  For the same masked targets, the model is trained to
assign lower NLL in the coherent row context than in a deterministic block-shuffled
context that preserves mask positions, labels, padding/special positions, and the
unmasked context token multiset.

Legal accounting: the disrupted view is a second exposed view of the same legal
row, so charged words = coherent_words * view_charge_multiplier (default 2.0).
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

HERE = _public_path('experiments/archive/frontier_consolidation/training/runs/cohmargin4M_scale1p75_seed43022/hf_model/chck_1M')
USER_ROOT = _public_path('experiments/archive/frontier_consolidation/training')
STUDY = _public_path('experiments/archive/frontier_consolidation')
WORKSPACE = _public_path('experiments/archive/frontier_consolidation')
BASE_TRAINER = _public_path('experiments/archive/compact_experience/scripts/masking_curriculum_trainer.py')
ADAPTER_MODELING = _public_path('experiments/archive/frontier_consolidation/training/runs/cohmargin4M_scale1p75_seed43022/hf_model/chck_1M/adapter_scaled_modeling.py')
DEFAULT_STREAM = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl')
DEFAULT_TOKENIZER = _public_path('experiments/archive/frontier_consolidation/data/compliant_tokenizer')


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(USER_ROOT))
    except Exception:
        return str(p)


def load_base():
    spec = importlib.util.spec_from_file_location("base_masking", BASE_TRAINER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load base trainer: {BASE_TRAINER}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


BASE = load_base()
sys.path.insert(0, str(HERE))
from adapter_scaled_modeling import AdapterDebertaV2ForMaskedLM  # noqa: E402


def reset_all_rng(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl_stream_upto(path: Path, max_charged_words: int, view_charge: float, limit_rows: int | None = None):
    """Load whole JSONL rows until adding another row would exceed charged cap."""
    examples = []
    sample_rows = []
    coherent_words = 0
    total_file_words = 0
    total_file_rows = 0
    target_coherent_cap = int(math.floor(max_charged_words / max(view_charge, 1e-9)))
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            total_file_rows += 1
            text = str(obj["text"])
            words = int(obj.get("words", len(text.split())))
            actual = len(text.split())
            if actual != words:
                raise RuntimeError(f"word mismatch at row {total_file_rows}: field={words} actual={actual}")
            total_file_words += words
            if limit_rows is not None and len(examples) >= limit_rows:
                continue
            if coherent_words + words <= target_coherent_cap:
                ex_id = int(obj.get("example_id", total_file_rows - 1))
                source = str(obj.get("source", "example_jsonl"))
                examples.append(BASE.Example(text=text, words=words, example_id=ex_id, source=source))
                coherent_words += words
                if len(sample_rows) < 10:
                    sample_rows.append({k: obj[k] for k in obj.keys() if k != "text"})
            elif limit_rows is None:
                break
            else:
                break
    charged_words = int(round(coherent_words * view_charge))
    return examples, coherent_words, charged_words, total_file_words, total_file_rows, sample_rows, target_coherent_cap


def build_adapter_model(args: argparse.Namespace, tokenizer):
    max_pos = max(args.max_position_embeddings, args.max_seq_length + 8)
    pos_att_type = [x.strip() for x in args.deberta_pos_att_type.split(",") if x.strip()]
    cfg = BASE.DebertaV2Config(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.n_layer,
        num_attention_heads=args.n_head,
        intermediate_size=args.hidden_size * args.ffn_mult,
        max_position_embeddings=max_pos,
        max_relative_positions=args.max_relative_positions,
        position_buckets=args.position_buckets,
        relative_attention=True,
        pos_att_type=pos_att_type,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    cfg.adapter_bottleneck = args.adapter_bottleneck
    cfg.adapter_activation = args.adapter_activation
    cfg.adapter_enabled = bool(args.adapter_enabled)
    cfg.adapter_scale = float(args.adapter_scale)
    model = AdapterDebertaV2ForMaskedLM(cfg)
    model.register_for_auto_class("AutoModelForMaskedLM")
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model


def optimizer_groups(model: torch.nn.Module, weight_decay: float):
    zero_ids = {id(p) for n, p in model.named_parameters() if ".adapter.up." in n}
    normal, zero_wd = [], []
    for p in model.parameters():
        (zero_wd if id(p) in zero_ids else normal).append(p)
    return [
        {"params": normal, "weight_decay": weight_decay},
        {"params": zero_wd, "weight_decay": 0.0},
    ]


def make_disrupted_context(masked_inputs: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor,
                           tokenizer, gen: torch.Generator, span_tokens: int) -> torch.Tensor:
    """Block-shuffle only unmasked context tokens; keep targets/masks/special/pad fixed."""
    bad = masked_inputs.clone()
    device = masked_inputs.device
    special_ids = torch.tensor(sorted(tokenizer.all_special_ids), device=device)
    selected = labels != -100
    non_special = ~torch.isin(masked_inputs, special_ids)
    for b in range(masked_inputs.shape[0]):
        ctx = (attention_mask[b].bool() & (~selected[b]) & non_special[b]).nonzero(as_tuple=False).flatten()
        if ctx.numel() < span_tokens * 2:
            continue
        n_full = int(ctx.numel()) // span_tokens
        if n_full < 2:
            continue
        blocks = [ctx[i * span_tokens:(i + 1) * span_tokens] for i in range(n_full)]
        order = torch.randperm(n_full, generator=gen, device=device)
        if bool(torch.equal(order, torch.arange(n_full, device=device))):
            order = torch.roll(order, shifts=1)
        src_vals = [masked_inputs[b, blocks[int(j.item())]].clone() for j in order]
        for dest, vals in zip(blocks, src_vals):
            bad[b, dest] = vals
    return bad


def masked_ce_vec(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    mask = labels != -100
    if not bool(mask.any()):
        return logits.new_zeros((0,), dtype=torch.float32)
    return F.cross_entropy(logits[mask].float(), labels[mask], reduction="none")


def verify_disruption(batch: dict[str, torch.Tensor], masked_inputs: torch.Tensor, labels: torch.Tensor,
                      bad_inputs: torch.Tensor, tokenizer) -> dict[str, Any]:
    attn = batch["attention_mask"].to(masked_inputs.device).bool()
    selected = labels != -100
    special_ids = torch.tensor(sorted(tokenizer.all_special_ids), device=masked_inputs.device)
    special_or_pad = (~attn) | torch.isin(masked_inputs, special_ids)
    target_mismatch = int(((masked_inputs != bad_inputs) & selected).sum().item())
    special_mismatch = int(((masked_inputs != bad_inputs) & special_or_pad).sum().item())
    mask_position_mismatch = int(((labels != -100) != selected).sum().item())
    rows_bad_multiset = 0
    moved_tokens = []
    for b in range(masked_inputs.shape[0]):
        ctx = attn[b] & (~selected[b]) & (~torch.isin(masked_inputs[b], special_ids))
        coh = sorted(int(x) for x in masked_inputs[b][ctx].detach().cpu().tolist())
        bad = sorted(int(x) for x in bad_inputs[b][ctx].detach().cpu().tolist())
        if coh != bad:
            rows_bad_multiset += 1
        moved_tokens.append(int(((masked_inputs[b] != bad_inputs[b]) & ctx).sum().item()))
    return {
        "target_mismatch_tokens": target_mismatch,
        "special_or_pad_mismatch_tokens": special_mismatch,
        "mask_position_mismatch_tokens": mask_position_mismatch,
        "rows_with_context_multiset_mismatch": rows_bad_multiset,
        "moved_context_tokens_mean": float(mean(moved_tokens)) if moved_tokens else 0.0,
        "moved_context_tokens_min": min(moved_tokens) if moved_tokens else 0,
        "moved_context_tokens_max": max(moved_tokens) if moved_tokens else 0,
    }


def save_hf_checkpoint(model, tokenizer, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dst, safe_serialization=True)
    tokenizer.save_pretrained(dst)
    BASE.force_portable_tokenizer_config(dst)
    for src in [_public_path('experiments/archive/frontier_consolidation/training/runs/cohmargin4M_scale1p75_seed43022/hf_model/chck_1M/coherence_margin_trainer.py'), ADAPTER_MODELING]:
        target = dst / src.name
        if not target.exists():
            shutil.copy2(src, target)


def build_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--example_jsonl", default=str(DEFAULT_STREAM))
    ap.add_argument("--tokenizer_path", default=str(DEFAULT_TOKENIZER))
    ap.add_argument("--tokenizer_label", default="compliant16k_reinvest10M")
    ap.add_argument("--max_word_exposure", type=int, default=4_000_000, help="charged words including disrupted views")
    ap.add_argument("--view_charge_multiplier", type=float, default=2.0)
    ap.add_argument("--checkpoint_words", type=int, default=1_000_000)
    ap.add_argument("--micro_batch_size", type=int, default=128)
    ap.add_argument("--grad_accum_steps", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seq_length", type=int, default=256)
    ap.add_argument("--max_seq_length", type=int, default=256)
    ap.add_argument("--max_position_embeddings", type=int, default=512)
    ap.add_argument("--hidden_size", type=int, default=480)
    ap.add_argument("--n_layer", type=int, default=8)
    ap.add_argument("--n_head", type=int, default=8)
    ap.add_argument("--ffn_mult", type=int, default=4)
    ap.add_argument("--position_buckets", type=int, default=256)
    ap.add_argument("--max_relative_positions", type=int, default=256)
    ap.add_argument("--deberta_pos_att_type", default="p2c,c2p")
    ap.add_argument("--adapter_bottleneck", type=int, default=128)
    ap.add_argument("--adapter_activation", default="gelu")
    ap.add_argument("--adapter_enabled", type=int, choices=[0, 1], default=1)
    ap.add_argument("--adapter_scale", type=float, default=1.75)
    ap.add_argument("--mask_prob", type=float, default=0.15)
    ap.add_argument("--margin_lambda", type=float, default=0.10)
    ap.add_argument("--margin", type=float, default=0.20)
    ap.add_argument("--disrupt_span_tokens", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_fraction", type=float, default=0.06)
    ap.add_argument("--lr_total_steps", type=int, default=1265, help="full 100M charged horizon for 2-view training")
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--extra_init_seed", type=int, default=-1)
    ap.add_argument("--train_rng_seed", type=int, default=43022)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--gradient_checkpointing", type=int, choices=[0, 1], default=1)
    ap.add_argument("--smoke_only", action="store_true")
    ap.add_argument("--smoke_rows", type=int, default=8)
    return ap.parse_args()


def main() -> None:
    args = build_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    start = time.time()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = BASE.make_portable_tokenizer(args.tokenizer_path)
    if args.smoke_only:
        examples, coherent_words, charged_words, total_file_words, total_file_rows, sample_rows, target_cap = load_jsonl_stream_upto(
            Path(args.example_jsonl), max_charged_words=10_000_000_000, view_charge=args.view_charge_multiplier, limit_rows=args.smoke_rows)
    else:
        examples, coherent_words, charged_words, total_file_words, total_file_rows, sample_rows, target_cap = load_jsonl_stream_upto(
            Path(args.example_jsonl), args.max_word_exposure, args.view_charge_multiplier, limit_rows=None)
    if not examples:
        raise RuntimeError("No examples loaded")

    dataset = BASE.MaskedChunkDataset(examples, tokenizer, args.max_seq_length)
    loader = DataLoader(dataset, batch_size=args.micro_batch_size, shuffle=False, collate_fn=BASE.collate,
                        num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    total_micro_steps = len(loader)
    total_optim_steps = math.ceil(total_micro_steps / max(1, args.grad_accum_steps))

    manifest = {
        "status": "COHERENCE_MARGIN_TRAINER_PREPARED" if args.smoke_only else "COHERENCE_MARGIN_TRAINING_RUN",
        "created_utc": now(),
        "example_jsonl": rel(args.example_jsonl),
        "example_jsonl_sha256": sha256_file(Path(args.example_jsonl)),
        "tokenizer_path": rel(args.tokenizer_path),
        "max_charged_word_exposure_cap": args.max_word_exposure,
        "view_charge_multiplier": args.view_charge_multiplier,
        "coherent_words_loaded": coherent_words,
        "charged_words_planned": charged_words,
        "target_coherent_cap": target_cap,
        "num_examples": len(examples),
        "total_file_words_seen_or_known": total_file_words,
        "total_file_rows_seen_or_known": total_file_rows,
        "sample_rows": sample_rows,
        "micro_batch_size": args.micro_batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "total_micro_steps": total_micro_steps,
        "total_optim_steps": total_optim_steps,
        "masking": {"mode": "wwm_fixed", "mask_prob": args.mask_prob},
        "coherence_margin": {"lambda": args.margin_lambda, "margin": args.margin, "disrupt_span_tokens": args.disrupt_span_tokens},
        "lr_horizon": {"lr_total_steps": args.lr_total_steps, "warmup_fraction": args.warmup_fraction},
        "legal_accounting": "charged words include both coherent and disrupted legal-row views; no external data or labels are used",
    }
    (out / "coherence_margin_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    reset_all_rng(args.seed)
    if args.extra_init_seed >= 0:
        reset_all_rng(args.extra_init_seed)
    model = build_adapter_model(args, tokenizer)
    if args.train_rng_seed >= 0:
        reset_all_rng(args.train_rng_seed)
    param_count = sum(p.numel() for p in model.parameters())
    adapter_n = sum(p.numel() for n, p in model.named_parameters() if ".adapter." in n)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.smoke_only else "cpu")
    model.to(device)
    curriculum_state = BASE.MaskingCurriculumState(curriculum="wwm_fixed", mask_prob_start=args.mask_prob, mask_prob_end=args.mask_prob)
    curriculum_state.initialize(vocab_size=len(tokenizer), total_steps=max(1, total_micro_steps))
    gen = torch.Generator(device=device)
    gen.manual_seed(args.train_rng_seed if args.train_rng_seed >= 0 else args.seed)

    # Mechanical smoke on the first batch before any optimizer step.
    first = next(iter(loader))
    input_ids = first["input_ids"].to(device)[:, :args.seq_length].contiguous()
    attn = first["attention_mask"].to(device)[:, :args.seq_length].contiguous()
    word_group = first["word_group"].to(device)[:, :args.seq_length].contiguous()
    masked_inputs, labels = BASE.apply_masking_curriculum(input_ids, attn, word_group, tokenizer, curriculum_state, gen)
    bad_inputs = make_disrupted_context(masked_inputs, labels, attn, tokenizer, gen, args.disrupt_span_tokens)
    smoke = verify_disruption({"attention_mask": attn}, masked_inputs, labels, bad_inputs, tokenizer)
    smoke.update({"param_count": param_count, "adapter_params": adapter_n, "device": str(device), "n_masked_tokens": int((labels != -100).sum().item())})
    (out / "coherence_margin_smoke.json").write_text(json.dumps(smoke, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "cohmargin_smoke", **smoke}), flush=True)
    if smoke["target_mismatch_tokens"] or smoke["special_or_pad_mismatch_tokens"] or smoke["rows_with_context_multiset_mismatch"]:
        raise RuntimeError({"bad_disruption_smoke": smoke})
    if args.smoke_only:
        print(json.dumps({"status": "SMOKE_ONLY_COMPLETE", "manifest": rel(out / "coherence_margin_manifest.json"), "smoke": rel(out / "coherence_margin_smoke.json")}, indent=2), flush=True)
        return

    optim = torch.optim.AdamW(optimizer_groups(model, args.weight_decay), lr=args.learning_rate, betas=(0.9, 0.98))
    warmup = max(1, int(args.lr_total_steps * args.warmup_fraction))
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=warmup, num_training_steps=args.lr_total_steps)
    log_path = out / "training_log.jsonl"
    saved_checkpoints = []
    next_ckpt = args.checkpoint_words if args.checkpoint_words > 0 else None
    cumulative_coherent_words = 0
    cumulative_charged_words = 0
    optim_step = 0
    accum_i = 0
    accum_loss = 0.0
    accum_mlm = 0.0
    accum_margin = 0.0
    loss_values = []
    model.train()
    optim.zero_grad(set_to_none=True)

    with log_path.open("w", encoding="utf-8") as logf:
        for micro_step, batch in enumerate(loader, 1):
            words = int(batch.pop("words").sum().item())
            input_ids = batch["input_ids"].to(device, non_blocking=True)[:, :args.seq_length].contiguous()
            attn = batch["attention_mask"].to(device, non_blocking=True)[:, :args.seq_length].contiguous()
            word_group = batch["word_group"].to(device, non_blocking=True)[:, :args.seq_length].contiguous()
            curriculum_state.current_step = micro_step - 1
            masked_inputs, labels = BASE.apply_masking_curriculum(input_ids, attn, word_group, tokenizer, curriculum_state, gen)
            bad_inputs = make_disrupted_context(masked_inputs, labels, attn, tokenizer, gen, args.disrupt_span_tokens)

            coh = model(input_ids=masked_inputs, attention_mask=attn, labels=labels)
            if coh.loss is None:
                raise RuntimeError("coherent MLM loss missing")
            bad = model(input_ids=bad_inputs, attention_mask=attn)
            coh_ce = masked_ce_vec(coh.logits, labels)
            bad_ce = masked_ce_vec(bad.logits, labels)
            if coh_ce.numel() == 0:
                margin_loss = coh.loss.new_tensor(0.0)
                nll_gap = 0.0
            else:
                margin_terms = F.softplus(args.margin + coh_ce - bad_ce)
                margin_loss = margin_terms.mean()
                nll_gap = float((bad_ce.detach() - coh_ce.detach()).mean().cpu())
            loss = coh.loss + args.margin_lambda * margin_loss
            (loss / args.grad_accum_steps).backward()

            cumulative_coherent_words += words
            cumulative_charged_words = int(round(cumulative_coherent_words * args.view_charge_multiplier))
            accum_i += 1
            accum_loss += float(loss.detach().cpu())
            accum_mlm += float(coh.loss.detach().cpu())
            accum_margin += float(margin_loss.detach().cpu())

            did_step = False
            if accum_i >= args.grad_accum_steps or micro_step == total_micro_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                optim_step += 1
                did_step = True
                rec = {
                    "micro_step": micro_step,
                    "optim_step": optim_step,
                    "loss": accum_loss / accum_i,
                    "mlm_loss": accum_mlm / accum_i,
                    "margin_loss": accum_margin / accum_i,
                    "nll_bad_minus_coh_last_micro": nll_gap,
                    "lr": float(sched.get_last_lr()[0]),
                    "coherent_words": cumulative_coherent_words,
                    "charged_word_exposure": cumulative_charged_words,
                    "batch_words_last_micro": words,
                    "effective_mask_rate_last_micro": int((labels != -100).sum().item()) / max(1, int(attn.sum().item())),
                    "elapsed_sec": round(time.time() - start, 1),
                }
                logf.write(json.dumps(rec) + "\n")
                loss_values.append(rec["loss"])
                if optim_step == 1 or optim_step % args.log_every == 0 or micro_step == total_micro_steps:
                    print(json.dumps({"event": "train", **rec}), flush=True)
                accum_i = 0
                accum_loss = accum_mlm = accum_margin = 0.0

            while did_step and next_ckpt is not None and cumulative_charged_words >= next_ckpt and next_ckpt <= args.max_word_exposure:
                name = f"chck_{next_ckpt // 1_000_000}M" if next_ckpt % 1_000_000 == 0 else f"chck_{next_ckpt}w"
                cp = out / "hf_model" / name
                save_hf_checkpoint(model, tokenizer, cp)
                saved_checkpoints.append({"name": name, "target_charged_word_exposure": next_ckpt, "actual_charged_word_exposure": cumulative_charged_words, "coherent_words": cumulative_coherent_words, "path": rel(cp)})
                print(json.dumps({"event": "checkpoint_saved", "name": name, "charged_words": cumulative_charged_words, "coherent_words": cumulative_coherent_words}), flush=True)
                next_ckpt += args.checkpoint_words

    save_hf_checkpoint(model, tokenizer, out / "hf_model" / "final")
    # Also keep base-compatible final model root for older local scripts if needed.
    legacy_root = out / "hf_model"
    if not (legacy_root / "config.json").exists():
        # Do not duplicate the safetensors; evaluation should use endpoint=final.
        pass

    metrics = {
        "status": "COHERENCE_MARGIN_TRAINING_DONE",
        "variant": "scale1p75_coherent_context_margin_mlm",
        "model_family": "AdapterDebertaV2ForMaskedLM",
        "parameter_count": param_count,
        "adapter_params": adapter_n,
        "vocab_size": len(tokenizer),
        "tokenizer_label": args.tokenizer_label,
        "coherent_words": cumulative_coherent_words,
        "charged_word_exposure": cumulative_charged_words,
        "max_charged_word_exposure_cap": args.max_word_exposure,
        "view_charge_multiplier": args.view_charge_multiplier,
        "actual_training_micro_steps": total_micro_steps,
        "actual_optimizer_steps": optim_step,
        "loss_first": loss_values[0] if loss_values else None,
        "loss_last": loss_values[-1] if loss_values else None,
        "margin_lambda": args.margin_lambda,
        "margin": args.margin,
        "disrupt_span_tokens": args.disrupt_span_tokens,
        "learning_rate": args.learning_rate,
        "warmup_fraction": args.warmup_fraction,
        "lr_total_steps": args.lr_total_steps,
        "seed": args.seed,
        "train_rng_seed": args.train_rng_seed,
        "saved_checkpoints": saved_checkpoints,
        "manifest": rel(out / "coherence_margin_manifest.json"),
        "smoke": rel(out / "coherence_margin_smoke.json"),
        "scientific_reading": "Pilot tests whether an MLM-channel coherent-vs-disrupted context margin creates a structured context-learning pressure under legal charged exposure. It is not private-scale endpoint tuning.",
    }
    (out / "scientific_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "done", "status": metrics["status"], "charged_word_exposure": cumulative_charged_words, "coherent_words": cumulative_coherent_words, "optim_steps": optim_step, "loss_first": metrics["loss_first"], "loss_last": metrics["loss_last"], "checkpoints": [c["name"] for c in saved_checkpoints], "final_model": rel(out / "hf_model" / "final")}), flush=True)


if __name__ == "__main__":
    main()
