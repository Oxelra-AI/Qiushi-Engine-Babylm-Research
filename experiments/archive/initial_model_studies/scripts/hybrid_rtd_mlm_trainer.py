#!/usr/bin/env python3
"""research Hybrid RTD+MLM Trainer for BabyLM Strict-Small.

Implements ELECTRA/DeBERTaV3-style replaced token detection PLUS standard MLM
on the same discriminator model. This gives:
- 100% of tokens get RTD training signal (vs 15% for pure MLM)
- MLM head is preserved for BabyLM evaluation compatibility
- Single changed factor vs baseline: training objective (WWM → WWM + RTD)

Architecture:
- Generator: small DeBERTa-v2 (2 layers, hidden 128, 4 heads)
- Discriminator: standard DeBERTa-v2 8×480 (same as protected backbone)
- GDES: discriminator embeddings shared to generator with stop_gradient

Training:
- Generator: MLM on masked 15% tokens → produces replacement tokens
- Discriminator input: original tokens with masked positions replaced by generator samples
- Discriminator loss: MLM on masked positions + λ_rtd × RTD on ALL positions
- RTD: binary classification (original vs replaced) at each position

Reference: He et al. 2021 "DeBERTaV3: Improving DeBERTa using ELECTRA-Style
Pre-Training with Gradient-Disentangled Embedding Sharing"
"""
from __future__ import annotations
import argparse, json, math, os, pathlib, sys, time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Reuse infrastructure from existing trainer
sys.path.insert(0, str(pathlib.Path('experiments/archive/initial_model_studies/training/scripts').resolve()))
from babylm_masked_train import (
    make_portable_tokenizer,
    MaskedChunkDataset,
    save_hf_checkpoint,
)

ROOT = pathlib.Path('experiments/archive/initial_model_studies')


# ─── Generator Model ───────────────────────────────────────────────────────────

class RTDGenerator(nn.Module):
    """Small DeBERTa-v2 generator for producing replacement tokens."""

    def __init__(self, vocab_size: int, hidden: int = 128, n_layers: int = 2,
                 n_heads: int = 4, max_position: int = 512,
                 disc_embeddings: Optional[nn.Embedding] = None):
        super().__init__()
        self.hidden = hidden
        self.vocab_size = vocab_size

        # Token embeddings: shared from discriminator via GDES (stop_gradient)
        if disc_embeddings is not None:
            self.embed_proj = nn.Linear(disc_embeddings.embedding_dim, hidden, bias=False)
            self.disc_embeddings = disc_embeddings  # reference, not owned
            self.own_embeddings = None
        else:
            self.own_embeddings = nn.Embedding(vocab_size, hidden)
            self.embed_proj = None
            self.disc_embeddings = None

        self.pos_embeddings = nn.Embedding(max_position, hidden)
        self.embed_norm = nn.LayerNorm(hidden)
        self.embed_dropout = nn.Dropout(0.1)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=hidden * 4,
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # MLM head
        self.mlm_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, vocab_size),
        )

    def get_token_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.disc_embeddings is not None:
            # GDES: use discriminator embeddings but stop gradient
            with torch.no_grad():
                emb = self.disc_embeddings(input_ids)
            emb = emb.detach()  # stop gradient from flowing back
            emb = self.embed_proj(emb)
        else:
            emb = self.own_embeddings(input_ids)
        return emb

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        seq_len = input_ids.size(1)
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        emb = self.get_token_embeddings(input_ids) + self.pos_embeddings(pos_ids)
        emb = self.embed_norm(emb)
        emb = self.embed_dropout(emb)

        if attention_mask is not None:
            # Convert 0/1 mask to additive mask for nn.TransformerEncoder
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None

        hidden = self.encoder(emb, src_key_padding_mask=src_key_padding_mask)
        logits = self.mlm_head(hidden)
        return logits


# ─── RTD Head for Discriminator ────────────────────────────────────────────────

class RTDHead(nn.Module):
    """Binary classification head: original (1) vs replaced (0) per token."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.dense(hidden_states)
        x = self.activation(x)
        x = self.classifier(x).squeeze(-1)
        return x


# ─── Hybrid Training Logic ─────────────────────────────────────────────────────

def create_rtd_inputs(
    input_ids: torch.Tensor,
    masked_positions: torch.Tensor,
    generator_logits: torch.Tensor,
    original_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Replace masked positions with generator samples; create RTD labels.

    Returns:
        rtd_input_ids: input with masked positions replaced by generator samples
        rtd_labels: 1 where token is original, 0 where replaced
    """
    # Sample from generator distribution at masked positions
    with torch.no_grad():
        masked_logits = generator_logits[masked_positions]
        probs = F.softmax(masked_logits, dim=-1)
        sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

    # Build discriminator input: original tokens with masked positions replaced
    rtd_input_ids = original_ids.clone()
    rtd_input_ids[masked_positions] = sampled_tokens

    # RTD labels: 1 if token is original, 0 if replaced
    rtd_labels = (rtd_input_ids == original_ids).long()

    return rtd_input_ids, rtd_labels


def compute_hybrid_loss(
    discriminator_model,
    rtd_head: RTDHead,
    rtd_input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    original_ids: torch.Tensor,
    masked_positions: torch.Tensor,
    rtd_labels: torch.Tensor,
    lambda_rtd: float = 50.0,
) -> tuple[torch.Tensor, dict]:
    """
    Compute hybrid MLM + RTD loss on discriminator.

    MLM loss: on masked positions only (discriminator predicts original token)
    RTD loss: on all positions (binary: original vs replaced)
    """
    # Forward pass through discriminator
    outputs = discriminator_model(
        input_ids=rtd_input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )

    # MLM loss on masked positions
    mlm_logits = outputs.logits  # (batch, seq_len, vocab_size)
    mlm_targets = original_ids.clone()
    mlm_targets[~masked_positions] = -100  # ignore non-masked
    mlm_loss = F.cross_entropy(
        mlm_logits.view(-1, mlm_logits.size(-1)),
        mlm_targets.view(-1),
        ignore_index=-100,
    )

    # RTD loss on all positions
    hidden_states = outputs.hidden_states[-1]  # last layer hidden
    rtd_logits = rtd_head(hidden_states)  # (batch, seq_len)

    # Only compute RTD loss where attention_mask is 1
    active_positions = attention_mask.bool().view(-1)
    active_rtd_logits = rtd_logits.view(-1)[active_positions]
    active_rtd_labels = rtd_labels.float().view(-1)[active_positions]

    rtd_loss = F.binary_cross_entropy_with_logits(
        active_rtd_logits, active_rtd_labels
    )

    # Combined loss
    total_loss = mlm_loss + lambda_rtd * rtd_loss

    metrics = {
        'mlm_loss': mlm_loss.item(),
        'rtd_loss': rtd_loss.item(),
        'total_loss': total_loss.item(),
        'rtd_accuracy': ((active_rtd_logits > 0).float() == active_rtd_labels).float().mean().item(),
    }

    return total_loss, metrics


# ─── Main Training Function ────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Hybrid RTD+MLM Trainer')
    # Data
    p.add_argument('--output_dir', required=True)
    p.add_argument('--example_jsonl', required=True)
    p.add_argument('--example_jsonl_meta', default='')
    p.add_argument('--example_jsonl_label', default='')
    p.add_argument('--max_word_exposure', type=int, default=100_000_000)
    p.add_argument('--example_pool_words', type=int, default=10_000_000)
    p.add_argument('--checkpoint_words', type=int, default=20_000_000)
    p.add_argument('--words_per_example', type=int, default=160)
    # Tokenizer
    p.add_argument('--tokenizer_label', default='baseline16k')
    p.add_argument('--tokenization_summary_limit', type=int, default=0)
    # Masking
    p.add_argument('--mask_mode', default='wwm', choices=['token', 'wwm'])
    p.add_argument('--mask_prob', type=float, default=0.15)
    # Model - discriminator
    p.add_argument('--model_type', default='deberta_v2')
    p.add_argument('--hidden_size', type=int, default=480)
    p.add_argument('--n_layer', type=int, default=8)
    p.add_argument('--n_head', type=int, default=12)
    p.add_argument('--ffn_mult', type=int, default=4)
    # Model - generator
    p.add_argument('--gen_hidden', type=int, default=128)
    p.add_argument('--gen_layers', type=int, default=2)
    p.add_argument('--gen_heads', type=int, default=4)
    # RTD
    p.add_argument('--lambda_rtd', type=float, default=50.0)
    # Training
    p.add_argument('--seq_length', type=int, default=256)
    p.add_argument('--max_seq_length', type=int, default=256)
    p.add_argument('--max_position_embeddings', type=int, default=512)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--learning_rate', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--warmup_fraction', type=float, default=0.05)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--extra_init_seed', type=int, default=456)
    p.add_argument('--train_rng_seed', type=int, default=789)
    p.add_argument('--log_every', type=int, default=20)
    return p.parse_args()


def build_discriminator(args, vocab_size: int):
    """Build discriminator DeBERTa-v2 for masked LM (same as protected backbone)."""
    from transformers import DebertaV2Config, DebertaV2ForMaskedLM
    config = DebertaV2Config(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.n_layer,
        num_attention_heads=args.n_head,
        intermediate_size=args.hidden_size * args.ffn_mult,
        max_position_embeddings=args.max_position_embeddings,
        type_vocab_size=0,
        hidden_act='gelu',
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        output_hidden_states=True,
    )
    torch.manual_seed(args.seed)
    model = DebertaV2ForMaskedLM(config)
    return model


def main():
    args = parse_args()
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}', flush=True)

    # Tokenizer
    tokenizer = make_portable_tokenizer(args.tokenizer_label)
    vocab_size = len(tokenizer)
    print(f'Vocab size: {vocab_size}', flush=True)

    # Build discriminator
    disc_model = build_discriminator(args, vocab_size)
    disc_model.to(device)

    # Get discriminator's word embeddings for GDES
    disc_word_emb = disc_model.deberta.embeddings.word_embeddings

    # Build generator with GDES
    torch.manual_seed(args.extra_init_seed)
    generator = RTDGenerator(
        vocab_size=vocab_size,
        hidden=args.gen_hidden,
        n_layers=args.gen_layers,
        n_heads=args.gen_heads,
        max_position=args.max_position_embeddings,
        disc_embeddings=disc_word_emb,
    )
    generator.to(device)

    # RTD head
    rtd_head = RTDHead(args.hidden_size)
    rtd_head.to(device)

    # Parameter counts
    disc_params = sum(p.numel() for p in disc_model.parameters())
    gen_params = sum(p.numel() for p in generator.parameters())
    rtd_params = sum(p.numel() for p in rtd_head.parameters())
    print(f'Discriminator params: {disc_params:,}', flush=True)
    print(f'Generator params: {gen_params:,}', flush=True)
    print(f'RTD head params: {rtd_params:,}', flush=True)
    print(f'Total trainable: {disc_params + gen_params + rtd_params:,}', flush=True)

    # Dataset
    dataset = MaskedChunkDataset(
        jsonl_path=args.example_jsonl,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        mask_prob=args.mask_prob,
        mask_mode=args.mask_mode,
        max_examples=None,
        seed=args.train_rng_seed,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # Optimizer: joint over all parameters
    all_params = list(disc_model.parameters()) + list(generator.parameters()) + list(rtd_head.parameters())
    total_steps = len(loader)
    warmup_steps = max(1, int(args.warmup_fraction * total_steps))

    optimizer = torch.optim.AdamW(all_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    # Linear warmup + linear decay schedule
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.0, 1.0 - (step - warmup_steps) / (total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    print(f'Total steps: {total_steps}, warmup: {warmup_steps}', flush=True)
    print(f'Lambda RTD: {args.lambda_rtd}', flush=True)

    log_path = out / 'training_log.jsonl'
    cumulative_words = 0
    saved_checkpoints = []
    next_ckpt = args.checkpoint_words if args.checkpoint_words > 0 else None
    loss_values = []

    disc_model.train()
    generator.train()
    rtd_head.train()

    with log_path.open('w', encoding='utf-8') as logf:
        for step, batch in enumerate(loader, 1):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)  # -100 for non-masked, original token for masked

            # Identify masked positions
            masked_positions = (labels != -100)
            original_ids = input_ids.clone()
            original_ids[masked_positions] = labels[masked_positions]

            # research: Generator forward on masked input
            gen_logits = generator(input_ids, attention_mask)

            # Generator MLM loss (on masked positions only)
            gen_mlm_targets = labels.clone()
            gen_loss = F.cross_entropy(
                gen_logits.view(-1, vocab_size),
                gen_mlm_targets.view(-1),
                ignore_index=-100,
            )

            # research: Create RTD inputs using generator samples
            rtd_input_ids, rtd_labels = create_rtd_inputs(
                input_ids, masked_positions, gen_logits, original_ids
            )

            # research: Discriminator forward with hybrid loss
            disc_loss, metrics = compute_hybrid_loss(
                disc_model, rtd_head, rtd_input_ids, attention_mask,
                original_ids, masked_positions, rtd_labels, args.lambda_rtd,
            )

            # Total loss: generator + discriminator
            total_loss = gen_loss + disc_loss

            # Backward and step
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()
            scheduler.step()

            # Track word exposure
            batch_words = batch.get('word_count', torch.tensor(0))
            if isinstance(batch_words, torch.Tensor):
                cumulative_words += batch_words.sum().item()
            else:
                cumulative_words += sum(batch_words)

            loss_values.append(metrics['total_loss'])

            if step % args.log_every == 0 or step == 1:
                rec = {
                    'step': step, 'cum_words': int(cumulative_words),
                    'gen_loss': round(gen_loss.item(), 4),
                    'disc_mlm_loss': round(metrics['mlm_loss'], 4),
                    'rtd_loss': round(metrics['rtd_loss'], 4),
                    'total_loss': round(metrics['total_loss'], 4),
                    'rtd_acc': round(metrics['rtd_accuracy'], 4),
                    'lr': round(scheduler.get_last_lr()[0], 6),
                }
                print(json.dumps(rec), flush=True)
                logf.write(json.dumps(rec) + '\n')
                logf.flush()

            # Checkpointing
            while next_ckpt is not None and cumulative_words >= next_ckpt and next_ckpt <= args.max_word_exposure:
                if next_ckpt < 1_000_000:
                    name = "chck_1M"
                elif next_ckpt % 1_000_000 == 0:
                    name = f"chck_{next_ckpt // 1_000_000}M"
                else:
                    name = f"chck_{next_ckpt}w"
                cp = out / "hf_model" / name
                save_hf_checkpoint(disc_model, tokenizer, cp)
                saved_checkpoints.append({
                    "name": name, "target_word_exposure": next_ckpt,
                    "actual_cumulative_word_exposure": int(cumulative_words),
                    "path": str(cp)
                })
                print(json.dumps({"event": "checkpoint_saved", "name": name, "cum_words": int(cumulative_words)}), flush=True)
                next_ckpt += args.checkpoint_words

    # Final save
    save_hf_checkpoint(disc_model, tokenizer, out / "hf_model")
    if not saved_checkpoints:
        cp = out / "hf_model" / "chck_1M"
        save_hf_checkpoint(disc_model, tokenizer, cp)
        saved_checkpoints.append({"name": "chck_1M", "target_word_exposure": args.checkpoint_words,
                                  "actual_cumulative_word_exposure": int(cumulative_words), "path": str(cp)})

    # Save metrics
    metrics_out = {
        "experiment": "hybrid_rtd_mlm",
        "parameter_count_discriminator": disc_params,
        "parameter_count_generator": gen_params,
        "parameter_count_rtd_head": rtd_params,
        "word_exposure": int(cumulative_words),
        "actual_training_steps": step,
        "loss_first": loss_values[0] if loss_values else None,
        "loss_last": loss_values[-1] if loss_values else None,
        "lambda_rtd": args.lambda_rtd,
        "gen_hidden": args.gen_hidden,
        "gen_layers": args.gen_layers,
        "mask_mode": args.mask_mode,
        "mask_prob": args.mask_prob,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "extra_init_seed": args.extra_init_seed,
        "train_rng_seed": args.train_rng_seed,
        "saved_checkpoints": saved_checkpoints,
    }
    (out / "scientific_metrics.json").write_text(
        json.dumps(metrics_out, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({"event": "done", "disc_params": disc_params, "gen_params": gen_params,
                      "loss_first": metrics_out["loss_first"], "loss_last": metrics_out["loss_last"],
                      "word_exposure": int(cumulative_words), "checkpoints": [c["name"] for c in saved_checkpoints]}), flush=True)


if __name__ == '__main__':
    main()
