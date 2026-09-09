#!/usr/bin/env python3
"""research: LAMB + Sequence-Length Curriculum Trainer for BabyLM Strict-Small.

Tests the two biggest untested factors from the visible leader's recipe:
1. LAMB optimizer (You et al., 2019) with layer-wise trust ratios
2. Sequence length curriculum (short → long)

Supports both LAMB and AdamW for controlled comparison.
Configurable DeBERTa-v2 architecture (default 12×384 matching the leader).
Fixed whole-word masking at 0.15.

"""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, sys, time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, PreTrainedTokenizerFast,
    DebertaV2Config, DebertaV2ForMaskedLM,
    get_cosine_schedule_with_warmup,
)

# ═══════════════════════════════════════════════════════════════════════════════
# LAMB Optimizer
# ═══════════════════════════════════════════════════════════════════════════════
class LAMB(torch.optim.Optimizer):
    """LAMB (Layer-wise Adaptive Moments) optimizer.

    Reference: You et al., 2019, "Large Batch Optimization for Deep Learning:
    Training BERT in 76 Minutes."

    Key difference from AdamW: per-layer trust ratio phi = ||w|| / ||adam_step||
    that enables stable training at much higher learning rates.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6,
                 weight_decay=0.01, max_trust_ratio=10.0,
                 exclude_from_layer_adapt=None):
        if exclude_from_layer_adapt is None:
            exclude_from_layer_adapt = set()
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        max_trust_ratio=max_trust_ratio,
                        exclude_from_layer_adapt=exclude_from_layer_adapt)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("LAMB does not support sparse gradients")
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                state['step'] += 1
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                bc1 = 1.0 - beta1 ** state['step']
                bc2 = 1.0 - beta2 ** state['step']
                adam_step = (exp_avg / bc1) / ((exp_avg_sq / bc2).sqrt() + group['eps'])
                if group['weight_decay'] != 0:
                    adam_step.add_(p, alpha=group['weight_decay'])
                # Trust ratio (skip for excluded params like bias/LayerNorm)
                if group.get('apply_layer_adapt', True):
                    w_norm = p.norm(2).clamp_min(1e-12)
                    s_norm = adam_step.norm(2).clamp_min(1e-12)
                    trust = min(w_norm.item() / s_norm.item(), group['max_trust_ratio'])
                else:
                    trust = 1.0
                p.add_(adam_step, alpha=-group['lr'] * trust)
        return loss


def build_optimizer(model, opt_name, lr, betas, eps, weight_decay, max_trust_ratio=10.0):
    """Build LAMB or AdamW optimizer with param group separation."""
    # Separate params: apply layer adaptation to everything except bias/LayerNorm
    adapt_params, noadapt_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'bias' in name or 'LayerNorm' in name or 'layernorm' in name:
            noadapt_params.append(p)
        else:
            adapt_params.append(p)
    if opt_name == 'lamb':
        return LAMB([
            {'params': adapt_params, 'apply_layer_adapt': True},
            {'params': noadapt_params, 'apply_layer_adapt': False, 'weight_decay': 0.0},
        ], lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
           max_trust_ratio=max_trust_ratio)
    elif opt_name == 'adamw':
        return torch.optim.AdamW([
            {'params': adapt_params},
            {'params': noadapt_params, 'weight_decay': 0.0},
        ], lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class TextRow:
    text: str
    words: int
    example_id: int
    source: str

def load_jsonl_rows(path: Path, max_words: int = 0) -> tuple[list[TextRow], int]:
    """Load rows from a pre-shuffled JSONL training file."""
    rows, total_w = [], 0
    with path.open('r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            text = str(obj['text'])
            w = len(text.split())
            stored_w = int(obj.get('words', w))
            if stored_w != w:
                raise RuntimeError(f"Row {i}: stored words={stored_w} != actual={w}")
            rows.append(TextRow(text=text, words=w,
                                example_id=int(obj.get('example_id', i)),
                                source=str(obj.get('source', ''))))
            total_w += w
            if max_words > 0 and total_w >= max_words:
                break
    return rows, total_w

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


# ═══════════════════════════════════════════════════════════════════════════════
# Curriculum Chunking Dataset
# ═══════════════════════════════════════════════════════════════════════════════
def is_word_start(tok_str: str) -> bool:
    return tok_str.startswith("Ġ") or tok_str.startswith("▁")

class CurriculumChunkedDataset(Dataset):
    """Tokenizes text rows and splits into fixed-length chunks with word groups.

    At shorter seq_len, each row produces multiple chunks, giving more gradient
    steps per word-pass (the core curriculum hypothesis).
    """
    def __init__(self, rows: list[TextRow], tokenizer, seq_len: int,
                 min_chunk_tokens: int = 8):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.pad_id = tokenizer.pad_token_id
        self.special_ids = set(tokenizer.all_special_ids)
        self._ws_cache: dict[int, bool] = {}
        self.chunks: list[tuple[list[int], int]] = []  # (token_ids, row_words_if_first_chunk)
        self.total_words = 0
        # Pre-tokenize and chunk
        for row in rows:
            enc = tokenizer(row.text, add_special_tokens=False, truncation=False)
            ids = enc['input_ids']
            self.total_words += row.words
            # Split into seq_len chunks
            n = len(ids)
            if n < min_chunk_tokens:
                continue
            for start in range(0, n, seq_len):
                chunk = ids[start:start + seq_len]
                if len(chunk) < min_chunk_tokens:
                    continue
                self.chunks.append((chunk, row.words if start == 0 else 0))

    def _word_start_flag(self, tid: int) -> bool:
        v = self._ws_cache.get(tid)
        if v is None:
            s = self.tokenizer.convert_ids_to_tokens(int(tid))
            v = bool(s is not None and is_word_start(str(s)))
            self._ws_cache[tid] = v
        return v

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        ids_list, row_words = self.chunks[idx]
        sl = self.seq_len
        # Pad to seq_len
        n = len(ids_list)
        if n < sl:
            padded = ids_list + [self.pad_id] * (sl - n)
        else:
            padded = ids_list[:sl]
            n = sl
        input_ids = torch.tensor(padded, dtype=torch.long)
        attention_mask = torch.zeros(sl, dtype=torch.long)
        attention_mask[:n] = 1
        # Build word groups for WWM
        group = torch.full((sl,), -1, dtype=torch.long)
        gid = -1
        for i in range(n):
            tid = int(input_ids[i])
            if tid in self.special_ids or attention_mask[i] == 0:
                continue
            if gid < 0 or self._word_start_flag(tid) or i == 0:
                gid += 1
            group[i] = gid
        return {'input_ids': input_ids, 'attention_mask': attention_mask,
                'word_group': group, 'row_words': row_words}


def collate_fn(batch):
    out = {}
    for k in batch[0]:
        vals = [b[k] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals)
        else:
            out[k] = vals  # row_words stays as list
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Whole-Word Masking
# ═══════════════════════════════════════════════════════════════════════════════
def apply_wwm(input_ids, word_group, attention_mask, mask_token_id, vocab_size,
              mask_prob=0.15):
    """Standard whole-word masking with 80/10/10 BERT replacement.
    Returns (masked_input_ids, labels) where labels=-100 for non-masked positions.
    """
    B, S = input_ids.shape
    labels = input_ids.clone()
    masked_ids = input_ids.clone()
    for b in range(B):
        groups = word_group[b]
        unique_g = groups[groups >= 0].unique()
        n_g = unique_g.numel()
        if n_g == 0:
            labels[b] = -100
            continue
        n_mask = max(1, round(n_g * mask_prob))
        perm = torch.randperm(n_g)[:n_mask]
        mask_group_ids = set(unique_g[perm].tolist())
        mask_pos = torch.tensor([int(groups[i].item()) in mask_group_ids
                                  for i in range(S)], dtype=torch.bool)
        labels[b, ~mask_pos] = -100
        for pos in mask_pos.nonzero(as_tuple=True)[0]:
            r = random.random()
            if r < 0.8:
                masked_ids[b, pos] = mask_token_id
            elif r < 0.9:
                masked_ids[b, pos] = random.randint(5, vocab_size - 1)
    return masked_ids, labels


# ═══════════════════════════════════════════════════════════════════════════════
# Model Creation
# ═══════════════════════════════════════════════════════════════════════════════
def create_model(vocab_size, hidden, layers, heads, ffn, init_seed, device,
                 position_biased_input=True):
    """Create a fresh DeBERTa-v2 for masked LM."""
    config = DebertaV2Config(
        vocab_size=vocab_size,
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        intermediate_size=ffn,
        max_position_embeddings=512,
        type_vocab_size=0,
        position_biased_input=position_biased_input,
        relative_attention=True,
        position_buckets=256,
        norm_rel_ebd="layer_norm",
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )
    torch.manual_seed(init_seed)
    torch.cuda.manual_seed_all(init_seed)
    model = DebertaV2ForMaskedLM(config)
    model = model.to(device)
    return model, config


# ═══════════════════════════════════════════════════════════════════════════════
# Checkpoint Saving
# ═══════════════════════════════════════════════════════════════════════════════
def save_checkpoint(model, tokenizer, out_dir: Path, name: str):
    """Save HF-compatible checkpoint."""
    dst = out_dir / name
    dst.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(dst))
    tokenizer.save_pretrained(str(dst))
    # Force portable tokenizer config
    tc = dst / 'tokenizer_config.json'
    if tc.exists():
        cfg = json.loads(tc.read_text())
        cfg['tokenizer_class'] = 'PreTrainedTokenizerFast'
        tc.write_text(json.dumps(cfg, indent=2) + '\n')
    return str(dst)


# ═══════════════════════════════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════════════════════════════
def train(args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(json.dumps({'event': 'start', 'device': str(device), 'args': vars(args)},
                     default=str), flush=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
    vocab_size = tokenizer.vocab_size
    mask_token_id = tokenizer.mask_token_id
    print(json.dumps({'event': 'tokenizer', 'vocab': vocab_size, 'mask_id': mask_token_id}),
          flush=True)

    # Load data
    t0 = time.time()
    rows, total_words = load_jsonl_rows(Path(args.data_path), max_words=args.max_words)
    print(json.dumps({'event': 'data_loaded', 'rows': len(rows), 'words': total_words,
                      'sec': round(time.time() - t0, 1)}), flush=True)

    # Parse curriculum phases: "64:20M,128:50M,256:100M"
    phases = []
    for spec in args.curriculum.split(','):
        sl_str, budget_str = spec.split(':')
        sl = int(sl_str)
        budget = int(budget_str.replace('M', '000000').replace('K', '000'))
        phases.append((sl, budget))
    print(json.dumps({'event': 'curriculum', 'phases': phases}), flush=True)

    # Split rows into phases by word count
    phase_rows: list[list[TextRow]] = [[] for _ in phases]
    cumulative = 0
    phase_idx = 0
    for row in rows:
        if phase_idx < len(phases) - 1 and cumulative >= phases[phase_idx][1]:
            phase_idx += 1
        phase_rows[phase_idx].append(row)
        cumulative += row.words

    # Create datasets per phase and count total steps
    total_steps = 0
    phase_datasets: list[CurriculumChunkedDataset] = []
    for i, (sl, budget) in enumerate(phases):
        ds = CurriculumChunkedDataset(phase_rows[i], tokenizer, sl)
        phase_datasets.append(ds)
        n_steps = math.ceil(len(ds) / args.batch_size)
        total_steps += n_steps
        print(json.dumps({'event': 'phase_dataset', 'phase': i, 'seq_len': sl,
                          'budget_words': budget, 'rows': len(phase_rows[i]),
                          'chunks': len(ds), 'steps': n_steps,
                          'words_in_phase': ds.total_words}), flush=True)

    # Create model
    model, config = create_model(
        vocab_size=vocab_size, hidden=args.hidden, layers=args.layers,
        heads=args.heads, ffn=args.ffn, init_seed=args.init_seed,
        device=device, position_biased_input=args.position_biased_input)
    n_params = sum(p.numel() for p in model.parameters())
    print(json.dumps({'event': 'model', 'params': n_params,
                      'config': {k: v for k, v in config.to_dict().items()
                                 if not isinstance(v, (dict, list))}}), flush=True)

    # Create optimizer
    optimizer = build_optimizer(model, args.optimizer, args.lr,
                                betas=(args.beta1, args.beta2),
                                eps=args.eps, weight_decay=args.weight_decay)
    # LR scheduler: cosine with warmup across all phases
    warmup_steps = max(1, int(total_steps * args.warmup_frac))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(json.dumps({'event': 'optimizer', 'name': args.optimizer, 'lr': args.lr,
                      'total_steps': total_steps, 'warmup_steps': warmup_steps}),
          flush=True)

    # Output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Training state
    global_step = 0
    words_seen = 0
    next_checkpoint_words = args.checkpoint_interval
    loss_accum = 0.0
    tokens_accum = 0
    training_log = []
    dynamics_traces = []
    use_amp = (args.amp and device.type == 'cuda')
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    # Set seeds
    random.seed(args.data_seed)
    np.random.seed(args.data_seed)
    torch.manual_seed(args.data_seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.data_seed)

    model.train()
    t_train = time.time()

    for phase_i, (sl, budget) in enumerate(phases):
        ds = phase_datasets[phase_i]
        if len(ds) == 0:
            continue
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=2, pin_memory=True,
                            drop_last=False)
        print(json.dumps({'event': 'phase_start', 'phase': phase_i, 'seq_len': sl,
                          'chunks': len(ds), 'batches': len(loader)}), flush=True)

        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            word_group = batch['word_group'].to(device)

            # Apply WWM masking
            masked_ids, labels = apply_wwm(
                input_ids, word_group, attention_mask,
                mask_token_id, vocab_size, args.mask_prob)

            # Forward
            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                outputs = model(input_ids=masked_ids, attention_mask=attention_mask,
                                labels=labels)
                loss = outputs.loss

            # Backward
            loss.backward()

            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()
            global_step += 1

            # Tracking
            n_masked = (labels != -100).sum().item()
            loss_accum += loss.item() * n_masked
            tokens_accum += n_masked

            # Word counting: sum words from first chunks of each row
            if isinstance(batch.get('row_words'), list):
                batch_words = sum(batch['row_words'])
            else:
                batch_words = batch['row_words'].sum().item()
            words_seen += batch_words

            # Log periodically
            if global_step % args.log_interval == 0:
                avg_loss = loss_accum / max(1, tokens_accum)
                lr_now = scheduler.get_last_lr()[0]
                elapsed = time.time() - t_train
                log_entry = {
                    'step': global_step, 'phase': phase_i, 'seq_len': sl,
                    'loss': round(avg_loss, 4), 'lr': lr_now,
                    'words_approx': words_seen,
                    'wps': round(words_seen / max(1, elapsed), 0),
                    'elapsed_sec': round(elapsed, 1),
                }
                print(json.dumps({'event': 'train_step', **log_entry}), flush=True)
                training_log.append(log_entry)
                loss_accum, tokens_accum = 0.0, 0

            # Checkpoint at word boundaries
            if words_seen >= next_checkpoint_words:
                chck_name = f"chck_{next_checkpoint_words // 1_000_000}M"
                chck_path = save_checkpoint(model, tokenizer, out_dir / 'hf_model', chck_name)
                print(json.dumps({'event': 'checkpoint', 'name': chck_name,
                                  'path': chck_path, 'step': global_step,
                                  'words': words_seen}), flush=True)
                dynamics_traces.append({
                    'checkpoint': chck_name, 'step': global_step,
                    'phase': phase_i, 'seq_len': sl, 'words': words_seen,
                    'loss': round(loss_accum / max(1, tokens_accum), 4) if tokens_accum else None,
                    'lr': scheduler.get_last_lr()[0],
                })
                next_checkpoint_words += args.checkpoint_interval

        print(json.dumps({'event': 'phase_end', 'phase': phase_i, 'seq_len': sl,
                          'words_after_phase': words_seen, 'step': global_step}),
              flush=True)

    # Final checkpoint
    final_path = save_checkpoint(model, tokenizer, out_dir / 'hf_model', 'chck_final')

    # Save scientific metrics
    elapsed = time.time() - t_train
    metrics = {
        'variant': f'{args.optimizer}_{args.curriculum}_s{args.init_seed}',
        'model_family': 'DebertaV2ForMaskedLM',
        'parameter_count': n_params,
        'vocab_size': vocab_size,
        'hidden_size': args.hidden,
        'num_layers': args.layers,
        'num_heads': args.heads,
        'ffn_size': args.ffn,
        'optimizer': args.optimizer,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'warmup_frac': args.warmup_frac,
        'curriculum': args.curriculum,
        'mask_prob': args.mask_prob,
        'position_biased_input': args.position_biased_input,
        'batch_size': args.batch_size,
        'total_steps': global_step,
        'word_exposure': words_seen,
        'amp': args.amp,
        'init_seed': args.init_seed,
        'data_seed': args.data_seed,
        'training_seconds': round(elapsed, 1),
        'final_checkpoint': final_path,
        'data_path': args.data_path,
        'tokenizer_path': args.tokenizer_path,
    }
    metrics_path = out_dir / 'scientific_metrics.json'
    metrics_path.write_text(json.dumps(metrics, indent=2) + '\n')

    # Save training log
    log_path = out_dir / 'training_log.jsonl'
    with log_path.open('w') as f:
        for entry in training_log:
            f.write(json.dumps(entry) + '\n')

    # Save dynamics traces
    traces_path = out_dir / 'dynamics_traces.jsonl'
    with traces_path.open('w') as f:
        for entry in dynamics_traces:
            f.write(json.dumps(entry) + '\n')

    print(json.dumps({
        'event': 'training_complete',
        'total_steps': global_step,
        'words': words_seen,
        'elapsed_sec': round(elapsed, 1),
        'final_checkpoint': final_path,
        'metrics': str(metrics_path),
    }), flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description='LAMB + Curriculum Trainer')
    # Data
    parser.add_argument('--data_path', required=True, help='Pre-shuffled JSONL training file')
    parser.add_argument('--tokenizer_path', required=True, help='HF tokenizer directory')
    parser.add_argument('--max_words', type=int, default=0, help='Max words to load (0=all)')
    # Architecture
    parser.add_argument('--hidden', type=int, default=384)
    parser.add_argument('--layers', type=int, default=12)
    parser.add_argument('--heads', type=int, default=12)
    parser.add_argument('--ffn', type=int, default=1280)
    parser.add_argument('--position_biased_input', type=int, default=1,
                        help='1=use absolute position embeddings (our default), 0=relative only')
    # Optimizer
    parser.add_argument('--optimizer', choices=['lamb', 'adamw'], default='lamb')
    parser.add_argument('--lr', type=float, default=0.007)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--eps', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_frac', type=float, default=0.06)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    # Curriculum: "seq_len:word_budget,seq_len:word_budget,..."
    parser.add_argument('--curriculum', default='64:20M,128:50M,256:100M',
                        help='Sequence length curriculum phases')
    # Masking
    parser.add_argument('--mask_prob', type=float, default=0.15)
    # Training
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--amp', type=int, default=1, help='1=use bf16 autocast')
    parser.add_argument('--gpu', type=int, default=0)
    # Seeds
    parser.add_argument('--init_seed', type=int, default=43022)
    parser.add_argument('--data_seed', type=int, default=43)
    # Output
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--checkpoint_interval', type=int, default=10_000_000,
                        help='Save checkpoint every N words')
    parser.add_argument('--log_interval', type=int, default=50)

    args = parser.parse_args()
    args.position_biased_input = bool(args.position_biased_input)
    args.amp = bool(args.amp)
    train(args)


if __name__ == '__main__':
    main()
