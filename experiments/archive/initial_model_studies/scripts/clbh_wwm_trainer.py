#!/usr/bin/env python3
"""research Contextual Lexical Binding Head (CLBH) trainer.

Extends protected DeBERTa-v2 WWM with an output-side pointer/copy path that mixes
standard MLM logits with copy evidence from visible context token identities.

Mechanism: for each position, compute bilinear attention over visible (non-MASK,
non-pad) context positions. Scatter-add raw attention scores to vocabulary entries
at those context positions. Gate and add to standard MLM logits.

Modes:
  clbh:       active pointer with correct token identities
  shuffled:   same pointer computation but vocab targets are fixed-permuted (control)
  adapter:    parameter-matched bottleneck adapter on hidden states, no pointer

All three train with standard WWM; the only difference is the output head computation.
"""
from __future__ import annotations
import argparse, json, math, os, pathlib, random, sys, time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
sys.path.insert(0, str((ROOT/'training/scripts').resolve()))
from babylm_masked_train import (
    TRAIN_FILES, Example, apply_masking, build_model, collate, download_dataset,
    iter_examples, make_portable_tokenizer, MaskedChunkDataset, reset_all_rng,
    save_hf_checkpoint, summarize_tokenization_coupling,
)


class CLBHModule(nn.Module):
    """Contextual Lexical Binding Head: pointer path over context tokens."""

    def __init__(self, hidden_size: int, ptr_dim: int = 64, vocab_size: int = 16384):
        super().__init__()
        self.ptr_dim = ptr_dim
        self.query_proj = nn.Linear(hidden_size, ptr_dim, bias=False)
        self.key_proj = nn.Linear(hidden_size, ptr_dim, bias=False)
        self.gate_proj = nn.Linear(hidden_size, 1, bias=True)
        self.vocab_size = vocab_size
        # For shuffled control: fixed random permutation of vocab
        perm = torch.randperm(vocab_size)
        self.register_buffer('shuffle_perm', perm)

    def forward(self, hidden_states, input_ids, attention_mask, mask_token_id, mode='clbh'):
        """
        Args:
            hidden_states: (B, L, H) encoder output
            input_ids: (B, L) token IDs in the input (with [MASK] at masked positions)
            attention_mask: (B, L) 1=real token, 0=pad
            mask_token_id: int, the [MASK] token ID
            mode: 'clbh' | 'shuffled'
        Returns:
            copy_logits: (B, L, V) additive logit contribution
            gate: (B, L, 1) gate values (sigmoid)
            diagnostics: dict with pointer statistics
        """
        B, L, H = hidden_states.shape
        # Queries and keys
        Q = self.query_proj(hidden_states)  # (B, L, ptr_dim)
        K = self.key_proj(hidden_states)    # (B, L, ptr_dim)

        # Raw attention scores
        raw_attn = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.ptr_dim)  # (B, L, L)

        # Visibility mask: context positions are non-MASK, non-pad, and not self
        is_visible = (input_ids != mask_token_id) & (attention_mask == 1)  # (B, L)
        # Expand for broadcasting: (B, 1, L) for key dimension
        vis_mask = is_visible.unsqueeze(1).expand(B, L, L).float()  # (B, L, L)
        # Self-mask: don't attend to self
        self_mask = torch.eye(L, device=hidden_states.device).unsqueeze(0).expand(B, L, L)
        vis_mask = vis_mask * (1.0 - self_mask)

        # Mask out invisible positions with large negative
        raw_attn = raw_attn.masked_fill(vis_mask == 0, -1e9)

        # Gate per position
        gate = torch.sigmoid(self.gate_proj(hidden_states))  # (B, L, 1)

        # Determine vocab IDs for scatter
        if mode == 'shuffled':
            scatter_ids = self.shuffle_perm[input_ids.clamp(0, self.vocab_size - 1)]  # (B, L)
        else:
            scatter_ids = input_ids  # (B, L)

        # Use ReLU'd raw scores (NOT softmax) to avoid probability dilution.
        # Only positions with positive query-key similarity get a boost;
        # tokens not in context get exactly zero additive contribution.
        positive_scores = F.relu(raw_attn) * vis_mask  # (B, L, L) — zero for invisible/negative

        # Scatter-add positive scores to vocab entries of context tokens
        # For each position i: copy_logits[i, v] = sum(positive_scores[i, j]) for j where scatter_ids[j]==v
        copy_logits = torch.zeros(B, L, self.vocab_size, device=hidden_states.device)
        scatter_expanded = scatter_ids.unsqueeze(1).expand(B, L, L)  # (B, L, L)
        copy_logits.scatter_add_(2, scatter_expanded, positive_scores)

        # No COPY_SCALE needed: raw Q·K/sqrt(d) magnitudes are naturally in logit-comparable range
        # Gate modulates the overall contribution

        # Diagnostics
        n_visible = is_visible.float().sum(dim=1).mean().item()
        gate_mean = gate.mean().item()
        max_copy = copy_logits.max().item()

        return copy_logits, gate, {
            'avg_visible_context': n_visible,
            'gate_mean': gate_mean,
            'max_copy_weight': max_copy,
        }


class AdapterModule(nn.Module):
    """Parameter-matched bottleneck adapter (no pointer, capacity control)."""

    def __init__(self, hidden_size: int, bottleneck: int = 64):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck, bias=False)
        self.up = nn.Linear(bottleneck, hidden_size, bias=False)
        self.gate_proj = nn.Linear(hidden_size, 1, bias=True)

    def forward(self, hidden_states):
        """Returns residual addition to hidden states + gate."""
        residual = self.up(F.gelu(self.down(hidden_states)))
        gate = torch.sigmoid(self.gate_proj(hidden_states))
        return residual * gate, gate


def build_args():
    p = argparse.ArgumentParser(description='CLBH WWM trainer')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--mode', choices=['clbh', 'shuffled', 'adapter', 'baseline'], default='clbh')
    p.add_argument('--ptr_dim', type=int, default=64)
    p.add_argument('--max_word_exposure', type=int, default=1_000_000)
    p.add_argument('--example_pool_words', type=int, default=1_000_000)
    p.add_argument('--checkpoint_words', type=int, default=1_000_000)
    p.add_argument('--words_per_example', type=int, default=160)
    p.add_argument('--tokenizer_label', default='baseline16k')
    p.add_argument('--tokenizer_path', default='')
    p.add_argument('--tokenization_summary_limit', type=int, default=0)
    p.add_argument('--mask_mode', choices=['token', 'wwm'], default='wwm')
    p.add_argument('--mask_prob', type=float, default=0.15)
    p.add_argument('--seq_length', type=int, default=256)
    p.add_argument('--max_seq_length', type=int, default=256)
    p.add_argument('--max_position_embeddings', type=int, default=512)
    p.add_argument('--model_type', default='deberta_v2')
    p.add_argument('--position_buckets', type=int, default=256)
    p.add_argument('--max_relative_positions', type=int, default=256)
    p.add_argument('--deberta_pos_att_type', default='p2c,c2p')
    p.add_argument('--deberta_relative_attention', default='true')
    p.add_argument('--hidden_size', type=int, default=480)
    p.add_argument('--n_layer', type=int, default=8)
    p.add_argument('--n_head', type=int, default=8)
    p.add_argument('--ffn_mult', type=int, default=4)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--learning_rate', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--warmup_fraction', type=float, default=0.05)
    p.add_argument('--lr_total_steps', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--extra_init_seed', type=int, default=456)
    p.add_argument('--train_rng_seed', type=int, default=789)
    p.add_argument('--log_every', type=int, default=5)
    p.add_argument('--dataset_id', default='BabyLM-community/BabyLM-2026-Strict-Small')
    p.add_argument('--dataset_revision', default='c92ab16b4f08858304b0815706065b3354d8fc0a')
    return p.parse_args()


def structural_verification(model, clbh_module, tokenizer, device, mode):
    """Verify that CLBH changes logits when gold token appears in context."""
    model.eval()
    if clbh_module:
        clbh_module.eval()

    # Construct a simple test: "The cat sat on the [MASK]" where answer is "mat"
    # and "mat" appears in visible context in one version but not another
    mask_id = tokenizer.mask_token_id
    text_with = "The cat sat on the mat . The dog found the [MASK] nearby ."
    text_without = "The cat sat on the rug . The dog found the [MASK] nearby ."

    # Tokenize (manually insert mask)
    def encode_with_mask(text):
        parts = text.split('[MASK]')
        ids = []
        for i, part in enumerate(parts):
            ids.extend(tokenizer(part.strip(), add_special_tokens=False)['input_ids'])
            if i < len(parts) - 1:
                ids.append(mask_id)
        return ids

    ids_with = encode_with_mask(text_with)
    ids_without = encode_with_mask(text_without)
    L = max(len(ids_with), len(ids_without))
    pad = tokenizer.pad_token_id

    def pad_to(seq, length):
        return seq + [pad] * (length - len(seq))

    inp = torch.tensor([pad_to(ids_with, L), pad_to(ids_without, L)], device=device)
    attn = torch.tensor([[1]*len(ids_with) + [0]*(L-len(ids_with)),
                         [1]*len(ids_without) + [0]*(L-len(ids_without))], device=device)

    # Find mask positions
    mask_pos_0 = (inp[0] == mask_id).nonzero(as_tuple=True)[0]
    mask_pos_1 = (inp[1] == mask_id).nonzero(as_tuple=True)[0]

    with torch.no_grad():
        outputs = model.deberta(input_ids=inp, attention_mask=attn)
        hidden = outputs.last_hidden_state
        vocab_logits = model.cls(hidden)  # standard MLM head

        if clbh_module:
            copy_logits, gate, diag = clbh_module(hidden, inp, attn, mask_id, mode=mode)
            combined = vocab_logits + gate * copy_logits
        else:
            combined = vocab_logits
            diag = {}

    # Find "mat" token ID
    mat_id = tokenizer(' mat', add_special_tokens=False)['input_ids']
    if not mat_id:
        mat_id = tokenizer('mat', add_special_tokens=False)['input_ids']

    results = {}
    if mat_id and mask_pos_0.numel() > 0 and mask_pos_1.numel() > 0:
        tid = mat_id[0]
        mp0 = mask_pos_0[0].item()
        mp1 = mask_pos_1[0].item()

        # Log-prob of "mat" at mask position in both cases
        lp_with = F.log_softmax(combined[0, mp0], dim=-1)[tid].item()
        lp_without = F.log_softmax(combined[1, mp1], dim=-1)[tid].item()
        lp_base_with = F.log_softmax(vocab_logits[0, mp0], dim=-1)[tid].item()
        lp_base_without = F.log_softmax(vocab_logits[1, mp1], dim=-1)[tid].item()

        results = {
            'mat_token_id': tid,
            'lp_mat_combined_with_context': lp_with,
            'lp_mat_combined_without_context': lp_without,
            'lp_mat_base_with': lp_base_with,
            'lp_mat_base_without': lp_base_without,
            'pointer_boost_with_context': lp_with - lp_base_with,
            'pointer_boost_without_context': lp_without - lp_base_without,
            'context_effect': lp_with - lp_without,
        }
    results['diagnostics'] = diag
    model.train()
    if clbh_module:
        clbh_module.train()
    return results


def main():
    args = build_args()
    start = time.time()
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Setup env
    hf = ROOT/'training/hf_home'
    os.environ['HF_HOME'] = str(hf.resolve())
    os.environ['HF_HUB_CACHE'] = str((hf/'hub').resolve())
    os.environ['HF_DATASETS_CACHE'] = str((hf/'datasets').resolve())
    os.environ['TRANSFORMERS_CACHE'] = str((hf/'transformers').resolve())
    os.environ['HF_MODULES_CACHE'] = str((ROOT/'training/hf_modules_cache').resolve())
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

    tokenizer = make_portable_tokenizer(args.tokenizer_path)
    # Data
    raw_dir, _ = download_dataset(args, out)
    files = [raw_dir / n for n in TRAIN_FILES]
    pool = list(iter_examples(files, args.example_pool_words, args.words_per_example))
    for i, ex in enumerate(pool):
        ex.example_id = i
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    examples = []
    actual = 0
    for ex in pool:
        if actual >= args.max_word_exposure:
            break
        if actual + ex.words <= args.max_word_exposure:
            examples.append(ex)
            actual += ex.words
        else:
            take = args.max_word_exposure - actual
            examples.append(Example(' '.join(ex.text.split()[:take]), take, ex.example_id, ex.source))
            actual += take
    if actual != args.max_word_exposure:
        raise RuntimeError(f'word mismatch {actual}')

    (out / 'example_order_manifest.json').write_text(json.dumps({
        'seed': args.seed, 'selected_words': actual,
        'num_examples': len(examples),
        'source_words': {s: sum(e.words for e in examples if e.source == s) for s in set(e.source for e in examples)},
    }, indent=2), encoding='utf-8')

    ds = MaskedChunkDataset(examples, tokenizer, args.max_seq_length)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
                        num_workers=2, pin_memory=torch.cuda.is_available())

    # Model
    if args.extra_init_seed >= 0:
        reset_all_rng(args.extra_init_seed)
    model = build_model(args, tokenizer)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # CLBH or adapter module
    clbh_module = None
    adapter_module = None
    if args.mode in ('clbh', 'shuffled'):
        clbh_module = CLBHModule(args.hidden_size, args.ptr_dim, len(tokenizer)).to(device)
    elif args.mode == 'adapter':
        adapter_module = AdapterModule(args.hidden_size, args.ptr_dim).to(device)

    if args.train_rng_seed >= 0:
        reset_all_rng(args.train_rng_seed)

    # Structural verification before training
    verify = structural_verification(model, clbh_module, tokenizer, device, args.mode)
    print(json.dumps({'event': 'structural_verification', 'mode': args.mode, **verify}), flush=True)

    # Optimizer
    all_params = list(model.parameters())
    if clbh_module:
        all_params += list(clbh_module.parameters())
    if adapter_module:
        all_params += list(adapter_module.parameters())

    opt = torch.optim.AdamW(all_params, lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    total_steps = len(loader)
    sched_total = args.lr_total_steps if args.lr_total_steps > 0 else total_steps
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(sched_total * args.warmup_fraction)), sched_total)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.train_rng_seed if args.train_rng_seed >= 0 else args.seed)

    base_params = sum(p.numel() for p in model.parameters())
    extra_params = sum(p.numel() for p in (clbh_module.parameters() if clbh_module else adapter_module.parameters() if adapter_module else []))
    total_params = base_params + extra_params
    print(f'Model params: {base_params:,} + {extra_params:,} ({args.mode}) = {total_params:,}', flush=True)

    model.train()
    if clbh_module:
        clbh_module.train()
    if adapter_module:
        adapter_module.train()

    cum = 0
    logs = []
    logf = (out / 'training_log.jsonl').open('w', encoding='utf-8')
    next_ckpt = args.checkpoint_words if args.checkpoint_words > 0 else None
    mask_id = tokenizer.mask_token_id

    for step, batch in enumerate(loader, 1):
        words = int(batch.pop('words').sum().item())
        input_ids = batch['input_ids'].to(device)
        attn = batch['attention_mask'].to(device)
        wg = batch['word_group'].to(device)

        # Truncate to seq_length
        cur_len = min(args.seq_length, args.max_seq_length)
        input_ids = input_ids[:, :cur_len].contiguous()
        attn = attn[:, :cur_len].contiguous()
        wg = wg[:, :cur_len].contiguous()

        # WWM masking
        masked_inputs, labels = apply_masking(input_ids, attn, wg, tokenizer, args.mask_mode, args.mask_prob, gen)

        # Forward through encoder
        opt.zero_grad(set_to_none=True)
        encoder_out = model.deberta(input_ids=masked_inputs, attention_mask=attn)
        hidden = encoder_out.last_hidden_state

        # Standard MLM logits
        if adapter_module:
            adapter_residual, adapter_gate = adapter_module(hidden)
            vocab_logits = model.cls(hidden + adapter_residual)
        else:
            vocab_logits = model.cls(hidden)

        # CLBH pointer contribution
        if clbh_module:
            copy_logits, gate, _ = clbh_module(hidden, masked_inputs, attn, mask_id, mode=args.mode)
            combined_logits = vocab_logits + gate * copy_logits
        else:
            combined_logits = vocab_logits

        # Loss on masked positions only
        loss = F.cross_entropy(combined_logits.view(-1, len(tokenizer)), labels.view(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        opt.step()
        sched.step()

        cum += words
        n_masked = int((labels != -100).sum().item())

        # Count how many masked targets have gold token in visible context
        gold_in_context = 0
        if step <= 5 or step % args.log_every == 0:
            mask_positions = (labels != -100)
            for b in range(min(4, masked_inputs.size(0))):
                visible_ids = set(masked_inputs[b][attn[b] == 1].tolist()) - {mask_id, tokenizer.pad_token_id}
                gold_ids = labels[b][mask_positions[b]].tolist()
                gold_in_context += sum(1 for g in gold_ids if g in visible_ids)

        rec = {
            'step': step, 'cum_words': cum, 'loss': float(loss.detach().cpu()),
            'n_masked': n_masked, 'lr': float(sched.get_last_lr()[0]),
            'elapsed': time.time() - start,
        }
        if clbh_module and (step <= 5 or step % args.log_every == 0):
            rec['gate_mean'] = float(gate.mean().detach().cpu())
            rec['gold_in_context_sample'] = gold_in_context
        if adapter_module and (step <= 5 or step % args.log_every == 0):
            rec['adapter_gate_mean'] = float(adapter_gate.mean().detach().cpu())
        logs.append(rec)
        logf.write(json.dumps(rec) + '\n')
        logf.flush()
        if step == 1 or step % args.log_every == 0 or step == total_steps:
            print(json.dumps({'event': 'train', **rec}), flush=True)

        while next_ckpt is not None and cum >= next_ckpt and next_ckpt <= args.max_word_exposure:
            name = f"chck_{next_ckpt // 1_000_000}M" if next_ckpt >= 1_000_000 and next_ckpt % 1_000_000 == 0 else f"chck_{next_ckpt}w"
            cp = out / 'hf_model' / name
            save_hf_checkpoint(model, tokenizer, cp)
            # Also save CLBH/adapter weights separately
            if clbh_module:
                torch.save(clbh_module.state_dict(), cp / 'clbh_module.pt')
            if adapter_module:
                torch.save(adapter_module.state_dict(), cp / 'adapter_module.pt')
            print(json.dumps({'event': 'checkpoint', 'name': name, 'cum': cum}), flush=True)
            next_ckpt += args.checkpoint_words

    logf.close()
    save_hf_checkpoint(model, tokenizer, out / 'hf_model')
    if clbh_module:
        torch.save(clbh_module.state_dict(), out / 'hf_model' / 'clbh_module.pt')
    if adapter_module:
        torch.save(adapter_module.state_dict(), out / 'hf_model' / 'adapter_module.pt')

    # Post-training verification
    verify_post = structural_verification(model, clbh_module, tokenizer, device, args.mode)
    print(json.dumps({'event': 'post_training_verification', **verify_post}), flush=True)

    metrics = {
        'mode': args.mode,
        'base_params': base_params,
        'extra_params': extra_params,
        'total_params': total_params,
        'ptr_dim': args.ptr_dim,
        'word_exposure': cum,
        'total_steps': total_steps,
        'loss_first': logs[0]['loss'] if logs else None,
        'loss_last': logs[-1]['loss'] if logs else None,
        'verification_pre': verify,
        'verification_post': verify_post,
    }
    (out / 'scientific_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'event': 'done', 'mode': args.mode, 'total_params': total_params,
                      'loss_last': metrics['loss_last']}), flush=True)


if __name__ == '__main__':
    main()
