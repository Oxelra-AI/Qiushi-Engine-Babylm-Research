#!/usr/bin/env python3
"""research CLBH binding discrimination test on frozen pre-trained encoder.

PURPOSE: Test whether the CLBH pointer mechanism can learn to discriminate
entity-property bindings (not just copy any token present in context).

Design:
1. Load frozen pre-trained DeBERTa-v2 encoder (100M seed42 checkpoint)
2. Add trainable CLBH pointer with softplus activation (gradient-preserving)
3. Initialize Q/K from word embeddings (structural warm-start, NOT evidence of binding)
4. Train ONLY the pointer on 100k words of WWM
5. Binding discrimination test: construct entity-property examples where both
   candidate tokens appear in context, but only one is the correct binding.
   Measure whether the pointer selectively boosts the correct one.

If the pointer cannot discriminate bindings: CLBH is fundamentally a copy
mechanism and does not support this route. Alternatives should change
relationship representation.
"""
from __future__ import annotations
import argparse, json, math, os, pathlib, random, sys, time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer, get_cosine_schedule_with_warmup

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
sys.path.insert(0, str((ROOT/'training/scripts').resolve()))
from babylm_masked_train import (
    TRAIN_FILES, Example, apply_masking, collate, download_dataset,
    iter_examples, MaskedChunkDataset, reset_all_rng,
)

FROZEN_CKPT = ROOT/'training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/hf_model/chck_100M'


class CLBHPointer(nn.Module):
    """CLBH pointer with softplus activation and embedding-based Q/K init."""

    def __init__(self, hidden_size: int, ptr_dim: int, vocab_size: int,
                 word_embeddings: torch.Tensor | None = None):
        super().__init__()
        self.ptr_dim = ptr_dim
        self.vocab_size = vocab_size
        self.query_proj = nn.Linear(hidden_size, ptr_dim, bias=False)
        self.key_proj = nn.Linear(hidden_size, ptr_dim, bias=False)
        self.gate_proj = nn.Linear(hidden_size, 1, bias=True)
        # Initialize gate bias negative so pointer starts weak
        nn.init.constant_(self.gate_proj.bias, -2.0)

        # Warm-start Q/K from word embeddings if provided
        if word_embeddings is not None:
            # word_embeddings: (vocab_size, hidden_size)
            # Use SVD to get ptr_dim principal components
            with torch.no_grad():
                U, S, V = torch.svd_lowrank(word_embeddings.float(), q=ptr_dim)
                # V: (hidden_size, ptr_dim) - projection from hidden to ptr space
                self.query_proj.weight.copy_(V.T)  # (ptr_dim, hidden_size)
                self.key_proj.weight.copy_(V.T)

        # Fixed random permutation for shuffled control
        perm = torch.randperm(vocab_size)
        self.register_buffer('shuffle_perm', perm)

    def forward(self, hidden_states, input_ids, attention_mask, mask_token_id, mode='clbh'):
        B, L, H = hidden_states.shape
        Q = self.query_proj(hidden_states)  # (B, L, ptr_dim)
        K = self.key_proj(hidden_states)    # (B, L, ptr_dim)
        raw_attn = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.ptr_dim)

        # Visibility: non-MASK, non-pad, non-self
        is_visible = (input_ids != mask_token_id) & (attention_mask == 1)
        vis_mask = is_visible.unsqueeze(1).expand(B, L, L).float()
        self_mask = torch.eye(L, device=hidden_states.device).unsqueeze(0)
        vis_mask = vis_mask * (1.0 - self_mask)

        # Softplus activation: smooth, always-positive-gradient, selective
        # Only visible positions contribute
        scores = F.softplus(raw_attn) * vis_mask  # (B, L, L)

        # Gate
        gate = torch.sigmoid(self.gate_proj(hidden_states))  # (B, L, 1)

        # Scatter to vocab
        if mode == 'shuffled':
            scatter_ids = self.shuffle_perm[input_ids.clamp(0, self.vocab_size - 1)]
        else:
            scatter_ids = input_ids

        copy_logits = torch.zeros(B, L, self.vocab_size, device=hidden_states.device)
        scatter_expanded = scatter_ids.unsqueeze(1).expand(B, L, L)
        copy_logits.scatter_add_(2, scatter_expanded, scores)

        diagnostics = {
            'gate_mean': gate.mean().item(),
            'max_copy': copy_logits.max().item(),
            'mean_positive_score': scores[scores > 0].mean().item() if (scores > 0).any() else 0,
            'n_visible_mean': is_visible.float().sum(1).mean().item(),
        }
        return copy_logits, gate, diagnostics


def binding_discrimination_test(model, pointer, tokenizer, device, mode='clbh'):
    """Test whether pointer discriminates entity-property bindings.

    Constructs pairs where both candidate tokens appear in context but belong
    to different entities. Measures whether pointer boosts the correct binding.
    """
    model.eval()
    pointer.eval()
    mask_id = tokenizer.mask_token_id

    # Test cases: (context, query_with_mask, correct_token, distractor_token)
    test_cases = [
        ("Alice found a ball . Bob found a hat .",
         "Bob lost the [MASK] yesterday .", "hat", "ball"),
        ("Alice found a ball . Bob found a hat .",
         "Alice lost the [MASK] yesterday .", "ball", "hat"),
        ("The cat ate fish . The dog ate bones .",
         "The dog wanted more [MASK] today .", "bones", "fish"),
        ("The cat ate fish . The dog ate bones .",
         "The cat wanted more [MASK] today .", "fish", "bones"),
        ("John lives in Paris . Mary lives in London .",
         "John returned to [MASK] last week .", "Paris", "London"),
        ("John lives in Paris . Mary lives in London .",
         "Mary returned to [MASK] last week .", "London", "Paris"),
        ("The red car is fast . The blue car is slow .",
         "The blue car drove [MASK] down the road .", "slow", "fast"),
        ("The red car is fast . The blue car is slow .",
         "The red car drove [MASK] down the road .", "fast", "slow"),
    ]

    results = []
    for context, query, correct, distractor in test_cases:
        # Build input sequence
        full_text = context + " " + query
        parts = full_text.split('[MASK]')
        ids = []
        for i, part in enumerate(parts):
            ids.extend(tokenizer(part.strip(), add_special_tokens=False)['input_ids'])
            if i < len(parts) - 1:
                ids.append(mask_id)

        # Get token IDs for correct and distractor
        correct_ids = tokenizer(' ' + correct, add_special_tokens=False)['input_ids']
        distractor_ids = tokenizer(' ' + distractor, add_special_tokens=False)['input_ids']
        if not correct_ids or not distractor_ids:
            continue
        correct_tid = correct_ids[0]
        distractor_tid = distractor_ids[0]

        # Forward pass
        inp = torch.tensor([ids], device=device)
        attn = torch.ones_like(inp)
        mask_pos = (inp[0] == mask_id).nonzero(as_tuple=True)[0]
        if mask_pos.numel() == 0:
            continue
        mp = mask_pos[0].item()

        with torch.no_grad():
            enc = model.deberta(input_ids=inp, attention_mask=attn)
            hidden = enc.last_hidden_state
            base_logits = model.cls(hidden)
            copy_logits, gate, _ = pointer(hidden, inp, attn, mask_id, mode=mode)
            combined = base_logits + gate * copy_logits

        # Extract logits at mask position
        base_correct = base_logits[0, mp, correct_tid].item()
        base_distractor = base_logits[0, mp, distractor_tid].item()
        copy_correct = copy_logits[0, mp, correct_tid].item()
        copy_distractor = copy_logits[0, mp, distractor_tid].item()
        combined_correct = combined[0, mp, correct_tid].item()
        combined_distractor = combined[0, mp, distractor_tid].item()

        results.append({
            'context': context[:60], 'query': query,
            'correct': correct, 'distractor': distractor,
            'base_margin': base_correct - base_distractor,
            'copy_margin': copy_correct - copy_distractor,
            'combined_margin': combined_correct - combined_distractor,
            'gate_at_mask': gate[0, mp, 0].item(),
            'copy_correct_val': copy_correct,
            'copy_distractor_val': copy_distractor,
        })

    # Aggregate
    if results:
        copy_margins = [r['copy_margin'] for r in results]
        combined_margins = [r['combined_margin'] for r in results]
        base_margins = [r['base_margin'] for r in results]
        binding_discrimination = sum(1 for m in copy_margins if m > 0) / len(copy_margins)
        mean_copy_margin = sum(copy_margins) / len(copy_margins)
        mean_base_margin = sum(base_margins) / len(base_margins)
        mean_combined_margin = sum(combined_margins) / len(combined_margins)
    else:
        binding_discrimination = 0
        mean_copy_margin = 0
        mean_base_margin = 0
        mean_combined_margin = 0

    model.train()
    pointer.train()
    return {
        'n_cases': len(results),
        'binding_discrimination_frac': binding_discrimination,
        'mean_copy_margin_correct_minus_distractor': mean_copy_margin,
        'mean_base_margin': mean_base_margin,
        'mean_combined_margin': mean_combined_margin,
        'cases': results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--mode', choices=['clbh', 'shuffled'], default='clbh')
    ap.add_argument('--ptr_dim', type=int, default=64)
    ap.add_argument('--max_word_exposure', type=int, default=100_000)
    ap.add_argument('--words_per_example', type=int, default=160)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--learning_rate', type=float, default=5e-4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--log_every', type=int, default=20)
    ap.add_argument('--dataset_id', default='BabyLM-community/BabyLM-2026-Strict-Small')
    ap.add_argument('--dataset_revision', default='c92ab16b4f08858304b0815706065b3354d8fc0a')
    args = ap.parse_args()
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()

    # Env
    hf = ROOT/'training/hf_home'
    os.environ['HF_HOME'] = str(hf.resolve())
    os.environ['HF_HUB_CACHE'] = str((hf/'hub').resolve())
    os.environ['HF_DATASETS_CACHE'] = str((hf/'datasets').resolve())
    os.environ['TRANSFORMERS_CACHE'] = str((hf/'transformers').resolve())
    os.environ['HF_MODULES_CACHE'] = str((ROOT/'training/hf_modules_cache').resolve())
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load frozen pre-trained model
    print(json.dumps({'event': 'loading_frozen_encoder', 'ckpt': str(FROZEN_CKPT)}), flush=True)
    model = AutoModelForMaskedLM.from_pretrained(str(FROZEN_CKPT.resolve()), trust_remote_code=True)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(str(FROZEN_CKPT.resolve()), use_fast=True)
    mask_id = tokenizer.mask_token_id

    # Build CLBH pointer with embedding warm-start
    word_emb = model.deberta.embeddings.word_embeddings.weight.detach()
    pointer = CLBHPointer(model.config.hidden_size, args.ptr_dim, len(tokenizer), word_emb).to(device)
    pointer_params = sum(p.numel() for p in pointer.parameters())
    print(f'Pointer params: {pointer_params:,} (trainable)', flush=True)

    # Pre-training binding test (should show structural bias only, NOT learned binding)
    pre_binding = binding_discrimination_test(model, pointer, tokenizer, device, args.mode)
    print(json.dumps({'event': 'pre_training_binding_test', 'mode': args.mode, **{k: v for k, v in pre_binding.items() if k != 'cases'}}), flush=True)

    # Data
    raw_dir, _ = download_dataset(args, out)
    files = [raw_dir / n for n in TRAIN_FILES]
    pool = list(iter_examples(files, args.max_word_exposure, args.words_per_example))
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

    ds = MaskedChunkDataset(examples, tokenizer, 256)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
                        num_workers=2, pin_memory=True)

    # Optimizer (only pointer params)
    opt = torch.optim.AdamW(pointer.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = len(loader)
    sched = get_cosine_schedule_with_warmup(opt, max(1, total_steps // 20), total_steps)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)

    # Training loop
    pointer.train()
    cum = 0
    logs = []
    for step, batch in enumerate(loader, 1):
        words = int(batch.pop('words').sum().item())
        input_ids = batch['input_ids'].to(device)[:, :256]
        attn = batch['attention_mask'].to(device)[:, :256]
        wg = batch['word_group'].to(device)[:, :256]

        masked_inputs, labels = apply_masking(input_ids, attn, wg, tokenizer, 'wwm', 0.15, gen)

        with torch.no_grad():
            enc = model.deberta(input_ids=masked_inputs, attention_mask=attn)
            hidden = enc.last_hidden_state
            base_logits = model.cls(hidden)

        copy_logits, gate, diag = pointer(hidden.detach(), masked_inputs, attn, mask_id, mode=args.mode)
        combined = base_logits.detach() + gate * copy_logits
        loss = F.cross_entropy(combined.view(-1, len(tokenizer)), labels.view(-1), ignore_index=-100)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pointer.parameters(), 1.0)
        opt.step()
        sched.step()

        cum += words
        rec = {'step': step, 'cum': cum, 'loss': float(loss.detach().cpu()),
               'lr': float(sched.get_last_lr()[0]), 'elapsed': time.time() - start}
        if step <= 5 or step % args.log_every == 0 or step == total_steps:
            rec.update(diag)
            print(json.dumps({'event': 'train', **rec}), flush=True)
        logs.append(rec)

    # Post-training binding test (THE DECISIVE RESULT)
    post_binding = binding_discrimination_test(model, pointer, tokenizer, device, args.mode)
    print(json.dumps({'event': 'post_training_binding_test', 'mode': args.mode, **{k: v for k, v in post_binding.items() if k != 'cases'}}), flush=True)

    # Also test shuffled control
    post_shuffled = binding_discrimination_test(model, pointer, tokenizer, device, 'shuffled')
    print(json.dumps({'event': 'post_training_binding_shuffled_control', **{k: v for k, v in post_shuffled.items() if k != 'cases'}}), flush=True)

    # Save results
    payload = {
        'status': 'CLBH_BINDING_DISCRIMINATION',
        'mode': args.mode, 'ptr_dim': args.ptr_dim, 'pointer_params': pointer_params,
        'frozen_encoder': str(FROZEN_CKPT), 'word_exposure': cum, 'total_steps': total_steps,
        'loss_first': logs[0]['loss'] if logs else None,
        'loss_last': logs[-1]['loss'] if logs else None,
        'pre_training_binding': pre_binding,
        'post_training_binding_clbh': post_binding,
        'post_training_binding_shuffled': post_shuffled,
        'interpretation': {
            'binding_passes': post_binding['binding_discrimination_frac'] > 0.6 and post_binding['mean_copy_margin_correct_minus_distractor'] > 0.05,
            'copy_margin_exceeds_shuffled': post_binding['mean_copy_margin_correct_minus_distractor'] > post_shuffled['mean_copy_margin_correct_minus_distractor'] + 0.02,
        },
        'elapsed_sec': time.time() - start,
    }
    (out / 'binding_discrimination_results.json').write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'event': 'done', 'binding_passes': payload['interpretation']['binding_passes'],
                      'copy_margin_exceeds_shuffled': payload['interpretation']['copy_margin_exceeds_shuffled'],
                      'discrimination_frac': post_binding['binding_discrimination_frac'],
                      'mean_copy_margin': post_binding['mean_copy_margin_correct_minus_distractor']}), flush=True)


if __name__ == '__main__':
    main()
