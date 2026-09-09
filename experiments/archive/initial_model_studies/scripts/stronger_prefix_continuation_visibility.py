#!/usr/bin/env python3
"""research stronger visibility test for strict prefix-continuation objective.

This is a structural verification before any 1M run. It tests multiple examples and
multiple suffix predictor positions t where hidden[t] predicts token[t+1].
For each tested t:
  - Mutating the predicted token x[t+1] must NOT change logits at position t.
  - Mutating all future suffix tokens > t must NOT change logits at position t.
  - Mutating an earlier suffix token and removing prefix SHOULD change logits at t.

It uses the exact research causal_forward and build_prefix_lm_mask implementations.
"""
from __future__ import annotations
import json, os, pathlib, random, sys, time

import torch
from torch.utils.data import DataLoader

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
sys.path.insert(0, str((ROOT/'scripts').resolve()))
sys.path.insert(0, str((ROOT/'training/scripts').resolve()))

from prefix_continuation_trainer import build_prefix_lm_mask, causal_forward  # noqa: E402
from babylm_masked_train import (  # noqa: E402
    TRAIN_FILES, iter_examples, Example, MaskedChunkDataset, collate, make_portable_tokenizer,
    build_model, reset_all_rng
)

OUT = ROOT/'data/prefix_continuation_visibility.json'


def setup_env():
    hf = ROOT/'training/hf_home'
    os.environ['HF_HOME'] = str(hf.resolve())
    os.environ['HF_HUB_CACHE'] = str((hf/'hub').resolve())
    os.environ['HF_DATASETS_CACHE'] = str((hf/'datasets').resolve())
    os.environ['TRANSFORMERS_CACHE'] = str((hf/'transformers').resolve())
    os.environ['HF_MODULES_CACHE'] = str((ROOT/'training/hf_modules_cache').resolve())
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


class Args:
    model_type='deberta_v2'; hidden_size=480; n_layer=8; n_head=8; ffn_mult=4
    max_position_embeddings=512; max_seq_length=256
    position_buckets=256; max_relative_positions=256
    deberta_relative_attention='true'; deberta_pos_att_type='p2c,c2p'


def reconstruct_examples(n_words=20000, words_per_example=160, seed=42):
    raw = ROOT/'training/runs/prefix_cont_smoke_20k/raw_dataset'
    if not raw.exists():
        raw = ROOT/'training/runs/wwm_debertav2_8x480_official_1M_b128_seed42_matched/raw_dataset'
    files = [raw/n for n in TRAIN_FILES]
    pool = list(iter_examples(files, n_words, words_per_example))
    for i, e in enumerate(pool):
        e.example_id = i
    rng = random.Random(seed); rng.shuffle(pool)
    out=[]; actual=0
    for ex in pool:
        if actual >= n_words: break
        if actual + ex.words <= n_words:
            out.append(ex); actual += ex.words
        else:
            take = n_words - actual
            out.append(Example(' '.join(ex.text.split()[:take]), take, ex.example_id, ex.source)); actual += take
    if actual != n_words:
        raise RuntimeError(f'word mismatch {actual}')
    return out


def mutate_token(x, vocab_size, offset):
    return (x + offset) % vocab_size


def main():
    setup_env(); t0=time.time()
    tokenizer = make_portable_tokenizer('')
    reset_all_rng(456)
    model = build_model(Args(), tokenizer)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    examples = reconstruct_examples()
    ds = MaskedChunkDataset(examples, tokenizer, 256)
    loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate, num_workers=0)
    batch = next(iter(loader))
    input_ids = batch['input_ids'].to(device)
    attn_2d = batch['attention_mask'].to(device)
    active_lens = attn_2d.sum(dim=1).int()
    L = input_ids.size(1)
    P = int(active_lens.min().item() * 0.5)
    P = max(10, min(P, L-20))
    causal_mask = build_prefix_lm_mask(attn_2d, P)
    with torch.no_grad():
        logits_orig, _ = causal_forward(model, input_ids, attn_2d, causal_mask)

    # Choose multiple predictor positions t in suffix. hidden[t] predicts token[t+1].
    max_active = int(active_lens.min().item())
    candidate_positions = [P+1, P+5, P+10, P+20, max(P+2, min(max_active-3, P+35))]
    candidate_positions = sorted(set([t for t in candidate_positions if P <= t < max_active-2]))
    rows=[]
    for t in candidate_positions:
        # A. Mutate exactly predicted token x[t+1]
        pred_mut = input_ids.clone()
        pred_mut[:, t+1] = mutate_token(pred_mut[:, t+1], len(tokenizer), 101)
        with torch.no_grad():
            logits_pred_mut, _ = causal_forward(model, pred_mut, attn_2d, causal_mask)
        pred_token_future_diff = (logits_pred_mut[:, t] - logits_orig[:, t]).abs().max().item()

        # B. Mutate all future suffix tokens > t, including predicted token and later tokens.
        future_mut = input_ids.clone()
        if t+1 < L:
            offsets = torch.arange(1, L-(t+1)+1, device=device).unsqueeze(0)
            future_mut[:, t+1:L] = (future_mut[:, t+1:L] + 97 + offsets) % len(tokenizer)
        with torch.no_grad():
            logits_future_mut, _ = causal_forward(model, future_mut, attn_2d, causal_mask)
        all_future_diff = (logits_future_mut[:, t] - logits_orig[:, t]).abs().max().item()

        # C. Mutate an earlier suffix token visible to t (if exists)
        earlier_mut = input_ids.clone()
        epos = max(P, t-3)
        earlier_mut[:, epos] = mutate_token(earlier_mut[:, epos], len(tokenizer), 103)
        with torch.no_grad():
            logits_earlier_mut, _ = causal_forward(model, earlier_mut, attn_2d, causal_mask)
        earlier_visible_diff = (logits_earlier_mut[:, t] - logits_orig[:, t]).abs().max().item()

        # D. Remove prefix from both tokens and attention visibility
        no_pref = input_ids.clone(); no_pref[:, :P] = tokenizer.pad_token_id
        mask_no_pref = causal_mask.clone(); mask_no_pref[:, :, :P] = 0
        with torch.no_grad():
            logits_no_pref, _ = causal_forward(model, no_pref, attn_2d, mask_no_pref)
        prefix_removal_diff = (logits_no_pref[:, t] - logits_orig[:, t]).abs().max().item()

        rows.append({
            'predictor_position_t': t,
            'predicted_token_position_t_plus_1': t+1,
            'predicted_token_mutation_max_logit_diff_at_t': pred_token_future_diff,
            'all_future_suffix_mutation_max_logit_diff_at_t': all_future_diff,
            'earlier_suffix_mutation_position': epos,
            'earlier_suffix_mutation_max_logit_diff_at_t': earlier_visible_diff,
            'prefix_removal_max_logit_diff_at_t': prefix_removal_diff,
        })

    max_pred = max(r['predicted_token_mutation_max_logit_diff_at_t'] for r in rows)
    max_future = max(r['all_future_suffix_mutation_max_logit_diff_at_t'] for r in rows)
    min_earlier = min(r['earlier_suffix_mutation_max_logit_diff_at_t'] for r in rows)
    min_prefix = min(r['prefix_removal_max_logit_diff_at_t'] for r in rows)
    passed = (max_pred < 1e-5 and max_future < 1e-5 and min_earlier > 1e-3 and min_prefix > 1e-2)
    payload = {
        'status': 'PASS' if passed else 'FAIL',
        'design': 'For multiple suffix predictor positions t, hidden[t] predicts x[t+1]. Mutating x[t+1] or any later token must not change logits at t; mutating earlier suffix or removing prefix should change logits.',
        'prefix_len': P,
        'num_examples': int(input_ids.size(0)),
        'active_len_min': int(active_lens.min().item()),
        'positions_tested': candidate_positions,
        'max_predicted_token_future_diff': max_pred,
        'max_all_future_diff': max_future,
        'min_earlier_visible_diff': min_earlier,
        'min_prefix_removal_diff': min_prefix,
        'rows': rows,
        'elapsed_sec': time.time()-t0,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps({'out': str(OUT), **payload}, indent=2))
    if not passed:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
