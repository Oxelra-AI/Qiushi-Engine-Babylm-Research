#!/usr/bin/env python3
"""research CPC-MLM trainer: counterfactual-prefix-conditioned logit margin.

Mechanism: For each training example split into prefix h and suffix s,
apply ordinary WWM to suffix tokens. Then find the hardest negative prefix
h- from the same batch (highest cosine similarity of prefix hidden states,
same-source preferred). Forward both (h+,s) and (h-,s) through the same
MLM model. Add margin loss:

  L_cpc = mean[ max(0, m - log_p(gold|h+,s) + log_p(gold|h-,s)) ]

over suffix masked targets only.

Total loss: L = L_wwm(true sequence) + lambda_cpc * L_cpc.

The model is the protected DeBERTa-v2 8x480 backbone; no auxiliary head,
no generator, no entity labels, no data filter. Only WWM + the margin
that forces prefix to matter for suffix logits.

Hard-negative selection: within-batch nearest-neighbor by prefix hidden-state
cosine, with same-source preference. This naturally gives same-register,
same-topic negatives without requiring entity annotations.

Matched-intervention constraint: the intervention probe tests same-source-nearby swaps
(not just cross-source), so the route cannot survive by learning topic coherence.
"""
from __future__ import annotations
import argparse, json, math, pathlib, random, sys, time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

sys.path.insert(0, str(pathlib.Path('experiments/archive/initial_model_studies/training/scripts').resolve()))
from babylm_masked_train import (
    TRAIN_FILES, Example, apply_masking, build_model, collate, download_dataset,
    iter_examples, load_examples_jsonl, make_portable_tokenizer, MaskedChunkDataset,
    reset_all_rng, save_hf_checkpoint, seq_length_for_progress, sha256_file,
    summarize_tokenization_coupling,
)

ROOT = pathlib.Path('experiments/archive/initial_model_studies')


def build_args():
    p = argparse.ArgumentParser(description='CPC-MLM trainer')
    p.add_argument('--dataset_id', default='BabyLM-community/BabyLM-2026-Strict-Small')
    p.add_argument('--dataset_revision', default='c92ab16b4f08858304b0815706065b3354d8fc0a')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--max_word_exposure', type=int, default=1_000_000)
    p.add_argument('--example_pool_words', type=int, default=1_000_000)
    p.add_argument('--checkpoint_words', type=int, default=1_000_000)
    p.add_argument('--words_per_example', type=int, default=160)
    p.add_argument('--example_jsonl', default=''); p.add_argument('--example_jsonl_label', default=''); p.add_argument('--example_jsonl_meta', default='')
    p.add_argument('--tokenizer_path', default=''); p.add_argument('--tokenizer_label', default='baseline16k')
    p.add_argument('--tokenization_summary_limit', type=int, default=0)
    p.add_argument('--mask_mode', choices=['token', 'wwm'], default='wwm')
    p.add_argument('--mask_prob', type=float, default=0.15)
    p.add_argument('--seq_length', type=int, default=256); p.add_argument('--max_seq_length', type=int, default=256)
    p.add_argument('--max_position_embeddings', type=int, default=512)
    p.add_argument('--seq_len_schedule', default='')
    p.add_argument('--model_type', choices=['bert', 'deberta_v2'], default='deberta_v2')
    p.add_argument('--position_buckets', type=int, default=256); p.add_argument('--max_relative_positions', type=int, default=256)
    p.add_argument('--deberta_pos_att_type', default='p2c,c2p'); p.add_argument('--deberta_relative_attention', choices=['true', 'false'], default='true')
    p.add_argument('--hidden_size', type=int, default=480); p.add_argument('--n_layer', type=int, default=8)
    p.add_argument('--n_head', type=int, default=8); p.add_argument('--ffn_mult', type=int, default=4)
    # CPC-specific
    p.add_argument('--lambda_cpc', type=float, default=1.0, help='Weight for CPC margin loss')
    p.add_argument('--cpc_margin', type=float, default=0.5, help='Margin m in max(0, m - lp_true + lp_neg)')
    p.add_argument('--prefix_fraction', type=float, default=0.5, help='Fraction of active tokens used as prefix')
    p.add_argument('--same_source_preference', type=float, default=0.8, help='Only used when --negative_source_mode=prefer_same: probability of restricting neg to same source')
    p.add_argument('--negative_source_mode', choices=['prefer_same','same','cross','any'], default='prefer_same', help='Hard-negative source constraint. cross explicitly excludes same-source candidates when possible; any is nearest neighbor excluding self; same forces same source when possible; prefer_same is the research behavior.')
    # Training
    p.add_argument('--batch_size', type=int, default=128); p.add_argument('--learning_rate', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=0.01); p.add_argument('--warmup_fraction', type=float, default=0.05)
    p.add_argument('--lr_total_steps', type=int, default=0)
    p.add_argument('--seed', type=int, default=42); p.add_argument('--extra_init_seed', type=int, default=456)
    p.add_argument('--train_rng_seed', type=int, default=789); p.add_argument('--log_every', type=int, default=5)
    return p.parse_args()


def load_examples_standard(args, out, tokenizer):
    """Exact same data selection as the protected baseline."""
    pool_words = max(args.example_pool_words, args.max_word_exposure)
    selected_words = args.max_word_exposure
    if args.example_jsonl:
        p = pathlib.Path(args.example_jsonl)
        examples, total_words, total_rows, sample_rows = load_examples_jsonl(p, selected_words)
        manifest_files = [{'path': str(p), 'name': p.name, 'bytes': p.stat().st_size,
                           'sha256': sha256_file(p), 'whitespace_words': total_words, 'rows': total_rows}]
        meta = {'data_source_type': 'example_jsonl', 'example_jsonl': str(p),
                'example_jsonl_label': args.example_jsonl_label}
    else:
        raw_dir, manifest_files = download_dataset(args, out)
        files = [raw_dir / n for n in TRAIN_FILES]
        pool = list(iter_examples(files, pool_words, args.words_per_example))
        for i, ex in enumerate(pool):
            ex.example_id = i
        rng = random.Random(args.seed); rng.shuffle(pool)
        examples = []; actual = 0
        for ex in pool:
            if actual >= selected_words:
                break
            if actual + ex.words <= selected_words:
                examples.append(ex); actual += ex.words
            else:
                take = selected_words - actual
                examples.append(Example(' '.join(ex.text.split()[:take]), take, ex.example_id, ex.source))
                actual += take
        meta = {'data_source_type': 'official_corpus'}
    actual_words = sum(ex.words for ex in examples)
    if actual_words != selected_words:
        raise RuntimeError(f'word mismatch {actual_words} vs {selected_words}')
    source_words = {}
    for ex in examples:
        source_words[ex.source] = source_words.get(ex.source, 0) + ex.words
    (out / 'example_order_manifest.json').write_text(json.dumps({
        'seed': args.seed, 'selected_for_training_words': actual_words,
        'num_consumed_examples': len(examples), 'source_words_consumed': source_words, **meta
    }, indent=2), encoding='utf-8')
    (out / 'data_manifest.json').write_text(json.dumps({
        'dataset_id': args.dataset_id, 'dataset_revision': args.dataset_revision,
        'files': manifest_files, 'selected_for_this_run_whitespace_words': actual_words, **meta
    }, indent=2), encoding='utf-8')
    return examples, actual_words, source_words


class CPCDataset(MaskedChunkDataset):
    """Extends MaskedChunkDataset to also return source label for same-source neg selection."""
    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        item['source_id'] = 0 if self.examples[idx].source == 'bnc_spoken.train.txt' else 1
        return item


def collate_cpc(batch):
    base = collate(batch)
    base['source_id'] = torch.tensor([x['source_id'] for x in batch], dtype=torch.long)
    return base


def select_hard_negatives(prefix_repr, source_ids, negative_source_mode='prefer_same', same_source_prob=0.8):
    """For each example, find the most similar OTHER prefix under an explicit source constraint.

    Modes:
      prefer_same: research behavior; restrict to same source with probability same_source_prob.
      same:        force same source when possible.
      cross:       force different source when possible (explicit source/style/topic diagnostic).
      any:         nearest neighbor excluding self; may still be same-source.

    Returns:
        neg_indices: (B,) index of negative for each example
    """
    B = prefix_repr.size(0)
    normed = F.normalize(prefix_repr, dim=-1)
    sim = normed @ normed.T
    sim.fill_diagonal_(-1e9)

    neg_indices = torch.zeros(B, dtype=torch.long, device=prefix_repr.device)
    for i in range(B):
        same_mask = (source_ids == source_ids[i])
        same_mask[i] = False
        cross_mask = (source_ids != source_ids[i])
        if negative_source_mode == 'same' and same_mask.any():
            allowed = same_mask
        elif negative_source_mode == 'cross' and cross_mask.any():
            allowed = cross_mask
        elif negative_source_mode == 'prefer_same' and same_mask.any() and random.random() < same_source_prob:
            allowed = same_mask
        else:
            allowed = torch.ones(B, dtype=torch.bool, device=prefix_repr.device)
            allowed[i] = False
        masked_sim = sim[i].clone()
        masked_sim[~allowed] = -1e9
        neg_indices[i] = masked_sim.argmax()
    return neg_indices


def compute_cpc_loss(model, input_ids, attention_mask, word_group, labels,
                     prefix_split, neg_indices, tokenizer, margin, device):
    """Compute CPC margin loss on suffix masked targets.

    For each example i:
      - true_logits = model(true_prefix_i + suffix_i)[suffix_masked_positions]
      - neg_logits = model(prefix_{neg[i]} + suffix_i)[same positions]
      - margin_loss = mean(max(0, m - lp_true + lp_neg))

    Returns: cpc_loss scalar, metrics dict
    """
    B, L = input_ids.shape
    # Build negative sequences: replace prefix with negative's prefix
    neg_input_ids = input_ids.clone()
    neg_attention_mask = attention_mask.clone()
    for i in range(B):
        j = neg_indices[i].item()
        split_i = prefix_split[i]
        split_j = prefix_split[j]
        # Copy prefix from j into i's sequence
        # Use min of the two prefix lengths to avoid overflow
        copy_len = min(split_i, split_j)
        neg_input_ids[i, :copy_len] = input_ids[j, :copy_len]
        neg_attention_mask[i, :copy_len] = attention_mask[j, :copy_len]
        # If j's prefix is shorter, pad remaining prefix positions
        if split_j < split_i:
            neg_input_ids[i, split_j:split_i] = tokenizer.pad_token_id
            neg_attention_mask[i, split_j:split_i] = 0

    # Forward negative sequences
    with torch.cuda.amp.autocast(enabled=False):
        neg_outputs = model(input_ids=neg_input_ids, attention_mask=neg_attention_mask)
    neg_logits = neg_outputs.logits  # (B, L, V)

    # Compute margin loss only on suffix masked positions
    # labels has -100 for non-masked, gold token for masked
    suffix_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
    for i in range(B):
        suffix_mask[i, prefix_split[i]:] = (labels[i, prefix_split[i]:] != -100)

    if not suffix_mask.any():
        return torch.tensor(0.0, device=device), {'cpc_loss': 0.0, 'margin_active_frac': 0.0, 'mean_lp_diff': 0.0}

    # Get gold token ids at suffix masked positions
    gold_ids = labels[suffix_mask]  # (N,)
    # Get true and negative log-probs at these positions
    # We need the true logits from the main forward (passed separately)
    # Actually we need to call forward on true sequence too for logits...
    # The true logits are from the main forward pass that computes WWM loss
    # We'll pass them in. For now, compute from neg_logits and return positions.

    # Return positions and gold_ids so caller can compute from true_logits
    return suffix_mask, gold_ids, neg_logits


def main():
    args = build_args()
    start = time.time()
    out = pathlib.Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    tokenizer = make_portable_tokenizer(args.tokenizer_path)
    examples, actual_words, source_words = load_examples_standard(args, out, tokenizer)
    tok_sum = summarize_tokenization_coupling(examples, tokenizer, args.max_seq_length, args.tokenization_summary_limit)
    (out / 'tokenization_coupling_summary.json').write_text(json.dumps(tok_sum, indent=2), encoding='utf-8')

    ds = CPCDataset(examples, tokenizer, args.max_seq_length)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_cpc,
                        num_workers=2, pin_memory=torch.cuda.is_available())

    if args.extra_init_seed >= 0:
        reset_all_rng(args.extra_init_seed)
    model = build_model(args, tokenizer)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    if args.train_rng_seed >= 0:
        reset_all_rng(args.train_rng_seed)

    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    total_steps = len(loader)
    sched_total = args.lr_total_steps if args.lr_total_steps > 0 else total_steps
    if sched_total < total_steps:
        raise RuntimeError('lr_total_steps < actual steps')
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(sched_total * args.warmup_fraction)), sched_total)
    mask_gen = torch.Generator(device=device)
    mask_gen.manual_seed(args.train_rng_seed if args.train_rng_seed >= 0 else args.seed)

    next_ckpt = args.checkpoint_words if args.checkpoint_words > 0 else None
    cum = 0; logs = []; saved = []
    seq_schedule = []
    if args.seq_len_schedule:
        for part in args.seq_len_schedule.split(','):
            t, L = part.split(':'); seq_schedule.append((float(t), int(L)))
        seq_schedule.sort()

    model.train()
    logf = (out / 'training_log.jsonl').open('w', encoding='utf-8')
    disc_params = sum(p.numel() for p in model.parameters())
    print(f'Model params: {disc_params:,}', flush=True)

    for step, batch in enumerate(loader, 1):
        words = int(batch.pop('words').sum().item())
        source_ids = batch.pop('source_id').to(device)
        input_ids = batch['input_ids'].to(device)
        attn = batch['attention_mask'].to(device)
        wg = batch['word_group'].to(device)
        B = input_ids.size(0)

        cur_len = min(
            seq_length_for_progress((step - 1) / max(1, sched_total), seq_schedule, args.seq_length)
            if seq_schedule else args.seq_length, args.max_seq_length)
        input_ids = input_ids[:, :cur_len].contiguous()
        attn = attn[:, :cur_len].contiguous()
        wg = wg[:, :cur_len].contiguous()
        L = input_ids.size(1)

        # Determine prefix/suffix split for each example
        active_lens = attn.sum(dim=1).int()  # (B,)
        prefix_split = (active_lens.float() * args.prefix_fraction).int().clamp(min=10, max=L - 10)

        # Apply WWM masking (to ALL positions, as in baseline)
        masked_inputs, labels = apply_masking(input_ids, attn, wg, tokenizer, args.mask_mode, args.mask_prob, mask_gen)

        # Forward true sequence
        outputs = model(input_ids=masked_inputs, attention_mask=attn, output_hidden_states=True)
        true_logits = outputs.logits  # (B, L, V)

        # Standard WWM loss (on all masked positions, same as baseline)
        wwm_loss = F.cross_entropy(true_logits.view(-1, true_logits.size(-1)), labels.view(-1), ignore_index=-100)

        # Select hard negatives using prefix hidden states
        last_hidden = outputs.hidden_states[-1]  # (B, L, H)
        prefix_repr = torch.zeros(B, args.hidden_size, device=device)
        for i in range(B):
            sp = prefix_split[i].item()
            prefix_mask = attn[i, :sp].bool()
            if prefix_mask.any():
                prefix_repr[i] = last_hidden[i, :sp][prefix_mask].mean(dim=0)

        neg_indices = select_hard_negatives(prefix_repr.detach(), source_ids, args.negative_source_mode, args.same_source_preference)

        # Build negative sequences and forward
        neg_masked = masked_inputs.clone()
        neg_attn = attn.clone()
        for i in range(B):
            j = neg_indices[i].item()
            sp_i = prefix_split[i].item()
            sp_j = prefix_split[j].item()
            copy_len = min(sp_i, sp_j)
            neg_masked[i, :copy_len] = masked_inputs[j, :copy_len]
            neg_attn[i, :copy_len] = attn[j, :copy_len]
            if sp_j < sp_i:
                neg_masked[i, sp_j:sp_i] = tokenizer.pad_token_id
                neg_attn[i, sp_j:sp_i] = 0

        neg_outputs = model(input_ids=neg_masked, attention_mask=neg_attn)
        neg_logits = neg_outputs.logits

        # CPC margin loss on suffix masked targets only
        suffix_masked = torch.zeros(B, L, dtype=torch.bool, device=device)
        for i in range(B):
            sp = prefix_split[i].item()
            suffix_masked[i, sp:] = (labels[i, sp:] != -100)

        n_suffix_targets = suffix_masked.sum().item()
        if n_suffix_targets > 0:
            gold_ids = labels[suffix_masked]
            true_lp = F.log_softmax(true_logits[suffix_masked], dim=-1)
            neg_lp = F.log_softmax(neg_logits[suffix_masked], dim=-1)
            true_gold_lp = true_lp.gather(1, gold_ids.unsqueeze(1)).squeeze(1)
            neg_gold_lp = neg_lp.gather(1, gold_ids.unsqueeze(1)).squeeze(1)
            # Margin loss: max(0, m - lp_true + lp_neg)
            margin_violations = F.relu(args.cpc_margin - true_gold_lp + neg_gold_lp)
            cpc_loss = margin_violations.mean()
            margin_active = (margin_violations > 0).float().mean().item()
            mean_lp_diff = (true_gold_lp - neg_gold_lp).mean().item()
        else:
            cpc_loss = torch.tensor(0.0, device=device)
            margin_active = 0.0; mean_lp_diff = 0.0

        # Total loss
        total_loss = wwm_loss + args.lambda_cpc * cpc_loss

        opt.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()

        cum += words
        # Report actual pairing proportions; source-mode arguments are not evidence without this.
        neg_same_source = (source_ids[neg_indices] == source_ids).float().mean().item()
        neg_cross_source = (source_ids[neg_indices] != source_ids).float().mean().item()

        rec = {
            'step': step, 'cumulative_word_exposure': cum, 'batch_words': words,
            'wwm_loss': float(wwm_loss.detach().cpu()),
            'cpc_loss': float(cpc_loss.detach().cpu()),
            'total_loss': float(total_loss.detach().cpu()),
            'margin_active_frac': margin_active,
            'mean_lp_diff_true_minus_neg': mean_lp_diff,
            'n_suffix_targets': n_suffix_targets,
            'neg_same_source_frac': neg_same_source,
            'neg_cross_source_frac': neg_cross_source,
            'negative_source_mode': args.negative_source_mode,
            'lr': float(sched.get_last_lr()[0]),
            'elapsed_sec': time.time() - start,
        }
        logs.append(rec); logf.write(json.dumps(rec) + '\n'); logf.flush()
        if step == 1 or step % args.log_every == 0 or step == total_steps:
            print(json.dumps({'event': 'train', **rec}), flush=True)

        # Checkpointing
        while next_ckpt is not None and cum >= next_ckpt and next_ckpt <= args.max_word_exposure:
            name = f"chck_{next_ckpt // 1_000_000}M" if next_ckpt >= 1_000_000 and next_ckpt % 1_000_000 == 0 else f"chck_{next_ckpt}w"
            cp = out / 'hf_model' / name
            save_hf_checkpoint(model, tokenizer, cp)
            saved.append({'name': name, 'target_word_exposure': next_ckpt,
                          'actual_cumulative_word_exposure': cum, 'path': str(cp)})
            print(json.dumps({'event': 'checkpoint_saved', 'name': name, 'cum_words': cum}), flush=True)
            next_ckpt += args.checkpoint_words

    logf.close()
    save_hf_checkpoint(model, tokenizer, out / 'hf_model')

    metrics = {
        'variant': 'cpc_mlm',
        'backend': 'mlm',
        'parameter_count': disc_params,
        'word_exposure': cum,
        'actual_training_steps': total_steps,
        'loss_first': logs[0]['total_loss'] if logs else None,
        'loss_last': logs[-1]['total_loss'] if logs else None,
        'wwm_loss_last': logs[-1]['wwm_loss'] if logs else None,
        'cpc_loss_last': logs[-1]['cpc_loss'] if logs else None,
        'margin_active_frac_last': logs[-1]['margin_active_frac'] if logs else None,
        'mean_lp_diff_last': logs[-1]['mean_lp_diff_true_minus_neg'] if logs else None,
        'neg_same_source_frac_last': logs[-1]['neg_same_source_frac'] if logs else None,
        'neg_cross_source_frac_last': logs[-1]['neg_cross_source_frac'] if logs else None,
        'negative_source_mode': args.negative_source_mode,
        'lambda_cpc': args.lambda_cpc,
        'cpc_margin': args.cpc_margin,
        'prefix_fraction': args.prefix_fraction,
        'same_source_preference': args.same_source_preference,
        'mask_mode': args.mask_mode,
        'mask_prob': args.mask_prob,
        'batch_size': args.batch_size,
        'seed': args.seed,
        'extra_init_seed': args.extra_init_seed,
        'train_rng_seed': args.train_rng_seed,
        'hidden_size': args.hidden_size,
        'n_layer': args.n_layer,
        'n_head': args.n_head,
        'tokenization_coupling_summary': tok_sum,
        'source_words_consumed': source_words,
        'saved_checkpoints': saved,
    }
    (out / 'scientific_metrics.json').write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({
        'event': 'done', 'params': disc_params, 'word_exposure': cum,
        'wwm_loss_last': metrics['wwm_loss_last'],
        'cpc_loss_last': metrics['cpc_loss_last'],
        'margin_active_frac_last': metrics['margin_active_frac_last'],
        'mean_lp_diff_last': metrics['mean_lp_diff_last'],
    }), flush=True)


if __name__ == '__main__':
    main()
