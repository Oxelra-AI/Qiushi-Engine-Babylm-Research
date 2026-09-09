#!/usr/bin/env python3
"""research Factorized WWM Cadence trainer.

Extends the protected DeBERTa-v2 WWM training with two schedule factors:
  A: Sequence-length schedule (e.g., 128→256→512) via existing --seq_len_schedule
  B: Mask-rate decay (e.g., 0.30→0.15) via --mask_prob_start/--mask_prob_end

Same data selection, model, tokenizer, checkpoint conventions as the baseline.
Designed for clean factorized comparison against research matched WWM baseline.

Factors:
  fixed:     seq_length=256, mask_prob=0.15 (≈ research baseline, already exists)
  len_only:  seq_len_schedule='0.0:128,0.4:256,0.7:512', mask_prob=0.15
  mask_only: seq_length=256, mask_prob 0.30→0.15
  combined:  seq_len_schedule + mask_prob decay
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


def mask_prob_for_progress(frac: float, start: float, end: float) -> float:
    """Linear interpolation of mask probability from start to end over training."""
    return start + (end - start) * min(1.0, max(0.0, frac))


def build_args():
    p = argparse.ArgumentParser(description='Factorized WWM Cadence trainer')
    p.add_argument('--dataset_id', default='BabyLM-community/BabyLM-2026-Strict-Small')
    p.add_argument('--dataset_revision', default='c92ab16b4f08858304b0815706065b3354d8fc0a')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--max_word_exposure', type=int, default=1_000_000)
    p.add_argument('--example_pool_words', type=int, default=1_000_000)
    p.add_argument('--checkpoint_words', type=int, default=1_000_000)
    p.add_argument('--words_per_example', type=int, default=160)
    p.add_argument('--example_jsonl', default='')
    p.add_argument('--example_jsonl_label', default='')
    p.add_argument('--example_jsonl_meta', default='')
    p.add_argument('--tokenizer_path', default='')
    p.add_argument('--tokenizer_label', default='baseline16k')
    p.add_argument('--tokenization_summary_limit', type=int, default=0)
    p.add_argument('--mask_mode', choices=['token', 'wwm'], default='wwm')
    # Mask prob scheduling
    p.add_argument('--mask_prob', type=float, default=0.15, help='Fixed mask prob (used if start==end)')
    p.add_argument('--mask_prob_start', type=float, default=-1, help='Start mask prob; -1 means use --mask_prob')
    p.add_argument('--mask_prob_end', type=float, default=-1, help='End mask prob; -1 means use --mask_prob')
    # Sequence length
    p.add_argument('--seq_length', type=int, default=256)
    p.add_argument('--max_seq_length', type=int, default=512)
    p.add_argument('--max_position_embeddings', type=int, default=512)
    p.add_argument('--seq_len_schedule', default='', help="e.g. '0.0:128,0.4:256,0.7:512'")
    # Model
    p.add_argument('--model_type', choices=['bert', 'deberta_v2'], default='deberta_v2')
    p.add_argument('--position_buckets', type=int, default=256)
    p.add_argument('--max_relative_positions', type=int, default=256)
    p.add_argument('--deberta_pos_att_type', default='p2c,c2p')
    p.add_argument('--deberta_relative_attention', choices=['true', 'false'], default='true')
    p.add_argument('--hidden_size', type=int, default=480)
    p.add_argument('--n_layer', type=int, default=8)
    p.add_argument('--n_head', type=int, default=8)
    p.add_argument('--ffn_mult', type=int, default=4)
    # Training
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--learning_rate', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--warmup_fraction', type=float, default=0.05)
    p.add_argument('--lr_total_steps', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--extra_init_seed', type=int, default=456)
    p.add_argument('--train_rng_seed', type=int, default=789)
    p.add_argument('--log_every', type=int, default=5)
    return p.parse_args()


def main():
    args = build_args()
    start = time.time()
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Resolve mask prob schedule
    mp_start = args.mask_prob_start if args.mask_prob_start >= 0 else args.mask_prob
    mp_end = args.mask_prob_end if args.mask_prob_end >= 0 else args.mask_prob
    mask_scheduled = (mp_start != mp_end)

    # Parse seq_len_schedule
    seq_schedule = []
    if args.seq_len_schedule:
        for part in args.seq_len_schedule.split(','):
            t, L = part.split(':')
            seq_schedule.append((float(t), int(L)))
        seq_schedule.sort()

    # Data selection (identical to protected baseline)
    tokenizer = make_portable_tokenizer(args.tokenizer_path)
    pool_words = max(args.example_pool_words, args.max_word_exposure)
    selected_words = args.max_word_exposure

    if args.example_jsonl:
        jsonl_path = pathlib.Path(args.example_jsonl)
        examples, _, _, _ = load_examples_jsonl(jsonl_path, selected_words)
    else:
        raw_dir, manifest_files = download_dataset(args, out)
        files = [raw_dir / n for n in TRAIN_FILES]
        pool = list(iter_examples(files, pool_words, args.words_per_example))
        for i, ex in enumerate(pool):
            ex.example_id = i
        rng = random.Random(args.seed)
        rng.shuffle(pool)
        examples = []
        actual = 0
        for ex in pool:
            if actual >= selected_words:
                break
            if actual + ex.words <= selected_words:
                examples.append(ex)
                actual += ex.words
            else:
                take = selected_words - actual
                examples.append(Example(' '.join(ex.text.split()[:take]), take, ex.example_id, ex.source))
                actual += take

    actual_words = sum(ex.words for ex in examples)
    if actual_words != selected_words:
        raise RuntimeError(f'word mismatch {actual_words} vs {selected_words}')

    source_words = {}
    for ex in examples:
        source_words[ex.source] = source_words.get(ex.source, 0) + ex.words

    # Save manifests
    (out / 'example_order_manifest.json').write_text(json.dumps({
        'seed': args.seed, 'selected_words': actual_words,
        'num_examples': len(examples), 'source_words': source_words,
    }, indent=2), encoding='utf-8')

    tok_sum = summarize_tokenization_coupling(examples, tokenizer, args.max_seq_length, args.tokenization_summary_limit)
    (out / 'tokenization_coupling_summary.json').write_text(json.dumps(tok_sum, indent=2), encoding='utf-8')

    # Dataset and loader — tokenize to max_seq_length (512) for later schedule use
    ds = MaskedChunkDataset(examples, tokenizer, args.max_seq_length)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
                        num_workers=2, pin_memory=torch.cuda.is_available())

    # Model
    if args.extra_init_seed >= 0:
        reset_all_rng(args.extra_init_seed)
    model = build_model(args, tokenizer)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    if args.train_rng_seed >= 0:
        reset_all_rng(args.train_rng_seed)

    # Optimizer and scheduler
    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    total_steps = len(loader)
    sched_total = args.lr_total_steps if args.lr_total_steps > 0 else total_steps
    if sched_total < total_steps:
        raise RuntimeError(f'lr_total_steps {sched_total} < actual steps {total_steps}')
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(sched_total * args.warmup_fraction)), sched_total)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.train_rng_seed if args.train_rng_seed >= 0 else args.seed)

    next_ckpt = args.checkpoint_words if args.checkpoint_words > 0 else None
    cum = 0
    logs = []
    saved = []
    disc_params = sum(p.numel() for p in model.parameters())
    print(f'Model params: {disc_params:,}', flush=True)
    print(json.dumps({'config': {'mask_scheduled': mask_scheduled, 'mp_start': mp_start, 'mp_end': mp_end,
                                  'seq_schedule': seq_schedule, 'batch_size': args.batch_size,
                                  'max_seq_length': args.max_seq_length, 'total_steps': total_steps}}), flush=True)

    model.train()
    logf = (out / 'training_log.jsonl').open('w', encoding='utf-8')

    for step, batch in enumerate(loader, 1):
        words = int(batch.pop('words').sum().item())
        input_ids = batch['input_ids'].to(device)
        attn = batch['attention_mask'].to(device)
        wg = batch['word_group'].to(device)

        frac = (step - 1) / max(1, sched_total)

        # Dynamic sequence length
        cur_len = seq_length_for_progress(frac, seq_schedule, args.seq_length) if seq_schedule else args.seq_length
        cur_len = min(cur_len, args.max_seq_length)
        input_ids = input_ids[:, :cur_len].contiguous()
        attn = attn[:, :cur_len].contiguous()
        wg = wg[:, :cur_len].contiguous()

        # Dynamic mask probability
        cur_mask_prob = mask_prob_for_progress(frac, mp_start, mp_end) if mask_scheduled else args.mask_prob

        # Standard WWM masking and forward
        masked_inputs, labels = apply_masking(input_ids, attn, wg, tokenizer, args.mask_mode, cur_mask_prob, gen)
        opt.zero_grad(set_to_none=True)
        out_model = model(input_ids=masked_inputs, attention_mask=attn, labels=labels)
        loss = out_model.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()

        cum += words
        n_masked = int((labels != -100).sum().item())
        n_active = int(attn.sum().item())

        rec = {
            'step': step, 'cumulative_word_exposure': cum, 'batch_words': words,
            'loss': float(loss.detach().cpu()),
            'cur_seq_len': cur_len, 'cur_mask_prob': round(cur_mask_prob, 4),
            'n_masked_targets': n_masked, 'n_active_tokens': n_active,
            'lr': float(sched.get_last_lr()[0]),
            'elapsed_sec': time.time() - start,
        }
        logs.append(rec)
        logf.write(json.dumps(rec) + '\n')
        logf.flush()
        if step == 1 or step % args.log_every == 0 or step == total_steps:
            print(json.dumps({'event': 'train', **rec}), flush=True)

        # Checkpointing
        while next_ckpt is not None and cum >= next_ckpt and next_ckpt <= args.max_word_exposure:
            name = f"chck_{next_ckpt // 1_000_000}M" if next_ckpt >= 1_000_000 and next_ckpt % 1_000_000 == 0 else f"chck_{next_ckpt}w"
            cp = out / 'hf_model' / name
            save_hf_checkpoint(model, tokenizer, cp)
            saved.append({'name': name, 'words': next_ckpt, 'cum': cum, 'path': str(cp)})
            print(json.dumps({'event': 'checkpoint', 'name': name, 'cum': cum}), flush=True)
            next_ckpt += args.checkpoint_words

    logf.close()
    save_hf_checkpoint(model, tokenizer, out / 'hf_model')

    metrics = {
        'variant': 'cadence_wwm',
        'parameter_count': disc_params,
        'word_exposure': cum,
        'total_steps': total_steps,
        'mask_scheduled': mask_scheduled,
        'mask_prob_start': mp_start,
        'mask_prob_end': mp_end,
        'seq_len_schedule': args.seq_len_schedule,
        'batch_size': args.batch_size,
        'loss_first': logs[0]['loss'] if logs else None,
        'loss_last': logs[-1]['loss'] if logs else None,
        'mask_prob_first': logs[0]['cur_mask_prob'] if logs else None,
        'mask_prob_last': logs[-1]['cur_mask_prob'] if logs else None,
        'seq_len_first': logs[0]['cur_seq_len'] if logs else None,
        'seq_len_last': logs[-1]['cur_seq_len'] if logs else None,
        'saved_checkpoints': saved,
    }
    (out / 'scientific_metrics.json').write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'event': 'done', 'params': disc_params, 'exposure': cum,
                      'loss_last': metrics['loss_last'], 'mask_last': metrics['mask_prob_last'],
                      'len_last': metrics['seq_len_last']}), flush=True)


if __name__ == '__main__':
    main()
