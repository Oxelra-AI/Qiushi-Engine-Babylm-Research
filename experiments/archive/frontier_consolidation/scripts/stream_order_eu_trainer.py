#!/usr/bin/env python3
"""Stream-order experience-utilization chunk trainer.

Why this exists: research implemented the word-boundary chunking mechanism, but
its first verified form repeats the 10M pool in canonical file order.  The existing
legal40k/depth baselines train on a materialized 100M stream: ten shuffled 10M
blocks that are each an exact multiset repetition of the same pool.  Launching the
research trainer unchanged would therefore bundle experience-utilization with a
large data-order change.

This trainer preserves the baseline materialized 100M stream order while replacing
prefix-hidden row slices with word-boundary chunks:
  * tokenize the exact 10M pool once;
  * read the exact 100M stream as ten 10M blocks of raw-line hashes;
  * for exposure epoch e, use stream block e's row order and stage-specific length;
  * build word-boundary chunks in that stream order;
  * distribute each 10M block into 253 optimizer updates, matching research's
    stage-reset chunk-stream accounting (2530 total updates for 10 blocks);
  * apply full-effective-batch WWM and masked-token-weighted microbatch gradients.

Padding uses tokenizer.pad_token_id (not id 0) so masked-LM inputs match the
ordinary tokenizer-padded baseline convention on ignored positions.
"""
from __future__ import annotations

import argparse
import collections
import csv
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
from transformers import get_cosine_schedule_with_warmup

USER_ROOT = Path('.').resolve()
WORKSPACE = USER_ROOT / 'experiments/archive/frontier_consolidation'
SCRIPTS = WORKSPACE / 'scripts'
A01_SCRIPTS = USER_ROOT / 'experiments/archive/representation_and_objectives/scripts'
COMPACT_EXPERIENCE_SCRIPTS = USER_ROOT / 'experiments/archive/compact_experience/scripts'
for p in (SCRIPTS, A01_SCRIPTS, COMPACT_EXPERIENCE_SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import masking_curriculum_trainer as base  # noqa: E402
import experience_utilization_trainer_ref as chunkbase  # noqa: E402

DEFAULT_BASE_10M = 'experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl'
DEFAULT_STREAM_100M = 'experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl'
POOL_WORDS = 10_000_000
TOTAL_EXPOSURE = 100_000_000
EXPECTED_EPOCHS = 10
STEPS_PER_EPOCH = 253
ARMS = chunkbase.ARMS


def now_utc() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()


def raw_line_hash(raw: bytes) -> str:
    return hashlib.sha256(raw.rstrip(b'\r\n')).hexdigest()


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Stream-order experience-utilization chunk trainer')
    p.add_argument('--arm', required=True, choices=sorted(ARMS.keys()))
    p.add_argument('--base_jsonl', default=DEFAULT_BASE_10M,
                   help='Exact 10M pool used for tokenizer/chunk tokenization')
    p.add_argument('--stream_jsonl', default=DEFAULT_STREAM_100M,
                   help='Exact materialized 100M stream whose row order is preserved')
    p.add_argument('--tokenizer_path', required=True)
    p.add_argument('--tokenizer_label', default='legal40k')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--micro_batch_size', type=int, default=64)
    # Model
    p.add_argument('--hidden_size', type=int, default=480)
    p.add_argument('--n_layer', type=int, default=8)
    p.add_argument('--n_head', type=int, default=8)
    p.add_argument('--ffn_mult', type=int, default=4)
    p.add_argument('--intermediate_size', type=int, default=0,
                   help='Explicit FFN intermediate size (0=hidden_size*ffn_mult)')
    p.add_argument('--position_buckets', type=int, default=256)
    p.add_argument('--max_relative_positions', type=int, default=256)
    p.add_argument('--max_position_embeddings', type=int, default=512)
    p.add_argument('--deberta_pos_att_type', default='p2c,c2p')
    # Optimization.  Defaults match legal40k/depth baselines unless overridden.
    p.add_argument('--learning_rate', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--warmup_fraction', type=float, default=0.06)
    p.add_argument('--mask_prob', type=float, default=0.15)
    # Reproducibility
    p.add_argument('--seed', type=int, default=43)
    p.add_argument('--extra_init_seed', type=int, default=-1)
    p.add_argument('--train_rng_seed', type=int, default=-1)
    # Checkpoints/logging
    p.add_argument('--checkpoint_words', type=int, default=1_000_000)
    p.add_argument('--log_every', type=int, default=50)
    # Dry-run mode
    p.add_argument('--dry_run', action='store_true')
    p.add_argument('--dry_run_epochs', type=int, default=-1,
                   help='Limit dry-run to this many exposure epochs (-1=all)')
    p.add_argument('--max_train_steps', type=int, default=0,
                   help='If >0, stop after this many optimizer steps for a real smoke run or focused dry-run')
    return p.parse_args()


def reset_all_rng(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_base_token_pool(path: Path, tokenizer) -> dict[str, Any]:
    token_by_hash: dict[str, dict[str, Any]] = {}
    counter: collections.Counter[str] = collections.Counter()
    source_words: collections.Counter[str] = collections.Counter()
    row_count = 0
    word_count = 0
    token_count = 0
    duplicate_hashes = 0
    with path.open('rb') as f:
        for row_idx, raw in enumerate(f):
            rec = json.loads(raw)
            h = raw_line_hash(raw)
            counter[h] += 1
            if counter[h] == 2:
                duplicate_hashes += 1
            text = str(rec['text'])
            words = int(rec['words'])
            source = str(rec.get('source', ''))
            source_words[source] += words
            if h not in token_by_hash:
                by_word, raw_tokens, unassigned = chunkbase.token_ids_by_whitespace_word(text, tokenizer)
                if unassigned:
                    raise RuntimeError(f'unassigned tokenizer offsets at base row {row_idx}')
                if len(by_word) != words:
                    raise RuntimeError(f'word count mismatch at base row {row_idx}: {len(by_word)} vs {words}')
                token_by_hash[h] = {
                    'word_tokens': by_word,
                    'raw_tokens': raw_tokens,
                    'words': words,
                    'source': source,
                    'first_row_index': row_idx,
                }
            word_count += words
            token_count += int(token_by_hash[h]['raw_tokens'])
            row_count += 1
            if row_count % 10000 == 0:
                print(json.dumps({'event': 'base_tokenize_progress', 'rows': row_count}), flush=True)
    if word_count != POOL_WORDS:
        raise RuntimeError(f'10M pool word count {word_count} != {POOL_WORDS}')
    return {
        'token_by_hash': token_by_hash,
        'counter': counter,
        'source_words': dict(source_words),
        'row_count': row_count,
        'word_count': word_count,
        'raw_tokens_per_epoch': token_count,
        'duplicate_raw_line_hashes': duplicate_hashes,
    }


def load_stream_blocks(path: Path, base_counter: collections.Counter[str], base_rows: int) -> dict[str, Any]:
    blocks: list[list[str]] = []
    block_records: list[dict[str, Any]] = []
    source_words_exposure: collections.Counter[str] = collections.Counter()
    total_rows = 0
    total_words = 0
    with path.open('rb') as f:
        for epoch in range(EXPECTED_EPOCHS):
            hs: list[str] = []
            c: collections.Counter[str] = collections.Counter()
            words = 0
            src_words: collections.Counter[str] = collections.Counter()
            for row_in_block in range(base_rows):
                raw = f.readline()
                if not raw:
                    raise RuntimeError(f'stream ended early at epoch {epoch}, row {row_in_block}')
                rec = json.loads(raw)
                h = raw_line_hash(raw)
                hs.append(h)
                c[h] += 1
                w = int(rec['words'])
                words += w
                src = str(rec.get('source', ''))
                src_words[src] += w
                source_words_exposure[src] += w
            leftover_missing = dict((base_counter - c).most_common(5))
            extra = dict((c - base_counter).most_common(5))
            block_records.append({
                'epoch': epoch,
                'rows': len(hs),
                'words': words,
                'words_match_10M': words == POOL_WORDS,
                'raw_line_multiset_matches_10M_pool': c == base_counter,
                'missing_hash_examples': leftover_missing,
                'extra_hash_examples': extra,
                'source_words': dict(src_words.most_common()),
                'first_hashes': hs[:5],
            })
            blocks.append(hs)
            total_rows += len(hs)
            total_words += words
        if f.readline():
            raise RuntimeError('stream has more rows than 10 complete 10M blocks')
    return {
        'blocks': blocks,
        'block_records': block_records,
        'total_rows': total_rows,
        'total_words': total_words,
        'source_words_exposure': dict(source_words_exposure),
        'all_blocks_words_10M': all(b['words_match_10M'] for b in block_records),
        'all_blocks_multiset_match': all(b['raw_line_multiset_matches_10M_pool'] for b in block_records),
    }


def collate_chunks(chunks: list[chunkbase.Chunk], L: int, pad_id: int) -> dict[str, torch.Tensor]:
    n = len(chunks)
    input_ids = torch.full((n, L), int(pad_id), dtype=torch.long)
    attention_mask = torch.zeros((n, L), dtype=torch.long)
    word_group = torch.full((n, L), -1, dtype=torch.long)
    words = torch.zeros(n, dtype=torch.long)
    for i, ch in enumerate(chunks):
        m = len(ch.input_ids)
        if m > L:
            raise RuntimeError(f'chunk length {m} > L{L}')
        if m != len(ch.word_group):
            raise RuntimeError('chunk input/word_group length mismatch')
        if m:
            input_ids[i, :m] = torch.tensor(ch.input_ids, dtype=torch.long)
            attention_mask[i, :m] = 1
            word_group[i, :m] = torch.tensor(ch.word_group, dtype=torch.long)
        words[i] = ch.charged_words
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'word_group': word_group, 'words': words}


def chunks_for_block_order(hashes: list[str], token_by_hash: dict[str, dict[str, Any]], L: int) -> tuple[list[chunkbase.Chunk], dict[str, Any]]:
    chunks: list[chunkbase.Chunk] = []
    stats = {
        'rows': 0,
        'charged_words': 0,
        'active_tokens': 0,
        'chunks': 0,
        'overlong_words': 0,
        'continuation_chunks': 0,
        'missing_hashes': 0,
    }
    for stream_pos, h in enumerate(hashes):
        info = token_by_hash.get(h)
        if info is None:
            stats['missing_hashes'] += 1
            continue
        row_chunks, meta = chunkbase.chunks_from_word_tokens(info['word_tokens'], L, stream_pos)
        chunks.extend(row_chunks)
        stats['rows'] += 1
        stats['charged_words'] += int(info['words'])
        stats['active_tokens'] += int(info['raw_tokens'])
        stats['chunks'] += len(row_chunks)
        stats['overlong_words'] += int(meta['overlong_words'])
        stats['continuation_chunks'] += int(meta['continuation_chunks'])
    if stats['missing_hashes']:
        raise RuntimeError(f"missing {stats['missing_hashes']} stream hashes from base token pool")
    if stats['charged_words'] != POOL_WORDS:
        raise RuntimeError(f"block charged words {stats['charged_words']} != {POOL_WORDS}")
    return chunks, stats


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    args = build_args()
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    start_time = time.time()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = base.make_portable_tokenizer(args.tokenizer_path)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    vocab_size = len(tokenizer)
    stages = ARMS[args.arm]
    total_epochs = sum(e for _, e in stages)
    if total_epochs != EXPECTED_EPOCHS:
        raise RuntimeError(f'arm {args.arm} has {total_epochs} epochs, expected {EXPECTED_EPOCHS}')
    total_steps = STEPS_PER_EPOCH * total_epochs
    args.max_seq_length = max(L for L, _ in stages)
    args.seq_length = args.max_seq_length
    effective_intermediate = args.intermediate_size if args.intermediate_size > 0 else args.hidden_size * args.ffn_mult

    base_path = Path(args.base_jsonl)
    stream_path = Path(args.stream_jsonl)
    base_sha = sha256_file(base_path)
    stream_sha = sha256_file(stream_path)

    seed_mapping = {
        'base_seed': args.seed,
        'initialization_seed': args.extra_init_seed if args.extra_init_seed >= 0 else args.seed,
        'training_rng_seed': args.train_rng_seed if args.train_rng_seed >= 0 else args.seed,
        'note': 'When extra_init_seed and train_rng_seed are non-negative, base seed is overwritten before model initialization and before masking/data RNG.',
    }

    print(json.dumps({
        'event': 'init',
        'created_utc': now_utc(),
        'arm': args.arm,
        'stages': stages,
        'total_epochs': total_epochs,
        'total_steps': total_steps,
        'max_train_steps': args.max_train_steps,
        'base_jsonl': str(base_path),
        'stream_jsonl': str(stream_path),
        'base_sha256': base_sha,
        'stream_sha256': stream_sha,
        'vocab_size': vocab_size,
        'pad_token_id': pad_id,
        'warmup_fraction': args.warmup_fraction,
        'seed_mapping': seed_mapping,
        'dry_run': args.dry_run,
    }), flush=True)

    t0 = time.time()
    pool = load_base_token_pool(base_path, tokenizer)
    tokenize_sec = time.time() - t0
    print(json.dumps({
        'event': 'base_pool_tokenized',
        'rows': pool['row_count'],
        'words': pool['word_count'],
        'raw_tokens': pool['raw_tokens_per_epoch'],
        'tokens_per_word': pool['raw_tokens_per_epoch'] / POOL_WORDS,
        'duplicate_raw_line_hashes': pool['duplicate_raw_line_hashes'],
        'seconds': round(tokenize_sec, 1),
    }), flush=True)

    stream = load_stream_blocks(stream_path, pool['counter'], pool['row_count'])
    if stream['total_words'] != TOTAL_EXPOSURE:
        raise RuntimeError(f"stream words {stream['total_words']} != {TOTAL_EXPOSURE}")
    if not stream['all_blocks_words_10M'] or not stream['all_blocks_multiset_match']:
        raise RuntimeError('100M stream is not ten exact 10M multiset blocks')

    # Save order/provenance manifests before any expensive training.
    stream_manifest = {
        'status': 'STREAM_ORDER_MANIFEST',
        'created_utc': now_utc(),
        'base_jsonl': str(base_path),
        'base_sha256': base_sha,
        'stream_jsonl': str(stream_path),
        'stream_sha256': stream_sha,
        'arm': args.arm,
        'stages': stages,
        'base_rows': pool['row_count'],
        'raw_tokens_per_epoch': pool['raw_tokens_per_epoch'],
        'stream_total_rows': stream['total_rows'],
        'stream_total_words': stream['total_words'],
        'all_blocks_words_10M': stream['all_blocks_words_10M'],
        'all_blocks_multiset_match': stream['all_blocks_multiset_match'],
        'block_records': stream['block_records'],
    }
    write_json(out / 'stream_order_manifest.json', stream_manifest)

    example_order_manifest = {
        'seed': args.seed,
        'selected_for_training_words': TOTAL_EXPOSURE,
        'num_consumed_examples': stream['total_rows'],
        'data_source_type': 'stream_order_chunked_jsonl',
        'base_jsonl': str(base_path),
        'base_sha256': base_sha,
        'stream_jsonl': str(stream_path),
        'stream_sha256': stream_sha,
        'arm': args.arm,
        'stages': stages,
        'chunking': 'word_boundary_stream_order_stage_reset',
        'steps_per_10M_block': STEPS_PER_EPOCH,
        'total_steps': total_steps,
        'max_train_steps': args.max_train_steps,
        'seed_mapping': seed_mapping,
        'masking_curriculum': 'wwm_fixed',
        'mask_prob_start': args.mask_prob,
        'mask_prob_end': args.mask_prob,
        'source_words_consumed': stream['source_words_exposure'],
    }
    write_json(out / 'example_order_manifest.json', example_order_manifest)

    if not args.dry_run:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        reset_all_rng(args.seed)
        if args.extra_init_seed >= 0:
            reset_all_rng(args.extra_init_seed)
        model = chunkbase._build_model(args, tokenizer)
        param_count = sum(p.numel() for p in model.parameters())
        if args.train_rng_seed >= 0:
            reset_all_rng(args.train_rng_seed)
        model.to(device)
        model.train()
        optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.98))
        warmup_steps = max(1, int(total_steps * args.warmup_fraction))
        sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        gen = torch.Generator(device=device)
        gen.manual_seed(args.train_rng_seed if args.train_rng_seed >= 0 else args.seed)
        print(json.dumps({'event': 'model_ready', 'device': str(device), 'param_count': param_count, 'warmup_steps': warmup_steps}), flush=True)
    else:
        device = torch.device('cpu')
        param_count = 0
        warmup_steps = max(1, int(total_steps * args.warmup_fraction))
        gen = torch.Generator(device='cpu')
        gen.manual_seed(args.train_rng_seed if args.train_rng_seed >= 0 else args.seed)

    curriculum_state = base.MaskingCurriculumState(
        curriculum='wwm_fixed',
        mask_prob_start=args.mask_prob,
        mask_prob_end=args.mask_prob,
        switch_frac=0.7,
        amlm_window=10,
        amlm_lambda=0.2,
    )
    curriculum_state.initialize(vocab_size=vocab_size, total_steps=total_steps)

    global_step = 0
    cumulative_words = 0
    saved_checkpoints: list[dict[str, Any]] = []
    stage_records: list[dict[str, Any]] = []
    epoch_records: list[dict[str, Any]] = []
    step_records: list[dict[str, Any]] = []
    loss_values: list[float] = []
    next_ckpt = args.checkpoint_words
    max_epochs_to_run = total_epochs if not args.dry_run or args.dry_run_epochs <= 0 else min(total_epochs, args.dry_run_epochs)
    logf = None if args.dry_run else (out / 'training_log.jsonl').open('w', encoding='utf-8')

    try:
        abs_epoch = 0
        stop = False
        for stage_idx, (stage_length, num_epochs) in enumerate(stages):
            print(json.dumps({'event': 'stage_start', 'stage_idx': stage_idx, 'length': stage_length, 'num_epochs': num_epochs, 'global_step': global_step, 'cumulative_words': cumulative_words}), flush=True)
            for ep_in_stage in range(num_epochs):
                if abs_epoch >= max_epochs_to_run:
                    stop = True
                    break
                t_build = time.time()
                epoch_hashes = stream['blocks'][abs_epoch]
                epoch_chunks, chunk_stats = chunks_for_block_order(epoch_hashes, pool['token_by_hash'], stage_length)
                build_sec = time.time() - t_build
                n_chunks = len(epoch_chunks)
                step_dist = chunkbase.distribute_into_steps(n_chunks, STEPS_PER_EPOCH)
                total_charged = sum(c.charged_words for c in epoch_chunks)
                total_active = sum(len(c.input_ids) for c in epoch_chunks)
                if total_charged != POOL_WORDS:
                    raise RuntimeError(f'epoch {abs_epoch} charged {total_charged} != {POOL_WORDS}')
                if total_active != pool['raw_tokens_per_epoch']:
                    raise RuntimeError(f"epoch {abs_epoch} active tokens {total_active} != {pool['raw_tokens_per_epoch']}")
                if ep_in_stage == 0:
                    stage_records.append({
                        'stage_idx': stage_idx,
                        'length': stage_length,
                        'num_epochs': num_epochs,
                        'first_epoch_chunks': n_chunks,
                        'charged_words_per_epoch': total_charged,
                        'active_tokens_per_epoch': total_active,
                        'steps_per_epoch': STEPS_PER_EPOCH,
                        'overlong_words_first_epoch': chunk_stats['overlong_words'],
                        'continuation_chunks_first_epoch': chunk_stats['continuation_chunks'],
                    })
                print(json.dumps({'event': 'chunks_built', 'epoch': abs_epoch, 'length': stage_length, 'n_chunks': n_chunks, 'charged_words': total_charged, 'active_tokens': total_active, 'overlong_words': chunk_stats['overlong_words'], 'continuation_chunks': chunk_stats['continuation_chunks'], 'build_sec': round(build_sec, 1)}), flush=True)

                ep_words = 0
                ep_active = 0
                ep_masked = 0
                ep_steps = 0
                for step_in_ep, (cs, ce) in enumerate(step_dist):
                    if args.max_train_steps > 0 and global_step >= args.max_train_steps:
                        stop = True
                        break
                    sc = epoch_chunks[cs:ce]
                    batch = collate_chunks(sc, stage_length, pad_id)
                    s_words = int(batch['words'].sum().item())
                    s_active = int(batch['attention_mask'].sum().item())
                    curriculum_state.current_step = global_step

                    if args.dry_run:
                        masked_inp, labels = base.apply_masking_curriculum(batch['input_ids'], batch['attention_mask'], batch['word_group'], tokenizer, curriculum_state, gen)
                        n_pred = int((labels != -100).sum().item())
                        loss_float = 0.0
                        del masked_inp, labels
                    else:
                        inp = batch['input_ids'].to(device, non_blocking=True)
                        att = batch['attention_mask'].to(device, non_blocking=True)
                        wg = batch['word_group'].to(device, non_blocking=True)
                        masked_inp, labels = base.apply_masking_curriculum(inp, att, wg, tokenizer, curriculum_state, gen)
                        n_pred = int((labels != -100).sum().item())
                        if n_pred <= 0:
                            raise RuntimeError(f'zero masked tokens at step {global_step}')
                        optim.zero_grad(set_to_none=True)
                        weighted_loss = 0.0
                        n_rows = int(inp.shape[0])
                        for mb_s in range(0, n_rows, args.micro_batch_size):
                            mb_e = min(mb_s + args.micro_batch_size, n_rows)
                            mb_lab = labels[mb_s:mb_e]
                            n_mb = int((mb_lab != -100).sum().item())
                            if n_mb == 0:
                                continue
                            out_m = model(input_ids=masked_inp[mb_s:mb_e], attention_mask=att[mb_s:mb_e], labels=mb_lab)
                            scale = n_mb / n_pred
                            (out_m.loss * scale).backward()
                            weighted_loss += float(out_m.loss.detach().cpu()) * scale
                            del out_m
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optim.step()
                        sched.step()
                        loss_float = weighted_loss
                        del inp, att, wg, masked_inp, labels

                    n_micro = math.ceil(len(sc) / args.micro_batch_size)
                    global_step += 1
                    cumulative_words += s_words
                    ep_words += s_words
                    ep_active += s_active
                    ep_masked += n_pred
                    ep_steps += 1

                    if args.dry_run:
                        step_records.append({
                            'global_step': global_step,
                            'epoch': abs_epoch,
                            'stage_length': stage_length,
                            'step_in_epoch': step_in_ep,
                            'n_chunks': len(sc),
                            'charged_words': s_words,
                            'active_tokens': s_active,
                            'masked_tokens': n_pred,
                            'microbatches': n_micro,
                            'cumulative_words': cumulative_words,
                        })
                    else:
                        loss_values.append(loss_float)
                        rec = {
                            'step': global_step,
                            'loss': loss_float,
                            'lr': float(sched.get_last_lr()[0]),
                            'batch_words': s_words,
                            'cumulative_word_exposure': cumulative_words,
                            'seq_len': stage_length,
                            'masked_tokens': n_pred,
                            'effective_mask_rate': round(n_pred / max(1, s_active), 4),
                            'n_chunks': len(sc),
                            'microbatches': n_micro,
                            'elapsed_sec': round(time.time() - start_time, 1),
                        }
                        assert logf is not None
                        logf.write(json.dumps(rec) + '\n')
                        logf.flush()
                        if global_step == 1 or global_step % args.log_every == 0 or global_step == total_steps:
                            print(json.dumps({'event': 'train', **rec}), flush=True)

                    while next_ckpt is not None and cumulative_words >= next_ckpt and next_ckpt <= TOTAL_EXPOSURE:
                        name = f'chck_{next_ckpt // 1_000_000}M' if next_ckpt % 1_000_000 == 0 else f'chck_{next_ckpt}w'
                        cp = out / 'hf_model' / name
                        if not args.dry_run:
                            base.save_hf_checkpoint(model, tokenizer, cp)
                            print(json.dumps({'event': 'checkpoint_saved', 'name': name, 'cum_words': cumulative_words}), flush=True)
                        saved_checkpoints.append({
                            'name': name,
                            'target_word_exposure': next_ckpt,
                            'actual_cumulative_word_exposure': cumulative_words,
                            'global_step': global_step,
                            'stage_length': stage_length,
                            'epoch': abs_epoch,
                            'path': str(cp),
                        })
                        next_ckpt += args.checkpoint_words

                epoch_records.append({
                    'epoch': abs_epoch,
                    'stage_idx': stage_idx,
                    'stage_length': stage_length,
                    'epoch_in_stage': ep_in_stage,
                    'stream_block_index': abs_epoch,
                    'steps': ep_steps,
                    'chunks': n_chunks,
                    'charged_words': ep_words,
                    'active_tokens': ep_active,
                    'masked_tokens': ep_masked,
                    'cumulative_words_after': cumulative_words,
                    'overlong_words': chunk_stats['overlong_words'],
                    'continuation_chunks': chunk_stats['continuation_chunks'],
                })
                print(json.dumps({'event': 'epoch_done', 'epoch': abs_epoch, 'stage_length': stage_length, 'steps': ep_steps, 'charged_words': ep_words, 'active_tokens': ep_active, 'masked_tokens': ep_masked, 'cumulative_words': cumulative_words, 'partial_epoch_due_to_max_train_steps': stop}), flush=True)
                abs_epoch += 1
                del epoch_chunks
                if stop:
                    break
            if stop:
                break
    finally:
        if logf is not None:
            logf.close()

    if not args.dry_run:
        base.save_hf_checkpoint(model, tokenizer, out / 'hf_model')

    completed_epochs = len(epoch_records)
    full_run = completed_epochs == total_epochs
    verification = {
        'completed_epochs': completed_epochs,
        'total_steps_executed': global_step,
        'total_charged_words': cumulative_words,
        'steps_match_expected': (global_step == total_steps) if full_run else None,
        'words_match_100M': (cumulative_words == TOTAL_EXPOSURE) if full_run else None,
        'all_epoch_words_10M': all(r['charged_words'] == POOL_WORDS for r in epoch_records),
        'all_epoch_steps_253': all(r['steps'] == STEPS_PER_EPOCH for r in epoch_records),
        'all_epoch_active_tokens_equal_base': all(r['active_tokens'] == pool['raw_tokens_per_epoch'] for r in epoch_records),
        'checkpoint_count': len(saved_checkpoints),
        'pad_token_id_used': pad_id,
        'stream_blocks_multiset_match': stream['all_blocks_multiset_match'],
    }
    result = {
        'status': 'STREAM_ORDER_EXPERIENCE_UTILIZATION_DRYRUN' if args.dry_run else 'STREAM_ORDER_EXPERIENCE_UTILIZATION_DONE',
        'created_utc': now_utc(),
        'arm': args.arm,
        'stages': stages,
        'base_jsonl': str(base_path),
        'base_sha256': base_sha,
        'stream_jsonl': str(stream_path),
        'stream_sha256': stream_sha,
        'tokenizer_label': args.tokenizer_label,
        'tokenizer_path': args.tokenizer_path,
        'tokenizer_vocab_hash': stable_hash(tokenizer.get_vocab()),
        'vocab_size': vocab_size,
        'pad_token_id': pad_id,
        'model_family': 'DebertaV2ForMaskedLM',
        'parameter_count': param_count,
        'hidden_size': args.hidden_size,
        'n_layer': args.n_layer,
        'n_head': args.n_head,
        'ffn_mult': args.ffn_mult,
        'intermediate_size': effective_intermediate,
        'learning_rate': args.learning_rate,
        'warmup_fraction': args.warmup_fraction,
        'warmup_steps': warmup_steps,
        'weight_decay': args.weight_decay,
        'max_train_steps': args.max_train_steps,
        'seed_mapping': seed_mapping,
        'masking_curriculum': 'wwm_fixed',
        'relation_masking': False,
        'mask_prob_start': args.mask_prob,
        'mask_prob_end': args.mask_prob,
        'mask_prob': args.mask_prob,
        'seed': args.seed,
        'extra_init_seed': args.extra_init_seed,
        'train_rng_seed': args.train_rng_seed,
        'raw_tokens_per_epoch': pool['raw_tokens_per_epoch'],
        'total_steps_executed': global_step,
        'total_charged_words': cumulative_words,
        'word_exposure': cumulative_words,
        'loss_first': loss_values[0] if loss_values else None,
        'loss_last': loss_values[-1] if loss_values else None,
        'stage_records': stage_records,
        'epoch_records': epoch_records,
        'checkpoints': saved_checkpoints,
        'verification': verification,
        'source_words_consumed': stream['source_words_exposure'],
        'elapsed_sec': round(time.time() - start_time, 1),
    }
    out_json = out / ('dryrun_metrics.json' if args.dry_run else 'scientific_metrics.json')
    write_json(out_json, result)
    if args.dry_run and step_records:
        write_csv(out / 'dryrun_step_accounting.csv', step_records)
    print(json.dumps({
        'event': 'done',
        'arm': args.arm,
        'dry_run': args.dry_run,
        'completed_epochs': completed_epochs,
        'total_steps': global_step,
        'total_words': cumulative_words,
        'verification': verification,
        'out_json': str(out_json),
        'elapsed_sec': round(time.time() - start_time, 1),
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
