#!/usr/bin/env python3
"""research WWM + Prefix-Continuation trainer with strict causal suffix masking.

Mechanism: standard bidirectional WWM on all masked tokens PLUS a dense
next-token continuation loss on suffix tokens where each suffix position
can ONLY attend to prefix + earlier suffix (enforced by a prefix-LM
attention mask). This tests whether dense directional prefix→suffix
supervision converts context variables into useful MLM logits.

The DeBERTa-v2 encoder accepts a 3D attention mask of shape (B, L, L)
directly. For the continuation forward:
- Prefix positions [0, P): bidirectional within prefix
- Suffix position t >= P: sees prefix [0, P) and earlier suffix [P, t]
- Continuation loss: predict token[t+1] from hidden[t] for t in [P, L-2]

Two forward passes per step:
1. Bidirectional (standard 2D mask) → WWM loss
2. Prefix-LM causal (3D mask) → continuation loss

Total: L = L_wwm + lambda_cont * L_cont

Built-in attention-visibility verification runs during smoke to prove:
(a) Later suffix changes do NOT affect earlier suffix logits (causal)
(b) Prefix changes DO affect suffix logits (directional dependence)
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
    p = argparse.ArgumentParser(description='WWM + Prefix-Continuation trainer')
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
    # Continuation-specific
    p.add_argument('--lambda_cont', type=float, default=1.0, help='Weight for continuation loss')
    p.add_argument('--prefix_fraction', type=float, default=0.5, help='Fraction of active tokens used as prefix')
    p.add_argument('--verify_causality', action='store_true', default=True,
                   help='Run attention-visibility verification during smoke (always True)')
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
        from babylm_masked_train import load_examples_jsonl
        p = pathlib.Path(args.example_jsonl)
        examples, total_words, total_rows, sample_rows = load_examples_jsonl(p, selected_words)
        manifest_files = [{'path': str(p), 'name': p.name, 'bytes': p.stat().st_size,
                           'sha256': sha256_file(p), 'whitespace_words': total_words, 'rows': total_rows}]
        meta = {'data_source_type': 'example_jsonl', 'example_jsonl': str(p), 'example_jsonl_label': args.example_jsonl_label}
    else:
        raw_dir, manifest_files = download_dataset(args, out)
        files = [raw_dir / n for n in TRAIN_FILES]
        pool = list(iter_examples(files, pool_words, args.words_per_example))
        for i, ex in enumerate(pool):
            ex.example_id = i
        rng = random.Random(args.seed); rng.shuffle(pool)
        examples = []; actual = 0
        for ex in pool:
            if actual >= selected_words: break
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


def build_prefix_lm_mask(attn_2d: torch.Tensor, prefix_len: int) -> torch.Tensor:
    """Construct a (B, L, L) prefix-LM attention mask.

    - Prefix positions [0, P): bidirectional (see all prefix positions)
    - Suffix positions [P, L): causal (see prefix + earlier suffix including self)
    - Padding positions: masked out in both directions

    The continuation loss uses hidden[t] to predict token[t+1], so position t
    must see up to and including itself (standard autoregressive convention).
    """
    B, L = attn_2d.shape
    P = prefix_len
    device = attn_2d.device

    # Build base causal pattern (L, L)
    base = torch.zeros(L, L, device=device)
    # Prefix bidirectional: positions [0,P) see all [0,P)
    base[:P, :P] = 1.0
    # All positions see prefix
    base[:, :P] = 1.0
    # Suffix causal: position t sees [P, t] (inclusive)
    suffix_size = L - P
    if suffix_size > 0:
        base[P:, P:] = torch.tril(torch.ones(suffix_size, suffix_size, device=device))

    # Combine with padding mask: (B, L, L) = base * row_active * col_active
    # row_active: position i is active → can attend (row i is valid)
    # col_active: position j is active → can be attended to (col j is valid)
    row_mask = attn_2d.unsqueeze(2)  # (B, L, 1)
    col_mask = attn_2d.unsqueeze(1)  # (B, 1, L)
    pad_mask = row_mask * col_mask    # (B, L, L)

    # Final mask: causal pattern AND padding
    causal_mask = base.unsqueeze(0) * pad_mask  # (B, L, L)
    return causal_mask


def causal_forward(model, input_ids, attn_2d, causal_mask_3d):
    """Forward with 3D causal mask, decomposed to avoid DeBERTa-v2 embedding mask issue.

    DeBERTa-v2's embedding layer uses the mask to zero padding embeddings and expects
    2D (B,L). The encoder accepts 3D (B,L,L) for attention control. We decompose:
    1. Embeddings with standard 2D padding mask
    2. Encoder with 3D causal mask
    3. Prediction head
    """
    embedding_output = model.deberta.embeddings(input_ids, mask=attn_2d)
    encoder_outputs = model.deberta.encoder(
        embedding_output, causal_mask_3d,
        output_hidden_states=False, output_attentions=False, return_dict=True
    )
    hidden = encoder_outputs.last_hidden_state
    logits = model.cls(hidden)
    return logits, hidden


def verify_attention_causality(model, tokenizer, batch, prefix_len: int, device):
    """Causal-attention verification: prove causal directionality before training.

    Tests:
    1. Changing a LATER suffix token does NOT change logits at an EARLIER suffix position.
    2. Changing an EARLIER suffix token DOES change logits at a LATER suffix position.
    3. Deleting prefix (replacing with pad) DOES change suffix logits.

    Returns dict with pass/fail and numerical evidence.
    """
    model.eval()
    input_ids = batch['input_ids'][:4].to(device)  # Use first 4 examples
    attn_2d = batch['attention_mask'][:4].to(device)
    B, L = input_ids.shape
    P = prefix_len

    # Build causal mask
    causal_mask = build_prefix_lm_mask(attn_2d, P)

    # Position indices for testing
    early_suffix = P + 5   # early suffix position
    late_suffix = P + 15   # late suffix position (we'll check if changing this affects early)
    target_pos = P + 10    # position where we measure logits

    if target_pos >= L or late_suffix >= L or early_suffix >= L:
        return {'status': 'SKIP', 'reason': f'sequence too short for test positions P={P} L={L}'}

    # Forward with original tokens using decomposed causal forward
    with torch.no_grad():
        logits_orig, _ = causal_forward(model, input_ids, attn_2d, causal_mask)

    # Test 1: Change LATER suffix token → logits at target_pos should NOT change
    input_modified_later = input_ids.clone()
    input_modified_later[:, late_suffix] = (input_ids[:, late_suffix] + 100) % len(tokenizer)
    with torch.no_grad():
        logits_later, _ = causal_forward(model, input_modified_later, attn_2d, causal_mask)

    # Max absolute difference at target_pos from later-suffix change
    later_diff = (logits_later[:, target_pos] - logits_orig[:, target_pos]).abs().max().item()

    # Test 2: Change EARLIER suffix token → logits at target_pos SHOULD change
    input_modified_earlier = input_ids.clone()
    input_modified_earlier[:, early_suffix] = (input_ids[:, early_suffix] + 100) % len(tokenizer)
    with torch.no_grad():
        logits_earlier, _ = causal_forward(model, input_modified_earlier, attn_2d, causal_mask)

    earlier_diff = (logits_earlier[:, target_pos] - logits_orig[:, target_pos]).abs().max().item()

    # Test 3: Delete prefix (replace with pad) → suffix logits SHOULD change
    input_no_prefix = input_ids.clone()
    input_no_prefix[:, :P] = tokenizer.pad_token_id
    # Also zero out prefix in causal mask (no position can attend to prefix)
    causal_mask_nopref = causal_mask.clone()
    causal_mask_nopref[:, :, :P] = 0
    with torch.no_grad():
        logits_nopref, _ = causal_forward(model, input_no_prefix, attn_2d, causal_mask_nopref)

    prefix_diff = (logits_nopref[:, target_pos] - logits_orig[:, target_pos]).abs().max().item()

    # Evaluation
    causality_ok = later_diff < 1e-5  # Later change must NOT affect target
    directionality_ok = earlier_diff > 0.01  # Earlier change MUST affect target
    prefix_dependence_ok = prefix_diff > 0.01  # Prefix removal MUST affect suffix

    result = {
        'status': 'PASS' if (causality_ok and directionality_ok and prefix_dependence_ok) else 'FAIL',
        'later_suffix_change_max_logit_diff': later_diff,
        'earlier_suffix_change_max_logit_diff': earlier_diff,
        'prefix_removal_max_logit_diff': prefix_diff,
        'causality_enforced': causality_ok,
        'earlier_context_affects_target': directionality_ok,
        'prefix_affects_suffix': prefix_dependence_ok,
        'test_positions': {'prefix_len': P, 'early_suffix': early_suffix,
                           'target_pos': target_pos, 'late_suffix': late_suffix},
    }
    model.train()
    return result


def main():
    args = build_args()
    start = time.time()
    out = pathlib.Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    tokenizer = make_portable_tokenizer(args.tokenizer_path)
    examples, actual_words, source_words = load_examples_standard(args, out, tokenizer)
    tok_sum = summarize_tokenization_coupling(examples, tokenizer, args.max_seq_length, args.tokenization_summary_limit)
    (out / 'tokenization_coupling_summary.json').write_text(json.dumps(tok_sum, indent=2), encoding='utf-8')

    ds = MaskedChunkDataset(examples, tokenizer, args.max_seq_length)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
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

    # === Run attention-visibility verification on first batch ===
    verification_result = None
    first_batch_for_verify = None

    for step, batch in enumerate(loader, 1):
        words = int(batch.pop('words').sum().item())
        input_ids = batch['input_ids'].to(device)
        attn_2d = batch['attention_mask'].to(device)
        wg = batch['word_group'].to(device)
        B = input_ids.size(0)

        cur_len = min(
            seq_length_for_progress((step - 1) / max(1, sched_total), seq_schedule, args.seq_length)
            if seq_schedule else args.seq_length, args.max_seq_length)
        input_ids = input_ids[:, :cur_len].contiguous()
        attn_2d = attn_2d[:, :cur_len].contiguous()
        wg = wg[:, :cur_len].contiguous()
        L = input_ids.size(1)

        # Determine prefix length (fixed for batch based on minimum active length)
        active_lens = attn_2d.sum(dim=1).int()
        P = int((active_lens.min().item()) * args.prefix_fraction)
        P = max(10, min(P, L - 20))  # ensure enough suffix for continuation

        # --- Attention-visibility verification on research ---
        if step == 1 and args.verify_causality:
            verification_result = verify_attention_causality(model, tokenizer, batch, P, device)
            print(json.dumps({'event': 'causality_verification', **verification_result}), flush=True)
            if verification_result['status'] == 'FAIL':
                raise RuntimeError(
                    f"CAUSALITY VERIFICATION FAILED — suffix is NOT strictly causal. "
                    f"Details: {json.dumps(verification_result)}")

        # === Forward 1: Bidirectional WWM (standard) ===
        masked_inputs, labels = apply_masking(input_ids, attn_2d, wg, tokenizer, args.mask_mode, args.mask_prob, mask_gen)
        outputs_bidi = model(input_ids=masked_inputs, attention_mask=attn_2d)
        wwm_loss = F.cross_entropy(
            outputs_bidi.logits.view(-1, outputs_bidi.logits.size(-1)),
            labels.view(-1), ignore_index=-100)

        # === Forward 2: Prefix-LM causal continuation ===
        causal_mask = build_prefix_lm_mask(attn_2d, P)  # (B, L, L)
        # Use ORIGINAL tokens (not masked) for continuation — we predict next token
        # Decomposed forward: 2D mask for embeddings, 3D causal mask for encoder
        logits_cont, _ = causal_forward(model, input_ids, attn_2d, causal_mask)  # (B, L, V)

        # Continuation loss: predict token[t+1] from hidden[t] for t in [P, L-2]
        # Only on active suffix positions
        cont_logits = logits_cont[:, P:L-1, :]  # (B, suffix_size-1, V)
        cont_targets = input_ids[:, P+1:L]       # (B, suffix_size-1)
        # Mask padding: only compute loss where target position is active
        cont_active = attn_2d[:, P+1:L]          # (B, suffix_size-1)
        n_cont_targets = int(cont_active.sum().item())

        if n_cont_targets > 0:
            cont_loss_flat = F.cross_entropy(
                cont_logits.reshape(-1, cont_logits.size(-1)),
                cont_targets.reshape(-1),
                reduction='none'
            ).reshape(B, -1)
            cont_loss = (cont_loss_flat * cont_active).sum() / cont_active.sum().clamp(min=1)
        else:
            cont_loss = torch.tensor(0.0, device=device)

        # === Total loss ===
        total_loss = wwm_loss + args.lambda_cont * cont_loss

        opt.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()

        cum += words
        n_wwm_targets = int((labels != -100).sum().item())

        rec = {
            'step': step, 'cumulative_word_exposure': cum, 'batch_words': words,
            'wwm_loss': float(wwm_loss.detach().cpu()),
            'cont_loss': float(cont_loss.detach().cpu()),
            'total_loss': float(total_loss.detach().cpu()),
            'n_wwm_targets': n_wwm_targets,
            'n_cont_targets': n_cont_targets,
            'prefix_len': P,
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
        'variant': 'wwm_prefix_continuation',
        'backend': 'mlm',
        'parameter_count': disc_params,
        'word_exposure': cum,
        'actual_training_steps': total_steps,
        'loss_first': logs[0]['total_loss'] if logs else None,
        'loss_last': logs[-1]['total_loss'] if logs else None,
        'wwm_loss_last': logs[-1]['wwm_loss'] if logs else None,
        'cont_loss_last': logs[-1]['cont_loss'] if logs else None,
        'lambda_cont': args.lambda_cont,
        'prefix_fraction': args.prefix_fraction,
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
        'causality_verification': verification_result,
    }
    (out / 'scientific_metrics.json').write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({
        'event': 'done', 'params': disc_params, 'word_exposure': cum,
        'wwm_loss_last': metrics['wwm_loss_last'],
        'cont_loss_last': metrics['cont_loss_last'],
        'causality_verification': verification_result['status'] if verification_result else 'NOT_RUN',
    }), flush=True)


if __name__ == '__main__':
    main()
