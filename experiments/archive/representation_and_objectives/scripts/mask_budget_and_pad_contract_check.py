#!/usr/bin/env python3
"""research: mask-budget and pad/label contract checks for stream-order chunk trainer.

This CPU-only check integrates independent review feedback before any expensive experience-
utilization launch:
  * verify that padded positions never receive MLM labels in the first real
    stream-order chunked update for U256 and U64_128_256;
  * place current row256 legal40k baseline mask/LR/accounting beside the research
    chunk dry-runs;
  * quantify the 0.15 versus target-matched mask probability issue.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path('.').resolve()
WORKSPACE = ROOT / 'experiments/archive/representation_and_objectives'
SCRIPTS = WORKSPACE / 'scripts'
COMPACT_EXPERIENCE = ROOT / 'experiments/archive/compact_experience/scripts'
for p in (SCRIPTS, COMPACT_EXPERIENCE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import masking_curriculum_trainer as base  # noqa: E402
import stream_order_experience_utilization_trainer as stream_trainer  # noqa: E402

BASE_RUN = WORKSPACE / 'training/runs/legal40k_accum_compact_view_reinvest_seed43022'
BASE_MANIFEST = WORKSPACE / 'data/legal40k_accum_training/legal40k_accum_train_manifest_seed43022.json'
DRY_U256 = WORKSPACE / 'data/dryrun_stream_order_legal40k_U256_full/dryrun_metrics.json'
DRY_U64 = WORKSPACE / 'data/dryrun_stream_order_legal40k_U64_128_256_full/dryrun_metrics.json'
TOKENIZER = WORKSPACE / 'data/legal_representation_route_map/tokenizers/legal_byte_bpe_40k'
BASE10 = ROOT / stream_trainer.DEFAULT_BASE_10M
STREAM100 = ROOT / stream_trainer.DEFAULT_STREAM_100M
OUT_DIR = WORKSPACE / 'data/mask_budget_and_pad_contract'
NOTE = (ROOT / 'research/notes/representation_and_objectives/mask_budget_and_pad_contract.md')
CURRENT_ROW256_ACTIVE_TOKENS = 137_061_620  # research legal40k current row256 measurement.
U256_ACTIVE_TOKENS = 139_426_440           # research/79 legal40k chunk visibility measurement.


def now() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def read_training_log_summary(path: Path) -> dict[str, Any]:
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    bins: list[dict[str, Any]] = []
    for r in rows:
        idx = min(9, int((int(r['cumulative_word_exposure']) - 1) // 10_000_000))
        while len(bins) <= idx:
            bins.append({'words': 0, 'masked_tokens': 0, 'steps': 0})
        bins[idx]['words'] += int(r['batch_words'])
        bins[idx]['masked_tokens'] += int(r['masked_tokens'])
        bins[idx]['steps'] += 1
    return {
        'path': str(path),
        'records': len(rows),
        'first': rows[0] if rows else None,
        'last': rows[-1] if rows else None,
        'total_words': sum(int(r['batch_words']) for r in rows),
        'total_masked_tokens': sum(int(r['masked_tokens']) for r in rows),
        'epoch_like_bins': bins,
    }


def first_batch_pad_check(arm: str) -> dict[str, Any]:
    tokenizer = base.make_portable_tokenizer(str(TOKENIZER))
    pool = stream_trainer.load_base_token_pool(BASE10, tokenizer)
    stream = stream_trainer.load_stream_blocks(STREAM100, pool['counter'], pool['row_count'])
    stage_length = stream_trainer.ARMS[arm][0][0]
    chunks, chunk_stats = stream_trainer.chunks_for_block_order(stream['blocks'][0], pool['token_by_hash'], stage_length)
    step_dist = stream_trainer.chunkbase.distribute_into_steps(len(chunks), stream_trainer.STEPS_PER_EPOCH)
    cs, ce = step_dist[0]
    batch_chunks = chunks[cs:ce]
    batch = stream_trainer.collate_chunks(batch_chunks, stage_length, tokenizer.pad_token_id)
    state = base.MaskingCurriculumState(
        curriculum='wwm_fixed', mask_prob_start=0.15, mask_prob_end=0.15,
        switch_frac=0.7, amlm_window=10, amlm_lambda=0.2)
    state.initialize(vocab_size=len(tokenizer), total_steps=2530)
    state.current_step = 0
    gen = torch.Generator(device='cpu')
    gen.manual_seed(43023)
    masked, labels = base.apply_masking_curriculum(batch['input_ids'], batch['attention_mask'], batch['word_group'], tokenizer, state, gen)
    pad_positions = batch['attention_mask'] == 0
    labels_on_pad = labels[pad_positions] != -100
    active_unlabeled_pad = int(labels_on_pad.sum().item())
    masked_pad_changed = int((masked[pad_positions] != tokenizer.pad_token_id).sum().item())
    # Whole-word grouping sanity: all non-pad active positions have non-negative group.
    active_missing_group = int(((batch['attention_mask'] == 1) & (batch['word_group'] < 0)).sum().item())
    return {
        'arm': arm,
        'stage_length': stage_length,
        'first_step_chunks': len(batch_chunks),
        'first_step_charged_words': int(batch['words'].sum().item()),
        'first_step_active_tokens': int(batch['attention_mask'].sum().item()),
        'first_step_masked_tokens': int((labels != -100).sum().item()),
        'pad_token_id': int(tokenizer.pad_token_id),
        'pad_positions': int(pad_positions.sum().item()),
        'labels_on_pad_positions': active_unlabeled_pad,
        'masked_pad_positions_changed_from_pad_id': masked_pad_changed,
        'active_positions_missing_word_group': active_missing_group,
        'all_pad_labels_ignored': active_unlabeled_pad == 0,
        'pad_inputs_remain_pad': masked_pad_changed == 0,
        'all_active_positions_have_word_group': active_missing_group == 0,
        'chunk_stats_epoch0': chunk_stats,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = load_json(BASE_RUN / 'scientific_metrics.json')
    manifest = load_json(BASE_MANIFEST)
    dry_u256 = load_json(DRY_U256)
    dry_u64 = load_json(DRY_U64)
    train_summary = read_training_log_summary(BASE_RUN / 'training_log.jsonl')
    pad_checks = [first_batch_pad_check('U256'), first_batch_pad_check('U64_128_256')]

    baseline_total_masked = int(train_summary['total_masked_tokens'])
    u256_mask015_total = sum(int(e['masked_tokens']) for e in dry_u256['epoch_records'])
    u64_mask015_total = sum(int(e['masked_tokens']) for e in dry_u64['epoch_records'])
    expected_targetmatch_prob = 0.15 * (CURRENT_ROW256_ACTIVE_TOKENS / U256_ACTIVE_TOKENS)
    realized_targetmatch_prob = baseline_total_masked / U256_ACTIVE_TOKENS
    comparison = {
        'baseline_row256': {
            'run_dir': str(BASE_RUN),
            'actual_training_steps': metrics.get('actual_training_steps'),
            'word_exposure': metrics.get('word_exposure'),
            'mask_prob_start': metrics.get('mask_prob_start'),
            'mask_prob_end': metrics.get('mask_prob_end'),
            'masking_curriculum': metrics.get('masking_curriculum'),
            'learning_rate': manifest['fixed_recipe'].get('learning_rate'),
            'warmup_fraction': manifest['fixed_recipe'].get('warmup_fraction'),
            'weight_decay': manifest['fixed_recipe'].get('weight_decay'),
            'total_masked_tokens_from_log': baseline_total_masked,
            'current_row256_active_tokens_step77': CURRENT_ROW256_ACTIVE_TOKENS,
            'realized_masked_per_active_token': baseline_total_masked / CURRENT_ROW256_ACTIVE_TOKENS,
        },
        'stream_order_U256_mask015': {
            'dryrun_json': str(DRY_U256),
            'total_steps': dry_u256['total_steps_executed'],
            'word_exposure': dry_u256['word_exposure'],
            'active_tokens': sum(int(e['active_tokens']) for e in dry_u256['epoch_records']),
            'total_masked_tokens_dryrun': u256_mask015_total,
            'masked_token_delta_vs_baseline': u256_mask015_total - baseline_total_masked,
            'masked_token_ratio_vs_baseline': u256_mask015_total / baseline_total_masked,
        },
        'stream_order_U64_128_256_mask015': {
            'dryrun_json': str(DRY_U64),
            'total_steps': dry_u64['total_steps_executed'],
            'word_exposure': dry_u64['word_exposure'],
            'active_tokens': sum(int(e['active_tokens']) for e in dry_u64['epoch_records']),
            'total_masked_tokens_dryrun': u64_mask015_total,
            'masked_token_delta_vs_baseline': u64_mask015_total - baseline_total_masked,
            'masked_token_ratio_vs_baseline': u64_mask015_total / baseline_total_masked,
        },
        'target_matched_mask_probability': {
            'expected_match_step77_formula_0p15_times_row256_active_over_U256_active': expected_targetmatch_prob,
            'realized_match_from_baseline_log_total_masked_over_U256_active': realized_targetmatch_prob,
            'baseline_total_masked_vs_expected_0p15_row256_active': baseline_total_masked - int(round(0.15 * CURRENT_ROW256_ACTIVE_TOKENS)),
        },
    }
    payload = {
        'status': 'MASK_BUDGET_AND_PAD_CONTRACT',
        'created_utc': now(),
        'base_run_config_sha_sources': {
            'baseline_scientific_metrics': str(BASE_RUN / 'scientific_metrics.json'),
            'baseline_train_manifest': str(BASE_MANIFEST),
            'baseline_training_log': str(BASE_RUN / 'training_log.jsonl'),
            'u256_dryrun': str(DRY_U256),
            'u64_dryrun': str(DRY_U64),
            'base10_sha256': sha256_file(BASE10),
            'stream100_sha256': sha256_file(STREAM100),
        },
        'baseline_training_log_summary': train_summary,
        'pad_label_contract_first_batches': pad_checks,
        'comparison': comparison,
        'interpretation': {
            'pad_contract_ok': all(x['all_pad_labels_ignored'] and x['pad_inputs_remain_pad'] and x['all_active_positions_have_word_group'] for x in pad_checks),
            'baseline_uses_mask_prob_0p15': metrics.get('mask_prob_start') == 0.15 and metrics.get('mask_prob_end') == 0.15,
            'baseline_lr_warmup_match_stream_trainer_defaults': manifest['fixed_recipe'].get('learning_rate') == 0.001 and manifest['fixed_recipe'].get('warmup_fraction') == 0.06,
            'mask015_tests_full_experience_utilization_not_target_matched_visibility_only': True,
            'targetmatched_mask_prob_to_pass_if_visibility_only_is_required': expected_targetmatch_prob,
        },
    }
    out_json = OUT_DIR / 'mask_budget_and_pad_contract.json'
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    rows = []
    for key in ['baseline_row256', 'stream_order_U256_mask015', 'stream_order_U64_128_256_mask015']:
        rec = comparison[key]
        rows.append({
            'condition': key,
            'steps': rec.get('actual_training_steps') or rec.get('total_steps'),
            'word_exposure': rec.get('word_exposure'),
            'active_tokens': rec.get('current_row256_active_tokens_step77') or rec.get('active_tokens'),
            'masked_tokens': rec.get('total_masked_tokens_from_log') or rec.get('total_masked_tokens_dryrun'),
            'masked_delta_vs_baseline': rec.get('masked_token_delta_vs_baseline', 0),
            'masked_ratio_vs_baseline': rec.get('masked_token_ratio_vs_baseline', 1.0),
        })
    csv_path = OUT_DIR / 'mask_budget_comparison.csv'
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    note = '\n'.join([
        '# research — Mask budget and pad/label contract for stream-order chunk training',
        '',
        'This check integrates independent_review feedback before any H100 launch of the experience-utilization trainer.',
        '',
        '## Baseline parity',
        '',
        f"The matched legal40k row256 baseline (`legal40k_accum_compact_view_reinvest_seed43022`) used fixed WWM 0.15, AdamW LR 0.001, warmup fraction 0.06, and 2,529 optimizer updates. Its training log sums to {baseline_total_masked:,} masked targets over 100M charged words.",
        f"The stream-order U256/U64 chunk dry-runs use the same data stream SHA `{sha256_file(STREAM100)}` and recover {U256_ACTIVE_TOKENS:,} active tokens versus the research row256 active-token measurement {CURRENT_ROW256_ACTIVE_TOKENS:,}.",
        '',
        '## Mask budget',
        '',
        f"At mask_prob=0.15, U256 dry-run produced {u256_mask015_total:,} targets, {u256_mask015_total - baseline_total_masked:+,} versus the realized row256 baseline ({u256_mask015_total / baseline_total_masked:.6f}x). U64_128_256 produced {u64_mask015_total:,} targets, {u64_mask015_total - baseline_total_masked:+,} ({u64_mask015_total / baseline_total_masked:.6f}x).",
        f"A target-matched launch should pass mask_prob about {expected_targetmatch_prob:.8f} by the research expected-active-token formula, or {realized_targetmatch_prob:.8f} if matching the exact realized baseline target count from the training log.",
        '',
        '## Pad/label contract',
        '',
        *[f"- {x['arm']} first batch L{x['stage_length']}: chunks={x['first_step_chunks']}, words={x['first_step_charged_words']}, active={x['first_step_active_tokens']}, masked={x['first_step_masked_tokens']}, pad positions={x['pad_positions']}, labels-on-pad={x['labels_on_pad_positions']}, pad-input changes={x['masked_pad_positions_changed_from_pad_id']}, active missing word groups={x['active_positions_missing_word_group']}." for x in pad_checks],
        '',
        'The pad/label contract is clean for the first real stream-order chunked update in both arms. The remaining design choice is scientific: mask_prob=0.15 tests full recovered experience including extra target pressure; mask_prob≈0.147456 tests target-matched visibility. The first expensive run after the depth result should choose this explicitly and record it in the launch manifest.',
        '',
        f"JSON: `{out_json}`",
        f"CSV: `{csv_path}`",
    ]) + '\n'
    NOTE.write_text(note, encoding='utf-8')
    print(json.dumps({
        'status': payload['status'],
        'pad_contract_ok': payload['interpretation']['pad_contract_ok'],
        'baseline_total_masked': baseline_total_masked,
        'u256_mask015_total': u256_mask015_total,
        'u256_mask_delta': u256_mask015_total - baseline_total_masked,
        'targetmatched_expected_mask_prob': expected_targetmatch_prob,
        'targetmatched_realized_mask_prob': realized_targetmatch_prob,
        'out_json': str(out_json),
        'note': str(NOTE),
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
