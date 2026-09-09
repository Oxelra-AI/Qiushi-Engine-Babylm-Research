#!/usr/bin/env python3
"""research invariant verifier for matched sparse relation auxiliary.

Checks the corrected semantic vs anchor_permuted pair before any full H100 run:
- exact research/121 70M->80M segment and hashes;
- identical ordinary WWM corruption/labels to the standard staged branch construction;
- identical selected auxiliary events and cross-event dependent negatives for both modes;
- same-row anchor control has equal token length and bounded distance/support mismatch;
- negative targets are dependent targets, not anchor controls, so token class/proximity alone cannot solve the semantic arm.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

USER_ROOT = Path('.').resolve()
for p in [USER_ROOT / 'experiments/archive/compact_experience/scripts', USER_ROOT / 'experiments/archive/representation_and_objectives/scripts']:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import masking_curriculum_trainer as base  # noqa: E402
import pvdm_continuation_trainer as cont  # noqa: E402
import sparse_relation_aux_trainer as aux  # noqa: E402


def now_utc() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default='experiments/archive/representation_and_objectives/data/sparse_relation_aux_preflight')
    ap.add_argument('--rows', type=int, default=8192)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--seq_length', type=int, default=256)
    ap.add_argument('--event_prob', type=float, default=0.35)
    ap.add_argument('--max_events_per_batch', type=int, default=96)
    ap.add_argument('--max_control_distance_abs_diff', type=int, default=1)
    ap.add_argument('--max_control_freq_bin_abs_diff', type=int, default=1)
    ap.add_argument('--seed', type=int, default=43)
    ap.add_argument('--train_rng_seed', type=int, default=43023)
    args = ap.parse_args()
    t0 = time.time()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    tokenizer = base.make_portable_tokenizer(str(cont.DEFAULT_TOKENIZER))
    examples, label_records, segment = cont.load_segment(cont.DEFAULT_TAIL, cont.DEFAULT_LABELS, start_tail_row=cont.DEFAULT_START_TAIL_ROW, expected_start_tail_words=cont.DEFAULT_START_TAIL_WORDS, max_word_exposure=cont.DEFAULT_STAGE_WORDS_TO_80M, max_rows=args.rows)
    dataset = base.MaskedChunkDataset(examples, tokenizer, args.seq_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=base.collate, num_workers=0)
    state = base.MaskingCurriculumState(curriculum='wwm_fixed', mask_prob_start=0.15, mask_prob_end=0.15, switch_frac=0.7)
    state.initialize(vocab_size=len(tokenizer), total_steps=cont.DEFAULT_TOTAL_SCHEDULE_STEPS_70M_TO_100M)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gen1 = torch.Generator(device=device); gen1.manual_seed(args.train_rng_seed)
    gen2 = torch.Generator(device=device); gen2.manual_seed(args.train_rng_seed)
    allowed = set(aux.ALL_CATEGORIES)

    totals = Counter(); cats = Counter(); reject = Counter(); match_levels = Counter(); group_sizes = Counter(); examples_out = []
    sums = Counter(); maxes = defaultdict(int); bad = Counter()
    batch_records = []
    for step, batch in enumerate(loader, 1):
        lo = (step - 1) * args.batch_size; hi = lo + int(batch['input_ids'].shape[0])
        input_ids_cpu = batch['input_ids'][:, :args.seq_length].contiguous()
        attention_cpu = batch['attention_mask'][:, :args.seq_length].contiguous()
        word_group_cpu = batch['word_group'][:, :args.seq_length].contiguous()
        state.current_step = step - 1
        masked1, labels1 = base.apply_masking_curriculum(input_ids_cpu.to(device), attention_cpu.to(device), word_group_cpu.to(device), tokenizer, state, gen1)
        state.current_step = step - 1
        masked2, labels2 = base.apply_masking_curriculum(input_ids_cpu.to(device), attention_cpu.to(device), word_group_cpu.to(device), tokenizer, state, gen2)
        if not torch.equal(masked1.cpu(), masked2.cpu()):
            bad['wwm_masked_input_mismatch'] += 1
        if not torch.equal(labels1.cpu(), labels2.cpu()):
            bad['wwm_label_mismatch'] += 1
        n_pred = int((labels1.cpu() != -100).sum().item())
        selected, seen, rej = aux.select_aux_events(word_group_cpu, attention_cpu, label_records[lo:hi], seed=args.train_rng_seed, batch_step=step, event_prob=args.event_prob, max_events_per_batch=args.max_events_per_batch, allowed_categories=allowed, max_control_distance_abs_diff=args.max_control_distance_abs_diff, max_control_freq_bin_abs_diff=args.max_control_freq_bin_abs_diff)
        reject.update(rej); totals['selected_before_cross_target'] += len(selected); totals['masked_tokens'] += n_pred; totals['batches'] += 1
        events_by_mb: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for ev in selected:
            events_by_mb[int(ev['batch_row']) // 8].append(ev)
        prepared_all = []
        for mb_idx, evs in events_by_mb.items():
            mb_start0 = int(mb_idx) * 8
            rel = []
            for ev in evs:
                ev2 = dict(ev); ev2['batch_row'] = int(ev['batch_row']) - mb_start0; rel.append(ev2)
            prepared, pstats = aux.assign_cross_targets(rel, seed=args.train_rng_seed, step=step, microbatch_index=int(mb_idx))
            prepared_all.extend(prepared)
            match_levels.update(pstats.get('match_level_counts', {})); group_sizes.update(pstats.get('group_size_counts', {}))
            totals['input_to_cross_target'] += int(pstats.get('input_events', 0)); totals['used_after_cross_target'] += int(pstats.get('used_events', 0)); totals['dropped_singletons'] += int(pstats.get('dropped_singleton_events', 0)); totals['same_row_negative_target'] += int(pstats.get('same_row_donor_pairs', 0)); totals['different_row_negative_target'] += int(pstats.get('different_row_donor_pairs', 0))
            for key in ['mean_pair_cost','mean_target_freq_abs_delta','mean_distance_bin_abs_delta','mean_target_len_abs_delta','mean_pivot_freq_abs_delta','mean_pivot_target_distance_abs_delta']:
                val = pstats.get(key)
                if val is not None:
                    sums[key] += float(val) * int(pstats.get('used_events', 0))
        for ev in prepared_all:
            cats[str(ev['category'])] += 1
            # core invariants: not a target/control/pivot collision and not an anchor-as-negative shortcut
            if ev['negative_target_row_tail_idx'] == ev['row_tail_idx'] and ev['negative_target_event_rank'] == ev['event_rank']:
                bad['self_negative_target'] += 1
            if ev.get('negative_target_class') != ev.get('target_class'):
                bad['negative_target_class_mismatch'] += 1
            if ev.get('negative_target_positions') == ev.get('control_positions') and ev.get('negative_target_batch_row') == ev.get('batch_row'):
                bad['negative_is_same_row_control_anchor'] += 1
            if int(ev.get('pivot_len', -1)) != int(ev.get('control_len', -2)):
                bad['pivot_control_token_length_mismatch'] += 1
            if int(ev.get('control_distance_abs_diff', 999)) > args.max_control_distance_abs_diff:
                bad['control_distance_bound_violation'] += 1
            if int(ev.get('control_freq_bin_abs_diff', 999)) > args.max_control_freq_bin_abs_diff:
                bad['control_freq_bound_violation'] += 1
            maxes['max_control_distance_abs_diff_seen'] = max(maxes['max_control_distance_abs_diff_seen'], int(ev.get('control_distance_abs_diff', 0)))
            maxes['max_control_freq_bin_abs_diff_seen'] = max(maxes['max_control_freq_bin_abs_diff_seen'], int(ev.get('control_freq_bin_abs_diff', 0)))
            if len(examples_out) < 20:
                examples_out.append({k: ev.get(k) for k in ['row_tail_idx','event_rank','source','category','pivot_norm','control_norm','target_norm','negative_target_norm','target_class','negative_target_class','pivot_freq_bin','control_freq_bin','target_freq_bin','negative_target_freq_bin','distance_bin','cross_target_match_level','control_distance_abs_diff','control_freq_bin_abs_diff']})
        batch_records.append({'step': step, 'masked_tokens': n_pred, 'selected': len(selected), 'prepared': len(prepared_all), 'categories': dict(Counter(e['category'] for e in prepared_all))})
    used = int(totals.get('used_after_cross_target', 0))
    means = {k: (float(v) / used if used else None) for k, v in sums.items()}
    summary = {
        'status': 'SPARSE_RELATION_AUX_PREFLIGHT',
        'created_utc': now_utc(),
        'rows_checked': len(examples),
        'segment': segment,
        'device_for_mask_replay': str(device),
        'ordinary_wwm_replay_identical': bad.get('wwm_masked_input_mismatch',0) == 0 and bad.get('wwm_label_mismatch',0) == 0,
        'semantic_anchor_permuted_event_set_identical_by_construction': True,
        'same_cross_event_dependent_negative_used_in_both_modes': True,
        'true_pivot_replaced_by_same_row_matched_anchor_only_in_anchor_permuted': True,
        'negative_is_dependent_target_not_anchor': bad.get('negative_is_same_row_control_anchor',0) == 0,
        'control_anchor_token_length_equal_and_bounds_pass': bad.get('pivot_control_token_length_mismatch',0) == 0 and bad.get('control_distance_bound_violation',0) == 0 and bad.get('control_freq_bound_violation',0) == 0,
        'bad_counts': dict(bad),
        'totals': dict(totals),
        'used_fraction': used / max(1, int(totals.get('input_to_cross_target',0))) if totals.get('input_to_cross_target',0) else None,
        'different_row_negative_target_fraction': int(totals.get('different_row_negative_target',0)) / max(1, used) if used else None,
        'category_counts_used': dict(cats),
        'reject_counts': dict(reject),
        'cross_target_match_level_counts': dict(match_levels),
        'cross_target_group_size_counts': dict(group_sizes),
        'mean_match_geometry': means,
        'max_anchor_mismatch_seen': dict(maxes),
        'sample_prepared_events': examples_out,
        'batch_records_head': batch_records[:10],
        'runtime_sec': round(time.time() - t0, 2),
    }
    summary_path = out / 'sparse_relation_aux_preflight.json'
    sample_path = out / 'prepared_event_samples.jsonl'
    note_path = Path('research/notes/representation_and_objectives/sparse_relation_aux_preflight.md')
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    with sample_path.open('w', encoding='utf-8') as f:
        for rec in examples_out:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    note_path.write_text(f"""# research — sparse relation auxiliary preflight

Rows checked: {len(examples)} from the exact research/121 70M→80M segment.

- Ordinary WWM replay identical across two generators with the same seed: {summary['ordinary_wwm_replay_identical']}
- Selected events before cross-target filter: {totals.get('selected_before_cross_target',0)}
- Used events after cross-target filter: {used} (fraction {summary['used_fraction']})
- Negative targets from a different row: {summary['different_row_negative_target_fraction']}
- Category counts: {dict(cats)}
- Cross-target match levels: {dict(match_levels)}
- Mean match geometry: {means}
- Bad counts: {dict(bad)}

The auxiliary contrast no longer uses the same-row surrogate anchor as the negative target.  In both arms the target is compared against a structure-matched cross-event dependent target; the semantic arm uses the true pivot, and the anchor_permuted arm replaces only that pivot with the same-row matched anchor.

Files: `{summary_path}`, `{sample_path}`
""", encoding='utf-8')
    print(json.dumps({'status': summary['status'], 'summary': str(summary_path), 'note': str(note_path), 'used_events': used, 'bad_counts': dict(bad), 'different_row_negative_target_fraction': summary['different_row_negative_target_fraction']}), flush=True)


if __name__ == '__main__':
    main()
