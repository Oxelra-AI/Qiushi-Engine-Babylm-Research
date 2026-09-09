#!/usr/bin/env python3
"""research budget audit for the unchanged research full-context auxiliary trainer.

The trainer reports only base-stream word exposure, but each auxiliary loss performs
extra full-context forward passes over selected rows. This script reuses the same
candidate and shuffle attachment code, without loading a model or taking optimizer
steps, to count base words, selected event views, and full-row word-pass debit.
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

USER_ROOT = Path('.').resolve()
for p in [USER_ROOT/'experiments/archive/compact_experience/scripts', USER_ROOT/'experiments/archive/representation_and_objectives/scripts']:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import masking_curriculum_trainer as base  # noqa: E402
import pvdm_continuation_trainer as cont  # noqa: E402
import fullctx_gradient_preflight as gp  # noqa: E402

OUT = Path('experiments/archive/representation_and_objectives/data/fullctx_aux_budget_audit')
NOTE = Path('research/notes/representation_and_objectives/fullctx_aux_budget_audit.md')


def summarize(xs: list[float]) -> dict[str, Any]:
    if not xs:
        return {'n': 0}
    ys = sorted(xs)
    def q(p: float) -> float:
        if len(ys) == 1: return ys[0]
        f = p*(len(ys)-1); lo = int(math.floor(f)); hi = int(math.ceil(f))
        if lo == hi: return ys[lo]
        return ys[lo]*(hi-f) + ys[hi]*(f-lo)
    return {'n': len(xs), 'mean': sum(xs)/len(xs), 'median': q(0.5), 'p05': q(0.05), 'p95': q(0.95), 'min': ys[0], 'max': ys[-1]}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tok_path = cont.DEFAULT_TOKENIZER
    tail = cont.DEFAULT_TAIL
    labels_p = cont.DEFAULT_LABELS
    hashes = {'tail_sha256': base.sha256_file(tail), 'labels_sha256': base.sha256_file(labels_p), 'tokenizer_sha256': base.sha256_file(tok_path/'tokenizer.json')}
    for k, exp in cont.EXPECTED.items():
        if hashes[k] != exp:
            raise RuntimeError(f'{k} mismatch {hashes[k]} != {exp}')
    tokenizer = base.make_portable_tokenizer(str(tok_path))
    examples, label_records, segment = cont.load_segment(tail, labels_p, start_tail_row=64255, expected_start_tail_words=10011326, max_word_exposure=9971308, max_rows=0)
    dataset = base.MaskedChunkDataset(examples, tokenizer, 256)
    loader = DataLoader(dataset, batch_size=256, shuffle=False, collate_fn=base.collate, num_workers=0)
    print(json.dumps({'event': 'loaded', 'rows': len(examples), 'segment': segment}), flush=True)
    donor_pools = gp.build_donor_pools(examples, label_records, tokenizer, batch_size=256, seq_length=256, max_control_distance_abs_diff=8, max_control_freq_bin_abs_diff=2, max_target_token_len=6, num_workers=0)

    base_words_total = 0
    aux_events_total = 0
    aux_rows_word_debit_two_views = 0
    aux_rows_word_debit_four_views = 0
    aux_seq_token_debit_two_views = 0
    aux_seq_token_debit_four_views = 0
    event_counts = Counter(); shuffle_levels = Counter(); row_repeat_counts = Counter(); per_batch = []
    batch_records_path = OUT / 'batch_budget_records.jsonl'
    with batch_records_path.open('w', encoding='utf-8') as f:
        for step, batch in enumerate(loader, 1):
            lo=(research)*256; hi=lo+int(batch['input_ids'].shape[0])
            words_by_row = [int(ex.words) for ex in examples[lo:hi]]
            words = int(sum(words_by_row)); base_words_total += words
            ids=batch['input_ids'][:,:256].contiguous(); attn=batch['attention_mask'][:,:256].contiguous(); wg=batch['word_group'][:,:256].contiguous()
            raw = gp.candidate_events_for_batch(ids, attn, wg, label_records[lo:hi], seed=43023, step=step, max_control_distance_abs_diff=8, max_control_freq_bin_abs_diff=2, max_target_token_len=6)
            evs, shuf_stats = gp.attach_shuffled(raw, donor_pools, seed=43023, step=step)
            aux_events_total += len(evs)
            event_counts.update(str(e['category']) for e in evs)
            shuffle_levels.update(str(e.get('shuffle_level', 'NA')) for e in evs)
            # The current trainer's compute_aux_loss builds four full-context views per event.
            # A repaired semantic-only implementation would need two views per event: true and shuffled.
            row_words = 0; seq_tokens = 0
            for e in evs:
                br = int(e['batch_row'])
                row_words += words_by_row[br]
                seq_tokens += int(attn[br].sum().item())
                row_repeat_counts[(step, br)] += 1
            aux_rows_word_debit_two_views += 2*row_words
            aux_rows_word_debit_four_views += 4*row_words
            aux_seq_token_debit_two_views += 2*seq_tokens
            aux_seq_token_debit_four_views += 4*seq_tokens
            rec = {'step': step, 'base_words': words, 'events': len(evs), 'event_categories': dict(Counter(str(e['category']) for e in evs)), 'shuffle_attach': shuf_stats, 'selected_row_words_one_view': row_words, 'selected_seq_tokens_one_view': seq_tokens, 'two_view_word_debit': 2*row_words, 'four_view_word_debit': 4*row_words, 'two_view_ratio_to_base': (2*row_words/words if words else None), 'four_view_ratio_to_base': (4*row_words/words if words else None)}
            per_batch.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False)+'\n')
            if step == 1 or step % 50 == 0:
                print(json.dumps({'event': 'progress', 'step': step, 'base_words': base_words_total, 'aux_events': aux_events_total, 'last_two_view_ratio': rec['two_view_ratio_to_base'], 'last_four_view_ratio': rec['four_view_ratio_to_base']}), flush=True)

    two_extra_ratio = aux_rows_word_debit_two_views / base_words_total
    four_extra_ratio = aux_rows_word_debit_four_views / base_words_total
    initial_actual = 80011326
    summary = {
        'status': 'FULLCTX_AUX_BUDGET_AUDIT',
        'purpose': 'count unreported full-context auxiliary word-pass exposure in the unchanged research trainer',
        'hashes': hashes,
        'segment': segment,
        'base_words_total': base_words_total,
        'aux_events_total': aux_events_total,
        'event_counts': dict(event_counts),
        'shuffle_level_counts': dict(shuffle_levels),
        'unique_selected_rows': len(row_repeat_counts),
        'selected_row_repeat_summary': summarize([float(v) for v in row_repeat_counts.values()]),
        'aux_debit_if_repaired_two_views_true_and_shuffle': {
            'extra_row_words': aux_rows_word_debit_two_views,
            'extra_seq_tokens': aux_seq_token_debit_two_views,
            'extra_ratio_to_base_words': two_extra_ratio,
            'total_debited_words_for_full_80_to_90_base_segment': base_words_total + aux_rows_word_debit_two_views,
            'actual_total_from_80m_start_if_debited': initial_actual + base_words_total + aux_rows_word_debit_two_views,
            'base_words_allowed_for_10m_debit_at_observed_ratio': int(10_000_000 / (1 + two_extra_ratio)),
        },
        'aux_debit_in_unchanged_trainer_four_views_true_anchor_shuffle_pivotmasked': {
            'extra_row_words': aux_rows_word_debit_four_views,
            'extra_seq_tokens': aux_seq_token_debit_four_views,
            'extra_ratio_to_base_words': four_extra_ratio,
            'total_debited_words_for_full_80_to_90_base_segment': base_words_total + aux_rows_word_debit_four_views,
            'actual_total_from_80m_start_if_debited': initial_actual + base_words_total + aux_rows_word_debit_four_views,
            'base_words_allowed_for_10m_debit_at_observed_ratio': int(10_000_000 / (1 + four_extra_ratio)),
        },
        'per_batch_ratio_summary': {
            'events': summarize([float(r['events']) for r in per_batch]),
            'two_view_ratio_to_base': summarize([float(r['two_view_ratio_to_base']) for r in per_batch]),
            'four_view_ratio_to_base': summarize([float(r['four_view_ratio_to_base']) for r in per_batch]),
        },
        'artifacts': {'summary': str(OUT/'fullctx_aux_budget_audit.json'), 'batch_records': str(batch_records_path), 'note': str(NOTE)},
    }
    (OUT/'fullctx_aux_budget_audit.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    lines = []
    lines.append('# research — full-context auxiliary budget audit\n')
    lines.append('The unchanged research trainer reports only the base WWM stream words. Its auxiliary loss, however, forwards full row contexts for selected events. This audit reuses the same selection and donor-attachment code without loading a model.\n')
    lines.append(f"- Base segment words: {base_words_total}\n")
    lines.append(f"- Auxiliary events: {aux_events_total}; categories: {dict(event_counts)}\n")
    lines.append(f"- Repaired two-view semantic objective (true+shuffle) would add {aux_rows_word_debit_two_views} row-word passes, ratio {two_extra_ratio:.3f} of base, total debit for the full base segment {base_words_total + aux_rows_word_debit_two_views}. Starting from 80,011,326, that would land at {initial_actual + base_words_total + aux_rows_word_debit_two_views}.\n")
    lines.append(f"- Unchanged four-view implementation adds {aux_rows_word_debit_four_views} row-word passes, ratio {four_extra_ratio:.3f} of base, total debit for the full base segment {base_words_total + aux_rows_word_debit_four_views}. Starting from 80,011,326, that would land at {initial_actual + base_words_total + aux_rows_word_debit_four_views}.\n")
    lines.append(f"- To spend only a 10M-word debit after 80M, observed ratios imply at most {summary['aux_debit_if_repaired_two_views_true_and_shuffle']['base_words_allowed_for_10m_debit_at_observed_ratio']} base words for a two-view objective, or {summary['aux_debit_in_unchanged_trainer_four_views_true_anchor_shuffle_pivotmasked']['base_words_allowed_for_10m_debit_at_observed_ratio']} base words for the unchanged four-view trainer.\n")
    lines.append('Consequence: no future relation-auxiliary continuation should call its endpoint 90M by base words alone. The WWM reference and auxiliary branches must share the same debited exposure, and the trainer should compute only the views required by the chosen loss.\n')
    lines.append(f"Files: `{OUT/'fullctx_aux_budget_audit.json'}`, `{batch_records_path}`\n")
    NOTE.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'status': summary['status'], 'summary': str(OUT/'fullctx_aux_budget_audit.json'), 'note': str(NOTE), 'two_view_extra_ratio': two_extra_ratio, 'four_view_extra_ratio': four_extra_ratio, 'two_view_total_debit': base_words_total + aux_rows_word_debit_two_views, 'four_view_total_debit': base_words_total + aux_rows_word_debit_four_views, 'aux_events_total': aux_events_total}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
