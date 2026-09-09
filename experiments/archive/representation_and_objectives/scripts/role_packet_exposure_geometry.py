#!/usr/bin/env python3
"""research: audit how much role-switch/control signal the legal 80M screen actually injects.

This is CPU-only and trains/evaluates nothing. It uses the already-built research
packet suite and research/147 in-place shared-tokenizer corpora to quantify the
credit-assignment geometry of the running 80M screen:

* packet words as a fraction of the 10M/80M stream;
* which optimizer steps contain packet rows under the trusted 256-row effective
  batch grouping;
* target-word exposure and expected WWM target-mask counts;
* contrast with the research disposable packet-only continuation, which used the
  same packet word exposure but with no natural-corpus dilution.

The goal is to make the eventual Route-B readout interpretable without polling
or touching the running H100 tasks.
"""
from __future__ import annotations

import json
import math
import pathlib
import statistics
from collections import Counter, defaultdict
from typing import Any

ROOT = pathlib.Path('.')
PKT_DIR = ROOT / 'experiments/archive/representation_and_objectives/data/role_switch_expanded_packets'
CORPUS_DIR = ROOT / 'experiments/archive/representation_and_objectives/data/inplace_packed_role_switch_replacement_corpus'
OUT_DIR = ROOT / 'experiments/archive/representation_and_objectives/data/role_packet_exposure_geometry'
BATCH_SIZE = 256
MICRO_BATCH_SIZE = 64
WORDS_PER_PASS = 10_000_000
ROWS_PER_PASS_EXPECTED = 64_740
SCREEN_WORDS = 80_000_000
PASSES_IN_80M = 8
MASK_PROB = 0.15
BERT_MASK_TOKEN_FRAC = 0.8
BERT_RANDOM_FRAC = 0.1
BERT_KEEP_FRAC = 0.1
DISPOSABLE_PACKET_BATCH_SIZE = 128
DISPOSABLE_REPEATS = 8


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def norm_word(w: str) -> str:
    return w.strip('.,!?;:"()[]{}').lower()


def target_for_row(row: dict[str, Any], pairs: dict[int, dict[str, Any]], mode: str) -> str:
    pair_id = int(row['pair_id'])
    p = pairs[pair_id]
    if mode == 'treatment':
        return str(p[f"correct_{row['direction']}"])
    if mode == 'role_fixed':
        return str(p['alt_0'] if (pair_id % 2 == 0) else p['alt_1'])
    raise ValueError(mode)


def locate_target_word(row: dict[str, Any], target: str) -> int:
    words = str(row['text']).split()
    target_n = norm_word(target)
    hits = [i for i, w in enumerate(words) if norm_word(w) == target_n]
    if not hits:
        raise RuntimeError(f"target {target!r} not found in row text {row['text']!r}")
    # Consequence target is the last occurrence for all current templates.
    return hits[-1]


def packet_events(short_rows: list[dict[str, Any]], pairs: dict[int, dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    out = []
    cursor = 0
    for short_i, row in enumerate(short_rows):
        wc = int(row.get('word_count', len(str(row['text']).split())))
        if wc != len(str(row['text']).split()):
            raise RuntimeError(f'{mode} short row {short_i} word mismatch')
        target = target_for_row(row, pairs, mode)
        local = locate_target_word(row, target)
        stream_offset = cursor + local
        out.append({
            'short_row_index': short_i,
            'pair_id': int(row['pair_id']),
            'family': row['family'],
            'template_id': row['template_id'],
            'style': row['style'],
            'direction': row['direction'],
            'word_count': wc,
            'target': target,
            'target_local_word_index': local,
            'target_stream_word_offset': stream_offset,
            'packed_row_index': stream_offset // 160,
            'packed_row_word_offset': stream_offset % 160,
        })
        cursor += wc
    return out


def summarize_numeric(xs: list[float]) -> dict[str, Any]:
    if not xs:
        return {'n': 0}
    ys = sorted(float(x) for x in xs)
    def q(p: float) -> float:
        if len(ys) == 1:
            return ys[0]
        pos = p * (len(ys) - 1)
        lo = int(math.floor(pos)); hi = int(math.ceil(pos))
        if lo == hi:
            return ys[lo]
        return ys[lo] * (hi - pos) + ys[hi] * (pos - lo)
    return {
        'n': len(ys),
        'min': ys[0],
        'mean': statistics.mean(ys),
        'median': q(0.5),
        'p90': q(0.9),
        'max': ys[-1],
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    scoring_pairs = {int(p['pair_id']): p for p in read_jsonl(PKT_DIR / 'scoring_pairs.jsonl')}
    treatment_short = read_jsonl(PKT_DIR / 'train_treatment.jsonl')
    fixed_short = read_jsonl(PKT_DIR / 'train_role_fixed_control.jsonl')
    treat10 = read_jsonl(CORPUS_DIR / 'role_switch_inplace_packed_replacement_10M.jsonl')
    fixed10 = read_jsonl(CORPUS_DIR / 'role_fixed_inplace_packed_replacement_10M.jsonl')

    if len(treat10) != ROWS_PER_PASS_EXPECTED or len(fixed10) != ROWS_PER_PASS_EXPECTED:
        raise RuntimeError(f'row count mismatch {len(treat10)} {len(fixed10)}')
    if sum(int(r['words']) for r in treat10) != WORDS_PER_PASS:
        raise RuntimeError('role_switch 10M word count mismatch')
    if sum(int(r['words']) for r in fixed10) != WORDS_PER_PASS:
        raise RuntimeError('role_fixed 10M word count mismatch')

    # Packet row map from corpus rows. In-place builder stored both original row index
    # and packed packet row index on the replacement examples.
    packet_rows_by_arm: dict[str, list[dict[str, Any]]] = {}
    for arm, rows in [('role_switch', treat10), ('role_fixed', fixed10)]:
        packet = []
        for row_index, r in enumerate(rows):
            src = str(r.get('source', ''))
            if src.startswith(f'step146_{arm}_packet_packed160'):
                packet.append({
                    'corpus_row_index': row_index,
                    'packed_row_index': int(r['packed_row_index']),
                    'words': int(r['words']),
                    'source': src,
                    'packet_family_touch_counts': r.get('packet_family_touch_counts', {}),
                })
        packet.sort(key=lambda x: x['packed_row_index'])
        if len(packet) != 138:
            raise RuntimeError(f'{arm}: expected 138 packet rows, found {len(packet)}')
        for j, r in enumerate(packet):
            if r['packed_row_index'] != j or r['words'] != 160:
                raise RuntimeError(f'{arm}: bad packet row order/length at {j}: {r}')
        packet_rows_by_arm[arm] = packet

    # Shared geometry should have identical row positions for both arms.
    switch_positions = [r['corpus_row_index'] for r in packet_rows_by_arm['role_switch']]
    fixed_positions = [r['corpus_row_index'] for r in packet_rows_by_arm['role_fixed']]
    if switch_positions != fixed_positions:
        raise RuntimeError('packet row positions differ between treatment and control')

    events = {
        'role_switch': packet_events(treatment_short, scoring_pairs, 'treatment'),
        'role_fixed': packet_events(fixed_short, scoring_pairs, 'role_fixed'),
    }

    by_arm_summary: dict[str, Any] = {}
    for arm, evs in events.items():
        # Map packed-row index to original corpus row index.
        packed_to_corpus = {r['packed_row_index']: r['corpus_row_index'] for r in packet_rows_by_arm[arm]}
        effective_steps = []
        per_step = defaultdict(lambda: {'packet_rows': 0, 'packet_words': 0, 'target_events': 0, 'families': Counter()})
        target_by_family = Counter()
        by_pass_packet_steps: dict[int, list[int]] = {}
        for pass_id in range(PASSES_IN_80M):
            step_set = set()
            for prow in packet_rows_by_arm[arm]:
                global_row = pass_id * ROWS_PER_PASS_EXPECTED + prow['corpus_row_index']
                step = global_row // BATCH_SIZE + 1
                step_set.add(step)
                rec = per_step[step]
                rec['packet_rows'] += 1
                rec['packet_words'] += prow['words']
            by_pass_packet_steps[pass_id] = sorted(step_set)
            for ev in evs:
                corpus_row = packed_to_corpus[ev['packed_row_index']]
                global_row = pass_id * ROWS_PER_PASS_EXPECTED + corpus_row
                step = global_row // BATCH_SIZE + 1
                per_step[step]['target_events'] += 1
                per_step[step]['families'][ev['family']] += 1
                target_by_family[ev['family']] += 1
                effective_steps.append(step)
        # Convert Counters for JSON.
        per_step_json = {}
        for step, rec in sorted(per_step.items()):
            per_step_json[str(step)] = {
                'packet_rows': rec['packet_rows'],
                'packet_words': rec['packet_words'],
                'target_events': rec['target_events'],
                'families': dict(sorted(rec['families'].items())),
            }
        unique_steps = sorted(per_step)
        packet_words_80m = 22_080 * PASSES_IN_80M
        target_events_80m = len(evs) * PASSES_IN_80M
        expected_masked_target_words = target_events_80m * MASK_PROB
        by_arm_summary[arm] = {
            'short_packet_texts_per_pass': len(evs),
            'packet_rows_per_pass_after_packing': len(packet_rows_by_arm[arm]),
            'packet_words_per_pass': 22_080,
            'packet_words_80M': packet_words_80m,
            'packet_word_fraction_of_stream': packet_words_80m / SCREEN_WORDS,
            'target_word_events_per_pass': len(evs),
            'target_word_events_80M': target_events_80m,
            'target_event_fraction_within_packet_words': len(evs) / 22_080,
            'expected_wwm_masked_target_word_events_80M': expected_masked_target_words,
            'expected_mask_token_replacements': expected_masked_target_words * BERT_MASK_TOKEN_FRAC,
            'expected_random_replacements': expected_masked_target_words * BERT_RANDOM_FRAC,
            'expected_keep_selected_targets': expected_masked_target_words * BERT_KEEP_FRAC,
            'target_events_by_family_80M': dict(sorted(target_by_family.items())),
            'unique_optimizer_steps_with_packet_rows': len(unique_steps),
            'total_optimizer_steps_80M': math.ceil((ROWS_PER_PASS_EXPECTED * PASSES_IN_80M) / BATCH_SIZE),
            'fraction_optimizer_steps_touching_packets': len(unique_steps) / math.ceil((ROWS_PER_PASS_EXPECTED * PASSES_IN_80M) / BATCH_SIZE),
            'packet_rows_per_touched_step': summarize_numeric([rec['packet_rows'] for rec in per_step.values()]),
            'packet_words_per_touched_step': summarize_numeric([rec['packet_words'] for rec in per_step.values()]),
            'target_events_per_touched_step': summarize_numeric([rec['target_events'] for rec in per_step.values()]),
            'packet_step_min_max': [unique_steps[0], unique_steps[-1]],
            'packet_steps_by_pass': by_pass_packet_steps,
            'per_step': per_step_json,
        }

    # Directional treatment/control semantics: role_fixed matches one treatment
    # direction per pair and contradicts the other relative to the exchange rule.
    same_direction = Counter()
    for tr, fx in zip(treatment_short, fixed_short):
        key = (tr['family'], tr['template_id'], tr['pair_id'], tr['direction'])
        if key != (fx['family'], fx['template_id'], fx['pair_id'], fx['direction']):
            raise RuntimeError(f'misaligned short packet rows: {key} vs fixed')
        pair_id = int(tr['pair_id'])
        p = scoring_pairs[pair_id]
        tr_t = target_for_row(tr, scoring_pairs, 'treatment')
        fx_t = target_for_row(fx, scoring_pairs, 'role_fixed')
        same_direction[(str(tr['family']), str(tr['direction']), 'same' if tr_t == fx_t else 'opposite')] += 1

    disposable_updates_per_epoch = math.ceil(len(treatment_short) / DISPOSABLE_PACKET_BATCH_SIZE)
    disposable_total_updates = disposable_updates_per_epoch * DISPOSABLE_REPEATS
    legal_total_steps = by_arm_summary['role_switch']['total_optimizer_steps_80M']
    legal_touched_steps = by_arm_summary['role_switch']['unique_optimizer_steps_with_packet_rows']

    summary = {
        'status': 'ROLE_PACKET_EXPOSURE_GEOMETRY',
        'scientific_question': 'How much and where does the legal from-scratch role-switch/control signal enter the 80M training stream?',
        'screen_context': {
            'running_tasks': ['s147_t30_tool1 role_switch shared-tokenizer 80M', 's147_t31_tool1 role_fixed shared-tokenizer 80M'],
            'this_audit_trains_or_evaluates_models': False,
            'shared_tokenizer_route': 'research shared legal 40k tokenizer trained on common 9,977,920-word intersection; arms differ only in packet rows.'
        },
        'stream_geometry': {
            'rows_per_pass': ROWS_PER_PASS_EXPECTED,
            'words_per_pass': WORDS_PER_PASS,
            'passes_in_80M': PASSES_IN_80M,
            'effective_batch_size_rows': BATCH_SIZE,
            'micro_batch_size_rows': MICRO_BATCH_SIZE,
            'total_effective_steps_80M_by_rows': legal_total_steps,
            'packet_row_positions_first_last': [switch_positions[0], switch_positions[-1]],
            'packet_row_positions_are_identical_across_arms': switch_positions == fixed_positions,
        },
        'by_arm': by_arm_summary,
        'treatment_vs_role_fixed_short_row_target_alignment': dict(sorted((f'{fam}|{direction}|{kind}', n) for (fam, direction, kind), n in same_direction.items())),
        'disposable_packet_only_contrast': {
            'disposable_repeats': DISPOSABLE_REPEATS,
            'disposable_unique_packet_words': 22_080,
            'disposable_packet_word_exposure': 22_080 * DISPOSABLE_REPEATS,
            'disposable_batch_size_rows': DISPOSABLE_PACKET_BATCH_SIZE,
            'disposable_updates_per_epoch': disposable_updates_per_epoch,
            'disposable_total_packet_only_updates': disposable_total_updates,
            'legal_screen_packet_word_exposure_same_as_disposable': (22_080 * DISPOSABLE_REPEATS) == by_arm_summary['role_switch']['packet_words_80M'],
            'legal_screen_packet_touched_updates': legal_touched_steps,
            'legal_screen_total_updates_80M': legal_total_steps,
            'legal_packet_updates_are_diluted_by_natural_rows': True,
            'packet_only_updates_per_legal_touched_update_ratio': disposable_total_updates / max(1, legal_touched_steps),
        },
        'interpretation': {
            'main_fact': 'The 80M legal screen gives exactly the same packet word exposure as the research disposable screen (176,640 words) but distributes it as a tiny fraction of the full MLM stream.',
            'credit_geometry': 'Only a small set of 256-row optimizer updates contains any packet rows; within those touched updates the packet rows are mixed with ordinary corpus rows. Thus a flat natural readout would not contradict synthetic learnability, but it would reject this low-dose sparse-primary-text insertion as a SOTA-relevant route.',
            'control_semantics': 'The role-fixed control is not a no-information baseline: for each pair it agrees with one treatment direction and trains the opposite target in the other direction, while preserving target counts. Role-switch-minus-role-fixed is therefore the exchange-specific signal.'
        }
    }
    out_path = OUT_DIR / 'role_packet_exposure_geometry_summary.json'
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({
        'event': 'ROLE_PACKET_EXPOSURE_GEOMETRY_DONE',
        'summary': str(out_path),
        'packet_word_fraction': by_arm_summary['role_switch']['packet_word_fraction_of_stream'],
        'target_events_80M': by_arm_summary['role_switch']['target_word_events_80M'],
        'expected_masked_targets_80M': by_arm_summary['role_switch']['expected_wwm_masked_target_word_events_80M'],
        'touched_steps': legal_touched_steps,
        'total_steps_80M': legal_total_steps,
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
