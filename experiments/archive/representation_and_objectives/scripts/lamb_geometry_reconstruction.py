#!/usr/bin/env python3
"""research: reconstruct the actual data/masking geometry of the research LAMB runs.

The research trainer is allowed to finish, but its results should be interpreted only
through the geometry it actually used.  This script mirrors the research
CurriculumChunkedDataset and WWM sampler and compares it with the matched fixed-256
legal40k AdamW coordinate.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer


def parse_curriculum(spec: str) -> list[tuple[int, int]]:
    phases: list[tuple[int, int]] = []
    for part in spec.split(','):
        left, right = part.split(':')
        phases.append((int(left), int(right.replace('M', '000000').replace('K', '000'))))
    return phases


def pct(x: float) -> float:
    return round(100.0 * x, 6)


class TokenGeometry:
    def __init__(self, tokenizer_path: str):
        self.tok = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
        self.special_ids = set(int(x) for x in self.tok.all_special_ids)
        self._start_cache: dict[int, bool] = {}
        self._tok_cache: dict[int, str] = {}

    def token_str(self, tid: int) -> str:
        v = self._tok_cache.get(tid)
        if v is None:
            v = str(self.tok.convert_ids_to_tokens(int(tid)))
            self._tok_cache[tid] = v
        return v

    def is_word_start(self, tid: int) -> bool:
        v = self._start_cache.get(tid)
        if v is None:
            s = self.token_str(tid)
            v = bool(s.startswith('Ġ') or s.startswith('▁'))
            self._start_cache[tid] = v
        return v

    def encode(self, text: str) -> list[int]:
        return list(self.tok(text, add_special_tokens=False, truncation=False)['input_ids'])

    def group_ids(self, ids: list[int]) -> list[int]:
        gids: list[int] = []
        gid = -1
        for i, tid in enumerate(ids):
            if int(tid) in self.special_ids:
                gids.append(-1)
                continue
            if gid < 0 or self.is_word_start(int(tid)) or i == 0:
                gid += 1
            gids.append(gid)
        return gids

    def group_count_for_span(self, ids: list[int], start: int, end: int) -> int:
        gid = -1
        for j, tid in enumerate(ids[start:end]):
            if int(tid) in self.special_ids:
                continue
            if gid < 0 or self.is_word_start(int(tid)) or j == 0:
                gid += 1
        return gid + 1


def make_phase_state(phases: list[tuple[int, int]]) -> dict[str, Any]:
    return {'phases': phases, 'phase_idx': 0, 'cumulative_before': 0}


def assign_phase(state: dict[str, Any]) -> int:
    phases = state['phases']
    if state['phase_idx'] < len(phases) - 1 and state['cumulative_before'] >= phases[state['phase_idx']][1]:
        state['phase_idx'] += 1
    return state['phase_idx']


def blank_phase(seq_len: int, target_budget: int) -> dict[str, Any]:
    return {
        'seq_len': seq_len,
        'target_budget_words': target_budget,
        'rows': 0,
        'row_words': 0,
        'trainer_charged_words': 0,
        'rows_token_lt8': 0,
        'chunks_kept': 0,
        'chunks_dropped_tail': 0,
        'rows_with_dropped_tail': 0,
        'dropped_tail_tokens': 0,
        'kept_tokens': 0,
        'full_row_tokens': 0,
        'kept_original_groups': 0,
        'trainer_word_groups': 0,
        'midword_chunk_boundaries': 0,
        'rows_with_midword_boundary': 0,
        'expected_lamb_mask_groups': 0.0,
        'expected_lamb_mask_tokens': 0.0,
        'expected_bernoulli_mask_groups_same_chunks': 0.0,
        'expected_bernoulli_mask_tokens_same_chunks': 0.0,
        'max_chunks_per_row': 0,
        'chunk_count_hist': Counter(),
        'tail_len_hist': Counter(),
        'midword_token_counter': Counter(),
        'row_token_len_p': [],
        'chunk_len_p': [],
        'group_count_p': [],
    }


def summarize_percentiles(vals: list[int | float]) -> dict[str, float] | None:
    if not vals:
        return None
    arr = np.asarray(vals, dtype=float)
    return {
        'mean': float(arr.mean()),
        'p05': float(np.percentile(arr, 5)),
        'p25': float(np.percentile(arr, 25)),
        'p50': float(np.percentile(arr, 50)),
        'p75': float(np.percentile(arr, 75)),
        'p95': float(np.percentile(arr, 95)),
        'max': float(arr.max()),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--data-path', required=True)
    p.add_argument('--tokenizer-path', required=True)
    p.add_argument('--curriculum', default='64:20000000,128:50000000,256:100000000')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--mask-prob', type=float, default=0.15)
    p.add_argument('--min-chunk-tokens', type=int, default=8)
    p.add_argument('--baseline-seq-len', type=int, default=256)
    p.add_argument('--max-rows', type=int, default=0)
    p.add_argument('--out-dir', required=True)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    phases = parse_curriculum(args.curriculum)
    geom = TokenGeometry(args.tokenizer_path)
    phase_state = make_phase_state(phases)
    phase_stats = [blank_phase(sl, budget) for sl, budget in phases]

    baseline = {
        'seq_len': args.baseline_seq_len,
        'rows': 0,
        'words': 0,
        'candidate_tokens': 0,
        'word_groups': 0,
        'rows_truncated': 0,
        'truncated_tokens': 0,
        'truncation_midword_cut_rows': 0,
        'expected_bernoulli_mask_tokens': 0.0,
        'expected_bernoulli_mask_groups': 0.0,
        'row_token_len_p': [],
        'visible_token_len_p': [],
        'visible_group_count_p': [],
    }

    global_counts = Counter()
    samples_midword: list[dict[str, Any]] = []
    samples_tail: list[dict[str, Any]] = []
    t0 = time.time()

    data_path = Path(args.data_path)
    with data_path.open('r', encoding='utf-8') as f:
        for row_i, line in enumerate(f):
            if args.max_rows and row_i >= args.max_rows:
                break
            if not line.strip():
                continue
            obj = json.loads(line)
            text = str(obj['text'])
            words = int(obj.get('words', len(text.split())))
            actual_words = len(text.split())
            if words != actual_words:
                raise RuntimeError(f'row {row_i} words field {words} != split {actual_words}')
            ph_i = assign_phase(phase_state)
            phase_state['cumulative_before'] += words
            st = phase_stats[ph_i]
            st['rows'] += 1
            st['row_words'] += words

            ids = geom.encode(text)
            n = len(ids)
            gids = geom.group_ids(ids)
            st['full_row_tokens'] += n
            st['row_token_len_p'].append(n)
            baseline['rows'] += 1
            baseline['words'] += words
            baseline['row_token_len_p'].append(n)
            visible_n = min(n, args.baseline_seq_len)
            baseline['candidate_tokens'] += visible_n
            baseline['visible_token_len_p'].append(visible_n)
            vis_g = len({g for g in gids[:visible_n] if g >= 0})
            baseline['word_groups'] += vis_g
            baseline['visible_group_count_p'].append(vis_g)
            baseline['expected_bernoulli_mask_tokens'] += args.mask_prob * visible_n
            baseline['expected_bernoulli_mask_groups'] += args.mask_prob * vis_g
            if n > args.baseline_seq_len:
                baseline['rows_truncated'] += 1
                baseline['truncated_tokens'] += n - args.baseline_seq_len
                if args.baseline_seq_len > 0 and gids[args.baseline_seq_len] == gids[args.baseline_seq_len - 1]:
                    baseline['truncation_midword_cut_rows'] += 1

            if n < args.min_chunk_tokens:
                st['rows_token_lt8'] += 1
                global_counts['rows_token_lt8'] += 1
                continue

            row_chunks = 0
            row_mid = False
            row_tail_dropped = False
            kept_original_groups = set()
            for start in range(0, n, st['seq_len']):
                end = min(n, start + st['seq_len'])
                clen = end - start
                if clen < args.min_chunk_tokens:
                    st['chunks_dropped_tail'] += 1
                    st['dropped_tail_tokens'] += clen
                    st['tail_len_hist'][clen] += 1
                    row_tail_dropped = True
                    if len(samples_tail) < 40:
                        samples_tail.append({
                            'row_index': row_i,
                            'phase': ph_i,
                            'seq_len': st['seq_len'],
                            'tail_start_token': start,
                            'tail_len': clen,
                            'row_words': words,
                            'row_tokens': n,
                            'tail_token_strings': [geom.token_str(int(x)) for x in ids[start:end]],
                            'text_prefix': text[:220],
                        })
                    continue
                row_chunks += 1
                st['chunks_kept'] += 1
                st['kept_tokens'] += clen
                st['chunk_len_p'].append(clen)
                for g in gids[start:end]:
                    if g >= 0:
                        kept_original_groups.add(g)
                if start == 0:
                    st['trainer_charged_words'] += words
                if start > 0 and gids[start] >= 0 and gids[start] == gids[start - 1]:
                    st['midword_chunk_boundaries'] += 1
                    row_mid = True
                    tok_here = geom.token_str(int(ids[start]))
                    st['midword_token_counter'][tok_here] += 1
                    if len(samples_midword) < 80:
                        samples_midword.append({
                            'row_index': row_i,
                            'phase': ph_i,
                            'seq_len': st['seq_len'],
                            'boundary_token_index': start,
                            'previous_token': geom.token_str(int(ids[start - 1])),
                            'first_token_after_boundary': tok_here,
                            'row_words': words,
                            'row_tokens': n,
                            'text_prefix': text[:260],
                        })
                trainer_g = geom.group_count_for_span(ids, start, end)
                st['trainer_word_groups'] += trainer_g
                st['group_count_p'].append(trainer_g)
                if trainer_g > 0:
                    n_mask = max(1, round(trainer_g * args.mask_prob))
                    st['expected_lamb_mask_groups'] += n_mask
                    st['expected_lamb_mask_tokens'] += n_mask * (clen / trainer_g)
                    st['expected_bernoulli_mask_groups_same_chunks'] += args.mask_prob * trainer_g
                    st['expected_bernoulli_mask_tokens_same_chunks'] += args.mask_prob * clen
            st['kept_original_groups'] += len(kept_original_groups)
            st['max_chunks_per_row'] = max(st['max_chunks_per_row'], row_chunks)
            st['chunk_count_hist'][row_chunks] += 1
            if row_mid:
                st['rows_with_midword_boundary'] += 1
            if row_tail_dropped:
                st['rows_with_dropped_tail'] += 1

            if (row_i + 1) % 50000 == 0:
                print(json.dumps({
                    'event': 'geometry_progress',
                    'rows': row_i + 1,
                    'words': phase_state['cumulative_before'],
                    'elapsed_sec': round(time.time() - t0, 1),
                }), flush=True)

    # Finalize phase summaries and write CSV.
    phase_rows = []
    for i, st in enumerate(phase_stats):
        steps = math.ceil(st['chunks_kept'] / args.batch_size) if st['chunks_kept'] else 0
        group_over = st['trainer_word_groups'] - st['kept_original_groups']
        row = {
            'phase': i,
            'seq_len': st['seq_len'],
            'target_budget_words': st['target_budget_words'],
            'rows': st['rows'],
            'row_words': st['row_words'],
            'trainer_charged_words': st['trainer_charged_words'],
            'rows_token_lt8': st['rows_token_lt8'],
            'chunks_kept': st['chunks_kept'],
            'optimizer_steps': steps,
            'words_per_step': st['trainer_charged_words'] / steps if steps else None,
            'kept_tokens': st['kept_tokens'],
            'full_row_tokens': st['full_row_tokens'],
            'visible_token_fraction': st['kept_tokens'] / st['full_row_tokens'] if st['full_row_tokens'] else None,
            'dropped_tail_tokens': st['dropped_tail_tokens'],
            'rows_with_dropped_tail': st['rows_with_dropped_tail'],
            'dropped_tail_token_fraction': st['dropped_tail_tokens'] / st['full_row_tokens'] if st['full_row_tokens'] else None,
            'midword_chunk_boundaries': st['midword_chunk_boundaries'],
            'rows_with_midword_boundary': st['rows_with_midword_boundary'],
            'rows_with_midword_boundary_frac': st['rows_with_midword_boundary'] / st['rows'] if st['rows'] else None,
            'kept_original_groups': st['kept_original_groups'],
            'trainer_word_groups': st['trainer_word_groups'],
            'group_overcount_from_midword_resets': group_over,
            'group_overcount_frac_of_kept_original': group_over / st['kept_original_groups'] if st['kept_original_groups'] else None,
            'expected_lamb_mask_groups': st['expected_lamb_mask_groups'],
            'expected_lamb_mask_tokens': st['expected_lamb_mask_tokens'],
            'expected_bernoulli_mask_groups_same_chunks': st['expected_bernoulli_mask_groups_same_chunks'],
            'expected_bernoulli_mask_tokens_same_chunks': st['expected_bernoulli_mask_tokens_same_chunks'],
            'lamb_round_min_mask_token_ratio_vs_bernoulli_same_chunks': st['expected_lamb_mask_tokens'] / st['expected_bernoulli_mask_tokens_same_chunks'] if st['expected_bernoulli_mask_tokens_same_chunks'] else None,
            'tokens_per_step': st['kept_tokens'] / steps if steps else None,
            'expected_lamb_mask_tokens_per_step': st['expected_lamb_mask_tokens'] / steps if steps else None,
            'chunks_per_row_mean': st['chunks_kept'] / st['rows'] if st['rows'] else None,
            'max_chunks_per_row': st['max_chunks_per_row'],
            'row_token_len_percentiles': summarize_percentiles(st['row_token_len_p']),
            'chunk_len_percentiles': summarize_percentiles(st['chunk_len_p']),
            'trainer_group_count_percentiles': summarize_percentiles(st['group_count_p']),
            'chunk_count_hist': {str(k): v for k, v in sorted(st['chunk_count_hist'].items())},
            'tail_len_hist': {str(k): v for k, v in sorted(st['tail_len_hist'].items())},
            'top_midword_start_tokens': st['midword_token_counter'].most_common(25),
        }
        phase_rows.append(row)

    total_lamb_steps = sum(r['optimizer_steps'] for r in phase_rows)
    total_chunks = sum(r['chunks_kept'] for r in phase_rows)
    total_words = sum(r['trainer_charged_words'] for r in phase_rows)
    total_kept_tokens = sum(r['kept_tokens'] for r in phase_rows)
    total_full_tokens = sum(r['full_row_tokens'] for r in phase_rows)
    total_mid = sum(r['midword_chunk_boundaries'] for r in phase_rows)
    total_rows_mid = sum(r['rows_with_midword_boundary'] for r in phase_rows)
    total_rows = sum(r['rows'] for r in phase_rows)
    total_tail = sum(r['dropped_tail_tokens'] for r in phase_rows)
    total_groups_trainer = sum(r['trainer_word_groups'] for r in phase_rows)
    total_groups_original = sum(r['kept_original_groups'] for r in phase_rows)
    total_lamb_mask_tok = sum(r['expected_lamb_mask_tokens'] for r in phase_rows)
    total_samechunk_bernoulli_mask_tok = sum(r['expected_bernoulli_mask_tokens_same_chunks'] for r in phase_rows)

    baseline['optimizer_steps'] = math.ceil(baseline['rows'] / args.batch_size) if baseline['rows'] else 0
    baseline['tokens_per_step'] = baseline['candidate_tokens'] / baseline['optimizer_steps'] if baseline['optimizer_steps'] else None
    baseline['words_per_step'] = baseline['words'] / baseline['optimizer_steps'] if baseline['optimizer_steps'] else None
    baseline['expected_bernoulli_mask_tokens_per_step'] = baseline['expected_bernoulli_mask_tokens'] / baseline['optimizer_steps'] if baseline['optimizer_steps'] else None
    baseline['row_token_len_percentiles'] = summarize_percentiles(baseline.pop('row_token_len_p'))
    baseline['visible_token_len_percentiles'] = summarize_percentiles(baseline.pop('visible_token_len_p'))
    baseline['visible_group_count_percentiles'] = summarize_percentiles(baseline.pop('visible_group_count_p'))

    totals = {
        'data_path': args.data_path,
        'tokenizer_path': args.tokenizer_path,
        'curriculum': args.curriculum,
        'batch_size_chunks': args.batch_size,
        'mask_prob': args.mask_prob,
        'min_chunk_tokens': args.min_chunk_tokens,
        'rows': total_rows,
        'trainer_charged_words': total_words,
        'chunks_kept': total_chunks,
        'optimizer_steps': total_lamb_steps,
        'baseline_fixed256_optimizer_steps': baseline['optimizer_steps'],
        'optimizer_step_ratio_lamb_curriculum_vs_fixed256': total_lamb_steps / baseline['optimizer_steps'] if baseline['optimizer_steps'] else None,
        'kept_tokens': total_kept_tokens,
        'full_row_tokens': total_full_tokens,
        'visible_token_fraction': total_kept_tokens / total_full_tokens if total_full_tokens else None,
        'baseline_fixed256_candidate_tokens': baseline['candidate_tokens'],
        'candidate_token_ratio_vs_fixed256': total_kept_tokens / baseline['candidate_tokens'] if baseline['candidate_tokens'] else None,
        'midword_chunk_boundaries': total_mid,
        'rows_with_midword_boundary': total_rows_mid,
        'rows_with_midword_boundary_frac': total_rows_mid / total_rows if total_rows else None,
        'dropped_tail_tokens': total_tail,
        'dropped_tail_token_fraction': total_tail / total_full_tokens if total_full_tokens else None,
        'trainer_word_groups': total_groups_trainer,
        'kept_original_groups': total_groups_original,
        'group_overcount_from_midword_resets': total_groups_trainer - total_groups_original,
        'group_overcount_frac_of_kept_original': (total_groups_trainer - total_groups_original) / total_groups_original if total_groups_original else None,
        'expected_lamb_mask_tokens': total_lamb_mask_tok,
        'expected_bernoulli_mask_tokens_same_chunks': total_samechunk_bernoulli_mask_tok,
        'lamb_round_min_mask_token_ratio_vs_bernoulli_same_chunks': total_lamb_mask_tok / total_samechunk_bernoulli_mask_tok if total_samechunk_bernoulli_mask_tok else None,
        'baseline_expected_bernoulli_mask_tokens': baseline['expected_bernoulli_mask_tokens'],
        'expected_mask_token_ratio_lamb_curriculum_vs_fixed256_bernoulli': total_lamb_mask_tok / baseline['expected_bernoulli_mask_tokens'] if baseline['expected_bernoulli_mask_tokens'] else None,
        'elapsed_sec': round(time.time() - t0, 1),
    }

    summary = {
        'status': 'LAMB_GEOMETRY_RECONSTRUCTION',
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'purpose': 'Reconstruct the actual research LAMB×sequence-curriculum exposure, chunking, and masking geometry before interpreting downstream scores.',
        'totals': totals,
        'phases': phase_rows,
        'matched_fixed256_reference_geometry': baseline,
        'interpretation': {
            'joint_intervention': 'research changes optimizer, length schedule, chunk construction, exact-per-chunk WWM selection, update count, candidate-token exposure, and token batch geometry together.',
            'not_word_boundary_chunked': True,
            'midword_reset_meaning': 'When a chunk starts inside a wordpiece group, research treats that continuation piece as a new WWM group in the new chunk.',
            'tail_drop_meaning': 'Final token tails shorter than min_chunk_tokens are not fed to the model, while the row word count is charged at the first chunk.',
            'cheap7_readout': 'Compare any research cheap7 vector to matched cheap7 vectors for legal40k fixed-256 AdamW baselines and the visible leader, not to nine-column Overall scores.',
        },
        'sample_files': {
            'midword_boundaries': str(out_dir / 'midword_boundary_samples.jsonl'),
            'dropped_tails': str(out_dir / 'dropped_tail_samples.jsonl'),
            'phase_csv': str(out_dir / 'phase_geometry.csv'),
        },
    }

    summary_path = out_dir / 'lamb_geometry_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    with (out_dir / 'midword_boundary_samples.jsonl').open('w', encoding='utf-8') as f:
        for rec in samples_midword:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    with (out_dir / 'dropped_tail_samples.jsonl').open('w', encoding='utf-8') as f:
        for rec in samples_tail:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    csv_cols = [
        'phase','seq_len','rows','row_words','trainer_charged_words','chunks_kept','optimizer_steps','words_per_step',
        'kept_tokens','full_row_tokens','visible_token_fraction','dropped_tail_tokens','rows_with_dropped_tail',
        'midword_chunk_boundaries','rows_with_midword_boundary','rows_with_midword_boundary_frac',
        'kept_original_groups','trainer_word_groups','group_overcount_from_midword_resets','group_overcount_frac_of_kept_original',
        'expected_lamb_mask_tokens','expected_bernoulli_mask_tokens_same_chunks','lamb_round_min_mask_token_ratio_vs_bernoulli_same_chunks',
        'tokens_per_step','expected_lamb_mask_tokens_per_step','chunks_per_row_mean','max_chunks_per_row'
    ]
    with (out_dir / 'phase_geometry.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=csv_cols)
        w.writeheader()
        for row in phase_rows:
            w.writerow({k: row.get(k) for k in csv_cols})

    note = out_dir.parent.parent / 'notes' / 'lamb_geometry_reconstruction.md'
    note.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append('# research — LAMB×sequence-curriculum geometry reconstruction')
    lines.append('')
    lines.append(f"Data: `{args.data_path}`")
    lines.append(f"Tokenizer: `{args.tokenizer_path}`")
    lines.append('')
    lines.append('## Totals')
    lines.append(f"- rows: {totals['rows']:,}; charged words: {totals['trainer_charged_words']:,}")
    lines.append(f"- kept chunks: {totals['chunks_kept']:,}; optimizer steps: {totals['optimizer_steps']:,} vs fixed-256 matched reference {baseline['optimizer_steps']:,} (ratio {totals['optimizer_step_ratio_lamb_curriculum_vs_fixed256']:.3f})")
    lines.append(f"- candidate tokens kept: {totals['kept_tokens']:,}; full tokenized stream: {totals['full_row_tokens']:,}; visible fraction {totals['visible_token_fraction']:.6f}; candidate-token ratio vs fixed-256 reference {totals['candidate_token_ratio_vs_fixed256']:.3f}")
    lines.append(f"- mid-word chunk boundaries: {totals['midword_chunk_boundaries']:,} in {totals['rows_with_midword_boundary']:,} rows ({pct(totals['rows_with_midword_boundary_frac']):.4f}% of rows)")
    lines.append(f"- dropped <{args.min_chunk_tokens}-token tails: {totals['dropped_tail_tokens']:,} tokens ({pct(totals['dropped_tail_token_fraction']):.6f}% of full tokens)")
    lines.append(f"- WWM groups after per-chunk restart: {totals['trainer_word_groups']:,}; original kept groups: {totals['kept_original_groups']:,}; overcount {totals['group_overcount_from_midword_resets']:,} ({pct(totals['group_overcount_frac_of_kept_original']):.4f}% of kept original groups)")
    lines.append(f"- expected masked tokens under research exact per-chunk WWM: {totals['expected_lamb_mask_tokens']:.1f}; same chunks with Bernoulli WWM would be {totals['expected_bernoulli_mask_tokens_same_chunks']:.1f} (ratio {totals['lamb_round_min_mask_token_ratio_vs_bernoulli_same_chunks']:.4f})")
    lines.append(f"- expected masked-token ratio versus fixed-256 Bernoulli reference: {totals['expected_mask_token_ratio_lamb_curriculum_vs_fixed256_bernoulli']:.3f}")
    lines.append('')
    lines.append('## Per phase')
    lines.append('| phase | seq_len | words | chunks | steps | words/step | tokens/step | mid-word boundaries | tail tokens | expected masked tokens/step |')
    lines.append('|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for r in phase_rows:
        lines.append(f"| {r['phase']} | {r['seq_len']} | {r['trainer_charged_words']:,} | {r['chunks_kept']:,} | {r['optimizer_steps']:,} | {r['words_per_step']:.1f} | {r['tokens_per_step']:.1f} | {r['midword_chunk_boundaries']:,} | {r['dropped_tail_tokens']:,} | {r['expected_lamb_mask_tokens_per_step']:.1f} |")
    lines.append('')
    lines.append('## Interpretation for research readout')
    lines.append('- The active runs are a joint LAMB×sequence-curriculum×chunking×masking intervention, not a separated optimizer or curriculum result.')
    lines.append('- The trainer uses fixed token slicing. A chunk can begin inside a wordpiece group and then restarts that group for WWM in the new chunk.')
    lines.append('- Tails shorter than eight tokens are not observed by the model, while row words are charged when the first chunk is consumed.')
    lines.append('- Because rows produce multiple chunks at shorter lengths, the optimizer update count and token batch geometry differ substantially from the matched fixed-256 AdamW baselines.')
    lines.append('- Any research cheap7 score should be compared to matched cheap7 vectors, not to nine-column Overall values.')
    lines.append('')
    lines.append(f"Summary JSON: `{summary_path}`")
    lines.append(f"Phase CSV: `{out_dir / 'phase_geometry.csv'}`")
    note.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    print(json.dumps({'status': summary['status'], 'summary': str(summary_path), 'note': str(note), 'elapsed_sec': totals['elapsed_sec']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
