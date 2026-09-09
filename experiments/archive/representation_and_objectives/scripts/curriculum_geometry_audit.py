#!/usr/bin/env python3
"""research: geometry/accounting audit for the repaired research curriculum trainer.

This is a CPU-only pre-computation using the exact train JSONL, tokenizer and
chunking code from curriculum_trainer.py. It reconstructs the 64->128->256
phase partition, counts chunks/steps/candidate tokens, and compares against the
fixed-256 legal40k baseline geometry. It does not train or evaluate a model.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path('.').resolve()
SCRIPTS = ROOT / 'experiments/archive/representation_and_objectives/scripts'
COMPACT_EXPERIENCE = ROOT / 'experiments/archive/compact_experience/scripts'
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(COMPACT_EXPERIENCE) not in sys.path:
    sys.path.insert(0, str(COMPACT_EXPERIENCE))

import curriculum_trainer as cur  # noqa: E402
import masking_curriculum_trainer as base  # noqa: E402


def pct(vals, q: float) -> float:
    if not vals:
        return 0.0
    vals = sorted(vals)
    if len(vals) == 1:
        return float(vals[0])
    x = (len(vals) - 1) * q
    lo = int(math.floor(x)); hi = int(math.ceil(x))
    if lo == hi:
        return float(vals[lo])
    return float(vals[lo] * (hi - x) + vals[hi] * (x - lo))


def summarize_lens(lens: list[int]) -> dict[str, float | int]:
    return {
        'n': len(lens),
        'mean': statistics.fmean(lens) if lens else 0.0,
        'median': pct(lens, 0.5),
        'p05': pct(lens, 0.05),
        'p25': pct(lens, 0.25),
        'p75': pct(lens, 0.75),
        'p95': pct(lens, 0.95),
        'min': min(lens) if lens else 0,
        'max': max(lens) if lens else 0,
    }


def sha_text(p: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--example_jsonl', required=True)
    ap.add_argument('--tokenizer_path', required=True)
    ap.add_argument('--max_word_exposure', type=int, default=100_000_000)
    ap.add_argument('--curriculum', default='64:20000000,128:50000000,256:100000000')
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--max_seq_length', type=int, default=256)
    ap.add_argument('--min_chunk_tokens', type=int, default=1)
    ap.add_argument('--output_dir', required=True)
    args = ap.parse_args()

    t0 = time.time()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_path = Path(args.example_jsonl)
    tok = base.make_portable_tokenizer(args.tokenizer_path)
    examples, total_words, total_rows, sample_rows = base.load_examples_jsonl(train_path, args.max_word_exposure)
    actual_words = sum(ex.words for ex in examples)
    print(json.dumps({'event': 'loaded', 'rows': len(examples), 'actual_words': actual_words, 'total_file_words': total_words, 'sec': round(time.time()-t0, 1)}), flush=True)

    t_pre = time.time()
    rows = cur.pre_tokenize_all(examples, tok, args.max_seq_length, progress_every=100000)
    print(json.dumps({'event': 'pretokenized', 'rows': len(rows), 'sec': round(time.time()-t_pre, 1)}), flush=True)

    curriculum = cur.parse_curriculum(args.curriculum)
    phase_summaries: list[dict[str, Any]] = []
    row_idx = 0
    cumulative_words = 0
    total_chunks = 0
    total_candidate_tokens = 0
    total_capacity_tokens = 0
    total_tail_rows_losing_tokens = 0
    total_tail_tokens_lost = 0
    source_words: dict[str, int] = {}

    for phase_i, (seq_len, word_budget) in enumerate(curriculum):
        phase_rows_start = row_idx
        phase_words = 0
        chunk_lens: list[int] = []
        chunks_per_row: list[int] = []
        phase_candidate_tokens = 0
        phase_capacity_tokens = 0
        split_rows = 0
        long_rows = 0
        tail_rows_losing_tokens = 0
        tail_tokens_lost = 0
        rows_with_short_tail = 0
        n_rows = 0
        source_phase_words: dict[str, int] = {}

        while row_idx < len(rows) and cumulative_words < word_budget:
            row = rows[row_idx]
            ex = examples[row_idx]
            row_chunks = cur.chunk_row(row, seq_len, tok.pad_token_id, args.min_chunk_tokens)
            lens = [int(c['attention_mask'].sum().item()) for c in row_chunks]
            chunk_lens.extend(lens)
            chunks_per_row.append(len(lens))
            phase_candidate_tokens += sum(lens)
            phase_capacity_tokens += len(lens) * seq_len
            if len(lens) > 1:
                split_rows += 1
            if row.n_tokens > seq_len:
                long_rows += 1
            if sum(lens) < row.n_tokens:
                tail_rows_losing_tokens += 1
                tail_tokens_lost += row.n_tokens - sum(lens)
            if lens and lens[-1] < 8 and len(lens) > 1:
                rows_with_short_tail += 1
            phase_words += row.words
            source_phase_words[ex.source] = source_phase_words.get(ex.source, 0) + ex.words
            source_words[ex.source] = source_words.get(ex.source, 0) + ex.words
            cumulative_words += row.words
            row_idx += 1
            n_rows += 1

        n_chunks = len(chunk_lens)
        n_steps = math.ceil(n_chunks / args.batch_size)
        total_chunks += n_chunks
        total_candidate_tokens += phase_candidate_tokens
        total_capacity_tokens += phase_capacity_tokens
        total_tail_rows_losing_tokens += tail_rows_losing_tokens
        total_tail_tokens_lost += tail_tokens_lost
        summary = {
            'phase': phase_i,
            'seq_len': seq_len,
            'word_budget': word_budget,
            'rows_start': phase_rows_start,
            'rows_end': row_idx,
            'n_rows': n_rows,
            'phase_words': phase_words,
            'n_chunks': n_chunks,
            'n_steps': n_steps,
            'chunk_len_summary': summarize_lens(chunk_lens),
            'chunks_per_row_summary': summarize_lens(chunks_per_row),
            'split_rows': split_rows,
            'long_rows': long_rows,
            'rows_with_sub8_noninitial_tail': rows_with_short_tail,
            'tail_rows_losing_tokens': tail_rows_losing_tokens,
            'tail_tokens_lost': tail_tokens_lost,
            'candidate_tokens': phase_candidate_tokens,
            'capacity_tokens': phase_capacity_tokens,
            'mean_candidate_tokens_per_step': phase_candidate_tokens / max(1, n_steps),
            'capacity_utilization': phase_candidate_tokens / max(1, phase_capacity_tokens),
            'source_words': source_phase_words,
        }
        phase_summaries.append(summary)
        print(json.dumps({'event': 'phase_done', **{k: summary[k] for k in ['phase','seq_len','n_rows','phase_words','n_chunks','n_steps','candidate_tokens','capacity_utilization','tail_tokens_lost']}}), flush=True)

    fixed_chunks = len(rows)
    fixed_steps = math.ceil(fixed_chunks / args.batch_size)
    fixed_candidate = sum(r.n_tokens for r in rows)
    fixed_capacity = fixed_chunks * args.max_seq_length

    manifest = {
        'status': 'CURRICULUM_GEOMETRY_AUDIT',
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'inputs': {
            'example_jsonl': str(train_path),
            'example_jsonl_sha256': sha_text(train_path),
            'tokenizer_path': args.tokenizer_path,
            'max_word_exposure': args.max_word_exposure,
            'actual_words': actual_words,
            'rows': len(examples),
            'total_file_words': total_words,
            'total_file_rows': total_rows,
            'curriculum': curriculum,
            'batch_size': args.batch_size,
            'max_seq_length': args.max_seq_length,
            'min_chunk_tokens': args.min_chunk_tokens,
        },
        'curriculum_geometry': {
            'total_steps': sum(p['n_steps'] for p in phase_summaries),
            'total_chunks': total_chunks,
            'total_candidate_tokens': total_candidate_tokens,
            'total_capacity_tokens': total_capacity_tokens,
            'capacity_utilization': total_candidate_tokens / max(1, total_capacity_tokens),
            'mean_candidate_tokens_per_step': total_candidate_tokens / max(1, sum(p['n_steps'] for p in phase_summaries)),
            'tail_rows_losing_tokens': total_tail_rows_losing_tokens,
            'tail_tokens_lost': total_tail_tokens_lost,
            'phase_summary': phase_summaries,
        },
        'fixed256_geometry_same_pretokenized_rows': {
            'total_steps': fixed_steps,
            'total_chunks': fixed_chunks,
            'total_candidate_tokens': fixed_candidate,
            'total_capacity_tokens': fixed_capacity,
            'capacity_utilization': fixed_candidate / max(1, fixed_capacity),
            'mean_candidate_tokens_per_step': fixed_candidate / max(1, fixed_steps),
        },
        'ratios_curriculum_vs_fixed256': {
            'steps': sum(p['n_steps'] for p in phase_summaries) / fixed_steps,
            'chunks': total_chunks / fixed_chunks,
            'candidate_tokens': total_candidate_tokens / fixed_candidate,
            'capacity_tokens': total_capacity_tokens / fixed_capacity,
            'mean_candidate_tokens_per_step': (total_candidate_tokens / max(1, sum(p['n_steps'] for p in phase_summaries))) / (fixed_candidate / max(1, fixed_steps)),
        },
        'source_words': source_words,
        'runtime_sec': round(time.time() - t0, 1),
    }

    json_path = out / 'curriculum_geometry_audit.json'
    json_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')

    lines = []
    lines.append('# research — repaired curriculum geometry/accounting audit')
    lines.append('')
    r = manifest['ratios_curriculum_vs_fixed256']
    cg = manifest['curriculum_geometry']
    fg = manifest['fixed256_geometry_same_pretokenized_rows']
    lines.append(f"Rows/words: {len(examples):,} rows, {actual_words:,} charged words.")
    lines.append(f"Curriculum total: {cg['total_chunks']:,} chunks, {cg['total_steps']:,} optimizer steps, {cg['total_candidate_tokens']:,} real candidate tokens, utilization {cg['capacity_utilization']:.4f}.")
    lines.append(f"Fixed-256 on same rows: {fg['total_chunks']:,} chunks, {fg['total_steps']:,} optimizer steps, {fg['total_candidate_tokens']:,} real candidate tokens, utilization {fg['capacity_utilization']:.4f}.")
    lines.append(f"Ratios curriculum/fixed: steps {r['steps']:.4f}, chunks {r['chunks']:.4f}, candidate tokens {r['candidate_tokens']:.4f}, capacity tokens {r['capacity_tokens']:.4f}, mean candidate tokens/step {r['mean_candidate_tokens_per_step']:.4f}.")
    lines.append(f"Tail tokens lost with min_chunk_tokens={args.min_chunk_tokens}: {cg['tail_tokens_lost']} across {cg['tail_rows_losing_tokens']} rows.")
    lines.append('')
    lines.append('| phase | seq | rows | words | chunks | steps | real tokens | utilization | mean real tokens/step | split rows | short tails |')
    lines.append('|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for p in phase_summaries:
        lines.append(f"| {p['phase']} | {p['seq_len']} | {p['n_rows']:,} | {p['phase_words']:,} | {p['n_chunks']:,} | {p['n_steps']:,} | {p['candidate_tokens']:,} | {p['capacity_utilization']:.4f} | {p['mean_candidate_tokens_per_step']:.1f} | {p['split_rows']:,} | {p['rows_with_sub8_noninitial_tail']:,} |")
    lines.append('')
    lines.append(f"JSON: `{json_path}`")
    note_path = out / 'curriculum_geometry_audit.md'
    note_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    print(json.dumps({'event': 'saved', 'json': str(json_path), 'note': str(note_path)}), flush=True)
    print(json.dumps({'status': manifest['status'], 'ratios': manifest['ratios_curriculum_vs_fixed256'], 'runtime_sec': manifest['runtime_sec']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
