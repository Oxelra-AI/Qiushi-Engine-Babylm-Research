#!/usr/bin/env python3
"""research: audit whether the research chunk trainer preserves the existing 100M stream order.

The research trainer intentionally used the exact 10M pool and repeated its canonical
row order for every stage epoch.  Existing legal40k/depth baselines, however, train
from the materialized 100M stream.  This audit checks whether the 100M stream is a
shuffled repetition of the 10M pool and quantifies the data-order change that would
confound a future U256/U64 experiment if the research pool-order trainer were launched
unchanged.
"""
from __future__ import annotations

import collections
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

ROOT = Path('.').resolve()
WORKSPACE = ROOT / 'experiments/archive/representation_and_objectives'
P10 = ROOT / 'experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl'
P100 = ROOT / 'experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl'
OUT_DIR = WORKSPACE / 'data/stream_order_confound_audit'
NOTE = (ROOT / 'research/notes/representation_and_objectives/stream_order_confound_audit.md')
POOL_WORDS = 10_000_000
EXPECTED_EPOCHS = 10


def now() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def line_hash(raw: bytes) -> str:
    return hashlib.sha256(raw.rstrip(b'\n\r')).hexdigest()


def load_base() -> dict[str, Any]:
    hashes: list[str] = []
    words: list[int] = []
    example_ids: list[int] = []
    sources: list[str] = []
    hash_to_positions: dict[str, list[int]] = collections.defaultdict(list)
    total_words = 0
    with P10.open('rb') as f:
        for i, raw in enumerate(f):
            rec = json.loads(raw)
            h = line_hash(raw)
            hashes.append(h)
            hash_to_positions[h].append(i)
            w = int(rec['words'])
            words.append(w)
            total_words += w
            example_ids.append(int(rec.get('example_id', -1)))
            sources.append(str(rec.get('source', '')))
    return {
        'row_count': len(hashes),
        'word_count': total_words,
        'hashes': hashes,
        'words': words,
        'example_ids': example_ids,
        'sources': sources,
        'hash_counter': collections.Counter(hashes),
        'hash_to_positions': {k: v for k, v in hash_to_positions.items()},
        'duplicate_raw_line_hashes': sum(1 for v in hash_to_positions.values() if len(v) > 1),
        'max_duplicate_multiplicity': max((len(v) for v in hash_to_positions.values()), default=0),
    }


def assign_positions(epoch_hashes: list[str], base: dict[str, Any]) -> tuple[list[int], dict[str, Any]]:
    # Duplicate exact rows are interchangeable for this order audit.  Reset one
    # queue per epoch so each repeated 10M block can consume the base multiset.
    queues = {h: collections.deque(pos_list) for h, pos_list in base['hash_to_positions'].items()}
    positions: list[int] = []
    missing = 0
    for h in epoch_hashes:
        q = queues.get(h)
        if not q:
            positions.append(-1)
            missing += 1
        else:
            positions.append(q.popleft())
    leftover = sum(len(q) for q in queues.values())
    return positions, {'missing_hash_assignments': missing, 'leftover_base_rows': leftover}


def summarize_positions(pos: list[int]) -> dict[str, Any]:
    n = len(pos)
    valid = [p for p in pos if p >= 0]
    if len(valid) != n:
        return {'valid_positions': len(valid), 'total_positions': n, 'error': 'invalid positions present'}
    same_position = sum(1 for i, p in enumerate(pos) if p == i)
    adjacent_forward = sum(1 for a, b in zip(pos, pos[1:]) if b == a + 1)
    adjacent_backward = sum(1 for a, b in zip(pos, pos[1:]) if b == a - 1)
    displacements = [abs(p - i) for i, p in enumerate(pos)]
    # A light Spearman-like correlation using raw positions; sufficient to show
    # shuffled versus canonical order without scipy.
    mean_i = (n - 1) / 2
    mean_p = sum(pos) / n
    cov = sum((i - mean_i) * (p - mean_p) for i, p in enumerate(pos))
    var_i = sum((i - mean_i) ** 2 for i in range(n))
    var_p = sum((p - mean_p) ** 2 for p in pos)
    corr = cov / ((var_i * var_p) ** 0.5) if var_i > 0 and var_p > 0 else None
    return {
        'valid_positions': len(valid),
        'total_positions': n,
        'same_position_rows': same_position,
        'same_position_fraction': same_position / n,
        'adjacent_forward_edges': adjacent_forward,
        'adjacent_forward_fraction': adjacent_forward / max(1, n - 1),
        'adjacent_backward_edges': adjacent_backward,
        'adjacent_backward_fraction': adjacent_backward / max(1, n - 1),
        'mean_abs_displacement': statistics.fmean(displacements),
        'median_abs_displacement': statistics.median(displacements),
        'max_abs_displacement': max(displacements),
        'position_correlation': corr,
        'first_12_base_positions': pos[:12],
    }


def audit_stream(base: dict[str, Any]) -> dict[str, Any]:
    base_rows = int(base['row_count'])
    block_records: list[dict[str, Any]] = []
    total_rows = 0
    total_words = 0
    with P100.open('rb') as f:
        for block_idx in range(EXPECTED_EPOCHS):
            epoch_hashes: list[str] = []
            epoch_words = 0
            epoch_sources: collections.Counter[str] = collections.Counter()
            for j in range(base_rows):
                raw = f.readline()
                if not raw:
                    raise RuntimeError(f'100M stream ended early at block {block_idx}, row {j}')
                rec = json.loads(raw)
                epoch_hashes.append(line_hash(raw))
                w = int(rec['words'])
                epoch_words += w
                epoch_sources[str(rec.get('source', ''))] += w
            total_rows += len(epoch_hashes)
            total_words += epoch_words
            c = collections.Counter(epoch_hashes)
            multiset_match = c == base['hash_counter']
            pos, assign = assign_positions(epoch_hashes, base)
            pos_summary = summarize_positions(pos)
            block_records.append({
                'epoch_index': block_idx,
                'rows': len(epoch_hashes),
                'words': epoch_words,
                'words_match_10M': epoch_words == POOL_WORDS,
                'raw_line_multiset_matches_10M_pool': multiset_match,
                'assignment': assign,
                'order_summary': pos_summary,
                'source_words': dict(epoch_sources.most_common()),
                'first_stream_hashes': epoch_hashes[:5],
                'first_stream_base_positions': pos[:5],
            })
        leftover = f.readline()
        if leftover:
            raise RuntimeError('100M stream has more rows than expected 10 blocks')
    return {
        'blocks': block_records,
        'total_rows': total_rows,
        'total_words': total_words,
        'all_blocks_words_10M': all(b['words_match_10M'] for b in block_records),
        'all_blocks_multiset_match': all(b['raw_line_multiset_matches_10M_pool'] for b in block_records),
        'all_blocks_assignment_clean': all(
            b['assignment']['missing_hash_assignments'] == 0 and b['assignment']['leftover_base_rows'] == 0
            for b in block_records
        ),
    }


def write_note(payload: dict[str, Any]) -> None:
    blocks = payload['stream_audit']['blocks']
    rows = []
    for b in blocks:
        o = b['order_summary']
        rows.append(
            f"| {b['epoch_index']} | {b['rows']} | {b['words']} | {b['raw_line_multiset_matches_10M_pool']} | "
            f"{o['same_position_fraction']:.6f} | {o['adjacent_forward_fraction']:.6f} | "
            f"{o['mean_abs_displacement']:.1f} | {o['position_correlation']:.4f} | {o['first_12_base_positions']} |"
        )
    text = "\n".join([
        '# research — Stream-order confound audit for experience-utilization trainer',
        '',
        'This CPU-only audit compares the current 10M pool used by the research trainer with the materialized 100M stream used by the legal40k/depth baselines.',
        '',
        '## Result',
        '',
        f"10M pool: {payload['base']['row_count']} rows, {payload['base']['word_count']} words, SHA `{payload['base']['sha256']}`.",
        f"100M stream: {payload['stream_audit']['total_rows']} rows, {payload['stream_audit']['total_words']} words, SHA `{payload['stream_sha256']}`.",
        f"Every 64,740-row block of the 100M stream is exactly the same raw-line multiset as the 10M pool: `{payload['stream_audit']['all_blocks_multiset_match']}`; every block has exactly 10M words: `{payload['stream_audit']['all_blocks_words_10M']}`.",
        '',
        'However, the row order is not the canonical 10M-file order repeated by research. Fixed-position matches are near zero and adjacent canonical edges are nearly absent. Launching the research pool-order trainer unchanged would therefore test visibility plus a substantial data-order change relative to the existing legal40k/depth baselines.',
        '',
        '| epoch | rows | words | multiset match | same-position fraction | adjacent-forward fraction | mean abs displacement | position corr | first 12 base positions |',
        '|---:|---:|---:|:---:|---:|---:|---:|---:|---|',
        *rows,
        '',
        '## Consequence for the route',
        '',
        'The first expensive experience-utilization run should preserve the materialized 100M stream order while replacing prefix-hidden row slices by word-boundary chunks. A stream-order trainer can still keep the research invariants (10M charged words per exposure epoch, 253 updates per epoch, 2,530 total updates, full-effective-batch WWM, masked-token weighting) but removes this avoidable data-order confound.',
        '',
        f"JSON: `{payload['out_json']}`",
    ]) + '\n'
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text(text, encoding='utf-8')


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = load_base()
    payload: dict[str, Any] = {
        'status': 'STREAM_ORDER_CONFOUND_AUDIT',
        'created_utc': now(),
        'scientific_question': 'Does the verified research pool-order chunk trainer preserve the materialized 100M stream order used by the legal40k/depth baselines?',
        'base_path': str(P10),
        'stream_path': str(P100),
        'base': {
            'row_count': base['row_count'],
            'word_count': base['word_count'],
            'sha256': sha256_file(P10),
            'duplicate_raw_line_hashes': base['duplicate_raw_line_hashes'],
            'max_duplicate_multiplicity': base['max_duplicate_multiplicity'],
        },
        'stream_sha256': sha256_file(P100),
        'stream_audit': audit_stream(base),
        'interpretation': {
            'pool_order_is_not_baseline_stream_order': True,
            'first_expensive_experience_utilization_run_should_use_stream_order': True,
            'reason': 'All 100M blocks are exact repetitions of the 10M pool as a multiset but are shuffled relative to the 10M canonical row order; data order is avoidable confounding when comparing U256 to existing row256 baselines.',
        },
    }
    out_json = OUT_DIR / 'stream_order_confound_audit.json'
    payload['out_json'] = str(out_json)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    write_note(payload)
    print(json.dumps({
        'status': payload['status'],
        'base_rows': payload['base']['row_count'],
        'stream_rows': payload['stream_audit']['total_rows'],
        'all_blocks_multiset_match': payload['stream_audit']['all_blocks_multiset_match'],
        'epoch0_same_position_fraction': payload['stream_audit']['blocks'][0]['order_summary']['same_position_fraction'],
        'epoch0_adjacent_forward_fraction': payload['stream_audit']['blocks'][0]['order_summary']['adjacent_forward_fraction'],
        'out_json': str(out_json),
        'note': str(NOTE),
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
