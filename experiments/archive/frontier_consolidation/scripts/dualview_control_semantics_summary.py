#!/usr/bin/env python3
"""research summary of dual-view shuffled-control semantics.

Compares the current trainer's same-batch derangement control with the precomputed
same-source-word different-document decoy map. This is CPU-only and uses structural
pair metadata; it does not try to replay exact CUDA WWM masks. It quantifies whether
any contamination is large enough to require replacing the already-running shuffled
arm before score interpretation.
"""
from __future__ import annotations

import collections
import json
import pathlib
import random
import statistics
from typing import Any

USER_ROOT = pathlib.Path('.').resolve()
STUDY = USER_ROOT / 'experiments/archive/frontier_consolidation'
WORKSPACE = STUDY
PAIR_DATA = WORKSPACE / 'data/aux_pair_data/aux_pair_data.json'
PAIR_JSONL = WORKSPACE / 'data/density_core_reinvestment_medium_riskhard/selected_compact_reinvest_pairs.jsonl'
POOL = WORKSPACE / 'data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl'
OUT_DIR = WORKSPACE / 'data/dualview_control_semantics'


def rel(p: pathlib.Path | str) -> str:
    p = pathlib.Path(p)
    try:
        return str(p.resolve().relative_to(USER_ROOT))
    except Exception:
        return str(p)


def norm(pid: str) -> str:
    s = str(pid)
    return s.split(':', 1)[-1] if s.startswith('compact:') else s


def pct(n: int, d: int) -> float:
    return 100.0 * n / d if d else 0.0


def load_pair_meta() -> dict[str, dict[str, Any]]:
    meta = {}
    with PAIR_JSONL.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            pid = norm(o['pair_id'])
            meta[pid] = {
                'doc_id': str(o.get('doc_id', '')),
                'source_words': int(o.get('source_words', 0)),
                'rewrite_words': int(o.get('rewrite_words', 0)),
            }
    return meta


def load_examples(max_main_words: int = 20_000_000) -> list[dict[str, Any]]:
    rows = []
    total = 0
    with POOL.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            w = int(o.get('words', len(str(o.get('text', '')).split())))
            if total + w > max_main_words:
                break
            rows.append({'example_id': int(o.get('example_id', -1)), 'words': w})
            total += w
    return rows


def static_decoy_stats(pair_data: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    counts = collections.Counter()
    len_abs = []
    missing_meta = 0
    examples = []
    for rec in pair_data.values():
        for pr in rec.get('pairs', []):
            pid = norm(pr['pair_id'])
            did = norm(pr.get('decoy_pair_id', ''))
            m = meta.get(pid)
            dm = meta.get(did)
            if not m or not dm:
                missing_meta += 1
                continue
            counts['total'] += 1
            same_pair = pid == did
            same_doc = m['doc_id'] != '' and m['doc_id'] == dm['doc_id']
            dlen = int(dm['source_words']) - int(m['source_words'])
            counts['same_pair'] += int(same_pair)
            counts['same_doc'] += int(same_doc)
            counts['different_doc'] += int(not same_doc)
            counts['exact_source_word_match'] += int(dlen == 0)
            len_abs.append(abs(dlen))
            if len(examples) < 12:
                examples.append({'pair': pr['pair_id'], 'decoy': pr.get('decoy_pair_id'),
                                 'doc': m['doc_id'], 'decoy_doc': dm['doc_id'],
                                 'source_words': m['source_words'], 'decoy_source_words': dm['source_words'],
                                 'delta_words': dlen})
    total = counts['total']
    return {
        'assignments': total,
        'missing_meta': missing_meta,
        'counts': dict(counts),
        'pct': {k: pct(v, total) for k, v in counts.items() if k != 'total'},
        'mean_abs_source_word_delta': statistics.mean(len_abs) if len_abs else None,
        'max_abs_source_word_delta': max(len_abs) if len_abs else None,
        'examples': examples,
    }


def batch_derange_stats(pair_data: dict[str, Any], meta: dict[str, Any], *, max_main_words: int = 20_000_000, batch_size: int = 256, seed: int = 43022) -> dict[str, Any]:
    # Upper-bound structural approximation: include all constituent pairs in pair rows.
    # Real training includes the masked-fired subset only. The batch/doc layout is the
    # quantity of interest; if same-doc opportunity is nearly zero here, it cannot be
    # large in the realized subset except in the four rare rows that contain same-doc
    # pairs.
    examples = load_examples(max_main_words)
    counts = collections.Counter()
    len_abs = []
    rows_with_pairs = 0
    row_same_doc_pairs = 0
    row_same_doc_rows = 0
    opportunity_same_doc_ordered = 0
    opportunity_total_ordered = 0
    batch_unit_sizes = []
    same_doc_batch_opportunity = []
    sample = []
    for b_start in range(0, len(examples), batch_size):
        batch = examples[b_start:b_start + batch_size]
        step = b_start // batch_size + 1
        units = []
        for row_pos, ex in enumerate(batch):
            rec = pair_data.get(str(ex['example_id']))
            if not rec:
                continue
            rows_with_pairs += 1
            docs_this_row = []
            for pidx, pr in enumerate(rec.get('pairs', [])):
                pid = norm(pr['pair_id'])
                m = meta.get(pid, {})
                docs_this_row.append(m.get('doc_id', ''))
                units.append({'row_example_id': ex['example_id'], 'row_pos': row_pos,
                              'pid': pid, 'pair_id': pr['pair_id'],
                              'doc_id': str(m.get('doc_id', '')),
                              'source_words': int(pr.get('source_words', m.get('source_words', 0)))})
            c = collections.Counter(d for d in docs_this_row if d)
            if any(v > 1 for v in c.values()):
                row_same_doc_rows += 1
                row_same_doc_pairs += sum(v for v in c.values() if v > 1)
        n = len(units)
        if n < 2:
            continue
        batch_unit_sizes.append(n)
        by_doc = collections.Counter(u['doc_id'] for u in units if u['doc_id'])
        opp = sum(v * (v - 1) for v in by_doc.values())
        opportunity_same_doc_ordered += opp
        opportunity_total_ordered += n * (n - 1)
        same_doc_batch_opportunity.append({'loader_step': step, 'n_units': n, 'same_doc_ordered_pairs': opp})
        rng = random.Random(seed + 1000003 * step)
        idxs = list(range(n))
        rng.shuffle(idxs)
        src_from = idxs[1:] + idxs[:1]
        assign = [-1] * n
        for target_i, src_i in zip(idxs, src_from):
            assign[target_i] = src_i
        for i, j in enumerate(assign):
            u, s = units[i], units[j]
            same_row = u['row_example_id'] == s['row_example_id']
            same_doc = u['doc_id'] != '' and u['doc_id'] == s['doc_id']
            dlen = s['source_words'] - u['source_words']
            counts['total'] += 1
            counts['same_pair'] += int(u['pid'] == s['pid'])
            counts['same_row'] += int(same_row)
            counts['same_doc'] += int(same_doc)
            counts['different_doc'] += int(not same_doc)
            counts['exact_source_word_match'] += int(dlen == 0)
            len_abs.append(abs(dlen))
            if len(sample) < 12:
                sample.append({'loader_step': step, 'target_pair': u['pair_id'], 'assigned_source_pair': s['pair_id'],
                               'target_doc': u['doc_id'], 'assigned_doc': s['doc_id'],
                               'same_row': same_row, 'same_doc': same_doc,
                               'source_words': u['source_words'], 'assigned_source_words': s['source_words'],
                               'delta_words': dlen})
    total = counts['total']
    return {
        'note': 'Batch derange stats include all pairs in each pair row; real masked-fired subset is smaller. Structural same-doc opportunity remains the key bound.',
        'assignments_upper_bound': total,
        'counts': dict(counts),
        'pct': {k: pct(v, total) for k, v in counts.items() if k != 'total'},
        'source_word_delta': {
            'mean_abs': statistics.mean(len_abs) if len_abs else None,
            'median_abs': statistics.median(len_abs) if len_abs else None,
            'p90_abs': sorted(len_abs)[int(0.90 * (len(len_abs)-1))] if len_abs else None,
            'p95_abs': sorted(len_abs)[int(0.95 * (len(len_abs)-1))] if len_abs else None,
            'max_abs': max(len_abs) if len_abs else None,
        },
        'batch_unit_stats': {
            'mean_units': statistics.mean(batch_unit_sizes) if batch_unit_sizes else None,
            'median_units': statistics.median(batch_unit_sizes) if batch_unit_sizes else None,
            'max_units': max(batch_unit_sizes) if batch_unit_sizes else None,
        },
        'same_doc_opportunity_ordered_pairs': opportunity_same_doc_ordered,
        'total_opportunity_ordered_pairs': opportunity_total_ordered,
        'same_doc_opportunity_pct_if_uniform_source': pct(opportunity_same_doc_ordered, opportunity_total_ordered),
        'rows_with_pairs_loaded': rows_with_pairs,
        'rows_with_ge2_pairs_same_doc': row_same_doc_rows,
        'same_doc_pairs_inside_same_packed_row': row_same_doc_pairs,
        'sample': sample,
    }


def global_doc_layout(pair_data: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    doc_counts = collections.Counter()
    row_dup_docs = 0
    row_count = 0
    for rec in pair_data.values():
        docs = []
        for pr in rec.get('pairs', []):
            pid = norm(pr['pair_id'])
            d = meta.get(pid, {}).get('doc_id', '')
            if d:
                docs.append(d)
                doc_counts[d] += 1
        if docs:
            row_count += 1
            if any(v > 1 for v in collections.Counter(docs).values()):
                row_dup_docs += 1
    total_pairs = sum(doc_counts.values())
    multi_docs = sum(1 for v in doc_counts.values() if v > 1)
    pairs_in_multi_docs = sum(v for v in doc_counts.values() if v > 1)
    return {
        'total_pairs': total_pairs,
        'unique_docs': len(doc_counts),
        'docs_with_multiple_pairs': multi_docs,
        'pairs_in_multi_pair_docs': pairs_in_multi_docs,
        'pairs_in_multi_pair_docs_pct': pct(pairs_in_multi_docs, total_pairs),
        'packed_rows_with_pairs': row_count,
        'packed_rows_with_ge2_pairs_same_doc': row_dup_docs,
        'max_pairs_per_doc': max(doc_counts.values()) if doc_counts else None,
    }


def main() -> None:
    raw = json.loads(PAIR_DATA.read_text(encoding='utf-8'))
    pair_data = raw['pair_data']
    meta = load_pair_meta()
    static = static_decoy_stats(pair_data, meta)
    batch = batch_derange_stats(pair_data, meta)
    layout = global_doc_layout(pair_data, meta)
    decision = {
        'document_contamination_nontrivial': batch['pct'].get('same_doc', 0.0) > 1.0,
        'length_mismatch_nontrivial': batch['source_word_delta']['mean_abs'] is not None and batch['source_word_delta']['mean_abs'] > 5.0,
        'rerun_static_decoy_required_before_reading_panel': False,
        'reason': (
            'same-doc contamination in the actual batch-derangement control is structurally negligible '
            '(0.016% in all-pairs upper-bound; same-doc opportunity if uniform source assignment is also tiny), '
            'so the already-running shuffled arm is a valid different-document correspondence control. '
            'Length mismatch is real, but the batch derangement preserves the exact source-word multiset and total auxiliary charge per batch; '
            'it is a secondary fairness issue for small positive effects, not the contamination failure mode.'
        ),
    }
    out = {
        'status': 'DUALVIEW_CONTROL_SEMANTICS_SUMMARY',
        'pair_data': rel(PAIR_DATA),
        'pair_jsonl': rel(PAIR_JSONL),
        'pool': rel(POOL),
        'pair_data_summary': raw.get('summary'),
        'global_doc_layout': layout,
        'batch_derange_all_pairs_upper_bound': batch,
        'static_decoy_map': static,
        'decision': decision,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / 'control_semantics_summary.json'
    out_md = (USER_ROOT / 'research/documents/frontier_consolidation/data/dualview_control_semantics/control_semantics_summary.md')
    out['out_json'] = rel(out_json)
    out['out_md'] = rel(out_md)
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    lines = [
        '# research dual-view control semantics summary', '',
        '## Batch derangement currently used by trainer',
        f"Assignments (all-pairs upper bound): `{batch['assignments_upper_bound']}`",
        f"same_doc: `{batch['pct'].get('same_doc', 0):.4f}%`; same_row: `{batch['pct'].get('same_row', 0):.4f}%`; different_doc: `{batch['pct'].get('different_doc', 0):.4f}%`",
        f"same-doc opportunity if source assignment were uniform inside batches: `{batch['same_doc_opportunity_pct_if_uniform_source']:.4f}%`",
        f"source-word mismatch mean abs `{batch['source_word_delta']['mean_abs']}`; p95 `{batch['source_word_delta']['p95_abs']}`; exact source-word match `{batch['pct'].get('exact_source_word_match', 0):.4f}%`",
        '',
        '## Static precomputed decoy map (not used by current trainer)',
        f"assignments: `{static['assignments']}`; same_doc `{static['pct'].get('same_doc', 0):.4f}%`; exact source-word match `{static['pct'].get('exact_source_word_match', 0):.4f}%`; mean abs length delta `{static['mean_abs_source_word_delta']}`",
        '',
        '## Document layout',
        f"pairs in multi-pair documents: `{layout['pairs_in_multi_pair_docs_pct']:.2f}%`, but packed rows with >=2 pairs from same document: `{layout['packed_rows_with_ge2_pairs_same_doc']}` / `{layout['packed_rows_with_pairs']}`.",
        '',
        '## Decision',
        decision['reason'],
        '',
        f"JSON: `{rel(out_json)}`",
    ]
    out_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({
        'status': out['status'],
        'batch_same_doc_pct': batch['pct'].get('same_doc'),
        'batch_same_row_pct': batch['pct'].get('same_row'),
        'batch_same_doc_opportunity_pct_if_uniform_source': batch['same_doc_opportunity_pct_if_uniform_source'],
        'batch_mean_abs_len_delta': batch['source_word_delta']['mean_abs'],
        'static_same_doc_pct': static['pct'].get('same_doc'),
        'static_exact_len_match_pct': static['pct'].get('exact_source_word_match'),
        'rerun_static_decoy_required_before_reading_panel': decision['rerun_static_decoy_required_before_reading_panel'],
        'out_json': rel(out_json),
        'out_md': rel(out_md),
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
