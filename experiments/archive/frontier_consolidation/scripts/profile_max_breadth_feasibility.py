#!/usr/bin/env python3
"""research: CPU feasibility profile for the MAX-dose source-breadth vertex.

This script does not create training data.  It measures whether the selected
2.64x compact-view MAX rows can be paired with independent whole FineWeb
sentences that spend the compact-rewrite word budget as source breadth.
"""
from __future__ import annotations

import collections
import hashlib
import json
import pathlib
import statistics
from typing import Any

ROOT = pathlib.Path('.').resolve()
WS = ROOT / 'experiments/archive/frontier_consolidation'
FULL_SOURCE = ROOT / 'experiments/archive/representation_and_objectives/training/data/fineweb_sentence_sources/fineweb_sentence_sources.jsonl'
MAX_PAIRS = WS / 'data/dose_distribution_select/selected_matched_max_pairs.jsonl'
ROW_META = WS / 'data/dose_2p64x_rowholdout_pools/compact_view_dose2p64x_changed_block_rows_meta.jsonl'
HIGHER_PROMPTS = WS / 'data/higher_dose_prompts/higher_dose_compact_prompts_all.jsonl'
OUT = WS / 'data/max_breadth_feasibility/max_breadth_feasibility_profile.json'


def norm(text: Any) -> str:
    return ' '.join(str(text or '').split())


def wc(text: Any) -> int:
    return len(norm(text).split())


def nh(text: Any) -> str:
    return hashlib.sha256(norm(text).lower().encode('utf-8')).hexdigest()[:32]


def stats(vals: list[int | float]) -> dict[str, Any]:
    if not vals:
        return {'n': 0}
    xs = sorted(float(v) for v in vals)
    def q(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        pos = p * (len(xs)-1)
        lo = int(pos); hi = min(lo+1, len(xs)-1)
        a = pos - lo
        return xs[lo]*(1-a)+xs[hi]*a
    return {'n': len(xs), 'sum': sum(xs), 'min': xs[0], 'p05': q(0.05), 'p10': q(0.10), 'p25': q(0.25), 'median': q(0.5), 'p75': q(0.75), 'p90': q(0.90), 'p95': q(0.95), 'max': xs[-1], 'mean': statistics.mean(xs)}


def main() -> None:
    pair_by_id: dict[str, dict[str, Any]] = {}
    used_sid: set[str] = set(); used_hash: set[str] = set(); used_doc: set[str] = set()
    totals = collections.Counter()
    origins = collections.Counter()
    with MAX_PAIRS.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            pid = str(d['pair_id'])
            sw = int(d.get('source_words') or wc(d.get('source_text')))
            rw = int(d.get('rewrite_words') or wc(d.get('rewrite_text')))
            pair_by_id[pid] = {**d, 'source_words': sw, 'rewrite_words': rw, 'source_hash': nh(d.get('source_text'))}
            used_sid.add(str(d.get('sentence_id')))
            used_hash.add(nh(d.get('source_text')))
            used_doc.add(str(d.get('doc_id')))
            totals['pairs'] += 1; totals['source_words'] += sw; totals['rewrite_words'] += rw; totals['pair_words'] += sw+rw
            origins[str(d.get('origin') or 'unknown')] += sw+rw

    pair_rows: list[dict[str, Any]] = []
    topup_rows: list[dict[str, Any]] = []
    with ROW_META.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            m = json.loads(line)
            pids = [str(x) for x in m.get('pair_ids') or []]
            if pids:
                sw = sum(pair_by_id[p]['source_words'] for p in pids)
                rw = sum(pair_by_id[p]['rewrite_words'] for p in pids)
                if sw + rw != int(m['words']):
                    raise RuntimeError(f"row {m['row_index']} words mismatch {sw}+{rw}!={m['words']}")
                pair_rows.append({'row_index': int(m['row_index']), 'example_id': int(m['example_id']), 'total_words': int(m['words']), 'source_words': sw, 'companion_words': rw, 'pair_count': len(pids)})
            else:
                topup_rows.append(m)

    # Sentence IDs prepared as unused after research's broader used filter.
    promptable_sid: set[str] = set()
    if HIGHER_PROMPTS.exists():
        with HIGHER_PROMPTS.open('r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    promptable_sid.add(str(d.get('sentence_id')))

    candidates_all = []
    candidates_promptable = []
    excluded_selected = 0
    with FULL_SOURCE.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            text = norm(d.get('text'))
            if not text:
                continue
            sid = str(d.get('sentence_id'))
            h = nh(text)
            w = int(d.get('words') or wc(text))
            if w != wc(text):
                raise RuntimeError(f"word mismatch sentence {sid}")
            if sid in used_sid or h in used_hash:
                excluded_selected += 1
                continue
            rec = {'sentence_id': sid, 'doc_id': str(d.get('doc_id') or ''), 'words': w, 'hash': h, 'flags': list(d.get('source_row_quality_flags') or [])}
            candidates_all.append(rec)
            if sid in promptable_sid:
                candidates_promptable.append(rec)

    budgets = [r['companion_words'] for r in pair_rows]
    by_len_all = collections.Counter(r['words'] for r in candidates_all)
    by_len_prompt = collections.Counter(r['words'] for r in candidates_promptable)
    need = sum(budgets)
    out = {
        'status': 'MAX_BREADTH_FEASIBILITY_PROFILE',
        'max_pair_totals': dict(totals),
        'max_pair_origin_pair_words': dict(origins),
        'pair_row_count': len(pair_rows),
        'topup_row_count': len(topup_rows),
        'budget_stats': stats(budgets),
        'budget_words_total': need,
        'candidate_all_excluding_selected': {'rows': len(candidates_all), 'words': sum(r['words'] for r in candidates_all), 'word_stats': stats([r['words'] for r in candidates_all]), 'by_len_1_80': {str(k): by_len_all[k] for k in range(1,81) if by_len_all[k]}},
        'candidate_promptable_after_step258_filter': {'rows': len(candidates_promptable), 'words': sum(r['words'] for r in candidates_promptable), 'word_stats': stats([r['words'] for r in candidates_promptable]), 'by_len_1_80': {str(k): by_len_prompt[k] for k in range(1,81) if by_len_prompt[k]}},
        'excluded_selected_source_sentences': excluded_selected,
        'selected_source_sentences': len(used_sid),
        'selected_source_docs': len(used_doc),
        'row_budget_counts': {str(k): v for k, v in sorted(collections.Counter(budgets).items())},
        'smallest_budgets': sorted(budgets)[:40],
        'largest_budgets': sorted(budgets)[-40:],
        'meaning': 'If candidate words exceed budget and row budgets have exact whole-sentence decompositions, build the MAX breadth arm as common MAX sources plus independent FineWeb companion sentences, keeping topup/filler identical.',
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': out['status'], 'budget_words_total': need, 'pair_rows': len(pair_rows), 'topup_rows': len(topup_rows), 'candidate_all_words': out['candidate_all_excluding_selected']['words'], 'candidate_promptable_words': out['candidate_promptable_after_step258_filter']['words'], 'min_budget': min(budgets), 'max_budget': max(budgets), 'out': str(OUT)}, indent=2), flush=True)

if __name__ == '__main__':
    main()
