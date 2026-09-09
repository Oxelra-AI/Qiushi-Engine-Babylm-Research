#!/usr/bin/env python3
"""research dual-view budget substitution audit.

Quantifies, under the corrected charged-word accounting, what main-stream text the
dual-view auxiliary exposure displaces relative to the exact mlm_only 20M reference,
and what auxiliary source/rewrite words replace it. This is not score evidence; it
explains the budget tradeoff behind aligned/shuffled/mlm_only comparisons.
"""
from __future__ import annotations

import collections
import json
import pathlib
from typing import Any

USER_ROOT = pathlib.Path('.').resolve()
WORKSPACE = USER_ROOT / 'experiments/archive/frontier_consolidation'
POOL = WORKSPACE / 'data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl'
PAIR_DATA = WORKSPACE / 'data/aux_pair_data/aux_pair_data.json'
PAIR_JSONL = WORKSPACE / 'data/density_core_reinvestment_medium_riskhard/selected_compact_reinvest_pairs.jsonl'
ALIGNED_RUN = WORKSPACE / 'training/runs/dualview_aligned_20M_seed43022'
MLM_RUN = WORKSPACE / 'training/runs/dualview_mlm_only_20M_seed43022'
OUT_DIR = WORKSPACE / 'data/dualview_budget_substitution'


def rel(p: pathlib.Path | str) -> str:
    p = pathlib.Path(p)
    try:
        return str(p.resolve().relative_to(USER_ROOT))
    except Exception:
        return str(p)


def read_json(p: pathlib.Path) -> Any:
    return json.loads(p.read_text(encoding='utf-8'))


def norm(pid: str) -> str:
    s = str(pid)
    return s.split(':', 1)[-1] if s.startswith('compact:') else s


def load_rows(max_words: int) -> list[dict[str, Any]]:
    rows = []
    total = 0
    with POOL.open('r', encoding='utf-8') as f:
        for idx,line in enumerate(f):
            if not line.strip():
                continue
            o = json.loads(line)
            w = int(o.get('words', len(str(o.get('text','')).split())))
            if total + w > max_words:
                break
            rows.append({'idx0': idx, 'example_id': int(o.get('example_id', -1)), 'words': w,
                         'source': str(o.get('source','')), 'text_head': str(o.get('text',''))[:120]})
            total += w
    return rows


def source_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    c = collections.Counter()
    for r in rows:
        c[r['source']] += int(r['words'])
    return dict(c.most_common())


def pair_doc_meta() -> dict[str, dict[str, Any]]:
    meta = {}
    with PAIR_JSONL.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                o = json.loads(line)
                meta[norm(o['pair_id'])] = {'doc_id': str(o.get('doc_id','')), 'source_words': int(o.get('source_words',0)), 'rewrite_words': int(o.get('rewrite_words',0))}
    return meta


def aux_theoretical_counts(pair_data: dict[str, Any], rows: list[dict[str, Any]], pair_meta: dict[str, Any]) -> dict[str, Any]:
    # Counts every constituent pair in the included main prefix, independent of WWM.
    # Actual auxiliary exposure is mask-subset dependent and read from logs/metrics.
    pair_rows = [r for r in rows if str(r['example_id']) in pair_data]
    n_pairs = 0
    source_words = 0
    rewrite_words = 0
    docs = collections.Counter()
    src_len = collections.Counter()
    for r in pair_rows:
        rec = pair_data[str(r['example_id'])]
        for pr in rec.get('pairs', []):
            n_pairs += 1
            sw = int(pr.get('source_words', 0))
            rw = int(pr.get('rewrite_words', 0))
            source_words += sw
            rewrite_words += rw
            pid = norm(pr['pair_id'])
            d = pair_meta.get(pid, {}).get('doc_id', '')
            docs[d] += 1
            src_len[sw] += 1
    return {
        'pair_rows': len(pair_rows),
        'constituent_pairs': n_pairs,
        'all_pair_source_words_once': source_words,
        'all_pair_rewrite_words_once': rewrite_words,
        'all_pair_aux_charge_if_all_fired_once': source_words + 2*rewrite_words,
        'unique_docs': len(docs),
        'top_docs_by_pair_count': docs.most_common(10),
        'source_word_length_hist_top': src_len.most_common(20),
    }


def load_log(run: pathlib.Path) -> list[dict[str, Any]]:
    out=[]
    with (run/'training_log.jsonl').open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): out.append(json.loads(line))
    return out


def main() -> None:
    am = read_json(ALIGNED_RUN / 'scientific_metrics.json')
    mm = read_json(MLM_RUN / 'scientific_metrics.json')
    alog, mlog = load_log(ALIGNED_RUN), load_log(MLM_RUN)
    dual_main_words = int(am['total_main_word_exposure'])
    mlm_main_words = int(mm['total_main_word_exposure'])
    dual_rows = load_rows(dual_main_words)
    mlm_rows = load_rows(mlm_main_words)
    # The 100M stream repeats the 10M pool, so example_id set subtraction is
    # wrong: later pass-2 rows often have the same example_id as pass-1 rows.
    # Compare by stream position. The dual-view arms process exactly the same
    # first 481 main batches as mlm_only, then stop early because aux words are
    # charged; mlm_only continues through additional stream rows.
    displaced = mlm_rows[len(dual_rows):]
    pair_data = read_json(PAIR_DATA)['pair_data']
    pmeta = pair_doc_meta()
    out = {
        'status': 'DUALVIEW_BUDGET_SUBSTITUTION_AUDIT',
        'aligned_run': rel(ALIGNED_RUN),
        'mlm_only_run': rel(MLM_RUN),
        'aligned_metrics': am,
        'mlm_only_metrics': mm,
        'same_main_prefix_updates': sum(1 for a,m in zip(alog, mlog) if int(a['loader_step']) == int(m['loader_step']) and int(a['batch_words']) == int(m['batch_words']) and int(a['masked_tokens']) == int(m['masked_tokens'])),
        'common_prefix_updates_compared': min(len(alog), len(mlog)),
        'dual_main_rows_loaded': len(dual_rows),
        'mlm_main_rows_loaded': len(mlm_rows),
        'displaced_main_rows_count': len(displaced),
        'displaced_main_words': sum(r['words'] for r in displaced),
        'dual_aux_words': int(am['total_aux_word_exposure']),
        'charged_gap_to_cap_dual': 20_000_000 - int(am['total_charged_words']),
        'displaced_source_word_counts': source_counts(displaced),
        'dual_main_source_word_counts': source_counts(dual_rows),
        'mlm_extra_rows_sample': displaced[:20],
        'dual_prefix_pair_structure_all_pairs': aux_theoretical_counts(pair_data, dual_rows, pmeta),
        'mlm_extra_pair_rows': sum(1 for r in displaced if str(r['example_id']) in pair_data),
        'mlm_extra_pair_constituent_pairs': sum(len(pair_data[str(r['example_id'])].get('pairs', [])) for r in displaced if str(r['example_id']) in pair_data),
        'interpretation': 'Dual-view spends ~0.97M legal words on source-conditioned/free auxiliary pair views; under the charged cap this replaces the final ~0.98M main-stream words seen by mlm_only. First 481 main updates share the same batch/mask geometry; mlm_only then receives additional main batches 482-506.',
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / 'budget_substitution_audit.json'
    out_md = (USER_ROOT / 'research/documents/frontier_consolidation/data/dualview_budget_substitution/budget_substitution_audit.md')
    out['out_json'] = rel(out_json); out['out_md'] = rel(out_md)
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    lines = ['# research dual-view budget substitution audit', '']
    lines.append(f"Aligned/shuffled main exposure: `{dual_main_words}`; aux exposure `{am['total_aux_word_exposure']}`; charged `{am['total_charged_words']}`.")
    lines.append(f"mlm_only main/charged exposure: `{mlm_main_words}`; updates `{mm['updates']}`.")
    lines.append(f"First `{out['same_main_prefix_updates']}` / `{out['common_prefix_updates_compared']}` common updates match in batch/masked-token geometry.")
    lines.append(f"Displaced main rows: `{out['displaced_main_rows_count']}`; displaced words `{out['displaced_main_words']}`.")
    lines.append(f"Displaced sources: `{out['displaced_source_word_counts']}`.")
    lines.append(f"Extra mlm_only pair rows: `{out['mlm_extra_pair_rows']}`; constituent pairs `{out['mlm_extra_pair_constituent_pairs']}`.")
    lines.append('')
    lines.append(out['interpretation'])
    lines.append(f"JSON: `{rel(out_json)}`")
    out_md.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({
        'status': out['status'],
        'same_main_prefix_updates': out['same_main_prefix_updates'],
        'common_prefix_updates_compared': out['common_prefix_updates_compared'],
        'displaced_main_rows_count': out['displaced_main_rows_count'],
        'displaced_main_words': out['displaced_main_words'],
        'dual_aux_words': out['dual_aux_words'],
        'displaced_source_word_counts': out['displaced_source_word_counts'],
        'mlm_extra_pair_rows': out['mlm_extra_pair_rows'],
        'mlm_extra_pair_constituent_pairs': out['mlm_extra_pair_constituent_pairs'],
        'out_json': rel(out_json),
        'out_md': rel(out_md),
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
