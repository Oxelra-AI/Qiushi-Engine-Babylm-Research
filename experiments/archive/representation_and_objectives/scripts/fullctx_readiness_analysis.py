#!/usr/bin/env python3
"""research readiness analysis for the full-context pivot-substitution route.

This script does not score models and does not train.  It repairs/derives the
readiness facts independent review requested from existing probe outputs plus a fresh legal
segment scan:
  * inferred actual shuffle donor relaxation level from saved per-event metadata;
  * comparative pivot-lexeme and target-length decomposition;
  * full 80M->90M trainable event density under the full-context substitution
    filters (same pivot/control token length, bounded distance/frequency,
    target_len cap), by batch and category.
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

USER_ROOT = Path('.').resolve()
for p in [USER_ROOT/'experiments/archive/compact_experience/scripts', USER_ROOT/'experiments/archive/representation_and_objectives/scripts']:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import masking_curriculum_trainer as base  # noqa: E402
import pvdm_continuation_trainer as cont  # noqa: E402
import pvdm_masking_lib as pvdm  # noqa: E402
import sparse_relation_aux_trainer as relaux  # noqa: E402
import symmetric_cue_view_likelihood_probe as cueprobe  # noqa: E402

OUTDIR = Path('experiments/archive/representation_and_objectives/data/fullctx_readiness_analysis')
NOTE = Path('research/notes/representation_and_objectives/fullctx_readiness_analysis.md')
SC80 = Path('experiments/archive/representation_and_objectives/data/full_context_pivot_substitution_probe/scored_events.jsonl')
SC70 = Path('experiments/archive/representation_and_objectives/data/full_context_pivot_substitution_probe_chck70/scored_events.jsonl')
START_TAIL_ROW = 64255
EXPECTED_START_WORDS = 10011326
MAX_ROWS = 64000
MAX_WORDS = 9971289
CATEGORIES = {'physical_change','causal_connector','temporal','spatial','comparative','negation'}


def num_add(d: dict[str, Any], pfx: str, val: float) -> None:
    if not math.isfinite(float(val)):
        return
    d[pfx+'_n'] = int(d.get(pfx+'_n', 0)) + 1
    d[pfx+'_sum'] = float(d.get(pfx+'_sum', 0.0)) + float(val)
    d[pfx+'_sumsq'] = float(d.get(pfx+'_sumsq', 0.0)) + float(val) * float(val)
    d[pfx+'_min'] = min(float(val), float(d.get(pfx+'_min', float('inf'))))
    d[pfx+'_max'] = max(float(val), float(d.get(pfx+'_max', float('-inf'))))


def finalize(d: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for pfx in sorted({k[:-2] for k in d if k.endswith('_n')}):
        n = int(d.get(pfx+'_n', 0)); s = float(d.get(pfx+'_sum', 0.0)); ss = float(d.get(pfx+'_sumsq', 0.0))
        mean = s/n if n else None
        var = max(0.0, ss/n - (mean or 0.0)**2) if n else None
        out[pfx] = {'n': n, 'mean': mean, 'std': math.sqrt(var) if var is not None else None, 'min': None if n==0 else float(d.get(pfx+'_min')), 'max': None if n==0 else float(d.get(pfx+'_max'))}
    for k, v in d.items():
        if not (k.endswith('_n') or k.endswith('_sum') or k.endswith('_sumsq') or k.endswith('_min') or k.endswith('_max')):
            out[k] = v
    return out


def infer_shuffle_level(e: dict[str, Any]) -> int | None:
    if str(e.get('category')) != str(e.get('shuffle_match_category')):
        return None
    same_target_class = str(e.get('target_class')) == str(e.get('shuffle_match_target_class'))
    same_distance = str(e.get('distance_bin')) == str(e.get('shuffle_match_distance_bin'))
    # full_context assign_shuffle levels:
    # 0 = category + target_class + distance_bin + pivot_len (pivot_len equality enforced for donors)
    # 1 = category + target_class + pivot_len
    # 2 = category + pivot_len
    # 3 = category
    if same_target_class and same_distance:
        return 0
    if same_target_class:
        return 1
    return 2


def analyze_scored(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.open(encoding='utf-8') if line.strip()]
    bycat: dict[str, dict[str, Any]] = defaultdict(dict)
    counters: dict[str, Counter] = defaultdict(Counter)
    for e in rows:
        cats = ['ALL', str(e['category'])]
        d_anchor = float(e['logp_true_pivot']) - float(e['logp_matched_anchor_replace'])
        d_shuffle = float(e['logp_true_pivot']) - float(e['logp_shuffled_pivot_replace'])
        d_masked = float(e['logp_true_pivot']) - float(e['logp_pivot_masked'])
        for cat in cats:
            d = bycat[cat]
            d['n'] = int(d.get('n', 0)) + 1
            for name, val in [('true_anchor', d_anchor), ('true_shuffle', d_shuffle), ('true_pivotmasked', d_masked)]:
                num_add(d, name, val)
                d[name+'_succ'] = int(d.get(name+'_succ', 0)) + int(val > 0)
                d[name+'_den'] = int(d.get(name+'_den', 0)) + 1
            lvl = infer_shuffle_level(e)
            counters[cat]['shuffle_level::' + ('NA' if lvl is None else str(lvl))] += 1
            for field in ['pivot_norm','control_norm','shuffle_pivot_norm','target_norm','target_class','target_len','source']:
                counters[cat][field+'::'+str(e.get(field))] += 1
    out = {}
    for cat, d in bycat.items():
        item = finalize(d); item['n'] = d['n']
        for name in ['true_anchor','true_shuffle','true_pivotmasked']:
            den = int(d.get(name+'_den', 0))
            item[name+'_success_rate'] = int(d.get(name+'_succ', 0))/den if den else None
        for field in ['shuffle_level','pivot_norm','control_norm','shuffle_pivot_norm','target_norm','target_class','target_len','source']:
            item[field+'_counts'] = dict(Counter({k.split(field+'::',1)[1]: v for k, v in counters[cat].items() if k.startswith(field+'::')}).most_common(40))
        out[cat] = item
    # Explicit comparative decomposition excluding high-signal rare scalar lexemes and high-frequency ambiguous lexemes.
    groups = {
        'all': [e for e in rows if e['category'] == 'comparative'],
        'diffuse_highcount_more_better_worse_less': [e for e in rows if e['category'] == 'comparative' and str(e.get('pivot_norm')) in {'more','better','worse','less'}],
        'rare_scalar_greater_higher_lower_larger_smaller_faster_slower_longer_shorter': [e for e in rows if e['category'] == 'comparative' and str(e.get('pivot_norm')) in {'greater','higher','lower','larger','smaller','faster','slower','longer','shorter'}],
        'without_greater_higher_lower': [e for e in rows if e['category'] == 'comparative' and str(e.get('pivot_norm')) not in {'greater','higher','lower'}],
        'without_more_better_worse_less': [e for e in rows if e['category'] == 'comparative' and str(e.get('pivot_norm')) not in {'more','better','worse','less'}],
    }
    comp = {}
    for name, sub in groups.items():
        d = defaultdict(float); dd = {}
        for e in sub:
            da = float(e['logp_true_pivot']) - float(e['logp_matched_anchor_replace'])
            ds = float(e['logp_true_pivot']) - float(e['logp_shuffled_pivot_replace'])
            dm = float(e['logp_true_pivot']) - float(e['logp_pivot_masked'])
            for key, val in [('true_anchor', da), ('true_shuffle', ds), ('true_pivotmasked', dm)]:
                num_add(dd, key, val); dd[key+'_succ'] = int(dd.get(key+'_succ', 0)) + int(val > 0); dd[key+'_den'] = int(dd.get(key+'_den', 0)) + 1
        item = finalize(dd); item['n'] = len(sub)
        for key in ['true_anchor','true_shuffle','true_pivotmasked']:
            den = int(dd.get(key+'_den', 0)); item[key+'_success_rate'] = int(dd.get(key+'_succ', 0))/den if den else None
        comp[name] = item
    out['comparative_decomposition'] = comp
    return out


def scan_density(batch_size: int = 256, seq_length: int = 256, max_control_distance_abs_diff: int = 8, max_control_freq_bin_abs_diff: int = 2, max_target_token_len: int = 6, num_workers: int = 0) -> dict[str, Any]:
    tokenizer = base.make_portable_tokenizer(str(cont.DEFAULT_TOKENIZER))
    examples, label_records, segment = cont.load_segment(cont.DEFAULT_TAIL, cont.DEFAULT_LABELS, start_tail_row=START_TAIL_ROW, expected_start_tail_words=EXPECTED_START_WORDS, max_word_exposure=MAX_WORDS, max_rows=MAX_ROWS)
    dataset = base.MaskedChunkDataset(examples, tokenizer, seq_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=base.collate, num_workers=num_workers)
    agg = Counter(); cats = Counter(); batch_records = []
    per_batch_totals = []; per_batch_by_cat: dict[str, list[int]] = defaultdict(list)
    one_per_row_totals = []; capped48_totals = []
    source_cats = Counter()
    for step, batch in enumerate(loader, 1):
        lo = (research)*batch_size; hi = lo + int(batch['input_ids'].shape[0])
        word_group = batch['word_group'][:, :seq_length].contiguous(); attention = batch['attention_mask'][:, :seq_length].contiguous()
        bcat = Counter(); brow = Counter(); selected_row_keys = set()
        for b in range(int(word_group.shape[0])):
            lr = label_records[lo+b]
            group_pos = pvdm.group_positions_for_row(word_group[b], attention[b])
            usable, rej = pvdm.collect_active_events(lr, group_pos, same_length_required=True)
            agg.update(rej)
            row_best = []
            for ev in usable:
                cat = str(ev.category)
                if cat not in CATEGORIES:
                    agg[f'category_excluded::{cat}'] += 1; continue
                if len({ev.pivot_gid, ev.target_gid, ev.control_gid}) < 3:
                    agg[f'role_overlap::{cat}'] += 1; continue
                ev_raw = cueprobe.raw_event(lr, ev.event_rank)
                meta = relaux._event_meta(ev_raw)
                if int(meta.get('control_distance_abs_diff', 999)) > max_control_distance_abs_diff:
                    agg[f'control_distance_mismatch_gt{max_control_distance_abs_diff}::{cat}'] += 1; continue
                if int(meta.get('control_freq_bin_abs_diff', 999)) > max_control_freq_bin_abs_diff:
                    agg[f'control_freq_mismatch_gt{max_control_freq_bin_abs_diff}::{cat}'] += 1; continue
                if len(group_pos[int(ev.target_gid)]) > max_target_token_len:
                    agg[f'target_len_gt{max_target_token_len}::{cat}'] += 1; continue
                bcat[cat] += 1; cats[cat] += 1; source_cats[str(lr.get('source',''))+'::'+cat] += 1
                row_best.append((cat, ev.event_rank))
            if row_best:
                # one event per row, deterministic category priority inherited from research.
                priority = {'physical_change':0, 'causal_connector':1, 'spatial':2, 'temporal':3, 'comparative':4, 'negation':5}
                row_best.sort(key=lambda x: (priority.get(x[0], 9), x[1]))
                brow[row_best[0][0]] += 1
        total = sum(bcat.values()); per_batch_totals.append(total)
        one_total = sum(brow.values()); one_per_row_totals.append(one_total)
        # A plausible minimum training cap: up to 8 per family, up to 48 total per effective batch.
        capcat = {k: min(v, 8) for k, v in bcat.items()}
        cap_total = min(48, sum(capcat.values()))
        capped48_totals.append(cap_total)
        for cat in sorted(CATEGORIES):
            per_batch_by_cat[cat].append(int(bcat.get(cat, 0)))
        batch_records.append({'step': step, 'events_all_passing': total, 'events_by_category': dict(bcat), 'one_per_row_events_by_category': dict(brow), 'one_per_row_total': one_total, 'cap8_each_up_to48_total': cap_total})
    def arr_stats(xs):
        arr = np.asarray(xs, dtype=float)
        return {'n_batches': int(len(xs)), 'mean': float(arr.mean()), 'median': float(np.median(arr)), 'min': int(arr.min()), 'p10': float(np.quantile(arr, 0.10)), 'p90': float(np.quantile(arr, 0.90)), 'max': int(arr.max()), 'zero_batches': int((arr == 0).sum())}
    return {
        'segment': segment,
        'filters': {'same_length_required': True, 'max_control_distance_abs_diff': max_control_distance_abs_diff, 'max_control_freq_bin_abs_diff': max_control_freq_bin_abs_diff, 'max_target_token_len': max_target_token_len},
        'events_all_passing_by_category': dict(cats),
        'events_all_passing_total': int(sum(cats.values())),
        'events_all_passing_per_batch': arr_stats(per_batch_totals),
        'one_per_row_per_batch': arr_stats(one_per_row_totals),
        'cap8_each_up_to48_per_batch': arr_stats(capped48_totals),
        'per_batch_by_category_stats': {cat: arr_stats(xs) for cat, xs in sorted(per_batch_by_cat.items())},
        'reject_top': dict(agg.most_common(50)),
        'source_category_top': dict(source_cats.most_common(40)),
        'batch_records': batch_records,
    }


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True); NOTE.parent.mkdir(parents=True, exist_ok=True)
    sc80 = analyze_scored(SC80)
    sc70 = analyze_scored(SC70)
    density = scan_density()
    summary = {'status': 'FULLCTX_READINESS_ANALYSIS', 'scored_analysis': {'chck80_standard': sc80, 'chck70_compact': sc70}, 'density_scan_80M_to_90M': density, 'artifacts': {'summary': str(OUTDIR/'fullctx_readiness_analysis.json'), 'batch_density_records': str(OUTDIR/'batch_density_records.jsonl'), 'note': str(NOTE)}}
    (OUTDIR/'fullctx_readiness_analysis.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    with (OUTDIR/'batch_density_records.jsonl').open('w', encoding='utf-8') as f:
        for rec in density['batch_records']:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    lines = ['# research — full-context pivot-substitution readiness analysis\n']
    lines.append('## Repaired donor/readiness facts\n')
    for label, sc in [('chck80_standard', sc80), ('chck70_compact', sc70)]:
        allitem = sc['ALL']; phys = sc['physical_change']; comp = sc['comparative']
        lines.append(f"- {label}: ALL true-shuffle mean={(allitem.get('true_shuffle') or {}).get('mean')} success={allitem.get('true_shuffle_success_rate')}; inferred shuffle levels={allitem.get('shuffle_level_counts')}\n")
        lines.append(f"  - physical_change n={phys.get('n')} true-anchor={(phys.get('true_anchor') or {}).get('mean')} / true-shuffle={(phys.get('true_shuffle') or {}).get('mean')} success_shuffle={phys.get('true_shuffle_success_rate')}\n")
        lines.append(f"  - comparative n={comp.get('n')} true-anchor={(comp.get('true_anchor') or {}).get('mean')} / true-shuffle={(comp.get('true_shuffle') or {}).get('mean')} success_shuffle={comp.get('true_shuffle_success_rate')}\n")
        lines.append(f"  - comparative without greater/higher/lower: {sc['comparative_decomposition']['without_greater_higher_lower']}\n")
        lines.append(f"  - comparative more/better/worse/less: {sc['comparative_decomposition']['diffuse_highcount_more_better_worse_less']}\n")
        lines.append(f"  - target-length counts ALL: {allitem.get('target_len_counts')}\n")
    lines.append('## Trainable event density on 80M→90M segment\n')
    lines.append(f"Events passing full-context substitution filters by category: {density['events_all_passing_by_category']} total={density['events_all_passing_total']}\n")
    lines.append(f"Per 256-row batch all-passing: {density['events_all_passing_per_batch']}\n")
    lines.append(f"One-per-row per batch: {density['one_per_row_per_batch']}\n")
    lines.append(f"Cap up to 8/family, 48 total per batch: {density['cap8_each_up_to48_per_batch']}\n")
    lines.append(f"Per-category batch stats: {density['per_batch_by_category_stats']}\n")
    lines.append(f"Top rejects: {density['reject_top']}\n")
    lines.append('## Files\n')
    lines.append(f"- summary: `{summary['artifacts']['summary']}`\n")
    lines.append(f"- batch records: `{summary['artifacts']['batch_density_records']}`\n")
    NOTE.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'status': summary['status'], 'summary': summary['artifacts']['summary'], 'note': str(NOTE), 'density_total': density['events_all_passing_total'], 'density_by_category': density['events_all_passing_by_category'], 'chck80_comparative_without_greater_higher_lower': sc80['comparative_decomposition']['without_greater_higher_lower'], 'chck80_target_len_counts_all': sc80['ALL']['target_len_counts']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
