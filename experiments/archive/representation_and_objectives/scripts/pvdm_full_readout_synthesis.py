#!/usr/bin/env python3
"""research: full three-target PVDM 80M synthesis after EWoK reference readout.

Reads the staged PVDM treatment/control readouts plus the uninterrupted compact 80M
reference.  The purpose is to update the mechanism-level understanding of the
context-conditioned world-relation problem, not to select a checkpoint for final
submission.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path('.').resolve()
WS = ROOT / 'experiments/archive/representation_and_objectives'
OUT = WS / 'data/pvdm_full_readout_synthesis'
NOTE = (ROOT / 'research/notes/representation_and_objectives/pvdm_full_readout_synthesis.md')
READOUT = WS / 'data/pvdm_80m_readouts'
GP = READOUT / 'globalpiqa_margin'
EWOK = READOUT / 'ewok_fourcell'
SUPP_ENT = READOUT / 'supplement_entity/per_target'
A02_ARCHIVED = ROOT / 'experiments/archive/frontier_consolidation/data/fw_comparison_eval/per_target/compact_view_chck_80M.json'
ANATOMY = WS / 'data/globalpiqa_parallel_anatomy/globalpiqa_parallel_anatomy.json'
RUNS = {
    'reference': ROOT / 'experiments/archive/frontier_consolidation/training/runs/fw_compact_view_shared16k_seed43022',
    'treatment': WS / 'training/runs/pvdm_treatment_70M_to_80M_seed43022',
    'control': WS / 'training/runs/pvdm_control_70M_to_80M_seed43022',
}
TARGETS = {
    'reference': 'compact_80m_reference',
    'treatment': 'pvdm_treatment_80m',
    'control': 'pvdm_control_80m',
}


def now_utc() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def qstats(vals: list[float]) -> dict[str, Any]:
    xs = sorted(v for v in vals if math.isfinite(v))
    if not xs:
        return {'n': 0}
    def q(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        idx = p * (len(xs) - 1)
        lo = math.floor(idx); hi = math.ceil(idx)
        if lo == hi:
            return xs[lo]
        return xs[lo] * (hi - idx) + xs[hi] * (idx - lo)
    return {'n': len(xs), 'min': xs[0], 'p05': q(0.05), 'mean': statistics.fmean(xs), 'median': statistics.median(xs), 'p95': q(0.95), 'max': xs[-1]}


def load_gp_rows(target: str, mode: str) -> dict[str, dict[str, Any]]:
    p = GP / f'{target}_{mode}_rows.csv'
    if not p.exists():
        raise FileNotFoundError(p)
    out = {}
    with p.open(newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rr = dict(r)
            rr['correct'] = str(rr.get('correct')).lower() == 'true'
            rr['choice'] = int(rr['choice']); rr['label'] = int(rr['label']); rr['correct_rank'] = int(rr['correct_rank'])
            rr['top_minus_correct'] = float(rr['top_minus_correct'])
            out[rr['example_id']] = rr
    return out


def load_ewok_rows(target: str) -> dict[str, dict[str, Any]]:
    p = EWOK / target / 'ewok_interaction_records.csv'
    if not p.exists():
        raise FileNotFoundError(p)
    out = {}
    with p.open(newline='', encoding='utf-8') as f:
        for i, r in enumerate(csv.DictReader(f)):
            rr = dict(r)
            key = rr.get('uid') or rr.get('example_id') or rr.get('row_id') or f'row_{i}'
            # The research record keys are stable in row order even if uid field is absent.
            rr['_row_index'] = i
            rr['_key'] = key
            bool_keys = {'saved_model_correct_flag', 'saved_model_wrong_flag', 'conditional_reversal_failure_stable', 'stable_nonpositive_interaction', 'both_official_sum_positive', 'both_within_context_sum_positive', 'both_within_context_mean_positive', 'local_both_actual_over_swapped_positive', 'deleted_contexts_identical'}
            float_keys = {'interaction_sum', 'interaction_mean', 'deletion_interaction_sum', 'interaction_minus_deletion_sum'}
            for k in bool_keys:
                if k in rr:
                    rr[k] = str(rr[k]).strip().lower() == 'true'
            for k in float_keys:
                if k in rr:
                    try: rr[k] = float(rr[k])
                    except Exception: pass
            out[key] = rr
    return out


def load_anatomy() -> tuple[set[str], dict[str, list[str]]]:
    if not ANATOMY.exists(): return set(), {}
    d = load_json(ANATOMY)
    rows = d.get('agreement', {}).get('parallel', {}).get('rows', [])
    return {r['example_id'] for r in rows if r.get('n_ok') == 0}, {r['example_id']: list(r.get('categories') or []) for r in rows}


def gp_summary(target: str) -> dict[str, Any]:
    d = load_json(GP / f'{target}_margins.json')
    out = {}
    for mode in ['parallel', 'nonparallel']:
        s = d['modes'][mode]['summary']
        aw = s.get('always_wrong_subset') or {}
        allm = s.get('all_rows_margin_summary') or {}
        out[mode] = {
            'accuracy': s.get('accuracy'),
            'correct_rank_counts': s.get('correct_rank_counts'),
            'mean_top_minus_correct': allm.get('mean_top_minus_correct'),
            'small_wrong_margin_le_0p50': allm.get('small_wrong_margin_le_0p50_nats'),
            'hard52_accuracy': aw.get('accuracy'),
            'hard52_rank_counts': aw.get('correct_rank_counts'),
            'hard52_mean_top_minus_correct': aw.get('mean_top_minus_correct'),
            'hard52_small_wrong_margin_le_0p50': aw.get('small_wrong_margin_le_0p50_nats'),
        }
    return out


def ewok_summary(target: str) -> dict[str, Any]:
    d = load_json(EWOK / target / 'ewok_interaction_summary.json')
    s = d['summary']
    return {
        'accuracy': s.get('accuracy'),
        'saved_wrong': s.get('saved_wrong'),
        'stable_failure': s.get('stable_failure'),
        'stable_failure_frac_wrong': s.get('stable_failure_frac_wrong'),
        'interaction_sum_wrong_mean': (s.get('interaction_sum_wrong') or {}).get('mean'),
        'interaction_sum_wrong_median': (s.get('interaction_sum_wrong') or {}).get('median'),
        'within_both_positive_wrong_frac': s.get('within_both_positive_wrong_frac'),
        'local_both_actual_over_swapped_positive_wrong_frac': s.get('local_both_actual_over_swapped_positive_wrong_frac'),
    }


def sentinels() -> dict[str, Any]:
    ref = load_json(A02_ARCHIVED)
    out = {'reference': {'Supplement': ref['tasks']['Supplement']['score'], 'Entity': ref['tasks']['Entity']['score'], 'source': str(A02_ARCHIVED)}}
    for name in ['treatment', 'control']:
        d = load_json(SUPP_ENT / f'{TARGETS[name]}.json')
        out[name] = {'Supplement': d['tasks']['Supplement']['score'], 'Entity': d['tasks']['Entity']['score'], 'source': str(SUPP_ENT / f'{TARGETS[name]}.json')}
    return out


def load_train_log(name: str) -> list[dict[str, Any]]:
    p = RUNS[name] / 'training_log.jsonl'
    if not p.exists(): return []
    return [json.loads(line) for line in p.read_text(encoding='utf-8').splitlines() if line.strip()]


def training_loss_delta() -> dict[str, Any]:
    t = {int(r['step']): r for r in load_train_log('treatment')}
    c = {int(r['step']): r for r in load_train_log('control')}
    steps = sorted(set(t) & set(c))
    vals = [t[s]['loss'] - c[s]['loss'] for s in steps]
    return {
        'matched_steps': len(steps),
        'treatment_lower_loss_steps': sum(v < 0 for v in vals),
        'control_lower_loss_steps': sum(v > 0 for v in vals),
        'masked_tokens_identical_each_step': all(t[s]['masked_tokens'] == c[s]['masked_tokens'] for s in steps),
        'loss_delta_treatment_minus_control': qstats(vals),
        'first': {'step': steps[0], 'delta': vals[0], 'treatment_loss': t[steps[0]]['loss'], 'control_loss': c[steps[0]]['loss']} if steps else None,
        'last': {'step': steps[-1], 'delta': vals[-1], 'treatment_loss': t[steps[-1]]['loss'], 'control_loss': c[steps[-1]]['loss']} if steps else None,
    }


def pairwise_gp(a: str, b: str, mode: str, hard_ids: set[str], cats: dict[str, list[str]]) -> dict[str, Any]:
    ra, rb = load_gp_rows(TARGETS[a], mode), load_gp_rows(TARGETS[b], mode)
    ids = sorted(set(ra) & set(rb))
    def sub(ids2: list[str]) -> dict[str, Any]:
        if not ids2: return {'n': 0}
        return {
            'n': len(ids2), 'a_correct': sum(ra[i]['correct'] for i in ids2), 'b_correct': sum(rb[i]['correct'] for i in ids2),
            'a_better_rank': sum(ra[i]['correct_rank'] < rb[i]['correct_rank'] for i in ids2),
            'same_rank': sum(ra[i]['correct_rank'] == rb[i]['correct_rank'] for i in ids2),
            'b_better_rank': sum(ra[i]['correct_rank'] > rb[i]['correct_rank'] for i in ids2),
            'a_lower_margin': sum(ra[i]['top_minus_correct'] < rb[i]['top_minus_correct'] for i in ids2),
            'b_lower_margin': sum(ra[i]['top_minus_correct'] > rb[i]['top_minus_correct'] for i in ids2),
            'mean_rank_delta_a_minus_b': statistics.fmean(ra[i]['correct_rank'] - rb[i]['correct_rank'] for i in ids2),
            'mean_margin_delta_a_minus_b': statistics.fmean(ra[i]['top_minus_correct'] - rb[i]['top_minus_correct'] for i in ids2),
        }
    out = sub(ids)
    out['both_correct'] = sum(ra[i]['correct'] and rb[i]['correct'] for i in ids)
    out['a_only_correct'] = sum(ra[i]['correct'] and not rb[i]['correct'] for i in ids)
    out['b_only_correct'] = sum((not ra[i]['correct']) and rb[i]['correct'] for i in ids)
    out['both_wrong'] = sum((not ra[i]['correct']) and (not rb[i]['correct']) for i in ids)
    if mode == 'parallel':
        out['hard52'] = sub([i for i in ids if i in hard_ids])
        grouped = defaultdict(list)
        for i in ids:
            for c in cats.get(i) or ['uncategorized']:
                grouped[c].append(i)
        out['category_rows'] = [{'category': c, **sub(v)} for c, v in sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0])) if len(v) >= 2]
    return out


def pairwise_ewok(a: str, b: str) -> dict[str, Any]:
    ra, rb = load_ewok_rows(TARGETS[a]), load_ewok_rows(TARGETS[b])
    ids = sorted(set(ra) & set(rb), key=lambda x: ra[x]['_row_index'])
    def boolv(r, k):
        v = r.get(k)
        if isinstance(v, bool): return v
        if isinstance(v, str): return v.strip().lower() == 'true'
        return bool(v)
    def fl(r,k):
        try: return float(r.get(k))
        except Exception: return float('nan')
    stable_a = [i for i in ids if boolv(ra[i], 'conditional_reversal_failure_stable')]
    stable_b = [i for i in ids if boolv(rb[i], 'conditional_reversal_failure_stable')]
    out = {
        'n': len(ids),
        'a_correct': sum(boolv(ra[i], 'saved_model_correct_flag') for i in ids),
        'b_correct': sum(boolv(rb[i], 'saved_model_correct_flag') for i in ids),
        'both_correct': sum(boolv(ra[i], 'saved_model_correct_flag') and boolv(rb[i], 'saved_model_correct_flag') for i in ids),
        'a_only_correct': sum(boolv(ra[i], 'saved_model_correct_flag') and not boolv(rb[i], 'saved_model_correct_flag') for i in ids),
        'b_only_correct': sum((not boolv(ra[i], 'saved_model_correct_flag')) and boolv(rb[i], 'saved_model_correct_flag') for i in ids),
        'both_wrong': sum((not boolv(ra[i], 'saved_model_correct_flag')) and (not boolv(rb[i], 'saved_model_correct_flag')) for i in ids),
        'stable_a': len(stable_a),
        'stable_b': len(stable_b),
        'stable_both': len(set(stable_a) & set(stable_b)),
        'stable_a_only': len(set(stable_a) - set(stable_b)),
        'stable_b_only': len(set(stable_b) - set(stable_a)),
        'interaction_sum_delta_a_minus_b': qstats([fl(ra[i], 'interaction_sum') - fl(rb[i], 'interaction_sum') for i in ids]),
    }
    grouped = defaultdict(list)
    for i in ids:
        dom = str(ra[i].get('domain') or '')
        grouped[dom].append(i)
    bydom = []
    for dom, ii in sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        if len(ii) < 10: continue
        bydom.append({
            'domain': dom,
            'n': len(ii),
            'a_correct': sum(boolv(ra[i], 'saved_model_correct_flag') for i in ii),
            'b_correct': sum(boolv(rb[i], 'saved_model_correct_flag') for i in ii),
              'stable_a': sum(boolv(ra[i], 'conditional_reversal_failure_stable') for i in ii),
              'stable_b': sum(boolv(rb[i], 'conditional_reversal_failure_stable') for i in ii),
            'mean_interaction_delta_a_minus_b': statistics.fmean(fl(ra[i], 'interaction_sum') - fl(rb[i], 'interaction_sum') for i in ii),
        })
    out['by_domain'] = bydom
    return out


def integrity() -> dict[str, Any]:
    out = {}
    refm = load_json(RUNS['reference'] / 'scientific_metrics.json')
    ref_ck = next((x for x in refm.get('saved_checkpoints', []) if x.get('name') == 'chck_80M'), {})
    out['reference'] = {'word_exposure': ref_ck.get('actual_cumulative_word_exposure'), 'path': ref_ck.get('path')}
    for name in ['treatment', 'control']:
        m = load_json(RUNS[name] / 'scientific_metrics.json')
        out[name] = {
            'word_exposure': m.get('word_exposure'), 'continuation_words': m.get('continuation_words'),
            'steps': m.get('actual_training_steps'), 'stage_stop_name': m.get('stage_stop_name'),
            'checkpoint_exists': (RUNS[name] / 'hf_model/chck_80M/model.safetensors').exists(),
            'selected_event_categories': m.get('pvdm_pairing', {}).get('selected_event_categories'),
            'aggregate_mask_stats': m.get('pvdm_pairing', {}).get('aggregate_mask_stats'),
        }
    return out


def write_row_tables(hard_ids: set[str], cats: dict[str, list[str]]):
    for mode in ['parallel', 'nonparallel']:
        rows = {name: load_gp_rows(TARGETS[name], mode) for name in TARGETS}
        ids = sorted(set.intersection(*(set(v) for v in rows.values())))
        flat = []
        for i in ids:
            rec = {'mode': mode, 'example_id': i}
            if mode == 'parallel':
                rec['hard52'] = i in hard_ids; rec['categories'] = ';'.join(cats.get(i) or [])
            for name in ['reference', 'control', 'treatment']:
                r = rows[name][i]
                rec[f'{name}_correct'] = r['correct']; rec[f'{name}_rank'] = r['correct_rank']; rec[f'{name}_margin'] = r['top_minus_correct']; rec[f'{name}_choice'] = r['choice']
            rec['treat_minus_control_rank'] = rows['treatment'][i]['correct_rank'] - rows['control'][i]['correct_rank']
            rec['treat_minus_control_margin'] = rows['treatment'][i]['top_minus_correct'] - rows['control'][i]['top_minus_correct']
            flat.append(rec)
        p = OUT / f'globalpiqa_{mode}_threeway_rows.csv'
        with p.open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
            w.writeheader(); w.writerows(flat)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    hard_ids, cats = load_anatomy()
    se = sentinels()
    gp = {name: gp_summary(TARGETS[name]) for name in TARGETS}
    ew = {name: ewok_summary(TARGETS[name]) for name in TARGETS}
    pairs = {
        'globalpiqa': {
            mode: {
                'treatment_vs_control': pairwise_gp('treatment', 'control', mode, hard_ids, cats),
                'control_vs_reference': pairwise_gp('control', 'reference', mode, hard_ids, cats),
                'treatment_vs_reference': pairwise_gp('treatment', 'reference', mode, hard_ids, cats),
            } for mode in ['parallel', 'nonparallel']
        },
        'ewok': {
            'treatment_vs_control': pairwise_ewok('treatment', 'control'),
            'control_vs_reference': pairwise_ewok('control', 'reference'),
            'treatment_vs_reference': pairwise_ewok('treatment', 'reference'),
        },
    }
    write_row_tables(hard_ids, cats)
    deltas = {
        'treatment_minus_control': {
            'Supplement': se['treatment']['Supplement'] - se['control']['Supplement'],
            'Entity': se['treatment']['Entity'] - se['control']['Entity'],
            'GlobalPIQA_parallel': gp['treatment']['parallel']['accuracy'] - gp['control']['parallel']['accuracy'],
            'GlobalPIQA_nonparallel': gp['treatment']['nonparallel']['accuracy'] - gp['control']['nonparallel']['accuracy'],
            'GP_hard52_mean_margin': gp['treatment']['parallel']['hard52_mean_top_minus_correct'] - gp['control']['parallel']['hard52_mean_top_minus_correct'],
            'EWoK_accuracy': ew['treatment']['accuracy'] - ew['control']['accuracy'],
            'EWoK_stable_failure_frac_wrong': ew['treatment']['stable_failure_frac_wrong'] - ew['control']['stable_failure_frac_wrong'],
            'EWoK_wrong_median_interaction': ew['treatment']['interaction_sum_wrong_median'] - ew['control']['interaction_sum_wrong_median'],
        },
        'control_minus_reference': {
            'Supplement': se['control']['Supplement'] - se['reference']['Supplement'],
            'Entity': se['control']['Entity'] - se['reference']['Entity'],
            'GlobalPIQA_parallel': gp['control']['parallel']['accuracy'] - gp['reference']['parallel']['accuracy'],
            'GlobalPIQA_nonparallel': gp['control']['nonparallel']['accuracy'] - gp['reference']['nonparallel']['accuracy'],
            'GP_hard52_mean_margin': gp['control']['parallel']['hard52_mean_top_minus_correct'] - gp['reference']['parallel']['hard52_mean_top_minus_correct'],
            'EWoK_accuracy': ew['control']['accuracy'] - ew['reference']['accuracy'],
            'EWoK_stable_failure_frac_wrong': ew['control']['stable_failure_frac_wrong'] - ew['reference']['stable_failure_frac_wrong'],
            'EWoK_wrong_median_interaction': ew['control']['interaction_sum_wrong_median'] - ew['reference']['interaction_sum_wrong_median'],
        },
        'treatment_minus_reference': {
            'Supplement': se['treatment']['Supplement'] - se['reference']['Supplement'],
            'Entity': se['treatment']['Entity'] - se['reference']['Entity'],
            'GlobalPIQA_parallel': gp['treatment']['parallel']['accuracy'] - gp['reference']['parallel']['accuracy'],
            'GlobalPIQA_nonparallel': gp['treatment']['nonparallel']['accuracy'] - gp['reference']['nonparallel']['accuracy'],
            'GP_hard52_mean_margin': gp['treatment']['parallel']['hard52_mean_top_minus_correct'] - gp['reference']['parallel']['hard52_mean_top_minus_correct'],
            'EWoK_accuracy': ew['treatment']['accuracy'] - ew['reference']['accuracy'],
            'EWoK_stable_failure_frac_wrong': ew['treatment']['stable_failure_frac_wrong'] - ew['reference']['stable_failure_frac_wrong'],
            'EWoK_wrong_median_interaction': ew['treatment']['interaction_sum_wrong_median'] - ew['reference']['interaction_sum_wrong_median'],
        }
    }
    payload = {
        'status': 'PVDM_FULL_READOUT_SYNTHESIS', 'created_utc': now_utc(),
        'research_question': 'Does true-pivot visible dependent masking repair the persistent context-conditioned world-relation weakness better than a target-identical matched anchor control and the uninterrupted compact trajectory?',
        'integrity': integrity(), 'sentinels': se, 'globalpiqa': gp, 'ewok': ew,
        'deltas': deltas, 'pairwise': pairs, 'training_loss_delta': training_loss_delta(),
        'interpretation': [
            'Treatment lower training loss on identical target/mask mass did not translate to relation competence; treatment is worse than control on GlobalPIQA parallel, hard52 margin/ranks, nonparallel GlobalPIQA, Supplement, and EWoK accuracy/stable failures, with only Entity improving.',
            'Control, which masks the true relation pivot while keeping the surrogate visible, is closer to or better than the uninterrupted compact 80M reference on Supplement and GlobalPIQA hard52 margin, but worse on EWoK than reference. This means the common continuation/target-redistribution cost is nontrivial, and true-pivot visibility adds damage rather than repair.',
            'The current dense PVDM formulation is closed as a continuation route to 100M. The broader context-conditioned world-relation problem remains open; the mechanism lesson is that hiding the relational pivot may be more valuable than making it visible, because predicting relation words and consequences jointly may be necessary to learn conditional compatibility.',
            'A coherent next research step should revise the relational objective/representation using this specific failure: preserve broad compact learning and train a relation-conditioned contrast or pivot-prediction signal that strengthens interaction without shifting broad probability mass, rather than spending more exposure on this PVDM treatment.',
        ],
        'row_tables': {'globalpiqa_parallel': str(OUT/'globalpiqa_parallel_threeway_rows.csv'), 'globalpiqa_nonparallel': str(OUT/'globalpiqa_nonparallel_threeway_rows.csv')},
    }
    out = OUT / 'pvdm_full_readout_synthesis.json'
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    lines = []
    lines.append('# research — PVDM 80M full readout synthesis\n')
    lines.append('The staged PVDM experiment tested the accumulated relation question with treatment/control/reference readouts at the same 80,011,326-word boundary.\n')
    lines.append('## Integrity\n')
    for name, rec in payload['integrity'].items(): lines.append(f'- `{name}`: {rec}\n')
    lines.append('\n## Main table\n')
    for name in ['reference', 'control', 'treatment']:
        lines.append(f"- `{name}`: Supplement={se[name]['Supplement']}, Entity={se[name]['Entity']}; GP_parallel={gp[name]['parallel']['accuracy']:.4f}, GP_nonparallel={gp[name]['nonparallel']['accuracy']:.4f}, GP_hard52_margin={gp[name]['parallel']['hard52_mean_top_minus_correct']:.4f}, GP_hard52_ranks={gp[name]['parallel']['hard52_rank_counts']}; EWoK_acc={ew[name]['accuracy']:.6f}, EWoK_stable_frac_wrong={ew[name]['stable_failure_frac_wrong']:.6f}, EWoK_wrong_median={ew[name]['interaction_sum_wrong_median']:.6f}\n")
    lines.append('\n## Deltas\n')
    for k,v in deltas.items(): lines.append(f'- `{k}`: {v}\n')
    lines.append('\n## Row-level movement\n')
    gph = pairs['globalpiqa']['parallel']['treatment_vs_control']['hard52']
    lines.append(f"- GlobalPIQA hard52 treatment vs control: treatment better rank {gph['a_better_rank']}, same {gph['same_rank']}, control better {gph['b_better_rank']}; treatment lower margin {gph['a_lower_margin']}, control lower margin {gph['b_lower_margin']}; mean margin delta {gph['mean_margin_delta_a_minus_b']:.6f}.\n")
    ew_tc = pairs['ewok']['treatment_vs_control']
    lines.append(f"- EWoK treatment vs control: treatment correct {ew_tc['a_correct']}, control correct {ew_tc['b_correct']}, treatment-only correct {ew_tc['a_only_correct']}, control-only correct {ew_tc['b_only_correct']}; stable treatment {ew_tc['stable_a']}, stable control {ew_tc['stable_b']}, stable both {ew_tc['stable_both']}.\n")
    tl = payload['training_loss_delta']
    lines.append('\n## Training signal\n')
    lines.append(f"Treatment lower loss steps={tl['treatment_lower_loss_steps']}/{tl['matched_steps']}; final loss delta={tl['last']['delta']:.6f}; mean delta={tl['loss_delta_treatment_minus_control']['mean']:.6f}; masked tokens identical each step={tl['masked_tokens_identical_each_step']}.\n")
    lines.append('\n## Mechanism interpretation\n')
    for x in payload['interpretation']: lines.append(f'- {x}\n')
    lines.append(f"\nFiles: `{out}`, `{OUT/'globalpiqa_parallel_threeway_rows.csv'}`, `{OUT/'globalpiqa_nonparallel_threeway_rows.csv'}`\n")
    NOTE.write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'status': payload['status'], 'out_json': str(out), 'note': str(NOTE), 'deltas': deltas['treatment_minus_control']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
