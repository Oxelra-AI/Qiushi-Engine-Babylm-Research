#!/usr/bin/env python3
"""research: synthesize staged PVDM 80M partial readouts before EWoK arrives.

This script stays within the paired PVDM experiment.  It reads only already
predeclared readouts (Supplement, Entity, GlobalPIQA all-option margins) and
training logs for the treatment/control/reference checkpoints.  It does not
launch training or invent a new metric; it makes the existing evidence usable
for deciding whether the PVDM formulation should be continued or revised after
EWoK four-cell completes.
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
OUT = WS / 'data/pvdm_partial_readout_synthesis'
NOTE = (ROOT / 'research/notes/representation_and_objectives/pvdm_partial_readout_synthesis.md')
READOUT = WS / 'data/pvdm_80m_readouts'
GP = READOUT / 'globalpiqa_margin'
SUPP_ENT = READOUT / 'supplement_entity/per_target'
A02_ARCHIVED = ROOT / 'experiments/archive/frontier_consolidation/data/fw_comparison_eval/per_target/compact_view_chck_80M.json'
ANATOMY = WS / 'data/globalpiqa_parallel_anatomy/globalpiqa_parallel_anatomy.json'
RUNS = {
    'reference': ROOT / 'experiments/archive/frontier_consolidation/training/runs/fw_compact_view_shared16k_seed43022',
    'treatment': WS / 'training/runs/pvdm_treatment_70M_to_80M_seed43022',
    'control': WS / 'training/runs/pvdm_control_70M_to_80M_seed43022',
}
TARGET_FILES = {
    'reference': 'compact_80m_reference',
    'treatment': 'pvdm_treatment_80m',
    'control': 'pvdm_control_80m',
}


def now_utc() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def fnum(x: Any) -> float | None:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


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
    return {
        'n': len(xs),
        'min': xs[0],
        'p05': q(0.05),
        'mean': statistics.fmean(xs),
        'median': statistics.median(xs),
        'p95': q(0.95),
        'max': xs[-1],
    }


def load_categories() -> tuple[set[str], dict[str, list[str]]]:
    if not ANATOMY.exists():
        return set(), {}
    d = load_json(ANATOMY)
    rows = d.get('agreement', {}).get('parallel', {}).get('rows', [])
    hard = {r['example_id'] for r in rows if r.get('n_ok') == 0}
    cats = {r['example_id']: list(r.get('categories') or []) for r in rows}
    return hard, cats


def load_gp_rows(target: str, mode: str) -> dict[str, dict[str, Any]]:
    p = GP / f'{target}_{mode}_rows.csv'
    if not p.exists():
        raise FileNotFoundError(p)
    out: dict[str, dict[str, Any]] = {}
    with p.open(encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            rr = dict(r)
            rr['correct'] = str(r.get('correct')).lower() == 'true'
            rr['choice'] = int(r['choice'])
            rr['label'] = int(r['label'])
            rr['correct_rank'] = int(r['correct_rank'])
            rr['top_minus_correct'] = float(r['top_minus_correct'])
            rr['correct_score'] = float(r['correct_score'])
            rr['top_score'] = float(r['top_score'])
            out[str(r['example_id'])] = rr
    return out


def load_margins_summary(target: str) -> dict[str, Any]:
    p = GP / f'{target}_margins.json'
    d = load_json(p)
    out = {}
    for mode in ['parallel', 'nonparallel']:
        s = d.get('modes', {}).get(mode, {}).get('summary', {})
        aw = s.get('always_wrong_subset') or {}
        allm = s.get('all_rows_margin_summary') or {}
        out[mode] = {
            'n': s.get('n'),
            'accuracy': s.get('accuracy'),
            'correct_rank_counts': s.get('correct_rank_counts'),
            'mean_top_minus_correct': allm.get('mean_top_minus_correct'),
            'median_top_minus_correct': allm.get('median_top_minus_correct'),
            'small_wrong_margin_le_0p50': allm.get('small_wrong_margin_le_0p50_nats'),
            'hard52_accuracy': aw.get('accuracy'),
            'hard52_correct_rank_counts': aw.get('correct_rank_counts'),
            'hard52_mean_top_minus_correct': aw.get('mean_top_minus_correct'),
            'hard52_median_top_minus_correct': aw.get('median_top_minus_correct'),
            'hard52_small_wrong_margin_le_0p50': aw.get('small_wrong_margin_le_0p50_nats'),
        }
    return out


def load_supp_entity() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    # Reference from archived compact full-eval JSON
    ref = load_json(A02_ARCHIVED)
    out['reference'] = {
        'Supplement': ref['tasks']['Supplement']['score'],
        'Entity': ref['tasks']['Entity']['score'],
        'source': str(A02_ARCHIVED),
    }
    for name in ['treatment', 'control']:
        p = SUPP_ENT / f'{TARGET_FILES[name]}.json'
        d = load_json(p)
        out[name] = {
            'Supplement': d['tasks']['Supplement']['score'],
            'Entity': d['tasks']['Entity']['score'],
            'source': str(p),
        }
    return out


def summarize_pair(a: dict[str, dict[str, Any]], b: dict[str, dict[str, Any]], *, hard_ids: set[str], cats: dict[str, list[str]], mode: str) -> dict[str, Any]:
    ids = sorted(set(a) & set(b))
    both_correct = [i for i in ids if a[i]['correct'] and b[i]['correct']]
    a_only = [i for i in ids if a[i]['correct'] and not b[i]['correct']]
    b_only = [i for i in ids if b[i]['correct'] and not a[i]['correct']]
    both_wrong = [i for i in ids if not a[i]['correct'] and not b[i]['correct']]
    rank_delta = [float(a[i]['correct_rank'] - b[i]['correct_rank']) for i in ids]  # negative means a better rank
    margin_delta = [float(a[i]['top_minus_correct'] - b[i]['top_minus_correct']) for i in ids]  # negative means a less-wrong margin
    better_rank = sum(a[i]['correct_rank'] < b[i]['correct_rank'] for i in ids)
    worse_rank = sum(a[i]['correct_rank'] > b[i]['correct_rank'] for i in ids)
    same_rank = len(ids) - better_rank - worse_rank
    lower_margin = sum(a[i]['top_minus_correct'] < b[i]['top_minus_correct'] for i in ids)
    higher_margin = sum(a[i]['top_minus_correct'] > b[i]['top_minus_correct'] for i in ids)
    equal_margin = len(ids) - lower_margin - higher_margin
    hard_subset = [i for i in ids if i in hard_ids]
    def sub(ids2: list[str]) -> dict[str, Any]:
        if not ids2:
            return {'n': 0}
        return {
            'n': len(ids2),
            'a_correct': sum(a[i]['correct'] for i in ids2),
            'b_correct': sum(b[i]['correct'] for i in ids2),
            'a_accuracy': 100.0 * sum(a[i]['correct'] for i in ids2) / len(ids2),
            'b_accuracy': 100.0 * sum(b[i]['correct'] for i in ids2) / len(ids2),
            'a_better_rank': sum(a[i]['correct_rank'] < b[i]['correct_rank'] for i in ids2),
            'same_rank': sum(a[i]['correct_rank'] == b[i]['correct_rank'] for i in ids2),
            'b_better_rank': sum(a[i]['correct_rank'] > b[i]['correct_rank'] for i in ids2),
            'a_lower_margin': sum(a[i]['top_minus_correct'] < b[i]['top_minus_correct'] for i in ids2),
            'b_lower_margin': sum(a[i]['top_minus_correct'] > b[i]['top_minus_correct'] for i in ids2),
            'mean_margin_delta_a_minus_b': statistics.fmean(a[i]['top_minus_correct'] - b[i]['top_minus_correct'] for i in ids2),
            'mean_rank_delta_a_minus_b': statistics.fmean(a[i]['correct_rank'] - b[i]['correct_rank'] for i in ids2),
        }
    cat_rows = []
    if mode == 'parallel':
        grouped: dict[str, list[str]] = defaultdict(list)
        for i in ids:
            for c in cats.get(i) or ['uncategorized']:
                grouped[c].append(i)
        for c, ids2 in sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            if len(ids2) >= 2:
                ss = sub(ids2)
                cat_rows.append({'category': c, **ss})
    return {
        'n': len(ids),
        'both_correct': len(both_correct),
        'a_only_correct': len(a_only),
        'b_only_correct': len(b_only),
        'both_wrong': len(both_wrong),
        'a_better_rank': better_rank,
        'same_rank': same_rank,
        'b_better_rank': worse_rank,
        'a_lower_top_minus_correct_margin': lower_margin,
        'same_margin': equal_margin,
        'b_lower_top_minus_correct_margin': higher_margin,
        'rank_delta_a_minus_b': qstats(rank_delta),
        'margin_delta_a_minus_b': qstats(margin_delta),
        'hard52': sub(hard_subset) if mode == 'parallel' else None,
        'category_rows': cat_rows[:20],
    }


def read_training_log(run_dir: Path) -> list[dict[str, Any]]:
    p = run_dir / 'training_log.jsonl'
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding='utf-8').splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def training_comparison() -> dict[str, Any]:
    t = read_training_log(RUNS['treatment'])
    c = read_training_log(RUNS['control'])
    by_t = {int(r['step']): r for r in t}
    by_c = {int(r['step']): r for r in c}
    steps = sorted(set(by_t) & set(by_c))
    deltas = []
    for s in steps:
        deltas.append({
            'step': s,
            'treatment_loss': by_t[s]['loss'],
            'control_loss': by_c[s]['loss'],
            'loss_delta_treatment_minus_control': by_t[s]['loss'] - by_c[s]['loss'],
            'masked_tokens_treatment': by_t[s]['masked_tokens'],
            'masked_tokens_control': by_c[s]['masked_tokens'],
            'word_exposure': by_t[s]['total_actual_word_exposure'],
        })
    loss_delta = [d['loss_delta_treatment_minus_control'] for d in deltas]
    masked_equal = all(d['masked_tokens_treatment'] == d['masked_tokens_control'] for d in deltas)
    return {
        'n_matched_steps': len(deltas),
        'treatment_lower_loss_steps': sum(d['loss_delta_treatment_minus_control'] < 0 for d in deltas),
        'control_lower_loss_steps': sum(d['loss_delta_treatment_minus_control'] > 0 for d in deltas),
        'masked_tokens_identical_each_step': masked_equal,
        'loss_delta_treatment_minus_control': qstats(loss_delta),
        'first_step': deltas[0] if deltas else None,
        'last_step': deltas[-1] if deltas else None,
        'sampled_deltas_every_25': [d for d in deltas if d['step'] == 1 or d['step'] % 25 == 0 or d['step'] == max(steps or [0])],
    }


def metrics_integrity() -> dict[str, Any]:
    out = {}
    ref_metrics = load_json(RUNS['reference'] / 'scientific_metrics.json')
    ref_ck = next((x for x in ref_metrics.get('saved_checkpoints', []) if x.get('name') == 'chck_80M'), {})
    out['reference'] = {
        'run_dir': str(RUNS['reference']),
        'checkpoint_actual_cumulative_word_exposure': ref_ck.get('actual_cumulative_word_exposure'),
        'checkpoint_path': ref_ck.get('path'),
    }
    for name in ['treatment', 'control']:
        m = load_json(RUNS[name] / 'scientific_metrics.json')
        ck = RUNS[name] / 'hf_model/chck_80M'
        out[name] = {
            'run_dir': str(RUNS[name]),
            'word_exposure': m.get('word_exposure'),
            'continuation_words': m.get('continuation_words'),
            'actual_training_steps': m.get('actual_training_steps'),
            'stage_stop_name': m.get('stage_stop_name'),
            'checkpoint_exists': ck.exists(),
            'model_safetensors_bytes': (ck / 'model.safetensors').stat().st_size if (ck / 'model.safetensors').exists() else None,
            'aggregate_mask_stats': m.get('pvdm_pairing', {}).get('aggregate_mask_stats'),
            'selected_event_categories': m.get('pvdm_pairing', {}).get('selected_event_categories'),
            'replacement_action_counts': m.get('pvdm_pairing', {}).get('replacement_action_counts'),
        }
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    keys = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); keys.append(k)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)


def row_shift_table(mode: str, hard_ids: set[str], cats: dict[str, list[str]]) -> list[dict[str, Any]]:
    rows = {name: load_gp_rows(TARGET_FILES[name], mode) for name in TARGET_FILES}
    ids = sorted(set.intersection(*(set(v) for v in rows.values())))
    out = []
    for i in ids:
        rr = {'mode': mode, 'example_id': i}
        if mode == 'parallel':
            rr['hard52'] = i in hard_ids
            rr['categories'] = ';'.join(cats.get(i) or [])
        for name in ['reference', 'control', 'treatment']:
            r = rows[name][i]
            rr[f'{name}_correct'] = r['correct']
            rr[f'{name}_rank'] = r['correct_rank']
            rr[f'{name}_top_minus_correct'] = r['top_minus_correct']
            rr[f'{name}_choice'] = r['choice']
        rr['treatment_minus_control_rank_delta'] = rows['treatment'][i]['correct_rank'] - rows['control'][i]['correct_rank']
        rr['treatment_minus_control_margin_delta'] = rows['treatment'][i]['top_minus_correct'] - rows['control'][i]['top_minus_correct']
        rr['control_minus_reference_rank_delta'] = rows['control'][i]['correct_rank'] - rows['reference'][i]['correct_rank']
        rr['control_minus_reference_margin_delta'] = rows['control'][i]['top_minus_correct'] - rows['reference'][i]['top_minus_correct']
        rr['treatment_minus_reference_rank_delta'] = rows['treatment'][i]['correct_rank'] - rows['reference'][i]['correct_rank']
        rr['treatment_minus_reference_margin_delta'] = rows['treatment'][i]['top_minus_correct'] - rows['reference'][i]['top_minus_correct']
        out.append(rr)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    hard_ids, cats = load_categories()
    gp_summaries = {name: load_margins_summary(TARGET_FILES[name]) for name in TARGET_FILES}
    rows_by_mode = {mode: {name: load_gp_rows(TARGET_FILES[name], mode) for name in TARGET_FILES} for mode in ['parallel', 'nonparallel']}
    pairwise = {}
    for mode in ['parallel', 'nonparallel']:
        pairwise[mode] = {
            'treatment_vs_control': summarize_pair(rows_by_mode[mode]['treatment'], rows_by_mode[mode]['control'], hard_ids=hard_ids, cats=cats, mode=mode),
            'control_vs_reference': summarize_pair(rows_by_mode[mode]['control'], rows_by_mode[mode]['reference'], hard_ids=hard_ids, cats=cats, mode=mode),
            'treatment_vs_reference': summarize_pair(rows_by_mode[mode]['treatment'], rows_by_mode[mode]['reference'], hard_ids=hard_ids, cats=cats, mode=mode),
        }
    supp_entity = load_supp_entity()
    sentinel_deltas = {
        'treatment_minus_control': {k: supp_entity['treatment'][k] - supp_entity['control'][k] for k in ['Supplement', 'Entity']},
        'control_minus_reference': {k: supp_entity['control'][k] - supp_entity['reference'][k] for k in ['Supplement', 'Entity']},
        'treatment_minus_reference': {k: supp_entity['treatment'][k] - supp_entity['reference'][k] for k in ['Supplement', 'Entity']},
    }
    for mode in ['parallel', 'nonparallel']:
        write_csv(OUT / f'globalpiqa_{mode}_row_shifts.csv', row_shift_table(mode, hard_ids, cats))
    payload = {
        'status': 'PVDM_PARTIAL_READOUT_SYNTHESIS',
        'created_utc': now_utc(),
        'scientific_boundary': 'PVDM staged 80M partial readout only: Supplement, Entity, GlobalPIQA margins, training logs. EWoK four-cell pending separately.',
        'targets': TARGET_FILES,
        'metrics_integrity': metrics_integrity(),
        'supplement_entity': supp_entity,
        'sentinel_deltas': sentinel_deltas,
        'globalpiqa_summaries': gp_summaries,
        'globalpiqa_pairwise': pairwise,
        'training_loss_comparison': training_comparison(),
        'row_shift_files': {
            'parallel': str(OUT / 'globalpiqa_parallel_row_shifts.csv'),
            'nonparallel': str(OUT / 'globalpiqa_nonparallel_row_shifts.csv'),
        },
        'interpretation': [
            'The paired treatment has lower PVDM training loss than the matched control at every recorded step, so the true-pivot-visible objective is locally easier on its own masked targets.',
            'The lower training loss does not transfer to GlobalPIQA: treatment is worse than control on parallel accuracy, nonparallel accuracy, and hard52 mean top-minus-correct; control also softens hard52 margin relative to uninterrupted compact 80M whereas treatment worsens it.',
            'The treatment-control contrast isolates true-pivot visibility on identical dependent targets and matched mask mass; the partial result therefore disfavors the current dense PVDM formulation as a world-relation repair, even before EWoK arrives.',
            'The result does not close the broader context-conditioned relation problem. It specifically suggests that making the pivot visible while predicting dependents may reduce pressure to represent relation pivots; the control, which masks true pivots, may preserve or improve relation-word learning at lower broad damage.',
        ],
    }
    out_json = OUT / 'pvdm_partial_readout_synthesis.json'
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    lines = []
    lines.append('# research — PVDM 80M partial readout synthesis before EWoK\n')
    lines.append('This note stays within the staged PVDM experiment and summarizes the completed broad sentinels, GlobalPIQA all-option margins, and treatment/control training traces. EWoK four-cell is still the pending relation readout.\n')
    lines.append('## Integrity\n')
    for name, rec in payload['metrics_integrity'].items():
        lines.append(f"- `{name}`: {rec}\n")
    lines.append('\n## Supplement/Entity sentinels\n')
    for name in ['reference', 'control', 'treatment']:
        lines.append(f"- `{name}`: Supplement={supp_entity[name]['Supplement']}, Entity={supp_entity[name]['Entity']}\n")
    lines.append(f"- treatment-control delta: {sentinel_deltas['treatment_minus_control']}\n")
    lines.append(f"- control-reference delta: {sentinel_deltas['control_minus_reference']}\n")
    lines.append(f"- treatment-reference delta: {sentinel_deltas['treatment_minus_reference']}\n")
    lines.append('\n## GlobalPIQA summaries\n')
    for name in ['reference', 'control', 'treatment']:
        par = gp_summaries[name]['parallel']; non = gp_summaries[name]['nonparallel']
        lines.append(f"- `{name}` parallel acc={par['accuracy']:.4f}, hard52 acc={par['hard52_accuracy']:.4f}, hard52 ranks={par['hard52_correct_rank_counts']}, hard52 mean top-minus-correct={par['hard52_mean_top_minus_correct']:.4f}; nonparallel acc={non['accuracy']:.4f}\n")
    lines.append('\n## Pairwise row movement\n')
    for mode in ['parallel', 'nonparallel']:
        lines.append(f"### {mode}\n")
        for key, rec in pairwise[mode].items():
            lines.append(f"- `{key}`: both_correct={rec['both_correct']}, a_only={rec['a_only_correct']}, b_only={rec['b_only_correct']}, both_wrong={rec['both_wrong']}, a_better_rank={rec['a_better_rank']}, same_rank={rec['same_rank']}, b_better_rank={rec['b_better_rank']}, mean_margin_delta={rec['margin_delta_a_minus_b'].get('mean')}\n")
            if mode == 'parallel':
                lines.append(f"  - hard52: {rec['hard52']}\n")
    tl = payload['training_loss_comparison']
    lines.append('\n## Training loss contrast\n')
    lines.append(f"Matched steps={tl['n_matched_steps']}, treatment lower loss steps={tl['treatment_lower_loss_steps']}, control lower={tl['control_lower_loss_steps']}, masked tokens identical each step={tl['masked_tokens_identical_each_step']}. Loss delta treatment-control stats={tl['loss_delta_treatment_minus_control']}\n")
    lines.append('\n## Current interpretation\n')
    for x in payload['interpretation']:
        lines.append(f"- {x}\n")
    lines.append(f"\nFiles: `{out_json}`, `{OUT/'globalpiqa_parallel_row_shifts.csv'}`, `{OUT/'globalpiqa_nonparallel_row_shifts.csv'}`\n")
    NOTE.write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'status': payload['status'], 'out_json': str(out_json), 'note': str(NOTE), 'key_delta': {'treatment_minus_control_sentinels': sentinel_deltas['treatment_minus_control'], 'treatment_minus_control_parallel': pairwise['parallel']['treatment_vs_control']['hard52']}}, indent=2), flush=True)


if __name__ == '__main__':
    main()
