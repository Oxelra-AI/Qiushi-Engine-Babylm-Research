#!/usr/bin/env python3
"""research: item-level complementarity across existing legal research checkpoints.

Reads existing official-compatible prediction files for the fully legal compact-view
reinvest trajectory at 20M, 70M, 80M, and 100M. Computes per-item correctness,
checkpoint score table, union/oracle ceilings, agreement, formed/lost behavior,
and simple vote ensembles for discrete columns. Reading is handled from its saved
column scores because it is continuous rather than item-accuracy.
"""
from __future__ import annotations

import csv
import json
import math
import pathlib
import re
from collections import Counter, defaultdict, OrderedDict
from statistics import mean

ROOT = pathlib.Path('.')
OUT_DIR = pathlib.Path('experiments/archive/frontier_consolidation/data/checkpoint_complementarity')
OUT_DIR.mkdir(parents=True, exist_ok=True)

EVAL_ROOT = pathlib.Path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval')
A01_EVAL_ROOT = pathlib.Path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval')

SUMMARY_100 = pathlib.Path('experiments/archive/frontier_consolidation/data/compliant_pristine_collate/complianttok_reinvest_seed43022/pristine_collate_complianttok_reinvest_seed43022_summary.json')
PER_TARGET = {
    20: pathlib.Path('experiments/archive/frontier_consolidation/data/legal20m_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_20M.json'),
    70: pathlib.Path('experiments/archive/frontier_consolidation/data/legal_mature_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_70M.json'),
    80: pathlib.Path('experiments/archive/frontier_consolidation/data/legal_mature_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_80M.json'),
}
CKPTS = [20, 70, 80, 100]
DISCRETE_COLUMNS = ['BLiMP', 'Supplement', 'EWoK', 'Entity', 'COMPS', 'GlobalPIQA']
CHEAP7_COLUMNS = ['BLiMP', 'Supplement', 'EWoK', 'Entity', 'COMPS', 'GlobalPIQA', 'Reading']

# Reference column scores from already validated summaries.
COLUMN_SCORES = {
    20: {'BLiMP': 59.69, 'Supplement': 55.45, 'EWoK': 50.73, 'Entity': 18.65, 'COMPS': 50.26, 'GlobalPIQA': 34.195, 'Reading': 8.67},
    70: {'BLiMP': 65.39, 'Supplement': 59.31, 'EWoK': 50.47, 'Entity': 26.98, 'COMPS': 51.82, 'GlobalPIQA': 35.55, 'Reading': 8.74},
    80: {'BLiMP': 66.11, 'Supplement': 60.66, 'EWoK': 51.01, 'Entity': 27.06, 'COMPS': 51.93, 'GlobalPIQA': 35.58, 'Reading': 8.29},
    100: {'BLiMP': 65.8707181799453, 'Supplement': 61.16566092036889, 'EWoK': 50.39323748109589, 'Entity': 27.400833994026197, 'COMPS': 52.00834536316919, 'GlobalPIQA': 36.0631067961165, 'Reading': 8.13816768550987, 'SuperGLUE': 70.27986764740969, 'AoA': 0.0},
}
for d in COLUMN_SCORES.values():
    d['cheap7'] = sum(d[c] for c in CHEAP7_COLUMNS) / 7.0


def load_json(p: pathlib.Path):
    with p.open('r', encoding='utf-8') as f:
        return json.load(f)


def normalize_text(s: str) -> str:
    return re.sub(r'\s+', ' ', str(s).strip())


def pred_map_from_nested(pred_path: pathlib.Path) -> dict[str, str]:
    data = load_json(pred_path)
    out = {}
    for group, rec in data.items():
        preds = rec.get('predictions', []) if isinstance(rec, dict) else []
        for item in preds:
            out[str(item['id'])] = normalize_text(item['pred'])
    return out


def read_jsonl(path: pathlib.Path):
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def get_pred_path(ckpt: int, column_or_sub: str) -> pathlib.Path:
    if ckpt in (20, 70, 80):
        payload = load_json(PER_TARGET[ckpt])
        if column_or_sub == 'GlobalPIQA_parallel':
            return pathlib.Path(payload['tasks']['GlobalPIQA_parallel']['predictions'])
        if column_or_sub == 'GlobalPIQA_nonparallel':
            return pathlib.Path(payload['tasks']['GlobalPIQA_nonparallel']['predictions'])
        return pathlib.Path(payload['tasks'][column_or_sub]['predictions'])
    # 100M paths from pristine summary staged source files.
    summary = load_json(SUMMARY_100)
    # use src paths to avoid relying on symlink resolution
    by_dst = {pathlib.Path(x['dst']).as_posix(): pathlib.Path(x['src']) for x in summary['staged_files']}
    mapping = {
        'BLiMP': 'experiments/archive/frontier_consolidation/data/compliant_pristine_collate/complianttok_reinvest_seed43022/results/hf_model/main/zero_shot/mlm/blimp/blimp_filtered/predictions.json',
        'Supplement': 'experiments/archive/frontier_consolidation/data/compliant_pristine_collate/complianttok_reinvest_seed43022/results/hf_model/main/zero_shot/mlm/blimp/supplement_filtered/predictions.json',
        'EWoK': 'experiments/archive/frontier_consolidation/data/compliant_pristine_collate/complianttok_reinvest_seed43022/results/hf_model/main/zero_shot/mlm/ewok/ewok_filtered/predictions.json',
        'Entity': 'experiments/archive/frontier_consolidation/data/compliant_pristine_collate/complianttok_reinvest_seed43022/results/hf_model/main/zero_shot/mlm/entity_tracking/entity_tracking/predictions.json',
        'COMPS': 'experiments/archive/frontier_consolidation/data/compliant_pristine_collate/complianttok_reinvest_seed43022/results/hf_model/main/zero_shot/mlm/comps/comps/predictions.json',
        'GlobalPIQA_parallel': 'experiments/archive/frontier_consolidation/data/compliant_pristine_collate/complianttok_reinvest_seed43022/results/hf_model/main/zero_shot/mlm/global_piqa_parallel/global_piqa_parallel/predictions.json',
        'GlobalPIQA_nonparallel': 'experiments/archive/frontier_consolidation/data/compliant_pristine_collate/complianttok_reinvest_seed43022/results/hf_model/main/zero_shot/mlm/global_piqa_nonparallel/global_piqa_nonparallel/predictions.json',
    }
    key = mapping[column_or_sub]
    return by_dst.get(key, pathlib.Path(key))


def load_blimp_like(column: str, ckpt: int):
    pred_path = get_pred_path(ckpt, column)
    preds = pred_map_from_nested(pred_path)
    data_dir = A01_EVAL_ROOT / ('blimp_filtered' if column == 'BLiMP' else 'supplement_filtered')
    rows = {}
    for f in sorted(data_dir.glob('*.jsonl')):
        group = f.stem
        for i, row in enumerate(read_jsonl(f)):
            item_id = f'{group}_{i}'
            gold = normalize_text(row['sentence_good'])
            bad = normalize_text(row['sentence_bad'])
            pred = preds.get(item_id)
            rows[item_id] = {
                'column': column, 'group': group, 'pred': pred,
                'gold': gold, 'bad': bad,
                'correct': pred == gold,
                'pred_is_bad': pred == bad,
            }
    return rows


def load_ewok(ckpt: int):
    pred_path = get_pred_path(ckpt, 'EWoK')
    preds = pred_map_from_nested(pred_path)
    data_dir = A01_EVAL_ROOT / 'ewok_filtered'
    rows = {}
    match_counts = Counter()
    for f in sorted(data_dir.glob('*.jsonl')):
        group = f.stem
        for i, row in enumerate(read_jsonl(f)):
            item_id = f'{group}_{i}'
            cand = {
                'c1t1': normalize_text(row['Context1'] + row['Target1']),
                'c1_space_t1': normalize_text(row['Context1'] + ' ' + row['Target1']),
                'c2t1': normalize_text(row['Context2'] + row['Target1']),
                'c2_space_t1': normalize_text(row['Context2'] + ' ' + row['Target1']),
                'c1t2': normalize_text(row['Context1'] + row['Target2']),
                'c2t2': normalize_text(row['Context2'] + row['Target2']),
            }
            pred = preds.get(item_id)
            matched = [k for k, v in cand.items() if pred == v]
            for m in matched:
                match_counts[m] += 1
            correct = bool(set(matched) & {'c1t1', 'c1_space_t1'})
            rows[item_id] = {'column': 'EWoK', 'group': group, 'pred': pred, 'gold': cand['c1_space_t1'], 'correct': correct, 'match': matched[:2]}
    return rows, match_counts


def load_entity(ckpt: int):
    pred_path = get_pred_path(ckpt, 'Entity')
    preds = pred_map_from_nested(pred_path)
    data_dir = A01_EVAL_ROOT / 'entity_tracking'
    rows = {}
    for f in sorted(data_dir.glob('*.jsonl')):
        family = f.stem
        counters = Counter()
        row_cache = []
        for row in read_jsonl(f):
            numops = int(row['numops'])
            group = f'{family}_{numops}_ops'
            idx = counters[group]
            counters[group] += 1
            item_id = f'{group}_{idx}'
            options = [normalize_text(x) for x in row['options']]
            pred = preds.get(item_id)
            # Official entity-tracking files store the correct answer as the first option.
            correct = pred == options[0]
            rows[item_id] = {'column': 'Entity', 'group': group, 'pred': pred, 'gold': options[0], 'correct': correct, 'options': options}
            row_cache.append(row)
    return rows


def load_comps(ckpt: int):
    pred_path = get_pred_path(ckpt, 'COMPS')
    preds = pred_map_from_nested(pred_path)
    files = {
        'wugs': 'comps_wugs.jsonl',
        'wugs_dist_in_between': 'comps_wugs_dist-in-between.jsonl',
        'wugs_dist_before': 'comps_wugs_dist-before.jsonl',
        'base': 'comps_base.jsonl',
    }
    rows = {}
    data_dir = A01_EVAL_ROOT / 'comps'
    for group, fn in files.items():
        for i, row in enumerate(read_jsonl(data_dir / fn)):
            item_id = f'{group}_{i}'
            acc = normalize_text(row['prefix_acceptable'] + ' ' + row['property_phrase'])
            unacc = normalize_text(row['prefix_unacceptable'] + ' ' + row['property_phrase'])
            pred = preds.get(item_id)
            rows[item_id] = {'column': 'COMPS', 'group': group, 'pred': pred, 'gold': acc, 'bad': unacc, 'correct': pred == acc, 'pred_is_bad': pred == unacc}
    return rows


def load_globalpiqa_sub(ckpt: int, sub: str):
    pred_path = get_pred_path(ckpt, sub)
    raw = load_json(pred_path)
    data_file = EVAL_ROOT / ('global_piqa_parallel' if sub.endswith('parallel') and not sub.endswith('nonparallel') else 'global_piqa_nonparallel') / 'eng_latn.jsonl'
    rows = {}
    # build by example_id and by prefixed ids
    eval_rows = {}
    for row in read_jsonl(data_file):
        eval_rows[str(row['example_id'])] = row
    for key, rec in raw.items():
        preds = rec.get('predictions', [])
        if not preds:
            continue
        item = preds[0]
        item_id = str(item['id'])
        # key is the example_id for current files.
        row = eval_rows.get(key) or eval_rows.get(item_id.rsplit('_', 1)[0])
        if row is None:
            # fallback: exact key may include language suffix used in eval file example_id
            row = eval_rows.get(key.replace('_0', ''))
        if row is None:
            continue
        pred = normalize_text(item['pred']).lstrip()
        label = int(row['label'])
        gold = normalize_text(row[f'solution{label}']).lstrip()
        options = []
        for j in range(4 if 'solution2' in row else 2):
            if f'solution{j}' in row:
                options.append(normalize_text(row[f'solution{j}']).lstrip())
        rows[f'{sub}:{key}'] = {'column': 'GlobalPIQA', 'group': sub, 'pred': pred, 'gold': gold, 'correct': pred == gold, 'options': options, 'label': label}
    return rows


def load_globalpiqa(ckpt: int):
    rows = {}
    rows.update(load_globalpiqa_sub(ckpt, 'GlobalPIQA_parallel'))
    rows.update(load_globalpiqa_sub(ckpt, 'GlobalPIQA_nonparallel'))
    return rows


def score(rows):
    return 100.0 * sum(1 for r in rows.values() if r['correct']) / max(1, len(rows))


def load_column(column: str, ckpt: int):
    if column in ('BLiMP', 'Supplement'):
        return load_blimp_like(column, ckpt), {}
    if column == 'EWoK':
        rows, meta = load_ewok(ckpt)
        return rows, {'match_counts': dict(meta)}
    if column == 'Entity':
        return load_entity(ckpt), {}
    if column == 'COMPS':
        return load_comps(ckpt), {}
    if column == 'GlobalPIQA':
        return load_globalpiqa(ckpt), {}
    raise KeyError(column)


def majority_vote(preds_by_ckpt, item_id, prefer_ckpt=100):
    preds = [preds_by_ckpt[c][item_id]['pred'] for c in CKPTS]
    counts = Counter(preds)
    best_count = max(counts.values())
    tied = {p for p, n in counts.items() if n == best_count}
    pref = preds_by_ckpt[prefer_ckpt][item_id]['pred']
    if pref in tied:
        return pref
    for c in reversed(CKPTS):
        p = preds_by_ckpt[c][item_id]['pred']
        if p in tied:
            return p
    return preds[-1]


def main():
    result = OrderedDict()
    all_ckpt_scores = {str(k): COLUMN_SCORES[k] for k in CKPTS}
    result['existing_scores'] = all_ckpt_scores
    result['score_bound'] = {
        'cheap7_100M': COLUMN_SCORES[100]['cheap7'],
        'cheap7_per_column_best_over_20_70_80_100': sum(max(COLUMN_SCORES[c][col] for c in CKPTS) for col in CHEAP7_COLUMNS)/7.0,
    }
    result['score_bound']['delta_best_columns_minus_100M'] = result['score_bound']['cheap7_per_column_best_over_20_70_80_100'] - result['score_bound']['cheap7_100M']

    discrete = OrderedDict()
    item_dump = []
    for col in DISCRETE_COLUMNS:
        preds_by_ckpt = OrderedDict()
        parse_meta = {}
        for ckpt in CKPTS:
            rows, meta = load_column(col, ckpt)
            preds_by_ckpt[ckpt] = rows
            parse_meta[str(ckpt)] = meta
        item_ids = sorted(set.intersection(*(set(v.keys()) for v in preds_by_ckpt.values())))
        missing_counts = {str(ckpt): len(set(preds_by_ckpt[ckpt]) - set(item_ids)) for ckpt in CKPTS}
        ckpt_item_scores = {str(ckpt): score({i: preds_by_ckpt[ckpt][i] for i in item_ids}) for ckpt in CKPTS}
        best_single = max(ckpt_item_scores.values())
        best_ckpt = max(ckpt_item_scores, key=lambda k: ckpt_item_scores[k])
        correct_vectors = {}
        for item_id in item_ids:
            vec = tuple(bool(preds_by_ckpt[c][item_id]['correct']) for c in CKPTS)
            correct_vectors[vec] = correct_vectors.get(vec, 0) + 1
        oracle = 100.0 * sum(1 for i in item_ids if any(preds_by_ckpt[c][i]['correct'] for c in CKPTS)) / max(1, len(item_ids))
        all_correct = 100.0 * sum(1 for i in item_ids if all(preds_by_ckpt[c][i]['correct'] for c in CKPTS)) / max(1, len(item_ids))
        never_correct = 100.0 * sum(1 for i in item_ids if not any(preds_by_ckpt[c][i]['correct'] for c in CKPTS)) / max(1, len(item_ids))
        lost_after_80 = 100.0 * sum(1 for i in item_ids if preds_by_ckpt[80][i]['correct'] and not preds_by_ckpt[100][i]['correct']) / max(1, len(item_ids))
        gained_after_80 = 100.0 * sum(1 for i in item_ids if (not preds_by_ckpt[80][i]['correct']) and preds_by_ckpt[100][i]['correct']) / max(1, len(item_ids))
        formed_by_70_lost_100 = 100.0 * sum(1 for i in item_ids if preds_by_ckpt[70][i]['correct'] and not preds_by_ckpt[100][i]['correct']) / max(1, len(item_ids))
        # majority vote using predicted string; correct if chosen prediction equals the gold for that item.
        vote_correct = 0
        vote_oracle_correct = 0
        for item_id in item_ids:
            vote = majority_vote(preds_by_ckpt, item_id, prefer_ckpt=100)
            if vote == preds_by_ckpt[100][item_id]['gold']:
                vote_correct += 1
            # two-out-of-four correct-state vote, preferring 100M on ties: this is an upper bound for vote over labels when pred strings differ.
            states = [preds_by_ckpt[c][item_id]['correct'] for c in CKPTS]
            if sum(states) > 2 or (sum(states) == 2 and preds_by_ckpt[100][item_id]['correct']):
                vote_oracle_correct += 1
        # Jaccard similarities of error sets to locate fixed errors.
        error_sets = {c: {i for i in item_ids if not preds_by_ckpt[c][i]['correct']} for c in CKPTS}
        pair_jacc = {}
        for a in CKPTS:
            for b in CKPTS:
                if a < b:
                    inter = len(error_sets[a] & error_sets[b])
                    union = len(error_sets[a] | error_sets[b])
                    pair_jacc[f'{a}-{b}'] = inter / union if union else 1.0
        rec = OrderedDict()
        rec['n_items'] = len(item_ids)
        rec['missing_extra_items_by_ckpt'] = missing_counts
        rec['score_from_items'] = ckpt_item_scores
        rec['reference_scores'] = {str(ckpt): COLUMN_SCORES[ckpt][col] for ckpt in CKPTS}
        rec['score_abs_max_error_vs_reference'] = max(abs(ckpt_item_scores[str(ckpt)] - COLUMN_SCORES[ckpt][col]) for ckpt in CKPTS)
        rec['best_single_ckpt'] = best_ckpt
        rec['best_single_score'] = best_single
        rec['oracle_any_checkpoint_score'] = oracle
        rec['oracle_minus_best_single'] = oracle - best_single
        rec['oracle_minus_100M'] = oracle - ckpt_item_scores['100']
        rec['all_checkpoints_correct_pct'] = all_correct
        rec['never_correct_pct'] = never_correct
        rec['lost_80_to_100_pct'] = lost_after_80
        rec['gained_80_to_100_pct'] = gained_after_80
        rec['formed_by_70_lost_at_100_pct'] = formed_by_70_lost_100
        rec['majority_string_vote_score'] = 100.0 * vote_correct / max(1, len(item_ids))
        rec['majority_correct_state_upper_bound'] = 100.0 * vote_oracle_correct / max(1, len(item_ids))
        rec['error_jaccard'] = pair_jacc
        rec['parse_meta'] = parse_meta
        # most common correctness states, e.g. 0/1 over 20,70,80,100
        rec['correct_state_patterns_top'] = [
            {'pattern_20_70_80_100': ''.join('1' if x else '0' for x in k), 'count': v, 'pct': 100.0*v/len(item_ids)}
            for k, v in sorted(correct_vectors.items(), key=lambda kv: -kv[1])[:12]
        ]
        discrete[col] = rec

    # Aggregated discrete-column means with official column weighting.
    aggregate = OrderedDict()
    aggregate['mean_discrete_best_single_score'] = sum(discrete[c]['best_single_score'] for c in DISCRETE_COLUMNS) / len(DISCRETE_COLUMNS)
    aggregate['mean_discrete_100M_score'] = sum(discrete[c]['score_from_items']['100'] for c in DISCRETE_COLUMNS) / len(DISCRETE_COLUMNS)
    aggregate['mean_discrete_oracle_any_checkpoint_score'] = sum(discrete[c]['oracle_any_checkpoint_score'] for c in DISCRETE_COLUMNS) / len(DISCRETE_COLUMNS)
    aggregate['mean_discrete_oracle_minus_100M'] = aggregate['mean_discrete_oracle_any_checkpoint_score'] - aggregate['mean_discrete_100M_score']
    aggregate['mean_discrete_oracle_minus_best_single'] = aggregate['mean_discrete_oracle_any_checkpoint_score'] - aggregate['mean_discrete_best_single_score']
    aggregate['mean_discrete_majority_string_vote_score'] = sum(discrete[c]['majority_string_vote_score'] for c in DISCRETE_COLUMNS) / len(DISCRETE_COLUMNS)
    aggregate['mean_discrete_majority_correct_state_upper_bound'] = sum(discrete[c]['majority_correct_state_upper_bound'] for c in DISCRETE_COLUMNS) / len(DISCRETE_COLUMNS)

    # Cheap7 hypothetical: discrete oracle + best observed Reading across checkpoints, and state-vote + current Reading.
    aggregate['cheap7_discrete_oracle_plus_best_reading'] = (sum(discrete[c]['oracle_any_checkpoint_score'] for c in DISCRETE_COLUMNS) + max(COLUMN_SCORES[k]['Reading'] for k in CKPTS)) / 7.0
    aggregate['cheap7_discrete_oracle_plus_100M_reading'] = (sum(discrete[c]['oracle_any_checkpoint_score'] for c in DISCRETE_COLUMNS) + COLUMN_SCORES[100]['Reading']) / 7.0
    aggregate['cheap7_majority_state_ub_plus_best_reading'] = (sum(discrete[c]['majority_correct_state_upper_bound'] for c in DISCRETE_COLUMNS) + max(COLUMN_SCORES[k]['Reading'] for k in CKPTS)) / 7.0
    aggregate['cheap7_majority_string_vote_plus_best_reading'] = (sum(discrete[c]['majority_string_vote_score'] for c in DISCRETE_COLUMNS) + max(COLUMN_SCORES[k]['Reading'] for k in CKPTS)) / 7.0
    aggregate['cheap7_per_column_best_score_table'] = result['score_bound']['cheap7_per_column_best_over_20_70_80_100']
    aggregate['cheap7_100M_score_table'] = result['score_bound']['cheap7_100M']
    aggregate['needed_cheap7_if_superglue_aoa_flat'] = 43.70291394373706  # 100M cheap7 + 0.6972 approx from research exact gap / cheap cols.
    aggregate['delta_cheap7_table_best_minus_needed'] = aggregate['cheap7_per_column_best_score_table'] - aggregate['needed_cheap7_if_superglue_aoa_flat']
    aggregate['delta_majority_string_vote_plus_best_reading_minus_100M'] = aggregate['cheap7_majority_string_vote_plus_best_reading'] - aggregate['cheap7_100M_score_table']
    aggregate['delta_majority_state_ub_plus_best_reading_minus_100M'] = aggregate['cheap7_majority_state_ub_plus_best_reading'] - aggregate['cheap7_100M_score_table']

    result['discrete_columns'] = discrete
    result['aggregate'] = aggregate

    out_json = OUT_DIR / 'checkpoint_complementarity_summary.json'
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    # TSV for manual inspection
    with (OUT_DIR / 'checkpoint_complementarity_columns.tsv').open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['column','n_items','score20','score70','score80','score100','best_ckpt','best_single','oracle_any','oracle_minus_best','majority_string','majority_state_ub','lost80to100','gained80to100','never_correct'])
        for c, r in discrete.items():
            w.writerow([c, r['n_items'], r['score_from_items']['20'], r['score_from_items']['70'], r['score_from_items']['80'], r['score_from_items']['100'], r['best_single_ckpt'], r['best_single_score'], r['oracle_any_checkpoint_score'], r['oracle_minus_best_single'], r['majority_string_vote_score'], r['majority_correct_state_upper_bound'], r['lost_80_to_100_pct'], r['gained_80_to_100_pct'], r['never_correct_pct']])
    md = []
    md.append('# research checkpoint complementarity across existing legal research checkpoints')
    md.append('')
    md.append('Existing checkpoints: 20M, 70M, 80M, 100M. Discrete columns use item-level saved predictions; Reading uses saved column scores only.')
    md.append('')
    md.append('## Score-table temporal bound')
    for k, v in result['score_bound'].items():
        md.append(f'- {k}: {v:.6f}' if isinstance(v, float) else f'- {k}: {v}')
    md.append('')
    md.append('## Aggregate discrete complementarity')
    for k, v in aggregate.items():
        md.append(f'- {k}: {v:.6f}' if isinstance(v, float) else f'- {k}: {v}')
    md.append('')
    md.append('## Per-column item complementarity')
    md.append('| column | n | 20M | 70M | 80M | 100M | best | oracle any | oracle-best | string vote | state-vote upper | lost80->100 | gained80->100 | never |')
    md.append('|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|')
    for c, r in discrete.items():
        md.append(f"| {c} | {r['n_items']} | {r['score_from_items']['20']:.3f} | {r['score_from_items']['70']:.3f} | {r['score_from_items']['80']:.3f} | {r['score_from_items']['100']:.3f} | {r['best_single_ckpt']} ({r['best_single_score']:.3f}) | {r['oracle_any_checkpoint_score']:.3f} | {r['oracle_minus_best_single']:.3f} | {r['majority_string_vote_score']:.3f} | {r['majority_correct_state_upper_bound']:.3f} | {r['lost_80_to_100_pct']:.3f} | {r['gained_80_to_100_pct']:.3f} | {r['never_correct_pct']:.3f} |")
    md.append('')
    md.append('## Interpretation')
    md.append('- Per-column checkpoint selection is small: the cheap7 score-table best over 20M/70M/80M/100M is only the listed delta above 100M.')
    md.append('- Item-level oracle is intentionally impossible for a submitted model; it measures whether errors are fixed or different. Vote rows measure a simple non-oracle combination of existing predictions.')
    md.append('- If vote or score-table recombination is not close to the needed cheap7, existing fixed checkpoints do not justify attention on checkpoint soups before a new learning signal.')
    out_md = (ROOT / 'research/documents/frontier_consolidation/data/checkpoint_complementarity/checkpoint_complementarity_summary.md')
    out_md.write_text('\n'.join(md), encoding='utf-8')
    print(json.dumps({'status':'CHECKPOINT_COMPLEMENTARITY_DONE','out_json':str(out_json),'out_md':str(out_md),'tsv':str(OUT_DIR/'checkpoint_complementarity_columns.tsv'),'score_bound':result['score_bound'],'aggregate':aggregate}, indent=2))

if __name__ == '__main__':
    main()
