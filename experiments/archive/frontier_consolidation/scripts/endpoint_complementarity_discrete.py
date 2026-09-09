#!/usr/bin/env python3
"""research: saved-prediction complementarity across research, scale1.75, and U256.

CPU/filesystem only. It reuses the validated research item parser to compare
three already-trained endpoints on the same discrete official rows. Majority
vote is a label-free diagnostic of whether endpoint predictions are compatible;
oracle-union is only an upper bound on complementary signal and is not a
submission result.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path('.').resolve()
PARSER_PATH = ROOT / 'experiments/archive/frontier_consolidation/scripts/pairwise_item_flip_analysis.py'
BASE_PAYLOAD = ROOT / 'experiments/archive/frontier_consolidation/data/compliant_full_eval/per_target/complianttok_reinvest_seed43022.json'
SCALE_PAYLOAD = ROOT / 'experiments/archive/frontier_consolidation/data/scale1p75_100M_repaired_merge/staged_full_eval/per_target/scale1p75_100M_seed43022.json'
U256_PAYLOAD = ROOT / 'experiments/archive/frontier_consolidation/data/u256_100M_full_eval_hardened/staged_full_eval/per_target/U256_100M_seed43022.json'
OUT_DIR = ROOT / 'experiments/archive/frontier_consolidation/data/endpoint_complementarity_discrete'
OUT_JSON = OUT_DIR / 'endpoint_complementarity_discrete.json'
OUT_MD = (ROOT / 'research/documents/frontier_consolidation/data/endpoint_complementarity_discrete/endpoint_complementarity_discrete.md')
DISCRETE_COLUMNS = ['BLiMP', 'Supplement', 'EWoK', 'Entity', 'COMPS', 'GlobalPIQA']
CHEAP_COLS = ['BLiMP', 'Supplement', 'EWoK', 'Entity', 'COMPS', 'GlobalPIQA', 'Reading']
REF = {
    'BLiMP': 65.8707181799453,
    'Supplement': 61.16566092036889,
    'EWoK': 50.39323748109589,
    'Entity': 27.400833994026197,
    'COMPS': 52.00834536316919,
    'SuperGLUE': 70.27986764740969,
    'GlobalPIQA': 36.0631067961165,
    'Reading': 8.13816768550987,
    'AoA': 0.0,
    'Overall': 41.257770896404615,
}
SCALE_REF = {
    'BLiMP': 68.63175326978445,
    'Supplement': 62.89517138691106,
    'EWoK': 49.08009312557581,
    'Entity': 27.464548753037228,
    'COMPS': 52.30391659251573,
    'SuperGLUE': 69.33459939303675,
    'GlobalPIQA': 36.10679611650485,
    'Reading': 8.319840190504358,
    'AoA': 0.0,
    'Overall': 41.57074653643003,
}
U256_PARTIAL_REF = {
    'BLiMP': 66.81,
    'Supplement': 60.85,
    'EWoK': 50.39,
    'Entity': 28.43,
    'COMPS': 51.65,
    'GlobalPIQA': 36.135,
    'Reading': 7.325,
    'AoA': 0.0,
}


def now() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def import_parser():
    spec = importlib.util.spec_from_file_location('pairwise_item_flip_analysis', PARSER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot load parser from {PARSER_PATH}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def row_maps(module, payload_path: Path, column: str):
    loader = module.PayloadLoader(payload_path)
    rows, meta = loader.load_column(column)
    return {r.item_id: r for r in rows}, meta, loader.payload


def official_score_from_bools(module, template_rows: list[Any], bools: dict[str, bool], column: str) -> float:
    rows = []
    for r in template_rows:
        # Reuse ItemRow shape so official_score handles UID/group aggregation.
        rows.append(module.ItemRow(r.item_id, r.uid, bool(bools[r.item_id]), r.pred, r.gold, r.column, r.sub, r.meta))
    return float(module.official_score(rows, column))


def majority_correct(preds: list[str | None], corrects: list[bool]) -> bool:
    cnt = Counter(preds)
    # If all three predictions differ, fall back to the first endpoint (research):
    # this is deterministic and label-free, but prevents oracle information use.
    pred, n = cnt.most_common(1)[0]
    return corrects[preds.index(pred)]


def uid_group(module, column: str, row: Any) -> str:
    return module.uid_group(column, row)


def analyze_column(module, column: str) -> dict[str, Any]:
    maps = {}
    metas = {}
    payloads = {}
    for name, path in [('research', BASE_PAYLOAD), ('scale1p75', SCALE_PAYLOAD), ('u256', U256_PAYLOAD)]:
        maps[name], metas[name], payloads[name] = row_maps(module, path, column)
    common = sorted(set(maps['research']) & set(maps['scale1p75']) & set(maps['u256']))
    base_rows = [maps['research'][i] for i in common]
    corrects = {name: {i: bool(maps[name][i].correct) for i in common} for name in maps}
    preds = {name: {i: maps[name][i].pred for i in common} for name in maps}
    score = {name: official_score_from_bools(module, base_rows, corrects[name], column) for name in maps}
    maj_bool = {i: majority_correct([preds['research'][i], preds['scale1p75'][i], preds['u256'][i]], [corrects['research'][i], corrects['scale1p75'][i], corrects['u256'][i]]) for i in common}
    all_wrong = {i: not (corrects['research'][i] or corrects['scale1p75'][i] or corrects['u256'][i]) for i in common}
    oracle_bool = {i: not all_wrong[i] for i in common}
    score['majority3'] = official_score_from_bools(module, base_rows, maj_bool, column)
    score['oracle_union3'] = official_score_from_bools(module, base_rows, oracle_bool, column)

    pattern_counts = Counter()
    pair_disagree = Counter()
    pair_unique_repair = Counter()
    examples = {'scale_only_gain': [], 'u256_only_gain': [], 'both_gain': [], 'both_loss': [], 'all_wrong': []}
    group = defaultdict(lambda: Counter())
    for i in common:
        b = corrects['research'][i]
        s = corrects['scale1p75'][i]
        u = corrects['u256'][i]
        pat = f"B{int(b)}S{int(s)}U{int(u)}"
        pattern_counts[pat] += 1
        if s != b:
            pair_disagree['scale_vs_step35'] += 1
        if u != b:
            pair_disagree['u256_vs_step35'] += 1
        if s != u:
            pair_disagree['scale_vs_u256'] += 1
        if (not b) and s and (not u):
            pair_unique_repair['scale_only'] += 1
        if (not b) and u and (not s):
            pair_unique_repair['u256_only'] += 1
        if (not b) and s and u:
            pair_unique_repair['both_repair'] += 1
        if b and (not s) and (not u):
            pair_unique_repair['both_damage'] += 1
        g = uid_group(module, column, maps['research'][i])
        group[g]['n'] += 1
        group[g]['correct'] += int(b)
        group[g]['scale_correct'] += int(s)
        group[g]['u256_correct'] += int(u)
        group[g]['majority_correct'] += int(maj_bool[i])
        group[g]['oracle_correct'] += int(oracle_bool[i])
        group[g]['scale_unique_repair'] += int((not b) and s and (not u))
        group[g]['u256_unique_repair'] += int((not b) and u and (not s))
        group[g]['both_repair'] += int((not b) and s and u)
        group[g]['both_damage'] += int(b and (not s) and (not u))
        key = None
        if (not b) and s and (not u):
            key = 'scale_only_gain'
        elif (not b) and u and (not s):
            key = 'u256_only_gain'
        elif (not b) and s and u:
            key = 'both_gain'
        elif b and (not s) and (not u):
            key = 'both_loss'
        elif (not b) and (not s) and (not u):
            key = 'all_wrong'
        if key and len(examples[key]) < 8:
            r = maps['research'][i]
            examples[key].append({'item_id': i, 'uid': r.uid, 'sub': r.sub, 'gold': r.gold, 'pred': preds['research'][i], 'scale_pred': preds['scale1p75'][i], 'u256_pred': preds['u256'][i], 'meta': r.meta})
    group_rows = []
    for g, c in group.items():
        n = c['n']
        group_rows.append({
            'group': g,
            'n': n,
            'pct': 100*c['correct']/n,
            'scale1p75_pct': 100*c['scale_correct']/n,
            'u256_pct': 100*c['u256_correct']/n,
            'majority3_pct': 100*c['majority_correct']/n,
            'oracle3_pct': 100*c['oracle_correct']/n,
            'scale_minus_step35_pct': 100*(c['scale_correct']-c['correct'])/n,
            'u256_minus_step35_pct': 100*(c['u256_correct']-c['correct'])/n,
            'majority_minus_best_pct': 100*c['majority_correct']/n - max(100*c['correct']/n, 100*c['scale_correct']/n, 100*c['u256_correct']/n),
            'oracle_minus_best_pct': 100*c['oracle_correct']/n - max(100*c['correct']/n, 100*c['scale_correct']/n, 100*c['u256_correct']/n),
            'scale_unique_repair': c['scale_unique_repair'],
            'u256_unique_repair': c['u256_unique_repair'],
            'both_repair': c['both_repair'],
            'both_damage': c['both_damage'],
        })
    group_rows_by_oracle = sorted(group_rows, key=lambda x: (x['oracle_minus_best_pct'], x['n']), reverse=True)
    group_rows_by_majority = sorted(group_rows, key=lambda x: (x['majority_minus_best_pct'], x['n']), reverse=True)

    return {
        'column': column,
        'n_common': len(common),
        'scores_reconstructed': score,
        'score_deltas_vs_step35': {k: v - score['research'] for k, v in score.items() if k != 'research'},
        'score_deltas_vs_best_single': {k: v - max(score['research'], score['scale1p75'], score['u256']) for k, v in score.items() if k in ['majority3', 'oracle_union3']},
        'pattern_counts': dict(pattern_counts),
        'pattern_pct': {k: 100*v/len(common) for k, v in pattern_counts.items()},
        'pair_disagree_counts': dict(pair_disagree),
        'pair_disagree_pct': {k: 100*v/len(common) for k, v in pair_disagree.items()},
        'pair_unique_repair_counts': dict(pair_unique_repair),
        'groups_by_oracle_headroom': group_rows_by_oracle[:20],
        'groups_by_majority_excess': group_rows_by_majority[:20],
        'worst_groups_by_majority_excess': sorted(group_rows, key=lambda x: (x['majority_minus_best_pct'], -x['n']))[:20],
        'examples': examples,
        'metas': {k: v for k, v in metas.items()},
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    module = import_parser()
    by_col = {col: analyze_column(module, col) for col in DISCRETE_COLUMNS}
    single_means = {
        name: mean(by_col[c]['scores_reconstructed'][name] for c in DISCRETE_COLUMNS)
        for name in ['research', 'scale1p75', 'u256', 'majority3', 'oracle_union3']
    }
    # Cheap projections: only discrete columns; Reading/SuperGLUE/AoA are not part of this diagnostic.
    aggregate = {
        'single_discrete_means': single_means,
        'majority3_minus_best_single_discrete_mean': single_means['majority3'] - max(single_means[k] for k in ['research', 'scale1p75', 'u256']),
        'oracle3_minus_best_single_discrete_mean': single_means['oracle_union3'] - max(single_means[k] for k in ['research', 'scale1p75', 'u256']),
        'columns_where_majority_beats_best_single': [c for c in DISCRETE_COLUMNS if by_col[c]['scores_reconstructed']['majority3'] > max(by_col[c]['scores_reconstructed'][k] for k in ['research','scale1p75','u256']) + 1e-9],
        'columns_where_oracle_headroom_gt_5': [c for c in DISCRETE_COLUMNS if by_col[c]['score_deltas_vs_best_single']['oracle_union3'] > 5.0],
        'total_common_rows': sum(by_col[c]['n_common'] for c in DISCRETE_COLUMNS),
        'interpretation': 'Majority is a label-free diagnostic only; oracle-union is an impossible upper bound showing whether errors are complementary. Additive model/data routes need majority or some plausible label-free rule to approach the oracle headroom, not merely high oracle headroom.',
    }
    out = {
        'status': 'ENDPOINT_COMPLEMENTARITY_DISCRETE',
        'created_utc': now(),
        'inputs': {
            'payload': rel(BASE_PAYLOAD),
            'scale1p75_payload': rel(SCALE_PAYLOAD),
            'u256_payload': rel(U256_PAYLOAD),
            'parser': rel(PARSER_PATH),
        },
        'columns': by_col,
        'aggregate': aggregate,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    lines = ['# research — endpoint complementarity on saved discrete predictions', '']
    lines.append('Majority vote is a label-free diagnostic; oracle-union is only an impossible upper bound on complementary decisions.')
    lines.append('')
    lines.append('| Column | research | scale1.75 | U256 | Majority3 | Oracle3 | Majority-best | Oracle-best | key patterns |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---|')
    for c in DISCRETE_COLUMNS:
        r = by_col[c]
        s = r['scores_reconstructed']
        best = max(s[k] for k in ['research','scale1p75','u256'])
        patterns = r['pattern_counts']
        key = ', '.join(f'{k}:{v}' for k, v in sorted(patterns.items()) if k in ['B0S1U0','B0S0U1','B0S1U1','B1S0U0'])
        lines.append(f"| {c} | {s['research']:.3f} | {s['scale1p75']:.3f} | {s['u256']:.3f} | {s['majority3']:.3f} | {s['oracle_union3']:.3f} | {s['majority3']-best:+.3f} | {s['oracle_union3']-best:+.3f} | {key} |")
    lines.append('')
    lines.append('## Aggregate')
    for k, v in aggregate.items():
        if k != 'interpretation':
            lines.append(f'- {k}: {v}')
    lines.append(f"- interpretation: {aggregate['interpretation']}")
    lines.append('')
    lines.append('## Column-level scientific readout')
    for c in DISCRETE_COLUMNS:
        r = by_col[c]
        lines.append(f"### {c}")
        lines.append(f"- Pair disagreement pct: {r['pair_disagree_pct']}")
        lines.append(f"- Unique repair counts: {r['pair_unique_repair_counts']}")
        lines.append(f"- Largest oracle headroom groups: {r['groups_by_oracle_headroom'][:5]}")
        lines.append(f"- Majority weak groups: {r['worst_groups_by_majority_excess'][:5]}")
    lines.append('')
    lines.append(f'JSON: `{rel(OUT_JSON)}`')
    OUT_MD.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    print(json.dumps({
        'status': out['status'],
        'out_json': rel(OUT_JSON),
        'out_md': rel(OUT_MD),
        'single_discrete_means': single_means,
        'majority3_minus_best_single_discrete_mean': aggregate['majority3_minus_best_single_discrete_mean'],
        'oracle3_minus_best_single_discrete_mean': aggregate['oracle3_minus_best_single_discrete_mean'],
        'columns_where_majority_beats_best_single': aggregate['columns_where_majority_beats_best_single'],
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
