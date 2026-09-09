#!/usr/bin/env python3
"""research: uncertainty and error-structure readout for completed endpoints.

CPU/filesystem only. Reads saved predictions for research, scale1.75, and U256,
then computes group-aware bootstrap intervals for endpoint deltas and pairwise
error correlations. This prevents tiny changes from being overinterpreted and
summarizes whether endpoint errors are structured enough to support combination
routes.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

ROOT = Path('.').resolve()
PARSER_PATH = ROOT / 'experiments/archive/frontier_consolidation/scripts/pairwise_item_flip_analysis.py'
PAYLOADS = {
    'research': ROOT / 'experiments/archive/frontier_consolidation/data/compliant_full_eval/per_target/complianttok_reinvest_seed43022.json',
    'scale1p75': ROOT / 'experiments/archive/frontier_consolidation/data/scale1p75_100M_repaired_merge/staged_full_eval/per_target/scale1p75_100M_seed43022.json',
    'u256': ROOT / 'experiments/archive/frontier_consolidation/data/u256_100M_full_eval_hardened/staged_full_eval/per_target/U256_100M_seed43022.json',
}
OUT_DIR = ROOT / 'experiments/archive/frontier_consolidation/data/endpoint_uncertainty_error_structure'
OUT_JSON = OUT_DIR / 'endpoint_uncertainty_error_structure.json'
OUT_MD = (ROOT / 'research/documents/frontier_consolidation/data/endpoint_uncertainty_error_structure/endpoint_uncertainty_error_structure.md')
DISCRETE_COLUMNS = ['BLiMP', 'Supplement', 'EWoK', 'Entity', 'COMPS', 'GlobalPIQA']
ENDPOINTS = ['research', 'scale1p75', 'u256', 'majority3']
PAIR_NAMES = [('scale1p75', 'research'), ('u256', 'research'), ('scale1p75', 'u256'), ('majority3', 'scale1p75')]
N_BOOT = 2000
RNG_SEED = 119119


def now() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def import_parser():
    spec = importlib.util.spec_from_file_location('pairwise_item_flip_analysis_for_uncertainty', PARSER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot load parser from {PARSER_PATH}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def majority_pred_correct(preds: list[str | None], corrects: list[bool]) -> bool:
    counts = Counter(preds)
    pred, _ = counts.most_common(1)[0]
    return bool(corrects[preds.index(pred)])


def load_maps(module, column: str):
    maps = {}
    metas = {}
    for name, path in PAYLOADS.items():
        loader = module.PayloadLoader(path)
        rows, meta = loader.load_column(column)
        maps[name] = {r.item_id: r for r in rows}
        metas[name] = meta
    common = sorted(set.intersection(*(set(m) for m in maps.values())))
    return maps, metas, common


def unit_key(module, column: str, row: Any) -> tuple[str, str]:
    if column == 'Entity':
        split = str(row.uid).split('_')[0]
        return split, str(row.uid)
    if column == 'GlobalPIQA':
        return str(row.sub), str(row.uid)
    return 'all', module.uid_group(column, row)


def build_unit_accs(module, column: str, maps: dict[str, dict[str, Any]], common: list[str]) -> tuple[dict[str, dict[tuple[str, str], float]], dict[str, list[tuple[str, str]]], dict[str, dict[str, bool]], dict[str, dict[str, str | None]]]:
    corrects: dict[str, dict[str, bool]] = {k: {} for k in ENDPOINTS}
    preds: dict[str, dict[str, str | None]] = {k: {} for k in ['research', 'scale1p75', 'u256']}
    units = defaultdict(lambda: defaultdict(list))
    for item_id in common:
        base_row = maps['research'][item_id]
        key = unit_key(module, column, base_row)
        b = bool(maps['research'][item_id].correct)
        s = bool(maps['scale1p75'][item_id].correct)
        u = bool(maps['u256'][item_id].correct)
        bp = maps['research'][item_id].pred
        sp = maps['scale1p75'][item_id].pred
        up = maps['u256'][item_id].pred
        corrects['research'][item_id] = b
        corrects['scale1p75'][item_id] = s
        corrects['u256'][item_id] = u
        corrects['majority3'][item_id] = majority_pred_correct([bp, sp, up], [b, s, u])
        preds['research'][item_id] = bp
        preds['scale1p75'][item_id] = sp
        preds['u256'][item_id] = up
        for ep in ENDPOINTS:
            units[ep][key].append(float(corrects[ep][item_id]))
    accs = {ep: {key: float(np.mean(vals)) for key, vals in units[ep].items()} for ep in ENDPOINTS}
    strata = defaultdict(list)
    for key in accs['research'].keys():
        strata[key[0]].append(key)
    return accs, dict(strata), corrects, preds


def score_from_unit_accs(column: str, acc: dict[tuple[str, str], float], strata: dict[str, list[tuple[str, str]]]) -> float:
    if column in ('Entity', 'GlobalPIQA'):
        vals = []
        for keys in strata.values():
            if keys:
                vals.append(float(np.mean([acc[k] for k in keys])))
        return 100.0 * float(np.mean(vals)) if vals else float('nan')
    keys = [k for keys in strata.values() for k in keys]
    return 100.0 * float(np.mean([acc[k] for k in keys])) if keys else float('nan')


def bootstrap_scores(column: str, accs: dict[str, dict[tuple[str, str], float]], strata: dict[str, list[tuple[str, str]]], rng: np.random.Generator) -> dict[str, np.ndarray]:
    samples = {ep: np.empty(N_BOOT, dtype=np.float64) for ep in ENDPOINTS}
    strata_items = {s: list(keys) for s, keys in strata.items()}
    for b in range(N_BOOT):
        sampled_by_stratum: dict[str, list[tuple[str, str]]] = {}
        for s, keys in strata_items.items():
            idx = rng.integers(0, len(keys), size=len(keys))
            sampled_by_stratum[s] = [keys[int(i)] for i in idx]
        for ep in ENDPOINTS:
            vals = []
            if column in ('Entity', 'GlobalPIQA'):
                for s, skeys in sampled_by_stratum.items():
                    vals.append(float(np.mean([accs[ep][k] for k in skeys])))
                samples[ep][b] = 100.0 * float(np.mean(vals))
            else:
                all_keys = sampled_by_stratum.get('all', [])
                samples[ep][b] = 100.0 * float(np.mean([accs[ep][k] for k in all_keys]))
    return samples


def interval(arr: np.ndarray) -> dict[str, float]:
    return {
        'mean': float(np.mean(arr)),
        'p2p5': float(np.quantile(arr, 0.025)),
        'p50': float(np.quantile(arr, 0.5)),
        'p97p5': float(np.quantile(arr, 0.975)),
        'p_gt_0': float(np.mean(arr > 0.0)),
    }


def phi_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def q_stat(correct_a: np.ndarray, correct_b: np.ndarray) -> float | None:
    # Kuncheva-style Q over correct/incorrect contingency: N11 both correct,
    # N00 both incorrect, N10/N01 one correct.
    a = correct_a.astype(bool)
    b = correct_b.astype(bool)
    n11 = int(np.sum(a & b))
    n00 = int(np.sum((~a) & (~b)))
    n10 = int(np.sum(a & (~b)))
    n01 = int(np.sum((~a) & b))
    den = n11 * n00 + n10 * n01
    if den == 0:
        return None
    return float((n11 * n00 - n10 * n01) / den)


def pair_error_stats(common: list[str], corrects: dict[str, dict[str, bool]]) -> dict[str, Any]:
    out = {}
    for a, b in [('scale1p75','research'), ('u256','research'), ('scale1p75','u256')]:
        ca = np.array([corrects[a][i] for i in common], dtype=bool)
        cb = np.array([corrects[b][i] for i in common], dtype=bool)
        ea = (~ca).astype(float)
        eb = (~cb).astype(float)
        out[f'{a}_vs_{b}'] = {
            'n': len(common),
            'both_correct': int(np.sum(ca & cb)),
            'both_wrong': int(np.sum((~ca) & (~cb))),
            'a_correct_b_wrong': int(np.sum(ca & (~cb))),
            'a_wrong_b_correct': int(np.sum((~ca) & cb)),
            'disagreement_pct': float(100.0 * np.mean(ca != cb)),
            'error_phi': phi_corr(ea, eb),
            'q_stat': q_stat(ca, cb),
        }
    return out


def analyze_column(module, column: str, rng: np.random.Generator) -> dict[str, Any]:
    maps, metas, common = load_maps(module, column)
    accs, strata, corrects, preds = build_unit_accs(module, column, maps, common)
    point = {ep: score_from_unit_accs(column, accs[ep], strata) for ep in ENDPOINTS}
    boot = bootstrap_scores(column, accs, strata, rng)
    delta_intervals = {}
    for a, b in PAIR_NAMES:
        arr = boot[a] - boot[b]
        rec = interval(arr)
        rec['point_delta'] = point[a] - point[b]
        delta_intervals[f'{a}_minus_{b}'] = rec
    patterns = Counter()
    for i in common:
        patterns[f"B{int(corrects['research'][i])}S{int(corrects['scale1p75'][i])}U{int(corrects['u256'][i])}"] += 1
    return {
        'column': column,
        'n_common': len(common),
        'n_units_by_stratum': {s: len(keys) for s, keys in strata.items()},
        'point_scores': point,
        'delta_intervals': delta_intervals,
        'error_pairs': pair_error_stats(common, corrects),
        'patterns': dict(patterns),
        'pattern_pct': {k: 100.0*v/len(common) for k, v in patterns.items()},
        'metas': metas,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    module = import_parser()
    rng = np.random.default_rng(RNG_SEED)
    columns = {c: analyze_column(module, c, rng) for c in DISCRETE_COLUMNS}
    mean_points = {ep: mean(columns[c]['point_scores'][ep] for c in DISCRETE_COLUMNS) for ep in ENDPOINTS}
    mean_delta = {f'{a}_minus_{b}': mean(columns[c]['delta_intervals'][f'{a}_minus_{b}']['point_delta'] for c in DISCRETE_COLUMNS) for a, b in PAIR_NAMES}
    out = {
        'status': 'ENDPOINT_UNCERTAINTY_ERROR_STRUCTURE',
        'created_utc': now(),
        'n_boot': N_BOOT,
        'rng_seed': RNG_SEED,
        'inputs': {k: rel(v) for k, v in PAYLOADS.items()} | {'parser': rel(PARSER_PATH)},
        'columns': columns,
        'aggregate': {
            'mean_point_scores_six_discrete': mean_points,
            'mean_point_deltas_six_discrete': mean_delta,
            'interpretation': 'Intervals are within-endpoint saved-prediction resampling, not pretraining-seed uncertainty. Tiny U256 EWoK/GlobalPIQA movements should be read as near-flat unless the interval strongly excludes zero.',
        },
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    lines = ['# research — endpoint uncertainty and error structure', '']
    lines.append('Bootstrap intervals resample official scoring units within each column. They quantify saved-prediction uncertainty, not pretraining-seed variance.')
    lines.append('')
    lines.append('| Column | scale-research Δ [2.5,50,97.5] | U256-research Δ [2.5,50,97.5] | majority-scale Δ [2.5,50,97.5] | U256/research error phi | scale/U256 error phi |')
    lines.append('|---|---:|---:|---:|---:|---:|')
    for c in DISCRETE_COLUMNS:
        r = columns[c]
        def fmt(key: str) -> str:
            q = r['delta_intervals'][key]
            return f"{q['point_delta']:+.3f} [{q['p2p5']:+.3f},{q['p50']:+.3f},{q['p97p5']:+.3f}]"
        e_u_b = r['error_pairs']['u256_vs_step35']['error_phi']
        e_s_u = r['error_pairs']['scale1p75_vs_u256']['error_phi']
        lines.append(f"| {c} | {fmt('scale1p75_minus_step35')} | {fmt('u256_minus_step35')} | {fmt('majority3_minus_scale1p75')} | {e_u_b if e_u_b is not None else 'n/a'} | {e_s_u if e_s_u is not None else 'n/a'} |")
    lines.append('')
    lines.append('## Aggregate')
    lines.append(f"- six-discrete point means: {mean_points}")
    lines.append(f"- six-discrete point deltas: {mean_delta}")
    lines.append('- Large oracle complementarity with high pairwise error correlation and weak hard majority is consistent with multidirectional boundary rotation, not an immediately exploitable endpoint combination.')
    lines.append('')
    lines.append('## Error-pair details')
    for c in DISCRETE_COLUMNS:
        lines.append(f'### {c}')
        for pair, rec in columns[c]['error_pairs'].items():
            lines.append(f"- {pair}: disagreement={rec['disagreement_pct']:.3f}%, error_phi={rec['error_phi']}, q={rec['q_stat']}, both_wrong={rec['both_wrong']}, one_correct={rec['a_correct_b_wrong'] + rec['a_wrong_b_correct']}")
    lines.append('')
    lines.append(f'JSON: `{rel(OUT_JSON)}`')
    OUT_MD.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({
        'status': out['status'],
        'out_json': rel(OUT_JSON),
        'out_md': rel(OUT_MD),
        'mean_point_scores_six_discrete': mean_points,
        'mean_point_deltas_six_discrete': mean_delta,
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
