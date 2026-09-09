#!/usr/bin/env python3
"""Corrected research synthesis.

This repairs the first research synthesis' broad-score extraction bug: the per-target
official-compatible evaluator stores GlobalPIQA under `official_overall.scores` and
Reading as a nested task score, so directly averaging only top-level task scores
incorrectly omitted GlobalPIQA/Reading. This script reuses the validated extractor
from `screen_readout.py` and adds correlation-style bridge-vs-hard readouts.
"""
from __future__ import annotations

import importlib.util
import json
import math
import pathlib
import statistics
import sys
from typing import Any

ROOT = pathlib.Path.cwd()
WS = ROOT / 'experiments/archive/representation_and_objectives'
OUT_DIR = WS / 'data/routeB_bridge_synthesis'
OUT_DIR.mkdir(parents=True, exist_ok=True)
NOTE = WS / 'notes/routeB_bridge_predictive_value.md'
CORR_NOTE = WS / 'notes/routeB_bridge_predictive_value_corrected.md'

ROLE_DIR = WS / 'data/role_screen_readout'
READ_ROOT = WS / 'data/small_update_hard_surface_readout'
SMALL_ROOT = WS / 'data/small_exchange_update'
EXTRACTOR_SCRIPT = WS / 'scripts/screen_readout.py'


def load(path: pathlib.Path) -> Any:
    if not path.exists():
        return {'MISSING': str(path)}
    return json.loads(path.read_text())


def load_extractor():
    spec = importlib.util.spec_from_file_location('screen_readout_extract', EXTRACTOR_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['screen_readout_extract'] = mod
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod.extract_score_payload

extract_score_payload = load_extractor()

ROLE_CHEAP = {
    'anchor_fixed256_80M': ROLE_DIR / 'cheap_anchor/cheap_eval/per_target/anchor_fixed256_80M.json',
    'role_switch_80M': ROLE_DIR / 'cheap_role_switch/cheap_eval/per_target/role_switch_80M.json',
    'role_fixed_80M': ROLE_DIR / 'cheap_role_fixed/cheap_eval/per_target/role_fixed_80M.json',
}
ROLE_GP = {
    'anchor_fixed256_80M': ROLE_DIR / 'globalpiqa_anchor/globalpiqa_margin_summary.json',
    'role_switch_80M': ROLE_DIR / 'globalpiqa_role_switch/globalpiqa_margin_summary.json',
    'role_fixed_80M': ROLE_DIR / 'globalpiqa_role_fixed/globalpiqa_margin_summary.json',
}
ROLE_EWOK = {
    'anchor_fixed256_80M': ROLE_DIR / 'ewok_anchor/ewok_interaction_summary.json',
    'role_switch_80M': ROLE_DIR / 'ewok_role_switch/ewok_interaction_summary.json',
    'role_fixed_80M': ROLE_DIR / 'ewok_role_fixed/ewok_interaction_summary.json',
}
ROLE_BRIDGE = {
    'anchor_fixed256_80M': WS / 'data/naturalistic_exchange_bridge_readout/naturalistic_bridge_readout_summary_score.json',
    'role_switch_80M': WS / 'data/naturalistic_exchange_bridge_readout_role_switch/naturalistic_bridge_readout_summary_score.json',
    'role_fixed_80M': WS / 'data/naturalistic_exchange_bridge_readout_role_fixed/naturalistic_bridge_readout_summary_score.json',
}
ROLE_SYN = ROLE_DIR / 'packet_synthetic_cpu/packet_synthetic_readout_summary.json'

SMALL = {
    'anchor80_exchange_lr5e-5_u80': {
        'summary': SMALL_ROOT / 'anchor80_exchange_lr5e-5_u80/small_update_summary.json',
        'gp': READ_ROOT / 'gp_anchor80_exchange_lr5e-5_u80/anchor80_exchange_lr5e-5_u80/globalpiqa/anchor80_exchange_lr5e-5_u80_globalpiqa_summary.json',
        'ewok': READ_ROOT / 'ewok_anchor80_exchange_lr5e-5_u80/anchor80_exchange_lr5e-5_u80/ewok/anchor80_exchange_lr5e-5_u80_ewok_summary.json',
    },
    'anchor80_correct_ce_lr5e-5_u80': {
        'summary': SMALL_ROOT / 'anchor80_correct_ce_lr5e-5_u80/small_update_summary.json',
        'gp': READ_ROOT / 'gp_anchor80_correct_ce_lr5e-5_u80/anchor80_correct_ce_lr5e-5_u80/globalpiqa/anchor80_correct_ce_lr5e-5_u80_globalpiqa_summary.json',
        'ewok': READ_ROOT / 'ewok_anchor80_correct_ce_lr5e-5_u80/anchor80_correct_ce_lr5e-5_u80/ewok/anchor80_correct_ce_lr5e-5_u80_ewok_summary.json',
    },
    'anchor80_fixed_ce_lr5e-5_u80': {
        'summary': SMALL_ROOT / 'anchor80_fixed_ce_lr5e-5_u80/small_update_summary.json',
        'gp': READ_ROOT / 'gp_anchor80_fixed_ce_lr5e-5_u80/anchor80_fixed_ce_lr5e-5_u80/globalpiqa/anchor80_fixed_ce_lr5e-5_u80_globalpiqa_summary.json',
        'ewok': READ_ROOT / 'ewok_anchor80_fixed_ce_lr5e-5_u80/anchor80_fixed_ce_lr5e-5_u80/ewok/anchor80_fixed_ce_lr5e-5_u80_ewok_summary.json',
    },
    'pareto_exchange_lr2e-5_u20': {
        'summary': SMALL_ROOT / 'pareto_anchor80_exchange_lr2e-5_u20/small_update_summary.json',
        'gp': READ_ROOT / 'gp_pareto_exchange_lr2e-5_u20/pareto_exchange_lr2e-5_u20/globalpiqa/pareto_exchange_lr2e-5_u20_globalpiqa_summary.json',
        'ewok': READ_ROOT / 'ewok_pareto_exchange_lr2e-5_u20/pareto_exchange_lr2e-5_u20/ewok/pareto_exchange_lr2e-5_u20_ewok_summary.json',
    },
    'pareto_correct_ce_lr2e-5_u20': {
        'summary': SMALL_ROOT / 'pareto_anchor80_correct_ce_lr2e-5_u20/small_update_summary.json',
        'gp': READ_ROOT / 'gp_pareto_correct_ce_lr2e-5_u20/pareto_correct_ce_lr2e-5_u20/globalpiqa/pareto_correct_ce_lr2e-5_u20_globalpiqa_summary.json',
        'ewok': READ_ROOT / 'ewok_pareto_correct_ce_lr2e-5_u20/pareto_correct_ce_lr2e-5_u20/ewok/pareto_correct_ce_lr2e-5_u20_ewok_summary.json',
    },
    'pareto_exchange_lr1e-5_u40': {
        'summary': SMALL_ROOT / 'pareto_anchor80_exchange_lr1e-5_u40/small_update_summary.json',
        'gp': READ_ROOT / 'gp_pareto_exchange_lr1e-5_u40/pareto_exchange_lr1e-5_u40/globalpiqa/pareto_exchange_lr1e-5_u40_globalpiqa_summary.json',
        'ewok': READ_ROOT / 'ewok_pareto_exchange_lr1e-5_u40/pareto_exchange_lr1e-5_u40/ewok/pareto_exchange_lr1e-5_u40_ewok_summary.json',
    },
    'pareto_correct_ce_lr1e-5_u40': {
        'summary': SMALL_ROOT / 'pareto_anchor80_correct_ce_lr1e-5_u40/small_update_summary.json',
        'gp': READ_ROOT / 'gp_pareto_correct_ce_lr1e-5_u40/pareto_correct_ce_lr1e-5_u40/globalpiqa/pareto_correct_ce_lr1e-5_u40_globalpiqa_summary.json',
        'ewok': READ_ROOT / 'ewok_pareto_correct_ce_lr1e-5_u40_retry/pareto_correct_ce_lr1e-5_u40/ewok/pareto_correct_ce_lr1e-5_u40_ewok_summary.json',
    },
}


def gp_summary(path: pathlib.Path) -> dict[str, Any]:
    d = load(path)
    if 'MISSING' in d:
        return d
    if 'modes' in d:
        modes = d['modes']
    else:
        modes = next(iter(d.get('targets', {}).values())).get('modes', {})
    par = modes.get('parallel', {})
    if 'summary' in par:
        par = par['summary']
    aw = par.get('always_wrong_subset') or {}
    return {
        'parallel_acc': par.get('accuracy'),
        'rank_counts': par.get('correct_rank_counts'),
        'hard52_acc': aw.get('accuracy'),
        'hard52_rank_counts': aw.get('correct_rank_counts'),
        'hard52_mean_top_minus_correct': aw.get('mean_top_minus_correct'),
        'hard52_small_wrong_margin_le_0p50': aw.get('small_wrong_margin_le_0p50_nats'),
        'path': str(path),
    }


def ewok_summary(path: pathlib.Path) -> dict[str, Any]:
    d = load(path)
    if 'MISSING' in d:
        return d
    if 'result' in d:
        s = d['result']['summary']
    else:
        s = next(iter(d.get('targets', {}).values())).get('summary', {})
    return {
        'accuracy': s.get('accuracy'),
        'saved_wrong': s.get('saved_wrong'),
        'stable_failure': s.get('stable_failure'),
        'stable_failure_frac_all': s.get('stable_failure_frac_all'),
        'stable_failure_frac_wrong': s.get('stable_failure_frac_wrong'),
        'interaction_sum_wrong_mean': (s.get('interaction_sum_wrong') or {}).get('mean'),
        'path': str(path),
    }


def bridge_existing(path: pathlib.Path, label: str) -> dict[str, Any]:
    d = load(path)
    if 'MISSING' in d:
        return d
    if 'target_summaries' in d:
        # role-specific readout files
        s = d['target_summaries'].get(label) or next(iter(d['target_summaries'].values()))
    else:
        # original multi-target readout
        key = 'anchor_80M' if label == 'anchor_fixed256_80M' else label
        s = d.get('target_summaries', {}).get(key) or d.get('summaries', {}).get(key) or next(iter(d.get('target_summaries', d.get('summaries', {})).values()))
    o = s.get('overall', {}) if isinstance(s, dict) else {}
    return {'M': o.get('M_mean'), 'both': o.get('both_correct'), 'path': str(path)}


def small_update_summary(path: pathlib.Path) -> dict[str, Any]:
    d = load(path)
    if 'MISSING' in d:
        return d
    def get(keys, default=None):
        cur = d
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur
    before = get(['before', 'bridge', 'overall'], {}) or {}
    after = get(['after', 'bridge', 'overall'], {}) or {}
    bm, am = before.get('M_mean'), after.get('M_mean')
    bb, ab = before.get('both_correct'), after.get('both_correct')
    return {
        'objective': get(['args', 'objective']),
        'lr': get(['args', 'lr']),
        'updates': get(['args', 'updates']),
        'bridge_delta_M': am - bm if isinstance(am, (int, float)) and isinstance(bm, (int, float)) else None,
        'bridge_delta_both': ab - bb if isinstance(ab, (int, float)) and isinstance(bb, (int, float)) else None,
        'bridge_after_M': am,
        'bridge_after_both': ab,
        'clean_loss_delta': get(['clean_probe', 'delta_loss_per_masked_token']),
        'path': str(path),
    }


def delta(a: Any, b: Any):
    return b - a if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None


def corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mx = statistics.fmean(xs); my = statistics.fmean(ys)
    vx = sum((x-mx)**2 for x in xs); vy = sum((y-my)**2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x-mx)*(y-my) for x,y in zip(xs,ys))/math.sqrt(vx*vy)

# Role screen
role_cheap = {k: extract_score_payload(load(p)) for k, p in ROLE_CHEAP.items()}
role_gp = {k: gp_summary(p) for k,p in ROLE_GP.items()}
role_ew = {k: ewok_summary(p) for k,p in ROLE_EWOK.items()}
role_bridge = {k: bridge_existing(p, k) for k,p in ROLE_BRIDGE.items()}
role_syn = load(ROLE_SYN).get('summaries', {}) if ROLE_SYN.exists() else {}

# Small updates
anchor_gp = role_gp['anchor_fixed256_80M']
anchor_ew = role_ew['anchor_fixed256_80M']
small = {}
rows = []
for name, paths in SMALL.items():
    ss = small_update_summary(paths['summary'])
    gs = gp_summary(paths['gp'])
    es = ewok_summary(paths['ewok'])
    ss['globalpiqa'] = gs
    ss['ewok'] = es
    ss['deltas_vs_anchor80'] = {
        'gp_parallel_acc': delta(anchor_gp.get('parallel_acc'), gs.get('parallel_acc')),
        'gp_hard52_acc': delta(anchor_gp.get('hard52_acc'), gs.get('hard52_acc')),
        'gp_hard52_mean_top_minus_correct': delta(anchor_gp.get('hard52_mean_top_minus_correct'), gs.get('hard52_mean_top_minus_correct')),
        'ewok_accuracy': delta(anchor_ew.get('accuracy'), es.get('accuracy')),
        'ewok_stable_failure': delta(anchor_ew.get('stable_failure'), es.get('stable_failure')),
    }
    row = {
        'target': name,
        'objective': ss.get('objective'),
        'lr': ss.get('lr'),
        'updates': ss.get('updates'),
        'bridge_delta_M': ss.get('bridge_delta_M'),
        'bridge_delta_both': ss.get('bridge_delta_both'),
        'clean_loss_delta': ss.get('clean_loss_delta'),
        'gp_parallel_acc': gs.get('parallel_acc'),
        'gp_parallel_delta': ss['deltas_vs_anchor80']['gp_parallel_acc'],
        'gp_hard52_acc': gs.get('hard52_acc'),
        'gp_hard52_margin_delta': ss['deltas_vs_anchor80']['gp_hard52_mean_top_minus_correct'],
        'ewok_accuracy': es.get('accuracy'),
        'ewok_accuracy_delta': ss['deltas_vs_anchor80']['ewok_accuracy'],
        'ewok_stable_failure': es.get('stable_failure'),
        'ewok_stable_failure_delta': ss['deltas_vs_anchor80']['ewok_stable_failure'],
    }
    small[name] = ss
    rows.append(row)
rows.sort(key=lambda r: (r['bridge_delta_M'] if isinstance(r['bridge_delta_M'], (int,float)) else -999), reverse=True)

corrs = {}
for metric in ['gp_parallel_delta', 'gp_hard52_margin_delta', 'ewok_accuracy_delta', 'ewok_stable_failure_delta']:
    xs=[]; ys=[]
    for r in rows:
        x=r.get('bridge_delta_M'); y=r.get(metric)
        if isinstance(x,(int,float)) and isinstance(y,(int,float)):
            xs.append(x); ys.append(y)
    corrs[f'bridge_delta_M_vs_{metric}'] = corr(xs, ys)
# One more: bridge both vs metrics
for metric in ['gp_parallel_delta', 'gp_hard52_margin_delta', 'ewok_accuracy_delta', 'ewok_stable_failure_delta']:
    xs=[]; ys=[]
    for r in rows:
        x=r.get('bridge_delta_both'); y=r.get(metric)
        if isinstance(x,(int,float)) and isinstance(y,(int,float)):
            xs.append(x); ys.append(y)
    corrs[f'bridge_delta_both_vs_{metric}'] = corr(xs, ys)

# Deltas for role screen with corrected cheap extraction.
role_cheap_deltas = {}
for base, tgt in [('role_fixed_80M','role_switch_80M'), ('anchor_fixed256_80M','role_switch_80M'), ('anchor_fixed256_80M','role_fixed_80M')]:
    role_cheap_deltas[f'{tgt}_minus_{base}'] = {k: delta(role_cheap[base].get(k), role_cheap[tgt].get(k)) for k in ['BLiMP','Supplement','EWoK','Entity','COMPS','GlobalPIQA','Reading','cheap7','GlobalPIQA_parallel','GlobalPIQA_nonparallel']}

out = {
    'status': 'ROUTEB_AND_BRIDGE_SYNTHESIS_CORRECTED',
    'correction': 'Corrected broad cheap7 extraction to include GlobalPIQA and Reading from official_overall/nested Reading score.',
    'role_screen': {
        'cheap': role_cheap,
        'cheap_deltas': role_cheap_deltas,
        'globalpiqa': role_gp,
        'ewok': role_ew,
        'bridge': role_bridge,
        'packet_synthetic_overall': {k: v.get('overall') for k,v in role_syn.items()},
    },
    'small_update_table_sorted_by_bridge_delta_M': rows,
    'small_updates': small,
    'correlations_across_7_small_updates': corrs,
    'scientific_interpretation': {
        'routeB_status': 'The exact legal sparse clustered WWM role-switch replacement recipe learned constructed packet exchange but did not beat role-fixed on GlobalPIQA_parallel/hard52 and remained below the anchor on broad cheap7 and EWoK stable failures.',
        'bridge_status': 'The research naturalistic bridge is an unsolved stress object, but its improvement is not a reliable selector for official hard behavior. Ordinary correct CE often improves it more than centered exchange, while GlobalPIQA hard52 and hard margins remain essentially unrepaired.',
        'next_direction': 'Do not optimize the bridge or vary packet dose/placement. Change representation or source of learning signal before any endpoint-scale work.'
    }
}
summary_path = OUT_DIR / 'routeB_bridge_synthesis_corrected.json'
summary_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n')

lines = []
lines.append('# research — corrected Route B and bridge predictive-value synthesis\n\n')
lines.append('This note replaces the first synthesis broad-score calculation, which omitted GlobalPIQA and Reading while reading per-target score JSONs. The hard-surface interpretation is unchanged.\n\n')
lines.append('## Legal research role-switch screen\n\n')
for k in ['anchor_fixed256_80M','role_fixed_80M','role_switch_80M']:
    c=role_cheap[k]; g=role_gp[k]; e=role_ew[k]; b=role_bridge[k]
    lines.append(f"- `{k}`: cheap7={c.get('cheap7'):.4f}; BLiMP={c.get('BLiMP')}, Supplement={c.get('Supplement')}, EWoK={c.get('EWoK')}, Entity={c.get('Entity')}, COMPS={c.get('COMPS')}, GlobalPIQA={c.get('GlobalPIQA')}, Reading={c.get('Reading')}; GP_parallel={g.get('parallel_acc'):.2f}, hard52={g.get('hard52_acc'):.2f}, hard52_margin={g.get('hard52_mean_top_minus_correct'):.3f}; EWoK_acc={e.get('accuracy'):.6f}, stable_fail={e.get('stable_failure')}; bridge_M={b.get('M'):.3f}, bridge_both={b.get('both'):.3f}.\n")
lines.append('\nCorrected role-switch minus role-fixed cheap deltas: ' + json.dumps(role_cheap_deltas['role_switch_80M_minus_role_fixed_80M'], ensure_ascii=False) + '\n')
lines.append('Corrected role-switch minus anchor cheap deltas: ' + json.dumps(role_cheap_deltas['role_switch_80M_minus_anchor_fixed256_80M'], ensure_ascii=False) + '\n\n')
lines.append('## research small updates: bridge movement versus official hard surfaces\n\n')
lines.append('| target | obj | lr | upd | bridge ΔM | bridge Δboth | clean Δloss | GP par Δ | GP hard52 | hard52 margin Δ | EWoK acc Δ | stable fail Δ |\n')
lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n')
for r in rows:
    lines.append(f"| {r['target']} | {r['objective']} | {r['lr']} | {r['updates']} | {r['bridge_delta_M']:.3f} | {r['bridge_delta_both']:.3f} | {r['clean_loss_delta']:.4f} | {r['gp_parallel_delta']:.2f} | {r['gp_hard52_acc']:.2f} | {r['gp_hard52_margin_delta']:.3f} | {r['ewok_accuracy_delta']:.6f} | {r['ewok_stable_failure_delta']} |\n")
lines.append('\nCorrelations across the seven small updates: ' + json.dumps(corrs, ensure_ascii=False) + '\n\n')
lines.append('## Scientific reading\n\n')
lines.append('- Route B exact recipe is closed: role-switch learned the packet grammar relative to role-fixed, but role-switch and role-fixed are identical on GlobalPIQA_parallel (23.30) and hard52 accuracy (1/52), and role-switch has worse hard52 mean top-minus-correct than role-fixed.\n')
lines.append('- Corrected broad scores do not rescue it: role-switch cheap7 is 42.9043, only +0.315 over role-fixed and +0.0136 over the 80M anchor, while losing Supplement and EWoK relative to anchor. This is not a SOTA trajectory and cannot justify 100M continuation.\n')
lines.append('- The constructed bridge is not a sufficient selector for official hard behavior. Correct-CE variants give the largest bridge improvements and some non-hard GlobalPIQA_parallel movement, but hard52 remains 0/52 or 1/52 with mean wrong margins not improved, and EWoK movement is inconsistent. Low-dose exchange preserves clean MLM better but leaves GP_parallel flat and worsens EWoK stable failures.\n')
lines.append('- The next mechanism should change the representation or source of learning signal, not vary packet dose/placement or optimize the bridge itself.\n')
lines.append(f'\nFull corrected JSON: `{summary_path}`\n')
CORR_NOTE.write_text(''.join(lines), encoding='utf-8')
print(json.dumps({'status': out['status'], 'summary': str(summary_path), 'note': str(CORR_NOTE), 'rows': rows, 'correlations': corrs}, indent=2), flush=True)
