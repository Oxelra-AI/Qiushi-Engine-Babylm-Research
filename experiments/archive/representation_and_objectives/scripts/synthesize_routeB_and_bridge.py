#!/usr/bin/env python3
"""research synthesis for Route B closure and bridge predictive value."""
from __future__ import annotations

import json
import pathlib
from typing import Any

ROOT = pathlib.Path.cwd()
WS = ROOT / 'experiments/archive/representation_and_objectives'
OUT_DIR = WS / 'data/routeB_bridge_synthesis'
OUT_DIR.mkdir(parents=True, exist_ok=True)

ROLE_DIR = WS / 'data/role_screen_readout'
SMALL_ROOT = WS / 'data/small_exchange_update'
READ_ROOT = WS / 'data/small_update_hard_surface_readout'

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
ROLE_SYN = ROLE_DIR / 'packet_synthetic_cpu/packet_synthetic_readout_summary.json'
ROLE_BRIDGE = {
    'role_switch_80M': WS / 'data/naturalistic_exchange_bridge_readout_role_switch/naturalistic_bridge_readout_summary_score.json',
    'role_fixed_80M': WS / 'data/naturalistic_exchange_bridge_readout_role_fixed/naturalistic_bridge_readout_summary_score.json',
    'anchor_80M': WS / 'data/naturalistic_exchange_bridge_readout/naturalistic_bridge_readout_summary_score.json',
}

SMALL_DIRS = {
    'anchor80_exchange_lr5e-5_u80': SMALL_ROOT / 'anchor80_exchange_lr5e-5_u80/small_update_summary.json',
    'anchor80_correct_ce_lr5e-5_u80': SMALL_ROOT / 'anchor80_correct_ce_lr5e-5_u80/small_update_summary.json',
    'anchor80_fixed_ce_lr5e-5_u80': SMALL_ROOT / 'anchor80_fixed_ce_lr5e-5_u80/small_update_summary.json',
    'pareto_exchange_lr2e-5_u20': SMALL_ROOT / 'pareto_anchor80_exchange_lr2e-5_u20/small_update_summary.json',
    'pareto_correct_ce_lr2e-5_u20': SMALL_ROOT / 'pareto_anchor80_correct_ce_lr2e-5_u20/small_update_summary.json',
    'pareto_exchange_lr1e-5_u40': SMALL_ROOT / 'pareto_anchor80_exchange_lr1e-5_u40/small_update_summary.json',
    'pareto_correct_ce_lr1e-5_u40': SMALL_ROOT / 'pareto_anchor80_correct_ce_lr1e-5_u40/small_update_summary.json',
}
SMALL_GP_DIR = {
    'anchor80_exchange_lr5e-5_u80': READ_ROOT / 'gp_anchor80_exchange_lr5e-5_u80/anchor80_exchange_lr5e-5_u80/globalpiqa/anchor80_exchange_lr5e-5_u80_globalpiqa_summary.json',
    'anchor80_correct_ce_lr5e-5_u80': READ_ROOT / 'gp_anchor80_correct_ce_lr5e-5_u80/anchor80_correct_ce_lr5e-5_u80/globalpiqa/anchor80_correct_ce_lr5e-5_u80_globalpiqa_summary.json',
    'anchor80_fixed_ce_lr5e-5_u80': READ_ROOT / 'gp_anchor80_fixed_ce_lr5e-5_u80/anchor80_fixed_ce_lr5e-5_u80/globalpiqa/anchor80_fixed_ce_lr5e-5_u80_globalpiqa_summary.json',
    'pareto_exchange_lr2e-5_u20': READ_ROOT / 'gp_pareto_exchange_lr2e-5_u20/pareto_exchange_lr2e-5_u20/globalpiqa/pareto_exchange_lr2e-5_u20_globalpiqa_summary.json',
    'pareto_correct_ce_lr2e-5_u20': READ_ROOT / 'gp_pareto_correct_ce_lr2e-5_u20/pareto_correct_ce_lr2e-5_u20/globalpiqa/pareto_correct_ce_lr2e-5_u20_globalpiqa_summary.json',
    'pareto_exchange_lr1e-5_u40': READ_ROOT / 'gp_pareto_exchange_lr1e-5_u40/pareto_exchange_lr1e-5_u40/globalpiqa/pareto_exchange_lr1e-5_u40_globalpiqa_summary.json',
    'pareto_correct_ce_lr1e-5_u40': READ_ROOT / 'gp_pareto_correct_ce_lr1e-5_u40/pareto_correct_ce_lr1e-5_u40/globalpiqa/pareto_correct_ce_lr1e-5_u40_globalpiqa_summary.json',
}
SMALL_EWOK_DIR = {
    'anchor80_exchange_lr5e-5_u80': READ_ROOT / 'ewok_anchor80_exchange_lr5e-5_u80/anchor80_exchange_lr5e-5_u80/ewok/anchor80_exchange_lr5e-5_u80_ewok_summary.json',
    'anchor80_correct_ce_lr5e-5_u80': READ_ROOT / 'ewok_anchor80_correct_ce_lr5e-5_u80/anchor80_correct_ce_lr5e-5_u80/ewok/anchor80_correct_ce_lr5e-5_u80_ewok_summary.json',
    'anchor80_fixed_ce_lr5e-5_u80': READ_ROOT / 'ewok_anchor80_fixed_ce_lr5e-5_u80/anchor80_fixed_ce_lr5e-5_u80/ewok/anchor80_fixed_ce_lr5e-5_u80_ewok_summary.json',
    'pareto_exchange_lr2e-5_u20': READ_ROOT / 'ewok_pareto_exchange_lr2e-5_u20/pareto_exchange_lr2e-5_u20/ewok/pareto_exchange_lr2e-5_u20_ewok_summary.json',
    'pareto_correct_ce_lr2e-5_u20': READ_ROOT / 'ewok_pareto_correct_ce_lr2e-5_u20/pareto_correct_ce_lr2e-5_u20/ewok/pareto_correct_ce_lr2e-5_u20_ewok_summary.json',
    'pareto_exchange_lr1e-5_u40': READ_ROOT / 'ewok_pareto_exchange_lr1e-5_u40/pareto_exchange_lr1e-5_u40/ewok/pareto_exchange_lr1e-5_u40_ewok_summary.json',
    'pareto_correct_ce_lr1e-5_u40': READ_ROOT / 'ewok_pareto_correct_ce_lr1e-5_u40_retry/pareto_correct_ce_lr1e-5_u40/ewok/pareto_correct_ce_lr1e-5_u40_ewok_summary.json',
}


def load(path: pathlib.Path) -> Any:
    if not path.exists():
        return {'MISSING': str(path)}
    return json.loads(path.read_text())


def scores_from_cheap(path: pathlib.Path) -> dict[str, Any]:
    d = load(path)
    if 'MISSING' in d:
        return d
    tasks = d.get('tasks', {})
    out = {k: tasks.get(k, {}).get('score') for k in ['BLiMP', 'Supplement', 'EWoK', 'Entity', 'COMPS', 'GlobalPIQA', 'Reading']}
    vals = [v for v in out.values() if isinstance(v, (int, float))]
    out['cheap7'] = sum(vals)/len(vals) if vals else None
    out['path'] = str(path)
    return out


def target_gp_summary(path: pathlib.Path) -> dict[str, Any]:
    d = load(path)
    if 'MISSING' in d:
        return d
    # wrapper format: modes; role-screen format: targets -> target -> modes
    if 'modes' in d:
        modes = d['modes']
    else:
        # first target
        modes = next(iter(d.get('targets', {}).values())).get('modes', {})
    par = modes.get('parallel', {})
    if 'summary' in par:
        par = par['summary']
    aw = par.get('always_wrong_subset') or {}
    allm = par.get('all_rows_margin_summary') or {}
    return {
        'path': str(path),
        'parallel_acc': par.get('accuracy'),
        'parallel_rank_counts': par.get('correct_rank_counts'),
        'hard52_acc': aw.get('accuracy'),
        'hard52_rank_counts': aw.get('correct_rank_counts'),
        'hard52_mean_top_minus_correct': aw.get('mean_top_minus_correct'),
        'hard52_small_margin_le_0p50': aw.get('small_wrong_margin_le_0p50_nats'),
        'all_mean_top_minus_correct': allm.get('mean_top_minus_correct'),
    }


def target_ewok_summary(path: pathlib.Path) -> dict[str, Any]:
    d = load(path)
    if 'MISSING' in d:
        return d
    if 'result' in d:
        s = d['result']['summary']
    else:
        s = next(iter(d.get('targets', {}).values())).get('summary', {})
    return {
        'path': str(path),
        'accuracy': s.get('accuracy'),
        'saved_wrong': s.get('saved_wrong'),
        'stable_failure': s.get('stable_failure'),
        'stable_failure_frac_all': s.get('stable_failure_frac_all'),
        'stable_failure_frac_wrong': s.get('stable_failure_frac_wrong'),
        'interaction_sum_wrong_mean': (s.get('interaction_sum_wrong') or {}).get('mean'),
        'interaction_sum_wrong_median': (s.get('interaction_sum_wrong') or {}).get('median'),
    }


def small_summary(path: pathlib.Path) -> dict[str, Any]:
    d = load(path)
    if 'MISSING' in d:
        return d
    def get(cur, keys, default=None):
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur
    before = get(d, ['before', 'bridge', 'overall'], {}) or {}
    after = get(d, ['after', 'bridge', 'overall'], {}) or {}
    pkt_before = get(d, ['before', 'synthetic', 'overall'], {}) or get(d, ['before', 'packet', 'overall'], {}) or {}
    pkt_after = get(d, ['after', 'synthetic', 'overall'], {}) or get(d, ['after', 'packet', 'overall'], {}) or {}
    bm, am = before.get('M_mean'), after.get('M_mean')
    bb, ab = before.get('both_correct'), after.get('both_correct')
    return {
        'path': str(path),
        'objective': get(d, ['args', 'objective']),
        'lr': get(d, ['args', 'lr']),
        'updates': get(d, ['args', 'updates']),
        'bridge_delta_M': am - bm if isinstance(am, (int, float)) and isinstance(bm, (int, float)) else None,
        'bridge_after_M': am,
        'bridge_delta_both': ab - bb if isinstance(ab, (int, float)) and isinstance(bb, (int, float)) else None,
        'bridge_after_both': ab,
        'clean_loss_delta': get(d, ['clean_probe', 'delta_loss_per_masked_token']),
        'synthetic_before_both': pkt_before.get('both_correct'),
        'synthetic_after_both': pkt_after.get('both_correct'),
    }


def bridge_existing(path: pathlib.Path, label: str) -> dict[str, Any]:
    d = load(path)
    if 'MISSING' in d:
        return d
    if 'target_summaries' in d:
        s = d['target_summaries'].get(label) or next(iter(d['target_summaries'].values()))
    else:
        s = d.get('summaries', {}).get(label) or next(iter(d.get('summaries', {}).values()))
    o = s.get('overall', {}) if isinstance(s, dict) else {}
    return {'path': str(path), 'M': o.get('M_mean'), 'both': o.get('both_correct'), 'acc_AB': o.get('acc_AB'), 'acc_BA': o.get('acc_BA')}


role = {
    'cheap': {k: scores_from_cheap(p) for k, p in ROLE_CHEAP.items()},
    'globalpiqa': {k: target_gp_summary(p) for k, p in ROLE_GP.items()},
    'ewok': {k: target_ewok_summary(p) for k, p in ROLE_EWOK.items()},
    'bridge': {k: bridge_existing(p, k) for k, p in ROLE_BRIDGE.items()},
}
# synthetic concise
syn = load(ROLE_SYN)
role['packet_synthetic'] = {}
if 'summaries' in syn:
    for k, v in syn['summaries'].items():
        role['packet_synthetic'][k] = {'overall': v.get('overall'), 'path': str(ROLE_SYN)}

# deltas for role screen
if all(k in role['cheap'] for k in ['role_switch_80M', 'role_fixed_80M', 'anchor_fixed256_80M']):
    role['cheap_deltas'] = {}
    for a, b in [('role_fixed_80M', 'role_switch_80M'), ('anchor_fixed256_80M', 'role_switch_80M'), ('anchor_fixed256_80M', 'role_fixed_80M')]:
        role['cheap_deltas'][f'{b}_minus_{a}'] = {m: (role['cheap'][b].get(m) - role['cheap'][a].get(m)) if isinstance(role['cheap'][b].get(m), (int, float)) and isinstance(role['cheap'][a].get(m), (int, float)) else None for m in ['BLiMP','Supplement','EWoK','Entity','COMPS','GlobalPIQA','Reading','cheap7']}

anchor_gp = role['globalpiqa']['anchor_fixed256_80M']
anchor_ew = role['ewok']['anchor_fixed256_80M']
small = {}
for k in SMALL_DIRS:
    ss = small_summary(SMALL_DIRS[k])
    gp = target_gp_summary(SMALL_GP_DIR[k])
    ew = target_ewok_summary(SMALL_EWOK_DIR[k])
    ss['globalpiqa'] = gp
    ss['ewok'] = ew
    if isinstance(gp.get('parallel_acc'), (int, float)) and isinstance(anchor_gp.get('parallel_acc'), (int, float)):
        ss['globalpiqa_delta_vs_anchor80'] = {
            'parallel_acc': gp['parallel_acc'] - anchor_gp['parallel_acc'],
            'hard52_acc': gp['hard52_acc'] - anchor_gp['hard52_acc'],
            'hard52_mean_top_minus_correct': gp['hard52_mean_top_minus_correct'] - anchor_gp['hard52_mean_top_minus_correct'],
            'hard52_small_margin_le_0p50': gp['hard52_small_margin_le_0p50'] - anchor_gp['hard52_small_margin_le_0p50'],
        }
    if isinstance(ew.get('accuracy'), (int, float)) and isinstance(anchor_ew.get('accuracy'), (int, float)):
        ss['ewok_delta_vs_anchor80'] = {
            'accuracy': ew['accuracy'] - anchor_ew['accuracy'],
            'stable_failure': ew['stable_failure'] - anchor_ew['stable_failure'],
            'stable_failure_frac_all': ew['stable_failure_frac_all'] - anchor_ew['stable_failure_frac_all'],
            'stable_failure_frac_wrong': ew['stable_failure_frac_wrong'] - anchor_ew['stable_failure_frac_wrong'],
            'interaction_sum_wrong_mean': ew['interaction_sum_wrong_mean'] - anchor_ew['interaction_sum_wrong_mean'],
        }
    small[k] = ss

# compact table sorted by bridge movement
small_table = []
for k, v in small.items():
    row = {
        'target': k,
        'objective': v.get('objective'),
        'lr': v.get('lr'),
        'updates': v.get('updates'),
        'bridge_delta_M': v.get('bridge_delta_M'),
        'bridge_delta_both': v.get('bridge_delta_both'),
        'clean_loss_delta': v.get('clean_loss_delta'),
        'gp_parallel_acc': v.get('globalpiqa', {}).get('parallel_acc'),
        'gp_parallel_delta_vs_anchor80': v.get('globalpiqa_delta_vs_anchor80', {}).get('parallel_acc'),
        'gp_hard52_acc': v.get('globalpiqa', {}).get('hard52_acc'),
        'gp_hard52_margin_delta_vs_anchor80': v.get('globalpiqa_delta_vs_anchor80', {}).get('hard52_mean_top_minus_correct'),
        'ewok_accuracy': v.get('ewok', {}).get('accuracy'),
        'ewok_accuracy_delta_vs_anchor80': v.get('ewok_delta_vs_anchor80', {}).get('accuracy'),
        'ewok_stable_failure': v.get('ewok', {}).get('stable_failure'),
        'ewok_stable_failure_delta_vs_anchor80': v.get('ewok_delta_vs_anchor80', {}).get('stable_failure'),
    }
    small_table.append(row)
small_table.sort(key=lambda r: (r['bridge_delta_M'] if isinstance(r['bridge_delta_M'], (int, float)) else -999), reverse=True)

out = {
    'status': 'ROUTEB_AND_BRIDGE_SYNTHESIS',
    'scientific_question': 'Does sparse role-switch experience or research bridge improvement predict natural official hard surfaces?',
    'role_screen': role,
    'small_updates': small,
    'small_update_table_sorted_by_bridge_delta_M': small_table,
    'interpretation': {
        'routeB_exact_recipe': 'role-switch learned synthetic packets but failed GlobalPIQA hard-rank requirement and broad cheap7 did not show a hidden large advantage; exact sparse clustered WWM replacement recipe should close.',
        'bridge_predictive_value': 'Bridge movement occurs under exchange, correct CE, and fixed CE, with correct CE often moving the bridge most, but GlobalPIQA_parallel and hard52 do not improve; bridge alone is not an endpoint-selection target.',
    },
}
summary_path = OUT_DIR / 'routeB_bridge_synthesis.json'
summary_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

# Markdown note for quick reading.
lines = []
lines.append('# research — Route B and bridge predictive-value synthesis\n')
lines.append('## Legal research role-switch screen\n')
for k in ['anchor_fixed256_80M','role_fixed_80M','role_switch_80M']:
    c = role['cheap'][k]; g = role['globalpiqa'][k]; e = role['ewok'][k]; b = role['bridge'].get(k.replace('anchor_fixed256_80M','anchor_80M'), {}) if k == 'anchor_fixed256_80M' else role['bridge'].get(k, {})
    lines.append(f"- `{k}`: cheap7={c.get('cheap7'):.4f}, BLiMP={c.get('BLiMP')}, Supplement={c.get('Supplement')}, EWoK={c.get('EWoK')}, Entity={c.get('Entity')}, COMPS={c.get('COMPS')}, GlobalPIQA={c.get('GlobalPIQA')}, Reading={c.get('Reading')}; GP_parallel={g.get('parallel_acc'):.2f}, hard52={g.get('hard52_acc'):.2f}, hard52_margin={g.get('hard52_mean_top_minus_correct'):.3f}; EWoK_acc={e.get('accuracy'):.6f}, stable_fail={e.get('stable_failure')}; bridge_M={b.get('M')}, bridge_both={b.get('both')}.\n")
lines.append('\nRole-switch minus role-fixed cheap deltas: ' + json.dumps(role.get('cheap_deltas', {}).get('role_switch_80M_minus_role_fixed_80M', {}), ensure_ascii=False) + '\n')
lines.append('\n## research small updates sorted by bridge delta\n')
lines.append('| target | obj | lr | upd | bridge ΔM | bridge Δboth | clean Δloss | GP par | GP hard52 | hard52 margin Δ | EWoK acc | stable fail Δ |\n')
lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n')
for r in small_table:
    lines.append(f"| {r['target']} | {r['objective']} | {r['lr']} | {r['updates']} | {r['bridge_delta_M']:.3f} | {r['bridge_delta_both']:.3f} | {r['clean_loss_delta']:.4f} | {r['gp_parallel_acc']:.2f} | {r['gp_hard52_acc']:.2f} | {r['gp_hard52_margin_delta_vs_anchor80']:.3f} | {r['ewok_accuracy']:.6f} | {r['ewok_stable_failure_delta_vs_anchor80']} |\n")
lines.append('\n## Interpretation\n')
lines.append('- The legal role-switch replacement arm learned more constructed packet behavior than role-fixed, but GlobalPIQA_parallel is identical between role-switch and role-fixed and the 52-row hard subset remains one row correct in both; role-switch has worse hard52 mean top-minus-correct than role-fixed.\n')
lines.append('- Broad cheap7 does not rescue the route: role-switch is below the fixed-256 80M anchor and only modestly above role-fixed through Entity/Supplement tradeoffs; it is not a SOTA seed.\n')
lines.append('- The low-dose bridge does not select the natural hard surfaces. Ordinary correct CE gives the largest bridge gain, but GlobalPIQA hard behavior stays at or below the anchor and EWoK movement is small or harmful. The fixed-CE arm also moves the bridge without the desired natural repair.\n')
lines.append('- Therefore the constructed bridge remains useful as an unsolved transfer object and stress test, but not as a direct optimization target. The next work should change representation or learning signal source rather than elaborate packet dose/placement or tune on the bridge.\n')
note_path = WS / 'notes/routeB_bridge_predictive_value.md'
note_path.write_text(''.join(lines), encoding='utf-8')

print(json.dumps({'status': out['status'], 'summary': str(summary_path), 'note': str(note_path), 'small_table': small_table}, indent=2), flush=True)
