#!/usr/bin/env python3
"""research complete CPU-side analysis for the dual-view 20M panel.

Run this after the three official-compatible cheap7 payloads exist.  It invokes the
validated sentinel comparison consumer and then combines:
  * cheap7/column deltas vs research and among aligned/shuffled/mlm_only
  * fragile-family sentinel readouts
  * training-accounting/budget substitution audits
  * parameter geometry from research

The output is a research-facing JSON/Markdown synthesis, not a final result.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time
from statistics import mean
from typing import Any

USER_ROOT = pathlib.Path('.').resolve()
STUDY = USER_ROOT / 'experiments/archive/frontier_consolidation'
WORKSPACE = STUDY
SCRIPTS = WORKSPACE / 'scripts'

SENTINEL_SCRIPT = SCRIPTS / 'dualview_sentinel_compare.py'
BASE_PAYLOAD = WORKSPACE / 'data/legal20m_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_20M.json'
PAYLOADS = {
    'aligned': WORKSPACE / 'data/dualview_aligned_20m_eval/per_target/dualview_aligned_20M.json',
    'shuffled': WORKSPACE / 'data/dualview_shuffled_20m_eval/per_target/dualview_shuffled_20M.json',
    'mlm_only': WORKSPACE / 'data/dualview_mlm_only_20m_eval/per_target/dualview_mlm_only_20M.json',
}
SUMMARIES = {
    'aligned': WORKSPACE / 'data/dualview_aligned_20m_summary/dualview_aligned_20M_summary.json',
    'shuffled': WORKSPACE / 'data/dualview_shuffled_20m_summary/dualview_shuffled_20M_summary.json',
    'mlm_only': WORKSPACE / 'data/dualview_mlm_only_20m_summary/dualview_mlm_only_20M_summary.json',
}
AUDIT = WORKSPACE / 'data/dualview_training_audit/aligned_shuffled_mlm_only_completed.json'
BUDGET = WORKSPACE / 'data/dualview_budget_substitution/budget_substitution_audit.json'
GEOMETRY = WORKSPACE / 'data/dualview_parameter_geometry/dualview_parameter_geometry.json'
TRAJECTORY = WORKSPACE / 'data/dualview_training_trajectory/aligned_vs_shuffled_training_trajectory.json'
OUT_DIR = WORKSPACE / 'data/dualview_complete_panel_analysis'

CHEAP_COLS = ['BLiMP', 'Supplement', 'EWoK', 'Entity', 'COMPS', 'GlobalPIQA', 'Reading']
FRAGILE = {
    'EWoK': ['material-dynamics', 'material-properties', 'spatial-relations', 'quantitative-properties', 'physical-dynamics', 'physical-interactions', 'social-relations'],
    'Supplement': ['subject_aux_inversion', 'qa_congruence_tricky', 'qa_congruence_easy'],
    'Entity': ['regular_5_ops', 'regular_4_ops', 'regular_3_ops'],
}


def now() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def rel(p: pathlib.Path | str) -> str:
    p = pathlib.Path(p)
    try:
        return str(p.resolve().relative_to(USER_ROOT))
    except Exception:
        return str(p)


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def cheap7(scores: dict[str, Any]) -> float:
    return float(mean(float(scores[c]) for c in CHEAP_COLS))


def require_files(paths: list[pathlib.Path]) -> list[str]:
    missing = [rel(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError('Missing required files:\n' + '\n'.join(missing))
    return [rel(p) for p in paths]


def run_sentinel(label: str) -> tuple[pathlib.Path, pathlib.Path, dict[str, Any]]:
    cmd = [
        sys.executable, '-B', str(SENTINEL_SCRIPT),
        '--base', str(BASE_PAYLOAD),
        '--base-label', 'reference_20M',
        '--candidate', f'aligned={PAYLOADS["aligned"]}',
        '--candidate', f'shuffled={PAYLOADS["shuffled"]}',
        '--candidate', f'mlm_only={PAYLOADS["mlm_only"]}',
        '--pairwise-candidates',
        '--label', label,
    ]
    proc = subprocess.run(cmd, cwd=str(USER_ROOT), text=True, capture_output=True, timeout=1200)
    (OUT_DIR / 'sentinel_stdout.log').write_text(proc.stdout, encoding='utf-8')
    (OUT_DIR / 'sentinel_stderr.log').write_text(proc.stderr, encoding='utf-8')
    if proc.returncode != 0:
        raise RuntimeError(f'sentinel comparison failed rc={proc.returncode}\n{proc.stderr[-4000:]}')
    sent_path = WORKSPACE / 'data/dualview_sentinel_compare' / f'{label}.json'
    sent_md = WORKSPACE / 'data/dualview_sentinel_compare' / f'{label}.md'
    if not sent_path.exists():
        raise FileNotFoundError(sent_path)
    return sent_path, sent_md, read_json(sent_path)


def summary_scores() -> dict[str, Any]:
    out = {}
    for lab, p in SUMMARIES.items():
        s = read_json(p)
        scores = {c: float(s['scores'][c]) for c in CHEAP_COLS}
        out[lab] = {
            'summary_path': rel(p),
            'payload_path': rel(PAYLOADS[lab]),
            'scores': scores,
            'cheap7': float(s.get('cheap7', cheap7(scores))),
            'cheap7_delta_vs_step35_20M': float(s.get('cheap7_delta_vs_step35_20M', 0.0)),
            'deltas_vs_step35_20M': {c: float(s['deltas_vs_step35_20M'][c]) for c in CHEAP_COLS},
            'training_metrics': s.get('training_metrics', {}),
            'elapsed_sec': s.get('elapsed_sec'),
        }
    # pairwise deltas in candidate score coordinates
    pairs = {}
    for a, b in [('aligned', 'shuffled'), ('aligned', 'mlm_only'), ('shuffled', 'mlm_only')]:
        pairs[f'{a}_minus_{b}'] = {
            'cheap7': out[a]['cheap7'] - out[b]['cheap7'],
            'columns': {c: out[a]['scores'][c] - out[b]['scores'][c] for c in CHEAP_COLS},
        }
    return {'arms': out, 'pairwise': pairs}


def find_comparison(sentinel: dict[str, Any], candidate: str, base: str) -> dict[str, Any] | None:
    for cmp in sentinel.get('comparisons', []):
        if cmp.get('candidate_label') == candidate and cmp.get('base_label') == base:
            return cmp
    return None


def summarize_fragile(sentinel: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for cname, candidate, base in [
        ('aligned_vs_step35', 'aligned', 'reference_20M'),
        ('shuffled_vs_step35', 'shuffled', 'reference_20M'),
        ('mlm_only_vs_step35', 'mlm_only', 'reference_20M'),
        ('shuffled_vs_aligned', 'shuffled', 'aligned'),
        ('mlm_only_vs_aligned', 'mlm_only', 'aligned'),
    ]:
        cmp = find_comparison(sentinel, candidate, base)
        if cmp is None:
            continue
        rows = {}
        for col, names in FRAGILE.items():
            sgs = cmp.get('sentinel_groups', {}).get(col, {})
            rows[col] = {name: sgs.get(name) for name in names}
        # negative net_gain_minus_loss means the candidate damages that family vs base.
        bad_nets = []
        for col, gs in rows.items():
            for name, r in gs.items():
                if r is not None:
                    bad_nets.append({'column': col, 'group': name, 'net_gain_minus_loss': r.get('net_gain_minus_loss'), 'n': r.get('n')})
        out[cname] = {
            'cheap7_delta': cmp.get('cheap7', {}).get('delta'),
            'column_deltas': cmp.get('deltas'),
            'fragile_groups': rows,
            'most_negative_fragile_nets': sorted(bad_nets, key=lambda x: x['net_gain_minus_loss'])[:10],
        }
    return out


def interpret(scores: dict[str, Any], fragile: dict[str, Any], geometry: dict[str, Any], trajectory: dict[str, Any], budget: dict[str, Any]) -> dict[str, Any]:
    a_vs_s = scores['pairwise']['aligned_minus_shuffled']['cheap7']
    a_vs_m = scores['pairwise']['aligned_minus_mlm_only']['cheap7']
    a_vs_step35 = scores['arms']['aligned']['cheap7_delta_vs_step35_20M']
    s_vs_step35 = scores['arms']['shuffled']['cheap7_delta_vs_step35_20M']
    m_vs_step35 = scores['arms']['mlm_only']['cheap7_delta_vs_step35_20M']
    # Scientific classification, not a stop rule. Interpretation still requires
    # inspection of the full artifacts; these labels summarize the evidence.
    if a_vs_s > 0 and a_vs_m > 0 and a_vs_step35 > 0:
        route_read = 'aligned_beats_both_controls_at_20M'
    elif a_vs_s > 0 and a_vs_m <= 0:
        route_read = 'true_correspondence_used_but_current_budget_tradeoff_under_mlm_only'
    elif a_vs_s <= 0 and a_vs_m > 0:
        route_read = 'dualview_aux_budget_may_help_but_not_via_true_source_correspondence'
    else:
        route_read = 'no_20M_score_support_for_current_dualview_construction'
    geom_pairs = geometry.get('pairwise', {})
    stock_rel = geom_pairs.get('aligned_vs_shuffled', {}).get('groups', {}).get('stock_all', {}).get('rel_l2_to_b')
    adapter_rel = geom_pairs.get('aligned_vs_shuffled', {}).get('groups', {}).get('adapter_all', {}).get('rel_l2_to_b')
    return {
        'route_read': route_read,
        'score_key_numbers': {
            'aligned_minus_shuffled_cheap7': a_vs_s,
            'aligned_minus_mlm_only_cheap7': a_vs_m,
            'aligned_minus_step35_20M_cheap7': a_vs_step35,
            'shuffled_minus_step35_20M_cheap7': s_vs_step35,
            'mlm_only_minus_step35_20M_cheap7': m_vs_step35,
        },
        'budget_key_numbers': {
            'dual_aux_words': budget.get('dual_aux_words'),
            'displaced_main_words': budget.get('displaced_main_words'),
            'displaced_main_rows_count': budget.get('displaced_main_rows_count'),
        },
        'training_loss_key_numbers': {
            'mean_main_loss_aligned_minus_shuffled': trajectory.get('loss_comparison', {}).get('mean_aligned_minus_shuffled'),
            'final_main_loss_aligned_minus_shuffled': trajectory.get('loss_comparison', {}).get('final_aligned_minus_shuffled'),
            'mean_aux_loss_aligned_minus_shuffled': trajectory.get('aux_loss_comparison', {}).get('mean_aligned_minus_shuffled'),
        },
        'geometry_key_numbers': {
            'aligned_vs_shuffled_stock_rel_l2': stock_rel,
            'aligned_vs_shuffled_adapter_rel_l2': adapter_rel,
            'aligned_vs_mlm_stock_rel_l2': geom_pairs.get('aligned_vs_mlm_only', {}).get('groups', {}).get('stock_all', {}).get('rel_l2_to_b'),
            'aligned_vs_mlm_adapter_rel_l2': geom_pairs.get('aligned_vs_mlm_only', {}).get('groups', {}).get('adapter_all', {}).get('rel_l2_to_b'),
        },
        'scientific_reading': (
            'A positive aligned-vs-shuffled score isolates useful true correspondence only because training accounting is matched. '
            'A positive aligned-vs-mlm_only score is harder: it means the dual-view private pathway overcame the 0.967M-word auxiliary budget substitution. '
            'Large stock rel-L2 by 20M means current construction is not purely private; if score support is weak, the next construction should reduce stock co-adaptation or make source-free adapter learning more budget-efficient rather than merely extending the same arm.'
        ),
    }


def build_md(out: dict[str, Any]) -> str:
    lines = [f"# research dual-view complete panel analysis — {out['label']}", '', f"Created: `{out['created_utc']}`", '']
    interp = out['interpretation']
    lines += ['## Route reading', '', f"`{interp['route_read']}`", '']
    lines += ['## Cheap7 and columns', '', '| Arm | cheap7 | vs research 20M | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | Reading |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for lab, rec in out['scores']['arms'].items():
        scores = rec['scores']
        lines.append(f"| {lab} | {rec['cheap7']:.6f} | {rec['cheap7_delta_vs_step35_20M']:+.6f} | " + ' | '.join(f"{scores[c]:.4f}" for c in CHEAP_COLS) + ' |')
    lines += ['', '### Pairwise cheap7 deltas', '', '| contrast | cheap7 delta |', '|---|---:|']
    for k, rec in out['scores']['pairwise'].items():
        lines.append(f"| {k} | {rec['cheap7']:+.6f} |")
    lines += ['', '## Mechanism/accounting context', '']
    for k, v in interp['training_loss_key_numbers'].items():
        lines.append(f"- {k}: `{v}`")
    for k, v in interp['budget_key_numbers'].items():
        lines.append(f"- {k}: `{v}`")
    for k, v in interp['geometry_key_numbers'].items():
        lines.append(f"- {k}: `{v}`")
    lines += ['', '## Most negative fragile sentinel nets', '']
    for cname, rec in out.get('fragile_summary', {}).items():
        lines.append(f"### {cname}")
        for r in rec.get('most_negative_fragile_nets', [])[:8]:
            lines.append(f"- {r['column']} / {r['group']}: {r['net_gain_minus_loss']} over n={r['n']}")
        lines.append('')
    lines += ['## Artifact paths', '']
    for k, v in out['artifacts'].items():
        lines.append(f"- {k}: `{v}`")
    return '\n'.join(lines) + '\n'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', default='dualview_panel_complete')
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    required = [BASE_PAYLOAD, AUDIT, BUDGET, GEOMETRY, TRAJECTORY, *PAYLOADS.values(), *SUMMARIES.values()]
    require_files(required)
    sent_json, sent_md, sentinel = run_sentinel(args.label)
    scores = summary_scores()
    fragile = summarize_fragile(sentinel)
    audit = read_json(AUDIT)
    budget = read_json(BUDGET)
    geometry = read_json(GEOMETRY)
    trajectory = read_json(TRAJECTORY)
    output = {
        'status': 'DUALVIEW_COMPLETE_PANEL_ANALYSIS',
        'label': args.label,
        'created_utc': now(),
        'scores': scores,
        'fragile_summary': fragile,
        'interpretation': interpret(scores, fragile, geometry, trajectory, budget),
        'artifacts': {
            'base_payload': rel(BASE_PAYLOAD),
            'aligned_payload': rel(PAYLOADS['aligned']),
            'shuffled_payload': rel(PAYLOADS['shuffled']),
            'mlm_only_payload': rel(PAYLOADS['mlm_only']),
            'aligned_summary': rel(SUMMARIES['aligned']),
            'shuffled_summary': rel(SUMMARIES['shuffled']),
            'mlm_only_summary': rel(SUMMARIES['mlm_only']),
            'sentinel_json': rel(sent_json),
            'sentinel_md': rel(sent_md),
            'training_audit': rel(AUDIT),
            'budget_audit': rel(BUDGET),
            'parameter_geometry': rel(GEOMETRY),
            'training_trajectory': rel(TRAJECTORY),
            'sentinel_stdout': rel(OUT_DIR / 'sentinel_stdout.log'),
            'sentinel_stderr': rel(OUT_DIR / 'sentinel_stderr.log'),
        },
        'training_audit_brief': {
            'aligned_shuffled_log_all_accounting_equal': audit.get('aligned_shuffled_log_all_accounting_equal'),
            'aligned_shuffled_metric_equal_selected': audit.get('aligned_shuffled_metric_equal_selected'),
        },
    }
    out_json = OUT_DIR / f'{args.label}.json'
    out_md = OUT_DIR / f'{args.label}.md'
    output['artifacts']['out_json'] = rel(out_json)
    output['artifacts']['out_md'] = rel(out_md)
    out_json.write_text(json.dumps(output, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    out_md.write_text(build_md(output), encoding='utf-8')
    print(json.dumps({'status': output['status'], 'label': args.label,
                      'route_read': output['interpretation']['route_read'],
                      'score_key_numbers': output['interpretation']['score_key_numbers'],
                      'out_json': rel(out_json), 'out_md': rel(out_md)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
