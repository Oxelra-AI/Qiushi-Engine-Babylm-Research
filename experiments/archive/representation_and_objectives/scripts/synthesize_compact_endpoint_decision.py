#!/usr/bin/env python3
"""research: synthesize compact directional endpoint transfer after eval completes.

Run this only after `eval_directional_fork_cheap7_parallel.py` has produced
FF/FR/RR/RF `cheap7_summary.json` files. It runs/loads full-EWoK domain analysis and
combines endpoint interactions with the internal pair-objective synthesis and the
source-conditioned ordering constraint.
"""
from __future__ import annotations
import argparse, json, pathlib, subprocess, sys, time
from typing import Any, Dict

INTERNAL = pathlib.Path('experiments/archive/representation_and_objectives/data/compact_directional_internal/internal_synthesis.json')
EWOK_ANALYZER = pathlib.Path('experiments/archive/representation_and_objectives/scripts/ewok_domain_directional_full_analyzer.py')
ARMS = ['ff','fr','rr','rf']
COLS = ['BLiMP','Supplement','EWoK','Entity','COMPS','GlobalPIQA','Reading','cheap7']


def load_json(p: pathlib.Path) -> Any:
    return json.loads(p.read_text(encoding='utf-8'))


def interaction(vals: Dict[str, float]) -> Dict[str, float]:
    one = 0.5*(vals['ff']+vals['rr'])
    mix = 0.5*(vals['fr']+vals['rf'])
    return {'values': vals, 'oneway_avg_ff_rr': one, 'mixed_avg_fr_rf': mix, 'mixed_minus_oneway': mix-one, 'direction_asymmetry_ff_minus_rr': vals['ff']-vals['rr'], 'order_asymmetry_fr_minus_rf': vals['fr']-vals['rf']}


def load_endpoint(eval_root: pathlib.Path) -> Dict[str, Any]:
    arm_summaries = {}
    for arm in ARMS:
        p = eval_root / arm / 'cheap7_summary.json'
        if not p.exists():
            raise FileNotFoundError(p)
        arm_summaries[arm] = load_json(p)
    interactions = {}
    for col in COLS:
        vals = {}
        for arm in ARMS:
            if col == 'cheap7':
                vals[arm] = arm_summaries[arm]['cheap7']
            else:
                vals[arm] = arm_summaries[arm]['cheap7_columns'].get(col)
        if all(v is not None for v in vals.values()):
            interactions[col] = interaction(vals)
    return {'arms': arm_summaries, 'interactions': interactions}


def ensure_ewok(eval_root: pathlib.Path) -> Dict[str, Any]:
    out = eval_root / 'ewok_domain_directional_full_analysis.json'
    if not out.exists():
        cmd = [sys.executable, '-B', str(EWOK_ANALYZER), '--eval_root', str(eval_root), '--output', str(out)]
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600)
        if proc.returncode != 0:
            raise RuntimeError(proc.stdout[-6000:])
    return load_json(out)


def signed_read(endpoint: Dict[str, Any], ewok: Dict[str, Any], internal: Dict[str, Any]) -> Dict[str, Any]:
    inter = endpoint['interactions']
    cheap7_I = inter.get('cheap7', {}).get('mixed_minus_oneway')
    ewok_I = inter.get('EWoK', {}).get('mixed_minus_oneway')
    rel_I = ewok.get('relational_domain_interaction', {}).get('mixed_minus_oneway')
    # No hard universal threshold: causal 20M endpoints are noisy. These bands simply
    # keep the route decision explicit and prevent a tiny scalar from becoming a claim.
    if cheap7_I is None or ewok_I is None or rel_I is None:
        status = 'incomplete_endpoint_interaction'
    elif cheap7_I <= 0 and ewok_I <= 0 and rel_I <= 0:
        status = 'close_compact_reciprocal_causal_instantiation'
    elif rel_I > 0 or ewok_I > 0 or cheap7_I > 0.05:
        status = 'endpoint_has_positive_signal_needing_controls'
    else:
        status = 'weak_or_mixed_signal_requires_judgment'
    return {
        'decision_status': status,
        'cheap7_I': cheap7_I,
        'EWoK_column_I': ewok_I,
        'full_EWoK_relational_domain_I': rel_I,
        'internal_forward_noncopied_I_loss': internal['central_numbers']['forward_pair_noncopied_I_loss'],
        'internal_reverse_noncopied_I_loss': internal['central_numbers']['reverse_pair_noncopied_I_loss'],
        'relation_context_constraint': internal.get('relation_context_constraint', {}),
        'interpretation': {
            'close_if': 'If endpoint cheap7, EWoK column, and full relational-domain I are all absent or negative, the compact 20M reciprocal causal instantiation learned local pair prediction but did not transfer to the target surface.',
            'controls_if': 'If endpoint transfer is positive, run semantic_extract and random_extract fork controls before attributing specificity to compact faithful rewrites, because A02 shows ordered source retrieval is not compact-specific.',
            'bounded_continuation_if': 'If endpoint is weak/mixed but internal noncopied I is the only favorable clue, continuation must be bounded and cheaper than two full controls for distinguishing delayed transfer from null.'
        }
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval_root', required=True)
    ap.add_argument('--output_json', default='')
    ap.add_argument('--output_md', default='')
    args = ap.parse_args()
    eval_root = pathlib.Path(args.eval_root)
    internal = load_json(INTERNAL)
    endpoint = load_endpoint(eval_root)
    ewok = ensure_ewok(eval_root)
    read = signed_read(endpoint, ewok, internal)
    summary = {
        'status': 'COMPACT_ENDPOINT_DECISION_SYNTHESIS',
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'eval_root': str(eval_root),
        'internal_synthesis': str(INTERNAL),
        'endpoint': {'interactions': endpoint['interactions']},
        'full_ewok': {
            'overall_interaction': ewok['overall_interaction'],
            'relational_domain_interaction': ewok['relational_domain_interaction'],
            'adjacency_independent_domain_interaction': ewok['adjacency_independent_domain_interaction'],
            'domain_interactions': ewok['domain_interactions'],
            'analysis_path': str(eval_root / 'ewok_domain_directional_full_analysis.json')
        },
        'scientific_read': read,
        'next_actions': {
            'if_close': 'Stop compact reciprocal-causal 20M route; hand route question back to mechanism/strategy around density, tail coverage, ordered retrieval, and DeBERTa MLM specificity.',
            'if_controls': 'Launch v4 fork branches for semantic_extract and random_extract using launch_copy_matched_control_branches.py, then repeat pair-objective and endpoint readouts and compare I(compact)-I(control).',
            'if_bounded_continuation': 'Design a short continuation with saved optimizer/fork states and intermediate checkpoints, but only if it can separate delayed endpoint maturation from null more cheaply than controls.'
        }
    }
    outj = pathlib.Path(args.output_json) if args.output_json else eval_root / 'compact_endpoint_decision_synthesis.json'
    outm = pathlib.Path(args.output_md) if args.output_md else eval_root / 'compact_endpoint_decision_synthesis.md'
    outj.parent.mkdir(parents=True, exist_ok=True)
    outj.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    md = ['# research compact directional endpoint decision synthesis', '', f'Eval root: `{eval_root}`', '', '## Endpoint interactions', '', '| metric | I = mixed - one-way | FF | RR | FR | RF |', '|---|---:|---:|---:|---:|---:|']
    for m, rec in summary['endpoint']['interactions'].items():
        v = rec['values']
        md.append(f"| {m} | {rec['mixed_minus_oneway']:.6f} | {v['ff']:.4f} | {v['rr']:.4f} | {v['fr']:.4f} | {v['rf']:.4f} |")
    md += ['', '## Full EWoK', '', f"- Overall full-EWoK I: `{ewok['overall_interaction']['mixed_minus_oneway']}`", f"- research relational-domain group I: `{ewok['relational_domain_interaction']['mixed_minus_oneway']}`", f"- Adjacency-independent-domain group I: `{ewok['adjacency_independent_domain_interaction']['mixed_minus_oneway']}`", '', '## Scientific read', '', f"Decision status: `{read['decision_status']}`", '', json.dumps(read, indent=2), '', f'JSON: `{outj}`']
    outm.write_text('\n'.join(md)+'\n', encoding='utf-8')
    print(json.dumps({'status': summary['status'], 'decision_status': read['decision_status'], 'cheap7_I': read['cheap7_I'], 'EWoK_I': read['EWoK_column_I'], 'relational_I': read['full_EWoK_relational_domain_I'], 'json': str(outj), 'md': str(outm)}, indent=2), flush=True)

if __name__ == '__main__':
    main()
