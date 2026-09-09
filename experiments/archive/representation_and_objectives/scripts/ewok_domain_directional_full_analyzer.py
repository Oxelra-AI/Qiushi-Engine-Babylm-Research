#!/usr/bin/env python3
"""research: full-EWoK directional endpoint analyzer for causal fork evals.

The research analyzer originally pointed to fast EWoK by default. This version is
explicitly tied to the full `ewok_filtered` data used by the research causal
endpoint evaluator. It reports full-domain accuracy and interaction

  I = 0.5*(FR + RF) - 0.5*(FF + RR)

and labels, but does not pretend that these full rows are item-identical to the
research fast EWoK transition table. research domains remain a prospective signature:
social-properties, physical-dynamics, spatial-relations, physical-relations.
"""
from __future__ import annotations
import argparse, json, pathlib, math
from typing import Any, Dict

ARMS = ['ff','fr','rr','rf']
RELATIONAL = ['social-properties','physical-dynamics','spatial-relations','physical-relations']
INDEPENDENT = ['material-properties','social-interactions']
DEFAULT_GOLD = pathlib.Path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/ewok_filtered')


def load_json(p: pathlib.Path) -> Any:
    return json.loads(p.read_text(encoding='utf-8'))


def find_pred(summary: Dict[str, Any]) -> pathlib.Path:
    rec = summary.get('tasks', {}).get('EWoK')
    if not rec:
        raise KeyError('EWoK task missing in cheap7_summary')
    p = pathlib.Path(rec.get('predictions') or '')
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def iter_gold(gold_dir: pathlib.Path):
    for gf in sorted(gold_dir.glob('*.jsonl')):
        sub = gf.stem
        rows = [json.loads(l) for l in gf.read_text(encoding='utf-8').splitlines() if l.strip()]
        yield sub, rows


def score(pred_path: pathlib.Path, gold_dir: pathlib.Path) -> Dict[str, Any]:
    pred = load_json(pred_path)
    total = correct = 0
    by_domain: Dict[str, Dict[str, int]] = {}
    by_subtask: Dict[str, Dict[str, int]] = {}
    wrong_examples = []
    for sub, gold_rows in iter_gold(gold_dir):
        if sub not in pred:
            raise KeyError(f'{sub} missing from predictions {pred_path}')
        preds = pred[sub].get('predictions', [])
        if len(preds) != len(gold_rows):
            raise ValueError(f'{sub} length mismatch: pred {len(preds)} gold {len(gold_rows)}')
        for i, (pr, g) in enumerate(zip(preds, gold_rows)):
            target = ' '.join([g['Context1'], g['Target1']]).strip()
            ok = (pr.get('pred') or '').strip() == target
            dom = g.get('Domain') or sub
            for tab, key in [(by_domain, dom), (by_subtask, sub)]:
                d = tab.setdefault(key, {'n':0, 'correct':0})
                d['n'] += 1; d['correct'] += int(ok)
            total += 1; correct += int(ok)
            if not ok and len(wrong_examples) < 6:
                wrong_examples.append({'subtask': sub, 'domain': dom, 'index': i, 'pred': pr.get('pred'), 'target': target})
    def finalize(tab):
        return {k: {**v, 'accuracy': 100.0*v['correct']/max(v['n'],1)} for k,v in sorted(tab.items())}
    return {'prediction_path': str(pred_path), 'total': total, 'correct': correct, 'accuracy': 100.0*correct/max(total,1), 'domains': finalize(by_domain), 'subtasks': finalize(by_subtask), 'wrong_examples': wrong_examples}


def weighted_group(arm_scores, domains):
    vals = {}
    for arm in ARMS:
        n = c = 0
        for d in domains:
            rec = arm_scores[arm]['domains'].get(d)
            if rec:
                n += rec['n']; c += rec['correct']
        vals[arm] = 100.0*c/max(n,1)
    return vals


def interaction(vals):
    one = 0.5*(vals['ff']+vals['rr'])
    mix = 0.5*(vals['fr']+vals['rf'])
    return {'values': vals, 'oneway_avg_ff_rr': one, 'mixed_avg_fr_rf': mix, 'mixed_minus_oneway': mix-one, 'direction_asymmetry_ff_minus_rr': vals['ff']-vals['rr'], 'order_asymmetry_fr_minus_rf': vals['fr']-vals['rf']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval_root', required=True)
    ap.add_argument('--gold_dir', default=str(DEFAULT_GOLD))
    ap.add_argument('--output', default='')
    args = ap.parse_args()
    eval_root = pathlib.Path(args.eval_root)
    gold_dir = pathlib.Path(args.gold_dir)
    arm_scores = {}
    for arm in ARMS:
        sp = eval_root / arm / 'cheap7_summary.json'
        if not sp.exists():
            raise FileNotFoundError(sp)
        arm_scores[arm] = score(find_pred(load_json(sp)), gold_dir)
    all_domains = sorted(set().union(*(set(arm_scores[a]['domains']) for a in ARMS)))
    domain_interactions = {}
    for d in all_domains:
        vals = {arm: arm_scores[arm]['domains'][d]['accuracy'] for arm in ARMS if d in arm_scores[arm]['domains']}
        if len(vals) == 4:
            domain_interactions[d] = {**interaction(vals), 'n': arm_scores['ff']['domains'][d]['n']}
    summary = {
        'status': 'FULL_EWOK_DOMAIN_DIRECTIONAL_ANALYSIS',
        'eval_root': str(eval_root),
        'gold_dir': str(gold_dir),
        'surface': 'full_eval/ewok_filtered, same as research causal endpoint evaluator',
        'not_item_identical_to_step211_fast_table': True,
        'arms': arm_scores,
        'overall_interaction': interaction({a: arm_scores[a]['accuracy'] for a in ARMS}),
        'domain_interactions': domain_interactions,
        'relational_domains': RELATIONAL,
        'relational_domain_interaction': interaction(weighted_group(arm_scores, RELATIONAL)),
        'adjacency_independent_domains': INDEPENDENT,
        'adjacency_independent_domain_interaction': interaction(weighted_group(arm_scores, INDEPENDENT)),
    }
    out = pathlib.Path(args.output) if args.output else eval_root / 'ewok_domain_directional_full_analysis.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    md = ['# research full EWoK directional analysis', '', f'Eval root: `{eval_root}`', f'Gold: `{gold_dir}`', '', f"Overall full-EWoK I: `{summary['overall_interaction']['mixed_minus_oneway']}`", f"research relational-domain group I: `{summary['relational_domain_interaction']['mixed_minus_oneway']}`", f"Adjacency-independent-domain group I: `{summary['adjacency_independent_domain_interaction']['mixed_minus_oneway']}`", '', '| domain | n | I | FF | RR | FR | RF |', '|---|---:|---:|---:|---:|---:|---:|']
    for d, rec in sorted(domain_interactions.items(), key=lambda kv: kv[1]['mixed_minus_oneway'], reverse=True):
        md.append(f"| {d} | {rec['n']} | {rec['mixed_minus_oneway']:.4f} | {rec['values']['ff']:.2f} | {rec['values']['rr']:.2f} | {rec['values']['fr']:.2f} | {rec['values']['rf']:.2f} |")
    md += ['', f'JSON: `{out}`']
    out.with_suffix('.md').write_text('\n'.join(md)+'\n', encoding='utf-8')
    print(json.dumps({'status': summary['status'], 'output': str(out), 'overall_I': summary['overall_interaction']['mixed_minus_oneway'], 'relational_I': summary['relational_domain_interaction']['mixed_minus_oneway'], 'independent_I': summary['adjacency_independent_domain_interaction']['mixed_minus_oneway']}, indent=2), flush=True)

if __name__ == '__main__':
    main()
