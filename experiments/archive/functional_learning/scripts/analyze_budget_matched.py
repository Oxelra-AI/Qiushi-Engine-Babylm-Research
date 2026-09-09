#!/usr/bin/env python3
"""Analyze research budget-matched run with unified, scale-robust criteria.

The training script writes rich records, but its auto-summary intentionally remains
simple.  This analyzer is the canonical research readout: it reports preparation-end
states, exact P+500 states, cumulative and post-switch acquisition times, sustained
success crossings, positive-fraction metrics, held corruption, and research/research
comparisons where available.
"""
import argparse, json, math
from pathlib import Path
from statistics import mean

DEFAULT_RESULTS = Path('experiments/archive/functional_learning/data/budget_matched_full_objective/results.json')
OUT_NOTE = Path('research/notes/functional_learning/budget_matched_full_objective_analysis.md')
QUERY_FIRST_RESULTS = Path('experiments/archive/functional_learning/data/query_first_binding/results.json')
BRANCHING_RESULTS = Path('experiments/archive/functional_learning/data/branching/results.json')


def gp(obj, path):
    for k in path:
        obj = obj[k]
    return obj


def records(d, seed=None, arm=None):
    rs = d.get('records', [])
    if seed is not None:
        rs = [r for r in rs if int(r['seed']) == int(seed)]
    if arm is not None:
        rs = [r for r in rs if r['arm'] == arm]
    return sorted(rs, key=lambda r: int(r['epoch']))


def exact(rs, epoch):
    m = [r for r in rs if int(r['epoch']) == int(epoch)]
    if not m:
        return None
    if len(m) != 1:
        raise ValueError(f'multiple records at epoch {epoch}')
    return m[0]


def metric_row(r):
    return dict(
        epoch=int(r['epoch']),
        phase=r.get('phase',''),
        mode=r.get('mode',''),
        target=r.get('target',''),
        nll=float(gp(r,['std','correct_nll'])),
        top4=float(gp(r,['std','ctx_top1'])),
        bag=float(gp(r,['std','bag_mass'])),
        bmean=float(gp(r,['bswap','mean'])),
        bfrac=float(gp(r,['bswap','frac_pos'])),
        qmean=float(gp(r,['qswap','margin'])),
        qfrac=float(gp(r,['qswap','frac_pos'])),
        qboth=float(gp(r,['qswap','both'])),
        sel=float(gp(r,['corrupt','novel_selectivity'])),
        held_top4=float(gp(r,['held','ctx_top1'])),
        held_nll=float(gp(r,['held','correct_nll'])),
        held_bmean=float(gp(r,['held_bswap','mean'])),
        held_bfrac=float(gp(r,['held_bswap','frac_pos'])),
        held_qmean=float(gp(r,['held_qswap','margin'])),
        held_qfrac=float(gp(r,['held_qswap','frac_pos'])),
        held_qboth=float(gp(r,['held_qswap','both'])),
        held_sel=float(gp(r,['held_corrupt','novel_selectivity'])),
    )


def strong(m):
    # Unified criterion: top4, margin scale, prevalence, q-both, and corruption.
    return (m['top4'] >= 0.95 and m['bmean'] >= 5.0 and m['qmean'] >= 5.0 and
            m['bfrac'] >= 0.95 and m['qfrac'] >= 0.95 and m['qboth'] >= 0.90 and
            m['sel'] >= 0.80)


def mid(m):
    return (m['top4'] >= 0.50 and m['bmean'] >= 1.0 and m['qmean'] >= 1.0 and
            m['bfrac'] >= 0.75 and m['qfrac'] >= 0.75)


def held_strong(m):
    # Held Q-swap is mixed, so require direct held B-swap and held corruption;
    # held q metrics are reported but not decisive.
    return (m['held_top4'] >= 0.75 and m['held_bmean'] >= 2.0 and
            m['held_bfrac'] >= 0.80 and m['held_sel'] >= 0.30)


def first_cross(ms, pred, min_epoch=1, sustained=False):
    ms = sorted([m for m in ms if m['epoch'] >= min_epoch], key=lambda x: x['epoch'])
    if not sustained:
        for m in ms:
            if pred(m):
                return m['epoch']
        return None
    for i in range(len(ms)-1):
        if pred(ms[i]) and pred(ms[i+1]):
            return ms[i]['epoch']
    return None


def fmt(x, digits=3):
    if x is None:
        return ''
    return f'{float(x):.{digits}f}'


def format_list(vals):
    return '[' + ', '.join('' if v is None else str(v) for v in vals) + ']'


def avg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return sum(vals)/len(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=str(DEFAULT_RESULTS))
    ap.add_argument('--out', default=str(OUT_NOTE))
    A = ap.parse_args()

    path = Path(A.results)
    d = json.load(open(path))
    seeds = [int(s) for s in d['config']['seeds']]
    arms = [a['name'] for a in d['arm_specs']]
    prep_epochs = {int(k): int(v) for k, v in d['config']['prep_epochs'].items()}
    total_epochs = int(d['config']['total_epochs'])

    # Build per-seed/arm table.
    rows = []
    for sd in seeds:
        P = prep_epochs[sd]
        p500 = P + 500
        for arm in arms:
            rs = records(d, seed=sd, arm=arm)
            ms = [metric_row(r) for r in rs]
            if not ms:
                continue
            prep = exact(rs, P)
            p1 = exact(rs, P+1)
            p25 = exact(rs, P+25)
            p50 = exact(rs, P+50)
            p500r = exact(rs, p500)
            final = exact(rs, total_epochs) or rs[-1]
            mp = metric_row(prep) if prep else None
            mf = metric_row(final)
            mp500 = metric_row(p500r) if p500r else None
            first_mid = first_cross(ms, mid)
            first_str = first_cross(ms, strong)
            first_str_sust = first_cross(ms, strong, sustained=True)
            first_str_after = first_cross(ms, strong, min_epoch=P+1)
            first_str_after_sust = first_cross(ms, strong, min_epoch=P+1, sustained=True)
            first_held = first_cross(ms, held_strong)
            first_held_after = first_cross(ms, held_strong, min_epoch=P+1)
            rows.append(dict(seed=sd, arm=arm, P=P, p500=p500,
                             prep=mp, p1=metric_row(p1) if p1 else None,
                             p25=metric_row(p25) if p25 else None,
                             p50=metric_row(p50) if p50 else None,
                             p500_metrics=mp500, final=mf,
                             first_mid=first_mid, first_str=first_str,
                             first_str_sust=first_str_sust,
                             first_str_after=first_str_after,
                             first_str_after_sust=first_str_after_sust,
                             post_switch_str=(None if first_str_after is None else first_str_after - P),
                             post_switch_str_sust=(None if first_str_after_sust is None else first_str_after_sust - P),
                             first_held=first_held,
                             first_held_after=first_held_after,
                             post_switch_held=(None if first_held_after is None else first_held_after - P)))

    lines = []
    lines.append('# research analysis: budget-matched full-objective trajectories')
    lines.append('')
    lines.append(f'Read from `{path}`. This analysis uses one unified trained-entity success definition: top4≥0.95, B-swap mean≥5, Q-swap mean≥5, B/Q positive fractions≥0.95, Q-swap both≥0.90, and query-novel selectivity≥0.80. It also reports a held-entity behavior summary using held top4, held B-swap, and held corruption; held Q-swap is mixed because one side queries a trained entity.')
    lines.append('')
    lines.append('## Arm-level timing and endpoint summary')
    lines.append('')
    lines.append('| arm | success at P+500 | success at 1000 | sustained success at 1000 | mean first strong | mean post-switch first strong | mean P+500 top4 | mean final top4 | mean final held_top4 | mean final held_B | mean final held_sel |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for arm in arms:
        ar = [r for r in rows if r['arm'] == arm]
        p500succ = sum(1 for r in ar if r['p500_metrics'] is not None and strong(r['p500_metrics']))
        fsucc = sum(1 for r in ar if strong(r['final']))
        fssucc = sum(1 for r in ar if r['first_str_sust'] is not None and r['first_str_sust'] <= total_epochs and strong(r['final']))
        lines.append(f"| {arm} | {p500succ}/{len(ar)} | {fsucc}/{len(ar)} | {fssucc}/{len(ar)} | {fmt(avg([r['first_str'] for r in ar]),1)} | {fmt(avg([r['post_switch_str'] for r in ar]),1)} | {fmt(avg([r['p500_metrics']['top4'] for r in ar if r['p500_metrics']]))} | {fmt(avg([r['final']['top4'] for r in ar]))} | {fmt(avg([r['final']['held_top4'] for r in ar]))} | {fmt(avg([r['final']['held_bmean'] for r in ar]))} | {fmt(avg([r['final']['held_sel'] for r in ar]))} |")
    lines.append('')

    lines.append('## Per-seed states at preparation end P')
    lines.append('')
    lines.append('| seed | arm | P | phase/mode | top4 | Bmean | Bfrac | Qmean | Qfrac | Qboth | sel | held4 | heldB | heldSel |')
    lines.append('|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for r in rows:
        m = r['prep']
        lines.append(f"| {r['seed']} | {r['arm']} | {r['P']} | {m['phase']}/{m['mode']} | {fmt(m['top4'])} | {fmt(m['bmean'])} | {fmt(m['bfrac'])} | {fmt(m['qmean'])} | {fmt(m['qfrac'])} | {fmt(m['qboth'])} | {fmt(m['sel'])} | {fmt(m['held_top4'])} | {fmt(m['held_bmean'])} | {fmt(m['held_sel'])} |")
    lines.append('')

    lines.append('## Per-seed matched research-total state P+500')
    lines.append('')
    lines.append('| seed | arm | epoch | top4 | Bmean | Bfrac | Qmean | Qfrac | Qboth | sel | held4 | heldB | heldBfrac | heldQ | heldSel | strong? |')
    lines.append('|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|')
    for r in rows:
        m = r['p500_metrics']
        if m is None:
            continue
        lines.append(f"| {r['seed']} | {r['arm']} | {m['epoch']} | {fmt(m['top4'])} | {fmt(m['bmean'])} | {fmt(m['bfrac'])} | {fmt(m['qmean'])} | {fmt(m['qfrac'])} | {fmt(m['qboth'])} | {fmt(m['sel'])} | {fmt(m['held_top4'])} | {fmt(m['held_bmean'])} | {fmt(m['held_bfrac'])} | {fmt(m['held_qmean'])} | {fmt(m['held_sel'])} | {strong(m)} |")
    lines.append('')

    lines.append('## Per-seed final state at 1000 cumulative epochs')
    lines.append('')
    lines.append('| seed | arm | top4 | Bmean | Bfrac | Qmean | Qfrac | Qboth | sel | held4 | heldB | heldBfrac | heldQ | heldSel | strong? | held-behavior? | first strong | sustained first | post-switch first |')
    lines.append('|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|')
    for r in rows:
        m = r['final']
        lines.append(f"| {r['seed']} | {r['arm']} | {fmt(m['top4'])} | {fmt(m['bmean'])} | {fmt(m['bfrac'])} | {fmt(m['qmean'])} | {fmt(m['qfrac'])} | {fmt(m['qboth'])} | {fmt(m['sel'])} | {fmt(m['held_top4'])} | {fmt(m['held_bmean'])} | {fmt(m['held_bfrac'])} | {fmt(m['held_qmean'])} | {fmt(m['held_sel'])} | {strong(m)} | {held_strong(m)} | {'' if r['first_str'] is None else r['first_str']} | {'' if r['first_str_sust'] is None else r['first_str_sust']} | {'' if r['post_switch_str'] is None else r['post_switch_str']} |")
    lines.append('')

    # Existing research weighted-full readout if available.
    if QUERY_FIRST_RESULTS.exists():
        d16 = json.load(open(QUERY_FIRST_RESULTS))
        lines.append('## research qfirst weighted-full w16 baseline at 500 epochs')
        lines.append('')
        lines.append('This is not budget-matched to the 1000-epoch research run, but it tests whether simply upweighting the answer within the full objective solved the task by 500 epochs in the previous experiment.')
        lines.append('')
        lines.append('| seed | arm | top4 | Bmean | Qmean | Qboth | sel | held4 |')
        lines.append('|---:|---|---:|---:|---:|---:|---:|---:|')
        for arm in ['qfirst_full_500','qfirst_w16_500','qfirst_ans_only_500']:
            for sd in seeds:
                rs = [r for r in d16.get('records',[]) if r.get('arm') == arm and int(r.get('seed')) == sd and int(r.get('epoch')) == 500]
                if not rs:
                    continue
                rr = rs[0]
                lines.append(f"| {sd} | {arm} | {fmt(gp(rr,['std','ctx_top1']))} | {fmt(gp(rr,['bswap','mean']))} | {fmt(gp(rr,['qswap','margin']))} | {fmt(gp(rr,['qswap','both']))} | {fmt(gp(rr,['corrupt','novel_selectivity']))} | {fmt(gp(rr,['held','ctx_top1']))} |")
        lines.append('')

    # Interpretation placeholders generated from actual counts.
    lines.append('## Direct interpretation from the observed research pattern')
    lines.append('')
    fresh = next((a for a in arms if a == 'fresh_qfirst_full'), None)
    bound = next((a for a in arms if a == 'prep_bound_ans_then_full'), None)
    bag = next((a for a in arms if a == 'prep_bag_ans_then_full'), None)
    ctx = next((a for a in arms if a == 'prep_ctx_then_full'), None)
    def arm_count(arm, key='p500_metrics'):
        ar = [r for r in rows if r['arm'] == arm]
        return sum(1 for r in ar if r.get(key) is not None and strong(r[key])), len(ar)
    if fresh and bound:
        bf, bn = arm_count(bound)
        ff, fn = arm_count(fresh)
        lines.append(f'- At the matched research-total point P+500, bound preparation succeeds in {bf}/{bn} seeds and fresh full succeeds in {ff}/{fn} seeds under the unified criterion.')
        if bag:
            cg, cn = arm_count(bag)
            lines.append(f'- The equally trained unbound bag-answer preparation succeeds in {cg}/{cn} seeds at P+500.')
        if ctx:
            cc, cn = arm_count(ctx)
            lines.append(f'- The context-only preparation succeeds in {cc}/{cn} seeds at P+500.')
        bpost = [r['post_switch_str'] for r in rows if r['arm'] == bound]
        ffirst = [r['first_str'] for r in rows if r['arm'] == fresh]
        lines.append(f'- Bound post-switch full-objective epochs to first strong: {format_list(bpost)}. Fresh cumulative full-objective epochs to first strong: {format_list(ffirst)}.')
    lines.append('- Treat these numbers as paired seed trajectories, not population rates. The next scientific step depends on whether bound preparation is uniquely earlier/stronger than fresh and unbound controls, and whether held-entity probes move with trained-entity binding.')

    out = Path(A.out)
    out.write_text('\n'.join(lines) + '\n')

    # Machine-readable compact analysis.
    compact = {
        'results': str(path),
        'seeds': seeds,
        'arms': arms,
        'prep_epochs': prep_epochs,
        'total_epochs': total_epochs,
        'rows': [{k:v for k,v in r.items() if k not in ('prep','p1','p25','p50','p500_metrics','final')} | {
            'prep': r['prep'], 'p500_metrics': r['p500_metrics'], 'final': r['final'],
            'p500_strong': (None if r['p500_metrics'] is None else strong(r['p500_metrics'])),
            'final_strong': strong(r['final']), 'final_held_behavior': held_strong(r['final'])
        } for r in rows]
    }
    jout = out.with_suffix('.json')
    jout.write_text(json.dumps(compact, indent=2))
    print(json.dumps({'status':'ok','note':str(out),'json':str(jout)}, indent=2))

if __name__ == '__main__':
    main()
