#!/usr/bin/env python3
"""research support: verify earlier research note facts and expose the budget gap.

This script is deliberately small and read-only with respect to research data.  It
extracts the exact per-seed numbers needed before launching a budget-matched
counterfactual.  It also records which held-entity measurements are already in
research and which require a new run.
"""
import json, math
from pathlib import Path

ROOT = Path('experiments/archive/functional_learning')
BRANCHING_RESULTS = ROOT / 'data' / 'branching' / 'results.json'
QUERY_FIRST_RESULTS = ROOT / 'data' / 'query_first_binding' / 'results.json'
OUT = (ROOT / 'notes'.parents[3] / 'research/notes/functional_learning/verification_and_budget_gap.md')


def get_path(obj, path):
    for k in path:
        obj = obj[k]
    return obj


def recs_for(data, branch, seed):
    return [r for r in data['branches'] if r['branch'] == branch and r['seed'] == seed]


def fmt(x):
    return f"{float(x):.3f}"


def main():
    d17 = json.load(open(BRANCHING_RESULTS))
    seeds = d17['config']['seeds']
    lines = []
    lines.append('# research precheck: research verification and budget-matching problem')
    lines.append('')
    lines.append('This note verifies the earlier research note directly from `data/branching/results.json` and records why a new budget-matched full-objective comparison is necessary.')
    lines.append('')

    # Acquisition epochs actually used for branches.
    lines.append('## Acquisition epochs actually used for branch starts')
    lines.append('')
    lines.append('| seed | branch-start acq_epoch | total after +500 branch |')
    lines.append('|---:|---:|---:|')
    acq = {}
    for sd in seeds:
        r0 = [r for r in d17['branches'] if r['seed'] == sd and r['branch'] == 'switch_qfirst_full' and r['branch_epoch'] == 0][0]
        acq[sd] = int(r0['acq_epoch'])
        lines.append(f"| {sd} | {acq[sd]} | {acq[sd] + 500} |")
    lines.append('')
    lines.append('The earlier 3/3 versus 1/3 comparison is not budget-matched: research `switch_qfirst_full` receives 300–500 focused-preparation epochs plus 500 full-objective epochs, while research `qfirst_full_500` receives only 500 full-objective epochs from scratch.')
    lines.append('')

    # Verify full recovery final thresholds.
    lines.append('## research full-objective recovery at final branch epoch')
    lines.append('')
    lines.append('| seed | total_epoch | top4 | B-swap | Q-swap | held_top4 | held_NLL |')
    lines.append('|---:|---:|---:|---:|---:|---:|---:|')
    all_full_ok = True
    for sd in seeds:
        rs = recs_for(d17, 'switch_qfirst_full', sd)
        mx = max(r['branch_epoch'] for r in rs)
        r = [x for x in rs if x['branch_epoch'] == mx][0]
        top4 = get_path(r, ['qf','std','ctx_top1'])
        bs = get_path(r, ['qf','bswap','mean'])
        qs = get_path(r, ['qf','qswap','margin'])
        held_top4 = get_path(r, ['qf','held','ctx_top1'])
        held_nll = get_path(r, ['qf','held','correct_nll'])
        all_full_ok = all_full_ok and (top4 > 0.95 and bs > 5 and qs > 5)
        lines.append(f"| {sd} | {int(r['global_epoch'])} | {fmt(top4)} | {fmt(bs)} | {fmt(qs)} | {fmt(held_top4)} | {fmt(held_nll)} |")
    lines.append('')
    lines.append(f"Threshold check (top4>0.95, B-swap>5, Q-swap>5 for all seeds): **{all_full_ok}**.")
    lines.append('')

    # Context-only final.
    lines.append('## Context-only branch final binding check')
    lines.append('')
    lines.append('| seed | final top4 | final B-swap | final Q-swap |')
    lines.append('|---:|---:|---:|---:|')
    ctx_ok = True
    for sd in seeds:
        rs = recs_for(d17, 'switch_qfirst_ctx_only', sd)
        mx = max(r['branch_epoch'] for r in rs)
        r = [x for x in rs if x['branch_epoch'] == mx][0]
        top4 = get_path(r, ['qf','std','ctx_top1'])
        bs = get_path(r, ['qf','bswap','mean'])
        qs = get_path(r, ['qf','qswap','margin'])
        ctx_ok = ctx_ok and (top4 < 0.30)
        lines.append(f"| {sd} | {fmt(top4)} | {fmt(bs)} | {fmt(qs)} |")
    lines.append('')
    lines.append(f"Context-only below top4 0.30 for all seeds: **{ctx_ok}**.")
    lines.append('')

    # Blocked eval at branch epoch 0.
    lines.append('## Blocked query-to-context evaluation at the bound checkpoint')
    lines.append('')
    lines.append('| seed | standard top4 | blocked top4 | standard B-swap | blocked B-swap |')
    lines.append('|---:|---:|---:|---:|---:|')
    block_ok = True
    for sd in seeds:
        r = [x for x in recs_for(d17, 'continue_qfirst_ans_only', sd) if x['branch_epoch'] == 0][0]
        st4 = get_path(r, ['qf','std','ctx_top1'])
        bt4 = get_path(r, ['qf_blocked','std','ctx_top1'])
        sbs = get_path(r, ['qf','bswap','mean'])
        bbs = get_path(r, ['qf_blocked','bswap','mean'])
        block_ok = block_ok and (st4 > 0.95 and bt4 < 0.30)
        lines.append(f"| {sd} | {fmt(st4)} | {fmt(bt4)} | {fmt(sbs)} | {fmt(bbs)} |")
    lines.append('')
    lines.append(f"Immediate blocked-evaluation collapse for all seeds: **{block_ok}**.")
    lines.append('')

    # Held comparison within research.
    lines.append('## Held-entity comparison available in research')
    lines.append('')
    lines.append('| seed | continue held_top4 | switch-full held_top4 | continue held_NLL | switch-full held_NLL |')
    lines.append('|---:|---:|---:|---:|---:|')
    for sd in seeds:
        cont = [x for x in recs_for(d17, 'continue_qfirst_ans_only', sd) if x['branch_epoch'] == 500][0]
        sw = [x for x in recs_for(d17, 'switch_qfirst_full', sd) if x['branch_epoch'] == 500][0]
        lines.append(f"| {sd} | {fmt(get_path(cont,['qf','held','ctx_top1']))} | {fmt(get_path(sw,['qf','held','ctx_top1']))} | {fmt(get_path(cont,['qf','held','correct_nll']))} | {fmt(get_path(sw,['qf','held','correct_nll']))} |")
    lines.append('')
    lines.append('research did **not** generate held-entity B-swap or held-entity Q-swap probes; it only has a held standard probe plus train-entity B/Q swap probes. research therefore needs to add held B-swap/Q-swap metrics in the budget-matched run.')
    lines.append('')

    if QUERY_FIRST_RESULTS.exists():
        d16 = json.load(open(QUERY_FIRST_RESULTS))
        lines.append('## research fresh full-objective baseline at 500 epochs (not budget matched)')
        lines.append('')
        lines.append('| seed | top4 | B-swap | Q-swap | held_top4 |')
        lines.append('|---:|---:|---:|---:|---:|')
        for sd in seeds:
            rs = [r for r in d16['records'] if r['arm'] == 'qfirst_full_500' and r['seed'] == sd and r['epoch'] == 500]
            if not rs:
                continue
            r = rs[0]
            lines.append(f"| {sd} | {fmt(get_path(r,['std','ctx_top1']))} | {fmt(get_path(r,['bswap','mean']))} | {fmt(get_path(r,['qswap','margin']))} | {fmt(get_path(r,['held','ctx_top1']))} |")
        lines.append('')
        lines.append('These research numbers are useful historical evidence but use only 500 epochs and a different probe bank, so they cannot decide the curriculum-efficiency question.')
        lines.append('')

    lines.append('## Consequence for research')
    lines.append('')
    lines.append('Run a new comparison on a common cumulative budget. For each seed, compare (i) fresh query-first full-objective training through the same total epochs, (ii) bound answer-only preparation followed by full objective, and (iii) an equally trained but unbound bag-answer preparation followed by full objective. Track acquisition/reacquisition curves and held-entity swap probes. A curriculum advantage requires earlier or more reliable strong full-objective binding under the same total experience/compute; endpoint success after extra preparation alone is insufficient.')

    OUT.write_text('\n'.join(lines) + '\n')
    print(json.dumps({
        'status': 'ok',
        'out': str(OUT),
        'full_recovery_threshold_all': all_full_ok,
        'ctx_only_below_0p30_all': ctx_ok,
        'blocked_eval_collapse_all': block_ok,
        'acq_epochs': acq,
    }, indent=2))

if __name__ == '__main__':
    main()
