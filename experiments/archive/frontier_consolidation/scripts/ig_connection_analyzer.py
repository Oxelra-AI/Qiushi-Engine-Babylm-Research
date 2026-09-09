#!/usr/bin/env python3
"""research: bounded analysis of research/163 context-dependence probes.

This script intentionally answers only two specified scientific questions:
1. Does the low-IG/high-error residual change under matched compact-view/reinvestment controls?
2. Does the IG/low-IG likelihood object plausibly track the 82M->100M relation/state loss?

It does not mine arbitrary token categories or design a trainer.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = _public_path('.')
OUT = _public_path('experiments/archive/frontier_consolidation/data/ig_bounded_connection_analysis')
COMMON = _public_path('experiments/archive/frontier_consolidation/data/ig_matched_clean80_reinvest80_common_sample/context_dependence_items.csv')
CHANGED = _public_path('experiments/archive/frontier_consolidation/data/ig_matched_clean80_reinvest80_changed_block/context_dependence_items.csv')
research = _public_path('experiments/archive/frontier_consolidation/data/context_dependence_ig_probe_chck82_chck100_256rows/context_dependence_items.csv')
LATE_LOSS = _public_path('experiments/archive/frontier_consolidation/data/scale1p75_late_window_item_flips/scale1p75_82M_to_100M_late_loss.json')


def rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def read_csv(path: Path) -> list[dict[str, Any]]:
    out=[]
    with path.open(newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rr=dict(r)
            for k,v in list(rr.items()):
                if v is None or v == '':
                    continue
                # keep ids/tokens as strings where needed
                if k in {'checkpoint','item_id','source','target_token'}:
                    continue
                try:
                    if any(c in v for c in ['.','e','E']):
                        rr[k]=float(v)
                    else:
                        rr[k]=int(v)
                except Exception:
                    pass
            out.append(rr)
    return out


def mean(xs):
    xs=[float(x) for x in xs if x is not None and not (isinstance(x,float) and math.isnan(x))]
    return sum(xs)/len(xs) if xs else None


def quantile(xs, q):
    xs=sorted(float(x) for x in xs)
    if not xs: return None
    i=(len(xs)-1)*q
    lo=math.floor(i); hi=math.ceil(i)
    if lo==hi: return xs[lo]
    return xs[lo]*(hi-i)+xs[hi]*(i-lo)


def pearson(x,y):
    x=[float(a) for a in x]; y=[float(b) for b in y]
    n=len(x)
    if n<3: return None
    mx=sum(x)/n; my=sum(y)/n
    vx=sum((a-mx)**2 for a in x); vy=sum((b-my)**2 for b in y)
    if vx<=0 or vy<=0: return None
    return sum((a-mx)*(b-my) for a,b in zip(x,y))/math.sqrt(vx*vy)


def residual(y, x):
    y=[float(v) for v in y]; x=[float(v) for v in x]
    n=len(y)
    if n<3: return None
    mx=sum(x)/n; my=sum(y)/n
    vx=sum((a-mx)**2 for a in x)
    if vx<=0: return [v-my for v in y]
    beta=sum((a-mx)*(b-my) for a,b in zip(x,y))/vx
    alpha=my-beta*mx
    return [b-(alpha+beta*a) for a,b in zip(x,y)]


def summarize_values(xs):
    xs=[float(x) for x in xs]
    return {'n':len(xs),'mean':mean(xs),'median':quantile(xs,0.5),'p10':quantile(xs,0.1),'p90':quantile(xs,0.9),'min':min(xs) if xs else None,'max':max(xs) if xs else None}


def paired_by_item(rows, ref, cand):
    by={(r['checkpoint'], r['item_id']): r for r in rows}
    items=sorted({r['item_id'] for r in rows})
    pairs=[]
    for it in items:
        a=by.get((ref,it)); b=by.get((cand,it))
        if a and b:
            pairs.append((a,b))
    return pairs


def matched_control(path: Path, label: str) -> dict[str, Any]:
    rows=read_csv(path)
    pairs=paired_by_item(rows,'clean80','reinvest80')
    out={'label':label,'csv':rel(path),'n_pairs':len(pairs),'radii':{}}
    for rad in [4,8,16]:
        ig=[float(a[f'ig_r{rad}']) for a,b in pairs]
        full=[float(a['full_nll']) for a,b in pairs]
        freq=[float(a['log_sample_freq']) for a,b in pairs]
        delta=[float(b['full_nll'])-float(a['full_nll']) for a,b in pairs]
        q_ig20=quantile(ig,0.2); q_ig80=quantile(ig,0.8)
        q_full80=quantile(full,0.8); q_full50=quantile(full,0.5)
        strata={
            'all': [i for i in range(len(pairs))],
            'low_ig_bottom20': [i for i,v in enumerate(ig) if v <= q_ig20],
            'high_ig_top20': [i for i,v in enumerate(ig) if v >= q_ig80],
            'high_error_top20': [i for i,v in enumerate(full) if v >= q_full80],
            'low_ig_and_high_error': [i for i,(g,n) in enumerate(zip(ig,full)) if g <= q_ig20 and n >= q_full80],
            'low_ig_and_above_median_error': [i for i,(g,n) in enumerate(zip(ig,full)) if g <= q_ig20 and n >= q_full50],
            'high_ig_and_low_error': [i for i,(g,n) in enumerate(zip(ig,full)) if g >= q_ig80 and n <= q_full50],
        }
        rad_out={
            'clean_ig_quantiles': {'q20':q_ig20,'q80':q_ig80},
            'clean_full_nll_quantiles': {'q50':q_full50,'q80':q_full80},
            'corr_clean_ig_with_reinvest_minus_clean_full_nll_delta': pearson(ig,delta),
            'partial_corr_clean_ig_with_delta_control_logfreq': pearson(residual(ig,freq), residual(delta,freq)),
            'strata': {}
        }
        for name, idxs in strata.items():
            if not idxs: continue
            rad_out['strata'][name]={
                'n':len(idxs),
                'clean_full_nll_mean':mean(full[i] for i in idxs),
                'clean_ig_mean':mean(ig[i] for i in idxs),
                'delta_reinvest_minus_clean_full_nll_mean':mean(delta[i] for i in idxs),
                'reinvest_full_nll_mean':mean(float(pairs[i][1]['full_nll']) for i in idxs),
                'improved_fraction_delta_lt0':mean(1.0 if delta[i] < 0 else 0.0 for i in idxs),
            }
        out['radii'][f'r{rad}']=rad_out
    return out


def late_relation_state_summary() -> dict[str, Any]:
    d=json.load(LATE_LOSS.open())
    out={'json':rel(LATE_LOSS),'columns':{}}
    relation_cols=['EWoK','Entity']
    for col in relation_cols:
        c=d['columns'][col]
        out['columns'][col]={
            'delta_score_payload': c['delta_score_payload'],
            'n_common': c['n_common'],
            'flip_counts': c['flip_counts'],
            'gain_minus_loss_items': c['gain_minus_loss_items'],
            'worst_groups': c.get('groups_all_by_item_net', [])[-8:],
            'best_groups': c.get('groups_all_by_item_net', [])[:8],
        }
    # A conservative statement: this artifact has official eval item groups, not legal-corpus token identities,
    # so direct joining to research legal-corpus token IG is impossible. Any tracking must be tested by a dedicated
    # official-item forward probe, not by token mining in corpus CSVs.
    return out


def late_likelihood_relation() -> dict[str, Any]:
    rows=read_csv(research)
    pairs=paired_by_item(rows,'chck82','chck100')
    out={'csv':rel(research),'n_pairs':len(pairs),'radii':{}}
    for rad in [4,8,16]:
        ig=[float(a[f'ig_r{rad}']) for a,b in pairs]
        full=[float(a['full_nll']) for a,b in pairs]
        freq=[float(a['log_sample_freq']) for a,b in pairs]
        delta=[float(b['full_nll'])-float(a['full_nll']) for a,b in pairs]
        qig20=quantile(ig,0.2); qfull80=quantile(full,0.8)
        idx_res=[i for i,(g,n) in enumerate(zip(ig,full)) if g <= qig20 and n >= qfull80]
        idx_not=[i for i in range(len(pairs)) if i not in set(idx_res)]
        out['radii'][f'r{rad}']={
            'corr_ig_with_100minus82_full_nll_delta': pearson(ig,delta),
            'partial_corr_ig_with_delta_control_logfreq': pearson(residual(ig,freq), residual(delta,freq)),
            'all_delta_summary': summarize_values(delta),
            'low_ig_high_error_stratum': {
                'n': len(idx_res),
                'mean_chck82_full_nll': mean(full[i] for i in idx_res),
                'mean_chck82_ig': mean(ig[i] for i in idx_res),
                'mean_100minus82_full_nll_delta': mean(delta[i] for i in idx_res),
                'improved_fraction_delta_lt0': mean(1.0 if delta[i] < 0 else 0.0 for i in idx_res),
            },
            'not_stratum': {
                'n': len(idx_not),
                'mean_100minus82_full_nll_delta': mean(delta[i] for i in idx_not),
                'improved_fraction_delta_lt0': mean(1.0 if delta[i] < 0 else 0.0 for i in idx_not),
            }
        }
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result={
        'status':'IG_BOUNDED_CONNECTION_ANALYSIS',
        'matched_control_common_sample': matched_control(COMMON,'random common/full-pool sample on reinvest corpus'),
        'matched_control_changed_block': matched_control(CHANGED,'first 3006 changed compact-view block rows'),
        'late_likelihood_relation_on_step162_corpus_sample': late_likelihood_relation(),
        'official_late_relation_state_summary': late_relation_state_summary(),
        'interpretation': []
    }
    # Pull key facts for interpretation.
    for key in ['matched_control_common_sample','matched_control_changed_block']:
        r8=result[key]['radii']['r8']
        st=r8['strata']
        result['interpretation'].append({
            'artifact': key,
            'r8_all_delta': st['all']['delta_reinvest_minus_clean_full_nll_mean'],
            'r8_low_ig_high_error_delta': st.get('low_ig_and_high_error',{}).get('delta_reinvest_minus_clean_full_nll_mean'),
            'r8_high_ig_top20_delta': st['high_ig_top20']['delta_reinvest_minus_clean_full_nll_mean'],
            'r8_partial_corr_clean_ig_delta_logfreq': r8['partial_corr_clean_ig_with_delta_control_logfreq'],
        })
    r8late=result['late_likelihood_relation_on_step162_corpus_sample']['radii']['r8']
    result['interpretation'].append({
        'artifact':'chck82_to_chck100_corpus_likelihood',
        'r8_all_100minus82_delta': r8late['all_delta_summary']['mean'],
        'r8_low_ig_high_error_100minus82_delta': r8late['low_ig_high_error_stratum']['mean_100minus82_full_nll_delta'],
        'r8_not_stratum_100minus82_delta': r8late['not_stratum']['mean_100minus82_full_nll_delta'],
        'r8_partial_corr_ig_delta_logfreq': r8late['partial_corr_ig_with_delta_control_logfreq']
    })
    out_json=_public_path('experiments/archive/frontier_consolidation/data/ig_bounded_connection_analysis/ig_bounded_connection_analysis.json')
    out_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding='utf-8')
    lines=[]
    lines.append('# research bounded IG connection analysis')
    lines.append('')
    lines.append('This note answers only the two allowed questions: matched compact-view/reinvestment connection and relation/state late-loss connection. It intentionally avoids arbitrary token mining.')
    lines.append('')
    lines.append('## Matched clean80 → reinvest80 controls')
    for key,title in [('matched_control_common_sample','Common/full-pool sample'),('matched_control_changed_block','Changed compact-view block')]:
        r=result[key]
        lines.append(f'### {title}')
        lines.append(f"CSV: `{r['csv']}`; paired items `{r['n_pairs']}`")
        for rad in ['r4','r8','r16']:
            rr=r['radii'][rad]
            st=rr['strata']
            lines.append(f"- {rad}: all ΔNLL(reinvest-clean)={st['all']['delta_reinvest_minus_clean_full_nll_mean']:.6f}; lowIG+highErr Δ={st.get('low_ig_and_high_error',{}).get('delta_reinvest_minus_clean_full_nll_mean'):.6f} (n={st.get('low_ig_and_high_error',{}).get('n')}); highIGtop20 Δ={st['high_ig_top20']['delta_reinvest_minus_clean_full_nll_mean']:.6f}; partial corr(cleanIG, Δ | logfreq)={rr['partial_corr_clean_ig_with_delta_control_logfreq']:.6f}")
        lines.append('')
    lines.append('## chck82 → chck100 corpus-likelihood relation')
    lr=result['late_likelihood_relation_on_step162_corpus_sample']
    lines.append(f"CSV: `{lr['csv']}`; paired items `{lr['n_pairs']}`")
    for rad in ['r4','r8','r16']:
        rr=lr['radii'][rad]
        lines.append(f"- {rad}: all mean ΔNLL(100-82)={rr['all_delta_summary']['mean']:.6f}; lowIG+highErr Δ={rr['low_ig_high_error_stratum']['mean_100minus82_full_nll_delta']:.6f} (n={rr['low_ig_high_error_stratum']['n']}); not-stratum Δ={rr['not_stratum']['mean_100minus82_full_nll_delta']:.6f}; partial corr(IG, Δ | logfreq)={rr['partial_corr_ig_with_delta_control_logfreq']:.6f}")
    lines.append('')
    lines.append('## Official relation/state late loss')
    off=result['official_late_relation_state_summary']
    for col,c in off['columns'].items():
        lines.append(f"- {col}: payload Δ={c['delta_score_payload']:.6f}, common={c['n_common']}, flips={c['flip_counts']}, net={c['gain_minus_loss_items']}")
    lines.append('')
    lines.append('## Direct scientific reading')
    lines.append('- Reinvestment changes corpus likelihood most strongly in the changed compact-view block and disproportionately helps the low-IG/high-error stratum there; in the random/common sample the effect is small. This is a local data-mechanism likelihood signature, not the original high-IG target-allocation story.')
    lines.append('- On the research chck82→chck100 sample, the low-IG/high-error stratum improves in NLL rather than worsening, while official EWoK/Entity scores decline. The likelihood residual therefore does not track relation/state loss in the only shared evidence currently available.')
    lines.append('- The late-loss JSON summarizes official EWoK/Entity item flips but has no legal-corpus token identity; direct tracking would require a separate official-item forward probe. Given the negative corpus-likelihood relation, likelihood-based mining should stop unless such a probe is explicitly needed by a new mechanism.')
    lines.append('')
    lines.append(f"JSON: `{rel(out_json)}`")
    out_md=_public_path('research/documents/frontier_consolidation/data/ig_bounded_connection_analysis/ig_bounded_connection_analysis.md')
    out_md.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'status':result['status'],'out_json':rel(out_json),'out_md':rel(out_md)}, indent=2))

if __name__=='__main__':
    main()
