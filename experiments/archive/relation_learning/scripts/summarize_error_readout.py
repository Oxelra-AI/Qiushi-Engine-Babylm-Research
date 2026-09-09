#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import csv, math, pathlib, statistics
ROOT=_public_path('experiments/archive/relation_learning/scripts/summarize_error_readout.py')
ROOT = _PUBLIC_ROOT
base=_public_path('experiments/archive/relation_learning/data/entity_error_comps_readout')
rows=list(csv.DictReader((_public_path('experiments/archive/relation_learning/data/entity_error_comps_readout/entity_prediction_late_contrasts.csv')).open()))
print('COMPACT ENTITY ERROR/ACCURACY LATE SUMMARY')
for seed in ['43022','43122']:
    print('\nseed', seed)
    for group in ['numops_0','numops_2','numops_3','numops_4','numops_5']:
        vals=[r for r in rows if r['seed']==seed and r['group']==group and r['contrast'] in ['VminusC','VminusR','CminusR']]
        for r in vals:
            def f(k):
                try: return float(r[k])
                except Exception: return float('nan')
            print(f"  {group:8s} {r['contrast']:7s} dAcc={f('late_mean_delta_accuracy_pct'):+6.2f} dStaleWrong={f('late_mean_delta_stale_pick_wrong_pct'):+7.2f} dStaleAllWrong={f('late_mean_delta_stale_pick_all_wrong_pct'):+7.2f}")
print('\nCROSS-SEED MEAN VminusR by depth')
for group in ['numops_0','numops_1','numops_2','numops_3','numops_4','numops_5']:
    vals=[r for r in rows if r['group']==group and r['contrast']=='VminusR']
    acc=[float(r['late_mean_delta_accuracy_pct']) for r in vals]
    st=[]
    for r in vals:
        try:
            x=float(r['late_mean_delta_stale_pick_wrong_pct'])
            if math.isfinite(x): st.append(x)
        except Exception: pass
    st_s = f"{statistics.mean(st):+7.2f}" if st else 'nan'
    print(f"  {group}: dAcc mean={statistics.mean(acc):+6.2f} spread={max(acc)-min(acc):5.2f}; dStaleWrong mean={st_s}")
srows=list(csv.DictReader((_public_path('experiments/archive/relation_learning/data/entity_error_comps_readout/entity_prediction_summary.csv')).open()))
print('\nABSOLUTE stale-pick among wrong-with-stale-candidate for V/R/C')
for seed in ['43022','43122']:
    print('seed', seed)
    for group in ['numops_2','numops_3','numops_4','numops_5']:
        parts=[]
        for arm in ['V','R','C']:
            vals=[]
            for r in srows:
                if r['seed']==seed and r['arm']==arm and r['group']==group and r['checkpoint'] in ['chck_80M','chck_90M','chck_100M']:
                    try: vals.append(float(r['stale_pick_pct_among_wrong_stale_available_not_gold']))
                    except Exception: pass
            parts.append(f"{arm}={statistics.mean(vals):5.1f}%" if vals else f"{arm}=nan")
        print(' ',group,' '.join(parts))
