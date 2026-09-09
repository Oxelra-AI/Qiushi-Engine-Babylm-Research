#!/usr/bin/env python3
"""Compact summary of research focused EWoK relation-margin phase dynamics."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json, statistics, math, time
from pathlib import Path

HERE = _public_path('experiments/archive/representation_and_objectives/scripts/summarize_relation_phase_dynamics.py'); A01 = _public_path('experiments/archive/representation_and_objectives'); USER_ROOT = _public_path('.'); WS = _public_path('experiments/archive/representation_and_objectives')
OUT = _public_path('experiments/archive/representation_and_objectives/data/relation_phase_dynamics_summary')

FILES = {
    "strictsmalltok_50M": _public_path('experiments/archive/representation_and_objectives/data/strictsmalltok_midtrain_ewok_margin_probe/strictsmalltok_chck_50M_focus553_margins.json'),
    "strictsmalltok_70M": _public_path('experiments/archive/representation_and_objectives/data/strictsmalltok_midtrain_ewok_margin_probe/strictsmalltok_chck_70M_focus553_margins.json'),
    "strictsmalltok_80M": _public_path('experiments/archive/representation_and_objectives/data/strictsmalltok_midtrain_ewok_margin_probe/strictsmalltok_chck_80M_focus553_margins.json'),
    "strictsmalltok_100M": _public_path('experiments/archive/representation_and_objectives/data/strictsmalltok_midtrain_ewok_margin_probe/strictsmalltok_chck_100M_focus553_margins.json'),
    "oldtok_50_80": _public_path('experiments/archive/representation_and_objectives/data/oldtok_checkpoint_ewok_phase_probe/oldtok_checkpoint_focus553_phase_probe.json'),
    "oldtok_100": _public_path('experiments/archive/representation_and_objectives/data/official_ewok_margin_subset/official_ewok_margin_focus553_analysis.json'),
}

def load(p): return json.loads(p.read_text(encoding='utf-8'))

def get_strict_pair(path):
    d = load(path)
    a = d['two_seed_focus553_analysis']
    return {
        'accuracy_43022': a['accuracy_43022'],
        'accuracy_43122': a['accuracy_43122'],
        'advantage_43022_minus_43122': a['accuracy_43022'] - a['accuracy_43122'],
        'opposite_sign_frac': a['opposite_sign_frac'],
        'margin_pearson_430_431': a['margin_pearson_430_431'],
        'delta_margin_431_minus_430_mean': a['delta_margin_431_minus_430_mean'],
        'old_endpoint_seed_delta_corr': d.get('old_inherited_tokenizer_endpoint_comparison', {}).get('corr_new50_seed_delta_with_old100_seed_delta'),
        'old_endpoint_sign_agree_430': d.get('old_inherited_tokenizer_endpoint_comparison', {}).get('sign_agreement_430_frac'),
        'old_endpoint_sign_agree_431': d.get('old_inherited_tokenizer_endpoint_comparison', {}).get('sign_agreement_431_frac'),
    }

def pearson(xs, ys):
    if len(xs) != len(ys) or len(xs) < 2: return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    vx = sum((x-mx)**2 for x in xs); vy = sum((y-my)**2 for y in ys)
    if vx <= 0 or vy <= 0: return None
    return sum((x-mx)*(y-my) for x,y in zip(xs,ys)) / math.sqrt(vx*vy)

def sign(x): return 1 if x > 0 else (-1 if x < 0 else 0)

def main():
    summary = {'status': 'RELATION_PHASE_DYNAMICS_SUMMARY', 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    strict = {}
    for k in ['strictsmalltok_50M','strictsmalltok_70M','strictsmalltok_80M','strictsmalltok_100M']:
        if FILES[k].exists():
            strict[k] = get_strict_pair(FILES[k])
    old = load(FILES['oldtok_50_80'])['phase_pair_summaries']
    old100 = load(FILES['oldtok_100'])['model_micro_accuracy_on_selected_subset']
    old_summary = {
        'oldtok_50M': {
            'accuracy_43022': old['oldtok_50M']['accuracy_430'],
            'accuracy_43122': old['oldtok_50M']['accuracy_431'],
            'advantage_43022_minus_43122': old['oldtok_50M']['accuracy_430'] - old['oldtok_50M']['accuracy_431'],
            'opposite_sign_frac': old['oldtok_50M']['opposite_sign_frac'],
            'margin_pearson_430_431': old['oldtok_50M']['margin_pearson_430_431'],
        },
        'oldtok_80M': {
            'accuracy_43022': old['oldtok_80M']['accuracy_430'],
            'accuracy_43122': old['oldtok_80M']['accuracy_431'],
            'advantage_43022_minus_43122': old['oldtok_80M']['accuracy_430'] - old['oldtok_80M']['accuracy_431'],
            'opposite_sign_frac': old['oldtok_80M']['opposite_sign_frac'],
            'margin_pearson_430_431': old['oldtok_80M']['margin_pearson_430_431'],
        },
        'oldtok_100M_endpoint': {
            'accuracy_43022': old100['reinv430'],
            'accuracy_43122': old100['reinv431'],
            'advantage_43022_minus_43122': old100['reinv430'] - old100['reinv431'],
            'source': str(FILES['oldtok_100'].relative_to(USER_ROOT)),
        },
    }
    summary['strictsmalltok_focused_ewok_553_trajectory'] = strict
    summary['old_inherited_tokenizer_focused_ewok_553_trajectory'] = old_summary
    summary['old_phase_to_endpoint_delta_comparisons'] = load(FILES['oldtok_50_80'])['comparisons']
    summary['main_reading'] = [
        'MLM loss does not select the downstream-winning seed; see loss_trajectory_vs_overall.',
        'Old-tokenizer relation advantage for seed43022 forms late: weak at 50M, strong by 80M, strongest at endpoint; old 80M seed-delta correlates 0.938 with endpoint delta.',
        'Corrected-tokenizer dynamics are different: 50M favors seed43122 strongly, 70M/80M favor seed43022 modestly, and 100M still shows only a small seed43022 advantage on the old-instability subset. Corrected 80M/100M seed-delta correlation with the old endpoint remains near zero/slightly negative.',
        'Therefore final compliant-coordinate judgment must come from corrected 100M official evaluation; old seed superiority and MLM loss are not valid endpoint predictors.'
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    out = _public_path('experiments/archive/representation_and_objectives/data/relation_phase_dynamics_summary/relation_phase_dynamics_summary.json')
    out.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({'status': summary['status'], 'out_json': str(out.relative_to(USER_ROOT)), 'strict_trajectory': strict, 'old_trajectory': old_summary}, indent=2))

if __name__ == '__main__': main()
