#!/usr/bin/env python3
"""research: seed43222 V-R Entity relevant-update crossover and pre-registered band check.

Uses existing D_V_43222 Entity prediction files (from research) and D_R_43222 files
(from research experiment_process or research script). Computes the V−R relevant-update crossover
and compares against pre-registered bands from research magnitude notes:
  - V−R zero-update:  [-12, -6]  (REPEAT wins)
  - V−R rel3/rel4:    [+4, +12]  (VIEW wins)

Also checks dose-1.82x V-R from research if Entity predictions exist.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import csv, json, math, os, pathlib, re, statistics, time
from collections import defaultdict
from typing import Any

ROOT = _public_path('experiments/archive/relation_learning/scripts/seed43222_vr_entity_crossover.py')
ROOT = _PUBLIC_ROOT

WS = _public_path('experiments/archive/relation_learning')
OUT = _public_path('experiments/archive/relation_learning/data/seed43222_vr_entity_crossover')
META = _public_path('experiments/archive/relation_learning/data/entity_relevant_update_analysis/entity_item_metadata.csv')
EVAL_DIR = _public_path('experiments/archive/relation_learning/data/seed43222_entity_eval')
CKS = ['chck_80M', 'chck_90M', 'chck_100M']

# Pre-registered V-R magnitude bands from research
PREREGISTERED_BANDS = {
    'rel_updates_0': (-12.0, -6.0),  # REPEAT wins at zero relevant updates
    'rel_updates_3': (4.0, 12.0),    # VIEW wins at 3+ relevant updates
    'rel_updates_4': (4.0, 12.0),
}
# Additional V-C band: zero-update about [-3,+2], rel4 about [+5,+10]
VC_BANDS = {
    'rel_updates_0': (-3.0, 2.0),
    'rel_updates_4': (5.0, 10.0),
}

# Prior two-seed late means for comparison (from research cross-seed summary)
PRIOR_VR_CROSS_SEED = {
    'rel_updates_0': -9.42,
    'rel_updates_1': +2.16,
    'rel_updates_2': +3.19,
    'rel_updates_3': +7.54,
    'rel_updates_4': +8.74,
    'rel_updates_5': +7.41,
}

def rel(p):
    try: return str(p.relative_to(ROOT))
    except: return str(p)

def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

def norm(s):
    return re.sub(r'\s+', ' ', str(s).strip().lower()).strip(' .')

def mean(xs):
    xs = [float(x) for x in xs if math.isfinite(float(x))]
    return statistics.mean(xs) if xs else float('nan')

def read_csv(p):
    with open(p, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f): yield r

def read_json(p):
    return json.loads(p.read_text(encoding='utf-8'))

def load_meta():
    by = defaultdict(list)
    for r in read_csv(META):
        r = dict(r)
        for k in ['item_index','reported_numops','relevant_updates','total_ops','irrelevant_ops','prefix_words','stale_available','stale_is_gold']:
            r[k] = int(r[k])
        by[r['uid']].append(r)
    for uid in by:
        by[uid].sort(key=lambda x: x['item_index'])
    return by

def find_prediction_file(arm_label, ck):
    """Find prediction JSON for D_V_43222 or D_R_43222."""
    payload_path = _public_path('experiments/archive/relation_learning/data/seed43222_entity_eval/per_target') / f'step009_{arm_label}_{ck}.json'
    if payload_path.exists():
        payload = read_json(payload_path)
        pred_rel = payload.get('tasks',{}).get('Entity',{}).get('predictions','')
        if pred_rel:
            pred_abs = ROOT / pred_rel
            if pred_abs.exists():
                return pred_abs
    return None

def score_arm(arm_code, arm_label, meta):
    """Score one arm across CKS, return list of item rows."""
    rows = []
    for ck in CKS:
        pred_file = find_prediction_file(arm_label, ck)
        if pred_file is None:
            print(f"[SKIP] {arm_label} {ck}: no prediction file found", flush=True)
            continue
        obj = read_json(pred_file)
        for uid, items in sorted(meta.items()):
            preds = obj.get(uid, {}).get('predictions', [])
            if len(preds) != len(items):
                print(f"[WARN] {uid} {arm_label} {ck}: pred={len(preds)} items={len(items)}", flush=True)
                continue
            for item, predrec in zip(items, preds):
                pred = str(predrec.get('pred', ''))
                correct = int(norm(pred) == norm(item['gold']))
                pred_stale = int(bool(item.get('stale_initial','')) and norm(pred) == norm(item.get('stale_initial','')))
                rows.append({
                    **item, 'seed': '43222', 'arm': arm_code,
                    'checkpoint': ck, 'pred': pred, 'correct': correct,
                    'pred_is_stale_initial': pred_stale,
                })
    return rows

def group_keys(r):
    relu = int(r['relevant_updates'])
    out = ['ALL', f'rel_updates_{relu}']
    if relu >= 1: out.append('rel_ge1')
    if relu >= 2: out.append('rel_ge2')
    if relu >= 3: out.append('rel_ge3')
    return out

def summarize(rows):
    d = defaultdict(list)
    for r in rows:
        for g in group_keys(r):
            d[(r['arm'], r['checkpoint'], g)].append(r)
    out = []
    for (arm, ck, g), vals in sorted(d.items()):
        n = len(vals)
        acc = 100 * sum(int(v['correct']) for v in vals) / n
        stale_cand = [v for v in vals if int(v['stale_available']) and not int(v['stale_is_gold'])]
        wrong_stale = [v for v in stale_cand if not int(v['correct'])]
        stale_pct = 100 * sum(int(v['pred_is_stale_initial']) for v in wrong_stale) / len(wrong_stale) if wrong_stale else float('nan')
        out.append({
            'arm': arm, 'checkpoint': ck, 'group': g,
            'n': n, 'accuracy_pct': acc,
            'stale_pick_pct': stale_pct,
            'mean_relevant_updates': mean([v['relevant_updates'] for v in vals]),
            'mean_total_ops': mean([v['total_ops'] for v in vals]),
            'mean_prefix_words': mean([v['prefix_words'] for v in vals]),
        })
    return out

def compute_late(summary):
    """Average over checkpoints."""
    d = defaultdict(list)
    for r in summary:
        d[(r['arm'], r['group'])].append(r)
    out = []
    for (arm, g), vals in sorted(d.items()):
        out.append({
            'arm': arm, 'group': g,
            'n_checkpoints': len(vals), 'n': int(vals[0]['n']),
            'late_mean_accuracy_pct': mean([v['accuracy_pct'] for v in vals]),
            'late_mean_stale_pct': mean([v['stale_pick_pct'] for v in vals]),
        })
    return out

def compute_contrasts(late_rows):
    """Compute V-R contrasts per group."""
    idx = {}
    for r in late_rows:
        idx[(r['arm'], r['group'])] = r

    contrasts = []
    for g in ['ALL', 'rel_updates_0', 'rel_updates_1', 'rel_updates_2',
              'rel_updates_3', 'rel_updates_4', 'rel_updates_5',
              'rel_ge1', 'rel_ge2', 'rel_ge3']:
        v = idx.get(('V', g))
        r = idx.get(('R', g))
        if v is None or r is None:
            continue
        vr = v['late_mean_accuracy_pct'] - r['late_mean_accuracy_pct']
        prior = PRIOR_VR_CROSS_SEED.get(g, float('nan'))
        band = PREREGISTERED_BANDS.get(g, None)
        in_band = None
        if band is not None:
            in_band = band[0] <= vr <= band[1]
        contrasts.append({
            'group': g, 'n': v['n'],
            'V_acc': round(v['late_mean_accuracy_pct'], 2),
            'R_acc': round(r['late_mean_accuracy_pct'], 2),
            'VminusR': round(vr, 2),
            'prior_cross_seed_VminusR': prior,
            'preregistered_band': f"[{band[0]}, {band[1]}]" if band else '',
            'in_band': in_band,
        })
    return contrasts

def write_csv_rows(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text('\n', encoding='utf-8'); return
    fields = list(rows[0].keys())
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

def write_note(contrasts, late_rows):
    lines = ['# research seed43222 V−R Entity crossover', '']
    lines.append('Pre-registered bands from research: V−R rel0 in [-12,-6], V−R rel3/rel4 in [+4,+12].')
    lines.append('Prior two-seed cross-seed V−R: rel0 -9.42, rel1 +2.16, rel2 +3.19, rel3 +7.54, rel4 +8.74, rel5 +7.41.')
    lines.append('')
    lines.append('## Seed43222 V−R contrasts (late mean 80M/90M/100M)')
    lines.append('')
    lines.append('| group | n | V acc | R acc | V−R | prior V−R | band | in_band |')
    lines.append('|---|---:|---:|---:|---:|---:|---|---|')
    for c in contrasts:
        lines.append(f"| {c['group']} | {c['n']} | {c['V_acc']:.2f} | {c['R_acc']:.2f} | {c['VminusR']:+.2f} | {c['prior_cross_seed_VminusR']:+.2f} | {c['preregistered_band']} | {c['in_band']} |")
    lines.append('')

    # Compute slope
    slope_data = [(int(c['group'].split('_')[-1]), c['VminusR'])
                  for c in contrasts if c['group'].startswith('rel_updates_')]
    slope_data.sort()
    if len(slope_data) >= 2:
        xs = [x for x, _ in slope_data]
        ys = [y for _, y in slope_data]
        x_mean = mean(xs); y_mean = mean(ys)
        num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
        den = sum((x - x_mean) ** 2 for x in xs)
        slope = num / den if den > 0 else float('nan')
        lines.append(f'V−R slope across relevant updates: {slope:+.3f} points/update')
        lines.append('')

    # Band check summary
    band_results = [(c['group'], c['in_band']) for c in contrasts if c['in_band'] is not None]
    all_pass = all(b for _, b in band_results)
    lines.append(f'## Pre-registered band check: {"ALL PASS" if all_pass else "SOME FAIL"}')
    for g, b in band_results:
        lines.append(f'- {g}: {"PASS" if b else "FAIL"}')
    lines.append('')

    lines.append('## Scientific reading')
    lines.append('')
    if all_pass:
        lines.append('The seed43222 V−R Entity crossover replicates within the pre-registered magnitude bands. '
                      'This is the third independent DeBERTa seed confirming the asymmetric fixed-budget dissociation: '
                      'exact recurrence supports unchanged-state retrieval, varied restatement supports multi-update '
                      'state discrimination.')
    else:
        lines.append('One or more pre-registered bands were missed. The crossover direction or magnitude must be '
                      'examined against the specific failures before claiming three-seed replication.')

    note_path = _public_path('research/notes/relation_learning/seed43222_vr_entity_crossover.md')
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return note_path

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = load_meta()

    # Score VIEW and REPEAT
    v_rows = score_arm('V', 'D_V_43222', meta)
    r_rows = score_arm('R', 'D_R_43222', meta)

    if not v_rows:
        print("ERROR: No VIEW seed43222 Entity predictions found", flush=True)
        return
    if not r_rows:
        print("ERROR: No REPEAT seed43222 Entity predictions found. Is the experiment_process complete?", flush=True)
        return

    all_rows = v_rows + r_rows
    summary = summarize(all_rows)
    late_rows = compute_late(summary)
    contrasts = compute_contrasts(late_rows)

    write_csv_rows(_public_path('experiments/archive/relation_learning/data/seed43222_vr_entity_crossover/seed43222_vr_entity_summary.csv'), summary)
    write_csv_rows(_public_path('experiments/archive/relation_learning/data/seed43222_vr_entity_crossover/seed43222_vr_entity_late.csv'), late_rows)
    write_csv_rows(_public_path('experiments/archive/relation_learning/data/seed43222_vr_entity_crossover/seed43222_vr_entity_contrasts.csv'), contrasts)

    note_path = write_note(contrasts, late_rows)

    result = {
        'status': 'SEED43222_VR_ENTITY_CROSSOVER_DONE',
        'finished_utc': now(),
        'v_checkpoints': len(set(r['checkpoint'] for r in v_rows)),
        'r_checkpoints': len(set(r['checkpoint'] for r in r_rows)),
        'contrasts': contrasts,
        'note': rel(note_path),
    }
    (_public_path('experiments/archive/relation_learning/data/seed43222_vr_entity_crossover/vr_crossover_result.json')).write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
