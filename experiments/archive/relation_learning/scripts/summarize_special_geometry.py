#!/usr/bin/env python3
"""Summarize research special-token geometry KL bins into interpretable mass shares."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import csv, json, pathlib, time
from collections import defaultdict

ROOT = _public_path('.')
SRC = _public_path('experiments/archive/relation_learning/data/special_geometry_kl_bins')
OUT = _public_path('experiments/archive/relation_learning/data/special_geometry_interpretation')


def rel(p):
    try: return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception: return str(p)


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def f(x):
    try: return float(x)
    except Exception: return 0.0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dist = read_csv(_public_path('experiments/archive/relation_learning/data/special_geometry_kl_bins/kl_by_distance.csv'))
    # totals by endpoint/form and useful grouped faces.
    totals = defaultdict(lambda: {"tokens":0.0, "kl_mass":0.0})
    groups = defaultdict(lambda: {"tokens":0.0, "kl_mass":0.0})
    for r in dist:
        key = (r['endpoint'], r['form'])
        tok = f(r['tokens']); mass = tok * f(r['kl_mean'])
        totals[key]['tokens'] += tok; totals[key]['kl_mass'] += mass
        db = r['distance_bin']; kind = r['token_kind']; lb = r['row_length_bin']
        labels = []
        if kind == 'special': labels.append('special_tokens')
        if kind == 'content' and db == '1': labels.append('content_distance_1')
        if kind == 'content' and db in {'1','2-3','4-7'}: labels.append('content_distance_1_to_7')
        if kind == 'content' and db not in {'none'}:
            # Parse near/far in a simple way.
            if db in {'1','2-3','4-7','8-15'}: labels.append('content_distance_1_to_15')
            else: labels.append('content_distance_ge16')
        if lb in {'0-15','16-31'}: labels.append('short_0_to_31_tokens')
        if lb in {'0-15','16-31','32-63'}: labels.append('short_0_to_63_tokens')
        for lab in labels:
            g = groups[(r['endpoint'], r['form'], lab)]
            g['tokens'] += tok; g['kl_mass'] += mass
    records=[]
    for (endpoint, form), tot in sorted(totals.items()):
        rec={'endpoint':endpoint,'form':form,'tokens':int(tot['tokens']),'kl_mean':tot['kl_mass']/max(1.0,tot['tokens']),'kl_mass':tot['kl_mass']}
        for lab in ['special_tokens','content_distance_1','content_distance_1_to_7','content_distance_1_to_15','content_distance_ge16','short_0_to_31_tokens','short_0_to_63_tokens']:
            g=groups.get((endpoint,form,lab), {'tokens':0.0,'kl_mass':0.0})
            rec[f'{lab}_token_share']=g['tokens']/max(1.0,tot['tokens'])
            rec[f'{lab}_kl_mass_share']=g['kl_mass']/max(1e-12,tot['kl_mass'])
            rec[f'{lab}_kl_mean']=g['kl_mass']/max(1.0,g['tokens'])
        records.append(rec)
    out_csv=_public_path('experiments/archive/relation_learning/data/special_geometry_interpretation/special_geometry_mass_shares.csv')
    fields=[]; seen=set()
    for r in records:
        for k in r:
            if k not in seen: fields.append(k); seen.add(k)
    with out_csv.open('w',newline='',encoding='utf-8') as fcsv:
        w=csv.DictWriter(fcsv,fieldnames=fields); w.writeheader(); w.writerows(records)
    summary={'status':'SPECIAL_GEOMETRY_INTERPRETATION','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'source':rel(_public_path('experiments/archive/relation_learning/data/special_geometry_kl_bins/special_geometry_kl_bins.json')),'records':records}
    out_json=_public_path('experiments/archive/relation_learning/data/special_geometry_interpretation/special_geometry_interpretation.json'); out_md=_public_path('research/documents/relation_learning/data/special_geometry_interpretation/special_geometry_interpretation.md')
    out_json.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    lines=['# research special-token geometry interpretation','', 'This summarizes KL mass from `special_geometry_kl_bins` on the same 32 matched examples used by research.  Mass share is the fraction of total private/slow KL in that form, not a count of items.','', '| endpoint | form | KL mean | special mass | content dist=1 mass | content dist<=15 mass | far-content mass | short<=31 mass |', '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in records:
        if r['form'] not in {'short_add_special','coherent_add_special'}: continue
        lines.append(f"| {r['endpoint']} | {r['form']} | {r['kl_mean']:.8f} | {100*r['special_tokens_kl_mass_share']:.1f}% | {100*r['content_distance_1_kl_mass_share']:.1f}% | {100*r['content_distance_1_to_15_kl_mass_share']:.1f}% | {100*r['content_distance_ge16_kl_mass_share']:.1f}% | {100*r['short_0_to_31_tokens_kl_mass_share']:.1f}% |")
    lines += ['', 'Main reading: compare coherent-special endpoints to coherent86.  If the extra KL mass concentrates at special tokens and adjacent content, then the branch is not merely off-leash globally; it is using a geometry channel created by adding special tokens to inputs whose relative-position geometry differs sharply between 256-token coherent rows and short evaluation rows.', '', f'CSV: `{rel(out_csv)}`', f'JSON: `{rel(out_json)}`']
    out_md.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':summary['status'],'out_json':rel(out_json),'out_md':rel(out_md)},indent=2),flush=True)

if __name__=='__main__': main()
