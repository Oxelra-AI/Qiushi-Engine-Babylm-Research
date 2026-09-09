#!/usr/bin/env python3
"""research: analyze delivered fit axes and COMPACT_EXPERIENCE benchmark columns.

This script is intentionally read-only with respect to older Sessions.  It writes
summary tables under relation_learning.  Goals:
  1. Check whether COMPACT_EXPERIENCE ALN-minus-OFF on the clean Strict-complement axis is
     broad across the 3,000 rows/sources or driven by one source.
  2. Preserve how the official_lengthmatched control filled its 10M-word budget,
     using existing COMPACT_EXPERIENCE audits rather than reconstructing streams.
  3. Summarize available COMPACT_EXPERIENCE cheap columns under the same research local
     accounting, especially OFF, ALN, and SHUF for seed43022.
  4. Summarize seed43022 up-dose ordinary-fit costs by subset and source where
     source is available.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import csv, json, math, pathlib, statistics, time
from collections import defaultdict

ROOT0 = _public_path('experiments/archive/relation_learning/scripts/analyze_fit_and_paired_context_columns.py')
ROOT = _PUBLIC_ROOT
OUT = ROOT/'experiments/archive/relation_learning/data/fit_and_paired_context_columns'
OUT.mkdir(parents=True, exist_ok=True)

STRICT_ROWS = ROOT/'experiments/archive/relation_learning/data/paired_context_off_aln_strict_complement/strict_fit_rows.csv'
STRICT_LATE = ROOT/'experiments/archive/relation_learning/data/paired_context_off_aln_strict_complement/strict_fit_late_summary.csv'
STRICT_AXIS_SUMMARY = ROOT/'experiments/archive/relation_learning/data/strict_complement_ngram_axis/summary.json'
DOSE_ROWS = ROOT/'experiments/archive/relation_learning/data/seed43022_trusted_ordinary_fit/ordinary_fit_rows.csv'
DOSE_RESULT = ROOT/'experiments/archive/relation_learning/data/seed43022_trusted_ordinary_fit/ordinary_fit_result.json'
COMPACT_EXPERIENCE_CONTROL = ROOT/'experiments/archive/compact_experience/data/control_eval_summary.json'
COMPACT_EXPERIENCE_EXPOSURE = ROOT/'experiments/archive/compact_experience/data/selected_exposure_audit.json'
COMPACT_EXPERIENCE_TOKEN = ROOT/'experiments/archive/compact_experience/data/tokenizer_factor_audit.json'
COMPACT_EXPERIENCE_PAIR = ROOT/'experiments/archive/compact_experience/data/pair_retention_and_interface_audit.json'

CHEAP = ['BLiMP','Supplement','EWoK','Entity','COMPS','GlobalPIQA','Reading']

def rel(p:pathlib.Path)->str:
    try: return str(p.relative_to(ROOT))
    except Exception: return str(p)

def mean(xs):
    xs=[float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return statistics.mean(xs) if xs else float('nan')

def se(xs):
    xs=[float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return statistics.stdev(xs)/math.sqrt(len(xs)) if len(xs)>1 else float('nan')

def read_csv(path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def write_csv(path, rows):
    if not rows:
        path.write_text('\n', encoding='utf-8'); return
    fields=[]
    seen=set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k); fields.append(k)
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def strict_per_source():
    rows=read_csv(STRICT_ROWS)
    # Average across late checkpoints per seed/relation/source/example first.
    by_key=defaultdict(list)
    for r in rows:
        if r['checkpoint'] not in {'chck_80M','chck_90M','chck_100M'}: continue
        key=(int(r['seed']), r['relation'], r['source'], r['example_id'])
        by_key[key].append(float(r['loss']))
    ex=[]
    for (seed,relat,source,eid), vals in by_key.items():
        ex.append({'seed':seed,'relation':relat,'source':source,'example_id':eid,'late_mean_loss':mean(vals)})
    by_src=defaultdict(list)
    for r in ex:
        by_src[(r['seed'], r['relation'], r['source'])].append(r['late_mean_loss'])
    summary=[]
    for (seed,relat,source), vals in sorted(by_src.items()):
        summary.append({'seed':seed,'relation':relat,'source':source,'n_rows':len(vals),'mean_loss':round(mean(vals),10),'se_rows':round(se(vals),10)})
    # paired ALN-OFF per seed/source/example
    idx={(r['seed'],r['source'],r['example_id'],r['relation']):r['late_mean_loss'] for r in ex}
    pair_by_src=defaultdict(list)
    for seed,relat,source,eid in by_key.keys():
        if relat!='ALN': continue
        a=idx.get((seed,source,eid,'ALN')); b=idx.get((seed,source,eid,'OFF'))
        if a is not None and b is not None:
            pair_by_src[(seed,source)].append(a-b)
    con=[]
    for (seed,source), vals in sorted(pair_by_src.items()):
        con.append({'seed':seed,'source':source,'n_rows':len(vals),'ALN_minus_OFF':round(mean(vals),10),'se_delta':round(se(vals),10),'fraction_ALN_lower_loss':round(sum(1 for v in vals if v<0)/len(vals),10)})
    pair_by_seed=defaultdict(list)
    for (seed,source), vals in pair_by_src.items(): pair_by_seed[seed].extend(vals)
    total=[]
    for seed, vals in sorted(pair_by_seed.items()):
        total.append({'seed':seed,'source':'ALL','n_rows':len(vals),'ALN_minus_OFF':round(mean(vals),10),'se_delta':round(se(vals),10),'fraction_ALN_lower_loss':round(sum(1 for v in vals if v<0)/len(vals),10)})
    write_csv(OUT/'paired_context_strict_complement_per_source_late_summary.csv', summary)
    write_csv(OUT/'paired_context_strict_complement_aln_minus_off_per_source.csv', total+con)
    return {'late_summary_csv': rel(OUT/'paired_context_strict_complement_per_source_late_summary.csv'), 'per_source_contrast_csv': rel(OUT/'paired_context_strict_complement_aln_minus_off_per_source.csv'), 'overall': total, 'per_source': con}

def dose_fit_per_source():
    if not DOSE_ROWS.exists(): return {'skipped':'missing dose rows'}
    rows=read_csv(DOSE_ROWS)
    by=defaultdict(list)
    for r in rows:
        if r.get('checkpoint') not in {'chck_80M','chck_90M','chck_100M'}: continue
        source=r.get('source','')
        subset=r.get('subset','') or r.get('axis','')
        by[(subset, r['arm'], r['dose'], source, r['example_id'])].append(float(r['loss']))
    ex=[]
    for (subset,arm,dose,source,eid), vals in by.items():
        ex.append({'subset':subset,'arm':arm,'dose':dose,'source':source,'example_id':eid,'late_mean_loss':mean(vals)})
    idx={(r['subset'],r['source'],r['example_id'],r['dose']):r['late_mean_loss'] for r in ex}
    con_by=defaultdict(list)
    for r in ex:
        if r['dose']=='base0': continue
        b=idx.get((r['subset'],r['source'],r['example_id'],'base0'))
        if b is not None:
            con_by[(r['subset'],r['source'],f"{r['dose']}-base0")].append(r['late_mean_loss']-b)
    out=[]
    for (subset,source,contrast), vals in sorted(con_by.items()):
        out.append({'subset':subset,'source':source or 'UNKNOWN','contrast':contrast,'n_rows':len(vals),'mean_delta_loss':round(mean(vals),10),'se_delta':round(se(vals),10),'fraction_dose_lower_loss':round(sum(1 for v in vals if v<0)/len(vals),10)})
    write_csv(OUT/'seed43022_dose_ordinary_fit_per_source_contrasts.csv', out)
    return {'per_source_contrast_csv': rel(OUT/'seed43022_dose_ordinary_fit_per_source_contrasts.csv'), 'sample': out[:20]}

def compact_experience_columns_and_budget():
    control=json.loads(COMPACT_EXPERIENCE_CONTROL.read_text(encoding='utf-8'))
    exposure=json.loads(COMPACT_EXPERIENCE_EXPOSURE.read_text(encoding='utf-8'))
    token=json.loads(COMPACT_EXPERIENCE_TOKEN.read_text(encoding='utf-8')) if COMPACT_EXPERIENCE_TOKEN.exists() else {}
    pair=json.loads(COMPACT_EXPERIENCE_PAIR.read_text(encoding='utf-8')) if COMPACT_EXPERIENCE_PAIR.exists() else {}
    targets=control['targets']
    wanted=['official_lengthmatched','qwen_clean_aligned','qwen_shuffled_control','official_lengthmatched_seed43122','qwen_clean_aligned_seed43122']
    rows=[]
    for name in wanted:
        if name not in targets: continue
        t=targets[name]
        row={'target':name}
        for c in CHEAP:
            row[c]=t.get(c)
        vals=[t.get(c) for c in CHEAP if t.get(c) is not None]
        row['cheap7']=round(mean(vals),10)
        rows.append(row)
    idx={r['target']:r for r in rows}
    contrasts=[]
    for a,b,label in [
        ('qwen_clean_aligned','official_lengthmatched','seed43022_ALN_minus_OFF'),
        ('qwen_shuffled_control','official_lengthmatched','seed43022_SHUF_minus_OFF'),
        ('qwen_clean_aligned','qwen_shuffled_control','seed43022_ALN_minus_SHUF'),
        ('qwen_clean_aligned_seed43122','official_lengthmatched_seed43122','seed43122_ALN_minus_OFF'),
    ]:
        if a in idx and b in idx:
            cr={'contrast':label,'a':a,'b':b}
            for c in CHEAP+['cheap7']:
                cr[c+'_delta']=round(float(idx[a][c])-float(idx[b][c]),10)
            contrasts.append(cr)
    write_csv(OUT/'paired_context_available_cheap_columns.csv', rows)
    write_csv(OUT/'paired_context_available_cheap_column_contrasts.csv', contrasts)
    # Budget/distribution facts available from audits.
    official_src={}
    try:
        # research stores official source words under one tokenizer section; robustly locate keys.
        data=token.get('audit', token)
        def walk(obj):
            if isinstance(obj, dict):
                if all(f'official_lengthmatched::{s}' in obj for s in ['bnc_spoken','childes','gutenberg','open_subtitles','simple_wiki','switchboard']):
                    return obj
                for v in obj.values():
                    got=walk(v)
                    if got: return got
            return None
        src_block=walk(data)
        if src_block:
            official_src={k.split('::',1)[1]: {'rows':v.get('rows'), 'words':v.get('words'), 'over256_rate':v.get('over256_rate'), 'tokens_per_word':v.get('tokens_per_word')} for k,v in src_block.items() if k.startswith('official_lengthmatched::')}
    except Exception as e:
        official_src={'error':str(e)}
    qwen_budget=exposure.get('arms',{}).get('qwen_clean_aligned',{})
    strict_axis=json.loads(STRICT_AXIS_SUMMARY.read_text(encoding='utf-8'))
    budget={
        'selected_pairs': exposure.get('selected_pairs'),
        'selected_pair_words': exposure.get('selected_pair_words'),
        'selected_pair_words_by_source': exposure.get('selected_pair_words_by_source'),
        'qwen_clean_aligned_budget': qwen_budget,
        'official_lengthmatched_source_words_from_token_audit': official_src,
        'strict_complement_axis_actual_rows_by_source': strict_axis.get('candidate_axis',{}).get('actual_rows_by_source'),
        'interpretation': 'official_lengthmatched is not a no-ordinary-text arm; it is the official-only lengthmatched 10M control, while ALN replaces a selected-row word budget with 1.6568M explicit original+rewrite pair words plus selected-ID filler to match the selected row budget. Source distributions differ, so OFF-ALN fit must be read with source/per-row breakdown.'
    }
    (OUT/'paired_context_budget_and_source_notes.json').write_text(json.dumps(budget,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return {'columns_csv': rel(OUT/'paired_context_available_cheap_columns.csv'), 'contrasts_csv': rel(OUT/'paired_context_available_cheap_column_contrasts.csv'), 'contrasts': contrasts, 'budget_json': rel(OUT/'paired_context_budget_and_source_notes.json'), 'budget': budget}

def main():
    strict=strict_per_source()
    dose=dose_fit_per_source()
    compact_experience=compact_experience_columns_and_budget()
    dose_result=json.loads(DOSE_RESULT.read_text(encoding='utf-8')) if DOSE_RESULT.exists() else {}
    summary={'status':'FIT_AND_COMPACT_EXPERIENCE_COLUMNS_DONE','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),'strict_off_aln_per_source':strict,'seed43022_dose_ordinary_fit':dose,'compact_experience_columns_and_budget':compact_experience,'seed43022_dose_ordered_triples_late':dose_result.get('ordered_triples_late'),'notes':['For strict-complement OFF-ALN, negative ALN_minus_OFF means ALN lower loss.','For dose ordinary fit, positive dose-base means added dose worsens fit on that axis.','COMPACT_EXPERIENCE cheap-column summaries are old research accounting and inherit the research-64 exposure qualification.']}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    md=['# research fit and COMPACT_EXPERIENCE column analysis','',f"Summary JSON: `{rel(OUT/'summary.json')}`",'','## Strict-complement OFF -> ALN per-source late contrasts','']
    for r in strict['overall']:
        md.append(f"- seed {r['seed']} ALL: ALN-minus-OFF {r['ALN_minus_OFF']:+.4f} nats, SE {r['se_delta']:.4f}, ALN lower on {100*r['fraction_ALN_lower_loss']:.1f}% rows.")
    md += ['',f"Per-source CSV: `{strict['per_source_contrast_csv']}`",'','## Seed43022 dose ordinary-fit costs','']
    for t in dose_result.get('ordered_triples_late',[]) or []:
        md.append(f"- {t['subset']}: dose21-base {t['dose21_minus_base0']:+.4f}, dose25-base {t['dose25_minus_base0']:+.4f}, dose25-dose21 {t['dose25_minus_dose21']:+.4f} nats.")
    md += ['',f"Dose per-source CSV: `{dose.get('per_source_contrast_csv')}`",'','## COMPACT_EXPERIENCE available cheap-column contrasts','']
    for cr in compact_experience['contrasts']:
        md.append(f"- {cr['contrast']}: cheap7 {cr['cheap7_delta']:+.3f}, BLiMP {cr['BLiMP_delta']:+.3f}, EWoK {cr['EWoK_delta']:+.3f}, Entity {cr['Entity_delta']:+.3f}, Reading {cr['Reading_delta']:+.3f}.")
    md += ['',f"Budget/source notes: `{compact_experience['budget_json']}`",'']
    ((_PUBLIC_ROOT / 'research/documents/relation_learning/data/fit_and_paired_context_columns/summary.md')).write_text('\n'.join(md),encoding='utf-8')
    print(json.dumps({'status':summary['status'],'summary':rel(OUT/'summary.json'),'md':rel((_PUBLIC_ROOT / 'research/documents/relation_learning/data/fit_and_paired_context_columns/summary.md'))},indent=2),flush=True)
if __name__=='__main__': main()
