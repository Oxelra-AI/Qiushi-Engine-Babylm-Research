#!/usr/bin/env python3
"""Synthesize all research GlobalPIQA margin-reader per-target JSONs."""
from __future__ import annotations
import csv, json
from pathlib import Path

ROOT=Path('experiments/archive/representation_and_objectives')
IN=ROOT/'data/globalpiqa_margin_reader'
OUT=ROOT/'data/globalpiqa_margin_synthesis'
NOTE=(ROOT.parents[2] / 'research/notes/representation_and_objectives/globalpiqa_margin_synthesis.md')
TARGETS=['legal40k8x480_43022','legal16k8x480_43022','depth12x384_43022','inherited16k_noncompliant_43022']
OUT.mkdir(parents=True, exist_ok=True)
rows=[]
details={}
for t in TARGETS:
    p=IN/f'{t}_margins.json'
    if not p.exists():
        rows.append({'target':t,'status':'missing'})
        continue
    d=json.load(open(p))
    details[t]=d
    for mode,md in d['modes'].items():
        s=md['summary']
        aw=s.get('always_wrong_subset') or {}
        rows.append({
            'target':t,
            'label':d['label'],
            'mode':mode,
            'n':s['n'],
            'accuracy':s['accuracy'],
            'chance_adjusted_accuracy':s['chance_adjusted_accuracy'],
            'rank1':s['correct_rank_counts'].get('1',0),
            'rank2':s['correct_rank_counts'].get('2',0),
            'rank3':s['correct_rank_counts'].get('3',0),
            'rank4':s['correct_rank_counts'].get('4',0),
            'always_wrong_n':aw.get('n'),
            'always_wrong_rank2':(aw.get('correct_rank_counts') or {}).get('2',0),
            'always_wrong_rank3':(aw.get('correct_rank_counts') or {}).get('3',0),
            'always_wrong_rank4':(aw.get('correct_rank_counts') or {}).get('4',0),
            'always_wrong_mean_top_minus_correct':aw.get('mean_top_minus_correct'),
            'always_wrong_median_top_minus_correct':aw.get('median_top_minus_correct'),
            'always_wrong_small_margin_le_0p25':aw.get('small_wrong_margin_le_0p25_nats'),
            'always_wrong_small_margin_le_0p50':aw.get('small_wrong_margin_le_0p50_nats'),
            'choice_counts_json':json.dumps(s['choice_counts'], sort_keys=True),
        })
with (OUT/'globalpiqa_margin_summary.csv').open('w',newline='') as f:
    fieldnames=list(rows[0].keys())
    w=csv.DictWriter(f,fieldnames=fieldnames,extrasaction='ignore')
    w.writeheader(); w.writerows(rows)
payload={
    'status':'GLOBALPIQA_MARGIN_SYNTHESIS_DONE',
    'inputs':[str(IN/f'{t}_margins.json') for t in TARGETS],
    'summary_rows':rows,
    'scientific_interpretation':{
        'key_result':'Across the best compliant legal40k, legal16k, depth, and inherited-tokenizer 42.033 endpoint, GlobalPIQA_parallel remains at chance (~22-26%). On the 52 rows wrong for all inspected endpoints, the correct option is usually rank 3 or 4, not a close rank-2 near miss.',
        'calibration_status':'This weakens simple inference-time calibration/prior subtraction as the main missing lever. A calibration might rescue a handful of rank-2 or small-margin rows, but the dominant failure is missing conditional physical/spatial/temporal/affordance structure.',
        'sota_relevance':'Because a 10-point parallel gain yields +5 GlobalPIQA and +0.56 Overall, this remains a high-leverage route, but it likely needs general contrastive/consequence data or objective work rather than just score postprocessing.',
        'next_use':'When A02/A01 FW compact/breadth checkpoints complete, run this same margin reader on their 70M/80M/100M checkpoints to see whether any arm actually moves rank distribution on hard rows rather than only perturbing chosen labels.'
    },
    'details':details,
}
(OUT/'globalpiqa_margin_synthesis.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False))
lines=[]
lines.append('# research — GlobalPIQA margin synthesis')
lines.append('')
lines.append('## Core result')
lines.append('')
lines.append('The official prediction artifacts had shown GlobalPIQA_parallel at ~22–28%, but not whether wrong answers were near ties. The CPU margin reader mirrors the official MLM length-normalized completion scoring and records all four option scores. It shows the failure is mostly not a near-tie/calibration issue.')
lines.append('')
lines.append('| target | parallel acc | chance-adjusted | rank1 | rank2 | rank3 | rank4 | 52-row hard rank2/rank3/rank4 | hard mean top-correct nats |')
lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
for r in rows:
    if r.get('mode')!='parallel': continue
    lines.append(f"| `{r['target']}` | {float(r['accuracy']):.2f} | {float(r['chance_adjusted_accuracy']):.3f} | {r['rank1']} | {r['rank2']} | {r['rank3']} | {r['rank4']} | {r['always_wrong_rank2']}/{r['always_wrong_rank3']}/{r['always_wrong_rank4']} | {float(r['always_wrong_mean_top_minus_correct']):.3f} |")
lines.append('')
lines.append('## Scientific reading')
lines.append('')
lines.append('- The best compliant endpoint (`legal40k8x480_43022`) gets 23/103 correct, with correct-option ranks 1/2/3/4 = 23/24/26/30. On the 52 cross-endpoint hard rows, ranks are 2/3/4 = 11/18/23 and mean top-minus-correct is 1.95 nats; only 8 hard wrong rows have margin <=0.50 nats.')
lines.append('- Legal16k, depth, and inherited-tokenizer endpoints show the same pattern. Even the noncompliant 42.033 endpoint has hard-row ranks 2/3/4 = 10/24/18 and mean top-minus-correct 1.84 nats.')
lines.append('- Therefore GlobalPIQA_parallel is a stable conditional-world-knowledge/reasoning weakness of the lineage, not a legal-tokenizer artifact and not mainly a simple answer-position bias or small-margin scoring artifact. Inference-time calibration could be studied only on corpus-derived validation, but it is unlikely to supply the whole SOTA gap.')
lines.append('- Future compact/breadth FW results should be read with this margin lens: a real mechanism should improve hard-row rank distribution and reduce top-minus-correct margins, not merely flip a few noisy labels at 100M.')
lines.append('')
lines.append('## Files')
lines.append('')
lines.append(f"- summary JSON: `{OUT/'globalpiqa_margin_synthesis.json'}`")
lines.append(f"- summary CSV: `{OUT/'globalpiqa_margin_summary.csv'}`")
lines.append(f"- per-target detailed margins: `{IN}/<target>_margins.json` and `<target>_parallel_rows.csv`")
NOTE.write_text('\n'.join(lines)+'\n')
print(json.dumps({'status':payload['status'],'json':str(OUT/'globalpiqa_margin_synthesis.json'),'csv':str(OUT/'globalpiqa_margin_summary.csv'),'note':str(NOTE)},indent=2))
