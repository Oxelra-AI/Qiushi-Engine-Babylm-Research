#!/usr/bin/env python3
from __future__ import annotations
import json, pathlib

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
S2 = ROOT/'data/true_s2_100m_available_coordinate.json'
S2_E = ROOT/'data/true_s2_100m_full_ewok_word_tokenize_score.json'
S1 = ROOT/'data/s1_100m_available_coordinate.json'
S1_E = ROOT/'data/s1_100m_full_ewok_word_tokenize_score.json'
P = ROOT/'data/debertav2_b256_true_9of9_coordinate.json'
OUT = ROOT/'data/true_s2_vs_references_comparison.json'
NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/true_s2_vs_references_comparison.md')

def load(p): return json.loads(p.read_text(encoding='utf-8'))
def rr(x): return round(float(x), 4)

def main():
    s2=load(S2); s2e=load(S2_E); s1=load(S1); s1e=load(S1_E); p=load(P)
    def avail(obj):
        sc=obj['scores']; return {
            'BLiMP': sc['blimp'],
            'Supplement': sc['supplement'],
            'Entity': sc['entity_tracking'],
            'COMPS': sc['comps'],
            'GlobalPIQA_parallel': sc['global_piqa_parallel'],
            'GlobalPIQA_nonparallel': sc['global_piqa_nonparallel'],
            'GlobalPIQA': obj['derived_columns']['GlobalPIQA_mean_parallel_nonparallel'],
            'Reading_eye': sc['reading_eye_tracking'],
            'Reading_self_paced': sc['reading_self_paced'],
            'Reading': obj['derived_columns']['Reading_mean_eye_selfpaced'],
        }
    s2a=avail(s2); s1a=avail(s1)
    s2a['EWoK']=s2e['ewok_full_score']; s1a['EWoK']=s1e['ewok_full_score']
    pa={
        'BLiMP': p['scores_official_columns']['BLiMP'],
        'Supplement': p['scores_official_columns']['BLiMP Supplement'],
        'EWoK': p['scores_official_columns']['EWoK'],
        'Entity': p['scores_official_columns']['Entity Tracking'],
        'COMPS': p['scores_official_columns']['COMPS'],
        'GlobalPIQA': p['scores_official_columns']['GlobalPIQA'],
        'Reading': p['scores_official_columns']['Reading'],
    }
    leader={'BLiMP':67.20,'Supplement':56.01,'EWoK':56.07,'Entity':28.45,'COMPS':53.57,'GlobalPIQA':39.67,'Reading':5.42,'SuperGLUE':69.79,'AoA':0.0,'Overall':41.80}
    columns=['BLiMP','Supplement','EWoK','Entity','COMPS','GlobalPIQA','Reading']
    rows=[]
    for c in columns:
        rows.append({'column':c,'true_s2_100M':rr(s2a[c]),'s1_100M':rr(s1a[c]),'s2_minus_s1':rr(s2a[c]-s1a[c]),'protected_8x480_100M':rr(pa[c]),'s2_minus_protected':rr(s2a[c]-pa[c]),'visible_leader':rr(leader[c]),'s2_minus_visible_leader':rr(s2a[c]-leader[c])})
    available_mean_s2=sum(s2a[c] for c in columns)/len(columns)
    available_mean_s1=sum(s1a[c] for c in columns)/len(columns)
    available_mean_p=sum(pa[c] for c in columns)/len(columns)
    nlp7_s2=(s2a['BLiMP']+s2a['Supplement']+s2a['EWoK']+s2a['Entity']+s2a['COMPS']+s2a['GlobalPIQA'])/6
    nlp7_s1=(s1a['BLiMP']+s1a['Supplement']+s1a['EWoK']+s1a['Entity']+s1a['COMPS']+s1a['GlobalPIQA'])/6
    payload={
        'status':'TRUE_S2_VS_REFERENCES_COMPARISON',
        'input_files':{'true_s2_available':str(S2),'true_s2_ewok':str(S2_E),'s1_available':str(S1),'s1_ewok':str(S1_E),'protected':str(P)},
        'rows':rows,
        'available_7col_means':{'true_s2':rr(available_mean_s2),'s1':rr(available_mean_s1),'protected':rr(available_mean_p),'true_s2_minus_s1':rr(available_mean_s2-available_mean_s1),'true_s2_minus_protected':rr(available_mean_s2-available_mean_p)},
        'available_6_nlp_no_superglue':{'true_s2':rr(nlp7_s2),'s1':rr(nlp7_s1),'true_s2_minus_s1':rr(nlp7_s2-nlp7_s1)},
        'leader_anchor':leader,
        'interpretation':[
            'True S2 is not a positive overall curriculum component on the official-corpus baseline16k S1 base: it improves only GlobalPIQA (+1.03 vs S1) and Reading (+0.27), while damaging BLiMP (-2.60), Supplement (-1.22), EWoK (-0.38), Entity (-1.77), and COMPS (-1.65).',
            'The 100M word-clock curriculum successfully tests the real leader-style schedule; the result must not be replaced by compressed 10M probes. Its column pattern says late token masking/length curriculum over official corpus/baseline16k trades grammar/entity/EWoK for GlobalPIQA/nonparallel PIQA rather than closing the leader package.',
            'S2 remains far from the visible leader on Entity (-9.98), EWoK (-4.43), COMPS (-2.96), BLiMP (-2.96), and GlobalPIQA (-1.04). It exceeds the leader on Supplement and Reading, but these are not the central gaps.',
            'Because both S1 and S2 fail to move Entity/EWoK upward, the missing leader factor is unlikely to be architecture shape plus curriculum alone under official-corpus/baseline16k/AdamW. The next high-value route should test the legal representation/data side (40k tokenizer and/or legally reconstructed simplification pairs) or the planned GPT-BERT/MNTP hybrid, not continue stacking curriculum variants on S1.'
        ]
    }
    OUT.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    lines=['# research — true S2 100M vs S1/protected/visible leader','',f'Evidence JSON: `{OUT}`','','| column | true S2 100M | S1 100M | S2-S1 | protected 8×480 | S2-protected | visible leader | S2-leader |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['column']} | {row['true_s2_100M']:.2f} | {row['s1_100M']:.2f} | {row['s2_minus_s1']:+.2f} | {row['protected_8x480_100M']:.2f} | {row['s2_minus_protected']:+.2f} | {row['visible_leader']:.2f} | {row['s2_minus_visible_leader']:+.2f} |")
    lines += ['','## Aggregate available columns','',f"- 7-column available mean (BLiMP/Supp/EWoK/Entity/COMPS/GlobalPIQA/Reading): true S2 {available_mean_s2:.3f}, S1 {available_mean_s1:.3f}, protected {available_mean_p:.3f}; S2-S1 {available_mean_s2-available_mean_s1:+.3f}.",f"- 6-column NLP subset without SuperGLUE: true S2 {nlp7_s2:.3f}, S1 {nlp7_s1:.3f}; S2-S1 {nlp7_s2-nlp7_s1:+.3f}.",'','## Interpretation','']
    lines += [f'- {x}' for x in payload['interpretation']]
    NOTE.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':payload['status'],'out':str(OUT),'note':str(NOTE),'available_7col_means':payload['available_7col_means'],'key_s2_minus_s1':{r['column']:r['s2_minus_s1'] for r in rows}},indent=2,ensure_ascii=False))
if __name__=='__main__': main()
