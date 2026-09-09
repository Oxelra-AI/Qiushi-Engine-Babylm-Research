#!/usr/bin/env python3
from __future__ import annotations
import json, os, pathlib, re, subprocess, sys
ROOT=pathlib.Path('experiments/archive/initial_model_studies')
STRICT=ROOT/'repos/babylm-eval/strict'; FULL=STRICT/'evaluation_data/full_eval'; BACKEND='mlm'
RUN=ROOT/'training/runs/babylm_uniform_s1_10M'; CKPT='chck_9999997w'; MODEL=(RUN/'hf_model'/CKPT).resolve()
OUTDIR=(RUN/'eval_results_available_direct').resolve()
OUT_JSON=ROOT/'data/uniform_10m_available_coordinate.json'
HIGH_JSON=ROOT/'data/high_entity_state_10m_available_coordinate.json'
LOW_JSON=ROOT/'data/matched_low_10m_available_coordinate.json'
EWOK_JSON=ROOT/'data/structure_arms_full_ewok.json'
THREE_JSON=ROOT/'data/structure_density_three_arm_comparison.json'
NOTE=(ROOT.parents[2] / 'research/notes/initial_model_studies/structure_density_three_arm_comparison.md')
LOG=(ROOT.parents[2] / 'research/notes/initial_model_studies/uniform_available_eval.log')
TASKS=[('blimp','blimp',FULL/'blimp_filtered','blimp_filtered'),('supplement','blimp',FULL/'supplement_filtered','supplement_filtered'),('entity_tracking','entity_tracking',FULL/'entity_tracking','entity_tracking'),('comps','comps',FULL/'comps','comps'),('global_piqa_parallel','global_piqa_parallel',FULL/'global_piqa_parallel','global_piqa_parallel'),('global_piqa_nonparallel','global_piqa_nonparallel',FULL/'global_piqa_nonparallel','global_piqa_nonparallel')]

def env():
    e=os.environ.copy(); hf=ROOT/'training/hf_home'
    e['HF_HOME']=str(hf.resolve()); e['HF_HUB_CACHE']=str((hf/'hub').resolve()); e['TRANSFORMERS_CACHE']=str((hf/'transformers').resolve()); e['HF_MODULES_CACHE']=str((ROOT/'training/hf_modules_cache').resolve()); e.setdefault('TOKENIZERS_PARALLELISM','false')
    for k in ['HF_HOME','HF_HUB_CACHE','TRANSFORMERS_CACHE','HF_MODULES_CACHE']: pathlib.Path(e[k]).mkdir(parents=True, exist_ok=True)
    return e

def run(cmd,e,logf):
    line='$ '+' '.join(map(str,cmd)); print(line,flush=True); logf.write('\n'+line+'\n'); logf.flush()
    p=subprocess.run([str(x) for x in cmd],cwd=str(STRICT),env=e,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    print(p.stdout[-2200:],flush=True); logf.write(p.stdout+f'\n[returncode={p.returncode}]\n'); logf.flush()
    if p.returncode!=0: raise RuntimeError(f'failed {p.returncode}: {line}\n{p.stdout[-8000:]}')

def avg(path):
    txt=path.read_text(encoding='utf-8',errors='replace'); m=re.search(r'AVERAGE ACCURACY\s*\n([0-9.\-]+)',txt)
    if m: return float(m.group(1))
    raise RuntimeError(f'no avg {path}\n{txt[:1000]}')

def read_reading(path):
    txt=path.read_text(encoding='utf-8',errors='replace'); out={}
    for lab,key in [('EYE TRACKING SCORE','reading_eye_tracking'),('SELF-PACED READING SCORE','reading_self_paced')]:
        m=re.search(re.escape(lab)+r':\s*([0-9.\-]+)',txt)
        if not m: raise RuntimeError(f'no {lab} in {path}')
        out[key]=float(m.group(1))
    return out

def find(task,dataset):
    hits=list(OUTDIR.rglob(f'zero_shot/{BACKEND}/{task}/{dataset}/best_temperature_report.txt'))
    if len(hits)!=1: raise RuntimeError(f'report not unique {task}/{dataset}: {hits[:8]}')
    return hits[0]

def main():
    if not MODEL.exists(): raise RuntimeError(f'missing {MODEL}')
    e=env(); OUTDIR.mkdir(parents=True,exist_ok=True); LOG.parent.mkdir(parents=True,exist_ok=True)
    scores={}; reports={}
    with LOG.open('w',encoding='utf-8') as logf:
        for col,task,data,dataset in TASKS:
            run([sys.executable,'-m','evaluation_pipeline.sentence_zero_shot.run','--model_path_or_name',str(MODEL),'--backend',BACKEND,'--task',task,'--data_path',str(data.resolve()),'--save_predictions','--revision_name','direct_'+CKPT,'--batch_size','64','--output_dir',str(OUTDIR)],e,logf)
            rep=find(task,dataset); scores[col]=avg(rep); reports[col]=str(rep)
        run([sys.executable,'-m','evaluation_pipeline.reading.run','--model_path_or_name',str(MODEL),'--backend',BACKEND,'--data_path',str((FULL/'reading/reading_data.csv').resolve()),'--revision_name','direct_'+CKPT,'--output_dir',str(OUTDIR)],e,logf)
    rhits=list(OUTDIR.rglob(f'zero_shot/{BACKEND}/reading/report.txt'))
    if not rhits: raise RuntimeError('missing reading report')
    reports['reading']=str(rhits[-1]); scores.update(read_reading(rhits[-1]))
    metrics=json.loads((RUN/'scientific_metrics.json').read_text())
    payload={'status':'UNIFORM_STRUCTURE_DENSITY_AVAILABLE_COORDINATE','arm':'uniform','run':str(RUN),'model_path_direct':str(MODEL),'scores':scores,'derived_columns':{'GlobalPIQA_mean_parallel_nonparallel':(scores['global_piqa_parallel']+scores['global_piqa_nonparallel'])/2.0,'Reading_mean_eye_selfpaced':(scores['reading_eye_tracking']+scores['reading_self_paced'])/2.0},'training_core':{k:metrics.get(k) for k in ['word_exposure','actual_training_steps','optimizer_steps','loss_first','loss_last','masked_tokens_total','masked_tokens_per_whitespace_word','example_jsonl_label','example_jsonl_total_words','example_jsonl_total_rows']},'tokenization_coupling_summary':metrics.get('tokenization_coupling_summary'),'reports':reports,'missing_columns':['SuperGLUE','AoA']}
    OUT_JSON.parent.mkdir(parents=True,exist_ok=True); OUT_JSON.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n')
    high=json.loads(HIGH_JSON.read_text()); low=json.loads(LOW_JSON.read_text()); ewok=json.loads(EWOK_JSON.read_text())
    arms={'high_entity_state':high,'matched_low':low,'uniform':payload}
    comp={'status':'STRUCTURE_DENSITY_THREE_ARM_COMPARISON','available_scores':{a:arms[a]['scores'] for a in arms},'derived_columns':{a:arms[a]['derived_columns'] for a in arms},'full_ewok_scores':ewok['scores'],'deltas':{},'jsons':{'high_entity_state':str(HIGH_JSON),'matched_low':str(LOW_JSON),'uniform':str(OUT_JSON),'ewok':str(EWOK_JSON)},'interpretation_warning':'Three-arm route screen only; within-source lexical score remains coupled to cue density/topic/syntax.'}
    for a,b in [('high_entity_state','matched_low'),('high_entity_state','uniform'),('uniform','matched_low')]:
        comp['deltas'][f'{a}_minus_{b}']={k:arms[a]['scores'][k]-arms[b]['scores'][k] for k in arms[a]['scores'] if k in arms[b]['scores']}
        comp['deltas'][f'{a}_minus_{b}'].update({f'EWoK_full':ewok['scores'][a]-ewok['scores'][b],'GlobalPIQA_mean':arms[a]['derived_columns']['GlobalPIQA_mean_parallel_nonparallel']-arms[b]['derived_columns']['GlobalPIQA_mean_parallel_nonparallel'],'Reading_mean':arms[a]['derived_columns']['Reading_mean_eye_selfpaced']-arms[b]['derived_columns']['Reading_mean_eye_selfpaced']})
    THREE_JSON.write_text(json.dumps(comp,indent=2,ensure_ascii=False)+'\n')
    order=['high_entity_state','matched_low','uniform']; rows=[('blimp','BLiMP'),('supplement','Supplement'),('entity_tracking','Entity'),('comps','COMPS'),('global_piqa_parallel','GPIQA parallel'),('global_piqa_nonparallel','GPIQA nonparallel')]
    lines=['# research — structure-density three-arm screen','',f'Evidence JSON: `{THREE_JSON}`','', '| column | high_entity_state | matched_low | uniform |','|---|---:|---:|---:|']
    for k,label in rows: lines.append(f'| {label} | {high["scores"][k]:.2f} | {low["scores"][k]:.2f} | {payload["scores"][k]:.2f} |')
    lines.append(f'| GlobalPIQA mean | {high["derived_columns"]["GlobalPIQA_mean_parallel_nonparallel"]:.2f} | {low["derived_columns"]["GlobalPIQA_mean_parallel_nonparallel"]:.2f} | {payload["derived_columns"]["GlobalPIQA_mean_parallel_nonparallel"]:.2f} |')
    lines.append(f'| Reading mean | {high["derived_columns"]["Reading_mean_eye_selfpaced"]:.2f} | {low["derived_columns"]["Reading_mean_eye_selfpaced"]:.2f} | {payload["derived_columns"]["Reading_mean_eye_selfpaced"]:.2f} |')
    lines.append(f'| Full EWoK | {ewok["scores"]["high_entity_state"]:.2f} | {ewok["scores"]["matched_low"]:.2f} | {ewok["scores"]["uniform"]:.2f} |')
    lines += ['', 'High entity/state selection did not improve Entity or EWoK over matched_low; any positive signal is mainly GlobalPIQA nonparallel/mean and must not be interpreted as recoverable relation structure without stronger controls.']
    NOTE.write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':comp['status'],'uniform_json':str(OUT_JSON),'three_arm':str(THREE_JSON),'uniform_scores':scores,'uniform_derived':payload['derived_columns'],'deltas':comp['deltas']},indent=2))
if __name__=='__main__': main()
