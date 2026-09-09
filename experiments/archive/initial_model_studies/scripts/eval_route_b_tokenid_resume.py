#!/usr/bin/env python3
"""research: resume only missing Route B token-ID evaluations and merge three-arm table."""
from __future__ import annotations
import json, os, pathlib, re, subprocess, sys, time

ROOT=pathlib.Path('experiments/archive/initial_model_studies')
STRICT=ROOT/'repos/babylm-eval/strict'
FULL=STRICT/'evaluation_data/full_eval'
MODEL=ROOT/'training/runs/route_b_token_id/hf_model/chck_10M'
OUTDIR=ROOT/'training/runs/route_b_token_id/eval_available_token_id'
OUT_JSON=ROOT/'data/route_b_threearm_scores.json'
LOG=(ROOT.parents[2] / 'research/notes/initial_model_studies/route_b_tokenid_resume_eval.log')

TASKS=[('blimp','blimp',FULL/'blimp_filtered','blimp_filtered'),
       ('supplement','blimp',FULL/'supplement_filtered','supplement_filtered'),
       ('entity_tracking','entity_tracking',FULL/'entity_tracking','entity_tracking'),
       ('comps','comps',FULL/'comps','comps'),
       ('global_piqa_parallel','global_piqa_parallel',FULL/'global_piqa_parallel','global_piqa_parallel'),
       ('global_piqa_nonparallel','global_piqa_nonparallel',FULL/'global_piqa_nonparallel','global_piqa_nonparallel'),
       ('ewok','ewok',FULL/'ewok_filtered_word_tokenize','ewok_filtered_word_tokenize')]

def env_setup():
    env=os.environ.copy(); env['NLTK_DATA']=str((ROOT/'data/nltk_data').resolve())
    hf=ROOT/'training/hf_home'
    env['HF_HOME']=str(hf.resolve()); env['HF_HUB_CACHE']=str((hf/'hub').resolve())
    env['TRANSFORMERS_CACHE']=str((hf/'transformers').resolve()); env['HF_MODULES_CACHE']=str((ROOT/'training/hf_modules_cache').resolve())
    env.setdefault('TOKENIZERS_PARALLELISM','false')
    return env

def run(cmd,env,logf):
    line='$ '+' '.join(map(str,cmd)); print(line,flush=True); logf.write('\n'+line+'\n'); logf.flush()
    p=subprocess.run([str(x) for x in cmd],cwd=str(STRICT),env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    print(p.stdout[-1500:],flush=True); logf.write(p.stdout+f'\n[rc={p.returncode}]\n'); logf.flush()
    if p.returncode: raise RuntimeError(f'failed {p.returncode}: {line}\n{p.stdout[-5000:]}')

def read_avg(report:pathlib.Path)->float:
    txt=report.read_text(encoding='utf-8',errors='replace')
    m=re.search(r'AVERAGE ACCURACY\s*\n([0-9.\-]+)',txt)
    if m: return float(m.group(1))
    vals=re.findall(r'(?:acc_norm|accuracy|acc)\s*[:=]\s*([0-9.]+)',txt,flags=re.I)
    if vals:
        v=float(vals[-1]); return v*(100 if v<=1 else 1)
    raise RuntimeError(f'parse failed {report}\n{txt[:1000]}')

def find_report(col,task,dataset):
    pat=f'zero_shot/mlm/{task}/{dataset}/best_temperature_report.txt'
    hits=sorted(OUTDIR.rglob(pat),key=lambda p:str(p))
    if not hits and col=='ewok': hits=sorted(OUTDIR.rglob('zero_shot/mlm/ewok/*/best_temperature_report.txt'),key=lambda p:str(p))
    return hits[-1] if hits else None

def read_reading(report:pathlib.Path)->dict[str,float]:
    txt=report.read_text(encoding='utf-8',errors='replace'); out={}
    for label,key in [('EYE TRACKING SCORE','reading_eye_tracking'),('SELF-PACED READING SCORE','reading_self_paced')]:
        m=re.search(re.escape(label)+r':\s*([0-9.\-]+)',txt)
        if not m: raise RuntimeError(f'parse {label} failed {report}')
        out[key]=float(m.group(1))
    return out

def main():
    t0=time.time(); assert MODEL.exists(), MODEL
    OUTDIR.mkdir(parents=True,exist_ok=True); LOG.parent.mkdir(parents=True,exist_ok=True)
    payload=json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {'status':'ROUTE_B_THREEARM','models':{}}
    scores={}; reports={}
    env=env_setup()
    with LOG.open('w',encoding='utf-8') as logf:
        for col,task,data,dataset in TASKS:
            rep=find_report(col,task,dataset)
            if rep is None:
                run([sys.executable,'-m','evaluation_pipeline.sentence_zero_shot.run','--model_path_or_name',str(MODEL.resolve()),'--backend','mlm','--task',task,'--data_path',str(data.resolve()),'--save_predictions','--revision_name','token_id','--batch_size','128' if col!='ewok' else '64','--output_dir',str(OUTDIR.resolve())],env,logf)
                rep=find_report(col,task,dataset)
            if rep is None: raise RuntimeError(f'missing report after run {col}')
            reports[col]=str(rep); scores[col]=read_avg(rep)
        rrep=next(iter(sorted(OUTDIR.rglob('zero_shot/mlm/reading/report.txt'),key=lambda p:str(p))),None)
        if rrep is None:
            reading=FULL/'reading/reading_data.csv'
            run([sys.executable,'-m','evaluation_pipeline.reading.run','--model_path_or_name',str(MODEL.resolve()),'--backend','mlm','--data_path',str(reading.resolve()),'--revision_name','token_id','--output_dir',str(OUTDIR.resolve())],env,logf)
            hits=sorted(OUTDIR.rglob('zero_shot/mlm/reading/report.txt'),key=lambda p:str(p))
            if hits: rrep=hits[-1]
        if rrep:
            reports['reading']=str(rrep); scores.update(read_reading(rrep))
    scores['GlobalPIQA_mean']=(scores['global_piqa_parallel']+scores['global_piqa_nonparallel'])/2
    scores['Reading_mean']=(scores.get('reading_eye_tracking',0)+scores.get('reading_self_paced',0))/2
    payload['models']['token_id']={'model_path':str(MODEL),'scores':scores,'reports':reports}
    if all(k in payload['models'] for k in ['baseline','surface','token_id']):
        b=payload['models']['baseline']['scores']; s=payload['models']['surface']['scores']; ti=payload['models']['token_id']['scores']
        keys=sorted(set(b)&set(s)&set(ti))
        payload['deltas']={
            'surface_minus_baseline':{k:s[k]-b[k] for k in keys},
            'token_id_minus_baseline':{k:ti[k]-b[k] for k in keys},
            'surface_minus_token_id':{k:s[k]-ti[k] for k in keys},
        }
    payload['status']='ROUTE_B_THREEARM_COMPLETE'
    payload['elapsed_sec_resume']=round(time.time()-t0,1)
    OUT_JSON.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps({'out':str(OUT_JSON),'token_id_scores':scores,'deltas':payload.get('deltas')},indent=2))

if __name__=='__main__': main()
