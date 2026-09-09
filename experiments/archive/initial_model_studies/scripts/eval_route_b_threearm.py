#!/usr/bin/env python3
"""research: official-compatible three-arm evaluation for Route B adapters.

Evaluates the fused exported chck_10M checkpoints for baseline, surface, and
token_id arms on the available columns plus full EWoK (word_tokenize path).
Decisive comparison: surface minus token_id and minus baseline on Entity/EWoK/COMPS
while preserving Supplement/Reading.
"""
from __future__ import annotations
import json, os, pathlib, re, subprocess, sys, time

ROOT=pathlib.Path('experiments/archive/initial_model_studies')
STRICT=ROOT/'repos/babylm-eval/strict'
FULL=STRICT/'evaluation_data/full_eval'
OUT_JSON=ROOT/'data/route_b_threearm_scores.json'
LOG=(ROOT.parents[2] / 'research/notes/initial_model_studies/route_b_eval.log')

MODELS={
 'baseline': ROOT/'training/runs/route_b_baseline/hf_model/chck_10M',
 'surface':  ROOT/'training/runs/route_b_surface/hf_model/chck_10M',
 'token_id': ROOT/'training/runs/route_b_token_id/hf_model/chck_10M',
}
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
    for k in ['HF_HOME','HF_HUB_CACHE','TRANSFORMERS_CACHE','HF_MODULES_CACHE']: pathlib.Path(env[k]).mkdir(parents=True,exist_ok=True)
    return env

def run(cmd,env,logf):
    line='$ '+' '.join(map(str,cmd)); print(line,flush=True); logf.write('\n'+line+'\n'); logf.flush()
    p=subprocess.run([str(x) for x in cmd],cwd=str(STRICT),env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    print(p.stdout[-1200:],flush=True); logf.write(p.stdout+f'\n[rc={p.returncode}]\n'); logf.flush()
    if p.returncode: raise RuntimeError(f'failed {p.returncode}: {line}\n{p.stdout[-4000:]}')

def read_avg(report):
    txt=pathlib.Path(report).read_text(encoding='utf-8',errors='replace')
    m=re.search(r'AVERAGE ACCURACY\s*\n([0-9.\-]+)',txt)
    if m: return float(m.group(1))
    vals=re.findall(r'(?:acc_norm|accuracy|acc)\s*[:=]\s*([0-9.]+)',txt,flags=re.I)
    if vals:
        v=float(vals[-1]); return v*(100 if v<=1 else 1)
    raise RuntimeError(f'parse failed {report}\n{txt[:800]}')

def read_reading(report):
    txt=pathlib.Path(report).read_text(encoding='utf-8',errors='replace'); out={}
    for label,key in [('EYE TRACKING SCORE','reading_eye_tracking'),('SELF-PACED READING SCORE','reading_self_paced')]:
        m=re.search(re.escape(label)+r':\s*([0-9.\-]+)',txt)
        if not m: raise RuntimeError(f'parse {label} failed {report}')
        out[key]=float(m.group(1))
    return out

def main():
    t0=time.time(); LOG.parent.mkdir(parents=True,exist_ok=True)
    for n,m in MODELS.items():
        if not m.exists(): raise RuntimeError(f'missing checkpoint {n}: {m}')
    env=env_setup(); payload={'status':'ROUTE_B_THREEARM','models':{}}
    with LOG.open('w',encoding='utf-8') as logf:
        for name,model in MODELS.items():
            outdir=model.parent.parent/f'eval_available_{name}'; outdir.mkdir(parents=True,exist_ok=True)
            scores={}; reports={}
            for col,task,data,dataset in TASKS:
                run([sys.executable,'-m','evaluation_pipeline.sentence_zero_shot.run','--model_path_or_name',str(model.resolve()),'--backend','mlm','--task',task,'--data_path',str(data.resolve()),'--save_predictions','--revision_name',f'step201_{name}','--batch_size','128' if col!='ewok' else '64','--output_dir',str(outdir.resolve())],env,logf)
                cand=sorted(outdir.rglob(f'zero_shot/mlm/{task}/{dataset}/best_temperature_report.txt'),key=lambda p:str(p))
                if not cand and col=='ewok': cand=sorted(outdir.rglob('zero_shot/mlm/ewok/*/best_temperature_report.txt'),key=lambda p:str(p))
                if not cand: raise RuntimeError(f'no report {name} {col}')
                reports[col]=str(cand[-1]); scores[col]=read_avg(cand[-1])
            reading=FULL/'reading/reading_data.csv'
            run([sys.executable,'-m','evaluation_pipeline.reading.run','--model_path_or_name',str(model.resolve()),'--backend','mlm','--data_path',str(reading.resolve()),'--revision_name',f'step201_{name}','--output_dir',str(outdir.resolve())],env,logf)
            cand=sorted(outdir.rglob('zero_shot/mlm/reading/report.txt'),key=lambda p:str(p))
            if cand: reports['reading']=str(cand[-1]); scores.update(read_reading(cand[-1]))
            scores['GlobalPIQA_mean']=(scores['global_piqa_parallel']+scores['global_piqa_nonparallel'])/2
            scores['Reading_mean']=(scores.get('reading_eye_tracking',0)+scores.get('reading_self_paced',0))/2
            payload['models'][name]={'model_path':str(model),'scores':scores,'reports':reports}
            OUT_JSON.write_text(json.dumps(payload,indent=2)+'\n')
    b=payload['models']['baseline']['scores']; s=payload['models']['surface']['scores']; t=payload['models']['token_id']['scores']
    keys=sorted(set(b)&set(s)&set(t))
    payload['deltas']={
      'surface_minus_baseline':{k:s[k]-b[k] for k in keys},
      'token_id_minus_baseline':{k:t[k]-b[k] for k in keys},
      'surface_minus_token_id':{k:s[k]-t[k] for k in keys},
    }
    payload['elapsed_sec']=round(time.time()-t0,1)
    OUT_JSON.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps({'out':str(OUT_JSON),'deltas':payload['deltas']},indent=2))

if __name__=='__main__': main()
