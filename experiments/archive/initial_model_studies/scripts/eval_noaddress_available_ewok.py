#!/usr/bin/env python3
from __future__ import annotations
import json, os, pathlib, re, subprocess, sys, time
ROOT=pathlib.Path('experiments/archive/initial_model_studies')
STRICT=ROOT/'repos/babylm-eval/strict'
FULL=STRICT/'evaluation_data/full_eval'
RUN=ROOT/'training/runs/wess_aux_base_transfer_screen/smoke_384x4_anneal'
MODEL=RUN/'no_address/hf_model'
OUT_JSON=ROOT/'data/noaddress_available_ewok_scores.json'
LOG=(ROOT.parents[2] / 'research/notes/initial_model_studies/noaddress_available_ewok_eval.log')
TASKS=[('blimp','blimp',FULL/'blimp_filtered','blimp_filtered'),('supplement','blimp',FULL/'supplement_filtered','supplement_filtered'),('entity_tracking','entity_tracking',FULL/'entity_tracking','entity_tracking'),('comps','comps',FULL/'comps','comps'),('global_piqa_parallel','global_piqa_parallel',FULL/'global_piqa_parallel','global_piqa_parallel'),('global_piqa_nonparallel','global_piqa_nonparallel',FULL/'global_piqa_nonparallel','global_piqa_nonparallel'),('ewok','ewok',FULL/'ewok_filtered_word_tokenize','ewok_filtered_word_tokenize')]

def env_setup():
    env=os.environ.copy(); env['NLTK_DATA']=str((ROOT/'data/nltk_data').resolve())
    hf=ROOT/'training/hf_home'
    env['HF_HOME']=str(hf.resolve()); env['HF_HUB_CACHE']=str((hf/'hub').resolve()); env['TRANSFORMERS_CACHE']=str((hf/'transformers').resolve()); env['HF_MODULES_CACHE']=str((ROOT/'training/hf_modules_cache').resolve()); env.setdefault('TOKENIZERS_PARALLELISM','false')
    for k in ['HF_HOME','HF_HUB_CACHE','TRANSFORMERS_CACHE','HF_MODULES_CACHE']: pathlib.Path(env[k]).mkdir(parents=True,exist_ok=True)
    return env

def run(cmd,env,logf):
    line='$ '+' '.join(map(str,cmd)); print(line,flush=True); logf.write('\n'+line+'\n'); logf.flush()
    p=subprocess.run([str(x) for x in cmd],cwd=str(STRICT),env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    print(p.stdout[-1600:],flush=True); logf.write(p.stdout+f'\n[returncode={p.returncode}]\n'); logf.flush()
    if p.returncode: raise RuntimeError(f'failed {p.returncode}: {line}\n{p.stdout[-6000:]}')

def read_avg(report):
    txt=pathlib.Path(report).read_text(encoding='utf-8',errors='replace')
    m=re.search(r'AVERAGE ACCURACY\s*\n([0-9.\-]+)',txt)
    if m: return float(m.group(1))
    vals=re.findall(r'(?:acc_norm|accuracy|acc)\s*[:=]\s*([0-9.]+)',txt,flags=re.I)
    if vals:
        v=float(vals[-1]); return v*(100 if v<=1 else 1)
    raise RuntimeError(f'parse failed {report}\n{txt[:1000]}')

def read_reading(report):
    txt=pathlib.Path(report).read_text(encoding='utf-8',errors='replace'); out={}
    for label,key in [('EYE TRACKING SCORE','reading_eye_tracking'),('SELF-PACED READING SCORE','reading_self_paced')]:
        m=re.search(re.escape(label)+r':\s*([0-9.\-]+)',txt)
        if not m: raise RuntimeError(f'parse {label} failed')
        out[key]=float(m.group(1))
    return out

def main():
    t0=time.time(); LOG.parent.mkdir(parents=True,exist_ok=True)
    if not MODEL.exists(): raise RuntimeError(f'missing {MODEL}')
    env=env_setup(); outdir=RUN/'eval_available_no_address'; outdir.mkdir(parents=True,exist_ok=True)
    scores={}; reports={}
    with LOG.open('w',encoding='utf-8') as logf:
        for col,task,data,dataset in TASKS:
            run([sys.executable,'-m','evaluation_pipeline.sentence_zero_shot.run','--model_path_or_name',str(MODEL.resolve()),'--backend','mlm','--task',task,'--data_path',str(data.resolve()),'--save_predictions','--revision_name','no_address','--batch_size','128' if col!='ewok' else '64','--output_dir',str(outdir.resolve())],env,logf)
            cand=sorted(outdir.rglob(f'zero_shot/mlm/{task}/{dataset}/best_temperature_report.txt'),key=lambda p:str(p))
            if not cand and col=='ewok': cand=sorted(outdir.rglob('zero_shot/mlm/ewok/*/best_temperature_report.txt'),key=lambda p:str(p))
            if not cand: raise RuntimeError(f'no report {col}')
            reports[col]=str(cand[-1]); scores[col]=read_avg(cand[-1])
        reading=FULL/'reading/reading_data.csv'
        run([sys.executable,'-m','evaluation_pipeline.reading.run','--model_path_or_name',str(MODEL.resolve()),'--backend','mlm','--data_path',str(reading.resolve()),'--revision_name','no_address','--output_dir',str(outdir.resolve())],env,logf)
        cand=sorted(outdir.rglob('zero_shot/mlm/reading/report.txt'),key=lambda p:str(p))
        if cand:
            reports['reading']=str(cand[-1]); scores.update(read_reading(cand[-1]))
    scores['GlobalPIQA_mean']=(scores['global_piqa_parallel']+scores['global_piqa_nonparallel'])/2
    scores['Reading_mean']=(scores['reading_eye_tracking']+scores['reading_self_paced'])/2
    payload={'status':'NOADDRESS_AVAILABLE_EWOK','model_path':str(MODEL),'scores':scores,'reports':reports,'elapsed_sec':round(time.time()-t0,1)}
    prev=json.loads((ROOT/'data/exported_available_scores.json').read_text())
    ewok=json.loads((ROOT/'data/exported_ewok_scores.json').read_text())
    comparison={'plain_mixed':{**prev['models']['plain_mixed']['scores'],'ewok':ewok['models']['plain_mixed']['ewok_full_score']},'wess_aux_exported_base':{**prev['models']['wess_aux_exported_base']['scores'],'ewok':ewok['models']['wess_aux_exported_base']['ewok_full_score']},'no_address_exported_base':{**scores,'ewok':scores['ewok']}}
    payload['comparison_table']=comparison
    payload['deltas_vs_plain']={m:{k:v-comparison['plain_mixed'][k] for k,v in sc.items() if k in comparison['plain_mixed'] and isinstance(v,(int,float))} for m,sc in comparison.items() if m!='plain_mixed'}
    OUT_JSON.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps({'out':str(OUT_JSON),'scores':scores,'deltas_vs_plain':payload['deltas_vs_plain']},indent=2))
if __name__=='__main__': main()
