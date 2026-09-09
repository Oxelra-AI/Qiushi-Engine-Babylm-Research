#!/usr/bin/env python3
from __future__ import annotations
import json, os, pathlib, re, subprocess, sys, time

ROOT=pathlib.Path('experiments/archive/initial_model_studies')
STRICT=ROOT/'repos/babylm-eval/strict'
EWOK_OUT=STRICT/'evaluation_data/full_eval/ewok_filtered_word_tokenize'
RUN=ROOT/'training/runs/wess_aux_base_transfer_screen/smoke_384x4_anneal'
OUT_JSON=ROOT/'data/exported_ewok_scores.json'
LOG=(ROOT.parents[2] / 'research/notes/initial_model_studies/exported_ewok_eval.log')
MODELS={
    'plain_mixed': RUN/'plain_mixed/hf_model',
    'wess_aux_exported_base': RUN/'wess_aux/hf_model',
}

def setup_env():
    env=os.environ.copy()
    env['NLTK_DATA']=str((ROOT/'data/nltk_data').resolve())
    hf=ROOT/'training/hf_home'
    env['HF_HOME']=str(hf.resolve()); env['HF_HUB_CACHE']=str((hf/'hub').resolve())
    env['TRANSFORMERS_CACHE']=str((hf/'transformers').resolve())
    env['HF_MODULES_CACHE']=str((ROOT/'training/hf_modules_cache').resolve())
    env.setdefault('TOKENIZERS_PARALLELISM','false')
    for k in ['HF_HOME','HF_HUB_CACHE','TRANSFORMERS_CACHE','HF_MODULES_CACHE']:
        pathlib.Path(env[k]).mkdir(parents=True,exist_ok=True)
    return env

def parse_avg(report:pathlib.Path)->float:
    txt=report.read_text(encoding='utf-8',errors='replace')
    m=re.search(r'AVERAGE ACCURACY\s*\n([0-9.\-]+)',txt)
    if not m:
        raise RuntimeError(f'Could not parse average from {report}\n{txt[:1200]}')
    return float(m.group(1))

def main():
    t0=time.time(); LOG.parent.mkdir(parents=True,exist_ok=True)
    if not EWOK_OUT.exists() or not any(EWOK_OUT.glob('*.jsonl')):
        raise RuntimeError(f'Missing filtered EWoK data {EWOK_OUT}')
    env=setup_env()
    payload={'status':'EXPORTED_BASE_FULL_EWOK','run':str(RUN),'backend':'mlm','data_path':str(EWOK_OUT),'models':{}}
    with LOG.open('w',encoding='utf-8') as logf:
        def log(msg:str):
            print(msg,flush=True); logf.write(msg+'\n'); logf.flush()
        for name,model in MODELS.items():
            if not model.exists(): raise RuntimeError(f'Missing model {model}')
            eval_out=RUN/f'eval_ewok_{name}'
            eval_out.mkdir(parents=True,exist_ok=True)
            cmd=[sys.executable,'-m','evaluation_pipeline.sentence_zero_shot.run',
                 '--model_path_or_name',str(model.resolve()),'--backend','mlm','--task','ewok',
                 '--data_path',str(EWOK_OUT.resolve()),'--save_predictions','--batch_size','64',
                 '--output_dir',str(eval_out.resolve())]
            log('$ '+' '.join(cmd))
            p=subprocess.run(cmd,cwd=str(STRICT),env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            log(p.stdout[-8000:])
            if p.returncode!=0:
                raise RuntimeError(f'EWoK eval failed for {name}: {p.returncode}\n{p.stdout[-10000:]}')
            hits=sorted(eval_out.rglob('best_temperature_report.txt'))
            if not hits: raise FileNotFoundError(f'No EWoK report under {eval_out}')
            report=hits[-1]
            score=parse_avg(report)
            payload['models'][name]={'model_path':str(model),'ewok_full_score':score,'report':str(report),'predictions':str(report.parent/'predictions.json'),'eval_output_dir':str(eval_out)}
            OUT_JSON.write_text(json.dumps(payload,indent=2)+'\n')
    p=payload['models']['plain_mixed']['ewok_full_score']; w=payload['models']['wess_aux_exported_base']['ewok_full_score']
    payload['delta_wess_exported_minus_plain']=w-p
    payload['elapsed_sec']=round(time.time()-t0,1)
    OUT_JSON.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps({'out':str(OUT_JSON),'plain_ewok':p,'wess_exported_ewok':w,'delta':w-p,'elapsed_sec':payload['elapsed_sec']},indent=2))

if __name__=='__main__': main()
