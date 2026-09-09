#!/usr/bin/env python3
"""Single-checkpoint SuperGLUE evaluator (official glue_filtered finetune).

Purpose: decide whether an earlier checkpoint (e.g. chck_80M) is a better single
training endpoint than chck_100M under the official scoring rule, which selects
ONE checkpoint for all 8 final-ability columns. We already have 7-column zero-shot
scores per checkpoint; this fills in the (Super)GLUE column so a true Overall
comparison at 80M vs 100M is possible with zero new pretraining.

Reuses the official evaluation_pipeline.finetune.run with absolute paths.
"""
from __future__ import annotations
import argparse, json, os, pathlib, subprocess, sys

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
STRICT = (ROOT/'repos/babylm-eval/strict').resolve()
DATA = STRICT/'evaluation_data/full_eval/glue_filtered'
TASKS = [
    {"task":"boolq","num_labels":2,"batch_size":16,"metric_for_valid":"accuracy","metrics":["accuracy","f1","mcc"],"epochs":10},
    {"task":"multirc","num_labels":2,"batch_size":16,"metric_for_valid":"accuracy","metrics":["accuracy","f1","mcc"],"epochs":10},
    {"task":"rte","num_labels":2,"batch_size":32,"metric_for_valid":"accuracy","metrics":["accuracy","f1","mcc"],"epochs":10},
    {"task":"wsc","num_labels":2,"batch_size":32,"metric_for_valid":"accuracy","metrics":["accuracy","f1","mcc"],"epochs":30},
    {"task":"mrpc","num_labels":2,"batch_size":32,"metric_for_valid":"f1","metrics":["accuracy","f1","mcc"],"epochs":10},
    {"task":"qqp","num_labels":2,"batch_size":32,"metric_for_valid":"f1","metrics":["accuracy","f1","mcc"],"epochs":10},
    {"task":"mnli","num_labels":3,"batch_size":32,"metric_for_valid":"accuracy","metrics":["accuracy"],"epochs":10},
]

def setup_env():
    env=os.environ.copy(); env['NLTK_DATA']=str((ROOT/'data/nltk_data').resolve())
    hf=ROOT/'training/hf_home'
    env['HF_HOME']=str(hf.resolve()); env['HF_HUB_CACHE']=str((hf/'hub').resolve())
    env['TRANSFORMERS_CACHE']=str((hf/'transformers').resolve())
    env['HF_MODULES_CACHE']=str((ROOT/'training/hf_modules_cache').resolve())
    env.setdefault('TOKENIZERS_PARALLELISM','false')
    for k in ['HF_HOME','HF_HUB_CACHE','TRANSFORMERS_CACHE','HF_MODULES_CACHE']:
        pathlib.Path(env[k]).mkdir(parents=True, exist_ok=True)
    return env

def labels_for(task):
    labels=[]
    for line in (DATA/f'{task}.valid.jsonl').read_text(encoding='utf-8').splitlines():
        if line.strip(): labels.append(int(json.loads(line)['label']))
    return labels

def score_predictions(task, pp):
    data=json.loads(pp.read_text(encoding='utf-8')); preds=[int(r['pred']) for r in data[task]['predictions']]
    labels=labels_for(task)
    if len(preds)!=len(labels): raise RuntimeError(f'{task}: preds={len(preds)} labels={len(labels)}')
    correct=sum(int(a==b) for a,b in zip(preds,labels))
    return {'task':task,'predictions':str(pp),'num_examples':len(labels),'correct':correct,'accuracy':correct/len(labels)*100.0}

def run_task(model, results_dir, spec, env, logf, gpu):
    results_dir=results_dir.resolve(); results_dir.mkdir(parents=True,exist_ok=True)
    models_dir=(results_dir/'models').resolve(); models_dir.mkdir(parents=True,exist_ok=True)
    t=spec['task']
    e=dict(env); e['CUDA_VISIBLE_DEVICES']=str(gpu)
    cmd=[sys.executable,'-m','evaluation_pipeline.finetune.run',
         '--model_name_or_path',str(model.resolve()),
         '--train_data',str((DATA/f'{t}.train.jsonl').resolve()),
         '--valid_data',str((DATA/f'{t}.valid.jsonl').resolve()),
         '--predict_data',str((DATA/f'{t}.valid.jsonl').resolve()),
         '--task',t,'--num_labels',str(spec['num_labels']),'--batch_size',str(spec['batch_size']),
         '--learning_rate','3e-5','--num_epochs',str(spec['epochs']),'--sequence_length','512',
         '--results_dir',str(results_dir),'--save','--save_dir',str(models_dir),'--metrics',*spec['metrics'],
         '--metric_for_valid',spec['metric_for_valid'],'--seed','42','--verbose','--padding_side','left','--take_final']
    line=f'[gpu{gpu}] $ '+' '.join(cmd); print(line,flush=True); logf.write('\n'+line+'\n'); logf.flush()
    p=subprocess.run(cmd,cwd=str(STRICT),env=e,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    print(p.stdout[-1500:],flush=True); logf.write(p.stdout+f'\n[rc={p.returncode}]\n'); logf.flush()
    if p.returncode: raise RuntimeError(f'{t} failed rc={p.returncode}')
    hits=sorted(results_dir.rglob(f'finetune/{t}/predictions.json'),key=lambda x:str(x))
    if not hits: raise FileNotFoundError(f'no predictions for {t} under {results_dir}')
    return score_predictions(t,hits[-1])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model_path',required=True)
    ap.add_argument('--run_name',required=True)
    ap.add_argument('--out_json',required=True)
    ap.add_argument('--results_dir',required=True)
    ap.add_argument('--log',required=True)
    ap.add_argument('--gpu',type=int,default=0)
    args=ap.parse_args()
    model=pathlib.Path(args.model_path).resolve()
    if not model.exists(): raise FileNotFoundError(model)
    log_path=pathlib.Path(args.log); log_path.parent.mkdir(parents=True,exist_ok=True)
    env=setup_env(); rows=[]
    with log_path.open('a',encoding='utf-8') as logf:
        for spec in TASKS:
            rows.append(run_task(model,pathlib.Path(args.results_dir)/spec['task'],spec,env,logf,args.gpu))
    superglue_mean=sum(r['accuracy'] for r in rows)/len(rows)
    payload={'run_name':args.run_name,'model_path':str(model),'superglue_mean':superglue_mean,'tasks':rows}
    out=pathlib.Path(args.out_json); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps({'status':'SUPERGLUE_DONE','out':str(out),'superglue_mean':superglue_mean,
                      'per_task':{r['task']:round(r['accuracy'],2) for r in rows}},indent=2))
if __name__=='__main__': main()
