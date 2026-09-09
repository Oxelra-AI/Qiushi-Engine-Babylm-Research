#!/usr/bin/env python3
"""Reusable SuperGLUE evaluator for 2x2 structured-experience arms.

Runs official BabyLM full_eval/glue_filtered fine-tuning tasks and computes mean
validation accuracy directly from predictions.json, with resume support.
Use on selected arms/seeds after the matching zero-shot screen.
"""
from __future__ import annotations
import argparse, json, os, pathlib, subprocess, sys, time

ROOT=pathlib.Path('experiments/archive/initial_model_studies')
STRICT=ROOT/'repos/babylm-eval/strict'
DATA=STRICT/'evaluation_data/full_eval/glue_filtered'
TASKS=[
    {"task":"boolq", "num_labels":2, "batch_size":16, "metric_for_valid":"accuracy", "metrics":["accuracy","f1","mcc"], "epochs":10},
    {"task":"multirc", "num_labels":2, "batch_size":16, "metric_for_valid":"accuracy", "metrics":["accuracy","f1","mcc"], "epochs":10},
    {"task":"rte", "num_labels":2, "batch_size":32, "metric_for_valid":"accuracy", "metrics":["accuracy","f1","mcc"], "epochs":10},
    {"task":"wsc", "num_labels":2, "batch_size":32, "metric_for_valid":"accuracy", "metrics":["accuracy","f1","mcc"], "epochs":30},
    {"task":"mrpc", "num_labels":2, "batch_size":32, "metric_for_valid":"f1", "metrics":["accuracy","f1","mcc"], "epochs":10},
    {"task":"qqp", "num_labels":2, "batch_size":32, "metric_for_valid":"f1", "metrics":["accuracy","f1","mcc"], "epochs":10},
    {"task":"mnli", "num_labels":3, "batch_size":32, "metric_for_valid":"accuracy", "metrics":["accuracy"], "epochs":10},
]

def arm_paths(seed:int, run_root:pathlib.Path)->dict[str,pathlib.Path]:
    return {
        'A_official_wwm': run_root/f'armA_official_lengthmatched_wwm_seed{seed}/hf_model/chck_9999969w',
        'B_structured_wwm': run_root/f'armB_structured_symmetric_wwm_seed{seed}/hf_model/chck_9999969w',
        'C_official_amlm': run_root/f'armC_official_lengthmatched_amlm_seed{seed}/hf_model/chck_9999969w',
        'D_structured_amlm': run_root/f'armD_structured_symmetric_amlm_seed{seed}/hf_model/chck_9999969w',
    }

def setup_env():
    env=os.environ.copy(); hf=ROOT/'training/hf_home'
    env['HF_HOME']=str(hf.resolve()); env['HF_HUB_CACHE']=str((hf/'hub').resolve())
    env['TRANSFORMERS_CACHE']=str((hf/'transformers').resolve()); env['HF_MODULES_CACHE']=str((ROOT/'training/hf_modules_cache').resolve())
    env.setdefault('TOKENIZERS_PARALLELISM','false')
    for k in ['HF_HOME','HF_HUB_CACHE','TRANSFORMERS_CACHE','HF_MODULES_CACHE']: pathlib.Path(env[k]).mkdir(parents=True,exist_ok=True)
    return env

def labels_for(task:str)->list[int]:
    labels=[]
    for line in (DATA/f'{task}.valid.jsonl').read_text(encoding='utf-8').splitlines():
        if line.strip(): labels.append(int(json.loads(line)['label']))
    return labels

def score_predictions(task:str, pp:pathlib.Path)->dict:
    data=json.loads(pp.read_text(encoding='utf-8')); preds=data[task]['predictions']; labels=labels_for(task)
    if len(preds)!=len(labels): raise RuntimeError(f'{task}: predictions={len(preds)} labels={len(labels)}')
    correct=sum(1 for r,y in zip(preds,labels) if int(r['pred'])==int(y))
    return {'task':task,'predictions':str(pp),'num_examples':len(labels),'correct':correct,'accuracy':correct/len(labels)*100.0}

def pred_path(results_dir:pathlib.Path, task:str)->pathlib.Path:
    return results_dir/'chck_9999969w'/'main'/'finetune'/task/'predictions.json'

def run_task(model:pathlib.Path, out_base:pathlib.Path, spec:dict, env:dict, logf, gpu:int)->dict:
    out_base = out_base.resolve()
    results_dir=(out_base/'results').resolve(); models_dir=(out_base/'models').resolve()
    results_dir.mkdir(parents=True,exist_ok=True); models_dir.mkdir(parents=True,exist_ok=True)
    t=spec['task']; pp=pred_path(results_dir,t)
    root_abs = ROOT.resolve()
    mistaken_results_dir = STRICT / str(results_dir.relative_to(root_abs)) if str(results_dir).startswith(str(root_abs)) else None
    fallback_hits=[]
    if mistaken_results_dir is not None and mistaken_results_dir.exists():
        fallback_hits=sorted(mistaken_results_dir.rglob(f'finetune/{t}/predictions.json'),key=lambda x:str(x))
    if pp.exists():
        row=score_predictions(t,pp); row['resumed_existing']=True; return row
    if fallback_hits:
        row=score_predictions(t,fallback_hits[-1]); row['resumed_existing']=True; row['fallback_from_relative_cwd_bug']=True; return row
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
    print(p.stdout[-2000:],flush=True); logf.write(p.stdout+f'\n[rc={p.returncode}]\n'); logf.flush()
    if p.returncode: raise RuntimeError(f'{t} failed {p.returncode}\n{p.stdout[-4000:]}')
    if not pp.exists():
        hits=sorted(results_dir.rglob(f'finetune/{t}/predictions.json'),key=lambda x:str(x))
        if not hits: raise FileNotFoundError(f'missing predictions for {t} under {results_dir}')
        pp=hits[-1]
    row=score_predictions(t,pp); row['resumed_existing']=False; return row

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seed_num',type=int,required=True); ap.add_argument('--run_root',required=True)
    ap.add_argument('--out_json',required=True); ap.add_argument('--log',required=True)
    ap.add_argument('--arms',nargs='+',required=True); ap.add_argument('--gpu',type=int,default=0)
    args=ap.parse_args(); t0=time.time()
    run_root=pathlib.Path(args.run_root); out_json=pathlib.Path(args.out_json); log=pathlib.Path(args.log)
    specs=arm_paths(args.seed_num,run_root); env=setup_env(); out_json.parent.mkdir(parents=True,exist_ok=True); log.parent.mkdir(parents=True,exist_ok=True)
    payload=json.loads(out_json.read_text()) if out_json.exists() else {'status':'SUPERGLUE_2X2','models':{}}
    with log.open('a',encoding='utf-8') as logf:
        for arm in args.arms:
            model=specs[arm]
            if not model.exists(): raise FileNotFoundError(f'missing {arm}: {model}')
            rec=payload['models'].setdefault(arm, {'model_path':str(model),'tasks':[]})
            done={r['task']:r for r in rec.get('tasks',[]) if 'accuracy' in r}
            rows=[]
            out_base=model.parent.parent/f'eval_step255_seed{args.seed_num}_{arm}_superglue'
            for spec in TASKS:
                if spec['task'] in done and pathlib.Path(done[spec['task']]['predictions']).exists():
                    row=score_predictions(spec['task'],pathlib.Path(done[spec['task']]['predictions'])); row['resumed_existing']=True
                else:
                    row=run_task(model,out_base,spec,env,logf,args.gpu)
                rows.append(row); rec['tasks']=rows; rec['superglue']=sum(r['accuracy'] for r in rows)/len(rows); rec['num_completed_tasks']=len(rows)
                out_json.write_text(json.dumps(payload,indent=2)+'\n')
    payload['elapsed_sec']=round(time.time()-t0,1); out_json.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps({'out':str(out_json),'models':{a:payload['models'].get(a,{}).get('superglue') for a in args.arms}},indent=2))
if __name__=='__main__': main()
