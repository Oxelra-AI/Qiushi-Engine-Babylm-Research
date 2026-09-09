#!/usr/bin/env python3
"""Reusable 2x2 zero-shot/reading evaluator for structured-experience screens.

Runs BLiMP, Supplement, Entity Tracking, COMPS, GlobalPIQA parallel/nonparallel,
EWoK, and Reading for four arms in a given seed directory and computes factorial
contrasts. This generalizes the research seed42 evaluator.
"""
from __future__ import annotations
import argparse, json, os, pathlib, re, subprocess, sys, time

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
STRICT = ROOT/'repos/babylm-eval/strict'
FULL = STRICT/'evaluation_data/full_eval'
TASKS=[('blimp','blimp',FULL/'blimp_filtered','blimp_filtered'),
       ('supplement','blimp',FULL/'supplement_filtered','supplement_filtered'),
       ('entity_tracking','entity_tracking',FULL/'entity_tracking','entity_tracking'),
       ('comps','comps',FULL/'comps','comps'),
       ('global_piqa_parallel','global_piqa_parallel',FULL/'global_piqa_parallel','global_piqa_parallel'),
       ('global_piqa_nonparallel','global_piqa_nonparallel',FULL/'global_piqa_nonparallel','global_piqa_nonparallel'),
       ('ewok','ewok',FULL/'ewok_filtered_word_tokenize','ewok_filtered_word_tokenize')]

def model_specs(seed:int, run_root:pathlib.Path)->dict[str,pathlib.Path]:
    return {
        'A_official_wwm': run_root/f'armA_official_lengthmatched_wwm_seed{seed}/hf_model/chck_9999969w',
        'B_structured_wwm': run_root/f'armB_structured_symmetric_wwm_seed{seed}/hf_model/chck_9999969w',
        'C_official_amlm': run_root/f'armC_official_lengthmatched_amlm_seed{seed}/hf_model/chck_9999969w',
        'D_structured_amlm': run_root/f'armD_structured_symmetric_amlm_seed{seed}/hf_model/chck_9999969w',
    }

def env_setup():
    env=os.environ.copy(); env['NLTK_DATA']=str((ROOT/'data/nltk_data').resolve())
    hf=ROOT/'training/hf_home'
    env['HF_HOME']=str(hf.resolve()); env['HF_HUB_CACHE']=str((hf/'hub').resolve())
    env['TRANSFORMERS_CACHE']=str((hf/'transformers').resolve()); env['HF_MODULES_CACHE']=str((ROOT/'training/hf_modules_cache').resolve())
    env.setdefault('TOKENIZERS_PARALLELISM','false')
    for k in ['HF_HOME','HF_HUB_CACHE','TRANSFORMERS_CACHE','HF_MODULES_CACHE']: pathlib.Path(env[k]).mkdir(parents=True,exist_ok=True)
    return env

def run(cmd, env, logf, gpu:int):
    e=dict(env); e['CUDA_VISIBLE_DEVICES']=str(gpu)
    line=f'[gpu{gpu}] $ '+' '.join(map(str,cmd)); print(line, flush=True); logf.write('\n'+line+'\n'); logf.flush()
    p=subprocess.run([str(x) for x in cmd], cwd=str(STRICT), env=e, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.stdout[-1200:], flush=True); logf.write(p.stdout+f'\n[rc={p.returncode}]\n'); logf.flush()
    if p.returncode: raise RuntimeError(f'failed {p.returncode}: {line}\n{p.stdout[-4000:]}')

def read_avg(report):
    txt=pathlib.Path(report).read_text(encoding='utf-8',errors='replace')
    m=re.search(r'AVERAGE ACCURACY\s*\n([0-9.\-]+)', txt)
    if m: return float(m.group(1))
    vals=re.findall(r'(?:acc_norm|accuracy|acc)\s*[:=]\s*([0-9.]+)', txt, flags=re.I)
    if vals:
        v=float(vals[-1]); return v*(100 if v<=1 else 1)
    raise RuntimeError(f'parse failed {report}\n{txt[:600]}')

def read_reading(report):
    txt=pathlib.Path(report).read_text(encoding='utf-8',errors='replace'); out={}
    for label,key in [('EYE TRACKING SCORE','reading_eye_tracking'),('SELF-PACED READING SCORE','reading_self_paced')]:
        m=re.search(re.escape(label)+r':\s*([0-9.\-]+)',txt)
        if not m: raise RuntimeError(f'parse {label} failed {report}')
        out[key]=float(m.group(1))
    return out

def eval_one(name, model, env, logf, gpu:int, seed:int):
    outdir=model.parent.parent/f'eval_step255_seed{seed}_{name}'; outdir.mkdir(parents=True,exist_ok=True)
    scores={}; reports={}
    for col,task,data,dataset in TASKS:
        run([sys.executable,'-m','evaluation_pipeline.sentence_zero_shot.run','--model_path_or_name',str(model.resolve()),'--backend','mlm','--task',task,'--data_path',str(data.resolve()),'--save_predictions','--revision_name',f'seed{seed}_{name}','--batch_size','128' if col!='ewok' else '64','--output_dir',str(outdir.resolve())], env, logf, gpu)
        cand=sorted(outdir.rglob(f'zero_shot/mlm/{task}/{dataset}/best_temperature_report.txt'), key=lambda p:str(p))
        if not cand and col=='ewok': cand=sorted(outdir.rglob('zero_shot/mlm/ewok/*/best_temperature_report.txt'), key=lambda p:str(p))
        if not cand: raise RuntimeError(f'no report {name} {col}')
        reports[col]=str(cand[-1]); scores[col]=read_avg(cand[-1])
    reading=FULL/'reading/reading_data.csv'
    run([sys.executable,'-m','evaluation_pipeline.reading.run','--model_path_or_name',str(model.resolve()),'--backend','mlm','--data_path',str(reading.resolve()),'--revision_name',f'seed{seed}_{name}','--output_dir',str(outdir.resolve())], env, logf, gpu)
    cand=sorted(outdir.rglob('zero_shot/mlm/reading/report.txt'), key=lambda p:str(p))
    if cand: reports['reading']=str(cand[-1]); scores.update(read_reading(cand[-1]))
    scores['GlobalPIQA_mean']=(scores['global_piqa_parallel']+scores['global_piqa_nonparallel'])/2
    scores['Reading_mean']=(scores.get('reading_eye_tracking',0)+scores.get('reading_self_paced',0))/2
    scores['NLP_mean_no_superglue_aoa']=sum(scores[k] for k in ['blimp','supplement','ewok','entity_tracking','comps','GlobalPIQA_mean','Reading_mean'])/7
    return {'model_path':str(model),'scores':scores,'reports':reports}

def factorial(models:dict)->dict:
    A,B,C,D=(models['A_official_wwm']['scores'],models['B_structured_wwm']['scores'],models['C_official_amlm']['scores'],models['D_structured_amlm']['scores'])
    keys=sorted(set(A)&set(B)&set(C)&set(D))
    return {
        'data_effect_wwm_BminusA':{k:B[k]-A[k] for k in keys},
        'data_effect_amlm_DminusC':{k:D[k]-C[k] for k in keys},
        'amlm_effect_official_CminusA':{k:C[k]-A[k] for k in keys},
        'amlm_effect_structured_DminusB':{k:D[k]-B[k] for k in keys},
        'interaction_DC_minus_BA':{k:(D[k]-C[k])-(B[k]-A[k]) for k in keys},
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seed_num', type=int, required=True)
    ap.add_argument('--run_root', required=True)
    ap.add_argument('--out_json', required=True)
    ap.add_argument('--log', required=True)
    ap.add_argument('--only', default='')
    ap.add_argument('--gpu', type=int, default=0)
    args=ap.parse_args()
    run_root=pathlib.Path(args.run_root); out_json=pathlib.Path(args.out_json); log=pathlib.Path(args.log)
    specs=model_specs(args.seed_num, run_root)
    for n,m in specs.items():
        if (not args.only or n in args.only.split(',')) and not m.exists(): raise RuntimeError(f'missing {n}: {m}')
    out_json.parent.mkdir(parents=True,exist_ok=True); log.parent.mkdir(parents=True,exist_ok=True)
    payload=json.loads(out_json.read_text()) if out_json.exists() else {'status':f'reference_2X2_SEED{args.seed_num}', 'models':{}}
    t0=time.time(); env=env_setup(); todo=[k for k in specs if not args.only or k in args.only.split(',')]
    with log.open('a', encoding='utf-8') as logf:
        for name in todo:
            payload['models'][name]=eval_one(name, specs[name], env, logf, args.gpu, args.seed_num)
            out_json.write_text(json.dumps(payload, indent=2)+'\n')
    if all(k in payload['models'] for k in specs):
        payload['factorial']=factorial(payload['models'])
        payload['elapsed_sec']=round(time.time()-t0,1)
        out_json.write_text(json.dumps(payload, indent=2)+'\n')
        print(json.dumps({'out':str(out_json),'factorial':payload['factorial']}, indent=2))
    else:
        print(json.dumps({'out':str(out_json),'done_models':list(payload['models'])}, indent=2))
if __name__=='__main__': main()
