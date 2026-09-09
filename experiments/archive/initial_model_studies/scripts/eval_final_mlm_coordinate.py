#!/usr/bin/env python3
"""Evaluate one MLM checkpoint on the zero-shot/Reading coordinate used for BabyLM screens.

This is a generic version for research full-budget AMLM candidates. It evaluates:
BLiMP, Supplement, Entity Tracking, COMPS, EWoK, GlobalPIQA parallel/nonparallel,
and Reading eye/self-paced. It does not run SuperGLUE or AoA; use separate
runners for those if the zero-shot result merits the full 9-column coordinate.
"""
from __future__ import annotations
import argparse, json, os, pathlib, re, subprocess, sys

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
STRICT = ROOT / 'repos/babylm-eval/strict'
FULL = STRICT / 'evaluation_data/full_eval'
TASKS = [
    ('blimp','blimp',FULL/'blimp_filtered','blimp_filtered'),
    ('supplement','blimp',FULL/'supplement_filtered','supplement_filtered'),
    ('entity_tracking','entity_tracking',FULL/'entity_tracking','entity_tracking'),
    ('comps','comps',FULL/'comps','comps'),
    ('global_piqa_parallel','global_piqa_parallel',FULL/'global_piqa_parallel','global_piqa_parallel'),
    ('global_piqa_nonparallel','global_piqa_nonparallel',FULL/'global_piqa_nonparallel','global_piqa_nonparallel'),
    ('ewok','ewok',FULL/'ewok_filtered_word_tokenize','ewok_filtered_word_tokenize'),
]

def setup_env():
    env=os.environ.copy(); env['NLTK_DATA']=str((ROOT/'data/nltk_data').resolve())
    hf=ROOT/'training/hf_home'
    env['HF_HOME']=str(hf.resolve()); env['HF_HUB_CACHE']=str((hf/'hub').resolve())
    env['TRANSFORMERS_CACHE']=str((hf/'transformers').resolve()); env['HF_MODULES_CACHE']=str((ROOT/'training/hf_modules_cache').resolve())
    env.setdefault('TOKENIZERS_PARALLELISM','false')
    for k in ['HF_HOME','HF_HUB_CACHE','TRANSFORMERS_CACHE','HF_MODULES_CACHE']: pathlib.Path(env[k]).mkdir(parents=True, exist_ok=True)
    return env

def run(cmd, env, logf, gpu):
    e=dict(env); e['CUDA_VISIBLE_DEVICES']=str(gpu)
    line=f'[gpu{gpu}] $ '+' '.join(map(str,cmd)); print(line, flush=True); logf.write('\n'+line+'\n'); logf.flush()
    p=subprocess.run([str(x) for x in cmd], cwd=str(STRICT), env=e, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.stdout[-2000:], flush=True); logf.write(p.stdout+f'\n[returncode={p.returncode}]\n'); logf.flush()
    if p.returncode: raise RuntimeError(f'failed {p.returncode}: {line}\n{p.stdout[-5000:]}')

def read_avg(report:pathlib.Path):
    txt=report.read_text(encoding='utf-8', errors='replace')
    m=re.search(r'AVERAGE ACCURACY\s*\n([0-9.\-]+)', txt)
    if m: return float(m.group(1))
    vals=re.findall(r'(?:acc_norm|accuracy|acc)\s*[:=]\s*([0-9.]+)', txt, flags=re.I)
    if vals:
        v=float(vals[-1]); return v*(100 if v<=1 else 1)
    raise RuntimeError(f'cannot parse average from {report}\n{txt[:1000]}')

def read_reading(report:pathlib.Path):
    txt=report.read_text(encoding='utf-8', errors='replace'); out={}
    for label,key in [('EYE TRACKING SCORE','reading_eye_tracking'),('SELF-PACED READING SCORE','reading_self_paced')]:
        m=re.search(re.escape(label)+r':\s*([0-9.\-]+)', txt)
        if not m: raise RuntimeError(f'cannot parse {label} from {report}\n{txt[:1000]}')
        out[key]=float(m.group(1))
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--run_name', required=True)
    ap.add_argument('--out_json', required=True)
    ap.add_argument('--out_note', required=True)
    ap.add_argument('--log', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--revision', default='chck_100M')
    ap.add_argument('--gpu', type=int, default=0)
    args=ap.parse_args()
    model=pathlib.Path(args.model_path).resolve(); outdir=pathlib.Path(args.output_dir).resolve(); outdir.mkdir(parents=True, exist_ok=True)
    if not model.exists(): raise FileNotFoundError(model)
    log_path=pathlib.Path(args.log); log_path.parent.mkdir(parents=True, exist_ok=True)
    env=setup_env(); scores={}; reports={}
    with log_path.open('a', encoding='utf-8') as logf:
        for col,task,data,dataset in TASKS:
            run([sys.executable,'-m','evaluation_pipeline.sentence_zero_shot.run','--model_path_or_name',str(model),'--backend','mlm','--task',task,'--data_path',str(data.resolve()),'--save_predictions','--revision_name',args.revision,'--batch_size','128' if col!='ewok' else '64','--output_dir',str(outdir)], env, logf, args.gpu)
            hits=sorted(outdir.rglob(f'zero_shot/mlm/{task}/{dataset}/best_temperature_report.txt'), key=lambda p:str(p))
            if not hits and col=='ewok': hits=sorted(outdir.rglob('zero_shot/mlm/ewok/*/best_temperature_report.txt'), key=lambda p:str(p))
            if not hits: raise FileNotFoundError(f'no report for {col} under {outdir}')
            scores[col]=read_avg(hits[-1]); reports[col]=str(hits[-1])
        reading=FULL/'reading/reading_data.csv'
        run([sys.executable,'-m','evaluation_pipeline.reading.run','--model_path_or_name',str(model),'--backend','mlm','--data_path',str(reading.resolve()),'--revision_name',args.revision,'--output_dir',str(outdir)], env, logf, args.gpu)
        hits=sorted(outdir.rglob('zero_shot/mlm/reading/report.txt'), key=lambda p:str(p))
        if not hits: raise FileNotFoundError(f'no reading report under {outdir}')
        reports['reading']=str(hits[-1]); scores.update(read_reading(hits[-1]))
    scores['GlobalPIQA_mean']=(scores['global_piqa_parallel']+scores['global_piqa_nonparallel'])/2
    scores['Reading_mean']=(scores['reading_eye_tracking']+scores['reading_self_paced'])/2
    scores['NLP_mean_no_superglue_aoa']=sum(scores[k] for k in ['blimp','supplement','ewok','entity_tracking','comps','GlobalPIQA_mean','Reading_mean'])/7
    payload={'run_name':args.run_name,'model_path':str(model),'revision':args.revision,'scores':scores,'reports':reports,'missing_columns':['SuperGLUE','AoA'],'note':'Zero-shot/Reading coordinate for candidate triage; run SuperGLUE/AoA only after this supports further evaluation.'}
    out_json=pathlib.Path(args.out_json); out_json.parent.mkdir(parents=True, exist_ok=True); out_json.write_text(json.dumps(payload, indent=2)+'\n')
    lines=[f'# research coordinate — {args.run_name}','',f'Evidence JSON: `{out_json}`','', '| column | score |','|---|---:|']
    for k in ['blimp','supplement','ewok','entity_tracking','comps','GlobalPIQA_mean','Reading_mean','NLP_mean_no_superglue_aoa']:
        lines.append(f'| {k} | {scores[k]:.3f} |')
    pathlib.Path(args.out_note).write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'status':'ZERO_SHOT_READING_DONE','out':str(out_json),'scores':scores}, indent=2))
if __name__=='__main__': main()
