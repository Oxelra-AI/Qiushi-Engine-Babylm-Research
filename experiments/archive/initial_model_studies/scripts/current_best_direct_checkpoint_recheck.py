#!/usr/bin/env python3
"""research direct checkpoint recheck: seed43 chck_80M vs chck_100M.

Avoids local parent+revision loading artifact by passing exact checkpoint directories
as --model_path_or_name. Evaluates the fast/direct columns that were used in the
research endpoint choice and are affordable in this step: BLiMP, Supplement, EWoK,
Entity, COMPS, Reading.
"""
from __future__ import annotations
import hashlib, json, os, pathlib, re, subprocess, sys, time
ROOT=pathlib.Path('experiments/archive/initial_model_studies')
STRICT=ROOT/'repos/babylm-eval/strict'
BASE=ROOT/'training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model'
CKPTS={'wwm43_chck_80M':BASE/'chck_80M','wwm43_chck_100M':BASE/'chck_100M'}
OUT=ROOT/'data/current_best_direct_checkpoint_recheck.json'
NOTE=(ROOT.parents[2] / 'research/notes/initial_model_studies/current_best_direct_checkpoint_recheck.md')
LOG=(ROOT.parents[2] / 'research/notes/initial_model_studies/current_best_direct_checkpoint_recheck.log')
TASKS=[
 ('BLiMP','blimp','evaluation_data/fast_eval/blimp_fast','blimp_fast','blimp_fast'),
 ('Supplement','blimp','evaluation_data/fast_eval/supplement_fast','supplement_fast','supplement_fast'),
 ('EWoK','ewok','evaluation_data/fast_eval/evaluation_data/fast_eval/ewok_fast','ewok_fast','ewok_fast'),
 ('Entity','entity_tracking','evaluation_data/fast_eval/entity_tracking_fast','entity_tracking_fast','entity_tracking_fast'),
 ('COMPS','comps','evaluation_data/full_eval/comps','comps','comps'),
]

def env_setup():
    env=os.environ.copy(); hf=ROOT/'training/hf_home'
    env['HF_HOME']=str(hf.resolve()); env['HF_HUB_CACHE']=str((hf/'hub').resolve())
    env['HF_DATASETS_CACHE']=str((hf/'datasets').resolve()); env['TRANSFORMERS_CACHE']=str((hf/'transformers').resolve())
    env['HF_MODULES_CACHE']=str((ROOT/'training/hf_modules_cache').resolve()); env.setdefault('TOKENIZERS_PARALLELISM','false')
    return env

def sha16(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for c in iter(lambda:f.read(1<<20),b''): h.update(c)
    return h.hexdigest()[:16]

def run_cmd(cmd,env,logf,cwd=None):
    line='$ '+' '.join(map(str,cmd)); print(line,flush=True); logf.write('\n'+line+'\n'); logf.flush()
    p=subprocess.run(cmd,cwd=cwd,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    print(p.stdout[-1400:],flush=True); logf.write(p.stdout+f'\n[returncode={p.returncode}]\n'); logf.flush()
    if p.returncode!=0: raise RuntimeError(line+'\n'+p.stdout[-6000:])

def parse_avg(report):
    txt=pathlib.Path(report).read_text(errors='replace')
    m=re.search(r'AVERAGE ACCURACY\s*\n([0-9.\-]+)',txt)
    if not m: raise RuntimeError(f'parse avg failed {report}')
    return float(m.group(1))

def parse_reading(report):
    txt=pathlib.Path(report).read_text(errors='replace'); out={}
    for lab,key in [('EYE TRACKING SCORE','reading_eye_tracking'),('SELF-PACED READING SCORE','reading_self_paced')]:
        m=re.search(re.escape(lab)+r':\s*([0-9.\-]+)',txt)
        if not m: raise RuntimeError(f'parse reading failed {report}')
        out[key]=float(m.group(1))
    out['Reading']=(out['reading_eye_tracking']+out['reading_self_paced'])/2
    return out

def eval_ckpt(label,ckpt,env,logf):
    outdir=ckpt.parent.parent/f'eval_step322_direct_{label}'; scores={}; reports={}
    for score_name,task,data_path,ds_name,key in TASKS:
        report=outdir/ckpt.name/'main'/'zero_shot'/'mlm'/task/ds_name/'best_temperature_report.txt'
        if not report.exists():
            run_cmd([sys.executable,'-m','evaluation_pipeline.sentence_zero_shot.run','--model_path_or_name',str(ckpt.resolve()),'--backend','mlm','--task',task,'--data_path',data_path,'--save_predictions','--batch_size','64','--output_dir',str(outdir.resolve())],env,logf,cwd=str(STRICT))
        scores[score_name]=parse_avg(report); reports[score_name]=str(report)
    rr=outdir/ckpt.name/'main'/'zero_shot'/'mlm'/'reading'/'report.txt'
    if not rr.exists():
        run_cmd([sys.executable,'-m','evaluation_pipeline.reading.run','--model_path_or_name',str(ckpt.resolve()),'--backend','mlm','--data_path','evaluation_data/fast_eval/reading/reading_data.csv','--output_dir',str(outdir.resolve())],env,logf,cwd=str(STRICT))
    scores.update(parse_reading(rr)); reports['Reading']=str(rr)
    return {'path':str(ckpt),'sha16':sha16(ckpt/'model.safetensors'),'scores':scores,'reports':reports}

def main():
    t0=time.time(); env=env_setup(); profiles={}; LOG.parent.mkdir(parents=True,exist_ok=True)
    with LOG.open('a',encoding='utf-8') as logf:
        logf.write('\n===== research direct checkpoint recheck =====\n')
        for label,ckpt in CKPTS.items(): profiles[label]=eval_ckpt(label,ckpt,env,logf)
    s80=profiles['wwm43_chck_80M']['scores']; s100=profiles['wwm43_chck_100M']['scores']
    delta={k:s80[k]-s100[k] for k in ['BLiMP','Supplement','EWoK','Entity','COMPS','Reading','reading_eye_tracking','reading_self_paced']}
    mean6_80=sum(s80[k] for k in ['BLiMP','Supplement','EWoK','Entity','COMPS','Reading'])/6
    mean6_100=sum(s100[k] for k in ['BLiMP','Supplement','EWoK','Entity','COMPS','Reading'])/6
    payload={'status':'CURRENT_BEST_DIRECT_CHECKPOINT_RECHECK','method':'exact checkpoint paths; no parent+revision loading','profiles':profiles,'chck80_minus_chck100':delta,'mean6':{'chck_80M':mean6_80,'chck_100M':mean6_100,'delta_80_minus_100':mean6_80-mean6_100},'elapsed_sec':time.time()-t0}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    lines=['# research current-best direct checkpoint recheck','',f'JSON: `{OUT}`',f'Log: `{LOG}`','','Exact checkpoint directories were passed as `--model_path_or_name`; no local parent+revision loading.','','## Scores','','| checkpoint | BLiMP | Supplement | EWoK | Entity | COMPS | Reading | mean6 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for lab,key in [('chck_80M','wwm43_chck_80M'),('chck_100M','wwm43_chck_100M')]:
        s=profiles[key]['scores']; m=mean6_80 if key.endswith('80M') else mean6_100
        lines.append(f"| {lab} | {s['BLiMP']:.2f} | {s['Supplement']:.2f} | {s['EWoK']:.2f} | {s['Entity']:.2f} | {s['COMPS']:.2f} | {s['Reading']:.3f} | {m:.3f} |")
    lines += ['','## chck_80M - chck_100M','','| BLiMP | Supplement | EWoK | Entity | COMPS | Reading | mean6 |','|---:|---:|---:|---:|---:|---:|---:|',f"| {delta['BLiMP']:+.2f} | {delta['Supplement']:+.2f} | {delta['EWoK']:+.2f} | {delta['Entity']:+.2f} | {delta['COMPS']:+.2f} | {delta['Reading']:+.3f} | {mean6_80-mean6_100:+.3f} |"]
    NOTE.parent.mkdir(parents=True,exist_ok=True); NOTE.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'out':str(OUT),'note':str(NOTE),'delta':delta,'mean6':payload['mean6']},indent=2))
if __name__=='__main__': main()
