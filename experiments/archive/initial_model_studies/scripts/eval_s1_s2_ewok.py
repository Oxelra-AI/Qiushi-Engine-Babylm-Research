#!/usr/bin/env python3
from __future__ import annotations
import json, os, pathlib, re, subprocess, sys, time

ROOT=pathlib.Path('experiments/archive/initial_model_studies')
STRICT=ROOT/'repos/babylm-eval/strict'
EWOK=STRICT/'evaluation_data/full_eval/ewok_filtered_word_tokenize'
OUT=ROOT/'data/s1_s2_100m_ewok_scores.json'
LOG=(ROOT.parents[2] / 'research/notes/initial_model_studies/s1_s2_ewok_eval.log')
MODELS={
  's1': ROOT/'training/runs/babylm_leadershape_s1_100M_aligned_micro128/hf_model/chck_100M',
  's2': ROOT/'training/runs/babylm_true_s2_100M_wordclock_wwm70_len64256/hf_model/chck_100M',
}
RUN_DIRS={
  's1': ROOT/'training/runs/babylm_leadershape_s1_100M_aligned_micro128/eval_results_ewok',
  's2': ROOT/'training/runs/babylm_true_s2_100M_wordclock_wwm70_len64256/eval_results_ewok',
}

def env_setup():
    env=os.environ.copy(); hf=ROOT/'training/hf_home'
    env['HF_HOME']=str(hf.resolve()); env['HF_HUB_CACHE']=str((hf/'hub').resolve())
    env['TRANSFORMERS_CACHE']=str((hf/'transformers').resolve())
    env['HF_MODULES_CACHE']=str((ROOT/'training/hf_modules_cache').resolve())
    env['NLTK_DATA']=str((ROOT/'data/nltk_data').resolve())
    env.setdefault('TOKENIZERS_PARALLELISM','false')
    return env

def parse(report:pathlib.Path)->float:
    txt=report.read_text(encoding='utf-8',errors='replace')
    m=re.search(r'AVERAGE ACCURACY\s*\n([0-9.\-]+)',txt)
    if not m: raise RuntimeError(f'parse failed {report}\n{txt[:1000]}')
    return float(m.group(1))

def find_report(outdir:pathlib.Path):
    hits=sorted(outdir.rglob('zero_shot/mlm/ewok/ewok_filtered_word_tokenize/best_temperature_report.txt'),key=lambda p:str(p))
    if not hits: hits=sorted(outdir.rglob('zero_shot/mlm/ewok/*/best_temperature_report.txt'),key=lambda p:str(p))
    return hits[-1] if hits else None

def main():
    t0=time.time(); LOG.parent.mkdir(parents=True,exist_ok=True); OUT.parent.mkdir(parents=True,exist_ok=True)
    if not EWOK.exists() or not any(EWOK.glob('*.jsonl')): raise RuntimeError(f'missing EWoK data {EWOK}')
    env=env_setup(); payload={'status':'S1_S2_EWOK','ewok_data':str(EWOK),'models':{}}
    with LOG.open('a',encoding='utf-8') as logf:
        for name,model in MODELS.items():
            if not model.exists(): raise RuntimeError(f'missing {name} {model}')
            outdir=RUN_DIRS[name]; outdir.mkdir(parents=True,exist_ok=True)
            rep=find_report(outdir)
            if rep is None:
                cmd=[sys.executable,'-m','evaluation_pipeline.sentence_zero_shot.run','--model_path_or_name',str(model.resolve()),'--backend','mlm','--task','ewok','--data_path',str(EWOK.resolve()),'--save_predictions','--revision_name','step203_'+name,'--batch_size','64','--output_dir',str(outdir.resolve())]
                line='$ '+' '.join(cmd); print(line,flush=True); logf.write('\n'+line+'\n'); logf.flush()
                p=subprocess.run(cmd,cwd=str(STRICT),env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
                print(p.stdout[-2000:],flush=True); logf.write(p.stdout+f'\n[rc={p.returncode}]\n'); logf.flush()
                if p.returncode: raise RuntimeError(f'EWoK failed for {name}: {p.returncode}\n{p.stdout[-5000:]}')
                rep=find_report(outdir)
            if rep is None: raise RuntimeError(f'no report for {name}')
            payload['models'][name]={'model_path':str(model),'ewok':parse(rep),'report':str(rep),'predictions':str(rep.parent/'predictions.json')}
            OUT.write_text(json.dumps(payload,indent=2)+'\n')
    payload['elapsed_sec']=round(time.time()-t0,1)
    OUT.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(payload,indent=2))
if __name__=='__main__': main()
