#!/usr/bin/env python3
from __future__ import annotations
import json, os, pathlib, re, subprocess, sys

ROOT=pathlib.Path('experiments/archive/initial_model_studies')
STRICT=ROOT/'repos/babylm-eval/strict'
FULL=STRICT/'evaluation_data/full_eval'
RUN=ROOT/'training/runs/wess_aux_base_transfer_screen/smoke_384x4_anneal'
OUT_JSON=ROOT/'data/exported_available_scores.json'
LOG=(ROOT.parents[2] / 'research/notes/initial_model_studies/exported_available_eval.log')
BACKEND='mlm'
MODELS={
    'plain_mixed': RUN/'plain_mixed/hf_model',
    'wess_aux_exported_base': RUN/'wess_aux/hf_model',
}
TASKS=[
 ('blimp','blimp',FULL/'blimp_filtered','blimp_filtered'),
 ('supplement','blimp',FULL/'supplement_filtered','supplement_filtered'),
 ('entity_tracking','entity_tracking',FULL/'entity_tracking','entity_tracking'),
 ('comps','comps',FULL/'comps','comps'),
 ('global_piqa_parallel','global_piqa_parallel',FULL/'global_piqa_parallel','global_piqa_parallel'),
 ('global_piqa_nonparallel','global_piqa_nonparallel',FULL/'global_piqa_nonparallel','global_piqa_nonparallel'),
]

def env_setup():
    env=os.environ.copy(); hf=ROOT/'training/hf_home'
    env['HF_HOME']=str(hf.resolve()); env['HF_HUB_CACHE']=str((hf/'hub').resolve()); env['TRANSFORMERS_CACHE']=str((hf/'transformers').resolve()); env['HF_MODULES_CACHE']=str((ROOT/'training/hf_modules_cache').resolve()); env.setdefault('TOKENIZERS_PARALLELISM','false')
    for k in ['HF_HOME','HF_HUB_CACHE','TRANSFORMERS_CACHE','HF_MODULES_CACHE']: pathlib.Path(env[k]).mkdir(parents=True,exist_ok=True)
    return env

def run(cmd,env,logf):
    line='$ '+' '.join(map(str,cmd)); print(line,flush=True); logf.write('\n'+line+'\n'); logf.flush()
    p=subprocess.run([str(x) for x in cmd],cwd=str(STRICT),env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    print(p.stdout[-2000:],flush=True); logf.write(p.stdout+f'\n[returncode={p.returncode}]\n'); logf.flush()
    if p.returncode: raise RuntimeError(f'command failed {p.returncode}: {line}\n{p.stdout[-6000:]}')

def read_avg(report:pathlib.Path)->float:
    txt=report.read_text(encoding='utf-8',errors='replace')
    m=re.search(r'AVERAGE ACCURACY\s*\n([0-9.\-]+)',txt)
    if m: return float(m.group(1))
    vals=re.findall(r'(?:acc_norm|accuracy|acc)\s*[:=]\s*([0-9.]+)',txt,flags=re.I)
    if vals:
        v=float(vals[-1]); return v*(100 if v<=1 else 1)
    raise RuntimeError(f'parse failed {report}\n{txt[:1000]}')

def read_reading(report:pathlib.Path)->dict[str,float]:
    txt=report.read_text(encoding='utf-8',errors='replace'); out={}
    for label,key in [('EYE TRACKING SCORE','reading_eye_tracking'),('SELF-PACED READING SCORE','reading_self_paced')]:
        m=re.search(re.escape(label)+r':\s*([0-9.\-]+)',txt)
        if not m: raise RuntimeError(f'parse {label} failed {report}')
        out[key]=float(m.group(1))
    return out

def main():
    env=env_setup(); LOG.parent.mkdir(parents=True,exist_ok=True)
    payload={'status':'EXPORTED_AVAILABLE_EVAL','run':str(RUN),'backend':BACKEND,'models':{}}
    with LOG.open('a',encoding='utf-8') as logf:
      for name,model in MODELS.items():
        if not model.exists(): raise RuntimeError(f'missing model {model}')
        outdir=RUN/f'eval_available_{name}'; outdir.mkdir(parents=True,exist_ok=True)
        scores={}; reports={}
        for col,task,data,dataset in TASKS:
            run([sys.executable,'-m','evaluation_pipeline.sentence_zero_shot.run','--model_path_or_name',str(model.resolve()),'--backend',BACKEND,'--task',task,'--data_path',str(data.resolve()),'--save_predictions','--revision_name',f'step192_{name}','--batch_size','128','--output_dir',str(outdir.resolve())],env,logf)
            cand=sorted(outdir.rglob(f'zero_shot/{BACKEND}/{task}/{dataset}/best_temperature_report.txt'),key=lambda p:str(p))
            if not cand: raise RuntimeError(f'no report {name} {col}')
            reports[col]=str(cand[-1]); scores[col]=read_avg(cand[-1])
        reading=FULL/'reading/reading_data.csv'
        run([sys.executable,'-m','evaluation_pipeline.reading.run','--model_path_or_name',str(model.resolve()),'--backend',BACKEND,'--data_path',str(reading.resolve()),'--revision_name',f'step192_{name}','--output_dir',str(outdir.resolve())],env,logf)
        cand=sorted(outdir.rglob(f'zero_shot/{BACKEND}/reading/report.txt'),key=lambda p:str(p))
        if cand:
            reports['reading']=str(cand[-1]); scores.update(read_reading(cand[-1]))
        scores['GlobalPIQA_mean']=(scores['global_piqa_parallel']+scores['global_piqa_nonparallel'])/2
        scores['Reading_mean']=(scores.get('reading_eye_tracking',0)+scores.get('reading_self_paced',0))/2
        payload['models'][name]={'model_path':str(model),'scores':scores,'reports':reports}
        OUT_JSON.write_text(json.dumps(payload,indent=2)+'\n')
    # deltas wess-base minus plain
    p=payload['models']['plain_mixed']['scores']; w=payload['models']['wess_aux_exported_base']['scores']
    payload['deltas_wess_exported_minus_plain']={k:w[k]-p[k] for k in sorted(set(p)&set(w)) if isinstance(p[k],(int,float))}
    OUT_JSON.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps({'out':str(OUT_JSON),'deltas':payload['deltas_wess_exported_minus_plain']},indent=2))
if __name__=='__main__': main()
