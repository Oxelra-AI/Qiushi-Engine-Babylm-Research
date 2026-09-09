#!/usr/bin/env python3
from __future__ import annotations
import json, os, pathlib, re, subprocess, sys, time
from collections import defaultdict

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
STRICT = ROOT / 'repos/babylm-eval/strict'
PARQUET = ROOT / 'data/ewok-core-1.0/data/test/ewok-core-1.0.parquet'
VOCAB = STRICT / 'evaluation_pipeline/ewok/vocab.txt'
NLTK_DATA = (ROOT / 'data/nltk_data').resolve()
EWOK_OUT = STRICT / 'evaluation_data/full_eval/ewok_filtered_word_tokenize'
ARMS = {
    'high_entity_state': (ROOT/'training/runs/babylm_high_entity_state_s1_10M','chck_9999947w'),
    'matched_low': (ROOT/'training/runs/babylm_matched_low_s1_10M','chck_9999960w'),
    'uniform': (ROOT/'training/runs/babylm_uniform_s1_10M','chck_9999997w'),
}
OUT = ROOT / 'data/structure_arms_full_ewok.json'
NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/structure_arms_full_ewok.md')
LOG = (ROOT.parents[2] / 'research/notes/initial_model_studies/structure_arms_full_ewok.log')
SUMMARY = ROOT / 'data/structure_ewok_filter_summary.json'

def read_parquet_rows(path):
    try:
        import pandas as pd; return pd.read_parquet(path).to_dict(orient='records')
    except Exception:
        import pyarrow.parquet as pq; return pq.read_table(path).to_pylist()

def parse_avg(path):
    txt=path.read_text(encoding='utf-8', errors='replace')
    m=re.search(r'AVERAGE ACCURACY\s*\n([0-9.\-]+)', txt)
    if not m: raise RuntimeError(f'no avg in {path}')
    return float(m.group(1))

def setup_env():
    env=os.environ.copy(); env['NLTK_DATA']=str(NLTK_DATA)
    hf=ROOT/'training/hf_home'
    env['HF_HOME']=str(hf.resolve()); env['HF_HUB_CACHE']=str((hf/'hub').resolve())
    env['TRANSFORMERS_CACHE']=str((hf/'transformers').resolve()); env['HF_MODULES_CACHE']=str((ROOT/'training/hf_modules_cache').resolve())
    env.setdefault('TOKENIZERS_PARALLELISM','false')
    for k in ['HF_HOME','HF_HUB_CACHE','TRANSFORMERS_CACHE','HF_MODULES_CACHE']: pathlib.Path(env[k]).mkdir(parents=True, exist_ok=True)
    return env

def build_filter(log):
    os.environ['NLTK_DATA']=str(NLTK_DATA)
    import nltk; nltk.data.path.insert(0,str(NLTK_DATA)); from nltk.tokenize import word_tokenize
    if word_tokenize('This is a test.') != ['This','is','a','test','.']: raise RuntimeError('word_tokenize mismatch')
    rows=read_parquet_rows(PARQUET); vocab={l.strip() for l in VOCAB.read_text().splitlines() if l.strip()}
    EWOK_OUT.mkdir(parents=True, exist_ok=True)
    for old in EWOK_OUT.glob('*.jsonl'): old.unlink()
    per=defaultdict(list); skipped=0
    req=['Domain','Context1','Context2','Target1','Target2','ContextType','ContextDiff','TargetDiff']
    for r in rows:
        if [k for k in req if k not in r]: raise RuntimeError('missing cols')
        dom=str(r['Domain']); bad=False
        for key in ('Context1','Context2','Target1','Target2'):
            for w in word_tokenize(str(r[key]).lower()):
                if w not in vocab: bad=True; break
            if bad: break
        if bad: skipped+=1; continue
        per[dom].append({k:(v.item() if hasattr(v,'item') else v) for k,v in r.items()})
    written={}
    for dom,items in sorted(per.items()):
        path=EWOK_OUT/f'{dom}.jsonl'
        with path.open('w', encoding='utf-8') as out:
            for it in items:
                out.write(json.dumps(it, ensure_ascii=False)+'\n')
                sw=dict(it); sw['Context1'],sw['Context2']=sw['Context2'],sw['Context1']; sw['Target1'],sw['Target2']=sw['Target2'],sw['Target1']
                out.write(json.dumps(sw, ensure_ascii=False)+'\n')
        written[dom]={'items':len(items),'lines':2*len(items)}
    summary={'raw_rows':len(rows),'filtered_items':sum(v['items'] for v in written.values()),'skipped':skipped,'domains':written,'output_dir':str(EWOK_OUT)}
    SUMMARY.write_text(json.dumps(summary,indent=2)+'\n'); log(json.dumps({'event':'filter','filtered':summary['filtered_items']}))
    return summary

def eval_arm(name, run_dir, ckpt, env, log):
    model=(run_dir/'hf_model'/ckpt).resolve()
    if not model.exists(): raise RuntimeError(f'missing {model}')
    out=run_dir/'eval_results_full_ewok_direct'; out.mkdir(parents=True, exist_ok=True)
    cmd=[sys.executable,'-m','evaluation_pipeline.sentence_zero_shot.run','--model_path_or_name',str(model),'--backend','mlm','--task','ewok','--data_path',str(EWOK_OUT.resolve()),'--save_predictions','--batch_size','64','--output_dir',str(out.resolve())]
    log('$ '+' '.join(cmd))
    p=subprocess.run(cmd, cwd=str(STRICT), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log(p.stdout[-3000:])
    if p.returncode!=0: raise RuntimeError(f'ewok fail {name} {p.returncode}\n{p.stdout[-6000:]}')
    hits=sorted(out.rglob('best_temperature_report.txt'))
    if len(hits)!=1: raise RuntimeError(f'report not unique {name}: {hits[:8]}')
    return parse_avg(hits[0]), str(hits[0])

def main():
    t0=time.time(); LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open('w', encoding='utf-8') as logf:
        def log(m): print(m, flush=True); logf.write(m+'\n'); logf.flush()
        build_filter(log); env=setup_env(); scores={}; reports={}
        for name,(rd,ck) in ARMS.items():
            s,rep=eval_arm(name,rd,ck,env,log); scores[name]=s; reports[name]=rep; log(json.dumps({'arm':name,'ewok':s}))
        payload={'status':'STRUCTURE_ARMS_FULL_EWOK','scores':scores,'reports':reports,'filter_summary':str(SUMMARY),
                 'high_minus_low':scores['high_entity_state']-scores['matched_low'],
                 'high_minus_uniform':scores['high_entity_state']-scores['uniform'],
                 'uniform_minus_low':scores['uniform']-scores['matched_low'],'elapsed_sec':time.time()-t0}
        OUT.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n')
        NOTE.write_text('\n'.join(['# research — structure arms full EWoK','',f'Evidence: `{OUT}`','', '| arm | EWoK |','|---|---:|',
            f'| high_entity_state | {scores["high_entity_state"]:.2f} |',f'| matched_low | {scores["matched_low"]:.2f} |',f'| uniform | {scores["uniform"]:.2f} |','',
            f'high - low = {payload["high_minus_low"]:+.2f}; high - uniform = {payload["high_minus_uniform"]:+.2f}; uniform - low = {payload["uniform_minus_low"]:+.2f}'])+'\n')
        log(json.dumps({'status':payload['status'],'scores':scores,'high_minus_low':payload['high_minus_low']}, indent=2))
if __name__=='__main__': main()
