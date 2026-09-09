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
RUN = ROOT / 'training/runs/babylm_s3_12x384_official40k_flatwwm_10M'
MODEL = (RUN / 'hf_model/chck_10M').resolve()
EVAL_OUT = RUN / 'eval_results_full_ewok_word_tokenize_direct'
SUMMARY = ROOT / 'data/s3_10m_ewok_word_tokenize_filter_summary.json'
SCORE_JSON = ROOT / 'data/s3_10m_full_ewok_word_tokenize_score.json'
NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/s3_10m_full_ewok_word_tokenize_score.md')
LOG = (ROOT.parents[2] / 'research/notes/initial_model_studies/s3_10m_full_ewok_eval.log')

def read_parquet_rows(path: pathlib.Path):
    try:
        import pandas as pd
        return pd.read_parquet(path).to_dict(orient='records')
    except Exception:
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()

def parse_avg(report: pathlib.Path) -> float:
    txt = report.read_text(encoding='utf-8', errors='replace')
    m = re.search(r'AVERAGE ACCURACY\s*\n([0-9.\-]+)', txt)
    if not m:
        raise RuntimeError(f'Could not parse average from {report}\n{txt[:1000]}')
    return float(m.group(1))

def setup_env():
    env = os.environ.copy(); env['NLTK_DATA'] = str(NLTK_DATA)
    hf = ROOT / 'training/hf_home'
    env['HF_HOME'] = str(hf.resolve()); env['HF_HUB_CACHE'] = str((hf/'hub').resolve())
    env['TRANSFORMERS_CACHE'] = str((hf/'transformers').resolve())
    env['HF_MODULES_CACHE'] = str((ROOT/'training/hf_modules_cache').resolve())
    env.setdefault('TOKENIZERS_PARALLELISM','false')
    for k in ['HF_HOME','HF_HUB_CACHE','TRANSFORMERS_CACHE','HF_MODULES_CACHE']:
        pathlib.Path(env[k]).mkdir(parents=True, exist_ok=True)
    return env

def main():
    t0=time.time(); LOG.parent.mkdir(parents=True, exist_ok=True); logf=LOG.open('w', encoding='utf-8')
    def log(msg): print(msg, flush=True); logf.write(msg+'\n'); logf.flush()
    if not MODEL.exists(): raise RuntimeError(f'Missing direct checkpoint {MODEL}')
    os.environ['NLTK_DATA'] = str(NLTK_DATA)
    import nltk; nltk.data.path.insert(0, str(NLTK_DATA))
    from nltk.tokenize import word_tokenize
    test_tokens = word_tokenize('This is a test.')
    if test_tokens != ['This','is','a','test','.']:
        raise RuntimeError(f'Unexpected word_tokenize output {test_tokens}')
    rows = read_parquet_rows(PARQUET)
    vocab = {line.strip() for line in VOCAB.read_text(encoding='utf-8').splitlines() if line.strip()}
    EWOK_OUT.mkdir(parents=True, exist_ok=True)
    for old in EWOK_OUT.glob('*.jsonl'): old.unlink()
    items_per_domain=defaultdict(list); skipped=0; skip_by_domain=defaultdict(int)
    required=['Domain','Context1','Context2','Target1','Target2','ContextType','ContextDiff','TargetDiff']
    for r in rows:
        missing=[k for k in required if k not in r]
        if missing: raise RuntimeError(f'Missing EWoK columns {missing}')
        domain=str(r['Domain']); bad=False
        for key in ('Context1','Context2','Target1','Target2'):
            for word in word_tokenize(str(r[key]).lower()):
                if word not in vocab:
                    bad=True; break
            if bad: break
        if bad:
            skipped += 1; skip_by_domain[domain] += 1; continue
        items_per_domain[domain].append({k:(v.item() if hasattr(v,'item') else v) for k,v in r.items()})
    written={}
    for domain,items in sorted(items_per_domain.items()):
        path=EWOK_OUT / f'{domain}.jsonl'
        with path.open('w', encoding='utf-8') as out:
            for item in items:
                out.write(json.dumps(item, ensure_ascii=False)+'\n')
                sw=dict(item); sw['Context1'],sw['Context2']=sw['Context2'],sw['Context1']; sw['Target1'],sw['Target2']=sw['Target2'],sw['Target1']
                out.write(json.dumps(sw, ensure_ascii=False)+'\n')
        written[domain]={'filtered_items':len(items),'jsonl_lines_with_swaps':2*len(items),'path':str(path)}
    total_filtered=sum(v['filtered_items'] for v in written.values()); total_lines=sum(v['jsonl_lines_with_swaps'] for v in written.values())
    summary={'source_parquet':str(PARQUET),'nltk_data':str(NLTK_DATA),'tokenizer_used_for_filter':'nltk.word_tokenize','word_tokenize_test':test_tokens,'raw_rows':len(rows),'filtered_items':total_filtered,'skipped_items':skipped,'jsonl_lines_with_swaps':total_lines,'domains':written,'skipped_by_domain':dict(sorted(skip_by_domain.items())),'output_dir':str(EWOK_OUT)}
    SUMMARY.parent.mkdir(parents=True, exist_ok=True); SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    log(json.dumps({'event':'ewok_filtered_word_tokenize','raw_rows':len(rows),'filtered_items':total_filtered,'lines':total_lines,'domains':len(written)}))
    env=setup_env(); EVAL_OUT.mkdir(parents=True, exist_ok=True)
    cmd=[sys.executable,'-m','evaluation_pipeline.sentence_zero_shot.run','--model_path_or_name',str(MODEL),'--backend','mlm','--task','ewok','--data_path',str(EWOK_OUT.resolve()),'--save_predictions','--batch_size','64','--output_dir',str(EVAL_OUT.resolve())]
    log('$ '+' '.join(cmd)); p=subprocess.run(cmd, cwd=str(STRICT), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log(p.stdout[-8000:])
    if p.returncode != 0: raise RuntimeError(f'EWoK eval failed {p.returncode}\n{p.stdout[-10000:]}')
    hits=sorted(EVAL_OUT.rglob('best_temperature_report.txt'))
    if not hits: raise FileNotFoundError(f'no best_temperature_report under {EVAL_OUT}')
    report=hits[0]; pred=report.parent/'predictions.json'; score=parse_avg(report)
    payload={'status':'S3_10M_FULL_EWOK_WORD_TOKENIZE','model':'S3 12x384 DeBERTa-v2 legal official40k WWM 10M','model_path_direct':str(MODEL),'backend':'mlm','ewok_full_score':score,'filter_summary':str(SUMMARY),'report':str(report),'predictions':str(pred),'eval_output_dir':str(EVAL_OUT),'elapsed_sec':time.time()-t0,'warning':'Full local EWoK generated from provided gated parquet with official nltk.word_tokenize resources and official vocab filter.'}
    SCORE_JSON.parent.mkdir(parents=True, exist_ok=True); SCORE_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    NOTE.write_text('\n'.join(['# research — S3 10M full EWoK with official word_tokenize','',f'Evidence JSON: `{SCORE_JSON}`',f'Filter summary: `{SUMMARY}`',f'Report: `{report}`','',f'Full EWoK score: **{score:.2f}**',f'Filtered items: {total_filtered}; swapped JSONL lines: {total_lines}','', 'This uses the local EWoK parquet, official BabyLM vocab filter, and NLTK `word_tokenize` with the local `data/nltk_data` resources.'])+'\n', encoding='utf-8')
    log(json.dumps({'status':payload['status'],'score':score,'score_json':str(SCORE_JSON),'elapsed_sec':payload['elapsed_sec']}, indent=2))
    logf.close()
if __name__=='__main__': main()
