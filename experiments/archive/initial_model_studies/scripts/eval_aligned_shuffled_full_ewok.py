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
    'aligned': ROOT / 'training/runs/babylm_aligned_s1_10M',
    'shuffled': ROOT / 'training/runs/babylm_shuffled_s1_10M',
}
OUT_JSONS = {
    'aligned': ROOT / 'data/aligned_10m_full_ewok_word_tokenize_score.json',
    'shuffled': ROOT / 'data/shuffled_10m_full_ewok_word_tokenize_score.json',
}
SUMMARY = ROOT / 'data/aligned_shuffled_10m_ewok_word_tokenize_filter_summary.json'
OUT_COMPARISON = ROOT / 'data/aligned_vs_shuffled_10m_full_ewok_comparison.json'
NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/aligned_vs_shuffled_10m_full_ewok_comparison.md')
LOG = (ROOT.parents[2] / 'research/notes/initial_model_studies/aligned_shuffled_full_ewok_eval.log')

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

def build_ewok_filter(log):
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
    return summary

def eval_arm(arm: str, run_dir: pathlib.Path, env, log):
    model=(run_dir/'hf_model/chck_9999996w').resolve()
    if not model.exists(): raise RuntimeError(f'Missing direct checkpoint {model}')
    eval_out=run_dir/'eval_results_full_ewok_word_tokenize_direct'
    eval_out.mkdir(parents=True, exist_ok=True)
    cmd=[sys.executable,'-m','evaluation_pipeline.sentence_zero_shot.run','--model_path_or_name',str(model),'--backend','mlm','--task','ewok','--data_path',str(EWOK_OUT.resolve()),'--save_predictions','--batch_size','64','--output_dir',str(eval_out.resolve())]
    log('$ '+' '.join(cmd))
    p=subprocess.run(cmd, cwd=str(STRICT), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log(p.stdout[-8000:])
    if p.returncode != 0: raise RuntimeError(f'EWoK eval failed {arm} {p.returncode}\n{p.stdout[-10000:]}')
    hits=sorted(eval_out.rglob('best_temperature_report.txt'))
    if not hits: raise FileNotFoundError(f'no best_temperature_report under {eval_out}')
    if len(hits) != 1: raise RuntimeError(f'expected one EWoK report for {arm}, got {hits[:10]}')
    report=hits[0]; pred=report.parent/'predictions.json'; score=parse_avg(report)
    metrics=json.loads((run_dir/'scientific_metrics.json').read_text())
    payload={'status':f'STEP154_{arm.upper()}_S1_10M_FULL_EWOK_WORD_TOKENIZE','arm':arm,'model':f'{arm} S1 12x384 DeBERTa-v2 baseline16k WWM 10M','model_path_direct':str(model),'backend':'mlm','ewok_full_score':score,'training_core':{k:metrics.get(k) for k in ['word_exposure','actual_training_steps','loss_first','loss_last','example_jsonl_label','example_jsonl_total_words','example_jsonl_total_rows']},'filter_summary':str(SUMMARY),'report':str(report),'predictions':str(pred),'eval_output_dir':str(eval_out),'warning':'Full local EWoK generated from provided parquet with official nltk.word_tokenize resources and official vocab filter.'}
    OUT_JSONS[arm].write_text(json.dumps(payload, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    return payload

def main():
    t0=time.time(); LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open('w', encoding='utf-8') as logf:
        def log(msg): print(msg, flush=True); logf.write(msg+'\n'); logf.flush()
        summary=build_ewok_filter(log)
        env=setup_env(); payloads={}
        for arm,run_dir in ARMS.items():
            payloads[arm]=eval_arm(arm, run_dir, env, log)
        delta=payloads['aligned']['ewok_full_score']-payloads['shuffled']['ewok_full_score']
        comparison={'status':'ALIGNED_MINUS_SHUFFLED_10M_FULL_EWOK_COMPARISON','mechanism':'correct semantic correspondence within MLM window vs identical source/target text multisets with wrong correspondence','aligned_json':str(OUT_JSONS['aligned']),'shuffled_json':str(OUT_JSONS['shuffled']),'scores':{'aligned':payloads['aligned']['ewok_full_score'],'shuffled':payloads['shuffled']['ewok_full_score']},'aligned_minus_shuffled_ewok':delta,'filter_summary':str(SUMMARY),'elapsed_sec':time.time()-t0}
        OUT_COMPARISON.write_text(json.dumps(comparison, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
        NOTE.write_text('\n'.join(['# research — aligned vs shuffled 10M full EWoK comparison','',f'Comparison JSON: `{OUT_COMPARISON}`',f'Filter summary: `{SUMMARY}`','', '| arm | full EWoK |','|---|---:|',f'| aligned | {payloads["aligned"]["ewok_full_score"]:.2f} |',f'| shuffled | {payloads["shuffled"]["ewok_full_score"]:.2f} |',f'| aligned - shuffled | {delta:+.2f} |','', 'This uses local EWoK parquet, official BabyLM vocab filter, and NLTK `word_tokenize` with local `data/nltk_data`.'])+'\n', encoding='utf-8')
        log(json.dumps({'status':comparison['status'],'comparison':str(OUT_COMPARISON),'aligned_minus_shuffled_ewok':delta,'elapsed_sec':comparison['elapsed_sec']}, indent=2))
if __name__=='__main__': main()
