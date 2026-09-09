#!/usr/bin/env python3
"""research: diagnose public RecGPT local Reading anomaly without rerunning model.

Compares saved Reading prediction distributions for RecGPT against a known causal
checkpoint that scores normally, and inspects RecGPT tokenizer word vs leading-space
word tokenization on the Reading targets.
"""
from __future__ import annotations
import json, pathlib, math
import pandas as pd
from transformers import AutoTokenizer

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
REC_PRED = ROOT/'training/runs/recgpt_public_causal_eval/patched_model/recgpt_public/zero_shot/causal/reading/prediction.jsonl'
BASE_PRED = ROOT/'training/runs/babylm_compare_dense6x384_10M_curve/eval_results_profile_curve/hf_model/chck_10M/zero_shot/causal/reading/prediction.jsonl'
READ = ROOT/'repos/babylm-eval/strict/evaluation_data/full_eval/reading/reading_data.csv'
TOKDIR = ROOT/'data/recgpt_local/patched_model'
OUT = ROOT/'data/recgpt_reading_anomaly_diagnostic.json'

def load_pred(path):
    rows=[]
    with path.open() as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)

def describe(series):
    s=pd.to_numeric(series, errors='coerce').dropna()
    return {k: float(v) for k,v in {
        'n': len(s), 'mean': s.mean(), 'std': s.std(), 'min': s.min(), 'p01': s.quantile(.01),
        'p10': s.quantile(.10), 'median': s.median(), 'p90': s.quantile(.90), 'p99': s.quantile(.99), 'max': s.max()
    }.items()}

def corr(df, cols):
    out={}
    for c in cols:
        if c in df:
            out[c]=float(df['pred'].corr(pd.to_numeric(df[c], errors='coerce')))
    return out

def main():
    rec=load_pred(REC_PRED).rename(columns={'Logprob':'pred','Prev_Logprob':'prev_pred'})
    base=load_pred(BASE_PRED).rename(columns={'Logprob':'pred','Prev_Logprob':'prev_pred'})
    data=pd.read_csv(READ, dtype={'item':str}).reset_index().rename(columns={'index':'Index'})
    recm=rec.merge(data, on='Index', how='left', suffixes=('','_data'))
    basem=base.merge(data, on='Index', how='left', suffixes=('','_data'))
    tok=AutoTokenizer.from_pretrained(str(TOKDIR), trust_remote_code=True, use_fast=True)
    words=data['word'].astype(str).fillna('').tolist()
    no_space=[tok(w, add_special_tokens=False)['input_ids'] for w in words]
    with_space=[tok(' '+w, add_special_tokens=False)['input_ids'] for w in words]
    diff=sum(1 for a,b in zip(no_space,with_space) if a!=b)
    out={
        'status':'RECGPT_READING_ANOMALY_DIAGNOSTIC',
        'recgpt_prediction_file':str(REC_PRED),
        'baseline_prediction_file':str(BASE_PRED),
        'recgpt_pred_summary':describe(recm['pred']),
        'baseline_pred_summary':describe(basem['pred']),
        'recgpt_prev_pred_summary':describe(recm['prev_pred']),
        'baseline_prev_pred_summary':describe(basem['prev_pred']),
        'recgpt_correlations_with_covariates_and_dvs':corr(recm,['length','context_length','Subtlex_log10','RTfirstfix','RTfirstpass','RTgopast','RTrightbound','self_paced_reading_time']),
        'baseline_correlations_with_covariates_and_dvs':corr(basem,['length','context_length','Subtlex_log10','RTfirstfix','RTfirstpass','RTgopast','RTrightbound','self_paced_reading_time']),
        'recgpt_tokenizer':{
            'class':tok.__class__.__name__, 'vocab_size':len(tok), 'pad_token_id':tok.pad_token_id,
            'targets_n':len(words),
            'no_space_len_mean':sum(map(len,no_space))/len(no_space),
            'with_space_len_mean':sum(map(len,with_space))/len(with_space),
            'no_space_single_frac':sum(1 for x in no_space if len(x)==1)/len(no_space),
            'with_space_single_frac':sum(1 for x in with_space if len(x)==1)/len(with_space),
            'word_vs_space_ids_differ_frac':diff/len(words),
            'examples':[{'word':words[i], 'no_space':no_space[i], 'no_space_decoded':[tok.decode([x]) for x in no_space[i]], 'with_space':with_space[i], 'with_space_decoded':[tok.decode([x]) for x in with_space[i]]} for i in range(min(20,len(words)))]
        },
        'interpretation_hint':'RecGPT Reading anomaly is likely not generic low-quality surprisal: raw correlations with RT are high, but unique variance after length/frequency is low. Tokenizer leading-space behavior is a plausible compatibility issue if word vs space-word ids often differ.'
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(out, indent=2, ensure_ascii=False)[:6000])
    print('Saved:', OUT)
if __name__=='__main__': main()
