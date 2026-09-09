#!/usr/bin/env python3
"""research: CDI endpoint NLL/rank profile including dense-mask/sparse-label.

This extends the research endpoint-only CDI diagnostic to the dense-mask/sparse-label
checkpoint. It applies temperatures fitted on ordinary legal-tail text by the research
five-model readout. It is not AoA scoring and does not alter official metrics.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse, hashlib, json, math, os, pathlib, random, statistics, sys, time
from collections import defaultdict
from typing import Any, Iterable, Optional

import torch
from transformers import AutoModelForMaskedLM, AutoProcessor

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
SCRIPTS = _public_path('experiments/archive/functional_learning/scripts')
STRICT = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict')
if str(SCRIPTS) not in sys.path: sys.path.insert(0, str(SCRIPTS))
if str(STRICT) not in sys.path: sys.path.insert(0, str(STRICT))
import batched_aoa_extractor as aoa_batch  # noqa: E402
from evaluation_pipeline.AoA_word.eval_util import load_eval  # noqa: E402

DEFAULT_TEMP = _public_path('experiments/archive/functional_learning/data/temperature_source_readout_five_models/temperature_source_readout.json')
DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/cdi_endpoint_temperature_rank_profile_five_models')
MODEL_PATHS = {
    'coherent86': _public_path('experiments/archive/functional_learning/data/automodel_repair/repaired_coherent86_alpha075'),
    'sparse_focus_seed62064': _public_path('experiments/archive/functional_learning/data/unchanged_focus_weighted_train/correspondence_focus_weighted/checkpoints/update_0080'),
    'dense_focus_seed62064': _public_path('experiments/archive/functional_learning/data/automodel_repair/repaired_dense_seed62064_u0080'),
    'dense_focus_seed62065': _public_path('experiments/archive/functional_learning/data/automodel_repair/repaired_dense_seed62065_u0080'),
    'densemask_sparselabel_seed62064': _public_path('experiments/archive/functional_learning/data/densemask_sparselabel_train_seed62064/correspondence_focus_weighted/checkpoints/update_0080'),
}

def rel(p):
    try: return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception: return str(p)

def now(): return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

def sha256_file(path, block_size=1<<20):
    p=pathlib.Path(path)
    if not p.is_file(): return None
    h=hashlib.sha256()
    with p.open('rb') as f:
        while True:
            b=f.read(block_size)
            if not b: break
            h.update(b)
    return h.hexdigest()

def finite(xs: Iterable[Any]) -> list[float]:
    out=[]
    for x in xs:
        try:
            y=float(x)
            if math.isfinite(y): out.append(y)
        except Exception: pass
    return out

def mean(xs):
    v=finite(xs); return sum(v)/len(v) if v else None

def median(xs):
    v=finite(xs); return statistics.median(v) if v else None

def rank_from_logits(logits_row, target_id): return int(torch.sum(logits_row > logits_row[int(target_id)]).item()) + 1

def nll_from_logits(logits_row, target_id, temperature):
    z=logits_row.float()/float(temperature)
    return float(torch.logsumexp(z, dim=-1).item() - z[int(target_id)].item())

def load_temperatures(path):
    obj=json.loads(path.read_text())
    temps={}
    for name, fit in obj.get('temperature_fits', {}).items():
        temps[name]=float(fit.get('best_temperature',1.0))
    return temps

def make_subset(max_words, seed):
    words, contexts=load_eval(aoa_batch.CDI_WORDS_PATH, min_context=0, debug=False)
    total=sum(len(c) for c in contexts)
    ids=list(range(len(words)))
    if max_words and 0<max_words<len(ids):
        rng=random.Random(seed); ids=sorted(rng.sample(ids,max_words))
    return [words[i] for i in ids], [contexts[i] for i in ids], {'cdi_words_path':rel(aoa_batch.CDI_WORDS_PATH),'cdi_words_sha256':sha256_file(aoa_batch.CDI_WORDS_PATH),'total_words_available':len(words),'total_contexts_available':total,'sampled_word_indices':ids,'sample_seed':seed,'words_evaluated':len(ids),'contexts_evaluated':sum(len(contexts[i]) for i in ids),'target_words_preview':[words[i] for i in ids[:16]]}

def ensure_cache(out_dir):
    # Use the repaired helper from research when available, and synchronize already-imported globals.
    helper=aoa_batch.run_extract.__globals__.get('_ensure_writable_cache_env')
    if helper:
        helper('HF_HOME', out_dir/'hf_cache'/'hf_home')
        helper('TRANSFORMERS_CACHE', out_dir/'hf_cache'/'transformers')
        helper('HF_MODULES_CACHE', out_dir/'hf_cache'/'modules')
    else:
        for k,p in [('HF_HOME',out_dir/'hf_cache'/'hf_home'),('TRANSFORMERS_CACHE',out_dir/'hf_cache'/'transformers'),('HF_MODULES_CACHE',out_dir/'hf_cache'/'modules')]:
            p.mkdir(parents=True, exist_ok=True); os.environ[k]=str(p)
    import transformers.utils.hub as hub, transformers.dynamic_module_utils as dyn
    hub.HF_MODULES_CACHE=os.environ['HF_MODULES_CACHE']; dyn.HF_MODULES_CACHE=os.environ['HF_MODULES_CACHE']

def score_model(name, model_path, temperature, target_words, contexts, device, batch_size):
    processor=AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True, padding_side='right', local_files_only=True)
    tokenizer=processor.tokenizer if hasattr(processor,'tokenizer') else processor
    model=AutoModelForMaskedLM.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True).to(device); model.eval()
    pad_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    batch=[]; rows=[]
    def flush():
        nonlocal batch, rows
        if not batch: return
        L=max(len(x['input_ids']) for x in batch)
        ids=torch.full((len(batch),L), int(pad_id), dtype=torch.long); att=torch.zeros((len(batch),L),dtype=torch.long)
        idx=torch.empty((len(batch),),dtype=torch.long); tids=torch.empty((len(batch),),dtype=torch.long)
        for i,item in enumerate(batch):
            l=len(item['input_ids']); ids[i,:l]=torch.tensor(item['input_ids']); att[i,:l]=torch.tensor(item['attention_mask']); idx[i]=int(item['index']); tids[i]=int(item['target'])
        ids=ids.to(device); att=att.to(device); idx=idx.to(device); tids=tids.to(device)
        with torch.no_grad():
            out=model(input_ids=ids, attention_mask=att); logits=out[0] if isinstance(out,tuple) else out['logits']; z=logits[torch.arange(logits.shape[0],device=logits.device),idx].float().detach().cpu()
        for item, zr, tid in zip(batch,z,tids.detach().cpu().tolist(), strict=False):
            rows.append({'model':name,'word_i':int(item['word_i']),'context_i':int(item['context_i']),'target_word':item['target_word'],'target_token_id':int(tid),'nll_T1':nll_from_logits(zr,tid,1.0),'nll_Tfit':nll_from_logits(zr,tid,temperature),'target_rank':rank_from_logits(zr,tid)})
        batch=[]
    variants=0; t0=time.time()
    for wi,(w,ctxs) in enumerate(zip(target_words,contexts,strict=False)):
        for ci,ctx in enumerate(ctxs):
            toks,atts,phrase_indices,target_tokens=aoa_batch.prepare_mlm_variants(processor, tokenizer, ctx, w, False)
            for ids,att,ix,tok in zip(toks,atts,phrase_indices,target_tokens,strict=False):
                batch.append({'word_i':wi,'context_i':ci,'target_word':w,'input_ids':ids,'attention_mask':att,'index':ix,'target':tok}); variants+=1
                if len(batch)>=batch_size: flush()
    flush(); del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return rows, {'model_name':name,'model_path':rel(model_path),'temperature':temperature,'n_rows':len(rows),'variants':variants,'elapsed_sec':round(time.time()-t0,2),'checkpoint':aoa_batch.checkpoint_info(name, pathlib.Path(model_path))}

def summarize(rows, parent_rows=None):
    parent={(r['word_i'],r['context_i'],r['target_token_id']):r for r in (parent_rows or [])}
    deltas=[]
    for r in rows:
        pr=parent.get((r['word_i'],r['context_i'],r['target_token_id']))
        if pr:
            deltas.append({'delta_nll_T1':r['nll_T1']-pr['nll_T1'],'delta_nll_Tfit':r['nll_Tfit']-pr['nll_Tfit'],'delta_rank':r['target_rank']-pr['target_rank'],'improved_nll_T1':r['nll_T1']<pr['nll_T1'],'improved_nll_Tfit':r['nll_Tfit']<pr['nll_Tfit'],'improved_rank':r['target_rank']<pr['target_rank']})
    by_word=defaultdict(list); p_by_word=defaultdict(list)
    for r in rows: by_word[r['word_i']].append(r)
    for r in (parent_rows or []): p_by_word[r['word_i']].append(r)
    wds=[]
    for wi,g in by_word.items():
        o={'word_i':wi,'target_word':g[0]['target_word'],'mean_nll_T1':mean(r['nll_T1'] for r in g),'mean_nll_Tfit':mean(r['nll_Tfit'] for r in g),'mean_rank':mean(r['target_rank'] for r in g)}
        pg=p_by_word.get(wi)
        if pg:
            o.update({'delta_word_mean_nll_T1':o['mean_nll_T1']-mean(r['nll_T1'] for r in pg),'delta_word_mean_nll_Tfit':o['mean_nll_Tfit']-mean(r['nll_Tfit'] for r in pg),'delta_word_mean_rank':o['mean_rank']-mean(r['target_rank'] for r in pg)})
        wds.append(o)
    return {'n_rows':len(rows),'mean_nll_T1':mean(r['nll_T1'] for r in rows),'mean_nll_Tfit':mean(r['nll_Tfit'] for r in rows),'median_nll_T1':median(r['nll_T1'] for r in rows),'mean_rank':mean(r['target_rank'] for r in rows),'median_rank':median(r['target_rank'] for r in rows),'mean_delta_nll_T1_vs_parent':mean(d['delta_nll_T1'] for d in deltas),'mean_delta_nll_Tfit_vs_parent':mean(d['delta_nll_Tfit'] for d in deltas),'mean_delta_rank_vs_parent':mean(d['delta_rank'] for d in deltas),'median_delta_rank_vs_parent':median(d['delta_rank'] for d in deltas),'improved_fraction_nll_T1':sum(d['improved_nll_T1'] for d in deltas)/len(deltas) if deltas else None,'improved_fraction_nll_Tfit':sum(d['improved_nll_Tfit'] for d in deltas)/len(deltas) if deltas else None,'improved_fraction_rank':sum(d['improved_rank'] for d in deltas)/len(deltas) if deltas else None,'word_mean_delta_nll_T1':mean(w.get('delta_word_mean_nll_T1') for w in wds),'word_mean_delta_nll_Tfit':mean(w.get('delta_word_mean_nll_Tfit') for w in wds),'word_mean_delta_rank':mean(w.get('delta_word_mean_rank') for w in wds),'word_improved_fraction_rank':sum(1 for w in wds if w.get('delta_word_mean_rank') is not None and w['delta_word_mean_rank']<0)/len([w for w in wds if w.get('delta_word_mean_rank') is not None]) if parent_rows else None}

def write_jsonl(path, rows):
    with pathlib.Path(path).open('w',encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--temperature-json', type=pathlib.Path, default=DEFAULT_TEMP)
    ap.add_argument('--out-dir', type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument('--models', nargs='+', default=list(MODEL_PATHS), choices=list(MODEL_PATHS))
    ap.add_argument('--max-words', type=int, default=96)
    ap.add_argument('--sample-seed', type=int, default=86032)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--device', default='cpu', choices=['cpu','cuda','auto'])
    ap.add_argument('--gpu', type=int, default=0)
    args=ap.parse_args(); args.out_dir.mkdir(parents=True,exist_ok=True); os.environ.setdefault('TOKENIZERS_PARALLELISM','false'); ensure_cache(args.out_dir)
    temps=load_temperatures(args.temperature_json)
    words,contexts,subset=make_subset(args.max_words,args.sample_seed)
    plan={'status':'CDI_ENDPOINT_TEMPERATURE_RANK_FIVE_MODEL_PLAN','created_utc':now(),'temperature_json':rel(args.temperature_json),'temperatures':temps,'models':{m:rel(MODEL_PATHS[m]) for m in args.models},'subset':subset,'boundary':'Endpoint-only CDI NLL/rank profile; temperature fitted on ordinary legal-tail text; not official AoA scoring.'}
    (args.out_dir/'plan.json').write_text(json.dumps(plan,indent=2,ensure_ascii=False)+'\n'); print(json.dumps(plan,indent=2),flush=True)
    device=f'cuda:{args.gpu}' if args.device=='cuda' or (args.device=='auto' and torch.cuda.is_available()) else 'cpu'
    infos={}; summaries={}; parent_rows=None
    for name in args.models:
        print(json.dumps({'event':'score_model','model':name,'temperature':temps.get(name,1.0),'device':device}),flush=True)
        rows,info=score_model(name,MODEL_PATHS[name],float(temps.get(name,1.0)),words,contexts,device,args.batch_size)
        write_jsonl(args.out_dir/f'cdi_scores_{name}.jsonl', rows)
        if name=='coherent86': parent_rows=rows
        summaries[name]=summarize(rows,parent_rows); infos[name]=info
        print(json.dumps({'event':'model_done','model':name,**summaries[name]},ensure_ascii=False),flush=True)
    result={'status':'CDI_ENDPOINT_TEMPERATURE_RANK_FIVE_MODEL_DONE','created_utc':now(),'plan':rel(args.out_dir/'plan.json'),'model_infos':infos,'summaries':summaries,'interpretation':'Dense-mask/sparse-label is compared to coherent86, sparse, and dense endpoints on CDI masked-token NLL and ranks. This tests preservation cost at the endpoint only; measured AoA remains the official trajectory score.'}
    out=args.out_dir/'cdi_endpoint_temperature_rank_profile_five_models.json'; out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    lines=['# research CDI endpoint temperature/rank profile including dense-mask/sparse-label\n\n',result['interpretation']+'\n\n']
    for n,s in summaries.items(): lines.append(f"- `{n}`: mean NLL T1={s.get('mean_nll_T1')}, Tfit={s.get('mean_nll_Tfit')}, mean rank={s.get('mean_rank')}, ΔNLL T1={s.get('mean_delta_nll_T1_vs_parent')}, ΔNLL Tfit={s.get('mean_delta_nll_Tfit_vs_parent')}, Δrank={s.get('mean_delta_rank_vs_parent')}, rank improved fraction={s.get('improved_fraction_rank')}\n")
    (args.out_dir/'cdi_endpoint_temperature_rank_profile_five_models.md').write_text(''.join(lines))
    print(json.dumps({'status':result['status'],'out_json':rel(out),'out_md':rel(args.out_dir/'cdi_endpoint_temperature_rank_profile_five_models.md')},indent=2),flush=True)
if __name__=='__main__': main()
