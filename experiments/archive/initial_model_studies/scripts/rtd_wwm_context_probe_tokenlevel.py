#!/usr/bin/env python3
from __future__ import annotations
import csv, json, math, os, pathlib, random, re, statistics, sys, time
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
ROOT=pathlib.Path('experiments/archive/initial_model_studies')
WWM_RUN=ROOT/'training/runs/wwm_debertav2_8x480_official_1M_b128_seed42_matched'
RTD_RUN=ROOT/'training/runs/hybrid_rtd_mlm_debertav2_8x480_official_1M_b128_seed42'
WWM_CKPT=WWM_RUN/'hf_model/chck_1M'; RTD_CKPT=RTD_RUN/'hf_model/chck_1M'
OUT_JSON=ROOT/'data/rtd_wwm_context_probe_tokenlevel.json'
OUT_CSV=ROOT/'data/context_probe_tokenlevel_rows.csv'
OUT_NOTE=(ROOT.parents[2] / 'research/notes/initial_model_studies/rtd_wwm_context_probe_tokenlevel.md')
sys.path.insert(0,str((ROOT/'training/scripts').resolve()))
from babylm_masked_train import TRAIN_FILES, iter_examples, Example
STOP=set('the a an and or but if then than as of in on at to from for with without by about into over under is are was were be been being do does did have has had this that these those it its they them he she we you i me my your our their his her not no yes can could would should will may might there here very just also only'.split())

def setup_env():
    hf=ROOT/'training/hf_home'
    os.environ['HF_HOME']=str(hf.resolve()); os.environ['HF_HUB_CACHE']=str((hf/'hub').resolve())
    os.environ['HF_DATASETS_CACHE']=str((hf/'datasets').resolve()); os.environ['TRANSFORMERS_CACHE']=str((hf/'transformers').resolve())
    os.environ['HF_MODULES_CACHE']=str((ROOT/'training/hf_modules_cache').resolve()); os.environ.setdefault('TOKENIZERS_PARALLELISM','false')

def reconstruct_examples():
    raw=WWM_RUN/'raw_dataset'; files=[raw/n for n in TRAIN_FILES]
    pool=list(iter_examples(files, 1_000_000, 160))
    for i,e in enumerate(pool): e.example_id=i
    rng=random.Random(42); rng.shuffle(pool); out=[]; actual=0
    for ex in pool:
        if actual>=1_000_000: break
        if actual+ex.words<=1_000_000: out.append(ex); actual+=ex.words
        else:
            take=1_000_000-actual; out.append(Example(' '.join(ex.text.split()[:take]), take, ex.example_id, ex.source)); actual+=take
    if actual!=1_000_000: raise RuntimeError(actual)
    return out

def ids(text, tok): return tok(text, add_special_tokens=False)['input_ids']
def good_token(t):
    s=str(t).replace('Ġ','').replace('▁','').strip()
    s=re.sub(r'^[^A-Za-z]+|[^A-Za-z]+$','',s)
    return len(s)>=3 and s.lower() not in STOP and re.match(r'^[A-Za-z][A-Za-z\'-]*$',s)

def make_cases(examples,tok,n=240):
    rng=random.Random(306); cands=[]
    for ex in examples:
        words=ex.text.split();
        if len(words)<70: continue
        split=max(30, min(len(words)-25, len(words)//2))
        ptxt=' '.join(words[:split]); ltxt=' '.join(words[split:])
        pids=ids(ptxt,tok); lids=ids(' '+ltxt,tok)
        if len(pids)<20 or len(lids)<20: continue
        start=max(5,len(lids)//4); target=None
        toks=tok.convert_ids_to_tokens(lids)
        for j in range(start, min(len(lids),120)):
            if lids[j] in tok.all_special_ids: continue
            if good_token(toks[j]): target=j; break
        if target is None: continue
        # keep total well below max length, preserving whole later suffix when possible
        pids=pids[-min(len(pids), 120):]; lids=lids[:min(len(lids), 120)]
        if target>=len(lids): continue
        cands.append({'id':ex.example_id,'source':ex.source,'pids':pids,'lids':lids,'target_idx':target,'target_id':lids[target],'token':toks[target]})
    if len(cands)<30: raise RuntimeError(f'too few token candidates {len(cands)}')
    rng.shuffle(cands); cases=[]
    for c in cands:
        opts=[o for o in cands if o['id']!=c['id'] and abs(len(o['pids'])-len(c['pids']))<=20]
        if not opts: opts=[o for o in cands if o['id']!=c['id']]
        u=rng.choice(opts); c=dict(c); c['upids']=u['pids'][-len(c['pids']):]; cases.append(c)
        if len(cases)>=n: break
    return cases

def block_shuffle(x,rng):
    blocks=[x[i:i+8] for i in range(0,len(x),8)]; rng.shuffle(blocks); return [z for b in blocks for z in b]
def logp(model, pids, lids, target_idx, target_id, tok, device):
    seq=(pids+lids)[:256]; pos=len(pids)+target_idx
    if pos>=len(seq): return float('nan')
    seq=list(seq); seq[pos]=tok.mask_token_id
    input_ids=torch.tensor([seq],device=device); attn=torch.ones_like(input_ids)
    with torch.no_grad(): return float(torch.log_softmax(model(input_ids=input_ids,attention_mask=attn).logits[0,pos],dim=-1)[target_id].item())
def med(xs): return float(statistics.median(xs)) if xs else float('nan')
def mean(xs): return float(sum(xs)/len(xs)) if xs else float('nan')
def pf(xs): return float(sum(v>0 for v in xs)/len(xs)) if xs else float('nan')
def main():
    t0=time.time(); setup_env(); tok=AutoTokenizer.from_pretrained(str(WWM_CKPT.resolve()),use_fast=True)
    examples=reconstruct_examples(); cases=make_cases(examples,tok,240); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models={'wwm':AutoModelForMaskedLM.from_pretrained(str(WWM_CKPT.resolve()),trust_remote_code=True).to(device).eval(), 'rtd_mlm':AutoModelForMaskedLM.from_pretrained(str(RTD_CKPT.resolve()),trust_remote_code=True).to(device).eval()}
    rng=random.Random(3061); rows=[]
    for i,c in enumerate(cases):
        variants={'full':c['pids'],'deleted':[],'shuffled':block_shuffle(c['pids'],rng),'unrelated':c['upids']}
        r={'case_id':i,'source':c['source'],'target_token':c['token'],'prefix_tokens':len(c['pids']),'later_tokens':len(c['lids']),'target_idx':c['target_idx']}
        for mn,m in models.items():
            vals={vn:logp(m,p,c['lids'],c['target_idx'],c['target_id'],tok,device) for vn,p in variants.items()}
            for vn,v in vals.items(): r[f'{mn}_lp_{vn}']=v
            r[f'{mn}_delta_deleted']=vals['full']-vals['deleted']; r[f'{mn}_delta_shuffled']=vals['full']-vals['shuffled']; r[f'{mn}_delta_unrelated']=vals['full']-vals['unrelated']
            r[f'{mn}_spec_deleted_vs_unrelated']=r[f'{mn}_delta_deleted']-r[f'{mn}_delta_unrelated']; r[f'{mn}_spec_shuffled_vs_unrelated']=r[f'{mn}_delta_shuffled']-r[f'{mn}_delta_unrelated']
        for k in ['delta_deleted','delta_shuffled','delta_unrelated','spec_deleted_vs_unrelated','spec_shuffled_vs_unrelated']:
            r[f'rtd_minus_wwm_{k}']=r[f'rtd_mlm_{k}']-r[f'wwm_{k}']
        rows.append(r)
    OUT_CSV.parent.mkdir(parents=True,exist_ok=True)
    with OUT_CSV.open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    keys=[k for k in rows[0] if k.startswith(('wwm_delta','rtd_mlm_delta','rtd_minus_wwm_delta','wwm_spec','rtd_mlm_spec','rtd_minus_wwm_spec'))]
    agg={}
    for k in keys:
        vals=[r[k] for r in rows if not math.isnan(r[k])]; agg[k]={'mean':mean(vals),'median':med(vals),'pos_fraction':pf(vals),'n':len(vals)}
    fast={'rtd_minus_wwm':{'blimp_fast':0.17,'supplement_fast':-2.80,'ewok_fast':2.18,'entity_tracking_fast':-1.39,'comps':0.13,'Reading_mean':0.045}}
    rtdm=json.loads((RTD_RUN/'scientific_metrics.json').read_text())
    payload={'status':'TOKENLEVEL_CONTEXT_PROBE','design':'Exact research examples, token-level fixed later MLM target; identical full/deleted/block-shuffled/unrelated prefix interventions applied to matched WWM and RTD+MLM; route-relevant quantities are rtd_minus_wwm deltas and specificity relative to unrelated prefix.','n_cases':len(rows),'rows_csv':str(OUT_CSV),'aggregate':agg,'fast_profile_from_step305':fast,'rtd_training':{k:rtdm.get(k) for k in ['rtd_replaced_recall_last','rtd_above_majority_last','rtd_pred_original_rate_last','rtd_label_original_rate_last']},'example_rows':rows[:5],'elapsed_sec':time.time()-t0}
    OUT_JSON.parent.mkdir(parents=True,exist_ok=True); OUT_JSON.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    lines=['# research token-level repaired RTD-vs-WWM context probe','',f'Evidence JSON: `{OUT_JSON}`',f'Rows CSV: `{OUT_CSV}`','','Fast profile from completed research direct evaluation: RTD-WWM BLiMP +0.17, Supplement -2.80, EWoK +2.18, Entity -1.39, COMPS +0.13, Reading +0.045.','','RTD shortcut metrics at 1M: replaced recall %.4f, above-majority %.4f, pred-original %.4f, label-original %.4f.'%(payload['rtd_training']['rtd_replaced_recall_last'],payload['rtd_training']['rtd_above_majority_last'],payload['rtd_training']['rtd_pred_original_rate_last'],payload['rtd_training']['rtd_label_original_rate_last']),'','| metric | mean | median | pos frac | n |','|---|---:|---:|---:|---:|']
    for k,v in agg.items(): lines.append(f"| {k} | {v['mean']:+.5f} | {v['median']:+.5f} | {v['pos_fraction']:.3f} | {v['n']} |")
    OUT_NOTE.parent.mkdir(parents=True,exist_ok=True); OUT_NOTE.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'out':str(OUT_JSON),'note':str(OUT_NOTE),'n_cases':len(rows),'key':{k:agg[k] for k in agg if k.startswith('rtd_minus_wwm')},'elapsed_sec':payload['elapsed_sec']},indent=2))
if __name__=='__main__': main()
