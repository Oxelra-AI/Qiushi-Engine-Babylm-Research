#!/usr/bin/env python3
"""research fast short-format readout on a bounded matched subset."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import argparse, json, os, pathlib, time
from typing import Any
import torch
import torch.nn.functional as F

ROOT=_public_path('.')
MANIFEST=_public_path('experiments/archive/relation_learning/data/short_format_readout/held_short_readout_manifest.json')
OUT=_public_path('experiments/archive/relation_learning/data/short_format_readout_fast')
ENDPOINTS={
 'coherent86_alpha075':'models/frontier',
 'coherent_special_98097_alpha075':'experiments/archive/relation_learning/data/format_replay_corrected/coherent_unsplit_special/seed98097/alpha_0.75',
 'coherent_special_98098_alpha075':'experiments/archive/relation_learning/data/format_replay_corrected/coherent_unsplit_special/seed98098/alpha_0.75',
}

def rel(p):
    try: return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception: return str(p)

def read_json(p): return json.loads(pathlib.Path(p).read_text(encoding='utf-8'))

def load_rows(p, max_rows):
    rows=[]
    with pathlib.Path(p).open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if len(rows)>=max_rows: break
    return rows

def set_private_enabled(model, enabled):
    if hasattr(model,'set_private_enabled'): model.set_private_enabled(bool(enabled))
    elif hasattr(model,'config') and hasattr(model.config,'private_adapter_enabled'): model.config.private_adapter_enabled=bool(enabled)

def collate(tok, encs, device):
    mx=max(e['input_ids'].numel() for e in encs); pad=int(tok.pad_token_id)
    ids=torch.full((len(encs),mx),pad,dtype=torch.long,device=device); att=torch.zeros((len(encs),mx),dtype=torch.long,device=device)
    for i,e in enumerate(encs):
        v=e['input_ids'].view(-1).to(device); a=e['attention_mask'].view(-1).to(device)
        ids[i,:v.numel()]=v; att[i,:a.numel()]=a
    return ids,att

@torch.no_grad()
def measure(model,tok,rows,device,batch_size,add_special):
    total=0.0; ntok=0; maxt=0.0
    for st in range(0,len(rows),batch_size):
        rb=rows[st:st+batch_size]
        encs=[tok(str(o.get('text','')),truncation=True,max_length=256,add_special_tokens=add_special,return_tensors='pt') for o in rb]
        ids,att=collate(tok,encs,device)
        set_private_enabled(model,False); slow=model(input_ids=ids,attention_mask=att).logits.detach()
        set_private_enabled(model,True); priv=model(input_ids=ids,attention_mask=att).logits.detach()
        kl=F.kl_div(F.log_softmax(priv,dim=-1),F.softmax(slow,dim=-1),reduction='none').sum(-1)
        m=att.bool(); total+=float(kl[m].sum().cpu()); ntok+=int(m.sum().item()); maxt=max(maxt,float(kl[m].max().cpu()) if m.any() else 0.0)
        del ids,att,slow,priv,kl
    return {'rows':len(rows),'tokens':ntok,'add_special_tokens':add_special,'kl_mean_per_attention_token':total/max(1,ntok),'kl_max_token':maxt}

def load_model(endpoint, device, cache):
    for k,p in {'HF_HOME':cache/'hf_home','TRANSFORMERS_CACHE':cache/'transformers','HF_MODULES_CACHE':cache/'modules'}.items():
        p.mkdir(parents=True,exist_ok=True); os.environ[k]=str(p)
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(endpoint),trust_remote_code=True,local_files_only=True)
    model=AutoModelForMaskedLM.from_pretrained(str(endpoint),trust_remote_code=True,local_files_only=True).to(device).eval()
    return model,tok

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--device',default='cpu'); ap.add_argument('--rows',type=int,default=64); ap.add_argument('--batch-size',type=int,default=32); args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    man=read_json(MANIFEST)
    coherent=load_rows(ROOT/man['coherent_readout_path'],args.rows)
    short=load_rows(ROOT/man['short_isolated_readout_path'],args.rows)
    device=torch.device(args.device if args.device!='cuda' or torch.cuda.is_available() else 'cpu')
    results={}
    for label,path in ENDPOINTS.items():
        t=time.time(); endpoint=ROOT/path
        if not (endpoint/'model.safetensors').exists():
            results[label]={'status':'missing','endpoint':rel(endpoint)}; continue
        model,tok=load_model(endpoint,device,_public_path('experiments/archive/relation_learning/data/short_format_readout_fast/hf_cache')/label)
        rec={'status':'measured','endpoint':rel(endpoint),'class':type(model).__name__,'private_scale':float(getattr(model.config,'private_adapter_scale',-1.0)),
             'coherent_add_special':measure(model,tok,coherent,device,args.batch_size,True),
             'coherent_no_special':measure(model,tok,coherent,device,args.batch_size,False),
             'short_add_special':measure(model,tok,short,device,args.batch_size,True),
             'elapsed_sec':round(time.time()-t,1)}
        results[label]=rec
        print(json.dumps({'event':'measured','label':label,'coh':rec['coherent_add_special']['kl_mean_per_attention_token'],'short':rec['short_add_special']['kl_mean_per_attention_token']}),flush=True)
        del model
    summary={'status':'SHORT_FORMAT_READOUT_FAST_DONE','rows_per_form':args.rows,'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'source_manifest':rel(MANIFEST),'results':results}
    outj=_public_path('experiments/archive/relation_learning/data/short_format_readout_fast/short_format_readout_fast.json'); outm=_public_path('research/documents/relation_learning/data/short_format_readout_fast/short_format_readout_fast.md')
    outj.write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    lines=['# research fast short-format readout','',f'Rows per form: {args.rows}.','', '| endpoint | scale | coherent+special KL | short+special KL | coherent no-special KL | ratio |','|---|---:|---:|---:|---:|---:|']
    for label,r in results.items():
        if r.get('status')!='measured': continue
        c=r['coherent_add_special']['kl_mean_per_attention_token']; s=r['short_add_special']['kl_mean_per_attention_token']; n=r['coherent_no_special']['kl_mean_per_attention_token']
        lines.append(f"| {label} | {r['private_scale']} | {c:.8f} | {s:.8f} | {n:.8f} | {s/c if c else 0:.3f} |")
    lines.append(''); lines.append(f'JSON: `{rel(outj)}`')
    outm.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':summary['status'],'out_json':rel(outj),'out_md':rel(outm)},indent=2),flush=True)
if __name__=='__main__': main()
