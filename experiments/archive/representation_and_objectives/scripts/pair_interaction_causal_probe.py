#!/usr/bin/env python3
"""research paired-interaction causal probe.

This is the second, stricter Route-2 probe.  It repairs the first research pilot,
which decoded affected-query identity but not an entity-bound final-state update.

Matched factorial object:
  * Two entities have the same contradictory prior state.
  * One entity is acted on by a state-changing verb.
  * For each queried entity we compare the hidden state when that same queried
    entity is affected vs when the other entity is affected.  This holds the
    query entity fixed and swaps only the action argument; action/state tokens and
    token count are held fixed.

Representation test:
  * Paired difference Δh = h(query fixed, query affected) - h(query fixed,
    other affected) is decoded for event/result polarity with state/action family
    held out.

Causal test:
  * A pre-fixed paired-update direction from train families is patched at layers
    4/5/6 into held-out-family affected-query cases.  Random and swapped-sign
    directions are matched controls.  No weights are updated.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse, csv, json, math, os, random, statistics, time
from collections import defaultdict
from pathlib import Path
from typing import Any

USER_ROOT = _public_path('.')
A01_WS = _public_path('experiments/archive/representation_and_objectives')
A02_WS = _public_path('experiments/archive/frontier_consolidation')
OUT_ROOT = _public_path('experiments/archive/representation_and_objectives/data/pair_interaction_causal_probe')
CACHE = _public_path('experiments/archive/representation_and_objectives/data/pair_interaction_causal_probe/hf_cache') / os.environ.get("CACHE_SUFFIX", "default")
for name, path in {"HOME": CACHE/"home", "XDG_CACHE_HOME": CACHE/"xdg", "HF_HOME": CACHE/"hf_home", "HF_MODULES_CACHE": CACHE/"hf_modules", "TRANSFORMERS_CACHE": CACHE/"transformers", "TORCH_HOME": CACHE/"torch", "TMPDIR": CACHE/"tmp"}.items():
    os.environ.setdefault(name, str(path)); Path(os.environ[name]).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForMaskedLM, AutoTokenizer

MODELS = {
    "chck82_scale1p75": _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M'),
    "legal16k_base100": _public_path('experiments/archive/representation_and_objectives/training/runs/strictsmalltok_compact_view_reinvest_seed43022/hf_model/chck_100M'),
    "scale1p75_100M": _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_100M'),
}
FAMILIES = [
    {"family":"open_closed", "state0":"open", "state1":"closed", "event0":"opened", "event1":"closed", "train_group":"train"},
    {"family":"empty_full", "state0":"empty", "state1":"full", "event0":"emptied", "event1":"filled", "train_group":"train"},
    {"family":"wet_dry", "state0":"wet", "state1":"dry", "event0":"wetted", "event1":"dried", "train_group":"train"},
    {"family":"clean_dirty", "state0":"clean", "state1":"dirty", "event0":"cleaned", "event1":"dirtied", "train_group":"heldout_family"},
    {"family":"warm_cool", "state0":"warm", "state1":"cool", "event0":"warmed", "event1":"cooled", "train_group":"heldout_family"},
    {"family":"loose_tight", "state0":"loose", "state1":"tight", "event0":"loosened", "event1":"tightened", "train_group":"heldout_family"},
]
OBJECT_PAIRS = [("box","cup"),("bag","bottle"),("door","drawer"),("window","jar"),("pot","pan")]
ACTORS = ["Kate","Jack","Anna","Emma","Mary","Seth","Amy","Tom"]


def rel(p: Path | str) -> str:
    try: return str(Path(p).resolve().relative_to(USER_ROOT))
    except Exception: return str(p)

def qstats(vals):
    xs=[]
    for v in vals:
        try:
            fv=float(v)
            if math.isfinite(fv): xs.append(fv)
        except Exception: pass
    xs.sort()
    if not xs: return {"n":0}
    def q(p):
        if len(xs)==1: return xs[0]
        idx=p*(len(xs)-1); lo=math.floor(idx); hi=math.ceil(idx)
        return xs[lo] if lo==hi else xs[lo]*(hi-idx)+xs[hi]*(idx-lo)
    return {"n":len(xs),"mean":statistics.fmean(xs),"median":statistics.median(xs),"p10":q(.1),"p25":q(.25),"p75":q(.75),"p90":q(.9),"min":xs[0],"max":xs[-1]}

def render(context, query_template, target):
    pre, post = query_template.split("{target}",1)
    lead=(context.strip()+" ") if context.strip() else ""
    sent=lead+pre+target+post
    return sent,(len(lead)+len(pre), len(lead)+len(pre)+len(target))

def mask_target(tok, sent, span):
    enc=tok(sent, return_offsets_mapping=True, return_tensors=None)
    ids=list(enc["input_ids"]); att=list(enc["attention_mask"]); offs=list(enc["offset_mapping"])
    s,e=span
    sel=[i for i,(a,b) in enumerate(offs) if not (a==b==0) and b>s and a<e]
    if len(sel)!=1: return None
    pos=sel[0]; tid=int(ids[pos]); ids[pos]=int(tok.mask_token_id)
    return ids,att,pos,tid

def single_id(tok, word):
    sent,span=render("", "The item is now {target}.", word)
    prep=mask_target(tok, sent, span)
    return None if prep is None else prep[3]

def token_count(tok, word):
    return len(tok.encode(" "+word, add_special_tokens=False))

def build_frames(tok):
    checks={"states":{}, "objects":{}, "events":{}}
    usable=[]
    for fam in FAMILIES:
        ok=True
        for s in [fam["state0"], fam["state1"]]:
            sid=single_id(tok,s); checks["states"][s]=sid; ok = ok and sid is not None
        for e in [fam["event0"], fam["event1"]]: checks["events"][e]=token_count(tok,e)
        if ok: usable.append(fam)
    for a,b in OBJECT_PAIRS:
        checks["objects"][a]=token_count(tok,a); checks["objects"][b]=token_count(tok,b)
    frames=[]; counter=0
    for fam in usable:
        for pair_i,(oa,ob) in enumerate(OBJECT_PAIRS):
            for actor in ACTORS:
                for ep in [0,1]:
                    event=fam[f"event{ep}"]; result=fam[f"state{ep}"]; prior=fam[f"state{1-ep}"]
                    for affected_slot, affected_obj in [("A",oa),("B",ob)]:
                        ctx=f"The {oa} was {prior}. The {ob} was {prior}. {actor} {event} the {affected_obj}."
                        for query_slot, query_obj in [("A",oa),("B",ob)]:
                            affected_query = int(query_slot==affected_slot)
                            final_pol = ep if affected_query else 1-ep
                            correct=fam[f"state{final_pol}"]; foil=fam[f"state{1-final_pol}"]
                            counter+=1
                            frames.append({"case_id":f"P196_{counter:05d}", "family":fam["family"], "train_group":fam["train_group"], "actor":actor,
                                           "obj_pair":f"{oa}/{ob}", "obj_a":oa, "obj_b":ob, "event_pol":ep, "event":event,
                                           "prior_state":prior, "result_state":result, "affected_slot":affected_slot, "query_slot":query_slot,
                                           "affected_query":affected_query, "final_pol":final_pol, "query_obj":query_obj, "affected_obj":affected_obj,
                                           "context":ctx, "query_template":f"The {query_obj} is now {{target}}.", "correct":correct, "foil":foil,
                                           "quad_key":f"{fam['family']}|{pair_i}|{actor}|{ep}", "query_key":f"{fam['family']}|{pair_i}|{actor}|{ep}|{query_slot}",
                                           "context_key":f"{fam['family']}|{pair_i}|{actor}|{ep}|{affected_slot}"})
    return frames, checks

@torch.inference_mode()
def collect(model,tok,device,frames,batch_size):
    n_layers=int(model.config.num_hidden_layers)+1
    examples=[]
    fam_by={f["family"]:f for f in FAMILIES}
    for i,r in enumerate(frames):
        sent,span=render(r["context"], r["query_template"], r["correct"])
        # Hidden state is independent of which one-token target is rendered before masking.
        for key,word in [("correct",r["correct"]),("foil",r["foil"]),("state0",fam_by[r["family"]]["state0"]),("state1",fam_by[r["family"]]["state1"] )]:
            sent2,span2=render(r["context"], r["query_template"], word)
            prep=mask_target(tok,sent2,span2)
            if prep is None: continue
            ids,att,pos,tid=prep
            examples.append({"i":i,"key":key,"ids":ids,"att":att,"pos":pos,"tid":tid,"length":len(ids)})
    examples.sort(key=lambda x:x["length"])
    pad=tok.pad_token_id if tok.pad_token_id is not None else 0
    reps={}; scores={}
    for st in range(0,len(examples),batch_size):
        batch=examples[st:st+batch_size]; ml=max(x["length"] for x in batch)
        ids=torch.tensor([x["ids"]+[pad]*(ml-x["length"]) for x in batch],dtype=torch.long,device=device)
        att=torch.tensor([x["att"]+[0]*(ml-x["length"]) for x in batch],dtype=torch.long,device=device)
        pos=torch.tensor([x["pos"] for x in batch],dtype=torch.long,device=device)
        tid=torch.tensor([x["tid"] for x in batch],dtype=torch.long,device=device)
        out=model(input_ids=ids, attention_mask=att, output_hidden_states=True, return_dict=True)
        mb=torch.arange(ids.shape[0],device=device)
        lp=torch.log_softmax(out.logits[mb,pos].float(),dim=-1).gather(-1,tid.unsqueeze(-1)).squeeze(-1)
        for bi,ex in enumerate(batch):
            scores[(ex["i"],ex["key"])] = float(lp[bi].cpu())
            if ex["key"]=="correct" and ex["i"] not in reps:
                reps[ex["i"]]=np.stack([out.hidden_states[li][bi,ex["pos"],:].float().cpu().numpy() for li in range(n_layers)],axis=0)
    rows=[]
    for i,r in enumerate(frames):
        if i not in reps: continue
        c=scores.get((i,"correct"),float('nan')); f=scores.get((i,"foil"),float('nan'))
        s0=scores.get((i,"state0"),float('nan')); s1=scores.get((i,"state1"),float('nan'))
        rr=dict(r); rr["hidden"]=reps[i]; rr["margin_correct_foil"]=c-f; rr["state0_minus_state1"]=s0-s1; rr["head_correct"]=bool(c>f)
        rows.append(rr)
    return rows,n_layers

def by_key(rows,key):
    return {r[key]:r for r in rows}

def make_pair_diffs(rows, layer):
    # Same queried entity, compare context where it is affected to context where other entity is affected.
    qgroups=defaultdict(dict)
    for r in rows:
        qgroups[r["query_key"]][r["affected_query"]]=r
    out=[]
    for qk,d in qgroups.items():
        if 0 not in d or 1 not in d: continue
        aff, unaff=d[1], d[0]
        dh=aff["hidden"][layer]-unaff["hidden"][layer]
        out.append({"kind":"same_query_contextdiff", "family":aff["family"], "train_group":aff["train_group"], "event_pol":aff["event_pol"], "query_slot":aff["query_slot"], "obj_pair":aff["obj_pair"], "actor":aff["actor"], "layer":layer, "diff":dh})
    # Same context, compare affected query to unaffected query.
    cgroups=defaultdict(dict)
    for r in rows:
        cgroups[r["context_key"]][r["affected_query"]]=r
    for ck,d in cgroups.items():
        if 0 not in d or 1 not in d: continue
        aff, unaff=d[1], d[0]
        dh=aff["hidden"][layer]-unaff["hidden"][layer]
        out.append({"kind":"same_context_querydiff", "family":aff["family"], "train_group":aff["train_group"], "event_pol":aff["event_pol"], "query_slot":aff["query_slot"], "obj_pair":aff["obj_pair"], "actor":aff["actor"], "layer":layer, "diff":dh})
    return out

def lr_acc(Xtr,ytr,Xte,yte,seed=0,permute=False):
    y=ytr.copy()
    if permute:
        rng=np.random.RandomState(seed); rng.shuffle(y)
    if len(np.unique(y))<2: return float('nan'), None, None, None
    mu=Xtr.mean(0); sd=Xtr.std(0)+1e-6
    clf=LogisticRegression(max_iter=2000,C=0.5,random_state=seed).fit((Xtr-mu)/sd,y)
    pred=clf.predict((Xte-mu)/sd)
    return float((pred==yte).mean()), clf, mu, sd

def pair_decoder_panel(rows,n_layers):
    res={}
    for kind in ["same_query_contextdiff","same_context_querydiff"]:
        layer_rows=[]
        for layer in range(n_layers):
            pds=[p for p in make_pair_diffs(rows,layer) if p["kind"]==kind]
            train=[p for p in pds if p["train_group"]=="train"]
            test=[p for p in pds if p["train_group"]=="heldout_family"]
            Xtr=np.stack([p["diff"] for p in train]); ytr=np.array([p["event_pol"] for p in train])
            Xte=np.stack([p["diff"] for p in test]); yte=np.array([p["event_pol"] for p in test])
            acc,_,_,_=lr_acc(Xtr,ytr,Xte,yte,seed=196+layer)
            pacc,_,_,_=lr_acc(Xtr,ytr,Xte,yte,seed=9196+layer,permute=True)
            layer_rows.append({"layer":layer,"acc_event_pol_heldout_family":acc,"perm_null":pacc,"n_train":len(train),"n_test":len(test)})
        best=max(layer_rows,key=lambda r:r["acc_event_pol_heldout_family"] if math.isfinite(r["acc_event_pol_heldout_family"]) else -1)
        res[kind]={"layers":layer_rows,"best_layer":best["layer"],"best_acc":best["acc_event_pol_heldout_family"]}
    return res

def component_from_pairdiff(rows, layer, kind="same_query_contextdiff"):
    pds=[p for p in make_pair_diffs(rows,layer) if p["kind"]==kind and p["train_group"]=="train"]
    X=np.stack([p["diff"] for p in pds]); y=np.array([p["event_pol"] for p in pds])
    m1=X[y==1].mean(0); m0=X[y==0].mean(0); w=m1-m0; w=w/(np.linalg.norm(w)+1e-12)
    proj=X@w; amp=0.5*abs(float(np.median(proj[y==1])-np.median(proj[y==0])))
    if not math.isfinite(amp) or amp<1e-6: amp=float(np.std(proj)+1e-6)
    rng=np.random.RandomState(19650+layer); rw=rng.normal(size=w.shape); rw=rw/(np.linalg.norm(rw)+1e-12)
    return {"layer":layer,"kind":kind,"w":w.astype('float32'),"random_w":rw.astype('float32'),"amp":amp,"train_n":len(pds),"proj_event1":qstats(proj[y==1]),"proj_event0":qstats(proj[y==0])}

def make_examples(tok, frames):
    fam_by={f["family"]:f for f in FAMILIES}
    exs=[]
    for i,r in enumerate(frames):
        for key,word in [("correct",r["correct"]),("foil",r["foil"]),("state0",fam_by[r["family"]]["state0"]),("state1",fam_by[r["family"]]["state1"] )]:
            sent,span=render(r["context"],r["query_template"],word); prep=mask_target(tok,sent,span)
            if prep is None: continue
            ids,att,pos,tid=prep; exs.append({"i":i,"key":key,"ids":ids,"att":att,"pos":pos,"tid":tid,"length":len(ids)})
    exs.sort(key=lambda x:x["length"]); return exs

@torch.inference_mode()
def hook_score(model,tok,device,frames,comp,mode,alpha,batch_size,eval_filter):
    eval_frames=[r for r in frames if eval_filter(r)]
    exs=make_examples(tok,eval_frames); pad=tok.pad_token_id if tok.pad_token_id is not None else 0
    layer=int(comp["layer"]); module_idx=layer-1
    w=torch.tensor(comp["w"],dtype=torch.float32,device=device); rw=torch.tensor(comp["random_w"],dtype=torch.float32,device=device); amp=float(comp["amp"])*alpha
    active={}
    def hook(_module,_inp,out):
        h=out[0] if isinstance(out,tuple) else out; hh=h.clone(); mb=torch.arange(h.shape[0],device=h.device); pos=active["pos"]; sign=active["sign"]
        hp=hh[mb,pos,:].float()
        if mode=="none": newp=hp
        elif mode=="add": newp=hp+(amp*sign).unsqueeze(-1)*w
        elif mode=="add_swapped": newp=hp-(amp*sign).unsqueeze(-1)*w
        elif mode=="add_random": newp=hp+(amp*sign).unsqueeze(-1)*rw
        elif mode=="remove":
            coeff=(hp@w).unsqueeze(-1); newp=hp-coeff*w
        elif mode=="remove_random":
            coeff=(hp@rw).unsqueeze(-1); newp=hp-coeff*rw
        else: raise ValueError(mode)
        hh[mb,pos,:]=newp.to(h.dtype)
        return (hh,)+tuple(out[1:]) if isinstance(out,tuple) else hh
    handle=model.deberta.encoder.layer[module_idx].register_forward_hook(hook)
    scores={}
    try:
        for st in range(0,len(exs),batch_size):
            batch=exs[st:st+batch_size]; ml=max(x["length"] for x in batch)
            ids=torch.tensor([x["ids"]+[pad]*(ml-x["length"]) for x in batch],dtype=torch.long,device=device)
            att=torch.tensor([x["att"]+[0]*(ml-x["length"]) for x in batch],dtype=torch.long,device=device)
            pos=torch.tensor([x["pos"] for x in batch],dtype=torch.long,device=device)
            tid=torch.tensor([x["tid"] for x in batch],dtype=torch.long,device=device)
            signs=[]
            for ex in batch:
                r=eval_frames[ex["i"]]
                # Only affected-query cases should receive update-to-result polarity; unaffected queries are left neutral.
                if int(r["affected_query"])==1:
                    signs.append(1.0 if int(r["event_pol"])==1 else -1.0)
                else:
                    signs.append(0.0)
            active["pos"]=pos; active["sign"]=torch.tensor(signs,dtype=torch.float32,device=device)
            out=model(input_ids=ids, attention_mask=att, return_dict=True)
            mb=torch.arange(ids.shape[0],device=device)
            lp=torch.log_softmax(out.logits[mb,pos].float(),dim=-1).gather(-1,tid.unsqueeze(-1)).squeeze(-1)
            for ex,val in zip(batch,lp.detach().cpu().tolist()): scores[(ex["i"],ex["key"])] = float(val)
    finally:
        handle.remove()
    rows=[]
    for i,r in enumerate(eval_frames):
        c=scores.get((i,"correct"),float('nan')); f=scores.get((i,"foil"),float('nan'))
        s0=scores.get((i,"state0"),float('nan')); s1=scores.get((i,"state1"),float('nan'))
        rows.append({"case_id":r["case_id"],"family":r["family"],"train_group":r["train_group"],"event_pol":r["event_pol"],"affected_query":r["affected_query"],"mode":mode,"layer":layer,"alpha":alpha,"margin_correct_foil":c-f,"state0_minus_state1":s0-s1,"head_correct":bool(c>f)})
    return rows

def summarize(rows):
    out={"n":len(rows),"accuracy":sum(1 for r in rows if r["head_correct"])/len(rows) if rows else None,"margin":qstats(r["margin_correct_foil"] for r in rows)}
    for f in ["family","affected_query","event_pol"]:
        dd=defaultdict(list)
        for r in rows: dd[str(r[f])].append(r)
        out["by_"+f]={k:{"n":len(v),"accuracy":sum(1 for x in v if x["head_correct"])/len(v),"margin_mean":statistics.fmean(x["margin_correct_foil"] for x in v)} for k,v in sorted(dd.items())}
    return out

def write_csv(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows: path.write_text("",encoding="utf-8"); return
    keys=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen: seen.add(k); keys.append(k)
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore'); w.writeheader(); w.writerows(rows)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',choices=list(MODELS),default='chck82_scale1p75'); ap.add_argument('--device',default='cuda:0'); ap.add_argument('--batch-size',type=int,default=64); ap.add_argument('--alpha',type=float,default=1.0)
    args=ap.parse_args(); device=torch.device(args.device if torch.cuda.is_available() else 'cpu')
    out_dir=OUT_ROOT/args.model; out_dir.mkdir(parents=True,exist_ok=True)
    print(json.dumps({"event":"load_model","model":args.model,"path":rel(MODELS[args.model]),"device":str(device)}),flush=True)
    tok=AutoTokenizer.from_pretrained(str(MODELS[args.model]),trust_remote_code=True,use_fast=True)
    model=AutoModelForMaskedLM.from_pretrained(str(MODELS[args.model]),trust_remote_code=True).to(device).eval()
    frames, checks=build_frames(tok)
    (out_dir/'pair_interaction_frames.jsonl').write_text(''.join(json.dumps(r,sort_keys=True,ensure_ascii=False)+'\n' for r in frames),encoding='utf-8')
    print(json.dumps({"event":"frames","n":len(frames),"checks":checks,"frames":rel(out_dir/'pair_interaction_frames.jsonl')}),flush=True)
    rows,n_layers=collect(model,tok,device,frames,args.batch_size)
    slim=[{k:v for k,v in r.items() if k!='hidden'} for r in rows]
    write_csv(out_dir/'baseline_pair_rows.csv',slim)
    baseline=summarize(slim)
    pair_panel=pair_decoder_panel(rows,n_layers)
    intervention_rows=[]; intervention_summaries={}
    for layer in [4,5,6]:
        if layer>=n_layers: continue
        comp=component_from_pairdiff(rows,layer,kind='same_query_contextdiff')
        comp_summary={k:v for k,v in comp.items() if k not in {'w','random_w'}}
        for mode in ['none','remove','remove_random','add','add_random','add_swapped']:
            irows=hook_score(model,tok,device,frames,comp,mode,args.alpha,args.batch_size,eval_filter=lambda r: r['train_group']=='heldout_family')
            intervention_rows.extend(irows); key=f'L{layer}_{mode}'
            intervention_summaries[key]={"component":comp_summary,"heldout_family":summarize(irows)}
            print(json.dumps({"event":"intervention","model":args.model,"layer":layer,"mode":mode,"acc":intervention_summaries[key]['heldout_family']['accuracy'],"margin_mean":intervention_summaries[key]['heldout_family']['margin'].get('mean')}),flush=True)
    write_csv(out_dir/'pair_intervention_rows.csv',intervention_rows)
    for kind,info in pair_panel.items():
        print(json.dumps({"event":"pair_decoder","model":args.model,"kind":kind,"best_layer":info['best_layer'],"best_acc":round(info['best_acc'],4),"curve":[round(x['acc_event_pol_heldout_family'],3) for x in info['layers']]}),flush=True)
    summary={"status":"PAIR_INTERACTION_CAUSAL_PROBE","created_utc":time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),"model":args.model,"model_path":rel(MODELS[args.model]),"n_layers":n_layers,"checks":checks,"n_frames":len(frames),"baseline_summary":baseline,"pair_decoder_panel":pair_panel,"intervention_summaries":intervention_summaries,"files":{"frames":rel(out_dir/'pair_interaction_frames.jsonl'),"baseline_rows":rel(out_dir/'baseline_pair_rows.csv'),"intervention_rows":rel(out_dir/'pair_intervention_rows.csv')},"scientific_scope":"Forward-only synthetic factorial causal test; no weights changed; no official labels used."}
    out_json=out_dir/'pair_interaction_causal_probe.json'; out_json.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({"status":summary['status'],"out_json":rel(out_json)},indent=2),flush=True)

if __name__=='__main__': main()
