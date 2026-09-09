#!/usr/bin/env python3
"""research: construct and measure source-attested continuity bridges.

CPU/file-only preflight. Starting from the exact extractive_balanced selector,
construct fixed-word-count source subsequences with the same per-pair content-word
count while increasing source adjacency. The optimizer also softly matches the
natural compact view's source-span and copied lexical anchors.

Outputs pair-level views/measurements and blinded proposition-preservation samples.
No training, evaluation, upload, AoA, or leaderboard action.
"""
import argparse
import hashlib
import json
import math
import random
import re
import statistics
import time
from collections import Counter
from pathlib import Path

STOPWORDS = {
    "a","an","the","and","or","but","if","then","else","so","because","as","than","to","of","in","on","for","with","without","by","from","at","into","onto","over","under","about","between","among","through","during","before","after","above","below","is","am","are","was","were","be","been","being","do","does","did","done","doing","have","has","had","having","can","could","may","might","must","shall","should","will","would","this","that","these","those","there","here","it","its","they","them","their","theirs","he","him","his","she","her","hers","we","us","our","ours","you","your","yours","i","me","my","mine","who","whom","whose","which","what","where","when","why","how","not","no","nor","only","just","also","very","more","most","less","least","much","many","some","any","all","each","every","other","another","such","own","same","too","again","still","already","yet","up","down","out","off","back","away","within","across","per","via","using","used","use","uses","become","became"
}
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['\u2019-][A-Za-z0-9]+)?")


def words(text): return [x for x in text.split() if x]
def norm(w):
    p = WORD_RE.findall(w)
    return "".join(p).lower() if p else ""
def content(n): return bool(n) and n not in STOPWORDS and (n.isdigit() or len(n) >= 4)
def now(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def evenly(indices,k):
    if k<=0:return []
    if k>=len(indices):return list(indices)
    step=len(indices)/k
    out=[]
    for j in range(k): out.append(indices[min(int(j*step+step/2),len(indices)-1)])
    out=sorted(set(out)); rem=[i for i in indices if i not in set(out)]
    while len(out)<k and rem: out.append(rem.pop(0));out.sort()
    return out[:k]

def balanced_indices(src,V,target_frac=.6474):
    ns=[norm(w) for w in src]; flags=[content(n) for n in ns]
    ci=[i for i,x in enumerate(flags) if x]; fi=[i for i,x in enumerate(flags) if not x]
    tc=min(len(ci),max(1,round(target_frac*V))); tf=V-tc
    if tf>len(fi):tf=len(fi);tc=V-tf
    if tc>len(ci):tc=len(ci);tf=V-tc
    kept=sorted(set(evenly(ci,tc)+evenly(fi,tf)))
    while len(kept)>V:
        cand=next((i for i in kept if not flags[i]),kept[0]);kept.remove(cand)
    while len(kept)<V:
        cand=max(i for i in range(len(src)) if i not in set(kept));kept.append(cand);kept.sort()
    return kept

def metrics_for_indices(src,idx,compact_norms):
    N=len(src); V=len(idx); S=set(idx); ns=[norm(w) for w in src]
    adj=sum(1 for a,b in zip(idx,idx[1:]) if b==a+1)
    runs=V-adj if V else 0
    span=(idx[-1]-idx[0]+1)/N if idx and N else 0
    c=sum(content(ns[i]) for i in idx)
    anchors=sum(1 for i in idx if ns[i] and ns[i] in compact_norms)
    return {"gap1":adj/max(V-1,1),"skip":1-adj/max(V-1,1),"runs":runs,
            "mean_run":V/max(runs,1),"span":span,"content_fraction":c/max(V,1),
            "compact_anchor_fraction":anchors/max(V,1)}

def objective(src,idx,compact_norms,target_gap,target_span=.8072):
    m=metrics_for_indices(src,idx,compact_norms)
    # Primary: continuity dose. Secondary: compact-like source span and copied anchors.
    return -12*abs(m["gap1"]-target_gap)-2.5*abs(m["span"]-target_span)+0.20*m["compact_anchor_fraction"]

def optimize(src,base,compact_norms,target_gap):
    idx=sorted(base); flags=[content(norm(w)) for w in src]
    score=objective(src,idx,compact_norms,target_gap)
    # Deterministic best-improvement swaps, preserving exact content count.
    for _ in range(40):
        S=set(idx); best=None
        for r in idx:
            for a in range(len(src)):
                if a in S or flags[a]!=flags[r]: continue
                cand=sorted((S-{r})|{a})
                sc=objective(src,cand,compact_norms,target_gap)
                key=(sc,-a,-r)
                if sc>score+1e-12 and (best is None or key>best[0]): best=(key,cand,sc)
        if best is None: break
        idx,score=best[1],best[2]
    return idx

def absent_content(src_text,view_text):
    bag=Counter(norm(w) for w in words(src_text) if content(norm(w)))
    absent=0
    for w in words(view_text):
        n=norm(w)
        if not content(n):continue
        if bag[n]>0:bag[n]-=1
        else:absent+=1
    return absent

def stat(xs):
    xs=sorted(xs)
    return {"n":len(xs),"mean":statistics.mean(xs),"median":statistics.median(xs),
            "p10":xs[int(.1*(len(xs)-1))],"p90":xs[int(.9*(len(xs)-1))]}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default="experiments/archive/frontier_consolidation")
    ap.add_argument("--out-dir",default="experiments/archive/frontier_consolidation/data/source_attested_continuity_bridge")
    ap.add_argument("--sample-n",type=int,default=240)
    args=ap.parse_args(); ws=Path(args.root); out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    pair_path=ws/"data/factorial_view_candidate_audit/compact_candidate_pairs.jsonl"
    tok_path=ws/"data/compliant_tokenizer"
    pairs=[json.loads(x) for x in open(pair_path)]
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(tok_path))
    variants={"balanced":[],"bridge_mid":[],"bridge_compactgap":[],"bridge_high":[],"compact":[]}
    targets={"bridge_mid":.66,"bridge_compactgap":.778,"bridge_high":.88}
    rows=[]
    for p in pairs:
        src=words(p["source_text"]); V=len(words(p["view_text"])); cn={norm(w) for w in words(p["view_text"]) if norm(w)}
        base=balanced_indices(src,V)
        idxs={"balanced":base}
        for name,t in targets.items():idxs[name]=optimize(src,base,cn,t)
        rec={"pair_id":p["pair_id"],"source":p["source_text"],"compact":p["view_text"],"view_words":V,"variants":{}}
        for name,idx in idxs.items():
            text=" ".join(src[i] for i in idx); m=metrics_for_indices(src,idx,cn)
            m.update({"active_tokens":len(tok.encode(text,add_special_tokens=False)),"absent_content":absent_content(p["source_text"],text)})
            rec["variants"][name]={"text":text,"indices":idx,"metrics":m};variants[name].append(m)
        compact_text=p["view_text"]
        cm={"active_tokens":len(tok.encode(compact_text,add_special_tokens=False)),"absent_content":absent_content(p["source_text"],compact_text),
            "content_fraction":sum(content(norm(w)) for w in words(compact_text))/max(V,1)}
        variants["compact"].append(cm);rows.append(rec)
    pair_out=out/"source_attested_continuity_pairs.jsonl"
    with open(pair_out,"w") as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+"\n")
    summary={}
    for name,ms in variants.items():
        keys=sorted(set.intersection(*(set(m) for m in ms)))
        summary[name]={k:stat([float(m[k]) for m in ms]) for k in keys if isinstance(ms[0].get(k),(int,float))}
        summary[name]["total_active_tokens"]=sum(m["active_tokens"] for m in ms)
        summary[name]["total_absent_content"]=sum(m["absent_content"] for m in ms)
    # Duplicate/template risk on view strings.
    risk={}
    for name in ["balanced"]+list(targets):
        texts=[r["variants"][name]["text"] for r in rows]
        c=Counter(texts)
        risk[name]={"unique_fraction":len(c)/len(texts),"duplicate_rows":sum(v-1 for v in c.values()),
                    "equals_source":sum(t==r["source"] for t,r in zip(texts,rows)),
                    "equals_compact":sum(t==r["compact"] for t,r in zip(texts,rows))}
    # High-repair sampling: largest balanced->compactgap continuity increase, stratified by compact absent count.
    ranked=sorted(rows,key=lambda r:(r["variants"]["bridge_compactgap"]["metrics"]["gap1"]-r["variants"]["balanced"]["metrics"]["gap1"],
                                      absent_content(r["source"],r["compact"])),reverse=True)
    bins=[[],[],[]]
    for r in ranked:
        a=absent_content(r["source"],r["compact"]);bins[0 if a==0 else 1 if a<=2 else 2].append(r)
    rng=random.Random(23343022); sample=[]
    each=args.sample_n//3
    for b in bins:
        pool=b[:max(each*4,each)];rng.shuffle(pool);sample.extend(pool[:each])
    rng.shuffle(sample)
    blind_path=out/"blinded_proposition_preservation_sample.jsonl"
    with open(blind_path,"w") as f:
        for j,r in enumerate(sample):
            opts=[("A",r["variants"]["balanced"]["text"]),("B",r["variants"]["bridge_compactgap"]["text"]),("C",r["compact"])]
            rr=random.Random(int(hashlib.sha256((r["pair_id"]+"233").encode()).hexdigest()[:16],16));rr.shuffle(opts)
            f.write(json.dumps({"sample_id":j,"pair_id":r["pair_id"],"source":r["source"],"options":dict(opts),
              "ratings":{"A":{"proposition_0_2":None,"fluency_0_2":None},"B":{"proposition_0_2":None,"fluency_0_2":None},"C":{"proposition_0_2":None,"fluency_0_2":None}}},ensure_ascii=False)+"\n")
    result={"status":"SOURCE_ATTESTED_CONTINUITY_BRIDGE_PREFLIGHT","created_utc":now(),"pairs":len(rows),
            "construction":"Exact source-word subsequences in source order; exact compact view word count; exact extractive_balanced per-pair content count; deterministic same-content-class swaps optimize continuity dose, compact-like source span, and compact lexical anchors.",
            "targets":targets,"summary":summary,"duplicate_template_risk":risk,
            "pair_file":str(pair_out),"pair_file_sha256":sha(pair_out),"blinded_sample":str(blind_path),"blinded_sample_n":len(sample),
            "promotion_requirements":{"zero_absent_content":True,"continuity_above_extractives":True,"compact_like_active_tokens":True,"human_proposition_preservation_needed":True},
            "no_training_eval_upload_aoa_or_leaderboard":True}
    with open(out/"source_attested_continuity_bridge_preflight.json","w") as f:json.dump(result,f,indent=2)
    lines=["# research source-attested continuity bridge preflight","",f"Pairs: {len(rows)}",""]
    lines.append("| variant | gap1 | skip | runs | span | content frac | active tokens | absent content |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name in variants:
        s=summary[name]
        lines.append(f"| {name} | {s.get('gap1',{}).get('mean',float('nan')):.3f} | {s.get('skip',{}).get('mean',float('nan')):.3f} | {s.get('runs',{}).get('mean',float('nan')):.2f} | {s.get('span',{}).get('mean',float('nan')):.3f} | {s['content_fraction']['mean']:.3f} | {s['total_active_tokens']} | {s['total_absent_content']} |")
    lines += ["","All bridge words are exact source words in source order. Human proposition/fluency ratings remain unfilled; this file does not authorize training."]
    (out/"source_attested_continuity_bridge_preflight.md").write_text("\n".join(lines))
    print(json.dumps({"status":result["status"],"out":str(out),"pairs":len(rows),"sample":len(sample)},indent=2))
if __name__=="__main__":main()
