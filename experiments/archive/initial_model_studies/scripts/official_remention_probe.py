#!/usr/bin/env python3
"""research official-text remention-state probe.

Scientific purpose:
Measure whether existing BabyLM checkpoints contain a transferable state variable
for "this entity/content word has already been introduced in prior context" on
legal official BabyLM text. This is a short evidence layer before any new
intermediate-state-supervision training.

Design:
- Extract repeated capitalized tokens and repeated content nouns/words from the
  official corpus raw_dataset.
- Build examples at mention positions: label 0 = first mention, label 1 = later
  mention in the same document/window.
- Lexical split by mention word: train and test words are disjoint, so a probe
  cannot solve by memorizing particular names/words.
- Compare full-history snippets to local-only snippets. A genuine context-state
  signal should be stronger in full history than local-only.
- Train small ridge/logistic probes on frozen hidden states at layers 2/4/6/8.

This does not train or modify the LM. It tests the representation premise behind
official-text intermediate-state supervision and relational-progress mechanisms.
"""
from __future__ import annotations
import argparse, collections, json, math, os, pathlib, random, re, sys, time
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, PreTrainedTokenizerFast

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
RAW_DEFAULT = ROOT/'training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/raw_dataset'
OUT_DEFAULT = ROOT/'data/official_remention_probe.json'
NOTE_DEFAULT = (ROOT.parents[2] / 'research/notes/initial_model_studies/official_remention_probe.md')
STOP = set('''the a an and or but if then else when while of in on at by for with without from to into onto over under as is are was were be been being do did done have has had i you he she it we they me him her us them my your his its our their this that these those there here not no yes can could would should may might will shall one two three four new old first last good bad little big small great mr mrs miss sir said says say like just very more most some any many much other another'''.split())
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]{2,}")
SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

@dataclass
class Case:
    word: str
    label: int
    source: str
    full_text: str
    full_start: int
    full_end: int
    local_text: str
    local_start: int
    local_end: int
    distance_words: int


def setup_env():
    hf=ROOT/'training/hf_home'
    os.environ['HF_HOME']=str(hf.resolve())
    os.environ['HF_HUB_CACHE']=str((hf/'hub').resolve())
    os.environ['TRANSFORMERS_CACHE']=str((hf/'transformers').resolve())
    os.environ['HF_MODULES_CACHE']=str((ROOT/'training/hf_modules_cache').resolve())
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')


def norm_word(w: str) -> str:
    return w.strip("'\".,!?;:()[]{}“”‘’").lower()


def candidate_word(w: str) -> bool:
    n=norm_word(w)
    if len(n)<4 or n in STOP: return False
    if not re.match(r"^[a-z][a-z'\-]+$", n): return False
    return True


def iter_docs(raw_dir: pathlib.Path, max_docs: int) -> Iterable[tuple[str,str]]:
    files=['gutenberg.train.txt','simple_wiki.train.txt','childes.train.txt','open_subtitles.train.txt','bnc_spoken.train.txt','switchboard.train.txt']
    n=0
    for fn in files:
        p=raw_dir/fn
        if not p.exists(): continue
        buf=[]
        with p.open('r',encoding='utf-8',errors='replace') as f:
            for line in f:
                s=line.strip()
                if s:
                    buf.append(s)
                    # keep docs bounded to avoid giant Gutenberg chunks
                    if sum(len(x.split()) for x in buf) >= 180:
                        yield fn, ' '.join(buf); n+=1; buf=[]
                        if n>=max_docs: return
                elif buf:
                    yield fn, ' '.join(buf); n+=1; buf=[]
                    if n>=max_docs: return
            if buf:
                yield fn, ' '.join(buf); n+=1
                if n>=max_docs: return


def word_offsets(text: str):
    return [(m.group(0), m.start(), m.end()) for m in WORD_RE.finditer(text)]


def build_snippet(tokens, mention_i: int, left: int, right: int) -> tuple[str,int,int]:
    lo=max(0, mention_i-left); hi=min(len(tokens), mention_i+right+1)
    base_start=tokens[lo][1]; base_end=tokens[hi-1][2]
    text_slice=tokens[lo][3][base_start:base_end]
    start=tokens[mention_i][1]-base_start; end=tokens[mention_i][2]-base_start
    return text_slice, start, end


def extract_cases(raw_dir: pathlib.Path, max_docs: int, max_cases: int, seed: int) -> list[Case]:
    rng=random.Random(seed)
    cases=[]
    # Store tokens as (surface,start,end,doc_text,norm)
    for source, doc in iter_docs(raw_dir, max_docs):
        toks=[]
        for surf,s,e in word_offsets(doc):
            nw=norm_word(surf)
            if candidate_word(surf):
                toks.append((surf,s,e,doc,nw))
        if len(toks)<20: continue
        seen={}
        # include first and repeated mentions for words with at least two occurrences
        counts=collections.Counter(t[-1] for t in toks)
        for i,t in enumerate(toks):
            nw=t[-1]
            if counts[nw] < 2: continue
            prev=seen.get(nw)
            label=0 if prev is None else 1
            if label==1:
                dist=i-prev
                if dist < 8:  # require non-local prior context
                    seen[nw]=i; continue
            else:
                dist=0
            full_text, fs, fe = build_snippet(toks, i, left=80, right=16)
            local_text, ls, le = build_snippet(toks, i, left=6, right=6)
            if len(full_text.split()) < 12 or len(local_text.split()) < 5:
                seen[nw]=i; continue
            cases.append(Case(nw,label,source,full_text,fs,fe,local_text,ls,le,dist))
            seen[nw]=i
    # balance per word and label approximately
    by_label={0:[],1:[]}
    for c in cases: by_label[c.label].append(c)
    rng.shuffle(by_label[0]); rng.shuffle(by_label[1])
    m=min(len(by_label[0]),len(by_label[1]),max_cases//2)
    out=by_label[0][:m]+by_label[1][:m]
    rng.shuffle(out)
    return out


def lexical_split(cases: list[Case], seed: int, test_frac: float=0.3):
    words=sorted(set(c.word for c in cases))
    rng=random.Random(seed); rng.shuffle(words)
    ntest=max(1,int(len(words)*test_frac)); test_words=set(words[:ntest]); train_words=set(words[ntest:])
    train=[c for c in cases if c.word in train_words]
    test=[c for c in cases if c.word in test_words]
    # rebalance labels inside split
    def rebalance(rows):
        a=[x for x in rows if x.label==0]; b=[x for x in rows if x.label==1]
        m=min(len(a),len(b)); return a[:m]+b[:m]
    return rebalance(train), rebalance(test), {'train_words':len(train_words),'test_words':len(test_words)}


def token_positions(tokenizer, text: str, start: int, end: int, max_len: int):
    enc=tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_len, return_offsets_mapping=True, return_tensors='pt')
    offs=enc.pop('offset_mapping')[0].tolist()
    pos=[i for i,(s,e) in enumerate(offs) if e>start and s<end]
    if not pos:
        # fallback nearest non-empty token
        pos=[min(range(len(offs)), key=lambda i: abs((offs[i][0]+offs[i][1])/2 - (start+end)/2))] if offs else [0]
    return enc, pos


def extract_features(model_path: pathlib.Path, cases: list[Case], context: str, layers: list[int], batch_size: int, max_len: int, device: str):
    tok=AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    model=AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True).to(device)
    model.eval()
    feats={l:[] for l in layers}; labels=[]; metas=[]
    with torch.no_grad():
        # batching variable-length tokenization manually
        for c in cases:
            if context=='full': text,start,end=c.full_text,c.full_start,c.full_end
            else: text,start,end=c.local_text,c.local_start,c.local_end
            enc,pos=token_positions(tok,text,start,end,max_len)
            enc={k:v.to(device) for k,v in enc.items()}
            out=model(**enc, output_hidden_states=True)
            for l in layers:
                h=out.hidden_states[l][0,pos,:].mean(dim=0).detach().cpu().numpy().astype('float32')
                feats[l].append(h)
            labels.append(c.label)
            metas.append({'word':c.word,'label':c.label,'source':c.source,'distance_words':c.distance_words})
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return {l:np.stack(feats[l]) for l in layers}, np.asarray(labels,dtype=np.int64), metas


def train_eval_probe(Xtr, ytr, Xte, yte, seed: int):
    # standardize using train statistics, then closed-form ridge regression to two logits
    mu=Xtr.mean(axis=0,keepdims=True); sd=Xtr.std(axis=0,keepdims=True)+1e-6
    Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd
    Xtr=np.concatenate([Xtr, np.ones((Xtr.shape[0],1),dtype=Xtr.dtype)],axis=1)
    Xte=np.concatenate([Xte, np.ones((Xte.shape[0],1),dtype=Xte.dtype)],axis=1)
    Y=np.zeros((len(ytr),2),dtype='float32'); Y[np.arange(len(ytr)),ytr]=1.0
    lam=10.0
    A=Xtr.T@Xtr + lam*np.eye(Xtr.shape[1],dtype='float32')
    W=np.linalg.solve(A, Xtr.T@Y)
    pred=(Xte@W).argmax(axis=1)
    acc=float((pred==yte).mean()*100.0)
    # balanced acc
    b=[]
    for cls in [0,1]:
        m=(yte==cls)
        if m.any(): b.append(float((pred[m]==yte[m]).mean()*100.0))
    return {'accuracy':acc,'balanced_accuracy':float(sum(b)/len(b)),'n_train':int(len(ytr)),'n_test':int(len(yte)),'test_label_counts':dict(collections.Counter(map(int,yte)))}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--raw_dir', default=str(RAW_DEFAULT))
    ap.add_argument('--max_docs', type=int, default=2500)
    ap.add_argument('--max_cases', type=int, default=1200)
    ap.add_argument('--max_len', type=int, default=128)
    ap.add_argument('--layers', nargs='+', type=int, default=[2,4,6,8])
    ap.add_argument('--seed', type=int, default=271)
    ap.add_argument('--out_json', default=str(OUT_DEFAULT))
    ap.add_argument('--out_note', default=str(NOTE_DEFAULT))
    ap.add_argument('--models', nargs='*', default=[
        'wwm43_40M=experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model/chck_40M',
        'wwm43_80M=experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model/chck_80M',
        'wwm43_100M=experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model/chck_100M',
        'amlm43_40M=experiments/archive/initial_model_studies/training/runs/fullcycle_official_amlm_debertav2_8x480_seed43_100M_b256/hf_model/chck_40M',
        'amlm43_100M=experiments/archive/initial_model_studies/training/runs/fullcycle_official_amlm_debertav2_8x480_seed43_100M_b256/hf_model/chck_100M',
    ])
    args=ap.parse_args()
    setup_env(); t0=time.time(); device='cuda' if torch.cuda.is_available() else 'cpu'
    cases=extract_cases(pathlib.Path(args.raw_dir), args.max_docs, args.max_cases, args.seed)
    train,test,split=lexical_split(cases,args.seed)
    if len(train)<50 or len(test)<30:
        raise RuntimeError(f'not enough split cases: train={len(train)} test={len(test)} cases={len(cases)} split={split}')
    model_specs=[]
    for m in args.models:
        name,path=m.split('=',1); model_specs.append((name,pathlib.Path(path)))
    payload={'status':'OFFICIAL_REMENTION_PROBE','task':'lexical-split first-vs-remention on official text','raw_dir':str(args.raw_dir),'num_cases':len(cases),'num_train':len(train),'num_test':len(test),'split':split,'layers':args.layers,'models':{},'elapsed_sec':None}
    for name,path in model_specs:
        if not path.exists(): raise FileNotFoundError(path)
        payload['models'][name]={}
        for ctx in ['local','full']:
            X,y,_=extract_features(path, train+test, ctx, args.layers, batch_size=1, max_len=args.max_len, device=device)
            ytr=y[:len(train)]; yte=y[len(train):]
            payload['models'][name][ctx]={}
            for l in args.layers:
                res=train_eval_probe(X[l][:len(train)],ytr,X[l][len(train):],yte,args.seed)
                payload['models'][name][ctx][f'layer_{l}']=res
        # compute best history gain
        best=[]
        for l in args.layers:
            lf=payload['models'][name]['full'][f'layer_{l}']['balanced_accuracy']
            ll=payload['models'][name]['local'][f'layer_{l}']['balanced_accuracy']
            best.append({'layer':l,'full_bal_acc':lf,'local_bal_acc':ll,'history_gain':lf-ll})
        payload['models'][name]['history_gain_by_layer']=best
        payload['models'][name]['best_full']=max(best,key=lambda r:r['full_bal_acc'])
        payload['models'][name]['best_gain']=max(best,key=lambda r:r['history_gain'])
        pathlib.Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out_json).write_text(json.dumps(payload,indent=2)+'\n')
    payload['elapsed_sec']=time.time()-t0
    pathlib.Path(args.out_json).write_text(json.dumps(payload,indent=2)+'\n')
    # note
    lines=['# research official-text remention-state probe','',f'Evidence JSON: `{args.out_json}`','',f'Cases: {len(cases)}; train {len(train)}; test {len(test)}; lexical split {split}','', '| model | best full layer | full bal acc | local bal acc | history gain | best gain layer | best gain |','|---|---:|---:|---:|---:|---:|---:|']
    for name,d in payload['models'].items():
        bf=d['best_full']; bg=d['best_gain']
        lines.append(f"| {name} | {bf['layer']} | {bf['full_bal_acc']:.2f} | {bf['local_bal_acc']:.2f} | {bf['history_gain']:+.2f} | {bg['layer']} | {bg['history_gain']:+.2f} |")
    lines += ['', 'Interpretation: a transferable cross-sentence remention-state signal should appear as full-history balanced accuracy above local-only under lexical split. If history gain is near zero or negative, ordinary frozen checkpoints do not contain an easily recoverable official-text state variable for this target.']
    pathlib.Path(args.out_note).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out_note).write_text('\n'.join(lines)+'\n')
    print(json.dumps({'out_json':args.out_json,'out_note':args.out_note,'num_cases':len(cases),'num_train':len(train),'num_test':len(test),'elapsed_sec':payload['elapsed_sec']},indent=2))

if __name__=='__main__': main()
