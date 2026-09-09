#!/usr/bin/env python3
from __future__ import annotations
import pathlib, random, re, sys
from transformers import AutoTokenizer
ROOT=pathlib.Path('experiments/archive/initial_model_studies')
sys.path.insert(0,str((ROOT/'training/scripts').resolve()))
from babylm_masked_train import TRAIN_FILES, iter_examples, Example
raw=ROOT/'training/runs/wwm_debertav2_8x480_official_1M_b128_seed42_matched/raw_dataset'
files=[raw/n for n in TRAIN_FILES]
print('files', [(f.name,f.exists(),f.stat().st_size if f.exists() else None) for f in files])
pool=list(iter_examples(files, 1000000, 160))
print('pool', len(pool), 'words sum', sum(e.words for e in pool), 'minmax', min(e.words for e in pool), max(e.words for e in pool))
for e in pool[:5]: print('EX', e.source, e.words, repr(e.text[:300]))
rng=random.Random(42); rng.shuffle(pool)
sel=[]; actual=0
for ex in pool:
    if actual>=1000000: break
    if actual+ex.words<=1000000:
        sel.append(ex); actual+=ex.words
    else:
        take=1000000-actual
        sel.append(Example(' '.join(ex.text.split()[:take]), take, ex.example_id, ex.source)); actual+=take
print('sel', len(sel), actual, 'minmax', min(e.words for e in sel), max(e.words for e in sel))
tok=AutoTokenizer.from_pretrained(str((ROOT/'training/runs/wwm_debertav2_8x480_official_1M_b128_seed42_matched/hf_model/chck_1M').resolve()), use_fast=True)
STOP=set('the a an and or but if then than as of in on at to from for with without by about into over under is are was were be been being do does did have has had this that these those it its they them he she we you i me my your our their his her not no yes can could would should will may might there here very just also only'.split())
WORD_RE=re.compile(r"^[A-Za-z][A-Za-z'-]{2,}$")
counts={'len>=110':0,'prefixlater':0,'content_examples':0,'single_examples':0,'content_tokens':0,'single_tokens':0}
examples=[]
for ex in sel[:2000]:
    words=ex.text.split()
    if len(words)>=110: counts['len>=110']+=1
    split=min(80,max(50,len(words)//2)); prefix=words[:split]; later=words[split:]
    if len(prefix)>=45 and len(later)>=35: counts['prefixlater']+=1
    sc=0; cc=0
    for w0 in later:
        w=re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$",'',w0)
        if WORD_RE.match(w) and w.lower() not in STOP:
            cc+=1
            ids=tok(' '+w, add_special_tokens=False)['input_ids']
            if len(ids)==1: sc+=1
    counts['content_tokens']+=cc; counts['single_tokens']+=sc
    if cc: counts['content_examples']+=1
    if sc:
        counts['single_examples']+=1
        if len(examples)<10: examples.append((ex.source, len(words), ' '.join(later[:40]), sc, cc))
print('counts first2000', counts)
for x in examples: print('CAND', x)
