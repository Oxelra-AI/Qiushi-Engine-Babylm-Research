#!/usr/bin/env python3
from __future__ import annotations
import pathlib, random, sys, collections
ROOT=pathlib.Path('experiments/archive/initial_model_studies')
sys.path.insert(0,str((ROOT/'training/scripts').resolve()))
from babylm_masked_train import TRAIN_FILES, iter_examples, Example
raw=ROOT/'training/runs/wwm_debertav2_8x480_official_1M_b128_seed42_matched/raw_dataset'
files=[raw/n for n in TRAIN_FILES]
pool=list(iter_examples(files,1_000_000,160))
for i,e in enumerate(pool): e.example_id=i
rng=random.Random(42); rng.shuffle(pool)
sel=[]; actual=0
for ex in pool:
    if actual>=1_000_000: break
    if actual+ex.words<=1_000_000:
        sel.append(ex); actual+=ex.words
    else:
        take=1_000_000-actual; sel.append(Example(' '.join(ex.text.split()[:take]), take, ex.example_id, ex.source)); actual+=take
print('n_examples',len(sel),'words',actual,'source_counts',collections.Counter(e.source for e in sel))
bs=128
mixed=0; only=collections.Counter(); counts=[]
for bi in range((len(sel)+bs-1)//bs):
    b=sel[bi*bs:(bi+1)*bs]
    c=collections.Counter(e.source for e in b)
    counts.append(dict(c))
    if len(c)>1: mixed+=1
    else: only.update(c.keys())
print('batches',len(counts),'mixed',mixed,'only',dict(only))
print('first10',counts[:10])
print('last5',counts[-5:])
