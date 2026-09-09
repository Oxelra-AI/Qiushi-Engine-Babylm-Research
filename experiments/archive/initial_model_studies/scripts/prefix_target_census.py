#!/usr/bin/env python3
"""research no-training prefix-dependent target census.

Purpose: determine whether official BabyLM text contains a usable population of
later MLM targets whose gold-token log-prob is helped by the true earlier prefix,
especially over same-source-near prefixes, under existing strong WWM checkpoints.

This is measurement only: no training, no new objective.
"""
from __future__ import annotations
import argparse, csv, json, math, os, pathlib, random, re, statistics, sys, time
from collections import Counter, defaultdict

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
RAW_DIR = ROOT/'training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/raw_dataset'
RUN = ROOT/'training/runs/wwm_debertav2_8x480_official_1M_b128_seed42_matched'
OUT_ROWS = ROOT/'data/prefix_target_census_rows.csv'
OUT_SUMMARY = ROOT/'data/prefix_target_census_summary.json'
OUT_NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/prefix_target_census.md')

sys.path.insert(0, str((ROOT/'training/scripts').resolve()))
from babylm_masked_train import TRAIN_FILES, iter_examples, Example  # noqa: E402

CKPTS = {
    'wwm42_40M': ROOT/'training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/hf_model/chck_40M',
    'wwm42_80M': ROOT/'training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/hf_model/chck_80M',
    'wwm42_100M': ROOT/'training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/hf_model/chck_100M',
    'wwm43_40M': ROOT/'training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model/chck_40M',
    'wwm43_80M': ROOT/'training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model/chck_80M',
    'wwm43_100M': ROOT/'training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model/chck_100M',
}
STOP=set('the a an and or but if then than as of in on at to from for with without by about into over under is are was were be been being do does did have has had this that these those it its they them he she we you i me my your our their his her not no yes can could would should will may might there here very just also only'.split())
WORD_RE=re.compile(r"^[A-Za-z][A-Za-z'-]{2,}$")
CAP_RE=re.compile(r'^[A-Z][A-Za-z]{2,}$')


def setup_env():
    hf=ROOT/'training/hf_home'
    os.environ['HF_HOME']=str(hf.resolve())
    os.environ['HF_HUB_CACHE']=str((hf/'hub').resolve())
    os.environ['HF_DATASETS_CACHE']=str((hf/'datasets').resolve())
    os.environ['TRANSFORMERS_CACHE']=str((hf/'transformers').resolve())
    os.environ['HF_MODULES_CACHE']=str((ROOT/'training/hf_modules_cache').resolve())
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')


def choose_raw_dir():
    if RAW_DIR.exists(): return RAW_DIR
    return RUN/'raw_dataset'


def reconstruct_exact_1m(words_per_example=160, seed=42):
    raw=RUN/'raw_dataset'
    files=[raw/n for n in TRAIN_FILES]
    pool=list(iter_examples(files,1_000_000,words_per_example))
    for i,e in enumerate(pool): e.example_id=i
    rng=random.Random(seed); rng.shuffle(pool)
    out=[]; actual=0
    for ex in pool:
        if actual>=1_000_000: break
        if actual+ex.words<=1_000_000:
            out.append(ex); actual+=ex.words
        else:
            take=1_000_000-actual
            out.append(Example(' '.join(ex.text.split()[:take]), take, ex.example_id, ex.source)); actual+=take
    if actual!=1_000_000: raise RuntimeError(actual)
    return out


def broad_official_examples(words_per_file=200_000, words_per_example=160):
    raw=choose_raw_dir(); out=[]; eid=10_000_000
    for name in TRAIN_FILES:
        f=raw/name
        if not f.exists(): continue
        examples=list(iter_examples([f], words_per_file, words_per_example))
        for ex in examples:
            ex.example_id=eid; eid+=1
        out.extend(examples)
    return out


def clean_word(w): return re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$",'',w)
def ids(text,tok): return tok(text, add_special_tokens=False)['input_ids']
def good_token(tokstr):
    s=str(tokstr).replace('Ġ','').replace('▁','').strip(); s=clean_word(s)
    return len(s)>=3 and s.lower() not in STOP and WORD_RE.match(s)
def tok_surface(tokstr): return clean_word(str(tokstr).replace('Ġ','').replace('▁','').strip()).lower()
def freq_bucket(count):
    if count<=1: return '1'
    if count<=3: return '2-3'
    if count<=10: return '4-10'
    if count<=30: return '11-30'
    return '31+'

def word_stats(examples):
    c=Counter()
    for ex in examples:
        for w in ex.text.split():
            cw=clean_word(w).lower()
            if cw: c[cw]+=1
    return c

def local_leak(later_toks, idx, surf):
    lo=max(0,idx-10); hi=min(len(later_toks),idx+11)
    for j in range(lo,hi):
        if j==idx: continue
        if tok_surface(later_toks[j])==surf and surf:
            return 1
    return 0

def repeated_cap_feature(prefix_words, later_words):
    p={clean_word(w) for w in prefix_words if CAP_RE.match(clean_word(w))}
    l={clean_word(w) for w in later_words if CAP_RE.match(clean_word(w))}
    return 1 if (p & l) else 0

def make_cases(examples, tok, n, label, rng, wordfreq):
    cands=[]
    for ex in examples:
        words=ex.text.split()
        if len(words)<70: continue
        split=max(30, min(len(words)-25, len(words)//2))
        prefix_words=words[:split]; later_words=words[split:]
        pids=ids(' '.join(prefix_words), tok); lids=ids(' '+' '.join(later_words), tok)
        if len(pids)<20 or len(lids)<20: continue
        pids=pids[-min(len(pids),120):]; lids=lids[:min(len(lids),120)]
        toks=tok.convert_ids_to_tokens(lids)
        possible=[]
        for j in range(max(5,len(lids)//4), min(len(lids),120)):
            if lids[j] in tok.all_special_ids: continue
            if good_token(toks[j]): possible.append(j)
        if not possible: continue
        # deterministic but varied target choice, independent of model response
        j=rng.choice(possible[:min(len(possible),8)])
        surf=tok_surface(toks[j])
        cands.append({
            'case_group': label, 'example_id': ex.example_id, 'source': ex.source,
            'pids': pids, 'lids': lids, 'target_idx': j, 'target_id': lids[j],
            'target_token': toks[j], 'target_surface': surf,
            'prefix_tokens': len(pids), 'later_tokens': len(lids), 'target_frac': j/max(1,len(lids)),
            'target_freq_bucket': freq_bucket(wordfreq.get(surf,0)),
            'target_subword_len_bucket': '1tok',
            'local_leak_10tok': local_leak(toks,j,surf),
            'repeated_capitalized_prefix_suffix': repeated_cap_feature(prefix_words,later_words),
        })
    rng.shuffle(cands)
    cases=[]
    for c in cands:
        same=[o for o in cands if o['example_id']!=c['example_id'] and o['source']==c['source'] and abs(len(o['pids'])-len(c['pids']))<=20]
        cross=[o for o in cands if o['example_id']!=c['example_id'] and o['source']!=c['source'] and abs(len(o['pids'])-len(c['pids']))<=25]
        anyo=[o for o in cands if o['example_id']!=c['example_id']]
        if not anyo: continue
        c=dict(c)
        c['same_pids']=(rng.choice(same if same else anyo)['pids'])[-len(c['pids']):]
        c['cross_pids']=(rng.choice(cross if cross else anyo)['pids'])[-len(c['pids']):]
        cases.append(c)
        if len(cases)>=n: break
    return cases


def block_shuffle(x, rng):
    blocks=[x[i:i+8] for i in range(0,len(x),8)]; rng.shuffle(blocks); return [z for b in blocks for z in b]


def build_variant_sequences(cases, tok):
    rng=random.Random(3167)
    seqs=[]; meta=[]
    for ci,c in enumerate(cases):
        variants={'full':c['pids'], 'deleted':[], 'block_shuffled':block_shuffle(c['pids'],rng), 'same_source_near':c['same_pids'], 'cross_source_near':c['cross_pids']}
        for vn,pids in variants.items():
            seq=(pids+c['lids'])[:256]
            pos=len(pids)+c['target_idx']
            if pos>=len(seq): continue
            seq=list(seq); seq[pos]=tok.mask_token_id
            seqs.append(seq); meta.append((ci,vn,pos,c['target_id']))
    return seqs, meta


def score_model(model, seqs, meta, device, batch_size=96):
    out=[None]*len(meta)
    for st in range(0,len(seqs),batch_size):
        chunk=seqs[st:st+batch_size]; mchunk=meta[st:st+batch_size]
        L=max(len(s) for s in chunk)
        input_ids=torch.full((len(chunk),L), 0, dtype=torch.long, device=device)
        attn=torch.zeros((len(chunk),L), dtype=torch.long, device=device)
        # pad id is 0 for this tokenizer; if not, attention mask excludes it for encoder
        positions=[]; targets=[]
        for i,s in enumerate(chunk):
            input_ids[i,:len(s)]=torch.tensor(s,dtype=torch.long,device=device); attn[i,:len(s)]=1
            positions.append(mchunk[i][2]); targets.append(mchunk[i][3])
        with torch.no_grad():
            logits=model(input_ids=input_ids, attention_mask=attn).logits
            lp=torch.log_softmax(logits, dim=-1)
        for i,(ci,vn,pos,tid) in enumerate(mchunk):
            out[st+i]=float(lp[i,pos,tid].item())
    return out


def aggregate(rows):
    thresholds=[0.02,0.05,0.10,0.20]
    exposures=['40M','80M','100M']
    summary={}
    strata=['case_group','source','target_freq_bucket','local_leak_10tok','repeated_capitalized_prefix_suffix']
    for exp in exposures:
        key=f'stable_full_minus_same_{exp}'
        vals=[float(r[key]) for r in rows if r.get(key) not in ('','nan',None)]
        summary[exp]={'n':len(vals),'mean_stable_full_minus_same':sum(vals)/len(vals) if vals else None,'median':statistics.median(vals) if vals else None}
        for th in thresholds:
            pos=[v for v in vals if v>=th]; neg=[v for v in vals if v<=-th]
            summary[exp][f'frac_pos_ge_{th}']=len(pos)/len(vals) if vals else None
            summary[exp][f'frac_neg_le_-{th}']=len(neg)/len(vals) if vals else None
        summary[exp]['by_strata']={}
        for s in strata:
            d=defaultdict(list)
            for r in rows:
                if r.get(key) not in ('','nan',None): d[str(r[s])].append(float(r[key]))
            summary[exp]['by_strata'][s]={}
            for sv,vs in sorted(d.items(), key=lambda kv: kv[0]):
                entry={'n':len(vs),'mean':sum(vs)/len(vs),'median':statistics.median(vs)}
                for th in thresholds:
                    entry[f'frac_pos_ge_{th}']=sum(v>=th for v in vs)/len(vs)
                summary[exp]['by_strata'][s][sv]=entry
    return summary


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--n_exact',type=int,default=300)
    ap.add_argument('--n_broad',type=int,default=300)
    ap.add_argument('--batch_size',type=int,default=96)
    args=ap.parse_args()
    setup_env(); t0=time.time()
    tok=AutoTokenizer.from_pretrained(str(CKPTS['wwm42_40M'].resolve()), use_fast=True)
    exact=reconstruct_exact_1m(); broad=broad_official_examples()
    wf=word_stats(exact+broad)
    rng=random.Random(316)
    cases=make_cases(exact,tok,args.n_exact,'exact_1m_slice',rng,wf)+make_cases(broad,tok,args.n_broad,'broad_official',rng,wf)
    if not cases: raise RuntimeError('no cases')
    seqs,meta=build_variant_sequences(cases,tok)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # rows indexed by case
    rows=[]
    for i,c in enumerate(cases):
        row={k:v for k,v in c.items() if k not in ['pids','lids','same_pids','cross_pids']}
        row['case_id']=i; rows.append(row)
    for mname,ckpt in CKPTS.items():
        print(json.dumps({'event':'load_model','model':mname,'ckpt':str(ckpt)}), flush=True)
        model=AutoModelForMaskedLM.from_pretrained(str(ckpt.resolve()), trust_remote_code=True).to(device).eval()
        vals=score_model(model,seqs,meta,device,args.batch_size)
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        # collect variant scores
        scores=defaultdict(dict)
        for val,(ci,vn,pos,tid) in zip(vals,meta): scores[ci][vn]=val
        for ci in range(len(cases)):
            sc=scores[ci]
            if 'full' not in sc: continue
            for vn,val in sc.items(): rows[ci][f'{mname}_lp_{vn}']=val
            for vn in ['deleted','block_shuffled','same_source_near','cross_source_near']:
                rows[ci][f'{mname}_full_minus_{vn}']=sc['full']-sc[vn] if vn in sc else ''
            if 'same_source_near' in sc and 'cross_source_near' in sc:
                rows[ci][f'{mname}_same_vs_cross_spec']=(sc['full']-sc['same_source_near'])-(sc['full']-sc['cross_source_near'])
    # stable seed42/43 same-source deltas by exposure: require same sign, else 0 with signed min magnitude? record both.
    for r in rows:
        for exp in ['40M','80M','100M']:
            a=r.get(f'wwm42_{exp}_full_minus_same_source_near','')
            b=r.get(f'wwm43_{exp}_full_minus_same_source_near','')
            if a=='' or b=='':
                r[f'stable_full_minus_same_{exp}']=''
            elif a*b>0:
                r[f'stable_full_minus_same_{exp}']=math.copysign(min(abs(float(a)),abs(float(b))), float(a)+float(b))
            else:
                r[f'stable_full_minus_same_{exp}']=0.0
            c=r.get(f'wwm42_{exp}_full_minus_deleted','')
            d=r.get(f'wwm43_{exp}_full_minus_deleted','')
            if c!='' and d!='' and c*d>0:
                r[f'stable_full_minus_deleted_{exp}']=math.copysign(min(abs(float(c)),abs(float(d))), float(c)+float(d))
            elif c!='' and d!='': r[f'stable_full_minus_deleted_{exp}']=0.0
            else: r[f'stable_full_minus_deleted_{exp}']=''
    OUT_ROWS.parent.mkdir(parents=True,exist_ok=True)
    # stable field order
    base_fields=list(rows[0].keys())
    extra=[]
    for r in rows:
        for k in r:
            if k not in base_fields and k not in extra: extra.append(k)
    fields=base_fields+extra
    with OUT_ROWS.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    summary=aggregate(rows)
    payload={'status':'PREFIX_TARGET_CENSUS','n_cases':len(rows),'n_exact':args.n_exact,'n_broad':args.n_broad,'checkpoints':{k:str(v) for k,v in CKPTS.items()},'variants':['full','deleted','block_shuffled','same_source_near','cross_source_near'],'rows_csv':str(OUT_ROWS),'summary':summary,'elapsed_sec':time.time()-t0}
    OUT_SUMMARY.parent.mkdir(parents=True,exist_ok=True); OUT_SUMMARY.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    lines=['# research prefix-dependent target census','',f'Rows: `{OUT_ROWS}`',f'Summary: `{OUT_SUMMARY}`','',f'Cases: {len(rows)} ({args.n_exact} exact 1M target, {args.n_broad} broad official target requested).','','## Sign-stable true-prefix over same-source-near fractions','', '| exposure | n | mean stable Δ | median | ≥0.02 | ≥0.05 | ≥0.10 | ≥0.20 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for exp in ['40M','80M','100M']:
        s=summary[exp]
        lines.append(f"| {exp} | {s['n']} | {s['mean_stable_full_minus_same']:+.5f} | {s['median']:+.5f} | {s['frac_pos_ge_0.02']:.4f} | {s['frac_pos_ge_0.05']:.4f} | {s['frac_pos_ge_0.1']:.4f} | {s['frac_pos_ge_0.2']:.4f} |")
    lines += ['','## Initial interpretation prompt','','A usable cross-sentence objective target population would require a nontrivial fraction of targets where `full - same_source_near` is positive with the same sign across seed42 and seed43 at the same exposure. Cross-source differences alone do not count. Inspect rows before using any target set for training.']
    OUT_NOTE.parent.mkdir(parents=True,exist_ok=True); OUT_NOTE.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'out':str(OUT_SUMMARY),'note':str(OUT_NOTE),'n_cases':len(rows),'summary':summary,'elapsed_sec':payload['elapsed_sec']},indent=2)[:6000])

if __name__=='__main__': main()
