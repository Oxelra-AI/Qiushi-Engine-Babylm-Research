#!/usr/bin/env python3
"""research v2 prefix-dependent target census with fixed suffix positions.

Repairs v1 artifact: deleted/replacement prefixes changed target absolute position.
Here every variant has the same prefix length P as the true prefix, so the suffix
and masked target remain at identical absolute positions. Deleted prefix is pad
ids with attention=0; same/cross prefixes are padded/truncated to length P.
Also restrict targets to word-start alphabetic tokens to avoid BPE-fragment cases.
"""
from __future__ import annotations
import argparse, csv, json, math, os, pathlib, random, re, statistics, sys, time
from collections import Counter, defaultdict
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
ROOT=pathlib.Path('experiments/archive/initial_model_studies')
RUN=ROOT/'training/runs/wwm_debertav2_8x480_official_1M_b128_seed42_matched'
RAW_FALLBACK=ROOT/'training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/raw_dataset'
OUT_ROWS=ROOT/'data/prefix_target_census_v2_fixedpos_rows.csv'
OUT_SUMMARY=ROOT/'data/prefix_target_census_v2_fixedpos_summary.json'
OUT_NOTE=(ROOT.parents[2] / 'research/notes/initial_model_studies/prefix_target_census_v2_fixedpos.md')
sys.path.insert(0,str((ROOT/'training/scripts').resolve()))
from babylm_masked_train import TRAIN_FILES, iter_examples, Example
CKPTS={
 'wwm42_40M':ROOT/'training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/hf_model/chck_40M',
 'wwm42_80M':ROOT/'training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/hf_model/chck_80M',
 'wwm42_100M':ROOT/'training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/hf_model/chck_100M',
 'wwm43_40M':ROOT/'training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model/chck_40M',
 'wwm43_80M':ROOT/'training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model/chck_80M',
 'wwm43_100M':ROOT/'training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model/chck_100M'}
STOP=set('the a an and or but if then than as of in on at to from for with without by about into over under is are was were be been being do does did have has had this that these those it its they them he she we you i me my your our their his her not no yes can could would should will may might there here very just also only'.split())
WORD_RE=re.compile(r"^[A-Za-z][A-Za-z'-]{2,}$"); CAP_RE=re.compile(r'^[A-Z][A-Za-z]{2,}$')
def setup_env():
 hf=ROOT/'training/hf_home'; os.environ['HF_HOME']=str(hf.resolve()); os.environ['HF_HUB_CACHE']=str((hf/'hub').resolve()); os.environ['HF_DATASETS_CACHE']=str((hf/'datasets').resolve()); os.environ['TRANSFORMERS_CACHE']=str((hf/'transformers').resolve()); os.environ['HF_MODULES_CACHE']=str((ROOT/'training/hf_modules_cache').resolve()); os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
def raw_dir(): return RAW_FALLBACK if RAW_FALLBACK.exists() else RUN/'raw_dataset'
def reconstruct_exact_1m(words_per_example=160,seed=42):
 files=[(RUN/'raw_dataset')/n for n in TRAIN_FILES]; pool=list(iter_examples(files,1_000_000,words_per_example))
 for i,e in enumerate(pool): e.example_id=i
 rng=random.Random(seed); rng.shuffle(pool); out=[]; actual=0
 for ex in pool:
  if actual>=1_000_000: break
  if actual+ex.words<=1_000_000: out.append(ex); actual+=ex.words
  else:
   take=1_000_000-actual; out.append(Example(' '.join(ex.text.split()[:take]),take,ex.example_id,ex.source)); actual+=take
 if actual!=1_000_000: raise RuntimeError(actual)
 return out
def broad_examples(words_per_file=200_000,words_per_example=160):
 out=[]; eid=10_000_000; rd=raw_dir()
 for n in TRAIN_FILES:
  f=rd/n
  if not f.exists(): continue
  exs=list(iter_examples([f],words_per_file,words_per_example))
  for ex in exs: ex.example_id=eid; eid+=1
  out.extend(exs)
 return out
def clean(w): return re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$",'',w)
def ids(text,tok): return tok(text,add_special_tokens=False)['input_ids']
def is_word_start(tokstr): return str(tokstr).startswith('Ġ') or str(tokstr).startswith('▁')
def surf(tokstr): return clean(str(tokstr).replace('Ġ','').replace('▁','').strip()).lower()
def good(tokstr):
 s=surf(tokstr)
 return is_word_start(tokstr) and len(s)>=3 and s not in STOP and WORD_RE.match(s)
def freq_bucket(c):
 return '1' if c<=1 else '2-3' if c<=3 else '4-10' if c<=10 else '11-30' if c<=30 else '31+'
def wordfreq(examples):
 c=Counter()
 for ex in examples:
  for w in ex.text.split():
   s=clean(w).lower()
   if s: c[s]+=1
 return c
def repeated_cap(prefix_words,later_words):
 p={clean(w) for w in prefix_words if CAP_RE.match(clean(w))}; l={clean(w) for w in later_words if CAP_RE.match(clean(w))}; return int(bool(p&l))
def local_leak(toks,idx,s):
 for j in range(max(0,idx-10),min(len(toks),idx+11)):
  if j!=idx and surf(toks[j])==s and s: return 1
 return 0
def make_cases(examples,tok,n,label,rng,wf):
 cands=[]
 for ex in examples:
  words=ex.text.split()
  if len(words)<70: continue
  split=max(30,min(len(words)-25,len(words)//2)); pw=words[:split]; lw=words[split:]
  pids=ids(' '.join(pw),tok); lids=ids(' '+' '.join(lw),tok)
  if len(pids)<20 or len(lids)<20: continue
  pids=pids[-min(len(pids),120):]; lids=lids[:min(len(lids),120)]; toks=tok.convert_ids_to_tokens(lids)
  poss=[j for j in range(max(5,len(lids)//4),min(len(lids),120)) if lids[j] not in tok.all_special_ids and good(toks[j])]
  if not poss: continue
  j=rng.choice(poss[:min(8,len(poss))]); s=surf(toks[j])
  cands.append({'case_group':label,'example_id':ex.example_id,'source':ex.source,'pids':pids,'lids':lids,'target_idx':j,'target_id':lids[j],'target_token':toks[j],'target_surface':s,'prefix_tokens':len(pids),'later_tokens':len(lids),'target_frac':j/max(1,len(lids)),'target_freq_bucket':freq_bucket(wf.get(s,0)),'local_leak_10tok':local_leak(toks,j,s),'repeated_capitalized_prefix_suffix':repeated_cap(pw,lw)})
 rng.shuffle(cands); cases=[]
 for c in cands:
  same=[o for o in cands if o['example_id']!=c['example_id'] and o['source']==c['source'] and abs(len(o['pids'])-len(c['pids']))<=20]
  cross=[o for o in cands if o['example_id']!=c['example_id'] and o['source']!=c['source'] and abs(len(o['pids'])-len(c['pids']))<=25]
  anyo=[o for o in cands if o['example_id']!=c['example_id']]
  if not anyo: continue
  c=dict(c); c['same_raw']=rng.choice(same if same else anyo)['pids']; c['cross_raw']=rng.choice(cross if cross else anyo)['pids']; cases.append(c)
  if len(cases)>=n: break
 return cases
def fit_prefix(pids,P,pad):
 p=list(pids)
 if len(p)>=P: return p[-P:], [1]*P
 # right-align prefix adjacent to suffix, pad earlier positions
 return [pad]*(P-len(p))+p, [0]*(P-len(p))+[1]*len(p)
def block_shuffle(x,rng):
 blocks=[x[i:i+8] for i in range(0,len(x),8)]; rng.shuffle(blocks); return [z for b in blocks for z in b]
def build_sequences(cases,tok):
 rng=random.Random(3168); seqs=[]; metas=[]
 for ci,c in enumerate(cases):
  P=len(c['pids']); pad=tok.pad_token_id
  full_p,full_m=fit_prefix(c['pids'],P,pad); del_p,del_m=[pad]*P,[0]*P; shuf_p,shuf_m=fit_prefix(block_shuffle(c['pids'],rng),P,pad); same_p,same_m=fit_prefix(c['same_raw'],P,pad); cross_p,cross_m=fit_prefix(c['cross_raw'],P,pad)
  variants={'full':(full_p,full_m),'deleted_fixedpos':(del_p,del_m),'block_shuffled_fixedpos':(shuf_p,shuf_m),'same_source_near_fixedpos':(same_p,same_m),'cross_source_near_fixedpos':(cross_p,cross_m)}
  for vn,(p,m) in variants.items():
   seq=p+c['lids']; am=m+[1]*len(c['lids']); pos=P+c['target_idx']
   if pos>=len(seq): continue
   seq=list(seq); seq[pos]=tok.mask_token_id; seqs.append((seq,am)); metas.append((ci,vn,pos,c['target_id']))
 return seqs,metas
def score(model,seqs,metas,device,pad,batch=96):
 out=[None]*len(metas)
 for st in range(0,len(seqs),batch):
  chunk=seqs[st:st+batch]; mchunk=metas[st:st+batch]; L=max(len(s) for s,a in chunk)
  inp=torch.full((len(chunk),L),pad,dtype=torch.long,device=device); attn=torch.zeros((len(chunk),L),dtype=torch.long,device=device)
  for i,(s,a) in enumerate(chunk): inp[i,:len(s)]=torch.tensor(s,dtype=torch.long,device=device); attn[i,:len(a)]=torch.tensor(a,dtype=torch.long,device=device)
  with torch.no_grad(): lp=torch.log_softmax(model(input_ids=inp,attention_mask=attn).logits,dim=-1)
  for i,(ci,vn,pos,tid) in enumerate(mchunk): out[st+i]=float(lp[i,pos,tid].item())
 return out
def aggregate(rows):
 summ={}; ths=[0.02,0.05,0.10,0.20]
 for exp in ['40M','80M','100M']:
  key=f'stable_full_minus_same_fixedpos_{exp}'; vals=[float(r[key]) for r in rows]
  summ[exp]={'n':len(vals),'mean':sum(vals)/len(vals),'median':statistics.median(vals),'frac_neg_le_-0.05':sum(v<=-0.05 for v in vals)/len(vals)}
  for th in ths: summ[exp][f'frac_pos_ge_{th}']=sum(v>=th for v in vals)/len(vals)
  summ[exp]['by_case_group']={}
  for g in sorted(set(r['case_group'] for r in rows)):
   vs=[float(r[key]) for r in rows if r['case_group']==g]; summ[exp]['by_case_group'][g]={'n':len(vs),'mean':sum(vs)/len(vs),'median':statistics.median(vs),'frac_pos_ge_0.05':sum(v>=0.05 for v in vs)/len(vs),'frac_pos_ge_0.2':sum(v>=0.2 for v in vs)/len(vs)}
 return summ
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--n_exact',type=int,default=300); ap.add_argument('--n_broad',type=int,default=300); ap.add_argument('--batch_size',type=int,default=96); args=ap.parse_args()
 setup_env(); t0=time.time(); tok=AutoTokenizer.from_pretrained(str(CKPTS['wwm42_40M'].resolve()),use_fast=True)
 exact=reconstruct_exact_1m(); broad=broad_examples(); wf=wordfreq(exact+broad); rng=random.Random(3162)
 cases=make_cases(exact,tok,args.n_exact,'exact_1m_slice',rng,wf)+make_cases(broad,tok,args.n_broad,'broad_official',rng,wf)
 seqs,metas=build_sequences(cases,tok); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 rows=[]
 for i,c in enumerate(cases): rows.append({k:v for k,v in c.items() if k not in ['pids','lids','same_raw','cross_raw']}|{'case_id':i})
 for name,ckpt in CKPTS.items():
  print(json.dumps({'event':'load_model','model':name,'ckpt':str(ckpt)}),flush=True); model=AutoModelForMaskedLM.from_pretrained(str(ckpt.resolve()),trust_remote_code=True).to(device).eval(); vals=score(model,seqs,metas,device,tok.pad_token_id,args.batch_size); del model; torch.cuda.empty_cache() if torch.cuda.is_available() else None
  sc=defaultdict(dict)
  for val,(ci,vn,pos,tid) in zip(vals,metas): sc[ci][vn]=val
  for ci in range(len(rows)):
   d=sc[ci]
   for vn,val in d.items(): rows[ci][f'{name}_lp_{vn}']=val
   for vn in ['deleted_fixedpos','block_shuffled_fixedpos','same_source_near_fixedpos','cross_source_near_fixedpos']:
    rows[ci][f'{name}_full_minus_{vn}']=d['full']-d[vn]
 for r in rows:
  for exp in ['40M','80M','100M']:
   a=float(r[f'wwm42_{exp}_full_minus_same_source_near_fixedpos']); b=float(r[f'wwm43_{exp}_full_minus_same_source_near_fixedpos'])
   r[f'stable_full_minus_same_fixedpos_{exp}']=math.copysign(min(abs(a),abs(b)),a+b) if a*b>0 else 0.0
   c=float(r[f'wwm42_{exp}_full_minus_deleted_fixedpos']); d=float(r[f'wwm43_{exp}_full_minus_deleted_fixedpos'])
   r[f'stable_full_minus_deleted_fixedpos_{exp}']=math.copysign(min(abs(c),abs(d)),c+d) if c*d>0 else 0.0
 OUT_ROWS.parent.mkdir(parents=True,exist_ok=True); fields=[]
 for r in rows:
  for k in r:
   if k not in fields: fields.append(k)
 with OUT_ROWS.open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
 summary=aggregate(rows); payload={'status':'PREFIX_TARGET_CENSUS_V2_FIXEDPOS','repair':'All variants preserve original prefix length and target absolute position; deleted prefix uses attention-masked pads; same/cross prefixes padded/truncated to same length; targets restricted to word-start alphabetic tokens.','n_cases':len(rows),'rows_csv':str(OUT_ROWS),'checkpoints':{k:str(v) for k,v in CKPTS.items()},'summary':summary,'elapsed_sec':time.time()-t0}
 OUT_SUMMARY.parent.mkdir(parents=True,exist_ok=True); OUT_SUMMARY.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
 lines=['# research v2 fixed-position prefix-dependent target census','',f'Rows: `{OUT_ROWS}`',f'Summary: `{OUT_SUMMARY}`','','This repairs the v1 artifact where replacement/deleted prefixes changed target absolute position. All variants now keep suffix and target at the same positions.','','| exposure | n | mean stable Δ(full-same) | median | ≥0.02 | ≥0.05 | ≥0.10 | ≥0.20 | neg≤-0.05 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for exp in ['40M','80M','100M']:
  s=summary[exp]; lines.append(f"| {exp} | {s['n']} | {s['mean']:+.5f} | {s['median']:+.5f} | {s['frac_pos_ge_0.02']:.4f} | {s['frac_pos_ge_0.05']:.4f} | {s['frac_pos_ge_0.1']:.4f} | {s['frac_pos_ge_0.2']:.4f} | {s['frac_neg_le_-0.05']:.4f} |")
 OUT_NOTE.parent.mkdir(parents=True,exist_ok=True); OUT_NOTE.write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print(json.dumps({'out':str(OUT_SUMMARY),'note':str(OUT_NOTE),'n_cases':len(rows),'summary':summary,'elapsed_sec':payload['elapsed_sec']},indent=2)[:5000])
if __name__=='__main__': main()
