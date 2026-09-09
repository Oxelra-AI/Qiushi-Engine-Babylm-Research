#!/usr/bin/env python3
"""research validated-state stability smoke.

This is NOT a corpus materializer. It tests the stability objection before any
1M data construction:

R_extra>0 under the current best model can simply select exact filler repetition
or templates the old model already likes. A candidate state thread is useful only
if its relation-specific ranking effect is (a) larger than a filler-only repeat
control and (b) directionally stable across checkpoints/seeds/stages.

For each FineWeb-Edu repeated relation candidate:
  - original: early entity-cue-y1 and later identical local y1 slot.
  - state_cf: replace early y1 -> y2.
  - unrelated_cf: replace matched unrelated prior content word.
  - filler_only_cf: insert/replace a prior unrelated word with y1 at similar
    distance, without the entity relation, to measure pure lexical y1 repetition.

Score with several frozen checkpoints:
  score = log p(y1 | masked later slot) - log p(y2 | masked later slot)
  R_state = score(original) - score(state_cf)
  R_unrelated = score(original) - score(unrelated_cf)
  R_extra = R_state - R_unrelated
  F_repeat = score(filler_only_cf) - score(unrelated_cf)
  relation_over_filler = R_extra - F_repeat

A useful pass needs relation_over_filler > 0 and sign stability across models.
"""
from __future__ import annotations
import argparse, collections, json, math, os, pathlib, random, re, statistics, time
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT=pathlib.Path('experiments/archive/initial_model_studies')
TOK_PATH=ROOT/'training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model/chck_80M'
OUT_JSON=ROOT/'data/validated_state_stability_smoke.json'
OUT_NOTE=(ROOT.parents[2] / 'research/notes/initial_model_studies/validated_state_stability_smoke.md')
WORD_RE=re.compile(r"[A-Za-z][A-Za-z'\-]{2,}")
STOP=set('''the a an and or but if then else when while of in on at by for with without from to into onto over under as is are was were be been being do did done have has had i you he she it we they me him her us them my your his its our their this that these those there here not no yes can could would should may might will shall one two three four new old first last good bad little big small great mr mrs miss sir said says say like just very more most some any many much other another about after before because through between where what which who whom whose than only also over again against every each such make made making man woman child children people person thing things time day way must soon quite during found lived existence text original added driving reports word thread foot look mother pinch'''.split())
LOCATION=set('in at on inside outside near beside behind under over across through from to into onto around within'.split())
POSSESS=set('has had held carried took brought found gave owned kept bought put left lost received holding carried'.split())
ROLE=set('is was became called named known made elected appointed born worked lived'.split())
CONTAINER=set('contained contains filled opened closed locked covered stored kept'.split())
ACTION=set('broke repaired opened closed moved changed turned killed saved built destroyed dropped raised lowered sent'.split())
FAM_CUES={'location':LOCATION,'possession':POSSESS,'role_attribute':ROLE,'container':CONTAINER,'action_consequence':ACTION}
ALL_CUES=set().union(*FAM_CUES.values())
LEAD_BAD={'The','This','That','These','Those','There','When','Where','What','How','Why','Because','For','And','But','New','All','Most','Some','Many','First','After','Before','During','Then','Each','Every','Page','Pages'}
BAD_SUBSTR=['privacy policy','terms of use','subscribe','copyright','all rights reserved','click here','<script','cookie policy']

@dataclass
class Cond:
    text:str; target_start:int; target_end:int
@dataclass
class RelOcc:
    family:str; entity:str; filler:str; ent_i:int; cue_i:int; fill_i:int; cue:str; fill_s:int; fill_e:int
@dataclass
class Case:
    case_id:str; doc_id:int; family:str; entity:str; cue:str; y1:str; y2:str; unrelated_word:str; unrelated_replacement:str; filler_only_word:str; early_gap:int; unrelated_gap:int; filler_only_gap:int; orig:Cond; state_cf:Cond; unrelated_cf:Cond; filler_only_cf:Cond; source_text_excerpt:str

def setup_env():
    hf=ROOT/'training/hf_home'
    os.environ['HF_HOME']=str(hf.resolve()); os.environ['HF_HUB_CACHE']=str((hf/'hub').resolve())
    os.environ['HF_DATASETS_CACHE']=str((hf/'datasets').resolve()); os.environ['TRANSFORMERS_CACHE']=str((hf/'transformers').resolve())
    os.environ['HF_MODULES_CACHE']=str((ROOT/'training/hf_modules_cache').resolve())
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    for k in ['HF_HOME','HF_HUB_CACHE','HF_DATASETS_CACHE','TRANSFORMERS_CACHE','HF_MODULES_CACHE']:
        pathlib.Path(os.environ[k]).mkdir(parents=True,exist_ok=True)

def norm(w): return w.strip("'\".,!?;:()[]{}“”‘’").lower()
def content(s):
    n=norm(s); return len(n)>=4 and n not in STOP and re.match(r"^[a-z][a-z'\-]+$",n) is not None
def cap_entity(s):
    if not s[:1].isupper(): return False
    first=s.split()[0]
    return first not in LEAD_BAD and len(norm(s))>=4 and norm(s) not in STOP
def clean_text(t):
    t=t.replace('\r',' ').replace('\t',' ')
    lines=[]
    for line in t.split('\n'):
        s=' '.join(line.strip().split())
        if len(s)>=25: lines.append(s)
    return ' '.join(lines)
def basic_ok(t):
    if any(b in t.lower() for b in BAD_SUBSTR): return False
    w=len(t.split())
    if w<80 or w>900: return False
    alpha=sum(ch.isalpha() for ch in t)/max(1,len(t))
    if alpha<0.65: return False
    return True
def token_ids(tok,w): return tok(w,add_special_tokens=False)['input_ids']
def same_case(repl,template): return repl.capitalize() if template[:1].isupper() else repl.lower()

def stream_docs(max_docs):
    from datasets import load_dataset
    ds=load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT', split='train', streaming=True)
    for i,row in enumerate(ds):
        if i>=max_docs: break
        t=clean_text(row.get('text',''))
        if basic_ok(t): yield i,t

def find_relations(ms,tok):
    rel=[]
    for a,(wi,surf,s,e,nw) in enumerate(ms):
        if not cap_entity(surf): continue
        for b in range(a+1,min(len(ms),a+8)):
            cue=ms[b][4]; fam=None
            for f,cues in FAM_CUES.items():
                if cue in cues: fam=f; break
            if fam is None: continue
            for c in range(b+1,min(len(ms),b+5)):
                fsurf=ms[c][1]; fn=ms[c][4]
                if fn==nw or fn in ALL_CUES or not content(fsurf): continue
                if len(token_ids(tok,fn))!=1: continue
                rel.append(RelOcc(fam,nw,fn,a,b,c,cue,ms[c][2],ms[c][3])); break
    return rel

def make_cond(doc,pstart,pend,left=650,right=220):
    lo=max(0,pstart-left); hi=min(len(doc),pend+right)
    return Cond(doc[lo:hi],pstart-lo,pend-lo)
def replace_span(doc,start,end,repl,later):
    nd=doc[:start]+repl+doc[end:]; shift=len(repl)-(end-start)
    nl=[]
    for s,e in later: nl.append((s+shift,e+shift) if start<s else (s,e))
    return nd,nl

def build_cases(max_docs,max_cases,seed,tok_path):
    rng=random.Random(seed); tok=AutoTokenizer.from_pretrained(tok_path,use_fast=True,trust_remote_code=True)
    docs=[]; freq=collections.Counter(); all_ms=[]; all_rels=[]
    for doc_id,doc in stream_docs(max_docs):
        ms=[]
        for wi,m in enumerate(WORD_RE.finditer(doc)):
            surf=m.group(0); nw=norm(surf)
            if content(surf) or cap_entity(surf):
                ms.append((wi,surf,m.start(),m.end(),nw))
                if content(surf) and len(token_ids(tok,nw))==1: freq[nw]+=1
        rels=find_relations(ms,tok)
        docs.append((doc_id,doc)); all_ms.append(ms); all_rels.append(rels)
    bands=collections.defaultdict(list)
    for w,c in freq.items(): bands[min(10,int(math.log2(c+1)))].append(w)
    for b in bands: rng.shuffle(bands[b])
    def repl_for(w,forbid):
        w=norm(w); b=min(10,int(math.log2(freq.get(w,1)+1)))
        cand=[]
        for bb in [b,b-1,b+1,b-2,b+2]:
            cand += [x for x in bands.get(bb,[]) if x not in forbid and x not in ALL_CUES and content(x) and len(token_ids(tok,x))==1]
        return rng.choice(cand) if cand else None
    cases=[]; seen=set()
    for di,(doc_id,doc) in enumerate(docs):
        ms=all_ms[di]; rels=all_rels[di]
        groups=collections.defaultdict(list)
        for r in rels: groups[(r.family,r.entity,r.filler)].append(r)
        for (fam,ent,y1),rs in groups.items():
            if len(rs)<2: continue
            rs=sorted(rs,key=lambda r:r.fill_i)
            for r1,r2 in zip(rs[:-1],rs[1:]):
                gap=r2.fill_i-r1.fill_i
                if gap<10 or gap>180: continue
                y2=repl_for(y1,{y1,ent})
                if not y2: continue
                y2surf=same_case(y2, doc[r1.fill_s:r1.fill_e])
                local=[x[4] for x in ms[max(0,r2.fill_i-8):min(len(ms),r2.fill_i+9)]]
                if local.count(y1)>1 or y2 in local: continue
                opts=[]
                for j,(wi,us,ust,uen,un) in enumerate(ms[:max(0,r2.fill_i-10)]):
                    if un in {y1,y2,ent} or un in ALL_CUES or not content(us) or len(token_ids(tok,un))!=1: continue
                    ugap=r2.fill_i-j
                    if abs(ugap-gap)>max(8,0.35*gap): continue
                    rr=repl_for(un,{un,y1,y2,ent})
                    if rr: opts.append((abs(ugap-gap),j,us,ust,uen,un,same_case(rr,us),ugap))
                if not opts: continue
                opts.sort(); _,uj,us,ust,uen,un,urepl,ugap=opts[0]
                # filler-only repeat: place y1 at matched unrelated prior content word, without entity/cue relation.
                fopts=[]
                for j,(wi,fs, fst, fen, fn) in enumerate(ms[:max(0,r2.fill_i-10)]):
                    if fn in {y1,y2,ent,un} or fn in ALL_CUES or not content(fs) or len(token_ids(tok,fn))!=1: continue
                    # Exclude words close to any relation occurrence for this entity.
                    if min(abs(j-r1.ent_i), abs(j-r1.cue_i), abs(j-r1.fill_i)) < 8: continue
                    fgap=r2.fill_i-j
                    if abs(fgap-gap)>max(8,0.35*gap): continue
                    fopts.append((abs(fgap-gap),j,fs,fst,fen,fn,fgap))
                if not fopts: continue
                fopts.sort(); _,fj,fs,fst,fen,fn,fgap=fopts[0]
                orig=make_cond(doc,r2.fill_s,r2.fill_e)
                sd,[(sps,spe)]=replace_span(doc,r1.fill_s,r1.fill_e,y2surf,[(r2.fill_s,r2.fill_e)])
                ud,[(ups,upe)]=replace_span(doc,ust,uen,urepl,[(r2.fill_s,r2.fill_e)])
                fd,[(fps,fpe)]=replace_span(doc,fst,fen,same_case(y1,fs),[(r2.fill_s,r2.fill_e)])
                state=make_cond(sd,sps,spe); unrel=make_cond(ud,ups,upe); filler=make_cond(fd,fps,fpe)
                target=orig.text[orig.target_start:orig.target_end]
                if state.text[state.target_start:state.target_end]!=target or unrel.text[unrel.target_start:unrel.target_end]!=target or filler.text[filler.target_start:filler.target_end]!=target: continue
                def loc(c):
                    lo=max(0,c.target_start-80); hi=min(len(c.text),c.target_end+80); return c.text[lo:hi]
                if loc(orig)!=loc(state) or loc(orig)!=loc(unrel) or loc(orig)!=loc(filler): continue
                cid=f'{doc_id}:{fam}:{ent}:{y1}:{r1.fill_i}:{r2.fill_i}'
                if cid in seen: continue
                seen.add(cid)
                excerpt=doc[max(0,r1.fill_s-220):min(len(doc),r2.fill_e+220)]
                cases.append(Case(cid,doc_id,fam,ent,r2.cue,y1,norm(y2surf),un,norm(urepl),fn,gap,ugap,fgap,orig,state,unrel,filler,' '.join(excerpt.split())[:900]))
                if len(cases)>=max_cases: return cases, {'docs_streamed':max_docs,'basic_docs':len(docs),'freq_vocab':len(freq)}
    return cases, {'docs_streamed':max_docs,'basic_docs':len(docs),'freq_vocab':len(freq)}

def tokpos(tok,text,start,end,max_len):
    enc=tok(text,add_special_tokens=False,truncation=True,max_length=max_len,return_offsets_mapping=True,return_tensors='pt')
    offs=enc.pop('offset_mapping')[0].tolist(); pos=[i for i,(s,e) in enumerate(offs) if e>start and s<end]
    if len(pos)!=1: return None,None
    return enc,pos[0]
def score_cond(model,tok,c,y1,y2,max_len,device):
    ids1=token_ids(tok,y1); ids2=token_ids(tok,y2)
    if len(ids1)!=1 or len(ids2)!=1: return None
    enc,pos=tokpos(tok,c.text,c.target_start,c.target_end,max_len)
    if enc is None: return None
    inp=enc['input_ids'].to(device); att=enc['attention_mask'].to(device); inp[0,pos]=tok.mask_token_id
    with torch.no_grad(): lp=F.log_softmax(model(input_ids=inp,attention_mask=att).logits[0,pos],dim=-1)
    return float(lp[ids1[0]]-lp[ids2[0]])
def eval_model(name,path,cases,max_len,device):
    tok=AutoTokenizer.from_pretrained(path,use_fast=True,trust_remote_code=True)
    model=AutoModelForMaskedLM.from_pretrained(path,trust_remote_code=True).to(device).eval()
    rows=[]
    for c in cases:
        vals=[score_cond(model,tok,cond,c.y1,c.y2,max_len,device) for cond in [c.orig,c.state_cf,c.unrelated_cf,c.filler_only_cf]]
        if any(v is None for v in vals): continue
        so,ss,su,sf=vals
        R_state=so-ss; R_unrel=so-su; R_extra=R_state-R_unrel; F_repeat=sf-su
        rows.append({'case_id':c.case_id,'family':c.family,'entity':c.entity,'y1':c.y1,'y2':c.y2,'score_orig':so,'R_state':R_state,'R_unrelated':R_unrel,'R_extra':R_extra,'F_repeat':F_repeat,'relation_over_filler':R_extra-F_repeat})
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    def stat(key):
        x=np.asarray([r[key] for r in rows],dtype='float64')
        return {'mean':float(x.mean()) if len(x) else None,'median':float(np.median(x)) if len(x) else None,'sem':float(x.std()/math.sqrt(max(1,len(x)))) if len(x) else None,'frac_pos':float((x>0).mean()) if len(x) else None}
    return {'num_used':len(rows),'R_extra':stat('R_extra'),'F_repeat':stat('F_repeat'),'relation_over_filler':stat('relation_over_filler'),'rows':rows}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--max_docs',type=int,default=3000); ap.add_argument('--max_cases',type=int,default=120); ap.add_argument('--max_len',type=int,default=224); ap.add_argument('--seed',type=int,default=282); ap.add_argument('--out_json',default=str(OUT_JSON)); ap.add_argument('--out_note',default=str(OUT_NOTE))
    ap.add_argument('--models',nargs='*',default=[
        'wwm43_40M=experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model/chck_40M',
        'wwm43_80M=experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model/chck_80M',
        'wwm43_100M=experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model/chck_100M',
        'wwm42_100M=experiments/archive/initial_model_studies/training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/hf_model/chck_100M'
    ])
    args=ap.parse_args(); setup_env(); t0=time.time(); device='cuda' if torch.cuda.is_available() else 'cpu'
    cases,build_stats=build_cases(args.max_docs,args.max_cases,args.seed,TOK_PATH)
    payload={'status':'VALIDATED_STATE_STABILITY_SMOKE','build_stats':build_stats,'num_cases':len(cases),'models':{},'cross_model':{},'samples':[asdict(c) for c in cases[:12]],'validity_note':'Adds filler-only repetition control and cross-model stability. Not a materializer.'}
    for spec in args.models:
        name,path=spec.split('=',1); p=pathlib.Path(path)
        if p.exists():
            payload['models'][name]=eval_model(name,p,cases,args.max_len,device)
            pathlib.Path(args.out_json).parent.mkdir(parents=True,exist_ok=True); pathlib.Path(args.out_json).write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n')
    # Cross-model stability on common cases.
    model_rows={m:{r['case_id']:r for r in d['rows']} for m,d in payload['models'].items()}
    common=set.intersection(*(set(x) for x in model_rows.values())) if model_rows else set()
    stable=[]
    for cid in common:
        vals=[model_rows[m][cid]['relation_over_filler'] for m in model_rows]
        rex=[model_rows[m][cid]['R_extra'] for m in model_rows]
        stable.append({'case_id':cid,'all_relation_over_filler_pos':all(v>0 for v in vals),'all_R_extra_pos':all(v>0 for v in rex),'mean_relation_over_filler':float(np.mean(vals)),'mean_R_extra':float(np.mean(rex))})
    payload['cross_model']={'common_cases':len(common),'all_models_relation_over_filler_pos_count':sum(x['all_relation_over_filler_pos'] for x in stable),'all_models_R_extra_pos_count':sum(x['all_R_extra_pos'] for x in stable),'all_models_relation_over_filler_pos_rate':sum(x['all_relation_over_filler_pos'] for x in stable)/max(1,len(stable)),'all_models_R_extra_pos_rate':sum(x['all_R_extra_pos'] for x in stable)/max(1,len(stable)),'stable_rows':stable[:40]}
    payload['elapsed_sec']=time.time()-t0
    pathlib.Path(args.out_json).write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n')
    lines=['# research validated-state stability smoke','',f'Evidence JSON: `{args.out_json}`','',f"Cases extracted: {len(cases)} from {build_stats['docs_streamed']} streamed docs / {build_stats['basic_docs']} basic-quality docs",'', '## Per-model effects','', '| model | n | R_extra mean | R_extra frac>0 | F_repeat mean | relation_over_filler mean | relation_over_filler frac>0 |','|---|---:|---:|---:|---:|---:|---:|']
    for name,d in payload['models'].items():
        lines.append(f"| {name} | {d['num_used']} | {d['R_extra']['mean']:+.4f} | {d['R_extra']['frac_pos']:.3f} | {d['F_repeat']['mean']:+.4f} | {d['relation_over_filler']['mean']:+.4f} | {d['relation_over_filler']['frac_pos']:.3f} |")
    cm=payload['cross_model']; lines += ['', '## Cross-model stability', '', f"Common cases: {cm['common_cases']}", f"All-models R_extra>0: {cm['all_models_R_extra_pos_count']} ({cm['all_models_R_extra_pos_rate']:.3f})", f"All-models relation_over_filler>0: {cm['all_models_relation_over_filler_pos_count']} ({cm['all_models_relation_over_filler_pos_rate']:.3f})", '', 'Interpretation: R_extra alone can select exact repetition. relation_over_filler = R_extra - filler-only repeat effect must be positive and stable before materializing validated-state 1M data.', '', '## Sample cases', '']
    for c in cases[:8]:
        lines.append(f"- {c.case_id} family={c.family} entity={c.entity} cue={c.cue} y1={c.y1} y2={c.y2} excerpt={c.source_text_excerpt[:450]}")
    pathlib.Path(args.out_note).parent.mkdir(parents=True,exist_ok=True); pathlib.Path(args.out_note).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'out_json':args.out_json,'out_note':args.out_note,'num_cases':len(cases),'cross_model':payload['cross_model'],'elapsed_sec':payload['elapsed_sec']},indent=2))
if __name__=='__main__': main()
