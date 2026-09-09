#!/usr/bin/env python3
"""research v2: stricter official-text state-counterfactual ranking test.

This replaces the permissive first miner after smoke inspection showed many targets
were not real state fillers. v2 keeps only explicit relation occurrences:
  Capitalized repeated entity ... relation cue ... single-token filler
appearing twice with the same entity, relation family, and filler. The later
filler is masked and scored against a matched single-token alternative after an
early filler replacement, with matched unrelated and surface edits.
"""
from __future__ import annotations
import argparse, collections, json, math, os, pathlib, random, re, time
from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT=pathlib.Path('experiments/archive/initial_model_studies')
RAW_DEFAULT=ROOT/'training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/raw_dataset'
TOK_PATH=ROOT/'training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model/chck_80M'
WORD_RE=re.compile(r"[A-Za-z][A-Za-z'\-]{2,}")
STOP=set('''the a an and or but if then else when while of in on at by for with without from to into onto over under as is are was were be been being do did done have has had i you he she it we they me him her us them my your his its our their this that these those there here not no yes can could would should may might will shall one two three four new old first last good bad little big small great mr mrs miss sir said says say like just very more most some any many much other another about after before because through between where what which who whom whose than only also over again against every each such make made making man woman child children people person thing things time day way must soon quite during found lived existence text original added driving reports word thread foot look mother pinch'''.split())
LOCATION=set('in at on inside outside near beside behind under over across through from to into onto around within'.split())
POSSESS=set('has had held carried took brought found gave owned kept bought put left lost received holding carried'.split())
ROLE=set('is was became called named known made elected appointed born worked lived'.split())
CONTAINER=set('contained contains filled opened closed locked covered stored kept'.split())
ACTION=set('broke repaired opened closed moved changed turned killed saved built destroyed dropped raised lowered sent'.split())
FAM_CUES={'location':LOCATION,'possession':POSSESS,'role_attribute':ROLE,'container':CONTAINER,'action_consequence':ACTION}
ALL_CUES=set().union(*FAM_CUES.values())

@dataclass
class Cond:
    text:str; target_start:int; target_end:int
@dataclass
class RelOcc:
    family:str; entity:str; filler:str; ent_i:int; cue_i:int; fill_i:int; cue:str; ent_s:int; ent_e:int; fill_s:int; fill_e:int
@dataclass
class Case:
    case_id:str; source:str; family:str; entity:str; cue:str; y1:str; y2:str; unrelated_word:str; unrelated_replacement:str; surface_word:str; surface_replacement:str; early_gap_tokens:int; unrelated_gap_tokens:int; orig:Cond; state_cf:Cond; unrelated_cf:Cond; surface_cf:Cond

def setup_env():
    hf=ROOT/'training/hf_home'
    os.environ['HF_HOME']=str(hf.resolve()); os.environ['HF_HUB_CACHE']=str((hf/'hub').resolve())
    os.environ['TRANSFORMERS_CACHE']=str((hf/'transformers').resolve()); os.environ['HF_MODULES_CACHE']=str((ROOT/'training/hf_modules_cache').resolve())
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')

def norm(w): return w.strip("'\".,!?;:()[]{}“”‘’").lower()
def cap_entity(s): return s[:1].isupper() and norm(s) not in STOP and len(norm(s))>=4
def content(s):
    n=norm(s); return len(n)>=4 and n not in STOP and re.match(r"^[a-z][a-z'\-]+$",n) is not None

def iter_docs(raw_dir,max_docs):
    files=['gutenberg.train.txt','simple_wiki.train.txt','childes.train.txt','open_subtitles.train.txt','bnc_spoken.train.txt','switchboard.train.txt']
    n=0
    for fn in files:
        p=raw_dir/fn
        if not p.exists(): continue
        buf=[]; wc=0
        for line in p.open('r',encoding='utf-8',errors='replace'):
            s=line.strip()
            if s:
                buf.append(s); wc+=len(s.split())
                if wc>=260:
                    yield fn,' '.join(buf); n+=1; buf=[]; wc=0
                    if n>=max_docs: return
            elif buf:
                yield fn,' '.join(buf); n+=1; buf=[]; wc=0
                if n>=max_docs: return
        if buf:
            yield fn,' '.join(buf); n+=1
            if n>=max_docs: return

def token_ids(tok,w): return tok(w,add_special_tokens=False)['input_ids']
def same_case(repl,template): return repl.capitalize() if template[:1].isupper() else repl.lower()

def find_relations(ms,tok):
    rel=[]
    # ms entries: (word_index, surface, start, end, norm)
    for a,(wi,surf,s,e,nw) in enumerate(ms):
        if not cap_entity(surf): continue
        # explicit entity ... cue ... filler, with cue after entity and filler after cue
        for b in range(a+1,min(len(ms),a+8)):
            cue=ms[b][4]
            fam=None
            for f,cues in FAM_CUES.items():
                if cue in cues: fam=f; break
            if fam is None: continue
            for c in range(b+1,min(len(ms),b+5)):
                fsurf=ms[c][1]; fn=ms[c][4]
                if fn==nw or fn in ALL_CUES or not content(fsurf): continue
                if len(token_ids(tok,fn))!=1: continue
                rel.append(RelOcc(fam,nw,fn,a,b,c,cue,s,e,ms[c][2],ms[c][3]))
                break
    return rel

def make_cond(doc,pstart,pend,left=650,right=220):
    lo=max(0,pstart-left); hi=min(len(doc),pend+right)
    return Cond(doc[lo:hi],pstart-lo,pend-lo)

def replace_span(doc,start,end,repl,later):
    nd=doc[:start]+repl+doc[end:]; shift=len(repl)-(end-start)
    nl=[]
    for s,e in later: nl.append((s+shift,e+shift) if start<s else (s,e))
    return nd,nl

def build(raw_dir,max_docs,max_cases,seed,tok_path):
    rng=random.Random(seed); tok=AutoTokenizer.from_pretrained(tok_path,use_fast=True,trust_remote_code=True)
    docs=list(iter_docs(raw_dir,max_docs)); freq=collections.Counter(); all_ms=[]; all_rels=[]
    for src,doc in docs:
        ms=[]
        for wi,m in enumerate(WORD_RE.finditer(doc)):
            surf=m.group(0); nw=norm(surf)
            if content(surf) or cap_entity(surf):
                ms.append((wi,surf,m.start(),m.end(),nw))
                if content(surf) and len(token_ids(tok,nw))==1: freq[nw]+=1
        all_ms.append(ms); all_rels.append(find_relations(ms,tok))
    bands=collections.defaultdict(list)
    for w,c in freq.items(): bands[min(10,int(math.log2(c+1)))].append(w)
    for b in bands: rng.shuffle(bands[b])
    def repl_for(w,forbid):
        w=norm(w); b=min(10,int(math.log2(freq.get(w,1)+1)))
        cand=[]
        for bb in [b,b-1,b+1,b-2,b+2]:
            cand += [x for x in bands.get(bb,[]) if x not in forbid and len(token_ids(tok,x))==1 and x not in ALL_CUES]
        return rng.choice(cand) if cand else None
    cases=[]; seen=set()
    for di,(src,doc) in enumerate(docs):
        ms=all_ms[di]; rels=all_rels[di]
        groups=collections.defaultdict(list)
        for r in rels: groups[(r.family,r.entity,r.filler)].append(r)
        for (fam,ent,y1),rs in groups.items():
            if len(rs)<2: continue
            rs=sorted(rs,key=lambda r:r.fill_i)
            for r1,r2 in zip(rs[:-1],rs[1:]):
                gap=r2.fill_i-r1.fill_i
                if gap<10 or gap>160: continue
                y2=repl_for(y1,{y1,ent})
                if not y2: continue
                y2surf=same_case(y2,doc[r1.fill_s:r1.fill_e])
                # local target context must not contain y1 besides target, or y2.
                local=[x[4] for x in ms[max(0,r2.fill_i-8):min(len(ms),r2.fill_i+9)]]
                if local.count(y1)>1 or y2 in local: continue
                # unrelated edit: a single-token content word before later target, similar distance from later slot, not a relation cue/filler/entity.
                opts=[]
                for j,(wi,us,ust,uen,un) in enumerate(ms[:max(0,r2.fill_i-10)]):
                    if un in {y1,y2,ent} or un in ALL_CUES or not content(us) or len(token_ids(tok,un))!=1: continue
                    ugap=r2.fill_i-j
                    if abs(ugap-gap)>max(8,0.35*gap): continue
                    rr=repl_for(un,{un,y1,y2,ent})
                    if rr: opts.append((abs(ugap-gap),j,us,ust,uen,un,same_case(rr,us),ugap))
                if not opts: continue
                opts.sort(); _,uj,us,ust,uen,un,urepl,ugap=opts[0]
                # surface edit near early relation but not cue/entity/filler
                sopts=[]
                for j in range(max(0,r1.ent_i-4),min(len(ms),r1.fill_i+5)):
                    _,ss,st,en,sn=ms[j]
                    if sn in {y1,y2,ent,un} or sn in ALL_CUES or not content(ss) or len(token_ids(tok,sn))!=1: continue
                    sr=repl_for(sn,{sn,y1,y2,ent,un})
                    if sr: sopts.append((abs(j-r1.fill_i),j,ss,st,en,sn,same_case(sr,ss)))
                if not sopts: continue
                sopts.sort(); _,sj,ss,st,en,sn,srepl=sopts[0]
                orig=make_cond(doc,r2.fill_s,r2.fill_e)
                sd,[(sps,spe)]=replace_span(doc,r1.fill_s,r1.fill_e,y2surf,[(r2.fill_s,r2.fill_e)])
                ud,[(ups,upe)]=replace_span(doc,ust,uen,urepl,[(r2.fill_s,r2.fill_e)])
                fd,[(fps,fpe)]=replace_span(doc,st,en,srepl,[(r2.fill_s,r2.fill_e)])
                state=make_cond(sd,sps,spe); unrel=make_cond(ud,ups,upe); surf=make_cond(fd,fps,fpe)
                target=orig.text[orig.target_start:orig.target_end]
                if state.text[state.target_start:state.target_end]!=target or unrel.text[unrel.target_start:unrel.target_end]!=target or surf.text[surf.target_start:surf.target_end]!=target: continue
                def loc(c):
                    lo=max(0,c.target_start-80); hi=min(len(c.text),c.target_end+80); return c.text[lo:hi]
                if loc(orig)!=loc(state) or loc(orig)!=loc(unrel) or loc(orig)!=loc(surf): continue
                cid=f'{src}:{di}:{fam}:{ent}:{y1}:{r1.fill_i}:{r2.fill_i}'
                if cid in seen: continue
                seen.add(cid)
                cases.append(Case(cid,src,fam,ent,r2.cue,y1,norm(y2surf),un,norm(urepl),sn,norm(srepl),gap,ugap,orig,state,unrel,surf))
                if len(cases)>=max_cases: return cases
    return cases

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
    with torch.no_grad():
        lp=F.log_softmax(model(input_ids=inp,attention_mask=att).logits[0,pos],dim=-1)
    return float(lp[ids1[0]]-lp[ids2[0]])
def eval_model(name,path,cases,max_len,device):
    tok=AutoTokenizer.from_pretrained(path,use_fast=True,trust_remote_code=True); model=AutoModelForMaskedLM.from_pretrained(path,trust_remote_code=True).to(device).eval()
    rows=[]
    for c in cases:
        vals=[score_cond(model,tok,cond,c.y1,c.y2,max_len,device) for cond in [c.orig,c.state_cf,c.unrelated_cf,c.surface_cf]]
        if any(v is None for v in vals): continue
        so,ss,su,sf=vals; rows.append({'case_id':c.case_id,'family':c.family,'score_orig':so,'R_state':so-ss,'R_unrelated':so-su,'R_surface':so-sf,'R_extra':(so-ss)-(so-su)})
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    def arr(k,sub=None): return np.asarray([r[k] for r in rows if sub is None or r['family']==sub],dtype='float64')
    fams=sorted(set(r['family'] for r in rows))
    def stats(x): return {'mean':float(x.mean()) if len(x) else None,'sem':float(x.std()/math.sqrt(max(1,len(x)))) if len(x) else None,'frac_pos':float((x>0).mean()) if len(x) else None,'n':int(len(x))}
    return {'num_used':len(rows),'score_orig_mean':float(arr('score_orig').mean()) if rows else None,'R_state':stats(arr('R_state')),'R_unrelated':stats(arr('R_unrelated')),'R_surface':stats(arr('R_surface')),'R_extra':stats(arr('R_extra')),'by_family':{f:{'R_extra':stats(arr('R_extra',f)),'R_state_mean':float(arr('R_state',f).mean()) if len(arr('R_state',f)) else None,'R_unrelated_mean':float(arr('R_unrelated',f).mean()) if len(arr('R_unrelated',f)) else None} for f in fams},'rows':rows[:30]}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--raw_dir',default=str(RAW_DEFAULT)); ap.add_argument('--max_docs',type=int,default=4000); ap.add_argument('--max_cases',type=int,default=200); ap.add_argument('--max_len',type=int,default=224); ap.add_argument('--seed',type=int,default=2752)
    ap.add_argument('--inventory_jsonl',default='experiments/archive/initial_model_studies/data/state_counterfactual_inventory_v2.jsonl'); ap.add_argument('--summary_json',default='experiments/archive/initial_model_studies/data/state_counterfactual_inventory_v2_summary.json'); ap.add_argument('--results_json',default='experiments/archive/initial_model_studies/data/state_counterfactual_ranking_v2_results.json'); ap.add_argument('--note',default='research/notes/initial_model_studies/state_counterfactual_ranking_v2_results.md')
    ap.add_argument('--models',nargs='*',default=['wwm43_80M=experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model/chck_80M','wwm43_100M=experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model/chck_100M','protected42_100M=experiments/archive/initial_model_studies/training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/hf_model/chck_100M'])
    args=ap.parse_args(); setup_env(); t0=time.time(); device='cuda' if torch.cuda.is_available() else 'cpu'
    cases=build(pathlib.Path(args.raw_dir),args.max_docs,args.max_cases,args.seed,TOK_PATH)
    pathlib.Path(args.inventory_jsonl).parent.mkdir(parents=True,exist_ok=True)
    with open(args.inventory_jsonl,'w',encoding='utf-8') as f:
        for c in cases: f.write(json.dumps(asdict(c),ensure_ascii=False)+'\n')
    fam=collections.Counter(c.family for c in cases); summary={'status':'V2_STRICT_INVENTORY','num_cases':len(cases),'families':dict(fam),'unique_entities':len(set(c.entity for c in cases)),'unique_y1':len(set(c.y1 for c in cases)),'mean_gap':float(np.mean([c.early_gap_tokens for c in cases])) if cases else None,'mean_unrelated_gap':float(np.mean([c.unrelated_gap_tokens for c in cases])) if cases else None,'inventory_jsonl':args.inventory_jsonl}
    pathlib.Path(args.summary_json).write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    results={'status':'V2_STRICT_STATE_RANKING','inventory_summary':summary,'models':{},'elapsed_sec':None,'validity_note':'explicit repeated capitalized entity + cue + single-token filler in early and later spans; identical later local text across conditions'}
    for spec in args.models:
        name,path=spec.split('=',1); p=pathlib.Path(path)
        if p.exists() and cases: results['models'][name]=eval_model(name,p,cases,args.max_len,device); pathlib.Path(args.results_json).parent.mkdir(parents=True,exist_ok=True); pathlib.Path(args.results_json).write_text(json.dumps(results,indent=2,ensure_ascii=False)+'\n')
    results['elapsed_sec']=time.time()-t0; pathlib.Path(args.results_json).write_text(json.dumps(results,indent=2,ensure_ascii=False)+'\n')
    lines=['# research v2 strict state-counterfactual ranking','',f'Inventory JSONL: `{args.inventory_jsonl}`',f'Summary JSON: `{args.summary_json}`',f'Results JSON: `{args.results_json}`','',f"Cases: {len(cases)}; families: {dict(fam)}; unique entities {summary['unique_entities']}; unique fillers {summary['unique_y1']}; mean gap {summary['mean_gap']}",'','| model | n | score_orig | R_state | R_unrelated | R_surface | R_extra | SEM | frac R_extra>0 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,d in results['models'].items():
        lines.append(f"| {name} | {d['num_used']} | {d['score_orig_mean']:.4f} | {d['R_state']['mean']:+.4f} | {d['R_unrelated']['mean']:+.4f} | {d['R_surface']['mean']:+.4f} | {d['R_extra']['mean']:+.4f} | {d['R_extra']['sem']:.4f} | {d['R_extra']['frac_pos']:.3f} |")
    lines += ['','## By family','', '| model | family | n | R_extra | frac pos | R_state | R_unrelated |','|---|---|---:|---:|---:|---:|---:|']
    for name,d in results['models'].items():
        for f,fd in d['by_family'].items(): lines.append(f"| {name} | {f} | {fd['R_extra']['n']} | {fd['R_extra']['mean']:+.4f} | {fd['R_extra']['frac_pos']:.3f} | {fd['R_state_mean']:+.4f} | {fd['R_unrelated_mean']:+.4f} |")
    pathlib.Path(args.note).parent.mkdir(parents=True,exist_ok=True); pathlib.Path(args.note).write_text('\n'.join(lines)+'\n')
    print(json.dumps({'inventory':args.inventory_jsonl,'summary':args.summary_json,'results':args.results_json,'note':args.note,'num_cases':len(cases),'elapsed_sec':results['elapsed_sec']},indent=2))
if __name__=='__main__': main()
