#!/usr/bin/env python3
"""research official-text state-counterfactual inventory and ranking test.

Purpose
-------
Before any new pretraining, test whether existing BabyLM checkpoints use an
earlier state/fact about an entity to rank a later masked filler over a matched
counterfactual filler when the later local context is identical.

For each case, mine from official BabyLM raw text:
  early relation:   entity ... filler_y1
  later slot:       entity ... filler_y1
Construct:
  original history
  state-changing counterfactual: replace only early filler_y1 -> matched y2
  unrelated-history control: replace a non-state earlier word at similar distance
  surface/state-preserving control: replace another non-state word near early span
The later local context and masked target span are identical across conditions.

Ranking statistic for one case/model:
  score(c) = log p(y1 | condition c, later slot masked)
             - log p(y2 | condition c, later slot masked)
  R_state = score(original) - score(state_cf)
  R_unrelated = score(original) - score(unrelated_cf)
  R_extra = R_state - R_unrelated
Positive R_extra indicates state-specific prior history matters more than a
matched upstream perturbation.

This is a test, not training. All text comes from counted official raw corpus.
"""
from __future__ import annotations
import argparse, collections, json, math, os, pathlib, random, re, time
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
RAW_DEFAULT = ROOT/'training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/raw_dataset'
BASE_TOK_PATH = ROOT/'training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model/chck_80M'
OUT_INV_DEFAULT = ROOT/'data/state_counterfactual_inventory.jsonl'
OUT_SUM_DEFAULT = ROOT/'data/state_counterfactual_inventory_summary.json'
OUT_RES_DEFAULT = ROOT/'data/state_counterfactual_ranking_results.json'
OUT_NOTE_DEFAULT = (ROOT.parents[2] / 'research/notes/initial_model_studies/state_counterfactual_ranking_results.md')

STOP = set('''the a an and or but if then else when while of in on at by for with without from to into onto over under as is are was were be been being do did done have has had i you he she it we they me him her us them my your his its our their this that these those there here not no yes can could would should may might will shall one two three four new old first last good bad little big small great mr mrs miss sir said says say like just very more most some any many much other another about after before because through between where what which who whom whose than only also over again against every each such make made making man woman child children people person thing things time day way said say says just like very went come came go get got could would should'''.split())
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]{2,}")
LOCATION_CUES = set('in at on inside outside near beside behind under over across through from to into onto around within'.split())
POSSESSION_CUES = set('has had have held carried took brought found gave owned kept bought put left lost received'.split())
ROLE_CUES = set('is was became called named known made elected appointed born worked lived'.split())
CONTAINER_CUES = set('contained contains holding inside filled opened closed locked covered stored'.split())
ACTION_CUES = set('broke repaired opened closed moved changed turned killed saved built destroyed dropped raised lowered'.split())
ALL_CUES = LOCATION_CUES | POSSESSION_CUES | ROLE_CUES | CONTAINER_CUES | ACTION_CUES


def setup_env():
    hf = ROOT/'training/hf_home'
    os.environ['HF_HOME'] = str(hf.resolve())
    os.environ['HF_HUB_CACHE'] = str((hf/'hub').resolve())
    os.environ['TRANSFORMERS_CACHE'] = str((hf/'transformers').resolve())
    os.environ['HF_MODULES_CACHE'] = str((ROOT/'training/hf_modules_cache').resolve())
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')


def norm(w: str) -> str:
    return w.strip("'\".,!?;:()[]{}“”‘’").lower()


def is_candidate(w: str) -> bool:
    n = norm(w)
    return len(n) >= 4 and n not in STOP and re.match(r"^[a-z][a-z'\-]+$", n) is not None


def iter_docs(raw_dir: pathlib.Path, max_docs: int):
    files = ['gutenberg.train.txt','simple_wiki.train.txt','childes.train.txt','open_subtitles.train.txt','bnc_spoken.train.txt','switchboard.train.txt']
    n = 0
    for fn in files:
        p = raw_dir/fn
        if not p.exists():
            continue
        buf=[]; wc=0
        with p.open('r', encoding='utf-8', errors='replace') as f:
            for line in f:
                s=line.strip()
                if s:
                    buf.append(s); wc += len(s.split())
                    if wc >= 220:
                        yield fn, ' '.join(buf); n += 1; buf=[]; wc=0
                        if n >= max_docs: return
                elif buf:
                    yield fn, ' '.join(buf); n += 1; buf=[]; wc=0
                    if n >= max_docs: return
            if buf:
                yield fn, ' '.join(buf); n += 1
                if n >= max_docs: return


@dataclass
class Cond:
    text: str
    target_start: int
    target_end: int

@dataclass
class Case:
    case_id: str
    source: str
    family: str
    entity: str
    y1: str
    y2: str
    unrelated_word: str
    unrelated_replacement: str
    surface_word: str
    surface_replacement: str
    early_gap_tokens: int
    unrelated_gap_tokens: int
    local_window_words: int
    orig: Cond
    state_cf: Cond
    unrelated_cf: Cond
    surface_cf: Cond


def token_ids(tok, w: str) -> list[int]:
    return tok(w, add_special_tokens=False)['input_ids']


def case_surface(repl: str, template: str) -> str:
    return repl.capitalize() if template[:1].isupper() else repl.lower()


def family_from(words: list[str]) -> str:
    s = set(norm(x) for x in words)
    if s & LOCATION_CUES: return 'location'
    if s & POSSESSION_CUES: return 'possession'
    if s & CONTAINER_CUES: return 'container'
    if s & ROLE_CUES: return 'role_attribute'
    if s & ACTION_CUES: return 'action_consequence'
    return 'other_relation'


def make_cond(doc: str, target_start: int, target_end: int, left_chars: int=560, right_chars: int=180) -> Cond:
    left=max(0, target_start-left_chars)
    right=min(len(doc), target_end+right_chars)
    return Cond(doc[left:right], target_start-left, target_end-left)


def replace_span(doc: str, start: int, end: int, repl: str, later_positions: list[tuple[int,int]]) -> tuple[str, list[tuple[int,int]]]:
    newdoc = doc[:start] + repl + doc[end:]
    shift = len(repl) - (end-start)
    newpos=[]
    for s,e in later_positions:
        if start < s:
            newpos.append((s+shift, e+shift))
        else:
            newpos.append((s,e))
    return newdoc, newpos


def build_inventory(raw_dir: pathlib.Path, max_docs: int, max_cases: int, seed: int, tok_path: pathlib.Path) -> list[Case]:
    rng=random.Random(seed)
    tok=AutoTokenizer.from_pretrained(tok_path, use_fast=True, trust_remote_code=True)
    docs=list(iter_docs(raw_dir, max_docs))
    freq=collections.Counter()
    all_matches=[]
    for di,(src,doc) in enumerate(docs):
        ms=[]
        for wi,m in enumerate(WORD_RE.finditer(doc)):
            surf=m.group(0); nw=norm(surf)
            if is_candidate(surf):
                ids=token_ids(tok, nw)
                if len(ids)==1:
                    freq[nw]+=1
                    ms.append((wi,surf,m.start(),m.end(),nw))
        all_matches.append(ms)
    # replacement candidates: single-token, similar frequency band, not too rare if possible
    bands=collections.defaultdict(list)
    for w,c in freq.items():
        bands[min(10, int(math.log2(c+1)))].append(w)
    for b in bands: rng.shuffle(bands[b])

    def repl_for(word: str, forbidden: set[str]) -> Optional[str]:
        c=freq.get(norm(word),1); b=min(10,int(math.log2(c+1)))
        cand=[]
        for bb in [b,b-1,b+1,b-2,b+2]:
            cand += [x for x in bands.get(bb,[]) if x not in forbidden and len(token_ids(tok,x))==1]
        if not cand: return None
        return rng.choice(cand)

    out=[]; seen=set()
    for di,(src,doc) in enumerate(docs):
        ms=all_matches[di]
        if len(ms)<30: continue
        by_word=collections.defaultdict(list)
        for j,t in enumerate(ms): by_word[t[4]].append(j)
        # later slot = a repeated filler y1. Need entity near earlier and later occurrences.
        for y1, yidxs in list(by_word.items()):
            if len(yidxs)<2 or y1 in STOP: continue
            # use several occurrence pairs, not only first/second
            for qj in yidxs[:-1]:
                for pj in yidxs[yidxs.index(qj)+1:]:
                    if pj-qj < 12 or pj-qj > 120: continue
                    _, ysurf_q, qstart, qend, _ = ms[qj]
                    _, ysurf_p, pstart, pend, _ = ms[pj]
                    # find an entity word near both q and p, preferring preceding mention near p
                    later_context = list(range(max(0,pj-8), min(len(ms),pj+5)))
                    early_context = list(range(max(0,qj-10), min(len(ms),qj+10)))
                    ents=[]
                    for lj in later_context:
                        ew=ms[lj][4]
                        if ew==y1 or ew in STOP: continue
                        for ej in early_context:
                            if ms[ej][4]==ew and abs(ej-qj)<=10 and abs(lj-pj)<=8:
                                ents.append((abs(lj-pj)+abs(ej-qj), ew, ej, lj))
                    if not ents: continue
                    ents.sort(); _, ent, ej, lj = ents[0]
                    rel_words=[ms[k][4] for k in range(min(ej,qj), max(ej,qj)+1)]
                    fam=family_from(rel_words)
                    if fam=='other_relation': continue
                    # local leakage: y1 not repeated in +/-8 word local context except target
                    local_js=list(range(max(0,pj-8), min(len(ms),pj+9)))
                    if sum(1 for k in local_js if ms[k][4]==y1) > 1: continue
                    y2=repl_for(ysurf_q, {y1, ent})
                    if not y2: continue
                    y2surf=case_surface(y2, ysurf_q)
                    # Choose unrelated control before target, similar distance to qj, not in local context, not ent/y1
                    unrelated_opts=[]
                    for rj in range(0, max(0,pj-10)):
                        _, rsurf, rstart, rend, rw = ms[rj]
                        if rw in {y1,ent,norm(y2surf)}: continue
                        if any(abs(rj-k)<=8 for k in [pj]): continue
                        if abs((pj-rj)-(pj-qj)) > max(8, 0.3*(pj-qj)): continue
                        rr=repl_for(rsurf, {rw,y1,ent,norm(y2surf)})
                        if rr: unrelated_opts.append((abs((pj-rj)-(pj-qj)), rj, rsurf, rstart, rend, rw, case_surface(rr,rsurf)))
                    if not unrelated_opts: continue
                    unrelated_opts.sort(); _, rj, rsurf, rstart, rend, rw, rrepl = unrelated_opts[0]
                    # surface/state-preserving-ish control: edit non y/ent word near early relation, not cue if possible.
                    surf_opts=[]
                    for sj in range(max(0,min(ej,qj)-5), min(len(ms),max(ej,qj)+6)):
                        _, ssurf, sstart, send, sw = ms[sj]
                        if sw in {y1,ent,rw,norm(y2surf)} or sw in ALL_CUES: continue
                        sr=repl_for(ssurf, {sw,y1,ent,rw,norm(y2surf)})
                        if sr: surf_opts.append((abs(sj-qj), sj, ssurf, sstart, send, sw, case_surface(sr,ssurf)))
                    if not surf_opts: surf_opts=unrelated_opts
                    surf_opts.sort(); _, sj, ssurf, sstart, send, sw, srepl = surf_opts[0]
                    # Build conditions, with separate shifted target offsets.
                    orig_cond=make_cond(doc,pstart,pend)
                    state_doc, [(spstart, spend)] = replace_span(doc,qstart,qend,y2surf,[(pstart,pend)])
                    state_cond=make_cond(state_doc,spstart,spend)
                    unrel_doc, [(upstart, upend)] = replace_span(doc,rstart,rend,rrepl,[(pstart,pend)])
                    unrel_cond=make_cond(unrel_doc,upstart,upend)
                    surf_doc, [(sfstart, sfend)] = replace_span(doc,sstart,send,srepl,[(pstart,pend)])
                    surf_cond=make_cond(surf_doc,sfstart,sfend)
                    # Verify identical later target substring and local context.
                    target=orig_cond.text[orig_cond.target_start:orig_cond.target_end]
                    if state_cond.text[state_cond.target_start:state_cond.target_end] != target: continue
                    if unrel_cond.text[unrel_cond.target_start:unrel_cond.target_end] != target: continue
                    if surf_cond.text[surf_cond.target_start:surf_cond.target_end] != target: continue
                    def loc(c):
                        lo=max(0,c.target_start-70); hi=min(len(c.text),c.target_end+70)
                        return c.text[lo:hi]
                    if loc(orig_cond)!=loc(state_cond) or loc(orig_cond)!=loc(unrel_cond) or loc(orig_cond)!=loc(surf_cond): continue
                    cid=f'{src}:{di}:{ent}:{y1}:{qj}:{pj}'
                    if cid in seen: continue
                    seen.add(cid)
                    out.append(Case(cid,src,fam,ent,y1,norm(y2surf),rw,norm(rrepl),sw,norm(srepl),pj-qj,pj-rj,len(local_js),orig_cond,state_cond,unrel_cond,surf_cond))
                    if len(out)>=max_cases: return out
                    break
                if len(out)>=max_cases: return out
    return out


def tokpos_for_target(tok, text: str, start: int, end: int, max_len: int):
    enc=tok(text, add_special_tokens=False, truncation=True, max_length=max_len, return_offsets_mapping=True, return_tensors='pt')
    offs=enc.pop('offset_mapping')[0].tolist()
    pos=[i for i,(s,e) in enumerate(offs) if e>start and s<end]
    if len(pos)!=1:
        return None, None
    return enc, pos[0]


def condition_score(model, tok, cond: Cond, y1: str, y2: str, max_len: int, device: str) -> Optional[float]:
    ids1=token_ids(tok,y1); ids2=token_ids(tok,y2)
    if len(ids1)!=1 or len(ids2)!=1: return None
    enc,pos=tokpos_for_target(tok,cond.text,cond.target_start,cond.target_end,max_len)
    if enc is None: return None
    input_ids=enc['input_ids'].to(device); attn=enc['attention_mask'].to(device)
    input_ids[0,pos]=tok.mask_token_id
    with torch.no_grad():
        logits=model(input_ids=input_ids, attention_mask=attn).logits[0,pos]
        logp=F.log_softmax(logits,dim=-1)
        return float(logp[ids1[0]]-logp[ids2[0]])


def evaluate_model(name: str, path: pathlib.Path, cases: list[Case], max_len: int, device: str) -> dict:
    tok=AutoTokenizer.from_pretrained(path, use_fast=True, trust_remote_code=True)
    model=AutoModelForMaskedLM.from_pretrained(path, trust_remote_code=True).to(device).eval()
    rows=[]
    for c in cases:
        so=condition_score(model,tok,c.orig,c.y1,c.y2,max_len,device)
        ss=condition_score(model,tok,c.state_cf,c.y1,c.y2,max_len,device)
        su=condition_score(model,tok,c.unrelated_cf,c.y1,c.y2,max_len,device)
        sf=condition_score(model,tok,c.surface_cf,c.y1,c.y2,max_len,device)
        if None in (so,ss,su,sf): continue
        rows.append({'case_id':c.case_id,'family':c.family,'score_orig':so,'score_state_cf':ss,'score_unrelated_cf':su,'score_surface_cf':sf,
                     'R_state':so-ss,'R_unrelated':so-su,'R_surface':so-sf,'R_extra':(so-ss)-(so-su)})
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    def mean(key, subset=None):
        rr=rows if subset is None else [r for r in rows if r['family']==subset]
        return float(np.mean([r[key] for r in rr])) if rr else None
    def sem(key):
        x=np.asarray([r[key] for r in rows],dtype='float64')
        return float(x.std(ddof=0)/math.sqrt(max(1,len(x)))) if len(x) else None
    fams=sorted(set(r['family'] for r in rows))
    return {'num_used':len(rows),'mean_score_orig':mean('score_orig'),'R_state_mean':mean('R_state'),'R_unrelated_mean':mean('R_unrelated'),'R_surface_mean':mean('R_surface'),'R_extra_mean':mean('R_extra'),'R_extra_sem':sem('R_extra'),'frac_R_extra_positive':float(np.mean([r['R_extra']>0 for r in rows])) if rows else None,'by_family':{f:{'n':sum(1 for r in rows if r['family']==f),'R_extra_mean':mean('R_extra',f),'R_state_mean':mean('R_state',f),'R_unrelated_mean':mean('R_unrelated',f)} for f in fams},'rows':rows[:50]}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--raw_dir',default=str(RAW_DEFAULT))
    ap.add_argument('--max_docs',type=int,default=3000)
    ap.add_argument('--max_cases',type=int,default=300)
    ap.add_argument('--max_len',type=int,default=192)
    ap.add_argument('--seed',type=int,default=275)
    ap.add_argument('--inventory_jsonl',default=str(OUT_INV_DEFAULT))
    ap.add_argument('--summary_json',default=str(OUT_SUM_DEFAULT))
    ap.add_argument('--results_json',default=str(OUT_RES_DEFAULT))
    ap.add_argument('--note',default=str(OUT_NOTE_DEFAULT))
    ap.add_argument('--models',nargs='*',default=[
        'wwm43_80M=experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model/chck_80M',
        'wwm43_100M=experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model/chck_100M',
        'protected42_100M=experiments/archive/initial_model_studies/training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/hf_model/chck_100M',
    ])
    args=ap.parse_args(); setup_env(); t0=time.time(); device='cuda' if torch.cuda.is_available() else 'cpu'
    cases=build_inventory(pathlib.Path(args.raw_dir),args.max_docs,args.max_cases,args.seed,BASE_TOK_PATH)
    pathlib.Path(args.inventory_jsonl).parent.mkdir(parents=True,exist_ok=True)
    with pathlib.Path(args.inventory_jsonl).open('w',encoding='utf-8') as f:
        for c in cases:
            f.write(json.dumps(asdict(c),ensure_ascii=False)+'\n')
    fam_counts=collections.Counter(c.family for c in cases)
    summary={'status':'STATE_COUNTERFACTUAL_INVENTORY','num_cases':len(cases),'families':dict(fam_counts),'mean_early_gap_tokens':float(np.mean([c.early_gap_tokens for c in cases])) if cases else None,'mean_unrelated_gap_tokens':float(np.mean([c.unrelated_gap_tokens for c in cases])) if cases else None,'unique_entities':len(set(c.entity for c in cases)),'unique_y1':len(set(c.y1 for c in cases)),'inventory_jsonl':args.inventory_jsonl}
    pathlib.Path(args.summary_json).write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    if len(cases)<20:
        raise RuntimeError(f'Too few cases: {len(cases)}')
    results={'status':'STATE_COUNTERFACTUAL_RANKING','inventory_summary':summary,'models':{},'elapsed_sec':None,'validity_note':'single-token y1/y2; separate offsets; later target and local context verified identical across original/state_cf/unrelated_cf/surface_cf'}
    for spec in args.models:
        name,path=spec.split('=',1); p=pathlib.Path(path)
        if not p.exists(): continue
        results['models'][name]=evaluate_model(name,p,cases,args.max_len,device)
        pathlib.Path(args.results_json).parent.mkdir(parents=True,exist_ok=True)
        pathlib.Path(args.results_json).write_text(json.dumps(results,indent=2,ensure_ascii=False)+'\n')
    results['elapsed_sec']=time.time()-t0
    pathlib.Path(args.results_json).write_text(json.dumps(results,indent=2,ensure_ascii=False)+'\n')
    lines=['# research state-counterfactual ranking results','',f'Inventory JSONL: `{args.inventory_jsonl}`',f'Summary JSON: `{args.summary_json}`',f'Results JSON: `{args.results_json}`','',f"Cases: {len(cases)}; families: {dict(fam_counts)}; mean early gap {summary['mean_early_gap_tokens']:.1f}; mean unrelated gap {summary['mean_unrelated_gap_tokens']:.1f}",'', '| model | n | score_orig | R_state | R_unrelated | R_surface | R_extra | SEM | frac R_extra>0 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,d in results['models'].items():
        lines.append(f"| {name} | {d['num_used']} | {d['mean_score_orig']:.4f} | {d['R_state_mean']:+.4f} | {d['R_unrelated_mean']:+.4f} | {d['R_surface_mean']:+.4f} | {d['R_extra_mean']:+.4f} | {d['R_extra_sem']:.4f} | {d['frac_R_extra_positive']:.3f} |")
    lines += ['','## By family: R_extra mean','', '| model | family | n | R_extra | R_state | R_unrelated |','|---|---|---:|---:|---:|---:|']
    for name,d in results['models'].items():
        for fam,fd in d['by_family'].items():
            lines.append(f"| {name} | {fam} | {fd['n']} | {fd['R_extra_mean']:+.4f} | {fd['R_state_mean']:+.4f} | {fd['R_unrelated_mean']:+.4f} |")
    lines += ['','Interpretation: positive R_extra means the early state-changing fact changes y1/y2 ranking more than a matched unrelated-history edit. This must be positive and robust before a delayed state-contrastive pretraining objective is justified.']
    pathlib.Path(args.note).parent.mkdir(parents=True,exist_ok=True)
    pathlib.Path(args.note).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'inventory':args.inventory_jsonl,'summary':args.summary_json,'results':args.results_json,'note':args.note,'num_cases':len(cases),'elapsed_sec':results['elapsed_sec']},indent=2))

if __name__=='__main__': main()
