#!/usr/bin/env python3
"""research strict ordered-prefix target surface.

Repairs research over-closure by adding:
  - same-source local-neighbor negatives using original contiguous example_id order;
  - independent selection/evaluation split: select targets with seed42@80M, evaluate with seed43@100M;
  - fixed suffix/target positions and word-start targets inherited from research v2;
  - decoded representative examples plus simple semantic/artifact tags.

No training. This measures whether an ordered cross-sentence target surface survives
stronger controls before any new objective is built.
"""
from __future__ import annotations
import argparse, csv, json, math, os, pathlib, random, re, statistics, sys, time
from collections import Counter, defaultdict
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT=pathlib.Path('experiments/archive/initial_model_studies')
OUT_ROWS=ROOT/'data/strict_ordered_target_surface_rows.csv'
OUT_SUMMARY=ROOT/'data/strict_ordered_target_surface_summary.json'
OUT_NOTE=(ROOT.parents[2] / 'research/notes/initial_model_studies/strict_ordered_target_surface.md')
sys.path.insert(0,str((ROOT/'scripts').resolve()))
sys.path.insert(0,str((ROOT/'training/scripts').resolve()))
import prefix_target_census_v2_fixedpos as v2

CKPT_SELECT=ROOT/'training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/hf_model/chck_80M'
CKPT_HELDOUT=ROOT/'training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model/chck_100M'
NUM_WORDS=set('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million percent per cent'.split())


def setup_env():
    hf=ROOT/'training/hf_home'
    os.environ['HF_HOME']=str(hf.resolve())
    os.environ['HF_HUB_CACHE']=str((hf/'hub').resolve())
    os.environ['HF_DATASETS_CACHE']=str((hf/'datasets').resolve())
    os.environ['TRANSFORMERS_CACHE']=str((hf/'transformers').resolve())
    os.environ['HF_MODULES_CACHE']=str((ROOT/'training/hf_modules_cache').resolve())
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')


def decode(tok, ids, maxchars=320):
    s=tok.decode(ids, skip_special_tokens=True)
    s=' '.join(s.split())
    return s[-maxchars:]


def make_cases_strict(examples, tok, n, label, rng, wf):
    # First build all candidates with original example_id/source retained.
    cands=[]
    for ex in examples:
        words=ex.text.split()
        if len(words)<70: continue
        split=max(30,min(len(words)-25,len(words)//2))
        pw=words[:split]; lw=words[split:]
        pids=v2.ids(' '.join(pw),tok); lids=v2.ids(' '+' '.join(lw),tok)
        if len(pids)<20 or len(lids)<20: continue
        pids=pids[-min(len(pids),120):]
        lids=lids[:min(len(lids),120)]
        toks=tok.convert_ids_to_tokens(lids)
        poss=[j for j in range(max(5,len(lids)//4),min(len(lids),120)) if lids[j] not in tok.all_special_ids and v2.good(toks[j])]
        if not poss: continue
        j=rng.choice(poss[:min(8,len(poss))])
        s=v2.surf(toks[j])
        cands.append({
            'case_group':label,'example_id':ex.example_id,'source':ex.source,
            'pids':pids,'lids':lids,'target_idx':j,'target_id':lids[j],
            'target_token':toks[j],'target_surface':s,'prefix_tokens':len(pids),'later_tokens':len(lids),
            'target_freq_bucket':v2.freq_bucket(wf.get(s,0)),
            'local_leak_10tok':v2.local_leak(toks,j,s),
            'repeated_capitalized_prefix_suffix':v2.repeated_cap(pw,lw),
            'prefix_text_tail': decode(tok,pids[-100:]),
            'suffix_window': tok.decode(lids[max(0,j-22):min(len(lids),j+30)], skip_special_tokens=True),
        })
    rng.shuffle(cands)
    by_source=defaultdict(list)
    for c in cands:
        by_source[c['source']].append(c)
    for src in by_source:
        by_source[src].sort(key=lambda x:x['example_id'])
    cases=[]
    for c in cands:
        P=len(c['pids'])
        # random same-source and cross-source as in v2
        same=[o for o in cands if o['example_id']!=c['example_id'] and o['source']==c['source'] and abs(len(o['pids'])-P)<=20]
        cross=[o for o in cands if o['example_id']!=c['example_id'] and o['source']!=c['source'] and abs(len(o['pids'])-P)<=25]
        anyo=[o for o in cands if o['example_id']!=c['example_id']]
        if not anyo: continue
        # same-source local neighbor: nearest original contiguous chunk from same source, non-self.
        local_pool=[o for o in by_source[c['source']] if o['example_id']!=c['example_id'] and abs(len(o['pids'])-P)<=35]
        local_pool.sort(key=lambda o:(abs(o['example_id']-c['example_id']), abs(len(o['pids'])-P)))
        if not local_pool:
            local_pool=same if same else anyo
        cc=dict(c)
        cc['same_random_raw']=rng.choice(same if same else anyo)['pids']
        cc['cross_raw']=rng.choice(cross if cross else anyo)['pids']
        cc['same_local_raw']=local_pool[0]['pids']
        cc['same_local_example_id']=local_pool[0]['example_id']
        cc['same_local_id_distance']=abs(local_pool[0]['example_id']-c['example_id'])
        cc['same_random_text_tail']=decode(tok,cc['same_random_raw'][-100:])
        cc['same_local_text_tail']=decode(tok,cc['same_local_raw'][-100:])
        cc['cross_text_tail']=decode(tok,cc['cross_raw'][-100:])
        cases.append(cc)
        if len(cases)>=n: break
    return cases


def block_shuffle(x,rng):
    blocks=[x[i:i+8] for i in range(0,len(x),8)]
    rng.shuffle(blocks)
    return [z for b in blocks for z in b]


def build_sequences(cases,tok):
    rng=random.Random(3198); seqs=[]; metas=[]
    for ci,c in enumerate(cases):
        P=len(c['pids']); pad=tok.pad_token_id
        full_p,full_m=v2.fit_prefix(c['pids'],P,pad)
        del_p,del_m=[pad]*P,[0]*P
        block_p,block_m=v2.fit_prefix(block_shuffle(c['pids'],rng),P,pad)
        same_random_p,same_random_m=v2.fit_prefix(c['same_random_raw'],P,pad)
        same_local_p,same_local_m=v2.fit_prefix(c['same_local_raw'],P,pad)
        cross_p,cross_m=v2.fit_prefix(c['cross_raw'],P,pad)
        variants={
            'full':(full_p,full_m),
            'deleted':(del_p,del_m),
            'block_shuffled':(block_p,block_m),
            'same_random':(same_random_p,same_random_m),
            'same_local':(same_local_p,same_local_m),
            'cross':(cross_p,cross_m),
        }
        for vn,(p,m) in variants.items():
            seq=p+c['lids']; am=m+[1]*len(c['lids']); pos=P+c['target_idx']
            if pos>=len(seq): continue
            seq=list(seq); seq[pos]=tok.mask_token_id
            seqs.append((seq,am)); metas.append((ci,vn,pos,c['target_id']))
    return seqs, metas


def score(model,seqs,metas,device,pad,batch=96):
    out=[None]*len(metas)
    for st in range(0,len(seqs),batch):
        chunk=seqs[st:st+batch]; mchunk=metas[st:st+batch]
        L=max(len(s) for s,a in chunk)
        inp=torch.full((len(chunk),L),pad,dtype=torch.long,device=device)
        attn=torch.zeros((len(chunk),L),dtype=torch.long,device=device)
        for i,(s,a) in enumerate(chunk):
            inp[i,:len(s)]=torch.tensor(s,dtype=torch.long,device=device)
            attn[i,:len(a)]=torch.tensor(a,dtype=torch.long,device=device)
        with torch.no_grad():
            lp=torch.log_softmax(model(input_ids=inp, attention_mask=attn).logits, dim=-1)
        for i,(ci,vn,pos,tid) in enumerate(mchunk):
            out[st+i]=float(lp[i,pos,tid].item())
    return out


def tag_case(c):
    target=c['target_surface'].lower()
    prefix=c['prefix_text_tail'].lower()
    suffix=c['suffix_window'].lower()
    tags=[]
    if target in prefix: tags.append('target_in_prefix')
    if target in suffix.replace('[mask]',''): tags.append('target_in_suffix_window')
    if target in NUM_WORDS or any(ch.isdigit() for ch in target): tags.append('numeric_quantity')
    if c['source']=='childes.train.txt': tags.append('childes_dialogue')
    if c['repeated_capitalized_prefix_suffix']: tags.append('repeated_capitalized')
    if c['local_leak_10tok']: tags.append('local_leak')
    if not tags: tags.append('uncategorized_discourse_or_topic')
    return tags


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--n_exact',type=int,default=600)
    ap.add_argument('--n_broad',type=int,default=600)
    ap.add_argument('--batch_size',type=int,default=96)
    ap.add_argument('--threshold',type=float,default=0.05)
    args=ap.parse_args()
    setup_env(); t0=time.time()
    tok=AutoTokenizer.from_pretrained(str(CKPT_SELECT.resolve()), use_fast=True)
    exact=v2.reconstruct_exact_1m(); broad=v2.broad_examples(); wf=v2.wordfreq(exact+broad)
    rng=random.Random(319)
    cases=make_cases_strict(exact,tok,args.n_exact,'exact_1m_slice',rng,wf)+make_cases_strict(broad,tok,args.n_broad,'broad_official',rng,wf)
    seqs,metas=build_sequences(cases,tok)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows=[]
    for i,c in enumerate(cases):
        rows.append({k:v for k,v in c.items() if k not in ['pids','lids','same_random_raw','same_local_raw','cross_raw']} | {'case_id':i,'tags':';'.join(tag_case(c))})
    for label,ckpt in [('select_wwm42_80M',CKPT_SELECT),('heldout_wwm43_100M',CKPT_HELDOUT)]:
        print(json.dumps({'event':'load_model','label':label,'ckpt':str(ckpt)}), flush=True)
        model=AutoModelForMaskedLM.from_pretrained(str(ckpt.resolve()), trust_remote_code=True).to(device).eval()
        vals=score(model,seqs,metas,device,tok.pad_token_id,args.batch_size)
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        sc=defaultdict(dict)
        for val,(ci,vn,pos,tid) in zip(vals,metas): sc[ci][vn]=val
        for ci in range(len(rows)):
            d=sc[ci]
            for vn,val in d.items(): rows[ci][f'{label}_lp_{vn}']=val
            for vn in ['deleted','block_shuffled','same_random','same_local','cross']:
                rows[ci][f'{label}_full_minus_{vn}']=d['full']-d[vn]
    th=args.threshold
    selected=[]; heldout_pass=[]
    selected_local=[]; heldout_local=[]
    for r in rows:
        sel_random=(float(r['select_wwm42_80M_full_minus_same_random'])>=th and float(r['select_wwm42_80M_full_minus_block_shuffled'])>=th)
        sel_local=(float(r['select_wwm42_80M_full_minus_same_local'])>=th and float(r['select_wwm42_80M_full_minus_block_shuffled'])>=th)
        hold_random=(float(r['heldout_wwm43_100M_full_minus_same_random'])>=th and float(r['heldout_wwm43_100M_full_minus_block_shuffled'])>=th)
        hold_local=(float(r['heldout_wwm43_100M_full_minus_same_local'])>=th and float(r['heldout_wwm43_100M_full_minus_block_shuffled'])>=th)
        r['select_ordered_random_pass']=int(sel_random); r['select_ordered_local_pass']=int(sel_local)
        r['heldout_ordered_random_pass']=int(hold_random); r['heldout_ordered_local_pass']=int(hold_local)
        if sel_random: selected.append(r)
        if sel_random and hold_random: heldout_pass.append(r)
        if sel_local: selected_local.append(r)
        if sel_local and hold_local: heldout_local.append(r)
    def counts_by(rows_subset, field):
        c=Counter(r[field] for r in rows_subset)
        return dict(c)
    tag_counter=Counter()
    for r in heldout_local:
        for t in r['tags'].split(';'): tag_counter[t]+=1
    summary={
        'n_cases':len(rows), 'threshold':th,
        'selection_model':'wwm42_80M', 'heldout_model':'wwm43_100M',
        'selected_random_same_and_block':len(selected),
        'selected_random_same_and_block_frac':len(selected)/len(rows),
        'selected_random_heldout_survive':len(heldout_pass),
        'selected_random_heldout_survive_frac_of_all':len(heldout_pass)/len(rows),
        'selected_random_heldout_survive_frac_of_selected':len(heldout_pass)/max(1,len(selected)),
        'selected_local_same_and_block':len(selected_local),
        'selected_local_same_and_block_frac':len(selected_local)/len(rows),
        'selected_local_heldout_survive':len(heldout_local),
        'selected_local_heldout_survive_frac_of_all':len(heldout_local)/len(rows),
        'selected_local_heldout_survive_frac_of_selected':len(heldout_local)/max(1,len(selected_local)),
        'heldout_local_by_group':counts_by(heldout_local,'case_group'),
        'heldout_local_by_source':counts_by(heldout_local,'source'),
        'heldout_local_tags':dict(tag_counter),
        'heldout_local_examples':[{
            'case_id':r['case_id'],'group':r['case_group'],'source':r['source'],'target':r['target_surface'],'tags':r['tags'],
            'local_distance':r['same_local_id_distance'],
            'select_same_local':float(r['select_wwm42_80M_full_minus_same_local']),
            'select_block':float(r['select_wwm42_80M_full_minus_block_shuffled']),
            'heldout_same_local':float(r['heldout_wwm43_100M_full_minus_same_local']),
            'heldout_block':float(r['heldout_wwm43_100M_full_minus_block_shuffled']),
            'prefix_tail':r['prefix_text_tail'], 'same_local_tail':r['same_local_text_tail'],
            'suffix_window':r['suffix_window']
        } for r in sorted(heldout_local, key=lambda x: min(float(x['heldout_wwm43_100M_full_minus_same_local']), float(x['heldout_wwm43_100M_full_minus_block_shuffled'])), reverse=True)[:20]]
    }
    OUT_ROWS.parent.mkdir(parents=True, exist_ok=True)
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with OUT_ROWS.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    payload={'status':'STRICT_ORDERED_TARGET_SURFACE','rows_csv':str(OUT_ROWS),'summary':summary,'elapsed_sec':time.time()-t0}
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    OUT_SUMMARY.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n', encoding='utf-8')
    lines=['# research strict ordered target surface','',f'Rows: `{OUT_ROWS}`',f'Summary: `{OUT_SUMMARY}`','',f"Selection: seed42@80M; heldout: seed43@100M; threshold {th}.",'', '| surface | selected | selected fraction | heldout survivors | survivor fraction of all | survivor fraction of selected |','|---|---:|---:|---:|---:|---:|', f"| random same + block | {summary['selected_random_same_and_block']} | {summary['selected_random_same_and_block_frac']:.4f} | {summary['selected_random_heldout_survive']} | {summary['selected_random_heldout_survive_frac_of_all']:.4f} | {summary['selected_random_heldout_survive_frac_of_selected']:.4f} |", f"| local-neighbor same + block | {summary['selected_local_same_and_block']} | {summary['selected_local_same_and_block_frac']:.4f} | {summary['selected_local_heldout_survive']} | {summary['selected_local_heldout_survive_frac_of_all']:.4f} | {summary['selected_local_heldout_survive_frac_of_selected']:.4f} |", '', 'The local-neighbor surface is the stricter one: it asks whether full prefix beats a nearby contiguous same-source chunk and block-shuffled true prefix, with independent heldout measurement. Representative heldout local survivors are stored in the JSON.']
    OUT_NOTE.parent.mkdir(parents=True, exist_ok=True)
    OUT_NOTE.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'out':str(OUT_SUMMARY),'note':str(OUT_NOTE),'summary':summary}, indent=2)[:7000])

if __name__=='__main__':
    main()
