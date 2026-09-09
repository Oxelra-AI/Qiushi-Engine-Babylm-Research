#!/usr/bin/env python3
"""Audit baseline16k seq256 visibility for the research seqsafe96 FineWeb candidate."""
from __future__ import annotations
import json, pathlib, statistics, argparse
from typing import Any
from transformers import AutoTokenizer

TOKENIZER_DEFAULT = "experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model"
TREAT_DEFAULT = pathlib.Path("experiments/archive/representation_and_objectives/training/data/cached_fineweb_seqsafe96_candidate/cleanqwen_seqsafe_fineweb_single_doc_10M.jsonl")
CTRL_DEFAULT = pathlib.Path("experiments/archive/representation_and_objectives/training/data/cached_fineweb_seqsafe96_candidate/cleanqwen_official_lengthmatched_seqsafe_control_10M.jsonl")
OUT_DEFAULT = pathlib.Path("experiments/archive/representation_and_objectives/training/data/cached_fineweb_seqsafe96_candidate/seq256_visibility_audit.json")
NOTE_DEFAULT = pathlib.Path("research/notes/representation_and_objectives/cached_fineweb_seqsafe96_visibility.md")


def stat(vals):
    xs=sorted(vals)
    if not xs: return {"n":0}
    def q(p): return xs[min(len(xs)-1, max(0, round((len(xs)-1)*p)))]
    return {"n":len(xs),"min":xs[0],"p05":q(0.05),"mean":statistics.mean(xs),"median":statistics.median(xs),"p95":q(0.95),"p99":q(0.99),"max":xs[-1]}


def token_len(tok,text):
    return len(tok(text, add_special_tokens=False, truncation=False)["input_ids"])


def visible_words(tok, words, seq_len):
    if not words: return 0
    if token_len(tok, " ".join(words)) <= seq_len: return len(words)
    lo, hi = 0, len(words)
    while lo < hi:
        mid = (lo + hi + 1)//2
        if token_len(tok, " ".join(words[:mid])) <= seq_len: lo=mid
        else: hi=mid-1
    return lo


def audit(path: pathlib.Path, tok, seq_len: int, prefixes: list[str]) -> dict[str, Any]:
    rows=0; words=0; trunc=0; visible=0; lens=[]; wlens=[]
    focus={p:{"rows":0,"words":0,"truncated_rows":0,"visible_words":0,"token_lens":[],"examples":[]} for p in prefixes}
    with path.open(encoding='utf-8') as f:
        for i,line in enumerate(f):
            if not line.strip(): continue
            o=json.loads(line); text=o['text']; w=int(o.get('words',len(text.split()))); source=str(o.get('source',''))
            tl=token_len(tok,text); v=w if tl<=seq_len else visible_words(tok,text.split(),seq_len)
            rows+=1; words+=w; lens.append(tl); wlens.append(w); visible+=v
            if tl>seq_len: trunc+=1
            for p in prefixes:
                if source.startswith(p):
                    r=focus[p]; r['rows']+=1; r['words']+=w; r['visible_words']+=v; r['token_lens'].append(tl)
                    if tl>seq_len:
                        r['truncated_rows']+=1
                        if len(r['examples'])<4:
                            r['examples'].append({'row':i,'source':source,'words':w,'tokens':tl,'visible_words':v,'text_excerpt':text[:500]})
    foc={}
    for p,r in focus.items():
        tl=r.pop('token_lens')
        r['truncated_fraction']=r['truncated_rows']/r['rows'] if r['rows'] else None
        r['visible_word_fraction']=r['visible_words']/r['words'] if r['words'] else None
        r['hidden_words']=r['words']-r['visible_words']
        r['token_length_stats']=stat(tl)
        foc[p]=r
    return {"path":str(path),"rows":rows,"words":words,"truncated_rows":trunc,"truncated_fraction":trunc/rows if rows else None,"visible_words":visible,"visible_word_fraction":visible/words if words else None,"hidden_words":words-visible,"token_length_stats":stat(lens),"word_length_stats":stat(wlens),"focus_by_source_prefix":foc}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--tokenizer',default=TOKENIZER_DEFAULT); ap.add_argument('--seq-len',type=int,default=256); ap.add_argument('--out',default=str(OUT_DEFAULT)); ap.add_argument('--note',default=str(NOTE_DEFAULT)); args=ap.parse_args()
    tok=AutoTokenizer.from_pretrained(args.tokenizer,use_fast=True)
    prefixes=['fineweb_edu_random_quality_single_doc_seqsafe_cached_initial_model_studies','official_lengthmatched_to_seqsafe_fineweb','qwen_pair_packed','official_identical_tail_after_seqsafe_fineweb_block']
    results={'treatment':audit(TREAT_DEFAULT,tok,args.seq_len,prefixes),'control':audit(CTRL_DEFAULT,tok,args.seq_len,prefixes)}
    fw=results['treatment']['focus_by_source_prefix']['fineweb_edu_random_quality_single_doc_seqsafe_cached_initial_model_studies']
    ct=results['control']['focus_by_source_prefix']['official_lengthmatched_to_seqsafe_fineweb']
    payload={'status':'SEQSAFE96_CANDIDATE_SEQ256_VISIBILITY','tokenizer':args.tokenizer,'seq_len':args.seq_len,'results':results,'replacement_block_delta':{'fineweb_minus_control_truncated_fraction':fw['truncated_fraction']-ct['truncated_fraction'],'fineweb_minus_control_visible_word_fraction':fw['visible_word_fraction']-ct['visible_word_fraction'],'fineweb_hidden_words':fw['hidden_words'],'control_hidden_words':ct['hidden_words']}}
    out=pathlib.Path(args.out); out.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    lines=['# research seqsafe96 FineWeb candidate seq256 visibility\n\n',f"Tokenizer: `{args.tokenizer}`; seq_len={args.seq_len}.\n\n",'| arm | whole trunc frac | whole visible word frac | focus | rows | words | trunc frac | visible word frac | hidden words | token p95 | token max |\n','|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|\n']
    for arm,res in results.items():
        items=[(k,v) for k,v in res['focus_by_source_prefix'].items() if v['rows']]
        for j,(k,v) in enumerate(items):
            lines.append(f"| {arm if j==0 else ''} | {res['truncated_fraction']:.4f} | {res['visible_word_fraction']:.4f} | {k} | {v['rows']} | {v['words']} | {v['truncated_fraction']:.4f} | {v['visible_word_fraction']:.4f} | {v['hidden_words']} | {v['token_length_stats'].get('p95',0):.1f} | {v['token_length_stats'].get('max',0):.1f} |\n")
    lines.append(f"\nReplacement block delta: FineWeb-control visible word fraction {payload['replacement_block_delta']['fineweb_minus_control_visible_word_fraction']:+.6f}; hidden words FineWeb {fw['hidden_words']}, control {ct['hidden_words']}.\n\nJSON: `{out}`\n")
    note=pathlib.Path(args.note); note.write_text(''.join(lines),encoding='utf-8')
    print(json.dumps({'status':payload['status'],'out':str(out),'note':str(note),'replacement_block_delta':payload['replacement_block_delta']},indent=2,ensure_ascii=False))

if __name__=='__main__': main()
