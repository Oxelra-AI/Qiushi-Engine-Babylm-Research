#!/usr/bin/env python3
"""Build high-precision research counterfactual exchange cases.

This builder is deliberately narrower than `counterfactual_exchange_probe.py`.
It stops the research-129 lexical-pair tightening loop and constructs a different
object: fully debitable, rule-certified relation counterfactuals grounded in
attested corpus relation sentences.  Each case keeps the same two alternatives
A/B and flips only the controlled relation, so the correct answer exchanges.

The generated contexts are *not* used for training here.  If later used for any
model update, every generated word-pass in `word_count_pair` must be counted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer

USER_ROOT = Path('.').resolve()
DEFAULT_CORPUS = USER_ROOT / 'experiments/archive/representation_and_objectives/data/fw_full_arms/fw_preserved_compact_view_10M.jsonl'
DEFAULT_TOKENIZER = USER_ROOT / 'experiments/archive/representation_and_objectives/data/shared_tokenizer/shared_16k_tokenizer'
DEFAULT_OUT = USER_ROOT / 'experiments/archive/representation_and_objectives/data/counterfactual_exchange_probe_v3/counterfactual_exchange_cases.jsonl'
DEFAULT_NOTE = USER_ROOT / 'research/notes/representation_and_objectives/counterfactual_exchange_v3_build.md'

WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]{1,}")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+")

STOP = {
    'a','an','the','and','or','but','if','then','else','so','because','although','though','while','when','where','before','after',
    'of','for','to','from','with','without','by','as','at','in','on','into','onto','under','over','above','below','beneath','inside','outside','behind','between','among','around','through','toward','towards','than','about','against','within','during','near','across','along','off','up','down','out',
    'is','are','was','were','be','been','being','am','do','does','did','have','has','had','having','can','could','will','would','should','may','might','must','shall','let','lets','made','make','makes','go','goes','went','come','came','get','got','said','say','says','told','tell','know','think','look','see','seen','use','used','using','help','helps','want','wants','need','needs','try','tried','turn','turns','move','moves','walk','walks','take','takes','took','find','found','meet','meets','choose','chooses','wish','wanna',
    'this','that','these','those','there','here','it','its','they','them','their','we','us','our','you','your','yours','yourself','he','him','his','she','her','i','me','my','who','whom','whose','what','which','why','how',
    'one','ones','thing','things','something','anything','everything','someone','everyone','people','person','way','time','day','year','years','month','months','week','weeks','hour','hours','minute','minutes','part','parts','kind','sort','same','other','another','first','last','new','old','good','bad','great','small','large','big','little','many','much','more','less','most','least','very','really','actually','just','only','also','well','right','left','ok','okay','yes','no','not','some','any','all','each','every','both','either','neither','few','several','own','such','rather','quite','often','typically','usually','always','never','ever','already','still','far','now',
    'chi','mot','fat','mar','bro','sis','mom','dad','par','inv','int','exp','add','com','act','pho','sit','gra','gpx','xxx','xx','uh','uhhuh','huh','hm','yeah','ooh','ah','oh','mhm'
}
BAD = STOP | {
    'baby','child','children','man','woman','men','women','boy','girl','place','area','system','court','government','city','state','country','world','school','company','family','group','home','house','room','water','food','work','life','name','number','example','problem','question','answer','story','book','line','point','case','side','end','start','percent','rate','level','levels','type','types','bit','half','pay','pick','grew','going','leaving','knocked','fought','dead','crucial','different','strong','flat','front','rear','hair'
}
VERBISH_SUFFIXES = ('ed','ing','ly')
MONTHS = {'january','february','march','april','may','june','july','august','september','october','november','december'}
DETERMINERS = {'the','a','an','this','that','these','those','his','her','its','their','our','your','my'}
PREP_OR_DET = DETERMINERS | {'of','for','with','without','in','on','at','by','from','to'}
SCALAR_OPPOSITE = {
    'higher':'lower','lower':'higher','larger':'smaller','smaller':'larger','greater':'lesser','lesser':'greater',
    'older':'younger','younger':'older','better':'worse','worse':'better','faster':'slower','slower':'faster',
    'longer':'shorter','shorter':'longer','stronger':'weaker','weaker':'stronger'
}
MORE_LESS_ADJ = {'important','common','likely','popular','powerful','useful','difficult','expensive','effective','efficient','complex','simple','serious','successful','dangerous','stable','active','specific','compact','severe'}


def now() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def norm(w: str) -> str:
    return re.sub(r"[^A-Za-z'\-]", '', w).strip("-'_").lower()


def tok_words(sent: str) -> list[dict[str, Any]]:
    rows=[]
    for m in WORD_RE.finditer(sent):
        surf=m.group(0); nw=norm(surf)
        rows.append({'surf':surf,'norm':nw,'start':m.start(),'end':m.end(),'title':surf[:1].isupper()})
    return rows


def split_sents(text: str) -> list[str]:
    text=' '.join(text.replace('\n',' ').split())
    out=[]
    for s in SENT_SPLIT_RE.split(text):
        s=s.strip()
        if 25 <= len(s) <= 260:
            out.append(s)
    return out


def wc(s: str) -> int:
    return len(re.findall(r"\b\S+\b", s))


def token_ids(tokenizer: Any, w: str) -> list[int]:
    return [int(x) for x in tokenizer(w, add_special_tokens=False)['input_ids']]


def freq_bin(c: int) -> str:
    if c <= 1: return '1'
    if c <= 3: return '2-3'
    if c <= 7: return '4-7'
    if c <= 15: return '8-15'
    if c <= 31: return '16-31'
    if c <= 63: return '32-63'
    if c <= 127: return '64-127'
    if c <= 255: return '128-255'
    if c <= 511: return '256-511'
    if c <= 1023: return '512-1023'
    return '1000+'


def candidate_class(words: list[dict[str, Any]], idx: int) -> str | None:
    if idx < 0 or idx >= len(words):
        return None
    w=words[idx]; nw=w['norm']
    if len(nw) < 3 or len(nw) > 18 or nw in BAD or nw in MONTHS:
        return None
    if not re.fullmatch(r"[a-z][a-z'\-]*", nw):
        return None
    if nw.endswith(VERBISH_SUFFIXES) or nw.endswith("n't") or nw.endswith("'s"):
        return None
    prev=words[idx-1]['norm'] if idx > 0 else ''
    nxt=words[idx+1]['norm'] if idx + 1 < len(words) else ''
    # High precision, no learned tagger: accept only surface clues that are
    # noun/entity-like.  This sacrifices density to avoid templates like
    # "trees caused knocked" or "pick is older than train".
    if w['title'] and idx > 0:
        return 'proper'
    if nw.endswith('s') and nw not in {'was','has','does'}:
        return 'plural'
    if prev in DETERMINERS:
        return 'det_noun'
    if nxt in {'of','with','in','on','for','from'} and nw not in STOP:
        return 'head_noun'
    return None


def nearest_left(words: list[dict[str, Any]], idx: int, max_gap: int) -> tuple[int, str] | None:
    for j in range(idx-1, max(-1, idx-max_gap-1), -1):
        cls=candidate_class(words,j)
        if cls:
            return j, cls
    return None


def nearest_right(words: list[dict[str, Any]], idx: int, max_gap: int, start_offset: int = 1) -> tuple[int, str] | None:
    for j in range(idx+start_offset, min(len(words), idx+max_gap+1)):
        cls=candidate_class(words,j)
        if cls:
            return j, cls
    return None


def sent_allowed(sent: str) -> bool:
    lo=sent.lower()
    if '*' in sent or 'xxx' in lo or re.search(r'\b(chi|mot|fat|sis|bro|inv|exp)\s*:', lo):
        return False
    return True


def build_freq(corpus: Path, max_rows: int = 0) -> Counter:
    freq=Counter(); rows=0
    with corpus.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            obj=json.loads(line); rows += 1
            for w in tok_words(str(obj.get('text',''))):
                if w['norm']:
                    freq[w['norm']] += 1
            if max_rows and rows >= max_rows:
                break
    return freq


def add_case(cases: list[dict[str, Any]], seen: set[str], tokenizer: Any, args: argparse.Namespace, *, family: str, row: int, sent: str, pivot: str, words: list[dict[str, Any]], ia: int, ib: int, rel0: str, rel1: str, ctx0: str, ctx1: str, target0: str, target1: str, freq: Counter, extra: dict[str, Any]) -> bool:
    a=words[ia]['norm']; b=words[ib]['norm']
    if a == b or target0 == target1:
        return False
    cls_a=candidate_class(words, ia); cls_b=candidate_class(words, ib)
    if cls_a is None or cls_b is None:
        return False
    # proper/plural/determiner/head classes are only rough, but enforcing the
    # broad surface type makes alternatives more plausible than arbitrary tokens.
    broad_a='proper' if cls_a=='proper' else 'common'
    broad_b='proper' if cls_b=='proper' else 'common'
    if broad_a != broad_b:
        return False
    if abs(math.log1p(freq[a]) - math.log1p(freq[b])) > args.max_logfreq_delta:
        return False
    ids0=token_ids(tokenizer, target0); ids1=token_ids(tokenizer, target1)
    if not ids0 or len(ids0) != len(ids1) or len(ids0) > args.max_target_pieces:
        return False
    mask=' '.join([tokenizer.mask_token or '<mask>']*len(ids0))
    c0=ctx0.format(A=a, B=b, MASK=mask)
    c1=ctx1.format(A=a, B=b, MASK=mask)
    if c0 == c1:
        return False
    case={
        'case_id':'', 'family':family, 'source_row':row, 'sentence':sent, 'attested_pivot':pivot,
        'arg_a':a, 'arg_b':b, 'arg_a_surface':words[ia]['surf'], 'arg_b_surface':words[ib]['surf'],
        'arg_class_a':cls_a, 'arg_class_b':cls_b, 'broad_arg_class':broad_a,
        'target0':target0, 'target1':target1, 'target0_ids':ids0, 'target1_ids':ids1, 'target_token_len':len(ids0), 'cap_class':broad_a,
        'relation0':rel0, 'relation1':rel1, 'context0':c0, 'context1':c1,
        'word_count_context0':wc(c0), 'word_count_context1':wc(c1), 'word_count_pair':wc(c0)+wc(c1),
        'freq_a':freq[a], 'freq_b':freq[b], 'freq_bin_a':freq_bin(freq[a]), 'freq_bin_b':freq_bin(freq[b]),
        'certification':'strict grammar: attested relation pivot with two noun/entity-like arguments; same A/B alternatives; generated relation flip exchanges correct target; generated word-passes fully debit if trained',
    }
    case.update(extra)
    raw='|'.join([family,a,b,c0,c1,target0,target1])
    h=hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]
    if h in seen:
        return False
    seen.add(h); case['case_id']='cfv3_'+h; cases.append(case)
    return True


def build(args: argparse.Namespace) -> dict[str, Any]:
    tokenizer=AutoTokenizer.from_pretrained(str(args.tokenizer), use_fast=True)
    freq=build_freq(args.corpus, args.freq_rows)
    cases=[]; seen=set(); rows=0; sent_count=0; rejects=Counter()
    with args.corpus.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            rows += 1
            obj=json.loads(line)
            for sent in split_sents(str(obj.get('text',''))):
                if not sent_allowed(sent):
                    rejects['speaker_or_xxx'] += 1; continue
                sent_count += 1
                words=tok_words(sent); lows=[w['norm'] for w in words]
                for i,piv in enumerate(lows):
                    if piv in {'above','over','below','under','beneath'}:
                        left=nearest_left(words,i,args.max_gap); right=nearest_right(words,i,args.max_gap)
                        if left and right:
                            ia,_=left; ib,_=right
                            if piv in {'above','over'}:
                                add_case(cases,seen,tokenizer,args,family='spatial_vertical',row=rows,sent=sent,pivot=piv,words=words,ia=ia,ib=ib,rel0='A_above_B_higher_A',rel1='A_below_B_higher_B',ctx0='{A} is above {B}. The higher one is {MASK}.',ctx1='{A} is below {B}. The higher one is {MASK}.',target0=words[ia]['norm'],target1=words[ib]['norm'],freq=freq,extra={'pivot_index':i,'left_gap':i-ia,'right_gap':ib-i})
                            else:
                                add_case(cases,seen,tokenizer,args,family='spatial_vertical',row=rows,sent=sent,pivot=piv,words=words,ia=ia,ib=ib,rel0='A_below_B_higher_B',rel1='A_above_B_higher_A',ctx0='{A} is below {B}. The higher one is {MASK}.',ctx1='{A} is above {B}. The higher one is {MASK}.',target0=words[ib]['norm'],target1=words[ia]['norm'],freq=freq,extra={'pivot_index':i,'left_gap':i-ia,'right_gap':ib-i})
                    if piv in {'before','after'}:
                        left=nearest_left(words,i,args.max_gap); right=nearest_right(words,i,args.max_gap)
                        if left and right:
                            ia,_=left; ib,_=right
                            if piv == 'before':
                                add_case(cases,seen,tokenizer,args,family='temporal_order',row=rows,sent=sent,pivot=piv,words=words,ia=ia,ib=ib,rel0='A_before_B_earlier_A',rel1='A_after_B_earlier_B',ctx0='{A} happened before {B}. The earlier one was {MASK}.',ctx1='{A} happened after {B}. The earlier one was {MASK}.',target0=words[ia]['norm'],target1=words[ib]['norm'],freq=freq,extra={'pivot_index':i,'left_gap':i-ia,'right_gap':ib-i})
                            else:
                                add_case(cases,seen,tokenizer,args,family='temporal_order',row=rows,sent=sent,pivot=piv,words=words,ia=ia,ib=ib,rel0='A_after_B_earlier_B',rel1='A_before_B_earlier_A',ctx0='{A} happened after {B}. The earlier one was {MASK}.',ctx1='{A} happened before {B}. The earlier one was {MASK}.',target0=words[ib]['norm'],target1=words[ia]['norm'],freq=freq,extra={'pivot_index':i,'left_gap':i-ia,'right_gap':ib-i})
                    if piv in SCALAR_OPPOSITE:
                        try:
                            than_idx=lows.index('than', i+1, min(len(lows), i+5))
                        except ValueError:
                            than_idx=-1
                        if than_idx > 0:
                            left=nearest_left(words,i,args.max_gap); right=nearest_right(words,than_idx,args.max_gap)
                            if left and right:
                                ia,_=left; ib,_=right; opp=SCALAR_OPPOSITE[piv]
                                add_case(cases,seen,tokenizer,args,family='comparative_scalar',row=rows,sent=sent,pivot=piv,words=words,ia=ia,ib=ib,rel0=f'A_{piv}_B_{piv}_A',rel1=f'A_{opp}_B_{piv}_B',ctx0='{A} is '+piv+' than {B}. The '+piv+' one is {MASK}.',ctx1='{A} is '+opp+' than {B}. The '+piv+' one is {MASK}.',target0=words[ia]['norm'],target1=words[ib]['norm'],freq=freq,extra={'pivot_index':i,'than_index':than_idx,'left_gap':i-ia,'right_gap':ib-than_idx,'opposite':opp})
                    if piv in {'more','less'} and i+2 < len(words) and lows[i+1] in MORE_LESS_ADJ:
                        try:
                            than_idx=lows.index('than', i+2, min(len(lows), i+7))
                        except ValueError:
                            than_idx=-1
                        if than_idx > 0:
                            left=nearest_left(words,i,args.max_gap); right=nearest_right(words,than_idx,args.max_gap)
                            if left and right:
                                ia,_=left; ib,_=right; adj=lows[i+1]; opp='less' if piv=='more' else 'more'
                                ask=piv+' '+adj
                                add_case(cases,seen,tokenizer,args,family='comparative_more_less',row=rows,sent=sent,pivot=piv+'_'+adj,words=words,ia=ia,ib=ib,rel0=f'A_{piv}_{adj}_B_{piv}_{adj}_A',rel1=f'A_{opp}_{adj}_B_{piv}_{adj}_B',ctx0='{A} is '+piv+' '+adj+' than {B}. The '+ask+' one is {MASK}.',ctx1='{A} is '+opp+' '+adj+' than {B}. The '+ask+' one is {MASK}.',target0=words[ia]['norm'],target1=words[ib]['norm'],freq=freq,extra={'pivot_index':i,'than_index':than_idx,'left_gap':i-ia,'right_gap':ib-than_idx,'opposite':opp+'_'+adj})
                    if piv in {'caused','causes'}:
                        left=nearest_left(words,i,args.max_gap); right=nearest_right(words,i,args.max_gap)
                        if left and right:
                            ia,_=left; ib,_=right
                            add_case(cases,seen,tokenizer,args,family='causal_direction',row=rows,sent=sent,pivot=piv,words=words,ia=ia,ib=ib,rel0='A_caused_B_cause_A',rel1='A_caused_by_B_cause_B',ctx0='{A} caused {B}. The cause was {MASK}.',ctx1='{A} was caused by {B}. The cause was {MASK}.',target0=words[ia]['norm'],target1=words[ib]['norm'],freq=freq,extra={'pivot_index':i,'left_gap':i-ia,'right_gap':ib-i})
            if rows % 10000 == 0:
                print(json.dumps({'event':'build_progress','rows':rows,'cases':len(cases),'utc':now()}), flush=True)
            if args.max_rows and rows >= args.max_rows:
                break
    # optional family-balanced cap
    if args.max_cases and len(cases) > args.max_cases:
        buckets=defaultdict(list)
        for c in cases:
            buckets[c['family']].append(c)
        kept=[]; quota=max(1, args.max_cases // max(1,len(buckets)))
        for fam in sorted(buckets):
            kept.extend(buckets[fam][:quota])
        rest=[c for c in cases if c not in kept]
        kept.extend(rest[:max(0,args.max_cases-len(kept))])
        cases=kept[:args.max_cases]
    # target-pair permutation donors within family/tokenlen/class/freq bins
    strata=defaultdict(list)
    for idx,c in enumerate(cases):
        strata[(c['family'], c['target_token_len'], c['cap_class'])].append(idx)
    donor_missing=0
    for key, idxs in strata.items():
        if len(idxs) < 2:
            for i in idxs:
                cases[i]['target_perm_case_id']=None
                donor_missing += 1
            continue
        for n,i in enumerate(idxs):
            donor=idxs[(n+1)%len(idxs)]
            # find a donor with distinct targets if possible
            for k in range(1,len(idxs)+1):
                cand=idxs[(n+k)%len(idxs)]
                if cases[i]['target0'] not in {cases[cand]['target0'],cases[cand]['target1']} and cases[i]['target1'] not in {cases[cand]['target0'],cases[cand]['target1']}:
                    donor=cand; break
            d=cases[donor]
            cases[i]['target_perm_case_id']=d['case_id']
            cases[i]['target_perm_target0']=d['target0']; cases[i]['target_perm_target1']=d['target1']
            cases[i]['target_perm_target0_ids']=d['target0_ids']; cases[i]['target_perm_target1_ids']=d['target1_ids']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False)+'\n')
    summary={
        'status':'COUNTERFACTUAL_EXCHANGE_V3_BUILD', 'created_utc':now(), 'corpus':str(args.corpus), 'tokenizer':str(args.tokenizer),
        'rows_seen':rows, 'sentences_seen':sent_count, 'cases':len(cases), 'by_family':dict(Counter(c['family'] for c in cases)),
        'by_arg_class':dict(Counter(c['broad_arg_class'] for c in cases)), 'target_token_len':dict(Counter(str(c['target_token_len']) for c in cases)),
        'top_pivots':Counter(c['attested_pivot'] for c in cases).most_common(40), 'donor_missing':donor_missing,
        'generated_pair_word_count_total':int(sum(c['word_count_pair'] for c in cases)),
        'generated_pair_word_count_mean':float(np.mean([c['word_count_pair'] for c in cases])) if cases else None,
        'cases_path':str(args.output), 'rejects':dict(rejects),
        'certification':'narrow hand-coded grammar with no pretrained parser; skips dialogue markup; noun/entity-like alternatives only; relation flip exchanges target among same A/B alternatives; all generated word-passes debit if trained',
    }
    (args.output.parent/'counterfactual_exchange_build_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    sample_path=args.output.parent/'counterfactual_exchange_case_samples.jsonl'
    with sample_path.open('w', encoding='utf-8') as f:
        for c in cases[:80]:
            f.write(json.dumps({k:c[k] for k in ['case_id','family','sentence','arg_a','arg_b','context0','target0','context1','target1','freq_bin_a','freq_bin_b','broad_arg_class']}, ensure_ascii=False)+'\n')
    note=[]
    note.append('# research — counterfactual exchange v3 build\n')
    note.append(f'Created: {summary["created_utc"]}\n')
    note.append(f'Rows seen: {rows}; sentences: {sent_count}; cases: **{len(cases)}**.\n')
    note.append(f'By family: `{summary["by_family"]}`.\n')
    note.append(f'Generated pair words if used once for training: **{summary["generated_pair_word_count_total"]}**.\n')
    note.append('This is frozen-probe material only in research; training would require full exposure accounting.\n')
    note.append(f'Cases: `{args.output}`; samples: `{sample_path}`; build summary: `{args.output.parent/"counterfactual_exchange_build_summary.json"}`\n')
    args.note.parent.mkdir(parents=True, exist_ok=True)
    args.note.write_text(''.join(note), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    ap=argparse.ArgumentParser()
    ap.add_argument('--corpus', type=Path, default=DEFAULT_CORPUS)
    ap.add_argument('--tokenizer', type=Path, default=DEFAULT_TOKENIZER)
    ap.add_argument('--output', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--note', type=Path, default=DEFAULT_NOTE)
    ap.add_argument('--max-rows', type=int, default=0)
    ap.add_argument('--freq-rows', type=int, default=0)
    ap.add_argument('--max-cases', type=int, default=6000)
    ap.add_argument('--max-target-pieces', type=int, default=1)
    ap.add_argument('--max-logfreq-delta', type=float, default=2.5)
    ap.add_argument('--max-gap', type=int, default=6)
    return ap.parse_args()


def main() -> None:
    build(parse_args())


if __name__ == '__main__':
    main()
