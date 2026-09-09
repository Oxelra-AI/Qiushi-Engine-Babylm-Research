#!/usr/bin/env python3
"""research: rule-certified counterfactual exchange probe.

Builds and scores a different relation object from the saturated research-129
lexical funnels.  Each item is generated from an attested corpus sentence that
contains a hand-coded relation pivot and two nearby content arguments.  The
scored cloze pair keeps the same two candidate entities/arguments and changes
only the relation world, so the correct answer exchanges between the two
alternatives:

  C0(A,B,R0): ... [MASK] ...   correct target = T0
  C1(A,B,R1): ... [MASK] ...   correct target = T1

The interaction margin
  M = s(C0,T0) + s(C1,T1) - s(C0,T1) - s(C1,T0)
cancels the target main effect and asks whether the model binds the same
alternatives differently under the controlled relation change.  A target-pair
permutation within family/token-shape strata is included as a no-update control.

No model weights are changed.  If the object is ever trained, every generated
counterfactual word-pass must be fully debited against the Strict-Small exposure
budget; this script reports the generated word count for that accounting.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, DebertaV2ForMaskedLM

USER_ROOT = Path('.').resolve()
DEFAULT_CORPUS = USER_ROOT / 'experiments/archive/representation_and_objectives/data/fw_full_arms/fw_preserved_compact_view_10M.jsonl'
DEFAULT_TOKENIZER = USER_ROOT / 'experiments/archive/representation_and_objectives/data/shared_tokenizer/shared_16k_tokenizer'
DEFAULT_CASES = USER_ROOT / 'experiments/archive/representation_and_objectives/data/counterfactual_exchange_probe/counterfactual_exchange_cases.jsonl'
DEFAULT_OUT = USER_ROOT / 'experiments/archive/representation_and_objectives/data/counterfactual_exchange_probe'
DEFAULT_NOTE = USER_ROOT / 'research/notes/representation_and_objectives/counterfactual_exchange_probe.md'
DEFAULT_ARMS = {
    'compact': USER_ROOT / 'experiments/archive/frontier_consolidation/training/runs/fw_compact_view_shared16k_seed43022/hf_model/chck_100M',
    'rowblock': USER_ROOT / 'experiments/archive/frontier_consolidation/training/runs/fw_source_breadth_shared16k_seed43022/hf_model/chck_100M',
    'interleaved': USER_ROOT / 'experiments/archive/representation_and_objectives/training/runs/fw_source_breadth_interleaved_wholesentence_fullbatch_shared16k_seed43022/hf_model/chck_100M',
}

WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]{1,}")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+")

STOP = {
    'a','an','the','and','or','but','if','then','else','so','because','although','though','while','when','where','before','after',
    'of','for','to','from','with','without','by','as','at','in','on','into','onto','under','over','above','below','beneath','inside','outside','behind','between','among','around','through','toward','towards','than','about','against','within','during','near','across','along','off','up','down','out',
    'is','are','was','were','be','been','being','am','do','does','did','have','has','had','having','can','could','will','would','should','may','might','must','shall','let','lets','made','make','makes','go','goes','went','come','came','get','got','said','say','says','told','tell','know','think','look','see','seen','use','used','using',
    'this','that','these','those','there','here','it','its','they','them','their','we','us','our','you','your','he','him','his','she','her','i','me','my','who','whom','whose','what','which','why','how',
    'one','ones','thing','things','something','anything','everything','someone','everyone','people','person','way','time','day','year','years','part','parts','kind','sort','same','other','another','first','last','new','old','good','bad','great','small','large','big','little','many','much','more','less','most','least','very','really','actually','just','only','also','well','right','left','ok','okay','yes','no','not','some','any','all','each','every','both','either','neither','few','several','own','such','rather','quite','often','typically','usually','always','never','ever','already','still',
    'chi','mot','fat','mar','bro','sis','mom','dad','par','inv','int','exp','add','com','act','pho','sit','gra','gpx','xxx','xx','uh','uhhuh','huh','hm','yeah','ooh','ah','oh','mhm'
}
BAD_TARGETS = STOP | {
    'baby','child','children','man','woman','men','women','boy','girl','thing','place','area','system','court','government','city','state','country','world','school','company','family','group','home','house','room','water','food','work','life','name','number','example','problem','question','answer','story','book','line','point','case','side','end','start',
    'find','take','takes','took','meet','meets','choose','chooses','wish','wanna','going','leaving','perform','occur','means','called','using','help','helps','tried','try','turn','move','walk','holding','knocked','fought','dead','crucial','different','significant','precisely','common','majority','easy','always','himself'
}
CONNECTORS = STOP | {'called','named','located','kept','known','found','born','built','created','included','including','called','become','became','held','set','given','based','led','shown','reported'}
VERBISH_SUFFIXES = ('ed', 'ing', 'ly')

FAMILY_PIVOTS = {
    'spatial_vertical': {'above','below','over','under','beneath'},
    'spatial_frontback': {'behind'},
    'temporal_order': {'before','after'},
    'comparative_scalar': {'higher','lower','larger','smaller','greater','older','younger','better','worse','faster','slower','longer','shorter','more','less'},
    'causal_direction': {'caused','causes','cause','because'},
}
SCALAR_ADJ = {'important','common','likely','popular','powerful','useful','difficult','expensive','effective','efficient','beautiful','complex','simple','serious','successful','dangerous','stable','active','open','formal','general','specific','recent','ancient','modern','local','global','dense','compact','severe','rapid','slow','strong','weak','high','low','large','small','old','young','fast'}


def now_utc() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def norm_word(w: str) -> str:
    return re.sub(r"[^A-Za-z'\-]", '', w).strip("-'_").lower()


def word_count(s: str) -> int:
    return len(re.findall(r"\b\S+\b", s))


def is_good_arg(w: str) -> bool:
    nw = norm_word(w)
    if not nw or len(nw) < 3 or len(nw) > 18:
        return False
    if nw in BAD_TARGETS:
        return False
    if not re.fullmatch(r"[a-z][a-z'\-]*", nw):
        return False
    if nw.endswith("n't") or nw.endswith("'s"):
        return False
    # High-precision counterfactual items should use entity/object-like alternatives,
    # not nearby verbs/adverbs that make templates such as "X caused choose".
    if nw.endswith(VERBISH_SUFFIXES):
        return False
    return True


def sentence_split(text: str) -> list[str]:
    out=[]
    for s in SENT_SPLIT_RE.split(text.replace('\n',' ')):
        s=' '.join(s.split()).strip()
        if 20 <= len(s) <= 360:
            out.append(s)
    return out


def tokenize_words(sentence: str) -> list[tuple[str, str, int, int]]:
    rows=[]
    for m in WORD_RE.finditer(sentence):
        surf=m.group(0)
        rows.append((surf, norm_word(surf), m.start(), m.end()))
    return rows


def nearest_left(words: list[tuple[str,str,int,int]], idx: int, max_gap: int = 8) -> tuple[str,str,int] | None:
    for j in range(idx-1, max(-1, idx-max_gap-1), -1):
        surf,nw,_,_=words[j]
        if is_good_arg(nw) and nw not in CONNECTORS:
            return surf, nw, idx-j
    return None


def nearest_right(words: list[tuple[str,str,int,int]], idx: int, max_gap: int = 8, start_offset: int = 1) -> tuple[str,str,int] | None:
    for j in range(idx+start_offset, min(len(words), idx+max_gap+1)):
        surf,nw,_,_=words[j]
        if is_good_arg(nw) and nw not in CONNECTORS:
            return surf, nw, j-idx
    return None


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


def cap_class(surf: str) -> str:
    if surf[:1].isupper():
        return 'title'
    return 'lower'


def make_masks(mask_token: str, n: int) -> str:
    return ' '.join([mask_token] * n)


def token_ids_for(tokenizer: Any, text: str) -> list[int]:
    return [int(x) for x in tokenizer(text, add_special_tokens=False)['input_ids']]


def case_key(case: dict[str, Any]) -> str:
    raw='|'.join([case.get('family',''), case.get('arg_a',''), case.get('arg_b',''), case.get('context0',''), case.get('context1','')])
    return hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]


def maybe_add_case(cases: list[dict[str, Any]], seen: set[str], tokenizer: Any, *, family: str, source_row: int, sentence: str, attested_pivot: str, arg_a_surf: str, arg_b_surf: str, context0_tpl: str, context1_tpl: str, target0: str, target1: str, relation0: str, relation1: str, extra: dict[str, Any] | None = None, max_target_pieces: int = 2) -> bool:
    a=norm_word(arg_a_surf); b=norm_word(arg_b_surf); t0=norm_word(target0); t1=norm_word(target1)
    if not (is_good_arg(a) and is_good_arg(b) and is_good_arg(t0) and is_good_arg(t1)):
        return False
    if a == b or t0 == t1:
        return False
    if cap_class(arg_a_surf) != cap_class(arg_b_surf):
        return False
    ids0=token_ids_for(tokenizer, t0); ids1=token_ids_for(tokenizer, t1)
    if len(ids0) <= 0 or len(ids0) != len(ids1) or len(ids0) > max_target_pieces:
        return False
    masks=make_masks(tokenizer.mask_token or '[MASK]', len(ids0))
    context0=context0_tpl.format(A=a, B=b, MASK=masks)
    context1=context1_tpl.format(A=a, B=b, MASK=masks)
    if context0 == context1:
        return False
    wc0=word_count(context0); wc1=word_count(context1)
    case={
        'case_id': '',
        'family': family,
        'source_row': source_row,
        'sentence': sentence,
        'attested_pivot': attested_pivot,
        'arg_a': a,
        'arg_b': b,
        'target0': t0,
        'target1': t1,
        'target0_ids': ids0,
        'target1_ids': ids1,
        'target_token_len': len(ids0),
        'cap_class': cap_class(arg_a_surf),
        'relation0': relation0,
        'relation1': relation1,
        'context0': context0,
        'context1': context1,
        'word_count_context0': wc0,
        'word_count_context1': wc1,
        'word_count_pair': wc0+wc1,
        'certification': 'attested pivot and two nearby same-case content arguments; generated relation flip changes which visible alternative is correct; all generated words must be debited if trained',
    }
    if extra:
        case.update(extra)
    h=case_key(case)
    if h in seen:
        return False
    case['case_id']='cf_'+h
    seen.add(h); cases.append(case)
    return True


def build_frequency(corpus: Path, max_rows: int = 0) -> Counter:
    freq=Counter(); rows=0
    with corpus.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            obj=json.loads(line); rows += 1
            for _,nw,_,_ in tokenize_words(str(obj.get('text',''))):
                if nw:
                    freq[nw] += 1
            if max_rows and rows >= max_rows:
                break
    return freq


def extract_cases(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    out=args.output_cases
    out.parent.mkdir(parents=True, exist_ok=True)
    tokenizer=AutoTokenizer.from_pretrained(str(args.tokenizer), use_fast=True)
    freq=build_frequency(args.corpus, args.freq_rows)
    cases=[]; seen=set(); scan=Counter(); byfam=Counter(); bypivot=Counter()
    rows=0
    with args.corpus.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            obj=json.loads(line); rows += 1
            text=str(obj.get('text',''))
            for sent in sentence_split(text):
                words=tokenize_words(sent)
                if len(words) < 5:
                    continue
                lows=[w[1] for w in words]
                for i,piv in enumerate(lows):
                    # vertical spatial opposition
                    if piv in {'above','over','below','under','beneath'}:
                        left=nearest_left(words,i); right=nearest_right(words,i)
                        if left and right:
                            a_s,a,lg=left; b_s,b,rg=right
                            if abs(math.log1p(freq[a])-math.log1p(freq[b])) <= args.max_logfreq_delta:
                                ok=maybe_add_case(
                                    cases, seen, tokenizer, family='spatial_vertical', source_row=rows, sentence=sent, attested_pivot=piv,
                                    arg_a_surf=a_s, arg_b_surf=b_s,
                                    context0_tpl='{A} is above {B}. The lower one is {MASK}.',
                                    context1_tpl='{A} is below {B}. The lower one is {MASK}.',
                                    target0=b, target1=a, relation0='above_lower_target_B', relation1='below_lower_target_A',
                                    extra={'left_gap':lg,'right_gap':rg,'freq_a':freq[a],'freq_b':freq[b],'freq_bin_a':freq_bin(freq[a]),'freq_bin_b':freq_bin(freq[b])},
                                    max_target_pieces=args.max_target_pieces)
                                if ok: byfam['spatial_vertical'] += 1; bypivot[piv] += 1
                    # temporal before/after
                    if piv in {'before','after'}:
                        left=nearest_left(words,i); right=nearest_right(words,i)
                        if left and right:
                            a_s,a,lg=left; b_s,b,rg=right
                            if abs(math.log1p(freq[a])-math.log1p(freq[b])) <= args.max_logfreq_delta:
                                ok=maybe_add_case(
                                    cases, seen, tokenizer, family='temporal_order', source_row=rows, sentence=sent, attested_pivot=piv,
                                    arg_a_surf=a_s, arg_b_surf=b_s,
                                    context0_tpl='{A} happened before {B}. The earlier one was {MASK}.',
                                    context1_tpl='{A} happened after {B}. The earlier one was {MASK}.',
                                    target0=a, target1=b, relation0='before_earlier_A', relation1='after_earlier_B',
                                    extra={'left_gap':lg,'right_gap':rg,'freq_a':freq[a],'freq_b':freq[b],'freq_bin_a':freq_bin(freq[a]),'freq_bin_b':freq_bin(freq[b])},
                                    max_target_pieces=args.max_target_pieces)
                                if ok: byfam['temporal_order'] += 1; bypivot[piv] += 1
                    # comparative: higher/lower/etc. directly before than
                    if piv in {'higher','larger','greater','older','better','faster','longer','stronger'}:
                        try:
                            than_idx = lows.index('than', i+1, min(len(lows), i+5))
                        except ValueError:
                            than_idx = -1
                        if than_idx > 0:
                            left=nearest_left(words,i); right=nearest_right(words,than_idx,max_gap=7)
                            if left and right:
                                a_s,a,lg=left; b_s,b,rg=right
                                adj=piv
                                if abs(math.log1p(freq[a])-math.log1p(freq[b])) <= args.max_logfreq_delta:
                                    ok=maybe_add_case(
                                        cases, seen, tokenizer, family='comparative_scalar', source_row=rows, sentence=sent, attested_pivot=piv,
                                        arg_a_surf=a_s, arg_b_surf=b_s,
                                        context0_tpl='{A} is '+adj+' than {B}. The '+adj+' one is {MASK}.',
                                        context1_tpl='{A} is lower than {B}. The '+adj+' one is {MASK}.' if adj not in {'better','older','faster','longer','stronger'} else '{A} is worse than {B}. The '+adj+' one is {MASK}.',
                                        target0=a, target1=b, relation0=adj+'_target_A', relation1='opposite_target_B',
                                        extra={'left_gap':lg,'right_gap':rg,'freq_a':freq[a],'freq_b':freq[b],'freq_bin_a':freq_bin(freq[a]),'freq_bin_b':freq_bin(freq[b]),'attribute':adj},
                                        max_target_pieces=args.max_target_pieces)
                                    if ok: byfam['comparative_scalar'] += 1; bypivot[piv] += 1
                    if piv in {'lower','smaller','younger','worse','slower','shorter','weaker'}:
                        try:
                            than_idx = lows.index('than', i+1, min(len(lows), i+5))
                        except ValueError:
                            than_idx = -1
                        if than_idx > 0:
                            left=nearest_left(words,i); right=nearest_right(words,than_idx,max_gap=7)
                            if left and right:
                                a_s,a,lg=left; b_s,b,rg=right
                                attr={'lower':'lower','smaller':'smaller','younger':'younger','worse':'worse','slower':'slower','shorter':'shorter','weaker':'weaker'}[piv]
                                high={'lower':'higher','smaller':'larger','younger':'older','worse':'better','slower':'faster','shorter':'longer','weaker':'stronger'}[piv]
                                if abs(math.log1p(freq[a])-math.log1p(freq[b])) <= args.max_logfreq_delta:
                                    ok=maybe_add_case(
                                        cases, seen, tokenizer, family='comparative_scalar', source_row=rows, sentence=sent, attested_pivot=piv,
                                        arg_a_surf=a_s, arg_b_surf=b_s,
                                        context0_tpl='{A} is '+attr+' than {B}. The '+attr+' one is {MASK}.',
                                        context1_tpl='{A} is '+high+' than {B}. The '+attr+' one is {MASK}.',
                                        target0=a, target1=b, relation0=attr+'_target_A', relation1=high+'_target_B',
                                        extra={'left_gap':lg,'right_gap':rg,'freq_a':freq[a],'freq_b':freq[b],'freq_bin_a':freq_bin(freq[a]),'freq_bin_b':freq_bin(freq[b]),'attribute':attr},
                                        max_target_pieces=args.max_target_pieces)
                                    if ok: byfam['comparative_scalar'] += 1; bypivot[piv] += 1
                    # more/less ADJ than
                    if piv in {'more','less'} and i+2 < len(words):
                        adj=lows[i+1]
                        if adj in SCALAR_ADJ:
                            try:
                                than_idx = lows.index('than', i+2, min(len(lows), i+7))
                            except ValueError:
                                than_idx = -1
                            if than_idx > 0:
                                left=nearest_left(words,i); right=nearest_right(words,than_idx,max_gap=7)
                                if left and right:
                                    a_s,a,lg=left; b_s,b,rg=right
                                    if abs(math.log1p(freq[a])-math.log1p(freq[b])) <= args.max_logfreq_delta:
                                        ok=maybe_add_case(
                                            cases, seen, tokenizer, family='comparative_more_less', source_row=rows, sentence=sent, attested_pivot=piv,
                                            arg_a_surf=a_s, arg_b_surf=b_s,
                                            context0_tpl='{A} is more '+adj+' than {B}. The more '+adj+' one is {MASK}.',
                                            context1_tpl='{A} is less '+adj+' than {B}. The more '+adj+' one is {MASK}.',
                                            target0=a, target1=b, relation0='more_'+adj+'_target_A', relation1='less_'+adj+'_target_B',
                                            extra={'left_gap':lg,'right_gap':rg,'freq_a':freq[a],'freq_b':freq[b],'freq_bin_a':freq_bin(freq[a]),'freq_bin_b':freq_bin(freq[b]),'attribute':adj},
                                            max_target_pieces=args.max_target_pieces)
                                        if ok: byfam['comparative_more_less'] += 1; bypivot[piv+'_'+adj] += 1
                    # causal direction templates
                    if piv in {'caused','causes','cause'}:
                        left=nearest_left(words,i); right=nearest_right(words,i)
                        if left and right:
                            a_s,a,lg=left; b_s,b,rg=right
                            if abs(math.log1p(freq[a])-math.log1p(freq[b])) <= args.max_logfreq_delta:
                                ok=maybe_add_case(
                                    cases, seen, tokenizer, family='causal_direction', source_row=rows, sentence=sent, attested_pivot=piv,
                                    arg_a_surf=a_s, arg_b_surf=b_s,
                                    context0_tpl='{A} caused {B}. The cause was {MASK}.',
                                    context1_tpl='{A} happened because of {B}. The cause was {MASK}.',
                                    target0=a, target1=b, relation0='caused_cause_A', relation1='because_cause_B',
                                    extra={'left_gap':lg,'right_gap':rg,'freq_a':freq[a],'freq_b':freq[b],'freq_bin_a':freq_bin(freq[a]),'freq_bin_b':freq_bin(freq[b])},
                                    max_target_pieces=args.max_target_pieces)
                                if ok: byfam['causal_direction'] += 1; bypivot[piv] += 1
                    if piv == 'because':
                        left=nearest_left(words,i); right=nearest_right(words,i, start_offset=1)
                        if left and right:
                            a_s,a,lg=left; b_s,b,rg=right
                            if abs(math.log1p(freq[a])-math.log1p(freq[b])) <= args.max_logfreq_delta:
                                ok=maybe_add_case(
                                    cases, seen, tokenizer, family='causal_direction', source_row=rows, sentence=sent, attested_pivot=piv,
                                    arg_a_surf=a_s, arg_b_surf=b_s,
                                    context0_tpl='{A} happened because of {B}. The cause was {MASK}.',
                                    context1_tpl='{A} caused {B}. The cause was {MASK}.',
                                    target0=b, target1=a, relation0='because_cause_B', relation1='caused_cause_A',
                                    extra={'left_gap':lg,'right_gap':rg,'freq_a':freq[a],'freq_b':freq[b],'freq_bin_a':freq_bin(freq[a]),'freq_bin_b':freq_bin(freq[b])},
                                    max_target_pieces=args.max_target_pieces)
                                if ok: byfam['causal_direction'] += 1; bypivot[piv] += 1
                scan['sentences'] += 1
            if rows % 5000 == 0:
                print(json.dumps({'event':'extract_progress','rows':rows,'cases':len(cases),'utc':now_utc()}), flush=True)
            if args.max_rows and rows >= args.max_rows:
                break
    # deterministic family balancing if requested
    if args.max_cases > 0 and len(cases) > args.max_cases:
        rng=random.Random(args.seed)
        fams=defaultdict(list)
        for c in cases:
            fams[c['family']].append(c)
        kept=[]
        quota=max(1, args.max_cases // max(1, len(fams)))
        for fam, xs in sorted(fams.items()):
            rng.shuffle(xs); kept.extend(xs[:quota])
        if len(kept) < args.max_cases:
            rest=[c for c in cases if c not in kept]
            rng.shuffle(rest); kept.extend(rest[:args.max_cases-len(kept)])
        cases=kept[:args.max_cases]
        byfam=Counter(c['family'] for c in cases); bypivot=Counter(c['attested_pivot'] for c in cases)
    # assign target permutation donors within strata
    strata=defaultdict(list)
    for idx,c in enumerate(cases):
        key=(c['family'], c['target_token_len'], c['cap_class'])
        strata[key].append(idx)
    rng=random.Random(args.seed + 17)
    donor_fail=0
    for key, idxs in strata.items():
        shuffled=idxs[:]; rng.shuffle(shuffled)
        if len(shuffled) < 2:
            for i in idxs:
                cases[i]['target_perm_case_id']=None
                donor_fail += 1
            continue
        for pos,i in enumerate(shuffled):
            j=shuffled[(pos+1) % len(shuffled)]
            # avoid sharing either target if possible
            for k in range(1, len(shuffled)):
                cand=shuffled[(pos+k) % len(shuffled)]
                if cases[i]['target0'] not in {cases[cand]['target0'], cases[cand]['target1']} and cases[i]['target1'] not in {cases[cand]['target0'], cases[cand]['target1']}:
                    j=cand; break
            cases[i]['target_perm_case_id']=cases[j]['case_id']
            cases[i]['target_perm_target0']=cases[j]['target0']
            cases[i]['target_perm_target1']=cases[j]['target1']
            cases[i]['target_perm_target0_ids']=cases[j]['target0_ids']
            cases[i]['target_perm_target1_ids']=cases[j]['target1_ids']
    with out.open('w', encoding='utf-8') as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False)+'\n')
    summary={
        'status':'COUNTERFACTUAL_EXCHANGE_BUILD',
        'created_utc':now_utc(),
        'corpus':str(args.corpus),
        'tokenizer':str(args.tokenizer),
        'rows_seen':rows,
        'sentences_seen':scan['sentences'],
        'cases':len(cases),
        'by_family':dict(Counter(c['family'] for c in cases)),
        'by_attested_pivot_top30':Counter(c['attested_pivot'] for c in cases).most_common(30),
        'target_token_len':dict(Counter(str(c['target_token_len']) for c in cases)),
        'target_perm_missing':donor_fail,
        'generated_pair_word_count_total':int(sum(c['word_count_pair'] for c in cases)),
        'generated_pair_word_count_mean':float(np.mean([c['word_count_pair'] for c in cases])) if cases else None,
        'note':'Generated contexts are not training data. If used for training, every generated word-pass must be counted under BabyLM Strict-Small exposure.',
        'cases_path':str(out),
    }
    (out.parent / 'counterfactual_exchange_build_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def qstats(vals: Iterable[float]) -> dict[str, Any]:
    xs=[float(v) for v in vals if math.isfinite(float(v))]
    if not xs:
        return {'n':0}
    arr=np.asarray(xs,dtype=np.float64)
    def q(p: float) -> float:
        return float(np.quantile(arr, p))
    return {'n':len(xs),'mean':float(arr.mean()),'std':float(arr.std()),'stderr':float(arr.std()/math.sqrt(len(xs))) if xs else None,'median':q(0.5),'p05':q(0.05),'p25':q(0.25),'p75':q(0.75),'p95':q(0.95),'min':float(arr.min()),'max':float(arr.max()),'success_gt0':float((arr>0).mean())}


def load_cases(path: Path, max_cases: int = 0) -> list[dict[str, Any]]:
    out=[]
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                c=json.loads(line)
                if c.get('target_perm_case_id') is not None:
                    out.append(c)
            if max_cases and len(out) >= max_cases:
                break
    return out


def build_encoded_contexts(tokenizer: Any, c: dict[str, Any], seq_length: int) -> dict[str, Any]:
    mask_id=int(tokenizer.mask_token_id)
    encs=[]
    for which in ['context0','context1']:
        text=c[which]
        enc=tokenizer(text, add_special_tokens=False, truncation=True, max_length=seq_length, padding='max_length', return_tensors='pt')
        ids=enc['input_ids'].squeeze(0).tolist(); attn=enc['attention_mask'].squeeze(0).tolist()
        pos=[i for i,t in enumerate(ids) if int(t)==mask_id and attn[i]==1]
        if len(pos) != int(c['target_token_len']):
            raise ValueError(f"mask count mismatch {c['case_id']} {which}: {len(pos)} vs {c['target_token_len']} text={text}")
        encs.append({'ids':ids,'attn':attn,'pos':pos})
    return {'c0':encs[0], 'c1':encs[1]}


def label_for(pos: list[int], target_ids: list[int], seq_length: int) -> list[int]:
    lab=[-100]*seq_length
    if len(pos) != len(target_ids):
        raise ValueError('label len mismatch')
    for q,t in zip(pos,target_ids):
        lab[int(q)] = int(t)
    return lab


def score_labels(logits_row: torch.Tensor, labels_row: torch.Tensor) -> float:
    """Mean target log-probability at masked positions without materializing full log_softmax.

    The earlier version converted the full [batch, seq, vocab] tensor to fp32 log
    probabilities, which is unnecessary and too memory-heavy when background
    evaluations occupy most H100 memory.  This function gathers only masked
    positions, then computes target_logit - logsumexp(vocab) for those rows.
    """
    pos=(labels_row != -100).nonzero(as_tuple=False).view(-1)
    tg=labels_row.index_select(0, pos)
    sel=logits_row.index_select(0, pos).float()
    vals=sel.gather(1, tg.view(-1,1)).view(-1) - torch.logsumexp(sel, dim=-1)
    return float(vals.mean().detach().cpu().item())


def precompute_items(cases: list[dict[str, Any]], tokenizer: Any, seq_length: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    items=[]; bad=[]
    for c in cases:
        try:
            ctx=build_encoded_contexts(tokenizer, c, seq_length)
            item={'case':c, 'ctx':ctx}
            # true labels and target-permutation labels on the same two contexts
            item['labels']={
                'c0_t0': label_for(ctx['c0']['pos'], c['target0_ids'], seq_length),
                'c0_t1': label_for(ctx['c0']['pos'], c['target1_ids'], seq_length),
                'c1_t0': label_for(ctx['c1']['pos'], c['target0_ids'], seq_length),
                'c1_t1': label_for(ctx['c1']['pos'], c['target1_ids'], seq_length),
                'c0_p0': label_for(ctx['c0']['pos'], c['target_perm_target0_ids'], seq_length),
                'c0_p1': label_for(ctx['c0']['pos'], c['target_perm_target1_ids'], seq_length),
                'c1_p0': label_for(ctx['c1']['pos'], c['target_perm_target0_ids'], seq_length),
                'c1_p1': label_for(ctx['c1']['pos'], c['target_perm_target1_ids'], seq_length),
            }
            items.append(item)
        except Exception as e:
            bad.append({'case_id':c.get('case_id'), 'error':str(e)[:300]})
    return items, {'input_cases':len(cases),'good_cases':len(items),'bad_contexts':len(bad),'bad_examples':bad[:20]}


def score_arm(arm: str, ckpt: Path, items: list[dict[str, Any]], *, device: torch.device, batch_size: int, dtype: str) -> list[dict[str, Any]]:
    torch_dtype=torch.bfloat16 if dtype == 'bf16' else torch.float16 if dtype == 'fp16' else torch.float32
    print(json.dumps({'event':'load_model','arm':arm,'checkpoint':str(ckpt),'device':str(device),'utc':now_utc()}), flush=True)
    model=DebertaV2ForMaskedLM.from_pretrained(str(ckpt), torch_dtype=torch_dtype).to(device)
    model.eval()
    rows=[]
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            chunk=items[start:start+batch_size]
            # Flatten two contexts per case, preserving order c0,c1.
            ids=[]; attn=[]
            for it in chunk:
                ids.append(it['ctx']['c0']['ids']); attn.append(it['ctx']['c0']['attn'])
                ids.append(it['ctx']['c1']['ids']); attn.append(it['ctx']['c1']['attn'])
            ids_t=torch.tensor(ids, dtype=torch.long, device=device)
            attn_t=torch.tensor(attn, dtype=torch.long, device=device)
            out=model(input_ids=ids_t, attention_mask=attn_t)
            logits=out.logits
            for r,it in enumerate(chunk):
                c=it['case']; labs=it['labels']
                lp0=logits[2*r]; lp1=logits[2*r+1]
                scores={k: score_labels(lp0 if k.startswith('c0') else lp1, torch.tensor(v, dtype=torch.long, device=device)) for k,v in labs.items()}
                ctx0_margin=scores['c0_t0'] - scores['c0_t1']
                ctx1_margin=scores['c1_t1'] - scores['c1_t0']
                true_m=ctx0_margin + ctx1_margin
                target_perm_m=(scores['c0_p0'] - scores['c0_p1']) + (scores['c1_p1'] - scores['c1_p0'])
                rec={
                    'arm':arm,
                    'case_id':c['case_id'],
                    'family':c['family'],
                    'attested_pivot':c['attested_pivot'],
                    'arg_a':c['arg_a'], 'arg_b':c['arg_b'],
                    'target0':c['target0'], 'target1':c['target1'],
                    'target_token_len':int(c['target_token_len']),
                    'freq_bin_a':c.get('freq_bin_a'), 'freq_bin_b':c.get('freq_bin_b'),
                    'word_count_pair':int(c.get('word_count_pair',0)),
                    'ctx0_correct':scores['c0_t0'], 'ctx0_wrong':scores['c0_t1'],
                    'ctx1_correct':scores['c1_t1'], 'ctx1_wrong':scores['c1_t0'],
                    'ctx0_margin':ctx0_margin, 'ctx1_margin':ctx1_margin,
                    'true_m':true_m,
                    'target_perm_m':target_perm_m,
                    'target_perm_case_id':c.get('target_perm_case_id'),
                    'target_main_bias':(scores['c0_t0'] + scores['c1_t0']) - (scores['c0_t1'] + scores['c1_t1']),
                    'both_contexts_correct': bool(ctx0_margin > 0 and ctx1_margin > 0),
                }
                rows.append(rec)
            del ids_t, attn_t, out, logits
            if start == 0 or (start // batch_size) % 20 == 0:
                print(json.dumps({'event':'score_progress','arm':arm,'cases_done':min(start+len(chunk),len(items)),'cases_total':len(items),'utc':now_utc()}), flush=True)
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return rows


def summarize(rows: list[dict[str, Any]], build_summary: dict[str, Any] | None = None) -> dict[str, Any]:
    byarm=defaultdict(list)
    for r in rows:
        byarm[r['arm']].append(r)
    metrics=['true_m','target_perm_m','ctx0_margin','ctx1_margin','target_main_bias']
    summary={'status':'COUNTERFACTUAL_EXCHANGE_SCORE','created_utc':now_utc(),'build_summary':build_summary or {},'arms':{},'arm_order':{},'paired_deltas':{}}
    for arm, xs in sorted(byarm.items()):
        asum={'n_cases':len(xs), 'overall':{}, 'by_family':{}, 'by_pivot_ge20':{}}
        for m in metrics:
            asum['overall'][m]=qstats([r[m] for r in xs])
        asum['overall']['both_contexts_correct_frac']=float(np.mean([r['both_contexts_correct'] for r in xs])) if xs else None
        for fam in sorted({r['family'] for r in xs}):
            fx=[r for r in xs if r['family']==fam]
            fsum={m:qstats([r[m] for r in fx]) for m in metrics}
            fsum['n_cases']=len(fx); fsum['both_contexts_correct_frac']=float(np.mean([r['both_contexts_correct'] for r in fx])) if fx else None
            asum['by_family'][fam]=fsum
        pivs=defaultdict(list)
        for r in xs:
            pivs[str(r['attested_pivot'])].append(r)
        for piv, px in sorted(pivs.items()):
            if len(px) >= 20:
                psum={m:qstats([r[m] for r in px]) for m in metrics}; psum['n_cases']=len(px)
                asum['by_pivot_ge20'][piv]=psum
        summary['arms'][arm]=asum
    for m in metrics:
        means={arm:summary['arms'][arm]['overall'][m]['mean'] for arm in byarm if summary['arms'][arm]['overall'][m].get('n')}
        if means:
            summary['arm_order'][m]={'means':means,'rank_high_to_low':sorted(means,key=lambda a:means[a], reverse=True),'range':max(means.values())-min(means.values())}
            byfam={}
            fams=sorted({r['family'] for r in rows})
            for fam in fams:
                fmeans={arm:summary['arms'][arm]['by_family'].get(fam,{}).get(m,{}).get('mean') for arm in byarm}
                fmeans={k:v for k,v in fmeans.items() if v is not None}
                if fmeans:
                    byfam[fam]={'means':fmeans,'rank_high_to_low':sorted(fmeans,key=lambda a:fmeans[a], reverse=True),'range':max(fmeans.values())-min(fmeans.values())}
            summary['arm_order'][m]['by_family']=byfam
    # paired deltas on common cases for true and null margins
    arms=sorted(byarm)
    by_key={(r['arm'],r['case_id']):r for r in rows}
    for i,a in enumerate(arms):
        for b in arms[i+1:]:
            common=sorted({r['case_id'] for r in byarm[a]} & {r['case_id'] for r in byarm[b]})
            dsum={'n_common':len(common)}
            for m in metrics:
                vals=[by_key[(b,k)][m]-by_key[(a,k)][m] for k in common]
                dsum[b+'_minus_'+a+'_'+m]=qstats(vals)
            summary['paired_deltas'][b+'_minus_'+a]=dsum
    return summary


def write_score_outputs(args: argparse.Namespace, rows: list[dict[str, Any]], summary: dict[str, Any], precompute_summary: dict[str, Any]) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary['precompute']=precompute_summary
    summary_path=args.out_dir / 'counterfactual_exchange_score_summary.json'
    rows_path=args.out_dir / 'counterfactual_exchange_scores.jsonl'
    csv_path=args.out_dir / 'counterfactual_exchange_scores.csv'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    with rows_path.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False)+'\n')
    if rows:
        cols=list(rows[0].keys())
        with csv_path.open('w', encoding='utf-8', newline='') as f:
            w=csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    note=[]
    note.append('# research — rule-certified counterfactual exchange probe\n')
    note.append(f'Created: {now_utc()}\n')
    bs=summary.get('build_summary') or {}
    note.append('## Build\n')
    note.append(f"Cases scored: **{precompute_summary.get('good_cases')}** / input {precompute_summary.get('input_cases')}\n")
    if bs:
        note.append(f"Generated pair word count if trained once: **{bs.get('generated_pair_word_count_total')}** words; by family: `{bs.get('by_family')}`.\n")
    note.append('\n## Frozen checkpoint arm means\n')
    note.append('| metric | compact | rowblock | interleaved | rank | range |\n')
    note.append('|---|---:|---:|---:|---|---:|\n')
    for m in ['true_m','target_perm_m','ctx0_margin','ctx1_margin','target_main_bias']:
        order=summary['arm_order'].get(m,{})
        means=order.get('means',{})
        note.append(f"| {m} | {means.get('compact')} | {means.get('rowblock')} | {means.get('interleaved')} | {' > '.join(order.get('rank_high_to_low',[]))} | {order.get('range')} |\n")
    note.append('\n## Family true_m means\n')
    note.append('| family | n | compact | rowblock | interleaved | rank | range |\n')
    note.append('|---|---:|---:|---:|---:|---|---:|\n')
    fam_order=summary['arm_order'].get('true_m',{}).get('by_family',{})
    for fam, fo in sorted(fam_order.items()):
        means=fo['means']; n=None
        for arm in ['compact','rowblock','interleaved']:
            if arm in summary['arms'] and fam in summary['arms'][arm]['by_family']:
                n=summary['arms'][arm]['by_family'][fam]['n_cases']; break
        note.append(f"| {fam} | {n} | {means.get('compact')} | {means.get('rowblock')} | {means.get('interleaved')} | {' > '.join(fo.get('rank_high_to_low',[]))} | {fo.get('range')} |\n")
    note.append('\n## Interpretation hook\n')
    note.append('The object is a no-update probe. It is useful as a possible training object only if true exchange margins are not merely saturated local copying, if arm separation exceeds the target-permutation control, and if the direction follows the known relation-hard-row tradeoff rather than the compact broad-capability ordering.\n')
    note.append('\nFiles:\n')
    note.append(f'- summary: `{summary_path}`\n')
    note.append(f'- scores: `{rows_path}`\n')
    note.append(f'- csv: `{csv_path}`\n')
    args.note.write_text(''.join(note), encoding='utf-8')


def score_cases(args: argparse.Namespace) -> dict[str, Any]:
    build_summary=None
    bpath=args.output_cases.parent / 'counterfactual_exchange_build_summary.json'
    if bpath.exists():
        build_summary=json.loads(bpath.read_text(encoding='utf-8'))
    tokenizer=AutoTokenizer.from_pretrained(str(args.tokenizer), use_fast=True)
    cases=load_cases(args.output_cases, args.score_max_cases)
    items, pre=precompute_items(cases, tokenizer, args.seq_length)
    if not items:
        raise SystemExit('No valid cases to score')
    device=torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    rows=[]
    arms={k:v for k,v in DEFAULT_ARMS.items() if (not args.arms or k in set(args.arms))}
    for arm, ckpt in arms.items():
        if not ckpt.exists():
            raise FileNotFoundError(f'{arm} checkpoint not found: {ckpt}')
        rows.extend(score_arm(arm, ckpt, items, device=device, batch_size=args.batch_size, dtype=args.dtype))
    summary=summarize(rows, build_summary)
    write_score_outputs(args, rows, summary, pre)
    print(json.dumps({'status':'COUNTERFACTUAL_EXCHANGE_DONE','cases':len(items),'arms':list(arms),'summary':str(args.out_dir / 'counterfactual_exchange_score_summary.json'),'note':str(args.note)}, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['build','score','build-score'], default='build-score')
    ap.add_argument('--corpus', type=Path, default=DEFAULT_CORPUS)
    ap.add_argument('--tokenizer', type=Path, default=DEFAULT_TOKENIZER)
    ap.add_argument('--output-cases', type=Path, default=DEFAULT_CASES)
    ap.add_argument('--out-dir', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--note', type=Path, default=DEFAULT_NOTE)
    ap.add_argument('--max-rows', type=int, default=0, help='limit corpus rows for build pilot; 0=all')
    ap.add_argument('--freq-rows', type=int, default=0, help='rows for corpus frequency table; 0=all')
    ap.add_argument('--max-cases', type=int, default=6000, help='family-balanced cap after build; 0=no cap')
    ap.add_argument('--max-target-pieces', type=int, default=1)
    ap.add_argument('--max-logfreq-delta', type=float, default=4.0)
    ap.add_argument('--score-max-cases', type=int, default=0, help='score first N valid cases; 0=all')
    ap.add_argument('--seq-length', type=int, default=256)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--dtype', choices=['bf16','fp16','fp32'], default='bf16')
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--arms', nargs='*', default=None)
    ap.add_argument('--seed', type=int, default=130)
    return ap.parse_args()


def main() -> None:
    args=parse_args()
    if args.mode in {'build','build-score'}:
        extract_cases(args)
    if args.mode in {'score','build-score'}:
        score_cases(args)


if __name__ == '__main__':
    main()
