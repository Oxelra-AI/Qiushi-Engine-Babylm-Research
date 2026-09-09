#!/usr/bin/env python3
"""research v4: witness-grounded exact microtemplate counterfactual exchange builder.

This is intentionally narrow. It does not keep tightening the old lexical funnels;
it requires exact surface frames that witness a relation between two noun/entity-like
spans before generating a fully debitable counterfactual exchange pair. The output
schema is compatible with counterfactual_exchange_probe.py score mode.
"""
from __future__ import annotations

import argparse, hashlib, json, math, re, time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer

ROOT = Path('.').resolve()
DEFAULT_CORPUS = ROOT / 'experiments/archive/representation_and_objectives/data/fw_full_arms/fw_preserved_compact_view_10M.jsonl'
DEFAULT_TOKENIZER = ROOT / 'experiments/archive/representation_and_objectives/data/shared_tokenizer/shared_16k_tokenizer'
DEFAULT_OUT = ROOT / 'experiments/archive/representation_and_objectives/data/counterfactual_exchange_v4_microtemplates/counterfactual_exchange_cases.jsonl'
DEFAULT_NOTE = ROOT / 'research/notes/representation_and_objectives/counterfactual_exchange_v4_microtemplates_build.md'

SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+")
WORD = re.compile(r"[A-Za-z][A-Za-z'\-]{1,}")
DET = r"(?:the|a|an|this|that|these|those|his|her|their|our|your|my)"
ADJ = r"(?:[a-z][a-z'\-]{2,}\s+){0,2}"
NP = rf"(?:(?:{DET})\s+)?{ADJ}[A-Za-z][A-Za-z'\-]{{2,}}(?:\s+[A-Za-z][A-Za-z'\-]{{2,}})?"

BAD_HEADS = {
    'a','an','the','this','that','these','those','his','her','their','our','your','my','it','its','they','them','we','you','he','she','i','me','him','who','what','which','where','when','why','how',
    'some','any','all','each','every','both','either','neither','many','much','more','less','most','least','one','ones','other','another','others','own','same','different','new','old','good','bad','great','small','large','big','little','first','last','next','previous','recent','current','past','future','present','far','near','now','then','here','there','well','just','only','also','actually','really','usually','often','sometimes','always','never','ever','still','already',
    'thing','things','something','anything','everything','person','people','way','time','day','year','years','month','week','part','parts','kind','sort','case','point','line','side','end','start','number','numbers','percent','rate','level','levels','type','types','area','areas','effect','effects','result','results','problem','problems','question','answer','example','study','report','research','paper','page','chapter',
    'is','are','was','were','be','been','being','have','has','had','do','does','did','can','could','will','would','should','may','might','must','shall','go','goes','went','come','came','get','got','make','makes','made','take','takes','took','use','used','using','see','seen','look','looks','show','shows','shown','find','found','meet','choose','wish','want','need','try','turn','move','walk','begin','begins','start','starts','end','ends','cover','covers','hide','hides','sit','sits','run','runs','rise','rises','occur','occurs','happen','happens',
    'chi','mot','fat','mar','bro','sis','mom','dad','xxx','xx','uh','yeah','okay','ok','oh','ah','huh','mhm'
}
ABSTRACT_BAD = {'important','significant','possible','likely','unlikely','usual','common','general','specific','basic','virtual','public','private','economic','political','social','legal','medical','physical','natural','human','major','minor','central','strong','weak','high','low','higher','lower','long','longer','short','shorter','large','larger','small','smaller','better','worse','best','worst','left','right','front','rear','top','bottom','middle','upper','lower','amount','quarter','half','hundred','thousand','million','billion','miles','kilometers','acres','degrees','temperature'}
VERBISH_SUFFIX = ('ed','ing','ly')
NOUNISH_SUFFIX = ('tion','sion','ment','ness','ity','ship','hood','ism','ist','ists','ers','ors','ies')
MONTHS = {'january','february','march','april','may','june','july','august','september','october','november','december'}
SCALAR_OPP = {'higher':'lower','lower':'higher','larger':'smaller','smaller':'larger','older':'younger','younger':'older','better':'worse','worse':'better','faster':'slower','slower':'faster','longer':'shorter','shorter':'longer','stronger':'weaker','weaker':'stronger'}

SPATIAL_PATTERNS = [
    re.compile(rf"\b(?P<A>{NP})\s+(?P<V>is|are|was|were|lies|lie|sits|sit|stands|stand|rests|rest|hangs|hang|floats|float|located|situated|placed|suspended|burrows|burrow|hides|hide)\s+(?:directly\s+|just\s+|well\s+)?(?P<R>above|below|beneath)\s+(?P<B>{NP})\b", re.I),
]
TEMPORAL_PATTERNS = [
    re.compile(rf"\b(?P<A>{NP})\s+(?P<V>was|were|is|are)?\s*(?P<EV>built|made|created|released|published|formed|founded|opened|closed|completed|introduced|settled|born|died)\s+(?P<R>before|after)\s+(?P<B>{NP})\b", re.I),
]
CAUSAL_PATTERNS = [
    re.compile(rf"\b(?P<A>{NP})\s+(?P<R>caused|causes)\s+(?P<B>{NP})\b", re.I),
    re.compile(rf"\b(?P<B>{NP})\s+(?:is|are|was|were|be|been)\s+(?P<R>caused)\s+by\s+(?P<A>{NP})\b", re.I),
]
COMP_PATTERNS = [
    re.compile(rf"\b(?P<A>{NP})\s+(?:is|are|was|were|becomes|became|seems|seemed)?\s*(?P<R>higher|lower|larger|smaller|older|younger|better|worse|faster|slower|longer|shorter|stronger|weaker)\s+than\s+(?P<B>{NP})\b", re.I),
]


def now(): return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

def norm(s: str) -> str:
    return re.sub(r"[^A-Za-z'\-]", '', s).strip("-'_").lower()

def head(np: str) -> str:
    ws=[norm(x.group(0)) for x in WORD.finditer(np)]
    ws=[w for w in ws if w and w not in {'the','a','an','this','that','these','those','his','her','their','our','your','my'}]
    return ws[-1] if ws else ''

def clean_np(np: str) -> str:
    return ' '.join(np.split()).strip(' ,.;:()[]"“”')

def good_head(h: str, np: str) -> bool:
    if not h or len(h) < 3 or len(h) > 18: return False
    if h in BAD_HEADS or h in ABSTRACT_BAD or h in MONTHS: return False
    if not re.fullmatch(r"[a-z][a-z'\-]*", h): return False
    if h.endswith(VERBISH_SUFFIX): return False
    toks=[norm(x.group(0)) for x in WORD.finditer(np)]
    prev_det = bool(toks and toks[0] in {'the','a','an','this','that','these','those','his','her','their','our','your','my'})
    # Accept proper-looking multiword surface, determiner NPs, plural heads, or nounish suffixes.
    if any(x[:1].isupper() for x in WORD.findall(np)) and h not in BAD_HEADS: return True
    if prev_det: return True
    if h.endswith('s') and h not in BAD_HEADS: return True
    if h.endswith(NOUNISH_SUFFIX): return True
    return False

def sentence_ok(sent: str) -> bool:
    lo=sent.lower()
    if '*' in sent or 'xxx' in lo or re.search(r"\b(chi|mot|fat|sis|bro|inv|exp)\s*:", lo): return False
    return True

def word_count(s: str) -> int: return len(re.findall(r"\b\S+\b", s))

def token_ids(tok: Any, t: str) -> list[int]: return [int(x) for x in tok(t, add_special_tokens=False)['input_ids']]

def freq_bin(c:int)->str:
    if c<=1:return '1'
    if c<=3:return '2-3'
    if c<=7:return '4-7'
    if c<=15:return '8-15'
    if c<=31:return '16-31'
    if c<=63:return '32-63'
    if c<=127:return '64-127'
    if c<=255:return '128-255'
    if c<=511:return '256-511'
    if c<=1023:return '512-1023'
    return '1000+'

def build_freq(corpus: Path, max_rows:int=0)->Counter:
    f=Counter(); rows=0
    with corpus.open(encoding='utf-8') as fh:
        for line in fh:
            if not line.strip(): continue
            rows+=1; obj=json.loads(line)
            for m in WORD.finditer(str(obj.get('text',''))):
                w=norm(m.group(0))
                if w: f[w]+=1
            if max_rows and rows>=max_rows: break
    return f

def add_case(cases:list[dict[str,Any]], seen:set[str], tok:Any, args:argparse.Namespace, freq:Counter, *, family:str, row:int, sent:str, pivot:str, np_a:str, np_b:str, target0:str, target1:str, ctx0_tpl:str, ctx1_tpl:str, relation0:str, relation1:str, evidence:str)->bool:
    np_a=clean_np(np_a); np_b=clean_np(np_b); a=head(np_a); b=head(np_b)
    if a==b or target0==target1: return False
    if not good_head(a,np_a) or not good_head(b,np_b): return False
    if abs(math.log1p(freq[a])-math.log1p(freq[b])) > args.max_logfreq_delta: return False
    ids0=token_ids(tok,target0); ids1=token_ids(tok,target1)
    if not ids0 or len(ids0)!=len(ids1) or len(ids0)>args.max_target_pieces: return False
    mask=' '.join([tok.mask_token or '<mask>']*len(ids0))
    ctx0=ctx0_tpl.format(A=np_a.lower(), B=np_b.lower(), MASK=mask)
    ctx1=ctx1_tpl.format(A=np_a.lower(), B=np_b.lower(), MASK=mask)
    raw='|'.join([family,sent,ctx0,ctx1,target0,target1])
    h=hashlib.sha1(raw.encode()).hexdigest()[:16]
    if h in seen: return False
    seen.add(h)
    c={'case_id':'cfv4_'+h,'family':family,'source_row':row,'sentence':sent,'attested_pivot':pivot,'arg_a':a,'arg_b':b,'arg_a_span':np_a,'arg_b_span':np_b,'target0':target0,'target1':target1,'target0_ids':ids0,'target1_ids':ids1,'target_token_len':len(ids0),'cap_class':'proper' if (np_a[:1].isupper() and np_b[:1].isupper()) else 'common','relation0':relation0,'relation1':relation1,'context0':ctx0,'context1':ctx1,'word_count_context0':word_count(ctx0),'word_count_context1':word_count(ctx1),'word_count_pair':word_count(ctx0)+word_count(ctx1),'freq_a':freq[a],'freq_b':freq[b],'freq_bin_a':freq_bin(freq[a]),'freq_bin_b':freq_bin(freq[b]),'evidence_pattern':evidence,'certification':'exact microtemplate witness; same A/B spans; controlled relation flip exchanges target; generated word-passes fully debit if trained'}
    cases.append(c); return True

def process_sentence(sent:str, row:int, tok:Any, args:argparse.Namespace, freq:Counter, cases:list[dict[str,Any]], seen:set[str], rejects:Counter):
    if not sentence_ok(sent): rejects['speaker_or_xxx']+=1; return
    lo=sent.lower()
    # reject common quantity/threshold uses before matching spatial/comparative.
    if re.search(r"\b(above|below|over|under|higher|lower|larger|smaller|more|less)\s+(one|two|three|four|five|six|seven|eight|nine|ten|\d|[0-9,.]+|a quarter|quarter|half|hundred|thousand|million|billion|percent|%)", lo):
        quantity_risky=True
    else:
        quantity_risky=False
    for pat in SPATIAL_PATTERNS:
        for m in pat.finditer(sent):
            if quantity_risky: rejects['spatial_quantity_risk']+=1; continue
            r=norm(m.group('R')); A=m.group('A'); B=m.group('B'); a=head(A); b=head(B)
            if r=='above':
                add_case(cases,seen,tok,args,freq,family='spatial_vertical_exact',row=row,sent=sent,pivot=r,np_a=A,np_b=B,target0=a,target1=b,ctx0_tpl='{A} is above {B}. The higher one is {MASK}.',ctx1_tpl='{A} is below {B}. The higher one is {MASK}.',relation0='above_higher_A',relation1='below_higher_B',evidence='copular_or_locative_above')
            else:
                add_case(cases,seen,tok,args,freq,family='spatial_vertical_exact',row=row,sent=sent,pivot=r,np_a=A,np_b=B,target0=b,target1=a,ctx0_tpl='{A} is below {B}. The higher one is {MASK}.',ctx1_tpl='{A} is above {B}. The higher one is {MASK}.',relation0='below_higher_B',relation1='above_higher_A',evidence='copular_or_locative_below')
    for pat in TEMPORAL_PATTERNS:
        for m in pat.finditer(sent):
            r=norm(m.group('R')); A=m.group('A'); B=m.group('B'); a=head(A); b=head(B)
            if r=='before':
                add_case(cases,seen,tok,args,freq,family='temporal_event_exact',row=row,sent=sent,pivot=r,np_a=A,np_b=B,target0=a,target1=b,ctx0_tpl='The {A} event happened before the {B} event. The earlier event was {MASK}.',ctx1_tpl='The {A} event happened after the {B} event. The earlier event was {MASK}.',relation0='before_earlier_A',relation1='after_earlier_B',evidence='event_verb_before')
            else:
                add_case(cases,seen,tok,args,freq,family='temporal_event_exact',row=row,sent=sent,pivot=r,np_a=A,np_b=B,target0=b,target1=a,ctx0_tpl='The {A} event happened after the {B} event. The earlier event was {MASK}.',ctx1_tpl='The {A} event happened before the {B} event. The earlier event was {MASK}.',relation0='after_earlier_B',relation1='before_earlier_A',evidence='event_verb_after')
    for pat in CAUSAL_PATTERNS:
        for m in pat.finditer(sent):
            # skip if this is the active pattern accidentally spanning auxiliary verbs only
            A=m.group('A'); B=m.group('B'); a=head(A); b=head(B)
            add_case(cases,seen,tok,args,freq,family='causal_exact',row=row,sent=sent,pivot=norm(m.group('R')),np_a=A,np_b=B,target0=a,target1=b,ctx0_tpl='{A} caused {B}. The cause was {MASK}.',ctx1_tpl='{A} was caused by {B}. The cause was {MASK}.',relation0='A_caused_B_cause_A',relation1='A_caused_by_B_cause_B',evidence='caused_or_caused_by_exact')
    for pat in COMP_PATTERNS:
        for m in pat.finditer(sent):
            if quantity_risky: rejects['comparative_quantity_risk']+=1; continue
            r=norm(m.group('R')); opp=SCALAR_OPP.get(r)
            if not opp: continue
            A=m.group('A'); B=m.group('B'); a=head(A); b=head(B)
            add_case(cases,seen,tok,args,freq,family='comparative_exact',row=row,sent=sent,pivot=r,np_a=A,np_b=B,target0=a,target1=b,ctx0_tpl='{A} is '+r+' than {B}. The '+r+' one is {MASK}.',ctx1_tpl='{A} is '+opp+' than {B}. The '+r+' one is {MASK}.',relation0=f'{r}_A',relation1=f'{opp}_B',evidence='comparative_than_exact')

def build(args):
    tok=AutoTokenizer.from_pretrained(str(args.tokenizer), use_fast=True)
    freq=build_freq(args.corpus,args.freq_rows)
    cases=[]; seen=set(); rows=0; sent_count=0; rejects=Counter()
    with args.corpus.open(encoding='utf-8') as fh:
        for line in fh:
            if not line.strip(): continue
            rows+=1; obj=json.loads(line); text=' '.join(str(obj.get('text','')).split())
            for sent in SENT_SPLIT.split(text):
                sent=sent.strip()
                if len(sent)<25 or len(sent)>280: continue
                sent_count+=1
                process_sentence(sent, rows, tok, args, freq, cases, seen, rejects)
            if rows%10000==0:
                print(json.dumps({'event':'v4_progress','rows':rows,'cases':len(cases),'utc':now()}), flush=True)
            if args.max_rows and rows>=args.max_rows: break
    if args.max_cases and len(cases)>args.max_cases:
        buckets=defaultdict(list)
        for c in cases: buckets[c['family']].append(c)
        kept=[]; quota=max(1,args.max_cases//max(1,len(buckets)))
        for fam in sorted(buckets): kept.extend(buckets[fam][:quota])
        rest=[c for c in cases if c not in kept]
        kept.extend(rest[:max(0,args.max_cases-len(kept))])
        cases=kept[:args.max_cases]
    # target pair permutation donors
    strata=defaultdict(list)
    for i,c in enumerate(cases): strata[(c['family'],c['target_token_len'],c['cap_class'])].append(i)
    missing=0
    for key,idxs in strata.items():
        if len(idxs)<2:
            for i in idxs: cases[i]['target_perm_case_id']=None; missing+=1
            continue
        for n,i in enumerate(idxs):
            donor=idxs[(n+1)%len(idxs)]
            for k in range(1,len(idxs)+1):
                cand=idxs[(n+k)%len(idxs)]
                if cases[i]['target0'] not in {cases[cand]['target0'],cases[cand]['target1']} and cases[i]['target1'] not in {cases[cand]['target0'],cases[cand]['target1']}:
                    donor=cand; break
            d=cases[donor]
            cases[i]['target_perm_case_id']=d['case_id']; cases[i]['target_perm_target0']=d['target0']; cases[i]['target_perm_target1']=d['target1']; cases[i]['target_perm_target0_ids']=d['target0_ids']; cases[i]['target_perm_target1_ids']=d['target1_ids']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w',encoding='utf-8') as f:
        for c in cases: f.write(json.dumps(c,ensure_ascii=False)+'\n')
    sample=args.output.parent/'counterfactual_exchange_v4_samples.jsonl'
    with sample.open('w',encoding='utf-8') as f:
        for c in cases[:120]:
            f.write(json.dumps({k:c.get(k) for k in ['case_id','family','sentence','arg_a_span','arg_b_span','context0','target0','context1','target1','evidence_pattern','freq_bin_a','freq_bin_b']}, ensure_ascii=False)+'\n')
    summary={'status':'COUNTERFACTUAL_EXCHANGE_V4_BUILD','created_utc':now(),'rows_seen':rows,'sentences_seen':sent_count,'cases':len(cases),'by_family':dict(Counter(c['family'] for c in cases)),'top_pivots':Counter(c['attested_pivot'] for c in cases).most_common(30),'target_token_len':dict(Counter(str(c['target_token_len']) for c in cases)),'target_perm_missing':missing,'generated_pair_word_count_total':int(sum(c['word_count_pair'] for c in cases)),'generated_pair_word_count_mean':float(np.mean([c['word_count_pair'] for c in cases])) if cases else None,'rejects':dict(rejects),'cases_path':str(args.output),'sample_path':str(sample),'certification':'exact microtemplate frames, no parser or external language model, generated contexts not used for training in research'}
    (args.output.parent/'counterfactual_exchange_v4_build_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    args.note.parent.mkdir(parents=True, exist_ok=True)
    args.note.write_text('# research — v4 exact microtemplate counterfactual exchange build\n\n'+f'Created: {summary["created_utc"]}\n\nRows seen: {rows}; sentences: {sent_count}; cases: **{len(cases)}**.\n\nBy family: `{summary["by_family"]}`.\n\nGenerated pair words if used once for training: **{summary["generated_pair_word_count_total"]}**.\n\nFiles: `{args.output}`, `{sample}`, `{args.output.parent/"counterfactual_exchange_v4_build_summary.json"}`.\n', encoding='utf-8')
    print(json.dumps(summary,indent=2), flush=True)

def parse():
    ap=argparse.ArgumentParser()
    ap.add_argument('--corpus',type=Path,default=DEFAULT_CORPUS)
    ap.add_argument('--tokenizer',type=Path,default=DEFAULT_TOKENIZER)
    ap.add_argument('--output',type=Path,default=DEFAULT_OUT)
    ap.add_argument('--note',type=Path,default=DEFAULT_NOTE)
    ap.add_argument('--max-rows',type=int,default=0)
    ap.add_argument('--freq-rows',type=int,default=0)
    ap.add_argument('--max-cases',type=int,default=4000)
    ap.add_argument('--max-target-pieces',type=int,default=1)
    ap.add_argument('--max-logfreq-delta',type=float,default=2.5)
    return ap.parse_args()

if __name__=='__main__':
    build(parse())
