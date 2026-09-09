#!/usr/bin/env python3
"""research: witness-grounded spatial counterfactual-exchange v5 probe.

This advances the research unsaturated spatial-exchange lead without H100 training.
It is a frozen/no-update probe builder and scorer for exact witnessed vertical
relations in the allowed BabyLM corpus.  It separates:
  * counterfactual exchange views (A above B vs A below B)
  * inverse-equivalent views (A above B vs B below A)
  * dual queries (higher and lower)
  * relation-erased controls (near)
  * target-pair permutation controls

No generated text is used for model updates here.  If any later training uses these
contexts, every generated word-pass stored in `generated_pair_words_*` must be
counted against the Strict-Small budget.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer, DebertaV2ForMaskedLM

ROOT = Path('.').resolve()
DEFAULT_CORPUS = ROOT / 'experiments/archive/representation_and_objectives/data/fw_full_arms/fw_preserved_compact_view_10M.jsonl'
DEFAULT_TOKENIZER = ROOT / 'experiments/archive/representation_and_objectives/data/shared_tokenizer/shared_16k_tokenizer'
DEFAULT_OUTDIR = ROOT / 'experiments/archive/representation_and_objectives/data/spatial_exchange_v5'
DEFAULT_NOTE = ROOT / 'research/notes/representation_and_objectives/spatial_exchange_v5.md'
DEFAULT_ARMS = {
    'compact': ROOT / 'experiments/archive/frontier_consolidation/training/runs/fw_compact_view_shared16k_seed43022/hf_model/chck_100M',
    'rowblock': ROOT / 'experiments/archive/frontier_consolidation/training/runs/fw_source_breadth_shared16k_seed43022/hf_model/chck_100M',
    'interleaved': ROOT / 'experiments/archive/representation_and_objectives/training/runs/fw_source_breadth_interleaved_wholesentence_fullbatch_shared16k_seed43022/hf_model/chck_100M',
}

SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+")
WORD = re.compile(r"[A-Za-z][A-Za-z'\-]{1,}")
DET_WORDS = {'the','a','an','this','that','these','those','his','her','their','our','your','my'}
DET = r"(?:the|a|an|this|that|these|those|his|her|their|our|your|my)"
# Keep the NP intentionally short to avoid swallowing full clauses. The final word
# is treated as the candidate head/target.
ADJ = r"(?:[A-Za-z][A-Za-z'\-]{2,}\s+){0,2}"
NP = rf"(?:(?:{DET})\s+)?{ADJ}[A-Za-z][A-Za-z'\-]{{2,}}(?:\s+[A-Za-z][A-Za-z'\-]{{2,}})?"

SPATIAL_PATTERNS = [
    # locative/copular exact vertical relations
    (re.compile(rf"\b(?P<A>{NP})\s+(?P<V>is|are|was|were|lies|lie|sits|sit|stands|stand|rests|rest|hangs|hang|floats|float|located|situated|placed|suspended|hides|hide)\s+(?:directly\s+|just\s+|well\s+)?(?P<R>above|below|beneath)\s+(?P<B>{NP})\b", re.I), 'locative_verb_vertical'),
    # on top of is a high-precision vertical witness
    (re.compile(rf"\b(?P<A>{NP})\s+(?P<V>is|are|was|were|lies|lie|sits|sit|stands|stand|rests|rest|located|placed)\s+(?:directly\s+|just\s+)?(?P<R>on\s+top\s+of)\s+(?P<B>{NP})\b", re.I), 'on_top_of_vertical'),
]

BAD_HEADS = {
    'a','an','the','this','that','these','those','his','her','their','our','your','my','it','its','they','them','we','you','he','she','i','me','him','who','what','which','where','when','why','how',
    'some','any','all','each','every','both','either','neither','many','much','more','less','most','least','one','ones','other','another','others','own','same','different','new','old','good','bad','great','small','large','big','little','first','last','next','previous','recent','current','past','future','present','far','near','now','then','here','there','well','just','only','also','actually','really','usually','often','sometimes','always','never','ever','still','already',
    'thing','things','something','anything','everything','person','people','way','time','day','year','years','month','week','part','parts','kind','sort','case','point','line','side','end','start','number','numbers','percent','rate','level','levels','type','types','area','areas','effect','effects','result','results','problem','problems','question','answer','example','study','report','research','paper','page','chapter','law','laws','rule','rules','section','sections','terms','term','condition','conditions',
    'is','are','was','were','be','been','being','have','has','had','do','does','did','can','could','will','would','should','may','might','must','shall','go','goes','went','come','came','get','got','make','makes','made','take','takes','took','use','used','using','see','seen','look','looks','show','shows','shown','find','found','meet','choose','wish','want','need','try','turn','move','walk','begin','begins','start','starts','end','ends','cover','covers','hide','hides','sit','sits','run','runs','rise','rises','occur','occurs','happen','happens','located','situated','placed','suspended',
    'chi','mot','fat','mar','bro','sis','mom','dad','xxx','xx','uh','yeah','okay','ok','oh','ah','huh','mhm'
}
ABSTRACT_OR_SCALAR = {
    'important','significant','possible','likely','unlikely','usual','common','general','specific','basic','virtual','public','private','economic','political','social','legal','medical','natural','human','major','minor','central','strong','weak','high','low','higher','lower','long','longer','short','shorter','large','larger','small','smaller','better','worse','best','worst','left','right','front','rear','top','bottom','middle','upper','lower','amount','quarter','half','hundred','thousand','million','billion','percent','percentage','miles','kilometers','acres','degrees','temperature','age','ages','score','scores','ranking','rankings','value','values','cost','costs','price','prices','income','population'
}
VERBISH_SUFFIX = ('ed','ing','ly')
NOUNISH_SUFFIX = ('tion','sion','ment','ness','ity','ship','hood','ism','ist','ists','ers','ors','ies','ium','age')
CONCRETE_HINTS = {
    'floor','ceiling','roof','table','desk','chair','bed','shelf','shelves','wall','walls','door','doors','window','windows','ground','soil','water','river','lake','sea','ocean','mountain','hill','valley','tree','trees','branch','branches','leaf','leaves','rock','rocks','stone','stones','bridge','road','street','building','buildings','tower','towers','room','rooms','house','houses','car','cars','train','trains','ship','ships','boat','boats','plane','planes','bird','birds','fish','plant','plants','box','boxes','cup','cups','bowl','bowls','plate','plates','hand','hands','head','heads','body','bodies','surface','surfaces','screen','screens','page','pages','map','maps','star','stars','sun','moon','cloud','clouds','sky','earth'
}
MONTHS = {'january','february','march','april','may','june','july','august','september','october','november','december'}


def now() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def norm(s: str) -> str:
    return re.sub(r"[^A-Za-z'\-]", '', s).strip("-'_").lower()


def words_in(s: str) -> list[str]:
    return [norm(m.group(0)) for m in WORD.finditer(s) if norm(m.group(0))]


def clean_np(s: str) -> str:
    return ' '.join(s.split()).strip(' ,.;:()[]"“”')


def head(np: str) -> str:
    ws = [w for w in words_in(np) if w not in DET_WORDS]
    return ws[-1] if ws else ''


def good_head(h: str, np: str, *, require_concrete_hint: bool) -> bool:
    if not h or len(h) < 3 or len(h) > 18:
        return False
    if h in BAD_HEADS or h in ABSTRACT_OR_SCALAR or h in MONTHS:
        return False
    if not re.fullmatch(r"[a-z][a-z'\-]*", h):
        return False
    if h.endswith(VERBISH_SUFFIX):
        return False
    toks = words_in(np)
    prev_det = bool(toks and toks[0] in DET_WORDS)
    properish = any(tok[:1].isupper() for tok in re.findall(r"\b[A-Z][A-Za-z'\-]{2,}\b", np))
    nounish = h.endswith('s') or h.endswith(NOUNISH_SUFFIX) or h in CONCRETE_HINTS
    if require_concrete_hint and h not in CONCRETE_HINTS and not properish and not prev_det:
        return False
    return bool(properish or prev_det or nounish)


def sentence_ok(sent: str) -> bool:
    lo = sent.lower()
    if '*' in sent or 'xxx' in lo or re.search(r"\b(chi|mot|fat|sis|bro|inv|exp)\s*:", lo):
        return False
    # remove quantity/comparison threshold uses that contaminated v3
    if re.search(r"\b(above|below|beneath|over|under|higher|lower)\s+(one|two|three|four|five|six|seven|eight|nine|ten|\d|[0-9,.]+|a quarter|quarter|half|hundred|thousand|million|billion|percent|%)", lo):
        return False
    if re.search(r"\b(above|below|beneath)\s+(average|threshold|level|rate|score|ranking|rank|age|years|months|weeks|days|price|cost|value)", lo):
        return False
    return True


def word_count(s: str) -> int:
    return len(re.findall(r"\b\S+\b", s))


def token_ids(tok: Any, text: str) -> list[int]:
    return [int(x) for x in tok(text, add_special_tokens=False)['input_ids']]


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


def build_freq(corpus: Path, max_rows: int = 0) -> Counter:
    freq = Counter(); rows = 0
    with corpus.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            rows += 1
            obj = json.loads(line)
            for w in words_in(str(obj.get('text',''))):
                freq[w] += 1
            if max_rows and rows >= max_rows:
                break
    return freq


def cap_class(np_a: str, np_b: str) -> str:
    pa = bool(re.search(r"\b[A-Z][A-Za-z'\-]{2,}\b", np_a))
    pb = bool(re.search(r"\b[A-Z][A-Za-z'\-]{2,}\b", np_b))
    return 'proper' if pa and pb else 'common'


def add_target_perm(cases: list[dict[str, Any]]) -> int:
    strata = defaultdict(list)
    for i, c in enumerate(cases):
        strata[(c['target_token_len'], c['cap_class'], c['freq_bin_a'], c['freq_bin_b'])].append(i)
    missing = 0
    for _, idxs in strata.items():
        if len(idxs) < 2:
            for i in idxs:
                cases[i]['target_perm_case_id'] = None
                missing += 1
            continue
        for n, i in enumerate(idxs):
            donor = idxs[(n + 1) % len(idxs)]
            for k in range(1, len(idxs) + 1):
                cand = idxs[(n + k) % len(idxs)]
                if cases[i]['target0'] not in {cases[cand]['target0'], cases[cand]['target1']} and cases[i]['target1'] not in {cases[cand]['target0'], cases[cand]['target1']}:
                    donor = cand
                    break
            d = cases[donor]
            cases[i]['target_perm_case_id'] = d['case_id']
            cases[i]['target_perm_target0'] = d['target0']
            cases[i]['target_perm_target1'] = d['target1']
            cases[i]['target_perm_target0_ids'] = d['target0_ids']
            cases[i]['target_perm_target1_ids'] = d['target1_ids']
    return missing


def make_case(tok: Any, args: argparse.Namespace, freq: Counter, *, row: int, sent: str, evidence: str, rel: str, np_a_raw: str, np_b_raw: str, rejects: Counter) -> dict[str, Any] | None:
    np_a = clean_np(np_a_raw); np_b = clean_np(np_b_raw)
    a = head(np_a); b = head(np_b)
    if a == b:
        rejects['same_head'] += 1; return None
    if not good_head(a, np_a, require_concrete_hint=args.require_concrete_hint) or not good_head(b, np_b, require_concrete_hint=args.require_concrete_hint):
        rejects['bad_head'] += 1; return None
    if abs(math.log1p(freq[a]) - math.log1p(freq[b])) > args.max_logfreq_delta:
        rejects['freq_mismatch'] += 1; return None
    ids_a = token_ids(tok, a); ids_b = token_ids(tok, b)
    if not ids_a or len(ids_a) != len(ids_b) or len(ids_a) > args.max_target_pieces:
        rejects['token_shape'] += 1; return None
    mask = ' '.join([tok.mask_token or '<mask>'] * len(ids_a))
    # Canonicalize to A_above_B as relation0.  If the witness is below/beneath,
    # the witnessed true world is A_below_B, but the generated factorial always
    # contains both worlds with known exchanged targets.
    rel_norm = 'above' if rel in {'above','on top of'} else 'below'
    case_base = {
        'case_id': '',
        'family': 'spatial_vertical_v5',
        'source_row': row,
        'sentence': sent,
        'evidence_pattern': evidence,
        'attested_relation': rel_norm,
        'arg_a_span': np_a,
        'arg_b_span': np_b,
        'arg_a': a,
        'arg_b': b,
        'target0': a,  # target for A_above_B / B_below_A higher query
        'target1': b,  # target for A_below_B / B_above_A higher query
        'target0_ids': ids_a,
        'target1_ids': ids_b,
        'target_token_len': len(ids_a),
        'cap_class': cap_class(np_a, np_b),
        'freq_a': freq[a],
        'freq_b': freq[b],
        'freq_bin_a': freq_bin(freq[a]),
        'freq_bin_b': freq_bin(freq[b]),
    }
    # Views are intentionally separated.  Context names encode the relation world
    # and the query word.  correct_target is expressed as t0/t1 where t0=head(A),
    # t1=head(B).
    A = np_a.lower(); B = np_b.lower()
    views = {
        'cf_above_higher': {'text': f'{A} is above {B}. The higher one is {mask}.', 'correct': 't0', 'wrong': 't1', 'role': 'counterfactual_above', 'query': 'higher'},
        'cf_below_higher': {'text': f'{A} is below {B}. The higher one is {mask}.', 'correct': 't1', 'wrong': 't0', 'role': 'counterfactual_below', 'query': 'higher'},
        'cf_above_lower': {'text': f'{A} is above {B}. The lower one is {mask}.', 'correct': 't1', 'wrong': 't0', 'role': 'counterfactual_above', 'query': 'lower'},
        'cf_below_lower': {'text': f'{A} is below {B}. The lower one is {mask}.', 'correct': 't0', 'wrong': 't1', 'role': 'counterfactual_below', 'query': 'lower'},
        'inv_above_higher': {'text': f'{B} is below {A}. The higher one is {mask}.', 'correct': 't0', 'wrong': 't1', 'role': 'inverse_equiv_above', 'query': 'higher'},
        'inv_below_higher': {'text': f'{B} is above {A}. The higher one is {mask}.', 'correct': 't1', 'wrong': 't0', 'role': 'inverse_equiv_below', 'query': 'higher'},
        'erased_higher': {'text': f'{A} is near {B}. The higher one is {mask}.', 'correct': 't0', 'wrong': 't1', 'role': 'relation_erased', 'query': 'higher'},
        'erased_lower': {'text': f'{A} is near {B}. The lower one is {mask}.', 'correct': 't1', 'wrong': 't0', 'role': 'relation_erased', 'query': 'lower'},
    }
    case_base['views'] = views
    case_base['generated_pair_words_counterfactual_dualquery'] = sum(word_count(views[k]['text']) for k in ['cf_above_higher','cf_below_higher','cf_above_lower','cf_below_lower'])
    case_base['generated_pair_words_full_factorial'] = sum(word_count(v['text']) for v in views.values())
    case_base['certification'] = 'v5 exact witnessed vertical relation; typed noun/entity-like spans; factorial relation flip, inverse-equivalent, dual-query and relation-erased views; generated text is no-update probe only'
    raw = '|'.join([sent, A, B, rel_norm])
    case_base['case_id'] = 'cfv5sp_' + hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]
    return case_base


def build(args: argparse.Namespace) -> dict[str, Any]:
    tok = AutoTokenizer.from_pretrained(str(args.tokenizer), use_fast=True)
    freq = build_freq(args.corpus, args.freq_rows)
    cases = []
    seen = set()
    rejects = Counter()
    rows = 0; sent_count = 0; match_count = 0
    with args.corpus.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            rows += 1
            obj = json.loads(line)
            text = ' '.join(str(obj.get('text','')).split())
            for sent in SENT_SPLIT.split(text):
                sent = sent.strip()
                if len(sent) < args.min_sentence_chars or len(sent) > args.max_sentence_chars:
                    continue
                sent_count += 1
                if not sentence_ok(sent):
                    rejects['sentence_filter'] += 1
                    continue
                for pat, evidence in SPATIAL_PATTERNS:
                    for m in pat.finditer(sent):
                        match_count += 1
                        rel = norm(m.group('R').replace(' ', ' '))
                        rel_key = 'on top of' if 'top' in m.group('R').lower() else rel
                        c = make_case(tok, args, freq, row=rows, sent=sent, evidence=evidence, rel=rel_key, np_a_raw=m.group('A'), np_b_raw=m.group('B'), rejects=rejects)
                        if c is None:
                            continue
                        if c['case_id'] in seen:
                            rejects['duplicate'] += 1
                            continue
                        seen.add(c['case_id'])
                        cases.append(c)
            if rows % 10000 == 0:
                print(json.dumps({'event':'v5_build_progress','rows':rows,'sentences':sent_count,'matches':match_count,'cases':len(cases),'utc':now()}), flush=True)
            if args.max_rows and rows >= args.max_rows:
                break
    # Orientation balancing: keep equal above/below attested witnesses when possible,
    # then cap. This does not change the generated factorial, but prevents source
    # orientation from dominating robustness summaries.
    if args.balance_attested and cases:
        buckets = defaultdict(list)
        for c in cases:
            buckets[c['attested_relation']].append(c)
        if len(buckets) >= 2:
            m = min(len(v) for v in buckets.values())
            balanced = []
            for rel in sorted(buckets):
                balanced.extend(buckets[rel][:m])
            cases = balanced
    if args.max_cases and len(cases) > args.max_cases:
        # retain attested orientation balance under cap
        buckets = defaultdict(list)
        for c in cases:
            buckets[c['attested_relation']].append(c)
        kept = []
        quota = max(1, args.max_cases // max(1, len(buckets)))
        for rel in sorted(buckets):
            kept.extend(buckets[rel][:quota])
        rest = [c for c in cases if c not in kept]
        kept.extend(rest[:max(0, args.max_cases - len(kept))])
        cases = kept[:args.max_cases]
    missing = add_target_perm(cases)
    args.outdir.mkdir(parents=True, exist_ok=True)
    cases_path = args.outdir / 'spatial_exchange_v5_cases.jsonl'
    with cases_path.open('w', encoding='utf-8') as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + '\n')
    sample_path = args.outdir / 'spatial_exchange_v5_samples.jsonl'
    with sample_path.open('w', encoding='utf-8') as f:
        for c in cases[:120]:
            f.write(json.dumps({
                'case_id': c['case_id'], 'attested_relation': c['attested_relation'], 'sentence': c['sentence'],
                'arg_a_span': c['arg_a_span'], 'arg_b_span': c['arg_b_span'],
                'target0': c['target0'], 'target1': c['target1'],
                'cf_above_higher': c['views']['cf_above_higher']['text'],
                'cf_below_higher': c['views']['cf_below_higher']['text'],
                'inv_above_higher': c['views']['inv_above_higher']['text'],
                'erased_higher': c['views']['erased_higher']['text'],
                'freq_bin_a': c['freq_bin_a'], 'freq_bin_b': c['freq_bin_b'], 'evidence_pattern': c['evidence_pattern'],
            }, ensure_ascii=False) + '\n')
    summary = {
        'status': 'SPATIAL_EXCHANGE_V5_BUILD',
        'created_utc': now(),
        'corpus': str(args.corpus),
        'tokenizer': str(args.tokenizer),
        'rows_seen': rows,
        'sentences_seen': sent_count,
        'regex_matches': match_count,
        'cases': len(cases),
        'by_attested_relation': dict(Counter(c['attested_relation'] for c in cases)),
        'by_evidence_pattern': dict(Counter(c['evidence_pattern'] for c in cases)),
        'target_token_len': dict(Counter(str(c['target_token_len']) for c in cases)),
        'cap_class': dict(Counter(c['cap_class'] for c in cases)),
        'target_perm_missing': missing,
        'generated_counterfactual_dualquery_words_total': int(sum(c['generated_pair_words_counterfactual_dualquery'] for c in cases)),
        'generated_full_factorial_words_total': int(sum(c['generated_pair_words_full_factorial'] for c in cases)),
        'rejects': dict(rejects),
        'cases_path': str(cases_path),
        'sample_path': str(sample_path),
        'certification': 'exact witnessed vertical relation with no parser/external model; no-update probe; all generated text must be debited if ever trained',
    }
    (args.outdir / 'spatial_exchange_v5_build_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    write_note(args.note, build_summary=summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def load_cases(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    cases = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            cases.append(json.loads(line))
            if limit and len(cases) >= limit:
                break
    return cases


def encode_with_mask(tok: Any, text: str, seq_len: int) -> tuple[list[int], list[int], list[int]]:
    enc = tok(text, add_special_tokens=False, truncation=True, max_length=seq_len, padding='max_length')
    ids = [int(x) for x in enc['input_ids']]
    attn = [int(x) for x in enc['attention_mask']]
    mask_id = tok.mask_token_id
    pos = [i for i, x in enumerate(ids) if x == mask_id]
    if not pos:
        raise ValueError('no mask token')
    return ids, attn, pos


def labels_for(pos: list[int], target_ids: list[int], seq_len: int) -> list[int]:
    if len(pos) != len(target_ids):
        raise ValueError(f'mask count {len(pos)} != target pieces {len(target_ids)}')
    lab = [-100] * seq_len
    for p, t in zip(pos, target_ids):
        lab[int(p)] = int(t)
    return lab


def precompute_items(cases: list[dict[str, Any]], tok: Any, seq_len: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    items = []
    bad = []
    for c in cases:
        try:
            enc_views = {}
            for name, v in c['views'].items():
                ids, attn, pos = encode_with_mask(tok, v['text'], seq_len)
                corr_ids = c['target0_ids'] if v['correct'] == 't0' else c['target1_ids']
                wrong_ids = c['target1_ids'] if v['wrong'] == 't1' else c['target0_ids']
                enc_views[name] = {
                    'ids': ids, 'attn': attn, 'pos': pos,
                    'correct_labels': labels_for(pos, corr_ids, seq_len),
                    'wrong_labels': labels_for(pos, wrong_ids, seq_len),
                }
                if c.get('target_perm_case_id'):
                    p0 = c['target_perm_target0_ids']; p1 = c['target_perm_target1_ids']
                    # Preserve t0/t1 orientation with donor target pair.
                    pcorr = p0 if v['correct'] == 't0' else p1
                    pwrong = p1 if v['wrong'] == 't1' else p0
                    enc_views[name]['perm_correct_labels'] = labels_for(pos, pcorr, seq_len)
                    enc_views[name]['perm_wrong_labels'] = labels_for(pos, pwrong, seq_len)
            items.append({'case': c, 'views': enc_views})
        except Exception as e:
            bad.append({'case_id': c.get('case_id'), 'error': str(e)[:300]})
    return items, {'input_cases': len(cases), 'good_cases': len(items), 'bad_contexts': len(bad), 'bad_examples': bad[:20]}


def score_labels(logits_row: torch.Tensor, labels: torch.Tensor) -> float:
    pos = (labels != -100).nonzero(as_tuple=False).view(-1)
    tg = labels.index_select(0, pos)
    sel = logits_row.index_select(0, pos).float()
    vals = sel.gather(1, tg.view(-1, 1)).view(-1) - torch.logsumexp(sel, dim=-1)
    return float(vals.mean().detach().cpu().item())


def margin_from_logits(logits_row: torch.Tensor, correct_labels: list[int], wrong_labels: list[int], device: torch.device) -> float:
    cl = torch.tensor(correct_labels, dtype=torch.long, device=device)
    wl = torch.tensor(wrong_labels, dtype=torch.long, device=device)
    return score_labels(logits_row, cl) - score_labels(logits_row, wl)


def qstats(vals: list[float]) -> dict[str, Any]:
    if not vals:
        return {'n': 0}
    a = np.array(vals, dtype=float)
    return {
        'n': int(a.size), 'mean': float(np.mean(a)), 'std': float(np.std(a)),
        'stderr': float(np.std(a) / math.sqrt(max(1, a.size))), 'median': float(np.median(a)),
        'p05': float(np.quantile(a, 0.05)), 'p25': float(np.quantile(a, 0.25)),
        'p75': float(np.quantile(a, 0.75)), 'p95': float(np.quantile(a, 0.95)),
        'min': float(np.min(a)), 'max': float(np.max(a)), 'success_gt0': float(np.mean(a > 0)),
    }


def score_arm(arm: str, ckpt: Path, items: list[dict[str, Any]], *, tok: Any, device: torch.device, batch_size: int, dtype: str) -> list[dict[str, Any]]:
    torch_dtype = torch.bfloat16 if dtype == 'bf16' else torch.float16 if dtype == 'fp16' else torch.float32
    print(json.dumps({'event': 'load_model', 'arm': arm, 'checkpoint': str(ckpt), 'device': str(device), 'utc': now()}), flush=True)
    model = DebertaV2ForMaskedLM.from_pretrained(str(ckpt), torch_dtype=torch_dtype).to(device)
    model.eval()
    view_names = list(items[0]['views']) if items else []
    rows = []
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            chunk = items[start:start+batch_size]
            ids = []; attn = []; index = []
            for ci, it in enumerate(chunk):
                for vn in view_names:
                    ids.append(it['views'][vn]['ids'])
                    attn.append(it['views'][vn]['attn'])
                    index.append((ci, vn))
            ids_t = torch.tensor(ids, dtype=torch.long, device=device)
            attn_t = torch.tensor(attn, dtype=torch.long, device=device)
            out = model(input_ids=ids_t, attention_mask=attn_t)
            logits = out.logits
            per_case_margins = [defaultdict(float) for _ in chunk]
            per_case_perm = [defaultdict(float) for _ in chunk]
            for ri, (ci, vn) in enumerate(index):
                ev = chunk[ci]['views'][vn]
                m = margin_from_logits(logits[ri], ev['correct_labels'], ev['wrong_labels'], device)
                per_case_margins[ci][vn] = m
                if 'perm_correct_labels' in ev:
                    pm = margin_from_logits(logits[ri], ev['perm_correct_labels'], ev['perm_wrong_labels'], device)
                    per_case_perm[ci][vn] = pm
            for ci, it in enumerate(chunk):
                c = it['case']; m = per_case_margins[ci]; pm = per_case_perm[ci]
                rec = {
                    'arm': arm,
                    'case_id': c['case_id'],
                    'source_row': c['source_row'],
                    'sentence_hash': hashlib.sha1(c['sentence'].encode('utf-8')).hexdigest()[:12],
                    'attested_relation': c['attested_relation'],
                    'evidence_pattern': c['evidence_pattern'],
                    'arg_a': c['arg_a'], 'arg_b': c['arg_b'],
                    'target0': c['target0'], 'target1': c['target1'],
                    'freq_bin_a': c['freq_bin_a'], 'freq_bin_b': c['freq_bin_b'],
                    'target_token_len': c['target_token_len'],
                    'cf_higher_m': m['cf_above_higher'] + m['cf_below_higher'],
                    'cf_lower_m': m['cf_above_lower'] + m['cf_below_lower'],
                    'cf_dual_m': m['cf_above_higher'] + m['cf_below_higher'] + m['cf_above_lower'] + m['cf_below_lower'],
                    'inverse_above_higher_m': m['inv_above_higher'],
                    'inverse_below_higher_m': m['inv_below_higher'],
                    'equiv_above_gap_abs': abs(m['cf_above_higher'] - m['inv_above_higher']),
                    'equiv_below_gap_abs': abs(m['cf_below_higher'] - m['inv_below_higher']),
                    'erased_higher_m': m['erased_higher'],
                    'erased_lower_m': m['erased_lower'],
                    'erased_dual_abs': abs(m['erased_higher']) + abs(m['erased_lower']),
                    'target_perm_dual_m': (pm.get('cf_above_higher', 0.0) + pm.get('cf_below_higher', 0.0) + pm.get('cf_above_lower', 0.0) + pm.get('cf_below_lower', 0.0)) if pm else None,
                    'both_higher_worlds_correct': bool(m['cf_above_higher'] > 0 and m['cf_below_higher'] > 0),
                    'all_four_cf_views_correct': bool(m['cf_above_higher'] > 0 and m['cf_below_higher'] > 0 and m['cf_above_lower'] > 0 and m['cf_below_lower'] > 0),
                }
                for vn in view_names:
                    rec[vn + '_margin'] = float(m[vn])
                rows.append(rec)
            del ids_t, attn_t, out, logits
            if start == 0 or (start // batch_size) % 10 == 0:
                print(json.dumps({'event': 'score_progress', 'arm': arm, 'cases_done': min(start+len(chunk), len(items)), 'cases_total': len(items), 'utc': now()}), flush=True)
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return rows


def summarize(rows: list[dict[str, Any]], build_summary: dict[str, Any] | None, precompute: dict[str, Any]) -> dict[str, Any]:
    byarm = defaultdict(list)
    for r in rows:
        byarm[r['arm']].append(r)
    metrics = ['cf_higher_m','cf_lower_m','cf_dual_m','target_perm_dual_m','erased_higher_m','erased_lower_m','erased_dual_abs','equiv_above_gap_abs','equiv_below_gap_abs']
    summary = {'status': 'SPATIAL_EXCHANGE_V5_SCORE', 'created_utc': now(), 'build_summary': build_summary or {}, 'precompute': precompute, 'arms': {}, 'arm_order': {}, 'paired_deltas': {}}
    for arm, xs in sorted(byarm.items()):
        asum = {'n_cases': len(xs), 'overall': {}, 'by_attested_relation': {}, 'by_evidence_pattern': {}}
        for m in metrics:
            vals = [r[m] for r in xs if r.get(m) is not None]
            asum['overall'][m] = qstats(vals)
        asum['overall']['both_higher_worlds_correct_frac'] = float(np.mean([r['both_higher_worlds_correct'] for r in xs])) if xs else None
        asum['overall']['all_four_cf_views_correct_frac'] = float(np.mean([r['all_four_cf_views_correct'] for r in xs])) if xs else None
        for key_name in ['attested_relation','evidence_pattern']:
            outkey = 'by_' + key_name
            for val in sorted({str(r[key_name]) for r in xs}):
                vx = [r for r in xs if str(r[key_name]) == val]
                vsum = {m: qstats([r[m] for r in vx if r.get(m) is not None]) for m in metrics}
                vsum['n_cases'] = len(vx)
                asum[outkey][val] = vsum
        summary['arms'][arm] = asum
    for m in metrics:
        means = {arm: summary['arms'][arm]['overall'][m].get('mean') for arm in byarm if summary['arms'][arm]['overall'][m].get('n')}
        means = {k: v for k, v in means.items() if v is not None}
        if means:
            summary['arm_order'][m] = {'means': means, 'rank_high_to_low': sorted(means, key=lambda a: means[a], reverse=True), 'range': max(means.values()) - min(means.values())}
    arms = sorted(byarm)
    by_key = {(r['arm'], r['case_id']): r for r in rows}
    for i, a in enumerate(arms):
        for b in arms[i+1:]:
            common = sorted({r['case_id'] for r in byarm[a]} & {r['case_id'] for r in byarm[b]})
            dsum = {'n_common': len(common)}
            for m in metrics:
                vals = [by_key[(b,k)][m] - by_key[(a,k)][m] for k in common if by_key[(b,k)].get(m) is not None and by_key[(a,k)].get(m) is not None]
                dsum[b + '_minus_' + a + '_' + m] = qstats(vals)
            summary['paired_deltas'][b + '_minus_' + a] = dsum
    return summary


def write_note(note_path: Path, *, build_summary: dict[str, Any] | None = None, score_summary: dict[str, Any] | None = None) -> None:
    note_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ['# research — witness-grounded spatial exchange v5\n\n']
    if build_summary:
        bs = build_summary
        lines += [
            f'Created: {bs.get("created_utc")}\n\n',
            '## Build\n\n',
            f'Rows seen: {bs.get("rows_seen")}; sentences: {bs.get("sentences_seen")}; regex matches: {bs.get("regex_matches")}; cases: **{bs.get("cases")}**.\n\n',
            f'By attested relation: `{bs.get("by_attested_relation")}`. Evidence patterns: `{bs.get("by_evidence_pattern")}`.\n\n',
            f'Generated words if trained once: counterfactual dual-query `{bs.get("generated_counterfactual_dualquery_words_total")}`, full factorial `{bs.get("generated_full_factorial_words_total")}`. This step uses no generated text for training.\n\n',
            f'Rejects: `{bs.get("rejects")}`.\n\n',
            f'Cases: `{bs.get("cases_path")}`; samples: `{bs.get("sample_path")}`.\n\n',
        ]
    if score_summary:
        ss = score_summary
        lines += ['## Frozen checkpoint score\n\n']
        lines += ['| metric | compact | rowblock | interleaved | rank | range |\n', '|---|---:|---:|---:|---|---:|\n']
        for m in ['cf_higher_m','cf_lower_m','cf_dual_m','target_perm_dual_m','erased_dual_abs','equiv_above_gap_abs','equiv_below_gap_abs']:
            order = ss.get('arm_order', {}).get(m, {})
            means = order.get('means', {})
            lines.append(f"| {m} | {means.get('compact')} | {means.get('rowblock')} | {means.get('interleaved')} | {' > '.join(order.get('rank_high_to_low', []))} | {order.get('range')} |\n")
        lines.append('\nInterpretation: useful relation signal requires unsaturated counterfactual margins, arm separation larger than target-permutation behavior, low relation-erased absolute margins, and preferably rowblock-positive movement on spatial exchange consistent with the previous GlobalPIQA_parallel tradeoff.\n\n')
        lines.append(f"Score summary: `{ss.get('summary_path')}`; rows: `{ss.get('rows_path')}`.\n")
    note_path.write_text(''.join(lines), encoding='utf-8')


def score(args: argparse.Namespace) -> dict[str, Any]:
    cases_path = args.cases or (args.outdir / 'spatial_exchange_v5_cases.jsonl')
    build_path = args.outdir / 'spatial_exchange_v5_build_summary.json'
    build_summary = json.loads(build_path.read_text()) if build_path.exists() else None
    tok = AutoTokenizer.from_pretrained(str(args.tokenizer), use_fast=True)
    cases = load_cases(cases_path, args.score_max_cases)
    items, pre = precompute_items(cases, tok, args.seq_length)
    if not items:
        raise SystemExit('no scoreable v5 cases')
    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else f'cuda:{args.gpu}')
    rows = []
    arms = {k: v for k, v in DEFAULT_ARMS.items() if (not args.arms or k in set(args.arms))}
    for arm, ckpt in arms.items():
        if not ckpt.exists():
            raise FileNotFoundError(f'{arm} checkpoint missing: {ckpt}')
        rows.extend(score_arm(arm, ckpt, items, tok=tok, device=device, batch_size=args.batch_size, dtype=args.dtype))
    summary = summarize(rows, build_summary, pre)
    score_dir = args.outdir / ('score_cpu' if device.type == 'cpu' else 'score_gpu')
    if args.score_max_cases:
        score_dir = args.outdir / f'score_first{args.score_max_cases}_' / ('cpu' if device.type == 'cpu' else 'gpu')
    score_dir.mkdir(parents=True, exist_ok=True)
    rows_path = score_dir / 'spatial_exchange_v5_scores.jsonl'
    csv_path = score_dir / 'spatial_exchange_v5_scores.csv'
    summary_path = score_dir / 'spatial_exchange_v5_score_summary.json'
    with rows_path.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    if rows:
        with csv_path.open('w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    summary['summary_path'] = str(summary_path); summary['rows_path'] = str(rows_path); summary['csv_path'] = str(csv_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    write_note(args.note, build_summary=build_summary, score_summary=summary)
    print(json.dumps({'status':'SPATIAL_EXCHANGE_V5_SCORE_DONE','cases':len(items),'arms':list(arms),'summary':str(summary_path),'note':str(args.note)}, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['build','score','build-score'], default='build')
    ap.add_argument('--corpus', type=Path, default=DEFAULT_CORPUS)
    ap.add_argument('--tokenizer', type=Path, default=DEFAULT_TOKENIZER)
    ap.add_argument('--outdir', type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument('--note', type=Path, default=DEFAULT_NOTE)
    ap.add_argument('--cases', type=Path, default=None)
    ap.add_argument('--max-rows', type=int, default=0)
    ap.add_argument('--freq-rows', type=int, default=0)
    ap.add_argument('--max-cases', type=int, default=0)
    ap.add_argument('--max-target-pieces', type=int, default=1)
    ap.add_argument('--max-logfreq-delta', type=float, default=2.5)
    ap.add_argument('--min-sentence-chars', type=int, default=25)
    ap.add_argument('--max-sentence-chars', type=int, default=280)
    ap.add_argument('--require-concrete-hint', action='store_true')
    ap.add_argument('--balance-attested', action='store_true')
    ap.add_argument('--score-max-cases', type=int, default=0)
    ap.add_argument('--seq-length', type=int, default=256)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--dtype', choices=['bf16','fp16','fp32'], default='bf16')
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--arms', nargs='*', default=None)
    return ap.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.mode in {'build','build-score'}:
        build(args)
    if args.mode in {'score','build-score'}:
        score(args)
