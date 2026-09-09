#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import random
import re
from collections import Counter, defaultdict

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
RAW_DIR_DEFAULT = ROOT/'data/reconstruct_tmp/raw_dataset'
OUT_DIR_DEFAULT = ROOT/'data/counterfactual_revision_109'
NOTE_DEFAULT = (ROOT.parents[2] / 'research/notes/initial_model_studies/linked_definition_counterfactual_materialization.md')
SENT_RE = re.compile(r'(?<=[.!?])\s+')
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-'][A-Za-z0-9]+)?")
STOP = {'the','a','an','and','or','but','of','for','to','in','on','at','by','with','from','as','is','are','was','were','be','been','being','that','which','who','it','its','they','them','their','this','these','those','there','have','has','had','can','could','will','would','should','may','might','many','some','several','most','more','other','same','new','old','first','second'}
DET = {'a','an','the'}
VERB_ANCHORS = {'is','are','was','were','means','refers','has','have','can','contains','includes','uses','covers','describes','represents','forms'}
DEPENDENT_STARTS = {'it','they','this','these','that','those'}
BAD_SUFFIX = ('ing','ly','ed')

def split_sents(line: str) -> list[str]:
    return [s.strip() for s in SENT_RE.split(line.strip()) if s.strip()]

def words(text: str) -> list[str]:
    return [m.group(0) for m in WORD_RE.finditer(text)]

def count_ws(text: str) -> int:
    return len(text.split())

def norm(w: str) -> str:
    return w.lower().strip("-'")

def is_vowel(w: str) -> bool:
    return bool(w) and w[0].lower() in 'aeiou'

def plural_flag(w: str) -> str:
    x = norm(w)
    return 'plural_s' if x.endswith('s') and not x.endswith('ss') else 'singular_or_mass'

def len_bin(n: int) -> str:
    if n <= 4: return 'l4'
    if n <= 7: return 'l5_7'
    if n <= 11: return 'l8_11'
    return 'l12p'

def freq_bin(c: int) -> str:
    if c < 5: return 'lt5'
    if c < 20: return '5_19'
    if c < 100: return '20_99'
    if c < 500: return '100_499'
    return '500p'

def sent_len_bin(n: int) -> str:
    if n <= 10: return 's_le10'
    if n <= 20: return 's_11_20'
    if n <= 40: return 's_21_40'
    return 's_41p'

def content_ok(w: str) -> bool:
    x = norm(w)
    return len(x) >= 4 and x not in STOP and not x.isdigit() and bool(re.search('[a-z]', x)) and not x.endswith(BAD_SUFFIX)

def dependent_span(s2: str) -> dict | None:
    ws = words(s2)
    if not ws:
        return None
    first = norm(ws[0])
    if first not in DEPENDENT_STARTS:
        return None
    m = WORD_RE.search(s2)
    if not m:
        return None
    return {'dependent_text': m.group(0), 'dependent_lower': first, 'dependent_span': [m.start(), m.end()], 'dependent_position': 's2_initial'}

def subject_head(s1: str) -> dict | None:
    toks = words(s1)
    if len(toks) < 5:
        return None
    det = norm(toks[0])
    if det not in DET:
        return None
    # Find first anchor verb after a short subject; head is the last content word before it.
    verb_i = None
    for i in range(1, min(len(toks), 10)):
        if norm(toks[i]) in VERB_ANCHORS:
            verb_i = i
            break
    if verb_i is None or verb_i < 2:
        return None
    head_i = None
    for j in range(verb_i - 1, 0, -1):
        if content_ok(toks[j]):
            head_i = j
            break
    if head_i is None:
        return None
    head = toks[head_i]
    # Locate char span by word sequence order.
    matches = list(WORD_RE.finditer(s1))
    if head_i >= len(matches):
        return None
    m = matches[head_i]
    # Article agreement; if det is a/an, replacement must preserve vowel class.
    article_vowel = 'na'
    if det in {'a','an'}:
        article_vowel = 'vowel' if is_vowel(norm(head)) else 'consonant'
    return {'det': det, 'head': head, 'head_lower': norm(head), 'head_token_index': head_i, 'head_span': [m.start(), m.end()], 'verb': toks[verb_i], 'verb_index': verb_i, 'article_vowel': article_vowel, 'plural': plural_flag(head)}

def apply_case(rep: str, orig: str) -> str:
    if orig[:1].isupper():
        return rep.capitalize()
    return rep.lower()

def collect(raw_dir: pathlib.Path, source: str):
    p = raw_dir/source
    rows=[]; freq=Counter(); heads=[]
    with p.open('r', encoding='utf-8', errors='replace') as f:
        for line_no,line in enumerate(f,1):
            sents = split_sents(line)
            for i in range(len(sents)-1):
                s1=sents[i]; s2=sents[i+1]
                h=subject_head(s1); dep=dependent_span(s2)
                if h and dep:
                    w1=len(words(s1)); w2=len(words(s2))
                    if 5 <= w1 <= 60 and 4 <= w2 <= 60:
                        rows.append({'source':source,'line_no':line_no,'sent_index':i,'s1':s1,'s2':s2,'s1_word_count':w1,'s2_word_count':w2,'subject':h,'dependent':dep})
                        heads.append(h['head_lower'])
                for w in words(line):
                    if content_ok(w):
                        freq[norm(w)] += 1
    return rows, freq, heads

def make_repl_index(heads: list[str], freq: Counter):
    idx=defaultdict(list)
    for h in set(heads):
        c=freq.get(h,0)
        if c < 5:
            continue
        key_base=(plural_flag(h), len_bin(len(h)), freq_bin(c), 'vowel' if is_vowel(h) else 'consonant')
        idx[key_base].append(h)
        idx[(plural_flag(h), len_bin(len(h)), freq_bin(c), 'ANY_VOWEL')].append(h)
    return {k:sorted(v) for k,v in idx.items() if len(v)>=2}

def choose_replacement(h: dict, freq: Counter, idx: dict, rng: random.Random):
    x=h['head_lower']; c=freq.get(x,0)
    vowel='vowel' if is_vowel(x) else 'consonant'
    key=(h['plural'], len_bin(len(x)), freq_bin(c), vowel if h['article_vowel']!='na' else 'ANY_VOWEL')
    pool=[z for z in idx.get(key,[]) if z!=x]
    if not pool and h['article_vowel']=='na':
        # relax frequency but keep plural and length.
        pool=[z for (pl,lb,fb,av),zs in idx.items() if pl==h['plural'] and lb==len_bin(len(x)) for z in zs if z!=x]
    if not pool:
        return None
    return rng.choice(pool)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--raw_dir', default=str(RAW_DIR_DEFAULT))
    ap.add_argument('--out_dir', default=str(OUT_DIR_DEFAULT))
    ap.add_argument('--target_counted_words', type=int, default=100000)
    ap.add_argument('--seed', type=int, default=1093)
    ap.add_argument('--source', default='simple_wiki.train.txt')
    args=ap.parse_args()
    rng=random.Random(args.seed)
    raw_dir=pathlib.Path(args.raw_dir); out_dir=pathlib.Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    candidates,freq,heads=collect(raw_dir,args.source)
    idx=make_repl_index(heads,freq)
    s2_index=defaultdict(list)
    for r in candidates:
        s2_index[sent_len_bin(r['s2_word_count'])].append(r)
    rng.shuffle(candidates)
    out=[]; counted=0; skipped=Counter(); dep_counts=Counter(); det_counts=Counter()
    for r in candidates:
        rep=choose_replacement(r['subject'],freq,idx,rng)
        if not rep:
            skipped['no_replacement']+=1; continue
        h=r['subject']; rep_surf=apply_case(rep,h['head'])
        s1p=r['s1'][:h['head_span'][0]]+rep_surf+r['s1'][h['head_span'][1]:]
        pool=[q for q in s2_index[sent_len_bin(r['s2_word_count'])] if q['line_no']!=r['line_no']]
        if not pool:
            skipped['no_random_s2']+=1; continue
        q=min(rng.sample(pool,min(len(pool),20)), key=lambda z: abs(z['s2_word_count']-r['s2_word_count']))
        text=r['s1']+' '+r['s2']; textp=s1p+' '+r['s2']; textr=s1p+' '+q['s2']
        aux=count_ws(text)+count_ws(textp)+count_ws(textr)
        if out and counted+aux>args.target_counted_words:
            break
        row={'example_id':len(out),'split':'heldout' if len(out)%20==0 else 'train','source':r['source'],'line_no':r['line_no'],'sent_index':r['sent_index'],
             's1':r['s1'],'s2':r['s2'],'s1_perturbed':s1p,'text':text,'text_perturbed':textp,'text_random_control':textr,
             'words':count_ws(text),'words_perturbed':count_ws(textp),'words_random_control':count_ws(textr),'aux_counted_words_original_perturbed_random':aux,
             's2_start_char_text':len(r['s1'])+1,'s2_start_char_perturbed':len(s1p)+1,'s2_start_char_random_control':len(s1p)+1,
             'linked_span':{'s1_subject_head':h,'s2_dependent':r['dependent'],'dependency_type':'definition_subject_to_initial_pronoun_or_deictic'},
             'edit':{'original':h['head'],'replacement':rep_surf,'original_lower':h['head_lower'],'replacement_lower':rep,'span':h['head_span'],'freq_original':freq.get(h['head_lower'],0),'freq_replacement':freq.get(rep,0),'plural':h['plural'],'article_vowel':h['article_vowel']},
             'random_s2':{'source':q['source'],'line_no':q['line_no'],'sent_index':q['sent_index'],'s2':q['s2'],'s2_word_count':q['s2_word_count']},
             'materializer_version':'v3_linked_definition_subject','auxiliary_labels':{'original_pair':0,'perturbed_s1_true_s2':1,'perturbed_s1_random_s2_control':1}}
        out.append(row); counted+=aux; dep_counts[r['dependent']['dependent_lower']]+=1; det_counts[h['det']]+=1
    out_path=out_dir/f'linked_definition_counterfactual_v3_seed{args.seed}_target{args.target_counted_words}_actual.jsonl'
    with out_path.open('w',encoding='utf-8') as f:
        for r in out: f.write(json.dumps(r,ensure_ascii=False)+'\n')
    summary={'status':'LINKED_DEFINITION_COUNTERFACTUAL_V3_MATERIALIZED','out_jsonl':str(out_path),'raw_dir':str(raw_dir),'source':args.source,'seed':args.seed,'candidate_linked_pairs':len(candidates),'num_rows':len(out),'actual_aux_counted_words_original_perturbed_random':counted,'dependent_counts':dict(dep_counts),'determiner_counts':dict(det_counts),'replacement_bucket_count':len(idx),'skipped':dict(skipped),'quality_intent':'High-precision subset: sentence 1 begins with determiner-headed subject before a short anchor verb; sentence 2 begins with pronoun/deictic dependent. Edit is the subject head, so sentence 2 explicitly uses the edited variable.'}
    summary_path=out_path.with_suffix('.summary.json')
    summary_path.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    note=NOTE_DEFAULT
    lines=['# research — linked definition counterfactual v3 materialization','',f'JSONL: `{out_path}`',f'Summary: `{summary_path}`','',f'Rows: **{len(out)}**',f'Counted aux words: **{counted}**',f'Dependent starts: {dict(dep_counts)}',f'Determiners: {dict(det_counts)}','',summary['quality_intent'],'','This v3 subset supersedes arbitrary v2 swaps for the intended state-propagation trainer; v2 should remain a matched leakage/unlinked control pool.']
    note.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':summary['status'],'jsonl':str(out_path),'summary':str(summary_path),'rows':len(out),'counted_words':counted,'candidate_linked_pairs':len(candidates)},indent=2))

if __name__=='__main__': main()
