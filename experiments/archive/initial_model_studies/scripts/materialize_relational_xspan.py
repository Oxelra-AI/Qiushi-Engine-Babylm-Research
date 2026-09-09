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
OUT_DIR_DEFAULT = ROOT/'data/xspan_revision_115'
NOTE_DEFAULT = (ROOT.parents[2] / 'research/notes/initial_model_studies/relational_xspan_materialization.md')
SENT_RE = re.compile(r'(?<=[.!?])\s+')
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-'][A-Za-z0-9]+)?")
DEP_STARTS = {'it','they','this','these','that','those'}
STOP_TARGET = {'is','are','was','were','be','been','being','a','an','the','and','or','but','of','to','for','with','by','as','in','on','at','from','which','who','that','this','these','those','it','they','them','their','its'}
ANCHOR_VERBS = {'is','are','was','were','means','refers','has','have','contains','includes','uses','covers','describes','represents','forms','consists','lives','live','accepts','works','stars','released','split','put','located','found','based'}
LOCATION_MARKERS = {'in','on','at','near','from','inside','outside','north','south','east','west','within','around','beside','between'}
ACTION_MARKERS = {'contains','includes','uses','accepts','works','stars','released','split','put','provides','features','covers','describes','represents','forms','consists','lives','live','located','found','based'}


def split_sents(line: str) -> list[str]:
    return [s.strip() for s in SENT_RE.split(line.strip()) if s.strip()]


def word_items(text: str):
    return [(m.group(0), m.start(), m.end()) for m in WORD_RE.finditer(text)]


def words(text: str) -> list[str]:
    return [w for w,_,_ in word_items(text)]


def norm(w: str) -> str:
    return w.lower().strip("-'")


def count_ws(text: str) -> int:
    return len(text.split())


def sent_len_bin(n: int) -> str:
    if n <= 10: return 's_le10'
    if n <= 20: return 's_11_20'
    if n <= 40: return 's_21_40'
    return 's_41p'


def has_s1_anchor(s1: str) -> bool:
    ws = [norm(w) for w in words(s1)]
    if len(ws) < 4:
        return False
    return any(w in ANCHOR_VERBS for w in ws[:14])


def phrase_end(items, start_i: int, max_words: int = 9) -> int:
    # Stop before relative clauses, punctuation-heavy continuations, or conjunction after at least two content words.
    end_i = start_i
    content_seen = 0
    for j in range(start_i, min(len(items), start_i + max_words)):
        w,a,b = items[j]
        x = norm(w)
        if j > start_i and x in {'which','who','where','when','while','because','although'}:
            break
        if j > start_i + 1 and x in {'and','or','but'}:
            break
        content_seen += int(x not in STOP_TARGET and len(x) >= 3)
        end_i = j
        # If phrase reaches a comma boundary in raw text before next word, stop after current word.
    return end_i


def choose_target(s2: str) -> dict | None:
    items = word_items(s2)
    if len(items) < 3:
        return None
    dep = norm(items[0][0])
    if dep not in DEP_STARTS:
        return None
    xs = [norm(w) for w,_,_ in items]
    # Pattern 1: location/spatial complements, e.g. It lives in India; They are in the Bering Sea.
    for i,x in enumerate(xs[1:], start=1):
        if x in LOCATION_MARKERS:
            # Include a preceding predicate if informative (lives/located/found/is/are) and close.
            start_i = i
            if i >= 2 and xs[i-1] in {'lives','live','located','found','based','is','are','was','were'}:
                start_i = i-1
            end_i = phrase_end(items, i, max_words=7)
            a = items[start_i][1]; b = items[end_i][2]
            txt = s2[a:b].strip()
            if valid_target_text(txt):
                return {'target_type':'location_spatial_phrase','target_span_s2':[a,b],'target_text':txt,'dependent':items[0][0]}
    # Pattern 2: object/affordance/action-result phrases after action markers.
    for i,x in enumerate(xs[1:], start=1):
        if x in ACTION_MARKERS:
            end_i = phrase_end(items, i, max_words=8)
            a = items[i][1]; b = items[end_i][2]
            txt = s2[a:b].strip()
            if valid_target_text(txt):
                return {'target_type':'action_object_result_phrase','target_span_s2':[a,b],'target_text':txt,'dependent':items[0][0]}
    # Pattern 3: definition/property complement after It/This is/are/was/were.
    for i,x in enumerate(xs[1:5], start=1):
        if x in {'is','are','was','were'}:
            # mask content complement after copula, not just article.
            start_i = i+1
            while start_i < len(items) and norm(items[start_i][0]) in {'a','an','the'}:
                start_i += 1
            if start_i < len(items):
                end_i = phrase_end(items, start_i, max_words=7)
                a = items[start_i][1]; b = items[end_i][2]
                txt = s2[a:b].strip()
                if valid_target_text(txt):
                    return {'target_type':'definition_property_complement','target_span_s2':[a,b],'target_text':txt,'dependent':items[0][0]}
    # Pattern 4: fallback first content phrase after dependent, if it contains a real content word and is short.
    start_i = None
    for i in range(1, min(len(items), 8)):
        x = xs[i]
        if x not in STOP_TARGET and len(x) >= 4:
            start_i = i; break
    if start_i is not None:
        end_i = phrase_end(items, start_i, max_words=5)
        a = items[start_i][1]; b = items[end_i][2]
        txt = s2[a:b].strip()
        if valid_target_text(txt):
            return {'target_type':'semantic_content_continuation','target_span_s2':[a,b],'target_text':txt,'dependent':items[0][0]}
    return None


def valid_target_text(txt: str) -> bool:
    ws = [norm(w) for w in words(txt)]
    if not ws:
        return False
    content = [w for w in ws if w not in STOP_TARGET and len(w) >= 4]
    if not content:
        return False
    if len(ws) > 10:
        return False
    return True


def collect_candidates(raw_dir: pathlib.Path, source: str, min_s1_words: int, max_s1_words: int, min_s2_words: int, max_s2_words: int):
    p = raw_dir/source
    rows=[]
    with p.open('r', encoding='utf-8', errors='replace') as f:
        for line_no,line in enumerate(f,1):
            sents=split_sents(line)
            for i in range(len(sents)-1):
                s1=sents[i]; s2=sents[i+1]
                w1=len(words(s1)); w2=len(words(s2))
                if not (min_s1_words <= w1 <= max_s1_words and min_s2_words <= w2 <= max_s2_words):
                    continue
                if not has_s1_anchor(s1):
                    continue
                t=choose_target(s2)
                if not t:
                    continue
                rows.append({'source':source,'line_no':line_no,'sent_index':i,'s1':s1,'s2':s2,'s1_word_count':w1,'s2_word_count':w2,**t})
    return rows


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--raw_dir', default=str(RAW_DIR_DEFAULT))
    ap.add_argument('--out_dir', default=str(OUT_DIR_DEFAULT))
    ap.add_argument('--source', default='simple_wiki.train.txt')
    ap.add_argument('--target_counted_words', type=int, default=200000)
    ap.add_argument('--seed', type=int, default=115)
    ap.add_argument('--min_s1_words', type=int, default=4)
    ap.add_argument('--max_s1_words', type=int, default=70)
    ap.add_argument('--min_s2_words', type=int, default=4)
    ap.add_argument('--max_s2_words', type=int, default=70)
    args=ap.parse_args()
    raw_dir=pathlib.Path(args.raw_dir); out_dir=pathlib.Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rng=random.Random(args.seed)
    cand=collect_candidates(raw_dir,args.source,args.min_s1_words,args.max_s1_words,args.min_s2_words,args.max_s2_words)
    # wrong-s1 controls: same source and rough s1 length, different line; no model scores used.
    idx=defaultdict(list)
    for r in cand:
        idx[sent_len_bin(r['s1_word_count'])].append(r)
    rng.shuffle(cand)
    rows=[]; counted=0; by_type=Counter(); skipped=Counter()
    for r in cand:
        pool=[q for q in idx[sent_len_bin(r['s1_word_count'])] if q['line_no']!=r['line_no']]
        if not pool:
            skipped['no_wrong_s1']+=1; continue
        wrong=min(rng.sample(pool, min(20,len(pool))), key=lambda q: abs(q['s1_word_count']-r['s1_word_count']))
        text_true=r['s1']+' '+r['s2']
        text_wrong=wrong['s1']+' '+r['s2']
        text_no=r['s2']
        words_true=count_ws(text_true); words_wrong=count_ws(text_wrong); words_no=count_ws(text_no)
        # Training true-context XSpan counts true stream; wrong/no streams are controls and counted only when used.
        if rows and counted + words_true > args.target_counted_words:
            break
        a,b=r['target_span_s2']
        row={'example_id':len(rows),'split':'heldout' if len(rows)%20==0 else 'train',
             'source':r['source'],'line_no':r['line_no'],'sent_index':r['sent_index'],
             's1':r['s1'],'s2':r['s2'],'text':text_true,'text_wrong_s1':text_wrong,'text_no_s1':text_no,
             'words':words_true,'words_wrong_s1':words_wrong,'words_no_s1':words_no,
             's2_start_char_text':len(r['s1'])+1,'s2_start_char_wrong_s1':len(wrong['s1'])+1,'s2_start_char_no_s1':0,
             'target_span_s2':[a,b],'target_text':r['target_text'],'target_type':r['target_type'],'dependent':r['dependent'],
             'target_span_text':[len(r['s1'])+1+a, len(r['s1'])+1+b],
             'target_span_wrong_s1':[len(wrong['s1'])+1+a, len(wrong['s1'])+1+b],
             'target_span_no_s1':[a,b],
             'wrong_s1':{'source':wrong['source'],'line_no':wrong['line_no'],'sent_index':wrong['sent_index'],'s1':wrong['s1'],'s1_word_count':wrong['s1_word_count']},
             'materializer_version':'model_independent_relational_xspan',
             'selection_policy':'rule_based_only_no_model_score_filtering'}
        rows.append(row); counted += words_true; by_type[r['target_type']]+=1
    out_path=out_dir/f'relational_xspan_rule_seed{args.seed}_target{args.target_counted_words}_actual.jsonl'
    with out_path.open('w', encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False)+'\n')
    summary={'status':'RELATIONAL_XSPAN_RULE_MATERIALIZED','out_jsonl':str(out_path),'raw_dir':str(raw_dir),'source':args.source,'seed':args.seed,
             'candidate_rows_before_budget':len(cand),'num_rows':len(rows),'true_context_counted_words':counted,'target_counted_words':args.target_counted_words,
             'by_target_type':dict(by_type),'heldout_rows':sum(r['split']=='heldout' for r in rows),'train_rows':sum(r['split']=='train' for r in rows),
             'skipped':dict(skipped),'policy':'No protected-model scores used for selection. Scores from research were used only to derive corpus-intrinsic target categories.'}
    summary_path=out_path.with_suffix('.summary.json')
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    lines=['# research — model-independent relational XSpan materialization','',f'JSONL: `{out_path}`',f'Summary: `{summary_path}`','',
           f'Rows: **{len(rows)}**',f'True-context counted words: **{counted}**',f'By target type: {dict(by_type)}','',
           '**Policy:** no model scores were used to choose rows. research protected-model scores only informed the rule family: do not target initial dependent tokens; target semantic s2 content phrases in definition/description continuations.','',
           'Each row stores true-s1, same-source length-matched wrong-s1, and no-s1 context forms for mechanism probes and controls. A trainer can use `text`/`target_span_text` for true XSpan and `text_wrong_s1`/`target_span_wrong_s1` for wrong-context control.']
    NOTE_DEFAULT.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'status':summary['status'],'jsonl':str(out_path),'summary':str(summary_path),'rows':len(rows),'counted_words':counted,'by_target_type':dict(by_type)}, indent=2))

if __name__=='__main__': main()
