#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, pathlib, re
from collections import Counter

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
DEFAULT_IN = ROOT/'data/xspan_revision_117/relational_xspan_v3_seed117_target200000_actual.jsonl'
DEFAULT_OUT_DIR = ROOT/'data/xspan_revision_118'
NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/relational_xspan_compact_v4_filter.md')
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-'][A-Za-z0-9]+)?")
HIDDEN_BOUNDARY = re.compile(r'[a-z]\.[A-Z]')
STOP = {'a','an','the','is','are','was','were','be','been','being','and','or','but','of','to','for','with','by','as','in','on','at','from','which','who','that','this','these','those','it','they','them','their','its','however','because'}
BAD_START = {'because','however','which','who','where','when','while','although','and','or','but','so','then','therefore','thus'}
BAD_END = {'in','of','to','by','for','with','as','on','at','from','until','according','that','the','a','an','and','or','but','into','onto','over','under','near','between','among','through','after','before','during','whenever','next','via','towards','toward'}
CLAUSE = {'which','who','whom','whose','where','when','whenever','while','although','because'}

def words(s: str):
    return [m.group(0) for m in WORD_RE.finditer(s)]

def clean(w: str) -> str:
    return w.strip('.,;:!?"()[]{}').lower()

def balanced(s: str) -> bool:
    if s.count('"') % 2:
        return False
    return all(s.count(a)==s.count(b) for a,b in [('(',')'),('[',']'),('{','}')])

def ok_target(txt: str):
    ws = [clean(w) for w in txt.split() if clean(w)]
    reasons=[]
    if not ws:
        reasons.append('empty')
        return False, reasons
    if len(ws) < 2 or len(ws) > 7:
        reasons.append('length_not_2_to_7_words')
    if ws[0] in BAD_START:
        reasons.append('bad_start')
    if ws[-1] in BAD_END:
        reasons.append('bad_end')
    if any(w in CLAUSE for w in ws[1:]):
        reasons.append('mid_clause_trigger')
    if not balanced(txt):
        reasons.append('unbalanced_quote_or_bracket')
    # Compact repair: no broad lists/comma tails or embedded sentence punctuation.
    if any(ch in txt for ch in ['.', ',', ';', ':']):
        reasons.append('punctuation_or_list_tail')
    subs=[w for w in ws if w not in STOP and len(w)>=4 and not w.isdigit()]
    if len(subs) < 2:
        reasons.append('too_few_substantive_tokens')
    return not reasons, reasons

def ok_row(r):
    reasons=[]
    if HIDDEN_BOUNDARY.search(r.get('s1','')) or HIDDEN_BOUNDARY.search(r.get('s2','')) or HIDDEN_BOUNDARY.search(r.get('target_text','')):
        reasons.append('hidden_sentence_boundary')
    ok, rr = ok_target(r.get('target_text',''))
    reasons.extend(rr)
    return (len(reasons)==0), reasons

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input', default=str(DEFAULT_IN))
    ap.add_argument('--out_dir', default=str(DEFAULT_OUT_DIR))
    ap.add_argument('--target_counted_words', type=int, default=200000)
    ap.add_argument('--seed_label', default='117')
    args=ap.parse_args()
    inp=pathlib.Path(args.input); out_dir=pathlib.Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows=[json.loads(l) for l in inp.open(encoding='utf-8') if l.strip()]
    out=[]; counted=0; reasons=Counter(); by_type=Counter(); kept_by_type=Counter()
    examples_bad=[]
    for r in rows:
        by_type[r.get('target_type','unknown')]+=1
        ok, rr = ok_row(r)
        if not ok:
            for x in rr: reasons[x]+=1
            if len(examples_bad)<25:
                examples_bad.append({'example_id':r.get('example_id'),'target_type':r.get('target_type'),'target_text':r.get('target_text'),'reasons':rr,'s2':r.get('s2')})
            continue
        if out and counted + int(r['words']) > args.target_counted_words:
            break
        nr=dict(r)
        nr['old_example_id']=r.get('example_id')
        nr['example_id']=len(out)
        nr['split']='heldout' if len(out)%20==0 else 'train'
        nr['materializer_version']='relational_xspan_compact_v4_filter_from_v3'
        nr['selection_policy']='rule_based_filter_no_model_score_filtering'
        out.append(nr); counted += int(nr['words']); kept_by_type[nr.get('target_type','unknown')]+=1
    out_path=out_dir/f'relational_xspan_compact_v4_from_v3_seed{args.seed_label}_target{args.target_counted_words}_actual.jsonl'
    with out_path.open('w', encoding='utf-8') as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False)+'\n')
    summary={'status':'RELATIONAL_XSPAN_COMPACT_V4_FILTERED','input':str(inp),'out_jsonl':str(out_path),'input_rows':len(rows),'kept_rows':len(out),'true_context_counted_words':counted,'target_counted_words':args.target_counted_words,'input_by_target_type':dict(by_type),'kept_by_target_type':dict(kept_by_type),'rejection_reason_counts':dict(reasons),'bad_examples':examples_bad,'policy':'Compact final repair only: filters v3 by corpus/text-intrinsic target quality; no protected-model score filtering.'}
    summary_path=out_path.with_suffix('.summary.json')
    summary_path.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    lines=['# research — compact relational XSpan v4 filter', '', f'Input: `{inp}`', f'Output JSONL: `{out_path}`', f'Summary: `{summary_path}`','',f'Kept rows: **{len(out)}** / {len(rows)}',f'True-context counted words: **{counted}**',f'Kept by target type: {dict(kept_by_type)}',f'Rejection reasons: {dict(reasons)}','', '**Policy:** this is the final compact data-boundary repair before likelihood testing; no model scores were used for selection. If same-target true/wrong/no-s1 likelihood signal is weak, change target representation rather than stacking more filters.']
    NOTE.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'status':summary['status'],'jsonl':str(out_path),'summary':str(summary_path),'kept_rows':len(out),'counted_words':counted,'kept_by_target_type':dict(kept_by_type),'rejection_reason_counts':dict(reasons)},indent=2))
if __name__=='__main__': main()
