#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, pathlib, collections, re

BAD_END = {'in','of','to','by','for','with','as','on','at','from','until','according','that','the','a','an','and','or','but','into','onto','over','under','near','between','among','through','after','before','during','whenever','next','via','towards','toward'}
BAD_START = {'because','however','which','who','where','when','while','although','and','or','but','so','then','therefore','thus'}
CLAUSE_MID = {'which','who','whom','whose','where','when','whenever','while','although','because'}
STOP = {'a','an','the','is','are','was','were','be','been','being','and','or','but','of','to','for','with','by','as','in','on','at','from','which','who','that','this','these','those','it','they','them','their','its','however','because'}
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-'][A-Za-z0-9]+)?")

def clean(w: str) -> str:
    return w.strip('.,;:!?"()[]{}').lower()

def words(txt: str):
    return [m.group(0) for m in WORD_RE.finditer(txt)]

def has_unbalanced(txt: str) -> bool:
    pairs = [('(',')'),('[',']'),('{','}'),('"','"')]
    for a,b in pairs:
        if a == b:
            if txt.count(a) % 2:
                return True
        elif txt.count(a) != txt.count(b):
            return True
    return False

def substantive(ws):
    return [w for w in ws if w not in STOP and len(w) >= 4 and not w.isdigit()]

def assess(target: str):
    ws = [clean(w) for w in target.split() if clean(w)]
    rr = []
    if not ws:
        rr.append('empty')
        return rr
    if ws[0] in BAD_START:
        rr.append('bad_start')
    if ws[-1] in BAD_END:
        rr.append('bad_end')
    if len(ws) > 10:
        rr.append('too_long')
    if len(substantive(ws)) < 2:
        rr.append('too_few_substantive_tokens')
    if any(w in CLAUSE_MID for w in ws[1:]):
        rr.append('mid_clause_trigger')
    if has_unbalanced(target):
        rr.append('unbalanced_quote_or_bracket')
    return rr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('jsonl')
    ap.add_argument('--out_json', default='')
    ap.add_argument('--out_note', default='')
    args = ap.parse_args()
    p = pathlib.Path(args.jsonl)
    rows = [json.loads(l) for l in p.open(encoding='utf-8') if l.strip()]
    by = collections.Counter(); reasons = collections.Counter(); bad=[]; good=[]
    for r in rows:
        rr = assess(r['target_text'])
        if rr:
            by[r.get('target_type','unknown')] += 1
            for x in rr: reasons[x]+=1
            if len(bad) < 50:
                bad.append({'example_id':r.get('example_id'), 'target_type':r.get('target_type'), 'target_text':r['target_text'], 'reasons':rr, 's2':r.get('s2','')})
        elif len(good) < 25:
            good.append({'example_id':r.get('example_id'), 'target_type':r.get('target_type'), 'target_text':r['target_text'], 's2':r.get('s2','')})
    flagged=sum(by.values())
    summary={'status':'RELATIONAL_XSPAN_QUALITY_SCAN','jsonl':str(p),'rows':len(rows),'flagged_rows':flagged,'flagged_frac':flagged/len(rows) if rows else None,'flagged_by_target_type':dict(by),'reason_counts':dict(reasons),'bad_examples':bad,'good_examples':good}
    out_json = pathlib.Path(args.out_json) if args.out_json else p.with_suffix('.quality_scan.json')
    out_note = pathlib.Path(args.out_note) if args.out_note else p.with_suffix('.quality_scan.md')
    out_json.parent.mkdir(parents=True, exist_ok=True); out_note.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    lines=['# Relational XSpan target quality scan','',f'JSONL: `{p}`',f'Evidence JSON: `{out_json}`','',f'Rows: **{len(rows)}**',f'Flagged rows: **{flagged}** ({(flagged/len(rows) if rows else 0):.3%})',f'Flagged by type: {dict(by)}',f'Reason counts: {dict(reasons)}','','## Flagged examples']
    for e in bad[:15]:
        lines.append(f"- {e['target_type']} `{e['target_text']}` {e['reasons']} :: {e['s2'][:120]}")
    lines += ['', '## Clean examples']
    for e in good[:12]:
        lines.append(f"- {e['target_type']} `{e['target_text']}` :: {e['s2'][:120]}")
    out_note.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'status':summary['status'],'out_json':str(out_json),'rows':len(rows),'flagged_frac':summary['flagged_frac'],'reason_counts':dict(reasons)}, indent=2))
if __name__ == '__main__': main()
