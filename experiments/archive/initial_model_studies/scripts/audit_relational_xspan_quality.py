#!/usr/bin/env python3
from __future__ import annotations
import json, pathlib, collections

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
INP = ROOT/'data/xspan_revision_115/relational_xspan_rule_seed115_target200000_actual.jsonl'
OUT = ROOT/'data/relational_xspan_quality_audit.json'
NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/relational_xspan_quality_audit.md')

BAD_END = {'in','of','to','by','for','with','as','on','at','from','until','according','because','however','which','who','that','and','or','but','the','a','an'}
BAD_START = {'because','however','which','who','where','when','while','although','and','or','but'}
STOP = {'a','an','the','is','are','was','were','be','been','being','and','or','but','of','to','for','with','by','as','in','on','at','from','which','who','that','this','these','those','it','they','them','their','its','however','because'}

def clean(w: str) -> str:
    return w.strip('.,;:!?"()[]{}').lower()

def main():
    rows = [json.loads(l) for l in INP.open(encoding='utf-8') if l.strip()]
    by = collections.Counter(); reasons = collections.Counter()
    bad = []; good = []
    for r in rows:
        ws = [clean(w) for w in r['target_text'].split() if clean(w)]
        rr = []
        if not ws:
            rr.append('empty')
        else:
            if ws[0] in BAD_START:
                rr.append('bad_start')
            if ws[-1] in BAD_END:
                rr.append('bad_end')
            if len(ws) > 10:
                rr.append('too_long')
            if sum(1 for w in ws if w not in STOP and len(w) >= 4) == 0:
                rr.append('no_content_word')
        if rr:
            by[r['target_type']] += 1
            for x in rr:
                reasons[x] += 1
            if len(bad) < 40:
                bad.append({'example_id': r['example_id'], 'target_type': r['target_type'], 'target_text': r['target_text'], 'reasons': rr, 's2': r['s2']})
        elif len(good) < 20:
            good.append({'example_id': r['example_id'], 'target_type': r['target_type'], 'target_text': r['target_text'], 's2': r['s2']})
    flagged = sum(by.values())
    summary = {
        'status': 'RELATIONAL_XSPAN_QUALITY_AUDIT',
        'jsonl': str(INP), 'rows': len(rows), 'flagged_rows': flagged,
        'flagged_frac': flagged / len(rows) if rows else None,
        'flagged_by_target_type': dict(by), 'reason_counts': dict(reasons),
        'bad_examples': bad, 'good_examples': good,
    }
    OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    lines = ['# research — relational XSpan quality audit', '', f'JSONL: `{INP}`', f'Evidence JSON: `{OUT}`', '',
             f'Rows: **{len(rows)}**', f'Flagged rows: **{flagged}** ({flagged/len(rows):.3%})',
             f'Flagged by type: {dict(by)}', f'Reason counts: {dict(reasons)}', '', '## Example flagged targets']
    for e in bad[:15]:
        lines.append(f"- {e['target_type']} `{e['target_text']}` {e['reasons']} :: {e['s2'][:110]}")
    lines += ['', '## Example clean targets']
    for e in good[:10]:
        lines.append(f"- {e['target_type']} `{e['target_text']}` :: {e['s2'][:110]}")
    NOTE.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'status': summary['status'], 'out': str(OUT), 'rows': len(rows), 'flagged_frac': summary['flagged_frac'], 'reason_counts': dict(reasons)}, indent=2))

if __name__ == '__main__':
    main()
