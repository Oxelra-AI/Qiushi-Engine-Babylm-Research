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
OUT_DIR_DEFAULT = ROOT/'data/xspan_revision_116'
NOTE_DEFAULT = (ROOT.parents[2] / 'research/notes/initial_model_studies/relational_xspan_materialization_v2.md')
SENT_RE = re.compile(r'(?<=[.!?])\s+')
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-'][A-Za-z0-9]+)?")
DEP_STARTS = {'it','they','this','these','that','those'}
BAD_START = {'because','however','which','who','where','when','while','although','and','or','but','so','then','therefore','thus'}
# Words that must not end a target span; a target should extend to a real complement head instead.
DANGLING_END = {'in','of','to','by','for','with','as','on','at','from','until','according','that','the','a','an','and','or','but','into','onto','over','under','near','between','among','through'}
STOP = {'a','an','the','is','are','was','were','be','been','being','and','or','but','of','to','for','with','by','as','in','on','at','from','which','who','that','this','these','those','it','they','them','their','its','however','because'}
ANCHOR_VERBS = {'is','are','was','were','means','refers','has','have','contains','includes','uses','covers','describes','represents','forms','consists','lives','live','accepts','works','stars','released','split','put','located','found','based','became','received','provides','features','distributed','named','extends','allows'}
LOCATION_MARKERS = {'in','on','at','near','from','inside','outside','north','south','east','west','within','around','beside','between'}
ACTION_MARKERS = {'contains','includes','uses','accepts','works','stars','released','split','put','provides','features','covers','describes','represents','forms','consists','lives','live','located','found','based','became','received','distributed','named','extends','allows','replaced','exported','discontinued','made','produced','directed','written','sung','announced'}
CLAUSE_BREAK = {'which','who','where','when','while','because','although','that','and','or','but'}


def split_sents(line: str) -> list[str]:
    return [s.strip() for s in SENT_RE.split(line.strip()) if s.strip()]

def word_items(text: str):
    return [(m.group(0), m.start(), m.end()) for m in WORD_RE.finditer(text)]

def words(text: str) -> list[str]:
    return [w for w, _, _ in word_items(text)]

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
    return any(w in ANCHOR_VERBS for w in ws[:16])

def extend_phrase(items, start_i: int, min_words: int = 2, max_words: int = 9) -> int:
    """Return end index (inclusive) of a target phrase that does not dangle on a preposition/complementizer.
    Stops at clause breaks (after >=min content), but if the current last word is a dangling connector,
    it keeps extending to include the complement head, up to max_words."""
    n = len(items)
    end_i = start_i
    content = 0
    for j in range(start_i, min(n, start_i + max_words)):
        x = norm(items[j][0])
        if j > start_i and x in CLAUSE_BREAK and content >= min_words:
            break
        end_i = j
        if x not in STOP and len(x) >= 3:
            content += 1
    # Repair dangling end: extend while the last token is a connector, adding its object head.
    guard = 0
    while end_i + 1 < n and norm(items[end_i][0]) in DANGLING_END and guard < 6:
        end_i += 1
        # after adding one object word, allow a short determiner+noun completion
        if norm(items[end_i][0]) in {'the', 'a', 'an'} and end_i + 1 < n:
            end_i += 1
        guard += 1
    return end_i

def valid_target_text(txt: str) -> bool:
    ws = [norm(w) for w in words(txt)]
    if not ws or len(ws) > 10:
        return False
    if ws[0] in BAD_START:
        return False
    if ws[-1] in DANGLING_END:
        return False
    content = [w for w in ws if w not in STOP and len(w) >= 4]
    return len(content) >= 1

def choose_target(s2: str):
    items = word_items(s2)
    if len(items) < 3:
        return None
    xs = [norm(w) for w, _, _ in items]
    if xs[0] not in DEP_STARTS:
        return None

    def build(start_i, ttype):
        end_i = extend_phrase(items, start_i)
        a = items[start_i][1]; b = items[end_i][2]
        txt = s2[a:b].strip()
        if valid_target_text(txt):
            return {'target_type': ttype, 'target_span_s2': [a, b], 'target_text': txt, 'dependent': items[0][0]}
        return None

    # Pattern 1: location/spatial complement.
    for i in range(1, len(xs)):
        if xs[i] in LOCATION_MARKERS:
            start_i = i
            if i >= 1 and xs[i-1] in {'lives', 'live', 'located', 'found', 'based', 'is', 'are', 'was', 'were'}:
                start_i = i - 1
            r = build(start_i, 'location_spatial_phrase')
            if r: return r
    # Pattern 2: action/object/result phrase.
    for i in range(1, len(xs)):
        if xs[i] in ACTION_MARKERS:
            r = build(i, 'action_object_result_phrase')
            if r: return r
    # Pattern 3: definition/property complement after copula.
    for i in range(1, min(len(xs), 5)):
        if xs[i] in {'is', 'are', 'was', 'were'}:
            start_i = i + 1
            while start_i < len(items) and norm(items[start_i][0]) in {'a', 'an', 'the'}:
                start_i += 1
            if start_i < len(items):
                r = build(start_i, 'definition_property_complement')
                if r: return r
    # Pattern 4: fallback first content phrase.
    for i in range(1, min(len(xs), 8)):
        if xs[i] not in STOP and len(xs[i]) >= 4:
            r = build(i, 'semantic_content_continuation')
            if r: return r
    return None

def collect(raw_dir, source, mn1, mx1, mn2, mx2):
    p = raw_dir/source
    rows = []
    with p.open('r', encoding='utf-8', errors='replace') as f:
        for line_no, line in enumerate(f, 1):
            sents = split_sents(line)
            for i in range(len(sents) - 1):
                s1 = sents[i]; s2 = sents[i + 1]
                w1 = len(words(s1)); w2 = len(words(s2))
                if not (mn1 <= w1 <= mx1 and mn2 <= w2 <= mx2):
                    continue
                if not has_s1_anchor(s1):
                    continue
                t = choose_target(s2)
                if not t:
                    continue
                rows.append({'source': source, 'line_no': line_no, 'sent_index': i, 's1': s1, 's2': s2, 's1_word_count': w1, 's2_word_count': w2, **t})
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw_dir', default=str(RAW_DIR_DEFAULT))
    ap.add_argument('--out_dir', default=str(OUT_DIR_DEFAULT))
    ap.add_argument('--source', default='simple_wiki.train.txt')
    ap.add_argument('--target_counted_words', type=int, default=200000)
    ap.add_argument('--seed', type=int, default=116)
    ap.add_argument('--min_s1_words', type=int, default=4)
    ap.add_argument('--max_s1_words', type=int, default=70)
    ap.add_argument('--min_s2_words', type=int, default=4)
    ap.add_argument('--max_s2_words', type=int, default=70)
    args = ap.parse_args()
    raw_dir = pathlib.Path(args.raw_dir); out_dir = pathlib.Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    cand = collect(raw_dir, args.source, args.min_s1_words, args.max_s1_words, args.min_s2_words, args.max_s2_words)
    idx = defaultdict(list)
    for r in cand:
        idx[sent_len_bin(r['s1_word_count'])].append(r)
    rng.shuffle(cand)
    rows = []; counted = 0; by_type = Counter(); skipped = Counter()
    for r in cand:
        pool = [q for q in idx[sent_len_bin(r['s1_word_count'])] if q['line_no'] != r['line_no']]
        if not pool:
            skipped['no_wrong_s1'] += 1; continue
        wrong = min(rng.sample(pool, min(20, len(pool))), key=lambda q: abs(q['s1_word_count'] - r['s1_word_count']))
        text_true = r['s1'] + ' ' + r['s2']
        text_wrong = wrong['s1'] + ' ' + r['s2']
        text_no = r['s2']
        wt = count_ws(text_true); ww = count_ws(text_wrong); wn = count_ws(text_no)
        if rows and counted + wt > args.target_counted_words:
            break
        a, b = r['target_span_s2']
        row = {'example_id': len(rows), 'split': 'heldout' if len(rows) % 20 == 0 else 'train',
               'source': r['source'], 'line_no': r['line_no'], 'sent_index': r['sent_index'],
               's1': r['s1'], 's2': r['s2'], 'text': text_true, 'text_wrong_s1': text_wrong, 'text_no_s1': text_no,
               'words': wt, 'words_wrong_s1': ww, 'words_no_s1': wn,
               's2_start_char_text': len(r['s1']) + 1, 's2_start_char_wrong_s1': len(wrong['s1']) + 1, 's2_start_char_no_s1': 0,
               'target_span_s2': [a, b], 'target_text': r['target_text'], 'target_type': r['target_type'], 'dependent': r['dependent'],
               'target_span_text': [len(r['s1']) + 1 + a, len(r['s1']) + 1 + b],
               'target_span_wrong_s1': [len(wrong['s1']) + 1 + a, len(wrong['s1']) + 1 + b],
               'target_span_no_s1': [a, b],
               'wrong_s1': {'source': wrong['source'], 'line_no': wrong['line_no'], 'sent_index': wrong['sent_index'], 's1': wrong['s1'], 's1_word_count': wrong['s1_word_count']},
               'materializer_version': 'relational_xspan_v2', 'selection_policy': 'rule_based_only_no_model_score_filtering'}
        rows.append(row); counted += wt; by_type[r['target_type']] += 1
    out_path = out_dir/f'relational_xspan_v2_seed{args.seed}_target{args.target_counted_words}_actual.jsonl'
    with out_path.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    summary = {'status': 'RELATIONAL_XSPAN_V2_MATERIALIZED', 'out_jsonl': str(out_path), 'raw_dir': str(raw_dir), 'source': args.source,
               'seed': args.seed, 'candidate_rows_before_budget': len(cand), 'num_rows': len(rows), 'true_context_counted_words': counted,
               'target_counted_words': args.target_counted_words, 'by_target_type': dict(by_type),
               'heldout_rows': sum(r['split'] == 'heldout' for r in rows), 'train_rows': sum(r['split'] == 'train' for r in rows), 'skipped': dict(skipped),
               'policy': 'No protected-model scores used. Phrase completion extends dangling prepositions to their complement; discourse-word starts rejected.'}
    summary_path = out_path.with_suffix('.summary.json')
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    lines = ['# research — relational XSpan v2 materialization', '', f'JSONL: `{out_path}`', f'Summary: `{summary_path}`', '',
             f'Rows: **{len(rows)}**', f'True-context counted words: **{counted}**', f'By target type: {dict(by_type)}', '',
             'v2 repairs: extend dangling prepositions to complement, reject discourse-word starts, add passive/event action markers.']
    NOTE_DEFAULT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'status': summary['status'], 'jsonl': str(out_path), 'rows': len(rows), 'counted_words': counted, 'by_target_type': dict(by_type)}, indent=2))

if __name__ == '__main__':
    main()
