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
NOTE_DEFAULT = (ROOT.parents[2] / 'research/notes/initial_model_studies/downstream_counterfactual_materialization_v2.md')
SENT_RE = re.compile(r'(?<=[.!?])\s+')
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-'][A-Za-z0-9]+)?")
FILES = ['gutenberg.train.txt','simple_wiki.train.txt']
STOP = {
    'the','a','an','and','or','but','of','for','to','in','on','at','by','with','from','as','is','are','was','were','be','been','being',
    'do','does','did','have','has','had','will','would','can','could','should','may','might','must','not','no','yes','this','that','these','those',
    'there','here','it','its','they','them','their','he','she','his','her','we','us','you','i','me','my','your','our','what','which','who','whom',
    'whose','when','where','why','how','than','then','so','if','because','about','into','up','down','out','over','under','again','very','just',
    'also','said','say','says','one','two','first','new','old','many','much','more','most','other','some','such','same','only','every','any','all'
}
DET = {'the','this','that','these','those','his','her','their','our','my','your','its'}
ART = {'a','an'}
PREP = {'of','in','on','at','from','to','with','by','for','near','under','over','through','between','among','inside','outside','before','after'}
PRONOUN_MARKERS = {'it','its','he','him','his','she','her','hers','they','them','their','theirs','this','that','these','those','there','then','such','same','another','other'}
BAD_SUFFIX = ('ing','ed','ly')

def split_sents(line: str) -> list[str]:
    return [s.strip() for s in SENT_RE.split(line.strip()) if s.strip()]

def word_matches(text: str):
    return list(WORD_RE.finditer(text))

def words(text: str) -> list[str]:
    return [m.group(0) for m in word_matches(text)]

def count_ws(text: str) -> int:
    return len(text.split())

def norm(w: str) -> str:
    return w.lower().strip("-'")

def is_vowel_start(w: str) -> bool:
    return bool(w) and w[0].lower() in 'aeiou'

def freq_bin(c: int) -> str:
    if c < 5: return 'lt5'
    if c < 20: return '5_19'
    if c < 100: return '20_99'
    if c < 500: return '100_499'
    return '500p'

def len_bin_n(n: int) -> str:
    if n <= 4: return 'l4'
    if n <= 7: return 'l5_7'
    if n <= 11: return 'l8_11'
    return 'l12p'

def sent_len_bin(n: int) -> str:
    if n <= 10: return 's_le10'
    if n <= 20: return 's_11_20'
    if n <= 40: return 's_21_40'
    return 's_41p'

def is_content_surface(w: str) -> bool:
    x = norm(w)
    return len(x) >= 4 and x not in STOP and bool(re.search('[a-z]', x)) and not x.isdigit()

def plural_flag(w: str) -> str:
    x = norm(w)
    return 'plural_s' if x.endswith('s') and not x.endswith('ss') else 'singular_or_mass'

def surface_shape(w: str, position: int) -> str:
    # proper_mid excludes first sentence word, because initial capitalization alone is not a proper-name signal.
    if position > 0 and w[:1].isupper() and not w.isupper() and is_content_surface(w):
        return 'proper_mid'
    return 'common'

def token_context(tokens: list[str], idx: int) -> dict:
    prev = norm(tokens[idx-1]) if idx > 0 else '<bos>'
    nxt = norm(tokens[idx+1]) if idx+1 < len(tokens) else '<eos>'
    return {'prev': prev, 'next': nxt}

def edit_category(tokens: list[str], idx: int) -> dict | None:
    w = tokens[idx]
    x = norm(w)
    if not is_content_surface(w):
        return None
    if x.endswith(BAD_SUFFIX):
        return None
    ctx = token_context(tokens, idx)
    shape = surface_shape(w, idx)
    if shape == 'proper_mid':
        return {'kind': 'proper_mid', 'prev_class': 'any', 'article_vowel': 'na', 'plural': plural_flag(w), 'suffix3': x[-3:]}
    prev = ctx['prev']
    # Restrict common edits to noun-like positions: determiner/article/preposition before the word, not verb/modal frames.
    if prev in ART:
        return {'kind': 'common_after_article', 'prev_class': prev, 'article_vowel': 'vowel' if is_vowel_start(x) else 'consonant', 'plural': plural_flag(w), 'suffix3': x[-3:]}
    if prev in DET:
        return {'kind': 'common_after_determiner', 'prev_class': prev, 'article_vowel': 'na', 'plural': plural_flag(w), 'suffix3': x[-3:]}
    if prev in PREP:
        return {'kind': 'common_after_prep', 'prev_class': prev, 'article_vowel': 'na', 'plural': plural_flag(w), 'suffix3': x[-3:]}
    return None

def apply_case(rep: str, orig: str) -> str:
    if orig.isupper(): return rep.upper()
    if orig[:1].isupper(): return rep.capitalize()
    return rep.lower()

def has_downstream_marker(s2: str) -> bool:
    ws = {norm(w) for w in words(s2)}
    return bool(ws & PRONOUN_MARKERS)

def collect_pairs_and_replacements(raw_dir: pathlib.Path, sources: list[str], min_words: int, max_words: int):
    pairs = []
    freq = Counter()
    raw_candidates = []
    for fn in sources:
        p = raw_dir/fn
        with p.open('r', encoding='utf-8', errors='replace') as f:
            for line_no, line in enumerate(f, 1):
                sents = []
                for sent in split_sents(line):
                    ws = words(sent)
                    if min_words <= len(ws) <= max_words:
                        sents.append((sent, ws))
                        for w in ws:
                            if is_content_surface(w):
                                freq[norm(w)] += 1
                for sent, ws in sents:
                    toks = ws
                    for i,w in enumerate(toks):
                        cat = edit_category(toks, i)
                        if cat:
                            raw_candidates.append({'word': norm(w), 'surface': w, 'cat': cat})
                for i in range(len(sents)-1):
                    s1, w1 = sents[i]; s2, w2 = sents[i+1]
                    if not has_downstream_marker(s2):
                        continue
                    pairs.append({'source': fn, 'line_no': line_no, 'sent_index': i, 's1': s1, 's2': s2, 's1_word_count': len(w1), 's2_word_count': len(w2)})
    repl = defaultdict(list)
    for c in raw_candidates:
        w = c['word']; f = freq.get(w,0)
        if f < 5:
            continue
        cat = c['cat']
        # coarse bucket keeps syntax shape but permits enough alternatives.
        key = (cat['kind'], cat['prev_class'], cat['article_vowel'], cat['plural'], len_bin_n(len(w)), freq_bin(f))
        repl[key].append(w)
        # relaxed prev-class fallback for sparse buckets.
        key2 = (cat['kind'], 'ANY_PREV', cat['article_vowel'], cat['plural'], len_bin_n(len(w)), freq_bin(f))
        repl[key2].append(w)
    repl = {k: sorted(set(v)) for k,v in repl.items() if len(set(v)) >= 2}
    return pairs, freq, repl

def choose_edit(s1: str, s2: str, freq: Counter, repl: dict, rng: random.Random):
    toks = words(s1)
    s2_norm = {norm(w) for w in words(s2)}
    candidates = []
    # Map token index to char span by WORD_RE sequence order.
    matches = word_matches(s1)
    for i,m in enumerate(matches):
        w = m.group(0); cat = edit_category(toks, i)
        if not cat:
            continue
        x = norm(w); f = freq.get(x,0)
        key = (cat['kind'], cat['prev_class'], cat['article_vowel'], cat['plural'], len_bin_n(len(x)), freq_bin(f))
        key2 = (cat['kind'], 'ANY_PREV', cat['article_vowel'], cat['plural'], len_bin_n(len(x)), freq_bin(f))
        candidates.append((i,m,w,cat,key,key2))
    rng.shuffle(candidates)
    for i,m,w,cat,key,key2 in candidates:
        x = norm(w)
        pool = [z for z in repl.get(key, []) if z != x and z not in s2_norm]
        if not pool:
            pool = [z for z in repl.get(key2, []) if z != x and z not in s2_norm]
        if not pool:
            continue
        rep_norm = rng.choice(pool)
        rep = apply_case(rep_norm, w)
        s1p = s1[:m.start()] + rep + s1[m.end():]
        edit = {'original': w, 'replacement': rep, 'original_lower': x, 'replacement_lower': rep_norm,
                'span': [m.start(), m.end()], 'token_index': i, 'category': cat, 'freq_original': freq.get(x,0),
                'freq_replacement': freq.get(rep_norm,0), 'length_bin': len_bin_n(len(x)), 'freq_bin': freq_bin(freq.get(x,0)),
                'same_plural_flag': plural_flag(w) == plural_flag(rep_norm), 'same_vowel_after_article': cat.get('article_vowel','na')}
        return s1p, edit
    return None, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw_dir', default=str(RAW_DIR_DEFAULT))
    ap.add_argument('--out_dir', default=str(OUT_DIR_DEFAULT))
    ap.add_argument('--target_counted_words', type=int, default=200000)
    ap.add_argument('--seed', type=int, default=109)
    ap.add_argument('--sources', nargs='+', default=FILES)
    ap.add_argument('--min_sent_words', type=int, default=6)
    ap.add_argument('--max_sent_words', type=int, default=70)
    args = ap.parse_args()
    raw_dir = pathlib.Path(args.raw_dir); out_dir = pathlib.Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    pairs, freq, repl = collect_pairs_and_replacements(raw_dir, args.sources, args.min_sent_words, args.max_sent_words)
    # random downstream controls: same source + sentence length bin, not same line.
    s2_index = defaultdict(list)
    for p in pairs:
        s2_index[(p['source'], sent_len_bin(p['s2_word_count']))].append(p)
    rng.shuffle(pairs)
    rows=[]; counted=0; skipped=Counter(); by_source=Counter(); by_edit_kind=Counter()
    for p in pairs:
        if counted >= args.target_counted_words:
            break
        s1p, edit = choose_edit(p['s1'], p['s2'], freq, repl, rng)
        if not edit:
            skipped['no_good_edit'] += 1; continue
        key = (p['source'], sent_len_bin(p['s2_word_count']))
        pool = [q for q in s2_index.get(key, []) if q['line_no'] != p['line_no']]
        if not pool:
            skipped['no_random_s2_same_source_len'] += 1; continue
        q = min(rng.sample(pool, min(len(pool), 20)), key=lambda z: abs(z['s2_word_count'] - p['s2_word_count']))
        text = p['s1'] + ' ' + p['s2']
        textp = s1p + ' ' + p['s2']
        textr = s1p + ' ' + q['s2']
        row_words = {'words': count_ws(text), 'words_perturbed': count_ws(textp), 'words_random_control': count_ws(textr)}
        aux_words = sum(row_words.values())
        if rows and counted + aux_words > args.target_counted_words:
            break
        row = {'example_id': len(rows), 'split': 'heldout' if len(rows)%20==0 else 'train',
               'source': p['source'], 'line_no': p['line_no'], 'sent_index': p['sent_index'],
               's1': p['s1'], 's2': p['s2'], 's1_perturbed': s1p,
               'text': text, 'text_perturbed': textp, 'text_random_control': textr,
               **row_words, 'aux_counted_words_original_perturbed_random': aux_words,
               's2_start_char_text': len(p['s1']) + 1, 's2_start_char_perturbed': len(s1p) + 1,
               's2_start_char_random_control': len(s1p) + 1,
               's1_word_count': p['s1_word_count'], 's2_word_count': p['s2_word_count'],
               'edit': edit,
               'random_s2': {'source': q['source'], 'line_no': q['line_no'], 'sent_index': q['sent_index'], 's2': q['s2'],
                             's2_word_count': q['s2_word_count'], 'length_abs_delta': abs(q['s2_word_count'] - p['s2_word_count'])},
               'auxiliary_labels': {'original_pair': 0, 'perturbed_s1_true_s2': 1, 'perturbed_s1_random_s2_control': 1},
               'materializer_version': 'v2_nounish_context_matched'}
        rows.append(row); counted += aux_words; by_source[p['source']] += 1; by_edit_kind[edit['category']['kind']] += 1
    out_path = out_dir/f"downstream_counterfactual_v2_seed{args.seed}_target{args.target_counted_words}_actual.jsonl"
    with out_path.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False)+'\n')
    summary = {'status':'DOWNSTREAM_COUNTERFACTUAL_V2_MATERIALIZED', 'raw_dir':str(raw_dir), 'out_jsonl':str(out_path),
               'seed':args.seed, 'target_counted_words':args.target_counted_words, 'actual_aux_counted_words_original_perturbed_random':counted,
               'num_rows':len(rows), 'sources':args.sources, 'by_source':dict(by_source), 'by_edit_kind':dict(by_edit_kind),
               'candidate_pairs_after_s2_marker_filter':len(pairs), 'replacement_bucket_count':len(repl), 'skipped':dict(skipped),
               'counting_note':'If all original/perturbed/random streams are processed, count actual_aux_counted_words_original_perturbed_random toward exposure. Ordinary WWM remains separate and unchanged.',
               'quality_intent':'v2 restricts edits to proper-noun or noun-like determiner/article/preposition contexts, preserves article-vowel/plural/capitalization coarse buckets, requires downstream pronoun/deictic marker, and source+length-matches random downstream controls.'}
    summary_path = out_path.with_suffix('.summary.json')
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    note = NOTE_DEFAULT
    lines=['# research — downstream counterfactual state-propagation v2 materialization','',f'JSONL: `{out_path}`',f'Summary: `{summary_path}`','',f'Rows: **{len(rows)}**',f'Counted aux words (original+perturbed+random): **{counted}**',f'By source: {dict(by_source)}',f'By edit kind: {dict(by_edit_kind)}','',summary['quality_intent'],'','This replaces the research v1 materialization for training, because v1 had obvious syntactic-artifact edits.']
    note.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'status':summary['status'], 'jsonl':str(out_path), 'summary':str(summary_path), 'rows':len(rows), 'counted_words':counted, 'by_edit_kind':dict(by_edit_kind)}, indent=2))

if __name__=='__main__':
    main()
