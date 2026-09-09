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
OUT_DIR_DEFAULT = ROOT/'data/counterfactual_revision_108'
NOTE_DEFAULT = (ROOT.parents[2] / 'research/notes/initial_model_studies/downstream_counterfactual_materialization.md')
SENT_RE = re.compile(r'(?<=[.!?])\s+')
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-'][A-Za-z0-9]+)?")
STOP = {
    'the','a','an','and','or','but','of','for','to','in','on','at','by','with','from','as','is','are','was','were','be','been','being',
    'do','does','did','have','has','had','will','would','can','could','should','may','might','must','not','no','yes','this','that',
    'these','those','there','here','it','its','they','them','their','he','she','his','her','we','us','you','i','me','my','your','our',
    'what','which','who','whom','whose','when','where','why','how','than','then','so','if','because','about','into','up','down',
    'out','over','under','again','very','just','also','said','say','says','one','two','first','new','old'
}
FILES = ['gutenberg.train.txt','simple_wiki.train.txt']

def split_sents(line: str) -> list[str]:
    return [s.strip() for s in SENT_RE.split(line.strip()) if s.strip()]

def wtokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", text)

def count_ws(text: str) -> int:
    return len(text.split())

def norm(w: str) -> str:
    return w.lower().strip("-'")

def is_content(w: str) -> bool:
    x = norm(w)
    return len(x) >= 4 and x not in STOP and bool(re.search('[a-z]', x)) and not x.isdigit()

def freq_bin(c: int) -> str:
    if c < 5: return 'lt5'
    if c < 20: return '5_19'
    if c < 100: return '20_99'
    if c < 500: return '100_499'
    return '500p'

def len_bin(w: str) -> str:
    n = len(w)
    if n <= 4: return 'l4'
    if n <= 7: return 'l5_7'
    if n <= 11: return 'l8_11'
    return 'l12p'

def apply_case(rep: str, orig: str) -> str:
    if orig.isupper(): return rep.upper()
    if orig[:1].isupper(): return rep.capitalize()
    return rep

def read_pairs(raw_dir: pathlib.Path, sources: list[str], min_words: int, max_words: int) -> tuple[list[dict], Counter]:
    pairs = []
    freq = Counter()
    for fn in sources:
        p = raw_dir/fn
        if not p.exists():
            raise FileNotFoundError(p)
        with p.open('r', encoding='utf-8', errors='replace') as f:
            for line_no, line in enumerate(f, 1):
                sents = split_sents(line)
                clean = []
                for sent in sents:
                    wc = len(wtokens(sent))
                    if min_words <= wc <= max_words:
                        clean.append((sent, wc))
                        for m in WORD_RE.finditer(sent):
                            x = norm(m.group(0))
                            if is_content(x):
                                freq[x] += 1
                for i in range(len(clean)-1):
                    s1,w1 = clean[i]; s2,w2 = clean[i+1]
                    pairs.append({'source': fn, 'line_no': line_no, 'sent_index': i, 's1': s1, 's2': s2, 's1_word_count': w1, 's2_word_count': w2})
    return pairs, freq

def build_replacement_index(freq: Counter, min_freq: int = 5) -> dict[tuple[str,str], list[str]]:
    idx = defaultdict(list)
    for w,c in freq.items():
        if c >= min_freq and is_content(w):
            idx[(len_bin(w), freq_bin(c))].append(w)
    for k in idx:
        idx[k] = sorted(set(idx[k]))
    return idx

def choose_edit(s1: str, s2: str, freq: Counter, repl_index: dict, rng: random.Random):
    matches = [m for m in WORD_RE.finditer(s1) if is_content(m.group(0))]
    rng.shuffle(matches)
    s2_words = {norm(w) for w in wtokens(s2)}
    for m in matches:
        orig = m.group(0); lo = norm(orig); c = freq.get(lo, 0)
        key = (len_bin(lo), freq_bin(c))
        pool = [x for x in repl_index.get(key, []) if x != lo and x not in s2_words]
        if not pool:
            # relax frequency bin but preserve length if needed
            pool = [x for (lb,fb), xs in repl_index.items() if lb == key[0] for x in xs if x != lo and x not in s2_words]
        if not pool:
            continue
        rep = apply_case(rng.choice(pool), orig)
        s1p = s1[:m.start()] + rep + s1[m.end():]
        return s1p, {'original': orig, 'replacement': rep, 'original_lower': lo, 'replacement_lower': norm(rep), 'span': [m.start(), m.end()], 'freq_original': c, 'freq_replacement': freq.get(norm(rep), 0), 'length_bin': key[0], 'freq_bin': key[1]}
    return None, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw_dir', default=str(RAW_DIR_DEFAULT))
    ap.add_argument('--out_dir', default=str(OUT_DIR_DEFAULT))
    ap.add_argument('--target_counted_words', type=int, default=200000)
    ap.add_argument('--seed', type=int, default=108)
    ap.add_argument('--min_sent_words', type=int, default=6)
    ap.add_argument('--max_sent_words', type=int, default=80)
    ap.add_argument('--sources', nargs='+', default=FILES)
    args = ap.parse_args()
    raw_dir = pathlib.Path(args.raw_dir); out_dir = pathlib.Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    pairs, freq = read_pairs(raw_dir, args.sources, args.min_sent_words, args.max_sent_words)
    repl_index = build_replacement_index(freq)
    # random s2 controls by source and length bin, from true downstream sentences
    s2_index = defaultdict(list)
    for p in pairs:
        s2_index[(p['source'], len_bin('x'*min(12, max(1, p['s2_word_count']))))].append(p)
    rng.shuffle(pairs)
    out_name = f"downstream_counterfactual_seed{args.seed}_target{args.target_counted_words}_actual.jsonl"
    out_path = out_dir/out_name
    rows = []
    skipped = Counter(); counted = 0; by_source = Counter()
    for p in pairs:
        if counted >= args.target_counted_words:
            break
        s1p, edit = choose_edit(p['s1'], p['s2'], freq, repl_index, rng)
        if not edit:
            skipped['no_edit'] += 1; continue
        key = (p['source'], len_bin('x'*min(12, max(1, p['s2_word_count']))))
        control_pool = [q for q in s2_index.get(key, []) if q['line_no'] != p['line_no']]
        if not control_pool:
            skipped['no_random_s2'] += 1; continue
        q = rng.choice(control_pool)
        text_original = p['s1'] + ' ' + p['s2']
        text_perturbed = s1p + ' ' + p['s2']
        text_random_control = s1p + ' ' + q['s2']
        aux_count = count_ws(text_original) + count_ws(text_perturbed) + count_ws(text_random_control)
        if counted + aux_count > args.target_counted_words and rows:
            break
        row = {
            'example_id': len(rows), 'split': 'heldout' if (len(rows) % 20 == 0) else 'train',
            'source': p['source'], 'line_no': p['line_no'], 'sent_index': p['sent_index'],
            's1': p['s1'], 's2': p['s2'], 's1_perturbed': s1p,
            'text': text_original, 'text_perturbed': text_perturbed, 'text_random_control': text_random_control,
            'words': count_ws(text_original), 'words_perturbed': count_ws(text_perturbed), 'words_random_control': count_ws(text_random_control),
            'aux_counted_words_original_perturbed_random': aux_count,
            'edit': edit,
            'random_s2': {'source': q['source'], 'line_no': q['line_no'], 'sent_index': q['sent_index'], 's2': q['s2']},
            'auxiliary_labels': {'original_pair': 0, 'perturbed_s1_true_s2': 1, 'perturbed_s1_random_s2_control': 1},
            'training_hypothesis': 'Downstream head must classify perturbation from s2-token representations only; local-edit control should be easier and random-nonadjacent control tests topic/artifact reliance.'
        }
        rows.append(row); counted += aux_count; by_source[p['source']] += 1
    with out_path.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False)+'\n')
    summary = {
        'status': 'DOWNSTREAM_COUNTERFACTUAL_MATERIALIZED',
        'raw_dir': str(raw_dir), 'out_jsonl': str(out_path), 'seed': args.seed,
        'target_counted_words': args.target_counted_words, 'actual_aux_counted_words_original_perturbed_random': counted,
        'num_rows': len(rows), 'sources': args.sources, 'by_source': dict(by_source), 'skipped': dict(skipped),
        'candidate_pairs_scanned': len(pairs), 'vocab_content_types': len(freq), 'replacement_buckets': {str(k): len(v) for k,v in repl_index.items()},
        'counting_note': 'For an auxiliary that processes original, perturbed, and random-control streams, count aux_counted_words_original_perturbed_random toward exposure. Ordinary WWM stream must remain separately counted and unchanged.',
        'schema_note': 'Rows preserve source/line/sentence IDs and include text/text_perturbed/text_random_control. A future trainer should feed [s1;s2] and classify perturbation from s2 representations only, with local-edit and random-nonadjacent controls.'
    }
    summary_path = out_path.with_suffix('.summary.json')
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    note_path = NOTE_DEFAULT
    lines = ['# research — downstream counterfactual state-propagation materialization','',f'JSONL: `{out_path}`',f'Summary: `{summary_path}`','',f'Rows: **{len(rows)}**',f'Counted aux words if original+perturbed+random streams are processed: **{counted}**',f'Sources: {args.sources}',f'By source: {dict(by_source)}','', 'This materializes true within-line adjacent sentence pairs from official raw data. Each row contains original `[s1;s2]`, perturbed `[s1\';s2]`, and a random-nonadjacent downstream control `[s1\';s2_random]` with source/line IDs and matched rough replacement metadata. It is a data object for the next custom trainer, not a final result.']
    note_path.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'status': summary['status'], 'jsonl': str(out_path), 'summary': str(summary_path), 'rows': len(rows), 'counted_words': counted}, indent=2))

if __name__ == '__main__':
    main()
