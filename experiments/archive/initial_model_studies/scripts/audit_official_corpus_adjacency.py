#!/usr/bin/env python3
from __future__ import annotations

import json, pathlib, re, statistics
from collections import Counter

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
RAW_DIRS = [
    ROOT/'data/reconstruct_tmp/raw_dataset',
    ROOT/'data/mixture_revision_61/official_raw',
    ROOT/'data/state_revision_68/official_raw',
]
FILES = ['bnc_spoken.train.txt','childes.train.txt','gutenberg.train.txt','open_subtitles.train.txt','simple_wiki.train.txt','switchboard.train.txt']
OUT = ROOT/'data/official_corpus_adjacency_audit.json'
NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/official_corpus_adjacency_audit.md')
SENT_RE = re.compile(r'(?<=[.!?])\s+')

def words(s: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", s)

def sent_split(line: str) -> list[str]:
    parts = [p.strip() for p in SENT_RE.split(line.strip()) if p.strip()]
    return [p for p in parts if len(words(p)) >= 3]

def main():
    raw_dir = next((d for d in RAW_DIRS if all((d/f).exists() for f in FILES)), None)
    if raw_dir is None:
        raise FileNotFoundError('no complete official raw dir found')
    summary = {'raw_dir': str(raw_dir), 'files': {}, 'total_words': 0, 'total_lines': 0, 'total_sentences': 0, 'total_adjacent_pairs': 0}
    examples = []
    for fn in FILES:
        p = raw_dir/fn
        line_count=0; word_count=0; sent_count=0; pair_count=0
        line_word_lens=[]; sent_word_lens=[]; sents_per_line=[]
        with p.open('r', encoding='utf-8', errors='replace') as f:
            for line_no, line in enumerate(f, 1):
                t=line.strip()
                if not t: continue
                ws=words(t); wc=len(ws)
                if wc==0: continue
                ss=sent_split(t)
                line_count += 1; word_count += wc; sent_count += len(ss); pair_count += max(0, len(ss)-1)
                line_word_lens.append(wc); sents_per_line.append(len(ss))
                for s in ss: sent_word_lens.append(len(words(s)))
                if len(ss)>=2 and len(examples)<20:
                    examples.append({'file': fn, 'line_no': line_no, 's1': ss[0][:160], 's2': ss[1][:160], 's1_words': len(words(ss[0])), 's2_words': len(words(ss[1]))})
        def q(xs):
            if not xs: return {}
            return {'mean': statistics.mean(xs), 'median': statistics.median(xs), 'min': min(xs), 'max': max(xs), 'n': len(xs)}
        summary['files'][fn] = {'lines': line_count, 'words_regex': word_count, 'sentences': sent_count, 'adjacent_pairs_within_line': pair_count, 'line_word_lens': q(line_word_lens), 'sent_word_lens': q(sent_word_lens), 'sents_per_line': q(sents_per_line)}
        summary['total_words'] += word_count; summary['total_lines'] += line_count; summary['total_sentences'] += sent_count; summary['total_adjacent_pairs'] += pair_count
    summary['examples_first20'] = examples
    summary['interpretation'] = 'Line breaks in recovered official raw files appear to be the only durable document/utterance boundary available locally. Adjacent sentence pairs within a line are plausible for same-document continuity; cross-line adjacency should be treated source-specifically and audited before use.'
    OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    lines=['# research — official corpus adjacency audit','',f'Evidence JSON: `{OUT}`','',f'Raw dir: `{raw_dir}`','', '| file | lines | sentences | within-line adjacent sentence pairs | mean sentences/line | mean sentence words |','|---|---:|---:|---:|---:|---:|']
    for fn,d in summary['files'].items():
        lines.append(f"| {fn} | {d['lines']} | {d['sentences']} | {d['adjacent_pairs_within_line']} | {d['sents_per_line'].get('mean',0):.2f} | {d['sent_word_lens'].get('mean',0):.2f} |")
    lines += ['', f"Total within-line adjacent sentence pairs: **{summary['total_adjacent_pairs']}**", '', summary['interpretation'], '', 'First 20 example pairs are in the JSON only to avoid clutter.']
    NOTE.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'status':'CORPUS_ADJACENCY_AUDIT_DONE','out':str(OUT),'pairs':summary['total_adjacent_pairs'],'raw_dir':str(raw_dir)}, indent=2))

if __name__=='__main__': main()
