#!/usr/bin/env python3
"""Equalize research token-aware rows to the same row count across arms.

Input rows already have baseline16k untruncated token length <=256 and exact 1M
words. Splitting rows preserves word stream/content and cannot create longer token
sequences. We target the maximum row count (true_pair_adjacent: 5988) so all arms
have identical DataLoader length / optimizer update count at b128 (47 steps).
"""
from __future__ import annotations
import json, pathlib

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
IN = ROOT/'data/same_entity_deranged_1M_tokenaware'
OUT = ROOT/'data/same_entity_deranged_1M_tokenaware_equalrows'
ARMS = ['true_pair_adjacent','hard_negative_same_entity','orig_only','shuffled_pair_adjacent']
TARGET_ROWS = 5988
BATCH = 128

def load_rows(arm):
    p = IN/f'{arm}_1000000w_tokenaware.jsonl'
    return [json.loads(line) for line in p.open(encoding='utf-8')]

def split_one(row):
    words = row['text'].split()
    if len(words) < 2:
        return None
    mid = len(words)//2
    a = {'source': row['source'], 'text': ' '.join(words[:mid]), 'words': mid}
    b = {'source': row['source'], 'text': ' '.join(words[mid:]), 'words': len(words)-mid}
    return a,b

def equalize(rows):
    rows = [{'source': r['source'], 'text': r['text'], 'words': len(r['text'].split())} for r in rows]
    while len(rows) < TARGET_ROWS:
        # split the longest row with >=2 words
        idx = max(range(len(rows)), key=lambda i: rows[i]['words'])
        sp = split_one(rows[idx])
        if sp is None:
            raise RuntimeError('cannot split further')
        rows[idx:idx+1] = list(sp)
    if len(rows) != TARGET_ROWS:
        raise RuntimeError(f'row count mismatch {len(rows)}')
    for i,r in enumerate(rows):
        r['example_id'] = i
    return rows

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stats={}
    for arm in ARMS:
        rows = equalize(load_rows(arm))
        outp = OUT/f'{arm}_1000000w_tokenaware_equalrows.jsonl'
        with outp.open('w', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False)+'\n')
        wcs=[r['words'] for r in rows]
        stats[arm] = {
            'path': str(outp), 'exact_words': sum(wcs), 'rows': len(rows),
            'steps_at_b128': (len(rows)+BATCH-1)//BATCH,
            'min_words_per_row': min(wcs), 'max_words_per_row': max(wcs),
            'mean_words_per_row': sum(wcs)/len(wcs)
        }
    meta={'status':'TOKENAWARE_EQUALROWS_READY','source_dir':str(IN),
          'target_rows_per_arm':TARGET_ROWS,'steps_at_b128':(TARGET_ROWS+BATCH-1)//BATCH,
          'max_untruncated_tokens_preserved':'<=256 inherited from token-aware input; splitting cannot increase token length',
          'arms':stats,
          'purpose':'No-truncation recheck of paired-restatement route with identical optimizer update count across arms.'}
    (OUT/'materialization_meta.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps({'out_dir':str(OUT),'meta':str(OUT/'materialization_meta.json'),'arms':stats}, indent=2))
if __name__ == '__main__': main()
