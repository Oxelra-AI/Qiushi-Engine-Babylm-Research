#!/usr/bin/env python3
"""research static profile of existing research extract controls under legal16k tokenizer.

This is NOT the final marginal-density protocol. It audits whether the existing
contiguous semantic/random extract scaffold is suitable or what it confounds.
"""
import json
import pathlib
import collections
import re
import statistics
from transformers import AutoTokenizer

BASE = pathlib.Path('experiments/archive/representation_and_objectives/data/prediction_geometry_scaffold_v2')
TOK_PATH = 'experiments/archive/representation_and_objectives/training/runs/gc_compact_repeat_reinvest_16k_seed43022_r2/hf_model/chck_1M'
OUT = pathlib.Path('experiments/archive/representation_and_objectives/data/existing_extract_profile')
WORD_RE = re.compile(r"[A-Za-z0-9']+")


def words(s: str):
    return WORD_RE.findall(s.lower())


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(int(len(xs) * p), len(xs) - 1)]


def summ(xs):
    return {
        'n': len(xs),
        'mean': round(statistics.mean(xs), 6),
        'median': round(statistics.median(xs), 6),
        'p10': round(pct(xs, 0.1), 6),
        'p90': round(pct(xs, 0.9), 6),
        'min': round(min(xs), 6),
        'max': round(max(xs), 6),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(TOK_PATH, use_fast=True)
    arms = ['compact_oneway', 'semantic_extract_oneway', 'random_extract_oneway']
    arm_data = {}
    for arm in arms:
        path = BASE / f'{arm}.jsonl'
        rows = [json.loads(line) for line in path.open(encoding='utf-8')]
        arm_data[arm] = rows
    source_counter = collections.Counter()
    for row in arm_data['compact_oneway']:
        source_counter.update(tok(row['source_text'], add_special_tokens=False)['input_ids'])
    compact_by_id = {r['pair_id']: r['side_text'] for r in arm_data['compact_oneway']}
    profile = {
        'status': 'EXISTING_EXTRACT_PROFILE',
        'meaning': 'Existing research extract arms are contiguous source spans. This profile audits their legal16k tokenizer/support geometry; it is not a final experiment design.',
        'tokenizer': TOK_PATH,
        'arms': {},
    }
    for arm, rows in arm_data.items():
        side_counter = collections.Counter()
        side_word_types = collections.Counter()
        overlap_compact = []
        overlap_source = []
        tok_per_word = []
        rare_frac = []
        source_support_mean = []
        side_word_lens = []
        side_bpe_lens = []
        for row in rows:
            w = words(row['side_text'])
            sw = set(words(row['source_text']))
            cw = set(words(compact_by_id[row['pair_id']]))
            if w:
                overlap_source.append(sum(x in sw for x in w) / len(w))
                overlap_compact.append(sum(x in cw for x in w) / len(w))
            ids = tok(row['side_text'], add_special_tokens=False)['input_ids']
            side_counter.update(ids)
            side_word_types.update(w)
            side_word_lens.append(len(w))
            side_bpe_lens.append(len(ids))
            tok_per_word.append(len(ids) / max(1, len(w)))
            if ids:
                supports = [source_counter[i] for i in ids]
                source_support_mean.append(sum(supports) / len(supports))
                rare_frac.append(sum(s <= 5 for s in supports) / len(supports))
        profile['arms'][arm] = {
            'rows': len(rows),
            'side_words': sum(side_word_lens),
            'side_bpe_tokens': sum(side_bpe_lens),
            'unique_word_types': len(side_word_types),
            'unique_bpe_types': len(side_counter),
            'overlap_source_word': summ(overlap_source),
            'overlap_compact_word': summ(overlap_compact),
            'side_words_per_row': summ(side_word_lens),
            'bpe_tokens_per_word': summ(tok_per_word),
            'side_bpe_tokens_per_row': summ(side_bpe_lens),
            'mean_source_support_per_side_bpe': summ(source_support_mean),
            'frac_side_bpe_source_support_le5': summ(rare_frac),
        }
    out_path = OUT / 'existing_extract_profile.json'
    out_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(profile, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
