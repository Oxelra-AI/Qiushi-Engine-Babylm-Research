#!/usr/bin/env python3
"""Inspect token offset assignment for the research faithful chunking prototype."""
from __future__ import annotations
import bisect
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, 'experiments/archive/compact_experience/scripts')
import masking_curriculum_trainer as base  # noqa: E402

TRAIN_10M = Path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl')
TOKENIZER = 'experiments/archive/representation_and_objectives/data/legal_representation_route_map/tokenizers/legal_byte_bpe_40k'
WORD_RE = re.compile(r'\S+')

def find(starts, spans, s, e):
    if e <= s or not spans:
        return None
    idx = bisect.bisect_right(starts, s) - 1
    for j in (idx, idx + 1):
        if 0 <= j < len(spans):
            a, b = spans[j]
            if s < b and e > a:
                return j
    mid = (s + e - 1) // 2
    idx = bisect.bisect_right(starts, mid) - 1
    if 0 <= idx < len(spans):
        a, b = spans[idx]
        if s < b and e > a:
            return idx
    return None

def main():
    tok = base.make_portable_tokenizer(TOKENIZER)
    shown = 0
    total = 0
    row_count = 0
    category_counts = {}
    examples = []
    for line_no, line in enumerate(TRAIN_10M.open(encoding='utf-8'), 1):
        if not line.strip():
            continue
        obj = json.loads(line)
        text = str(obj['text'])
        spans3 = [(m.start(), m.end(), m.group(0)) for m in WORD_RE.finditer(text)]
        spans = [(a, b) for a, b, _ in spans3]
        starts = [a for a, _ in spans]
        enc = tok(text, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
        for i, (tid, off) in enumerate(zip(enc['input_ids'], enc['offset_mapping'])):
            s, e = int(off[0]), int(off[1])
            idx = find(starts, spans, s, e)
            if idx is not None:
                continue
            total += 1
            piece = text[max(0, s):min(len(text), e)] if 0 <= s <= e <= len(text) else ''
            tokstr = str(tok.convert_ids_to_tokens(int(tid)))
            if s == e:
                cat = 'zero_length'
            elif piece and piece.strip() == '':
                cat = 'whitespace_only_offset'
            elif tokstr.startswith('Ġ') or tokstr.startswith('▁'):
                cat = 'leading_space_marker_no_overlap'
            else:
                cat = 'other'
            category_counts[cat] = category_counts.get(cat, 0) + 1
            if shown < 40:
                near = []
                for j, (a, b, w) in enumerate(spans3):
                    if abs(a - s) <= 25 or abs(b - e) <= 25 or (a <= s <= b) or (a <= e <= b):
                        near.append({'j': j, 'span': [a, b], 'word': w})
                examples.append({
                    'line': line_no,
                    'token_index': i,
                    'token': tokstr,
                    'offset': [s, e],
                    'offset_text': piece,
                    'context': text[max(0, s-35):min(len(text), e+35)],
                    'category': cat,
                    'near_words': near[:8],
                })
                shown += 1
        row_count += 1
        if shown >= 40 and row_count >= 1000:
            break
    out = {
        'rows_scanned': row_count,
        'unassigned_seen_in_scanned_rows': total,
        'category_counts': category_counts,
        'examples': examples,
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
