#!/usr/bin/env python3
"""research token-aware no-truncation repack for research pair arms.

Reads the exact same word streams from research deranged 1M arms and writes new
JSONLs whose rows are greedily packed so baseline16k token length <=256. This
removes the research input truncation confound while preserving exact 1M word
exposure and arm content/order. Binary search reduces tokenizer calls.
"""
from __future__ import annotations
import json, os, pathlib, sys, time

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
IN = ROOT/'data/same_entity_deranged_1M'
OUT = ROOT/'data/same_entity_deranged_1M_tokenaware'
ARMS = ['true_pair_adjacent','hard_negative_same_entity','orig_only','shuffled_pair_adjacent']
MAX_TOK = 256
BATCH = 128


def setup_env():
    hf = ROOT/'training/hf_home'
    os.environ['HF_HOME'] = str(hf.resolve())
    os.environ['HF_HUB_CACHE'] = str((hf/'hub').resolve())
    os.environ['HF_DATASETS_CACHE'] = str((hf/'datasets').resolve())
    os.environ['TRANSFORMERS_CACHE'] = str((hf/'transformers').resolve())
    os.environ['HF_MODULES_CACHE'] = str((ROOT/'training/hf_modules_cache').resolve())
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')


def load_tokenizer():
    sys.path.insert(0, str((ROOT/'training/scripts').resolve()))
    from babylm_masked_train import make_portable_tokenizer
    return make_portable_tokenizer('')


def read_words(path: pathlib.Path) -> list[str]:
    words = []
    for line in path.open(encoding='utf-8'):
        obj = json.loads(line)
        words.extend(obj['text'].split())
    return words


def token_len(tok, words: list[str], start: int, end: int) -> int:
    return len(tok(' '.join(words[start:end]), add_special_tokens=False, truncation=False)['input_ids'])


def repack_arm(tok, arm: str) -> dict:
    in_path = IN/f'{arm}_1000000w.jsonl'
    out_path = OUT/f'{arm}_1000000w_tokenaware.jsonl'
    words = read_words(in_path)
    if len(words) != 1_000_000:
        raise RuntimeError(f'{arm}: expected 1M words, got {len(words)}')
    rows = []
    i = 0
    n = len(words)
    # Greedy: at each position choose largest row whose true tokenizer length <=256.
    while i < n:
        lo = i + 1
        hi = min(n, i + 300)
        # Expand high if still fits; rare but keeps exact greediness.
        while hi < n and token_len(tok, words, i, hi) <= MAX_TOK:
            nxt = min(n, i + 2*(hi-i))
            if nxt == hi:
                break
            hi = nxt
        if token_len(tok, words, i, lo) > MAX_TOK:
            # Extremely pathological single whitespace word; still write it and record.
            best = lo
        else:
            # If hi fits, best is hi; otherwise binary search [lo, hi)
            if token_len(tok, words, i, hi) <= MAX_TOK:
                best = hi
            else:
                low, high = lo, hi
                best = lo
                while low <= high:
                    mid = (low + high) // 2
                    tl = token_len(tok, words, i, mid)
                    if tl <= MAX_TOK:
                        best = mid
                        low = mid + 1
                    else:
                        high = mid - 1
        text = ' '.join(words[i:best])
        tl = len(tok(text, add_special_tokens=False, truncation=False)['input_ids'])
        rows.append({'example_id': len(rows), 'source': arm, 'text': text, 'words': best-i, 'untruncated_tokens': tl})
        i = best
        if len(rows) % 1000 == 0:
            print(f'{arm}: rows {len(rows)} words {i}', flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        for r in rows:
            # Trainer ignores untruncated_tokens; kept as metadata for inspection.
            f.write(json.dumps(r, ensure_ascii=False)+'\n')
    toks = [r['untruncated_tokens'] for r in rows]
    wcs = [r['words'] for r in rows]
    stats = {
        'path': str(out_path),
        'exact_words': sum(wcs),
        'rows': len(rows),
        'steps_at_b128': (len(rows)+BATCH-1)//BATCH,
        'max_tokens': max(toks),
        'min_tokens': min(toks),
        'mean_tokens_per_row': sum(toks)/len(toks),
        'max_words_per_row': max(wcs),
        'min_words_per_row': min(wcs),
        'mean_words_per_row': sum(wcs)/len(wcs),
        'too_long_rows': sum(1 for x in toks if x > MAX_TOK),
    }
    return stats


def main():
    t0 = time.time(); setup_env(); tok = load_tokenizer(); OUT.mkdir(parents=True, exist_ok=True)
    stats = {}
    for arm in ARMS:
        print('repacking', arm, flush=True)
        stats[arm] = repack_arm(tok, arm)
        print(json.dumps(stats[arm], indent=2), flush=True)
    meta = {
        'status': 'TOKENAWARE_REPACK_READY',
        'source_dir': str(IN),
        'max_untruncated_tokens': MAX_TOK,
        'tokenizer': 'baseline16k via babylm_masked_train.make_portable_tokenizer',
        'arms': stats,
        'elapsed_sec': time.time()-t0,
        'purpose': 'Remove research truncation mismatch before one decisive paired-restatement recheck; content/order preserved as exact word streams from research arms.'
    }
    (OUT/'materialization_meta.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps({'out_dir': str(OUT), 'meta': str(OUT/'materialization_meta.json'), 'arms': stats}, indent=2))

if __name__ == '__main__':
    main()
