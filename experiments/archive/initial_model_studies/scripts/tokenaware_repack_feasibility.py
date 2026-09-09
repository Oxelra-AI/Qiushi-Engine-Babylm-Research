#!/usr/bin/env python3
from __future__ import annotations
import json, os, pathlib, sys
ROOT = pathlib.Path('experiments/archive/initial_model_studies')
DATA = ROOT/'data/same_entity_deranged_1M'
OUT = ROOT/'data/tokenaware_repack_feasibility.json'
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

def iter_words(path):
    for line in path.open(encoding='utf-8'):
        obj=json.loads(line); txt=obj['text']
        for w in txt.split():
            yield w

def tok_len(tok, words):
    return len(tok(' '.join(words), add_special_tokens=False, truncation=False)['input_ids'])

def repack_stats(tok, arm):
    path = DATA/f'{arm}_1000000w.jsonl'
    rows=[]; buf=[]; total_words=0; too_long_single=0
    for w in iter_words(path):
        if not buf:
            buf=[w]; total_words += 1; continue
        cand = buf + [w]
        if tok_len(tok, cand) <= MAX_TOK:
            buf = cand
        else:
            rows.append((len(buf), tok_len(tok, buf)))
            buf = [w]
        total_words += 1
    if buf:
        rows.append((len(buf), tok_len(tok, buf)))
    for wc, tl in rows:
        if tl > MAX_TOK: too_long_single += 1
    return {
        'exact_words': total_words,
        'rows': len(rows),
        'steps_at_b128': (len(rows)+BATCH-1)//BATCH,
        'max_tokens': max(t for _,t in rows),
        'mean_words_per_row': sum(w for w,_ in rows)/len(rows),
        'mean_tokens_per_row': sum(t for _,t in rows)/len(rows),
        'min_words_per_row': min(w for w,_ in rows),
        'max_words_per_row': max(w for w,_ in rows),
        'too_long_rows': too_long_single,
    }

def main():
    setup_env(); tok=load_tokenizer(); out={}
    for arm in ARMS:
        print('processing', arm, flush=True)
        out[arm]=repack_stats(tok, arm)
        print(out[arm], flush=True)
    out['interpretation'] = 'Greedy no-truncation repack of existing JSONL word sequences at baseline16k max 256 tokens. If row/step counts differ materially across arms, a token-aware recheck changes update geometry unless trainer or packing is further controlled.'
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps({'out':str(OUT), 'stats':out}, indent=2))
if __name__=='__main__': main()
