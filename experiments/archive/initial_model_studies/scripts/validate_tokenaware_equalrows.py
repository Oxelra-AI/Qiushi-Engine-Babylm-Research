#!/usr/bin/env python3
from __future__ import annotations
import json, os, pathlib, sys
ROOT = pathlib.Path('experiments/archive/initial_model_studies')
DATA = ROOT/'data/same_entity_deranged_1M_tokenaware_equalrows'
OUT = ROOT/'data/tokenaware_equalrows_validation.json'
ARMS = ['true_pair_adjacent','hard_negative_same_entity','orig_only','shuffled_pair_adjacent']
MAX_SEQ = 256
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

def validate(tok, arm):
    path = DATA/f'{arm}_1000000w_tokenaware_equalrows.jsonl'
    rows = 0; words = 0; toks = []; bad = []
    for line in path.open(encoding='utf-8'):
        obj = json.loads(line); rows += 1; words += int(obj['words'])
        tl = len(tok(obj['text'], add_special_tokens=False, truncation=False)['input_ids'])
        toks.append(tl)
        if tl > MAX_SEQ and len(bad) < 5:
            bad.append({'example_id': obj.get('example_id'), 'tokens': tl, 'words': obj['words'], 'text_prefix': obj['text'][:200]})
    return {
        'path': str(path),
        'rows': rows,
        'words': words,
        'steps_at_b128': (rows+BATCH-1)//BATCH,
        'max_untruncated_tokens': max(toks),
        'min_untruncated_tokens': min(toks),
        'mean_untruncated_tokens': sum(toks)/len(toks),
        'truncated_examples_at_256': sum(1 for x in toks if x > MAX_SEQ),
        'truncated_example_fraction': sum(1 for x in toks if x > MAX_SEQ)/len(toks),
        'bad_examples': bad,
    }

def main():
    setup_env(); tok = load_tokenizer(); payload = {}
    for arm in ARMS:
        payload[arm] = validate(tok, arm)
    payload['status'] = 'TOKENAWARE_EQUALROWS_VALIDATION'
    payload['interpretation'] = 'If each arm has 1M words, 5988 rows, 47 steps, and zero examples over 256 baseline16k tokens, research recheck removes research truncation and update-count mismatch.'
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps({'out': str(OUT), **payload}, indent=2, ensure_ascii=False))
if __name__ == '__main__': main()
