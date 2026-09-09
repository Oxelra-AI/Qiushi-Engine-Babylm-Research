#!/usr/bin/env python3
"""research fast pre-flight verifier for 4M BSM trace.

The full BSMDataset verifier timed out at 4M because it recomputed word groups for
all rows. This verifier preserves the scientific checks needed before training:
  - row count, words, word-length sequence;
  - training-mode sequence (T=targeted, W=standard WWM);
  - target spans in official reference are complete whitespace-word/phrase spans;
  - actual target-token counts under the trainer's tokenizer/offset-overlap rule;
  - shuffled batch target-token counts/targeted-row counts/batch words using the
    same seed and RandomSampler order as training.

It does not build word groups because those affect ordinary WWM token selection,
not the targeted-token supervision-strength confound being checked here.
"""
from __future__ import annotations
import argparse, hashlib, json, os, pathlib, sys
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
sys.path.insert(0, str((ROOT/'training/scripts').resolve()))
from babylm_masked_train_fullcycle import make_portable_tokenizer, reset_all_rng


def env_setup():
    hf = ROOT / 'training/hf_home'
    os.environ['HF_HOME'] = str(hf.resolve())
    os.environ['HF_HUB_CACHE'] = str((hf/'hub').resolve())
    os.environ['HF_DATASETS_CACHE'] = str((hf/'datasets').resolve())
    os.environ['TRANSFORMERS_CACHE'] = str((hf/'transformers').resolve())
    os.environ['HF_MODULES_CACHE'] = str((ROOT/'training/hf_modules_cache').resolve())
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def load_jsonl(path: pathlib.Path) -> List[dict]:
    rows=[]
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def seq_hash(xs) -> str:
    return hashlib.sha256(','.join(map(str, xs)).encode()).hexdigest()


def mode(row):
    return 'T' if row.get('kind') in ('binding','official_targeted') else 'W'


def is_word_boundary_span(text: str, c0: int, c1: int) -> bool:
    if c0 < 0 and c1 < 0:
        return True
    if c0 < 0 or c1 <= c0 or c1 > len(text):
        return False
    left_ok = (c0 == 0) or text[c0-1].isspace()
    right_ok = (c1 == len(text)) or text[c1].isspace()
    span = text[c0:c1]
    return left_ok and right_ok and span.strip() == span and any(ch.isalnum() for ch in span)


def token_counts_batched(tokenizer, rows: List[dict], seq_length: int, batch_size: int=1024) -> List[int]:
    counts = [0] * len(rows)
    idxs=[]; texts=[]; spans=[]
    for i,r in enumerate(rows):
        if mode(r) == 'T':
            c0=int(r.get('mask_char_start',-1)); c1=int(r.get('mask_char_end',-1))
            idxs.append(i); texts.append(r['text']); spans.append((c0,c1))
    for off in range(0, len(texts), batch_size):
        bt = texts[off:off+batch_size]
        bs = spans[off:off+batch_size]
        bi = idxs[off:off+batch_size]
        enc = tokenizer(bt, add_special_tokens=False, truncation=True, max_length=seq_length,
                        padding='max_length', return_offsets_mapping=True)
        for j, offsets in enumerate(enc['offset_mapping']):
            c0,c1 = bs[j]
            n=0
            for s,e in offsets:
                s=int(s); e=int(e)
                if e <= 0:
                    continue
                if s < c1 and e > c0:
                    n += 1
            counts[bi[j]] = n
    return counts


def shuffled_batches(values: List[int], batch_size: int, seed: int) -> List[int]:
    # Match DataLoader(RandomSampler) order after reset_all_rng(seed).
    reset_all_rng(seed)
    ds = TensorDataset(torch.arange(len(values)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    vals = torch.tensor(values, dtype=torch.long)
    out=[]
    for (idx,) in loader:
        out.append(int(vals[idx].sum().item()))
    return out


def summarize(name: str, path: pathlib.Path, tokenizer, seq_length: int, batch_size: int, seed: int) -> dict:
    rows = load_jsonl(path)
    word_lengths=[int(r['words']) for r in rows]
    modes=[mode(r) for r in rows]
    target_counts=token_counts_batched(tokenizer, rows, seq_length)
    target_rows=[1 if m=='T' else 0 for m in modes]
    batch_targets=shuffled_batches(target_counts, batch_size, seed)
    batch_targeted_rows=shuffled_batches(target_rows, batch_size, seed)
    batch_words=shuffled_batches(word_lengths, batch_size, seed)
    # natural span check only matters for official reference targeted rows; for BSM answer spans are whole values.
    bad_spans=[]
    for i,r in enumerate(rows):
        if mode(r) == 'T':
            if not is_word_boundary_span(r['text'], int(r.get('mask_char_start',-1)), int(r.get('mask_char_end',-1))):
                bad_spans.append(i)
                if len(bad_spans) >= 20:
                    break
    return {
        'path': str(path), 'rows': len(rows), 'words': sum(word_lengths),
        'word_lengths_hash': seq_hash(word_lengths),
        'training_modes_hash': seq_hash(modes),
        'per_row_target_tokens_hash': seq_hash(target_counts),
        'target_tokens_total': sum(target_counts),
        'targeted_rows_total': sum(target_rows),
        'batch_target_tokens': batch_targets,
        'batch_targeted_rows': batch_targeted_rows,
        'batch_words': batch_words,
        'batch_target_tokens_hash': seq_hash(batch_targets),
        'batch_targeted_rows_hash': seq_hash(batch_targeted_rows),
        'batch_words_hash': seq_hash(batch_words),
        'batch_target_tokens_range': [min(batch_targets), max(batch_targets)] if batch_targets else [0,0],
        'bad_target_span_count_capped20': len(bad_spans),
        'bad_target_span_examples': bad_spans,
    }


def compare(a: dict, b: dict) -> dict:
    return {
        'same_rows': a['rows']==b['rows'],
        'same_words': a['words']==b['words'],
        'same_word_lengths': a['word_lengths_hash']==b['word_lengths_hash'],
        'same_training_modes': a['training_modes_hash']==b['training_modes_hash'],
        'same_per_row_target_tokens': a['per_row_target_tokens_hash']==b['per_row_target_tokens_hash'],
        'same_batch_target_tokens': a['batch_target_tokens_hash']==b['batch_target_tokens_hash'],
        'same_batch_targeted_rows': a['batch_targeted_rows_hash']==b['batch_targeted_rows_hash'],
        'same_batch_words': a['batch_words_hash']==b['batch_words_hash'],
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--corpora', nargs='+', required=True)
    ap.add_argument('--names', nargs='+', required=True)
    ap.add_argument('--out_json', required=True)
    ap.add_argument('--seq_length', type=int, default=128)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--seed', type=int, default=42)
    args=ap.parse_args()
    env_setup(); tokenizer=make_portable_tokenizer("")
    summaries={}
    for n,p in zip(args.names,args.corpora):
        print(json.dumps({'event':'summarize_start','name':n}), flush=True)
        summaries[n]=summarize(n, pathlib.Path(p), tokenizer, args.seq_length, args.batch_size, args.seed)
        print(json.dumps({'event':'summarize_done','name':n,'rows':summaries[n]['rows'],'target_total':summaries[n]['target_tokens_total'],'batch_range':summaries[n]['batch_target_tokens_range'],'bad_spans':summaries[n]['bad_target_span_count_capped20']}), flush=True)
    comps={}
    for i in range(len(args.names)):
        for j in range(i+1,len(args.names)):
            a=args.names[i]; b=args.names[j]
            comps[f'{a}_vs_{b}']=compare(summaries[a], summaries[b])
    payload={'status':'FAST_BATCH_MATCH_VERIFICATION','params':vars(args),'summaries':summaries,'comparisons':comps}
    out=pathlib.Path(args.out_json); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps({'out':str(out),'comparisons':comps}, indent=2), flush=True)

if __name__=='__main__':
    main()
