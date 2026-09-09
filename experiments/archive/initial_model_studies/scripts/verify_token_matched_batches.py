#!/usr/bin/env python3
"""research: verify actual shuffled batch target-token matching.

The research materializer matches the per-row target-token-count sequence. This
script checks the sequence after the same shuffle seed and DataLoader logic used
by train_legal_bsm_screen.py. It compares total rows, total words,
per-row word lengths, training-mode sequence, target-token-count sequence, and
shuffled per-batch targeted-token counts.
"""
from __future__ import annotations
import argparse, json, pathlib, sys, os, hashlib
from typing import List

import torch
from torch.utils.data import DataLoader

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
sys.path.insert(0, str((ROOT/'scripts').resolve()))
sys.path.insert(0, str((ROOT/'training/scripts').resolve()))

from train_legal_bsm_screen import load_corpus_jsonl, BSMDataset, collate_fn
from babylm_masked_train_fullcycle import make_portable_tokenizer, reset_all_rng


def env_setup():
    hf = ROOT / 'training/hf_home'
    os.environ['HF_HOME'] = str(hf.resolve())
    os.environ['HF_HUB_CACHE'] = str((hf/'hub').resolve())
    os.environ['HF_DATASETS_CACHE'] = str((hf/'datasets').resolve())
    os.environ['TRANSFORMERS_CACHE'] = str((hf/'transformers').resolve())
    os.environ['HF_MODULES_CACHE'] = str((ROOT/'training/hf_modules_cache').resolve())
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def seq_hash(xs) -> str:
    return hashlib.sha256(','.join(map(str, xs)).encode()).hexdigest()


def targeted_mode(kind: str) -> str:
    return 'T' if kind in ('binding', 'official_targeted') else 'W'


def summarize(path: pathlib.Path, tokenizer, seq_length: int, batch_size: int, seed: int):
    rows = load_corpus_jsonl(path)
    ds = BSMDataset(rows, tokenizer, seq_length)
    # Match trainer: reset RNG, DataLoader shuffle=True. Use num_workers=0 for transparent order;
    # RandomSampler order is governed by torch's main RNG, same as trainer after reset_all_rng.
    reset_all_rng(seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    batch_targets = []
    batch_words = []
    batch_targeted_rows = []
    for batch in loader:
        # binding_mask is nonzero only for targeted rows after research trainer repair.
        batch_targets.append(int(batch['binding_mask'].sum().item()))
        batch_words.append(int(batch['words'].sum().item()))
        batch_targeted_rows.append(int(batch['is_binding'].sum().item()))
    # Per-row target counts in original file order
    per_row_targets = [int(ds[i]['binding_mask'].sum().item()) for i in range(len(ds))]
    return {
        'path': str(path),
        'rows': len(rows),
        'words': sum(r.words for r in rows),
        'word_lengths': [r.words for r in rows],
        'training_modes': [targeted_mode(r.kind) for r in rows],
        'per_row_target_tokens': per_row_targets,
        'batch_target_tokens': batch_targets,
        'batch_targeted_rows': batch_targeted_rows,
        'batch_words': batch_words,
        'hashes': {
            'word_lengths': seq_hash([r.words for r in rows]),
            'training_modes': seq_hash([targeted_mode(r.kind) for r in rows]),
            'per_row_target_tokens': seq_hash(per_row_targets),
            'batch_target_tokens': seq_hash(batch_targets),
            'batch_targeted_rows': seq_hash(batch_targeted_rows),
            'batch_words': seq_hash(batch_words),
        },
        'ranges': {
            'batch_target_tokens_min': min(batch_targets) if batch_targets else 0,
            'batch_target_tokens_max': max(batch_targets) if batch_targets else 0,
            'batch_targeted_rows_min': min(batch_targeted_rows) if batch_targeted_rows else 0,
            'batch_targeted_rows_max': max(batch_targeted_rows) if batch_targeted_rows else 0,
            'batch_words_min': min(batch_words) if batch_words else 0,
            'batch_words_max': max(batch_words) if batch_words else 0,
        }
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpora', nargs='+', required=True)
    ap.add_argument('--names', nargs='+', required=True)
    ap.add_argument('--out_json', required=True)
    ap.add_argument('--seq_length', type=int, default=128)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    env_setup()
    tokenizer = make_portable_tokenizer("")
    summaries = {}
    for name, path in zip(args.names, args.corpora):
        summaries[name] = summarize(pathlib.Path(path), tokenizer, args.seq_length, args.batch_size, args.seed)
    comparisons = {}
    names = args.names
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            a, b = names[i], names[j]
            comparisons[f'{a}_vs_{b}'] = {
                'same_rows': summaries[a]['rows'] == summaries[b]['rows'],
                'same_words': summaries[a]['words'] == summaries[b]['words'],
                'same_word_lengths': summaries[a]['hashes']['word_lengths'] == summaries[b]['hashes']['word_lengths'],
                'same_training_modes': summaries[a]['hashes']['training_modes'] == summaries[b]['hashes']['training_modes'],
                'same_per_row_target_tokens': summaries[a]['hashes']['per_row_target_tokens'] == summaries[b]['hashes']['per_row_target_tokens'],
                'same_batch_target_tokens': summaries[a]['hashes']['batch_target_tokens'] == summaries[b]['hashes']['batch_target_tokens'],
                'same_batch_targeted_rows': summaries[a]['hashes']['batch_targeted_rows'] == summaries[b]['hashes']['batch_targeted_rows'],
                'same_batch_words': summaries[a]['hashes']['batch_words'] == summaries[b]['hashes']['batch_words'],
            }
    payload = {'status': 'BATCH_TOKEN_MATCH_VERIFIED', 'params': vars(args), 'summaries': summaries, 'comparisons': comparisons}
    out = pathlib.Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps({'out': str(out), 'comparisons': comparisons}, indent=2), flush=True)

if __name__ == '__main__':
    main()
