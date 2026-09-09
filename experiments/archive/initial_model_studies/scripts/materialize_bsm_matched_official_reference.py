#!/usr/bin/env python3
"""research: build an official reference matched to a BSM arm's structure.

The research official_short_targeted reference was too short-row-heavy: in the
100k smoke it had 6875 rows / 107 updates, whereas BSM had 1899 rows / 30
updates. This script repairs that confound by taking a BSM corpus as a schedule
and replacing each row with official text of the SAME word length and SAME row
training mode:

  - if BSM row kind == binding: official_targeted row with one metadata target
  - if BSM row kind == official: official row with standard WWM

Thus row count, word-count sequence, row order, batch count, and per-batch target
opportunity are matched to the BSM arm. Content is official-only.
"""
from __future__ import annotations
import argparse, json, os, pathlib, random, sys, time
from collections import Counter
from typing import List, Dict, Tuple

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
sys.path.insert(0, str((ROOT / 'training/scripts').resolve()))
from babylm_masked_train_fullcycle import TRAIN_FILES, download_dataset, iter_examples, Example

OUT_DIR = ROOT / 'data/bsm_matched_reference'


def env_setup():
    hf = ROOT / 'training/hf_home'
    os.environ['HF_HOME'] = str(hf.resolve())
    os.environ['HF_HUB_CACHE'] = str((hf/'hub').resolve())
    os.environ['HF_DATASETS_CACHE'] = str((hf/'datasets').resolve())
    os.environ['TRANSFORMERS_CACHE'] = str((hf/'transformers').resolve())
    os.environ['HF_MODULES_CACHE'] = str((ROOT/'training/hf_modules_cache').resolve())
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def load_jsonl(path: pathlib.Path) -> List[dict]:
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def official_word_stream(args, max_words: int) -> List[str]:
    from types import SimpleNamespace
    dargs = SimpleNamespace(dataset_id=args.dataset_id, dataset_revision=args.dataset_revision)
    raw_dir, _ = download_dataset(dargs, OUT_DIR)
    files = [raw_dir / n for n in TRAIN_FILES]
    words = []
    for ex in iter_examples(files, max_words, 160):
        words.extend(ex.text.split())
        if len(words) >= max_words:
            break
    return words


def choose_target_span(words: List[str], rng: random.Random) -> Tuple[str, int, int]:
    """Choose one target word and return (word, char_start, char_end)."""
    candidates = [i for i, w in enumerate(words) if len(w) > 2 and any(c.isalpha() for c in w)]
    if not candidates:
        candidates = list(range(len(words)))
    idx = rng.choice(candidates)
    start = sum(len(words[j]) + 1 for j in range(idx))
    word = words[idx]
    return word, start, start + len(word)


def make_reference(schedule_rows: List[dict], official_words: List[str], seed: int) -> List[dict]:
    rng = random.Random(seed)
    out = []
    ptr = 0
    for i, sched in enumerate(schedule_rows):
        w = int(sched['words'])
        if ptr + w > len(official_words):
            ptr = 0
        chunk = official_words[ptr:ptr+w]
        ptr += w
        if len(chunk) < w:
            raise RuntimeError('official stream unexpectedly too short')
        text = ' '.join(chunk)
        if sched.get('kind') == 'binding':
            # Match a BSM targeted-answer row with an official targeted row.
            mask_word, c0, c1 = choose_target_span(chunk, rng)
            kind = 'official_targeted'
        else:
            # Match a BSM official row with an official WWM row.
            mask_word, c0, c1 = '', -1, -1
            kind = 'official'
        out.append({
            'text': text,
            'words': w,
            'kind': kind,
            'event_id': sched.get('event_id', -1),
            'condition': 'official_bsm_matched_reference',
            'schedule_index': i,
            'schedule_kind': sched.get('kind', 'unknown'),
            'schedule_words': w,
            'query_entity': '',
            'bound_value': '',
            'distractor_value': '',
            'correct_tid': -1,
            'distractor_tid': -1,
            'mask_word': mask_word,
            'mask_char_start': c0,
            'mask_char_end': c1,
            'template_idx': -1,
            'assign_e1_v1': False,
            'source': 'official_matched_to_bsm_schedule',
        })
    return out


def write_jsonl(rows: List[dict], path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def summarize(rows: List[dict], batch_size: int = 64) -> dict:
    kinds = Counter(r.get('kind') for r in rows)
    words_by_kind = Counter()
    for r in rows:
        words_by_kind[r.get('kind')] += int(r['words'])
    return {
        'rows': len(rows),
        'words': sum(int(r['words']) for r in rows),
        'mean_words_per_row': sum(int(r['words']) for r in rows) / max(1, len(rows)),
        'estimated_updates': (len(rows) + batch_size - 1) // batch_size,
        'kind_rows': dict(kinds),
        'kind_words': dict(words_by_kind),
        'word_length_sequence_hash': __import__('hashlib').sha256(','.join(str(r['words']) for r in rows).encode()).hexdigest(),
        'kind_sequence_hash': __import__('hashlib').sha256(','.join(str(r.get('kind')) for r in rows).encode()).hexdigest(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--schedule_corpus', required=True, help='BSM corpus whose row structure to match')
    ap.add_argument('--out_jsonl', required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--dataset_id', default='BabyLM-community/BabyLM-2026-Strict-Small')
    ap.add_argument('--dataset_revision', default='c92ab16b4f08858304b0815706065b3354d8fc0a')
    args = ap.parse_args()
    env_setup(); OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    sched = load_jsonl(pathlib.Path(args.schedule_corpus))
    need_words = sum(int(r['words']) for r in sched) + 10000
    words = official_word_stream(args, need_words)
    ref = make_reference(sched, words, args.seed)
    out_path = pathlib.Path(args.out_jsonl)
    write_jsonl(ref, out_path)
    meta = {
        'status': 'OFFICIAL_BSM_MATCHED_REFERENCE',
        'schedule_corpus': args.schedule_corpus,
        'out_jsonl': str(out_path),
        'schedule_summary': summarize(sched),
        'reference_summary': summarize(ref),
        'same_row_count': len(sched) == len(ref),
        'same_word_sequence': [int(r['words']) for r in sched] == [int(r['words']) for r in ref],
        'targeted_rows_match_binding_rows': sum(1 for r in ref if r['kind']=='official_targeted') == sum(1 for r in sched if r.get('kind')=='binding'),
        'elapsed_sec': time.time() - t0,
    }
    meta_path = out_path.with_suffix('.metadata.json')
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps({'event': 'done', 'out': str(out_path), 'meta': str(meta_path), **{k: meta[k] for k in ['same_row_count','same_word_sequence','targeted_rows_match_binding_rows']}}, indent=2), flush=True)

if __name__ == '__main__':
    main()
