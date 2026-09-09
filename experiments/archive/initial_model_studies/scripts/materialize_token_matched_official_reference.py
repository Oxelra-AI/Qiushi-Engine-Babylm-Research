#!/usr/bin/env python3
"""research: token-level matched official reference for paired BSM trace.

Repairs the research reference: row count, word-length sequence, and update count
were matched, but actual loss-bearing target-token counts still differed because
official target words often tokenize into more subwords than BSM answer words.

This script takes a BSM schedule corpus and produces an official-only reference
with:
  - exact same row count and word-length sequence;
  - same row kind sequence at the training-mode level:
      BSM binding -> official_targeted, BSM official -> official;
  - for every targeted row, the SAME actual target-token count under the trainer's
    tokenizer/offset-overlap rule.

It records per-row target-token counts and hashes so batch target-token
opportunity can be audited before running any 4M trace.
"""
from __future__ import annotations
import argparse, hashlib, json, os, pathlib, random, sys, time
from collections import Counter
from typing import List, Tuple

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
sys.path.insert(0, str((ROOT / 'training/scripts').resolve()))
from babylm_masked_train_fullcycle import TRAIN_FILES, download_dataset, iter_examples, make_portable_tokenizer

OUT_DIR = ROOT / 'data/token_matched_reference'


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


def token_overlap_count(tokenizer, text: str, c0: int, c1: int, seq_length: int) -> int:
    if c0 < 0 or c1 <= c0:
        return 0
    enc = tokenizer(
        text, add_special_tokens=False, truncation=True, max_length=seq_length,
        padding='max_length', return_tensors=None, return_offsets_mapping=True,
    )
    n = 0
    for start, end in enc['offset_mapping']:
        if end <= 0:
            continue
        if start < c1 and end > c0:
            n += 1
    return n


def word_spans(words: List[str]) -> List[Tuple[int, int, str]]:
    spans = []
    pos = 0
    for w in words:
        spans.append((pos, pos + len(w), w))
        pos += len(w) + 1
    return spans


def _count_offsets_in_span(offsets, c0: int, c1: int) -> int:
    return sum(1 for s, e in offsets if e > 0 and s < c1 and e > c0)


def choose_span_with_token_count(tokenizer, words: List[str], desired: int, rng: random.Random,
                                 seq_length: int) -> Tuple[str, int, int, int, bool]:
    """Choose a complete natural official word/short-phrase span with exact token count.

    This intentionally rejects the unsafe offset-fragment repair from research:
    target spans must align to whitespace word boundaries in the official text.
    We allow short contiguous phrases because the BSM answers can have more than
    one tokenizer target token, but the span is always made of complete official
    words, not word-internal fragments or punctuation-only substrings.
    """
    text = ' '.join(words)
    if desired <= 0:
        return '', -1, -1, 0, True
    spans = word_spans(words)
    enc = tokenizer(
        text, add_special_tokens=False, truncation=True, max_length=seq_length,
        padding='max_length', return_tensors=None, return_offsets_mapping=True,
    )
    offsets = [(int(s), int(e)) for s, e in enc['offset_mapping'] if int(e) > int(s)]
    candidates = []
    closest = []
    # Search complete-word contiguous phrases. Prefer short phrases and content text.
    for i in range(len(spans)):
        for j in range(i, len(spans)):
            c0 = spans[i][0]
            c1 = spans[j][1]
            span_text = text[c0:c1]
            if not span_text.strip() or not any(ch.isalnum() for ch in span_text):
                continue
            cnt = _count_offsets_in_span(offsets, c0, c1)
            n_words = j - i + 1
            if cnt == desired:
                candidates.append((n_words, span_text, c0, c1, cnt))
            closest.append((abs(cnt - desired), n_words, span_text, c0, c1, cnt))
            # Token counts are monotone with longer phrases, so stop very overshooting spans.
            if cnt > desired + 4:
                break
    if candidates:
        min_words = min(c[0] for c in candidates)
        short = [c for c in candidates if c[0] == min_words]
        _, span_text, c0, c1, cnt = rng.choice(short)
        return span_text, c0, c1, cnt, True
    if not closest:
        return '', -1, -1, 0, False
    closest.sort(key=lambda x: (x[0], x[1]))
    _, _, span_text, c0, c1, cnt = closest[0]
    return span_text, c0, c1, cnt, False


def make_reference(schedule_rows: List[dict], official_words: List[str], tokenizer, seed: int,
                   seq_length: int, chunk_search_tries: int = 2048) -> Tuple[List[dict], List[int], List[int], int]:
    rng = random.Random(seed)
    out = []
    sched_counts = []
    ref_counts = []
    failures = 0
    ptr = 0
    max_start = max(0, len(official_words) - 1)
    for i, sched in enumerate(schedule_rows):
        w = int(sched['words'])
        if ptr + w > len(official_words):
            ptr = 0
        chunk_start = ptr
        chunk = official_words[chunk_start:chunk_start+w]
        if len(chunk) < w:
            raise RuntimeError('official stream unexpectedly too short')
        if sched.get('kind') == 'binding':
            desired = token_overlap_count(
                tokenizer, sched['text'], int(sched.get('mask_char_start', -1)),
                int(sched.get('mask_char_end', -1)), seq_length
            )
            # First try the sequential official chunk. If it lacks a complete natural
            # word/phrase with the exact token count, search later same-length chunks.
            # This preserves the BSM row-length sequence while avoiding unnatural
            # word-internal target fragments.
            best = None
            for attempt in range(chunk_search_tries):
                cand_start = chunk_start + attempt * max(1, w)
                if cand_start + w > len(official_words):
                    cand_start = (cand_start % max(1, len(official_words) - w))
                cand = official_words[cand_start:cand_start+w]
                mask_word, c0, c1, actual, exact = choose_span_with_token_count(
                    tokenizer, cand, desired, rng, seq_length
                )
                if best is None or abs(actual - desired) < abs(best[3] - desired):
                    best = (cand, mask_word, c0, c1, actual, exact, cand_start)
                if exact:
                    break
            chunk, mask_word, c0, c1, actual, exact, used_start = best
            ptr = used_start + w
            if not exact:
                failures += 1
            kind = 'official_targeted'
            sched_counts.append(desired)
            ref_counts.append(actual)
        else:
            desired = 0
            actual = 0
            mask_word, c0, c1 = '', -1, -1
            kind = 'official'
            ptr = chunk_start + w
            sched_counts.append(0)
            ref_counts.append(0)
        text = ' '.join(chunk)
        out.append({
            'text': text,
            'words': w,
            'kind': kind,
            'event_id': sched.get('event_id', -1),
            'condition': 'official_bsm_token_matched_reference',
            'schedule_index': i,
            'schedule_kind': sched.get('kind', 'unknown'),
            'schedule_words': w,
            'schedule_target_token_count': desired,
            'reference_target_token_count': actual,
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
            'source': 'official_token_matched_to_bsm_schedule',
        })
    return out, sched_counts, ref_counts, failures


def write_jsonl(rows: List[dict], path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def sequence_hash(seq) -> str:
    return hashlib.sha256(','.join(map(str, seq)).encode()).hexdigest()


def summarize(rows: List[dict], target_counts: List[int], batch_size: int = 64) -> dict:
    kinds = Counter(r.get('kind') for r in rows)
    words_by_kind = Counter()
    targets_by_kind = Counter()
    for r, tc in zip(rows, target_counts):
        k = r.get('kind')
        words_by_kind[k] += int(r['words'])
        targets_by_kind[k] += int(tc)
    batch_targets = [sum(target_counts[i:i+batch_size]) for i in range(0, len(target_counts), batch_size)]
    return {
        'rows': len(rows),
        'words': sum(int(r['words']) for r in rows),
        'mean_words_per_row': sum(int(r['words']) for r in rows) / max(1, len(rows)),
        'estimated_updates': (len(rows) + batch_size - 1) // batch_size,
        'kind_rows': dict(kinds),
        'kind_words': dict(words_by_kind),
        'kind_target_tokens': dict(targets_by_kind),
        'target_tokens_total': sum(target_counts),
        'target_tokens_per_batch': batch_targets,
        'target_tokens_per_batch_min': min(batch_targets) if batch_targets else 0,
        'target_tokens_per_batch_max': max(batch_targets) if batch_targets else 0,
        'target_count_sequence_hash': sequence_hash(target_counts),
        'word_length_sequence_hash': sequence_hash([r['words'] for r in rows]),
        'training_mode_sequence_hash': sequence_hash(['T' if r.get('kind') in ('binding','official_targeted') else 'W' for r in rows]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--schedule_corpus', required=True, help='BSM corpus whose row/token target structure to match')
    ap.add_argument('--out_jsonl', required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--seq_length', type=int, default=128)
    ap.add_argument('--dataset_id', default='BabyLM-community/BabyLM-2026-Strict-Small')
    ap.add_argument('--dataset_revision', default='c92ab16b4f08858304b0815706065b3354d8fc0a')
    args = ap.parse_args()
    env_setup(); OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tokenizer = make_portable_tokenizer("")
    sched = load_jsonl(pathlib.Path(args.schedule_corpus))
    need_words = sum(int(r['words']) for r in sched) + 10000
    words = official_word_stream(args, need_words)
    ref, sched_counts, ref_counts, failures = make_reference(sched, words, tokenizer, args.seed, args.seq_length)
    out_path = pathlib.Path(args.out_jsonl)
    write_jsonl(ref, out_path)
    meta = {
        'status': 'OFFICIAL_BSM_TOKEN_MATCHED_REFERENCE',
        'schedule_corpus': args.schedule_corpus,
        'out_jsonl': str(out_path),
        'seq_length': args.seq_length,
        'schedule_summary': summarize(sched, sched_counts),
        'reference_summary': summarize(ref, ref_counts),
        'same_row_count': len(sched) == len(ref),
        'same_word_sequence': [int(r['words']) for r in sched] == [int(r['words']) for r in ref],
        'same_target_token_count_sequence': sched_counts == ref_counts,
        'target_match_failures': failures,
        'targeted_rows_match_binding_rows': sum(1 for r in ref if r['kind']=='official_targeted') == sum(1 for r in sched if r.get('kind')=='binding'),
        'elapsed_sec': time.time() - t0,
    }
    meta_path = out_path.with_suffix('.metadata.json')
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps({
        'event': 'done', 'out': str(out_path), 'meta': str(meta_path),
        'same_row_count': meta['same_row_count'],
        'same_word_sequence': meta['same_word_sequence'],
        'same_target_token_count_sequence': meta['same_target_token_count_sequence'],
        'target_match_failures': failures,
        'schedule_targets_per_batch': [meta['schedule_summary']['target_tokens_per_batch_min'], meta['schedule_summary']['target_tokens_per_batch_max']],
        'reference_targets_per_batch': [meta['reference_summary']['target_tokens_per_batch_min'], meta['reference_summary']['target_tokens_per_batch_max']],
    }, indent=2), flush=True)

if __name__ == '__main__':
    main()
