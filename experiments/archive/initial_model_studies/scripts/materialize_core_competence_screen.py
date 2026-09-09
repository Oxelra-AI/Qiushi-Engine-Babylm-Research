#!/usr/bin/env python3
"""research/349: materialize replicated curriculum-ordering mechanism test.

Replicated-comparison constraints:
  - A single random order vs one curriculum order cannot carry a mechanism
    conclusion because ordering effects and fast EWoK are noisy.
  - Both arms must be read strictly in file order during training.
  - Use independent random orders or a second seed as replication before scaling.

This materializer therefore creates SAME-CONTENT arms:
  1. official_random_order_a: fixed 4M row set, random order seed A
  2. official_random_order_b: exact same row set, independent random order seed B
  3. official_curriculum: exact same row set, sorted simple -> complex

Rows, words, sources, and text hashes are identical across arms; only file order differs.
"""
from __future__ import annotations
import argparse, hashlib, json, os, pathlib, random, sys, time
from collections import Counter
from typing import List, Dict

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
sys.path.insert(0, str((ROOT / 'training/scripts').resolve()))
from babylm_masked_train_fullcycle import TRAIN_FILES, download_dataset, iter_examples

OUT_DIR = ROOT / 'data/core_competence_screen'


def env_setup():
    hf = ROOT / 'training/hf_home'
    os.environ['HF_HOME'] = str(hf.resolve())
    os.environ['HF_HUB_CACHE'] = str((hf / 'hub').resolve())
    os.environ['HF_DATASETS_CACHE'] = str((hf / 'datasets').resolve())
    os.environ['TRANSFORMERS_CACHE'] = str((hf / 'transformers').resolve())
    os.environ['HF_MODULES_CACHE'] = str((ROOT / 'training/hf_modules_cache').resolve())
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def curriculum_complexity(text: str) -> float:
    """Parse-free progressive-complexity score; higher = later in curriculum."""
    words = text.split()
    n = len(words)
    if n < 3:
        return 0.0
    clean = [w.lower().strip('.,!?;:\'"()-[]{}') for w in words]
    mean_wlen = sum(len(w) for w in clean) / n
    terminals = sum(1 for w in words if w.rstrip('"\')')[-1:] in '.!?')
    words_per_sent = n / max(1, terminals)
    ttr = len(set(clean)) / n
    subs = {'because','although','though','while','when','whenever','where','wherever','if','unless','until','before','after','since','whereas','whether','which','who','whom','whose'}
    sub_density = sum(1 for w in clean if w in subs) / n
    dialogue_fillers = {'yeah','yep','okay','ok','uh','um','hmm','hm','oh','ah','er','mhm','huh'}
    filler_density = sum(1 for w in clean if w in dialogue_fillers) / n
    score = (
        0.24 * min(mean_wlen / 7.0, 1.0) +
        0.30 * min(words_per_sent / 25.0, 1.0) +
        0.24 * ttr +
        0.18 * min(sub_density * 20.0, 1.0) -
        0.10 * min(filler_density * 10.0, 1.0)
    )
    return float(score)


def row_hash(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


def seq_hash(xs) -> str:
    return hashlib.sha256('\n'.join(map(str, xs)).encode('utf-8')).hexdigest()


def write_jsonl(rows: List[dict], path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def summarize(rows: List[dict], name: str) -> dict:
    scores = [r['curriculum_score'] for r in rows]
    words = sum(r['words'] for r in rows)
    n = len(rows)
    quartiles = []
    for q in range(4):
        a = q * n // 4
        b = (q + 1) * n // 4
        part = rows[a:b]
        quartiles.append({
            'quartile': q + 1,
            'rows': len(part),
            'words': sum(r['words'] for r in part),
            'mean_curriculum_score': sum(r['curriculum_score'] for r in part) / max(1, len(part)),
            'source_top3': dict(Counter(r['source'] for r in part).most_common(3)),
        })
    return {
        'name': name,
        'rows': n,
        'words': words,
        'mean_curriculum_score': sum(scores) / max(1, n),
        'min_curriculum_score': min(scores) if scores else None,
        'max_curriculum_score': max(scores) if scores else None,
        'source_dist': dict(Counter(r['source'] for r in rows).most_common()),
        'content_multiset_hash': seq_hash(sorted(r['row_hash'] for r in rows)),
        'order_hash': seq_hash([r['row_hash'] for r in rows]),
        'quartile_progression': quartiles,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target_words', type=int, default=4_000_000)
    ap.add_argument('--row_max_words', type=int, default=160)
    ap.add_argument('--content_seed', type=int, default=42, help='Seed selecting fixed 4M content set')
    ap.add_argument('--random_a_seed', type=int, default=142)
    ap.add_argument('--random_b_seed', type=int, default=243)
    ap.add_argument('--dataset_id', default='BabyLM-community/BabyLM-2026-Strict-Small')
    ap.add_argument('--dataset_revision', default='c92ab16b4f08858304b0815706065b3354d8fc0a')
    args = ap.parse_args()

    env_setup(); OUT_DIR.mkdir(parents=True, exist_ok=True); t0 = time.time()
    from types import SimpleNamespace
    dargs = SimpleNamespace(dataset_id=args.dataset_id, dataset_revision=args.dataset_revision)
    raw_dir, _ = download_dataset(dargs, OUT_DIR)
    files = [raw_dir / n for n in TRAIN_FILES]
    all_examples = list(iter_examples(files, 10_000_000, args.row_max_words))
    print(json.dumps({'event': 'loaded', 'rows': len(all_examples), 'words': sum(e.words for e in all_examples)}), flush=True)

    rows = []
    for idx, ex in enumerate(all_examples):
        h = row_hash(ex.text)
        rows.append({
            'text': ex.text, 'words': ex.words, 'source': ex.source,
            'kind': 'official', 'global_row_id': idx, 'row_hash': h,
            'curriculum_score': curriculum_complexity(ex.text),
        })

    # Select one fixed 4M content set. Since official rows are 160 words, target_words=4M -> 25k rows.
    rng_content = random.Random(args.content_seed)
    pool = rows[:]
    rng_content.shuffle(pool)
    fixed = []
    cum = 0
    for r in pool:
        if cum + r['words'] > args.target_words:
            continue
        fixed.append(r)
        cum += r['words']
        if cum == args.target_words:
            break
    if cum != args.target_words:
        raise RuntimeError(f'Could not hit exact target_words={args.target_words}; got {cum}')

    # Same content, three file orders.
    random_a = fixed[:]
    random_b = fixed[:]
    random.Random(args.random_a_seed).shuffle(random_a)
    random.Random(args.random_b_seed).shuffle(random_b)
    curriculum = sorted(fixed, key=lambda r: (r['curriculum_score'], r['global_row_id']))

    arms = {
        'official_random_order_a': random_a,
        'official_random_order_b': random_b,
        'official_curriculum': curriculum,
    }

    stats = {}
    for name, arm_rows in arms.items():
        out_rows = []
        for pos, r in enumerate(arm_rows):
            out_rows.append({
                'text': r['text'], 'words': r['words'], 'source': r['source'], 'kind': 'official',
                'condition': name, 'global_row_id': r['global_row_id'], 'row_hash': r['row_hash'],
                'file_position': pos, 'curriculum_score': round(r['curriculum_score'], 6),
            })
        write_jsonl(out_rows, OUT_DIR / f'{name}.jsonl')
        stats[name] = summarize(out_rows, name)

    content_hashes = {name: stats[name]['content_multiset_hash'] for name in arms}
    word_counts = {name: stats[name]['words'] for name in arms}
    meta = {
        'status': 'REPLICATED_CURRICULUM_ORDER_ARMS_MATERIALIZED',
        'mechanism': 'progressive_complexity_ordering_same_content',
        'hypothesis': 'If order matters, curriculum order should beat the distribution of independent random file orders on weighted multi-column proxy despite identical content.',
        'params': vars(args),
        'same_content_all_arms': len(set(content_hashes.values())) == 1,
        'same_word_count_all_arms': len(set(word_counts.values())) == 1,
        'content_hashes': content_hashes,
        'word_counts': word_counts,
        'stats': stats,
        'elapsed_sec': time.time() - t0,
    }
    (OUT_DIR / 'metadata.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({
        'event': 'done',
        'same_content_all_arms': meta['same_content_all_arms'],
        'same_word_count_all_arms': meta['same_word_count_all_arms'],
        'words': word_counts,
        'curriculum_q1': stats['official_curriculum']['quartile_progression'][0]['mean_curriculum_score'],
        'curriculum_q4': stats['official_curriculum']['quartile_progression'][3]['mean_curriculum_score'],
        'random_a_q1': stats['official_random_order_a']['quartile_progression'][0]['mean_curriculum_score'],
        'random_b_q1': stats['official_random_order_b']['quartile_progression'][0]['mean_curriculum_score'],
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
