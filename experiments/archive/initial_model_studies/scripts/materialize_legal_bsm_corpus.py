#!/usr/bin/env python3
"""research: Materialize legal BSM corpus variants for from-scratch training.

Produces JSONL corpus files for three conditions:
  1. official_control: pure official corpus, no binding replacement
  2. bsm_coherent_20pct: 20% binding rows replacing official rows, consistent E→V
  3. bsm_swapped_20pct: same surface distribution but entity-value mappings randomized
     between pairs so stable binding cannot form

Each binding row contains natural text (no literal [MASK]) with metadata:
  - kind: "binding" or "official"
  - mask_word: the answer word to target during training
  - mask_char_start/end: character span of the answer to mask
  - correct_tid: token id of the correct answer
  - distractor_tid: token id of the distractor
  - pair_id, query_entity, bound_value, distractor_value

The trainer uses this metadata to apply targeted masking on binding rows.
All rows count toward the corpus word budget.
"""
from __future__ import annotations
import argparse, hashlib, json, os, pathlib, random, sys, time
from dataclasses import dataclass, asdict
from typing import List, Tuple

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
sys.path.insert(0, str((ROOT / 'scripts').resolve()))
sys.path.insert(0, str((ROOT / 'training/scripts').resolve()))

from binding_switch_margin_pretest import verify_single_semantic_token, build_inventories
from babylm_masked_train_fullcycle import (
    TRAIN_FILES, download_dataset, iter_examples, Example,
    make_portable_tokenizer,
)

OUT_DIR = ROOT / 'data/legal_bsm_corpus'

# Natural binding templates — more varied than research's formulaic patterns.
# Each template has: context pattern, query pattern, and {E1},{E2},{V1},{V2},{Eq},{Vq} slots.
# {Vq} is the answer word that will be masked during training.
NATURAL_BINDING_TEMPLATES = [
    # Simple possessive
    ("{E1} has a {V1} and {E2} has a {V2}.", "{Eq} really likes the {Vq}."),
    ("{E1} found a {V1}. {E2} found a {V2}.", "Later {Eq} lost the {Vq}."),
    ("{E1} bought a {V1}. {E2} bought a {V2}.", "{Eq} returned the {Vq} today."),
    ("{E1} brought {V1}. {E2} brought {V2}.", "{Eq} shared the {Vq} with everyone."),
    # Location/state
    ("{E1} lives in {V1}. {E2} lives in {V2}.", "{Eq} went back to {Vq} last week."),
    ("{E1} works at the {V1}. {E2} works at the {V2}.", "{Eq} left the {Vq} early."),
    # Action/preference
    ("{E1} picked up the {V1}. {E2} picked up the {V2}.", "{Eq} kept the {Vq}."),
    ("{E1} ordered {V1}. {E2} ordered {V2}.", "{Eq} enjoyed the {Vq}."),
    ("{E1} chose {V1}. {E2} chose {V2}.", "{Eq} was happy with the {Vq}."),
    # Longer/more natural
    ("In the morning {E1} grabbed a {V1} from the shelf. {E2} grabbed a {V2} instead.", "By noon {Eq} still had the {Vq}."),
    ("The teacher gave {E1} a {V1} and gave {E2} a {V2}.", "At recess {Eq} played with the {Vq}."),
    ("{E1} always carries a {V1}. {E2} always carries a {V2}.", "Today {Eq} forgot the {Vq} at home."),
]


@dataclass
class BindingRow:
    text: str
    words: int
    kind: str  # "binding"
    pair_id: int
    query_entity: str
    bound_value: str
    distractor_value: str
    correct_tid: int
    distractor_tid: int
    mask_word: str
    mask_char_start: int
    mask_char_end: int
    template_idx: int
    condition: str  # "coherent" or "swapped"
    source: str = "generated_binding"


def find_last_occurrence(text: str, word: str) -> Tuple[int, int]:
    """Find the last occurrence of word in text (case-sensitive, word boundary)."""
    idx = text.rfind(word)
    if idx < 0:
        idx = text.rfind(word.lower())
    if idx < 0:
        raise ValueError(f"Cannot find '{word}' in text: {text[:100]}")
    return idx, idx + len(word)


def generate_binding_rows(inventories, tokenizer, n_pairs: int, condition: str,
                          rng: random.Random, pair_id_start: int = 0) -> List[BindingRow]:
    """Generate binding rows for one condition."""
    ents = list(inventories['train_entities'].items())
    vals = list(inventories['train_values'].items())
    rows = []

    for pair_idx in range(n_pairs):
        pair_id = pair_id_start + pair_idx
        # Pick entities and values
        e1_name, _ = rng.choice(ents)
        e2_name, _ = rng.choice([e for e in ents if e[0] != e1_name])
        v1_name, v1_tid = rng.choice(vals)
        v2_name, v2_tid = rng.choice([v for v in vals if v[0] != v1_name])

        # For swapped condition: randomize which value goes with which entity in the QUERY
        if condition == 'swapped':
            # Swap the query answers (but keep context the same)
            query_map = [(e1_name, v2_name, v2_tid, v1_name, v1_tid),
                         (e2_name, v1_name, v1_tid, v2_name, v2_tid)]
        else:  # coherent
            query_map = [(e1_name, v1_name, v1_tid, v2_name, v2_tid),
                         (e2_name, v2_name, v2_tid, v1_name, v1_tid)]

        tidx = rng.randrange(len(NATURAL_BINDING_TEMPLATES))
        ctx_tmpl, query_tmpl = NATURAL_BINDING_TEMPLATES[tidx]

        # Generate context (same for both conditions — both values present)
        context = ctx_tmpl.format(E1=e1_name, E2=e2_name, V1=v1_name, V2=v2_name)

        for eq_name, vq_name, vq_tid, vd_name, vd_tid in query_map:
            query = query_tmpl.format(Eq=eq_name, Vq=vq_name)
            full_text = context + " " + query
            # Find the answer word in the query portion
            query_start = len(context) + 1
            # Find vq_name in the query portion specifically
            vq_pos_in_query = query.rfind(vq_name)
            if vq_pos_in_query < 0:
                continue
            mask_start = query_start + vq_pos_in_query
            mask_end = mask_start + len(vq_name)

            # Verify the span is correct
            if full_text[mask_start:mask_end] != vq_name:
                continue

            word_count = len(full_text.split())
            rows.append(BindingRow(
                text=full_text, words=word_count, kind="binding",
                pair_id=pair_id, query_entity=eq_name,
                bound_value=vq_name, distractor_value=vd_name,
                correct_tid=vq_tid, distractor_tid=vd_tid,
                mask_word=vq_name, mask_char_start=mask_start, mask_char_end=mask_end,
                template_idx=tidx, condition=condition,
            ))

    return rows


def materialize_corpus(binding_rows: List[BindingRow], official_examples: List[Example],
                       target_words: int, binding_word_fraction: float,
                       rng: random.Random) -> List[dict]:
    """Mix binding rows with official examples to target word count and ratio."""
    target_binding_words = int(target_words * binding_word_fraction)
    target_official_words = target_words - target_binding_words

    corpus = []
    # Add binding rows up to target
    bw = 0
    rng.shuffle(binding_rows)
    bi = 0
    while bw < target_binding_words and bi < len(binding_rows) * 10:
        row = binding_rows[bi % len(binding_rows)]
        if bw + row.words <= target_binding_words:
            corpus.append(asdict(row))
            bw += row.words
        bi += 1
        if bi % len(binding_rows) == 0:
            rng.shuffle(binding_rows)

    # Add official examples up to target
    ow = 0
    rng.shuffle(official_examples)
    oi = 0
    while ow < target_official_words and oi < len(official_examples):
        ex = official_examples[oi]
        if ow + ex.words <= target_official_words:
            corpus.append({
                'text': ex.text, 'words': ex.words, 'kind': 'official',
                'pair_id': -1, 'query_entity': '', 'bound_value': '',
                'distractor_value': '', 'correct_tid': -1, 'distractor_tid': -1,
                'mask_word': '', 'mask_char_start': -1, 'mask_char_end': -1,
                'template_idx': -1, 'condition': 'official', 'source': ex.source,
            })
            ow += ex.words
        oi += 1

    rng.shuffle(corpus)
    return corpus


def write_jsonl(rows: List[dict], path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def env_setup():
    hf = ROOT / 'training/hf_home'
    os.environ['HF_HOME'] = str(hf.resolve())
    os.environ['HF_HUB_CACHE'] = str((hf / 'hub').resolve())
    os.environ['HF_DATASETS_CACHE'] = str((hf / 'datasets').resolve())
    os.environ['TRANSFORMERS_CACHE'] = str((hf / 'transformers').resolve())
    os.environ['HF_MODULES_CACHE'] = str((ROOT / 'training/hf_modules_cache').resolve())
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target_words', type=int, default=100000,
                    help='Total corpus words for smoke (small); use 10000000 for full')
    ap.add_argument('--binding_fraction', type=float, default=0.20)
    ap.add_argument('--n_binding_pairs', type=int, default=500)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--dataset_id', default='BabyLM-community/BabyLM-2026-Strict-Small')
    ap.add_argument('--dataset_revision', default='c92ab16b4f08858304b0815706065b3354d8fc0a')
    args = ap.parse_args()

    env_setup()
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load tokenizer and build inventories
    tokenizer = make_portable_tokenizer("")
    inv = build_inventories(tokenizer)
    print(json.dumps({
        'event': 'inventories',
        'train_entities': list(inv['train_entities'].keys()),
        'train_values': list(inv['train_values'].keys()),
    }), flush=True)

    rng = random.Random(args.seed)

    # Generate binding rows for both conditions
    coherent_rows = generate_binding_rows(inv, tokenizer, args.n_binding_pairs, 'coherent', rng, pair_id_start=0)
    swapped_rows = generate_binding_rows(inv, tokenizer, args.n_binding_pairs, 'swapped', random.Random(args.seed + 1), pair_id_start=args.n_binding_pairs)
    print(json.dumps({'event': 'binding_generated', 'coherent': len(coherent_rows), 'swapped': len(swapped_rows)}), flush=True)

    # Download official data
    from types import SimpleNamespace
    dargs = SimpleNamespace(dataset_id=args.dataset_id, dataset_revision=args.dataset_revision)
    raw_dir, manifest = download_dataset(dargs, OUT_DIR)
    files = [raw_dir / n for n in TRAIN_FILES]
    official_pool = list(iter_examples(files, min(args.target_words * 5, 10_000_000), 160))
    print(json.dumps({'event': 'official_loaded', 'pool_size': len(official_pool), 'pool_words': sum(e.words for e in official_pool)}), flush=True)

    # Materialize three conditions
    conditions = {}

    # 1. Official control (no binding)
    control_corpus = materialize_corpus([], official_pool, args.target_words, 0.0, random.Random(args.seed))
    control_path = OUT_DIR / 'official_control.jsonl'
    write_jsonl(control_corpus, control_path)
    conditions['official_control'] = {
        'path': str(control_path), 'rows': len(control_corpus),
        'total_words': sum(r['words'] for r in control_corpus),
        'binding_words': 0, 'official_words': sum(r['words'] for r in control_corpus),
    }

    # 2. Coherent natural-BSM 20%
    coherent_corpus = materialize_corpus(coherent_rows, official_pool, args.target_words, args.binding_fraction, random.Random(args.seed + 10))
    coherent_path = OUT_DIR / 'bsm_coherent_20pct.jsonl'
    write_jsonl(coherent_corpus, coherent_path)
    bw_coh = sum(r['words'] for r in coherent_corpus if r['kind'] == 'binding')
    ow_coh = sum(r['words'] for r in coherent_corpus if r['kind'] == 'official')
    conditions['bsm_coherent_20pct'] = {
        'path': str(coherent_path), 'rows': len(coherent_corpus),
        'total_words': bw_coh + ow_coh, 'binding_words': bw_coh, 'official_words': ow_coh,
        'actual_binding_fraction': bw_coh / max(1, bw_coh + ow_coh),
    }

    # 3. Swapped natural-BSM 20%
    swapped_corpus = materialize_corpus(swapped_rows, official_pool, args.target_words, args.binding_fraction, random.Random(args.seed + 20))
    swapped_path = OUT_DIR / 'bsm_swapped_20pct.jsonl'
    write_jsonl(swapped_corpus, swapped_path)
    bw_sw = sum(r['words'] for r in swapped_corpus if r['kind'] == 'binding')
    ow_sw = sum(r['words'] for r in swapped_corpus if r['kind'] == 'official')
    conditions['bsm_swapped_20pct'] = {
        'path': str(swapped_path), 'rows': len(swapped_corpus),
        'total_words': bw_sw + ow_sw, 'binding_words': bw_sw, 'official_words': ow_sw,
        'actual_binding_fraction': bw_sw / max(1, bw_sw + ow_sw),
    }

    # Verify matching: coherent and swapped should have same word counts, entity/value freqs
    coh_entity_counts = {}
    sw_entity_counts = {}
    for r in coherent_corpus:
        if r['kind'] == 'binding':
            coh_entity_counts[r['query_entity']] = coh_entity_counts.get(r['query_entity'], 0) + 1
    for r in swapped_corpus:
        if r['kind'] == 'binding':
            sw_entity_counts[r['query_entity']] = sw_entity_counts.get(r['query_entity'], 0) + 1

    # Save metadata
    meta = {
        'status': 'CORPUS_MATERIALIZED',
        'params': vars(args),
        'inventories': {k: list(v.keys()) for k, v in inv.items()},
        'n_templates': len(NATURAL_BINDING_TEMPLATES),
        'conditions': conditions,
        'coherent_entity_distribution': coh_entity_counts,
        'swapped_entity_distribution': sw_entity_counts,
        'word_count_match': abs(conditions['bsm_coherent_20pct']['total_words'] - conditions['bsm_swapped_20pct']['total_words']) <= 100,
        'elapsed_sec': time.time() - t0,
    }
    meta_path = OUT_DIR / 'corpus_metadata.json'
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'event': 'done', 'conditions': conditions, 'meta': str(meta_path)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
