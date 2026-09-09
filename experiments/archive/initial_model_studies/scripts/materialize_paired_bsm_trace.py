#!/usr/bin/env python3
"""research: materialize STRICTLY PAIRED BSM trace corpora.

Design principles (from research route decision):
  - Coherent and swapped arms share the same master binding events.
  - Both have internally correct supervision (no wrong-label training).
  - The ONLY difference is cross-row entity-value CONSISTENCY:
      coherent: stable global E→V mapping across all rows
      swapped: per-event random assignment, destroying cross-row consistency
  - Same row count, entity/value/template frequency, row order, target positions.
  - Includes a matched official short-row/targeted-mask reference corpus.

Four output corpora:
  1. official_standard: normal official rows, standard WWM
  2. official_short_targeted: official rows split to BSM row lengths, targeted mask
  3. bsm_paired_coherent: coherent binding + official, paired with swapped
  4. bsm_paired_swapped: text-level swapped binding + same official, per-event random
"""
from __future__ import annotations
import argparse, hashlib, json, os, pathlib, random, sys, time
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
sys.path.insert(0, str((ROOT / 'scripts').resolve()))
sys.path.insert(0, str((ROOT / 'training/scripts').resolve()))

from binding_switch_margin_pretest import verify_single_semantic_token, build_inventories
from babylm_masked_train_fullcycle import TRAIN_FILES, download_dataset, iter_examples, Example, make_portable_tokenizer

OUT_DIR = ROOT / 'data/paired_bsm_trace'

# More natural templates with varied structure
BINDING_TEMPLATES = [
    ("{E1} has a {V1} and {E2} has a {V2}.", "{Eq} really likes the {Vq}."),
    ("{E1} found a {V1}. {E2} found a {V2}.", "Later {Eq} lost the {Vq}."),
    ("{E1} bought a {V1}. {E2} bought a {V2}.", "{Eq} returned the {Vq} today."),
    ("{E1} brought {V1}. {E2} brought {V2}.", "{Eq} shared the {Vq} with everyone."),
    ("{E1} lives in {V1}. {E2} lives in {V2}.", "{Eq} went back to {Vq} last week."),
    ("{E1} works at the {V1}. {E2} works at the {V2}.", "{Eq} left the {Vq} early."),
    ("{E1} picked up the {V1}. {E2} picked up the {V2}.", "{Eq} kept the {Vq}."),
    ("{E1} ordered {V1}. {E2} ordered {V2}.", "{Eq} enjoyed the {Vq}."),
    ("{E1} chose {V1}. {E2} chose {V2}.", "{Eq} was happy with the {Vq}."),
    ("In the morning {E1} grabbed a {V1}. {E2} grabbed a {V2}.", "By noon {Eq} still had the {Vq}."),
    ("The teacher gave {E1} a {V1} and gave {E2} a {V2}.", "At recess {Eq} played with the {Vq}."),
    ("{E1} always carries a {V1}. {E2} always carries a {V2}.", "Today {Eq} forgot the {Vq} at home."),
]


@dataclass
class MasterEvent:
    """A master binding event that generates both coherent and swapped rows."""
    event_id: int
    e1: str
    e2: str
    v1: str  # v1 and v2 are the two values; assignment to entities differs by condition
    v2: str
    v1_tid: int
    v2_tid: int
    template_idx: int
    query_entity_is_e1: bool  # True = query about e1, False = query about e2


def generate_master_events(entities: Dict[str, int], values: Dict[str, int],
                           n_events: int, seed: int) -> List[MasterEvent]:
    """Generate the master event table shared by coherent and swapped arms."""
    rng = random.Random(seed)
    ent_list = list(entities.items())
    val_list = list(values.items())
    events = []
    for i in range(n_events):
        e1_name, _ = rng.choice(ent_list)
        e2_name, _ = rng.choice([e for e in ent_list if e[0] != e1_name])
        v1_name, v1_tid = rng.choice(val_list)
        v2_name, v2_tid = rng.choice([v for v in val_list if v[0] != v1_name])
        tidx = rng.randrange(len(BINDING_TEMPLATES))
        query_is_e1 = rng.random() < 0.5
        events.append(MasterEvent(
            event_id=i, e1=e1_name, e2=e2_name,
            v1=v1_name, v2=v2_name, v1_tid=v1_tid, v2_tid=v2_tid,
            template_idx=tidx, query_entity_is_e1=query_is_e1
        ))
    return events


def realize_event(event: MasterEvent, condition: str, swap_rng: random.Random) -> dict:
    """Realize a master event into a corpus row for the given condition.

    coherent: E1→V1, E2→V2 (stable global assignment)
    swapped: per-event 50% chance of E1→V1 or E1→V2 (destroys consistency)
    """
    ctx_tmpl, query_tmpl = BINDING_TEMPLATES[event.template_idx]

    if condition == 'coherent':
        # Stable: E1 always gets V1, E2 always gets V2
        assign_e1_v1 = True
    else:
        # Random per event: 50% chance E1 gets V1 or V2
        assign_e1_v1 = swap_rng.random() < 0.5

    if assign_e1_v1:
        ctx_v1, ctx_v2 = event.v1, event.v2
        ctx_v1_tid, ctx_v2_tid = event.v1_tid, event.v2_tid
    else:
        ctx_v1, ctx_v2 = event.v2, event.v1
        ctx_v1_tid, ctx_v2_tid = event.v2_tid, event.v1_tid

    # Build context: E1 has ctx_v1, E2 has ctx_v2
    context = ctx_tmpl.format(E1=event.e1, E2=event.e2, V1=ctx_v1, V2=ctx_v2)

    # Query entity and its correct answer
    if event.query_entity_is_e1:
        eq = event.e1
        vq = ctx_v1  # E1's value in this realization
        vq_tid = ctx_v1_tid
        distractor = ctx_v2
        distractor_tid = ctx_v2_tid
    else:
        eq = event.e2
        vq = ctx_v2  # E2's value in this realization
        vq_tid = ctx_v2_tid
        distractor = ctx_v1
        distractor_tid = ctx_v1_tid

    query = query_tmpl.format(Eq=eq, Vq=vq)
    full_text = context + " " + query

    # Find answer word span in query portion
    query_start = len(context) + 1
    vq_pos = query.rfind(vq)
    if vq_pos < 0:
        return None
    mask_start = query_start + vq_pos
    mask_end = mask_start + len(vq)

    if full_text[mask_start:mask_end] != vq:
        return None

    return {
        'text': full_text, 'words': len(full_text.split()),
        'kind': 'binding', 'event_id': event.event_id,
        'condition': condition,
        'query_entity': eq, 'bound_value': vq, 'distractor_value': distractor,
        'correct_tid': vq_tid, 'distractor_tid': distractor_tid,
        'mask_word': vq, 'mask_char_start': mask_start, 'mask_char_end': mask_end,
        'template_idx': event.template_idx,
        'assign_e1_v1': assign_e1_v1,
        'source': 'generated_binding_paired',
    }


def build_official_short_rows(official_pool: List[Example], target_row_words: int,
                              max_total_words: int, rng: random.Random) -> List[dict]:
    """Split official text into short rows matching BSM binding row length."""
    rows = []
    total_w = 0
    for ex in official_pool:
        words = ex.text.split()
        i = 0
        while i < len(words) and total_w < max_total_words:
            chunk_words = words[i:i + target_row_words]
            chunk_text = ' '.join(chunk_words)
            wc = len(chunk_words)
            if wc < 3:
                i += wc
                continue
            # Pick a random content word to target-mask
            candidates = [j for j in range(wc) if len(chunk_words[j]) > 2 and chunk_words[j].isalpha()]
            if not candidates:
                candidates = list(range(wc))
            mask_idx = rng.choice(candidates)
            mask_word = chunk_words[mask_idx]
            # Character span
            char_pos = sum(len(chunk_words[k]) + 1 for k in range(mask_idx))
            rows.append({
                'text': chunk_text, 'words': wc, 'kind': 'official_short',
                'event_id': -1, 'condition': 'official_short_targeted',
                'query_entity': '', 'bound_value': '', 'distractor_value': '',
                'correct_tid': -1, 'distractor_tid': -1,
                'mask_word': mask_word,
                'mask_char_start': char_pos, 'mask_char_end': char_pos + len(mask_word),
                'template_idx': -1, 'assign_e1_v1': False,
                'source': ex.source,
            })
            total_w += wc
            i += target_row_words
    return rows


def interleave_binding_official(binding_rows: List[dict], official_rows: List[dict],
                                binding_fraction: float, total_words: int,
                                rng: random.Random) -> List[dict]:
    """Interleave binding and official rows to target word count and fraction."""
    target_bind = int(total_words * binding_fraction)
    target_off = total_words - target_bind
    corpus = []
    bw, ow = 0, 0
    bi, oi = 0, 0
    while (bw < target_bind or ow < target_off) and (bi < len(binding_rows) * 5 or oi < len(official_rows)):
        if bw < target_bind and bi < len(binding_rows) * 3:
            row = binding_rows[bi % len(binding_rows)]
            corpus.append(row)
            bw += row['words']
            bi += 1
        if ow < target_off and oi < len(official_rows):
            row = official_rows[oi % len(official_rows)]
            corpus.append({**row, 'kind': 'official'})
            ow += row['words']
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
    ap.add_argument('--target_words', type=int, default=100000, help='Per-arm total corpus words')
    ap.add_argument('--binding_fraction', type=float, default=0.20)
    ap.add_argument('--n_events', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--target_binding_row_words', type=int, default=15)
    ap.add_argument('--dataset_id', default='BabyLM-community/BabyLM-2026-Strict-Small')
    ap.add_argument('--dataset_revision', default='c92ab16b4f08858304b0815706065b3354d8fc0a')
    args = ap.parse_args()

    env_setup()
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = make_portable_tokenizer("")
    inv = build_inventories(tokenizer)
    print(json.dumps({'event': 'inventories', 'train_entities': list(inv['train_entities'].keys()),
                      'train_values': list(inv['train_values'].keys())}), flush=True)

    # Generate master events
    events = generate_master_events(inv['train_entities'], inv['train_values'], args.n_events, args.seed)
    print(json.dumps({'event': 'master_events', 'count': len(events)}), flush=True)

    # Realize both conditions from the SAME master events
    coherent_rows = []
    swapped_rows = []
    swap_rng = random.Random(args.seed + 7777)  # independent per-event randomness for swapped

    for ev in events:
        coh = realize_event(ev, 'coherent', swap_rng)
        sw = realize_event(ev, 'swapped', swap_rng)
        if coh is not None and sw is not None:
            coherent_rows.append(coh)
            swapped_rows.append(sw)

    print(json.dumps({'event': 'binding_realized', 'coherent': len(coherent_rows),
                      'swapped': len(swapped_rows), 'paired': len(coherent_rows) == len(swapped_rows)}), flush=True)

    # Download official data
    from types import SimpleNamespace
    dargs = SimpleNamespace(dataset_id=args.dataset_id, dataset_revision=args.dataset_revision)
    raw_dir, _ = download_dataset(dargs, OUT_DIR)
    files = [raw_dir / n for n in TRAIN_FILES]
    official_pool = list(iter_examples(files, min(args.target_words * 10, 10_000_000), 160))
    print(json.dumps({'event': 'official_loaded', 'pool_size': len(official_pool)}), flush=True)

    # Build four corpora
    off_rng = random.Random(args.seed + 100)

    # 1. official_standard: normal official rows
    std_corpus = []
    cum = 0
    for ex in official_pool:
        if cum >= args.target_words:
            break
        std_corpus.append({'text': ex.text, 'words': ex.words, 'kind': 'official',
                           'event_id': -1, 'condition': 'official_standard',
                           'query_entity': '', 'bound_value': '', 'distractor_value': '',
                           'correct_tid': -1, 'distractor_tid': -1,
                           'mask_word': '', 'mask_char_start': -1, 'mask_char_end': -1,
                           'template_idx': -1, 'assign_e1_v1': False, 'source': ex.source})
        cum += ex.words
    off_rng.shuffle(std_corpus)
    write_jsonl(std_corpus, OUT_DIR / 'official_standard.jsonl')

    # 2. official_short_targeted: split into short rows
    short_rows = build_official_short_rows(official_pool, args.target_binding_row_words,
                                           args.target_words, random.Random(args.seed + 200))
    off_rng2 = random.Random(args.seed + 201)
    off_rng2.shuffle(short_rows)
    write_jsonl(short_rows[:args.target_words // max(1, args.target_binding_row_words)],
                OUT_DIR / 'official_short_targeted.jsonl')

    # 3 & 4. BSM paired coherent and swapped with same official rows
    # Use same official rows for both BSM arms
    bsm_official_rows = []
    cum = 0
    target_off = int(args.target_words * (1 - args.binding_fraction))
    for ex in official_pool:
        if cum >= target_off:
            break
        bsm_official_rows.append({'text': ex.text, 'words': ex.words, 'kind': 'official',
                                  'event_id': -1, 'condition': 'official_in_bsm',
                                  'query_entity': '', 'bound_value': '', 'distractor_value': '',
                                  'correct_tid': -1, 'distractor_tid': -1,
                                  'mask_word': '', 'mask_char_start': -1, 'mask_char_end': -1,
                                  'template_idx': -1, 'assign_e1_v1': False, 'source': ex.source})
        cum += ex.words

    coherent_corpus = interleave_binding_official(
        coherent_rows, bsm_official_rows, args.binding_fraction, args.target_words,
        random.Random(args.seed + 300))
    swapped_corpus = interleave_binding_official(
        swapped_rows, bsm_official_rows, args.binding_fraction, args.target_words,
        random.Random(args.seed + 300))  # SAME shuffle seed for matched order

    write_jsonl(coherent_corpus, OUT_DIR / 'bsm_paired_coherent.jsonl')
    write_jsonl(swapped_corpus, OUT_DIR / 'bsm_paired_swapped.jsonl')

    # Verify pairing
    coh_bind = [r for r in coherent_corpus if r['kind'] == 'binding']
    sw_bind = [r for r in swapped_corpus if r['kind'] == 'binding']
    n_paired = sum(1 for a, b in zip(coh_bind, sw_bind)
                   if a['event_id'] == b['event_id'] and a['query_entity'] == b['query_entity'])

    # Compute stats
    stats = {}
    for name, corpus in [('official_standard', std_corpus),
                         ('official_short_targeted', short_rows),
                         ('bsm_paired_coherent', coherent_corpus),
                         ('bsm_paired_swapped', swapped_corpus)]:
        words = sum(r['words'] for r in corpus)
        bind_w = sum(r['words'] for r in corpus if r['kind'] == 'binding')
        stats[name] = {'rows': len(corpus), 'words': words, 'binding_words': bind_w,
                       'official_words': words - bind_w,
                       'mean_words_per_row': words / max(1, len(corpus)),
                       'estimated_updates_b64': len(corpus) // 64}

    meta = {
        'status': 'PAIRED_CORPUS_MATERIALIZED',
        'params': vars(args),
        'n_master_events': len(events),
        'n_paired_binding_rows': n_paired,
        'total_binding_rows_coherent': len(coh_bind),
        'total_binding_rows_swapped': len(sw_bind),
        'pairing_exact': n_paired == len(coh_bind) == len(sw_bind),
        'stats': stats,
        'elapsed_sec': time.time() - t0,
    }
    (OUT_DIR / 'corpus_metadata.json').write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'event': 'done', **{k: v for k, v in meta.items() if k != 'params'}}, indent=2), flush=True)


if __name__ == '__main__':
    main()
