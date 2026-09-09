#!/usr/bin/env python3
"""research: deterministic execution-path replay for relation-biased WWM masking.

No optimization, no model forward, no GPU, no official evaluation text.  This script
checks the actual 100M launch container consumed by the research trainer and replays
its masking path with disabled/base and boost-2 relation masking.

It answers the main remaining cheap questions before any expensive relation route:
1. Does the 100M trainer prefix consumed as 10M words equal the audited 10M pool?
2. With the resolved launch arguments, what visible groups/tokens and selected
   relation/non-relation masks are actually produced by the trainer code?
3. Are there fallback activations, zero-target rows, partial WWM groups, or special
   IDs used as random replacements?
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import statistics
import time
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path('.').resolve()
STUDY = ROOT / 'experiments/archive/representation_and_objectives'
WS = STUDY
SCRIPT = WS / 'scripts/relation_bias_accumulated_trainer.py'
POOL10 = ROOT / 'experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl'
STREAM100 = ROOT / 'experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl'
TOKENIZER_40K = WS / 'data/legal_representation_route_map/tokenizers/legal_byte_bpe_40k'
OUT_DIR = WS / 'data/relation_bias_execution_replay'
OUT_JSON = OUT_DIR / 'relation_bias_execution_replay.json'
OUT_STEP_JSONL = OUT_DIR / 'replay_step_records.jsonl'
NOTE = (ROOT / 'research/notes/representation_and_objectives/relation_bias_execution_replay.md')

EXPECTED_POOL10_SHA = '215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23'
EXPECTED_STREAM100_SHA = '3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691'
EXPECTED_TOKENIZER_SHA = '94b44bcf5901d097e20294e8220740758eac29af3ec20d58d8da867c85197758'
BASE_PROB = 0.15
BOOST = 2.0
PROB_MAX = 0.6


def now_utc() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def sha256_text_stream(texts: list[str]) -> str:
    h = hashlib.sha256()
    for s in texts:
        h.update(s.encode('utf-8'))
        h.update(b'\n')
    return h.hexdigest()


def row_fingerprint(obj: dict[str, Any]) -> str:
    # Exact row-object identity, independent of JSON key ordering.
    blob = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(blob).hexdigest()


def counter_digest(counter: Counter[str]) -> str:
    h = hashlib.sha256()
    for key, count in sorted(counter.items()):
        h.update(key.encode('ascii'))
        h.update(b':')
        h.update(str(count).encode('ascii'))
        h.update(b'\n')
    return h.hexdigest()


def q(xs: list[float], p: float) -> float | None:
    vals = sorted(float(x) for x in xs if math.isfinite(float(x)))
    if not vals:
        return None
    pos = p * (len(vals) - 1)
    lo = int(math.floor(pos)); hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def mean(xs: list[float]) -> float | None:
    return float(sum(xs) / len(xs)) if xs else None


def import_trainer():
    spec = importlib.util.spec_from_file_location('relation_bias_accumulated_trainer', SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot import {SCRIPT}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def read_rows_until_words(path: Path, target_words: int, max_rows: int | None = None) -> tuple[list[dict[str, Any]], int, int]:
    rows = []
    words = 0
    with path.open('r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows:
                break
            if not line.strip():
                continue
            obj = json.loads(line)
            text = str(obj['text'])
            w = int(obj.get('words', len(text.split())))
            if len(text.split()) != w:
                raise RuntimeError({'path': rel(path), 'row': i, 'word_mismatch': [len(text.split()), w]})
            if max_rows is None and words + w > target_words:
                raise RuntimeError({'path': rel(path), 'row': i, 'partial_example_needed': {'current_words': words, 'row_words': w, 'target': target_words}})
            rows.append(obj)
            words += w
            if max_rows is None and words == target_words:
                break
    return rows, words, len(rows)


def build_state(rb, current_step: int, total_steps: int):
    st = rb.base.MaskingCurriculumState(
        curriculum='wwm_fixed',
        mask_prob_start=0.15,
        mask_prob_end=0.15,
        switch_frac=0.7,
        amlm_window=10,
        amlm_lambda=0.2,
    )
    st.initialize(vocab_size=40_000, total_steps=total_steps)
    st.current_step = current_step
    return st


def tensor_hash(t: torch.Tensor) -> str:
    arr = t.detach().cpu().contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def count_selected_groups(select: torch.Tensor, word_group: torch.Tensor, relation_group: torch.Tensor) -> dict[str, int]:
    out = {
        'candidate_groups': 0,
        'relation_groups': 0,
        'selected_groups': 0,
        'selected_relation_groups': 0,
        'rows_zero_selected': 0,
        'partial_selected_groups': 0,
    }
    bsz = word_group.shape[0]
    for b in range(bsz):
        groups = word_group[b]
        valid_groups = torch.unique(groups[groups >= 0])
        out['candidate_groups'] += int(valid_groups.numel())
        selected_any_row = False
        for gid_t in valid_groups.tolist():
            gid = int(gid_t)
            gm = groups == gid
            rel = bool((gm & relation_group[b]).any().item())
            sel_count = int((gm & select[b]).sum().item())
            group_count = int(gm.sum().item())
            if rel:
                out['relation_groups'] += 1
            if sel_count > 0:
                selected_any_row = True
                out['selected_groups'] += 1
                if rel:
                    out['selected_relation_groups'] += 1
                if sel_count != group_count:
                    out['partial_selected_groups'] += 1
        if not selected_any_row:
            out['rows_zero_selected'] += 1
    return out


def observe_replacement_specials(masked_inputs: torch.Tensor, labels: torch.Tensor, original_inputs: torch.Tensor, tokenizer) -> dict[str, Any]:
    """Observe special IDs actually placed in selected positions.

    This does not consume or perturb the masking RNG.  It counts selected positions
    whose masked input is neither [MASK] nor the original token; this misses random
    replacements that coincidentally sampled the original ID, but safely detects
    whether special tokens were inserted by the corruption policy.
    """
    select = labels != -100
    random_like = select & (masked_inputs != tokenizer.mask_token_id) & (masked_inputs != original_inputs)
    n = int(random_like.sum().item())
    special_counts = Counter()
    specials = set(int(x) for x in tokenizer.all_special_ids)
    if n > 0:
        for x in masked_inputs[random_like].tolist():
            if int(x) in specials:
                tok = tokenizer.convert_ids_to_tokens(int(x))
                special_counts[str(tok)] += 1
    return {'observed_random_like_positions': n, 'observed_special_replacement_counts': dict(special_counts)}


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    s: dict[str, Any] = defaultdict(float)
    hist_zero_rows = []
    for r in records:
        for k, v in r.items():
            if isinstance(v, (int, float)):
                s[k] += float(v)
        hist_zero_rows.append(float(r.get('boost_rows_zero_selected', 0)))
    out = dict(s)
    # Derived ratios.
    out['base_selected_relation_group_frac'] = out.get('base_selected_relation_groups', 0.0) / out.get('base_selected_groups', 1.0)
    out['boost_selected_relation_group_frac'] = out.get('boost_selected_relation_groups', 0.0) / out.get('boost_selected_groups', 1.0)
    out['base_selected_relation_token_frac'] = out.get('base_selected_relation_tokens', 0.0) / out.get('base_selected_tokens', 1.0)
    out['boost_selected_relation_token_frac'] = out.get('boost_selected_relation_tokens', 0.0) / out.get('boost_selected_tokens', 1.0)
    out['boost_selected_group_multiplier_vs_base_realized'] = out.get('boost_selected_groups', 0.0) / out.get('base_selected_groups', 1.0)
    out['boost_selected_token_multiplier_vs_base_realized'] = out.get('boost_selected_tokens', 0.0) / out.get('base_selected_tokens', 1.0)
    out['boost_expected_group_multiplier_vs_uniform'] = out.get('boost_expected_selected_groups', 0.0) / (BASE_PROB * out.get('candidate_groups', 1.0))
    out['boost_expected_token_multiplier_vs_uniform'] = out.get('boost_expected_selected_tokens', 0.0) / (BASE_PROB * out.get('candidate_tokens', 1.0))
    out['base_rows_zero_selected_frac'] = out.get('base_rows_zero_selected', 0.0) / out.get('rows', 1.0)
    out['boost_rows_zero_selected_frac'] = out.get('boost_rows_zero_selected', 0.0) / out.get('rows', 1.0)
    out['boost_rows_clipped_frac'] = out.get('boost_rows_clipped', 0.0) / out.get('rows', 1.0)
    out['boost_special_random_replacements_total'] = sum(v for k, v in out.items() if k.startswith('boost_random_special_'))
    out['base_special_random_replacements_total'] = sum(v for k, v in out.items() if k.startswith('base_random_special_'))
    return out


def replay(max_rows: int | None, mask_seeds: list[int]) -> dict[str, Any]:
    start = time.time()
    rb = import_trainer()
    tok = rb.base.make_portable_tokenizer(str(TOKENIZER_40K))
    stream_rows, stream_words, stream_n = read_rows_until_words(STREAM100, 10_000_000, max_rows=max_rows)
    pool_rows, pool_words, pool_n = read_rows_until_words(POOL10, 10_000_000, max_rows=max_rows)
    texts_stream = [str(o['text']) for o in stream_rows]
    texts_pool = [str(o['text']) for o in pool_rows]
    stream_counter = Counter(row_fingerprint(o) for o in stream_rows)
    pool_counter = Counter(row_fingerprint(o) for o in pool_rows)
    prefix_identity = {
        'stream_rows': stream_n,
        'stream_words': stream_words,
        'pool_rows': pool_n,
        'pool_words': pool_words,
        'text_sequence_equal': texts_stream == texts_pool,
        'row_multiset_equal': stream_counter == pool_counter,
        'text_sequence_sha_stream': sha256_text_stream(texts_stream),
        'text_sequence_sha_pool': sha256_text_stream(texts_pool),
        'row_multiset_sha_stream': counter_digest(stream_counter),
        'row_multiset_sha_pool': counter_digest(pool_counter),
        'first_mismatch_index': None,
        'stream_only_row_fingerprint_count': sum((stream_counter - pool_counter).values()),
        'pool_only_row_fingerprint_count': sum((pool_counter - stream_counter).values()),
    }
    if texts_stream != texts_pool:
        for i, (a, b) in enumerate(zip(texts_stream, texts_pool)):
            if a != b:
                prefix_identity['first_mismatch_index'] = i
                break
        if prefix_identity['first_mismatch_index'] is None and len(texts_stream) != len(texts_pool):
            prefix_identity['first_mismatch_index'] = min(len(texts_stream), len(texts_pool))

    examples = [rb.base.Example(text=str(o['text']), words=int(o.get('words', len(str(o['text']).split()))), example_id=int(o.get('example_id', i)), source=str(o.get('source', 'example_jsonl'))) for i, o in enumerate(stream_rows)]
    cue_lexicon, multi_cues = rb.load_step54_lexicon()
    cue_single = rb.build_cue_single(cue_lexicon)
    rel_ds = rb.RelationMaskedChunkDataset(examples, tok, 256, cue_single, multi_cues)
    base_ds = rb.base.MaskedChunkDataset(examples, tok, 256)

    # Check first min(256, n) item identity with base dataset.
    mismatches = []
    for i in range(min(256, len(examples))):
        a = base_ds[i]
        b = rel_ds[i]
        for field in ['input_ids', 'attention_mask', 'word_group']:
            if not torch.equal(a[field], b[field]):
                mismatches.append({'row': i, 'field': field})

    loader = DataLoader(rel_ds, batch_size=64, shuffle=False, collate_fn=rb.relation_collate, num_workers=0)
    total_steps = math.ceil(len(rel_ds) / 256)
    step_records_all: list[dict[str, Any]] = []
    source_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_by_row = [ex.source for ex in examples]
    step_jsonl_lines = []

    cached_batches: list[tuple[int, dict[str, torch.Tensor], list[str]]] = []
    rows_seen_cache = 0
    for step_idx, batch in enumerate(rb.effective_batch_iterator(loader, 4), 1):
        bsz = int(batch['input_ids'].shape[0])
        sources = source_by_row[rows_seen_cache: rows_seen_cache + bsz]
        cached_batches.append((step_idx, batch, sources))
        rows_seen_cache += bsz
    if rows_seen_cache != len(examples):
        raise RuntimeError({'cached_rows_seen': rows_seen_cache, 'examples': len(examples)})

    for mask_seed in mask_seeds:
        gen_base = torch.Generator(device='cpu'); gen_base.manual_seed(mask_seed)
        gen_wrap = torch.Generator(device='cpu'); gen_wrap.manual_seed(mask_seed)
        gen_boost = torch.Generator(device='cpu'); gen_boost.manual_seed(mask_seed)
        for step_idx, batch, sources in cached_batches:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            word_group = batch['word_group']
            relation_group = batch['relation_group'].bool()
            bsz = int(input_ids.shape[0])
            st_base = build_state(rb, step_idx - 1, total_steps)
            st_wrap = build_state(rb, step_idx - 1, total_steps)
            st_boost = build_state(rb, step_idx - 1, total_steps)
            masked_base, labels_base = rb.base.apply_masking_curriculum(input_ids, attention_mask, word_group, tok, st_base, gen_base)
            masked_wrap, labels_wrap, disabled_stats = rb.apply_masking_with_relation(
                input_ids, attention_mask, word_group, relation_group, tok, st_wrap, gen_wrap,
                relation_enabled=False, relation_cue_boost=1.0,
            )
            if not torch.equal(masked_base, masked_wrap) or not torch.equal(labels_base, labels_wrap):
                raise RuntimeError({'disabled_wrapper_mismatch': {'mask_seed': mask_seed, 'step': step_idx}})
            masked_boost, labels_boost, boost_stats = rb.apply_masking_with_relation(
                input_ids, attention_mask, word_group, relation_group, tok, st_boost, gen_boost,
                relation_enabled=True, relation_cue_boost=BOOST, relation_prob_max=PROB_MAX,
            )
            select_base = labels_base != -100
            select_boost = labels_boost != -100
            group_base = count_selected_groups(select_base, word_group, relation_group)
            group_boost = count_selected_groups(select_boost, word_group, relation_group)
            base_special = observe_replacement_specials(masked_base, labels_base, input_ids, tok)
            boost_special = observe_replacement_specials(masked_boost, labels_boost, input_ids, tok)

            candidate_tokens = int((attention_mask.bool() & ~torch.isin(input_ids, torch.tensor(tok.all_special_ids))).sum().item())
            candidate_relation_tokens = int((attention_mask.bool() & relation_group & ~torch.isin(input_ids, torch.tensor(tok.all_special_ids))).sum().item())
            # Analytical expectation for boost on this effective batch.
            exp_selected_groups = 0.0
            exp_selected_tokens = 0.0
            exp_relation_groups = 0.0
            exp_relation_tokens = 0.0
            clipped_rows = 0
            for b in range(bsz):
                groups = word_group[b]
                valid_groups = torch.unique(groups[groups >= 0])
                n_rel = 0
                n_non = 0
                rel_tok = 0
                non_tok = 0
                for gid_t in valid_groups.tolist():
                    gm = groups == int(gid_t)
                    is_rel = bool((gm & relation_group[b]).any().item())
                    toks = int(gm.sum().item())
                    if is_rel:
                        n_rel += 1; rel_tok += toks
                    else:
                        n_non += 1; non_tok += toks
                rel_p, non_p, clipped = rb.normalized_relation_probs(BASE_PROB, BOOST, n_rel, n_non, PROB_MAX)
                clipped_rows += int(clipped)
                exp_selected_groups += rel_p * n_rel + non_p * n_non
                exp_selected_tokens += rel_p * rel_tok + non_p * non_tok
                exp_relation_groups += rel_p * n_rel
                exp_relation_tokens += rel_p * rel_tok

            rec = {
                'mask_seed': mask_seed,
                'step': step_idx,
                'rows': bsz,
                'charged_words': int(batch['words'].sum().item()),
                'candidate_groups': group_base['candidate_groups'],
                'candidate_tokens': candidate_tokens,
                'candidate_relation_groups': group_base['relation_groups'],
                'candidate_relation_tokens': candidate_relation_tokens,
                'base_selected_groups': group_base['selected_groups'],
                'base_selected_relation_groups': group_base['selected_relation_groups'],
                'base_selected_tokens': int(select_base.sum().item()),
                'base_selected_relation_tokens': int((select_base & relation_group).sum().item()),
                'base_rows_zero_selected': group_base['rows_zero_selected'],
                'base_partial_selected_groups': group_base['partial_selected_groups'],
                'boost_selected_groups': group_boost['selected_groups'],
                'boost_selected_relation_groups': group_boost['selected_relation_groups'],
                'boost_selected_tokens': int(select_boost.sum().item()),
                'boost_selected_relation_tokens': int((select_boost & relation_group).sum().item()),
                'boost_rows_zero_selected': group_boost['rows_zero_selected'],
                'boost_partial_selected_groups': group_boost['partial_selected_groups'],
                'boost_expected_selected_groups': exp_selected_groups,
                'boost_expected_selected_tokens': exp_selected_tokens,
                'boost_expected_relation_selected_groups': exp_relation_groups,
                'boost_expected_relation_selected_tokens': exp_relation_tokens,
                'boost_rows_clipped': clipped_rows,
                'disabled_base_masked_hash': tensor_hash(masked_base),
                'disabled_wrap_masked_hash': tensor_hash(masked_wrap),
                'disabled_base_labels_hash': tensor_hash(labels_base),
                'disabled_wrap_labels_hash': tensor_hash(labels_wrap),
                'boost_masked_hash': tensor_hash(masked_boost),
                'boost_labels_hash': tensor_hash(labels_boost),
            }
            for tok_name, c in base_special['observed_special_replacement_counts'].items():
                rec['base_random_special_' + tok_name] = int(c)
            rec['base_observed_random_like_positions'] = base_special['observed_random_like_positions']
            for tok_name, c in boost_special['observed_special_replacement_counts'].items():
                rec['boost_random_special_' + tok_name] = int(c)
            rec['boost_observed_random_like_positions'] = boost_special['observed_random_like_positions']
            step_records_all.append(rec)
            step_jsonl_lines.append(json.dumps(rec))

            # Also aggregate by source by making one-source pseudo-records cheaply at row granularity.
            # Compute per-source group and selected counts exactly by subselecting rows.
            for source in sorted(set(sources)):
                idxs = [i for i, s in enumerate(sources) if s == source]
                if not idxs:
                    continue
                idx = torch.tensor(idxs, dtype=torch.long)
                wb = word_group.index_select(0, idx)
                rbm = relation_group.index_select(0, idx)
                bsel = select_base.index_select(0, idx)
                tsel = select_boost.index_select(0, idx)
                gb = count_selected_groups(bsel, wb, rbm)
                gt = count_selected_groups(tsel, wb, rbm)
                sub_attn = attention_mask.index_select(0, idx)
                sub_ids = input_ids.index_select(0, idx)
                special_tensor = torch.tensor(tok.all_special_ids)
                cand = sub_attn.bool() & ~torch.isin(sub_ids, special_tensor)
                source_records[source].append({
                    'rows': len(idxs),
                    'candidate_groups': gb['candidate_groups'],
                    'candidate_tokens': int(cand.sum().item()),
                    'candidate_relation_groups': gb['relation_groups'],
                    'candidate_relation_tokens': int((cand & rbm).sum().item()),
                    'base_selected_groups': gb['selected_groups'],
                    'base_selected_relation_groups': gb['selected_relation_groups'],
                    'base_selected_tokens': int(bsel.sum().item()),
                    'base_selected_relation_tokens': int((bsel & rbm).sum().item()),
                    'base_rows_zero_selected': gb['rows_zero_selected'],
                    'base_partial_selected_groups': gb['partial_selected_groups'],
                    'boost_selected_groups': gt['selected_groups'],
                    'boost_selected_relation_groups': gt['selected_relation_groups'],
                    'boost_selected_tokens': int(tsel.sum().item()),
                    'boost_selected_relation_tokens': int((tsel & rbm).sum().item()),
                    'boost_rows_zero_selected': gt['rows_zero_selected'],
                    'boost_partial_selected_groups': gt['partial_selected_groups'],
                })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_STEP_JSONL.write_text('\n'.join(step_jsonl_lines) + ('\n' if step_jsonl_lines else ''), encoding='utf-8')
    global_agg = aggregate_records(step_records_all)
    by_source = {s: aggregate_records(recs) for s, recs in sorted(source_records.items())}

    payload = {
        'status': 'RELATION_BIAS_EXECUTION_REPLAY',
        'created_utc': now_utc(),
        'elapsed_sec': round(time.time() - start, 1),
        'purpose': 'CPU-only execution-path replay of disabled/base WWM and boost-2 relation WWM on the exact launch stream prefix.',
        'inputs': {
            'pool10': rel(POOL10),
            'pool10_file_sha256': sha256_file(POOL10),
            'expected_pool10_sha256': EXPECTED_POOL10_SHA,
            'stream100': rel(STREAM100),
            'stream100_file_sha256': sha256_file(STREAM100),
            'expected_stream100_sha256': EXPECTED_STREAM100_SHA,
            'tokenizer_40k': rel(TOKENIZER_40K),
            'tokenizer_json_sha256': sha256_file(TOKENIZER_40K / 'tokenizer.json'),
            'expected_tokenizer_json_sha256': EXPECTED_TOKENIZER_SHA,
            'trainer': rel(SCRIPT),
            'trainer_sha256': sha256_file(SCRIPT),
            'max_rows': max_rows,
            'mask_seeds': mask_seeds,
            'boost': BOOST,
            'base_prob': BASE_PROB,
            'relation_prob_max': PROB_MAX,
        },
        'prefix_identity': prefix_identity,
        'checks': {
            'pool10_file_sha_matches': sha256_file(POOL10) == EXPECTED_POOL10_SHA,
            'stream100_file_sha_matches': sha256_file(STREAM100) == EXPECTED_STREAM100_SHA,
            'tokenizer_sha_matches': sha256_file(TOKENIZER_40K / 'tokenizer.json') == EXPECTED_TOKENIZER_SHA,
            'prefix_text_sequence_equal': prefix_identity['text_sequence_equal'],
            'relation_dataset_matches_base_first256': len(mismatches) == 0,
            'relation_base_mismatches_first256': mismatches[:10],
        },
        'global_realized_masking': global_agg,
        'by_source_realized_masking': by_source,
        'step_records_file': rel(OUT_STEP_JSONL),
        'files': {'json': rel(OUT_JSON), 'note': rel(NOTE)},
        'interpretation': interpretation(prefix_identity, global_agg, by_source),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    write_note(payload)
    return payload


def interpretation(prefix_identity: dict[str, Any], global_agg: dict[str, Any], by_source: dict[str, Any]) -> list[str]:
    out: list[str] = []
    if prefix_identity['text_sequence_equal']:
        out.append('The 100M launch stream prefix consumed as the first 10M charged words is text-identical to the separately audited 10M JSONL; research large-pool relation doses apply to the actual trainer prefix.')
    elif prefix_identity.get('row_multiset_equal'):
        out.append('The 100M launch stream prefix is a permutation of the audited 10M JSONL row multiset, not the same sequence. The research large-pool dose applies to one pass of the actual trainer prefix, while order-specific learning effects remain part of the fixed stream design.')
    else:
        out.append(f"The checked 100M launch stream prefix subset is not the same row multiset as the compared 10M rows; first sequence mismatch {prefix_identity['first_mismatch_index']}, stream-only rows {prefix_identity.get('stream_only_row_fingerprint_count')}, pool-only rows {prefix_identity.get('pool_only_row_fingerprint_count')}. Full-prefix equality is required before a relation run is interpreted.")
    out.append(
        f"Realized boost-2 masks over the replayed mask seeds moved relation selected groups from {global_agg['base_selected_relation_group_frac']:.4f} to {global_agg['boost_selected_relation_group_frac']:.4f} and relation selected tokens from {global_agg['base_selected_relation_token_frac']:.4f} to {global_agg['boost_selected_relation_token_frac']:.4f}."
    )
    out.append(
        f"Realized selected group count under boost/base ratio was {global_agg['boost_selected_group_multiplier_vs_base_realized']:.4f}; target-token ratio was {global_agg['boost_selected_token_multiplier_vs_base_realized']:.4f}."
    )
    out.append(
        f"Boost expected group multiplier against uniform was {global_agg['boost_expected_group_multiplier_vs_uniform']:.6f}; expected token multiplier was {global_agg['boost_expected_token_multiplier_vs_uniform']:.6f}; row clipping fraction was {global_agg['boost_rows_clipped_frac']:.6f}."
    )
    out.append(
        f"Zero-selected rows occurred in base fraction {global_agg['base_rows_zero_selected_frac']:.6f} and boost fraction {global_agg['boost_rows_zero_selected_frac']:.6f}; partial selected groups were base={int(global_agg.get('base_partial_selected_groups',0))}, boost={int(global_agg.get('boost_partial_selected_groups',0))}."
    )
    out.append(
        f"Observed random-like replacements can include special IDs in both paths: base total {int(global_agg.get('base_special_random_replacements_total',0))}, boost total {int(global_agg.get('boost_special_random_replacements_total',0))} over the replayed masks. This is inherited from the base corruption policy rather than unique to relation bias."
    )
    vals = []
    for s, rec in by_source.items():
        if rec.get('boost_selected_relation_group_frac') is not None:
            vals.append((s, rec['boost_selected_relation_group_frac']))
    if vals:
        vals.sort(key=lambda x: x[1])
        out.append(f"Realized source-class boost dose ranges from {vals[0][1]:.4f} ({vals[0][0]}) to {vals[-1][1]:.4f} ({vals[-1][0]}), consistent with broad but source-dependent pressure.")
    return out


def write_note(payload: dict[str, Any]) -> None:
    g = payload['global_realized_masking']
    lines = [
        '# research relation-biased masking execution replay\n\n',
        'CPU-only replay of the exact research masking code on the 100M launch stream prefix. No optimization, model forward, GPU, or official evaluation text.\n\n',
        '## Prefix identity and integrity\n\n',
        f"- 100M stream SHA matched expected: `{payload['checks']['stream100_file_sha_matches']}`; 10M pool SHA matched expected: `{payload['checks']['pool10_file_sha_matches']}`; tokenizer SHA matched expected: `{payload['checks']['tokenizer_sha_matches']}`.\n",
        f"- Prefix text sequence equal: `{payload['prefix_identity']['text_sequence_equal']}`; row multiset equal: `{payload['prefix_identity']['row_multiset_equal']}`; stream prefix rows/words `{payload['prefix_identity']['stream_rows']}` / `{payload['prefix_identity']['stream_words']}`, pool rows/words `{payload['prefix_identity']['pool_rows']}` / `{payload['prefix_identity']['pool_words']}`.\n",
        f"- Relation dataset matched base input/attention/word_group for first 256 rows: `{payload['checks']['relation_dataset_matches_base_first256']}`.\n",
        '\n## Realized masking replay\n\n',
        f"- Mask seeds replayed: `{payload['inputs']['mask_seeds']}`.\n",
        f"- Relation selected-group fraction base -> boost2: `{g['base_selected_relation_group_frac']:.6f}` -> `{g['boost_selected_relation_group_frac']:.6f}`.\n",
        f"- Relation selected-token fraction base -> boost2: `{g['base_selected_relation_token_frac']:.6f}` -> `{g['boost_selected_relation_token_frac']:.6f}`.\n",
        f"- Realized boost/base selected-group ratio: `{g['boost_selected_group_multiplier_vs_base_realized']:.6f}`; selected-token ratio: `{g['boost_selected_token_multiplier_vs_base_realized']:.6f}`.\n",
        f"- Expected boost selected-group multiplier vs uniform: `{g['boost_expected_group_multiplier_vs_uniform']:.6f}`; expected token multiplier: `{g['boost_expected_token_multiplier_vs_uniform']:.6f}`.\n",
        f"- Boost clipped rows fraction: `{g['boost_rows_clipped_frac']:.6f}`.\n",
        f"- Zero-selected row fraction base/boost: `{g['base_rows_zero_selected_frac']:.6f}` / `{g['boost_rows_zero_selected_frac']:.6f}`.\n",
        f"- Partial selected groups base/boost: `{int(g.get('base_partial_selected_groups',0))}` / `{int(g.get('boost_partial_selected_groups',0))}`.\n",
        f"- Special random replacements base/boost: `{int(g.get('base_special_random_replacements_total',0))}` / `{int(g.get('boost_special_random_replacements_total',0))}`.\n",
        '\n## Interpretation\n\n',
    ]
    for x in payload['interpretation']:
        lines.append(f'- {x}\n')
    lines.extend([
        '\n## Files\n\n',
        f"- JSON: `{rel(OUT_JSON)}`\n",
        f"- Step records: `{rel(OUT_STEP_JSONL)}`\n",
    ])
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text(''.join(lines), encoding='utf-8')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--max_rows', type=int, default=0, help='0 = full 10M prefix')
    ap.add_argument('--mask_seeds', default='43023,43123,77777')
    args = ap.parse_args()
    mask_seeds = [int(x.strip()) for x in args.mask_seeds.split(',') if x.strip()]
    max_rows = args.max_rows if args.max_rows and args.max_rows > 0 else None
    payload = replay(max_rows=max_rows, mask_seeds=mask_seeds)
    print(json.dumps({
        'status': payload['status'],
        'elapsed_sec': payload['elapsed_sec'],
        'prefix_text_sequence_equal': payload['prefix_identity']['text_sequence_equal'],
        'prefix_row_multiset_equal': payload['prefix_identity']['row_multiset_equal'],
        'stream_prefix_rows_words': [payload['prefix_identity']['stream_rows'], payload['prefix_identity']['stream_words']],
        'pool_rows_words': [payload['prefix_identity']['pool_rows'], payload['prefix_identity']['pool_words']],
        'relation_dataset_matches_base_first256': payload['checks']['relation_dataset_matches_base_first256'],
        'boost_selected_relation_group_frac': payload['global_realized_masking']['boost_selected_relation_group_frac'],
        'boost_selected_relation_token_frac': payload['global_realized_masking']['boost_selected_relation_token_frac'],
        'boost_group_ratio_vs_base_realized': payload['global_realized_masking']['boost_selected_group_multiplier_vs_base_realized'],
        'boost_rows_clipped_frac': payload['global_realized_masking']['boost_rows_clipped_frac'],
        'json': rel(OUT_JSON),
        'note': rel(NOTE),
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
