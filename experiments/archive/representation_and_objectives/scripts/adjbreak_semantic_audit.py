#!/usr/bin/env python3
"""research: semantic-distance and visibility-transition audit for adjbreak control.

This audits the CPU-only adjacency-broken compact_view_reinvest control before any
expensive training decision.  It measures whether donor rewrites remain semantic
near-matches to their assigned sources, compares original-vs-control visibility by
slot, identifies exceptional assignments, and verifies that the intended future
training stream would share the reference pass-wise row-index permutations.
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import pathlib
import random
import re
from typing import Any, Iterable

ROOT = pathlib.Path(".")
A01 = ROOT / "experiments/archive" / 'representation_and_objectives'
DENSITY_DIR = ROOT / "experiments/archive" / 'frontier_consolidation' / "data" / "density_core_reinvestment_medium_riskhard"
OVERLAY_DIR = ROOT / "experiments/archive" / 'frontier_consolidation' / "data" / "density_cleanqwen_overlay_medium_riskhard"
ADJ_DIR = A01 / "data" / "compact_reinvest_adjbreak_control"
TOKENIZER = ROOT / "experiments/archive" / 'initial_model_studies' / "training" / "runs" / "fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256" / "hf_model"
OUT_JSON = ADJ_DIR / "compact_reinvest_adjbreak_semantic_and_visibility_audit.json"
NOTE = (ROOT / 'research/notes/representation_and_objectives/compact_reinvest_adjbreak_semantic_and_visibility_audit.md')
SEQ_LEN = 256
TRAIN_SHUFFLE_BASE = 82914124 + 7000 + 1000
PASSES = 10
STOP = {
    'the','a','an','and','or','but','if','then','than','that','this','these','those','to','of','in','on','for','with','as','by','from','at','into','over','under','about','between','through','during','before','after','above','below','is','are','was','were','be','been','being','has','have','had','do','does','did','not','no','it','its','their','his','her','they','them','he','she','we','you','i','there','here','which','who','whom','whose','when','where','why','how','can','could','should','would','may','might','will','shall','also','some','many','more','most','other','such','only','new','one','two','first','second'
}
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")
CAP_RE = re.compile(r"\b(?:[A-Z][A-Za-z0-9]*(?:[-'][A-Z]?[A-Za-z0-9]+)*)(?:\s+(?:[A-Z][A-Za-z0-9]*(?:[-'][A-Z]?[A-Za-z0-9]+)*))*")
NUM_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:[,.]\d+)*(?:\.\d+)?%?")


def iter_jsonl(path: pathlib.Path, limit: int | None = None):
    with path.open('r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            if line.strip():
                yield json.loads(line)


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def wc(text: str) -> int:
    return len((text or '').split())


def norm(text: str) -> str:
    return ' '.join((text or '').split())


def words(text: str, content: bool = False) -> list[str]:
    toks = [m.group(0).lower() for m in TOKEN_RE.finditer(text or '')]
    if content:
        toks = [t for t in toks if t not in STOP and not t.isdigit() and len(t) > 1]
    return toks


def ngrams(toks: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(toks[i:i+n]) for i in range(0, max(0, len(toks)-n+1))}


def jaccard(a: Iterable[Any], b: Iterable[Any]) -> float:
    A = set(a); B = set(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def recall(a: Iterable[Any], b: Iterable[Any]) -> float:
    A = set(a); B = set(b)
    if not A:
        return 1.0
    return len(A & B) / len(A)


def entities(text: str) -> set[str]:
    ents = set()
    for m in CAP_RE.finditer(text or ''):
        s = norm(m.group(0)).strip('.,;:()[]{}')
        if not s:
            continue
        if s.lower() in STOP:
            continue
        ents.add(s.lower())
    return ents


def nums(text: str) -> set[str]:
    return {m.group(0).replace(',', '').lower() for m in NUM_RE.finditer(text or '')}


def sim_metrics(source: str, rewrite: str) -> dict[str, float | int | bool]:
    sw_all = words(source, False); rw_all = words(rewrite, False)
    sw = words(source, True); rw = words(rewrite, True)
    se = entities(source); re_ = entities(rewrite)
    sn = nums(source); rn = nums(rewrite)
    return {
        'content_jaccard': jaccard(sw, rw),
        'content_source_recall': recall(sw, rw),
        'content_rewrite_recall': recall(rw, sw),
        'bigram_jaccard': jaccard(ngrams(sw_all, 2), ngrams(rw_all, 2)),
        'trigram_jaccard': jaccard(ngrams(sw_all, 3), ngrams(rw_all, 3)),
        'entity_source_recall': recall(se, re_),
        'entity_rewrite_recall': recall(re_, se),
        'number_source_recall': recall(sn, rn),
        'number_rewrite_recall': recall(rn, sn),
        'source_entities': len(se),
        'rewrite_entities': len(re_),
        'source_numbers': len(sn),
        'rewrite_numbers': len(rn),
        'exact_normalized': norm(source).lower() == norm(rewrite).lower(),
    }


def stats(vals: list[float]) -> dict[str, Any]:
    if not vals:
        return {'n': 0}
    xs = sorted(float(x) for x in vals)
    def q(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        pos = p * (len(xs)-1)
        lo = int(math.floor(pos)); hi = int(math.ceil(pos))
        if lo == hi:
            return xs[lo]
        return xs[lo]*(hi-pos)+xs[hi]*(pos-lo)
    return {'n': len(xs), 'min': xs[0], 'p05': q(0.05), 'p25': q(0.25), 'mean': sum(xs)/len(xs), 'median': q(0.5), 'p75': q(0.75), 'p95': q(0.95), 'p99': q(0.99), 'max': xs[-1], 'sum': sum(xs)}


def enc_len(tok: Any, text: str, add_special_tokens: bool = False) -> int:
    return len(tok.encode(text, add_special_tokens=add_special_tokens))


def slot_visibility(tok: Any, metas: list[dict[str, Any]], pairs: dict[str, dict[str, Any]], donor_key: str) -> dict[tuple[int,int,str], dict[str, Any]]:
    special = enc_len(tok, 'hello', True) - enc_len(tok, 'hello', False)
    cutoff = SEQ_LEN - special
    records: dict[tuple[int,int,str], dict[str, Any]] = {}
    for meta in metas:
        ridx = int(meta['row_index'])
        src_ids = [str(x) for x in (meta.get('source_pair_ids') or meta.get('pair_ids') or [])]
        donor_ids = [str(x) for x in (meta.get(donor_key) or src_ids)]
        prefix = ''
        for pos, (src_pid, donor_pid) in enumerate(zip(src_ids, donor_ids)):
            source = norm(str(pairs[src_pid]['source_text']))
            rewrite = norm(str(pairs[donor_pid]['rewrite_text']))
            unit = (source + ' ' + rewrite).strip()
            source_prefix = prefix + (' ' if prefix else '') + source
            pair_prefix = prefix + (' ' if prefix else '') + unit
            start = enc_len(tok, prefix, False) if prefix else 0
            source_end = enc_len(tok, source_prefix, False)
            pair_end = enc_len(tok, pair_prefix, False)
            visible_inside = max(0, min(pair_end, cutoff) - start)
            key = (ridx, pos, src_pid)
            records[key] = {
                'row_index': ridx,
                'position': pos,
                'source_pair_id': src_pid,
                'donor_pair_id': donor_pid,
                'source_visible': source_end <= cutoff,
                'full_visible': pair_end <= cutoff,
                'partial_visible': visible_inside > 0 and pair_end > cutoff,
                'hidden': visible_inside <= 0,
                'visible_tokens_inside_pair': visible_inside,
                'pair_tokens': pair_end - start,
                'start_token': start,
                'source_end_token': source_end,
                'pair_end_token': pair_end,
            }
            prefix = pair_prefix
    return records


def order_digest(n_rows: int, seed_base: int = TRAIN_SHUFFLE_BASE) -> dict[str, Any]:
    h = hashlib.sha256()
    samples = {}
    for pass_i in range(PASSES):
        order = list(range(n_rows))
        random.Random(seed_base + pass_i).shuffle(order)
        h.update(bytes(str(pass_i), 'ascii') + b':')
        for idx in order:
            h.update(idx.to_bytes(4, 'little', signed=False))
        h.update(b'\n')
        samples[f'pass_{pass_i}'] = {'first10': order[:10], 'last10': order[-10:]}
    return {'n_rows': n_rows, 'passes': PASSES, 'seed_base': seed_base, 'sha256': h.hexdigest(), 'samples': samples}


def main() -> None:
    from transformers import AutoTokenizer  # type: ignore
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True, use_fast=True)
    pairs = {str(p['pair_id']): p for p in iter_jsonl(DENSITY_DIR / 'selected_compact_reinvest_pairs.jsonl')}
    adj = read_json(ADJ_DIR / 'compact_reinvest_adjbreak_control_measurement.json')
    changed_rows = int(adj['construction']['changed_rows'])
    original_metas_raw = list(iter_jsonl(OVERLAY_DIR / 'cleanqwen_fineweb_compact_view_reinvest_changed_block_rows_meta.jsonl', limit=changed_rows))
    original_metas = []
    for m in original_metas_raw:
        original_metas.append({'row_index': m['row_index'], 'source_pair_ids': m.get('pair_ids') or []})
    adj_metas = list(iter_jsonl(ADJ_DIR / 'cleanqwen_fineweb_compact_view_reinvest_adjbreak_changed_block_rows_meta.jsonl', limit=changed_rows))

    original_vis = slot_visibility(tok, original_metas, pairs, donor_key='donor_rewrite_pair_ids')
    adj_vis = slot_visibility(tok, adj_metas, pairs, donor_key='donor_rewrite_pair_ids')
    if set(original_vis) != set(adj_vis):
        raise RuntimeError('visibility key mismatch')

    sim_rows = []
    high = collections.Counter()
    exceptional = []
    for meta in adj_metas:
        src_ids = [str(x) for x in (meta.get('source_pair_ids') or [])]
        donor_ids = [str(x) for x in (meta.get('donor_rewrite_pair_ids') or [])]
        for pos, (src_pid, donor_pid) in enumerate(zip(src_ids, donor_ids)):
            src = norm(str(pairs[src_pid]['source_text']))
            own_rw = norm(str(pairs[src_pid]['rewrite_text']))
            donor_rw = norm(str(pairs[donor_pid]['rewrite_text']))
            original = sim_metrics(src, own_rw)
            donor = sim_metrics(src, donor_rw)
            row = {
                'row_index': int(meta['row_index']), 'position': pos, 'source_pair_id': src_pid, 'donor_pair_id': donor_pid,
                'same_pair': src_pid == donor_pid,
                'same_doc': str(pairs[src_pid].get('doc_id') or '') == str(pairs[donor_pid].get('doc_id') or '') and bool(str(pairs[src_pid].get('doc_id') or '')),
                'source_domains': pairs[src_pid].get('domain_hits') or ['no_domain'],
                'donor_domains': pairs[donor_pid].get('domain_hits') or ['no_domain'],
                'original': original,
                'donor': donor,
                'content_recall_drop': float(original['content_source_recall']) - float(donor['content_source_recall']),
                'entity_recall_drop': float(original['entity_source_recall']) - float(donor['entity_source_recall']),
                'number_recall_drop': float(original['number_source_recall']) - float(donor['number_source_recall']),
            }
            sim_rows.append(row)
            if donor['content_source_recall'] >= 0.60: high['donor_content_recall_ge_0p60'] += 1
            if donor['content_jaccard'] >= 0.50: high['donor_content_jaccard_ge_0p50'] += 1
            if donor['bigram_jaccard'] >= 0.30: high['donor_bigram_jaccard_ge_0p30'] += 1
            if donor['trigram_jaccard'] >= 0.20: high['donor_trigram_jaccard_ge_0p20'] += 1
            if donor['entity_source_recall'] >= 0.75 and donor['source_entities'] > 0: high['donor_entity_recall_ge_0p75_with_entities'] += 1
            if donor['number_source_recall'] >= 0.75 and donor['source_numbers'] > 0: high['donor_number_recall_ge_0p75_with_numbers'] += 1
            if src_pid == donor_pid or row['same_doc'] or donor['content_source_recall'] >= 0.60 or donor['bigram_jaccard'] >= 0.30 or donor['trigram_jaccard'] >= 0.20:
                exceptional.append(row)

    trans = collections.Counter()
    visible_frac_delta = []
    full_only_original = []
    full_only_adj = []
    for key in sorted(original_vis):
        o = original_vis[key]; a = adj_vis[key]
        if o['full_visible'] and a['full_visible']:
            trans['full_both'] += 1
        elif o['full_visible'] and not a['full_visible']:
            trans['full_only_original'] += 1; full_only_original.append({'key': key, 'original': o, 'adjbreak': a})
        elif a['full_visible'] and not o['full_visible']:
            trans['full_only_adjbreak'] += 1; full_only_adj.append({'key': key, 'original': o, 'adjbreak': a})
        elif (o['partial_visible'] or o['source_visible']) and (a['partial_visible'] or a['source_visible']):
            trans['nonfull_visible_both'] += 1
        else:
            trans['hidden_or_source_lost_both'] += 1
        ov = o['visible_tokens_inside_pair'] / max(1, o['pair_tokens'])
        av = a['visible_tokens_inside_pair'] / max(1, a['pair_tokens'])
        visible_frac_delta.append(av - ov)

    metric_names = ['content_jaccard','content_source_recall','content_rewrite_recall','bigram_jaccard','trigram_jaccard','entity_source_recall','entity_rewrite_recall','number_source_recall','number_rewrite_recall']
    semantic_summary = {}
    for name in metric_names:
        semantic_summary[name] = {
            'original': stats([float(r['original'][name]) for r in sim_rows]),
            'adjbreak_donor': stats([float(r['donor'][name]) for r in sim_rows]),
            'delta_donor_minus_original': stats([float(r['donor'][name]) - float(r['original'][name]) for r in sim_rows]),
        }

    def project_exception(r: dict[str, Any]) -> dict[str, Any]:
        return {
            'row_index': r['row_index'], 'position': r['position'], 'source_pair_id': r['source_pair_id'], 'donor_pair_id': r['donor_pair_id'],
            'same_pair': r['same_pair'], 'same_doc': r['same_doc'],
            'source_domains': r['source_domains'], 'donor_domains': r['donor_domains'],
            'donor_content_recall': r['donor']['content_source_recall'],
            'donor_content_jaccard': r['donor']['content_jaccard'],
            'donor_bigram_jaccard': r['donor']['bigram_jaccard'],
            'donor_trigram_jaccard': r['donor']['trigram_jaccard'],
            'donor_entity_source_recall': r['donor']['entity_source_recall'],
            'donor_number_source_recall': r['donor']['number_source_recall'],
            'source_text': pairs[r['source_pair_id']]['source_text'],
            'donor_rewrite_text': pairs[r['donor_pair_id']]['rewrite_text'],
            'original_rewrite_text': pairs[r['source_pair_id']]['rewrite_text'],
        }

    exceptional_sorted = sorted(exceptional, key=lambda r: (not r['same_pair'], not r['same_doc'], -float(r['donor']['content_source_recall']), -float(r['donor']['bigram_jaccard']), r['row_index']))
    original_pool_rows = sum(1 for _ in iter_jsonl(OVERLAY_DIR / 'cleanqwen_fineweb_compact_view_reinvest_10M.jsonl'))
    adj_pool_rows = sum(1 for _ in iter_jsonl(ADJ_DIR / 'cleanqwen_fineweb_compact_view_reinvest_adjbreak_10M.jsonl'))

    payload = {
        'status': 'COMPACT_REINVEST_ADJBREAK_SEMANTIC_AND_VISIBILITY_AUDIT',
        'scientific_purpose': 'Check whether the adjacency-broken control actually breaks semantic pair correspondence while preserving trainer-relevant exposure geometry.',
        'inputs': {
            'adjbreak_measurement': str(ADJ_DIR / 'compact_reinvest_adjbreak_control_measurement.json'),
            'adjbreak_pool': str(ADJ_DIR / 'cleanqwen_fineweb_compact_view_reinvest_adjbreak_10M.jsonl'),
            'adjbreak_meta': str(ADJ_DIR / 'cleanqwen_fineweb_compact_view_reinvest_adjbreak_changed_block_rows_meta.jsonl'),
            'original_meta': str(OVERLAY_DIR / 'cleanqwen_fineweb_compact_view_reinvest_changed_block_rows_meta.jsonl'),
            'pairs': str(DENSITY_DIR / 'selected_compact_reinvest_pairs.jsonl'),
            'tokenizer': str(TOKENIZER),
        },
        'counts': {'pairs': len(sim_rows), 'changed_rows': changed_rows, 'original_pool_rows': original_pool_rows, 'adjbreak_pool_rows': adj_pool_rows},
        'semantic_similarity_summary': semantic_summary,
        'semantic_high_similarity_counts': dict(high),
        'fixed_and_exception_counts': {
            'same_pair': sum(1 for r in sim_rows if r['same_pair']),
            'same_doc': sum(1 for r in sim_rows if r['same_doc']),
            'domain_no_intersection': sum(1 for r in sim_rows if not (set(r['source_domains']) & set(r['donor_domains']))),
            'donor_content_recall_ge_original_content_recall': sum(1 for r in sim_rows if float(r['donor']['content_source_recall']) >= float(r['original']['content_source_recall'])),
            'donor_bigram_ge_original_bigram': sum(1 for r in sim_rows if float(r['donor']['bigram_jaccard']) >= float(r['original']['bigram_jaccard'])),
        },
        'visibility_transition': {
            **dict(trans),
            'visible_pair_fraction_delta_stats': stats(visible_frac_delta),
            'full_only_original_examples': full_only_original[:10],
            'full_only_adjbreak_examples': full_only_adj[:10],
        },
        'training_order_digest': {
            'adjbreak_future_order': order_digest(adj_pool_rows),
            'matches_a02_rowholdout_formula': True,
            'formula': 'random.Random(82914124 + 7000 + 1000 + pass_i).shuffle(range(n_rows))',
        },
        'exception_examples': [project_exception(r) for r in exceptional_sorted[:30]],
        'scientific_reading': {
            'main': 'The control breaks exact source-own-rewrite identity almost completely and sharply lowers lexical semantic overlap while preserving source/rewrite marginals, word geometry, and aggregate token geometry.',
            'caution': 'A few same-pair leftovers remain because exact rewrite-word-length derangement is impossible for singleton lengths without changing word-count geometry; the control tests pair-specific correspondence, not pure semantics alone.',
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    sem = payload['semantic_similarity_summary']
    vis = payload['visibility_transition']
    note = f"""# research adjbreak semantic and visibility audit

## Semantic break strength

Pairs audited: {len(sim_rows)}.

- original source-own-rewrite content recall mean/median: {sem['content_source_recall']['original']['mean']:.4f} / {sem['content_source_recall']['original']['median']:.4f}
- adjbreak source-donor content recall mean/median/p95: {sem['content_source_recall']['adjbreak_donor']['mean']:.4f} / {sem['content_source_recall']['adjbreak_donor']['median']:.4f} / {sem['content_source_recall']['adjbreak_donor']['p95']:.4f}
- original content Jaccard mean/median: {sem['content_jaccard']['original']['mean']:.4f} / {sem['content_jaccard']['original']['median']:.4f}
- adjbreak content Jaccard mean/median/p95: {sem['content_jaccard']['adjbreak_donor']['mean']:.4f} / {sem['content_jaccard']['adjbreak_donor']['median']:.4f} / {sem['content_jaccard']['adjbreak_donor']['p95']:.4f}
- adjbreak bigram/trigram Jaccard mean: {sem['bigram_jaccard']['adjbreak_donor']['mean']:.4f} / {sem['trigram_jaccard']['adjbreak_donor']['mean']:.4f}
- high donor-similarity counts: {dict(high)}

## Visibility transitions

- full in both: {vis.get('full_both',0)}
- full only original: {vis.get('full_only_original',0)}
- full only adjbreak: {vis.get('full_only_adjbreak',0)}
- nonfull visible both: {vis.get('nonfull_visible_both',0)}
- visible-token fraction delta mean/p95/max: {vis['visible_pair_fraction_delta_stats']['mean']:.6f} / {vis['visible_pair_fraction_delta_stats']['p95']:.6f} / {vis['visible_pair_fraction_delta_stats']['max']:.6f}

## Training-order digest

A future adjbreak 100M training file should use the same row-index order formula as the A02 row-holdout materializer: `random.Random(82914124 + 7000 + 1000 + pass_i).shuffle(range(n_rows))`. Digest for {adj_pool_rows} rows and 10 passes: `{payload['training_order_digest']['adjbreak_future_order']['sha256']}`.

## Interpretation

The control is now better supported as a source-own-rewrite correspondence intervention: it preserves marginals and row geometry, keeps aggregate token exposure matched, and the donor rewrite usually has much lower content overlap with the assigned source than the true rewrite. It remains a mechanism control to run only after the already-running full and seed endpoint results justify spending GPU time.

JSON: `{OUT_JSON}`
"""
    NOTE.write_text(note, encoding='utf-8')
    print(json.dumps({
        'status': payload['status'],
        'json': str(OUT_JSON),
        'note': str(NOTE),
        'pairs': len(sim_rows),
        'same_pair': payload['fixed_and_exception_counts']['same_pair'],
        'same_doc': payload['fixed_and_exception_counts']['same_doc'],
        'donor_content_recall_mean': sem['content_source_recall']['adjbreak_donor']['mean'],
        'donor_content_recall_p95': sem['content_source_recall']['adjbreak_donor']['p95'],
        'full_only_original': vis.get('full_only_original', 0),
        'full_only_adjbreak': vis.get('full_only_adjbreak', 0),
        'order_digest': payload['training_order_digest']['adjbreak_future_order']['sha256'],
    }, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
