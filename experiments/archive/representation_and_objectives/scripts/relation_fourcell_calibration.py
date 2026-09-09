#!/usr/bin/env python3
"""research no-update four-cell calibration for active relation pairs.

The research active-token pair pool is only worth training on if it is connected to
known conditional-interaction failures rather than just attested local coherence.
This script scores frozen MLM checkpoints on a two-context/two-target interaction:

    M = s(Ca, Ta) + s(Cb, Tb) - s(Ca, Tb) - s(Cb, Ta)

where Ca/Cb are the original contexts with their own target word-group masked, and
Ta/Tb are matched same-length target token sequences. The score s is the mean
masked-token log-probability at the saved target positions. Because pairs require
the same target token length/pattern/frequency stratum, target length main effects
are tightly controlled; the four-cell contrast also cancels context and target main
effects inside each pair.

Two deterministic nulls test whether any arm separation is real interaction:
  - target_permuted: keep contexts Ca/Cb but replace (Ta,Tb) by another pair's
    target pair from the same stratum.
  - context_permuted: keep targets Ta/Tb but replace (Ca,Cb) by another pair's
    context pair from the same stratum.

No optimizer step, data mutation, or training is performed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, DebertaV2ForMaskedLM

USER_ROOT = Path('.').resolve()
COMPACT_EXPERIENCE_SCRIPTS = USER_ROOT / 'experiments/archive/compact_experience/scripts'
if str(COMPACT_EXPERIENCE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(COMPACT_EXPERIENCE_SCRIPTS))
import masking_curriculum_trainer as base  # noqa: E402

DEFAULT_PAIR_POOL = USER_ROOT / 'experiments/archive/representation_and_objectives/data/relation_active_pair_funnel/active_relation_pair_pool.jsonl'
DEFAULT_TOKENIZER = USER_ROOT / 'experiments/archive/representation_and_objectives/data/shared_tokenizer/shared_16k_tokenizer'
DEFAULT_OUT = USER_ROOT / 'experiments/archive/representation_and_objectives/data/relation_fourcell_calibration'
DEFAULT_NOTE = USER_ROOT / 'research/notes/representation_and_objectives/relation_fourcell_calibration.md'
DEFAULT_ARMS = {
    'compact': USER_ROOT / 'experiments/archive/frontier_consolidation/training/runs/fw_compact_view_shared16k_seed43022/hf_model/chck_100M',
    'rowblock': USER_ROOT / 'experiments/archive/frontier_consolidation/training/runs/fw_source_breadth_shared16k_seed43022/hf_model/chck_100M',
    'interleaved': USER_ROOT / 'experiments/archive/representation_and_objectives/training/runs/fw_source_breadth_interleaved_wholesentence_fullbatch_shared16k_seed43022/hf_model/chck_100M',
}
FAMILIES = ['causal_connector', 'spatial', 'temporal', 'negation', 'physical_change', 'comparative']


def now_utc() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def stable_u(seed: int, *items: Any) -> float:
    h = hashlib.blake2b(digest_size=8)
    h.update(str(seed).encode())
    for it in items:
        h.update(b'\0')
        h.update(str(it).encode('utf-8', errors='ignore'))
    return int.from_bytes(h.digest(), 'big') / float(2**64)


def qstats(vals: Iterable[float]) -> dict[str, Any]:
    xs = sorted(float(v) for v in vals if math.isfinite(float(v)))
    if not xs:
        return {'n': 0}
    arr = np.asarray(xs, dtype=np.float64)
    def q(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        idx = p * (len(xs) - 1)
        lo = int(math.floor(idx)); hi = int(math.ceil(idx))
        if lo == hi:
            return xs[lo]
        return xs[lo] * (hi - idx) + xs[hi] * (idx - lo)
    return {
        'n': len(xs), 'mean': float(arr.mean()), 'std': float(arr.std()),
        'stderr': float(arr.std() / math.sqrt(len(xs))) if len(xs) else None,
        'median': q(0.5), 'p05': q(0.05), 'p25': q(0.25), 'p75': q(0.75),
        'p95': q(0.95), 'min': xs[0], 'max': xs[-1],
        'success_gt0': float((arr > 0).mean()),
    }


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    x = np.asarray(xs, dtype=np.float64); y = np.asarray(ys, dtype=np.float64)
    sx = float(x.std()); sy = float(y.std())
    if sx <= 1e-12 or sy <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def load_pairs(path: Path) -> list[dict[str, Any]]:
    pairs = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    return pairs


def pair_key(p: dict[str, Any], *, loose: bool = False) -> tuple[Any, ...]:
    if loose:
        return (p['category'], p['target_class'], p['target_token_len'], p['target_token_pattern'])
    return (p['category'], p['target_class'], p['target_token_len'], p['target_token_pattern'], p['target_freq_bin_a'], p['target_freq_bin_b'], p['distance_bin_a'], p['distance_bin_b'])


def select_pairs(pairs: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    families = set(args.categories) if args.categories else set(FAMILIES)
    pairs = [p for p in pairs if str(p.get('category')) in families]
    if args.max_pairs_per_category <= 0 and args.max_pairs <= 0:
        return pairs
    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in pairs:
        by_cat[str(p.get('category'))].append(p)
    selected = []
    for cat in sorted(by_cat):
        xs = sorted(by_cat[cat], key=lambda p: stable_u(args.seed, 'sample', cat, p['pair_id']))
        cap = args.max_pairs_per_category if args.max_pairs_per_category > 0 else len(xs)
        selected.extend(xs[:cap])
    if args.max_pairs > 0 and len(selected) > args.max_pairs:
        selected = sorted(selected, key=lambda p: stable_u(args.seed, 'global_sample', p['pair_id']))[:args.max_pairs]
    selected = sorted(selected, key=lambda p: (str(p.get('category')), int(p.get('match_level', 99)), str(p.get('pair_id'))))
    return selected


def assign_donors(pairs: list[dict[str, Any]], seed: int) -> tuple[dict[str, str], dict[str, Any]]:
    """Assign one null donor pair to each pair, preferring exact strata."""
    id_to_pair = {p['pair_id']: p for p in pairs}
    assignments: dict[str, str] = {}
    levels = [False, True]
    counts = Counter(); failed = 0
    remaining = set(id_to_pair)
    for loose in levels:
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for pid in sorted(remaining):
            groups[pair_key(id_to_pair[pid], loose=loose)].append(id_to_pair[pid])
        newly = set()
        for key, group in groups.items():
            if len(group) < 2:
                continue
            ordered = sorted(group, key=lambda p: stable_u(seed, 'donor_order', loose, p['pair_id']))
            n = len(ordered)
            group_newly: set[str] = set()
            for pos, p in enumerate(ordered):
                if p['pair_id'] not in remaining:
                    continue
                donor = None
                for shift in range(1, n):
                    cand = ordered[(pos + shift) % n]
                    if cand['pair_id'] == p['pair_id']:
                        continue
                    # Avoid reusing an identical target pair or exactly same row pair.
                    if {cand['target_norm_a'], cand['target_norm_b']} == {p['target_norm_a'], p['target_norm_b']}:
                        continue
                    if {cand['row_a'], cand['row_b']} & {p['row_a'], p['row_b']}:
                        continue
                    if int(cand['target_token_len']) != int(p['target_token_len']):
                        continue
                    donor = cand; break
                if donor is not None:
                    assignments[p['pair_id']] = donor['pair_id']; newly.add(p['pair_id']); group_newly.add(p['pair_id'])
            counts['loose' if loose else 'exact'] += len(group_newly)
        remaining -= newly
    failed = len(remaining)
    donor_ids = set(assignments.values())
    stats = {'pairs': len(pairs), 'assigned': len(assignments), 'failed': failed, 'donor_unique': len(donor_ids), 'level_counts': dict(counts)}
    return assignments, stats


def build_context(tokenizer: Any, pair: dict[str, Any], side: str, seq_length: int, strict_verify: bool = True) -> dict[str, Any]:
    text = pair[f'text_{side}']
    positions = [int(x) for x in pair[f'target_positions_{side}']]
    target_ids = [int(x) for x in pair[f'target_ids_{side}']]
    if len(positions) != len(target_ids):
        raise ValueError(f'{pair["pair_id"]}:{side} positions/ids length mismatch')
    enc = tokenizer(text, add_special_tokens=False, truncation=True, max_length=seq_length, padding='max_length', return_tensors='pt')
    ids = enc['input_ids'].squeeze(0).tolist()
    attn = enc['attention_mask'].squeeze(0).tolist()
    if any(p < 0 or p >= seq_length or attn[p] == 0 for p in positions):
        raise ValueError(f'{pair["pair_id"]}:{side} target position not active')
    actual = [int(ids[p]) for p in positions]
    if strict_verify and actual != target_ids:
        raise ValueError(f'{pair["pair_id"]}:{side} target ids mismatch actual={actual} saved={target_ids}')
    mask_id = int(tokenizer.mask_token_id)
    for p in positions:
        ids[p] = mask_id
    return {'ids': ids, 'attn': attn, 'positions': positions, 'own_target_ids': target_ids}


def precompute_contexts(pairs: list[dict[str, Any]], tokenizer: Any, seq_length: int, strict_verify: bool) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    ctx: dict[tuple[str, str], dict[str, Any]] = {}
    bad = []
    for p in pairs:
        for side in ['a', 'b']:
            try:
                ctx[(p['pair_id'], side)] = build_context(tokenizer, p, side, seq_length, strict_verify=strict_verify)
            except Exception as e:
                bad.append({'pair_id': p.get('pair_id'), 'side': side, 'error': str(e)[:300]})
    if bad and strict_verify:
        # Keep this failure loud: the pool would no longer be the exact active-token view.
        raise RuntimeError('context verification failed: ' + json.dumps(bad[:5], ensure_ascii=False))
    good_pair_ids = [p['pair_id'] for p in pairs if (p['pair_id'], 'a') in ctx and (p['pair_id'], 'b') in ctx]
    return ctx, {'input_pairs': len(pairs), 'good_pairs': len(good_pair_ids), 'bad_contexts': len(bad), 'bad_examples': bad[:20]}


def target_ids_for(pair: dict[str, Any], side: str) -> list[int]:
    return [int(x) for x in pair[f'target_ids_{side}']]


def add_view(items: list[tuple[str, str, str, list[int], list[int], list[int], list[int]]], *, pair_id: str, cell: str, ctx_rec: dict[str, Any], label_ids: list[int]) -> None:
    pos = ctx_rec['positions']
    if len(pos) != len(label_ids):
        raise ValueError(f'view {pair_id}:{cell} length mismatch pos={len(pos)} labels={len(label_ids)}')
    labels = [-100] * len(ctx_rec['ids'])
    for p, tid in zip(pos, label_ids):
        labels[p] = int(tid)
    items.append((pair_id, cell, '', list(ctx_rec['ids']), list(ctx_rec['attn']), labels, pos))


def build_views(pairs: list[dict[str, Any]], contexts: dict[tuple[str, str], dict[str, Any]], donor: dict[str, str], id_to_pair: dict[str, dict[str, Any]]) -> tuple[list[tuple[str, str, str, list[int], list[int], list[int], list[int]]], dict[str, Any]]:
    items = []
    skipped = Counter()
    for p in pairs:
        pid = p['pair_id']
        if (pid, 'a') not in contexts or (pid, 'b') not in contexts:
            skipped['missing_context'] += 1; continue
        if pid not in donor:
            skipped['missing_donor'] += 1; continue
        d = id_to_pair[donor[pid]]
        # True four cells.
        add_view(items, pair_id=pid, cell='true_s_aa', ctx_rec=contexts[(pid, 'a')], label_ids=target_ids_for(p, 'a'))
        add_view(items, pair_id=pid, cell='true_s_ab', ctx_rec=contexts[(pid, 'a')], label_ids=target_ids_for(p, 'b'))
        add_view(items, pair_id=pid, cell='true_s_bb', ctx_rec=contexts[(pid, 'b')], label_ids=target_ids_for(p, 'b'))
        add_view(items, pair_id=pid, cell='true_s_ba', ctx_rec=contexts[(pid, 'b')], label_ids=target_ids_for(p, 'a'))
        # Target-permuted null: own contexts, donor targets.
        add_view(items, pair_id=pid, cell='tperm_s_aa', ctx_rec=contexts[(pid, 'a')], label_ids=target_ids_for(d, 'a'))
        add_view(items, pair_id=pid, cell='tperm_s_ab', ctx_rec=contexts[(pid, 'a')], label_ids=target_ids_for(d, 'b'))
        add_view(items, pair_id=pid, cell='tperm_s_bb', ctx_rec=contexts[(pid, 'b')], label_ids=target_ids_for(d, 'b'))
        add_view(items, pair_id=pid, cell='tperm_s_ba', ctx_rec=contexts[(pid, 'b')], label_ids=target_ids_for(d, 'a'))
        # Context-permuted null: donor contexts, own targets.
        if (d['pair_id'], 'a') not in contexts or (d['pair_id'], 'b') not in contexts:
            skipped['missing_donor_context'] += 1; continue
        add_view(items, pair_id=pid, cell='cperm_s_aa', ctx_rec=contexts[(d['pair_id'], 'a')], label_ids=target_ids_for(p, 'a'))
        add_view(items, pair_id=pid, cell='cperm_s_ab', ctx_rec=contexts[(d['pair_id'], 'a')], label_ids=target_ids_for(p, 'b'))
        add_view(items, pair_id=pid, cell='cperm_s_bb', ctx_rec=contexts[(d['pair_id'], 'b')], label_ids=target_ids_for(p, 'b'))
        add_view(items, pair_id=pid, cell='cperm_s_ba', ctx_rec=contexts[(d['pair_id'], 'b')], label_ids=target_ids_for(p, 'a'))
    stats = {'views': len(items), 'skipped': dict(skipped)}
    return items, stats


def score_views(model: torch.nn.Module, items: list[tuple[str, str, str, list[int], list[int], list[int], list[int]]], *, device: torch.device, batch_size: int) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = defaultdict(dict)
    model.eval()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            chunk = items[start:start+batch_size]
            ids = torch.tensor([x[3] for x in chunk], dtype=torch.long, device=device)
            attn = torch.tensor([x[4] for x in chunk], dtype=torch.long, device=device)
            labels = torch.tensor([x[5] for x in chunk], dtype=torch.long, device=device)
            out = model(input_ids=ids, attention_mask=attn)
            logp = F.log_softmax(out.logits.float(), dim=-1)
            for r, (pid, cell, _unused, _ids, _attn, _labels, _pos) in enumerate(chunk):
                pos = (labels[r] != -100).nonzero(as_tuple=False).view(-1)
                tg = labels[r].index_select(0, pos)
                vals = logp[r].index_select(0, pos).gather(1, tg.view(-1, 1)).view(-1)
                scores[pid][cell] = float(vals.mean().detach().cpu().item())
            del ids, attn, labels, out, logp
            if start == 0 or (start // batch_size) % 50 == 0:
                print(json.dumps({'event': 'score_progress', 'views_done': min(start+len(chunk), len(items)), 'views_total': len(items)}), flush=True)
    return scores


def margins_from_cells(cell: dict[str, float], prefix: str) -> dict[str, float]:
    saa = cell[f'{prefix}_s_aa']; sab = cell[f'{prefix}_s_ab']; sbb = cell[f'{prefix}_s_bb']; sba = cell[f'{prefix}_s_ba']
    ma = saa - sab
    mb = sbb - sba
    return {f'{prefix}_m': ma + mb, f'{prefix}_m_a': ma, f'{prefix}_m_b': mb, f'{prefix}_s_aa': saa, f'{prefix}_s_ab': sab, f'{prefix}_s_bb': sbb, f'{prefix}_s_ba': sba}


def combine_scores(arm: str, pairs: list[dict[str, Any]], scores: dict[str, dict[str, float]], donor: dict[str, str]) -> list[dict[str, Any]]:
    rows = []
    for p in pairs:
        pid = p['pair_id']
        cell = scores.get(pid, {})
        need = [f'{pref}_s_{xy}' for pref in ['true', 'tperm', 'cperm'] for xy in ['aa', 'ab', 'bb', 'ba']]
        if not all(k in cell for k in need):
            continue
        rec = {
            'arm': arm, 'pair_id': pid, 'donor_pair_id': donor.get(pid),
            'category': p['category'], 'match_level': int(p.get('match_level', -1)),
            'pair_cost': float(p.get('pair_cost', 0.0)),
            'target_class': p.get('target_class'), 'target_token_len': int(p.get('target_token_len', -1)),
            'target_token_pattern': p.get('target_token_pattern'),
            'target_freq_bin_a': p.get('target_freq_bin_a'), 'target_freq_bin_b': p.get('target_freq_bin_b'),
            'distance_bin_a': p.get('distance_bin_a'), 'distance_bin_b': p.get('distance_bin_b'),
            'target_norm_a': p.get('target_norm_a'), 'target_norm_b': p.get('target_norm_b'),
            'pivot_norm_a': p.get('pivot_norm_a'), 'pivot_norm_b': p.get('pivot_norm_b'),
            'row_a': p.get('row_a'), 'row_b': p.get('row_b'),
        }
        rec.update(margins_from_cells(cell, 'true'))
        rec.update(margins_from_cells(cell, 'tperm'))
        rec.update(margins_from_cells(cell, 'cperm'))
        rows.append(rec)
    return rows


def summarize_rows(rows_by_arm: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    metrics = ['true_m', 'true_m_a', 'true_m_b', 'tperm_m', 'cperm_m']
    summary: dict[str, Any] = {'arms': {}, 'paired_deltas': {}, 'correlations': {}, 'arm_order': {}}
    for arm, rows in rows_by_arm.items():
        arm_sum: dict[str, Any] = {'n_pairs': len(rows), 'overall': {}, 'by_category': {}}
        for m in metrics:
            arm_sum['overall'][m] = qstats([r[m] for r in rows])
        for cat in sorted(set(str(r['category']) for r in rows)):
            cr = [r for r in rows if str(r['category']) == cat]
            arm_sum['by_category'][cat] = {m: qstats([r[m] for r in cr]) for m in metrics}
            arm_sum['by_category'][cat]['n_pairs'] = len(cr)
        summary['arms'][arm] = arm_sum
    for m in ['true_m', 'tperm_m', 'cperm_m']:
        means = {arm: summary['arms'][arm]['overall'][m]['mean'] for arm in rows_by_arm if summary['arms'][arm]['overall'][m].get('n', 0)}
        summary['arm_order'][m] = {'means': means, 'rank_high_to_low': sorted(means, key=lambda a: means[a], reverse=True), 'range': (max(means.values()) - min(means.values()) if means else None)}
        # category arm order
        bycat = {}
        cats = sorted({str(r['category']) for rows in rows_by_arm.values() for r in rows})
        for cat in cats:
            cm = {}
            for arm in rows_by_arm:
                item = summary['arms'][arm]['by_category'].get(cat, {}).get(m, {})
                if item.get('n', 0):
                    cm[arm] = item.get('mean')
            if cm:
                bycat[cat] = {'means': cm, 'rank_high_to_low': sorted(cm, key=lambda a: cm[a], reverse=True), 'range': max(cm.values()) - min(cm.values())}
        summary['arm_order'][m]['by_category'] = bycat
    # Paired deltas on common pair IDs for each arm pair.
    arms = sorted(rows_by_arm)
    rowdict = {arm: {r['pair_id']: r for r in rows} for arm, rows in rows_by_arm.items()}
    for i, a in enumerate(arms):
        for b in arms[i+1:]:
            common = sorted(set(rowdict[a]) & set(rowdict[b]))
            key = f'{a}_minus_{b}'
            dsum: dict[str, Any] = {'n_common': len(common), 'overall': {}, 'by_category': {}}
            for m in ['true_m', 'tperm_m', 'cperm_m']:
                vals = [rowdict[a][pid][m] - rowdict[b][pid][m] for pid in common]
                dsum['overall'][m] = qstats(vals)
            cats = sorted({str(rowdict[a][pid]['category']) for pid in common})
            for cat in cats:
                pids = [pid for pid in common if str(rowdict[a][pid]['category']) == cat]
                dsum['by_category'][cat] = {'n_pairs': len(pids)}
                for m in ['true_m', 'tperm_m', 'cperm_m']:
                    dsum['by_category'][cat][m] = qstats([rowdict[a][pid][m] - rowdict[b][pid][m] for pid in pids])
            summary['paired_deltas'][key] = dsum
    # Event-level correlations of true margins across arms.
    for i, a in enumerate(arms):
        for b in arms[i+1:]:
            common = sorted(set(rowdict[a]) & set(rowdict[b]))
            summary['correlations'][f'{a}__{b}'] = {
                m: pearson([rowdict[a][pid][m] for pid in common], [rowdict[b][pid][m] for pid in common])
                for m in ['true_m', 'tperm_m', 'cperm_m']
            }
    # Compact-vs-relation expected contrasts in human-readable orientation.
    if all(a in rows_by_arm for a in ['compact', 'rowblock', 'interleaved']):
        rd = rowdict
        common = sorted(set(rd['compact']) & set(rd['rowblock']) & set(rd['interleaved']))
        rel = {}
        for hi, lo in [('rowblock','compact'), ('interleaved','compact'), ('rowblock','interleaved')]:
            rel[f'{hi}_minus_{lo}'] = {}
            for m in ['true_m','tperm_m','cperm_m']:
                rel[f'{hi}_minus_{lo}'][m] = qstats([rd[hi][pid][m] - rd[lo][pid][m] for pid in common])
            rel[f'{hi}_minus_{lo}']['by_category'] = {}
            for cat in sorted({rd['compact'][pid]['category'] for pid in common}):
                pids = [pid for pid in common if rd['compact'][pid]['category'] == cat]
                rel[f'{hi}_minus_{lo}']['by_category'][cat] = {
                    m: qstats([rd[hi][pid][m] - rd[lo][pid][m] for pid in pids]) for m in ['true_m','tperm_m','cperm_m']
                }
                rel[f'{hi}_minus_{lo}']['by_category'][cat]['n_pairs'] = len(pids)
        summary['expected_relation_oriented_deltas'] = {'common_pairs': len(common), **rel}
    return summary


def write_outputs(outdir: Path, note_path: Path, args: argparse.Namespace, pairs: list[dict[str, Any]], donor_stats: dict[str, Any], ctx_stats: dict[str, Any], view_stats: dict[str, Any], rows_by_arm: dict[str, list[dict[str, Any]]], summary: dict[str, Any], runtime_sec: float, arm_paths: dict[str, str]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    score_path = outdir / 'fourcell_pair_scores.jsonl'
    with score_path.open('w', encoding='utf-8') as f:
        for arm in sorted(rows_by_arm):
            for r in rows_by_arm[arm]:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
    csv_path = outdir / 'fourcell_pair_scores_summary.csv'
    fields = ['arm','pair_id','category','match_level','pair_cost','target_class','target_token_len','target_freq_bin_a','target_freq_bin_b','distance_bin_a','distance_bin_b','target_norm_a','target_norm_b','pivot_norm_a','pivot_norm_b','true_m','true_m_a','true_m_b','tperm_m','cperm_m']
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for arm in sorted(rows_by_arm):
            for r in rows_by_arm[arm]:
                w.writerow({k: r.get(k) for k in fields})
    full = {
        'status': 'RELATION_FOURCELL_CALIBRATION', 'created_utc': now_utc(),
        'purpose': 'No-update four-cell test of whether research active relation-pair pool tracks known compact/rowblock/interleaved conditional-relation tradeoff and whether matched permutations erase it.',
        'inputs': {'pair_pool': str(args.pair_pool), 'tokenizer_path': str(args.tokenizer_path), 'arm_paths': arm_paths},
        'parameters': vars(args), 'selected_pair_count': len(pairs),
        'selected_pairs_by_category': dict(Counter(str(p['category']) for p in pairs)),
        'donor_stats': donor_stats, 'context_stats': ctx_stats, 'view_stats': view_stats,
        'summary': summary, 'runtime_sec': runtime_sec,
        'artifacts': {'score_jsonl': str(score_path), 'score_csv': str(csv_path), 'summary_json': str(outdir / 'fourcell_calibration_summary.json'), 'note': str(note_path)},
    }
    summary_path = outdir / 'fourcell_calibration_summary.json'
    summary_path.write_text(json.dumps(full, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    # Compact note with the decision-changing numbers.
    lines = ['# research — no-update four-cell calibration on active relation pairs', '']
    lines.append(f"Created: {full['created_utc']}  Runtime: {runtime_sec:.1f} s")
    lines.append('')
    lines.append('## What was scored')
    lines.append('For each pair, the scorer masked the saved target token positions and computed `M = s(Ca,Ta)+s(Cb,Tb)-s(Ca,Tb)-s(Cb,Ta)` with frozen checkpoint weights. Two controls used same-stratum target and context permutations. No training or optimizer update was performed.')
    lines.append('')
    lines.append(f"Selected pairs: **{len(pairs)}**; by family: `{full['selected_pairs_by_category']}`")
    lines.append(f"Donor assignment: `{donor_stats}`")
    lines.append(f"Context verification: `{ctx_stats}`")
    lines.append('')
    lines.append('## Arm means for four-cell margin')
    lines.append('| margin | compact | rowblock | interleaved | rank | range |')
    lines.append('|---|---:|---:|---:|---|---:|')
    for m in ['true_m', 'tperm_m', 'cperm_m']:
        item = summary['arm_order'].get(m, {})
        means = item.get('means', {})
        rank = ' > '.join(item.get('rank_high_to_low', []))
        lines.append(f"| {m} | {means.get('compact')} | {means.get('rowblock')} | {means.get('interleaved')} | {rank} | {item.get('range')} |")
    lines.append('')
    lines.append('## Family means for true four-cell margin')
    lines.append('| family | n | compact | rowblock | interleaved | true rank | true range | target-perm range | context-perm range |')
    lines.append('|---|---:|---:|---:|---:|---|---:|---:|---:|')
    cats = sorted(summary['arm_order'].get('true_m', {}).get('by_category', {}))
    for cat in cats:
        tm = summary['arm_order']['true_m']['by_category'][cat]
        tpm = summary['arm_order']['tperm_m']['by_category'].get(cat, {})
        cpm = summary['arm_order']['cperm_m']['by_category'].get(cat, {})
        means = tm.get('means', {})
        n = summary['arms'].get('compact', {}).get('by_category', {}).get(cat, {}).get('n_pairs')
        lines.append(f"| {cat} | {n} | {means.get('compact')} | {means.get('rowblock')} | {means.get('interleaved')} | {' > '.join(tm.get('rank_high_to_low', []))} | {tm.get('range')} | {tpm.get('range')} | {cpm.get('range')} |")
    lines.append('')
    lines.append('## Relation-oriented paired deltas')
    rel = summary.get('expected_relation_oriented_deltas', {})
    for dk in ['rowblock_minus_compact','interleaved_minus_compact','rowblock_minus_interleaved']:
        if dk not in rel: continue
        lines.append(f"### {dk}")
        for m in ['true_m','tperm_m','cperm_m']:
            st = rel[dk][m]
            lines.append(f"- {m}: mean={st.get('mean')} stderr={st.get('stderr')} success_gt0={st.get('success_gt0')} n={st.get('n')}")
        lines.append('')
    lines.append('## Files')
    lines.append(f"- JSON summary: `{summary_path}`")
    lines.append(f"- Pair scores: `{score_path}`")
    lines.append(f"- CSV: `{csv_path}`")
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'status': full['status'], 'pairs': len(pairs), 'arms': sorted(rows_by_arm), 'summary': str(summary_path), 'note': str(note_path)}, indent=2), flush=True)


def parse_arm_specs(specs: list[str]) -> dict[str, Path]:
    if not specs:
        return dict(DEFAULT_ARMS)
    out: dict[str, Path] = {}
    for s in specs:
        if '=' not in s:
            raise ValueError(f'arm spec must be name=path, got {s}')
        name, path = s.split('=', 1)
        out[name.strip()] = Path(path.strip())
    return out


def build_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair_pool', default=str(DEFAULT_PAIR_POOL))
    ap.add_argument('--tokenizer_path', default=str(DEFAULT_TOKENIZER))
    ap.add_argument('--output_dir', default=str(DEFAULT_OUT))
    ap.add_argument('--note_path', default=str(DEFAULT_NOTE))
    ap.add_argument('--arm', action='append', default=[], help='name=checkpoint_path; default compact,rowblock,interleaved 100M')
    ap.add_argument('--categories', nargs='*', default=FAMILIES)
    ap.add_argument('--seq_length', type=int, default=256)
    ap.add_argument('--view_batch_size', type=int, default=160)
    ap.add_argument('--max_pairs_per_category', type=int, default=0, help='pilot cap; 0 means all selected pairs')
    ap.add_argument('--max_pairs', type=int, default=0)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--dtype', default='bf16', choices=['fp32','fp16','bf16'])
    ap.add_argument('--seed', type=int, default=43028)
    ap.add_argument('--no_strict_verify', action='store_true')
    return ap.parse_args()


def main() -> None:
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    t0 = time.time()
    args = build_args()
    random.seed(args.seed); np.random.seed(args.seed % (2**32 - 1)); torch.manual_seed(args.seed)
    pair_path = Path(args.pair_pool); tok_path = Path(args.tokenizer_path); outdir = Path(args.output_dir); note_path = Path(args.note_path)
    if not pair_path.exists():
        raise RuntimeError(f'pair pool not found: {pair_path}')
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    if tokenizer.mask_token_id is None:
        raise RuntimeError('tokenizer has no mask_token_id')
    all_pairs = load_pairs(pair_path)
    pairs = select_pairs(all_pairs, args)
    if not pairs:
        raise RuntimeError('no pairs selected')
    donors, donor_stats = assign_donors(pairs, args.seed)
    selected_id_to_pair = {p['pair_id']: p for p in pairs}
    score_pairs = [p for p in pairs if p['pair_id'] in donors]
    needed_pair_ids = set(donors.keys()) | set(donors.values())
    context_pairs = [selected_id_to_pair[pid] for pid in sorted(needed_pair_ids) if pid in selected_id_to_pair]
    id_to_pair = selected_id_to_pair
    contexts, ctx_stats = precompute_contexts(context_pairs, tokenizer, args.seq_length, strict_verify=(not args.no_strict_verify))
    views, view_stats = build_views(score_pairs, contexts, donors, id_to_pair)
    arm_paths = parse_arm_specs(args.arm)
    for name, path in arm_paths.items():
        if not Path(path).exists():
            raise RuntimeError(f'checkpoint for arm {name} not found: {path}')
    if args.device == 'cuda' and not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    torch_dtype = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}[args.dtype]
    rows_by_arm: dict[str, list[dict[str, Any]]] = {}
    for arm, path in arm_paths.items():
        print(json.dumps({'event': 'load_model', 'arm': arm, 'checkpoint': str(path), 'device': str(device), 'dtype': args.dtype, 'score_pairs': len(score_pairs), 'context_pairs': len(context_pairs), 'views': len(views)}), flush=True)
        model = DebertaV2ForMaskedLM.from_pretrained(path, torch_dtype=torch_dtype)
        model.to(device)
        scores = score_views(model, views, device=device, batch_size=args.view_batch_size)
        rows_by_arm[arm] = combine_scores(arm, score_pairs, scores, donors)
        del model, scores
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    summary = summarize_rows(rows_by_arm)
    write_outputs(outdir, note_path, args, score_pairs, donor_stats, ctx_stats, view_stats, rows_by_arm, summary, time.time() - t0, {k: str(v) for k, v in arm_paths.items()})


if __name__ == '__main__':
    main()
