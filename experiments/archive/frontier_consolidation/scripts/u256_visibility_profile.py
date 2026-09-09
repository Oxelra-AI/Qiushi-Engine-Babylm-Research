#!/usr/bin/env python3
"""Legal16k visibility/source-tail accounting for U256.

CPU-only analysis.  It compares the research row256 training example object with
U256 faithful word-boundary chunking on exactly the legal compact-view reinvest
10M pool and 100M stream.  The purpose is to interpret U256 endpoint evidence
without substituting the legal40k profile.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

USER_ROOT = Path('.').resolve()
STUDY = USER_ROOT / 'experiments/archive/frontier_consolidation'
WORKSPACE = STUDY
SCRIPTS = WORKSPACE / 'scripts'
COMPACT_EXPERIENCE_SCRIPTS = USER_ROOT / 'experiments/archive/compact_experience/scripts'
for p in (SCRIPTS, COMPACT_EXPERIENCE_SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import masking_curriculum_trainer as base  # noqa: E402
import experience_utilization_trainer_ref as chunkbase  # noqa: E402

BASE_10M = WORKSPACE / 'data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl'
STREAM_100M = WORKSPACE / 'data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl'
TOKENIZER_DIR = WORKSPACE / 'data/compliant_tokenizer'
LOG = WORKSPACE / 'training/runs/complianttok_reinvest_seed43022_r2/training_log.jsonl'
U256_LOG_20M = WORKSPACE / 'training/runs/eu_U256_legal16k_seed43022_20M/training_log.jsonl'
U256_DRYRUN_STEP110 = WORKSPACE / 'data/eu_U256_dryrun/dryrun_step_accounting.csv'
U256_DRYRUN_FULL = WORKSPACE / 'data/experience_utilization_dryruns/U256/dryrun_step_accounting.csv'
U64_DRYRUN_FULL = WORKSPACE / 'data/experience_utilization_dryruns/U64_128_256/dryrun_step_accounting.csv'
U256_DRYRUN_METRICS_FULL = WORKSPACE / 'data/experience_utilization_dryruns/U256/dryrun_metrics.json'
U64_DRYRUN_METRICS_FULL = WORKSPACE / 'data/experience_utilization_dryruns/U64_128_256/dryrun_metrics.json'

OUT_DIR = WORKSPACE / 'data/u256_visibility_profile'
OUT_JSON = OUT_DIR / 'u256_visibility_profile.json'
SOURCE_CSV = OUT_DIR / 'recovered_suffix_by_source_legal16k.csv'
ROW_CSV = OUT_DIR / 'top_recovered_suffix_rows_legal16k.csv'
UPDATE_CSV = OUT_DIR / 'update_mass_summary_legal16k.csv'
PREFIX_BUCKET_CSV = OUT_DIR / 'row_token_length_buckets_legal16k.csv'
NOTE = (USER_ROOT / 'research/notes/frontier_consolidation/u256_visibility_profile.md')

EXPECTED_BASE_SHA = '215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23'
EXPECTED_STREAM_SHA = '3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691'
EXPECTED_TOKENIZER_SHA = '91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9'
POOL_WORDS = 10_000_000
TOTAL_WORDS = 100_000_000
SEQ_LEN = 256
STEPS_PER_EPOCH = 253
EXPECTED_EPOCHS = 10

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[\u2019'][A-Za-z0-9]+)?")
NUMBER_RE = re.compile(r"\b\d+(?:[.,:/-]\d+)*(?:%|st|nd|rd|th)?\b", re.I)
CAP_RE = re.compile(r"\b[A-Z][A-Za-z]+(?:[-'][A-Z]?[A-Za-z]+)?\b")

# Broad training-text cue lexicons, independent of evaluation item text.
ACTION = {
    'put', 'take', 'get', 'give', 'go', 'come', 'make', 'made', 'open', 'close',
    'hold', 'held', 'push', 'pull', 'move', 'moved', 'turn', 'turned', 'drop',
    'dropped', 'pick', 'picked', 'use', 'used', 'using', 'play', 'eat', 'drink',
    'wash', 'throw', 'threw', 'bring', 'brought', 'carry', 'carried', 'keep',
    'kept', 'let', 'help', 'build', 'built', 'break', 'broke', 'change', 'changed',
    'find', 'found', 'fall', 'fell', 'run', 'ran', 'walk', 'walked', 'cut', 'pour',
    'poured', 'mix', 'mixed', 'cover', 'covered', 'remove', 'removed', 'fill',
    'filled', 'leave', 'left', 'start', 'started', 'stop', 'stopped',
}
CAUSAL_TEMPORAL = {
    'because', 'cause', 'caused', 'causes', 'so', 'therefore', 'after', 'before',
    'when', 'while', 'if', 'then', 'result', 'results', 'resulted', 'became',
    'become', 'becomes', 'makes', 'made', 'prevent', 'prevents', 'allow', 'allows',
    'requires', 'required', 'during', 'until', 'since', 'leads', 'led', 'again',
    'first', 'next', 'later', 'finally', 'once', 'then', 'now', 'soon',
}
PHYSICAL_OBJECT = {
    'water', 'fire', 'box', 'ball', 'door', 'cup', 'table', 'toy', 'paper',
    'hand', 'body', 'food', 'stone', 'wood', 'glass', 'machine', 'tool', 'wheel',
    'container', 'bag', 'room', 'floor', 'wall', 'book', 'plant', 'animal', 'air',
    'light', 'heat', 'material', 'metal', 'house', 'car', 'window', 'cloth',
    'sock', 'handle', 'head', 'bottle', 'chair', 'bed', 'kitchen', 'garden',
}
SPATIAL_STATE = {
    'in', 'on', 'under', 'over', 'inside', 'outside', 'behind', 'front', 'near',
    'across', 'through', 'between', 'below', 'above', 'around', 'left', 'right',
    'top', 'bottom', 'beside', 'into', 'out', 'off', 'down', 'up', 'back', 'onto',
    'from', 'toward', 'towards', 'beside', 'along', 'against', 'within',
}
MENTAL_SOCIAL = {
    'think', 'thought', 'know', 'knew', 'want', 'wanted', 'say', 'said', 'ask',
    'asked', 'tell', 'told', 'see', 'saw', 'look', 'looked', 'feel', 'felt',
    'believe', 'learn', 'teach', 'child', 'mother', 'father', 'person', 'people',
    'friend', 'teacher', 'man', 'woman', 'boy', 'girl', 'family', 'name', 'talk',
    'talked', 'answer', 'answered', 'question', 'call', 'called', 'share', 'shared',
}
MATERIAL_PROPERTY = {
    'hot', 'cold', 'warm', 'dry', 'wet', 'hard', 'soft', 'heavy', 'light', 'rough',
    'smooth', 'sharp', 'flat', 'round', 'sticky', 'clean', 'dirty', 'broken',
    'full', 'empty', 'open', 'closed', 'wooden', 'metal', 'plastic', 'glass',
}
QUANTITATIVE = {
    'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
    'many', 'much', 'more', 'less', 'few', 'several', 'all', 'none', 'some', 'both',
    'half', 'twice', 'first', 'second', 'third', 'same', 'different', 'another',
}


def now_utc() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(USER_ROOT))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def raw_line_hash(raw: bytes) -> str:
    return hashlib.sha256(raw.rstrip(b'\r\n')).hexdigest()


def quantile(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    vals = sorted(float(x) for x in xs)
    if len(vals) == 1:
        return vals[0]
    pos = p * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def summarize(xs: list[float]) -> dict[str, Any]:
    vals = [float(x) for x in xs if x is not None]
    if not vals:
        return {'n': 0}
    mean = statistics.fmean(vals)
    sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return {
        'n': len(vals), 'sum': sum(vals), 'mean': mean, 'sd': sd,
        'min': min(vals), 'p05': quantile(vals, 0.05), 'p25': quantile(vals, 0.25),
        'p50': quantile(vals, 0.50), 'p75': quantile(vals, 0.75),
        'p95': quantile(vals, 0.95), 'max': max(vals),
        'cv': sd / mean if mean else None,
    }


def words_to_tokens(words: list[str]) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(' '.join(words))]


def word_feature_counts(words: list[str]) -> dict[str, int]:
    text = ' '.join(words)
    toks = words_to_tokens(words)
    cnt = collections.Counter(toks)
    action = sum(cnt[w] for w in ACTION)
    causal = sum(cnt[w] for w in CAUSAL_TEMPORAL)
    physical = sum(cnt[w] for w in PHYSICAL_OBJECT)
    spatial = sum(cnt[w] for w in SPATIAL_STATE)
    social = sum(cnt[w] for w in MENTAL_SOCIAL)
    material = sum(cnt[w] for w in MATERIAL_PROPERTY)
    quantitative = sum(cnt[w] for w in QUANTITATIVE) + len(NUMBER_RE.findall(text))
    caps = len(CAP_RE.findall(text))
    return {
        'token_count': len(toks),
        'action': action,
        'causal_temporal': causal,
        'physical_object': physical,
        'spatial_state': spatial,
        'mental_social': social,
        'material_property': material,
        'quantitative': quantitative,
        'numbers': len(NUMBER_RE.findall(text)),
        'capitalized': caps,
        'action_physical_rowsignal': int(action > 0 and physical > 0),
        'spatial_physical_rowsignal': int(spatial > 0 and physical > 0),
        'causal_physical_rowsignal': int(causal > 0 and physical > 0),
        'material_physical_rowsignal': int(material > 0 and physical > 0),
        'quantitative_physical_rowsignal': int(quantitative > 0 and physical > 0),
        'social_action_rowsignal': int(social > 0 and action > 0),
    }


def add_counter(dst: dict[str, int], src: dict[str, int], mul: int = 1) -> None:
    for k, v in src.items():
        dst[k] = int(dst.get(k, 0)) + int(v) * mul


def prefix_visibility(word_tokens: list[list[int]], limit: int = SEQ_LEN) -> dict[str, int]:
    cum = 0
    visible_full_words = 0
    split_word_visible = 0
    split_word_hidden_tokens = 0
    hidden_full_words = 0
    hidden_full_word_tokens = 0
    for ids in word_tokens:
        n = len(ids)
        start = cum
        end = cum + n
        if n == 0:
            if start < limit:
                visible_full_words += 1
            else:
                hidden_full_words += 1
            continue
        if end <= limit:
            visible_full_words += 1
        elif start < limit:
            split_word_visible += 1
            split_word_hidden_tokens += end - limit
        else:
            hidden_full_words += 1
            hidden_full_word_tokens += n
        cum = end
    raw_tokens = cum
    active = min(raw_tokens, limit)
    hidden_tokens = max(0, raw_tokens - limit)
    return {
        'raw_tokens': raw_tokens,
        'row256_active_tokens': active,
        'row256_visible_full_words': visible_full_words,
        'row256_boundary_split_words': split_word_visible,
        'row256_boundary_hidden_tokens': split_word_hidden_tokens,
        'row256_hidden_full_words': hidden_full_words,
        'row256_hidden_full_word_tokens': hidden_full_word_tokens,
        'row256_hidden_tokens_total': hidden_tokens,
        'row256_any_recovered_token': int(hidden_tokens > 0),
        'row256_any_hidden_full_word': int(hidden_full_words > 0),
    }


def tokenize_profile_rows(tokenizer) -> dict[str, Any]:
    row_by_hash: dict[str, dict[str, Any]] = {}
    counter: collections.Counter[str] = collections.Counter()
    source_rows: collections.Counter[str] = collections.Counter()
    source_words: collections.Counter[str] = collections.Counter()
    source_raw_tokens: collections.Counter[str] = collections.Counter()
    source_row256_active_tokens: collections.Counter[str] = collections.Counter()
    source_hidden_tokens: collections.Counter[str] = collections.Counter()
    source_hidden_words: collections.Counter[str] = collections.Counter()
    source_boundary_tokens: collections.Counter[str] = collections.Counter()
    source_features: dict[str, dict[str, int]] = collections.defaultdict(lambda: collections.defaultdict(int))  # type: ignore[assignment]
    rows = 0
    words_total = 0
    raw_tokens_total = 0
    row256_active_total = 0
    visible_full_words_total = 0
    hidden_full_words_total = 0
    hidden_tokens_total = 0
    boundary_hidden_tokens_total = 0
    boundary_split_rows = 0
    recovered_rows: list[dict[str, Any]] = []
    token_buckets: dict[str, dict[str, int]] = collections.defaultdict(lambda: collections.defaultdict(int))  # type: ignore[assignment]

    with BASE_10M.open('rb') as f:
        for row_idx, raw in enumerate(f):
            if not raw.strip():
                continue
            rec = json.loads(raw)
            h = raw_line_hash(raw)
            counter[h] += 1
            text = str(rec['text'])
            words = int(rec['words'])
            split_words = text.split()
            if len(split_words) != words:
                raise RuntimeError(f'row {row_idx} word mismatch {len(split_words)} != {words}')
            source = str(rec.get('source', ''))
            word_tokens, raw_tokens, unassigned = chunkbase.token_ids_by_whitespace_word(text, tokenizer)
            if unassigned:
                raise RuntimeError(f'row {row_idx} unassigned tokenizer offsets {unassigned}')
            if len(word_tokens) != words:
                raise RuntimeError(f'row {row_idx} tokenized word mismatch {len(word_tokens)} != {words}')
            vis = prefix_visibility(word_tokens)
            if vis['raw_tokens'] != raw_tokens:
                raise RuntimeError(f'row {row_idx} raw token mismatch')
            visible_full_words = int(vis['row256_visible_full_words'])
            split_words_count = int(vis['row256_boundary_split_words'])
            suffix_start = min(words, visible_full_words + split_words_count)
            recovered_words = split_words[suffix_start:]
            recovered_features = word_feature_counts(recovered_words) if recovered_words else {k: 0 for k in word_feature_counts([])}
            # In row256, a boundary-split word receives some visible tokens and a group label; U256 additionally
            # recovers its hidden suffix pieces.  We separately account for full hidden words and boundary pieces.
            rec_profile = {
                'hash': h,
                'row_index': row_idx,
                'source': source,
                'words': words,
                'raw_tokens': raw_tokens,
                **vis,
                'u256_recovered_tokens': int(vis['row256_hidden_tokens_total']),
                'u256_recovered_full_words': int(vis['row256_hidden_full_words']),
                'u256_recovered_boundary_tokens': int(vis['row256_boundary_hidden_tokens']),
                'recovered_word_sample': recovered_words[:50],
                'text_prefix': text[:260],
                'suffix_features': recovered_features,
            }
            row_by_hash[h] = rec_profile
            rows += 1
            words_total += words
            raw_tokens_total += raw_tokens
            row256_active_total += int(vis['row256_active_tokens'])
            visible_full_words_total += visible_full_words
            hidden_full_words_total += int(vis['row256_hidden_full_words'])
            hidden_tokens_total += int(vis['row256_hidden_tokens_total'])
            boundary_hidden_tokens_total += int(vis['row256_boundary_hidden_tokens'])
            boundary_split_rows += int(vis['row256_boundary_split_words'] > 0)
            source_rows[source] += 1
            source_words[source] += words
            source_raw_tokens[source] += raw_tokens
            source_row256_active_tokens[source] += int(vis['row256_active_tokens'])
            source_hidden_tokens[source] += int(vis['row256_hidden_tokens_total'])
            source_hidden_words[source] += int(vis['row256_hidden_full_words'])
            source_boundary_tokens[source] += int(vis['row256_boundary_hidden_tokens'])
            add_counter(source_features[source], recovered_features)
            if int(vis['row256_hidden_tokens_total']) > 0:
                cue_score = (
                    recovered_features.get('action_physical_rowsignal', 0)
                    + recovered_features.get('spatial_physical_rowsignal', 0)
                    + recovered_features.get('causal_physical_rowsignal', 0)
                    + recovered_features.get('material_physical_rowsignal', 0)
                    + recovered_features.get('quantitative_physical_rowsignal', 0)
                    + recovered_features.get('social_action_rowsignal', 0)
                    + min(4, recovered_features.get('action', 0) + recovered_features.get('causal_temporal', 0))
                    + min(3, recovered_features.get('material_property', 0) + recovered_features.get('quantitative', 0))
                    + min(2, recovered_features.get('capitalized', 0) + recovered_features.get('numbers', 0))
                )
                rr = dict(rec_profile)
                rr['cue_score'] = int(cue_score)
                recovered_rows.append(rr)
            raw_bucket = '000-128' if raw_tokens <= 128 else '129-256' if raw_tokens <= 256 else '257-384' if raw_tokens <= 384 else '385-512' if raw_tokens <= 512 else '513+'
            b = token_buckets[raw_bucket]
            b['rows'] += 1
            b['words'] += words
            b['raw_tokens'] += raw_tokens
            b['row256_active_tokens'] += int(vis['row256_active_tokens'])
            b['hidden_tokens'] += int(vis['row256_hidden_tokens_total'])
            b['hidden_full_words'] += int(vis['row256_hidden_full_words'])
            b['boundary_hidden_tokens'] += int(vis['row256_boundary_hidden_tokens'])
            if rows % 10000 == 0:
                print(json.dumps({'event': 'base_tokenized', 'rows': rows, 'words': words_total}), flush=True)

    if words_total != POOL_WORDS:
        raise RuntimeError(f'base words {words_total} != {POOL_WORDS}')
    source_table = []
    for source in sorted(source_words, key=lambda s: (-source_hidden_tokens[s], s)):
        feats = dict(source_features[source])
        source_table.append({
            'source': source,
            'rows': int(source_rows[source]),
            'words': int(source_words[source]),
            'raw_tokens': int(source_raw_tokens[source]),
            'row256_active_tokens': int(source_row256_active_tokens[source]),
            'u256_recovered_tokens': int(source_hidden_tokens[source]),
            'u256_recovered_token_fraction_of_source_raw': source_hidden_tokens[source] / max(1, source_raw_tokens[source]),
            'hidden_full_words': int(source_hidden_words[source]),
            'hidden_full_word_fraction_of_source_words': source_hidden_words[source] / max(1, source_words[source]),
            'boundary_hidden_tokens': int(source_boundary_tokens[source]),
            'suffix_action': int(feats.get('action', 0)),
            'suffix_causal_temporal': int(feats.get('causal_temporal', 0)),
            'suffix_physical_object': int(feats.get('physical_object', 0)),
            'suffix_spatial_state': int(feats.get('spatial_state', 0)),
            'suffix_mental_social': int(feats.get('mental_social', 0)),
            'suffix_material_property': int(feats.get('material_property', 0)),
            'suffix_quantitative': int(feats.get('quantitative', 0)),
            'suffix_numbers': int(feats.get('numbers', 0)),
            'suffix_capitalized': int(feats.get('capitalized', 0)),
            'suffix_action_physical_rowsignal': int(feats.get('action_physical_rowsignal', 0)),
            'suffix_spatial_physical_rowsignal': int(feats.get('spatial_physical_rowsignal', 0)),
            'suffix_causal_physical_rowsignal': int(feats.get('causal_physical_rowsignal', 0)),
            'suffix_material_physical_rowsignal': int(feats.get('material_physical_rowsignal', 0)),
            'suffix_quantitative_physical_rowsignal': int(feats.get('quantitative_physical_rowsignal', 0)),
            'suffix_social_action_rowsignal': int(feats.get('social_action_rowsignal', 0)),
        })
    bucket_rows = []
    for name in ['000-128', '129-256', '257-384', '385-512', '513+']:
        b = dict(token_buckets.get(name, {}))
        if not b:
            b = {'rows': 0, 'words': 0, 'raw_tokens': 0, 'row256_active_tokens': 0, 'hidden_tokens': 0,
                 'hidden_full_words': 0, 'boundary_hidden_tokens': 0}
        b['bucket'] = name
        b['u256_recovered_token_fraction_of_raw'] = b['hidden_tokens'] / max(1, b['raw_tokens'])
        bucket_rows.append(b)
    recovered_rows_sorted = sorted(
        recovered_rows,
        key=lambda r: (int(r.get('cue_score', 0)), int(r['u256_recovered_full_words']), int(r['u256_recovered_tokens']), int(r['raw_tokens'])),
        reverse=True,
    )
    profile = {
        'rows': rows,
        'unique_raw_line_hashes': len(counter),
        'duplicate_raw_line_hashes': sum(1 for v in counter.values() if v > 1),
        'words_total': words_total,
        'raw_tokens_total': raw_tokens_total,
        'row256_active_tokens_total': row256_active_total,
        'u256_active_tokens_total': raw_tokens_total,
        'u256_recovered_tokens_total': raw_tokens_total - row256_active_total,
        'u256_active_token_ratio_vs_row256': raw_tokens_total / max(1, row256_active_total),
        'u256_recovered_token_fraction_of_raw': (raw_tokens_total - row256_active_total) / max(1, raw_tokens_total),
        'row256_visible_full_words_total': visible_full_words_total,
        'u256_recovered_full_words_total': hidden_full_words_total,
        'u256_recovered_full_word_fraction': hidden_full_words_total / max(1, words_total),
        'row256_boundary_split_rows': boundary_split_rows,
        'u256_recovered_boundary_tokens_total': boundary_hidden_tokens_total,
        'source_table': source_table,
        'token_length_buckets': bucket_rows,
        'top_recovered_suffix_rows': [trim_recovered_row(r) for r in recovered_rows_sorted[:40]],
    }
    return {'row_by_hash': row_by_hash, 'counter': counter, 'profile': profile}


def trim_recovered_row(r: dict[str, Any]) -> dict[str, Any]:
    feats = r.get('suffix_features', {})
    return {
        'row_index': r.get('row_index'),
        'source': r.get('source'),
        'words': r.get('words'),
        'raw_tokens': r.get('raw_tokens'),
        'row256_active_tokens': r.get('row256_active_tokens'),
        'u256_recovered_tokens': r.get('u256_recovered_tokens'),
        'u256_recovered_full_words': r.get('u256_recovered_full_words'),
        'u256_recovered_boundary_tokens': r.get('u256_recovered_boundary_tokens'),
        'cue_score': r.get('cue_score', 0),
        'suffix_action': feats.get('action', 0),
        'suffix_causal_temporal': feats.get('causal_temporal', 0),
        'suffix_physical_object': feats.get('physical_object', 0),
        'suffix_spatial_state': feats.get('spatial_state', 0),
        'suffix_mental_social': feats.get('mental_social', 0),
        'suffix_material_property': feats.get('material_property', 0),
        'suffix_quantitative': feats.get('quantitative', 0),
        'suffix_numbers': feats.get('numbers', 0),
        'suffix_capitalized': feats.get('capitalized', 0),
        'recovered_word_sample': ' '.join(r.get('recovered_word_sample', [])[:40]),
        'text_prefix': r.get('text_prefix'),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open('r', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def summarize_log_updates(rows: list[dict[str, Any]], arm: str, max_steps: int | None = None) -> dict[str, Any]:
    if max_steps is not None:
        rows = rows[:max_steps]
    fields = ['batch_words', 'masked_tokens', 'effective_mask_rate']
    out = {
        'arm': arm,
        'steps': len(rows),
        'total_words': sum(int(r.get('batch_words', 0)) for r in rows),
        'total_masked_tokens': sum(int(r.get('masked_tokens', 0)) for r in rows),
        'loss_first': float(rows[0]['loss']) if rows and 'loss' in rows[0] else None,
        'loss_last': float(rows[-1]['loss']) if rows and 'loss' in rows[-1] else None,
        'field_summaries': {f: summarize([float(r[f]) for r in rows if f in r and r[f] is not None]) for f in fields},
    }
    out['masked_per_word_total'] = out['total_masked_tokens'] / max(1, out['total_words'])
    return out


def summarize_dryrun_csv(path: Path, arm: str, max_steps: int | None = None) -> dict[str, Any]:
    rows = read_csv(path)
    if max_steps is not None:
        rows = rows[:max_steps]
    for r in rows:
        for key in ['global_step', 'epoch', 'stage_length', 'step_in_epoch', 'n_chunks', 'charged_words', 'active_tokens', 'masked_tokens', 'microbatches', 'cumulative_words']:
            if key in r and r[key] != '':
                r[key] = int(r[key])  # type: ignore[assignment]
    fields = ['charged_words', 'active_tokens', 'masked_tokens', 'n_chunks', 'microbatches']
    out = {
        'arm': arm,
        'steps': len(rows),
        'total_words': sum(int(r.get('charged_words', 0)) for r in rows),
        'total_active_tokens': sum(int(r.get('active_tokens', 0)) for r in rows),
        'total_masked_tokens': sum(int(r.get('masked_tokens', 0)) for r in rows),
        'total_nominal_slots': sum(int(r.get('n_chunks', 0)) * int(r.get('stage_length', 0)) for r in rows),
        'length_counts': dict(collections.Counter(str(r.get('stage_length')) for r in rows)),
        'field_summaries': {f: summarize([float(r[f]) for r in rows if f in r]) for f in fields},
    }
    out['active_tokens_per_word_total'] = out['total_active_tokens'] / max(1, out['total_words'])
    out['masked_per_active_total'] = out['total_masked_tokens'] / max(1, out['total_active_tokens'])
    out['masked_per_word_total'] = out['total_masked_tokens'] / max(1, out['total_words'])
    out['pad_slots_total'] = out['total_nominal_slots'] - out['total_active_tokens']
    out['pad_fraction'] = out['pad_slots_total'] / max(1, out['total_nominal_slots'])
    return out


def reconstruct_row256_updates(row_by_hash: dict[str, dict[str, Any]], base_counter: collections.Counter[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    log = read_jsonl(LOG)
    updates: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    cur: list[dict[str, Any]] = []
    total_words = 0
    stream_counter: collections.Counter[str] = collections.Counter()
    with STREAM_100M.open('rb') as f:
        for raw in f:
            if not raw.strip():
                continue
            h = raw_line_hash(raw)
            stream_counter[h] += 1
            info = row_by_hash.get(h)
            if info is None:
                raise RuntimeError(f'stream row hash absent from 10M pool: {h}')
            cur.append(info)
            total_words += int(info['words'])
            if len(cur) == 256:
                flush_row_batch(cur, updates, log, mismatches)
                cur = []
    if cur:
        flush_row_batch(cur, updates, log, mismatches)
    if total_words != TOTAL_WORDS:
        raise RuntimeError(f'stream words {total_words} != {TOTAL_WORDS}')
    expected_counter = collections.Counter({k: int(v) * EXPECTED_EPOCHS for k, v in base_counter.items()})
    if stream_counter != expected_counter:
        missing = dict((expected_counter - stream_counter).most_common(5))
        extra = dict((stream_counter - expected_counter).most_common(5))
        raise RuntimeError({'stream_multiset_mismatch': True, 'missing_examples': missing, 'extra_examples': extra})
    return updates, mismatches


def flush_row_batch(cur: list[dict[str, Any]], updates: list[dict[str, Any]], log_rows: list[dict[str, Any]], mismatches: list[dict[str, Any]]) -> None:
    idx = len(updates)
    words = sum(int(r['words']) for r in cur)
    active = sum(int(r['row256_active_tokens']) for r in cur)
    recovered = sum(int(r['u256_recovered_tokens']) for r in cur)
    hidden_words = sum(int(r['u256_recovered_full_words']) for r in cur)
    boundary = sum(int(r['u256_recovered_boundary_tokens']) for r in cur)
    rec_log = log_rows[idx] if idx < len(log_rows) else {}
    if rec_log and int(rec_log.get('batch_words', -1)) != words:
        mismatches.append({'step': idx + 1, 'computed_words': words, 'log_words': int(rec_log.get('batch_words', -1))})
    updates.append({
        'arm': 'row256_log_order',
        'step': idx + 1,
        'rows_or_chunks': len(cur),
        'charged_words': words,
        'row256_active_tokens': active,
        'u256_recoverable_tokens': recovered,
        'u256_recoverable_full_words': hidden_words,
        'u256_recoverable_boundary_tokens': boundary,
        'masked_tokens': int(rec_log.get('masked_tokens')) if rec_log.get('masked_tokens') is not None else None,
        'log_loss': rec_log.get('loss'),
        'log_lr': rec_log.get('lr'),
        'cumulative_word_exposure': rec_log.get('cumulative_word_exposure'),
        'active_tokens_per_word': active / max(1, words),
        'recoverable_tokens_per_word': recovered / max(1, words),
        'masked_per_active': (int(rec_log.get('masked_tokens')) / active) if rec_log.get('masked_tokens') is not None and active else None,
    })


def summarize_reconstructed_updates(updates: list[dict[str, Any]], arm: str, max_steps: int | None = None) -> dict[str, Any]:
    if max_steps is not None:
        updates = updates[:max_steps]
    fields = ['charged_words', 'row256_active_tokens', 'u256_recoverable_tokens', 'u256_recoverable_full_words', 'u256_recoverable_boundary_tokens', 'masked_tokens', 'active_tokens_per_word', 'recoverable_tokens_per_word', 'masked_per_active']
    total_words = sum(int(r['charged_words']) for r in updates)
    total_active = sum(int(r['row256_active_tokens']) for r in updates)
    total_recovered = sum(int(r['u256_recoverable_tokens']) for r in updates)
    total_masked = sum(int(r['masked_tokens']) for r in updates if r.get('masked_tokens') is not None)
    out = {
        'arm': arm,
        'steps': len(updates),
        'total_words': total_words,
        'total_active_tokens': total_active,
        'total_u256_recoverable_tokens': total_recovered,
        'total_masked_tokens': total_masked,
        'loss_first': updates[0].get('log_loss') if updates else None,
        'loss_last': updates[-1].get('log_loss') if updates else None,
        'field_summaries': {f: summarize([float(r[f]) for r in updates if r.get(f) is not None]) for f in fields},
    }
    out['active_tokens_per_word_total'] = total_active / max(1, total_words)
    out['recoverable_tokens_per_word_total'] = total_recovered / max(1, total_words)
    out['masked_per_active_total'] = total_masked / max(1, total_active)
    out['masked_per_word_total'] = total_masked / max(1, total_words)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def flatten_update_summaries(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for s in summaries:
        base_fields = {k: v for k, v in s.items() if k not in {'field_summaries'}}
        for field, stats in s.get('field_summaries', {}).items():
            row = dict(base_fields)
            row['metric'] = field
            for k, v in stats.items():
                row[k] = v
            rows.append(row)
    return rows


def load_dryrun_metrics(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def write_note(result: dict[str, Any]) -> None:
    profile = result['row256_visibility_profile']
    comp = result['update_mass_comparison']
    src = profile['source_table']
    lines: list[str] = []
    lines.append('# research — A02 legal16k U256 visibility/source-tail profile')
    lines.append('')
    lines.append('CPU-only accounting on the exact A02 legal compact-view reinvest corpus and research legal16k tokenizer. It does not launch training or evaluation.')
    lines.append('')
    lines.append('## Row256 versus U256 exposure')
    lines.append('')
    lines.append(f"- 10M pool words: {profile['words_total']:,}; legal16k raw tokens: {profile['raw_tokens_total']:,}.")
    lines.append(f"- research row256 active tokens: {profile['row256_active_tokens_total']:,}; U256 active tokens: {profile['u256_active_tokens_total']:,}.")
    lines.append(f"- U256 recovers {profile['u256_recovered_tokens_total']:,} tokens per 10M epoch, an active-token ratio of {profile['u256_active_token_ratio_vs_row256']:.6f} and recovered fraction {profile['u256_recovered_token_fraction_of_raw']:.4%} of raw tokens.")
    lines.append(f"- Fully hidden words recovered by U256: {profile['u256_recovered_full_words_total']:,} / {profile['words_total']:,} ({profile['u256_recovered_full_word_fraction']:.4%}); boundary-hidden token pieces recovered: {profile['u256_recovered_boundary_tokens_total']:,}; boundary-split rows: {profile['row256_boundary_split_rows']:,}.")
    lines.append('')
    lines.append('This confirms U256 is a small but real visibility repair. Its 20M score movement (+0.9179 cheap7) is much larger than the +2.59% token-count change, so the endpoint question is whether the newly visible row tails change useful credit flow or only rotate competence.')
    lines.append('')
    lines.append('## Update mass and masking')
    lines.append('')
    row20 = comp['row256_first20M']
    u20 = comp['U256_actual_first20M_log']
    u20_dry = comp['U256_step110_dryrun_first20M']
    ufull = comp['U256_full100M_dryrun']
    lines.append(f"- First 20M row256 reconstructed updates: {row20['steps']} steps, {row20['total_active_tokens']:,} active tokens, {row20['total_masked_tokens']:,} masked targets, loss {row20['loss_first']:.6f}→{row20['loss_last']:.6f}.")
    lines.append(f"- First 20M U256 actual log: {u20['steps']} steps, {u20_dry['total_active_tokens']:,} active tokens from the matching dry-run accounting, {u20['total_masked_tokens']:,} realized masked targets, loss {u20['loss_first']:.6f}→{u20['loss_last']:.6f}.")
    lines.append(f"- U256/row256 first-20M active-token ratio: {comp['u25620_vs_row25620']['active_token_ratio']:.6f}; masked-target ratio: {comp['u25620_vs_row25620']['masked_target_ratio']:.6f}; masked-per-active ratio: {comp['u25620_vs_row25620']['masked_per_active_ratio']:.6f}.")
    lines.append(f"- Full U256 dry-run: {ufull['steps']} steps, {ufull['total_active_tokens']:,} active tokens, {ufull['total_masked_tokens']:,} masked targets, pad fraction {ufull['pad_fraction']:.4f}.")
    if 'U64_128_256_full100M_dryrun' in comp:
        u64 = comp['U64_128_256_full100M_dryrun']
        lines.append(f"- U64_128_256 full dry-run has the same charged/raw-token exposure but {u64['total_nominal_slots']:,} nominal slots and pad fraction {u64['pad_fraction']:.4f}; it adds short-context/order effects, so U256 remains the cleaner first endpoint.")
    lines.append('')
    lines.append('## Source-tail distribution')
    lines.append('')
    for row in src[:10]:
        lines.append(f"- {row['source']}: recovered tokens {row['u256_recovered_tokens']:,} / raw {row['raw_tokens']:,} ({row['u256_recovered_token_fraction_of_source_raw']:.3%}); hidden full words {row['hidden_full_words']:,}; suffix cues action={row['suffix_action']}, causal={row['suffix_causal_temporal']}, physical={row['suffix_physical_object']}, spatial={row['suffix_spatial_state']}, material={row['suffix_material_property']}, quantitative={row['suffix_quantitative']}, social={row['suffix_mental_social']}.")
    lines.append('')
    lines.append('The recovered suffix mass is source-concentrated rather than uniform. Later U256 endpoint interpretation should compare any EWoK material/social/quantitative or Reading movement against these recovered-tail cues, but these lexicon counts are descriptive only and must not be used as benchmark-conditioned data selection.')
    lines.append('')
    lines.append('## Files')
    lines.append('')
    lines.append(f"- JSON: `{rel(OUT_JSON)}`")
    lines.append(f"- Source CSV: `{rel(SOURCE_CSV)}`")
    lines.append(f"- Top recovered rows CSV: `{rel(ROW_CSV)}`")
    lines.append(f"- Update summary CSV: `{rel(UPDATE_CSV)}`")
    lines.append(f"- Token-length bucket CSV: `{rel(PREFIX_BUCKET_CSV)}`")
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    base_sha = sha256_file(BASE_10M)
    stream_sha = sha256_file(STREAM_100M)
    tokenizer_sha = sha256_file(TOKENIZER_DIR / 'tokenizer.json')
    if base_sha != EXPECTED_BASE_SHA:
        raise RuntimeError(f'base SHA mismatch {base_sha}')
    if stream_sha != EXPECTED_STREAM_SHA:
        raise RuntimeError(f'stream SHA mismatch {stream_sha}')
    if tokenizer_sha != EXPECTED_TOKENIZER_SHA:
        raise RuntimeError(f'tokenizer SHA mismatch {tokenizer_sha}')
    tokenizer = base.make_portable_tokenizer(str(TOKENIZER_DIR))
    profile_bundle = tokenize_profile_rows(tokenizer)
    print(json.dumps({'event': 'profile_tokenized', 'elapsed_sec': round(time.time() - t0, 1)}), flush=True)
    row_updates, row_mismatches = reconstruct_row256_updates(profile_bundle['row_by_hash'], profile_bundle['counter'])
    print(json.dumps({'event': 'row256_reconstructed', 'updates': len(row_updates), 'mismatches': len(row_mismatches), 'elapsed_sec': round(time.time() - t0, 1)}), flush=True)

    first20 = summarize_reconstructed_updates(row_updates, 'row256_first20M', max_steps=506)
    full = summarize_reconstructed_updates(row_updates, 'row256_full100M', max_steps=None)
    u256_actual20_log = summarize_log_updates(read_jsonl(U256_LOG_20M), 'U256_actual_first20M_log')
    u256_actual20_dry = summarize_dryrun_csv(U256_DRYRUN_STEP110, 'U256_step110_dryrun_first20M', max_steps=506)
    u256_full_dry = summarize_dryrun_csv(U256_DRYRUN_FULL, 'U256_full100M_dryrun') if U256_DRYRUN_FULL.exists() else summarize_dryrun_csv(U256_DRYRUN_STEP110, 'U256_step110_dryrun_available')
    update_summaries = [first20, u256_actual20_log, u256_actual20_dry, full, u256_full_dry]
    comp: dict[str, Any] = {
        'row256_first20M': first20,
        'row256_full100M': full,
        'U256_actual_first20M_log': u256_actual20_log,
        'U256_step110_dryrun_first20M': u256_actual20_dry,
        'U256_full100M_dryrun': u256_full_dry,
        'u25620_vs_row25620': {
            'active_token_ratio': u256_actual20_dry['total_active_tokens'] / max(1, first20['total_active_tokens']),
            'masked_target_ratio': u256_actual20_log['total_masked_tokens'] / max(1, first20['total_masked_tokens']),
            'dryrun_masked_target_ratio': u256_actual20_dry['total_masked_tokens'] / max(1, first20['total_masked_tokens']),
            'masked_per_active_ratio': (u256_actual20_log['total_masked_tokens'] / max(1, u256_actual20_dry['total_active_tokens'])) / max(1e-12, first20['masked_per_active_total']),
            'word_exposure_ratio': u256_actual20_log['total_words'] / max(1, first20['total_words']),
            'step_delta': u256_actual20_log['steps'] - first20['steps'],
        },
        'u256full_vs_row256full': {
            'active_token_ratio': u256_full_dry['total_active_tokens'] / max(1, full['total_active_tokens']),
            'dryrun_masked_target_ratio': u256_full_dry['total_masked_tokens'] / max(1, full['total_masked_tokens']),
            'step_delta': u256_full_dry['steps'] - full['steps'],
        },
    }
    if U64_DRYRUN_FULL.exists():
        u64_full = summarize_dryrun_csv(U64_DRYRUN_FULL, 'U64_128_256_full100M_dryrun')
        comp['U64_128_256_full100M_dryrun'] = u64_full
        comp['u64_vs_u256_full_dryrun'] = {
            'active_token_ratio': u64_full['total_active_tokens'] / max(1, u256_full_dry['total_active_tokens']),
            'masked_target_ratio': u64_full['total_masked_tokens'] / max(1, u256_full_dry['total_masked_tokens']),
            'nominal_slot_ratio': u64_full['total_nominal_slots'] / max(1, u256_full_dry['total_nominal_slots']),
            'pad_fraction_delta': u64_full['pad_fraction'] - u256_full_dry['pad_fraction'],
            'step_delta': u64_full['steps'] - u256_full_dry['steps'],
        }
        update_summaries.append(u64_full)

    result = {
        'status': 'A02_LEGAL16K_U256_VISIBILITY_PROFILE',
        'created_utc': now_utc(),
        'purpose': 'CPU-only A02-local accounting for interpreting U256 faithful visibility endpoint evidence.',
        'no_gpu_training_or_model_evaluation': True,
        'inputs': {
            'base_10m': rel(BASE_10M),
            'base_10m_sha256': base_sha,
            'stream_100m': rel(STREAM_100M),
            'stream_100m_sha256': stream_sha,
            'tokenizer_dir': rel(TOKENIZER_DIR),
            'tokenizer_json_sha256': tokenizer_sha,
            'log': rel(LOG),
            'u256_log_20m': rel(U256_LOG_20M),
            'u256_dryrun_step110': rel(U256_DRYRUN_STEP110),
            'u256_dryrun_full_step084': rel(U256_DRYRUN_FULL) if U256_DRYRUN_FULL.exists() else None,
            'u64_128_256_dryrun_full_step084': rel(U64_DRYRUN_FULL) if U64_DRYRUN_FULL.exists() else None,
            'u256_dryrun_metrics_full': load_dryrun_metrics(U256_DRYRUN_METRICS_FULL),
            'u64_dryrun_metrics_full': load_dryrun_metrics(U64_DRYRUN_METRICS_FULL),
        },
        'row256_visibility_profile': profile_bundle['profile'],
        'row256_reconstruction': {
            'updates': len(row_updates),
            'word_mismatch_count': len(row_mismatches),
            'word_mismatch_examples': row_mismatches[:10],
        },
        'update_mass_comparison': comp,
        'interpretation': {
            'u256_is_visibility_repair_not_new_text': True,
            'legal_word_budget_unchanged': True,
            'tokenizer_unchanged_from_step35': True,
            'u256_changes_training_example_object': 'row prefixes -> word-boundary chunks preserving stream order and all token pieces',
            'u256_20m_score_delta_exceeds_token_mass_change': True,
            'endpoint_risk_to_check': ['EWoK material-dynamics/social/quantitative losses', 'Reading loss', 'fragile GlobalPIQA small-n gain'],
            'u64_128_256_is_not_same_mechanism': 'same visibility but adds short-context/order/packing effects; do not interpret from U256 alone',
        },
        'outputs': {
            'json': rel(OUT_JSON),
            'note': rel(NOTE),
            'source_csv': rel(SOURCE_CSV),
            'top_recovered_rows_csv': rel(ROW_CSV),
            'update_summary_csv': rel(UPDATE_CSV),
            'token_length_bucket_csv': rel(PREFIX_BUCKET_CSV),
        },
        'elapsed_sec': round(time.time() - t0, 1),
    }

    write_csv(SOURCE_CSV, profile_bundle['profile']['source_table'])
    write_csv(ROW_CSV, profile_bundle['profile']['top_recovered_suffix_rows'])
    write_csv(PREFIX_BUCKET_CSV, profile_bundle['profile']['token_length_buckets'])
    write_csv(UPDATE_CSV, flatten_update_summaries(update_summaries))
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    write_note(result)
    print(json.dumps({
        'status': result['status'],
        'u256_recovered_tokens_per_epoch': profile_bundle['profile']['u256_recovered_tokens_total'],
        'u256_active_ratio_vs_row256': profile_bundle['profile']['u256_active_token_ratio_vs_row256'],
        'u25620_active_ratio_vs_row25620': comp['u25620_vs_row25620']['active_token_ratio'],
        'u25620_masked_ratio_vs_row25620': comp['u25620_vs_row25620']['masked_target_ratio'],
        'row256_reconstruction_mismatches': len(row_mismatches),
        'top_source_by_recovered_tokens': profile_bundle['profile']['source_table'][0]['source'] if profile_bundle['profile']['source_table'] else None,
        'out_json': rel(OUT_JSON),
        'note': rel(NOTE),
        'elapsed_sec': result['elapsed_sec'],
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
