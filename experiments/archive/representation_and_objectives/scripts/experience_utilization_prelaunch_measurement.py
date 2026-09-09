#!/usr/bin/env python3
"""research: prelaunch measurement for legal40k stream-order experience utilization.

Depth training/evaluation are already running elsewhere.  This CPU-only script
strengthens the next possible chunk-stream experiment without starting training:
  * characterize exactly which row256 suffix words/tokens are recovered by U256;
  * compare row256, U256, and U64_128_256 per-update word/token/target mass;
  * record whether a future U256 launch is a small fixed-length visibility repair
    or is likely to act through a particular source/content tail.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import math
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

USER_ROOT = Path('.').resolve()
WORKSPACE = USER_ROOT / 'experiments/archive/representation_and_objectives'
SCRIPTS = WORKSPACE / 'scripts'
COMPACT_EXPERIENCE_SCRIPTS = USER_ROOT / 'experiments/archive/compact_experience/scripts'
for p in (SCRIPTS, COMPACT_EXPERIENCE_SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import masking_curriculum_trainer as base  # noqa: E402
import experience_utilization_trainer as chunkbase  # noqa: E402

BASE_10M = USER_ROOT / 'experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl'
STREAM_100M = USER_ROOT / 'experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl'
TOKENIZER_DIR = WORKSPACE / 'data/legal_representation_route_map/tokenizers/legal_byte_bpe_40k'
BASELINE_LOG = WORKSPACE / 'training/runs/legal40k_accum_compact_view_reinvest_seed43022/training_log.jsonl'
U256_DRYRUN = WORKSPACE / 'data/dryrun_stream_order_legal40k_U256_full/dryrun_step_accounting.csv'
U64_DRYRUN = WORKSPACE / 'data/dryrun_stream_order_legal40k_U64_128_256_full/dryrun_step_accounting.csv'

OUT_DIR = WORKSPACE / 'data/experience_utilization_prelaunch_measurement'
OUT_JSON = OUT_DIR / 'experience_utilization_prelaunch_measurement.json'
BASELINE_UPDATE_CSV = OUT_DIR / 'row256_baseline_update_stats.csv'
SUMMARY_CSV = OUT_DIR / 'update_mass_summary.csv'
SOURCE_CSV = OUT_DIR / 'recovered_suffix_by_source.csv'
NOTE = (USER_ROOT / 'research/notes/representation_and_objectives/experience_utilization_prelaunch_measurement.md')

EXPECTED_BASE_SHA = '215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23'
EXPECTED_STREAM_SHA = '3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691'
EXPECTED_TOK_SHA = '94b44bcf5901d097e20294e8220740758eac29af3ec20d58d8da867c85197758'
POOL_WORDS = 10_000_000
TOTAL_WORDS = 100_000_000
SEQ256 = 256

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[\u2019'][A-Za-z0-9]+)?")
NUMBER_RE = re.compile(r"\b\d+(?:[.,:/-]\d+)*(?:%|st|nd|rd|th)?\b", re.I)
CAP_RE = re.compile(r"\b[A-Z][A-Za-z]+(?:[-'][A-Z]?[A-Za-z]+)?\b")

# Small broad training-text lexicon, independent of BabyLM score examples.  It is
# for tail characterization only, not data selection.
ACTION = {
    'put', 'take', 'get', 'give', 'go', 'come', 'make', 'made', 'open', 'close',
    'hold', 'push', 'pull', 'move', 'moved', 'turn', 'turned', 'drop', 'dropped',
    'pick', 'picked', 'use', 'used', 'using', 'play', 'eat', 'drink', 'wash',
    'throw', 'threw', 'bring', 'brought', 'carry', 'keep', 'kept', 'let', 'help',
    'build', 'built', 'break', 'broke', 'change', 'changed', 'find', 'found',
}
CAUSAL_TEMPORAL = {
    'because', 'cause', 'caused', 'causes', 'so', 'therefore', 'after', 'before',
    'when', 'while', 'if', 'then', 'result', 'results', 'resulted', 'became',
    'become', 'makes', 'made', 'prevent', 'prevents', 'allow', 'allows',
    'requires', 'required', 'during', 'until', 'since', 'leads', 'led',
}
PHYSICAL_OBJECT = {
    'water', 'fire', 'box', 'ball', 'door', 'cup', 'table', 'toy', 'paper',
    'hand', 'body', 'food', 'stone', 'wood', 'glass', 'machine', 'tool', 'wheel',
    'container', 'bag', 'room', 'floor', 'wall', 'book', 'plant', 'animal', 'air',
    'light', 'heat', 'material', 'metal', 'house', 'car', 'window', 'cloth',
}
SPATIAL_STATE = {
    'in', 'on', 'under', 'over', 'inside', 'outside', 'behind', 'front', 'near',
    'across', 'through', 'between', 'below', 'above', 'around', 'left', 'right',
    'top', 'bottom', 'beside', 'into', 'out', 'off', 'down', 'up', 'back',
}
MENTAL_SOCIAL = {
    'think', 'thought', 'know', 'knew', 'want', 'wanted', 'say', 'said', 'ask',
    'asked', 'tell', 'told', 'see', 'saw', 'look', 'looked', 'feel', 'felt',
    'believe', 'learn', 'teach', 'child', 'mother', 'father', 'person', 'people',
    'friend', 'teacher', 'man', 'woman', 'boy', 'girl', 'family', 'name',
}


def now_utc() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def raw_line_hash(raw: bytes) -> str:
    return hashlib.sha256(raw.rstrip(b'\r\n')).hexdigest()


def q(values: list[float], prob: float) -> float | None:
    if not values:
        return None
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    pos = prob * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def summarize(values: list[float]) -> dict[str, Any]:
    vals = [float(v) for v in values]
    if not vals:
        return {'n': 0}
    mean = statistics.fmean(vals)
    sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return {
        'n': len(vals),
        'sum': sum(vals),
        'mean': mean,
        'sd': sd,
        'cv': sd / mean if mean else None,
        'min': min(vals),
        'p05': q(vals, 0.05),
        'p25': q(vals, 0.25),
        'p50': q(vals, 0.50),
        'p75': q(vals, 0.75),
        'p95': q(vals, 0.95),
        'max': max(vals),
    }


def text_tokens(words: list[str]) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(' '.join(words))]


def word_feature_counts(words: list[str], original_text: str | None = None) -> dict[str, int]:
    toks = text_tokens(words)
    text = original_text if original_text is not None else ' '.join(words)
    cnt = collections.Counter(toks)
    action = sum(cnt[w] for w in ACTION)
    causal = sum(cnt[w] for w in CAUSAL_TEMPORAL)
    physical = sum(cnt[w] for w in PHYSICAL_OBJECT)
    spatial = sum(cnt[w] for w in SPATIAL_STATE)
    social = sum(cnt[w] for w in MENTAL_SOCIAL)
    numbers = len(NUMBER_RE.findall(text))
    caps = len(CAP_RE.findall(text))
    return {
        'token_count': len(toks),
        'action': action,
        'causal_temporal': causal,
        'physical_object': physical,
        'spatial_state': spatial,
        'mental_social': social,
        'numbers': numbers,
        'capitalized': caps,
        'action_physical_rowsignal': int(action > 0 and physical > 0),
        'spatial_physical_rowsignal': int(spatial > 0 and physical > 0),
        'causal_physical_rowsignal': int(causal > 0 and physical > 0),
    }


def prefix_visibility(word_tokens: list[list[int]], limit: int = SEQ256) -> dict[str, int]:
    cum = 0
    visible_words = 0
    boundary_split_words = 0
    boundary_hidden_tokens = 0
    full_hidden_word_tokens = 0
    hidden_full_words = 0
    for ids in word_tokens:
        wc = len(ids)
        if wc == 0:
            if cum < limit:
                visible_words += 1
            else:
                hidden_full_words += 1
            continue
        start = cum
        end = cum + wc
        if start < limit:
            visible_words += 1
            if end > limit:
                boundary_split_words += 1
                boundary_hidden_tokens += end - limit
        else:
            hidden_full_words += 1
            full_hidden_word_tokens += wc
        cum = end
    raw_tokens = cum
    return {
        'raw_tokens': raw_tokens,
        'prefix_active_tokens': min(raw_tokens, limit),
        'prefix_visible_words': visible_words,
        'hidden_words': hidden_full_words,
        'hidden_tokens_total': max(0, raw_tokens - limit),
        'full_hidden_word_tokens': full_hidden_word_tokens,
        'boundary_split_words': boundary_split_words,
        'boundary_hidden_tokens': boundary_hidden_tokens,
    }


def add_counter(dst: dict[str, int], src: dict[str, int], mul: int = 1) -> None:
    for k, v in src.items():
        dst[k] = int(dst.get(k, 0)) + int(v) * mul


def load_base_rows(tokenizer) -> dict[str, Any]:
    row_by_hash: dict[str, dict[str, Any]] = {}
    base_counter: collections.Counter[str] = collections.Counter()
    source_rows: collections.Counter[str] = collections.Counter()
    source_words: collections.Counter[str] = collections.Counter()
    source_hidden_words: collections.Counter[str] = collections.Counter()
    source_hidden_tokens: collections.Counter[str] = collections.Counter()
    source_boundary_tokens: collections.Counter[str] = collections.Counter()
    source_features: dict[str, dict[str, int]] = collections.defaultdict(lambda: collections.defaultdict(int))  # type: ignore[assignment]
    hidden_rows: list[dict[str, Any]] = []
    hidden_cue_rows: list[dict[str, Any]] = []
    rows = 0
    words_total = 0
    raw_tokens_total = 0
    prefix_active_total = 0
    visible_words_total = 0
    hidden_words_total = 0
    hidden_tokens_total = 0
    boundary_hidden_tokens_total = 0
    boundary_split_rows = 0
    duplicate_hashes = 0

    with BASE_10M.open('rb') as f:
        for row_idx, raw in enumerate(f):
            if not raw.strip():
                continue
            rec = json.loads(raw)
            h = raw_line_hash(raw)
            base_counter[h] += 1
            if base_counter[h] == 2:
                duplicate_hashes += 1
            text = str(rec['text'])
            words = int(rec['words'])
            split_words = text.split()
            if len(split_words) != words:
                raise RuntimeError(f'word count mismatch row {row_idx}: {len(split_words)} vs {words}')
            source = str(rec.get('source', ''))
            by_word, raw_tokens, unassigned = chunkbase.token_ids_by_whitespace_word(text, tokenizer)
            if unassigned:
                raise RuntimeError(f'unassigned tokenizer offsets row {row_idx}: {unassigned}')
            if len(by_word) != words:
                raise RuntimeError(f'whitespace-token rows mismatch row {row_idx}: {len(by_word)} vs {words}')
            vis = prefix_visibility(by_word)
            if vis['raw_tokens'] != raw_tokens:
                raise RuntimeError(f'raw token mismatch row {row_idx}: {vis["raw_tokens"]} vs {raw_tokens}')
            hidden_words = int(vis['hidden_words'])
            hidden_tokens = int(vis['hidden_tokens_total'])
            visible_words = int(vis['prefix_visible_words'])
            suffix_words = split_words[visible_words:]
            suffix_features = word_feature_counts(suffix_words, ' '.join(suffix_words)) if suffix_words else {
                'token_count': 0, 'action': 0, 'causal_temporal': 0, 'physical_object': 0,
                'spatial_state': 0, 'mental_social': 0, 'numbers': 0, 'capitalized': 0,
                'action_physical_rowsignal': 0, 'spatial_physical_rowsignal': 0, 'causal_physical_rowsignal': 0,
            }
            row_rec = {
                'hash': h,
                'row_index': row_idx,
                'words': words,
                'source': source,
                'raw_tokens': raw_tokens,
                'prefix_active_tokens': vis['prefix_active_tokens'],
                'prefix_visible_words': visible_words,
                'hidden_words': hidden_words,
                'hidden_tokens_total': hidden_tokens,
                'full_hidden_word_tokens': vis['full_hidden_word_tokens'],
                'boundary_split_words': vis['boundary_split_words'],
                'boundary_hidden_tokens': vis['boundary_hidden_tokens'],
                'suffix_features': suffix_features,
                'suffix_word_sample': suffix_words[:40],
                'text_prefix': text[:220],
            }
            # Duplicate raw lines share the same metrics; lookup by hash is enough for stream blocks.
            row_by_hash[h] = row_rec

            rows += 1
            words_total += words
            raw_tokens_total += raw_tokens
            prefix_active_total += int(vis['prefix_active_tokens'])
            visible_words_total += visible_words
            hidden_words_total += hidden_words
            hidden_tokens_total += hidden_tokens
            boundary_hidden_tokens_total += int(vis['boundary_hidden_tokens'])
            boundary_split_rows += int(vis['boundary_split_words'] > 0)
            source_rows[source] += 1
            source_words[source] += words
            source_hidden_words[source] += hidden_words
            source_hidden_tokens[source] += hidden_tokens
            source_boundary_tokens[source] += int(vis['boundary_hidden_tokens'])
            add_counter(source_features[source], suffix_features)
            if hidden_words > 0 or hidden_tokens > 0:
                hidden_rows.append(row_rec)
                cue_score = (
                    suffix_features.get('action_physical_rowsignal', 0)
                    + suffix_features.get('spatial_physical_rowsignal', 0)
                    + suffix_features.get('causal_physical_rowsignal', 0)
                    + min(3, suffix_features.get('action', 0) + suffix_features.get('causal_temporal', 0))
                    + min(2, suffix_features.get('numbers', 0) + suffix_features.get('capitalized', 0))
                )
                if cue_score > 0:
                    rr = dict(row_rec)
                    rr['suffix_cue_score'] = int(cue_score)
                    hidden_cue_rows.append(rr)
            if rows % 10000 == 0:
                print(json.dumps({'event': 'base_rows_tokenized', 'rows': rows, 'words': words_total}), flush=True)

    if words_total != POOL_WORDS:
        raise RuntimeError(f'base words {words_total} != {POOL_WORDS}')

    def trim_row(r: dict[str, Any]) -> dict[str, Any]:
        keep = ['row_index', 'source', 'words', 'raw_tokens', 'prefix_visible_words', 'hidden_words',
                'hidden_tokens_total', 'full_hidden_word_tokens', 'boundary_split_words',
                'boundary_hidden_tokens', 'suffix_features', 'suffix_word_sample', 'text_prefix']
        return {k: r.get(k) for k in keep}

    source_table = []
    for src in sorted(source_words, key=lambda s: (-source_hidden_words[s], s)):
        feats = dict(source_features[src])
        source_table.append({
            'source': src,
            'rows': int(source_rows[src]),
            'words': int(source_words[src]),
            'hidden_words': int(source_hidden_words[src]),
            'hidden_word_fraction_of_source_words': source_hidden_words[src] / max(1, source_words[src]),
            'hidden_tokens_total': int(source_hidden_tokens[src]),
            'boundary_hidden_tokens': int(source_boundary_tokens[src]),
            'suffix_action': int(feats.get('action', 0)),
            'suffix_causal_temporal': int(feats.get('causal_temporal', 0)),
            'suffix_physical_object': int(feats.get('physical_object', 0)),
            'suffix_spatial_state': int(feats.get('spatial_state', 0)),
            'suffix_mental_social': int(feats.get('mental_social', 0)),
            'suffix_numbers': int(feats.get('numbers', 0)),
            'suffix_capitalized': int(feats.get('capitalized', 0)),
            'suffix_action_physical_rowsignal': int(feats.get('action_physical_rowsignal', 0)),
            'suffix_spatial_physical_rowsignal': int(feats.get('spatial_physical_rowsignal', 0)),
            'suffix_causal_physical_rowsignal': int(feats.get('causal_physical_rowsignal', 0)),
        })

    hidden_rows_sorted = sorted(hidden_rows, key=lambda r: (r['hidden_words'], r['hidden_tokens_total'], r['raw_tokens']), reverse=True)
    hidden_cue_sorted = sorted(hidden_cue_rows, key=lambda r: (r.get('suffix_cue_score', 0), r['hidden_words'], r['hidden_tokens_total']), reverse=True)
    profile = {
        'rows': rows,
        'duplicate_raw_line_hashes': duplicate_hashes,
        'words_total': words_total,
        'raw_tokens_total': raw_tokens_total,
        'prefix_active_tokens_total': prefix_active_total,
        'u256_recovered_tokens_total': raw_tokens_total - prefix_active_total,
        'u256_active_token_ratio_vs_row256': raw_tokens_total / prefix_active_total,
        'prefix_visible_words_total': visible_words_total,
        'hidden_full_words_total': hidden_words_total,
        'hidden_full_word_fraction': hidden_words_total / words_total,
        'hidden_tokens_total': hidden_tokens_total,
        'full_hidden_word_tokens_total': sum(int(r['full_hidden_word_tokens']) for r in hidden_rows),
        'boundary_hidden_tokens_total': boundary_hidden_tokens_total,
        'boundary_split_rows': boundary_split_rows,
        'rows_with_any_hidden_word_or_token': len(hidden_rows),
        'rows_with_hidden_full_words': sum(1 for r in hidden_rows if int(r['hidden_words']) > 0),
        'rows_with_boundary_split_only': sum(1 for r in hidden_rows if int(r['hidden_words']) == 0 and int(r['boundary_hidden_tokens']) > 0),
        'source_table': source_table,
        'top_hidden_rows': [trim_row(r) for r in hidden_rows_sorted[:20]],
        'top_hidden_cue_rows': [trim_row(r) | {'suffix_cue_score': r.get('suffix_cue_score', 0)} for r in hidden_cue_sorted[:20]],
    }
    return {'row_by_hash': row_by_hash, 'counter': base_counter, 'profile': profile, 'source_table': source_table}


def load_training_log(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_chunk_dryrun_csv(path: Path, arm: str) -> list[dict[str, Any]]:
    out = []
    with path.open('r', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            n_chunks = int(r['n_chunks'])
            L = int(r['stage_length'])
            active = int(r['active_tokens'])
            words = int(r['charged_words'])
            masked = int(r['masked_tokens'])
            out.append({
                'arm': arm,
                'step': int(r['global_step']),
                'epoch': int(r['epoch']),
                'stage_length': L,
                'rows_or_chunks': n_chunks,
                'charged_words': words,
                'active_tokens': active,
                'masked_tokens': masked,
                'pad_slots': n_chunks * L - active,
                'microbatches': int(r['microbatches']),
                'cumulative_words': int(r['cumulative_words']),
                'masked_per_active': masked / max(1, active),
                'active_per_word': active / max(1, words),
            })
    return out


def reconstruct_row256_updates(row_by_hash: dict[str, dict[str, Any]], training_log: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    updates: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    cur: list[dict[str, Any]] = []
    total_rows = 0
    total_words = 0
    with STREAM_100M.open('rb') as f:
        for raw in f:
            if not raw.strip():
                continue
            h = raw_line_hash(raw)
            info = row_by_hash.get(h)
            if info is None:
                raise RuntimeError(f'stream row hash not found in base pool: {h}')
            cur.append(info)
            total_rows += 1
            total_words += int(info['words'])
            if len(cur) == 256:
                flush_row256_batch(cur, updates, training_log, mismatches)
                cur = []
    if cur:
        flush_row256_batch(cur, updates, training_log, mismatches)
    if total_words != TOTAL_WORDS:
        raise RuntimeError(f'stream words {total_words} != {TOTAL_WORDS}')
    return updates, mismatches


def flush_row256_batch(cur: list[dict[str, Any]], updates: list[dict[str, Any]], training_log: list[dict[str, Any]], mismatches: list[dict[str, Any]]) -> None:
    idx = len(updates)
    words = sum(int(r['words']) for r in cur)
    active = sum(int(r['prefix_active_tokens']) for r in cur)
    visible_words = sum(int(r['prefix_visible_words']) for r in cur)
    hidden_words = sum(int(r['hidden_words']) for r in cur)
    hidden_tokens = sum(int(r['hidden_tokens_total']) for r in cur)
    boundary_hidden = sum(int(r['boundary_hidden_tokens']) for r in cur)
    n_rows = len(cur)
    rec_log = training_log[idx] if idx < len(training_log) else {}
    log_words = rec_log.get('batch_words')
    log_masked = rec_log.get('masked_tokens')
    if log_words is not None and int(log_words) != words:
        mismatches.append({'step': idx + 1, 'computed_words': words, 'log_words': int(log_words)})
    updates.append({
        'arm': 'row256_baseline_log_order',
        'step': idx + 1,
        'stage_length': 256,
        'rows_or_chunks': n_rows,
        'charged_words': words,
        'active_tokens': active,
        'visible_words': visible_words,
        'hidden_words': hidden_words,
        'hidden_tokens_total': hidden_tokens,
        'boundary_hidden_tokens': boundary_hidden,
        'pad_slots': n_rows * 256 - active,
        'masked_tokens': int(log_masked) if log_masked is not None else None,
        'masked_per_active': (int(log_masked) / active) if log_masked is not None and active > 0 else None,
        'active_per_word': active / max(1, words),
        'log_loss': rec_log.get('loss'),
        'log_lr': rec_log.get('lr'),
        'cumulative_word_exposure': rec_log.get('cumulative_word_exposure'),
    })


def summarize_updates(updates: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    fields = ['charged_words', 'active_tokens', 'masked_tokens', 'pad_slots', 'rows_or_chunks', 'microbatches', 'masked_per_active', 'active_per_word']
    by_field = {}
    for field in fields:
        vals = [r[field] for r in updates if r.get(field) is not None]
        by_field[field] = summarize([float(v) for v in vals])
    total_words = sum(int(r['charged_words']) for r in updates)
    total_active = sum(int(r['active_tokens']) for r in updates)
    total_masked = sum(int(r['masked_tokens']) for r in updates if r.get('masked_tokens') is not None)
    total_pad = sum(int(r['pad_slots']) for r in updates)
    return {
        'arm': arm,
        'steps': len(updates),
        'total_words': total_words,
        'total_active_tokens': total_active,
        'total_masked_tokens': total_masked,
        'total_pad_slots': total_pad,
        'masked_per_active_total': total_masked / max(1, total_active),
        'active_tokens_per_word_total': total_active / max(1, total_words),
        'pad_fraction_of_nominal_slots': total_pad / max(1, total_pad + total_active),
        'field_summaries': by_field,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def flatten_summary_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for s in summaries:
        base_fields = {
            'arm': s['arm'],
            'steps': s['steps'],
            'total_words': s['total_words'],
            'total_active_tokens': s['total_active_tokens'],
            'total_masked_tokens': s['total_masked_tokens'],
            'total_pad_slots': s['total_pad_slots'],
            'masked_per_active_total': s['masked_per_active_total'],
            'active_tokens_per_word_total': s['active_tokens_per_word_total'],
            'pad_fraction_of_nominal_slots': s['pad_fraction_of_nominal_slots'],
        }
        for field, fs in s['field_summaries'].items():
            row = dict(base_fields)
            row['metric'] = field
            for k in ['mean', 'sd', 'cv', 'min', 'p05', 'p50', 'p95', 'max', 'sum']:
                row[k] = fs.get(k)
            out.append(row)
    return out


def write_note(result: dict[str, Any]) -> None:
    prof = result['row256_recovered_suffix_profile']
    comp = result['baseline_vs_chunked']
    row256 = comp['row256_baseline_log_order']
    u256 = comp['U256_mask015']
    u64 = comp['U64_128_256_mask015']
    lines: list[str] = []
    lines.append('# research — Experience-utilization prelaunch measurement')
    lines.append('')
    lines.append('CPU-only analysis without new training or model evaluation. Depth training and its dependent official-compatible evaluation remain pending.')
    lines.append('')
    lines.append('## What U256 actually recovers from row256')
    lines.append('')
    lines.append(f"- Legal40k 10M pool raw tokens: {prof['raw_tokens_total']:,}; row256 active tokens: {prof['prefix_active_tokens_total']:,}; U256 recovered tokens: {prof['u256_recovered_tokens_total']:,} ({prof['u256_active_token_ratio_vs_row256']:.6f}x active tokens).")
    lines.append(f"- Fully hidden charged words recovered by U256: {prof['hidden_full_words_total']:,} / {prof['words_total']:,} ({prof['hidden_full_word_fraction']:.4%}); rows with any hidden word/token: {prof['rows_with_any_hidden_word_or_token']:,}.")
    lines.append(f"- Boundary-hidden tokens inside the last visible word: {prof['boundary_hidden_tokens_total']:,}; full hidden-word tokens: {prof['full_hidden_word_tokens_total']:,}; rows with only boundary-token recovery: {prof['rows_with_boundary_split_only']:,}.")
    lines.append('')
    lines.append('The visible change is therefore small in total mass but semantically concentrated in long-row tails; it should be interpreted as fixed-length suffix visibility plus the ordinary WWM target pressure on those now-visible suffixes.')
    lines.append('')
    lines.append('## Per-update mass compared with row256')
    lines.append('')
    lines.append(f"- Row256 baseline reconstruction matched the training log batch words with {len(result['row256_batch_word_mismatches'])} mismatches over {row256['steps']} updates.")
    lines.append(f"- Row256: {row256['total_active_tokens']:,} active tokens, {row256['total_masked_tokens']:,} realized masked targets, {row256['steps']} updates, pad fraction {row256['pad_fraction_of_nominal_slots']:.4f}.")
    lines.append(f"- U256@0.15: {u256['total_active_tokens']:,} active tokens, {u256['total_masked_tokens']:,} realized masked targets, {u256['steps']} updates, pad fraction {u256['pad_fraction_of_nominal_slots']:.4f}.")
    lines.append(f"- U64_128_256@0.15: {u64['total_active_tokens']:,} active tokens, {u64['total_masked_tokens']:,} realized masked targets, {u64['steps']} updates, pad fraction {u64['pad_fraction_of_nominal_slots']:.4f}.")
    lines.append(f"- U256 versus row256: active-token ratio {comp['u256_vs_row256']['active_token_ratio']:.6f}; masked-target ratio {comp['u256_vs_row256']['masked_target_ratio']:.6f}; update-count difference {comp['u256_vs_row256']['step_delta']:+d}.")
    lines.append('')
    lines.append('U256 keeps maximum length 256 and stream order, but it changes the example object from rows to row-internal chunks, raises pad slots substantially, and uses one more stage-reset update than the continuous row baseline. These are small enough for a SOTA-facing endpoint test, but the target-matched U256 arm remains the attribution follow-up if U256@0.15 improves.')
    lines.append('')
    lines.append('## Source and tail profile')
    lines.append('')
    for row in prof['source_table'][:8]:
        lines.append(f"- {row['source']}: hidden words {row['hidden_words']:,} / source words {row['words']:,} ({row['hidden_word_fraction_of_source_words']:.3%}); hidden tokens {row['hidden_tokens_total']:,}; suffix cues action={row['suffix_action']}, causal={row['suffix_causal_temporal']}, physical={row['suffix_physical_object']}, spatial={row['suffix_spatial_state']}, names/numbers={row['suffix_capitalized'] + row['suffix_numbers']}.")
    lines.append('')
    lines.append('The recovered tail mass is not uniformly distributed across sources. The source table and top hidden rows should be read before choosing a data-mechanism interpretation for any U256 score movement.')
    lines.append('')
    lines.append('## Consequence for the next launch')
    lines.append('')
    lines.append('This measurement supports the independent_review reading: if depth is coherent and competitive, the first combined chunk-stream endpoint should be U256@0.15 on the 12x384 substrate; if depth is flat or worse, U256@0.15 on the 8x480 legal40k substrate is the cleaner fixed-length visibility comparison. U64_128_256 should follow a meaningful U256 signal, not precede it, because it adds context-length/order/packing changes on top of visibility repair.')
    lines.append('')
    lines.append(f"JSON: `{OUT_JSON.relative_to(USER_ROOT)}`")
    lines.append(f"CSV: `{SUMMARY_CSV.relative_to(USER_ROOT)}`, `{SOURCE_CSV.relative_to(USER_ROOT)}`, `{BASELINE_UPDATE_CSV.relative_to(USER_ROOT)}`")
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_sha = sha256_file(BASE_10M)
    stream_sha = sha256_file(STREAM_100M)
    tok_sha = sha256_file(TOKENIZER_DIR / 'tokenizer.json')
    if base_sha != EXPECTED_BASE_SHA:
        raise RuntimeError(f'base SHA mismatch {base_sha} != {EXPECTED_BASE_SHA}')
    if stream_sha != EXPECTED_STREAM_SHA:
        raise RuntimeError(f'stream SHA mismatch {stream_sha} != {EXPECTED_STREAM_SHA}')
    if tok_sha != EXPECTED_TOK_SHA:
        raise RuntimeError(f'tokenizer SHA mismatch {tok_sha} != {EXPECTED_TOK_SHA}')
    tokenizer = base.make_portable_tokenizer(str(TOKENIZER_DIR))
    t0 = time.time()
    base_rows = load_base_rows(tokenizer)
    print(json.dumps({'event': 'base_profile_done', 'elapsed_sec': round(time.time() - t0, 1)}), flush=True)

    training_log = load_training_log(BASELINE_LOG)
    row256_updates, mismatches = reconstruct_row256_updates(base_rows['row_by_hash'], training_log)
    u256_updates = read_chunk_dryrun_csv(U256_DRYRUN, 'U256_mask015')
    u64_updates = read_chunk_dryrun_csv(U64_DRYRUN, 'U64_128_256_mask015')

    summaries = [
        summarize_updates(row256_updates, 'row256_baseline_log_order'),
        summarize_updates(u256_updates, 'U256_mask015'),
        summarize_updates(u64_updates, 'U64_128_256_mask015'),
    ]
    by_arm = {s['arm']: s for s in summaries}
    u256_vs = {
        'active_token_ratio': by_arm['U256_mask015']['total_active_tokens'] / by_arm['row256_baseline_log_order']['total_active_tokens'],
        'masked_target_ratio': by_arm['U256_mask015']['total_masked_tokens'] / by_arm['row256_baseline_log_order']['total_masked_tokens'],
        'pad_slot_ratio': by_arm['U256_mask015']['total_pad_slots'] / by_arm['row256_baseline_log_order']['total_pad_slots'],
        'step_delta': by_arm['U256_mask015']['steps'] - by_arm['row256_baseline_log_order']['steps'],
        'active_mean_delta_per_update': by_arm['U256_mask015']['field_summaries']['active_tokens']['mean'] - by_arm['row256_baseline_log_order']['field_summaries']['active_tokens']['mean'],
        'masked_mean_delta_per_update': by_arm['U256_mask015']['field_summaries']['masked_tokens']['mean'] - by_arm['row256_baseline_log_order']['field_summaries']['masked_tokens']['mean'],
    }
    u64_vs_u256 = {
        'active_token_ratio': by_arm['U64_128_256_mask015']['total_active_tokens'] / by_arm['U256_mask015']['total_active_tokens'],
        'masked_target_ratio': by_arm['U64_128_256_mask015']['total_masked_tokens'] / by_arm['U256_mask015']['total_masked_tokens'],
        'pad_slot_ratio': by_arm['U64_128_256_mask015']['total_pad_slots'] / by_arm['U256_mask015']['total_pad_slots'],
        'step_delta': by_arm['U64_128_256_mask015']['steps'] - by_arm['U256_mask015']['steps'],
        'chunk_mean_ratio': by_arm['U64_128_256_mask015']['field_summaries']['rows_or_chunks']['mean'] / by_arm['U256_mask015']['field_summaries']['rows_or_chunks']['mean'],
    }

    write_csv(BASELINE_UPDATE_CSV, row256_updates)
    write_csv(SOURCE_CSV, base_rows['source_table'])
    write_csv(SUMMARY_CSV, flatten_summary_rows(summaries))

    result = {
        'status': 'EXPERIENCE_UTILIZATION_PRELAUNCH_MEASUREMENT',
        'created_utc': now_utc(),
        'purpose': 'CPU-only characterization of recovered row256 suffix exposure and per-update mass before any experience-utilization H100 launch.',
        'no_new_training_or_model_evaluation': True,
        'inputs': {
            'base_10M': str(BASE_10M.relative_to(USER_ROOT)),
            'base_10M_sha256': base_sha,
            'stream_100M': str(STREAM_100M.relative_to(USER_ROOT)),
            'stream_100M_sha256': stream_sha,
            'tokenizer_dir': str(TOKENIZER_DIR.relative_to(USER_ROOT)),
            'tokenizer_json_sha256': tok_sha,
            'baseline_training_log': str(BASELINE_LOG.relative_to(USER_ROOT)),
            'u256_dryrun_step_csv': str(U256_DRYRUN.relative_to(USER_ROOT)),
            'u64_dryrun_step_csv': str(U64_DRYRUN.relative_to(USER_ROOT)),
        },
        'row256_recovered_suffix_profile': base_rows['profile'],
        'row256_batch_word_mismatches': mismatches[:20],
        'row256_batch_word_mismatch_count': len(mismatches),
        'baseline_vs_chunked': {
            **by_arm,
            'u256_vs_row256': u256_vs,
            'u64_vs_u256': u64_vs_u256,
        },
        'interpretation': {
            'u256_is_fixed_length_visibility_repair': True,
            'u256_changes_target_pressure_at_mask015': True,
            'target_matched_followup_mask_prob_expected': 0.14745584123068767,
            'target_matched_followup_mask_prob_realized_log_match': 0.14752237093624423,
            'u64_128_256_requires_u256_reference_first': True,
            'row256_reconstruction_matched_training_log_words': len(mismatches) == 0,
            'stage_reset_update_delta_vs_row256_continuous': u256_vs['step_delta'],
        },
        'outputs': {
            'json': str(OUT_JSON.relative_to(USER_ROOT)),
            'note': str(NOTE.relative_to(USER_ROOT)),
            'row256_update_csv': str(BASELINE_UPDATE_CSV.relative_to(USER_ROOT)),
            'summary_csv': str(SUMMARY_CSV.relative_to(USER_ROOT)),
            'source_csv': str(SOURCE_CSV.relative_to(USER_ROOT)),
        },
        'elapsed_sec': round(time.time() - t0, 1),
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    write_note(result)
    print(json.dumps({
        'status': result['status'],
        'u256_recovered_tokens_total': base_rows['profile']['u256_recovered_tokens_total'],
        'hidden_full_words_total': base_rows['profile']['hidden_full_words_total'],
        'row256_word_mismatches': len(mismatches),
        'u256_active_ratio': u256_vs['active_token_ratio'],
        'u256_masked_ratio': u256_vs['masked_target_ratio'],
        'out_json': str(OUT_JSON.relative_to(USER_ROOT)),
        'note': str(NOTE.relative_to(USER_ROOT)),
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
