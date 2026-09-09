#!/usr/bin/env python3
"""research: compare U256-recovered row suffixes to row256-visible prefixes.

CPU-only mechanism profile.  It asks whether the extra U256-visible mass has a
systematically different broad lexical/source composition from the row256-visible
mass already seen by research.  These are descriptive, benchmark-independent text
features; they are not used for training or selection.
"""
from __future__ import annotations

import collections
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

USER_ROOT = Path('.').resolve()
WORKSPACE = USER_ROOT / 'experiments/archive/frontier_consolidation'
SCRIPTS = WORKSPACE / 'scripts'
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import u256_visibility_profile as prof  # noqa: E402
import u256_tail_quality as tq  # noqa: E402

OUT_DIR = WORKSPACE / 'data/u256_tail_balance_profile'
OUT_JSON = OUT_DIR / 'u256_tail_balance_profile.json'
BY_SOURCE_CSV = OUT_DIR / 'tail_balance_by_source.csv'
TOP_SHIFT_CSV = OUT_DIR / 'top_tail_enriched_rows.csv'
NOTE = (USER_ROOT / 'research/notes/frontier_consolidation/u256_tail_balance_profile.md')

FEATURE_KEYS = [
    'action', 'causal_temporal', 'physical_object', 'spatial_state', 'mental_social',
    'material_property', 'quantitative', 'numbers', 'capitalized',
]
QUALITY_KEYS = [
    'cleanish_flag', 'encoding_noise_flag', 'allcaps_dense_flag', 'is_childes_like',
    'is_subtitle_like', 'is_list_like',
]


def now() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(USER_ROOT))
    except ValueError:
        return str(path)


def safe_div(a: float, b: float) -> float | None:
    if b == 0:
        return None
    return a / b


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fields: list[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def word_span_features(words: list[str]) -> dict[str, int]:
    return prof.word_feature_counts(words)


def quality_features(text: str, source: str) -> dict[str, Any]:
    return tq.text_quality_features(text, source)


def add_counts(dst: dict[str, float], counts: dict[str, Any], prefix: str = '') -> None:
    for k, v in counts.items():
        if isinstance(v, (int, float)):
            dst[prefix + k] = float(dst.get(prefix + k, 0.0)) + float(v)


def ratio_block(d: dict[str, float], span: str) -> dict[str, Any]:
    toks = max(1.0, float(d.get(f'{span}_token_count', 0.0)))
    words = max(1.0, float(d.get(f'{span}_words', 0.0)))
    out: dict[str, Any] = {
        f'{span}_rows': int(d.get(f'{span}_rows', 0.0)),
        f'{span}_words': int(d.get(f'{span}_words', 0.0)),
        f'{span}_bpe_tokens': int(d.get(f'{span}_bpe_tokens', 0.0)),
        f'{span}_lexical_tokens': int(d.get(f'{span}_token_count', 0.0)),
    }
    for k in FEATURE_KEYS:
        out[f'{span}_{k}_count'] = int(d.get(f'{span}_{k}', 0.0))
        out[f'{span}_{k}_per_lexical_token'] = float(d.get(f'{span}_{k}', 0.0)) / toks
        out[f'{span}_{k}_per_word'] = float(d.get(f'{span}_{k}', 0.0)) / words
    for k in QUALITY_KEYS:
        out[f'{span}_{k}_rows'] = int(d.get(f'{span}_{k}', 0.0))
        out[f'{span}_{k}_row_fraction'] = float(d.get(f'{span}_{k}', 0.0)) / max(1.0, float(d.get(f'{span}_rows', 0.0)))
    return out


def row_shift_score(prefix_feats: dict[str, int], suffix_feats: dict[str, int], prefix_words: int, suffix_words: int) -> float:
    # Broad descriptive enrichment score: extra density of physical/action/social/quantity cues in suffix.
    pf_den = sum(prefix_feats.get(k, 0) for k in ['action', 'physical_object', 'spatial_state', 'material_property', 'quantitative', 'mental_social']) / max(1, prefix_words)
    sf_den = sum(suffix_feats.get(k, 0) for k in ['action', 'physical_object', 'spatial_state', 'material_property', 'quantitative', 'mental_social']) / max(1, suffix_words)
    return sf_den - pf_den


def main() -> None:
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NOTE.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = prof.base.make_portable_tokenizer(str(prof.TOKENIZER_DIR))
    base_sha = prof.sha256_file(prof.BASE_10M)
    tok_sha = prof.sha256_file(prof.TOKENIZER_DIR / 'tokenizer.json')
    if base_sha != prof.EXPECTED_BASE_SHA or tok_sha != prof.EXPECTED_TOKENIZER_SHA:
        raise RuntimeError({'base_sha': base_sha, 'tok_sha': tok_sha})

    totals: dict[str, float] = collections.defaultdict(float)
    by_source: dict[str, dict[str, float]] = collections.defaultdict(lambda: collections.defaultdict(float))  # type: ignore[assignment]
    top_rows: list[dict[str, Any]] = []
    recovered_rows = 0
    recovered_tokens = 0
    recovered_words = 0
    boundary_tokens = 0
    words_total = 0
    row256_active_tokens_total = 0
    raw_tokens_total = 0

    with prof.BASE_10M.open('r', encoding='utf-8') as f:
        for row_idx, line in enumerate(f):
            if not line.strip():
                continue
            rec = json.loads(line)
            text = str(rec['text'])
            words = int(rec['words'])
            split_words = text.split()
            if len(split_words) != words:
                raise RuntimeError({'row_idx': row_idx, 'split_words': len(split_words), 'words': words})
            source = str(rec.get('source', ''))
            by_word, raw_tokens, unassigned = prof.chunkbase.token_ids_by_whitespace_word(text, tokenizer)
            if unassigned:
                raise RuntimeError({'row_idx': row_idx, 'unassigned': unassigned[:10]})
            vis = prof.prefix_visibility(by_word)
            raw_tokens_total += raw_tokens
            row256_active_tokens_total += int(vis['row256_active_tokens'])
            words_total += words
            visible_full = int(vis['row256_visible_full_words'])
            boundary_split = int(vis['row256_boundary_split_words'])
            suffix_start = min(words, visible_full + boundary_split)
            prefix_words_list = split_words[:suffix_start]
            suffix_words_list = split_words[suffix_start:]
            # Prefix includes the boundary-split word as research already labels that WWM group, while suffix counts fully hidden words.
            prefix_feats = word_span_features(prefix_words_list)
            suffix_feats = word_span_features(suffix_words_list)
            prefix_quality = quality_features(' '.join(prefix_words_list), source)
            suffix_quality = quality_features(' '.join(suffix_words_list), source) if suffix_words_list else {k: 0 for k in quality_features('', source)}
            prefix_bpe = int(vis['row256_active_tokens'])
            suffix_bpe = int(vis['row256_hidden_tokens_total'])
            suffix_full_words = int(vis['row256_hidden_full_words'])
            bnd = int(vis['row256_boundary_hidden_tokens'])
            row_common = {
                'rows': 1,
                'words_total': words,
                'raw_tokens_total': raw_tokens,
                'row256_active_tokens_total': prefix_bpe,
                'u256_recovered_tokens_total': suffix_bpe,
                'u256_recovered_words_total': suffix_full_words,
                'u256_boundary_tokens_total': bnd,
            }
            for dst in [totals, by_source[source]]:
                add_counts(dst, row_common)
                dst['prefix_rows'] = dst.get('prefix_rows', 0.0) + 1.0
                dst['prefix_words'] = dst.get('prefix_words', 0.0) + float(len(prefix_words_list))
                dst['prefix_bpe_tokens'] = dst.get('prefix_bpe_tokens', 0.0) + float(prefix_bpe)
                add_counts(dst, prefix_feats, 'prefix_')
                for k in QUALITY_KEYS:
                    dst[f'prefix_{k}'] = dst.get(f'prefix_{k}', 0.0) + float(prefix_quality.get(k, 0))
                if suffix_bpe > 0:
                    dst['suffix_rows'] = dst.get('suffix_rows', 0.0) + 1.0
                    dst['suffix_words'] = dst.get('suffix_words', 0.0) + float(len(suffix_words_list))
                    dst['suffix_bpe_tokens'] = dst.get('suffix_bpe_tokens', 0.0) + float(suffix_bpe)
                    add_counts(dst, suffix_feats, 'suffix_')
                    for k in QUALITY_KEYS:
                        dst[f'suffix_{k}'] = dst.get(f'suffix_{k}', 0.0) + float(suffix_quality.get(k, 0))
            if suffix_bpe > 0:
                recovered_rows += 1
                recovered_tokens += suffix_bpe
                recovered_words += suffix_full_words
                boundary_tokens += bnd
                shift = row_shift_score(prefix_feats, suffix_feats, len(prefix_words_list), max(1, len(suffix_words_list)))
                top_rows.append({
                    'row_index': row_idx,
                    'source': source,
                    'words': words,
                    'raw_tokens': raw_tokens,
                    'row256_active_tokens': prefix_bpe,
                    'u256_recovered_tokens': suffix_bpe,
                    'u256_recovered_full_words': suffix_full_words,
                    'u256_boundary_tokens': bnd,
                    'shift_score': shift,
                    'prefix_cue_density': (sum(prefix_feats.get(k, 0) for k in ['action', 'physical_object', 'spatial_state', 'material_property', 'quantitative', 'mental_social']) / max(1, len(prefix_words_list))),
                    'suffix_cue_density': (sum(suffix_feats.get(k, 0) for k in ['action', 'physical_object', 'spatial_state', 'material_property', 'quantitative', 'mental_social']) / max(1, len(suffix_words_list))),
                    'suffix_cue_counts': {k: suffix_feats.get(k, 0) for k in FEATURE_KEYS},
                    'suffix_quality': {k: suffix_quality.get(k, 0) for k in QUALITY_KEYS},
                    'suffix_sample': ' '.join(suffix_words_list[:70]).replace('\n', ' '),
                    'prefix_tail_sample': ' '.join(prefix_words_list[-35:]).replace('\n', ' '),
                })
            if (row_idx + 1) % 10000 == 0:
                print(json.dumps({'event': 'rows_scanned', 'rows': row_idx + 1, 'recovered_rows': recovered_rows}), flush=True)

    source_rows: list[dict[str, Any]] = []
    for source, d in by_source.items():
        row = {
            'source': source,
            'rows': int(d.get('rows', 0)),
            'words_total': int(d.get('words_total', 0)),
            'raw_tokens_total': int(d.get('raw_tokens_total', 0)),
            'row256_active_tokens_total': int(d.get('row256_active_tokens_total', 0)),
            'u256_recovered_tokens_total': int(d.get('u256_recovered_tokens_total', 0)),
            'u256_recovered_words_total': int(d.get('u256_recovered_words_total', 0)),
            'recovered_token_share': safe_div(float(d.get('u256_recovered_tokens_total', 0)), max(1.0, float(totals.get('u256_recovered_tokens_total', 0)))),
            'recovered_token_fraction_of_source_raw': safe_div(float(d.get('u256_recovered_tokens_total', 0)), max(1.0, float(d.get('raw_tokens_total', 0)))),
        }
        row.update(ratio_block(d, 'prefix'))
        row.update(ratio_block(d, 'suffix'))
        # Compact comparison fields for quick reading.
        for k in FEATURE_KEYS:
            row[f'{k}_suffix_minus_prefix_per_word'] = row[f'suffix_{k}_per_word'] - row[f'prefix_{k}_per_word']
        for k in QUALITY_KEYS:
            row[f'{k}_suffix_minus_prefix_row_fraction'] = row[f'suffix_{k}_row_fraction'] - row[f'prefix_{k}_row_fraction']
        source_rows.append(row)
    source_rows.sort(key=lambda r: (-int(r['u256_recovered_tokens_total']), str(r['source'])))

    top_rows_sorted = sorted(top_rows, key=lambda r: (r['shift_score'], r['u256_recovered_tokens']), reverse=True)[:80]
    csv_top_rows = []
    for r in top_rows_sorted:
        rr = dict(r)
        rr.update({f'suffix_{k}': r['suffix_cue_counts'].get(k, 0) for k in FEATURE_KEYS})
        rr.update({f'suffix_{k}': r['suffix_quality'].get(k, 0) for k in QUALITY_KEYS})
        rr.pop('suffix_cue_counts', None)
        rr.pop('suffix_quality', None)
        csv_top_rows.append(rr)

    summary = {
        'status': 'U256_TAIL_BALANCE_PROFILE',
        'utc': now(),
        'base10m': rel(prof.BASE_10M),
        'tokenizer': rel(prof.TOKENIZER_DIR),
        'hashes': {
            'base10m_sha256': base_sha,
            'tokenizer_json_sha256': tok_sha,
        },
        'rows_total': int(totals.get('rows', 0)),
        'words_total': words_total,
        'raw_tokens_total': raw_tokens_total,
        'row256_active_tokens_total': row256_active_tokens_total,
        'u256_recovered_tokens_total': recovered_tokens,
        'u256_recovered_words_total': recovered_words,
        'u256_boundary_tokens_total': boundary_tokens,
        'u256_active_token_ratio_vs_row256': raw_tokens_total / max(1, row256_active_tokens_total),
        'recovered_rows': recovered_rows,
        'recovered_row_fraction': recovered_rows / max(1, int(totals.get('rows', 0))),
        'global_prefix': ratio_block(totals, 'prefix'),
        'global_suffix': ratio_block(totals, 'suffix'),
        'global_suffix_minus_prefix_per_word': {
            k: (float(totals.get(f'suffix_{k}', 0.0)) / max(1.0, float(totals.get('suffix_words', 0.0)))) - (float(totals.get(f'prefix_{k}', 0.0)) / max(1.0, float(totals.get('prefix_words', 0.0))))
            for k in FEATURE_KEYS
        },
        'global_suffix_minus_prefix_quality_fraction': {
            k: (float(totals.get(f'suffix_{k}', 0.0)) / max(1.0, float(totals.get('suffix_rows', 0.0)))) - (float(totals.get(f'prefix_{k}', 0.0)) / max(1.0, float(totals.get('prefix_rows', 0.0))))
            for k in QUALITY_KEYS
        },
        'source_table': source_rows,
        'top_tail_enriched_rows': csv_top_rows[:40],
        'outputs': {
            'json': rel(OUT_JSON),
            'by_source_csv': rel(BY_SOURCE_CSV),
            'top_shift_csv': rel(TOP_SHIFT_CSV),
            'note': rel(NOTE),
        },
        'elapsed_sec': round(time.time() - t0, 3),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    write_csv(BY_SOURCE_CSV, source_rows)
    write_csv(TOP_SHIFT_CSV, csv_top_rows)

    gspw = summary['global_suffix_minus_prefix_per_word']
    gq = summary['global_suffix_minus_prefix_quality_fraction']
    NOTE.write_text(
        '# research — U256 tail balance profile\n\n'
        'This CPU-only profile compares the row256-visible prefix mass with the U256-recovered suffix mass in the exact legal 10M pool. Features are broad corpus-text descriptors only.\n\n'
        f'- Rows: {summary["rows_total"]:,}; recovered-tail rows: {recovered_rows:,} ({summary["recovered_row_fraction"]:.4%}).\n'
        f'- Row256 active BPE tokens: {row256_active_tokens_total:,}; U256 recovered BPE tokens: {recovered_tokens:,}; active-token ratio: {summary["u256_active_token_ratio_vs_row256"]:.6f}.\n'
        f'- Recovered full words: {recovered_words:,}; boundary hidden BPE pieces: {boundary_tokens:,}.\n'
        f'- Global suffix-minus-prefix per-word shifts: action {gspw["action"]:+.4f}, physical {gspw["physical_object"]:+.4f}, spatial {gspw["spatial_state"]:+.4f}, material {gspw["material_property"]:+.4f}, quantitative {gspw["quantitative"]:+.4f}, mental/social {gspw["mental_social"]:+.4f}.\n'
        f'- Global suffix-minus-prefix row fractions: cleanish {gq["cleanish_flag"]:+.4f}, encoding-noise {gq["encoding_noise_flag"]:+.4f}, allcaps/glued {gq["allcaps_dense_flag"]:+.4f}, CHILDES-like {gq["is_childes_like"]:+.4f}, subtitle-like {gq["is_subtitle_like"]:+.4f}, list/TOC-like {gq["is_list_like"]:+.4f}.\n\n'
        'Largest source shares of recovered tokens:\n'
        + ''.join(f'- {r["source"]}: {int(r["u256_recovered_tokens_total"]):,} tokens, share {float(r["recovered_token_share"] or 0):.2%}, source raw fraction {float(r["recovered_token_fraction_of_source_raw"] or 0):.2%}.\n' for r in source_rows[:6])
        + '\nInterpretation: use this when the U256 endpoint score arrives. A positive endpoint would mean this small, source-tail-concentrated visibility repair added useful credit; a negative or mixed endpoint should be read against the suffix enrichment/noise/source mix rather than as evidence against compact-view reinvestment itself.\n\n'
        f'JSON: `{rel(OUT_JSON)}`\n',
        encoding='utf-8',
    )
    print(json.dumps({'status': summary['status'], 'out_json': rel(OUT_JSON), 'note': rel(NOTE), 'recovered_tokens': recovered_tokens, 'active_ratio': summary['u256_active_token_ratio_vs_row256']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
