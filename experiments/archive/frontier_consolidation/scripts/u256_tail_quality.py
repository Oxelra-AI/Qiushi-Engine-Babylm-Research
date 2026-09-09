#!/usr/bin/env python3
"""research: quality/type profile of U256-recovered row tails.

CPU-only.  It reuses the legal16k tokenizer and exact 10M pool to classify
which kinds of row suffix text U256 exposes that research row256 truncates.
The categories are descriptive mechanism interpretation, not data selection.
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
SCRIPTS = USER_ROOT / 'experiments/archive/frontier_consolidation/scripts'
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import u256_visibility_profile as prof  # noqa: E402

OUT_DIR = USER_ROOT / 'experiments/archive/frontier_consolidation/data/u256_tail_quality'
OUT_JSON = OUT_DIR / 'u256_tail_quality.json'
BY_SOURCE_CSV = OUT_DIR / 'tail_quality_by_source.csv'
TOP_NOISE_CSV = OUT_DIR / 'top_recovered_tail_noise_rows.csv'
NOTE = USER_ROOT / 'research/notes/frontier_consolidation/u256_tail_quality.md'

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[\u2019'][A-Za-z0-9]+)?")
MOJIBAKE_RE = re.compile(r"[�ÆÂÃÄÅÇÐÑÓÍÎÏÙÚÛÝÞß¢£¤¥§¨©®º¼½¾]|1⁄2|3⁄4|\?\s*\?")
CHILDES_RE = re.compile(r"\*(?:MOT|CHI|FAT|BRO|SIS|BR\d|INV|EXP|PAR|ADU):|%\w+:", re.I)
SUBTITLE_MARKER_RE = re.compile(r"\b(?:subtitle|original air date|sync|episode|season|www\.|http|\.com|@)\b", re.I)
LIST_TOC_RE = re.compile(r"\b(?:chapter|contents|illustrations|page|vol\.|no\.|appendix|table of contents)\b|={2,}|\b\d{2,4}\b.*\b\d{2,4}\b", re.I)
GLUED_RE = re.compile(r"\b[A-Z]{8,}\b|\b[A-Za-z]{18,}\b")


def now() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(USER_ROOT))
    except ValueError:
        return str(path)


def q(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    vals = sorted(xs)
    if len(vals) == 1:
        return vals[0]
    pos = p * (len(vals) - 1)
    lo = int(math.floor(pos)); hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def summarize(xs: list[float]) -> dict[str, Any]:
    if not xs:
        return {'n': 0}
    return {
        'n': len(xs), 'sum': sum(xs), 'mean': statistics.fmean(xs),
        'sd': statistics.pstdev(xs) if len(xs) > 1 else 0.0,
        'min': min(xs), 'p50': q(xs, 0.5), 'p90': q(xs, 0.9), 'p95': q(xs, 0.95), 'max': max(xs),
    }


def text_quality_features(text: str, source: str) -> dict[str, Any]:
    toks = WORD_RE.findall(text)
    n_tok = max(1, len(toks))
    upper = sum(1 for t in toks if len(t) > 1 and t.isupper())
    digit = sum(1 for t in toks if any(ch.isdigit() for ch in t))
    long_glued = len(GLUED_RE.findall(text))
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    replacement = text.count('�')
    mojibake = len(MOJIBAKE_RE.findall(text))
    childes = len(CHILDES_RE.findall(text))
    subtitle = len(SUBTITLE_MARKER_RE.findall(text))
    list_toc = len(LIST_TOC_RE.findall(text))
    punct = sum(1 for ch in text if ch in '.!?;:')
    source_l = source.lower()
    is_childes_like = int(childes > 0 or source_l == 'childes')
    is_subtitle_like = int(subtitle > 0 or 'subtitle' in source_l)
    is_list_like = int(list_toc > 0)
    encoding_noise = int(replacement > 0 or mojibake >= 3 or (non_ascii / max(1, len(text)) > 0.08 and source_l != 'childes'))
    allcaps_dense = int(upper / n_tok > 0.35 or long_glued >= 3)
    cleanish = int(not encoding_noise and not allcaps_dense and childes == 0 and subtitle == 0 and list_toc == 0)
    return {
        'word_tokens': len(toks),
        'uppercase_token_count': upper,
        'uppercase_token_fraction': upper / n_tok,
        'digit_token_count': digit,
        'digit_token_fraction': digit / n_tok,
        'long_or_glued_token_count': long_glued,
        'non_ascii_char_count': non_ascii,
        'unicode_replacement_count': replacement,
        'mojibake_marker_count': mojibake,
        'childes_marker_count': childes,
        'subtitle_marker_count': subtitle,
        'list_toc_marker_count': list_toc,
        'punctuation_count': punct,
        'is_childes_like': is_childes_like,
        'is_subtitle_like': is_subtitle_like,
        'is_list_like': is_list_like,
        'encoding_noise_flag': encoding_noise,
        'allcaps_dense_flag': allcaps_dense,
        'cleanish_flag': cleanish,
        'noise_score': int(encoding_noise) * 3 + int(allcaps_dense) * 2 + min(4, mojibake) + min(3, childes + subtitle + list_toc) + min(3, long_glued),
    }


def add(dst: dict[str, float], key: str, val: float) -> None:
    dst[key] = float(dst.get(key, 0.0)) + float(val)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fields=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); fields.append(k)
    with path.open('w', encoding='utf-8', newline='') as f:
        w=csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def main() -> None:
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tokenizer = prof.base.make_portable_tokenizer(str(prof.TOKENIZER_DIR))
    base_sha = prof.sha256_file(prof.BASE_10M)
    tok_sha = prof.sha256_file(prof.TOKENIZER_DIR / 'tokenizer.json')
    if base_sha != prof.EXPECTED_BASE_SHA or tok_sha != prof.EXPECTED_TOKENIZER_SHA:
        raise RuntimeError({'sha_mismatch': True, 'base_sha': base_sha, 'tok_sha': tok_sha})

    totals: dict[str, float] = collections.defaultdict(float)
    by_source: dict[str, dict[str, float]] = collections.defaultdict(lambda: collections.defaultdict(float))  # type: ignore[assignment]
    noise_rows: list[dict[str, Any]] = []
    recovered_token_values: list[float] = []
    recovered_word_values: list[float] = []

    with prof.BASE_10M.open('rb') as f:
        for row_idx, raw in enumerate(f):
            if not raw.strip():
                continue
            rec = json.loads(raw)
            text = str(rec['text'])
            words = int(rec['words'])
            split_words = text.split()
            source = str(rec.get('source', ''))
            by_word, raw_tokens, unassigned = prof.chunkbase.token_ids_by_whitespace_word(text, tokenizer)
            if unassigned:
                raise RuntimeError({'row_idx': row_idx, 'unassigned': unassigned})
            vis = prof.prefix_visibility(by_word)
            rec_tokens = int(vis['row256_hidden_tokens_total'])
            if rec_tokens <= 0:
                continue
            suffix_start = min(words, int(vis['row256_visible_full_words']) + int(vis['row256_boundary_split_words']))
            suffix_words = split_words[suffix_start:]
            suffix_text = ' '.join(suffix_words)
            feats = text_quality_features(suffix_text, source)
            rec_words = int(vis['row256_hidden_full_words'])
            recovered_token_values.append(float(rec_tokens))
            recovered_word_values.append(float(rec_words))
            row_core = {
                'row_index': row_idx,
                'source': source,
                'words': words,
                'raw_tokens': raw_tokens,
                'u256_recovered_tokens': rec_tokens,
                'u256_recovered_full_words': rec_words,
                'recovered_text_sample': suffix_text[:280].replace('\n', ' '),
                **feats,
            }
            noise_rows.append(row_core)
            for k in ['rows', 'u256_recovered_tokens', 'u256_recovered_full_words']:
                if k == 'rows':
                    totals[k] += 1; by_source[source][k] += 1
                else:
                    totals[k] += float(row_core[k]); by_source[source][k] += float(row_core[k])
            for k, v in feats.items():
                if isinstance(v, (int, float)):
                    totals[k] += float(v)
                    by_source[source][k] += float(v)
            if (row_idx + 1) % 10000 == 0:
                print(json.dumps({'event': 'rows_scanned', 'rows': row_idx + 1, 'recovered_rows': int(totals.get('rows', 0))}), flush=True)

    by_source_rows=[]
    for source, d in by_source.items():
        rec_tokens = d.get('u256_recovered_tokens', 0.0)
        rows = max(1.0, d.get('rows', 0.0))
        by_source_rows.append({
            'source': source,
            'rows_with_recovered_tail': int(d.get('rows', 0)),
            'u256_recovered_tokens': int(rec_tokens),
            'u256_recovered_full_words': int(d.get('u256_recovered_full_words', 0)),
            'share_of_all_recovered_tokens': rec_tokens / max(1.0, totals.get('u256_recovered_tokens', 0.0)),
            'mean_recovered_tokens_per_tail_row': rec_tokens / rows,
            'cleanish_rows': int(d.get('cleanish_flag', 0)),
            'cleanish_row_fraction': d.get('cleanish_flag', 0.0) / rows,
            'encoding_noise_rows': int(d.get('encoding_noise_flag', 0)),
            'encoding_noise_row_fraction': d.get('encoding_noise_flag', 0.0) / rows,
            'allcaps_dense_rows': int(d.get('allcaps_dense_flag', 0)),
            'allcaps_dense_row_fraction': d.get('allcaps_dense_flag', 0.0) / rows,
            'childes_like_rows': int(d.get('is_childes_like', 0)),
            'subtitle_like_rows': int(d.get('is_subtitle_like', 0)),
            'list_like_rows': int(d.get('is_list_like', 0)),
            'mojibake_marker_count': int(d.get('mojibake_marker_count', 0)),
            'childes_marker_count': int(d.get('childes_marker_count', 0)),
            'subtitle_marker_count': int(d.get('subtitle_marker_count', 0)),
            'list_toc_marker_count': int(d.get('list_toc_marker_count', 0)),
            'long_or_glued_token_count': int(d.get('long_or_glued_token_count', 0)),
            'uppercase_token_fraction_mean_proxy': d.get('uppercase_token_count', 0.0) / max(1.0, d.get('word_tokens', 0.0)),
            'digit_token_fraction_mean_proxy': d.get('digit_token_count', 0.0) / max(1.0, d.get('word_tokens', 0.0)),
        })
    by_source_rows.sort(key=lambda r: (-r['u256_recovered_tokens'], r['source']))
    top_noise = sorted(noise_rows, key=lambda r: (r['noise_score'], r['u256_recovered_tokens']), reverse=True)[:80]

    total_tail_rows = max(1.0, totals.get('rows', 0.0))
    result = {
        'status': 'U256_RECOVERED_TAIL_QUALITY',
        'created_utc': now(),
        'purpose': 'Descriptive mechanism profile of the row suffix text newly visible under U256 but truncated under research row256.',
        'no_gpu_training_or_model_evaluation': True,
        'inputs': {'base_10m': rel(prof.BASE_10M), 'base_sha256': base_sha, 'tokenizer': rel(prof.TOKENIZER_DIR), 'tokenizer_sha256': tok_sha},
        'aggregate': {
            'rows_with_recovered_tail': int(totals.get('rows', 0)),
            'u256_recovered_tokens': int(totals.get('u256_recovered_tokens', 0)),
            'u256_recovered_full_words': int(totals.get('u256_recovered_full_words', 0)),
            'cleanish_rows': int(totals.get('cleanish_flag', 0)),
            'cleanish_row_fraction': totals.get('cleanish_flag', 0.0) / total_tail_rows,
            'encoding_noise_rows': int(totals.get('encoding_noise_flag', 0)),
            'encoding_noise_row_fraction': totals.get('encoding_noise_flag', 0.0) / total_tail_rows,
            'allcaps_dense_rows': int(totals.get('allcaps_dense_flag', 0)),
            'allcaps_dense_row_fraction': totals.get('allcaps_dense_flag', 0.0) / total_tail_rows,
            'childes_like_rows': int(totals.get('is_childes_like', 0)),
            'childes_like_row_fraction': totals.get('is_childes_like', 0.0) / total_tail_rows,
            'subtitle_like_rows': int(totals.get('is_subtitle_like', 0)),
            'subtitle_like_row_fraction': totals.get('is_subtitle_like', 0.0) / total_tail_rows,
            'list_like_rows': int(totals.get('is_list_like', 0)),
            'list_like_row_fraction': totals.get('is_list_like', 0.0) / total_tail_rows,
            'recovered_tokens_per_tail_row': summarize(recovered_token_values),
            'recovered_full_words_per_tail_row': summarize(recovered_word_values),
        },
        'by_source': by_source_rows,
        'top_noise_rows': top_noise,
        'interpretation': {
            'descriptive_only_not_selection': True,
            'mechanism_question': 'Does exposure of source-tail suffixes improve credit flow, or does it add transcript/list/encoding noise that rotates competence?',
            'links_to_endpoint_risks': ['EWoK material/social/quantitative movement', 'Reading loss/recovery', 'GlobalPIQA small-n fragility'],
        },
        'outputs': {'json': rel(OUT_JSON), 'by_source_csv': rel(BY_SOURCE_CSV), 'top_noise_csv': rel(TOP_NOISE_CSV), 'note': rel(NOTE)},
        'elapsed_sec': round(time.time() - t0, 1),
    }
    write_csv(BY_SOURCE_CSV, by_source_rows)
    write_csv(TOP_NOISE_CSV, top_noise)
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    lines=[]
    a=result['aggregate']
    lines.append('# research — U256 recovered-tail quality profile')
    lines.append('')
    lines.append('CPU-only descriptive profile of text newly visible under U256 and hidden under research row256. It must not be used as benchmark-conditioned data selection.')
    lines.append('')
    lines.append(f"- Tail rows: {a['rows_with_recovered_tail']:,}; recovered tokens: {a['u256_recovered_tokens']:,}; recovered full words: {a['u256_recovered_full_words']:,}.")
    lines.append(f"- Cleanish rows by heuristic: {a['cleanish_rows']:,} ({a['cleanish_row_fraction']:.2%}); encoding-noise rows: {a['encoding_noise_rows']:,} ({a['encoding_noise_row_fraction']:.2%}); all-caps/glued rows: {a['allcaps_dense_rows']:,} ({a['allcaps_dense_row_fraction']:.2%}).")
    lines.append(f"- CHILDES-like rows: {a['childes_like_rows']:,} ({a['childes_like_row_fraction']:.2%}); subtitle-like rows: {a['subtitle_like_rows']:,} ({a['subtitle_like_row_fraction']:.2%}); list/TOC-like rows: {a['list_like_rows']:,} ({a['list_like_row_fraction']:.2%}).")
    lines.append('')
    lines.append('## By source')
    for r in by_source_rows:
        lines.append(f"- {r['source']}: {r['u256_recovered_tokens']:,} recovered tokens ({r['share_of_all_recovered_tokens']:.2%} of all), tail rows {r['rows_with_recovered_tail']:,}, cleanish {r['cleanish_row_fraction']:.2%}, encoding-noise {r['encoding_noise_row_fraction']:.2%}, allcaps/glued {r['allcaps_dense_row_fraction']:.2%}.")
    lines.append('')
    lines.append('Scientific consequence: U256 is not just a uniform token-count increase. It exposes concentrated row tails dominated by CHILDES, SimpleWiki, OpenSubtitles, and Gutenberg; a nontrivial part contains transcript, list, all-caps/glued, or encoding artifacts. Endpoint interpretation should distinguish useful long-row suffix credit from noise-driven competence rotation.')
    lines.append('')
    lines.append(f"JSON: `{rel(OUT_JSON)}`")
    lines.append(f"CSV: `{rel(BY_SOURCE_CSV)}`, `{rel(TOP_NOISE_CSV)}`")
    NOTE.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({
        'status': result['status'],
        'tail_rows': a['rows_with_recovered_tail'],
        'recovered_tokens': a['u256_recovered_tokens'],
        'cleanish_fraction': a['cleanish_row_fraction'],
        'encoding_noise_fraction': a['encoding_noise_row_fraction'],
        'allcaps_dense_fraction': a['allcaps_dense_row_fraction'],
        'out_json': rel(OUT_JSON),
        'note': rel(NOTE),
        'elapsed_sec': result['elapsed_sec'],
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
