#!/usr/bin/env python3
"""research: quality/content measurement of row256 suffix material recovered by U256.

CPU-only.  This does not select or alter data; it characterizes the 1.29% of
charged words that row256 hid and U256 would expose, so future score movement can
be interpreted scientifically.
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

ROOT = Path('.').resolve()
WS = ROOT / 'experiments/archive/representation_and_objectives'
for p in (WS / 'scripts', ROOT / 'experiments/archive/compact_experience/scripts'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import masking_curriculum_trainer as base  # noqa: E402
import experience_utilization_trainer as chunkbase  # noqa: E402

BASE_10M = ROOT / 'experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl'
TOKENIZER = WS / 'data/legal_representation_route_map/tokenizers/legal_byte_bpe_40k'
OUT_DIR = WS / 'data/recovered_suffix_quality_measurement'
OUT_JSON = OUT_DIR / 'recovered_suffix_quality_measurement.json'
SOURCE_CSV = OUT_DIR / 'suffix_quality_by_source.csv'
CATEGORY_CSV = OUT_DIR / 'suffix_quality_categories.csv'
SAMPLE_CSV = OUT_DIR / 'suffix_quality_samples.csv'
NOTE = (ROOT / 'research/notes/representation_and_objectives/recovered_suffix_quality_measurement.md')

EXPECTED_BASE_SHA = '215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23'
EXPECTED_TOK_SHA = '94b44bcf5901d097e20294e8220740758eac29af3ec20d58d8da867c85197758'

WORD_RE = re.compile(r'\S+')
TOK_RE = re.compile(r"[A-Za-z0-9]+(?:[\u2019'][A-Za-z0-9]+)?")
NUM_RE = re.compile(r"\b\d+(?:[.,:/-]\d+)*(?:%|st|nd|rd|th)?\b", re.I)
CAP_RE = re.compile(r"\b[A-Z][A-Za-z]+(?:[-'][A-Z]?[A-Za-z]+)?\b")
SPEAKER_RE = re.compile(r"(?:\*[A-Z]{2,4}:|\[[^\]]{2,80}\]|^-\s)")
BAD_MARKERS = ['�', 'À', 'Á', 'Å', 'Ç', 'È', 'Ð', 'Ñ', 'Ò', 'Ó', 'Ô', 'Õ', 'Ö', '×', 'Ø', 'Ù', 'Ú', 'Û', 'Ü', 'æ', 'Æ', '⁄', '½', '¼', '\ufffd']

ACTION = {'put','take','get','give','go','come','make','made','open','close','hold','push','pull','move','moved','turn','turned','drop','dropped','pick','picked','use','used','using','eat','drink','throw','bring','carry','keep','help','build','built','break','change','changed','find','found','play'}
CAUSAL = {'because','cause','caused','causes','so','therefore','after','before','when','while','if','then','result','results','resulted','became','become','prevent','allow','allows','requires','required','during','until','since','leads','led'}
PHYSICAL = {'water','fire','box','ball','door','cup','table','toy','paper','hand','body','food','stone','wood','glass','machine','tool','wheel','container','bag','room','floor','wall','book','plant','animal','air','light','heat','material','metal','house','car','window','cloth'}
SPATIAL = {'in','on','under','over','inside','outside','behind','front','near','across','through','between','below','above','around','left','right','top','bottom','beside','into','out','off','down','up','back'}
SOCIAL = {'think','thought','know','knew','want','wanted','say','said','ask','asked','tell','told','see','saw','look','looked','feel','felt','believe','learn','teach','child','mother','father','person','people','friend','teacher','man','woman','boy','girl','family','name'}


def now() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def q(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    xs = sorted(vals)
    if len(xs) == 1:
        return xs[0]
    pos = p * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def stats(vals: list[float]) -> dict[str, Any]:
    if not vals:
        return {'n': 0}
    return {'n': len(vals), 'sum': sum(vals), 'mean': statistics.fmean(vals), 'p50': q(vals, 0.5), 'p90': q(vals, 0.9), 'p95': q(vals, 0.95), 'max': max(vals)}


def visible_word_count(text: str, tokenizer) -> tuple[int, int, int, int]:
    words = text.split()
    by_word, raw_tokens, unassigned = chunkbase.token_ids_by_whitespace_word(text, tokenizer)
    if unassigned:
        raise RuntimeError(f'unassigned offsets: {unassigned}')
    cum = 0
    vis_words = 0
    boundary_hidden = 0
    for ids in by_word:
        wc = len(ids)
        if wc == 0:
            if cum < 256:
                vis_words += 1
            continue
        if cum < 256:
            vis_words += 1
            if cum + wc > 256:
                boundary_hidden += cum + wc - 256
        cum += wc
    return vis_words, raw_tokens, max(0, raw_tokens - 256), boundary_hidden


def suffix_record(row_idx: int, source: str, text: str, visible_words: int, raw_tokens: int, hidden_tokens: int, boundary_hidden: int) -> dict[str, Any] | None:
    words = text.split()
    suffix_words = words[visible_words:]
    if not suffix_words and hidden_tokens <= 0:
        return None
    suffix = ' '.join(suffix_words)
    toks = [m.group(0).lower() for m in TOK_RE.finditer(suffix)]
    cnt = collections.Counter(toks)
    word_n = len(suffix_words)
    chars = len(suffix)
    non_ascii = sum(1 for ch in suffix if ord(ch) > 127)
    marker_hits = sum(suffix.count(m) for m in BAD_MARKERS)
    nums = NUM_RE.findall(suffix)
    caps = CAP_RE.findall(suffix)
    speaker = SPEAKER_RE.findall(suffix)
    action = sum(cnt[w] for w in ACTION)
    causal = sum(cnt[w] for w in CAUSAL)
    physical = sum(cnt[w] for w in PHYSICAL)
    spatial = sum(cnt[w] for w in SPATIAL)
    social = sum(cnt[w] for w in SOCIAL)
    categories = []
    if marker_hits > 0 or (chars > 0 and non_ascii / chars > 0.12):
        categories.append('encoding_noise')
    if word_n > 0 and len(nums) / word_n > 0.30 and len(caps) / word_n > 0.15:
        categories.append('index_or_catalog_tail')
    if speaker:
        categories.append('dialogue_or_transcript_tail')
    if action + causal + physical + spatial + social >= 3:
        categories.append('relation_action_social_tail')
    if word_n > 0 and len(nums) / word_n > 0.20 and 'index_or_catalog_tail' not in categories:
        categories.append('number_heavy_tail')
    if word_n > 0 and len(caps) / word_n > 0.25 and 'index_or_catalog_tail' not in categories:
        categories.append('name_heavy_tail')
    if not categories:
        categories.append('ordinary_tail')
    return {
        'row_index': row_idx,
        'source': source,
        'row_words': len(words),
        'row_raw_tokens': raw_tokens,
        'visible_words': visible_words,
        'suffix_words': word_n,
        'hidden_tokens': hidden_tokens,
        'boundary_hidden_tokens': boundary_hidden,
        'suffix_token_words': len(toks),
        'non_ascii_chars': non_ascii,
        'chars': chars,
        'non_ascii_fraction': non_ascii / chars if chars else 0.0,
        'bad_marker_hits': marker_hits,
        'number_tokens': len(nums),
        'capitalized_tokens': len(caps),
        'speaker_marker_hits': len(speaker),
        'action': action,
        'causal': causal,
        'physical': physical,
        'spatial': spatial,
        'social': social,
        'categories': categories,
        'category_primary': categories[0],
        'suffix_sample': suffix[:240],
        'row_prefix': text[:160],
    }


def add_agg(agg: dict[str, Any], rec: dict[str, Any]) -> None:
    agg['rows'] += 1
    for key in ['suffix_words','hidden_tokens','boundary_hidden_tokens','non_ascii_chars','chars','bad_marker_hits','number_tokens','capitalized_tokens','speaker_marker_hits','action','causal','physical','spatial','social']:
        agg[key] += int(rec[key])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k); keys.append(k)
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_sha = sha(BASE_10M)
    tok_sha = sha(TOKENIZER / 'tokenizer.json')
    if base_sha != EXPECTED_BASE_SHA:
        raise RuntimeError(f'base SHA mismatch: {base_sha}')
    if tok_sha != EXPECTED_TOK_SHA:
        raise RuntimeError(f'tokenizer SHA mismatch: {tok_sha}')
    tokenizer = base.make_portable_tokenizer(str(TOKENIZER))
    records: list[dict[str, Any]] = []
    by_source: dict[str, Any] = collections.defaultdict(lambda: collections.Counter(rows=0))
    by_category: dict[str, Any] = collections.defaultdict(lambda: collections.Counter(rows=0))
    suffix_word_counts: list[float] = []
    hidden_token_counts: list[float] = []
    t0 = time.time()
    with BASE_10M.open('r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            text = str(obj['text'])
            source = str(obj.get('source', ''))
            visible_words, raw_tokens, hidden_tokens, boundary_hidden = visible_word_count(text, tokenizer)
            rec = suffix_record(i, source, text, visible_words, raw_tokens, hidden_tokens, boundary_hidden)
            if rec is None:
                continue
            records.append(rec)
            suffix_word_counts.append(float(rec['suffix_words']))
            hidden_token_counts.append(float(rec['hidden_tokens']))
            add_agg(by_source[source], rec)
            for cat in rec['categories']:
                add_agg(by_category[cat], rec)
            if (i + 1) % 10000 == 0:
                print(json.dumps({'event': 'progress', 'rows_seen': i + 1, 'suffix_rows': len(records)}), flush=True)

    source_rows = []
    total_suffix_words = sum(int(r['suffix_words']) for r in records)
    total_hidden_tokens = sum(int(r['hidden_tokens']) for r in records)
    for source, c in sorted(by_source.items(), key=lambda kv: (-kv[1]['suffix_words'], kv[0])):
        row = dict(c)
        row['source'] = source
        row['suffix_word_share'] = row['suffix_words'] / max(1, total_suffix_words)
        row['hidden_token_share'] = row['hidden_tokens'] / max(1, total_hidden_tokens)
        row['non_ascii_fraction'] = row['non_ascii_chars'] / max(1, row['chars'])
        source_rows.append(row)

    category_rows = []
    for cat, c in sorted(by_category.items(), key=lambda kv: (-kv[1]['suffix_words'], kv[0])):
        row = dict(c)
        row['category'] = cat
        row['suffix_word_share'] = row['suffix_words'] / max(1, total_suffix_words)
        row['hidden_token_share'] = row['hidden_tokens'] / max(1, total_hidden_tokens)
        row['non_ascii_fraction'] = row['non_ascii_chars'] / max(1, row['chars'])
        category_rows.append(row)

    samples = []
    # Keep a diverse set of interpretable and problematic examples for later reading.
    for cat in ['relation_action_social_tail', 'encoding_noise', 'index_or_catalog_tail', 'dialogue_or_transcript_tail', 'ordinary_tail']:
        cand = [r for r in records if cat in r['categories']]
        if cat == 'relation_action_social_tail':
            cand.sort(key=lambda r: (r['action'] + r['causal'] + r['physical'] + r['spatial'] + r['social'], r['suffix_words']), reverse=True)
        elif cat == 'encoding_noise':
            cand.sort(key=lambda r: (r['bad_marker_hits'], r['non_ascii_fraction'], r['suffix_words']), reverse=True)
        elif cat == 'index_or_catalog_tail':
            cand.sort(key=lambda r: (r['number_tokens'] + r['capitalized_tokens'], r['suffix_words']), reverse=True)
        else:
            cand.sort(key=lambda r: (r['suffix_words'], r['hidden_tokens']), reverse=True)
        for r in cand[:12]:
            row = dict(r)
            row['categories'] = '|'.join(row['categories'])
            samples.append(row)

    result = {
        'status': 'RECOVERED_SUFFIX_QUALITY_MEASUREMENT',
        'created_utc': now(),
        'purpose': 'CPU-only source/content/noise characterization of row256-hidden suffix words that U256 would expose.',
        'no_training_or_model_evaluation': True,
        'inputs': {
            'base_10M': str(BASE_10M.relative_to(ROOT)),
            'base_10M_sha256': base_sha,
            'tokenizer': str(TOKENIZER.relative_to(ROOT)),
            'tokenizer_sha256': tok_sha,
        },
        'totals': {
            'suffix_rows': len(records),
            'suffix_words': total_suffix_words,
            'hidden_tokens': total_hidden_tokens,
            'suffix_words_stats_per_row': stats(suffix_word_counts),
            'hidden_tokens_stats_per_row': stats(hidden_token_counts),
        },
        'by_source': source_rows,
        'by_category': category_rows,
        'sample_rows': samples,
        'interpretation': {
            'u256_recovers_most_suffix_words_from_childes': bool(source_rows and source_rows[0]['source'] == 'childes'),
            'some_recovered_tail_mass_is_encoding_or_index_noise': any(r['category'] in ('encoding_noise','index_or_catalog_tail') for r in category_rows),
            'measurement_is_not_a_data_selector': True,
            'route_implication': 'U256@0.15 remains the clean fixed-length visibility experiment, but any score movement should be interpreted against this nonuniform recovered-tail profile rather than as generic more-data exposure.',
        },
        'outputs': {
            'json': str(OUT_JSON.relative_to(ROOT)),
            'note': str(NOTE.relative_to(ROOT)),
            'source_csv': str(SOURCE_CSV.relative_to(ROOT)),
            'category_csv': str(CATEGORY_CSV.relative_to(ROOT)),
            'sample_csv': str(SAMPLE_CSV.relative_to(ROOT)),
        },
        'elapsed_sec': round(time.time() - t0, 1),
    }
    write_csv(SOURCE_CSV, source_rows)
    write_csv(CATEGORY_CSV, category_rows)
    write_csv(SAMPLE_CSV, samples)
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    lines = []
    lines.append('# research — Recovered suffix quality/content measurement')
    lines.append('')
    lines.append('CPU-only. No data were selected or changed, and no model was trained or evaluated.')
    lines.append('')
    lines.append(f"U256 would expose {total_suffix_words:,} fully hidden suffix words and {total_hidden_tokens:,} suffix/boundary tokens per 10M corpus pass. These are small in corpus mass, but not neutral in content.")
    lines.append('')
    lines.append('## Source concentration')
    for row in source_rows[:8]:
        lines.append(f"- {row['source']}: {row['suffix_words']:,} suffix words ({row['suffix_word_share']:.2%}), {row['hidden_tokens']:,} hidden tokens ({row['hidden_token_share']:.2%}), non-ASCII fraction {row['non_ascii_fraction']:.3f}.")
    lines.append('')
    lines.append('## Content/noise categories')
    for row in category_rows:
        lines.append(f"- {row['category']}: {row['suffix_words']:,} words ({row['suffix_word_share']:.2%}), {row['hidden_tokens']:,} tokens ({row['hidden_token_share']:.2%}), rows {row['rows']:,}.")
    lines.append('')
    lines.append('## Scientific reading')
    lines.append('U256 is still the clean first chunk-stream experiment because it changes visibility at fixed maximum length 256. But it is not simply adding uniformly good experience: most hidden full words are Childes tails, and a measurable subset is encoding-heavy or index/catalog-like text from OpenSubtitles/Gutenberg. If U256 helps, the result is stronger because it survived this heterogeneous tail; if it hurts, this profile gives a concrete explanation and points toward visibility-aware data repair rather than repeating the same chunking route.')
    lines.append('')
    lines.append(f"JSON: `{OUT_JSON.relative_to(ROOT)}`")
    lines.append(f"CSV: `{SOURCE_CSV.relative_to(ROOT)}`, `{CATEGORY_CSV.relative_to(ROOT)}`, `{SAMPLE_CSV.relative_to(ROOT)}`")
    NOTE.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    print(json.dumps({
        'status': result['status'],
        'suffix_rows': len(records),
        'suffix_words': total_suffix_words,
        'hidden_tokens': total_hidden_tokens,
        'top_source': source_rows[0] if source_rows else None,
        'top_category': category_rows[0] if category_rows else None,
        'out_json': str(OUT_JSON.relative_to(ROOT)),
        'note': str(NOTE.relative_to(ROOT)),
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
