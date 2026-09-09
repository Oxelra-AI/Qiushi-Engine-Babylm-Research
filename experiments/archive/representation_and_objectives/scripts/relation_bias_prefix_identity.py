#!/usr/bin/env python3
"""research: exact 10M-pool vs 100M first-pass identity check.

No model work. Reads the exact 10M compact_view_reinvest pool and the first
10,000,000 charged words of the 100M launch stream used by trainers.  The 100M
stream is allowed to be a permutation, so this checks row-object multiset equality,
not sequence equality.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import time
from typing import Any

ROOT = Path('.').resolve()
WS = ROOT / 'experiments/archive/representation_and_objectives'
POOL10 = ROOT / 'experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl'
STREAM100 = ROOT / 'experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl'
OUT_DIR = WS / 'data/relation_bias_prefix_identity'
OUT_JSON = OUT_DIR / 'relation_bias_prefix_identity.json'
NOTE = (ROOT / 'research/notes/representation_and_objectives/relation_bias_prefix_identity.md')
EXPECTED_POOL10_SHA = '215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23'
EXPECTED_STREAM100_SHA = '3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691'


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def now_utc() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def row_fingerprint(obj: dict[str, Any]) -> str:
    blob = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(blob).hexdigest()


def counter_digest(counter: Counter[str]) -> str:
    h = hashlib.sha256()
    for k, v in sorted(counter.items()):
        h.update(k.encode('ascii')); h.update(b':'); h.update(str(v).encode('ascii')); h.update(b'\n')
    return h.hexdigest()


def read_full_pool(path: Path) -> dict[str, Any]:
    c: Counter[str] = Counter()
    rows = 0; words = 0
    source_words: dict[str, int] = defaultdict(int)
    source_rows: dict[str, int] = defaultdict(int)
    sample = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = str(obj['text'])
            w = int(obj.get('words', len(text.split())))
            if len(text.split()) != w:
                raise RuntimeError({'path': rel(path), 'row': rows, 'word_mismatch': [len(text.split()), w]})
            fp = row_fingerprint(obj)
            c[fp] += 1
            rows += 1; words += w
            src = str(obj.get('source', 'unknown'))
            source_words[src] += w; source_rows[src] += 1
            if len(sample) < 5:
                sample.append({k: v for k, v in obj.items() if k != 'text'})
    return {'counter': c, 'rows': rows, 'words': words, 'source_words': dict(source_words), 'source_rows': dict(source_rows), 'sample': sample}


def read_stream_prefix(path: Path, target_words: int) -> dict[str, Any]:
    c: Counter[str] = Counter()
    rows = 0; words = 0
    source_words: dict[str, int] = defaultdict(int)
    source_rows: dict[str, int] = defaultdict(int)
    sample = []
    first_overrun = None
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = str(obj['text'])
            w = int(obj.get('words', len(text.split())))
            if len(text.split()) != w:
                raise RuntimeError({'path': rel(path), 'row': rows, 'word_mismatch': [len(text.split()), w]})
            if words + w > target_words:
                first_overrun = {'row': rows, 'words_before': words, 'row_words': w, 'target_words': target_words}
                break
            fp = row_fingerprint(obj)
            c[fp] += 1
            rows += 1; words += w
            src = str(obj.get('source', 'unknown'))
            source_words[src] += w; source_rows[src] += 1
            if len(sample) < 5:
                sample.append({k: v for k, v in obj.items() if k != 'text'})
            if words == target_words:
                break
    return {'counter': c, 'rows': rows, 'words': words, 'source_words': dict(source_words), 'source_rows': dict(source_rows), 'sample': sample, 'first_overrun': first_overrun}


def main() -> None:
    start = time.time()
    pool = read_full_pool(POOL10)
    prefix = read_stream_prefix(STREAM100, 10_000_000)
    pc = pool.pop('counter')
    sc = prefix.pop('counter')
    stream_minus_pool = sc - pc
    pool_minus_stream = pc - sc
    payload = {
        'status': 'RELATION_BIAS_PREFIX_IDENTITY',
        'created_utc': now_utc(),
        'elapsed_sec': round(time.time() - start, 1),
        'purpose': 'Exact no-model check that the 100M launch stream first 10M charged-word pass is a permutation of the audited legal 10M pool.',
        'inputs': {
            'pool10': rel(POOL10),
            'pool10_sha256': sha256_file(POOL10),
            'expected_pool10_sha256': EXPECTED_POOL10_SHA,
            'stream100': rel(STREAM100),
            'stream100_sha256': sha256_file(STREAM100),
            'expected_stream100_sha256': EXPECTED_STREAM100_SHA,
            'target_prefix_words': 10_000_000,
        },
        'pool10': pool,
        'stream_prefix': prefix,
        'row_multiset_equal': sc == pc,
        'row_multiset_sha_pool10': counter_digest(pc),
        'row_multiset_sha_stream_prefix': counter_digest(sc),
        'stream_minus_pool_row_count': sum(stream_minus_pool.values()),
        'pool_minus_stream_row_count': sum(pool_minus_stream.values()),
        'stream_only_fingerprints_sample': list(stream_minus_pool.keys())[:10],
        'pool_only_fingerprints_sample': list(pool_minus_stream.keys())[:10],
        'source_words_equal': pool['source_words'] == prefix['source_words'],
        'source_rows_equal': pool['source_rows'] == prefix['source_rows'],
        'checks': {
            'pool10_sha_matches': sha256_file(POOL10) == EXPECTED_POOL10_SHA,
            'stream100_sha_matches': sha256_file(STREAM100) == EXPECTED_STREAM100_SHA,
            'pool_counts_64740_10M': pool['rows'] == 64740 and pool['words'] == 10_000_000,
            'prefix_counts_64740_10M': prefix['rows'] == 64740 and prefix['words'] == 10_000_000,
            'row_multiset_equal': sc == pc,
            'source_words_equal': pool['source_words'] == prefix['source_words'],
        },
        'files': {'json': rel(OUT_JSON), 'note': rel(NOTE)},
        'interpretation': [],
    }
    if payload['row_multiset_equal']:
        payload['interpretation'].append('The 100M launch stream first 10M charged words are exactly a row-object multiset permutation of the audited 10M pool; research full-pool relation-dose statistics describe the actual first training pass, though not its sequence order.')
    else:
        payload['interpretation'].append('The 100M launch stream prefix is not the same row-object multiset as the audited 10M pool; a future relation-biased launch would need repair before interpretation.')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '# research relation-bias prefix identity\n\n',
        'No-model check comparing the exact 10M pool against the first 10M charged words consumed from the 100M launch stream.\n\n',
        f"- Pool SHA matched expected: `{payload['checks']['pool10_sha_matches']}`; stream SHA matched expected: `{payload['checks']['stream100_sha_matches']}`.\n",
        f"- Pool rows/words: `{pool['rows']}` / `{pool['words']}`; stream prefix rows/words: `{prefix['rows']}` / `{prefix['words']}`.\n",
        f"- Row-object multiset equal: `{payload['row_multiset_equal']}`; source words equal: `{payload['source_words_equal']}`.\n",
        f"- Stream-only row count: `{payload['stream_minus_pool_row_count']}`; pool-only row count: `{payload['pool_minus_stream_row_count']}`.\n\n",
        '## Interpretation\n\n',
    ]
    for item in payload['interpretation']:
        lines.append(f'- {item}\n')
    lines.append(f"\nJSON: `{rel(OUT_JSON)}`\n")
    NOTE.write_text(''.join(lines), encoding='utf-8')
    print(json.dumps({'status': payload['status'], 'elapsed_sec': payload['elapsed_sec'], 'row_multiset_equal': payload['row_multiset_equal'], 'pool_rows_words': [pool['rows'], pool['words']], 'prefix_rows_words': [prefix['rows'], prefix['words']], 'json': rel(OUT_JSON), 'note': rel(NOTE)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
