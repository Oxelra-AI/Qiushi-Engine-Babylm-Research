#!/usr/bin/env python3
"""research: CPU support-sharing potential for the legal representation corner.

This does not train or evaluate a model.  It measures whether low-count legal40k
BPE tokens can be decomposed into higher-support legal16k pieces learned from the
same allowed 10M corpus.  The result informs a possible support-aware embedding
or rare-token sharing construction if the pending depth and minfreq50 vectors
show complementary recovery.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from tokenizers import Tokenizer

ROOT = Path('.').resolve()
WS = ROOT / 'experiments/archive/representation_and_objectives'
if str(WS / 'scripts') not in sys.path:
    sys.path.insert(0, str(WS / 'scripts'))
import tokenizer_support_spectrum as supportbase  # noqa: E402

SUPPORT_JSON = WS / 'data/tokenizer_support_spectrum/tokenizer_support_spectrum.json'
LEGAL40K = WS / 'data/legal_representation_route_map/tokenizers/legal_byte_bpe_40k'
LEGAL16K = WS / 'data/strictsmall_tokenizer_retrain/strictsmall_compact_reinvest_16k_tokenizer'
LEGAL40K_TWO_SEED = WS / 'data/legal40k_two_seed_comparison/legal40k_two_seed_comparison.json'
OUT_DIR = WS / 'data/support_shared_token_rescue'
OUT_JSON = OUT_DIR / 'support_shared_token_rescue.json'
TOKEN_CSV = OUT_DIR / 'rare40k_token_component_support.csv'
EVAL_CSV = OUT_DIR / 'eval_family_component_rescue_summary.csv'
NOTE = (ROOT / 'research/notes/representation_and_objectives/support_shared_token_rescue.md')

EXPECTED_40K_SHA = '94b44bcf5901d097e20294e8220740758eac29af3ec20d58d8da867c85197758'
EXPECTED_16K_SHA = '4a95a2a2ead21813a409c5ad7c2c008fbf418dcf3e0dc17bd970bd9ce25ea738'
THRESHOLDS = [20, 50, 100, 200]
BATCH = 512
LEADER = {
    'BLiMP': 67.20,
    'Supplement': 56.01,
    'EWoK': 56.07,
    'Entity': 28.45,
    'COMPS': 53.57,
    'GlobalPIQA': 39.67,
    'SuperGLUE': 69.79,
    'Reading': 5.42,
    'AoA': 0.0,
    'Overall': 41.80,
}


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
    lo = int(math.floor(pos)); hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def finite_float(x: Any) -> float | None:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def geo_mean(vals: list[int]) -> float | None:
    xs = [max(1, int(v)) for v in vals]
    if not xs:
        return None
    return float(math.exp(sum(math.log(v) for v in xs) / len(xs)))


def read_support() -> dict[str, Any]:
    payload = json.loads(SUPPORT_JSON.read_text(encoding='utf-8'))
    return payload['pool_support']


def load_tok(path: Path) -> Tokenizer:
    tok = Tokenizer.from_file(str(path / 'tokenizer.json'))
    try:
        tok.no_padding()
    except Exception:
        pass
    try:
        tok.no_truncation()
    except Exception:
        pass
    return tok


def special_ids(rec: dict[str, Any]) -> set[int]:
    return {int(v) for v in rec['special_ids'].values() if v is not None}


def decode_piece(tok: Tokenizer, token_id: int, token_text: str) -> str:
    try:
        s = tok.decode([token_id], skip_special_tokens=False)
        if s:
            return s
    except Exception:
        pass
    # Fallback keeps the raw BPE surface only when decoder cannot produce text.
    return token_text


def component_map() -> tuple[dict[int, dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    # research's durable JSON intentionally strips heavy counts/id-to-token maps.
    # Recompute them from the exact allowed 10M pool here so the measurement is
    # independent and still uses no score text for tokenizer training.
    tok40 = load_tok(LEGAL40K)
    tok16 = load_tok(LEGAL16K)
    rec40 = supportbase.count_pool_support('legal_byte_bpe_40k_step82_recount', tok40)
    rec16 = supportbase.count_pool_support('legal_a01_16k_step82_recount', tok16)
    counts40 = [int(x) for x in rec40['counts_by_id']]
    counts16 = [int(x) for x in rec16['counts_by_id']]
    idtok40 = {int(k): v for k, v in rec40['id_to_token'].items()}
    idtok16 = {int(k): v for k, v in rec16['id_to_token'].items()}
    specials40 = special_ids(rec40)
    cmap: dict[int, dict[str, Any]] = {}
    rare_rows: list[dict[str, Any]] = []
    by_count_band = collections.defaultdict(lambda: collections.Counter(total=0))
    for token_id, token_text in sorted(idtok40.items()):
        if token_id in specials40:
            continue
        direct = counts40[token_id] if token_id < len(counts40) else 0
        surface = decode_piece(tok40, token_id, token_text)
        enc16 = tok16.encode(surface, add_special_tokens=False)
        comp_ids = [int(x) for x in enc16.ids]
        comp_counts = [counts16[x] if x < len(counts16) else 0 for x in comp_ids]
        comp_tokens = [idtok16.get(x, '') for x in comp_ids]
        cmin = min(comp_counts) if comp_counts else 0
        csum = sum(comp_counts)
        cgeo = geo_mean(comp_counts)
        comp = {
            'token_id': token_id,
            'token': token_text,
            'surface': surface,
            'direct_count40': direct,
            'component_ids16': comp_ids,
            'component_tokens16': comp_tokens,
            'component_counts16': comp_counts,
            'component_n16': len(comp_ids),
            'component_min_count16': cmin,
            'component_sum_count16': csum,
            'component_geo_count16': cgeo,
            'component_all_ge50': bool(comp_counts and cmin >= 50),
            'component_all_ge100': bool(comp_counts and cmin >= 100),
            'component_all_ge200': bool(comp_counts and cmin >= 200),
        }
        cmap[token_id] = comp
        if direct < 20:
            band = 'lt20'
        elif direct < 50:
            band = '20_49'
        elif direct < 100:
            band = '50_99'
        elif direct < 200:
            band = '100_199'
        else:
            band = 'ge200'
        b = by_count_band[band]
        b['total'] += 1
        b['comp_all_ge50'] += int(comp['component_all_ge50'])
        b['comp_all_ge100'] += int(comp['component_all_ge100'])
        b['comp_all_ge200'] += int(comp['component_all_ge200'])
        if direct < 100 or (direct < 200 and cmin >= 500):
            rare_rows.append({
                'token_id': token_id,
                'token': token_text,
                'surface_repr': repr(surface),
                'direct_count40': direct,
                'component_n16': len(comp_ids),
                'component_min_count16': cmin,
                'component_geo_count16': cgeo,
                'component_sum_count16': csum,
                'component_tokens16': ' '.join(comp_tokens[:16]),
                'component_counts16': ' '.join(str(x) for x in comp_counts[:16]),
            })
    summary_bands = {}
    for band, c in by_count_band.items():
        total = max(1, c['total'])
        summary_bands[band] = {
            'tokens': c['total'],
            'component_all_ge50_frac': c['comp_all_ge50'] / total,
            'component_all_ge100_frac': c['comp_all_ge100'] / total,
            'component_all_ge200_frac': c['comp_all_ge200'] / total,
        }
    rare_rows.sort(key=lambda r: (int(r['direct_count40']), -int(r['component_min_count16']), str(r['token'])))
    return cmap, {'by_direct_count_band': summary_bands}, rare_rows


def consume_eval_batch(tok40: Tokenizer, cmap: dict[int, dict[str, Any]], acc: dict[str, collections.Counter], token_freq: collections.Counter[tuple[str, int]], batch: list[tuple[str, str]]) -> None:
    encs = tok40.encode_batch([x[1] for x in batch], add_special_tokens=False)
    for (family, _text), enc in zip(batch, encs):
        c = acc[family]
        for tid in enc.ids:
            tid = int(tid)
            comp = cmap.get(tid)
            if comp is None:
                continue
            direct = int(comp['direct_count40'])
            c['tokens'] += 1
            token_freq[(family, tid)] += 1
            for th in THRESHOLDS:
                if direct < th:
                    c[f'direct_lt{th}'] += 1
                    # For th=50 this is the same counter as the generic
                    # ge50-within-lt50 monitor, so count it only once.
                    if comp['component_min_count16'] >= th:
                        c[f'rescued_min_ge{th}_within_direct_lt{th}'] += 1
                    if th != 50 and comp['component_min_count16'] >= 50:
                        c[f'rescued_min_ge50_within_direct_lt{th}'] += 1
                    if th != 100 and comp['component_min_count16'] >= 100:
                        c[f'rescued_min_ge100_within_direct_lt{th}'] += 1
            if direct < 100:
                c['component_n_sum_for_direct_lt100'] += int(comp['component_n16'])
                c['component_min_sum_for_direct_lt100'] += int(comp['component_min_count16'])


def eval_summary(cmap: dict[int, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tok40 = load_tok(LEGAL40K)
    acc: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    token_freq: collections.Counter[tuple[str, int]] = collections.Counter()
    batch: list[tuple[str, str]] = []
    for family, text in supportbase.iter_eval_texts():
        batch.append((family, text))
        if len(batch) >= BATCH:
            consume_eval_batch(tok40, cmap, acc, token_freq, batch)
            batch = []
    if batch:
        consume_eval_batch(tok40, cmap, acc, token_freq, batch)
    rows: list[dict[str, Any]] = []
    for family in sorted(acc):
        c = acc[family]
        total = max(1, c['tokens'])
        row: dict[str, Any] = {'family': family, 'eval_tokens': int(c['tokens'])}
        for th in THRESHOLDS:
            dlt = int(c[f'direct_lt{th}'])
            row[f'direct_frac_lt{th}'] = dlt / total
            row[f'rescued_min_ge{th}_frac_of_all'] = int(c[f'rescued_min_ge{th}_within_direct_lt{th}']) / total
            row[f'rescued_min_ge{th}_frac_of_low'] = int(c[f'rescued_min_ge{th}_within_direct_lt{th}']) / max(1, dlt)
            row[f'rescued_min_ge50_within_lt{th}_frac_of_low'] = int(c[f'rescued_min_ge50_within_direct_lt{th}']) / max(1, dlt)
        row['direct_lt100_mean_component_n'] = c['component_n_sum_for_direct_lt100'] / max(1, c['direct_lt100'])
        row['direct_lt100_mean_component_min_support'] = c['component_min_sum_for_direct_lt100'] / max(1, c['direct_lt100'])
        rows.append(row)

    top_rows: list[dict[str, Any]] = []
    for (family, tid), freq in token_freq.most_common():
        comp = cmap[tid]
        if int(comp['direct_count40']) >= 100:
            continue
        top_rows.append({
            'family': family,
            'eval_frequency': int(freq),
            'token_id': tid,
            'token': comp['token'],
            'surface_repr': repr(comp['surface']),
            'direct_count40': int(comp['direct_count40']),
            'component_n16': int(comp['component_n16']),
            'component_min_count16': int(comp['component_min_count16']),
            'component_geo_count16': comp['component_geo_count16'],
            'component_tokens16': ' '.join(comp['component_tokens16'][:16]),
            'component_counts16': ' '.join(str(x) for x in comp['component_counts16'][:16]),
        })
        if len(top_rows) >= 200:
            break
    return rows, top_rows


def score_context() -> dict[str, Any]:
    payload = json.loads(LEGAL40K_TWO_SEED.read_text(encoding='utf-8'))
    base = payload['by_seed']['43022']['legal40k']
    return {
        'legal40k_8x480_seed43022': base,
        'gap_to_leader': {k: LEADER[k] - base[k] for k in LEADER if k in base},
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def fmt(x: Any, nd=4) -> str:
    v = finite_float(x)
    if v is None:
        return 'NA'
    return f'{v:.{nd}f}'


def write_note(payload: dict[str, Any]) -> None:
    rows = payload['eval_family_summary']
    byfam = {r['family']: r for r in rows}
    bands = payload['vocab_component_summary']['by_direct_count_band']
    ctx = payload['score_context']
    lines: list[str] = []
    lines.append('# research — Support-shared rare-token representation measurement')
    lines.append('')
    lines.append('CPU-only. No training, no model evaluation, and no managed-task query. This measures whether the legal40k low-support-token problem can be attacked by sharing representation with legal16k components learned from the same allowed 10M corpus.')
    lines.append('')
    lines.append('## Vocabulary-level decomposition')
    lines.append('')
    lines.append('| direct legal40k count band | tokens | all legal16 components >=50 | all components >=100 | all components >=200 |')
    lines.append('|---|---:|---:|---:|---:|')
    for band in ['lt20', '20_49', '50_99', '100_199', 'ge200']:
        b = bands.get(band, {'tokens': 0})
        lines.append(f"| {band} | {int(b.get('tokens', 0))} | {fmt(b.get('component_all_ge50_frac'))} | {fmt(b.get('component_all_ge100_frac'))} | {fmt(b.get('component_all_ge200_frac'))} |")
    lines.append('')
    lines.append('## Official-text surface exposure through legal40k low-count tokens')
    lines.append('')
    lines.append('| family | legal40k frac<50 | frac<100 | among <50, legal16 component-min >=50 | among <100, component-min >=50 | mean component-min support for <100 | leader gap from legal40k 8x480 |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|')
    for fam in ['EWoK', 'GlobalPIQA', 'Supplement', 'SuperGLUE', 'BLiMP', 'COMPS', 'Entity', 'Reading']:
        r = byfam.get(fam)
        if not r:
            continue
        gap = ctx['gap_to_leader'].get(fam)
        lines.append(f"| {fam} | {fmt(r['direct_frac_lt50'])} | {fmt(r['direct_frac_lt100'])} | {fmt(r['rescued_min_ge50_within_lt50_frac_of_low'])} | {fmt(r['rescued_min_ge50_within_lt100_frac_of_low'])} | {fmt(r['direct_lt100_mean_component_min_support'], 1)} | {fmt(gap, 3)} |")
    lines.append('')
    lines.append('## Scientific reading')
    lines.append('')
    lines.append('A support-shared 40k representation is mechanically plausible if many score-text low-support 40k tokens decompose into legal16 pieces with much higher corpus support. It would attack a different bottleneck than U256: preserve the shorter 40k segmentation and high Supplement/BLiMP behavior while reducing the undertrained-row problem visible in EWoK and GlobalPIQA. This is construction evidence only; it becomes a serious next route only if the pending depth and A02 minfreq50 score vectors show complementary recovery consistent with representation support rather than merely dialogue-tail visibility.')
    lines.append('')
    lines.append(f'JSON: `{OUT_JSON}`')
    lines.append(f'CSV: `{EVAL_CSV}`, `{TOKEN_CSV}`')
    NOTE.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sha40 = sha(LEGAL40K / 'tokenizer.json')
    sha16 = sha(LEGAL16K / 'tokenizer.json')
    if sha40 != EXPECTED_40K_SHA:
        raise RuntimeError(f'legal40k tokenizer SHA mismatch: {sha40}')
    if sha16 != EXPECTED_16K_SHA:
        raise RuntimeError(f'legal16k tokenizer SHA mismatch: {sha16}')
    cmap, vocab_summary, rare_rows = component_map()
    eval_rows, top_eval_rare = eval_summary(cmap)
    payload = {
        'status': 'SUPPORT_SHARED_TOKEN_RESCUE',
        'created_utc': now(),
        'purpose': 'Measure whether low-support legal40k tokens can borrow support from legal16k components learned on the same legal 10M pool.',
        'no_training_or_model_evaluation': True,
        'no_managed_task_state_query': True,
        'inputs': {
            'support_json': str(SUPPORT_JSON),
            'legal40k_tokenizer': str(LEGAL40K),
            'legal40k_tokenizer_sha256': sha40,
            'legal16k_tokenizer': str(LEGAL16K),
            'legal16k_tokenizer_sha256': sha16,
        },
        'vocab_component_summary': vocab_summary,
        'eval_family_summary': eval_rows,
        'top_eval_low_support_40k_tokens': top_eval_rare,
        'score_context': score_context(),
        'route_reading': [
            'This is a legal-representation-corner measurement, not score evidence.',
            'If depth and minfreq50 show complementary recovery, support-aware rare-token embedding/composition is a candidate comparison before another flat tokenizer run or U256.',
            'Any custom representation must preserve official compatibility before H100 launch.',
        ],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    write_csv(EVAL_CSV, eval_rows)
    write_csv(TOKEN_CSV, rare_rows[:1000])
    write_note(payload)
    print(json.dumps({
        'status': payload['status'],
        'eval_summary_rows': len(eval_rows),
        'rare_token_rows_written': min(len(rare_rows), 1000),
        'EWoK_direct_lt50': next((r['direct_frac_lt50'] for r in eval_rows if r['family'] == 'EWoK'), None),
        'EWoK_lt50_rescue_frac': next((r['rescued_min_ge50_within_lt50_frac_of_low'] for r in eval_rows if r['family'] == 'EWoK'), None),
        'GlobalPIQA_direct_lt50': next((r['direct_frac_lt50'] for r in eval_rows if r['family'] == 'GlobalPIQA'), None),
        'GlobalPIQA_lt50_rescue_frac': next((r['rescued_min_ge50_within_lt50_frac_of_low'] for r in eval_rows if r['family'] == 'GlobalPIQA'), None),
        'out_json': str(OUT_JSON),
        'note': str(NOTE),
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
