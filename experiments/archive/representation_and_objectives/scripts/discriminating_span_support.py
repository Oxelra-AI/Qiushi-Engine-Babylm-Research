#!/usr/bin/env python3
"""research: does low legal40k token support fall in answer-discriminating spans?

CPU-only. No training, no model evaluation, no managed-task query.

The support-shared representation route (research) is only worth an H100 run if the
undertrained legal40k tokens actually sit in the part of each item that decides the
answer, not only in shared context. This script recomputes legal40k pool support
from the exact allowed 10M corpus and, for EWoK / GlobalPIQA / COMPS / Entity,
separates each item's SHARED text from its DISCRIMINATING text (the parts that
differ between candidates or between the two contexts), then measures where the
low-support tokens concentrate.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import math
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

LEGAL40K = WS / 'data/legal_representation_route_map/tokenizers/legal_byte_bpe_40k'
PRISTINE = WS / 'data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval'
GLOBALPIQA = WS / 'data/globalpiqa_official_lineage/official_dl_scratch/generated_by_current_official_dl/evaluation_data/full_eval'
OUT_DIR = WS / 'data/discriminating_span_support'
OUT_JSON = OUT_DIR / 'discriminating_span_support.json'
FAMILY_CSV = OUT_DIR / 'discriminating_span_support_by_family.csv'
NOTE = (ROOT / 'research/notes/representation_and_objectives/discriminating_span_support.md')

EXPECTED_40K_SHA = '94b44bcf5901d097e20294e8220740758eac29af3ec20d58d8da867c85197758'
THRESHOLDS = [20, 50, 100]


def now() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open('r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_tok() -> Tokenizer:
    tok = Tokenizer.from_file(str(LEGAL40K / 'tokenizer.json'))
    try:
        tok.no_padding()
    except Exception:
        pass
    try:
        tok.no_truncation()
    except Exception:
        pass
    return tok


def build_support() -> tuple[list[int], set[int], Tokenizer]:
    tok = load_tok()
    rec = supportbase.count_pool_support('legal40k_step82_disc', tok)
    counts = [int(x) for x in rec['counts_by_id']]
    specials = {int(v) for v in rec['special_ids'].values() if v is not None}
    return counts, specials, tok


def token_multiset(tok: Tokenizer, text: str, specials: set[int]) -> collections.Counter[int]:
    if not text or not text.strip():
        return collections.Counter()
    ids = [int(x) for x in tok.encode(text, add_special_tokens=False).ids]
    return collections.Counter(i for i in ids if i not in specials)


def diff_spans(shared: collections.Counter[int], variants: list[collections.Counter[int]]) -> tuple[collections.Counter[int], collections.Counter[int]]:
    """Return (shared_effective, discriminating). A token id is discriminating if
    its multiplicity is not identical across all variant candidates; shared count
    is the minimum multiplicity common to every candidate (plus explicit shared)."""
    all_ids: set[int] = set(shared)
    for v in variants:
        all_ids |= set(v)
    shared_eff = collections.Counter()
    disc = collections.Counter()
    for tid in all_ids:
        mults = [v.get(tid, 0) for v in variants]
        base = shared.get(tid, 0)
        common = min(mults) if mults else 0
        # shared portion = explicit shared text + the common floor across candidates
        shared_eff[tid] += base + common
        # discriminating portion = variation above the common floor
        extra = sum(m - common for m in mults)
        if extra:
            disc[tid] += extra
    return shared_eff, disc


def accumulate(counts: list[int], specials: set[int], multiset: collections.Counter[int], agg: dict[str, int]) -> None:
    for tid, mult in multiset.items():
        if tid in specials:
            continue
        c = counts[tid] if tid < len(counts) else 0
        agg['tokens'] += mult
        for th in THRESHOLDS:
            if c < th:
                agg[f'lt{th}'] += mult


def new_agg() -> dict[str, int]:
    d = {'tokens': 0}
    for th in THRESHOLDS:
        d[f'lt{th}'] = 0
    return d


def process_ewok(counts: list[int], specials: set[int], tok: Tokenizer) -> dict[str, Any]:
    shared_agg, disc_agg = new_agg(), new_agg()
    n = 0
    for p in sorted((PRISTINE / 'ewok_filtered').glob('*.jsonl')):
        for obj in iter_jsonl(p):
            c1, c2 = obj.get('Context1', ''), obj.get('Context2', '')
            t1 = obj.get('Target1', '')
            t2 = obj.get('Target2', c1 and t1)
            # The two scored candidates are (Context_i + Target1); the shared token
            # is Target1, the discriminating token is Context1 vs Context2 (and the
            # optional Target2). Model must use context differences to choose.
            cand1 = token_multiset(tok, (c1 + ' ' + t1).strip(), specials)
            cand2 = token_multiset(tok, (c2 + ' ' + t1).strip(), specials)
            shared_eff, disc = diff_spans(collections.Counter(), [cand1, cand2])
            accumulate(counts, specials, shared_eff, shared_agg)
            accumulate(counts, specials, disc, disc_agg)
            n += 1
    return {'family': 'EWoK', 'items': n, 'shared': shared_agg, 'discriminating': disc_agg}


def process_globalpiqa(counts: list[int], specials: set[int], tok: Tokenizer) -> dict[str, Any]:
    shared_agg, disc_agg = new_agg(), new_agg()
    n = 0
    for sub in ['global_piqa_parallel', 'global_piqa_nonparallel']:
        p = GLOBALPIQA / sub / 'eng_latn.jsonl'
        if not p.exists():
            continue
        for obj in iter_jsonl(p):
            prompt = obj.get('prompt', '') or ''
            sols = [obj.get(f'solution{i}') for i in range(4)]
            sols = [s for s in sols if isinstance(s, str) and s.strip()]
            prompt_ms = token_multiset(tok, prompt, specials)
            sol_ms = [token_multiset(tok, s, specials) for s in sols]
            shared_eff, disc = diff_spans(prompt_ms, sol_ms)
            accumulate(counts, specials, shared_eff, shared_agg)
            accumulate(counts, specials, disc, disc_agg)
            n += 1
    return {'family': 'GlobalPIQA', 'items': n, 'shared': shared_agg, 'discriminating': disc_agg}


def process_comps(counts: list[int], specials: set[int], tok: Tokenizer) -> dict[str, Any]:
    shared_agg, disc_agg = new_agg(), new_agg()
    n = 0
    for p in sorted((PRISTINE / 'comps').glob('*.jsonl')):
        for obj in iter_jsonl(p):
            prop = obj.get('property_phrase', '') or ''
            pa = obj.get('prefix_acceptable', '') or ''
            pu = obj.get('prefix_unacceptable', '') or ''
            cand_a = token_multiset(tok, (pa + ' ' + prop).strip(), specials)
            cand_u = token_multiset(tok, (pu + ' ' + prop).strip(), specials)
            shared_eff, disc = diff_spans(collections.Counter(), [cand_a, cand_u])
            accumulate(counts, specials, shared_eff, shared_agg)
            accumulate(counts, specials, disc, disc_agg)
            n += 1
    return {'family': 'COMPS', 'items': n, 'shared': shared_agg, 'discriminating': disc_agg}


def process_entity(counts: list[int], specials: set[int], tok: Tokenizer) -> dict[str, Any]:
    shared_agg, disc_agg = new_agg(), new_agg()
    n = 0
    for p in sorted((PRISTINE / 'entity_tracking').glob('*.jsonl')):
        for obj in iter_jsonl(p):
            prefix = obj.get('input_prefix', '') or ''
            opts = obj.get('options', []) or []
            opts = [o for o in opts if isinstance(o, str) and o.strip()]
            prefix_ms = token_multiset(tok, prefix, specials)
            opt_ms = [token_multiset(tok, o, specials) for o in opts]
            shared_eff, disc = diff_spans(prefix_ms, opt_ms)
            accumulate(counts, specials, shared_eff, shared_agg)
            accumulate(counts, specials, disc, disc_agg)
            n += 1
    return {'family': 'Entity', 'items': n, 'shared': shared_agg, 'discriminating': disc_agg}


def summarize(rec: dict[str, Any]) -> dict[str, Any]:
    out = {'family': rec['family'], 'items': rec['items']}
    for part in ['shared', 'discriminating']:
        agg = rec[part]
        tot = max(1, agg['tokens'])
        out[f'{part}_tokens'] = agg['tokens']
        for th in THRESHOLDS:
            out[f'{part}_frac_lt{th}'] = agg[f'lt{th}'] / tot
    # Enrichment: how much more low-support mass sits in discriminating vs shared spans.
    for th in THRESHOLDS:
        s = out.get(f'shared_frac_lt{th}', 0.0)
        d = out.get(f'discriminating_frac_lt{th}', 0.0)
        out[f'disc_over_shared_lt{th}'] = (d / s) if s > 0 else None
        # Fraction of ALL low-support token mass that is in the discriminating span.
        low_disc = rec['discriminating'][f'lt{th}']
        low_all = rec['shared'][f'lt{th}'] + rec['discriminating'][f'lt{th}']
        out[f'disc_share_of_low_lt{th}'] = (low_disc / low_all) if low_all > 0 else None
    disc_tok = rec['discriminating']['tokens']
    all_tok = rec['shared']['tokens'] + rec['discriminating']['tokens']
    out['discriminating_token_share'] = disc_tok / max(1, all_tok)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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
    if x is None:
        return 'NA'
    try:
        v = float(x)
        return f'{v:.{nd}f}' if math.isfinite(v) else 'NA'
    except Exception:
        return str(x)


def write_note(summaries: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    lines.append('# research — Low-support tokens in answer-discriminating spans')
    lines.append('')
    lines.append('CPU-only. No training, no model evaluation, no managed-task query. This isolates the decisive object flagged by independent_review: whether legal40k low-support tokens actually sit in the discriminating part of each item (the parts that differ across candidates or contexts), which the model must use to choose the answer, versus the shared context.')
    lines.append('')
    lines.append('For each item the SHARED span is text common to every candidate; the DISCRIMINATING span is the multiset difference across candidates (EWoK: Context1 vs Context2; GlobalPIQA: solutions; COMPS: acceptable vs unacceptable prefix; Entity: options).')
    lines.append('')
    lines.append('| family | items | disc token share | shared frac<50 | disc frac<50 | disc/shared<50 | disc share of all <50 mass | shared frac<100 | disc frac<100 | disc share of all <100 mass |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for s in summaries:
        lines.append(
            f"| {s['family']} | {s['items']} | {fmt(s['discriminating_token_share'])} | "
            f"{fmt(s['shared_frac_lt50'])} | {fmt(s['discriminating_frac_lt50'])} | {fmt(s['disc_over_shared_lt50'])} | "
            f"{fmt(s['disc_share_of_low_lt50'])} | {fmt(s['shared_frac_lt100'])} | {fmt(s['discriminating_frac_lt100'])} | {fmt(s['disc_share_of_low_lt100'])} |"
        )
    lines.append('')
    lines.append('## Scientific reading')
    lines.append('')
    lines.append('If the discriminating span has clearly higher low-support fraction than the shared span (disc/shared > 1) and holds a large share of the total low-support mass, then undertrained legal40k rows plausibly sit exactly where the answer is decided, strengthening the support-aware representation route. If low-support mass is dominated by shared context, then the support corner is weaker: rare tokens are seen by both candidates and cannot directly explain discrimination, so U256, depth, or a different objective is more likely the operative lever. Read together with the pending depth and A02 minfreq50 score vectors before any launch.')
    lines.append('')
    lines.append(f'JSON: `{OUT_JSON}`')
    lines.append(f'CSV: `{FAMILY_CSV}`')
    NOTE.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sha40 = sha(LEGAL40K / 'tokenizer.json')
    if sha40 != EXPECTED_40K_SHA:
        raise RuntimeError(f'legal40k tokenizer SHA mismatch: {sha40}')
    counts, specials, tok = build_support()
    recs = [
        process_ewok(counts, specials, tok),
        process_globalpiqa(counts, specials, tok),
        process_comps(counts, specials, tok),
        process_entity(counts, specials, tok),
    ]
    summaries = [summarize(r) for r in recs]
    payload = {
        'status': 'DISCRIMINATING_SPAN_SUPPORT',
        'created_utc': now(),
        'purpose': 'Locate low-support legal40k tokens in shared vs answer-discriminating spans to test the support-aware representation route before any GPU launch.',
        'no_training_or_model_evaluation': True,
        'no_managed_task_state_query': True,
        'legal40k_tokenizer_sha256': sha40,
        'thresholds': THRESHOLDS,
        'family_summaries': summaries,
        'span_definition': {
            'EWoK': 'shared=Target1 (+common tokens); discriminating=Context1 vs Context2 difference',
            'GlobalPIQA': 'shared=prompt (+common solution tokens); discriminating=solution differences',
            'COMPS': 'shared=property_phrase (+common); discriminating=acceptable vs unacceptable prefix',
            'Entity': 'shared=input_prefix (+common); discriminating=option differences',
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    write_csv(FAMILY_CSV, summaries)
    write_note(summaries)
    print(json.dumps({
        'status': payload['status'],
        'summaries': [{
            'family': s['family'],
            'items': s['items'],
            'disc_token_share': round(s['discriminating_token_share'], 4),
            'shared_frac_lt50': round(s['shared_frac_lt50'], 4),
            'disc_frac_lt50': round(s['discriminating_frac_lt50'], 4),
            'disc_over_shared_lt50': round(s['disc_over_shared_lt50'], 4) if s['disc_over_shared_lt50'] is not None else None,
            'disc_share_of_low_lt50': round(s['disc_share_of_low_lt50'], 4) if s['disc_share_of_low_lt50'] is not None else None,
        } for s in summaries],
        'out_json': str(OUT_JSON),
        'note': str(NOTE),
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
