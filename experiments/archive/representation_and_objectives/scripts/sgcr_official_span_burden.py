#!/usr/bin/env python3
"""research: exact-prefix SGCR residual burden on official answer-discriminating text.

CPU-only. Does not train, evaluate a model.

The pending SGCR endpoint should be interpreted against the actual text positions
where candidate answers differ. This script uses the repaired exact legal40k ->
legal16k prefix decomposition from sgcr_module.py and measures the K=50
SGCR residual weight (1-rho) in shared vs answer-discriminating spans for the
same official-coordinate EWoK, GlobalPIQA, COMPS, and Entity text used by the
post-training evaluator.
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

import torch
from tokenizers import Tokenizer

ROOT = Path('.').resolve()
WS = ROOT / 'experiments/archive/representation_and_objectives'
SCRIPT_DIR = WS / 'scripts'
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import sgcr_module as sgcr_mod  # noqa: E402

TOK40 = WS / 'data/legal_representation_route_map/tokenizers/legal_byte_bpe_40k'
TOK16 = WS / 'data/strictsmall_tokenizer_retrain/strictsmall_compact_reinvest_16k_tokenizer'
POOL = ROOT / 'experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl'
PRISTINE_FULL = WS / 'data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval'
GLOBALPIQA_FULL = WS / 'data/globalpiqa_official_lineage/official_dl_scratch/generated_by_current_official_dl/evaluation_data/full_eval'
OUT_DIR = WS / 'data/sgcr_official_span_burden'
OUT_JSON = OUT_DIR / 'sgcr_official_span_burden.json'
OUT_CSV = OUT_DIR / 'sgcr_official_span_burden_by_group.csv'
NOTE = (ROOT / 'research/notes/representation_and_objectives/sgcr_official_span_burden.md')

EXPECTED = {
    'tok40_sha': '94b44bcf5901d097e20294e8220740758eac29af3ec20d58d8da867c85197758',
    'tok16_sha': '4a95a2a2ead21813a409c5ad7c2c008fbf418dcf3e0dc17bd970bd9ce25ea738',
    'pool_sha': '215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23',
    'decomp_sha': 'b450cc45f7d66564a894fb8cc12e0b0ab8e3ba7db0339cad5eec6f30c6edfd8b',
    'legal40k_total_tokens': 13942644,
    'legal16k_total_tokens': 14669276,
}
THRESHOLDS = [20, 50, 100]
FORCE_STANDARD_IDS = [0, 1, 2, 3, 4]
K = 50.0


def now() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def map_hash(mapping: dict[int, list[int]]) -> str:
    payload = json.dumps({str(k): mapping[k] for k in sorted(mapping)}, separators=(',', ':'), sort_keys=True)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open('r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_tokenizer(path: Path) -> Tokenizer:
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


def token_multiset(tok: Tokenizer, text: str, force_ids: set[int]) -> collections.Counter[int]:
    if not isinstance(text, str) or not text.strip():
        return collections.Counter()
    enc = tok.encode(text, add_special_tokens=False)
    return collections.Counter(int(t) for t in enc.ids if int(t) not in force_ids)


def diff_spans(explicit_shared: collections.Counter[int], variants: list[collections.Counter[int]]) -> tuple[collections.Counter[int], collections.Counter[int]]:
    """Multiset split into candidate-common and candidate-varying token mass."""
    all_ids: set[int] = set(explicit_shared)
    for v in variants:
        all_ids.update(v)
    shared = collections.Counter()
    disc = collections.Counter()
    for tid in all_ids:
        mults = [v.get(tid, 0) for v in variants]
        common = min(mults) if mults else 0
        base = explicit_shared.get(tid, 0)
        if base or common:
            shared[tid] = base + common
        extra = sum(m - common for m in mults)
        if extra:
            disc[tid] = extra
    return shared, disc


def new_acc() -> dict[str, Any]:
    d: dict[str, Any] = {
        'items': 0,
        'tokens': 0,
        'support_sum': 0.0,
        'residual_sum': 0.0,
        'decomp_len_sum': 0.0,
        'comp_min_support_sum': 0.0,
        'comp_ge50_occurrences': 0,
        'comp_ge50_residual_sum': 0.0,
        'types': set(),
        'comp_ge50_types': set(),
    }
    for th in THRESHOLDS:
        d[f'lt{th}_occurrences'] = 0
        d[f'lt{th}_residual_sum'] = 0.0
        d[f'lt{th}_comp_ge50_occurrences'] = 0
        d[f'lt{th}_comp_ge50_residual_sum'] = 0.0
        d[f'lt{th}_types'] = set()
        d[f'lt{th}_comp_ge50_types'] = set()
    return d


def add_multiset(
    acc: dict[str, Any],
    multiset: collections.Counter[int],
    counts40: dict[int, int],
    counts16: dict[int, int],
    rho: list[float],
    decomp_map: dict[int, list[int]],
) -> None:
    for tid, mult in multiset.items():
        if mult <= 0:
            continue
        tid = int(tid)
        c40 = int(counts40.get(tid, 0))
        resid = float(1.0 - rho[tid]) if tid < len(rho) else 1.0
        comps = decomp_map.get(tid, [])
        if comps:
            comp_min = min(int(counts16.get(c, 0)) for c in comps)
            comp_ge50 = comp_min >= 50
            dlen = len(comps)
        else:
            comp_min = 0
            comp_ge50 = False
            dlen = 0
        acc['tokens'] += int(mult)
        acc['support_sum'] += c40 * mult
        acc['residual_sum'] += resid * mult
        acc['decomp_len_sum'] += dlen * mult
        acc['comp_min_support_sum'] += comp_min * mult
        acc['types'].add(tid)
        if comp_ge50:
            acc['comp_ge50_occurrences'] += int(mult)
            acc['comp_ge50_residual_sum'] += resid * mult
            acc['comp_ge50_types'].add(tid)
        for th in THRESHOLDS:
            if c40 < th:
                acc[f'lt{th}_occurrences'] += int(mult)
                acc[f'lt{th}_residual_sum'] += resid * mult
                acc[f'lt{th}_types'].add(tid)
                if comp_ge50:
                    acc[f'lt{th}_comp_ge50_occurrences'] += int(mult)
                    acc[f'lt{th}_comp_ge50_residual_sum'] += resid * mult
                    acc[f'lt{th}_comp_ge50_types'].add(tid)


def finalize_acc(acc: dict[str, Any]) -> dict[str, Any]:
    tokens = max(1, int(acc['tokens']))
    residual = float(acc['residual_sum'])
    out: dict[str, Any] = {
        'items': int(acc['items']),
        'tokens': int(acc['tokens']),
        'unique_types': len(acc['types']),
        'mean_legal40k_pool_count': float(acc['support_sum']) / tokens,
        'mean_sgcr_residual_weight': residual / tokens,
        'mean_decomp_len': float(acc['decomp_len_sum']) / tokens,
        'mean_component_min_pool_count16': float(acc['comp_min_support_sum']) / tokens,
        'component_ge50_occurrence_fraction': int(acc['comp_ge50_occurrences']) / tokens,
        'component_ge50_type_fraction': len(acc['comp_ge50_types']) / max(1, len(acc['types'])),
        'component_ge50_residual_fraction': float(acc['comp_ge50_residual_sum']) / residual if residual > 0 else 0.0,
    }
    for th in THRESHOLDS:
        low_occ = int(acc[f'lt{th}_occurrences'])
        low_res = float(acc[f'lt{th}_residual_sum'])
        out[f'frac_lt{th}'] = low_occ / tokens
        out[f'lt{th}_occurrences'] = low_occ
        out[f'lt{th}_unique_types'] = len(acc[f'lt{th}_types'])
        out[f'lt{th}_residual_fraction'] = low_res / residual if residual > 0 else 0.0
        out[f'lt{th}_component_ge50_occurrence_fraction'] = int(acc[f'lt{th}_comp_ge50_occurrences']) / max(1, low_occ)
        out[f'lt{th}_component_ge50_type_fraction'] = len(acc[f'lt{th}_comp_ge50_types']) / max(1, len(acc[f'lt{th}_types']))
        out[f'lt{th}_component_ge50_residual_fraction'] = float(acc[f'lt{th}_comp_ge50_residual_sum']) / low_res if low_res > 0 else 0.0
    return out


def new_group(label: str) -> dict[str, Any]:
    return {'label': label, 'shared': new_acc(), 'discriminating': new_acc()}


def add_item(
    groups: dict[str, dict[str, Any]],
    label: str,
    shared: collections.Counter[int],
    disc: collections.Counter[int],
    counts40: dict[int, int],
    counts16: dict[int, int],
    rho: list[float],
    decomp_map: dict[int, list[int]],
) -> None:
    if label not in groups:
        groups[label] = new_group(label)
    g = groups[label]
    g['shared']['items'] += 1
    g['discriminating']['items'] += 1
    add_multiset(g['shared'], shared, counts40, counts16, rho, decomp_map)
    add_multiset(g['discriminating'], disc, counts40, counts16, rho, decomp_map)


def finalize_group(g: dict[str, Any]) -> dict[str, Any]:
    shared = finalize_acc(g['shared'])
    disc = finalize_acc(g['discriminating'])
    all_tokens = shared['tokens'] + disc['tokens']
    shared_residual = shared['mean_sgcr_residual_weight'] * shared['tokens']
    disc_residual = disc['mean_sgcr_residual_weight'] * disc['tokens']
    out: dict[str, Any] = {
        'label': g['label'],
        'items': max(shared['items'], disc['items']),
        'shared': shared,
        'discriminating': disc,
        'discriminating_token_share': disc['tokens'] / max(1, all_tokens),
        'discriminating_residual_share': disc_residual / max(1e-12, shared_residual + disc_residual),
    }
    for th in THRESHOLDS:
        sfrac = shared[f'frac_lt{th}']
        dfrac = disc[f'frac_lt{th}']
        slow = shared[f'lt{th}_occurrences']
        dlow = disc[f'lt{th}_occurrences']
        sres = shared[f'lt{th}_residual_fraction'] * shared_residual
        dres = disc[f'lt{th}_residual_fraction'] * disc_residual
        out[f'disc_over_shared_frac_lt{th}'] = (dfrac / sfrac) if sfrac > 0 else None
        out[f'disc_share_of_lt{th}_occurrences'] = dlow / max(1, slow + dlow)
        out[f'disc_share_of_lt{th}_residual'] = dres / max(1e-12, sres + dres)
    smean = shared['mean_sgcr_residual_weight']
    dmean = disc['mean_sgcr_residual_weight']
    out['disc_over_shared_mean_residual'] = (dmean / smean) if smean > 0 else None
    return out


def process_ewok(groups: dict[str, dict[str, Any]], tok: Tokenizer, force_ids: set[int], counts40, counts16, rho, decomp_map) -> int:
    n = 0
    for p in sorted((PRISTINE_FULL / 'ewok_filtered').glob('*.jsonl')):
        for obj in iter_jsonl(p):
            c1 = obj.get('Context1', '') or ''
            c2 = obj.get('Context2', '') or ''
            t1 = obj.get('Target1', '') or ''
            cand1 = token_multiset(tok, (c1 + ' ' + t1).strip(), force_ids)
            cand2 = token_multiset(tok, (c2 + ' ' + t1).strip(), force_ids)
            shared, disc = diff_spans(collections.Counter(), [cand1, cand2])
            add_item(groups, 'EWoK', shared, disc, counts40, counts16, rho, decomp_map)
            add_item(groups, f'EWoK::{obj.get("Domain", p.stem)}', shared, disc, counts40, counts16, rho, decomp_map)
            n += 1
    return n


def process_globalpiqa(groups: dict[str, dict[str, Any]], tok: Tokenizer, force_ids: set[int], counts40, counts16, rho, decomp_map) -> int:
    n = 0
    specs = [('global_piqa_parallel', 4), ('global_piqa_nonparallel', 2)]
    for sub, nsol in specs:
        p = GLOBALPIQA_FULL / sub / 'eng_latn.jsonl'
        for obj in iter_jsonl(p):
            prompt = token_multiset(tok, obj.get('prompt', '') or '', force_ids)
            sols = [token_multiset(tok, obj.get(f'solution{i}', '') or '', force_ids) for i in range(nsol)]
            shared, disc = diff_spans(prompt, sols)
            label = 'GlobalPIQA_parallel' if sub.endswith('parallel') and sub != 'global_piqa_nonparallel' else 'GlobalPIQA_nonparallel'
            add_item(groups, 'GlobalPIQA', shared, disc, counts40, counts16, rho, decomp_map)
            add_item(groups, label, shared, disc, counts40, counts16, rho, decomp_map)
            cat = obj.get('categories') or 'unknown'
            add_item(groups, f'{label}::{cat}', shared, disc, counts40, counts16, rho, decomp_map)
            n += 1
    return n


def process_comps(groups: dict[str, dict[str, Any]], tok: Tokenizer, force_ids: set[int], counts40, counts16, rho, decomp_map) -> int:
    n = 0
    for p in sorted((PRISTINE_FULL / 'comps').glob('*.jsonl')):
        for obj in iter_jsonl(p):
            prop = token_multiset(tok, obj.get('property_phrase', '') or '', force_ids)
            pa = token_multiset(tok, obj.get('prefix_acceptable', '') or '', force_ids)
            pu = token_multiset(tok, obj.get('prefix_unacceptable', '') or '', force_ids)
            shared, disc = diff_spans(prop, [pa, pu])
            add_item(groups, 'COMPS', shared, disc, counts40, counts16, rho, decomp_map)
            add_item(groups, f'COMPS::{p.stem}', shared, disc, counts40, counts16, rho, decomp_map)
            neg = obj.get('negative_sample_type') or 'unknown'
            add_item(groups, f'COMPS_negative::{neg}', shared, disc, counts40, counts16, rho, decomp_map)
            n += 1
    return n


def process_entity(groups: dict[str, dict[str, Any]], tok: Tokenizer, force_ids: set[int], counts40, counts16, rho, decomp_map) -> tuple[int, int]:
    kept = 0
    skipped = 0
    for p in sorted((PRISTINE_FULL / 'entity_tracking').glob('*.jsonl')):
        for obj in iter_jsonl(p):
            options = [o for o in (obj.get('options') or []) if isinstance(o, str)]
            if any('nothing' in o for o in options):
                skipped += 1
                continue
            prefix = token_multiset(tok, obj.get('input_prefix', '') or '', force_ids)
            opts = [token_multiset(tok, o, force_ids) for o in options if o.strip()]
            if not opts:
                skipped += 1
                continue
            shared, disc = diff_spans(prefix, opts)
            subset = f'{p.stem}_{obj.get("numops", -1)}_ops'
            add_item(groups, 'Entity', shared, disc, counts40, counts16, rho, decomp_map)
            add_item(groups, f'Entity::{subset}', shared, disc, counts40, counts16, rho, decomp_map)
            kept += 1
    return kept, skipped


def flatten_row(group: dict[str, Any]) -> dict[str, Any]:
    row = {
        'label': group['label'],
        'items': group['items'],
        'disc_token_share': group['discriminating_token_share'],
        'disc_residual_share': group['discriminating_residual_share'],
        'disc_over_shared_mean_residual': group['disc_over_shared_mean_residual'],
    }
    for part in ['shared', 'discriminating']:
        rec = group[part]
        prefix = 'shared' if part == 'shared' else 'disc'
        for k in [
            'tokens', 'unique_types', 'mean_legal40k_pool_count', 'mean_sgcr_residual_weight',
            'mean_decomp_len', 'mean_component_min_pool_count16', 'frac_lt50',
            'lt50_component_ge50_occurrence_fraction', 'lt50_component_ge50_residual_fraction',
            'component_ge50_residual_fraction',
        ]:
            row[f'{prefix}_{k}'] = rec.get(k)
    for th in THRESHOLDS:
        row[f'disc_over_shared_frac_lt{th}'] = group.get(f'disc_over_shared_frac_lt{th}')
        row[f'disc_share_of_lt{th}_occurrences'] = group.get(f'disc_share_of_lt{th}_occurrences')
        row[f'disc_share_of_lt{th}_residual'] = group.get(f'disc_share_of_lt{th}_residual')
    return row


def write_csv(rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return 'NA'
    try:
        v = float(x)
        return f'{v:.{nd}f}' if math.isfinite(v) else 'NA'
    except Exception:
        return str(x)


def write_note(payload: dict[str, Any]) -> None:
    focus = payload['family_summaries']
    lines: list[str] = []
    lines.append('# research — exact-prefix SGCR residual burden in official answer text')
    lines.append('')
    lines.append('CPU-only measurement. It uses the repaired exact BPE-prefix legal40k→legal16k decomposition and K=50 rho weights from the SGCR training code. It does not read or query the running SGCR training/evaluation tasks.')
    lines.append('')
    lines.append('The quantity `mean residual` is average `(1-rho_t)` over token occurrences in a span. For K=50, tokens with legal40k pool count below 50 receive more component-path weight than standard-row weight. `component>=50 among low<50` asks whether low-count legal40k tokens are decomposed into legal16k components each seen at least 50 times in the same allowed 10M pool.')
    lines.append('')
    lines.append('| family | items | disc token share | shared mean residual | disc mean residual | residual ratio | shared frac<50 | disc frac<50 | frac<50 ratio | disc share of low<50 residual | disc low<50 component>=50 |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for s in focus:
        sh = s['shared']; di = s['discriminating']
        lines.append(
            f"| {s['label']} | {s['items']} | {fmt(s['discriminating_token_share'])} | "
            f"{fmt(sh['mean_sgcr_residual_weight'])} | {fmt(di['mean_sgcr_residual_weight'])} | {fmt(s['disc_over_shared_mean_residual'])} | "
            f"{fmt(sh['frac_lt50'])} | {fmt(di['frac_lt50'])} | {fmt(s['disc_over_shared_frac_lt50'])} | "
            f"{fmt(s['disc_share_of_lt50_residual'])} | {fmt(di['lt50_component_ge50_occurrence_fraction'])} |"
        )
    lines.append('')
    lines.append('## Reading for the pending endpoint')
    lines.append('')
    lines.append('- The exact-prefix SGCR K=50 run is most directly aimed at candidate-varying tokens when the discriminating spans carry much higher residual pressure than shared spans and their low-count tokens are usually component-supported.')
    lines.append('- COMPS has the strongest measured alignment: discriminating spans have about 2.6× the mean residual of shared spans, while their below-50 support fraction is about 2.92× the shared fraction; almost all low-count discriminating occurrences decompose into well-supported legal16k components. A real endpoint should therefore move COMPS if the mechanism is effective.')
    lines.append('- GlobalPIQA is weaker but still aligned: candidate solutions carry about 1.5× the shared residual pressure in aggregate, with both parallel and nonparallel aligned. Parallel and nonparallel should be read separately because prior vectors showed they can move differently, not because the burden map predicts nonparallel should benefit more.')
    lines.append('- Entity has a high low<50 enrichment in answer options after the official skip, but its mean residual ratio is modest because many shared prefix tokens sit in the 50–99 support band. A positive Entity movement would support the component-sharing story; a flat Entity vector would not by itself refute SGCR.')
    lines.append('- EWoK remains poorly aligned with this mechanism: candidate-varying context tokens are not meaningfully more low-supported or residual-heavy than shared tokens. Large EWoK recovery should not be attributed to simple rare-token sharing without further evidence.')
    lines.append('')
    lines.append(f"JSON: `{OUT_JSON}`")
    lines.append(f"CSV: `{OUT_CSV}`")
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sha40 = sha256_file(TOK40 / 'tokenizer.json')
    sha16 = sha256_file(TOK16 / 'tokenizer.json')
    pool_sha = sha256_file(POOL)
    if sha40 != EXPECTED['tok40_sha']:
        raise RuntimeError(f'tok40 SHA mismatch: {sha40}')
    if sha16 != EXPECTED['tok16_sha']:
        raise RuntimeError(f'tok16 SHA mismatch: {sha16}')
    if pool_sha != EXPECTED['pool_sha']:
        raise RuntimeError(f'pool SHA mismatch: {pool_sha}')

    tok40 = load_tokenizer(TOK40)
    decomp_map = sgcr_mod.build_decomposition_map(TOK40, TOK16)
    decomp_sha = map_hash(decomp_map)
    if decomp_sha != EXPECTED['decomp_sha']:
        raise RuntimeError(f'decomp map SHA mismatch: {decomp_sha}')
    counts40 = sgcr_mod.build_pool_counts(TOK40, POOL)
    counts16 = sgcr_mod.build_pool_counts(TOK16, POOL)
    total40 = sum(counts40.values())
    total16 = sum(counts16.values())
    if total40 != EXPECTED['legal40k_total_tokens']:
        raise RuntimeError(f'legal40k total token mismatch: {total40}')
    if total16 != EXPECTED['legal16k_total_tokens']:
        raise RuntimeError(f'legal16k total token mismatch: {total16}')

    cfg = sgcr_mod.SGCRConfig(
        vocab_40k=40000,
        vocab_16k=16384,
        hidden_size=384,
        d_comp=64,
        K=K,
        force_standard_ids=FORCE_STANDARD_IDS,
    )
    buffers = sgcr_mod.build_sgcr_buffers(decomp_map, counts40, cfg)
    rho = [float(x) for x in buffers['rho'].tolist()]
    force_ids = set(FORCE_STANDARD_IDS)

    groups: dict[str, dict[str, Any]] = {}
    n_ewok = process_ewok(groups, tok40, force_ids, counts40, counts16, rho, decomp_map)
    n_gpiqa = process_globalpiqa(groups, tok40, force_ids, counts40, counts16, rho, decomp_map)
    n_comps = process_comps(groups, tok40, force_ids, counts40, counts16, rho, decomp_map)
    n_entity, n_entity_skipped = process_entity(groups, tok40, force_ids, counts40, counts16, rho, decomp_map)

    finalized = {k: finalize_group(v) for k, v in groups.items()}
    family_labels = ['EWoK', 'GlobalPIQA', 'GlobalPIQA_parallel', 'GlobalPIQA_nonparallel', 'COMPS', 'Entity']
    family_summaries = [finalized[k] for k in family_labels if k in finalized]
    rows = [flatten_row(finalized[k]) for k in sorted(finalized)]
    write_csv(rows)

    payload = {
        'status': 'SGCR_OFFICIAL_SPAN_BURDEN',
        'created_utc': now(),
        'no_training_or_model_evaluation': True,
        'no_managed_task_state_query': True,
        'purpose': 'Prepare interpretation of the running exact-prefix SGCR endpoint by locating its K=50 residual pressure in official shared vs answer-discriminating spans.',
        'inputs': {
            'tok40_sha256': sha40,
            'tok16_sha256': sha16,
            'pool_sha256': pool_sha,
            'decomposition_map_sha256': decomp_sha,
            'legal40k_pool_tokens': total40,
            'legal16k_pool_tokens': total16,
            'K': K,
            'force_standard_ids': FORCE_STANDARD_IDS,
            'max_decomposition_length': int(buffers['max_comp_len'].item()),
        },
        'official_row_counts': {
            'EWoK': n_ewok,
            'GlobalPIQA_total': n_gpiqa,
            'COMPS': n_comps,
            'Entity_kept_after_official_nothing_skip': n_entity,
            'Entity_skipped_by_official_nothing_skip_or_empty': n_entity_skipped,
        },
        'span_definitions': {
            'EWoK': 'official candidates are Context1+Target1 vs Context2+Target1; shared is common multiset mass, discriminating is the candidate-varying context mass',
            'GlobalPIQA': 'shared prompt plus any common solution mass; discriminating solution differences; parallel uses 4 options and nonparallel uses 2 options as in read_files.py',
            'COMPS': 'shared property_phrase plus common prefix mass; discriminating acceptable vs unacceptable prefix differences',
            'Entity': 'official skip if any option contains "nothing"; shared input_prefix plus common option mass; discriminating option differences',
        },
        'family_summaries': family_summaries,
        'all_group_summaries': finalized,
        'csv': str(OUT_CSV),
        'note': str(NOTE),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    write_note(payload)
    print(json.dumps({
        'status': payload['status'],
        'row_counts': payload['official_row_counts'],
        'family_quick': [
            {
                'label': s['label'],
                'items': s['items'],
                'disc_token_share': round(s['discriminating_token_share'], 4),
                'shared_mean_residual': round(s['shared']['mean_sgcr_residual_weight'], 4),
                'disc_mean_residual': round(s['discriminating']['mean_sgcr_residual_weight'], 4),
                'disc_over_shared_mean_residual': None if s['disc_over_shared_mean_residual'] is None else round(s['disc_over_shared_mean_residual'], 4),
                'shared_frac_lt50': round(s['shared']['frac_lt50'], 4),
                'disc_frac_lt50': round(s['discriminating']['frac_lt50'], 4),
                'disc_over_shared_frac_lt50': None if s['disc_over_shared_frac_lt50'] is None else round(s['disc_over_shared_frac_lt50'], 4),
                'disc_low50_component_ge50': round(s['discriminating']['lt50_component_ge50_occurrence_fraction'], 4),
            }
            for s in family_summaries
        ],
        'json': str(OUT_JSON),
        'csv': str(OUT_CSV),
        'note': str(NOTE),
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
