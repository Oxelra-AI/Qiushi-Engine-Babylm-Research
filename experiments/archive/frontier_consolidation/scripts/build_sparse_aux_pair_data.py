#!/usr/bin/env python3
"""Build sparse high-utility auxiliary pair-data variants and forecast charged exposure.

The current broad dual-view auxiliary route fires on all masked rewrite word groups in
all 12,155 legal pairs.  Steps122-123 supported a narrower phenomenon: changed tokens
whose token id is absent from the source.  This script ranks pairs by potential
changed-source-absent pieces per auxiliary charged word, writes filtered
`aux_pair_data`-compatible JSON files for several top fractions, and replays
the corrected trainer's deterministic WWM/batch stream to forecast aux words/targets
and stop points under a 20M charged cap.

No model loading, no training, no official evaluation.
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import pathlib
import random
import re
import sys
import time
from collections import Counter, defaultdict
from typing import Any

# Writable cache before transformers import.
_CACHE = pathlib.Path('experiments/archive/frontier_consolidation/data/sparse_aux_pair_data/import_cache').resolve()
for _k, _p in {
    'HF_HOME': _CACHE / 'hf_home',
    'HF_HUB_CACHE': _CACHE / 'hf_home' / 'hub',
    'HUGGINGFACE_HUB_CACHE': _CACHE / 'hf_home' / 'hub',
    'TRANSFORMERS_CACHE': _CACHE / 'transformers',
    'HF_MODULES_CACHE': _CACHE / 'modules',
    'HF_DATASETS_CACHE': _CACHE / 'datasets',
    'TMPDIR': _CACHE / 'tmp',
}.items():
    _p.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(_k, str(_p))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

USER_ROOT = pathlib.Path('.').resolve()
STUDY = USER_ROOT / 'experiments/archive/frontier_consolidation'
WORKSPACE = STUDY
SCRIPTS = WORKSPACE / 'scripts'
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import dual_view_corrected_trainer as tr  # noqa: E402

PAIR_DATA = WORKSPACE / 'data/aux_pair_data/aux_pair_data.json'
PAIR_JSONL = WORKSPACE / 'data/density_core_reinvestment_medium_riskhard/selected_compact_reinvest_pairs.jsonl'
TOKENIZER_DIR = WORKSPACE / 'data/compliant_tokenizer'
STREAM = WORKSPACE / 'data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl'
OUT_ROOT = WORKSPACE / 'data/sparse_aux_pair_data'
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*|[^\w\s]", re.UNICODE)


def now() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def rel(p: pathlib.Path | str) -> str:
    p = pathlib.Path(p)
    try:
        return str(p.resolve().relative_to(USER_ROOT))
    except Exception:
        return str(p)


def norm_pid(pid: str) -> str:
    pid = str(pid)
    return pid.split(':', 1)[-1] if pid.startswith('compact:') else pid


def token_spans(text: str) -> list[dict[str, Any]]:
    out = []
    for m in WORD_RE.finditer(text):
        t = m.group()
        out.append({'text': t, 'norm': t.lower(), 'start': m.start(), 'end': m.end(), 'is_word': bool(re.search(r'[A-Za-z0-9]', t))})
    return out


def changed_char_set(source: str, rewrite: str) -> set[int]:
    s = [x['norm'] for x in token_spans(source)]
    r_spans = token_spans(rewrite)
    r = [x['norm'] for x in r_spans]
    sm = difflib.SequenceMatcher(a=s, b=r, autojunk=False)
    chars: set[int] = set()
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in {'insert', 'replace'}:
            for sp in r_spans[j1:j2]:
                if sp['is_word']:
                    chars.update(range(int(sp['start']), int(sp['end'])))
    return chars


def is_word_start(tok: str) -> bool:
    return tok.startswith('Ġ') or tok.startswith('▁')


def rewrite_piece_signal(tokenizer, source: str, rewrite: str, source_ids: list[int]) -> dict[str, int]:
    enc = tokenizer(rewrite, add_special_tokens=False, return_offsets_mapping=True)
    ids = [int(x) for x in enc['input_ids']]
    offs = [(int(a), int(b)) for a, b in enc['offset_mapping']]
    src_set = set(int(x) for x in source_ids)
    changed = changed_char_set(source, rewrite)
    counts = Counter()
    gid = -1
    group_flags: dict[int, Counter] = defaultdict(Counter)
    for i, (tid, (a, b)) in enumerate(zip(ids, offs)):
        tok = str(tokenizer.convert_ids_to_tokens(tid))
        if gid < 0 or is_word_start(tok) or i == 0:
            gid += 1
        if b <= a or not any(ch.isalnum() for ch in rewrite[a:b]):
            continue
        is_changed = bool(set(range(a, b)) & changed)
        absent = tid not in src_set
        counts['alnum_pieces'] += 1
        counts['changed_pieces'] += int(is_changed)
        counts['source_absent_pieces'] += int(absent)
        counts['changed_source_absent_pieces'] += int(is_changed and absent)
        group_flags[gid]['alnum'] += 1
        group_flags[gid]['changed_absent'] += int(is_changed and absent)
    counts['alnum_groups'] = sum(1 for c in group_flags.values() if c.get('alnum', 0) > 0)
    counts['changed_source_absent_groups'] = sum(1 for c in group_flags.values() if c.get('changed_absent', 0) > 0)
    return {k: int(v) for k, v in counts.items()}


def load_pair_texts() -> dict[str, dict[str, Any]]:
    out = {}
    with PAIR_JSONL.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            pid = str(obj.get('pair_id'))
            out[pid] = obj
            out[norm_pid(pid)] = obj
            out['compact:' + norm_pid(pid)] = obj
    return out


def rank_pairs(tokenizer, pair_data: dict[str, Any]) -> list[dict[str, Any]]:
    texts = load_pair_texts()
    rows = []
    for eid, rec in pair_data.items():
        for pr in rec.get('pairs', []):
            pid = str(pr['pair_id'])
            txt = texts.get(pid) or texts.get(norm_pid(pid)) or texts.get('compact:' + norm_pid(pid))
            if not txt:
                continue
            source = str(txt.get('source_text', ''))
            rewrite = str(txt.get('rewrite_text', ''))
            signal = rewrite_piece_signal(tokenizer, source, rewrite, list(pr.get('source_ids', [])))
            charge = int(pr.get('source_words', 0)) + 2 * int(pr.get('rewrite_words', 0))
            rows.append({
                'row_eid': str(eid),
                'pair_id': pid,
                'charge': charge,
                **signal,
                'utility_piece_per_charge': signal.get('changed_source_absent_pieces', 0) / charge if charge else 0.0,
                'utility_group_per_charge': signal.get('changed_source_absent_groups', 0) / charge if charge else 0.0,
            })
    rows.sort(key=lambda r: (r['utility_piece_per_charge'], r['changed_source_absent_pieces'], -r['charge']), reverse=True)
    return rows


def filter_pair_data(raw: dict[str, Any], selected: set[str], fraction: float, ranked: list[dict[str, Any]]) -> dict[str, Any]:
    out_pd = {}
    kept_pairs = 0
    kept_rows = 0
    for eid, rec in raw['pair_data'].items():
        pairs = [pr for pr in rec.get('pairs', []) if str(pr.get('pair_id')) in selected]
        if pairs:
            new = dict(rec)
            new['pairs'] = pairs
            out_pd[eid] = new
            kept_rows += 1
            kept_pairs += len(pairs)
    selected_rank_rows = [r for r in ranked if r['pair_id'] in selected]
    summary = dict(raw.get('summary', {}))
    summary.update({
        'status': 'SPARSE_AUX_PAIR_DATA',
        'source_pair_data': rel(PAIR_DATA),
        'selection': f'top_{fraction:g}_by_changed_source_absent_piece_per_charge',
        'top_fraction_pairs': fraction,
        'n_pair_rows': kept_rows,
        'n_used_pairs': kept_pairs,
        'selected_aux_charge_per_full_activation': sum(r['charge'] for r in selected_rank_rows),
        'selected_changed_source_absent_pieces': sum(r['changed_source_absent_pieces'] for r in selected_rank_rows),
        'selected_alnum_pieces': sum(r['alnum_pieces'] for r in selected_rank_rows),
        'created_utc': now(),
    })
    return {'summary': summary, 'pair_data': out_pd}


def aux_words_targets_for_units(units: list[tr.AuxUnit]) -> tuple[int, int, int]:
    words = 0
    targets = 0
    units_kept = 0
    for unit in units:
        # Match trainer length filtering.
        if len(unit.source_ids) + len(unit.rewrite_ids) + 3 > tr.DEFAULTS['aux_max_length']:
            continue
        if len(unit.rewrite_ids) + 2 > tr.DEFAULTS['aux_max_length']:
            continue
        n_t = 0
        for g in unit.rewrite_word_group:
            if g in unit.rewrite_word_groups_to_mask:
                n_t += 1
        if n_t <= 0:
            continue
        words += int(unit.source_words) + 2 * int(unit.rewrite_words)
        targets += 2 * n_t
        units_kept += 1
    return words, targets, units_kept


def forecast_variants(tokenizer, variants: dict[str, dict[str, Any]], charged_cap: int) -> dict[str, Any]:
    max_main_words = 20_000_000
    examples = tr.load_examples(str(STREAM), max_main_words)
    # Union pair eids for marking pair rows. The base pair_eids includes all pair rows so masking replay matches potential rows.
    base_pair_data = variants['all']['pair_data']
    pair_eids = {int(k) for k in base_pair_data.keys()}
    dataset = tr.DualViewDataset(examples, tokenizer, tr.DEFAULTS['seq_length'], pair_eids)
    loader = DataLoader(dataset, batch_size=tr.DEFAULTS['batch_size'], shuffle=False, collate_fn=tr.collate, num_workers=0)
    gen = torch.Generator(device='cpu')
    gen.manual_seed(tr.DEFAULTS['train_rng_seed'])
    # Per variant dynamic totals.
    totals = {name: {'updates': 0, 'main_words': 0, 'aux_words': 0, 'charged_words': 0, 'aux_targets': 0, 'aux_units': 0, 'stopped_before_cap': False, 'stop_loader_step': None, 'first_stop_record': None} for name in variants}
    active = {name: True for name in variants}
    logs_sample = {name: [] for name in variants}
    for loader_step, batch in enumerate(loader, 1):
        words = int(batch['words'].sum().item())
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        word_group = batch['word_group']
        _, labels = tr.apply_wwm(input_ids, attention_mask, word_group, tokenizer, tr.DEFAULTS['mask_prob'], gen)
        example_ids = batch['example_id']
        is_pair = batch['is_pair']
        pidx = is_pair.nonzero(as_tuple=False).squeeze(-1).tolist()
        if isinstance(pidx, int):
            pidx = [pidx]
        for name, raw in variants.items():
            if not active[name]:
                continue
            pdata = raw['pair_data']
            recs = []
            valid = []
            for bi in pidx:
                k = str(int(example_ids[bi].item()))
                if k in pdata:
                    valid.append(bi)
                    recs.append(pdata[k])
            units = tr.collect_aux_units(word_group, labels, valid, recs) if valid else []
            aux_words, aux_targets, aux_units = aux_words_targets_for_units(units)
            next_charged = totals[name]['charged_words'] + words + aux_words
            if next_charged > charged_cap:
                active[name] = False
                totals[name]['stopped_before_cap'] = True
                totals[name]['stop_loader_step'] = loader_step
                totals[name]['first_stop_record'] = {'loader_step': loader_step, 'current_charged': totals[name]['charged_words'], 'next_main_words': words, 'next_aux_words': aux_words}
                continue
            totals[name]['updates'] += 1
            totals[name]['main_words'] += words
            totals[name]['aux_words'] += aux_words
            totals[name]['charged_words'] = totals[name]['main_words'] + totals[name]['aux_words']
            totals[name]['aux_targets'] += aux_targets
            totals[name]['aux_units'] += aux_units
            if len(logs_sample[name]) < 3 or loader_step > 500:
                logs_sample[name].append({'loader_step': loader_step, 'main_words': words, 'aux_words': aux_words, 'aux_targets': aux_targets, 'aux_units': aux_units, 'charged': totals[name]['charged_words']})
        if not any(active.values()):
            break
    for name in totals:
        totals[name]['logs_sample'] = logs_sample[name][:3] + logs_sample[name][-3:]
    return totals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--fractions', nargs='+', type=float, default=[0.05, 0.10, 0.20, 0.40, 0.60])
    ap.add_argument('--charged-cap', type=int, default=20_000_000)
    args = ap.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), use_fast=True)
    raw = json.loads(PAIR_DATA.read_text(encoding='utf-8'))
    ranked = rank_pairs(tok, raw['pair_data'])
    total_signal = sum(r['changed_source_absent_pieces'] for r in ranked)
    total_charge = sum(r['charge'] for r in ranked)
    variants = {'all': raw}
    paths = {}
    curve = []
    for frac in args.fractions:
        n = max(1, int(round(frac * len(ranked))))
        selected_rows = ranked[:n]
        selected = {r['pair_id'] for r in selected_rows}
        var = filter_pair_data(raw, selected, frac, ranked)
        name = f'top{int(round(frac*100)):02d}'
        variants[name] = var
        out_path = OUT_ROOT / name / f'sparse_aux_pair_data_{name}.json'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(var, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        paths[name] = rel(out_path)
        sig = sum(r['changed_source_absent_pieces'] for r in selected_rows)
        ch = sum(r['charge'] for r in selected_rows)
        curve.append({'name': name, 'fraction': frac, 'n_pairs': n, 'full_activation_charge': ch, 'charge_frac': ch / total_charge, 'changed_source_absent_pieces': sig, 'signal_frac': sig / total_signal if total_signal else None, 'signal_per_charge': sig / ch if ch else 0.0, 'path': rel(out_path)})
    forecast = forecast_variants(tok, variants, args.charged_cap)
    result = {
        'status': 'BUILD_SPARSE_AUX_PAIR_DATA',
        'created_utc': now(),
        'source_pair_data': rel(PAIR_DATA),
        'total_pairs_ranked': len(ranked),
        'total_full_activation_charge_all_pairs': total_charge,
        'total_changed_source_absent_pieces_all_pairs': total_signal,
        'sparse_curve': curve,
        'variant_paths': paths,
        'wwm_replay_forecast_20m_charged_cap': forecast,
        'interpretation': 'Filtered pair-data variants are compatible with dual_view_corrected_trainer via --aux_pair_data_path. Forecast uses deterministic WWM replay, no model; it estimates how much auxiliary charge and target count each sparse variant would spend before the 20M charged cap.',
    }
    out_json = OUT_ROOT / 'sparse_aux_pair_data_summary.json'
    out_md = (USER_ROOT / 'research/documents/frontier_consolidation/data/sparse_aux_pair_data/sparse_aux_pair_data_summary.md')
    result['out_json'] = rel(out_json)
    result['out_md'] = rel(out_md)
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    lines = ['# research sparse auxiliary pair-data variants', '', '| variant | pair frac | full charge frac | signal frac | forecast updates | forecast main M | forecast aux M | forecast aux targets | path |', '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for c in curve:
        f = forecast.get(c['name'], {})
        lines.append(f"| {c['name']} | {c['fraction']:.2f} | {c['charge_frac']:.4f} | {c['signal_frac']:.4f} | {f.get('updates')} | {f.get('main_words',0)/1e6:.3f} | {f.get('aux_words',0)/1e6:.3f} | {f.get('aux_targets')} | `{c['path']}` |")
    f = forecast.get('all', {})
    lines += ['', f"All-pairs forecast: updates `{f.get('updates')}`, main `{f.get('main_words')}`, aux `{f.get('aux_words')}`, charged `{f.get('charged_words')}`, aux targets `{f.get('aux_targets')}`.", '', f"JSON: `{rel(out_json)}`"]
    out_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'variant_paths': paths, 'forecast': {k: {kk: vv for kk, vv in v.items() if kk != 'logs_sample'} for k, v in forecast.items()}, 'out_json': rel(out_json), 'out_md': rel(out_md)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
