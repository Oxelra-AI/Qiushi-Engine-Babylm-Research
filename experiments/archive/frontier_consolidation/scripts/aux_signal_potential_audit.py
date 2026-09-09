#!/usr/bin/env python3
"""research potential auxiliary-signal audit for legal compact source/rewrite pairs.

This CPU-only audit asks whether the current dual-view auxiliary target family is
mostly the non-copy edit phenomenon supported by Steps122-123, or diluted by copied
/ unchanged rewrite tokens.  It does not replay WWM masks; instead it analyzes the
potential target inventory for each legal pair in `aux_pair_data.json`.
"""
from __future__ import annotations

import difflib
import json
import pathlib
import re
import statistics
import time
from collections import Counter, defaultdict
from typing import Any

from transformers import AutoTokenizer

USER_ROOT = pathlib.Path('.').resolve()
STUDY = USER_ROOT / 'experiments/archive/frontier_consolidation'
WORKSPACE = STUDY
PAIR_DATA = WORKSPACE / 'data/aux_pair_data/aux_pair_data.json'
PAIR_JSONL = WORKSPACE / 'data/density_core_reinvestment_medium_riskhard/selected_compact_reinvest_pairs.jsonl'
TOKENIZER_DIR = WORKSPACE / 'data/compliant_tokenizer'
OUT_ROOT = WORKSPACE / 'data/aux_signal_potential_audit'

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
                # Restrict to real word/number material, not pure punctuation.
                if sp['is_word']:
                    chars.update(range(int(sp['start']), int(sp['end'])))
    return chars


def is_word_start(tok: str) -> bool:
    return tok.startswith('Ġ') or tok.startswith('▁')


def rewrite_token_groups(tokenizer, rewrite: str) -> tuple[list[int], list[tuple[int, int]], list[int]]:
    enc = tokenizer(rewrite, add_special_tokens=False, return_offsets_mapping=True)
    ids = [int(x) for x in enc['input_ids']]
    offs = [(int(a), int(b)) for a, b in enc['offset_mapping']]
    groups = []
    gid = -1
    for i, tid in enumerate(ids):
        ts = str(tokenizer.convert_ids_to_tokens(tid))
        if gid < 0 or is_word_start(ts) or i == 0:
            gid += 1
        groups.append(gid)
    return ids, offs, groups


def load_pair_texts() -> dict[str, dict[str, Any]]:
    out = {}
    with PAIR_JSONL.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            pid_raw = str(obj.get('pair_id'))
            for pid in {pid_raw, norm_pid(pid_raw), 'compact:' + norm_pid(pid_raw)}:
                out[pid] = obj
    return out


def quantiles(xs: list[float]) -> dict[str, float | None]:
    if not xs:
        return {'mean': None, 'p50': None, 'p90': None, 'p95': None, 'max': None}
    ys = sorted(xs)
    def q(p: float) -> float:
        idx = min(len(ys)-1, max(0, int(round(p * (len(ys)-1)))))
        return ys[idx]
    return {'mean': statistics.mean(xs), 'p50': q(0.5), 'p90': q(0.9), 'p95': q(0.95), 'max': max(xs)}


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), use_fast=True)
    raw = json.loads(PAIR_DATA.read_text(encoding='utf-8'))
    pdata = raw['pair_data']
    summary = raw.get('summary', {})
    pair_texts = load_pair_texts()
    rows = []
    missing_text = 0
    mismatch_lengths = 0
    for row_eid, rec in pdata.items():
        for pr in rec.get('pairs', []):
            pid = str(pr['pair_id'])
            txt = pair_texts.get(pid) or pair_texts.get(norm_pid(pid)) or pair_texts.get('compact:' + norm_pid(pid))
            if txt is None:
                missing_text += 1
                continue
            source = str(txt.get('source_text', ''))
            rewrite = str(txt.get('rewrite_text', ''))
            ids, offs, groups = rewrite_token_groups(tok, rewrite)
            if len(ids) != len(pr.get('rw_ids', [])):
                mismatch_lengths += 1
            src_ids = set(int(x) for x in pr.get('source_ids', []))
            changed_chars = changed_char_set(source, rewrite)
            # Piece-level counters and group-level counters.
            piece = Counter()
            group_flags: dict[int, Counter] = defaultdict(Counter)
            for tid, (a, b), g in zip(ids, offs, groups):
                if b <= a:
                    continue
                token_has_alnum = any(ch.isalnum() for ch in rewrite[a:b])
                if not token_has_alnum:
                    piece['punct_or_other'] += 1
                    group_flags[g]['punct_or_other'] += 1
                    continue
                is_changed = bool(set(range(a, b)) & changed_chars)
                is_absent = int(tid) not in src_ids
                piece['total_alnum_pieces'] += 1
                piece['changed_pieces'] += int(is_changed)
                piece['source_absent_pieces'] += int(is_absent)
                piece['changed_source_absent_pieces'] += int(is_changed and is_absent)
                piece['unchanged_or_copy_pieces'] += int(not (is_changed and is_absent))
                group_flags[g]['total_alnum_pieces'] += 1
                group_flags[g]['changed'] += int(is_changed)
                group_flags[g]['source_absent'] += int(is_absent)
                group_flags[g]['changed_source_absent'] += int(is_changed and is_absent)
            group_count = len([g for g, c in group_flags.items() if c.get('total_alnum_pieces', 0) > 0])
            changed_groups = sum(1 for c in group_flags.values() if c.get('changed', 0) > 0)
            absent_groups = sum(1 for c in group_flags.values() if c.get('source_absent', 0) > 0)
            changed_absent_groups = sum(1 for c in group_flags.values() if c.get('changed_source_absent', 0) > 0)
            source_words = int(pr.get('source_words', txt.get('source_words', 0)))
            rewrite_words = int(pr.get('rewrite_words', txt.get('rewrite_words', 0)))
            charge = source_words + 2 * rewrite_words
            rows.append({
                'row_example_id': int(row_eid),
                'pair_id': pid,
                'doc_id': str(txt.get('doc_id', '')),
                'source_words': source_words,
                'rewrite_words': rewrite_words,
                'aux_charge_per_activation': charge,
                'total_alnum_pieces': int(piece['total_alnum_pieces']),
                'changed_pieces': int(piece['changed_pieces']),
                'source_absent_pieces': int(piece['source_absent_pieces']),
                'changed_source_absent_pieces': int(piece['changed_source_absent_pieces']),
                'changed_groups': changed_groups,
                'source_absent_groups': absent_groups,
                'changed_source_absent_groups': changed_absent_groups,
                'alnum_groups': group_count,
                'utility_changed_absent_piece_per_charge': (piece['changed_source_absent_pieces'] / charge) if charge else 0.0,
                'utility_changed_absent_group_per_charge': (changed_absent_groups / charge) if charge else 0.0,
                'content_overlap': float(txt.get('content_overlap', 0.0)),
                'rewrite_head': rewrite[:120].replace('\n', ' '),
            })
    totals = Counter()
    charge_total = 0
    for r in rows:
        charge_total += r['aux_charge_per_activation']
        for k in ['total_alnum_pieces', 'changed_pieces', 'source_absent_pieces', 'changed_source_absent_pieces', 'changed_groups', 'source_absent_groups', 'changed_source_absent_groups', 'alnum_groups']:
            totals[k] += int(r[k])
    sorted_rows = sorted(rows, key=lambda r: (r['utility_changed_absent_piece_per_charge'], r['changed_source_absent_pieces']), reverse=True)
    total_signal = max(1, totals['changed_source_absent_pieces'])
    curves = []
    for frac in [0.05, 0.10, 0.20, 0.30, 0.40, 0.60, 0.80, 1.00]:
        n = max(1, int(round(frac * len(sorted_rows))))
        part = sorted_rows[:n]
        ch = sum(r['aux_charge_per_activation'] for r in part)
        sig = sum(r['changed_source_absent_pieces'] for r in part)
        curves.append({
            'top_fraction_pairs': frac,
            'n_pairs': n,
            'aux_charge_per_full_activation': ch,
            'fraction_charge': ch / charge_total if charge_total else None,
            'changed_source_absent_pieces': sig,
            'fraction_signal_captured': sig / total_signal,
            'signal_per_charge': sig / ch if ch else 0.0,
        })
    by_charge_eff = quantiles([r['utility_changed_absent_piece_per_charge'] for r in rows])
    result = {
        'status': 'AUX_SIGNAL_POTENTIAL_AUDIT',
        'created_utc': now(),
        'pair_data': rel(PAIR_DATA),
        'pair_jsonl': rel(PAIR_JSONL),
        'tokenizer': rel(TOKENIZER_DIR),
        'aux_pair_data_summary': summary,
        'n_pairs_analyzed': len(rows),
        'missing_text': missing_text,
        'rewrite_token_length_mismatches_vs_pair_data': mismatch_lengths,
        'total_aux_charge_per_activation_all_pairs': charge_total,
        'totals': dict(totals),
        'fractions': {
            'changed_piece_frac_of_alnum': totals['changed_pieces'] / totals['total_alnum_pieces'] if totals['total_alnum_pieces'] else None,
            'source_absent_piece_frac_of_alnum': totals['source_absent_pieces'] / totals['total_alnum_pieces'] if totals['total_alnum_pieces'] else None,
            'changed_source_absent_piece_frac_of_alnum': totals['changed_source_absent_pieces'] / totals['total_alnum_pieces'] if totals['total_alnum_pieces'] else None,
            'changed_source_absent_group_frac_of_alnum_groups': totals['changed_source_absent_groups'] / totals['alnum_groups'] if totals['alnum_groups'] else None,
        },
        'utility_quantiles': by_charge_eff,
        'sparse_utility_curve': curves,
        'top_pairs_by_utility': sorted_rows[:40],
        'interpretation': 'Potential inventory only: actual WWM chooses a subset of word groups. If changed_source_absent signal is a small fraction of current aux targets/charge, a sparse source-absent changed-span auxiliary sampler may preserve the research-123 mechanism while displacing fewer main-stream words.',
    }
    out_json = OUT_ROOT / 'aux_signal_potential_audit.json'
    out_md = (USER_ROOT / 'research/documents/frontier_consolidation/data/aux_signal_potential_audit/aux_signal_potential_audit.md')
    result['out_json'] = rel(out_json)
    result['out_md'] = rel(out_md)
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    lines = ['# research auxiliary signal potential audit', '', f"Pairs analyzed: `{len(rows)}`; total full-activation aux charge: `{charge_total}` words.", '', '## Fractions', '', '```json', json.dumps(result['fractions'], indent=2), '```', '', '## Sparse utility curve', '', '| top pair fraction | pairs | charge frac | signal captured | signal/charge |', '|---:|---:|---:|---:|---:|']
    for c in curves:
        lines.append(f"| {c['top_fraction_pairs']:.2f} | {c['n_pairs']} | {c['fraction_charge']:.4f} | {c['fraction_signal_captured']:.4f} | {c['signal_per_charge']:.6f} |")
    lines += ['', f"JSON: `{rel(out_json)}`"]
    out_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'n_pairs_analyzed': len(rows), 'fractions': result['fractions'], 'sparse_utility_curve': curves, 'out_json': rel(out_json), 'out_md': rel(out_md)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
