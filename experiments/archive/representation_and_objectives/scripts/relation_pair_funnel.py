#!/usr/bin/env python3
"""research structural pair-funnel for a legal two-context/two-target relation objective.

Purpose
-------
After research showed local pivot/target compatibility is saturated in failing
models, a relation route is only scientifically live if the legal corpus contains
diverse pairs of events that can form a true two-context/two-target interaction:
  m_a = s(C_a,T_a)-s(C_a,T_b), m_b = s(C_b,T_b)-s(C_b,T_a).
This script performs the no-model structural part of that test. It uses only the
research hand-coded legal event labels, the legal compact tail text, and the
shared legal tokenizer. It does not score with any model and does not train.

The pair construction is frozen before any checkpoint scoring. It enforces equal
token length and token word-start/continuation pattern for the two targets,
relation-family/target-class/frequency/distance strata, different rows, different
target strings, no direct target-string leakage in the opposite context, and a
cap on target multiplicity.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from transformers import AutoTokenizer

USER_ROOT = Path('.').resolve()
for p in [USER_ROOT/'experiments/archive/compact_experience/scripts', USER_ROOT/'experiments/archive/representation_and_objectives/scripts']:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pvdm_continuation_trainer as cont  # noqa: E402
import sparse_relation_aux_trainer as relaux  # noqa: E402

DEFAULT_OUT = Path('experiments/archive/representation_and_objectives/data/relation_pair_funnel')
DEFAULT_NOTE = Path('research/notes/representation_and_objectives/relation_pair_funnel.md')
CATEGORIES = {'physical_change','causal_connector','temporal','spatial','comparative','negation'}
WORD_RE = re.compile(r"\S+")
PUNCT_STRIP = "\"'“”‘’.,!?;:()[]{}<>«»‹›*_-=+/\\|`~"


def now_utc() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def stable_u(seed: int, *items: Any) -> float:
    h = hashlib.blake2b(digest_size=8)
    h.update(str(seed).encode())
    for it in items:
        h.update(b'\0'); h.update(str(it).encode('utf-8', errors='ignore'))
    return int.from_bytes(h.digest(), 'big') / float(2**64)


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            yield json.loads(line)


def qstats(vals: list[float]) -> dict[str, Any]:
    xs = sorted(float(v) for v in vals if math.isfinite(float(v)))
    if not xs:
        return {'n': 0}
    def q(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        idx = p * (len(xs)-1); lo = int(math.floor(idx)); hi = int(math.ceil(idx))
        if lo == hi:
            return xs[lo]
        return xs[lo]*(hi-idx) + xs[hi]*(idx-lo)
    return {'n': len(xs), 'mean': float(sum(xs)/len(xs)), 'median': q(0.5), 'std': float(np.std(xs)), 'p05': q(0.05), 'p95': q(0.95), 'min': xs[0], 'max': xs[-1]}


def word_spans(text: str) -> list[tuple[str, str, int, int]]:
    out=[]
    for m in WORD_RE.finditer(text):
        raw=m.group(0)
        norm=raw.lower().strip(PUNCT_STRIP)
        out.append((raw,norm,m.start(),m.end()))
    return out


def context_without_target(text: str, target_i: int) -> str:
    ws = word_spans(text)
    if 0 <= target_i < len(ws):
        s,e = ws[target_i][2], ws[target_i][3]
        return (text[:s] + ' [MASKTARGET] ' + text[e:]).lower()
    return text.lower()


def word_norm_at(text: str, idx: int) -> str:
    ws = word_spans(text)
    if 0 <= idx < len(ws):
        return ws[idx][1]
    return ''


def contains_norm_as_word(context_lc: str, norm: str) -> bool:
    if not norm:
        return False
    # for short/function targets, substring tests overreject; use token normalization
    return any(tok == norm for _raw, tok, _s, _e in word_spans(context_lc))


def target_token_signature(tokenizer, target_text: str) -> dict[str, Any]:
    text = str(target_text).strip()
    enc = tokenizer(text, add_special_tokens=False)
    ids = [int(x) for x in enc.get('input_ids', [])]
    toks = tokenizer.convert_ids_to_tokens(ids)
    # Byte-level BPE word-start markers vary; this robustly separates starts from continuations.
    pattern=[]
    for j,tok in enumerate(toks):
        s=str(tok)
        starts = (j == 0) or s.startswith('Ġ') or s.startswith('▁') or s.startswith(' ')
        pattern.append('S' if starts else 'C')
    return {'token_ids': ids, 'tokens': toks, 'token_len': len(ids), 'token_pattern': ''.join(pattern)}


def event_to_rec(row: dict[str, Any], ev: dict[str, Any], tokenizer, *, row_order: int, max_target_token_len: int, strict_context_leak: bool) -> tuple[dict[str, Any] | None, str | None]:
    meta = relaux._event_meta(ev)
    cat = str(ev.get('category'))
    if cat not in CATEGORIES:
        return None, 'category_not_allowed'
    if ev.get('control_match') is None:
        return None, 'no_control_match'
    target_i = int(ev.get('target_i', -1)); pivot_i = int(ev.get('pivot_i', -1)); control_i = int(ev.get('control_i', -1))
    if len({target_i, pivot_i, control_i}) < 3:
        return None, 'role_overlap'
    target_norm = str(meta.get('target_norm', '')).lower().strip(PUNCT_STRIP)
    pivot_norm = str(meta.get('pivot_norm', '')).lower().strip(PUNCT_STRIP)
    control_norm = str(meta.get('control_norm', '')).lower().strip(PUNCT_STRIP)
    if not target_norm or target_norm in {'[',']','*'}:
        return None, 'bad_target_norm'
    sig = target_token_signature(tokenizer, ev.get('target', target_norm))
    if sig['token_len'] <= 0:
        return None, 'target_token_len_zero'
    if sig['token_len'] > max_target_token_len:
        return None, 'target_token_len_gt_max'
    text = str(row.get('text',''))
    masked_context_lc = context_without_target(text, target_i)
    if strict_context_leak and contains_norm_as_word(masked_context_lc, target_norm):
        return None, 'target_leaks_in_own_context'
    cm = ev.get('control_match', {}) if isinstance(ev.get('control_match', {}), dict) else {}
    rec = {
        'uid': f"{row.get('tail_row_idx')}:{ev.get('category')}:{ev.get('pivot_i')}:{ev.get('target_i')}:{ev.get('event_rank','')}",
        'row_order': int(row_order),
        'tail_row_idx': int(row.get('tail_row_idx', -1)),
        'orig_row_idx': int(row.get('orig_row_idx', -1)),
        'example_id': row.get('example_id'),
        'source': row.get('source'),
        'words': int(row.get('words', 0)),
        'category': cat,
        'pivot_i': pivot_i,
        'target_i': target_i,
        'control_i': control_i,
        'pivot': ev.get('pivot'),
        'target': ev.get('target'),
        'control': ev.get('control'),
        'pivot_norm': pivot_norm,
        'target_norm': target_norm,
        'control_norm': control_norm,
        'target_class': str(meta.get('target_class')),
        'pivot_anchor_class': str(meta.get('pivot_anchor_class')),
        'target_freq_bin': str(meta.get('target_freq_bin')),
        'target_freq_bin_id': int(meta.get('target_freq_bin_id', -1)),
        'pivot_freq_bin': str(meta.get('pivot_freq_bin')),
        'pivot_freq_bin_id': int(meta.get('pivot_freq_bin_id', -1)),
        'distance_bin': str(meta.get('distance_bin')),
        'distance_bin_id': int(meta.get('distance_bin_id', -1)),
        'pivot_target_distance': int(meta.get('pivot_target_distance', -1)),
        'control_distance_abs_diff': int(meta.get('control_distance_abs_diff', cm.get('distance_abs_diff', 999))),
        'control_freq_bin_abs_diff': int(meta.get('control_freq_bin_abs_diff', cm.get('freq_bin_abs_diff', 999))),
        'token_ids': sig['token_ids'],
        'token_text': ' '.join(str(x) for x in sig['tokens']),
        'target_token_len': sig['token_len'],
        'target_token_pattern': sig['token_pattern'],
        'context_no_target_lc': masked_context_lc,
        'text': text,
    }
    return rec, None


def strict_key(rec: dict[str, Any]) -> tuple[Any, ...]:
    return (rec['category'], rec['target_class'], rec['target_token_len'], rec['target_token_pattern'], rec['target_freq_bin'], rec['distance_bin'])


def relaxed_keys(rec: dict[str, Any]) -> list[tuple[Any, ...]]:
    return [
        ('L0', rec['category'], rec['target_class'], rec['target_token_len'], rec['target_token_pattern'], rec['target_freq_bin'], rec['distance_bin']),
        ('L1', rec['category'], rec['target_class'], rec['target_token_len'], rec['target_token_pattern'], rec['target_freq_bin']),
        ('L2', rec['category'], rec['target_class'], rec['target_token_len'], rec['target_token_pattern']),
        ('L3', rec['category'], rec['target_token_len'], rec['target_token_pattern']),
    ]


def pair_cost(a: dict[str, Any], b: dict[str, Any]) -> float:
    return (
        1000.0 * (a['category'] != b['category']) +
        100.0 * (a['target_class'] != b['target_class']) +
        50.0 * (a['target_token_len'] != b['target_token_len']) +
        25.0 * (a['target_token_pattern'] != b['target_token_pattern']) +
        5.0 * abs(int(a['target_freq_bin_id']) - int(b['target_freq_bin_id'])) +
        2.0 * abs(int(a['distance_bin_id']) - int(b['distance_bin_id'])) +
        0.1 * abs(int(a['pivot_target_distance']) - int(b['pivot_target_distance'])) +
        0.001 * abs(int(a['words']) - int(b['words']))
    )


def compatible(a: dict[str, Any], b: dict[str, Any], *, max_freq_delta: int, max_distance_delta: int, max_same_source: bool, cross_leak_filter: bool) -> tuple[bool, str]:
    if a['tail_row_idx'] == b['tail_row_idx']:
        return False, 'same_row'
    if str(a['target_norm']) == str(b['target_norm']):
        return False, 'same_target_norm'
    if a['target_token_len'] != b['target_token_len'] or a['target_token_pattern'] != b['target_token_pattern']:
        return False, 'token_signature_mismatch'
    if a['category'] != b['category']:
        return False, 'category_mismatch'
    if a['target_class'] != b['target_class']:
        return False, 'target_class_mismatch'
    if abs(int(a['target_freq_bin_id']) - int(b['target_freq_bin_id'])) > max_freq_delta:
        return False, 'target_freq_delta'
    if abs(int(a['distance_bin_id']) - int(b['distance_bin_id'])) > max_distance_delta:
        return False, 'distance_delta'
    if not max_same_source and str(a.get('source')) == str(b.get('source')):
        return False, 'same_source'
    if cross_leak_filter:
        if contains_norm_as_word(a['context_no_target_lc'], str(b['target_norm'])):
            return False, 'b_target_leaks_in_a_context'
        if contains_norm_as_word(b['context_no_target_lc'], str(a['target_norm'])):
            return False, 'a_target_leaks_in_b_context'
    return True, 'ok'


def load_events(args, tokenizer) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rejects=Counter(); counts=Counter(); by_cat=Counter(); sources=Counter(); rows_with=Counter()
    events=[]
    t0=time.time()
    for row_order, (label_row, text_row) in enumerate(zip(read_jsonl(Path(args.labels_jsonl)), read_jsonl(Path(args.tail_jsonl)))):
        if args.max_rows and row_order >= args.max_rows:
            break
        # Prefer text from the compact tail file; labels duplicate it but this makes lineage explicit.
        label_row = dict(label_row)
        label_row['text'] = text_row.get('text', label_row.get('text',''))
        label_row['words'] = text_row.get('words', label_row.get('words',0))
        counts['rows_seen'] += 1
        row_added = 0
        for erank, ev in enumerate(label_row.get('events', []) or []):
            ev = dict(ev); ev.setdefault('event_rank', erank)
            counts['raw_events'] += 1
            rec, why = event_to_rec(label_row, ev, tokenizer, row_order=row_order, max_target_token_len=args.max_target_token_len, strict_context_leak=args.strict_own_context_leak)
            if rec is None:
                rejects[str(why)] += 1; rejects[f'{why}::{ev.get("category")}'] += 1; continue
            if rec['control_distance_abs_diff'] > args.max_control_distance_abs_diff:
                rejects['control_distance_mismatch'] += 1; rejects[f'control_distance_mismatch::{rec["category"]}'] += 1; continue
            if rec['control_freq_bin_abs_diff'] > args.max_control_freq_bin_abs_diff:
                rejects['control_freq_mismatch'] += 1; rejects[f'control_freq_mismatch::{rec["category"]}'] += 1; continue
            events.append(rec); by_cat[rec['category']] += 1; sources[str(rec.get('source'))] += 1; row_added += 1
        if row_added:
            rows_with['rows_with_events'] += 1
        if counts['rows_seen'] == 1 or counts['rows_seen'] % 50000 == 0:
            print(json.dumps({'event':'load_progress','rows_seen':counts['rows_seen'],'raw_events':counts['raw_events'],'kept_events':len(events),'elapsed_sec':round(time.time()-t0,1)}), flush=True)
    summary={'counts':dict(counts), 'kept_events':len(events), 'rejects':dict(rejects), 'kept_by_category':dict(by_cat), 'kept_sources':dict(sources), 'rows_with_events':rows_with.get('rows_with_events',0)}
    return events, summary


def build_pairs(events: list[dict[str, Any]], args) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # Process high-quality exact strata first, then relaxed fallback. Greedy matching with target multiplicity caps.
    by_key=defaultdict(list)
    for ev in events:
        for key in relaxed_keys(ev):
            by_key[key].append(ev)
    # main candidate order is deterministic and stratified by category to preserve rare families.
    ev_order=sorted(events, key=lambda e: (e['category'], e['target_class'], e['target_token_len'], e['target_freq_bin_id'], stable_u(args.seed, 'event_order', e['uid'])))
    used_event=set(); target_use=Counter(); pairs=[]; failure=Counter(); level_count=Counter(); costs=[]
    by_cat=Counter(); source_pairs=Counter()
    samples=[]
    for a in ev_order:
        if a['uid'] in used_event:
            continue
        if target_use[a['target_norm']] >= args.max_pairs_per_target:
            failure['target_cap_a'] += 1; continue
        best=None; best_level=None; best_cost=None; best_reason=Counter()
        for level, key in enumerate(relaxed_keys(a)):
            pool = by_key.get(key, [])
            if not pool:
                continue
            # Do not scan very large strata in arbitrary order; inspect stable best subset plus local neighboring candidates.
            candidates = sorted(pool, key=lambda b: (pair_cost(a,b), stable_u(args.seed, 'cand', a['uid'], b['uid'])))[:args.candidate_scan_per_level]
            for b in candidates:
                if b['uid'] == a['uid'] or b['uid'] in used_event:
                    best_reason['used_or_self'] += 1; continue
                if target_use[b['target_norm']] >= args.max_pairs_per_target:
                    best_reason['target_cap_b'] += 1; continue
                ok, why = compatible(a,b, max_freq_delta=args.max_pair_target_freq_delta, max_distance_delta=args.max_pair_distance_delta, max_same_source=args.allow_same_source, cross_leak_filter=args.cross_context_leak_filter)
                if not ok:
                    best_reason[why] += 1; continue
                c=pair_cost(a,b)
                best=b; best_level=level; best_cost=c; break
            if best is not None:
                break
        if best is None:
            failure['no_partner'] += 1
            for k,v in best_reason.items():
                failure[f'no_partner_detail::{k}'] += v
            continue
        b=best
        used_event.add(a['uid']); used_event.add(b['uid'])
        target_use[a['target_norm']] += 1; target_use[b['target_norm']] += 1
        level_count[str(best_level)] += 1; costs.append(float(best_cost)); by_cat[a['category']] += 1
        source_pairs[f"{a.get('source')}||{b.get('source')}"] += 1
        rec={
            'pair_id': f"p{len(pairs):07d}",
            'match_level': int(best_level),
            'pair_cost': float(best_cost),
            'category': a['category'],
            'target_class': a['target_class'],
            'target_token_len': a['target_token_len'],
            'target_token_pattern': a['target_token_pattern'],
            'target_freq_bin_a': a['target_freq_bin'],
            'target_freq_bin_b': b['target_freq_bin'],
            'target_freq_bin_delta': abs(int(a['target_freq_bin_id'])-int(b['target_freq_bin_id'])),
            'distance_bin_a': a['distance_bin'],
            'distance_bin_b': b['distance_bin'],
            'distance_bin_delta': abs(int(a['distance_bin_id'])-int(b['distance_bin_id'])),
            'row_a': a['tail_row_idx'],
            'row_b': b['tail_row_idx'],
            'source_a': a.get('source'),
            'source_b': b.get('source'),
            'target_norm_a': a['target_norm'],
            'target_norm_b': b['target_norm'],
            'target_ids_a': a['token_ids'],
            'target_ids_b': b['token_ids'],
            'pivot_norm_a': a['pivot_norm'],
            'pivot_norm_b': b['pivot_norm'],
            'pivot_i_a': a['pivot_i'],
            'pivot_i_b': b['pivot_i'],
            'target_i_a': a['target_i'],
            'target_i_b': b['target_i'],
            'text_a': a['text'],
            'text_b': b['text'],
        }
        pairs.append(rec)
        if len(samples) < args.sample_pairs:
            samples.append(rec)
        if args.max_pairs and len(pairs) >= args.max_pairs:
            break
    summary={
        'input_events': len(events),
        'pairs': len(pairs),
        'events_used': len(used_event),
        'event_use_fraction': len(used_event)/len(events) if events else None,
        'target_unique_used': len(target_use),
        'target_use_top20': target_use.most_common(20),
        'pairs_by_category': dict(by_cat),
        'match_levels': dict(level_count),
        'pair_cost': qstats(costs),
        'failure_counts': dict(failure),
        'source_pair_top20': source_pairs.most_common(20),
        'sample_pairs': samples,
    }
    return pairs, summary


def audit_natural_mask_inclusion(pairs: list[dict[str, Any]], args) -> dict[str, Any]:
    # No model/data-loader replay here. Estimate independent WWM inclusion probability from word-group masking target probability.
    # For target word-groups, exact-count WWM chooses about 15% groups. Pair needs both targets masked in their own rows for a zero-extra-forward four-cell gather.
    p = float(args.wwm_group_mask_prob)
    est_pair_both = p*p
    return {
        'wwm_group_mask_prob_assumed': p,
        'estimated_pairs_with_both_targets_masked_per_epoch': est_pair_both * len(pairs),
        'estimated_pairs_with_both_targets_masked_10_epochs': est_pair_both * len(pairs) * 10,
        'interpretation': 'This is an independence estimate only. A future no-update calibration should replay the exact WWM mask generator and count realized pairs; forced target masking would require a same-mask WWM reference.'
    }


def write_outputs(events: list[dict[str, Any]], load_summary: dict[str, Any], pairs: list[dict[str, Any]], pair_summary: dict[str, Any], args) -> None:
    out=Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    pairs_path=out/'relation_pair_pool.jsonl'
    with pairs_path.open('w', encoding='utf-8') as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
    csv_path=out/'relation_pair_pool_summary.csv'
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        keys=['pair_id','match_level','pair_cost','category','target_class','target_token_len','target_token_pattern','target_freq_bin_a','target_freq_bin_b','target_freq_bin_delta','distance_bin_a','distance_bin_b','distance_bin_delta','row_a','row_b','source_a','source_b','target_norm_a','target_norm_b','pivot_norm_a','pivot_norm_b']
        w=csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for p in pairs:
            w.writerow({k:p.get(k) for k in keys})
    event_strata=Counter()
    for e in events:
        event_strata[str(strict_key(e))] += 1
    nat=audit_natural_mask_inclusion(pairs,args)
    summary={
        'status':'RELATION_PAIR_FUNNEL',
        'created_utc':now_utc(),
        'purpose':'Structural legal pair yield for two-context/two-target relation interaction objective before any model scoring or H100 training.',
        'inputs':{
            'labels_jsonl':str(args.labels_jsonl),
            'tail_jsonl':str(args.tail_jsonl),
            'tokenizer_path':str(args.tokenizer_path),
        },
        'parameters':vars(args),
        'load_summary':load_summary,
        'strict_strata_count':len(event_strata),
        'strict_strata_top20':event_strata.most_common(20),
        'pair_summary':pair_summary,
        'natural_mask_estimate':nat,
        'outputs':{'pairs_jsonl':str(pairs_path),'pairs_csv':str(csv_path),'note':str(args.note_path)},
    }
    summary_path=out/'relation_pair_funnel_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    lines=[
        '# research — legal relation pair funnel',
        '',
        'This no-model structural audit tests whether the research legal event labels can support a two-context/two-target interaction objective after research showed local relation compatibility is saturated.',
        '',
        f"Events kept after legal/control/token/leak filters: **{load_summary['kept_events']}** from {load_summary['counts'].get('raw_events')} raw events in {load_summary['counts'].get('rows_seen')} rows.",
        f"Pairs constructed: **{pair_summary['pairs']}** using {pair_summary['events_used']} events; event-use fraction {pair_summary['event_use_fraction']:.4f}.",
        '',
        '## Pairs by relation family',
        '',
        '| family | pairs |',
        '|---|---:|',
    ]
    for k,v in sorted(pair_summary['pairs_by_category'].items(), key=lambda kv:(-kv[1],kv[0])):
        lines.append(f'| {k} | {v} |')
    lines += ['', '## Matching', '', f"Match levels: `{pair_summary['match_levels']}`", f"Pair cost: `{pair_summary['pair_cost']}`", '', 'Top target multiplicities:', '']
    for t,c in pair_summary['target_use_top20'][:10]:
        lines.append(f'- {t}: {c}')
    lines += ['', '## Natural-mask exposure estimate', '', f"With ordinary WWM p≈{nat['wwm_group_mask_prob_assumed']}, expected pairs with both target groups naturally masked per epoch ≈ {nat['estimated_pairs_with_both_targets_masked_per_epoch']:.1f}, over 10 epochs ≈ {nat['estimated_pairs_with_both_targets_masked_10_epochs']:.1f}. This only estimates a zero-extra-forward gather path; exact replay is still needed.", '', 'Files:', f'- summary: `{summary_path}`', f'- pairs: `{pairs_path}`', f'- csv: `{csv_path}`']
    Path(args.note_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.note_path).write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'status':summary['status'], 'events_kept':load_summary['kept_events'], 'pairs':pair_summary['pairs'], 'summary':str(summary_path), 'note':str(args.note_path)}, indent=2), flush=True)


def build_args():
    ap=argparse.ArgumentParser()
    ap.add_argument('--labels_jsonl', default=str(cont.DEFAULT_LABELS))
    ap.add_argument('--tail_jsonl', default=str(cont.DEFAULT_TAIL))
    ap.add_argument('--tokenizer_path', default=str(cont.DEFAULT_TOKENIZER))
    ap.add_argument('--output_dir', default=str(DEFAULT_OUT))
    ap.add_argument('--note_path', default=str(DEFAULT_NOTE))
    ap.add_argument('--max_rows', type=int, default=0)
    ap.add_argument('--max_pairs', type=int, default=0)
    ap.add_argument('--max_target_token_len', type=int, default=4)
    ap.add_argument('--max_control_distance_abs_diff', type=int, default=1)
    ap.add_argument('--max_control_freq_bin_abs_diff', type=int, default=1)
    ap.add_argument('--max_pair_target_freq_delta', type=int, default=0)
    ap.add_argument('--max_pair_distance_delta', type=int, default=1)
    ap.add_argument('--max_pairs_per_target', type=int, default=32)
    ap.add_argument('--candidate_scan_per_level', type=int, default=512)
    ap.add_argument('--sample_pairs', type=int, default=40)
    ap.add_argument('--strict_own_context_leak', action='store_true', default=True)
    ap.add_argument('--no_strict_own_context_leak', dest='strict_own_context_leak', action='store_false')
    ap.add_argument('--cross_context_leak_filter', action='store_true', default=True)
    ap.add_argument('--no_cross_context_leak_filter', dest='cross_context_leak_filter', action='store_false')
    ap.add_argument('--allow_same_source', action='store_true')
    ap.add_argument('--wwm_group_mask_prob', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=43023)
    return ap.parse_args()


def main():
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    t0=time.time(); args=build_args()
    tokenizer=AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    events, load_summary = load_events(args, tokenizer)
    pairs, pair_summary = build_pairs(events, args)
    pair_summary['runtime_sec_before_write'] = round(time.time()-t0,2)
    write_outputs(events, load_summary, pairs, pair_summary, args)


if __name__ == '__main__':
    main()
