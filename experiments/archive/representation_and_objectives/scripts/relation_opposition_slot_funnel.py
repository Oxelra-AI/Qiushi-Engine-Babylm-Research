#!/usr/bin/env python3
"""research: active-token opposition/shared-slot relation-pair funnel.

This builds a stricter legal pair reservoir after research showed that arbitrary
same-frequency target pairing mostly measures attested sentence fit. Every event
is inherited from the research active-token mapping, so target token positions and
ids correspond to the actual 256-token training view. The new pairing requires
hand-coded opposite relation pivots and a more similar target-side local slot.

No model is loaded and no training or evaluation examples are scored.
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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer

USER_ROOT = Path('.').resolve()
for p in [USER_ROOT/'experiments/archive/compact_experience/scripts', USER_ROOT/'experiments/archive/representation_and_objectives/scripts']:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import relation_active_pair_funnel as active_funnel  # noqa: E402
import pvdm_continuation_trainer as cont  # noqa: E402

DEFAULT_OUT = Path('experiments/archive/representation_and_objectives/data/relation_opposition_slot_funnel')
DEFAULT_NOTE = Path('research/notes/representation_and_objectives/relation_opposition_slot_funnel.md')
PUNCT_STRIP = "\"'“”‘’.,!?;:()[]{}<>«»‹›*_-=+/\\|`~"
WORD_RE = re.compile(r"\S+")

SPEAKER_OR_ARTIFACT = {
    'chi','mot','fat','mar','bro','sis','mom','dad','par','inv','int','exp','add','com','act','pho','sit','gra','gpx','x','xx','xxx','uh','uhhuh','huh','hm','yeah','okay','ok','ooh','ah','oh'
}
STOP_LOCAL = {
    'a','an','the','and','or','but','of','for','to','from','with','without','by','as','at','in','on','up','down','out','off','over',
    'this','that','these','those','there','here','then','than','also','very','just','only','some','any','all','another','other',
    'i','you','he','she','it','we','they','me','him','her','us','them','my','your','his','its','our','their','what','which','who',
    'where','why','how','when','yes','no','not','do','does','did','have','has','had','is','are','was','were','be','been','being',
    'can','could','will','would','should','may','might','must','shall','one','two','three','four','five','six','seven','eight','nine','ten',
    'said','say','says','get','got','go','come','came','see','look','like','make','made','take','took','put','let','want','need','know','think'
} | SPEAKER_OR_ARTIFACT
GENERIC_TARGETS = STOP_LOCAL | {'thing','things','something','anything','everything','someone','people','person','way','time','day','year','part','kind','sort','place','work','use','used','using','good','bad','new','old','first','last','many','much','more','less','same','different','right','left','anyway','actually','really','well','course'}

# Legal, hand-coded opposition axes. Words not present in this table are simply
# not used for this stricter reservoir.
OPPOSITION: dict[str, tuple[str, int]] = {}

def add_axis(axis: str, pos: list[str], neg: list[str]) -> None:
    for w in pos:
        OPPOSITION[w] = (axis, +1)
    for w in neg:
        OPPOSITION[w] = (axis, -1)

add_axis('temporal_before_after', ['after'], ['before'])
add_axis('spatial_vertical', ['above'], ['below','under','beneath'])
add_axis('spatial_inclusion', ['inside','within','into'], ['outside'])
add_axis('comparative_amount', ['more','greater','larger','higher','faster','longer','better','most'], ['less','smaller','lower','slower','shorter','worse','least'])
add_axis('change_amount', ['increase','increases','increased','increasing','rise','rises','rose','risen','rising'], ['decrease','decreases','decreased','decreasing','fall','falls','fell','fallen','falling'])
add_axis('size_change', ['grow','grows','grew','grown','growing','expand','expands','expanded','expanding'], ['shrink','shrinks','shrank','shrunk','shrinking','contract','contracts','contracted','contracting'])
add_axis('add_remove', ['add','adds','added','adding','fill','fills','filled','filling'], ['remove','removes','removed','removing','empty','empties','emptied','emptying'])
add_axis('open_close', ['open','opens','opened','opening'], ['close','closes','closed','closing'])
add_axis('enter_leave', ['enter','enters','entered','entering'], ['leave','leaves','left','leaving'])
add_axis('build_destroy', ['build','builds','built','building','create','creates','created','creating','fix','fixes','fixed','fixing'], ['destroy','destroys','destroyed','destroying','break','breaks','broke','broken','breaking'])
add_axis('push_pull', ['push','pushes','pushed','pushing'], ['pull','pulls','pulled','pulling'])
add_axis('heat_cool', ['heat','heats','heated','heating','melt','melts','melted','melting'], ['cool','cools','cooled','cooling','freeze','freezes','froze','frozen','freezing'])
add_axis('condition_exception', ['if'], ['unless'])
add_axis('cause_concession', ['because','therefore','thereby','thus','hence'], ['although','though','whereas'])


def now_utc() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def norm_word(w: str) -> str:
    return str(w).strip(PUNCT_STRIP).lower()


def stable_u(seed: int, *items: Any) -> float:
    h = hashlib.blake2b(digest_size=8)
    h.update(str(seed).encode())
    for it in items:
        h.update(b'\0'); h.update(str(it).encode('utf-8', errors='ignore'))
    return int.from_bytes(h.digest(), 'big') / float(2**64)


def qstats(vals: list[float]) -> dict[str, Any]:
    xs = sorted(float(v) for v in vals if math.isfinite(float(v)))
    if not xs:
        return {'n': 0}
    arr = np.asarray(xs, dtype=np.float64)
    def q(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        idx = p * (len(xs)-1); lo = int(math.floor(idx)); hi = int(math.ceil(idx))
        if lo == hi:
            return xs[lo]
        return xs[lo]*(hi-idx)+xs[hi]*(idx-lo)
    return {'n': len(xs), 'mean': float(arr.mean()), 'std': float(arr.std()), 'median': q(0.5), 'p05': q(0.05), 'p25': q(0.25), 'p75': q(0.75), 'p95': q(0.95), 'min': xs[0], 'max': xs[-1]}


def side_of(ev: dict[str, Any]) -> str:
    return 'after' if int(ev['target_gid']) > int(ev['pivot_gid']) else 'before'


def words_for(ev: dict[str, Any]) -> list[str]:
    return [m.group(0) for m in WORD_RE.finditer(str(ev.get('text','')))]


def event_slot(ev: dict[str, Any], radius: int) -> dict[str, Any]:
    ws = words_for(ev)
    t = int(ev['target_gid']); p = int(ev['pivot_gid']); c = int(ev.get('control_gid', -9999))
    lo, hi = max(0, t-radius), min(len(ws), t+radius+1)
    surface = []
    content = []
    skeleton = []
    for i in range(lo, hi):
        nw = norm_word(ws[i])
        if not nw:
            continue
        if i == t:
            skeleton.append('<T>'); continue
        if i == p:
            skeleton.append('<P>'); continue
        if i == c:
            skeleton.append('<C>'); continue
        skeleton.append(nw)
        surface.append(nw)
        if nw not in STOP_LOCAL and len(nw) > 2 and not nw.isdigit():
            content.append(nw)
    # Immediate two-token shell around the target with the pivot marked, useful for
    # exposing shared syntactic slots without learned parsing.
    shell = []
    for i in range(max(0, t-2), min(len(ws), t+3)):
        if i == t:
            shell.append('<T>')
        elif i == p:
            shell.append('<P>')
        else:
            nw = norm_word(ws[i])
            if nw:
                shell.append(nw)
    return {'surface_set': sorted(set(surface)), 'content_set': sorted(set(content)), 'skeleton': ' '.join(skeleton), 'shell': ' '.join(shell), 'left_word': norm_word(ws[t-1]) if t-1 >= 0 else '', 'right_word': norm_word(ws[t+1]) if t+1 < len(ws) else ''}


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def enrich_events(events: list[dict[str, Any]], radius: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out = []
    reject = Counter(); axes = Counter(); pivots = Counter(); by_cat = Counter()
    for ev in events:
        piv = str(ev.get('pivot_norm','')).lower()
        if piv not in OPPOSITION:
            reject['pivot_not_opposition'] += 1; reject[f'pivot_not_opposition::{ev.get("category")}'] += 1; continue
        axis, pol = OPPOSITION[piv]
        tg = str(ev.get('target_norm','')).lower()
        if tg in GENERIC_TARGETS:
            reject['generic_target'] += 1; reject[f'generic_target::{ev.get("category")}'] += 1; continue
        slot = event_slot(ev, radius)
        ev2 = dict(ev)
        ev2['opposition_axis'] = axis
        ev2['opposition_polarity'] = pol
        ev2['target_side'] = side_of(ev)
        ev2['slot_surface_set'] = slot['surface_set']
        ev2['slot_content_set'] = slot['content_set']
        ev2['slot_skeleton'] = slot['skeleton']
        ev2['slot_shell'] = slot['shell']
        ev2['slot_left_word'] = slot['left_word']
        ev2['slot_right_word'] = slot['right_word']
        out.append(ev2)
        axes[axis] += 1; pivots[piv] += 1; by_cat[str(ev.get('category'))] += 1
    return out, {'input_events': len(events), 'kept_opposition_events': len(out), 'rejects': dict(reject), 'events_by_axis': dict(axes), 'events_by_category': dict(by_cat), 'top_pivots': pivots.most_common(40)}


def pair_keys(ev: dict[str, Any]) -> list[tuple[Any, ...]]:
    return [
        ('L0', ev['opposition_axis'], ev['target_class'], ev['target_token_len'], ev['target_token_pattern'], ev['target_freq_bin'], ev['distance_bin'], ev['target_side']),
        ('L1', ev['opposition_axis'], ev['target_class'], ev['target_token_len'], ev['target_token_pattern'], ev['target_freq_bin'], ev['target_side']),
        ('L2', ev['opposition_axis'], ev['target_class'], ev['target_token_len'], ev['target_token_pattern'], ev['target_side']),
        ('L3', ev['opposition_axis'], ev['target_class'], ev['target_token_len'], ev['target_token_pattern']),
    ]


def token_shape_ok(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return int(a['target_token_len']) == int(b['target_token_len']) and str(a['target_token_pattern']) == str(b['target_token_pattern'])


def compatible(a: dict[str, Any], b: dict[str, Any], args: argparse.Namespace) -> tuple[bool, str, dict[str, Any]]:
    if a['uid'] == b['uid'] or int(a['tail_row_idx']) == int(b['tail_row_idx']):
        return False, 'same_row_or_self', {}
    if a['opposition_axis'] != b['opposition_axis']:
        return False, 'axis_mismatch', {}
    if int(a['opposition_polarity']) == int(b['opposition_polarity']):
        return False, 'same_polarity', {}
    if str(a['target_norm']) == str(b['target_norm']):
        return False, 'same_target_norm', {}
    if str(a['target_class']) != str(b['target_class']):
        return False, 'target_class_mismatch', {}
    if not token_shape_ok(a, b):
        return False, 'token_shape_mismatch', {}
    if abs(int(a['target_freq_bin_id']) - int(b['target_freq_bin_id'])) > args.max_target_freq_delta:
        return False, 'target_freq_delta', {}
    if abs(int(a['distance_bin_id']) - int(b['distance_bin_id'])) > args.max_distance_delta:
        return False, 'distance_delta', {}
    if (not args.allow_side_mismatch) and str(a['target_side']) != str(b['target_side']):
        return False, 'side_mismatch', {}
    if (not args.allow_same_source) and str(a.get('source')) == str(b.get('source')):
        return False, 'same_source', {}
    if active_funnel.contains_norm_as_word(a['context_no_target_lc'], str(b['target_norm'])):
        return False, 'b_target_leaks_in_a_context', {}
    if active_funnel.contains_norm_as_word(b['context_no_target_lc'], str(a['target_norm'])):
        return False, 'a_target_leaks_in_b_context', {}
    content_j = jaccard(a['slot_content_set'], b['slot_content_set'])
    surface_j = jaccard(a['slot_surface_set'], b['slot_surface_set'])
    shared_content = sorted(set(a['slot_content_set']) & set(b['slot_content_set']))
    shared_surface = sorted(set(a['slot_surface_set']) & set(b['slot_surface_set']))
    shell_same = str(a.get('slot_shell')) == str(b.get('slot_shell'))
    lr_same = (str(a.get('slot_left_word')) == str(b.get('slot_left_word')) and str(a.get('slot_right_word')) == str(b.get('slot_right_word')))
    if args.slot_accept_mode == 'loose':
        slot_ok = not (content_j < args.min_content_jaccard and surface_j < args.min_surface_jaccard and len(shared_content) < args.min_shared_content and not shell_same and not lr_same)
    else:
        # The stricter mode is the intended research repair: surface overlap from
        # function words can no longer admit a pair by itself. Either the target
        # shell is effectively the same, or the contexts share real local content.
        content_ok = (len(shared_content) >= args.min_shared_content and content_j >= args.min_content_jaccard)
        shell_ok = shell_same or lr_same
        slot_ok = content_ok or shell_ok
    if not slot_ok:
        return False, 'weak_slot_overlap', {'content_jaccard': content_j, 'surface_jaccard': surface_j, 'shared_content': shared_content, 'shared_surface': shared_surface, 'shell_same': shell_same, 'lr_same': lr_same, 'slot_accept_mode': args.slot_accept_mode}
    info = {'content_jaccard': content_j, 'surface_jaccard': surface_j, 'shared_content': shared_content[:12], 'shared_surface': shared_surface[:20], 'shared_content_count': len(shared_content), 'shared_surface_count': len(shared_surface), 'shell_same': shell_same, 'left_right_same': lr_same}
    return True, 'ok', info


def pair_cost(a: dict[str, Any], b: dict[str, Any], overlap: dict[str, Any]) -> float:
    fd = abs(int(a['target_freq_bin_id']) - int(b['target_freq_bin_id']))
    dd = abs(int(a['distance_bin_id']) - int(b['distance_bin_id']))
    dist = abs(int(a.get('pivot_target_distance', 99)) - int(b.get('pivot_target_distance', 99)))
    cj = float(overlap.get('content_jaccard', 0.0)); sj = float(overlap.get('surface_jaccard', 0.0))
    shared = float(overlap.get('shared_content_count', 0))
    shell_bonus = 2.0 if overlap.get('shell_same') else 0.0
    lr_bonus = 1.0 if overlap.get('left_right_same') else 0.0
    return 10*fd + 4*dd + 0.15*dist - 18*cj - 6*sj - 0.8*shared - shell_bonus - lr_bonus + 0.001*abs(int(a.get('words',0))-int(b.get('words',0)))


def build_pairs(events: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_key: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for e in events:
        for key in pair_keys(e):
            by_key[key].append(e)
    order = sorted(events, key=lambda e: (e['opposition_axis'], e['target_class'], int(e['target_token_len']), int(e['target_freq_bin_id']), stable_u(args.seed, 'order', e['uid'])))
    used = set(); target_use = Counter(); pivot_pair_use = Counter(); target_pair_use = Counter()
    pairs = []; fail = Counter(); levels = Counter(); costs = []; by_axis = Counter(); by_cat = Counter(); overlaps = {'content_jaccard': [], 'surface_jaccard': [], 'shared_content_count': [], 'shared_surface_count': []}; examples = []
    for a in order:
        if a['uid'] in used:
            continue
        if target_use[a['target_norm']] >= args.max_pairs_per_target:
            fail['target_cap_a'] += 1; continue
        best = None; best_level = None; best_cost = None; best_info = None; local_fail = Counter()
        for level, key in enumerate(pair_keys(a)):
            cands = by_key.get(key, [])
            if not cands:
                continue
            cands = sorted(cands, key=lambda b: (0 if int(b['opposition_polarity']) != int(a['opposition_polarity']) else 1, abs(int(a['target_freq_bin_id'])-int(b['target_freq_bin_id'])), stable_u(args.seed, 'cand', a['uid'], b['uid'])))[:args.candidate_scan_per_level]
            for b in cands:
                if b['uid'] in used or b['uid'] == a['uid']:
                    local_fail['used_or_self'] += 1; continue
                if target_use[b['target_norm']] >= args.max_pairs_per_target:
                    local_fail['target_cap_b'] += 1; continue
                ok, why, info = compatible(a, b, args)
                if not ok:
                    local_fail[why] += 1; continue
                c = pair_cost(a, b, info)
                if best is None or c < best_cost:
                    best = b; best_level = level; best_cost = c; best_info = info
            if best is not None:
                break
        if best is None:
            fail['no_partner'] += 1
            for k, v in local_fail.items():
                fail[f'no_partner_detail::{k}'] += v
            continue
        b = best; info = best_info or {}
        used.add(a['uid']); used.add(b['uid']); target_use[a['target_norm']] += 1; target_use[b['target_norm']] += 1
        levels[str(best_level)] += 1; costs.append(float(best_cost)); by_axis[a['opposition_axis']] += 1; by_cat[a['category']] += 1
        piv_key = tuple(sorted([str(a['pivot_norm']), str(b['pivot_norm'])])); tar_key = tuple(sorted([str(a['target_norm']), str(b['target_norm'])]))
        pivot_pair_use[piv_key] += 1; target_pair_use[tar_key] += 1
        for k in overlaps:
            overlaps[k].append(float(info.get(k, 0.0)))
        rec = {
            'pair_id': f'p{len(pairs):07d}', 'match_level': int(best_level), 'pair_cost': float(best_cost),
            'category': a['category'], 'opposition_axis': a['opposition_axis'], 'polarity_a': int(a['opposition_polarity']), 'polarity_b': int(b['opposition_polarity']),
            'target_class': a['target_class'], 'target_token_len': int(a['target_token_len']), 'target_token_pattern': a['target_token_pattern'],
            'target_freq_bin_a': a['target_freq_bin'], 'target_freq_bin_b': b['target_freq_bin'], 'target_freq_bin_delta': abs(int(a['target_freq_bin_id'])-int(b['target_freq_bin_id'])),
            'distance_bin_a': a['distance_bin'], 'distance_bin_b': b['distance_bin'], 'distance_bin_delta': abs(int(a['distance_bin_id'])-int(b['distance_bin_id'])),
            'target_side_a': a['target_side'], 'target_side_b': b['target_side'],
            'row_a': int(a['tail_row_idx']), 'row_b': int(b['tail_row_idx']), 'source_a': a.get('source'), 'source_b': b.get('source'),
            'uid_a': a['uid'], 'uid_b': b['uid'], 'target_norm_a': a['target_norm'], 'target_norm_b': b['target_norm'],
            'target_ids_a': a['target_ids'], 'target_ids_b': b['target_ids'], 'target_positions_a': a['target_positions'], 'target_positions_b': b['target_positions'],
            'pivot_norm_a': a['pivot_norm'], 'pivot_norm_b': b['pivot_norm'], 'pivot_gid_a': int(a['pivot_gid']), 'pivot_gid_b': int(b['pivot_gid']),
            'target_gid_a': int(a['target_gid']), 'target_gid_b': int(b['target_gid']), 'text_a': a['text'], 'text_b': b['text'],
            'slot_content_jaccard': float(info.get('content_jaccard', 0.0)), 'slot_surface_jaccard': float(info.get('surface_jaccard', 0.0)),
            'shared_content_count': int(info.get('shared_content_count', 0)), 'shared_surface_count': int(info.get('shared_surface_count', 0)),
            'shared_content': info.get('shared_content', []), 'shared_surface': info.get('shared_surface', []),
            'slot_shell_a': a.get('slot_shell'), 'slot_shell_b': b.get('slot_shell'), 'slot_shell_same': bool(info.get('shell_same', False)),
        }
        pairs.append(rec)
        if len(examples) < args.sample_pairs:
            examples.append(rec)
        if args.max_pairs and len(pairs) >= args.max_pairs:
            break
    overlap_summary = {k: qstats(v) for k, v in overlaps.items()}
    repeated_targets = [(list(k), v) for k, v in target_pair_use.items() if v >= 2]
    repeated_targets.sort(key=lambda x: (-x[1], x[0]))
    summary = {'input_events': len(events), 'pairs': len(pairs), 'events_used': len(used), 'event_use_fraction': len(used)/len(events) if events else None, 'pairs_by_axis': dict(by_axis), 'pairs_by_category': dict(by_cat), 'match_levels': dict(levels), 'pair_cost': qstats(costs), 'slot_overlap': overlap_summary, 'target_unique_used': len(target_use), 'target_use_top20': target_use.most_common(20), 'pivot_pair_top20': [(list(k), v) for k, v in pivot_pair_use.most_common(20)], 'repeated_target_pair_count': len(repeated_targets), 'repeated_target_pair_top20': repeated_targets[:20], 'failure_counts': dict(fail), 'sample_pairs': examples}
    return pairs, summary


def write_outputs(events: list[dict[str, Any]], load_summary: dict[str, Any], opposition_summary: dict[str, Any], pairs: list[dict[str, Any]], pair_summary: dict[str, Any], args: argparse.Namespace, runtime: float) -> None:
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    pair_path = out/'opposition_slot_pair_pool.jsonl'
    with pair_path.open('w', encoding='utf-8') as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
    csv_path = out/'opposition_slot_pair_pool_summary.csv'
    fields = ['pair_id','match_level','pair_cost','category','opposition_axis','pivot_norm_a','pivot_norm_b','target_norm_a','target_norm_b','target_class','target_token_len','target_freq_bin_a','target_freq_bin_b','distance_bin_a','distance_bin_b','target_side_a','target_side_b','slot_content_jaccard','slot_surface_jaccard','shared_content_count','row_a','row_b']
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for p in pairs:
            w.writerow({k: p.get(k) for k in fields})
    summary = {
        'status': 'OPPOSITION_SLOT_PAIR_FUNNEL', 'created_utc': now_utc(),
        'purpose': 'Legal active-token relation pairs requiring hand-coded opposite pivots and shared target-side local slots before any no-update scoring or training.',
        'inputs': {'labels_jsonl': str(args.labels_jsonl), 'tail_jsonl': str(args.tail_jsonl), 'tokenizer_path': str(args.tokenizer_path)},
        'parameters': vars(args), 'load_summary': load_summary, 'opposition_event_summary': opposition_summary,
        'pair_summary': pair_summary, 'runtime_sec': runtime,
        'outputs': {'pairs_jsonl': str(pair_path), 'pairs_csv': str(csv_path), 'summary_json': str(out/'opposition_slot_pair_funnel_summary.json'), 'note': str(args.note_path)},
    }
    (out/'opposition_slot_pair_funnel_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    lines = ['# research — opposition/shared-slot active relation pair funnel', '']
    lines.append('This pool rebuilds the research/128 relation object by requiring both a hand-coded opposite relation pivot and a more similar target-side local slot. It is built from the same active-token 256-token view used by the PVDM trainers; no model weights are loaded.')
    lines.append('')
    lines.append(f"Segment: {load_summary['segment']['selected_rows']} rows / {load_summary['segment']['selected_words']} words.")
    lines.append(f"Active events before opposition filtering: {load_summary['kept_events']:,}.")
    lines.append(f"Opposition-pivot events kept: {opposition_summary['kept_opposition_events']:,}.")
    lines.append(f"Pairs constructed: **{pair_summary['pairs']:,}** using {pair_summary['events_used']:,} events.")
    lines.append('')
    lines.append('## Pairs by opposition axis')
    lines.append('')
    lines.append('| axis | pairs |')
    lines.append('|---|---:|')
    for k, v in sorted(pair_summary['pairs_by_axis'].items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f'| {k} | {v} |')
    lines.append('')
    lines.append('## Pair quality summaries')
    lines.append('')
    lines.append(f"Match levels: `{pair_summary['match_levels']}`")
    lines.append(f"Pair cost: `{pair_summary['pair_cost']}`")
    lines.append(f"Slot overlap: `{pair_summary['slot_overlap']}`")
    lines.append(f"Repeated target-pair count (>=2 contexts): {pair_summary['repeated_target_pair_count']}")
    lines.append('')
    lines.append('Top pivot pairs:')
    for pp, c in pair_summary['pivot_pair_top20'][:12]:
        lines.append(f'- {pp}: {c}')
    lines.append('')
    lines.append('Top repeated target pairs:')
    for tp, c in pair_summary['repeated_target_pair_top20'][:12]:
        lines.append(f'- {tp}: {c}')
    lines.append('')
    lines.append('Files:')
    lines.append(f"- summary: `{out/'opposition_slot_pair_funnel_summary.json'}`")
    lines.append(f"- pairs: `{pair_path}`")
    lines.append(f"- csv: `{csv_path}`")
    Path(args.note_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.note_path).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'status': summary['status'], 'opposition_events': opposition_summary['kept_opposition_events'], 'pairs': pair_summary['pairs'], 'summary': str(out/'opposition_slot_pair_funnel_summary.json'), 'note': str(args.note_path)}, indent=2), flush=True)


def build_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels_jsonl', default=str(cont.DEFAULT_LABELS)); ap.add_argument('--tail_jsonl', default=str(cont.DEFAULT_TAIL)); ap.add_argument('--tokenizer_path', default=str(cont.DEFAULT_TOKENIZER))
    ap.add_argument('--output_dir', default=str(DEFAULT_OUT)); ap.add_argument('--note_path', default=str(DEFAULT_NOTE))
    ap.add_argument('--start_tail_row', type=int, default=cont.DEFAULT_START_TAIL_ROW); ap.add_argument('--expected_start_tail_words', type=int, default=cont.DEFAULT_START_TAIL_WORDS); ap.add_argument('--max_word_exposure', type=int, default=cont.DEFAULT_STAGE_WORDS_TO_80M)
    ap.add_argument('--max_rows', type=int, default=0); ap.add_argument('--batch_size', type=int, default=256); ap.add_argument('--seq_length', type=int, default=256); ap.add_argument('--num_workers', type=int, default=0)
    ap.add_argument('--max_target_token_len', type=int, default=4); ap.add_argument('--max_control_distance_abs_diff', type=int, default=1); ap.add_argument('--max_control_freq_bin_abs_diff', type=int, default=1); ap.add_argument('--strict_own_context_leak', action='store_true', default=True); ap.add_argument('--no_strict_own_context_leak', dest='strict_own_context_leak', action='store_false')
    ap.add_argument('--slot_radius', type=int, default=8); ap.add_argument('--min_content_jaccard', type=float, default=0.12); ap.add_argument('--min_surface_jaccard', type=float, default=0.16); ap.add_argument('--min_shared_content', type=int, default=2); ap.add_argument('--slot_accept_mode', choices=['loose','content_or_shell'], default='content_or_shell')
    ap.add_argument('--max_target_freq_delta', type=int, default=1); ap.add_argument('--max_distance_delta', type=int, default=1); ap.add_argument('--allow_side_mismatch', action='store_true'); ap.add_argument('--allow_same_source', action='store_true')
    ap.add_argument('--max_pairs_per_target', type=int, default=8); ap.add_argument('--candidate_scan_per_level', type=int, default=1200); ap.add_argument('--max_pairs', type=int, default=0); ap.add_argument('--sample_pairs', type=int, default=60); ap.add_argument('--seed', type=int, default=43029)
    return ap.parse_args()


def main() -> None:
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    t0 = time.time(); args = build_args()
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    events, load_summary = active_funnel.load_segment_and_active_events(args, tok)
    op_events, op_summary = enrich_events(events, args.slot_radius)
    pairs, pair_summary = build_pairs(op_events, args)
    write_outputs(op_events, load_summary, op_summary, pairs, pair_summary, args, round(time.time()-t0, 2))


if __name__ == '__main__':
    main()
