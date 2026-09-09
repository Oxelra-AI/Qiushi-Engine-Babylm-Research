#!/usr/bin/env python3
"""research active-token pair-funnel for a two-context/two-target relation object.

This repairs the first structural funnel by requiring every event to survive the
actual 256-token training view. It uses MaskedChunkDataset word groups and the
same research `collect_active_events` mapping used by PVDM trainers, so target
positions and token ids are real forwarded-token positions. No model is loaded;
no training or official-example scoring is performed.
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
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

USER_ROOT = Path('.').resolve()
for p in [USER_ROOT/'experiments/archive/compact_experience/scripts', USER_ROOT/'experiments/archive/representation_and_objectives/scripts']:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import masking_curriculum_trainer as base  # noqa: E402
import pvdm_continuation_trainer as cont  # noqa: E402
import pvdm_masking_lib as pvdm  # noqa: E402
import sparse_relation_aux_trainer as relaux  # noqa: E402

DEFAULT_OUT = Path('experiments/archive/representation_and_objectives/data/relation_active_pair_funnel')
DEFAULT_NOTE = Path('research/notes/representation_and_objectives/relation_active_pair_funnel.md')
CATEGORIES = {'physical_change','causal_connector','temporal','spatial','comparative','negation'}
PUNCT_STRIP = "\"'“”‘’.,!?;:()[]{}<>«»‹›*_-=+/\\|`~"
WORD_RE = re.compile(r"\S+")


def now_utc() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def stable_u(seed: int, *items: Any) -> float:
    h=hashlib.blake2b(digest_size=8)
    h.update(str(seed).encode())
    for it in items:
        h.update(b'\0'); h.update(str(it).encode('utf-8', errors='ignore'))
    return int.from_bytes(h.digest(), 'big')/float(2**64)


def qstats(vals: list[float]) -> dict[str, Any]:
    xs=sorted(float(v) for v in vals if math.isfinite(float(v)))
    if not xs:
        return {'n':0}
    def q(p: float) -> float:
        if len(xs)==1: return xs[0]
        idx=p*(len(xs)-1); lo=int(math.floor(idx)); hi=int(math.ceil(idx))
        if lo==hi: return xs[lo]
        return xs[lo]*(hi-idx)+xs[hi]*(idx-lo)
    return {'n':len(xs),'mean':sum(xs)/len(xs),'median':q(0.5),'std':float(np.std(xs)),'p05':q(0.05),'p95':q(0.95),'min':xs[0],'max':xs[-1]}


def word_spans(text: str) -> list[tuple[str,str,int,int]]:
    return [(m.group(0), m.group(0).lower().strip(PUNCT_STRIP), m.start(), m.end()) for m in WORD_RE.finditer(text)]


def context_without_target(text: str, target_gid: int) -> str:
    ws=word_spans(text)
    if 0 <= target_gid < len(ws):
        s,e=ws[target_gid][2], ws[target_gid][3]
        return (text[:s]+' [MASKTARGET] '+text[e:]).lower()
    return text.lower()


def contains_norm_as_word(context_lc: str, norm: str) -> bool:
    norm=str(norm).lower().strip(PUNCT_STRIP)
    if not norm:
        return False
    return any(tok == norm for _raw,tok,_s,_e in word_spans(context_lc))


def token_pattern(tokenizer, ids: list[int]) -> str:
    toks=tokenizer.convert_ids_to_tokens([int(x) for x in ids])
    pat=[]
    for j,t in enumerate(toks):
        s=str(t)
        starts=(j==0) or s.startswith('Ġ') or s.startswith('▁') or s.startswith(' ')
        pat.append('S' if starts else 'C')
    return ''.join(pat)


def ev_to_rec(lr: dict[str, Any], active: pvdm.ActiveEvent, input_ids_row: torch.Tensor, group_pos: dict[int,list[int]], tokenizer, *, max_target_token_len: int, strict_own_context_leak: bool) -> tuple[dict[str,Any] | None, str | None]:
    raw = lr.get('events', [])[int(active.event_rank)] if int(active.event_rank) < len(lr.get('events', [])) else {}
    meta = relaux._event_meta(raw)
    cat=str(active.category)
    if cat not in CATEGORIES:
        return None, 'category_not_allowed'
    tpos=[int(x) for x in group_pos.get(int(active.target_gid), [])]
    ppos=[int(x) for x in group_pos.get(int(active.pivot_gid), [])]
    cpos=[int(x) for x in group_pos.get(int(active.control_gid), [])]
    if not tpos or not ppos or not cpos:
        return None, 'missing_token_group'
    if len(tpos) > max_target_token_len:
        return None, 'target_token_len_gt_max'
    tids=[int(input_ids_row[p].item()) for p in tpos]
    target_norm=str(meta.get('target_norm','')).lower().strip(PUNCT_STRIP)
    if not target_norm:
        return None, 'empty_target_norm'
    ctx=context_without_target(str(lr.get('text','')), int(active.target_gid))
    if strict_own_context_leak and contains_norm_as_word(ctx, target_norm):
        return None, 'target_leaks_in_own_context'
    cm=raw.get('control_match', {}) if isinstance(raw.get('control_match', {}), dict) else {}
    rec={
        'uid': f"{lr.get('tail_row_idx')}:{active.event_rank}:{active.category}:{active.pivot_gid}:{active.target_gid}",
        'tail_row_idx': int(lr.get('tail_row_idx', -1)),
        'orig_row_idx': int(lr.get('orig_row_idx', -1)),
        'example_id': lr.get('example_id'),
        'source': lr.get('source'),
        'words': int(lr.get('words',0)),
        'text': str(lr.get('text','')),
        'category': cat,
        'event_rank': int(active.event_rank),
        'pivot_gid': int(active.pivot_gid),
        'target_gid': int(active.target_gid),
        'control_gid': int(active.control_gid),
        'pivot_norm': str(meta.get('pivot_norm','')).lower().strip(PUNCT_STRIP),
        'target_norm': target_norm,
        'control_norm': str(meta.get('control_norm','')).lower().strip(PUNCT_STRIP),
        'target_class': str(meta.get('target_class')),
        'target_freq_bin': str(meta.get('target_freq_bin')),
        'target_freq_bin_id': int(meta.get('target_freq_bin_id',-1)),
        'pivot_freq_bin': str(meta.get('pivot_freq_bin')),
        'pivot_freq_bin_id': int(meta.get('pivot_freq_bin_id',-1)),
        'pivot_anchor_class': str(meta.get('pivot_anchor_class')),
        'distance_bin': str(meta.get('distance_bin')),
        'distance_bin_id': int(meta.get('distance_bin_id',-1)),
        'pivot_target_distance': int(meta.get('pivot_target_distance',-1)),
        'control_distance_abs_diff': int(meta.get('control_distance_abs_diff', cm.get('distance_abs_diff',999))),
        'control_freq_bin_abs_diff': int(meta.get('control_freq_bin_abs_diff', cm.get('freq_bin_abs_diff',999))),
        'target_positions': tpos,
        'pivot_positions': ppos,
        'control_positions': cpos,
        'target_ids': tids,
        'target_token_len': len(tids),
        'target_token_pattern': token_pattern(tokenizer, tids),
        'context_no_target_lc': ctx,
        'raw_target': raw.get('target'),
        'raw_pivot': raw.get('pivot'),
        'raw_control': raw.get('control'),
    }
    return rec, None


def load_segment_and_active_events(args, tokenizer) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    examples, labels, segment = cont.load_segment(Path(args.tail_jsonl), Path(args.labels_jsonl), start_tail_row=args.start_tail_row, expected_start_tail_words=args.expected_start_tail_words, max_word_exposure=args.max_word_exposure, max_rows=args.max_rows)
    # Add text/source fields to labels for downstream leak checks and samples.
    for ex, lr in zip(examples, labels):
        lr['text']=ex.text; lr['words']=ex.words; lr['source']=getattr(ex,'source',''); lr['example_id']=getattr(ex,'example_id', lr.get('example_id'))
    dataset=base.MaskedChunkDataset(examples, tokenizer, args.seq_length)
    loader=DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=base.collate, num_workers=args.num_workers)
    events=[]; counts=Counter(); rejects=Counter(); kept_by_cat=Counter(); usable_by_cat=Counter(); raw_by_cat=Counter(); rows_with=0
    t0=time.time()
    for step,batch in enumerate(loader,1):
        lo=(research)*args.batch_size; hi=lo+int(batch['input_ids'].shape[0])
        input_ids=batch['input_ids'][:,:args.seq_length].contiguous()
        attn=batch['attention_mask'][:,:args.seq_length].contiguous()
        wg=batch['word_group'][:,:args.seq_length].contiguous()
        batch_labels=labels[lo:hi]
        for b,lr in enumerate(batch_labels):
            counts['rows_seen'] += 1
            for raw in lr.get('events',[]) or []:
                raw_by_cat[str(raw.get('category'))] += 1
            group_pos=pvdm.group_positions_for_row(wg[b], attn[b])
            usable, rej = pvdm.collect_active_events(lr, group_pos, same_length_required=False)
            rejects.update(rej)
            for ae in usable:
                usable_by_cat[str(ae.category)] += 1
                rec, why = ev_to_rec(lr, ae, input_ids[b], group_pos, tokenizer, max_target_token_len=args.max_target_token_len, strict_own_context_leak=args.strict_own_context_leak)
                if rec is None:
                    rejects[str(why)] += 1; rejects[f'{why}::{ae.category}'] += 1; continue
                if rec['control_distance_abs_diff'] > args.max_control_distance_abs_diff:
                    rejects['control_distance_mismatch'] += 1; rejects[f'control_distance_mismatch::{rec["category"]}'] += 1; continue
                if rec['control_freq_bin_abs_diff'] > args.max_control_freq_bin_abs_diff:
                    rejects['control_freq_mismatch'] += 1; rejects[f'control_freq_mismatch::{rec["category"]}'] += 1; continue
                events.append(rec); kept_by_cat[rec['category']] += 1
            if any(e['tail_row_idx'] == int(lr.get('tail_row_idx',-1)) for e in events[-len(usable):]):
                rows_with += 1
        if step == 1 or step % 50 == 0:
            print(json.dumps({'event':'active_load_progress','step':step,'rows_seen':counts['rows_seen'],'events_kept':len(events),'elapsed_sec':round(time.time()-t0,1)}), flush=True)
    summary={'segment':segment,'rows_with_kept_events':rows_with,'counts':dict(counts),'raw_events_by_category':dict(raw_by_cat),'usable_events_by_category':dict(usable_by_cat),'kept_events':len(events),'kept_by_category':dict(kept_by_cat),'rejects':dict(rejects)}
    return events, summary


def relaxed_keys(e: dict[str,Any]) -> list[tuple[Any,...]]:
    return [
        ('L0',e['category'],e['target_class'],e['target_token_len'],e['target_token_pattern'],e['target_freq_bin'],e['distance_bin']),
        ('L1',e['category'],e['target_class'],e['target_token_len'],e['target_token_pattern'],e['target_freq_bin']),
        ('L2',e['category'],e['target_class'],e['target_token_len'],e['target_token_pattern']),
        ('L3',e['category'],e['target_token_len'],e['target_token_pattern']),
    ]


def strict_key(e: dict[str,Any]) -> tuple[Any,...]:
    return (e['category'],e['target_class'],e['target_token_len'],e['target_token_pattern'],e['target_freq_bin'],e['distance_bin'])


def pair_cost(a: dict[str,Any], b: dict[str,Any]) -> float:
    return 100*(a['target_class']!=b['target_class']) + 50*(a['target_token_len']!=b['target_token_len']) + 25*(a['target_token_pattern']!=b['target_token_pattern']) + 5*abs(a['target_freq_bin_id']-b['target_freq_bin_id']) + 2*abs(a['distance_bin_id']-b['distance_bin_id']) + 0.1*abs(a['pivot_target_distance']-b['pivot_target_distance']) + 0.001*abs(a['words']-b['words'])


def compatible(a: dict[str,Any], b: dict[str,Any], args) -> tuple[bool,str]:
    if a['uid'] == b['uid'] or a['tail_row_idx'] == b['tail_row_idx']:
        return False,'same_row_or_self'
    if a['category'] != b['category']:
        return False,'category_mismatch'
    if a['target_class'] != b['target_class']:
        return False,'target_class_mismatch'
    if a['target_token_len'] != b['target_token_len'] or a['target_token_pattern'] != b['target_token_pattern']:
        return False,'token_signature_mismatch'
    if a['target_norm'] == b['target_norm']:
        return False,'same_target_norm'
    if abs(a['target_freq_bin_id']-b['target_freq_bin_id']) > args.max_pair_target_freq_delta:
        return False,'target_freq_delta'
    if abs(a['distance_bin_id']-b['distance_bin_id']) > args.max_pair_distance_delta:
        return False,'distance_delta'
    if (not args.allow_same_source) and str(a.get('source')) == str(b.get('source')):
        return False,'same_source'
    if args.cross_context_leak_filter:
        if contains_norm_as_word(a['context_no_target_lc'], b['target_norm']):
            return False,'b_target_leaks_in_a_context'
        if contains_norm_as_word(b['context_no_target_lc'], a['target_norm']):
            return False,'a_target_leaks_in_b_context'
    return True,'ok'


def build_pairs(events: list[dict[str,Any]], args) -> tuple[list[dict[str,Any]], dict[str,Any]]:
    by_key=defaultdict(list)
    for e in events:
        for key in relaxed_keys(e): by_key[key].append(e)
    order=sorted(events, key=lambda e:(e['category'], e['target_class'], e['target_token_len'], e['target_freq_bin_id'], stable_u(args.seed,'order',e['uid'])))
    used=set(); target_use=Counter(); pairs=[]; fail=Counter(); levels=Counter(); costs=[]; by_cat=Counter(); source_pairs=Counter(); examples=[]
    for a in order:
        if a['uid'] in used: continue
        if target_use[a['target_norm']] >= args.max_pairs_per_target:
            fail['target_cap_a'] += 1; continue
        best=None; best_level=None; best_cost=None; local_fail=Counter()
        for level,key in enumerate(relaxed_keys(a)):
            pool=by_key.get(key,[])
            if not pool: continue
            cand=sorted(pool, key=lambda b:(pair_cost(a,b), stable_u(args.seed,'cand',a['uid'],b['uid'])))[:args.candidate_scan_per_level]
            for b in cand:
                if b['uid'] in used or b['uid']==a['uid']:
                    local_fail['used_or_self'] += 1; continue
                if target_use[b['target_norm']] >= args.max_pairs_per_target:
                    local_fail['target_cap_b'] += 1; continue
                ok,why=compatible(a,b,args)
                if not ok:
                    local_fail[why] += 1; continue
                best=b; best_level=level; best_cost=pair_cost(a,b); break
            if best is not None: break
        if best is None:
            fail['no_partner'] += 1
            for k,v in local_fail.items(): fail[f'no_partner_detail::{k}'] += v
            continue
        b=best
        used.add(a['uid']); used.add(b['uid']); target_use[a['target_norm']] += 1; target_use[b['target_norm']] += 1
        levels[str(best_level)] += 1; costs.append(float(best_cost)); by_cat[a['category']] += 1; source_pairs[f"{a.get('source')}||{b.get('source')}"] += 1
        rec={
            'pair_id': f"p{len(pairs):07d}", 'match_level': int(best_level), 'pair_cost': float(best_cost),
            'category': a['category'], 'target_class': a['target_class'], 'target_token_len': a['target_token_len'], 'target_token_pattern': a['target_token_pattern'],
            'target_freq_bin_a': a['target_freq_bin'], 'target_freq_bin_b': b['target_freq_bin'], 'target_freq_bin_delta': abs(a['target_freq_bin_id']-b['target_freq_bin_id']),
            'distance_bin_a': a['distance_bin'], 'distance_bin_b': b['distance_bin'], 'distance_bin_delta': abs(a['distance_bin_id']-b['distance_bin_id']),
            'row_a': a['tail_row_idx'], 'row_b': b['tail_row_idx'], 'source_a': a.get('source'), 'source_b': b.get('source'),
            'uid_a': a['uid'], 'uid_b': b['uid'], 'target_norm_a': a['target_norm'], 'target_norm_b': b['target_norm'],
            'target_ids_a': a['target_ids'], 'target_ids_b': b['target_ids'], 'target_positions_a': a['target_positions'], 'target_positions_b': b['target_positions'],
            'pivot_norm_a': a['pivot_norm'], 'pivot_norm_b': b['pivot_norm'], 'pivot_gid_a': a['pivot_gid'], 'pivot_gid_b': b['pivot_gid'],
            'target_gid_a': a['target_gid'], 'target_gid_b': b['target_gid'], 'text_a': a['text'], 'text_b': b['text'],
        }
        pairs.append(rec)
        if len(examples) < args.sample_pairs: examples.append(rec)
        if args.max_pairs and len(pairs) >= args.max_pairs: break
    return pairs, {'input_events':len(events),'pairs':len(pairs),'events_used':len(used),'event_use_fraction':len(used)/len(events) if events else None,'target_unique_used':len(target_use),'target_use_top20':target_use.most_common(20),'pairs_by_category':dict(by_cat),'match_levels':dict(levels),'pair_cost':qstats(costs),'failure_counts':dict(fail),'source_pair_top20':source_pairs.most_common(20),'sample_pairs':examples}


def write_outputs(events, load_summary, pairs, pair_summary, args, runtime):
    out=Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    pairs_path=out/'active_relation_pair_pool.jsonl'
    with pairs_path.open('w',encoding='utf-8') as f:
        for p in pairs: f.write(json.dumps(p, ensure_ascii=False)+'\n')
    csv_path=out/'active_relation_pair_pool_summary.csv'
    keys=['pair_id','match_level','pair_cost','category','target_class','target_token_len','target_token_pattern','target_freq_bin_a','target_freq_bin_b','target_freq_bin_delta','distance_bin_a','distance_bin_b','distance_bin_delta','row_a','row_b','source_a','source_b','target_norm_a','target_norm_b','pivot_norm_a','pivot_norm_b']
    with csv_path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow({k:p.get(k) for k in keys}) for p in pairs]
    strata=Counter(str(strict_key(e)) for e in events)
    p=float(args.wwm_group_mask_prob); nat={'wwm_group_mask_prob_assumed':p,'estimated_pairs_with_both_targets_masked_per_epoch':p*p*len(pairs),'estimated_pairs_with_both_targets_masked_10_epochs':p*p*len(pairs)*10,'interpretation':'Independence estimate; exact mask replay is needed before zero-extra-forward training, and forced target masking needs same-mask WWM reference.'}
    summary={'status':'ACTIVE_RELATION_PAIR_FUNNEL','created_utc':now_utc(),'purpose':'Active-token structural pair yield for legal two-context/two-target interaction objective; every event survived actual 256-token training-view mapping.','inputs':{'labels_jsonl':str(args.labels_jsonl),'tail_jsonl':str(args.tail_jsonl),'tokenizer_path':str(args.tokenizer_path)},'parameters':vars(args),'load_summary':load_summary,'strict_strata_count':len(strata),'strict_strata_top20':strata.most_common(20),'pair_summary':pair_summary,'natural_mask_estimate':nat,'runtime_sec':runtime,'outputs':{'pairs_jsonl':str(pairs_path),'pairs_csv':str(csv_path),'note':str(args.note_path)}}
    summary_path=out/'active_relation_pair_funnel_summary.json'; summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n',encoding='utf-8')
    lines=['# research — active-token legal relation pair funnel','', 'This repairs the earlier raw structural pilot by requiring each event to survive the actual 256-token training view and by saving target token positions/ids from that view.', '', f"Segment rows: {load_summary['segment']['selected_rows']} rows / {load_summary['segment']['selected_words']} words starting tail row {load_summary['segment']['start_tail_row']}.", f"Events kept after active-token/control/token/leak filters: **{load_summary['kept_events']}**.", f"Pairs constructed: **{pair_summary['pairs']}** using {pair_summary['events_used']} events; event-use fraction {pair_summary['event_use_fraction']:.4f}.", '', '## Pairs by relation family', '', '| family | pairs |', '|---|---:|']
    for k,v in sorted(pair_summary['pairs_by_category'].items(), key=lambda kv:(-kv[1],kv[0])): lines.append(f'| {k} | {v} |')
    lines += ['', '## Matching', '', f"Match levels: `{pair_summary['match_levels']}`", f"Pair cost: `{pair_summary['pair_cost']}`", '', 'Top target multiplicities:']
    for t,c in pair_summary['target_use_top20'][:10]: lines.append(f'- {t}: {c}')
    lines += ['', '## Natural-mask exposure estimate', '', f"Ordinary WWM p≈{p}: expected both-target-masked pairs per epoch ≈ {nat['estimated_pairs_with_both_targets_masked_per_epoch']:.1f}, over 10 epochs ≈ {nat['estimated_pairs_with_both_targets_masked_10_epochs']:.1f}. Exact replay remains required.", '', 'Files:', f'- summary: `{summary_path}`', f'- pairs: `{pairs_path}`', f'- csv: `{csv_path}`']
    Path(args.note_path).parent.mkdir(parents=True, exist_ok=True); Path(args.note_path).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':summary['status'],'events_kept':load_summary['kept_events'],'pairs':pair_summary['pairs'],'summary':str(summary_path),'note':str(args.note_path)}, indent=2), flush=True)


def build_args():
    ap=argparse.ArgumentParser()
    ap.add_argument('--labels_jsonl', default=str(cont.DEFAULT_LABELS)); ap.add_argument('--tail_jsonl', default=str(cont.DEFAULT_TAIL)); ap.add_argument('--tokenizer_path', default=str(cont.DEFAULT_TOKENIZER))
    ap.add_argument('--output_dir', default=str(DEFAULT_OUT)); ap.add_argument('--note_path', default=str(DEFAULT_NOTE))
    ap.add_argument('--start_tail_row', type=int, default=cont.DEFAULT_START_TAIL_ROW); ap.add_argument('--expected_start_tail_words', type=int, default=cont.DEFAULT_START_TAIL_WORDS); ap.add_argument('--max_word_exposure', type=int, default=cont.DEFAULT_STAGE_WORDS_TO_80M)
    ap.add_argument('--max_rows', type=int, default=0); ap.add_argument('--batch_size', type=int, default=256); ap.add_argument('--seq_length', type=int, default=256); ap.add_argument('--num_workers', type=int, default=0)
    ap.add_argument('--max_pairs', type=int, default=0); ap.add_argument('--max_target_token_len', type=int, default=4); ap.add_argument('--max_control_distance_abs_diff', type=int, default=1); ap.add_argument('--max_control_freq_bin_abs_diff', type=int, default=1); ap.add_argument('--max_pair_target_freq_delta', type=int, default=0); ap.add_argument('--max_pair_distance_delta', type=int, default=1); ap.add_argument('--max_pairs_per_target', type=int, default=32); ap.add_argument('--candidate_scan_per_level', type=int, default=512); ap.add_argument('--sample_pairs', type=int, default=40)
    ap.add_argument('--strict_own_context_leak', action='store_true', default=True); ap.add_argument('--no_strict_own_context_leak', dest='strict_own_context_leak', action='store_false'); ap.add_argument('--cross_context_leak_filter', action='store_true', default=True); ap.add_argument('--no_cross_context_leak_filter', dest='cross_context_leak_filter', action='store_false'); ap.add_argument('--allow_same_source', action='store_true'); ap.add_argument('--wwm_group_mask_prob', type=float, default=0.15); ap.add_argument('--seed', type=int, default=43023)
    return ap.parse_args()


def main():
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    t0=time.time(); args=build_args(); tok=AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    events, load_summary = load_segment_and_active_events(args, tok)
    pairs, pair_summary = build_pairs(events, args)
    write_outputs(events, load_summary, pairs, pair_summary, args, round(time.time()-t0,2))


if __name__=='__main__':
    main()
