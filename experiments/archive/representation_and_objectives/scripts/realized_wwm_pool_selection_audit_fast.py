#!/usr/bin/env python3
"""research fast audit: actual 100M CUDA-WWM realized pool-level selection match.

This is the decision audit before launching the copied-content 100M control.  It
simulates the same CUDA torch.Generator WWM schedule as the research packed trainer,
then compares realized masked source-absent-content whole-word events to realized
masked copied-content events selected by the research pool-level whole-word copied
selection.  It deliberately does NOT do expensive rematching.
"""
from __future__ import annotations

import argparse, collections, hashlib, json, math, pathlib, statistics, sys, time
from typing import Any
import torch

USER_ROOT = pathlib.Path('.').resolve()
SCRIPT_DIR = USER_ROOT / 'experiments/archive/representation_and_objectives/scripts'
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import packed_target_selective_trainer as train228  # noqa: E402
from annotate_packed_pool import word_class  # noqa: E402

DEFAULT_STREAM = 'experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl'
DEFAULT_ANNOTATION = 'experiments/archive/representation_and_objectives/data/packed_pool_annotation/packed_pool_annotations.jsonl'
DEFAULT_SELECTION = 'experiments/archive/representation_and_objectives/data/packed_pool_annotation/packed_wholeword_copied_selection.jsonl'
DEFAULT_TOKENIZER = 'experiments/archive/representation_and_objectives/training/runs/gc_compact_repeat_reinvest_16k_seed43022_r2/hf_model/chck_1M'
DEFAULT_OUT = 'experiments/archive/representation_and_objectives/data/realized_wwm_pool_selection_audit_fast'

CAT_RW_COPIED = train228.CAT_TO_IDX[train228.CAT_RW_COPIED]
CAT_RW_ABS_CONTENT = train228.CAT_TO_IDX[train228.CAT_RW_ABS_CONTENT]


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def q_stats(xs: list[float]) -> dict[str, Any]:
    if not xs:
        return {'n': 0}
    s = sorted(float(x) for x in xs)
    n = len(s)
    def q(p: float) -> float:
        return s[min(n - 1, max(0, int(round(p * (n - 1)))))]
    return {
        'n': n, 'mean': round(statistics.mean(s), 6), 'median': round(statistics.median(s), 6),
        'p05': round(q(0.05), 6), 'p10': round(q(0.10), 6), 'p25': round(q(0.25), 6),
        'p75': round(q(0.75), 6), 'p90': round(q(0.90), 6), 'p95': round(q(0.95), 6),
        'min': round(s[0], 6), 'max': round(s[-1], 6),
    }


def summarize(events: list[dict[str, Any]]) -> dict[str, Any]:
    out = {
        'n_events': len(events),
        'piece_total': int(sum(e['bpe_len'] for e in events)),
        'unique_example_groups': len({(e['example_id'], e['group_id']) for e in events}),
        'epoch_counts': dict(sorted(collections.Counter(str(e['epoch']) for e in events).items(), key=lambda kv: int(kv[0]))),
        'bpe_len_counts': dict(sorted(collections.Counter(str(e['bpe_len']) for e in events).items(), key=lambda kv: int(kv[0]))),
        'word_class_counts': dict(collections.Counter(e.get('word_class','') for e in events)),
    }
    for k in ['bpe_len','support_log_mean','support_log_min','support_mean','support_min','seq_token_mid','rel_group_pos','row_order_frac']:
        out[k] = q_stats([float(e[k]) for e in events])
    return out


def deltas(abs_s: dict[str, Any], sel_s: dict[str, Any]) -> dict[str, Any]:
    d = {'event_delta': sel_s['n_events'] - abs_s['n_events'], 'piece_delta': sel_s['piece_total'] - abs_s['piece_total']}
    for k in ['bpe_len','support_log_mean','support_log_min','support_mean','support_min','seq_token_mid','rel_group_pos','row_order_frac']:
        d[f'{k}_mean_delta'] = round(float(sel_s[k]['mean']) - float(abs_s[k]['mean']), 6)
    return d


def load_annotations(path: pathlib.Path):
    annotations, row_to_eid = {}, {}
    with path.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            a = json.loads(line)
            eid = int(a['example_id'])
            annotations[eid] = [train228.CAT_TO_IDX.get(c,0) for c in a['word_categories']]
            row_to_eid[int(a['row_index'])] = eid
    return annotations, row_to_eid


def load_selection(path: pathlib.Path, row_to_eid: dict[int,int]) -> set[tuple[int,int]]:
    out = set()
    with path.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            g = json.loads(line)
            eid = row_to_eid.get(int(g['row_index']))
            if eid is not None:
                out.add((eid, int(g['word_index'])))
    return out


def load_stream(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding='utf-8') as f:
        for i,line in enumerate(f):
            if line.strip():
                r=json.loads(line)
                text=str(r['text'])
                rows.append({'stream_index': i, 'example_id': int(r.get('example_id', i)), 'text': text, 'words': int(r.get('words', len(text.split())))})
    return rows


class Cache:
    def __init__(self, tok, annotations, seq_len):
        self.tok=tok; self.annotations=annotations; self.seq_len=seq_len
        self.special=set(int(x) for x in tok.all_special_ids)
        self.ws={}; self.cache={}
    def is_start(self, tid):
        if int(tid) not in self.ws:
            s=self.tok.convert_ids_to_tokens(int(tid))
            self.ws[int(tid)] = bool(s is not None and train228.is_word_start(str(s)))
        return self.ws[int(tid)]
    def enc(self, e):
        eid=int(e['example_id'])
        if eid in self.cache:
            return self.cache[eid]
        enc=self.tok(e['text'], add_special_tokens=False, truncation=True, max_length=self.seq_len, padding='max_length', return_tensors='pt')
        ids=enc['input_ids'].squeeze(0).long(); att=enc['attention_mask'].squeeze(0).long()
        group=torch.full_like(ids, -1); gid=-1
        for i in range(ids.shape[0]):
            if int(att[i])==0: continue
            tid=int(ids[i])
            if tid in self.special: continue
            if gid<0 or self.is_start(tid) or i==0: gid+=1
            group[i]=gid
        cats=torch.zeros_like(ids)
        wc=self.annotations.get(eid)
        if wc is not None:
            for i in range(ids.shape[0]):
                g=int(group[i])
                if g>=0 and g < len(wc): cats[i]=wc[g]
        out={'input_ids':ids,'attention_mask':att,'word_group':group,'token_cat':cats,'words':e['text'].split(),'max_gid': int(group.max().item()) if (group>=0).any() else -1,'has_interest': bool(((cats==CAT_RW_ABS_CONTENT)|(cats==CAT_RW_COPIED)).any().item())}
        self.cache[eid]=out
        return out


def token_freq(stream, cache):
    occ=collections.Counter(int(e['example_id']) for e in stream)
    first={}
    for e in stream:
        first.setdefault(int(e['example_id']), e)
    freq=collections.Counter()
    for eid,e in first.items():
        en=cache.enc(e); mult=occ[eid]
        for tid in en['input_ids'][en['attention_mask'].bool()].tolist():
            it=int(tid)
            if it not in cache.special: freq[it]+=mult
    return dict(freq)


def make_event(origin, entry, epoch, gid, pos, labs, freq, en, n):
    supports=[int(freq.get(int(t),0)) for t in labs]
    logs=[math.log1p(x) for x in supports]
    words=en['words']; text=words[gid] if 0<=gid<len(words) else ''
    mid=0.5*(min(pos)+max(pos))
    return {
        'origin': origin, 'stream_index': int(entry['stream_index']), 'example_id': int(entry['example_id']),
        'epoch': int(epoch), 'group_id': int(gid), 'word_text': text, 'word_class': word_class(text) if text else '',
        'positions': [int(p) for p in pos], 'token_ids': [int(t) for t in labs], 'bpe_len': len(labs),
        'seq_token_mid': mid/255.0, 'rel_group_pos': gid/max(1,en['max_gid']), 'row_order_frac': entry['stream_index']/max(1,n-1),
        'support_min': min(supports) if supports else 0, 'support_mean': sum(supports)/max(1,len(supports)),
        'support_log_min': min(logs) if logs else 0.0, 'support_log_mean': sum(logs)/max(1,len(logs)),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--stream', default=DEFAULT_STREAM)
    ap.add_argument('--annotation', default=DEFAULT_ANNOTATION)
    ap.add_argument('--selection', default=DEFAULT_SELECTION)
    ap.add_argument('--tokenizer_path', default=DEFAULT_TOKENIZER)
    ap.add_argument('--output_dir', default=DEFAULT_OUT)
    ap.add_argument('--device', default='cuda:1')
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--max_seq_length', type=int, default=256)
    ap.add_argument('--mask_prob', type=float, default=0.15)
    ap.add_argument('--train_rng_seed', type=int, default=43023)
    ap.add_argument('--progress_every', type=int, default=100000)
    args=ap.parse_args()
    t0=time.time(); out=pathlib.Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    ann,row_to_eid=load_annotations(pathlib.Path(args.annotation)); selected=load_selection(pathlib.Path(args.selection), row_to_eid)
    stream=load_stream(pathlib.Path(args.stream))
    tok=train228.make_portable_tokenizer(args.tokenizer_path)
    cache=Cache(tok, ann, args.max_seq_length)
    freq=token_freq(stream, cache)
    device=torch.device(args.device)
    gen=torch.Generator(device=device); gen.manual_seed(args.train_rng_seed)
    abs_events=[]; sel_events=[]; copied_all=0; selected_noncontent=0; occ=collections.Counter(); n=len(stream)
    for start in range(0,n,args.batch_size):
        entries=stream[start:start+args.batch_size]
        encs=[cache.enc(e) for e in entries]
        ids_cpu=torch.stack([x['input_ids'] for x in encs]); att_cpu=torch.stack([x['attention_mask'] for x in encs]); group_cpu=torch.stack([x['word_group'] for x in encs]); cat_cpu=torch.stack([x['token_cat'] for x in encs])
        ids=ids_cpu.to(device, non_blocking=True); att=att_cpu.to(device, non_blocking=True); group=group_cpu.to(device, non_blocking=True)
        _, labels=train228.apply_wwm_masking(ids, att, group, tok, args.mask_prob, gen)
        labels=labels.cpu(); mask=labels!=-100
        for b,e in enumerate(entries):
            eid=int(e['example_id']); epoch=occ[eid]; occ[eid]+=1
            if not encs[b]['has_interest']: continue
            groups=group_cpu[b]; cats=cat_cpu[b]; labs=labels[b]; m=mask[b]
            gids=torch.unique(groups[m & ((cats==CAT_RW_ABS_CONTENT)|(cats==CAT_RW_COPIED))]).tolist()
            for gid0 in gids:
                gid=int(gid0)
                if gid < 0: continue
                pos=((groups==gid)&m).nonzero(as_tuple=False).view(-1).tolist()
                if not pos: continue
                cat_vals=[int(cats[p]) for p in pos]
                cat=collections.Counter(cat_vals).most_common(1)[0][0]
                labids=[int(labs[p]) for p in pos]
                if cat==CAT_RW_ABS_CONTENT:
                    abs_events.append(make_event('absent_content', e, epoch, gid, pos, labids, freq, encs[b], n))
                elif cat==CAT_RW_COPIED:
                    ev=make_event('copied_content', e, epoch, gid, pos, labids, freq, encs[b], n)
                    if ev['word_class'] in {'content','number'}:
                        copied_all += 1
                        if (eid,gid) in selected:
                            sel_events.append(ev)
                    elif (eid,gid) in selected:
                        selected_noncontent += 1
        if args.progress_every and (start+len(entries)) % args.progress_every < args.batch_size:
            print(json.dumps({'event':'progress','rows':start+len(entries),'abs_events':len(abs_events),'pool_selected_events':len(sel_events),'copied_content_events_seen':copied_all,'elapsed_sec':round(time.time()-t0,1)}), flush=True)
    abs_s=summarize(abs_events); sel_s=summarize(sel_events); dd=deltas(abs_s, sel_s)
    audit={
        'status':'REALIZED_WWM_POOL_SELECTION_AUDIT_FAST',
        'meaning':'Actual CUDA-WWM 100M schedule comparison of source-absent-content deletion events versus research pool-level copied-content whole-word selection. Use this to decide whether the pool-level copied arm is matched enough or requires event-level rematch.',
        'inputs':{'stream':args.stream,'stream_sha256':sha256_file(pathlib.Path(args.stream)),'annotation':args.annotation,'annotation_sha256':sha256_file(pathlib.Path(args.annotation)),'selection':args.selection,'selection_sha256':sha256_file(pathlib.Path(args.selection)),'tokenizer':args.tokenizer_path,'device':args.device,'train_rng_seed':args.train_rng_seed,'mask_prob':args.mask_prob},
        'stream_rows':len(stream), 'selection_positions':len(selected), 'occurrence_count_distribution':dict(collections.Counter(str(v) for v in occ.values())),
        'source_absent_content_realized':abs_s, 'pool_selected_copied_content_realized':sel_s, 'pool_selected_minus_abs':dd,
        'copied_content_events_seen':copied_all, 'selected_noncontent_copied_events':selected_noncontent,
        'interpretation': {'launch_pool_level_control_if':'event_delta and piece_delta are small relative to abs events, and support/position mean deltas are scientifically minor; otherwise use event-level rematch trainer prepared in research.', 'aggregate_endpoint_warning':'Even a matched control only tests mechanism if the endpoint readout reconstructs Supplement and relational-EWoK transitions, not aggregate score alone.'},
        'elapsed_sec':round(time.time()-t0,1),
    }
    out_json=out/'realized_wwm_pool_selection_audit_fast.json'
    out_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps({'status':audit['status'],'out':str(out_json),'abs_events':abs_s['n_events'],'abs_pieces':abs_s['piece_total'],'pool_selected_events':sel_s['n_events'],'pool_selected_pieces':sel_s['piece_total'],'pool_selected_minus_abs':dd,'elapsed_sec':audit['elapsed_sec']}, indent=2, ensure_ascii=False), flush=True)

if __name__=='__main__':
    main()
