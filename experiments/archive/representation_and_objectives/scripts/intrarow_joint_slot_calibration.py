#!/usr/bin/env python3
"""research: joint masked two-argument slot calibration.

The research intra-row reservoir gives one relation pivot with a left and right
content argument in the same active 256-token view. The research four-cell scorer
masked one argument at a time, leaving the other argument visible. This sharper
no-update scorer masks both argument token groups simultaneously, then compares:

  original = log p(T_left at left_positions) + log p(T_right at right_positions)
  swapped  = log p(T_right at left_positions) + log p(T_left at right_positions)

using the same frozen logits. The target token shape equality required by the
funnel makes the swap well-defined. This tests same-context role/slot assignment
rather than cross-row attested coherence. No training, optimizer update, or
official evaluation text is used.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, DebertaV2ForMaskedLM

USER_ROOT = Path('.').resolve()
COMPACT_EXPERIENCE_SCRIPTS = USER_ROOT / 'experiments/archive/compact_experience/scripts'
if str(COMPACT_EXPERIENCE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(COMPACT_EXPERIENCE_SCRIPTS))

DEFAULT_PAIR_POOL = USER_ROOT / 'experiments/archive/representation_and_objectives/data/intrarow_twoarg_relation_funnel_pilot20k/intrarow_twoarg_pair_pool.jsonl'
DEFAULT_TOKENIZER = USER_ROOT / 'experiments/archive/representation_and_objectives/data/shared_tokenizer/shared_16k_tokenizer'
DEFAULT_OUT = USER_ROOT / 'experiments/archive/representation_and_objectives/data/intrarow_joint_slot_calibration_pilot20k'
DEFAULT_NOTE = USER_ROOT / 'research/notes/representation_and_objectives/intrarow_joint_slot_calibration_pilot20k.md'
DEFAULT_ARMS = {
    'compact': USER_ROOT / 'experiments/archive/frontier_consolidation/training/runs/fw_compact_view_shared16k_seed43022/hf_model/chck_100M',
    'rowblock': USER_ROOT / 'experiments/archive/frontier_consolidation/training/runs/fw_source_breadth_shared16k_seed43022/hf_model/chck_100M',
    'interleaved': USER_ROOT / 'experiments/archive/representation_and_objectives/training/runs/fw_source_breadth_interleaved_wholesentence_fullbatch_shared16k_seed43022/hf_model/chck_100M',
}
BAD_TARGETS = {
    'a','an','the','and','or','but','of','for','to','from','with','without','by','as','at','in','on','up','down','out','off','over',
    'this','that','these','those','there','here','then','than','also','very','just','only','some','any','all','another','other',
    'i','you','he','she','it','we','they','me','him','her','us','them','my','your','his','its','our','their','what','which','who','where','why','how','when',
    'yes','no','not','do','does','did','have','has','had','is','are','was','were','be','been','being','can','could','will','would','should','may','might','must','shall',
    'one','two','three','four','five','six','seven','eight','nine','ten','said','say','says','get','got','go','come','came','see','look','like','make','made','take','took','put','let','want','need','know','think',
    'chi','mot','fat','mar','bro','sis','mom','dad','par','inv','int','exp','add','com','act','pho','sit','gra','gpx','x','xx','xxx','uh','uhhuh','huh','hm','yeah','okay','ok','ooh','ah','oh',
    'thing','things','something','anything','everything','someone','people','person','way','time','day','year','part','kind','sort','place','work','use','used','using','good','bad','new','old','first','last','many','much','same','different','right','left','actually','really','well','course','maybe','probably','number','mhm',
}


def now_utc() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def qstats(vals: Iterable[float]) -> dict[str, Any]:
    xs = sorted(float(v) for v in vals if math.isfinite(float(v)))
    if not xs:
        return {'n': 0}
    arr = np.asarray(xs, dtype=np.float64)
    def q(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        idx = p*(len(xs)-1); lo=int(math.floor(idx)); hi=int(math.ceil(idx))
        if lo == hi:
            return xs[lo]
        return xs[lo]*(hi-idx)+xs[hi]*(idx-lo)
    return {'n': len(xs), 'mean': float(arr.mean()), 'std': float(arr.std()), 'stderr': float(arr.std()/math.sqrt(len(xs))) if len(xs) else None, 'median': q(0.5), 'p05': q(0.05), 'p25': q(0.25), 'p75': q(0.75), 'p95': q(0.95), 'min': xs[0], 'max': xs[-1], 'success_gt0': float((arr > 0).mean())}


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    x=np.asarray(xs, dtype=np.float64); y=np.asarray(ys, dtype=np.float64)
    if float(x.std()) <= 1e-12 or float(y.std()) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0,1])


def load_pairs(path: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    pairs=[]; reject=Counter()
    with path.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            p=json.loads(line)
            if args.categories and str(p.get('category')) not in set(args.categories):
                reject['category'] += 1; continue
            ta=str(p.get('target_norm_a','')).lower(); tb=str(p.get('target_norm_b','')).lower()
            if args.exclude_bad_targets and (ta in BAD_TARGETS or tb in BAD_TARGETS):
                reject['bad_target'] += 1; continue
            if int(p.get('target_token_len', -1)) <= 0 or len(p.get('target_ids_a', [])) != len(p.get('target_ids_b', [])):
                reject['bad_token_length'] += 1; continue
            if str(p.get('target_norm_a')) == str(p.get('target_norm_b')):
                reject['same_target'] += 1; continue
            pairs.append(p)
    if args.max_pairs > 0:
        pairs=pairs[:args.max_pairs]
    return pairs


def build_masked_context(tokenizer: Any, p: dict[str, Any], seq_length: int, strict_verify: bool) -> dict[str, Any]:
    text=str(p['text_a'])
    posa=[int(x) for x in p['target_positions_a']]
    posb=[int(x) for x in p['target_positions_b']]
    ida=[int(x) for x in p['target_ids_a']]
    idb=[int(x) for x in p['target_ids_b']]
    if len(posa) != len(ida) or len(posb) != len(idb) or len(ida) != len(idb):
        raise ValueError(f"bad target position/id lengths for {p.get('pair_id')}")
    if set(posa) & set(posb):
        raise ValueError(f"overlapping target positions for {p.get('pair_id')}")
    enc=tokenizer(text, add_special_tokens=False, truncation=True, max_length=seq_length, padding='max_length', return_tensors='pt')
    ids=enc['input_ids'].squeeze(0).tolist(); attn=enc['attention_mask'].squeeze(0).tolist()
    for pos, target_ids, side in [(posa, ida, 'a'), (posb, idb, 'b')]:
        if any(q < 0 or q >= seq_length or attn[q] == 0 for q in pos):
            raise ValueError(f"inactive target pos {p.get('pair_id')} {side}")
        actual=[int(ids[q]) for q in pos]
        if strict_verify and actual != target_ids:
            raise ValueError(f"target ids mismatch {p.get('pair_id')} {side}: actual={actual} saved={target_ids}")
    mask_id=int(tokenizer.mask_token_id)
    for q in posa+posb:
        ids[q]=mask_id
    labels_orig=[-100]*len(ids); labels_swap=[-100]*len(ids)
    for q, tid in zip(posa, ida): labels_orig[q]=tid
    for q, tid in zip(posb, idb): labels_orig[q]=tid
    for q, tid in zip(posa, idb): labels_swap[q]=tid
    for q, tid in zip(posb, ida): labels_swap[q]=tid
    return {'ids': ids, 'attn': attn, 'labels_orig': labels_orig, 'labels_swap': labels_swap, 'pos_a': posa, 'pos_b': posb}


def precompute(pairs: list[dict[str, Any]], tokenizer: Any, seq_length: int, strict_verify: bool) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], dict[str, Any]]:
    items=[]; bad=[]
    for p in pairs:
        try:
            items.append((p, build_masked_context(tokenizer, p, seq_length, strict_verify)))
        except Exception as e:
            bad.append({'pair_id': p.get('pair_id'), 'error': str(e)[:300]})
    if bad and strict_verify:
        raise RuntimeError('context verification failed: '+json.dumps(bad[:10], ensure_ascii=False))
    return items, {'input_pairs': len(pairs), 'good_pairs': len(items), 'bad_contexts': len(bad), 'bad_examples': bad[:20]}


def score_label(logp_row: torch.Tensor, labels_row: torch.Tensor) -> float:
    pos=(labels_row != -100).nonzero(as_tuple=False).view(-1)
    tg=labels_row.index_select(0, pos)
    vals=logp_row.index_select(0, pos).gather(1, tg.view(-1,1)).view(-1)
    return float(vals.mean().detach().cpu().item())


def score_items(model: torch.nn.Module, items: list[tuple[dict[str, Any], dict[str, Any]]], *, device: torch.device, batch_size: int) -> list[dict[str, Any]]:
    rows=[]; model.eval()
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            chunk=items[start:start+batch_size]
            ids=torch.tensor([c[1]['ids'] for c in chunk], dtype=torch.long, device=device)
            attn=torch.tensor([c[1]['attn'] for c in chunk], dtype=torch.long, device=device)
            lab_o=torch.tensor([c[1]['labels_orig'] for c in chunk], dtype=torch.long, device=device)
            lab_s=torch.tensor([c[1]['labels_swap'] for c in chunk], dtype=torch.long, device=device)
            out=model(input_ids=ids, attention_mask=attn)
            logp=F.log_softmax(out.logits.float(), dim=-1)
            for r,(p,ctx) in enumerate(chunk):
                orig=score_label(logp[r], lab_o[r]); sw=score_label(logp[r], lab_s[r])
                # Position-specific two halves help distinguish role effect from one side dominating.
                la=[-100]*len(ctx['ids']); lb=[-100]*len(ctx['ids']); lsa=[-100]*len(ctx['ids']); lsb=[-100]*len(ctx['ids'])
                for q,tid in zip(ctx['pos_a'], p['target_ids_a']): la[q]=int(tid)
                for q,tid in zip(ctx['pos_b'], p['target_ids_b']): lb[q]=int(tid)
                for q,tid in zip(ctx['pos_a'], p['target_ids_b']): lsa[q]=int(tid)
                for q,tid in zip(ctx['pos_b'], p['target_ids_a']): lsb[q]=int(tid)
                la_t=torch.tensor(la, dtype=torch.long, device=device); lb_t=torch.tensor(lb, dtype=torch.long, device=device); lsa_t=torch.tensor(lsa, dtype=torch.long, device=device); lsb_t=torch.tensor(lsb, dtype=torch.long, device=device)
                score_a=score_label(logp[r], la_t); score_b=score_label(logp[r], lb_t); score_swap_a=score_label(logp[r], lsa_t); score_swap_b=score_label(logp[r], lsb_t)
                rec={
                    'pair_id': p['pair_id'], 'category': p.get('category'), 'pivot': p.get('pivot_norm_a'), 'source': p.get('source_a'),
                    'target_a': p.get('target_norm_a'), 'target_b': p.get('target_norm_b'), 'target_class': p.get('target_class'), 'target_token_len': int(p.get('target_token_len', -1)),
                    'target_freq_bin_a': p.get('target_freq_bin_a'), 'target_freq_bin_b': p.get('target_freq_bin_b'), 'left_distance': int(p.get('left_distance', -1)), 'right_distance': int(p.get('right_distance', -1)), 'distance_delta': abs(int(p.get('left_distance', -1))-int(p.get('right_distance', -1))), 'between_content_count': int(p.get('between_content_count', 0)),
                    'orig_score': orig, 'swap_score': sw, 'slot_margin': orig-sw, 'left_margin': score_a-score_swap_a, 'right_margin': score_b-score_swap_b,
                }
                rows.append(rec)
            del ids, attn, lab_o, lab_s, out, logp
            if start == 0 or (start//batch_size) % 50 == 0:
                print(json.dumps({'event':'joint_score_progress','views_done':min(start+len(chunk),len(items)),'views_total':len(items)}), flush=True)
    return rows


def summarize(rows_by_arm: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    summary={'arms':{}, 'arm_order':{}, 'paired_deltas':{}, 'correlations':{}}
    metrics=['slot_margin','left_margin','right_margin','orig_score','swap_score']
    for arm, rows in rows_by_arm.items():
        arm_sum={'n_pairs':len(rows), 'overall':{}, 'by_category':{}, 'by_pivot_ge50':{}}
        for m in metrics:
            arm_sum['overall'][m]=qstats([r[m] for r in rows])
        for cat in sorted({str(r['category']) for r in rows}):
            xs=[r for r in rows if str(r['category'])==cat]
            arm_sum['by_category'][cat]={m:qstats([r[m] for r in xs]) for m in metrics}; arm_sum['by_category'][cat]['n_pairs']=len(xs)
        byp=defaultdict(list)
        for r in rows: byp[str(r['pivot'])].append(r)
        for piv,xs in sorted(byp.items()):
            if len(xs)>=50:
                arm_sum['by_pivot_ge50'][piv]={m:qstats([r[m] for r in xs]) for m in metrics}; arm_sum['by_pivot_ge50'][piv]['n_pairs']=len(xs)
        summary['arms'][arm]=arm_sum
    for m in metrics:
        means={a: summary['arms'][a]['overall'][m]['mean'] for a in rows_by_arm if summary['arms'][a]['overall'][m].get('n')}
        summary['arm_order'][m]={'means':means,'rank_high_to_low':sorted(means,key=lambda a:means[a], reverse=True),'range':max(means.values())-min(means.values()) if means else None}
        cats=sorted({str(r['category']) for rows in rows_by_arm.values() for r in rows})
        bycat={}
        for cat in cats:
            cm={}
            for a in rows_by_arm:
                item=summary['arms'][a]['by_category'].get(cat,{}).get(m,{})
                if item.get('n'): cm[a]=item.get('mean')
            if cm: bycat[cat]={'means':cm,'rank_high_to_low':sorted(cm,key=lambda a:cm[a], reverse=True),'range':max(cm.values())-min(cm.values())}
        summary['arm_order'][m]['by_category']=bycat
    arms=sorted(rows_by_arm); rowdict={a:{r['pair_id']:r for r in rows_by_arm[a]} for a in arms}
    for i,a in enumerate(arms):
        for b in arms[i+1:]:
            common=sorted(set(rowdict[a]) & set(rowdict[b])); key=f'{a}_minus_{b}'
            d={'n_common':len(common),'overall':{},'by_category':{}}
            for m in ['slot_margin','left_margin','right_margin']:
                d['overall'][m]=qstats([rowdict[a][pid][m]-rowdict[b][pid][m] for pid in common])
            for cat in sorted({str(rowdict[a][pid]['category']) for pid in common}):
                pids=[pid for pid in common if str(rowdict[a][pid]['category'])==cat]
                d['by_category'][cat]={'n_pairs':len(pids)}
                for m in ['slot_margin','left_margin','right_margin']:
                    d['by_category'][cat][m]=qstats([rowdict[a][pid][m]-rowdict[b][pid][m] for pid in pids])
            summary['paired_deltas'][key]=d
            summary['correlations'][f'{a}__{b}']={m:pearson([rowdict[a][pid][m] for pid in common],[rowdict[b][pid][m] for pid in common]) for m in ['slot_margin','left_margin','right_margin']}
    if all(a in rows_by_arm for a in ['compact','rowblock','interleaved']):
        rd=rowdict; common=sorted(set(rd['compact']) & set(rd['rowblock']) & set(rd['interleaved']))
        rel={'common_pairs':len(common)}
        for hi,lo in [('rowblock','compact'),('interleaved','compact'),('rowblock','interleaved')]:
            rel[f'{hi}_minus_{lo}']={}
            for m in ['slot_margin','left_margin','right_margin']:
                rel[f'{hi}_minus_{lo}'][m]=qstats([rd[hi][pid][m]-rd[lo][pid][m] for pid in common])
        summary['relation_oriented_deltas']=rel
    return summary


def parse_arm_specs(specs: list[str]) -> dict[str, Path]:
    if not specs:
        return dict(DEFAULT_ARMS)
    out={}
    for s in specs:
        if '=' not in s:
            raise ValueError(f'arm spec must be name=path, got {s}')
        k,v=s.split('=',1); out[k.strip()]=Path(v.strip())
    return out


def write_outputs(outdir: Path, note_path: Path, args: argparse.Namespace, pairs: list[dict[str, Any]], ctx_stats: dict[str, Any], rows_by_arm: dict[str, list[dict[str, Any]]], summary: dict[str, Any], runtime: float, arm_paths: dict[str, str]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    score_path=outdir/'joint_slot_pair_scores.jsonl'
    with score_path.open('w',encoding='utf-8') as f:
        for arm in sorted(rows_by_arm):
            for r in rows_by_arm[arm]:
                rec=dict(r); rec['arm']=arm; f.write(json.dumps(rec, ensure_ascii=False)+'\n')
    csv_path=outdir/'joint_slot_pair_scores_summary.csv'
    fields=['arm','pair_id','category','pivot','target_a','target_b','target_class','target_token_len','left_distance','right_distance','between_content_count','slot_margin','left_margin','right_margin','orig_score','swap_score']
    with csv_path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for arm in sorted(rows_by_arm):
            for r in rows_by_arm[arm]:
                rec=dict(r); rec['arm']=arm; w.writerow({k:rec.get(k) for k in fields})
    full={'status':'INTRAROW_JOINT_SLOT_CALIBRATION','created_utc':now_utc(),'purpose':'Frozen-weight same-context two-argument joint-mask slot assignment test: original argument placement vs swapped placement from identical logits.','inputs':{'pair_pool':str(args.pair_pool),'tokenizer_path':str(args.tokenizer_path),'arm_paths':arm_paths},'parameters':vars(args),'selected_pair_count':len(pairs),'selected_pairs_by_category':dict(Counter(str(p.get('category')) for p in pairs)),'context_stats':ctx_stats,'summary':summary,'runtime_sec':runtime,'artifacts':{'score_jsonl':str(score_path),'score_csv':str(csv_path),'summary_json':str(outdir/'joint_slot_calibration_summary.json'),'note':str(note_path)}}
    (outdir/'joint_slot_calibration_summary.json').write_text(json.dumps(full, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    lines=['# research — intra-row joint slot calibration','']
    lines.append(f"Created: {full['created_utc']}  Runtime: {runtime:.1f} s")
    lines.append('')
    lines.append('Both argument positions are masked simultaneously. The margin is original left/right assignment minus swapped left/right assignment from the same frozen logits. No training was performed.')
    lines.append(f"Selected pairs: **{len(pairs):,}**; by family: `{full['selected_pairs_by_category']}`")
    lines.append(f"Context verification: `{ctx_stats}`")
    lines.append('')
    lines.append('## Arm means')
    lines.append('| metric | compact | rowblock | interleaved | rank | range |')
    lines.append('|---|---:|---:|---:|---|---:|')
    for m in ['slot_margin','left_margin','right_margin','orig_score','swap_score']:
        item=summary['arm_order'].get(m,{}); means=item.get('means',{})
        lines.append(f"| {m} | {means.get('compact')} | {means.get('rowblock')} | {means.get('interleaved')} | {' > '.join(item.get('rank_high_to_low', []))} | {item.get('range')} |")
    lines.append('')
    lines.append('## Family slot-margin means')
    lines.append('| family | n | compact | rowblock | interleaved | rank | range |')
    lines.append('|---|---:|---:|---:|---:|---|---:|')
    for cat,item in sorted(summary['arm_order'].get('slot_margin',{}).get('by_category',{}).items()):
        means=item.get('means',{}); n=summary['arms'].get('compact',{}).get('by_category',{}).get(cat,{}).get('n_pairs')
        lines.append(f"| {cat} | {n} | {means.get('compact')} | {means.get('rowblock')} | {means.get('interleaved')} | {' > '.join(item.get('rank_high_to_low', []))} | {item.get('range')} |")
    lines.append('')
    lines.append('## Relation-oriented paired deltas')
    rel=summary.get('relation_oriented_deltas',{})
    for dk in ['rowblock_minus_compact','interleaved_minus_compact','rowblock_minus_interleaved']:
        if dk not in rel: continue
        lines.append(f'### {dk}')
        for m in ['slot_margin','left_margin','right_margin']:
            st=rel[dk][m]
            lines.append(f"- {m}: mean={st.get('mean')} stderr={st.get('stderr')} success_gt0={st.get('success_gt0')} n={st.get('n')}")
        lines.append('')
    lines.append('Files:')
    lines.append(f"- JSON summary: `{outdir/'joint_slot_calibration_summary.json'}`")
    lines.append(f"- pair scores: `{score_path}`")
    lines.append(f"- CSV: `{csv_path}`")
    note_path.parent.mkdir(parents=True, exist_ok=True); note_path.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'status':full['status'],'pairs':len(pairs),'arms':sorted(rows_by_arm),'summary':str(outdir/'joint_slot_calibration_summary.json'),'note':str(note_path)}, indent=2), flush=True)


def build_args() -> argparse.Namespace:
    ap=argparse.ArgumentParser()
    ap.add_argument('--pair_pool', default=str(DEFAULT_PAIR_POOL)); ap.add_argument('--tokenizer_path', default=str(DEFAULT_TOKENIZER)); ap.add_argument('--output_dir', default=str(DEFAULT_OUT)); ap.add_argument('--note_path', default=str(DEFAULT_NOTE))
    ap.add_argument('--arm', action='append', default=[]); ap.add_argument('--categories', nargs='*', default=[]); ap.add_argument('--exclude_bad_targets', action='store_true', default=True); ap.add_argument('--include_bad_targets', dest='exclude_bad_targets', action='store_false')
    ap.add_argument('--seq_length', type=int, default=256); ap.add_argument('--batch_size', type=int, default=256); ap.add_argument('--device', default='cuda'); ap.add_argument('--dtype', default='bf16', choices=['fp32','fp16','bf16']); ap.add_argument('--max_pairs', type=int, default=0); ap.add_argument('--no_strict_verify', action='store_true')
    return ap.parse_args()


def main() -> None:
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false'); os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
    t0=time.time(); args=build_args()
    tok=AutoTokenizer.from_pretrained(Path(args.tokenizer_path), trust_remote_code=True)
    if tok.mask_token_id is None:
        raise RuntimeError('tokenizer has no mask_token_id')
    pairs=load_pairs(Path(args.pair_pool), args)
    if not pairs:
        raise RuntimeError('no pairs selected')
    items, ctx_stats=precompute(pairs, tok, args.seq_length, strict_verify=(not args.no_strict_verify))
    arm_paths=parse_arm_specs(args.arm)
    for k,v in arm_paths.items():
        if not Path(v).exists():
            raise RuntimeError(f'checkpoint for arm {k} not found: {v}')
    device=torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else args.device)
    dtype={'fp32':torch.float32,'fp16':torch.float16,'bf16':torch.bfloat16}[args.dtype]
    rows_by_arm={}
    for arm,path in arm_paths.items():
        print(json.dumps({'event':'load_model','arm':arm,'checkpoint':str(path),'device':str(device),'dtype':args.dtype,'pairs':len(items)}), flush=True)
        model=DebertaV2ForMaskedLM.from_pretrained(path, torch_dtype=dtype)
        model.to(device)
        rows_by_arm[arm]=score_items(model, items, device=device, batch_size=args.batch_size)
        del model
        if device.type == 'cuda': torch.cuda.empty_cache()
    summary=summarize(rows_by_arm)
    write_outputs(Path(args.output_dir), Path(args.note_path), args, pairs, ctx_stats, rows_by_arm, summary, time.time()-t0, {k:str(v) for k,v in arm_paths.items()})


if __name__=='__main__':
    main()
