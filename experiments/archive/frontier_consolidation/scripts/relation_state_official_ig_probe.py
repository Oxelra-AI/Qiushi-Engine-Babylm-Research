#!/usr/bin/env python3
"""research: bounded official EWoK/Entity context-dependence probe.

Purpose: test the specified relation-state connection. For the exact
EWoK/Entity items that flip between scale1.75 chck82 and chck100, measure whether
the 82M losses have a distinctive low-context-dependence/high-gold-NLL signature.
No training, no token-category mining, no leaderboard submission.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/frontier_consolidation')
WORKSPACE = _public_path('experiments/archive/frontier_consolidation')
SCRIPTS = _public_path('experiments/archive/frontier_consolidation/scripts')
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from pairwise_item_flip_analysis import PayloadLoader  # noqa: E402
from anchor_margin_alpha_census import build_candidate_maps  # noqa: E402

BASE_PAYLOAD = _public_path('experiments/archive/representation_and_objectives/data/scale1p75_chck82_full_eval_reproduction/staged_full_eval/per_target/scale1p75_chck82_independent.json')
CAND_PAYLOAD = _public_path('experiments/archive/frontier_consolidation/data/scale1p75_100M_repaired_merge/staged_full_eval/per_target/scale1p75_100M_seed43022.json')
CHCK82 = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M')
CHCK100 = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_100M')
DEFAULT_OUT = _public_path('experiments/archive/frontier_consolidation/data/relation_state_official_ig_probe')


def rel(p: Path | str) -> str:
    q = Path(p)
    try:
        return str(q.resolve().relative_to(ROOT))
    except Exception:
        return str(q)


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def set_cache_env(out_dir: Path, gpu: int | None) -> None:
    cache = out_dir / 'runtime_cache'
    for key, sub in {
        'HF_HOME': 'home',
        'HF_HUB_CACHE': 'hub',
        'HUGGINGFACE_HUB_CACHE': 'hub',
        'HF_DATASETS_CACHE': 'datasets',
        'TRANSFORMERS_CACHE': 'transformers',
        'HF_MODULES_CACHE': 'modules',
        'XDG_CACHE_HOME': 'xdg',
    }.items():
        p = cache / sub
        p.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(p.resolve())
    if gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)


def mean(xs):
    xs=[float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return sum(xs)/len(xs) if xs else None


def quantile(xs, q):
    xs=sorted(float(x) for x in xs if math.isfinite(float(x)))
    if not xs: return None
    i=(len(xs)-1)*q
    lo=math.floor(i); hi=math.ceil(i)
    if lo==hi: return xs[lo]
    return xs[lo]*(hi-i)+xs[hi]*(i-lo)


def summarize(xs):
    xs=[float(x) for x in xs if math.isfinite(float(x))]
    if not xs: return {'n':0}
    return {'n':len(xs),'mean':mean(xs),'median':quantile(xs,0.5),'p10':quantile(xs,0.1),'p90':quantile(xs,0.9),'min':min(xs),'max':max(xs)}


def pearson(xs, ys):
    xs=[float(x) for x in xs]; ys=[float(y) for y in ys]
    if len(xs)<3 or len(xs)!=len(ys): return None
    mx=sum(xs)/len(xs); my=sum(ys)/len(ys)
    vx=sum((x-mx)**2 for x in xs); vy=sum((y-my)**2 for y in ys)
    if vx<=0 or vy<=0: return None
    return sum((x-mx)*(y-my) for x,y in zip(xs,ys))/math.sqrt(vx*vy)


def classify_items(seed: int, stable_sample_per_status: int) -> list[dict[str, Any]]:
    base_loader = PayloadLoader(BASE_PAYLOAD)
    cand_loader = PayloadLoader(CAND_PAYLOAD)
    # Candidate reconstruction from the base payload's official data coordinate.
    candidate_maps = build_candidate_maps(base_loader)
    rng = random.Random(seed)
    selected=[]
    summary={}
    for col in ['EWoK','Entity']:
        base_rows,_ = base_loader.load_column(col)
        cand_rows,_ = cand_loader.load_column(col)
        bmap={r.item_id:r for r in base_rows}
        cmap={r.item_id:r for r in cand_rows}
        by_status=defaultdict(list)
        for iid in sorted(set(bmap)&set(cmap)):
            b=bmap[iid]; c=cmap[iid]
            if b.correct and not c.correct:
                st='loss'
            elif (not b.correct) and c.correct:
                st='gain'
            elif b.correct and c.correct:
                st='both_correct'
            else:
                st='both_wrong'
            by_status[st].append((b,c))
        summary[col]={k:len(v) for k,v in by_status.items()}
        for st, rows in by_status.items():
            rows=list(rows)
            # All flips; bounded stable samples.
            if st in {'loss','gain'}:
                keep=rows
            else:
                keep=rows[:] if len(rows) <= stable_sample_per_status else rng.sample(rows, stable_sample_per_status)
            for b,c in keep:
                meta = candidate_maps[col].get(b.item_id)
                if not meta:
                    continue
                selected.append({
                    'column': col,
                    'item_id': b.item_id,
                    'uid': b.uid,
                    'status_82_to_100': st,
                    'base_correct': bool(b.correct),
                    'cand_correct': bool(c.correct),
                    'base_pred': b.pred,
                    'cand_pred': c.pred,
                    'gold': b.gold,
                    'candidate_meta': meta,
                    'official_meta': b.meta or {},
                })
    return selected, summary


def token_positions_for_completion(tokenizer, text: str, completion: str) -> tuple[list[int], list[int], list[tuple[int,int]]]:
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True, return_tensors=None)
    ids = [int(x) for x in enc['input_ids']]
    offsets = list(enc['offset_mapping'])
    # Special tokens usually have (0,0). Real content positions have end > start.
    start_char = len(text) - len(completion)
    positions=[]
    for i,(s,e) in enumerate(offsets):
        if e > s and e > start_char:
            positions.append(i)
    return ids, positions, offsets


def make_masked(ids: list[int], pos: int, tokenizer, mode: str, radius: int, max_seq_len: int) -> tuple[list[int], int]:
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        raise RuntimeError('missing mask token')
    cls_id = getattr(tokenizer,'cls_token_id',None) or getattr(tokenizer,'bos_token_id',None)
    sep_id = getattr(tokenizer,'sep_token_id',None) or getattr(tokenizer,'eos_token_id',None)
    special = {x for x in [getattr(tokenizer,'bos_token_id',None), getattr(tokenizer,'eos_token_id',None), getattr(tokenizer,'cls_token_id',None), getattr(tokenizer,'sep_token_id',None), getattr(tokenizer,'pad_token_id',None)] if x is not None}
    if mode == 'full':
        seq=list(ids)
        seq[pos]=int(mask_id)
        mask_pos=pos
    elif mode == 'local':
        # Map to real-token stream and add fresh boundaries, matching the research corpus probe.
        real_positions=[i for i,x in enumerate(ids) if int(x) not in special]
        if pos not in real_positions:
            raise RuntimeError('target pos not in real positions')
        j=real_positions.index(pos)
        real_ids=[int(ids[i]) for i in real_positions]
        start=max(0,j-radius); end=min(len(real_ids), j+radius+1)
        left=real_ids[start:j]
        right=real_ids[j+1:end]
        seq=[int(cls_id)] + left + [int(mask_id)] + right + [int(sep_id)]
        mask_pos=1+len(left)
    else:
        raise ValueError(mode)
    if len(seq) > max_seq_len:
        # Official candidates can be longer than 256; this probe is a bounded likelihood reader, so skip rather than truncate silently.
        raise ValueError(f'seq too long {len(seq)}>{max_seq_len}')
    return seq, mask_pos


def batch_nll(model, tokenizer, contexts: list[dict[str, Any]], device: str, batch_size: int) -> list[float]:
    import torch
    import torch.nn.functional as F
    pad = tokenizer.pad_token_id
    out=[]
    for start in range(0,len(contexts),batch_size):
        batch=contexts[start:start+batch_size]
        max_len=max(len(c['input_ids']) for c in batch)
        x=[]; am=[]; pos=[]; targ=[]
        for c in batch:
            ids=list(c['input_ids'])
            padn=max_len-len(ids)
            x.append(ids+[pad]*padn)
            am.append([1]*len(ids)+[0]*padn)
            pos.append(int(c['mask_index']))
            targ.append(int(c['target_id']))
        with torch.no_grad():
            xt=torch.tensor(x,dtype=torch.long,device=device)
            mt=torch.tensor(am,dtype=torch.long,device=device)
            logits=model(input_ids=xt, attention_mask=mt).logits
            pi=torch.tensor(pos,dtype=torch.long,device=device)
            ti=torch.tensor(targ,dtype=torch.long,device=device)
            ri=torch.arange(len(batch),device=device)
            nll=F.cross_entropy(logits[ri,pi,:], ti, reduction='none')
            out.extend(float(v) for v in nll.detach().cpu().tolist())
    return out


def build_contexts_for_items(items: list[dict[str, Any]], tokenizer, radius: int, max_seq_len: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    full=[]; local=[]; item_units=[]
    skipped=Counter()
    for idx,it in enumerate(items):
        meta=it['candidate_meta']
        lab=int(meta['label_index'])
        text=str(meta['candidates'][lab])
        completion=str(meta['completions'][lab])
        try:
            ids, positions, offsets = token_positions_for_completion(tokenizer, text, completion)
            if not positions:
                skipped['no_completion_tokens'] += 1
                continue
            for p in positions:
                tid=int(ids[p])
                try:
                    fseq, fpos = make_masked(ids,p,tokenizer,'full',radius,max_seq_len)
                    lseq, lpos = make_masked(ids,p,tokenizer,'local',radius,max_seq_len)
                except ValueError:
                    skipped['too_long'] += 1
                    continue
                unit={
                    'item_index': idx,
                    'token_pos': int(p),
                    'target_id': tid,
                    'target_token': tokenizer.convert_ids_to_tokens(tid),
                    'completion_token_count_item': len(positions),
                }
                full.append({'input_ids':fseq,'mask_index':fpos,'target_id':tid})
                local.append({'input_ids':lseq,'mask_index':lpos,'target_id':tid})
                item_units.append(unit)
        except Exception as e:
            skipped[type(e).__name__] += 1
    return full, local, item_units, dict(skipped)


def run(args):
    out_dir=Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_cache_env(out_dir, args.gpu)
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    device='cuda' if args.gpu is not None and torch.cuda.is_available() else 'cpu'
    items, class_summary = classify_items(args.seed, args.stable_sample_per_status)
    ckpts={'chck82': CHCK82, 'chck100': CHCK100}
    all_records=[]
    timing={}
    for label, ckpt in ckpts.items():
        t0=time.time()
        tokenizer=AutoTokenizer.from_pretrained(str(ckpt), trust_remote_code=True, local_files_only=True)
        model=AutoModelForMaskedLM.from_pretrained(str(ckpt), trust_remote_code=True, local_files_only=True)
        model.to(device); model.eval()
        full_ctx, local_ctx, units, skipped = build_contexts_for_items(items, tokenizer, args.radius, args.max_seq_len)
        full_nll=batch_nll(model, tokenizer, full_ctx, device, args.batch_size)
        local_nll=batch_nll(model, tokenizer, local_ctx, device, args.batch_size)
        by_item=defaultdict(lambda: {'full':[], 'local':[], 'tokens':[]})
        for u,fn,ln in zip(units, full_nll, local_nll):
            by_item[u['item_index']]['full'].append(fn)
            by_item[u['item_index']]['local'].append(ln)
            by_item[u['item_index']]['tokens'].append(u['target_token'])
        for idx,it in enumerate(items):
            vals=by_item.get(idx)
            if not vals:
                continue
            rec={k:it[k] for k in ['column','item_id','uid','status_82_to_100','base_correct','cand_correct']}
            rec.update({
                'checkpoint': label,
                'gold_completion_token_count_scored': len(vals['full']),
                'gold_completion_full_nll_mean': mean(vals['full']),
                'gold_completion_full_nll_sum': sum(vals['full']),
                f'gold_completion_local_r{args.radius}_nll_mean': mean(vals['local']),
                f'gold_completion_ig_r{args.radius}_mean': mean([l-f for l,f in zip(vals['local'], vals['full'])]),
                'target_tokens_preview': ' '.join(vals['tokens'][:12]),
            })
            all_records.append(rec)
        del model
        if device=='cuda':
            torch.cuda.empty_cache()
        timing[label]={'elapsed_sec':time.time()-t0,'skipped':skipped,'scored_item_records':sum(1 for r in all_records if r['checkpoint']==label),'token_contexts':len(units)}
    # Add deltas chck100 - chck82.
    by={(r['checkpoint'], r['item_id']):r for r in all_records}
    for iid in sorted({r['item_id'] for r in all_records}):
        a=by.get(('chck82',iid)); b=by.get(('chck100',iid))
        if a and b:
            dn=b['gold_completion_full_nll_mean']-a['gold_completion_full_nll_mean']
            dig=b[f'gold_completion_ig_r{args.radius}_mean']-a[f'gold_completion_ig_r{args.radius}_mean']
            for r in (a,b):
                r['gold_full_nll_delta_chck100_minus_chck82']=dn
                r[f'gold_ig_r{args.radius}_delta_chck100_minus_chck82']=dig
    # Summary by column/status using chck82 rows.
    summary={'status':'RELATION_STATE_OFFICIAL_IG_PROBE','radius':args.radius,'device':device,'class_summary_full_common':class_summary,'selected_items':len(items),'timing':timing,'checkpoints':{k:rel(v) for k,v in ckpts.items()},'paths':{'base_payload':rel(BASE_PAYLOAD),'candidate_payload':rel(CAND_PAYLOAD)}}
    base_rows=[r for r in all_records if r['checkpoint']=='chck82']
    # thresholds within all EWoK+Entity selected items at chck82: analysis only, not a trainer.
    ig_vals=[r[f'gold_completion_ig_r{args.radius}_mean'] for r in base_rows]
    nll_vals=[r['gold_completion_full_nll_mean'] for r in base_rows]
    qig20=quantile(ig_vals,0.2); qnll80=quantile(nll_vals,0.8)
    summary['chck82_thresholds_selected_population']={'ig_q20':qig20,'full_nll_q80':qnll80}
    summary['by_column_status']={}
    for col in ['EWoK','Entity']:
        for st in ['loss','gain','both_correct','both_wrong']:
            rows=[r for r in base_rows if r['column']==col and r['status_82_to_100']==st]
            if not rows: continue
            delta=[r.get('gold_full_nll_delta_chck100_minus_chck82') for r in rows if 'gold_full_nll_delta_chck100_minus_chck82' in r]
            summary['by_column_status'][f'{col}:{st}']={
                'n':len(rows),
                'chck82_full_nll': summarize([r['gold_completion_full_nll_mean'] for r in rows]),
                f'chck82_ig_r{args.radius}': summarize([r[f'gold_completion_ig_r{args.radius}_mean'] for r in rows]),
                'delta_full_nll_100minus82': summarize(delta),
                'low_ig_high_error_fraction': mean(1.0 if (r[f'gold_completion_ig_r{args.radius}_mean'] <= qig20 and r['gold_completion_full_nll_mean'] >= qnll80) else 0.0 for r in rows),
                'worsened_gold_nll_fraction_delta_gt0': mean(1.0 if r.get('gold_full_nll_delta_chck100_minus_chck82',0.0) > 0 else 0.0 for r in rows),
            }
    # Loss-vs-gain contrasts.
    contrasts={}
    for col in ['EWoK','Entity']:
        loss=[r for r in base_rows if r['column']==col and r['status_82_to_100']=='loss']
        gain=[r for r in base_rows if r['column']==col and r['status_82_to_100']=='gain']
        if loss and gain:
            contrasts[col]={
                'loss_minus_gain_chck82_full_nll_mean': mean(r['gold_completion_full_nll_mean'] for r in loss)-mean(r['gold_completion_full_nll_mean'] for r in gain),
                f'loss_minus_gain_chck82_ig_r{args.radius}_mean': mean(r[f'gold_completion_ig_r{args.radius}_mean'] for r in loss)-mean(r[f'gold_completion_ig_r{args.radius}_mean'] for r in gain),
                'loss_minus_gain_delta_full_nll_100minus82_mean': mean(r['gold_full_nll_delta_chck100_minus_chck82'] for r in loss)-mean(r['gold_full_nll_delta_chck100_minus_chck82'] for r in gain),
                'loss_low_ig_high_error_fraction': summary['by_column_status'][f'{col}:loss']['low_ig_high_error_fraction'],
                'gain_low_ig_high_error_fraction': summary['by_column_status'][f'{col}:gain']['low_ig_high_error_fraction'],
            }
    summary['loss_vs_gain_contrasts']=contrasts
    # Save CSV.
    csv_path=out_dir/'relation_state_official_ig_items.csv'
    fields=sorted({k for r in all_records for k in r.keys()})
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(all_records)
    summary['item_csv']=rel(csv_path)
    out_json=out_dir/'relation_state_official_ig_summary.json'
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True)+'\n', encoding='utf-8')
    lines=['# research relation/state official IG probe','',f"Selected official EWoK/Entity items: `{len(items)}`; radius `{args.radius}`; device `{device}`.",'', '## Status counts in full common set']
    for col,cs in class_summary.items():
        lines.append(f'- {col}: {cs}')
    lines += ['', '## chck82 by official flip status']
    for key in sorted(summary['by_column_status']):
        s=summary['by_column_status'][key]
        lines.append(f"- {key}: n={s['n']}; fullNLL mean={s['chck82_full_nll']['mean']:.6f}; IG mean={s[f'chck82_ig_r{args.radius}']['mean']:.6f}; ΔNLL100-82 mean={s['delta_full_nll_100minus82']['mean']:.6f}; lowIG+highErr frac={s['low_ig_high_error_fraction']:.3f}; worsened gold NLL frac={s['worsened_gold_nll_fraction_delta_gt0']:.3f}")
    lines += ['', '## Loss minus gain contrasts']
    for col,c in contrasts.items():
        lines.append(f"- {col}: loss-gain fullNLL={c['loss_minus_gain_chck82_full_nll_mean']:+.6f}; loss-gain IG={c[f'loss_minus_gain_chck82_ig_r{args.radius}_mean']:+.6f}; loss-gain ΔNLL100-82={c['loss_minus_gain_delta_full_nll_100minus82_mean']:+.6f}; lowIG+highErr frac loss/gain={c['loss_low_ig_high_error_fraction']:.3f}/{c['gain_low_ig_high_error_fraction']:.3f}")
    lines += ['', '## Direct reading', '- If official relation/state losses were the same low-IG/high-error residual, loss items should have higher chck82 gold NLL, lower IG, and/or larger positive 100M-minus-82M gold-NLL deltas than gains. The numbers above are the bounded test of that connection.', '', f'CSV: `{rel(csv_path)}`', f'JSON: `{rel(out_json)}`']
    out_md=out_dir/'relation_state_official_ig_summary.md'
    out_md.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'status':summary['status'],'selected_items':len(items),'out_json':rel(out_json),'out_md':rel(out_md)}, indent=2))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=str(DEFAULT_OUT))
    ap.add_argument('--gpu', type=int, default=None)
    ap.add_argument('--radius', type=int, default=8)
    ap.add_argument('--max-seq-len', type=int, default=512)
    ap.add_argument('--batch-size', type=int, default=96)
    ap.add_argument('--seed', type=int, default=16303)
    ap.add_argument('--stable-sample-per-status', type=int, default=300)
    args=ap.parse_args()
    run(args)

if __name__ == '__main__':
    main()
