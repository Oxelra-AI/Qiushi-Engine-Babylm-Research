#!/usr/bin/env python3
"""research masked alternative consequence-selection calibration.

Instead of adding full-context auxiliary views, use the ordinary WWM masked forward.
Only event target groups that happen to be masked by ordinary WWM are used. At the
masked target positions, compare the MLM log-likelihood of the true dependent
/consequence group against a matched cross-event dependent target with same
category/target class/length/frequency when possible. This adds no extra input
views and no extra word-pass exposure if used as a future auxiliary.

This script performs no optimizer step. It measures coverage, matching quality,
pre-update margins and a small gradient ratio for the margin loss.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import DebertaV2ForMaskedLM

USER_ROOT = Path('.').resolve()
for p in [USER_ROOT/'experiments/archive/compact_experience/scripts', USER_ROOT/'experiments/archive/representation_and_objectives/scripts']:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import masking_curriculum_trainer as base  # noqa: E402
import pvdm_continuation_trainer as cont  # noqa: E402
import pvdm_masking_lib as pvdm  # noqa: E402
import sparse_relation_aux_trainer as relaux  # noqa: E402
import symmetric_cue_view_likelihood_probe as cueprobe  # noqa: E402

DEFAULT_OUT = Path('experiments/archive/representation_and_objectives/data/masked_alt_consequence_calibration')
DEFAULT_NOTE = Path('research/notes/representation_and_objectives/masked_alt_consequence_calibration.md')
DEFAULT_CKPT = Path('experiments/archive/representation_and_objectives/training/runs/standard_legacy_70M_to_80M_seed43022/hf_model/chck_80M')
START_TAIL_ROW = 64255
EXPECTED_START_WORDS = 10011326
MAX_WORDS = 9971308
CATEGORIES = {'physical_change','causal_connector','temporal','spatial','comparative','negation'}


def now_utc() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def reset_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed % (2**32 - 1)); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def summarize(xs: list[float]) -> dict[str, Any]:
    if not xs:
        return {'n': 0}
    ys = sorted(xs)
    def q(p: float) -> float:
        if len(ys) == 1: return ys[0]
        f = p*(len(ys)-1); lo = int(math.floor(f)); hi = int(math.ceil(f))
        if lo == hi: return ys[lo]
        return ys[lo]*(hi-f) + ys[hi]*(f-lo)
    return {'n': len(xs), 'mean': sum(xs)/len(xs), 'median': q(0.5), 'std': float(np.std(xs)), 'stderr': float(np.std(xs)/math.sqrt(len(xs))), 'p05': q(0.05), 'p95': q(0.95), 'success_gt0': sum(x > 0 for x in xs)/len(xs)}


def grad_l2(model: torch.nn.Module) -> float:
    s = 0.0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach().float(); s += float((g*g).sum().item())
    return math.sqrt(s)


def clone_grads(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


def grad_cos(model: torch.nn.Module, ref: dict[str, torch.Tensor]) -> float | None:
    dot=sq=rsq=0.0
    for n,p in model.named_parameters():
        if p.grad is None or n not in ref: continue
        g=p.grad.detach().float(); r=ref[n].detach().float().to(g.device)
        dot += float((g*r).sum().item()); sq += float((g*g).sum().item()); rsq += float((r*r).sum().item())
    if sq <= 0 or rsq <= 0: return None
    return dot / math.sqrt(sq*rsq)


def zero(model: torch.nn.Module) -> None:
    model.zero_grad(set_to_none=True)


def candidate_events_for_batch(input_ids: torch.Tensor, attention: torch.Tensor, word_group: torch.Tensor, label_records: list[dict[str, Any]], *, max_control_distance_abs_diff: int, max_control_freq_bin_abs_diff: int, max_target_token_len: int) -> list[dict[str, Any]]:
    out=[]
    for b, lr in enumerate(label_records):
        group_pos = pvdm.group_positions_for_row(word_group[b], attention[b])
        usable, _rej = pvdm.collect_active_events(lr, group_pos, same_length_required=False)
        for ev in usable:
            cat = str(ev.category)
            if cat not in CATEGORIES: continue
            if len({ev.pivot_gid, ev.target_gid, ev.control_gid}) < 3: continue
            ev_raw = cueprobe.raw_event(lr, ev.event_rank)
            meta = relaux._event_meta(ev_raw)
            if int(meta.get('control_distance_abs_diff', 999)) > max_control_distance_abs_diff: continue
            if int(meta.get('control_freq_bin_abs_diff', 999)) > max_control_freq_bin_abs_diff: continue
            tpos = [int(x) for x in group_pos[int(ev.target_gid)]]
            tids = [int(input_ids[b, x].item()) for x in tpos]
            if not tpos or len(tpos) > max_target_token_len: continue
            out.append({'batch_row': int(b), 'tail_row_idx': int(lr.get('tail_row_idx', -1)), 'event_rank': int(ev.event_rank), 'category': cat, 'target_positions': tpos, 'target_ids': tids, 'target_len': len(tids), 'target_norm': str(meta.get('target_norm')), 'target_class': str(meta.get('target_class')), 'target_freq_bin': str(meta.get('target_freq_bin')), 'target_freq_bin_id': int(meta.get('target_freq_bin_id', -1)), 'distance_bin': str(meta.get('distance_bin')), 'pivot_norm': str(meta.get('pivot_norm')), **meta})
    return out


def build_negative_pools(examples, labels, tokenizer, *, batch_size: int, seq_length: int, max_control_distance_abs_diff: int, max_control_freq_bin_abs_diff: int, max_target_token_len: int, num_workers: int) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    dataset = base.MaskedChunkDataset(examples, tokenizer, seq_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=base.collate, num_workers=num_workers)
    pools: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    total=0
    for step,batch in enumerate(loader,1):
        lo=(research)*batch_size; hi=lo+int(batch['input_ids'].shape[0])
        evs = candidate_events_for_batch(batch['input_ids'][:,:seq_length].contiguous(), batch['attention_mask'][:,:seq_length].contiguous(), batch['word_group'][:,:seq_length].contiguous(), labels[lo:hi], max_control_distance_abs_diff=max_control_distance_abs_diff, max_control_freq_bin_abs_diff=max_control_freq_bin_abs_diff, max_target_token_len=max_target_token_len)
        for e in evs:
            donor = {k:e[k] for k in ['tail_row_idx','event_rank','category','target_len','target_ids','target_norm','target_class','target_freq_bin','target_freq_bin_id','distance_bin'] if k in e}
            keys = [
                (e['category'], e['target_class'], e['target_len'], e['target_freq_bin'], e['distance_bin']),
                (e['category'], e['target_class'], e['target_len'], e['target_freq_bin']),
                (e['category'], e['target_class'], e['target_len']),
                (e['category'], e['target_len']),
            ]
            for key in keys: pools[key].append(donor)
            total += 1
        if step == 1 or step % 50 == 0:
            print(json.dumps({'event':'negative_pool_progress','step':step,'donors_added':total,'keys':len(pools)}), flush=True)
    return pools


def neg_keys(ev: dict[str, Any]) -> list[tuple[Any, ...]]:
    return [
        (ev['category'], ev['target_class'], ev['target_len'], ev['target_freq_bin'], ev['distance_bin']),
        (ev['category'], ev['target_class'], ev['target_len'], ev['target_freq_bin']),
        (ev['category'], ev['target_class'], ev['target_len']),
        (ev['category'], ev['target_len']),
    ]


def attach_negative(evs: list[dict[str, Any]], pools: dict[tuple[Any, ...], list[dict[str, Any]]], *, seed: int, step: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out=[]; levels=Counter(); drops=Counter()
    for ev in evs:
        donor=None; used=None
        for level,key in enumerate(neg_keys(ev)):
            pool=pools.get(key, [])
            if not pool: continue
            order=sorted(range(len(pool)), key=lambda i: pvdm.hash_uniform(seed, 'altneg', step, level, ev['tail_row_idx'], ev['event_rank'], i))
            for i in order[:256]:
                cand=pool[i]
                if int(cand['tail_row_idx']) == int(ev['tail_row_idx']): continue
                if str(cand.get('target_norm')) == str(ev.get('target_norm')): continue
                if len(cand.get('target_ids', [])) != len(ev.get('target_ids', [])): continue
                donor=cand; used=level; break
            if donor is not None: break
        if donor is None:
            drops[str(ev['category'])]+=1; continue
        e2=dict(ev); e2['negative_target_ids']=[int(x) for x in donor['target_ids']]; e2['negative_target_norm']=str(donor.get('target_norm')); e2['negative_tail_row_idx']=int(donor['tail_row_idx']); e2['negative_match_level']=int(used); out.append(e2); levels[str(used)] += 1
    return out, {'input_events':len(evs),'used_events':len(out),'drop_by_category':dict(drops),'match_levels':dict(levels),'used_categories':dict(Counter(e['category'] for e in out))}


def filter_naturally_masked(evs: list[dict[str, Any]], labels_cpu: torch.Tensor) -> list[dict[str, Any]]:
    out=[]
    for e in evs:
        br=int(e['batch_row'])
        if all(int(labels_cpu[br, int(p)].item()) != -100 for p in e['target_positions']):
            out.append(e)
    return out


def cap_balanced(evs: list[dict[str, Any]], *, seed: int, step: int, cap_per_cat: int, cap_total: int) -> list[dict[str, Any]]:
    chosen=[]
    for cat in sorted(CATEGORIES):
        sub=[e for e in evs if e['category']==cat]
        sub.sort(key=lambda e: pvdm.hash_uniform(seed, 'masked_alt_cap', step, cat, e['tail_row_idx'], e['event_rank'], e.get('target_norm')))
        chosen.extend(sub[:cap_per_cat])
    chosen.sort(key=lambda e: pvdm.hash_uniform(seed, 'masked_alt_order', step, e['tail_row_idx'], e['event_rank'], e['category']))
    return chosen[:cap_total]


def logp_targets_from_logits(logits: torch.Tensor, evs: list[dict[str, Any]], *, device: torch.device) -> tuple[torch.Tensor, list[float]]:
    logp = F.log_softmax(logits.float(), dim=-1)
    margins=[]; losses=[]
    for e in evs:
        br=int(e['batch_row']); pos=torch.tensor([int(p) for p in e['target_positions']], dtype=torch.long, device=device)
        true_ids=torch.tensor([int(x) for x in e['target_ids']], dtype=torch.long, device=device)
        neg_ids=torch.tensor([int(x) for x in e['negative_target_ids']], dtype=torch.long, device=device)
        vals=logp[br].index_select(0,pos)
        lp_true=vals.gather(1,true_ids.view(-1,1)).view(-1).mean()
        lp_neg=vals.gather(1,neg_ids.view(-1,1)).view(-1).mean()
        m=lp_true-lp_neg
        margins.append(float(m.detach().cpu().item()))
        losses.append(F.softplus(torch.tensor(0.2,device=device)-m))
    if not losses:
        return torch.tensor(0.0,device=device), margins
    return torch.stack(losses).mean(), margins


def run_model_margin(model, masked_cpu, attn_cpu, evs, device, *, micro_batch_size:int) -> list[float]:
    margins=[]
    model.eval()
    with torch.no_grad():
        # For simplicity and correctness, run the whole batch at once if memory permits; otherwise microbatch.
        all_logits=[]
        for s in range(0, int(masked_cpu.shape[0]), micro_batch_size):
            e=min(s+micro_batch_size, int(masked_cpu.shape[0]))
            out=model(input_ids=masked_cpu[s:e].to(device), attention_mask=attn_cpu[s:e].to(device))
            all_logits.append(out.logits.detach().cpu())
            del out
        logits_cpu=torch.cat(all_logits, dim=0)
        logp=F.log_softmax(logits_cpu.float(), dim=-1)
        for ev in evs:
            br=int(ev['batch_row']); pos=torch.tensor([int(p) for p in ev['target_positions']], dtype=torch.long)
            true_ids=torch.tensor([int(x) for x in ev['target_ids']], dtype=torch.long)
            neg_ids=torch.tensor([int(x) for x in ev['negative_target_ids']], dtype=torch.long)
            vals=logp[br].index_select(0,pos)
            margins.append(float((vals.gather(1,true_ids.view(-1,1)).view(-1).mean() - vals.gather(1,neg_ids.view(-1,1)).view(-1).mean()).item()))
    model.train()
    return margins


def build_args():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output_dir', default=str(DEFAULT_OUT))
    ap.add_argument('--note_path', default=str(DEFAULT_NOTE))
    ap.add_argument('--init_checkpoint', default=str(DEFAULT_CKPT))
    ap.add_argument('--checkpoint_label', default='standard80')
    ap.add_argument('--tail_jsonl', default=str(cont.DEFAULT_TAIL))
    ap.add_argument('--labels_jsonl', default=str(cont.DEFAULT_LABELS))
    ap.add_argument('--tokenizer_path', default=str(cont.DEFAULT_TOKENIZER))
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--seq_length', type=int, default=256)
    ap.add_argument('--micro_batch_size', type=int, default=8)
    ap.add_argument('--margin_batches', type=int, default=48)
    ap.add_argument('--grad_batches', type=int, default=4)
    ap.add_argument('--cap_per_cat', type=int, default=8)
    ap.add_argument('--cap_total', type=int, default=48)
    ap.add_argument('--max_control_distance_abs_diff', type=int, default=8)
    ap.add_argument('--max_control_freq_bin_abs_diff', type=int, default=2)
    ap.add_argument('--max_target_token_len', type=int, default=6)
    ap.add_argument('--seed', type=int, default=43023)
    ap.add_argument('--num_workers', type=int, default=0)
    return ap.parse_args()


def main():
    args=build_args(); os.environ.setdefault('TOKENIZERS_PARALLELISM','false'); os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
    t0=time.time(); out=Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    tail=Path(args.tail_jsonl); labels_p=Path(args.labels_jsonl); tok_path=Path(args.tokenizer_path); ckpt=Path(args.init_checkpoint)
    hashes={'tail_sha256':base.sha256_file(tail),'labels_sha256':base.sha256_file(labels_p),'tokenizer_sha256':base.sha256_file(tok_path/'tokenizer.json')}
    for k,exp in cont.EXPECTED.items():
        if hashes[k]!=exp: raise RuntimeError(f'{k} mismatch {hashes[k]} != {exp}')
    tokenizer=base.make_portable_tokenizer(str(tok_path))
    examples,label_records,segment=cont.load_segment(tail,labels_p,start_tail_row=START_TAIL_ROW,expected_start_tail_words=EXPECTED_START_WORDS,max_word_exposure=MAX_WORDS,max_rows=0)
    dataset=base.MaskedChunkDataset(examples,tokenizer,args.seq_length)
    loader=DataLoader(dataset,batch_size=args.batch_size,shuffle=False,collate_fn=base.collate,num_workers=args.num_workers)
    print(json.dumps({'event':'loaded','checkpoint_label':args.checkpoint_label,'rows':len(examples),'segment':segment}),flush=True)
    pools=build_negative_pools(examples,label_records,tokenizer,batch_size=args.batch_size,seq_length=args.seq_length,max_control_distance_abs_diff=args.max_control_distance_abs_diff,max_control_freq_bin_abs_diff=args.max_control_freq_bin_abs_diff,max_target_token_len=args.max_target_token_len,num_workers=args.num_workers)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    reset_all(args.seed)
    model=DebertaV2ForMaskedLM.from_pretrained(str(ckpt)); model.to(device); model.train()
    state=base.MaskingCurriculumState(curriculum='wwm_fixed',mask_prob_start=0.15,mask_prob_end=0.15,switch_frac=0.7); state.initialize(vocab_size=len(tokenizer), total_steps=752)
    gen=torch.Generator(device=device); gen.manual_seed(args.seed)
    coverage=Counter(); margins_by_cat=defaultdict(list); margins_all=[]; match_levels=Counter(); examples_out=[]; grad_records=[]
    batch_records=[]
    for step,batch in enumerate(loader,1):
        if step > max(args.margin_batches,args.grad_batches): break
        lo=(research)*args.batch_size; hi=lo+int(batch['input_ids'].shape[0])
        ids=batch['input_ids'][:,:args.seq_length].contiguous(); attn=batch['attention_mask'][:,:args.seq_length].contiguous(); wg=batch['word_group'][:,:args.seq_length].contiguous()
        state.current_step=250+research
        masked_dev, labels_dev=base.apply_masking_curriculum(ids.to(device),attn.to(device),wg.to(device),tokenizer,state,gen)
        masked_cpu, labels_cpu=masked_dev.cpu(), labels_dev.cpu(); del masked_dev, labels_dev
        raw=candidate_events_for_batch(ids,attn,wg,label_records[lo:hi],max_control_distance_abs_diff=args.max_control_distance_abs_diff,max_control_freq_bin_abs_diff=args.max_control_freq_bin_abs_diff,max_target_token_len=args.max_target_token_len)
        naturally=filter_naturally_masked(raw,labels_cpu)
        capped=cap_balanced(naturally,seed=args.seed,step=step,cap_per_cat=args.cap_per_cat,cap_total=args.cap_total)
        evs,negstats=attach_negative(capped,pools,seed=args.seed,step=step)
        coverage.update({'batches':1,'raw_events':len(raw),'naturally_masked_events':len(naturally),'capped_events':len(capped),'used_events':len(evs),'masked_tokens':int((labels_cpu!=-100).sum().item())})
        coverage.update({'used_'+k:v for k,v in Counter(e['category'] for e in evs).items()})
        match_levels.update(negstats.get('match_levels',{}))
        br={'step':step,'raw_events':len(raw),'naturally_masked_events':len(naturally),'capped_events':len(capped),'used_events':len(evs),'used_categories':dict(Counter(e['category'] for e in evs)),'negstats':negstats}
        if step <= args.margin_batches and evs:
            ms=run_model_margin(model,masked_cpu,attn,evs,device,micro_batch_size=args.micro_batch_size)
            for e,m in zip(evs,ms):
                margins_all.append(m); margins_by_cat[e['category']].append(m)
                if len(examples_out)<120:
                    slim={k:v for k,v in e.items() if k not in {'target_ids','negative_target_ids'} or True}
                    slim['margin_true_minus_negative']=m; examples_out.append(slim)
        if step <= args.grad_batches and evs:
            zero(model)
            # MLM gradient reference.
            total=int((labels_cpu!=-100).sum().item()); mlm_loss=0.0
            for s in range(0,int(masked_cpu.shape[0]),args.micro_batch_size):
                e=min(s+args.micro_batch_size,int(masked_cpu.shape[0])); labs=labels_cpu[s:e]
                n=int((labs!=-100).sum().item())
                if n<=0: continue
                outm=model(input_ids=masked_cpu[s:e].to(device),attention_mask=attn[s:e].to(device),labels=labs.to(device))
                loss=outm.loss*(n/total); loss.backward(); mlm_loss += float(outm.loss.detach().cpu())*(n/total); del outm,loss
            mlm_l2=grad_l2(model); ref=clone_grads(model)
            zero(model)
            outm=model(input_ids=masked_cpu.to(device),attention_mask=attn.to(device))
            aux_loss, ms_grad=logp_targets_from_logits(outm.logits,evs,device=device)
            (aux_loss*0.03).backward()
            aux_l2=grad_l2(model); cos=grad_cos(model,ref)
            grad_records.append({'step':step,'events_used':len(evs),'mlm_loss':mlm_loss,'aux_loss_raw':float(aux_loss.detach().cpu().item()),'mlm_grad_l2':mlm_l2,'scaled_aux_grad_l2':aux_l2,'ratio_to_mlm':aux_l2/mlm_l2 if mlm_l2 else None,'cosine_with_mlm':cos,'margin_mean':float(np.mean(ms_grad)) if ms_grad else None,'categories':dict(Counter(e['category'] for e in evs))})
            zero(model); del outm, aux_loss
        batch_records.append(br)
        if step==1 or step%10==0:
            print(json.dumps({'event':'batch','step':step,'raw':len(raw),'naturally':len(naturally),'used':len(evs),'used_categories':br['used_categories']}),flush=True)
        del ids,attn,wg,masked_cpu,labels_cpu
    summary={'status':'MASKED_ALT_CONSEQUENCE_CALIBRATION','created_utc':now_utc(),'checkpoint_label':args.checkpoint_label,'init_checkpoint':str(ckpt),'runtime_sec':round(time.time()-t0,2),'device':str(device),'hashes':hashes,'segment':segment,'args':vars(args),'coverage':dict(coverage),'match_levels':dict(match_levels),'margin_summary':{'ALL':summarize(margins_all),**{k:summarize(v) for k,v in sorted(margins_by_cat.items())}},'gradient_records':grad_records,'gradient_summary':{'ratio_to_mlm':summarize([float(r['ratio_to_mlm']) for r in grad_records if r.get('ratio_to_mlm') is not None]),'cosine_with_mlm':summarize([float(r['cosine_with_mlm']) for r in grad_records if r.get('cosine_with_mlm') is not None])},'artifacts':{'summary':str(out/'masked_alt_consequence_calibration_summary.json'),'batch_records':str(out/'batch_records.jsonl'),'sample_events':str(out/'sample_events.jsonl'),'gradient_records':str(out/'gradient_records.jsonl'),'note':str(Path(args.note_path))}}
    (out/'masked_alt_consequence_calibration_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    with (out/'batch_records.jsonl').open('w',encoding='utf-8') as f:
        for r in batch_records: f.write(json.dumps(r,ensure_ascii=False)+'\n')
    with (out/'sample_events.jsonl').open('w',encoding='utf-8') as f:
        for r in examples_out: f.write(json.dumps(r,ensure_ascii=False)+'\n')
    with (out/'gradient_records.jsonl').open('w',encoding='utf-8') as f:
        for r in grad_records: f.write(json.dumps(r,ensure_ascii=False)+'\n')
    lines=['# research — masked alternative consequence calibration\n',f"Checkpoint: {args.checkpoint_label} `{ckpt}` runtime={summary['runtime_sec']} device={device}\n",'## Coverage\n',f"{dict(coverage)}\n",f"Negative match levels: {dict(match_levels)}\n",'## Margin true target vs matched alternative\n',json.dumps(summary['margin_summary'],indent=2,ensure_ascii=False)+'\n','## Gradient\n',json.dumps(summary['gradient_summary'],indent=2,ensure_ascii=False)+'\n',f"Files: `{out/'masked_alt_consequence_calibration_summary.json'}`\n"]
    Path(args.note_path).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':summary['status'],'summary':summary['artifacts']['summary'],'note':summary['artifacts']['note'],'coverage':summary['coverage'],'margin_all':summary['margin_summary']['ALL'],'grad_ratio':summary['gradient_summary']['ratio_to_mlm'],'runtime_sec':summary['runtime_sec']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
