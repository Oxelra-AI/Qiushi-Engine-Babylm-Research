#!/usr/bin/env python3
"""No-update gradient preflight for full-context pivot-substitution auxiliary.

Measures, on the legal 80M->90M segment and a fixed 80M checkpoint, whether the
masked-consequence substitution losses produce a non-negligible and differentiated
backbone gradient relative to ordinary WWM.  No optimizer step is performed.
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

DEFAULT_OUT = Path('experiments/archive/representation_and_objectives/data/fullctx_gradient_preflight')
DEFAULT_NOTE = Path('research/notes/representation_and_objectives/fullctx_gradient_preflight.md')
DEFAULT_CKPT = Path('experiments/archive/representation_and_objectives/training/runs/standard_legacy_70M_to_80M_seed43022/hf_model/chck_80M')
START_TAIL_ROW = 64255
EXPECTED_START_WORDS = 10011326
MAX_ROWS = 64000
MAX_WORDS = 9971289
CATEGORIES = {'physical_change','causal_connector','temporal','spatial','comparative','negation'}


def now_utc() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def reset_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed % (2**32 - 1)); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def grad_blocks(model: torch.nn.Module) -> dict[str, dict[str, float]]:
    blocks = defaultdict(lambda: {'sq': 0.0, 'n_params': 0, 'n_elems': 0})
    for name, p in model.named_parameters():
        if p.grad is None: continue
        g = p.grad.detach().float()
        sq = float((g*g).sum().item())
        if name.startswith('deberta.embeddings'): block = 'embeddings'
        elif name.startswith('deberta.encoder'): block = 'encoder'
        elif name.startswith('cls'): block = 'mlm_head'
        else: block = 'other'
        for key in ['model_all', block]:
            blocks[key]['sq'] += sq; blocks[key]['n_params'] += 1; blocks[key]['n_elems'] += int(g.numel())
    out = {}
    for key in ['model_all','embeddings','encoder','mlm_head','other']:
        v = blocks[key]
        out[key] = {'l2': math.sqrt(float(v['sq'])), 'n_params_with_grad': int(v['n_params']), 'n_elems_with_grad': int(v['n_elems'])}
    return out


def save_current_grads(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None}


def grad_dot_cos(model: torch.nn.Module, ref: dict[str, torch.Tensor]) -> dict[str, float]:
    dot = 0.0; sq = 0.0; rsq = 0.0; n = 0
    for name, p in model.named_parameters():
        if p.grad is None or name not in ref: continue
        g = p.grad.detach().float(); r = ref[name].detach().float().to(g.device)
        dot += float((g*r).sum().item()); sq += float((g*g).sum().item()); rsq += float((r*r).sum().item()); n += 1
    cos = dot / max(1e-30, math.sqrt(sq) * math.sqrt(rsq)) if sq > 0 and rsq > 0 else None
    return {'dot_with_mlm': dot, 'cosine_with_mlm': cos, 'shared_grad_params': n}


def zero(model: torch.nn.Module) -> None:
    model.zero_grad(set_to_none=True)


def candidate_events_for_batch(input_ids: torch.Tensor, attention: torch.Tensor, word_group: torch.Tensor, label_records: list[dict[str, Any]], *, seed: int, step: int, max_control_distance_abs_diff: int, max_control_freq_bin_abs_diff: int, max_target_token_len: int) -> list[dict[str, Any]]:
    all_events = []
    for b, lr in enumerate(label_records):
        group_pos = pvdm.group_positions_for_row(word_group[b], attention[b])
        usable, _rej = pvdm.collect_active_events(lr, group_pos, same_length_required=True)
        for ev in usable:
            cat = str(ev.category)
            if cat not in CATEGORIES: continue
            if len({ev.pivot_gid, ev.target_gid, ev.control_gid}) < 3: continue
            ev_raw = cueprobe.raw_event(lr, ev.event_rank)
            meta = relaux._event_meta(ev_raw)
            if int(meta.get('control_distance_abs_diff', 999)) > max_control_distance_abs_diff: continue
            if int(meta.get('control_freq_bin_abs_diff', 999)) > max_control_freq_bin_abs_diff: continue
            if len(group_pos[int(ev.target_gid)]) > max_target_token_len: continue
            rec = {
                'batch_row': int(b), 'tail_row_idx': int(lr.get('tail_row_idx', -1)), 'event_rank': int(ev.event_rank),
                'category': cat, 'pivot_gid': int(ev.pivot_gid), 'target_gid': int(ev.target_gid), 'control_gid': int(ev.control_gid),
                'pivot_positions': [int(x) for x in group_pos[int(ev.pivot_gid)]],
                'target_positions': [int(x) for x in group_pos[int(ev.target_gid)]],
                'control_positions': [int(x) for x in group_pos[int(ev.control_gid)]],
                'pivot_ids': [int(input_ids[b, x].item()) for x in group_pos[int(ev.pivot_gid)]],
                'control_ids': [int(input_ids[b, x].item()) for x in group_pos[int(ev.control_gid)]],
                'target_ids': [int(input_ids[b, x].item()) for x in group_pos[int(ev.target_gid)]],
                'pivot_len': int(ev.pivot_len), 'control_len': int(ev.control_len), 'target_len': int(ev.target_len),
                **meta,
            }
            all_events.append(rec)
    # Cap to a trainable balanced set: up to 8 per family, 48 per effective batch.
    chosen = []
    for cat in sorted(CATEGORIES):
        sub = [e for e in all_events if e['category'] == cat]
        sub.sort(key=lambda e: pvdm.hash_uniform(seed, 'grad_select', step, e['tail_row_idx'], e['event_rank'], cat, e.get('target_norm')))
        chosen.extend(sub[:8])
    chosen.sort(key=lambda e: pvdm.hash_uniform(seed, 'grad_order', step, e['tail_row_idx'], e['event_rank'], e['category']))
    return chosen[:48]


def donor_key(ev: dict[str, Any], level: int) -> tuple[Any, ...]:
    if level == 0: return (ev.get('category'), ev.get('target_class'), ev.get('distance_bin'), ev.get('pivot_len'))
    if level == 1: return (ev.get('category'), ev.get('target_class'), ev.get('pivot_len'))
    if level == 2: return (ev.get('category'), ev.get('pivot_len'))
    return (ev.get('category'),)


def build_donor_pools(examples, labels, tokenizer, *, batch_size: int, seq_length: int, max_control_distance_abs_diff: int, max_control_freq_bin_abs_diff: int, max_target_token_len: int, num_workers: int) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    dataset = base.MaskedChunkDataset(examples, tokenizer, seq_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=base.collate, num_workers=num_workers)
    pools: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    total = 0
    for step, batch in enumerate(loader, 1):
        lo = (research)*batch_size; hi = lo + int(batch['input_ids'].shape[0])
        ids = batch['input_ids'][:, :seq_length].contiguous(); attn = batch['attention_mask'][:, :seq_length].contiguous(); wg = batch['word_group'][:, :seq_length].contiguous()
        evs = candidate_events_for_batch(ids, attn, wg, labels[lo:hi], seed=98765, step=step, max_control_distance_abs_diff=max_control_distance_abs_diff, max_control_freq_bin_abs_diff=max_control_freq_bin_abs_diff, max_target_token_len=max_target_token_len)
        # candidate_events_for_batch is capped; for donor pool use enough but not exhaustive. Also add uncapped not necessary for first batches.
        for e in evs:
            donor = {'tail_row_idx': e['tail_row_idx'], 'event_rank': e['event_rank'], 'category': e['category'], 'pivot_norm': e.get('pivot_norm'), 'target_class': e.get('target_class'), 'distance_bin': e.get('distance_bin'), 'pivot_len': e.get('pivot_len'), 'pivot_ids': e['pivot_ids']}
            for level in range(4): pools[donor_key(donor, level)].append(donor)
            total += 1
        if step == 1 or step % 50 == 0:
            print(json.dumps({'event': 'donor_pool_progress', 'step': step, 'donors_added': total, 'keys': len(pools)}), flush=True)
    return pools


def attach_shuffled(evs: list[dict[str, Any]], pools: dict[tuple[Any, ...], list[dict[str, Any]]], *, seed: int, step: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out = []; levels = Counter(); drops = Counter()
    for ev in evs:
        donor = None; used_level = None
        for level in range(4):
            pool = pools.get(donor_key(ev, level), [])
            if not pool: continue
            order = sorted(range(len(pool)), key=lambda i: pvdm.hash_uniform(seed, 'grad_donor', step, level, ev['tail_row_idx'], ev['event_rank'], i))
            for i in order[:128]:
                cand = pool[i]
                if int(cand['tail_row_idx']) == int(ev['tail_row_idx']): continue
                if str(cand.get('pivot_norm')) == str(ev.get('pivot_norm')): continue
                if int(cand.get('pivot_len', -1)) != int(ev.get('pivot_len', -2)): continue
                donor = cand; used_level = level; break
            if donor is not None: break
        if donor is None:
            drops[str(ev.get('category'))] += 1; continue
        e2 = dict(ev); e2['shuffle_pivot_ids'] = [int(x) for x in donor['pivot_ids']]; e2['shuffle_pivot_norm'] = str(donor.get('pivot_norm')); e2['shuffle_level'] = int(used_level)
        out.append(e2); levels[str(used_level)] += 1
    return out, {'input_events': len(evs), 'used_events': len(out), 'drop_by_category': dict(drops), 'shuffle_level_counts': dict(levels), 'used_categories': dict(Counter(e['category'] for e in out))}


def make_views_for_events(input_ids: torch.Tensor, attention: torch.Tensor, evs: list[dict[str, Any]], *, mask_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[int, str]]]:
    kinds = ['true', 'anchor', 'shuffle', 'pivotmasked']
    ids_list=[]; attn_list=[]; labels_list=[]; refs=[]
    seq_length = int(input_ids.shape[1])
    for i, ev in enumerate(evs):
        for kind in kinds:
            ids = input_ids[int(ev['batch_row'])].clone(); att = attention[int(ev['batch_row'])].clone(); lab = torch.full_like(ids, -100)
            for p, tid in zip(ev['target_positions'], ev['target_ids']):
                ids[int(p)] = int(mask_id); lab[int(p)] = int(tid)
            if kind == 'anchor':
                for p, tid in zip(ev['pivot_positions'], ev['control_ids']): ids[int(p)] = int(tid)
            elif kind == 'shuffle':
                for p, tid in zip(ev['pivot_positions'], ev['shuffle_pivot_ids']): ids[int(p)] = int(tid)
            elif kind == 'pivotmasked':
                for p in ev['pivot_positions']: ids[int(p)] = int(mask_id)
            ids_list.append(ids); attn_list.append(att); labels_list.append(lab); refs.append((i, kind))
    return torch.stack(ids_list), torch.stack(attn_list), torch.stack(labels_list), refs


def event_logps(model, view_ids, view_attn, view_labels, refs, device, view_batch_size: int) -> dict[tuple[int, str], torch.Tensor]:
    vals: dict[tuple[int, str], torch.Tensor] = {}
    for start in range(0, int(view_ids.shape[0]), view_batch_size):
        end = min(start+view_batch_size, int(view_ids.shape[0]))
        ids = view_ids[start:end].to(device, non_blocking=True); attn = view_attn[start:end].to(device, non_blocking=True); labs = view_labels[start:end].to(device, non_blocking=True)
        out = model(input_ids=ids, attention_mask=attn)
        logp = F.log_softmax(out.logits.float(), dim=-1)
        for row in range(end-start):
            pos = (labs[row] != -100).nonzero(as_tuple=False).view(-1)
            target = labs[row].index_select(0, pos)
            tok = logp[row].index_select(0, pos).gather(1, target.view(-1,1)).view(-1)
            vals[refs[start+row]] = tok.mean()
        del ids, attn, labs, out, logp
    return vals


def margin_loss(logps: dict[tuple[int, str], torch.Tensor], n: int, positive: str, negative: str, margin: float) -> torch.Tensor:
    losses = []
    for i in range(n):
        losses.append(F.softplus(torch.tensor(float(margin), device=logps[(i, positive)].device) - (logps[(i, positive)] - logps[(i, negative)])))
    return torch.stack(losses).mean()


def true_nll_loss(logps: dict[tuple[int, str], torch.Tensor], n: int) -> torch.Tensor:
    return torch.stack([-logps[(i, 'true')] for i in range(n)]).mean()


def accumulate_mlm_grad(model, batch, tokenizer, legacy_state, legacy_gen, device, *, micro_batch_size: int, train_seed: int, step: int) -> dict[str, Any]:
    ids_cpu = batch['input_ids'].contiguous(); attn_cpu = batch['attention_mask'].contiguous(); wg_cpu = batch['word_group'].contiguous()
    legacy_state.current_step = 250 + step - 1
    masked_dev, labels_dev = base.apply_masking_curriculum(ids_cpu.to(device), attn_cpu.to(device), wg_cpu.to(device), tokenizer, legacy_state, legacy_gen)
    masked_cpu, labels_cpu = masked_dev.cpu(), labels_dev.cpu(); del masked_dev, labels_dev
    total = int((labels_cpu != -100).sum().item())
    zero(model); loss_sum = 0.0; active=0
    for mb_start in range(0, int(ids_cpu.shape[0]), micro_batch_size):
        mb_end = min(mb_start+micro_batch_size, int(ids_cpu.shape[0]))
        labs = labels_cpu[mb_start:mb_end]
        n_pred = int((labs != -100).sum().item())
        if n_pred <= 0: continue
        reset_all(train_seed + 900000 + step*1000 + mb_start)
        out = model(input_ids=masked_cpu[mb_start:mb_end].to(device), attention_mask=attn_cpu[mb_start:mb_end].to(device), labels=labs.to(device))
        loss = out.loss * (n_pred / total)
        loss.backward(); loss_sum += float(out.loss.detach().cpu()) * (n_pred / total); active += 1
        del out, loss
    norms = grad_blocks(model); ref = save_current_grads(model)
    zero(model)
    return {'loss': loss_sum, 'masked_tokens': total, 'active_microbatches': active, 'grad_norms': norms, 'grads_ref': ref}


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output_dir', default=str(DEFAULT_OUT))
    ap.add_argument('--note_path', default=str(DEFAULT_NOTE))
    ap.add_argument('--init_checkpoint', default=str(DEFAULT_CKPT))
    ap.add_argument('--tail_jsonl', default=str(cont.DEFAULT_TAIL))
    ap.add_argument('--labels_jsonl', default=str(cont.DEFAULT_LABELS))
    ap.add_argument('--tokenizer_path', default=str(cont.DEFAULT_TOKENIZER))
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--micro_batch_size', type=int, default=8)
    ap.add_argument('--seq_length', type=int, default=256)
    ap.add_argument('--grad_batches', type=int, default=4)
    ap.add_argument('--view_batch_size', type=int, default=96)
    ap.add_argument('--margin', type=float, default=0.2)
    ap.add_argument('--aux_weight', type=float, default=0.03)
    ap.add_argument('--max_control_distance_abs_diff', type=int, default=8)
    ap.add_argument('--max_control_freq_bin_abs_diff', type=int, default=2)
    ap.add_argument('--max_target_token_len', type=int, default=6)
    ap.add_argument('--seed', type=int, default=43)
    ap.add_argument('--train_rng_seed', type=int, default=43023)
    ap.add_argument('--num_workers', type=int, default=0)
    return ap.parse_args()


def main():
    args = build_args(); os.environ.setdefault('TOKENIZERS_PARALLELISM','false'); os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
    t0=time.time(); outdir=Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True); note_path=Path(args.note_path); note_path.parent.mkdir(parents=True, exist_ok=True)
    tail=Path(args.tail_jsonl); labels_p=Path(args.labels_jsonl); tok=Path(args.tokenizer_path); ckpt=Path(args.init_checkpoint)
    hashes={'tail_sha256': base.sha256_file(tail), 'labels_sha256': base.sha256_file(labels_p), 'tokenizer_sha256': base.sha256_file(tok/'tokenizer.json')}
    for k, exp in cont.EXPECTED.items():
        if hashes[k] != exp: raise RuntimeError(f'{k} mismatch {hashes[k]} != {exp}')
    tokenizer=base.make_portable_tokenizer(str(tok))
    examples, labels, segment = cont.load_segment(tail, labels_p, start_tail_row=START_TAIL_ROW, expected_start_tail_words=EXPECTED_START_WORDS, max_word_exposure=MAX_WORDS, max_rows=MAX_ROWS)
    print(json.dumps({'event':'loaded_segment','examples':len(examples),'segment':segment}), flush=True)
    pools = build_donor_pools(examples, labels, tokenizer, batch_size=args.batch_size, seq_length=args.seq_length, max_control_distance_abs_diff=args.max_control_distance_abs_diff, max_control_freq_bin_abs_diff=args.max_control_freq_bin_abs_diff, max_target_token_len=args.max_target_token_len, num_workers=args.num_workers)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    reset_all(args.seed); model=DebertaV2ForMaskedLM.from_pretrained(str(ckpt)); model.to(device); model.train()
    if model.config.vocab_size != len(tokenizer): raise RuntimeError('vocab mismatch')
    legacy_state=base.MaskingCurriculumState(curriculum='wwm_fixed', mask_prob_start=0.15, mask_prob_end=0.15, switch_frac=0.7); legacy_state.initialize(vocab_size=len(tokenizer), total_steps=752)
    legacy_gen=torch.Generator(device=device); legacy_gen.manual_seed(args.train_rng_seed)
    dataset=base.MaskedChunkDataset(examples, tokenizer, args.seq_length)
    loader=DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=base.collate, num_workers=args.num_workers)
    records=[]; event_counts=Counter(); loss_names=['true_vs_shuffle','true_vs_anchor','true_vs_pivotmasked','placebo_shuffle_vs_anchor','true_nll']
    rec_path=outdir/'gradient_records.jsonl'
    with rec_path.open('w', encoding='utf-8') as rf:
        for step, batch in enumerate(loader, 1):
            if step > args.grad_batches: break
            lo=(research)*args.batch_size; hi=lo+int(batch['input_ids'].shape[0])
            ids=batch['input_ids'][:,:args.seq_length].contiguous(); attn=batch['attention_mask'][:,:args.seq_length].contiguous(); wg=batch['word_group'][:,:args.seq_length].contiguous()
            evs=candidate_events_for_batch(ids, attn, wg, labels[lo:hi], seed=args.train_rng_seed, step=step, max_control_distance_abs_diff=args.max_control_distance_abs_diff, max_control_freq_bin_abs_diff=args.max_control_freq_bin_abs_diff, max_target_token_len=args.max_target_token_len)
            evs, shuf_stats = attach_shuffled(evs, pools, seed=args.train_rng_seed, step=step)
            event_counts.update(e['category'] for e in evs)
            mlm=accumulate_mlm_grad(model, {'input_ids':ids,'attention_mask':attn,'word_group':wg}, tokenizer, legacy_state, legacy_gen, device, micro_batch_size=args.micro_batch_size, train_seed=args.train_rng_seed, step=step)
            mlm_ref=mlm.pop('grads_ref')
            mlm_model_l2=float(mlm['grad_norms']['model_all']['l2']); mlm_encoder_l2=float(mlm['grad_norms']['encoder']['l2'])
            rf.write(json.dumps({'step':step,'key':'mlm',**mlm}, ensure_ascii=False)+'\n'); rf.flush()
            records.append({'step':step,'key':'mlm',**mlm})
            if not evs:
                continue
            view_ids, view_attn, view_labels, refs=make_views_for_events(ids, attn, evs, mask_id=int(tokenizer.mask_token_id))
            # For each candidate loss, recompute forward and backward independently for exact gradients.
            for key in loss_names:
                zero(model); reset_all(args.train_rng_seed + 950000 + step*100 + loss_names.index(key))
                logps = event_logps(model, view_ids, view_attn, view_labels, refs, device, args.view_batch_size)
                if key == 'true_vs_shuffle': raw_loss = margin_loss(logps, len(evs), 'true', 'shuffle', args.margin)
                elif key == 'true_vs_anchor': raw_loss = margin_loss(logps, len(evs), 'true', 'anchor', args.margin)
                elif key == 'true_vs_pivotmasked': raw_loss = margin_loss(logps, len(evs), 'true', 'pivotmasked', args.margin)
                elif key == 'placebo_shuffle_vs_anchor': raw_loss = margin_loss(logps, len(evs), 'shuffle', 'anchor', args.margin)
                elif key == 'true_nll': raw_loss = true_nll_loss(logps, len(evs))
                else: raise ValueError(key)
                scaled_loss = raw_loss * float(args.aux_weight)
                scaled_loss.backward()
                norms=grad_blocks(model); dotcos=grad_dot_cos(model, mlm_ref)
                rec={'step':step,'key':key,'events_used':len(evs),'event_categories':dict(Counter(e['category'] for e in evs)),'shuffle_stats':shuf_stats,'raw_loss':float(raw_loss.detach().cpu()),'scaled_loss':float(scaled_loss.detach().cpu()),'grad_norms':norms,'ratio_to_mlm_model_l2': norms['model_all']['l2']/mlm_model_l2 if mlm_model_l2 else None,'ratio_to_mlm_encoder_l2': norms['encoder']['l2']/mlm_encoder_l2 if mlm_encoder_l2 else None,**dotcos}
                rf.write(json.dumps(rec, ensure_ascii=False)+'\n'); rf.flush(); records.append(rec); zero(model)
                del logps, raw_loss, scaled_loss
            del ids, attn, wg, view_ids, view_attn, view_labels
            print(json.dumps({'event':'grad_batch_done','step':step,'events':len(evs),'event_categories':dict(Counter(e['category'] for e in evs)),'elapsed_sec':round(time.time()-t0,1)}), flush=True)
    # summarize
    by=defaultdict(lambda: defaultdict(list)); cats=Counter()
    for r in records:
        key=r['key']; cats.update(r.get('event_categories', {})) if isinstance(r.get('event_categories'), dict) else None
        if key=='mlm':
            by[key]['model_l2'].append(float(r['grad_norms']['model_all']['l2'])); by[key]['encoder_l2'].append(float(r['grad_norms']['encoder']['l2'])); by[key]['loss'].append(float(r['loss']))
        else:
            by[key]['model_l2'].append(float(r['grad_norms']['model_all']['l2'])); by[key]['encoder_l2'].append(float(r['grad_norms']['encoder']['l2'])); by[key]['ratio_model'].append(float(r['ratio_to_mlm_model_l2'])); by[key]['ratio_encoder'].append(float(r['ratio_to_mlm_encoder_l2'])); by[key]['cosine_with_mlm'].append(float(r['cosine_with_mlm'])); by[key]['raw_loss'].append(float(r['raw_loss'])); by[key]['events_used'].append(float(r['events_used']))
    def stat(xs):
        if not xs: return {'n':0,'mean':None,'median':None,'min':None,'max':None}
        arr=np.asarray(xs,dtype=float); return {'n':int(arr.size),'mean':float(arr.mean()),'median':float(np.median(arr)),'min':float(arr.min()),'max':float(arr.max())}
    summary={'status':'FULLCTX_GRADIENT_PREFLIGHT','created_utc':now_utc(),'runtime_sec':round(time.time()-t0,2),'device':str(device),'purpose':'no-update gradient preflight for full-context masked-consequence pivot substitution objective','checkpoint':str(ckpt),'hashes':hashes,'segment':segment,'args':vars(args),'summary_by_key':{k:{m:stat(v) for m,v in d.items()} for k,d in by.items()},'aggregate_event_categories':dict(event_counts),'artifacts':{'summary':str(outdir/'fullctx_gradient_preflight_summary.json'),'records':str(rec_path),'note':str(note_path)}}
    (outdir/'fullctx_gradient_preflight_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    lines=['# research — full-context gradient preflight\n',f"Created: {summary['created_utc']} runtime={summary['runtime_sec']} device={device}\n",'## Gradient summaries\n']
    for key, item in summary['summary_by_key'].items(): lines.append(f"- {key}: {item}\n")
    lines.append(f"Aggregate event categories: {dict(event_counts)}\n")
    lines.append(f"Files: `{summary['artifacts']['summary']}`, `{summary['artifacts']['records']}`\n")
    note_path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':summary['status'],'summary':summary['artifacts']['summary'],'note':str(note_path),'by_key':summary['summary_by_key'],'event_categories':dict(event_counts),'runtime_sec':summary['runtime_sec']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
