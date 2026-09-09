#!/usr/bin/env python3
"""research full-context pivot-substitution auxiliary trainer.

Minimum 80M->90M continuation test derived from research probes.  The ordinary
WWM pass is left intact.  A transient auxiliary duplicate masks the same legal
research dependent target and keeps all non-target context visible; it compares
masked-target likelihood under true pivot, same-category shuffled pivot, and
matched same-row anchor substitutions at the original pivot positions.

Modes:
  semantic_true_vs_shuffle:  softplus(m - (logp_true - logp_shuffle))
  placebo_shuffle_vs_anchor: softplus(m - (logp_shuffle - logp_anchor))
  true_vs_anchor:            auxiliary/debug only, confounded by anchor syntax

No final endpoint should be inferred from this trainer alone; fixed readouts must
compare any branch with a matched standard WWM continuation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import DebertaV2ForMaskedLM, get_cosine_schedule_with_warmup

USER_ROOT = Path('.').resolve()
for p in [USER_ROOT/'experiments/archive/compact_experience/scripts', USER_ROOT/'experiments/archive/representation_and_objectives/scripts']:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import masking_curriculum_trainer as base  # noqa: E402
import pvdm_continuation_trainer as cont  # noqa: E402
import fullctx_gradient_preflight as gp  # noqa: E402

DEFAULT_INIT = Path('experiments/archive/representation_and_objectives/training/runs/standard_legacy_70M_to_80M_seed43022/hf_model/chck_80M')
DEFAULT_START_TAIL_ROW = 64255
DEFAULT_EXPECTED_START_WORDS = 10011326
DEFAULT_STAGE_WORDS = 9971308
DEFAULT_INITIAL_EXPOSURE = 80011326
DEFAULT_TOTAL_SCHEDULE_STEPS = 752


def now_utc() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def reset_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed % (2**32 - 1)); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def convert(obj: Any) -> Any:
    if isinstance(obj, Counter): return dict(obj)
    if isinstance(obj, dict): return {str(k): convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [convert(x) for x in obj]
    return obj


def compute_aux_loss(model: torch.nn.Module, input_ids_cpu: torch.Tensor, attention_cpu: torch.Tensor, evs: list[dict[str, Any]], tokenizer: Any, device: torch.device, *, mode: str, margin: float, view_batch_size: int) -> tuple[torch.Tensor | None, dict[str, Any]]:
    if not evs:
        return None, {'events_used': 0}
    view_ids, view_attn, view_labels, refs = gp.make_views_for_events(input_ids_cpu, attention_cpu, evs, mask_id=int(tokenizer.mask_token_id))
    logps = gp.event_logps(model, view_ids, view_attn, view_labels, refs, device, view_batch_size)
    n = len(evs)
    if mode == 'semantic_true_vs_shuffle':
        loss = gp.margin_loss(logps, n, 'true', 'shuffle', margin)
        pos, neg = 'true', 'shuffle'
    elif mode == 'placebo_shuffle_vs_anchor':
        loss = gp.margin_loss(logps, n, 'shuffle', 'anchor', margin)
        pos, neg = 'shuffle', 'anchor'
    elif mode == 'true_vs_anchor':
        loss = gp.margin_loss(logps, n, 'true', 'anchor', margin)
        pos, neg = 'true', 'anchor'
    elif mode == 'true_vs_pivotmasked':
        loss = gp.margin_loss(logps, n, 'true', 'pivotmasked', margin)
        pos, neg = 'true', 'pivotmasked'
    else:
        raise ValueError(mode)
    with torch.no_grad():
        deltas = []
        for i in range(n):
            deltas.append(float((logps[(i, pos)] - logps[(i, neg)]).detach().cpu().item()))
        cats = Counter(str(e['category']) for e in evs)
        levels = Counter(str(e.get('shuffle_level', 'NA')) for e in evs)
    stats = {
        'events_used': int(n),
        'event_categories': dict(cats),
        'shuffle_levels': dict(levels),
        'mode': mode,
        'positive_view': pos,
        'negative_view': neg,
        'raw_aux_loss': float(loss.detach().cpu().item()),
        'margin_delta_mean': float(np.mean(deltas)) if deltas else None,
        'margin_delta_success_rate': float(np.mean([x > 0 for x in deltas])) if deltas else None,
    }
    del view_ids, view_attn, view_labels, logps
    return loss, stats


def build_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='80M->90M full-context pivot-substitution auxiliary trainer')
    ap.add_argument('--mode', choices=['semantic_true_vs_shuffle','placebo_shuffle_vs_anchor','true_vs_anchor','true_vs_pivotmasked'], required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--init_checkpoint', default=str(DEFAULT_INIT))
    ap.add_argument('--tail_jsonl', default=str(cont.DEFAULT_TAIL))
    ap.add_argument('--labels_jsonl', default=str(cont.DEFAULT_LABELS))
    ap.add_argument('--tokenizer_path', default=str(cont.DEFAULT_TOKENIZER))
    ap.add_argument('--start_tail_row', type=int, default=DEFAULT_START_TAIL_ROW)
    ap.add_argument('--expected_start_tail_words', type=int, default=DEFAULT_EXPECTED_START_WORDS)
    ap.add_argument('--initial_actual_word_exposure', type=int, default=DEFAULT_INITIAL_EXPOSURE)
    ap.add_argument('--max_word_exposure', type=int, default=DEFAULT_STAGE_WORDS)
    ap.add_argument('--max_rows', type=int, default=0)
    ap.add_argument('--checkpoint_name', default='chck_90M')
    ap.add_argument('--total_schedule_steps', type=int, default=DEFAULT_TOTAL_SCHEDULE_STEPS)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--micro_batch_size', type=int, default=8)
    ap.add_argument('--view_batch_size', type=int, default=96)
    ap.add_argument('--seq_length', type=int, default=256)
    ap.add_argument('--learning_rate', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=0.01)
    ap.add_argument('--warmup_fraction', type=float, default=0.06)
    ap.add_argument('--mask_prob', type=float, default=0.15)
    ap.add_argument('--aux_weight', type=float, default=0.03)
    ap.add_argument('--aux_margin', type=float, default=0.2)
    ap.add_argument('--max_control_distance_abs_diff', type=int, default=8)
    ap.add_argument('--max_control_freq_bin_abs_diff', type=int, default=2)
    ap.add_argument('--max_target_token_len', type=int, default=6)
    ap.add_argument('--seed', type=int, default=43)
    ap.add_argument('--train_rng_seed', type=int, default=43023)
    ap.add_argument('--num_workers', type=int, default=0)
    ap.add_argument('--log_every', type=int, default=25)
    ap.add_argument('--save_trainer_state', action='store_true')
    return ap.parse_args()


def main() -> None:
    args = build_args()
    if args.batch_size % args.micro_batch_size != 0:
        raise ValueError('batch_size must be divisible by micro_batch_size')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    t0 = time.time(); out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    tail = Path(args.tail_jsonl); labels_p = Path(args.labels_jsonl); tok_path = Path(args.tokenizer_path); init_ckpt = Path(args.init_checkpoint)
    hashes = {'tail_sha256': base.sha256_file(tail), 'labels_sha256': base.sha256_file(labels_p), 'tokenizer_sha256': base.sha256_file(tok_path/'tokenizer.json')}
    for k, exp in cont.EXPECTED.items():
        if hashes[k] != exp: raise RuntimeError(f'{k} mismatch {hashes[k]} != {exp}')
    if not init_ckpt.exists(): raise RuntimeError(f'init checkpoint not found: {init_ckpt}')
    tokenizer = base.make_portable_tokenizer(str(tok_path))
    examples, label_records, segment = cont.load_segment(tail, labels_p, start_tail_row=args.start_tail_row, expected_start_tail_words=args.expected_start_tail_words, max_word_exposure=args.max_word_exposure, max_rows=args.max_rows)
    dataset = base.MaskedChunkDataset(examples, tokenizer, args.seq_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=base.collate, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    stage_steps = math.ceil(len(dataset) / args.batch_size)
    schedule_total = max(args.total_schedule_steps, stage_steps)
    warmup = max(1, int(schedule_total * args.warmup_fraction))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(json.dumps({'event':'building_donor_pools','rows':len(examples),'device':str(device)}), flush=True)
    donor_pools = gp.build_donor_pools(examples, label_records, tokenizer, batch_size=args.batch_size, seq_length=args.seq_length, max_control_distance_abs_diff=args.max_control_distance_abs_diff, max_control_freq_bin_abs_diff=args.max_control_freq_bin_abs_diff, max_target_token_len=args.max_target_token_len, num_workers=args.num_workers)
    reset_all(args.seed)
    model = DebertaV2ForMaskedLM.from_pretrained(str(init_ckpt))
    if model.config.vocab_size != len(tokenizer): raise RuntimeError(f'vocab mismatch {model.config.vocab_size} != {len(tokenizer)}')
    model.to(device); model.train()
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats()
    param_count = sum(p.numel() for p in model.parameters())
    reset_all(args.train_rng_seed)
    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=warmup, num_training_steps=schedule_total)
    legacy_state = base.MaskingCurriculumState(curriculum='wwm_fixed', mask_prob_start=args.mask_prob, mask_prob_end=args.mask_prob, switch_frac=0.7)
    legacy_state.initialize(vocab_size=len(tokenizer), total_steps=schedule_total)
    legacy_gen = torch.Generator(device=device); legacy_gen.manual_seed(args.train_rng_seed)

    source_words: dict[str, int] = {}
    for ex in examples: source_words[ex.source] = source_words.get(ex.source, 0) + ex.words
    manifest = {'status':'FULLCTX_AUX_TRAINER_MANIFEST','created_utc':now_utc(),'mode':args.mode,'output_dir':str(out),'init_checkpoint':str(init_ckpt),'initial_actual_word_exposure':args.initial_actual_word_exposure,'target_stage_stop':args.checkpoint_name,'segment':segment,'hashes':hashes,'param_count':param_count,'batch_size':args.batch_size,'micro_batch_size':args.micro_batch_size,'view_batch_size':args.view_batch_size,'stage_steps':stage_steps,'schedule_total':schedule_total,'optimizer_reset':'AdamW reset symmetrically for all research 80M->90M branches','learning_rate':args.learning_rate,'warmup_fraction':args.warmup_fraction,'weight_decay':args.weight_decay,'mask_prob':args.mask_prob,'aux_weight':args.aux_weight,'aux_margin':args.aux_margin,'source_words_consumed':source_words,'probe_evidence':['experiments/archive/representation_and_objectives/data/full_context_pivot_substitution_probe/full_context_pivot_substitution_summary.json','experiments/archive/representation_and_objectives/data/fullctx_gradient_preflight/fullctx_gradient_preflight_summary.json']}
    (out/'example_order_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps({'event':'fullctx_aux_start','mode':args.mode,'output_dir':str(out),'device':str(device),'param_count':param_count,'stage_rows':len(examples),'stage_words':segment['selected_words'],'stage_steps':stage_steps,'expected_total_at_stage_end':args.initial_actual_word_exposure + segment['selected_words']}), flush=True)

    log_path = out/'training_log.jsonl'; aux_path = out/'aux_stats.jsonl'
    cumulative_words = 0; loss_values=[]; mlm_values=[]; aux_values=[]; event_counts=Counter(); shuffle_level_counts=Counter(); total_aux_events=0
    with log_path.open('w', encoding='utf-8') as logf, aux_path.open('w', encoding='utf-8') as auxf:
        for step, batch in enumerate(loader, 1):
            lo=(research)*args.batch_size; hi=lo+int(batch['input_ids'].shape[0])
            words=int(batch.pop('words').sum().item())
            input_ids_cpu=batch['input_ids'][:,:args.seq_length].contiguous(); attention_cpu=batch['attention_mask'][:,:args.seq_length].contiguous(); word_group_cpu=batch['word_group'][:,:args.seq_length].contiguous()
            legacy_state.current_step = step - 1
            masked_dev, labels_dev = base.apply_masking_curriculum(input_ids_cpu.to(device, non_blocking=True), attention_cpu.to(device, non_blocking=True), word_group_cpu.to(device, non_blocking=True), tokenizer, legacy_state, legacy_gen)
            masked_cpu, labels_cpu = masked_dev.cpu(), labels_dev.cpu(); del masked_dev, labels_dev
            n_pred_total=int((labels_cpu != -100).sum().item())
            if n_pred_total <= 0: raise RuntimeError(f'no MLM masked tokens at step {step}')
            raw_events = gp.candidate_events_for_batch(input_ids_cpu, attention_cpu, word_group_cpu, label_records[lo:hi], seed=args.train_rng_seed, step=step, max_control_distance_abs_diff=args.max_control_distance_abs_diff, max_control_freq_bin_abs_diff=args.max_control_freq_bin_abs_diff, max_target_token_len=args.max_target_token_len)
            evs, shuf_stats = gp.attach_shuffled(raw_events, donor_pools, seed=args.train_rng_seed, step=step)
            optim.zero_grad(set_to_none=True)
            mlm_loss_float=0.0; active_micro=0
            for mb_start in range(0, int(masked_cpu.shape[0]), args.micro_batch_size):
                mb_end=min(mb_start+args.micro_batch_size, int(masked_cpu.shape[0]))
                sl_labels=labels_cpu[mb_start:mb_end]
                n_pred_i=int((sl_labels != -100).sum().item())
                if n_pred_i <= 0: continue
                outm=model(input_ids=masked_cpu[mb_start:mb_end].to(device, non_blocking=True), attention_mask=attention_cpu[mb_start:mb_end].to(device, non_blocking=True), labels=sl_labels.to(device, non_blocking=True))
                loss=outm.loss * (n_pred_i / n_pred_total)
                loss.backward(); mlm_loss_float += float(outm.loss.detach().cpu()) * (n_pred_i / n_pred_total); active_micro += 1
                del outm, loss
            aux_loss_float=None; aux_stats={'events_used':0}
            if evs:
                aux_loss, aux_stats = compute_aux_loss(model, input_ids_cpu, attention_cpu, evs, tokenizer, device, mode=args.mode, margin=args.aux_margin, view_batch_size=args.view_batch_size)
                if aux_loss is not None:
                    (aux_loss * args.aux_weight).backward()
                    aux_loss_float = float(aux_loss.detach().cpu().item())
            if active_micro <= 0: raise RuntimeError(f'no active MLM microbatches at step {step}')
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); sched.step()
            cumulative_words += words
            loss_values.append(mlm_loss_float + (args.aux_weight * aux_loss_float if aux_loss_float is not None else 0.0)); mlm_values.append(mlm_loss_float)
            if aux_loss_float is not None: aux_values.append(aux_loss_float)
            total_aux_events += int(aux_stats.get('events_used', 0) or 0)
            event_counts.update(aux_stats.get('event_categories', {})); shuffle_level_counts.update(aux_stats.get('shuffle_levels', {}))
            rec={'step':step,'mode':args.mode,'loss_total_for_optimization':loss_values[-1],'mlm_loss':mlm_loss_float,'aux_loss_raw':aux_loss_float,'aux_weight':args.aux_weight,'lr':float(sched.get_last_lr()[0]),'batch_words':words,'cumulative_continuation_words':cumulative_words,'total_actual_word_exposure':args.initial_actual_word_exposure+cumulative_words,'masked_tokens':n_pred_total,'active_microbatches':active_micro,'aux_events_used':int(aux_stats.get('events_used', 0) or 0),'aux_event_categories':aux_stats.get('event_categories', {}),'shuffle_levels':aux_stats.get('shuffle_levels', {}),'margin_delta_mean':aux_stats.get('margin_delta_mean'),'margin_delta_success_rate':aux_stats.get('margin_delta_success_rate'),'elapsed_sec':round(time.time()-t0,1)}
            logf.write(json.dumps(rec, ensure_ascii=False)+'\n'); logf.flush()
            auxf.write(json.dumps({'step':step, **convert(aux_stats), 'shuffle_attach': convert(shuf_stats)}, ensure_ascii=False)+'\n'); auxf.flush()
            if step == 1 or step % args.log_every == 0 or step == stage_steps:
                if device.type == 'cuda': rec['cuda_peak_memory_gb'] = torch.cuda.max_memory_allocated(device) / (1024**3)
                print(json.dumps({'event':'train', **rec}, ensure_ascii=False), flush=True)
            del input_ids_cpu, attention_cpu, word_group_cpu, masked_cpu, labels_cpu
    ckpt_dir = out/'hf_model'/args.checkpoint_name
    base.save_hf_checkpoint(model, tokenizer, ckpt_dir)
    base.save_hf_checkpoint(model, tokenizer, out/'hf_model')
    trainer_state_path=None
    if args.save_trainer_state:
        state_dir=out/'trainer_state'/args.checkpoint_name; state_dir.mkdir(parents=True, exist_ok=True); trainer_state_path=state_dir/'optimizer_scheduler_rng.pt'
        torch.save({'optimizer':optim.state_dict(),'scheduler':sched.state_dict(),'torch_rng_state':torch.get_rng_state(),'cuda_rng_state_all':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,'python_random_state':random.getstate(),'numpy_random_state':np.random.get_state(),'completed_stage_steps':stage_steps,'cumulative_continuation_words':cumulative_words,'total_actual_word_exposure':args.initial_actual_word_exposure+cumulative_words,'last_tail_row':segment['last_tail_row'],'next_tail_row':int(segment['last_tail_row'])+1 if segment['last_tail_row'] is not None else None,'schedule_total_steps':schedule_total}, trainer_state_path)
    metrics={'variant':'fullctx_aux_'+args.mode,'backend':'mlm_plus_fullctx_pivot_substitution_aux','model_family':'DebertaV2ForMaskedLM','parameter_count':param_count,'vocab_size':len(tokenizer),'word_exposure':args.initial_actual_word_exposure+cumulative_words,'continuation_words':cumulative_words,'loss_first':loss_values[0] if loss_values else None,'loss_last':loss_values[-1] if loss_values else None,'mlm_loss_first':mlm_values[0] if mlm_values else None,'mlm_loss_last':mlm_values[-1] if mlm_values else None,'aux_loss_first':aux_values[0] if aux_values else None,'aux_loss_last':aux_values[-1] if aux_values else None,'actual_training_steps':stage_steps,'effective_batch_size':args.batch_size,'micro_batch_size':args.micro_batch_size,'saved_checkpoints':[{'name':args.checkpoint_name,'target_word_exposure':args.initial_actual_word_exposure+args.max_word_exposure,'actual_cumulative_word_exposure':args.initial_actual_word_exposure+cumulative_words,'path':str(ckpt_dir)}],'auxiliary':{'mode':args.mode,'aux_weight':args.aux_weight,'aux_margin':args.aux_margin,'total_aux_events':total_aux_events,'event_counts':dict(event_counts),'shuffle_level_counts':dict(shuffle_level_counts)},'trainer_state_path':str(trainer_state_path) if trainer_state_path else None,'training_log':str(log_path),'aux_stats':str(aux_path), **manifest}
    if device.type == 'cuda': metrics['cuda_peak_memory_gb'] = torch.cuda.max_memory_allocated(device)/(1024**3)
    (out/'scientific_metrics.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps({'event':'done','mode':args.mode,'output_dir':str(out),'metrics':str(out/'scientific_metrics.json'),'word_exposure':metrics['word_exposure'],'loss_last':metrics['loss_last'],'mlm_loss_last':metrics['mlm_loss_last'],'aux_loss_last':metrics['aux_loss_last'],'total_aux_events':total_aux_events,'cuda_peak_memory_gb':metrics.get('cuda_peak_memory_gb')}), flush=True)


if __name__ == '__main__':
    main()
