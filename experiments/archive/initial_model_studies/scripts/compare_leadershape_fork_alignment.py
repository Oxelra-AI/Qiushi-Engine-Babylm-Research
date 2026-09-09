#!/usr/bin/env python3
from __future__ import annotations
import json, math, pathlib
import torch
from safetensors.torch import load_file

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
ORIG = ROOT / 'training/runs/babylm_align_orig_eb16'
FORK = ROOT / 'training/runs/babylm_align_fork_eb16_acc2'
OUT = ROOT / 'data/leadershape_fork_alignment.json'
NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/leadershape_fork_alignment.md')


def read_json(p: pathlib.Path):
    return json.loads(p.read_text(encoding='utf-8'))


def read_logs(p: pathlib.Path):
    return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]


def aggregate_fork_by_opt(logs):
    groups = []
    cur = []
    last = None
    for rec in logs:
        cur.append(rec)
        # opt_step increments on the microbatch that performs the step
        if rec.get('opt_step', 0) != last and rec.get('opt_step', 0) > 0:
            groups.append(cur)
            cur = []
        last = rec.get('opt_step', last)
    if cur:
        groups.append(cur)
    agg = []
    for g in groups:
        toks = sum(int(r['masked_tokens']) for r in g)
        words = sum(int(r['batch_words']) for r in g)
        loss_tok = sum(float(r['loss']) * int(r['masked_tokens']) for r in g) / toks if toks else None
        step_rec = g[-1]
        agg.append({
            'micro_steps': [r['step'] for r in g],
            'opt_step': step_rec.get('opt_step'),
            'batch_words_sum': words,
            'cumulative_word_exposure': step_rec['cumulative_word_exposure'],
            'masked_tokens_sum': toks,
            'token_weighted_loss': loss_tok,
            'lr': step_rec['lr'],
            'seq_len': step_rec['seq_len'],
        })
    return agg


def tensor_diff(orig_path: pathlib.Path, fork_path: pathlib.Path):
    a = load_file(str(orig_path))
    b = load_file(str(fork_path))
    common = sorted(set(a) & set(b))
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    total_num = 0
    total_abs = 0.0
    total_sq = 0.0
    max_abs = 0.0
    max_name = None
    per = []
    for k in common:
        ta = a[k].float().cpu(); tb = b[k].float().cpu()
        d = (ta - tb).abs()
        n = d.numel()
        ma = float(d.max().item()) if n else 0.0
        mean = float(d.mean().item()) if n else 0.0
        rms = float(torch.sqrt((d*d).mean()).item()) if n else 0.0
        per.append({'name': k, 'numel': n, 'max_abs': ma, 'mean_abs': mean, 'rms_abs': rms})
        if ma > max_abs:
            max_abs = ma; max_name = k
        total_num += n
        total_abs += float(d.sum().item())
        total_sq += float((d*d).sum().item())
    per_sorted = sorted(per, key=lambda x: x['max_abs'], reverse=True)[:20]
    return {
        'common_tensors': len(common), 'only_orig': only_a, 'only_fork': only_b,
        'total_numel': total_num,
        'global_mean_abs': total_abs / total_num if total_num else None,
        'global_rms_abs': math.sqrt(total_sq / total_num) if total_num else None,
        'global_max_abs': max_abs, 'global_max_tensor': max_name,
        'top20_by_max_abs': per_sorted,
    }


def main():
    om = read_json(ORIG/'scientific_metrics.json')
    fm = read_json(FORK/'scientific_metrics.json')
    ol = read_logs(ORIG/'training_log.jsonl')
    fl = read_logs(FORK/'training_log.jsonl')
    fa = aggregate_fork_by_opt(fl)
    exact = {
        'example_order_manifest_sha_equal': (ORIG/'example_order_manifest.json').read_bytes() == (FORK/'example_order_manifest.json').read_bytes(),
        'tokenization_summary_equal': (ORIG/'tokenization_coupling_summary.json').read_bytes() == (FORK/'tokenization_coupling_summary.json').read_bytes(),
        'config_equal': read_json(ORIG/'hf_model/config.json') == read_json(FORK/'hf_model/config.json'),
        'total_word_exposure_equal': om.get('word_exposure') == fm.get('word_exposure'),
        'source_words_equal': om.get('source_words_consumed') == fm.get('source_words_consumed'),
        'masked_tokens_total_equal': om.get('masked_tokens_total') == fm.get('masked_tokens_total'),
        'optimizer_steps_fork_equals_orig_steps': fm.get('optimizer_steps') == om.get('actual_training_steps'),
        'effective_batch_equals_orig_batch': fm.get('effective_batch_size') == 16,
    }
    per_step = []
    for i, (o, f) in enumerate(zip(ol, fa), 1):
        per_step.append({
            'step': i,
            'orig_cum_words': o['cumulative_word_exposure'],
            'fork_cum_words': f['cumulative_word_exposure'],
            'cum_words_equal': o['cumulative_word_exposure'] == f['cumulative_word_exposure'],
            'orig_batch_words': o['batch_words'],
            'fork_batch_words_sum': f['batch_words_sum'],
            'batch_words_equal': o['batch_words'] == f['batch_words_sum'],
            'orig_masked_tokens': o['masked_tokens'],
            'fork_masked_tokens_sum': f['masked_tokens_sum'],
            'masked_tokens_equal': o['masked_tokens'] == f['masked_tokens_sum'],
            'orig_lr': o['lr'],
            'fork_lr': f['lr'],
            'lr_abs_diff': abs(float(o['lr']) - float(f['lr'])),
            'orig_loss': o['loss'],
            'fork_token_weighted_loss': f['token_weighted_loss'],
            'loss_abs_diff': abs(float(o['loss']) - float(f['token_weighted_loss'])) if f['token_weighted_loss'] is not None else None,
        })
    log_summary = {
        'orig_log_steps': len(ol), 'fork_micro_steps': len(fl), 'fork_aggregated_opt_steps': len(fa),
        'all_cum_words_equal': all(x['cum_words_equal'] for x in per_step),
        'all_batch_words_equal': all(x['batch_words_equal'] for x in per_step),
        'all_masked_tokens_equal': all(x['masked_tokens_equal'] for x in per_step),
        'max_lr_abs_diff': max(x['lr_abs_diff'] for x in per_step) if per_step else None,
        'max_loss_abs_diff': max(x['loss_abs_diff'] for x in per_step if x['loss_abs_diff'] is not None) if per_step else None,
    }
    td = tensor_diff(ORIG/'hf_model/model.safetensors', FORK/'hf_model/model.safetensors')
    payload = {
        'status': 'LEADERSHAPE_FORK_ALIGNMENT',
        'orig_run': str(ORIG), 'fork_run': str(FORK),
        'orig_metrics_core': {k: om.get(k) for k in ['parameter_count','word_exposure','actual_training_steps','lr_schedule_total_steps','masked_tokens_total','loss_first','loss_last','hidden_size','n_layer','n_head','ffn_mult','intermediate_size']},
        'fork_metrics_core': {k: fm.get(k) for k in ['parameter_count','word_exposure','actual_training_steps','grad_accum_steps','optimizer_steps','effective_batch_size','lr_schedule_total_steps','masked_tokens_total','loss_first','loss_last','hidden_size','n_layer','n_head','ffn_mult','intermediate_size']},
        'exact_accounting': exact,
        'log_summary': log_summary,
        'per_effective_step': per_step,
        'tensor_diff': td,
        'interpretation': []
    }
    if all(exact.values()) and log_summary['all_cum_words_equal'] and log_summary['all_batch_words_equal'] and log_summary['all_masked_tokens_equal'] and (log_summary['max_lr_abs_diff'] or 0) < 1e-12:
        payload['interpretation'].append('Accounting alignment is exact: same examples/tokenization/config, same effective word exposure per optimizer step, same masked-token totals, same LR trajectory, same HF config.')
    else:
        payload['interpretation'].append('Accounting alignment has differences; inspect exact_accounting/log_summary before using the fork for S1.')
    if td['global_max_abs'] is not None:
        payload['interpretation'].append(f"Final tensors are not bit-identical (max_abs={td['global_max_abs']:.6g}, mean_abs={td['global_mean_abs']:.6g}); this is expected under microbatch vs full-batch dropout/RNG unless all stochastic masks are synchronized, but accounting and optimizer-step equivalence are the load-bearing checks for the 10M S1 run.")
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    lines = ['# research — leader-shape fork alignment', '', f'Evidence JSON: `{OUT}`', '', '## Exact accounting checks', '', '| check | value |', '|---|---|']
    for k,v in exact.items(): lines.append(f'| {k} | `{v}` |')
    lines += ['', '## Log summary', '', '```json', json.dumps(log_summary, indent=2), '```', '', '## Tensor diff', '', f"- global max abs: `{td['global_max_abs']}` in `{td['global_max_tensor']}`", f"- global mean abs: `{td['global_mean_abs']}`", f"- global RMS abs: `{td['global_rms_abs']}`", '', '## Interpretation']
    lines += [f'- {x}' for x in payload['interpretation']]
    NOTE.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'status':payload['status'], 'out':str(OUT), 'note':str(NOTE), 'exact_accounting': exact, 'log_summary': log_summary, 'tensor_diff_core': {k: td[k] for k in ['global_max_abs','global_mean_abs','global_rms_abs','global_max_tensor']}}, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
