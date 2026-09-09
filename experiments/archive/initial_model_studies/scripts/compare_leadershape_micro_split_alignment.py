#!/usr/bin/env python3
from __future__ import annotations
import json, math, pathlib
import torch
from safetensors.torch import load_file

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
ORIG = ROOT / 'training/runs/babylm_align_orig_eb16'
FORK = ROOT / 'training/runs/babylm_align_fork_eb16_micro8'
OUT = ROOT / 'data/leadershape_micro_split_alignment.json'
NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/leadershape_micro_split_alignment.md')

def read_json(p):
    return json.loads(pathlib.Path(p).read_text(encoding='utf-8'))

def read_logs(p):
    return [json.loads(x) for x in pathlib.Path(p).read_text(encoding='utf-8').splitlines() if x.strip()]

def tensor_diff(a_path, b_path):
    a=load_file(str(a_path)); b=load_file(str(b_path))
    common=sorted(set(a)&set(b)); only_a=sorted(set(a)-set(b)); only_b=sorted(set(b)-set(a))
    total_n=0; total_abs=0.0; total_sq=0.0; max_abs=-1.0; max_name=None
    top=[]
    for k in common:
        da=(a[k].float().cpu()-b[k].float().cpu()).abs(); n=da.numel()
        ma=float(da.max().item()) if n else 0.0; mean=float(da.mean().item()) if n else 0.0; rms=float(torch.sqrt((da*da).mean()).item()) if n else 0.0
        top.append({'name':k,'numel':n,'max_abs':ma,'mean_abs':mean,'rms_abs':rms})
        if ma>max_abs: max_abs=ma; max_name=k
        total_n += n; total_abs += float(da.sum().item()); total_sq += float((da*da).sum().item())
    return {'common_tensors':len(common),'only_orig':only_a,'only_fork':only_b,'total_numel':total_n,'global_max_abs':max_abs,'global_max_tensor':max_name,'global_mean_abs':total_abs/total_n if total_n else None,'global_rms_abs':math.sqrt(total_sq/total_n) if total_n else None,'top20_by_max_abs':sorted(top,key=lambda x:x['max_abs'],reverse=True)[:20]}

def main():
    om=read_json(ORIG/'scientific_metrics.json'); fm=read_json(FORK/'scientific_metrics.json')
    ol=read_logs(ORIG/'training_log.jsonl'); fl=read_logs(FORK/'training_log.jsonl')
    exact={
        'example_order_manifest_bytes_equal': (ORIG/'example_order_manifest.json').read_bytes()==(FORK/'example_order_manifest.json').read_bytes(),
        'tokenization_summary_bytes_equal': (ORIG/'tokenization_coupling_summary.json').read_bytes()==(FORK/'tokenization_coupling_summary.json').read_bytes(),
        'hf_config_equal': read_json(ORIG/'hf_model/config.json')==read_json(FORK/'hf_model/config.json'),
        'word_exposure_equal': om.get('word_exposure')==fm.get('word_exposure'),
        'source_words_equal': om.get('source_words_consumed')==fm.get('source_words_consumed'),
        'masked_tokens_total_equal': om.get('masked_tokens_total')==fm.get('masked_tokens_total'),
        'orig_steps_equal_fork_steps': om.get('actual_training_steps')==fm.get('actual_training_steps')==fm.get('optimizer_steps'),
        'effective_batch_equal': fm.get('effective_batch_size')==16,
        'micro_batch_size_recorded': fm.get('micro_batch_size')==8,
    }
    per=[]
    for i,(o,f) in enumerate(zip(ol,fl),1):
        per.append({
            'step':i,
            'cum_words_equal': o['cumulative_word_exposure']==f['cumulative_word_exposure'],
            'batch_words_equal': o['batch_words']==f['batch_words'],
            'masked_tokens_equal': o['masked_tokens']==f['masked_tokens'],
            'seq_len_equal': o['seq_len']==f['seq_len'],
            'lr_abs_diff': abs(float(o['lr'])-float(f['lr'])),
            'loss_abs_diff': abs(float(o['loss'])-float(f['loss'])),
            'orig': {'cum_words':o['cumulative_word_exposure'],'batch_words':o['batch_words'],'masked_tokens':o['masked_tokens'],'lr':o['lr'],'loss':o['loss']},
            'fork': {'cum_words':f['cumulative_word_exposure'],'batch_words':f['batch_words'],'masked_tokens':f['masked_tokens'],'lr':f['lr'],'loss':f['loss'],'opt_step':f.get('opt_step')},
        })
    log_summary={
        'orig_log_steps':len(ol),'fork_log_steps':len(fl),
        'all_cum_words_equal':all(x['cum_words_equal'] for x in per),
        'all_batch_words_equal':all(x['batch_words_equal'] for x in per),
        'all_masked_tokens_equal':all(x['masked_tokens_equal'] for x in per),
        'all_seq_len_equal':all(x['seq_len_equal'] for x in per),
        'max_lr_abs_diff':max((x['lr_abs_diff'] for x in per), default=None),
        'max_loss_abs_diff':max((x['loss_abs_diff'] for x in per), default=None),
    }
    td=tensor_diff(ORIG/'hf_model/model.safetensors', FORK/'hf_model/model.safetensors')
    alignment_ok = all(exact.values()) and log_summary['all_cum_words_equal'] and log_summary['all_batch_words_equal'] and log_summary['all_masked_tokens_equal'] and log_summary['all_seq_len_equal'] and (log_summary['max_lr_abs_diff'] or 0.0)<1e-12
    interpretation=[]
    if alignment_ok:
        interpretation.append('The isolated fork now preserves the protected trainer accounting and masking semantics exactly at this short-run level: same examples, tokenizer summary, HF config, exposure, masked-token totals/per-step masked tokens, sequence length, optimizer steps, and LR trajectory.')
    else:
        interpretation.append('The isolated fork still differs in an accounting or masking field; do not launch S1 10M until inspected.')
    interpretation.append('Final tensors/losses need not be bit-identical because the micro-split forward/backward changes dropout RNG grouping relative to one full-batch forward. This does not change the data, masks, loss reduction, optimizer-step schedule, or checkpoint accounting; it is an unavoidable memory-rescue difference unless dropout RNG is specially synchronized.')
    payload={'status':'LEADERSHAPE_MICRO_SPLIT_ALIGNMENT','orig_run':str(ORIG),'fork_run':str(FORK),'orig_metrics_core':{k:om.get(k) for k in ['parameter_count','word_exposure','actual_training_steps','lr_schedule_total_steps','masked_tokens_total','loss_first','loss_last','hidden_size','n_layer','n_head','ffn_mult','intermediate_size']},'fork_metrics_core':{k:fm.get(k) for k in ['parameter_count','word_exposure','actual_training_steps','optimizer_steps','effective_batch_size','micro_batch_size','lr_schedule_total_steps','masked_tokens_total','loss_first','loss_last','hidden_size','n_layer','n_head','ffn_mult','intermediate_size']},'exact_accounting':exact,'log_summary':log_summary,'per_step':per,'tensor_diff':td,'alignment_accounting_ok':alignment_ok,'interpretation':interpretation}
    OUT.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    lines=['# research — leader-shape fork micro-split alignment','',f'Evidence JSON: `{OUT}`','','## Exact checks','','| check | value |','|---|---|']
    for k,v in exact.items(): lines.append(f'| {k} | `{v}` |')
    lines += ['','## Log summary','', '```json', json.dumps(log_summary,indent=2), '```','','## Tensor diff','',f"- global max abs: `{td['global_max_abs']}` in `{td['global_max_tensor']}`",f"- global mean abs: `{td['global_mean_abs']}`",f"- global RMS abs: `{td['global_rms_abs']}`",'','## Interpretation']
    lines += [f'- {x}' for x in interpretation]
    NOTE.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':payload['status'],'out':str(OUT),'note':str(NOTE),'alignment_accounting_ok':alignment_ok,'exact_accounting':exact,'log_summary':log_summary,'tensor_diff_core':{k:td[k] for k in ['global_max_abs','global_mean_abs','global_rms_abs','global_max_tensor']}},indent=2,ensure_ascii=False))
if __name__=='__main__': main()
