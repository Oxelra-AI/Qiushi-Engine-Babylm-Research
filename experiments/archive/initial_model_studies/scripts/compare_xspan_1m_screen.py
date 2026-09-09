#!/usr/bin/env python3
from __future__ import annotations
import json, pathlib
ROOT=pathlib.Path('experiments/archive/initial_model_studies')
RUNS={
 'xspan_true':ROOT/'training/runs/babylm_xspan_true_1M',
 'xspan_wrong':ROOT/'training/runs/babylm_xspan_wrong_1M',
 'wwm_only':ROOT/'training/runs/babylm_wwm_only_rho0_1M',
}
OUT=ROOT/'data/xspan_1m_mechanism_comparison.json'
NOTE=(ROOT.parents[2] / 'research/notes/initial_model_studies/xspan_1m_mechanism_comparison.md')

def load_logs(p):
    return [json.loads(l) for l in (p/'training_log.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]

def summ(k,p):
    m=json.loads((p/'scientific_metrics.json').read_text(encoding='utf-8'))
    logs=load_logs(p)
    hf_files=sorted(x.name for x in (p/'hf_model').iterdir())
    pollution=any(('head' in n or 'aux' in n or 'xspan' in n) for n in hf_files)
    return {
        'run_dir':str(p),'context':m.get('xspan_context'),'rho':m.get('xspan_rho'),
        'wwm_word_exposure':m.get('wwm_word_exposure'),'xspan_word_exposure':m.get('xspan_word_exposure'),
        'total_counted_word_exposure':m.get('total_counted_word_exposure'),'actual_training_steps':m.get('actual_training_steps'),
        'example_order_hash_hint':'see example_order_manifest sha in training artifact output',
        'parameter_count':m.get('parameter_count'),'loss_last':m.get('loss_last'),'loss_wwm_last':m.get('loss_wwm_last'),'loss_xspan_last':m.get('loss_xspan_last'),
        'loss_wwm_first':m.get('loss_wwm_first'),'loss_xspan_first':m.get('loss_xspan_first'),
        'xspan_eval':m.get('xspan_eval'),'xspan_target_tokens_total':m.get('xspan_target_tokens_total'),'xspan_valid_rows_total':m.get('xspan_valid_rows_total'),
        'masked_tokens_total':m.get('masked_tokens_total'),'masked_tokens_per_wwm_whitespace_word':m.get('masked_tokens_per_wwm_whitespace_word'),
        'xspan_train_meta':m.get('xspan_train_meta'),
        'hf_files':hf_files,'hf_pollution':pollution,'last_log':logs[-1] if logs else None,
    }

def val(s,run,path):
    x=s[run]
    for p in path:
        x=x[p]
    return x

def d(s,a,b,path):
    va=val(s,a,path); vb=val(s,b,path)
    return None if va is None or vb is None else va-vb

def main():
    s={k:summ(k,p) for k,p in RUNS.items()}
    comp={
      'matched_wwm_exposure':len({s[k]['wwm_word_exposure'] for k in s})==1,
      'matched_official_steps':len({s[k]['actual_training_steps'] for k in s})==1,
      'all_hf_clean':all(not s[k]['hf_pollution'] for k in s),
      'true_minus_wrong_delta_logprob_true_minus_wrong':d(s,'xspan_true','xspan_wrong',['xspan_eval','delta_logprob_true_minus_wrong']),
      'true_minus_wwm_delta_logprob_true_minus_wrong':d(s,'xspan_true','wwm_only',['xspan_eval','delta_logprob_true_minus_wrong']),
      'true_minus_wrong_delta_logprob_true_minus_no':d(s,'xspan_true','xspan_wrong',['xspan_eval','delta_logprob_true_minus_no']),
      'true_minus_wwm_delta_logprob_true_minus_no':d(s,'xspan_true','wwm_only',['xspan_eval','delta_logprob_true_minus_no']),
      'true_minus_wwm_true_s1_loss':d(s,'xspan_true','wwm_only',['xspan_eval','true_s1','loss_per_token']),
      'wrong_minus_wwm_true_s1_loss':d(s,'xspan_wrong','wwm_only',['xspan_eval','true_s1','loss_per_token']),
      'true_minus_wwm_final_wwm_loss':d(s,'xspan_true','wwm_only',['loss_wwm_last']),
      'wrong_minus_wwm_final_wwm_loss':d(s,'xspan_wrong','wwm_only',['loss_wwm_last']),
    }
    interp=[]
    if comp['true_minus_wrong_delta_logprob_true_minus_wrong'] is not None:
        if comp['true_minus_wrong_delta_logprob_true_minus_wrong'] < 0.01:
            interp.append('At 1M, true-s1 XSpan does not induce a meaningful heldout true-vs-wrong specificity over the wrong-s1 control; the gain is only a few thousandths of logprob/token.')
        else:
            interp.append('At 1M, true-s1 XSpan separates from wrong-s1 control on heldout true-vs-wrong specificity.')
    if comp['true_minus_wwm_true_s1_loss'] is not None and comp['true_minus_wwm_true_s1_loss'] < -0.5:
        interp.append('Both XSpan arms greatly improve absolute XSpan target likelihood versus WWM-only, showing the primary span objective is learned, but this can be target-span familiarity rather than s1 binding.')
    if comp['true_minus_wwm_final_wwm_loss'] is not None and comp['true_minus_wwm_final_wwm_loss'] > 0.03:
        interp.append('XSpan rho=0.15 slightly worsens ordinary WWM loss at 1M, so guard columns may be vulnerable.')
    payload={'status':'XSPAN_1M_MECHANISM_COMPARISON','runs':s,'comparison':comp,'interpretation':interp}
    OUT.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    lines=['# research — XSpan 1M mechanism comparison','',f'Evidence JSON: `{OUT}`','','| arm | rho | context | counted words | WWM loss last | XSpan loss last | true loss | wrong loss | no loss | Δ true-wrong | Δ true-no |','|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for k in ['xspan_true','xspan_wrong','wwm_only']:
        r=s[k]; xe=r['xspan_eval']
        lines.append(f"| {k} | {r['rho']} | {r['context']} | {r['total_counted_word_exposure']} | {r['loss_wwm_last']:.4f} | {r['loss_xspan_last']:.4f} | {xe['true_s1']['loss_per_token']:.4f} | {xe['wrong_s1']['loss_per_token']:.4f} | {xe['no_s1']['loss_per_token']:.4f} | {xe['delta_logprob_true_minus_wrong']:+.4f} | {xe['delta_logprob_true_minus_no']:+.4f} |")
    lines += ['','## Deltas','```json',json.dumps(comp,indent=2),'```','','## Interpretation']+[f'- {x}' for x in interp]
    NOTE.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':payload['status'],'out':str(OUT),'comparison':comp,'interpretation':interp},indent=2))
if __name__=='__main__': main()
