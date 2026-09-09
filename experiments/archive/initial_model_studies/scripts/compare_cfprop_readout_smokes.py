#!/usr/bin/env python3
from __future__ import annotations
import json, pathlib
ROOT = pathlib.Path('experiments/archive/initial_model_studies')
RUNS = {
    'dependent': ROOT/'training/runs/babylm_cfprop_v3_readout_dependent_smoke20k',
    's2_nondependent': ROOT/'training/runs/babylm_cfprop_v3_readout_s2_nondep_smoke20k',
    's1_local': ROOT/'training/runs/babylm_cfprop_v3_readout_s1_local_smoke20k',
}
OUT = ROOT/'data/cfprop_readout_smoke_comparison.json'
NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/cfprop_readout_smoke_comparison.md')

def load_logs(p):
    return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]

def summarise(k,p):
    m=json.loads((p/'scientific_metrics.json').read_text(encoding='utf-8'))
    logs=load_logs(p/'training_log.jsonl')
    hf_files=sorted(x.name for x in (p/'hf_model').iterdir())
    pollution=any(('counterfactual' in n or 'head' in n or 'aux' in n) for n in hf_files)
    def tail_mean(field):
        vals=[float(x[field]) for x in logs[-min(10,len(logs)):]]
        return sum(vals)/len(vals) if vals else None
    return {
        'run_dir': str(p),
        'cf_readout_mode': m.get('cf_readout_mode'),
        'wwm_word_exposure': m.get('wwm_word_exposure'),
        'aux_pair_words_seen': m.get('aux_pair_words_seen'),
        'aux_triple_words_seen': m.get('aux_triple_words_seen'),
        'total_counted_words_pair_rule': m.get('total_counted_words_pair_rule'),
        'total_counted_words_triple_rule': m.get('total_counted_words_triple_rule'),
        'actual_training_steps': m.get('actual_training_steps'),
        'parameter_count': m.get('parameter_count'),
        'cf_head_params': m.get('counterfactual_head_parameter_count'),
        'loss_mlm_first': m.get('loss_mlm_first'),
        'loss_mlm_last': m.get('loss_mlm_last'),
        'loss_cf_first': m.get('loss_cf_first'),
        'loss_cf_last': m.get('loss_cf_last'),
        'cf_train_pair_accuracy_weighted': m.get('cf_train_pair_accuracy_weighted'),
        'cf_train_valid_pair_total': m.get('cf_train_valid_pair_total'),
        'cf_heldout_eval': m.get('cf_heldout_eval'),
        'num_logs': len(logs),
        'valid_pair_min_max': [min(int(x['cf_valid_pair']) for x in logs), max(int(x['cf_valid_pair']) for x in logs)],
        'valid_random_min_max': [min(int(x['cf_valid_random']) for x in logs), max(int(x['cf_valid_random']) for x in logs)],
        'tail_mean_pair_acc': tail_mean('cf_acc_pair'),
        'tail_mean_random_acc': tail_mean('cf_acc_random'),
        'tail_mean_pair_loss': tail_mean('cf_loss_pair'),
        'tail_mean_random_loss': tail_mean('cf_loss_random'),
        'last_log': logs[-1],
        'hf_files': hf_files,
        'hf_pollution': pollution,
    }

def diff(s,a,b,field):
    va=s[a].get(field); vb=s[b].get(field)
    return None if va is None or vb is None else va-vb

def nested_diff(s,a,b,outer,inner):
    va=s[a][outer][inner]; vb=s[b][outer][inner]
    return None if va is None or vb is None else va-vb

def main():
    missing=[str(p) for p in RUNS.values() if not (p/'scientific_metrics.json').exists()]
    if missing: raise FileNotFoundError(missing)
    s={k:summarise(k,p) for k,p in RUNS.items()}
    comp={
        'all_wwm_exposure_equal': len({s[k]['wwm_word_exposure'] for k in s})==1,
        'all_aux_pair_words_equal': len({s[k]['aux_pair_words_seen'] for k in s})==1,
        'all_hf_clean': all(not s[k]['hf_pollution'] for k in s),
        'dependent_minus_s2_nondep_heldout_pair_acc': nested_diff(s,'dependent','s2_nondependent','cf_heldout_eval','pair_accuracy'),
        'dependent_minus_s2_nondep_heldout_random_acc': nested_diff(s,'dependent','s2_nondependent','cf_heldout_eval','random_accuracy'),
        'dependent_minus_s1_local_heldout_pair_acc': nested_diff(s,'dependent','s1_local','cf_heldout_eval','pair_accuracy'),
        's1_local_minus_dependent_train_pair_acc': diff(s,'s1_local','dependent','cf_train_pair_accuracy_weighted'),
        's2_nondep_minus_dependent_tail_pair_acc': diff(s,'s2_nondependent','dependent','tail_mean_pair_acc'),
        's1_local_minus_dependent_tail_pair_acc': diff(s,'s1_local','dependent','tail_mean_pair_acc'),
        'dependent_minus_s2_nondep_cf_loss_last': diff(s,'dependent','s2_nondependent','loss_cf_last'),
    }
    interp=[]
    if abs(comp['dependent_minus_s2_nondep_heldout_pair_acc']) < 0.05:
        interp.append('20k smoke does not localize heldout pair signal to the dependent token; dependent and s2-nondependent are similar at this scale.')
    if comp['all_wwm_exposure_equal'] and comp['all_aux_pair_words_equal'] and comp['all_hf_clean']:
        interp.append('Readout smokes are matched in WWM exposure, auxiliary exposure, and HF cleanliness.')
    if s['s1_local']['cf_heldout_eval']['pair_accuracy'] is not None and s['s1_local']['cf_heldout_eval']['pair_accuracy'] > max(s['dependent']['cf_heldout_eval']['pair_accuracy'], s['s2_nondependent']['cf_heldout_eval']['pair_accuracy']) + 0.1:
        interp.append('s1-local readout is much easier, suggesting local anomaly is an upper-bound shortcut.')
    payload={'status':'CFPROP_READOUT_SMOKE_COMPARISON','runs':s,'comparison':comp,'interpretation':interp}
    OUT.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    lines=['# research — counterfactual propagation readout-location smoke comparison','',f'Evidence JSON: `{OUT}`','','| mode | heldout pair | heldout random | train pair | tail pair | final CF loss | aux pair words | HF clean |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for k in ['dependent','s2_nondependent','s1_local']:
        r=s[k]
        lines.append(f"| {k} | {r['cf_heldout_eval']['pair_accuracy']} | {r['cf_heldout_eval']['random_accuracy']} | {r['cf_train_pair_accuracy_weighted']} | {r['tail_mean_pair_acc']} | {r['loss_cf_last']} | {r['aux_pair_words_seen']} | {not r['hf_pollution']} |")
    lines += ['', '## Key deltas', json.dumps(comp, indent=2), '', '## Interpretation'] + [f'- {x}' for x in interp]
    NOTE.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':payload['status'],'out':str(OUT),'comparison':comp,'interpretation':interp},indent=2))
if __name__=='__main__': main()
