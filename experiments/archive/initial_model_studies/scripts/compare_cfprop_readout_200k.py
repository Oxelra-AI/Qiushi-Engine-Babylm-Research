#!/usr/bin/env python3
from __future__ import annotations
import json, pathlib
ROOT = pathlib.Path('experiments/archive/initial_model_studies')
RUNS = {
    'dependent': ROOT/'training/runs/babylm_cfprop_v3_readout_dependent_200k',
    's2_nondependent': ROOT/'training/runs/babylm_cfprop_v3_readout_s2_nondep_200k',
    's1_local': ROOT/'training/runs/babylm_cfprop_v3_readout_s1_local_200k',
}
OUT = ROOT/'data/cfprop_readout_200k_comparison.json'
NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/cfprop_readout_200k_comparison.md')

def logs(p):
    return [json.loads(x) for x in (p/'training_log.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]

def summ(k,p):
    m=json.loads((p/'scientific_metrics.json').read_text(encoding='utf-8'))
    lg=logs(p)
    hf_files=sorted(x.name for x in (p/'hf_model').iterdir())
    pollution=any(('counterfactual' in n or 'head' in n or 'aux' in n) for n in hf_files)
    def tail(field,n=20):
        vals=[float(x[field]) for x in lg[-min(n,len(lg)):]]
        return sum(vals)/len(vals) if vals else None
    return {
        'run_dir': str(p), 'mode': m.get('cf_readout_mode'),
        'wwm_word_exposure': m.get('wwm_word_exposure'), 'aux_pair_words_seen': m.get('aux_pair_words_seen'),
        'aux_triple_words_seen': m.get('aux_triple_words_seen'), 'total_counted_words_pair_rule': m.get('total_counted_words_pair_rule'),
        'total_counted_words_triple_rule': m.get('total_counted_words_triple_rule'), 'steps': m.get('actual_training_steps'),
        'loss_mlm_first': m.get('loss_mlm_first'), 'loss_mlm_last': m.get('loss_mlm_last'),
        'loss_cf_first': m.get('loss_cf_first'), 'loss_cf_last': m.get('loss_cf_last'),
        'train_pair_acc': m.get('cf_train_pair_accuracy_weighted'), 'train_valid_pair': m.get('cf_train_valid_pair_total'),
        'heldout_pair_acc': m.get('cf_heldout_eval',{}).get('pair_accuracy'),
        'heldout_random_acc': m.get('cf_heldout_eval',{}).get('random_accuracy'),
        'heldout_loss': m.get('cf_heldout_eval',{}).get('loss_mean'),
        'valid_pair_minmax': [min(int(x['cf_valid_pair']) for x in lg), max(int(x['cf_valid_pair']) for x in lg)],
        'valid_random_minmax': [min(int(x['cf_valid_random']) for x in lg), max(int(x['cf_valid_random']) for x in lg)],
        'tail20_pair_acc': tail('cf_acc_pair'), 'tail20_random_acc': tail('cf_acc_random'),
        'tail20_pair_loss': tail('cf_loss_pair'), 'tail20_random_loss': tail('cf_loss_random'),
        'last_log': lg[-1], 'hf_files': hf_files, 'hf_pollution': pollution,
    }

def d(s,a,b,field):
    va=s[a].get(field); vb=s[b].get(field)
    return None if va is None or vb is None else va-vb

def main():
    missing=[str(p) for p in RUNS.values() if not (p/'scientific_metrics.json').exists()]
    if missing:
        raise FileNotFoundError('missing completed metrics: '+', '.join(missing))
    s={k:summ(k,p) for k,p in RUNS.items()}
    comp={
        'matched_wwm_exposure': len({s[k]['wwm_word_exposure'] for k in s})==1,
        'matched_aux_pair_words': len({s[k]['aux_pair_words_seen'] for k in s})==1,
        'all_hf_clean': all(not s[k]['hf_pollution'] for k in s),
        'dependent_minus_s2_nondep_heldout_pair_acc': d(s,'dependent','s2_nondependent','heldout_pair_acc'),
        'dependent_minus_s2_nondep_tail20_pair_acc': d(s,'dependent','s2_nondependent','tail20_pair_acc'),
        'dependent_minus_s2_nondep_final_cf_loss': d(s,'dependent','s2_nondependent','loss_cf_last'),
        's1_local_minus_dependent_heldout_pair_acc': d(s,'s1_local','dependent','heldout_pair_acc'),
        's1_local_minus_dependent_tail20_pair_acc': d(s,'s1_local','dependent','tail20_pair_acc'),
        's1_local_minus_dependent_final_cf_loss': d(s,'s1_local','dependent','loss_cf_last'),
    }
    interp=[]
    if comp['dependent_minus_s2_nondep_heldout_pair_acc'] is not None:
        if comp['dependent_minus_s2_nondep_heldout_pair_acc'] > 0.08 and s['dependent']['heldout_pair_acc'] >= 0.58:
            interp.append('Dependent readout localizes a stronger heldout signal than matched s2 nondependent readout.')
        elif abs(comp['dependent_minus_s2_nondep_heldout_pair_acc']) < 0.05:
            interp.append('Dependent and s2 nondependent heldout pair accuracies remain similar; no convincing localization to dependent token.')
        else:
            interp.append('Readout localization is mixed; inspect logs and losses before scaling.')
    if comp['s1_local_minus_dependent_heldout_pair_acc'] is not None and comp['s1_local_minus_dependent_heldout_pair_acc'] > 0.08:
        interp.append('s1-local readout is easier than dependent readout, indicating local anomaly shortcut is strong.')
    payload={'status':'CFPROP_READOUT_200K_COMPARISON','runs':s,'comparison':comp,'interpretation':interp}
    OUT.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    lines=['# research — counterfactual propagation v3 readout localization at 200k','',f'Evidence JSON: `{OUT}`','','| mode | heldout pair | heldout random | train pair | tail20 pair | final CF loss | final MLM loss | aux pair words | HF clean |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for k in ['dependent','s2_nondependent','s1_local']:
        r=s[k]
        lines.append(f"| {k} | {r['heldout_pair_acc']} | {r['heldout_random_acc']} | {r['train_pair_acc']} | {r['tail20_pair_acc']} | {r['loss_cf_last']} | {r['loss_mlm_last']} | {r['aux_pair_words_seen']} | {not r['hf_pollution']} |")
    lines += ['', '## Key deltas', '```json', json.dumps(comp,indent=2), '```', '', '## Interpretation'] + [f'- {x}' for x in interp]
    NOTE.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':payload['status'],'out':str(OUT),'comparison':comp,'interpretation':interp},indent=2))
if __name__=='__main__': main()
