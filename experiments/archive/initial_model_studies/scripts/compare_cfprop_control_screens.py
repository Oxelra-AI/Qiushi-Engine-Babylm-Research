#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
RUNS = {
    'linked_v3': ROOT/'training/runs/babylm_cfprop_linked_v3_random_200k',
    'unlinked_v2': ROOT/'training/runs/babylm_cfprop_unlinked_v2_random_200k',
}
OUT = ROOT/'data/cfprop_control_screen_comparison.json'
NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/cfprop_control_screen_comparison.md')


def load_json(p: pathlib.Path):
    return json.loads(p.read_text(encoding='utf-8'))


def load_logs(p: pathlib.Path):
    return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]


def summarise(key: str, run: pathlib.Path) -> dict:
    m = load_json(run/'scientific_metrics.json')
    logs = load_logs(run/'training_log.jsonl')
    hf_files = sorted(x.name for x in (run/'hf_model').iterdir())
    pollution = any(('counterfactual' in n or 'head' in n or 'aux' in n) for n in hf_files)
    cf_losses = [float(x['loss_cf']) for x in logs]
    pair_losses = [float(x['cf_loss_pair']) for x in logs]
    rand_losses = [float(x['cf_loss_random']) for x in logs]
    pair_accs = [float(x['cf_acc_pair']) for x in logs]
    rand_accs = [float(x['cf_acc_random']) for x in logs]
    valid_pair = [int(x['cf_valid_pair']) for x in logs]
    valid_rand = [int(x['cf_valid_random']) for x in logs]
    return {
        'run_dir': str(run),
        'exists': run.exists(),
        'variant': m.get('variant'),
        'wwm_word_exposure': m.get('wwm_word_exposure'),
        'aux_pair_words_seen': m.get('aux_pair_words_seen'),
        'aux_triple_words_seen': m.get('aux_triple_words_seen'),
        'total_counted_words_pair_rule': m.get('total_counted_words_pair_rule'),
        'total_counted_words_triple_rule': m.get('total_counted_words_triple_rule'),
        'actual_training_steps': m.get('actual_training_steps'),
        'lr_schedule_total_steps': m.get('lr_schedule_total_steps'),
        'parameter_count': m.get('parameter_count'),
        'cf_head_params': m.get('counterfactual_head_parameter_count'),
        'loss_mlm_first': m.get('loss_mlm_first'),
        'loss_mlm_last': m.get('loss_mlm_last'),
        'loss_cf_first': m.get('loss_cf_first'),
        'loss_cf_last': m.get('loss_cf_last'),
        'cf_train_pair_accuracy_weighted': m.get('cf_train_pair_accuracy_weighted'),
        'cf_train_valid_pair_total': m.get('cf_train_valid_pair_total'),
        'cf_heldout_eval': m.get('cf_heldout_eval'),
        'aux_meta': m.get('counterfactual_aux_meta'),
        'heldout_meta': m.get('counterfactual_heldout_meta'),
        'hf_files': hf_files,
        'hf_pollution': pollution,
        'checkpoint_names': [c.get('name') for c in m.get('saved_checkpoints', [])],
        'num_logs': len(logs),
        'first_log': logs[0] if logs else None,
        'last_log': logs[-1] if logs else None,
        'tail_mean_cf_loss': sum(cf_losses[-10:])/min(10, len(cf_losses)) if cf_losses else None,
        'tail_mean_pair_loss': sum(pair_losses[-10:])/min(10, len(pair_losses)) if pair_losses else None,
        'tail_mean_random_loss': sum(rand_losses[-10:])/min(10, len(rand_losses)) if rand_losses else None,
        'tail_mean_pair_acc': sum(pair_accs[-10:])/min(10, len(pair_accs)) if pair_accs else None,
        'tail_mean_random_acc': sum(rand_accs[-10:])/min(10, len(rand_accs)) if rand_accs else None,
        'valid_pair_min_max': [min(valid_pair), max(valid_pair)] if valid_pair else None,
        'valid_random_min_max': [min(valid_rand), max(valid_rand)] if valid_rand else None,
    }


def diff(a, b, field):
    va = a.get(field); vb = b.get(field)
    return None if va is None or vb is None else va - vb


def main():
    missing = [str(p) for p in RUNS.values() if not (p/'scientific_metrics.json').exists()]
    if missing:
        raise FileNotFoundError('missing completed run metrics: '+', '.join(missing))
    s = {k: summarise(k, p) for k,p in RUNS.items()}
    linked = s['linked_v3']; unlinked = s['unlinked_v2']
    comparison = {
        'linked_minus_unlinked_train_pair_acc': diff(linked, unlinked, 'cf_train_pair_accuracy_weighted'),
        'linked_minus_unlinked_heldout_pair_acc': linked['cf_heldout_eval']['pair_accuracy'] - unlinked['cf_heldout_eval']['pair_accuracy'],
        'linked_minus_unlinked_heldout_random_acc': linked['cf_heldout_eval']['random_accuracy'] - unlinked['cf_heldout_eval']['random_accuracy'],
        'linked_minus_unlinked_tail_pair_acc': diff(linked, unlinked, 'tail_mean_pair_acc'),
        'linked_minus_unlinked_tail_random_acc': diff(linked, unlinked, 'tail_mean_random_acc'),
        'linked_minus_unlinked_cf_loss_last': diff(linked, unlinked, 'loss_cf_last'),
        'linked_minus_unlinked_mlm_loss_last': diff(linked, unlinked, 'loss_mlm_last'),
        'both_hf_clean': (not linked['hf_pollution']) and (not unlinked['hf_pollution']),
    }
    interpretation = []
    lp = linked['cf_heldout_eval']['pair_accuracy']; up = unlinked['cf_heldout_eval']['pair_accuracy']
    if lp is not None and up is not None:
        if lp > up + 0.08 and lp >= 0.58:
            interpretation.append('Linked-v3 heldout pair accuracy is meaningfully higher than unlinked control in this auxiliary screen.')
        elif abs(lp-up) < 0.05:
            interpretation.append('Linked-v3 and unlinked-v2 heldout pair accuracies are similar; the screen does not separate true linked dependency from leakage/control artifacts.')
        else:
            interpretation.append('Auxiliary learnability pattern is mixed; do not scale without more controls.')
    payload = {'status':'CFPROP_CONTROL_SCREEN_COMPARISON','runs':s,'comparison':comparison,'interpretation':interpretation}
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    lines = ['# research — counterfactual propagation control screen comparison','',f'Evidence JSON: `{OUT}`','','| quantity | linked v3 | unlinked v2 | linked-unlinked |','|---|---:|---:|---:|']
    rows = [
        ('wwm_word_exposure','WWM words'),('aux_pair_words_seen','aux pair words'),('total_counted_words_pair_rule','counted words pair-rule'),
        ('loss_mlm_last','final MLM loss'),('loss_cf_last','final CF loss'),('cf_train_pair_accuracy_weighted','train pair acc'),
        ('tail_mean_pair_acc','tail mean pair acc'),('tail_mean_random_acc','tail mean random acc')]
    for field,label in rows:
        lv=linked.get(field); uv=unlinked.get(field); dv=diff(linked,unlinked,field)
        lines.append(f"| {label} | {lv} | {uv} | {dv} |")
    lines.append(f"| heldout pair acc | {linked['cf_heldout_eval']['pair_accuracy']} | {unlinked['cf_heldout_eval']['pair_accuracy']} | {comparison['linked_minus_unlinked_heldout_pair_acc']} |")
    lines.append(f"| heldout random acc | {linked['cf_heldout_eval']['random_accuracy']} | {unlinked['cf_heldout_eval']['random_accuracy']} | {comparison['linked_minus_unlinked_heldout_random_acc']} |")
    lines += ['', '## Interpretation'] + [f'- {x}' for x in interpretation]
    lines += ['', 'HF cleanliness: linked hf_pollution=%s, unlinked hf_pollution=%s.' % (linked['hf_pollution'], unlinked['hf_pollution'])]
    NOTE.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'status':payload['status'],'out':str(OUT),'comparison':comparison,'interpretation':interpretation}, indent=2))

if __name__ == '__main__':
    main()
