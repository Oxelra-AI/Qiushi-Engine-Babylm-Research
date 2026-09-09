#!/usr/bin/env python3
"""Synthesize research paired-tail dry-run manifests."""
from __future__ import annotations
import json
from pathlib import Path

root = Path('experiments/archive/representation_and_objectives/data')
paths = {
    'residual_low_lr': root / 'paired_tail_dryrun_residual/paired_tail_manifest.json',
    'reheat_cosine': root / 'paired_tail_dryrun_reheat/paired_tail_manifest.json',
}
items = {k: json.loads(p.read_text(encoding='utf-8')) for k, p in paths.items()}
res = items['residual_low_lr']
reh = items['reheat_cosine']
plain_keys = ['input_ids_sha256', 'attention_mask_sha256', 'word_group_sha256', 'words_sha256', 'words_sum']
mask_keys = ['masked_inputs_sha256', 'labels_sha256', 'selected_positions_sha256', 'masked_tokens']
summary = {
    'status': 'PAIRED_TAIL_DRYRUN_SYNTHESIS',
    'script': 'experiments/archive/representation_and_objectives/scripts/paired_tail_continuation_trainer.py',
    'manifests': {k: str(paths[k]) for k in paths},
    'same_tail_geometry': {
        'tail_start_row_index': res['stream']['tail_start_row_index'] == reh['stream']['tail_start_row_index'],
        'tail_examples': res['stream']['tail_examples'] == reh['stream']['tail_examples'],
        'continuation_words': res['stream']['continuation_words'] == reh['stream']['continuation_words'],
        'total_tail_steps': res['method']['total_tail_steps'] == reh['method']['total_tail_steps'],
        'values': {
            'tail_start_row_index': res['stream']['tail_start_row_index'],
            'tail_examples': res['stream']['tail_examples'],
            'continuation_words': res['stream']['continuation_words'],
            'total_tail_steps': res['method']['total_tail_steps'],
            'source_step': res['source_checkpoint']['source_step'],
            'final_global_step': res['method']['final_global_step'],
        },
    },
    'same_first_plain_batch': all(
        res['stream']['first_effective_batch_plain_hashes'][k] == reh['stream']['first_effective_batch_plain_hashes'][k]
        for k in plain_keys
    ),
    'same_first_cpu_mask_replay': all(
        res['stream']['first_effective_batch_cpu_mask_hashes'][k] == reh['stream']['first_effective_batch_cpu_mask_hashes'][k]
        for k in mask_keys
    ),
    'first_batch_hashes': {
        'plain': res['stream']['first_effective_batch_plain_hashes'],
        'cpu_mask': res['stream']['first_effective_batch_cpu_mask_hashes'],
    },
    'lr_geometry': {
        'residual_low_lr': res['lr_geometry'],
        'reheat_cosine': reh['lr_geometry'],
    },
    'interpretation': (
        'Both arms reset AdamW moments from identical 80M weights and share tail rows, first-batch '
        'tokenization, and first CPU WWM mask replay; intended difference is residual baseline LR '
        'versus reheated tail cosine.'
    ),
}
out = root / 'paired_tail_dryrun_synthesis.json'
out.write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps({
    'event': 'summary_saved',
    'path': str(out),
    'same_plain': summary['same_first_plain_batch'],
    'same_cpu_mask': summary['same_first_cpu_mask_replay'],
    'resid_first_lr': summary['lr_geometry']['residual_low_lr']['first_update_lr'],
    'reheat_first_lr': summary['lr_geometry']['reheat_cosine']['first_update_lr'],
}))


if __name__ == '__main__':
    pass
