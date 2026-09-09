#!/usr/bin/env python3
import json
from pathlib import Path
p = Path('experiments/archive/representation_and_objectives/data/tokenizer_surface_contingency/tokenizer_surface_contingency.json')
d = json.loads(p.read_text(encoding='utf-8'))
for section, groups in [
    ('pool', ['ALL','reinvest_changed_block','inherited_qwen_pairs','shared_filler::childes','shared_filler::gutenberg','shared_filler::open_subtitles','shared_filler::simple_wiki','shared_filler::bnc_spoken','shared_filler::switchboard']),
    ('eval', ['ALL','family::BLiMP','family::Supplement','family::EWoK','family::EWoK_concat','family::Entity','family::COMPS','family::GlobalPIQA_parallel','family::GlobalPIQA_nonparallel','family::Reading_sentence','family::Reading_word','family::SuperGLUE'])
]:
    rows = d['reinvest_pool_occurrence_rows'] if section == 'pool' else d['eval_occurrence_rows']
    by = {r['group']: r for r in rows}
    print('\nSECTION', section)
    for g in groups:
        r = by.get(g)
        if not r:
            continue
        print(g, 'n', r['n_texts'], 'words', r['total_words_field'], 'ratio', round(r['new_over_old_token_ratio'], 6), 'oldonly_pct', round(100*r['old_only_occ_fraction'], 3), 'newonly_pct', round(100*r['new_only_occ_fraction'], 3), 'mean_delta', round(r['delta_len']['mean'], 3))
print('\nEWOK domains')
for r in d['eval_occurrence_rows']:
    if r['group'].startswith('ewok::'):
        print(r['group'], 'ratio', round(r['new_over_old_token_ratio'], 6), 'oldonly_pct', round(100*r['old_only_occ_fraction'], 3), 'mean_delta', round(r['delta_len']['mean'], 3))
print('\nfrag selected')
for k in ['family::EWoK','family::COMPS','family::GlobalPIQA_parallel','family::GlobalPIQA_nonparallel','ewok_domain::spatial-relations','ewok_domain::material-dynamics','ewok_domain::physical-dynamics','ewok_domain::social-relations','family::SuperGLUE']:
    r = d['word_fragmentation'].get(k)
    if not r:
        continue
    print('\n', k, 'changed', r['changed_word_types'], '/', r['unique_eval_words'], 'longer_mass', r['eval_weighted_extra_pieces_new_longer'], 'shorter_mass', r['eval_weighted_extra_pieces_new_shorter'])
    for x in r['top_new_longer_eval_words'][:8]:
        print(' ', x['word'], 'eval', x['eval_count'], 'train', x['train_count'], 'old', x['old_len'], x['old_tokens'], 'new', x['new_len'], x['new_tokens'])
