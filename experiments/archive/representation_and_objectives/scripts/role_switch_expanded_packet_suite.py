#!/usr/bin/env python3
"""research: expanded role-switch packet suite with structurally matched controls.

Purpose: build a disposable learning-test packet suite without starting any
submission-facing retrain.  The treatment keeps role exchange; the role-fixed
control preserves contexts, templates, entities, target-slot adjacency and word
counts while removing the sign flip (one fixed alternative is used in both
context directions).  The suite also keeps held-out templates and an entirely
held-out container family for transfer scoring.

This script imports the research hand-written grammar/vocabulary and increases
entity combinations per template.  It does not use pretrained parsers, teacher
models, external corpora, or official evaluation labels.
"""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import hashlib
import importlib.util
import json
import pathlib
import random
from collections import Counter, defaultdict

USER_ROOT = pathlib.Path('.')
PATH = USER_ROOT / 'experiments/archive/representation_and_objectives/scripts/role_switch_packet_builder.py'
OUT_DIR = USER_ROOT / 'experiments/archive/representation_and_objectives/data/role_switch_expanded_packets'
EVAL_ROOT = USER_ROOT / 'experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data'
TOK_DIR = USER_ROOT / 'experiments/archive/representation_and_objectives/data/legal_representation_route_map/tokenizers/legal_byte_bpe_40k'

SEED = 145045
N_PER_TEMPLATE = 60
TRAIN_FAMILIES = {'spatial', 'transfer', 'comparative', 'state_change'}
EXCLUDED_TRAIN_FAMILIES = {'temporal', 'container'}


def load_step144():
    spec = importlib.util.spec_from_file_location('role_switch_packet_builder', PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    mod.N_PER_TEMPLATE = N_PER_TEMPLATE
    return mod


def fill_consequence(pair, direction, mode):
    """Return context + first consequence for treatment or role_fixed control."""
    ctx = pair[f'context_{direction}']
    alt0, alt1 = pair['alt_0'], pair['alt_1']
    correct = pair[f'correct_{direction}']
    true_other = alt1 if correct == alt0 else alt0
    if mode == 'treatment':
        target, other = correct, true_other
    elif mode == 'role_fixed':
        # Fixed per pair, not per direction; alternate by pair_id to keep target
        # frequencies balanced while destroying role exchange.
        fixed = alt0 if (pair['pair_id'] % 2 == 0) else alt1
        target = fixed
        other = alt1 if fixed == alt0 else alt0
    else:
        raise ValueError(mode)
    con = pair['consequence_masked'].replace('__TARGET__', target).replace('__OTHER__', other)
    return ctx + ' ' + con


def build_training_examples(pairs, mode):
    rows = []
    for p in pairs:
        if p['family'] not in TRAIN_FAMILIES or p['style'] != 'train':
            continue
        for direction in ['AB', 'BA']:
            text = fill_consequence(p, direction, mode)
            rows.append({
                'text': text,
                'mode': mode,
                'family': p['family'],
                'template_id': p['template_id'],
                'style': p['style'],
                'direction': direction,
                'pair_id': p['pair_id'],
                'word_count': len(text.split()),
            })
    return rows


def compute_balance(rows):
    return {
        'n_texts': len(rows),
        'words': sum(r['word_count'] for r in rows),
        'by_family_texts': dict(sorted(Counter(r['family'] for r in rows).items())),
        'by_family_words': dict(sorted((fam, sum(r['word_count'] for r in rows if r['family'] == fam))
                                      for fam in set(r['family'] for r in rows))),
        'by_direction': dict(sorted(Counter(r['direction'] for r in rows).items())),
        'by_template': dict(sorted(Counter(r['template_id'] for r in rows).items())),
    }


def target_counts_in_consequence(rows, pairs):
    pair_by_id = {p['pair_id']: p for p in pairs}
    counts = Counter()
    for r in rows:
        p = pair_by_id[r['pair_id']]
        # Count occurrences of scoring alternatives after the context.
        ctx_len = len(p[f"context_{r['direction']}"].split())
        for w in r['text'].split()[ctx_len:]:
            clean = w.strip('.,!?;:').lower()
            if clean == p['alt_0'].lower():
                counts[f"{p['family']}::alt0"] += 1
            if clean == p['alt_1'].lower():
                counts[f"{p['family']}::alt1"] += 1
    return dict(sorted(counts.items()))


def simple_wwm_density(rows, tok_dir, n_passes=10, mask_prob=0.15):
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(tok_dir))
    except Exception as e:
        return {'error': str(e)}
    rng = random.Random(145099)
    stats = defaultdict(lambda: {'total_words': 0, 'total_tokens': 0, 'masked_groups': 0})
    for row in rows:
        words = row['text'].split()
        fam = row['family']
        stats[fam]['total_words'] += len(words)
        word_groups = []
        for wi, w in enumerate(words):
            ids = tok.encode(w, add_special_tokens=False)
            stats[fam]['total_tokens'] += len(ids)
            word_groups.extend([wi] * len(ids))
        unique_groups = sorted(set(word_groups))
        for _ in range(n_passes):
            stats[fam]['masked_groups'] += sum(1 for _g in unique_groups if rng.random() < mask_prob)
    out = {}
    for fam, s in sorted(stats.items()):
        s = dict(s)
        s['masked_groups_per_pass'] = s['masked_groups'] / n_passes
        s['masked_per_word_exposure'] = s['masked_groups'] / max(1, s['total_words'] * n_passes)
        out[fam] = s
    return {'per_family': out, 'n_passes': n_passes, 'mask_prob': mask_prob}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    research = load_step144()
    rng = random.Random(SEED)
    e_pairs, _e_texts = research.generate_entity_swap(rng)
    s_pairs, _s_texts = research.generate_state_change(rng)
    all_pairs = e_pairs + s_pairs
    # Add stable pair IDs before any rendering.
    for i, p in enumerate(all_pairs):
        p['pair_id'] = i

    treatment_rows = build_training_examples(all_pairs, 'treatment')
    role_fixed_rows = build_training_examples(all_pairs, 'role_fixed')
    if sum(r['word_count'] for r in treatment_rows) != sum(r['word_count'] for r in role_fixed_rows):
        raise RuntimeError('treatment/control word-count mismatch')
    if len(treatment_rows) != len(role_fixed_rows):
        raise RuntimeError('treatment/control text-count mismatch')

    # Text set for provenance includes training rows from both arms and scoring full texts.
    prov_texts = []
    for row in treatment_rows + role_fixed_rows:
        prov_texts.append({'text': row['text']})
    for p in all_pairs:
        prov_texts.append({'text': p['full_text_AB']})
        prov_texts.append({'text': p['full_text_BA']})

    print(f'Generated {len(all_pairs)} scoring pairs from research grammar at N={N_PER_TEMPLATE}', flush=True)
    print(f'Treatment train: {len(treatment_rows)} texts / {sum(r["word_count"] for r in treatment_rows)} words', flush=True)
    print(f'Role-fixed control train: {len(role_fixed_rows)} texts / {sum(r["word_count"] for r in role_fixed_rows)} words', flush=True)
    print(f'Training families: {sorted(TRAIN_FAMILIES)}; excluded from training: {sorted(EXCLUDED_TRAIN_FAMILIES)}', flush=True)

    # Save pair and training files.
    pairs_path = OUT_DIR / 'scoring_pairs.jsonl'
    treat_path = OUT_DIR / 'train_treatment.jsonl'
    fixed_path = OUT_DIR / 'train_role_fixed_control.jsonl'
    with pairs_path.open('w', encoding='utf-8') as f:
        for p in all_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
    for path, rows in [(treat_path, treatment_rows), (fixed_path, role_fixed_rows)]:
        with path.open('w', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print('Running official-text n-gram provenance check...', flush=True)
    prov = research.provenance_check(prov_texts, EVAL_ROOT)
    print('Running WWM density approximation on candidate training rows...', flush=True)
    wwm_treat = simple_wwm_density(treatment_rows, TOK_DIR)
    wwm_fixed = simple_wwm_density(role_fixed_rows, TOK_DIR)

    fam_counts = defaultdict(lambda: {'train': 0, 'held_out': 0})
    for p in all_pairs:
        fam_counts[p['family']][p['style']] += 1
    manifest = {
        'status': 'EXPANDED_ROLE_SWITCH_PACKET_SUITE',
        'seed': SEED,
        'source_step144_script': str(PATH),
        'n_per_template': N_PER_TEMPLATE,
        'total_scoring_pairs': len(all_pairs),
        'family_counts_pairs': {k: dict(v) for k, v in sorted(fam_counts.items())},
        'train_families': sorted(TRAIN_FAMILIES),
        'excluded_train_families': sorted(EXCLUDED_TRAIN_FAMILIES),
        'treatment_train': compute_balance(treatment_rows),
        'role_fixed_control_train': compute_balance(role_fixed_rows),
        'target_counts_treatment': target_counts_in_consequence(treatment_rows, all_pairs),
        'target_counts_role_fixed': target_counts_in_consequence(role_fixed_rows, all_pairs),
        'provenance': prov,
        'wwm_density_treatment': wwm_treat,
        'wwm_density_role_fixed': wwm_fixed,
        'control_semantics': 'role_fixed: same contexts and consequence target slots as treatment; fixed pair alternative is used in both AB/BA directions, alternating by pair_id to balance target counts; this removes context-conditioned role exchange while preserving word count, entities, templates, adjacency, and repetition.',
        'output_files': {
            'scoring_pairs': str(pairs_path),
            'train_treatment': str(treat_path),
            'train_role_fixed_control': str(fixed_path),
        },
        'code_hash': hashlib.sha256(_public_path('experiments/archive/representation_and_objectives/scripts/role_switch_expanded_packet_suite.py').read_bytes()).hexdigest(),
    }
    manifest_path = OUT_DIR / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({
        'event': 'expanded_packets_done',
        'scoring_pairs': len(all_pairs),
        'train_texts': len(treatment_rows),
        'train_words': sum(r['word_count'] for r in treatment_rows),
        'provenance_overlap_7gram': prov.get('7gram', {}).get('overlap_count'),
        'manifest': str(manifest_path),
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
