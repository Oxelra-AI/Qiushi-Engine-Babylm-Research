#!/usr/bin/env python3
"""research Binding-Switch Margin (BSM) pre-scale measurement.

Tests whether the ordinary DeBERTa MLM interface can learn to make the query
entity addressable: in a context with two entity-value bindings, does the model
switch its masked-position preference when only the queried entity changes?

Four arms from the same base checkpoint:
  1. base:        frozen 100M checkpoint, no fine-tuning
  2. mlm_only:    fine-tune on same texts with standard MLM, no binding margin
  3. bsm:         fine-tune with MLM + binding-switch margin loss
  4. random_corr: same BSM structure but entity-value labels randomized

Zero added inference parameters. Uses the research semantic-token verification rule.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, math, os, pathlib, random, sys, time
from dataclasses import dataclass, field
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
FROZEN_CKPT = ROOT / 'training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/hf_model/chck_100M'
OUT_DIR = ROOT / 'training/runs/bsm_pretest'
OUT_JSON = ROOT / 'data/bsm_pretest_results.json'
OUT_NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/bsm_pretest_results.md')


# ─── Entity/Value inventory ───────────────────────────────────────────────────

def verify_single_semantic_token(tokenizer, word: str):
    """research rule: tokenizer(' '+word) must be [space_marker, single_semantic_word_token]."""
    ids = tokenizer(' ' + word, add_special_tokens=False)['input_ids']
    toks = tokenizer.convert_ids_to_tokens(ids)
    if len(ids) == 2 and toks[0] in ('Ġ', '▁'):
        surf = toks[1].replace('Ġ', '').replace('▁', '').lower()
        if surf == word.lower():
            return ids[1]
    if len(ids) == 1:
        surf = toks[0].replace('Ġ', '').replace('▁', '').lower()
        if surf == word.lower():
            return ids[0]
    return None


def build_inventories(tokenizer):
    """Build verified entity/value inventories with train/held-out splits."""
    # Entity candidates (proper names and common nouns as entity markers)
    entity_candidates = [
        'Alice', 'Bob', 'John', 'Mary', 'Tom', 'Sarah', 'David', 'Emma',
        'Peter', 'Lisa', 'Mike', 'Kate', 'Jack', 'Anna', 'Sam', 'Jane',
    ]
    # Value candidates (concrete nouns, locations, states, colors)
    value_candidates = [
        'hat', 'ball', 'fish', 'cake', 'milk', 'book', 'apple', 'bread',
        'coffee', 'tea', 'Paris', 'London', 'Rome', 'fast', 'slow',
        'red', 'blue', 'green', 'north', 'south', 'home', 'school',
        'river', 'forest', 'doctor', 'teacher', 'gold', 'silver',
    ]

    entities = {}
    for w in entity_candidates:
        tid = verify_single_semantic_token(tokenizer, w)
        if tid is not None:
            entities[w] = tid
    values = {}
    for w in value_candidates:
        tid = verify_single_semantic_token(tokenizer, w)
        if tid is not None and tid not in entities.values():
            values[w] = tid

    # Split: first 60% train, last 40% held-out
    ent_list = list(entities.items())
    val_list = list(values.items())
    ent_split = max(4, int(len(ent_list) * 0.6))
    val_split = max(6, int(len(val_list) * 0.6))

    return {
        'train_entities': dict(ent_list[:ent_split]),
        'heldout_entities': dict(ent_list[ent_split:]),
        'train_values': dict(val_list[:val_split]),
        'heldout_values': dict(val_list[val_split:]),
    }


# ─── Template system ──────────────────────────────────────────────────────────

TRAIN_TEMPLATES = [
    ("{E1} found a {V1}. {E2} found a {V2}. {Eq} lost the [MASK] yesterday.",
     "{E1} found a {V1}. {E2} found a {V2}. {Eq} lost the [MASK] yesterday."),
    ("{E1} brought {V1}. {E2} brought {V2}. {Eq} shared the [MASK] later.",
     "{E1} brought {V1}. {E2} brought {V2}. {Eq} shared the [MASK] later."),
    ("{E1} bought a {V1}. {E2} bought a {V2}. {Eq} returned the [MASK] today.",
     "{E1} bought a {V1}. {E2} bought a {V2}. {Eq} returned the [MASK] today."),
    ("{E1} likes {V1}. {E2} likes {V2}. {Eq} chose {V1} over everything.",
     "{E1} likes {V1}. {E2} likes {V2}. {Eq} chose [MASK] over everything."),
    ("{E1} has a {V1}. {E2} has a {V2}. {Eq} dropped the [MASK] on the floor.",
     "{E1} has a {V1}. {E2} has a {V2}. {Eq} dropped the [MASK] on the floor."),
    ("{E1} picked up the {V1}. {E2} picked up the {V2}. {Eq} kept the [MASK].",
     "{E1} picked up the {V1}. {E2} picked up the {V2}. {Eq} kept the [MASK]."),
]

HELDOUT_TEMPLATES = [
    ("{E1} carried a {V1}. {E2} carried a {V2}. {Eq} dropped the [MASK].",
     "{E1} carried a {V1}. {E2} carried a {V2}. {Eq} dropped the [MASK]."),
    ("{E1} ordered {V1}. {E2} ordered {V2}. {Eq} received the [MASK] first.",
     "{E1} ordered {V1}. {E2} ordered {V2}. {Eq} received the [MASK] first."),
    ("{E1} wanted {V1}. {E2} wanted {V2}. Later {Eq} got the [MASK].",
     "{E1} wanted {V1}. {E2} wanted {V2}. Later {Eq} got the [MASK]."),
]


@dataclass
class BindingPair:
    """A single binding-switch pair: context with E1→V1, E2→V2, two query directions."""
    text_q1: str  # query about E1 (answer should be V1)
    text_q2: str  # query about E2 (answer should be V2)
    e1: str
    e2: str
    v1: str
    v2: str
    v1_tid: int
    v2_tid: int
    split: str  # 'train' or one of the heldout split names
    template_idx: int
    order_flipped: bool = False


def generate_pairs(inventories, templates, split_name, rng, n_pairs=100, flip_order=False):
    """Generate binding pairs from entity/value inventories and templates."""
    ents = list(inventories['entities'].items())
    vals = list(inventories['values'].items())
    pairs = []
    for _ in range(n_pairs):
        if len(ents) < 2 or len(vals) < 2:
            break
        e1_name, _ = rng.choice(ents)
        e2_name, _ = rng.choice([e for e in ents if e[0] != e1_name])
        v1_name, v1_tid = rng.choice(vals)
        v2_name, v2_tid = rng.choice([v for v in vals if v[0] != v1_name])
        tidx = rng.randrange(len(templates))
        tmpl = templates[tidx][0]

        if flip_order:
            # Swap entity order in context but keep query→answer mapping
            text_q1 = tmpl.format(E1=e2_name, E2=e1_name, V1=v2_name, V2=v1_name, Eq=e1_name)
            text_q2 = tmpl.format(E1=e2_name, E2=e1_name, V1=v2_name, V2=v1_name, Eq=e2_name)
        else:
            text_q1 = tmpl.format(E1=e1_name, E2=e2_name, V1=v1_name, V2=v2_name, Eq=e1_name)
            text_q2 = tmpl.format(E1=e1_name, E2=e2_name, V1=v1_name, V2=v2_name, Eq=e2_name)

        pairs.append(BindingPair(
            text_q1=text_q1, text_q2=text_q2,
            e1=e1_name, e2=e2_name, v1=v1_name, v2=v2_name,
            v1_tid=v1_tid, v2_tid=v2_tid,
            split=split_name, template_idx=tidx, order_flipped=flip_order
        ))
    return pairs


# ─── Tokenization and measurement ────────────────────────────────────────────

def tokenize_binding_query(tokenizer, text: str, mask_id: int):
    """Tokenize text with [MASK] placeholder."""
    parts = text.split('[MASK]')
    ids = []
    for i, part in enumerate(parts):
        ids.extend(tokenizer(part, add_special_tokens=False)['input_ids'])
        if i < len(parts) - 1:
            ids.append(mask_id)
    mask_positions = [i for i, x in enumerate(ids) if x == mask_id]
    return ids, mask_positions


def measure_binding_pairs(model, tokenizer, pairs: List[BindingPair], device):
    """Measure pair-level binding discrimination."""
    model.eval()
    mask_id = tokenizer.mask_token_id
    results = []

    with torch.no_grad():
        for p in pairs:
            ids_q1, mpos_q1 = tokenize_binding_query(tokenizer, p.text_q1, mask_id)
            ids_q2, mpos_q2 = tokenize_binding_query(tokenizer, p.text_q2, mask_id)

            if not mpos_q1 or not mpos_q2:
                continue

            # Verify both value tokens appear in context before mask
            pre_mask_q1 = set(ids_q1[:mpos_q1[0]])
            pre_mask_q2 = set(ids_q2[:mpos_q2[0]])
            if p.v1_tid not in pre_mask_q1 or p.v2_tid not in pre_mask_q1:
                continue
            if p.v1_tid not in pre_mask_q2 or p.v2_tid not in pre_mask_q2:
                continue

            # Forward pass for q1
            inp_q1 = torch.tensor([ids_q1], device=device)
            logits_q1 = model(input_ids=inp_q1).logits[0, mpos_q1[0]]

            # Forward pass for q2
            inp_q2 = torch.tensor([ids_q2], device=device)
            logits_q2 = model(input_ids=inp_q2).logits[0, mpos_q2[0]]

            # Margins: correct - distractor
            margin_q1 = (logits_q1[p.v1_tid] - logits_q1[p.v2_tid]).item()
            margin_q2 = (logits_q2[p.v2_tid] - logits_q2[p.v1_tid]).item()

            results.append({
                'e1': p.e1, 'e2': p.e2, 'v1': p.v1, 'v2': p.v2,
                'split': p.split, 'order_flipped': p.order_flipped,
                'margin_q1': margin_q1, 'margin_q2': margin_q2,
                'both_correct': margin_q1 > 0 and margin_q2 > 0,
                'same_value_pref': (margin_q1 > 0) == (margin_q2 < 0),  # prefers same value both directions
            })

    # Aggregate
    if results:
        n = len(results)
        both_correct_frac = sum(r['both_correct'] for r in results) / n
        same_value_pref_frac = sum(r['same_value_pref'] for r in results) / n
        mean_margin_q1 = sum(r['margin_q1'] for r in results) / n
        mean_margin_q2 = sum(r['margin_q2'] for r in results) / n
    else:
        n = 0
        both_correct_frac = 0
        same_value_pref_frac = 0
        mean_margin_q1 = 0
        mean_margin_q2 = 0

    return {
        'n_pairs': n,
        'both_correct_frac': both_correct_frac,
        'same_value_pref_frac': same_value_pref_frac,
        'mean_margin_q1': mean_margin_q1,
        'mean_margin_q2': mean_margin_q2,
        'mean_margin_overall': (mean_margin_q1 + mean_margin_q2) / 2 if n > 0 else 0,
    }, results


# ─── BSM training ────────────────────────────────────────────────────────────

def bsm_training_step(model, tokenizer, pairs: List[BindingPair], device,
                       mode='bsm', margin_target=1.0, lambda_bsm=1.0, rng=None):
    """One training step: MLM on full text + optional binding-switch margin."""
    mask_id = tokenizer.mask_token_id
    total_mlm_loss = 0.0
    total_bsm_loss = 0.0
    n_valid = 0

    for p in pairs:
        ids_q1, mpos_q1 = tokenize_binding_query(tokenizer, p.text_q1, mask_id)
        ids_q2, mpos_q2 = tokenize_binding_query(tokenizer, p.text_q2, mask_id)
        if not mpos_q1 or not mpos_q2:
            continue

        # MLM targets: the correct value at mask position
        labels_q1 = [-100] * len(ids_q1)
        labels_q1[mpos_q1[0]] = p.v1_tid
        labels_q2 = [-100] * len(ids_q2)
        labels_q2[mpos_q2[0]] = p.v2_tid

        # Forward q1
        inp_q1 = torch.tensor([ids_q1], device=device)
        lab_q1 = torch.tensor([labels_q1], device=device)
        out_q1 = model(input_ids=inp_q1, labels=lab_q1)
        mlm_loss_q1 = out_q1.loss

        # Forward q2
        inp_q2 = torch.tensor([ids_q2], device=device)
        lab_q2 = torch.tensor([labels_q2], device=device)
        out_q2 = model(input_ids=inp_q2, labels=lab_q2)
        mlm_loss_q2 = out_q2.loss

        total_mlm_loss = total_mlm_loss + mlm_loss_q1 + mlm_loss_q2

        # BSM margin loss
        if mode in ('bsm', 'random_corr'):
            logits_q1 = out_q1.logits[0, mpos_q1[0]]
            logits_q2 = out_q2.logits[0, mpos_q2[0]]

            if mode == 'random_corr':
                # Randomize which value should be preferred
                if rng and rng.random() < 0.5:
                    correct_q1, distract_q1 = p.v2_tid, p.v1_tid
                    correct_q2, distract_q2 = p.v1_tid, p.v2_tid
                else:
                    correct_q1, distract_q1 = p.v1_tid, p.v2_tid
                    correct_q2, distract_q2 = p.v2_tid, p.v1_tid
            else:
                correct_q1, distract_q1 = p.v1_tid, p.v2_tid
                correct_q2, distract_q2 = p.v2_tid, p.v1_tid

            margin_q1 = logits_q1[correct_q1] - logits_q1[distract_q1]
            margin_q2 = logits_q2[correct_q2] - logits_q2[distract_q2]
            bsm_loss = (F.relu(margin_target - margin_q1) + F.relu(margin_target - margin_q2))
            total_bsm_loss = total_bsm_loss + bsm_loss

        n_valid += 1

    if n_valid == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), 0, 0

    loss = total_mlm_loss / n_valid
    if mode in ('bsm', 'random_corr'):
        loss = loss + lambda_bsm * total_bsm_loss / n_valid

    return loss, float(total_mlm_loss.detach().cpu()) / n_valid, float(total_bsm_loss.detach().cpu()) / n_valid if isinstance(total_bsm_loss, torch.Tensor) else 0.0


def train_arm(model, tokenizer, train_pairs, device, mode, n_epochs=3,
              batch_size=8, lr=2e-5, margin_target=1.0, lambda_bsm=1.0, seed=42):
    """Train one arm and return the model."""
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    rng = random.Random(seed)
    logs = []

    for epoch in range(n_epochs):
        epoch_pairs = train_pairs[:]
        rng.shuffle(epoch_pairs)
        for i in range(0, len(epoch_pairs), batch_size):
            batch = epoch_pairs[i:i+batch_size]
            opt.zero_grad(set_to_none=True)
            loss, mlm_loss, bsm_loss = bsm_training_step(
                model, tokenizer, batch, device, mode=mode,
                margin_target=margin_target, lambda_bsm=lambda_bsm, rng=rng
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if i == 0 or (i // batch_size) % 10 == 0:
                logs.append({'epoch': epoch, 'batch': i // batch_size,
                             'loss': float(loss.detach().cpu()),
                             'mlm_loss': mlm_loss, 'bsm_loss': bsm_loss})

    return model, logs


# ─── Main ─────────────────────────────────────────────────────────────────────

def env_setup():
    hf = ROOT / 'training/hf_home'
    os.environ['HF_HOME'] = str(hf.resolve())
    os.environ['HF_HUB_CACHE'] = str((hf / 'hub').resolve())
    os.environ['HF_DATASETS_CACHE'] = str((hf / 'datasets').resolve())
    os.environ['TRANSFORMERS_CACHE'] = str((hf / 'transformers').resolve())
    os.environ['HF_MODULES_CACHE'] = str((ROOT / 'training/hf_modules_cache').resolve())
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_train_pairs', type=int, default=200)
    ap.add_argument('--n_heldout_pairs', type=int, default=80)
    ap.add_argument('--n_epochs', type=int, default=3)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--margin_target', type=float, default=1.0)
    ap.add_argument('--lambda_bsm', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    env_setup()
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load tokenizer and build inventories
    tokenizer = AutoTokenizer.from_pretrained(str(FROZEN_CKPT.resolve()), use_fast=True)
    inv = build_inventories(tokenizer)
    print(json.dumps({
        'event': 'inventories',
        'train_entities': list(inv['train_entities'].keys()),
        'heldout_entities': list(inv['heldout_entities'].keys()),
        'train_values': list(inv['train_values'].keys()),
        'heldout_values': list(inv['heldout_values'].keys()),
    }), flush=True)

    # Generate pairs
    rng = random.Random(args.seed)
    train_pairs = generate_pairs(
        {'entities': inv['train_entities'], 'values': inv['train_values']},
        TRAIN_TEMPLATES, 'train', rng, n_pairs=args.n_train_pairs
    )
    # Held-out splits
    heldout_ent_pairs = generate_pairs(
        {'entities': inv['heldout_entities'], 'values': inv['train_values']},
        TRAIN_TEMPLATES, 'heldout_entities', rng, n_pairs=args.n_heldout_pairs
    )
    heldout_val_pairs = generate_pairs(
        {'entities': inv['train_entities'], 'values': inv['heldout_values']},
        TRAIN_TEMPLATES, 'heldout_values', rng, n_pairs=args.n_heldout_pairs
    )
    heldout_tmpl_pairs = generate_pairs(
        {'entities': inv['train_entities'], 'values': inv['train_values']},
        HELDOUT_TEMPLATES, 'heldout_templates', rng, n_pairs=args.n_heldout_pairs
    )
    heldout_order_pairs = generate_pairs(
        {'entities': inv['train_entities'], 'values': inv['train_values']},
        TRAIN_TEMPLATES, 'heldout_order_flip', rng, n_pairs=args.n_heldout_pairs,
        flip_order=True
    )

    all_eval_sets = {
        'train': train_pairs,
        'heldout_entities': heldout_ent_pairs,
        'heldout_values': heldout_val_pairs,
        'heldout_templates': heldout_tmpl_pairs,
        'heldout_order_flip': heldout_order_pairs,
    }
    print(json.dumps({'event': 'pairs_generated',
                      'train': len(train_pairs),
                      'heldout_entities': len(heldout_ent_pairs),
                      'heldout_values': len(heldout_val_pairs),
                      'heldout_templates': len(heldout_tmpl_pairs),
                      'heldout_order_flip': len(heldout_order_pairs)}), flush=True)

    # Run four arms
    arms = ['base', 'mlm_only', 'bsm', 'random_corr']
    arm_results = {}

    for arm_name in arms:
        print(f'\n===== ARM: {arm_name} =====', flush=True)

        # Load fresh model for each arm
        model = AutoModelForMaskedLM.from_pretrained(
            str(FROZEN_CKPT.resolve()), trust_remote_code=True
        ).to(device)

        train_logs = []
        if arm_name != 'base':
            mode_map = {'mlm_only': 'mlm_only', 'bsm': 'bsm', 'random_corr': 'random_corr'}
            # For mlm_only, we still pass through bsm_training_step but mode='mlm_only'
            # which means no BSM loss is applied
            if arm_name == 'mlm_only':
                # Simple MLM fine-tuning: use the BSM framework with lambda_bsm=0
                model, train_logs = train_arm(
                    model, tokenizer, train_pairs, device,
                    mode='bsm', n_epochs=args.n_epochs, batch_size=args.batch_size,
                    lr=args.lr, margin_target=args.margin_target, lambda_bsm=0.0,
                    seed=args.seed
                )
            else:
                model, train_logs = train_arm(
                    model, tokenizer, train_pairs, device,
                    mode=arm_name, n_epochs=args.n_epochs, batch_size=args.batch_size,
                    lr=args.lr, margin_target=args.margin_target,
                    lambda_bsm=args.lambda_bsm, seed=args.seed
                )

        # Measure on all eval sets
        arm_evals = {}
        for eval_name, eval_pairs in all_eval_sets.items():
            agg, _ = measure_binding_pairs(model, tokenizer, eval_pairs, device)
            arm_evals[eval_name] = agg
            print(json.dumps({'event': 'eval', 'arm': arm_name, 'split': eval_name, **agg}), flush=True)

        arm_results[arm_name] = {
            'train_logs': train_logs[-5:] if train_logs else [],
            'evaluations': arm_evals,
        }

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # Interpretation
    base_train = arm_results['base']['evaluations']['train']['both_correct_frac']
    mlm_train = arm_results['mlm_only']['evaluations']['train']['both_correct_frac']
    bsm_train = arm_results['bsm']['evaluations']['train']['both_correct_frac']
    rand_train = arm_results['random_corr']['evaluations']['train']['both_correct_frac']

    bsm_heldout_ent = arm_results['bsm']['evaluations'].get('heldout_entities', {}).get('both_correct_frac', 0)
    bsm_heldout_val = arm_results['bsm']['evaluations'].get('heldout_values', {}).get('both_correct_frac', 0)
    bsm_heldout_tmpl = arm_results['bsm']['evaluations'].get('heldout_templates', {}).get('both_correct_frac', 0)
    bsm_heldout_order = arm_results['bsm']['evaluations'].get('heldout_order_flip', {}).get('both_correct_frac', 0)

    interp = {
        'bsm_learns_binding_on_train': bsm_train > base_train + 0.1 and bsm_train > mlm_train + 0.05,
        'bsm_exceeds_random_corr': bsm_train > rand_train + 0.05,
        'bsm_transfers_to_heldout_entities': bsm_heldout_ent > base_train + 0.05,
        'bsm_transfers_to_heldout_values': bsm_heldout_val > base_train + 0.05,
        'bsm_transfers_to_heldout_templates': bsm_heldout_tmpl > base_train + 0.05,
        'bsm_transfers_to_order_flips': bsm_heldout_order > base_train + 0.05,
        'bsm_route_viable': (bsm_train > 0.6) and (bsm_heldout_ent > 0.4) and (bsm_train > rand_train + 0.05),
    }

    payload = {
        'status': 'BSM_PRETEST',
        'frozen_ckpt': str(FROZEN_CKPT),
        'params': vars(args),
        'inventories': {k: list(v.keys()) for k, v in inv.items()},
        'pair_counts': {k: len(v) for k, v in all_eval_sets.items()},
        'arm_results': arm_results,
        'interpretation': interp,
        'elapsed_sec': time.time() - t0,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    # Note
    lines = [
        '# research — BSM pre-scale measurement results', '',
        f'Evidence: `{OUT_JSON}`', '',
        '## Train-set both_correct_frac by arm', '',
        f'| arm | both_correct | same_value_pref | mean_margin |',
        f'|---|---:|---:|---:|',
    ]
    for arm in arms:
        e = arm_results[arm]['evaluations']['train']
        lines.append(f"| {arm} | {e['both_correct_frac']:.3f} | {e['same_value_pref_frac']:.3f} | {e['mean_margin_overall']:+.3f} |")
    lines += ['', '## BSM held-out transfer', '',
              '| split | both_correct | mean_margin |', '|---|---:|---:|']
    for split in ['heldout_entities', 'heldout_values', 'heldout_templates', 'heldout_order_flip']:
        e = arm_results['bsm']['evaluations'].get(split, {})
        lines.append(f"| {split} | {e.get('both_correct_frac', 0):.3f} | {e.get('mean_margin_overall', 0):+.3f} |")
    lines += ['', '## Interpretation', '', json.dumps(interp, indent=2)]
    OUT_NOTE.parent.mkdir(parents=True, exist_ok=True)
    OUT_NOTE.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    print(json.dumps({'event': 'done', 'interpretation': interp,
                      'elapsed_sec': payload['elapsed_sec']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
