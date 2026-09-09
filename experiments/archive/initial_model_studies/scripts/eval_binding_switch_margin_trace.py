#!/usr/bin/env python3
"""research: continuous binding-switch margin evaluator.

Goes beyond pair-level both-correct by measuring continuous logit margins
between correct and distractor values. Reports per-split:
  - mean/median margin
  - bootstrap 95% CI
  - switch-direction accuracy (fraction where margin > 0)
  - pair-level both-correct
  - same-value preference
  - correct-value and distractor-value mean log-probability

Designed to detect sub-threshold relation learning that binary probes miss.
"""
from __future__ import annotations
import argparse, json, math, os, pathlib, random, sys, time
from typing import List, Dict, Any

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
sys.path.insert(0, str((ROOT / 'scripts').resolve()))

from binding_switch_margin_pretest import (
    TRAIN_TEMPLATES, HELDOUT_TEMPLATES, BindingPair,
    build_inventories, generate_pairs, tokenize_binding_query,
)
from bsm_density_persistence import NATURAL_TEMPLATES


def env_setup():
    hf = ROOT / 'training/hf_home'
    os.environ['HF_HOME'] = str(hf.resolve())
    os.environ['HF_HUB_CACHE'] = str((hf / 'hub').resolve())
    os.environ['HF_DATASETS_CACHE'] = str((hf / 'datasets').resolve())
    os.environ['TRANSFORMERS_CACHE'] = str((hf / 'transformers').resolve())
    os.environ['HF_MODULES_CACHE'] = str((ROOT / 'training/hf_modules_cache').resolve())
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def measure_continuous_margins(model, tokenizer, pairs: List[BindingPair], device):
    """Measure continuous switch margins for a set of binding pairs.
    
    For each pair (context with E1→V1, E2→V2, two query directions):
      margin_q1 = logit(V1) - logit(V2) at query-E1 mask position
      margin_q2 = logit(V2) - logit(V1) at query-E2 mask position
    """
    model.eval()
    mask_id = tokenizer.mask_token_id
    margins_q1 = []
    margins_q2 = []
    lp_correct = []
    lp_distractor = []

    with torch.no_grad():
        for p in pairs:
            ids_q1, mpos_q1 = tokenize_binding_query(tokenizer, p.text_q1, mask_id)
            ids_q2, mpos_q2 = tokenize_binding_query(tokenizer, p.text_q2, mask_id)
            if not mpos_q1 or not mpos_q2:
                continue

            # Forward q1
            inp_q1 = torch.tensor([ids_q1], device=device)
            logits_q1 = model(input_ids=inp_q1).logits[0, mpos_q1[0]]
            lp_q1 = torch.log_softmax(logits_q1, dim=-1)

            # Forward q2
            inp_q2 = torch.tensor([ids_q2], device=device)
            logits_q2 = model(input_ids=inp_q2).logits[0, mpos_q2[0]]
            lp_q2 = torch.log_softmax(logits_q2, dim=-1)

            # Margins: correct - distractor
            m1 = (logits_q1[p.v1_tid] - logits_q1[p.v2_tid]).item()
            m2 = (logits_q2[p.v2_tid] - logits_q2[p.v1_tid]).item()
            margins_q1.append(m1)
            margins_q2.append(m2)

            # Log-probabilities
            lp_correct.append(lp_q1[p.v1_tid].item())
            lp_correct.append(lp_q2[p.v2_tid].item())
            lp_distractor.append(lp_q1[p.v2_tid].item())
            lp_distractor.append(lp_q2[p.v1_tid].item())

    if not margins_q1:
        return {'n_pairs': 0}

    n = len(margins_q1)
    all_margins = margins_q1 + margins_q2
    arr = np.array(all_margins)

    # Bootstrap 95% CI for mean margin
    n_boot = 1000
    boot_means = []
    rng_np = np.random.default_rng(42)
    for _ in range(n_boot):
        sample = rng_np.choice(arr, size=len(arr), replace=True)
        boot_means.append(float(sample.mean()))
    boot_means.sort()
    ci_lo = boot_means[int(0.025 * n_boot)]
    ci_hi = boot_means[int(0.975 * n_boot)]

    # Pair-level metrics
    both_correct = sum(1 for m1, m2 in zip(margins_q1, margins_q2) if m1 > 0 and m2 > 0)
    same_value_pref = sum(1 for m1, m2 in zip(margins_q1, margins_q2)
                          if (m1 > 0) == (m2 < 0))  # prefers same value both directions

    return {
        'n_pairs': n,
        'mean_margin': float(arr.mean()),
        'median_margin': float(np.median(arr)),
        'std_margin': float(arr.std()),
        'ci_95_lo': ci_lo,
        'ci_95_hi': ci_hi,
        'switch_direction_accuracy': float(np.mean(arr > 0)),
        'both_correct_frac': both_correct / n,
        'same_value_pref_frac': same_value_pref / n,
        'mean_lp_correct': float(np.mean(lp_correct)),
        'mean_lp_distractor': float(np.mean(lp_distractor)),
        'mean_lp_diff': float(np.mean(lp_correct) - np.mean(lp_distractor)),
        'margin_q1_mean': float(np.mean(margins_q1)),
        'margin_q2_mean': float(np.mean(margins_q2)),
    }


def make_probe_sets(tokenizer, seed=42, n=80):
    inv = build_inventories(tokenizer)
    rng = random.Random(seed)
    return {
        'train': generate_pairs({'entities': inv['train_entities'], 'values': inv['train_values']},
                                TRAIN_TEMPLATES, 'train', rng, n_pairs=n),
        'heldout_entities': generate_pairs({'entities': inv['heldout_entities'], 'values': inv['train_values']},
                                           TRAIN_TEMPLATES, 'heldout_entities', rng, n_pairs=n),
        'heldout_values': generate_pairs({'entities': inv['train_entities'], 'values': inv['heldout_values']},
                                         TRAIN_TEMPLATES, 'heldout_values', rng, n_pairs=n),
        'heldout_templates': generate_pairs({'entities': inv['train_entities'], 'values': inv['train_values']},
                                            HELDOUT_TEMPLATES, 'heldout_templates', rng, n_pairs=n),
        'heldout_order_flip': generate_pairs({'entities': inv['train_entities'], 'values': inv['train_values']},
                                             TRAIN_TEMPLATES, 'heldout_order_flip', rng, n_pairs=n, flip_order=True),
        'natural_templates': generate_pairs({'entities': inv['train_entities'], 'values': inv['train_values']},
                                            NATURAL_TEMPLATES, 'natural_templates', rng, n_pairs=n),
    }


def evaluate_model(model_path: str, device, probe_sets=None, tokenizer=None):
    """Evaluate one model on all probe sets with continuous margins."""
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if probe_sets is None:
        probe_sets = make_probe_sets(tokenizer)
    model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True).to(device)
    results = {}
    for split_name, pairs in probe_sets.items():
        results[split_name] = measure_continuous_margins(model, tokenizer, pairs, device)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_paths', nargs='+', required=True, help='One or more model paths to evaluate')
    ap.add_argument('--model_names', nargs='+', required=True, help='Corresponding names for each model')
    ap.add_argument('--out_json', required=True)
    ap.add_argument('--n_pairs', type=int, default=80)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    env_setup()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t0 = time.time()

    # Load tokenizer from first model for probe generation
    tokenizer = AutoTokenizer.from_pretrained(args.model_paths[0], use_fast=True)
    probe_sets = make_probe_sets(tokenizer, seed=args.seed, n=args.n_pairs)

    results = {}
    for name, path in zip(args.model_names, args.model_paths):
        print(json.dumps({'event': 'eval_start', 'model': name}), flush=True)
        results[name] = evaluate_model(path, device, probe_sets, tokenizer)
        # Print summary for this model
        train_m = results[name]['train']['mean_margin']
        nat_m = results[name]['natural_templates']['mean_margin']
        train_bc = results[name]['train']['both_correct_frac']
        print(json.dumps({'event': 'eval_done', 'model': name,
                          'train_margin': round(train_m, 4), 'natural_margin': round(nat_m, 4),
                          'train_both_correct': round(train_bc, 3)}), flush=True)

    # Compute deltas if there are at least two models
    deltas = {}
    names = args.model_names
    if len(names) >= 2:
        for split in probe_sets:
            for metric in ['mean_margin', 'both_correct_frac', 'switch_direction_accuracy']:
                for i in range(len(names)):
                    for j in range(i+1, len(names)):
                        key = f"{names[i]}_minus_{names[j]}_{split}_{metric}"
                        vi = results[names[i]][split].get(metric)
                        vj = results[names[j]][split].get(metric)
                        if vi is not None and vj is not None:
                            deltas[key] = vi - vj

    payload = {
        'status': 'CONTINUOUS_MARGIN_EVAL',
        'model_paths': dict(zip(args.model_names, args.model_paths)),
        'probe_counts': {k: len(v) for k, v in probe_sets.items()},
        'results': results,
        'deltas': deltas,
        'elapsed_sec': time.time() - t0,
    }

    out_path = pathlib.Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'event': 'saved', 'path': str(out_path)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
