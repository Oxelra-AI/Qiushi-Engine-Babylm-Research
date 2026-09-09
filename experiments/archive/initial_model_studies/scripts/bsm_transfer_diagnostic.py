#!/usr/bin/env python3
"""research diagnostic: does BSM binding transfer to official Entity/EWoK?

This is a NON-COMPLIANT diagnostic only (exceeds strict-small budget).
Purpose: determine whether synthetic binding-switch learning transfers to
official BabyLM evaluation columns AT ALL under ideal conditions, before
investing in a compliant from-scratch integration.

If positive: shift to incorporating binding within legal ≤100M exposure.
If negative: synthetic templated binding does not help official evaluation;
the route needs fundamental rethinking.
"""
from __future__ import annotations
import json, os, pathlib, random, subprocess, sys, time
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
sys.path.insert(0, str((ROOT/'scripts').resolve()))
sys.path.insert(0, str((ROOT/'training/scripts').resolve()))

from binding_switch_margin_pretest import (
    FROZEN_CKPT, TRAIN_TEMPLATES, HELDOUT_TEMPLATES, BindingPair,
    build_inventories, generate_pairs, measure_binding_pairs, tokenize_binding_query,
)
from babylm_masked_train import save_hf_checkpoint, make_portable_tokenizer

MODEL_ROOT = ROOT / 'training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/hf_model'
OUT_DIR = ROOT / 'training/runs/bsm_transfer_diagnostic'
OUT_JSON = ROOT / 'data/bsm_transfer_diagnostic.json'
EVAL_REPO = ROOT / 'repos/babylm-eval/strict'


def env_setup():
    hf = ROOT / 'training/hf_home'
    os.environ['HF_HOME'] = str(hf.resolve())
    os.environ['HF_HUB_CACHE'] = str((hf/'hub').resolve())
    os.environ['HF_DATASETS_CACHE'] = str((hf/'datasets').resolve())
    os.environ['TRANSFORMERS_CACHE'] = str((hf/'transformers').resolve())
    os.environ['HF_MODULES_CACHE'] = str((ROOT/'training/hf_modules_cache').resolve())
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def train_binding_only(model, tokenizer, pairs, device, n_epochs=3, batch_size=8, lr=2e-5, seed=42):
    """Train binding-only MLM from research code."""
    mask_id = tokenizer.mask_token_id
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    rng = random.Random(seed)
    for epoch in range(n_epochs):
        epoch_pairs = pairs[:]
        rng.shuffle(epoch_pairs)
        for i in range(0, len(epoch_pairs), batch_size):
            batch = epoch_pairs[i:i+batch_size]
            opt.zero_grad(set_to_none=True)
            losses = []
            for p in batch:
                ids1, mp1 = tokenize_binding_query(tokenizer, p.text_q1, mask_id)
                ids2, mp2 = tokenize_binding_query(tokenizer, p.text_q2, mask_id)
                if not mp1 or not mp2:
                    continue
                lab1 = [-100]*len(ids1); lab1[mp1[0]] = p.v1_tid
                lab2 = [-100]*len(ids2); lab2[mp2[0]] = p.v2_tid
                inp1 = torch.tensor([ids1], device=device)
                inp2 = torch.tensor([ids2], device=device)
                l1 = model(input_ids=inp1, labels=torch.tensor([lab1], device=device)).loss
                l2 = model(input_ids=inp2, labels=torch.tensor([lab2], device=device)).loss
                losses.append(0.5*(l1+l2))
            if losses:
                loss = torch.stack(losses).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
    return model


def run_fast_eval(model_path: str, task_name: str, task_type: str, data_path: str):
    """Run one official fast evaluation task."""
    cmd = [
        sys.executable, str(EVAL_REPO/'evaluation_pipeline/run_eval.py'),
        '--model_path_or_name', model_path,
        '--task', task_type,
        '--task_name', task_name,
        '--data_path', str(EVAL_REPO/data_path),
        '--backend', 'mlm',
        '--output_dir', str(OUT_DIR/'eval_results'),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    # Parse score from output
    score = None
    for line in result.stdout.splitlines():
        if '"score"' in line or '"accuracy"' in line:
            try:
                d = json.loads('{'+line.split('{',1)[-1])
                score = d.get('score', d.get('accuracy'))
            except Exception:
                pass
    # Also check result file
    res_dir = OUT_DIR/'eval_results'
    for f in res_dir.rglob('*.json'):
        try:
            data = json.loads(f.read_text())
            if isinstance(data, dict) and 'score' in data:
                score = data['score']
        except Exception:
            pass
    return {'task': task_name, 'score': score, 'returncode': result.returncode,
            'stdout_tail': result.stdout[-500:] if result.stdout else '',
            'stderr_tail': result.stderr[-500:] if result.stderr else ''}


def main():
    env_setup()
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build model and binding pairs
    ckpt_path = MODEL_ROOT / 'chck_100M'
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_path.resolve()), use_fast=True)
    inv = build_inventories(tokenizer)
    rng = random.Random(42)
    train_pairs = generate_pairs(
        {'entities': inv['train_entities'], 'values': inv['train_values']},
        TRAIN_TEMPLATES, 'train', rng, n_pairs=200
    )

    # Train BSM model (100% binding, ~20k words)
    print(json.dumps({'event': 'training_bsm_model', 'n_pairs': len(train_pairs), 'epochs': 3}), flush=True)
    model = AutoModelForMaskedLM.from_pretrained(str(ckpt_path.resolve()), trust_remote_code=True).to(device)
    model = train_binding_only(model, tokenizer, train_pairs, device, n_epochs=3, batch_size=8, lr=2e-5, seed=42)

    # Measure binding (quick sanity)
    eval_pairs = generate_pairs(
        {'entities': inv['heldout_entities'], 'values': inv['train_values']},
        HELDOUT_TEMPLATES, 'heldout', rng, n_pairs=40
    )
    binding_agg, _ = measure_binding_pairs(model, tokenizer, eval_pairs, device)
    print(json.dumps({'event': 'binding_check', **binding_agg}), flush=True)

    # Save checkpoint for evaluation
    save_path = OUT_DIR / 'hf_model'
    save_hf_checkpoint(model, tokenizer, save_path)
    print(json.dumps({'event': 'saved', 'path': str(save_path)}), flush=True)
    del model
    torch.cuda.empty_cache()

    # Also evaluate the ORIGINAL 100M checkpoint as baseline
    baseline_path = str(ckpt_path.resolve())
    bsm_path = str(save_path.resolve())

    # Fast evaluations: Entity Tracking, EWoK, BLiMP, Supplement
    TASKS = [
        ('entity_tracking_fast', 'entity_tracking', 'evaluation_data/fast_eval/entity_tracking_fast'),
        ('ewok_fast', 'ewok', 'evaluation_data/fast_eval/evaluation_data/fast_eval/ewok_fast'),
        ('blimp_fast', 'blimp', 'evaluation_data/fast_eval/blimp_fast'),
        ('supplement_fast', 'blimp', 'evaluation_data/fast_eval/supplement_fast'),
    ]

    results = {'baseline': {}, 'bsm': {}}
    for name, ttype, dpath in TASKS:
        print(json.dumps({'event': 'eval_start', 'model': 'baseline', 'task': name}), flush=True)
        results['baseline'][name] = run_fast_eval(baseline_path, name, ttype, dpath)
        print(json.dumps({'event': 'eval_start', 'model': 'bsm', 'task': name}), flush=True)
        results['bsm'][name] = run_fast_eval(bsm_path, name, ttype, dpath)

    # Summary
    deltas = {}
    for name in [t[0] for t in TASKS]:
        bs = results['baseline'][name].get('score')
        bsm_s = results['bsm'][name].get('score')
        if bs is not None and bsm_s is not None:
            deltas[name] = bsm_s - bs
        else:
            deltas[name] = None

    payload = {
        'status': 'BSM_TRANSFER_DIAGNOSTIC',
        'note': 'NON-COMPLIANT diagnostic only. Model exceeds strict-small budget.',
        'binding_check': binding_agg,
        'baseline_ckpt': baseline_path,
        'bsm_ckpt': bsm_path,
        'results': results,
        'deltas': deltas,
        'elapsed_sec': time.time() - t0,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'event': 'done', 'deltas': deltas, 'binding_heldout': binding_agg['both_correct_frac']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
