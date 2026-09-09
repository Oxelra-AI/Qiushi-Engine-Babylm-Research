#!/usr/bin/env python3
"""research: multi-seed replication of the research short WESS-MLM bridge pilot.

Keeps the same balanced evaluation suite and official-text validation slice so arm/seed
variation is interpretable. Varies training episode seed and model/training seed.
This is still a short transfer probe, not a BabyLM-scale candidate.
"""
from __future__ import annotations
import importlib.util, json, pathlib, statistics, sys, time
import torch
from transformers import AutoTokenizer

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
SRC = ROOT / 'scripts/wess_mlm_transfer_pilot.py'
OUT = ROOT / 'data/wess_mlm_transfer_multiseed.json'

spec = importlib.util.spec_from_file_location('bridge', SRC)
bridge = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bridge
spec.loader.exec_module(bridge)  # type: ignore


def mean_std(vals):
    return {'mean': sum(vals)/len(vals), 'std': statistics.pstdev(vals), 'values': vals}


def main():
    bridge.setup_env()
    t0 = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tok = AutoTokenizer.from_pretrained(str(bridge.TOK_PATH), use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token is not None else tok.mask_token

    seeds = [185, 286, 387]
    arms = ['plain_mlm', 'no_address', 'wess_gold', 'wess_eventwise_random', 'wess_wrong_entity']
    # Fixed eval and official validation slices for comparability
    eval_eps = bridge.build_pairs(tok, 120, 2851)
    official_eval = bridge.load_official_texts(tok, 600, 1852)
    audit = bridge.shortcut_audit(eval_eps)

    per_seed = {}
    for seed in seeds:
        print(f'\n=== seed {seed} ===', flush=True)
        train_eps = bridge.build_pairs(tok, 240, seed * 10 + 1)
        official_train = bridge.load_official_texts(tok, 600, seed * 10 + 2)
        per_seed[str(seed)] = {}
        for arm in arms:
            print(f'ARM {arm}', flush=True)
            model = bridge.train_arm(tok, arm, train_eps, official_train, eval_eps, seed=seed, device=device, steps=180, batch=16)
            rec = {
                'binding': bridge.eval_binding(tok, model, eval_eps, device),
                'official_mlm_loss': bridge.eval_official_loss(tok, model, official_eval, device),
                'params': sum(p.numel() for p in model.parameters()),
            }
            if arm != 'plain_mlm':
                rec['interventions'] = bridge.eval_interventions(tok, model, eval_eps[:80], device)
            per_seed[str(seed)][arm] = rec
            print(json.dumps({arm: rec}, indent=2), flush=True)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    aggregate = {}
    for arm in arms:
        aggregate[arm] = {
            'mean_logodds': mean_std([per_seed[str(s)][arm]['binding']['mean_logodds'] for s in seeds]),
            'example_acc': mean_std([per_seed[str(s)][arm]['binding']['example_acc'] for s in seeds]),
            'pair_acc': mean_std([per_seed[str(s)][arm]['binding']['pair_acc'] for s in seeds]),
            'official_mlm_loss': mean_std([per_seed[str(s)][arm]['official_mlm_loss'] for s in seeds]),
        }
        if arm != 'plain_mlm':
            aggregate[arm]['swap_delta'] = mean_std([per_seed[str(s)][arm]['interventions']['swap_logodds_delta_other_minus_answer'] for s in seeds])
            aggregate[arm]['ablation_delta'] = mean_std([per_seed[str(s)][arm]['interventions']['ablation_logodds_delta_prev_minus_answer'] for s in seeds])
            aggregate[arm]['swap_top1'] = mean_std([per_seed[str(s)][arm]['interventions']['swap_top1_to_other'] for s in seeds])
            aggregate[arm]['ablation_top1'] = mean_std([per_seed[str(s)][arm]['interventions']['ablation_top1_to_prev'] for s in seeds])

    payload = {
        'status': 'WESS_MLM_TRANSFER_MULTI_SEED',
        'source_script': str(SRC),
        'description': 'Three-seed replication of the short DeBERTa-v2 WESS MLM transfer bridge. Same eval suite; varied train/init seeds. 180 steps per arm.',
        'seeds': seeds,
        'arms': arms,
        'shortcut_audit_fixed_eval': audit,
        'aggregate': aggregate,
        'per_seed': per_seed,
        'elapsed_sec': round(time.time() - t0, 1),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + '\n')

    print('\nAGGREGATE')
    for arm in arms:
        a = aggregate[arm]
        line = f"{arm}: logodds={a['mean_logodds']['mean']:.4f}±{a['mean_logodds']['std']:.4f}, acc={a['example_acc']['mean']:.4f}, pair={a['pair_acc']['mean']:.4f}, loss={a['official_mlm_loss']['mean']:.4f}"
        if arm != 'plain_mlm':
            line += f", swap={a['swap_delta']['mean']:.4f}, ablate={a['ablation_delta']['mean']:.4f}"
        print(line)
    print(f'Saved {OUT}')

if __name__ == '__main__':
    main()
