#!/usr/bin/env python3
"""research — replicate the corrected nonce-state v2 pilot across seeds.

Imports the research v2 implementation and runs the same structured-interface
three-arm comparison over 5 seeds. Saves aggregate results for interpretation.
"""
from __future__ import annotations
import importlib.util, json, pathlib, statistics, sys, time
import torch

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
SRC = ROOT / 'scripts/nonce_state_tracking_v2.py'
OUT = ROOT / 'data/nonce_state_tracking/v2_replicate_5seed.json'

spec = importlib.util.spec_from_file_location('nonce_v2', SRC)
nonce = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = nonce
spec.loader.exec_module(nonce)  # type: ignore


def run_seed(seed: int, device):
    trainset = nonce.dataset(seed, 500, nonce.ENT_TR, nonce.ST_TR, (5, 7), 1)
    splits = {
        'iid': nonce.dataset(1001, 200, nonce.ENT_TR, nonce.ST_TR, (5, 7), 1),
        'new_ent': nonce.dataset(1002, 200, nonce.ENT_HO, nonce.ST_TR, (5, 7), 1),
        'new_state': nonce.dataset(1003, 200, nonce.ENT_TR, nonce.ST_HO, (5, 7), 1),
        'new_both': nonce.dataset(1004, 200, nonce.ENT_HO, nonce.ST_HO, (5, 7), 1),
        'long': nonce.dataset(1005, 200, nonce.ENT_TR, nonce.ST_TR, (9, 12), 3),
        'overwrite': nonce.dataset(1006, 200, nonce.ENT_TR, nonce.ST_TR, (7, 9), 3),
    }
    rec = {}
    for arm in ['endpoint', 'step', 'wess']:
        torch.manual_seed(seed)
        model = (nonce.WESS() if arm == 'wess' else nonce.Endpoint(step=(arm == 'step'))).to(device)
        loss = nonce.train(model, arm, trainset, seed)
        scores = {k: nonce.acc(model, v) for k, v in splits.items()}
        arm_rec = {
            'loss_last': loss,
            'params': sum(p.numel() for p in model.parameters()),
            'accuracy': scores,
        }
        if arm == 'wess':
            arm_rec['interventions'] = nonce.interventions(model, splits['overwrite'])
        rec[arm] = arm_rec
        print(f'seed={seed} arm={arm} {json.dumps(arm_rec, sort_keys=True)}', flush=True)
    return rec


def mean_std(xs):
    return {'mean': sum(xs)/len(xs), 'std': statistics.pstdev(xs), 'values': xs}


def main():
    t0 = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seeds = [42, 123, 456, 789, 2024]
    per_seed = {str(s): run_seed(s, device) for s in seeds}
    splits = ['iid','new_ent','new_state','new_both','long','overwrite']
    arms = ['endpoint','step','wess']
    aggregate = {}
    for arm in arms:
        aggregate[arm] = {}
        for split in splits:
            aggregate[arm][split] = mean_std([per_seed[str(s)][arm]['accuracy'][split] for s in seeds])
        if arm == 'wess':
            aggregate[arm]['slot_swap_transfer'] = mean_std([per_seed[str(s)][arm]['interventions']['slot_swap_transfer'] for s in seeds])
            aggregate[arm]['last_write_ablation_revert'] = mean_std([per_seed[str(s)][arm]['interventions']['last_write_ablation_revert'] for s in seeds])
    payload = {
        'status': 'NONCE_STATE_TRACKING_V2_5SEED',
        'source_script': str(SRC),
        'seeds': seeds,
        'random_baseline': 0.25,
        'interface': 'structured nonce-symbol event stream with gold participant routing for WESS and episode-local four-candidate state retrieval',
        'important_scope_note': 'This tests entity-indexed update/readout as a micro-world mechanism. It is not yet a BabyLM text-pretraining result and not yet a DeBERTa-v2 natural-language interface.',
        'aggregate': aggregate,
        'per_seed': per_seed,
        'elapsed_sec': time.time() - t0,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + '\n')
    print('\nAGGREGATE')
    print(json.dumps(aggregate, indent=2))
    print(f'Saved {OUT}')

if __name__ == '__main__':
    main()
