#!/usr/bin/env python3
"""Aggregate two-seed 2x2 structured-experience screen results.

Reads per-seed zero-shot/reading JSON files (from eval_2x2_seed.py or
eval_2x2_seed42.py) and optional SuperGLUE JSONs, then computes per-seed
and mean factorial effects. This script deliberately reports evidence; it does
not decide whether to scale the route.
"""
from __future__ import annotations
import argparse, json, pathlib, statistics

ARMS = ['A_official_wwm','B_structured_wwm','C_official_amlm','D_structured_amlm']
CONTRASTS = ['data_effect_wwm_BminusA','data_effect_amlm_DminusC','amlm_effect_official_CminusA','amlm_effect_structured_DminusB','interaction_DC_minus_BA']


def load_json(p: pathlib.Path) -> dict:
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding='utf-8'))


def attach_superglue(seed_payload: dict, sg_payload: dict | None) -> None:
    if not sg_payload:
        return
    sg_models = sg_payload.get('models', {})
    for arm in ARMS:
        if arm in seed_payload.get('models', {}) and arm in sg_models and 'superglue' in sg_models[arm]:
            seed_payload['models'][arm]['scores']['superglue'] = float(sg_models[arm]['superglue'])


def recompute_factorial(seed_payload: dict) -> dict:
    M = seed_payload['models']
    if not all(a in M for a in ARMS):
        return {}
    A,B,C,D = (M['A_official_wwm']['scores'], M['B_structured_wwm']['scores'], M['C_official_amlm']['scores'], M['D_structured_amlm']['scores'])
    keys = sorted(set(A) & set(B) & set(C) & set(D))
    return {
        'data_effect_wwm_BminusA': {k:B[k]-A[k] for k in keys},
        'data_effect_amlm_DminusC': {k:D[k]-C[k] for k in keys},
        'amlm_effect_official_CminusA': {k:C[k]-A[k] for k in keys},
        'amlm_effect_structured_DminusB': {k:D[k]-B[k] for k in keys},
        'interaction_DC_minus_BA': {k:(D[k]-C[k])-(B[k]-A[k]) for k in keys},
    }


def summarize(seeds: dict[str, dict]) -> dict:
    per_seed = {s: recompute_factorial(p) for s,p in seeds.items()}
    out = {'per_seed_factorial': per_seed, 'mean_factorial': {}, 'seed_values': {}}
    for contrast in CONTRASTS:
        keys = sorted(set.intersection(*(set(per_seed[s].get(contrast, {})) for s in per_seed if per_seed[s].get(contrast)))) if per_seed else []
        out['mean_factorial'][contrast] = {}
        out['seed_values'][contrast] = {}
        for k in keys:
            vals = [per_seed[s][contrast][k] for s in sorted(per_seed) if k in per_seed[s].get(contrast, {})]
            out['seed_values'][contrast][k] = vals
            rec = {'n': len(vals), 'mean': statistics.mean(vals)}
            if len(vals) > 1:
                rec['stdev'] = statistics.stdev(vals)
                rec['min'] = min(vals); rec['max'] = max(vals)
            out['mean_factorial'][contrast][k] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed42', required=True)
    ap.add_argument('--seed43', required=True)
    ap.add_argument('--superglue42', default='')
    ap.add_argument('--superglue43', default='')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    seeds = {'42': load_json(pathlib.Path(args.seed42)), '43': load_json(pathlib.Path(args.seed43))}
    if args.superglue42:
        attach_superglue(seeds['42'], load_json(pathlib.Path(args.superglue42)) if pathlib.Path(args.superglue42).exists() else None)
    if args.superglue43:
        attach_superglue(seeds['43'], load_json(pathlib.Path(args.superglue43)) if pathlib.Path(args.superglue43).exists() else None)
    payload = {
        'status': 'TWO_SEED_2X2_AGGREGATE',
        'inputs': {'seed42': args.seed42, 'seed43': args.seed43, 'superglue42': args.superglue42, 'superglue43': args.superglue43},
        'seeds': seeds,
        'summary': summarize(seeds),
        'interpretation_note': 'Evidence aggregation only. Route judgment requires checking reproducibility of Entity/EWoK/GlobalPIQA/SuperGLUE gains with Supplement/Reading preservation; do not infer from a single seed.'
    }
    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'out': str(out), 'contrasts': list(payload['summary']['mean_factorial'])}, indent=2))

if __name__ == '__main__':
    main()
