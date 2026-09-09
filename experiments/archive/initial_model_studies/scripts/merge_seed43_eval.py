#!/usr/bin/env python3
"""Merge split seed43 A/B and C/D zero-shot evaluation JSONs and recompute factorial."""
from __future__ import annotations
import json, pathlib

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
AB = ROOT/'data/seed43_scores_AB.json'
CD = ROOT/'data/seed43_scores_CD.json'
OUT = ROOT/'data/2x2_seed43_scores.json'
ARMS = ['A_official_wwm','B_structured_wwm','C_official_amlm','D_structured_amlm']

def factorial(M):
    A,B,C,D=(M['A_official_wwm']['scores'],M['B_structured_wwm']['scores'],M['C_official_amlm']['scores'],M['D_structured_amlm']['scores'])
    keys=sorted(set(A)&set(B)&set(C)&set(D))
    return {
        'data_effect_wwm_BminusA':{k:B[k]-A[k] for k in keys},
        'data_effect_amlm_DminusC':{k:D[k]-C[k] for k in keys},
        'amlm_effect_official_CminusA':{k:C[k]-A[k] for k in keys},
        'amlm_effect_structured_DminusB':{k:D[k]-B[k] for k in keys},
        'interaction_DC_minus_BA':{k:(D[k]-C[k])-(B[k]-A[k]) for k in keys},
    }

def main():
    if not AB.exists() or not CD.exists():
        raise FileNotFoundError(f'missing partials: AB={AB.exists()} CD={CD.exists()}')
    ab=json.loads(AB.read_text()); cd=json.loads(CD.read_text())
    models={}; models.update(ab.get('models',{})); models.update(cd.get('models',{}))
    missing=[a for a in ARMS if a not in models]
    if missing: raise RuntimeError(f'missing arms: {missing}')
    payload={'status':'reference_2X2_SEED43_MERGED','models':models,'factorial':factorial(models),'inputs':{'AB':str(AB),'CD':str(CD)}}
    OUT.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps({'out':str(OUT),'factorial':payload['factorial']},indent=2))
if __name__=='__main__': main()
