#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import csv, json, pathlib, time

ROOT = _public_path('experiments/archive/frontier_consolidation/scripts/update_prediction_commitment_ablation.py')
ROOT = _PUBLIC_ROOT
WS = _public_path('experiments/archive/frontier_consolidation')
commit_path = _public_path('experiments/archive/frontier_consolidation/data/distribution_proximity_prediction/prediction_commitment.json')
ablation_csv = _public_path('experiments/archive/frontier_consolidation/data/profile_ablation_prediction/aggregate_prediction_rows.csv')
decomp_json = _public_path('experiments/archive/frontier_consolidation/data/profile_distance_decomposition/profile_distance_decomposition_summary.json')
commit = json.loads(commit_path.read_text(encoding='utf-8'))
ab = {}
with ablation_csv.open(encoding='utf-8', newline='') as f:
    for r in csv.DictReader(f):
        ab.setdefault(r['ablation'], {})[r['quantity']] = float(r['pred_points'])
decomp = json.loads(decomp_json.read_text(encoding='utf-8'))
commit['profile_ablation_robustness'] = {
    'created_before_register_scores': True,
    'source_summary_md': 'research/documents/frontier_consolidation/data/profile_ablation_prediction/profile_ablation_prediction_summary.md',
    'source_decomposition_md': 'research/documents/frontier_consolidation/data/profile_distance_decomposition/profile_distance_decomposition_summary.md',
    'aggregate_predictions': ab,
    'exEntity_component_means': {r['feature_group']: r['exEntity_mean_contribution'] for r in decomp['group_summary_rows']},
    'scientific_implication': 'The positive profile prediction survives removing transcript/markup (+0.658 exEntity5) and remains positive after removing both transcript/markup and punctuation (+0.380 exEntity5); explicit transcript markers contribute but do not solely carry the committed structural/profile-register prediction.'
}
commit['last_updated_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
commit_path.write_text(json.dumps(commit, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
print(json.dumps({'status':'commitment_ablation_updated','path':str(commit_path.relative_to(ROOT)),'ablations':ab}, indent=2))
