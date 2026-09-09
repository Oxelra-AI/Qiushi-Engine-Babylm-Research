#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import hashlib, json, pathlib, time
ROOT = _public_path('experiments/archive/representation_and_objectives/scripts/checkpoint_identity.py')
parts = ROOT.parts
USER_ROOT = pathlib.Path(*parts[:parts.index("experiments")])
files = {
    'model_safetensors_80M_nonladder': 'experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M80M_seed43022/hf_model/chck_80M/model.safetensors',
    'model_safetensors_80M_in_100M_ladder': 'experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_80M/model.safetensors',
    'config_80M_nonladder': 'experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M80M_seed43022/hf_model/chck_80M/config.json',
    'config_80M_in_100M_ladder': 'experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_80M/config.json',
}
def sha(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()
out = {
    'status': 'SCALE1P75_80M_CHECKPOINT_IDENTITY',
    'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'files': {},
}
for k, rel in files.items():
    p = USER_ROOT / rel
    out['files'][k] = {'path': rel, 'exists': p.exists(), 'sha256': sha(p) if p.exists() else None, 'size_bytes': p.stat().st_size if p.exists() else None}
out['model_safetensors_identical'] = out['files']['model_safetensors_80M_nonladder']['sha256'] == out['files']['model_safetensors_80M_in_100M_ladder']['sha256']
out['config_identical'] = out['files']['config_80M_nonladder']['sha256'] == out['files']['config_80M_in_100M_ladder']['sha256']
out['interpretation'] = 'The research/153-scored scale1.75 chck_80M is bit-identical to chck_80M saved inside the research 100M official-ladder run; this ties the internal 80M evidence to the complete continuation lineage. It does not change the measured 80M Overall, which remains below 41.80 with AoA=0.0.'
out_path = USER_ROOT / 'experiments/archive/representation_and_objectives/data/scale1p75_score_collation/scale1p75_80M_checkpoint_identity.json'
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'status': out['status'], 'path': str(out_path.relative_to(USER_ROOT)), 'model_identical': out['model_safetensors_identical'], 'config_identical': out['config_identical'], 'sha256': out['files']['model_safetensors_80M_nonladder']['sha256']}))
