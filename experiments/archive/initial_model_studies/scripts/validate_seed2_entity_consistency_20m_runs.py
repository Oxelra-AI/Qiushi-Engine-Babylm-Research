#!/usr/bin/env python3
from __future__ import annotations

import pathlib, sys
ROOT = pathlib.Path('experiments/archive/initial_model_studies')
sys.path.insert(0, str((ROOT/'scripts').resolve()))
import validate_entity_consistency_20m_runs as v

v.RUNS = {
    'consistency': ROOT/'training/runs/babylm_seed2_entity_consistency_deberta_20M',
    'shuffled_pair': ROOT/'training/runs/babylm_seed2_entity_shuffled_pair_deberta_20M',
}
v.OUT = ROOT/'data/seed2_entity_consistency_20m_training_validation.json'
v.NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/seed2_entity_consistency_20m_training_validation.md')

if __name__ == '__main__':
    v.main()
