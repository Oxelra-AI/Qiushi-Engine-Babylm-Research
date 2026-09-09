#!/usr/bin/env python3
"""Full official-compatible evaluation wrapper for the conditional research
legal40k compact_view_reinvest SGCR(K=50,d=64) 12x384 seed43122 reproduction.

Use only if seed43022 SGCR treatment clears or nearly clears the 41.80 frontier
and a second-seed endpoint becomes scientifically decisive. The coordinate matches
research seed43022 treatment except for extra_init_seed=43122/train_rng_seed=43123.
"""
from __future__ import annotations

import pathlib
import sys

USER_ROOT = pathlib.Path('.').resolve()
COMPACT_EXPERIENCE_SCRIPTS = USER_ROOT / "experiments/archive" / 'compact_experience' / 'scripts'
if str(COMPACT_EXPERIENCE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(COMPACT_EXPERIENCE_SCRIPTS))

import full_overall_eval_runner as base  # noqa: E402

A01_WORKSPACE = USER_ROOT / "experiments/archive" / 'representation_and_objectives'
A01_RUN_BASE = A01_WORKSPACE / 'training' / 'runs'
PRISTINE_STRICT = A01_WORKSPACE / 'data' / 'pristine_official_coordinate' / 'babylm-eval' / 'strict'
GLOBALPIQA_GENERATED = A01_WORKSPACE / 'data' / 'globalpiqa_official_lineage' / 'official_dl_scratch' / 'generated_by_current_official_dl' / 'evaluation_data' / 'full_eval'

base.STRICT = PRISTINE_STRICT
for _task in base.ZERO_SHOT_TASKS:
    if _task['column'] == 'GlobalPIQA_parallel':
        _task['data_path'] = str((GLOBALPIQA_GENERATED / 'global_piqa_parallel').resolve())
    elif _task['column'] == 'GlobalPIQA_nonparallel':
        _task['data_path'] = str((GLOBALPIQA_GENERATED / 'global_piqa_nonparallel').resolve())

base.OUT_ROOT = A01_WORKSPACE / 'data' / 'legal40k_12x384_sgcrK50d64_seed43122_full_eval'
base.PER_TARGET_DIR = base.OUT_ROOT / 'per_target'
base.TARGETS = {
    'legal40k_12x384_sgcrK50d64_seed43122': {
        'run_dir': A01_RUN_BASE / 'legal40k_12x384_sgcrK50d64_compact_view_reinvest_seed43122',
        'endpoint': 'chck_100M',
        'description': (
            'Compact_view_reinvest seed43122 retrained with legal 40k byte-BPE, '
            'true 12-layer hidden-384 DeBERTa-v2 geometry, and exact-prefix SGCR '
            'K=50 d=64 support-gated legal16k component sharing. This is the '
            'second-seed reproduction coordinate for a promising seed43022 SGCR endpoint.'
        ),
        'family': 'representation_and_objectives_legal40k_12x384_sgcrK50d64_seed43122',
    }
}
base.ZERO_BY_COL = {t['column']: t for t in base.ZERO_SHOT_TASKS}

if __name__ == '__main__':
    base.main()
