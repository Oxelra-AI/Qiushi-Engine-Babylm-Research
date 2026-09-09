#!/usr/bin/env python3
"""Full official-compatible evaluation wrapper for the research legal40k
compact_view_reinvest 12x384 SGCR mass-matched uniform-residual control.

This wrapper reuses the same robust COMPACT_EXPERIENCE full evaluator and pristine official
coordinate used for depth and SGCR treatment. EWoK and AoA are computed by the
research post-training controller, then staged by the hardened pristine collator.

Scientific interpretation: relative to the pending research SGCR treatment, this
control preserves data, tokenizer, WWM labels, stream order, optimizer, schedule,
model geometry, seed identity, component table/projection parameterization, and
training-token residual mass. It removes support-dependent allocation by setting
one ordinary-token residual weight for all legal40k tokens.
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

base.OUT_ROOT = A01_WORKSPACE / 'data' / 'legal40k_12x384_sgcrUniformK50d64_seed43022_full_eval'
base.PER_TARGET_DIR = base.OUT_ROOT / 'per_target'
base.TARGETS = {
    'legal40k_12x384_sgcrUniformK50d64_seed43022': {
        'run_dir': A01_RUN_BASE / 'legal40k_12x384_sgcrUniformK50d64_compact_view_reinvest_seed43022',
        'endpoint': 'chck_100M',
        'description': (
            'Compact_view_reinvest seed43022 retrained with legal 40k byte-BPE, '
            'true 12-layer hidden-384 DeBERTa-v2 geometry, and SGCR mass-matched '
            'uniform residual control K=50 d=64. Baked HF checkpoints are standard '
            'DebertaV2ForMaskedLM models for evaluation.'
        ),
        'family': 'representation_and_objectives_legal40k_12x384_sgcrUniformK50d64_seed43022',
    }
}
base.ZERO_BY_COL = {t['column']: t for t in base.ZERO_SHOT_TASKS}

if __name__ == '__main__':
    base.main()
