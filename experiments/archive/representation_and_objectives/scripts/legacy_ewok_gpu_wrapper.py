#!/usr/bin/env python3
"""GPU-only EWoK four-cell readout for research standard-legacy 80M.

This intentionally evaluates only the missing EWoK readout for the staged
standard-WWM 70M->80M branch.  Completed Supplement/Entity and GlobalPIQA files
from the cancelled combined CPU task are not rerun.  Output is isolated from the
cancelled CPU attempt.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path('.').resolve()
A01_WS = ROOT / 'experiments/archive/representation_and_objectives'
BASE_SCRIPT = A01_WS / 'scripts/fw_ewok_interaction_reader.py'
OUT_ROOT = A01_WS / 'data/legacy_80m_ewok_gpu'
NOTE = (ROOT / 'research/notes/representation_and_objectives/legacy_80m_ewok_gpu_reader.md')
TARGET = 'standard_legacy_80m_gpu'
MODEL_PATH = A01_WS / 'training/runs/standard_legacy_70M_to_80M_seed43022/hf_model/chck_80M'


def now_utc() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def load_base():
    spec = importlib.util.spec_from_file_location('fw_ewok_step121_gpu', BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot import {BASE_SCRIPT}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    preflight = {
        'status': 'READY',
        'created_utc': now_utc(),
        'target': TARGET,
        'model_path': str(MODEL_PATH),
        'model_path_exists': MODEL_PATH.exists(),
        'output_root': str(OUT_ROOT),
        'note': str(NOTE),
        'purpose': 'missing research standard-legacy EWoK four-cell readout only; completed Supp/Entity and GlobalPIQA are preserved from existing files and not rerun',
    }
    (OUT_ROOT / 'preflight.json').write_text(json.dumps(preflight, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    mod = load_base()
    mod.OUT_ROOT = OUT_ROOT
    mod.NOTE = NOTE
    mod.DEFAULT_TARGETS.clear()
    mod.DEFAULT_TARGETS[TARGET] = {
        'model_path': MODEL_PATH,
        'label': 'research staged legacy-WWM 70M->80M replay, GPU isolated EWoK readout',
    }
    old_argv = sys.argv
    try:
        sys.argv = [str(BASE_SCRIPT), '--targets', TARGET, '--device', 'cuda', '--threads', '8', '--row_batch_size', '64', '--masked_batch_size', '128']
        mod.main()
    finally:
        sys.argv = old_argv
    print(json.dumps({'status': 'LEGACY_EWOK_GPU_WRAPPER_DONE', 'out_root': str(OUT_ROOT), 'summary': str(OUT_ROOT / TARGET / 'ewok_interaction_summary.json'), 'note': str(NOTE)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
