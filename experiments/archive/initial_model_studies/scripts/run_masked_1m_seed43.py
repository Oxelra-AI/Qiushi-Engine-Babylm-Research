#!/usr/bin/env python3
"""research seed-43 repeat of the controlled token-MLM vs WWM 1M experiment.

This deliberately reuses the research position-capacity-repaired runner and changes
only run IDs, output paths, log path, and random seeds. It does not add length
curriculum, tokenizer changes, paired data, optimizer changes, or architecture changes.
"""
from __future__ import annotations

import importlib.util
import pathlib

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
BASE_SCRIPT = ROOT / "scripts/run_masked_1m_grid_pos512.py"

spec = importlib.util.spec_from_file_location("runner", BASE_SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(mod)

# New artifacts for seed-43 repeat.
mod.RUN_SPECS = [
    ("token", "babylm_masked_token_pos512_seed43_1M"),
    ("wwm", "babylm_masked_wwm_pos512_seed43_1M"),
]
mod.OUT_JSON = ROOT / "data/masked_1m_seed43_profile.json"
mod.OUT_TRAIN_JSON = ROOT / "data/masked_1m_seed43_training_summary.json"
mod.OUT_NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/masked_1m_seed43_token_vs_wwm_profile.md')
mod.LOG = (ROOT.parents[2] / 'research/notes/initial_model_studies/masked_1m_seed43_train_and_profile.log')

# Change only the random condition while preserving the research controlled design.
def replace_arg(args: list[str], key: str, value: str) -> None:
    i = args.index(key)
    args[i + 1] = value

replace_arg(mod.COMMON, "--seed", "43")
replace_arg(mod.COMMON, "--extra_init_seed", "457")
replace_arg(mod.COMMON, "--train_rng_seed", "790")

if __name__ == "__main__":
    mod.main()
