#!/usr/bin/env python3
"""research extension: evaluate the missing DeBERTa CLEAN seed43122 arm.

This reuses the research deterministic-mask per-row MLM-loss evaluator but adds the
fourth DeBERTa cell needed for a proper 2(seed) x 2(data) decomposition:

    VIEW  seed43022   VIEW  seed43122
    CLEAN seed43022   CLEAN seed43122

No training, no official benchmark evaluation, no upload, no leaderboard action.
"""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


from pathlib import Path
import sys

SCRIPT_DIR = _public_path('experiments/archive/relation_learning/scripts')
sys.path.insert(0, str(SCRIPT_DIR))

import shared_private_loss as s3  # noqa: E402

s3.ARM_CONFIGS["D_C_43122"] = {
    "model_family": "deberta",
    "run_dir": s3.frontier_consolidation_RUNS / "full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43122",
}

if __name__ == "__main__":
    s3.main()
