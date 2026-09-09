#!/usr/bin/env python3
"""research trainer wrapper: minfreq50 support-floor with matched body initialization.

The ordinary COMPACT_EXPERIENCE trainer builds the model after setting the global random seed.
Changing the tokenizer vocabulary size changes how many random numbers are consumed
by vocab-shaped tensors, so the same seed no longer leaves later same-shape body
weights identical.  For the support-floor experiment, this wrapper keeps the
COMPACT_EXPERIENCE training loop and corruption policy unchanged but monkeypatches model
construction:

1. build a research legal-16k reference model under the same seed/extra_init_seed;
2. build the target minfreq50 model under the same seed/extra_init_seed;
3. copy every same-shape tensor from the 16k reference into the target;
4. leave only vocab-shaped tensors (word embeddings / decoder / bias) as target
   tokenizer-specific random tensors.

This isolates the support-floor tokenizer/representation factor much better than a
plain vocab-size run.  It is still a representation package, not final evidence by
itself: segmentation, target-token count, visible groups, and vocab-shaped
parameters all change.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import pathlib
import random
import sys
import time
from typing import Any

import numpy as np
import torch


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
COMPACT_EXPERIENCE_SCRIPTS = USER_ROOT / "experiments/archive/compact_experience/scripts"
if str(COMPACT_EXPERIENCE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(COMPACT_EXPERIENCE_SCRIPTS))

import masking_curriculum_trainer as base  # noqa: E402

TOKENIZER = USER_ROOT / "experiments/archive/frontier_consolidation/data/compliant_tokenizer"
ORIGINAL_BUILD_MODEL = base.build_model


def reset_all_rng(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_for_init(args: Any) -> int:
    extra = int(getattr(args, "extra_init_seed", -1))
    if extra >= 0:
        return extra
    return int(getattr(args, "seed", 43))


def same_shape_initmatched_build_model(args: Any, tokenizer: Any):
    """Build target model and copy same-shape tensors from research legal16k init."""
    if not TOKENIZER.exists():
        raise FileNotFoundError(TOKENIZER)
    init_seed = seed_for_init(args)

    # Reference: exactly what the COMPACT_EXPERIENCE trainer would build for research legal16k.
    reset_all_rng(init_seed)
    ref_tok = base.make_portable_tokenizer(str(TOKENIZER))
    ref_model = ORIGINAL_BUILD_MODEL(args, ref_tok)
    ref_sd = {k: v.detach().cpu().clone() for k, v in ref_model.state_dict().items()}
    ref_param_count = int(sum(p.numel() for p in ref_model.parameters()))

    # Target: normal random vocab-shaped tensors for the target tokenizer.
    reset_all_rng(init_seed)
    target_model = ORIGINAL_BUILD_MODEL(args, tokenizer)
    target_param_count = int(sum(p.numel() for p in target_model.parameters()))

    copied = []
    skipped = []
    target_sd = target_model.state_dict()
    with torch.no_grad():
        for name, ref_tensor in ref_sd.items():
            if name not in target_sd:
                continue
            tgt = target_sd[name]
            if tuple(tgt.shape) == tuple(ref_tensor.shape):
                tgt.copy_(ref_tensor.to(dtype=tgt.dtype, device=tgt.device))
                copied.append({"name": name, "numel": int(tgt.numel())})
            else:
                skipped.append({"name": name, "ref_shape": list(ref_tensor.shape), "target_shape": list(tgt.shape), "target_numel": int(tgt.numel())})

    # Lightweight verification before training; only same-shape tensors should now match.
    verify_sd = target_model.state_dict()
    exact_same_shape = 0
    same_shape = 0
    exact_random_like = 0
    random_like = 0
    exact_numel = 0
    same_numel = 0
    for name, ref_tensor in ref_sd.items():
        if name not in verify_sd:
            continue
        tgt = verify_sd[name].detach().cpu()
        if tuple(tgt.shape) != tuple(ref_tensor.shape):
            continue
        same_shape += 1
        same_numel += int(tgt.numel())
        eq = torch.equal(tgt, ref_tensor)
        if eq:
            exact_same_shape += 1
            exact_numel += int(tgt.numel())
        is_random_like = torch.is_floating_point(ref_tensor) and float(ref_tensor.float().std(unbiased=False).item()) > 0.0
        if is_random_like:
            random_like += 1
            if eq:
                exact_random_like += 1
    event = {
        "event": "initmatched_model_build",
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reference_tokenizer": str(TOKENIZER),
        "target_vocab_size": len(tokenizer),
        "reference_vocab_size": len(ref_tok),
        "init_seed": init_seed,
        "copied_same_shape_tensor_count": len(copied),
        "skipped_shape_mismatch_count": len(skipped),
        "skipped_shape_mismatches": skipped,
        "same_shape_exact_tensor_count_after_copy": exact_same_shape,
        "same_shape_tensor_count_after_copy": same_shape,
        "same_shape_exact_numel_fraction_after_copy": exact_numel / same_numel if same_numel else None,
        "random_like_exact_tensor_count_after_copy": exact_random_like,
        "random_like_tensor_count_after_copy": random_like,
        "ref_param_count": ref_param_count,
        "target_param_count": target_param_count,
        "target_minus_ref_params": target_param_count - ref_param_count,
    }
    print(json.dumps(event, ensure_ascii=False), flush=True)
    del ref_model, ref_sd
    return target_model


def main() -> None:
    base.build_model = same_shape_initmatched_build_model
    base.main()


if __name__ == "__main__":
    main()
