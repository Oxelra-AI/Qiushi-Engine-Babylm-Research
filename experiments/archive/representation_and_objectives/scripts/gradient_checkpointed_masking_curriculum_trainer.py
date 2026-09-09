#!/usr/bin/env python3
"""Gradient-checkpointed wrapper around the COMPACT_EXPERIENCE masking-curriculum trainer.

research purpose: repair the research H100 OOM without changing the compact-view
mechanism contrast. The original COMPACT_EXPERIENCE trainer with batch_size=256 now peaks above
available free H100 memory because both GPUs show ~19.5 GiB resident memory with
no visible compute process. This wrapper imports the trusted COMPACT_EXPERIENCE trainer and
only enables model activation checkpointing immediately after model construction.
The command-line interface, data loading, masking, optimizer, scheduler, checkpoint
word positions, batch size, seeds, and logging are inherited unchanged.

Scientific status: this is an operational memory repair. It should preserve the
full effective 256-row update much more directly than microbatch accumulation.
A first-update loss pilot should be checked before long runs are interpreted.
"""
from __future__ import annotations

import pathlib
import sys

USER_ROOT = pathlib.Path('.').resolve()
COMPACT_EXPERIENCE_SCRIPTS = USER_ROOT / "experiments/archive" / 'compact_experience' / 'scripts'
if str(COMPACT_EXPERIENCE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(COMPACT_EXPERIENCE_SCRIPTS))

import masking_curriculum_trainer as base  # noqa: E402

_ORIGINAL_BUILD_MODEL = base.build_model


def build_model_with_activation_checkpointing(args, tokenizer):
    model = _ORIGINAL_BUILD_MODEL(args, tokenizer)
    # Transformers models expose gradient_checkpointing_enable(); DeBERTa-v2
    # supports it through the encoder. Use reentrant checkpointing defaults from
    # the installed Transformers/PyTorch stack so dropout RNG state is preserved.
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    else:
        raise RuntimeError('model does not expose gradient_checkpointing_enable')
    if hasattr(model.config, 'use_cache'):
        model.config.use_cache = False
    setattr(model.config, 'activation_checkpointing', True)
    return model


base.build_model = build_model_with_activation_checkpointing


if __name__ == '__main__':
    base.main()
