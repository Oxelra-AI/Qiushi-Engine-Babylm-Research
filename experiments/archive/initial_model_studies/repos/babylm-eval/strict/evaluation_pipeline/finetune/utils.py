from __future__ import annotations

import torch
import os
import random
import math

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.optim.lr_scheduler import LambdaLR


def cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int, min_factor: float) -> LambdaLR:
    """A cosine decay scheduler with linear warmup. The learning
    rate decays to a factor of the maximum learining rate specified
    by the user. The amount of decay and warmup steps is also
    specified by the user.

    Args:
        num_warmup_steps(int): The total number of warmup steps to
            perform.
        num_training_steps(int): The total number of steps until
            reaching the smallest learning rate. It is the sum of
            the warmup and decay steps.
        min_factor(float): The minimum factor of the maximum
            learning rate that the scheduler decays to.

    Returns:
        LambdaLR: The cosine decay scheduler object.
    """
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        lr = max(min_factor, min_factor + (1 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress)))
        return lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)


def seed_everything(seed_value: int = 42) -> None:
    """A utility function that seeds all the different random
    number generators to make the process using it
    deterministic (barring hardware configuration).

    Args:
        seed_value(int): The state number for the random
            number generators.
    """
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
