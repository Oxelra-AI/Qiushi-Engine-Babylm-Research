# Residual Incremental Learning

Each layer first computes the existing encoder output, then adds bottleneck residual paths. The first path participated in earlier training. The second starts with zero output and supports new learning while the existing parameters remain frozen. The "dedicated increment" corresponds to `private_adapter` in the public code: it is a separately trained branch, not a privacy mechanism.

Both released model generations have 36,458,592 parameters. The second path contains 995,584 parameters in 48 tensors, and only these parameters are updated in Stage III. Its parameter fraction is not its fraction of computational cost. The first path has scale 1.75; the released second path has scale 0.75.

The same configuration registers entry points both with and without a prediction head. Downstream fine-tuning must retain both residual paths; a stock encoder missing the incremental branch is not a valid substitute.

- [Model implementation](../models/frontier/modeling_frozen_slow_private_debertav2.py)
- [Architecture configuration](../models/frontier/config.json)
- [Increment-scale comparison](../results/private_scale_sweep.csv)
- [Model lineage](../evidence/model_lineage.md)

The residual architecture provides a controlled object for further study. Mechanistic explanations and final scores require their respective experimental evidence; neither can be inferred from an architecture diagram.
