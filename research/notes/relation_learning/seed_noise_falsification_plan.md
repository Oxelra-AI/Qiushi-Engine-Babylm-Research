# Falsifying the Seed-Noise Explanation

Scientific status: prospective proposal. The tests below are not results merely because they were proposed.

The initial alternative explanation was that a register-level sign reversal reflected aggregate seed noise rather than a stable data mechanism. A scalar benchmark difference is therefore insufficient evidence that one material has greater intrinsic training value.

The planned fitting index used deterministic per-row MLM losses on 6,992 reference rows and 5,000 common filler rows, at five checkpoints from 60M through 100M. The comparison arms were DeBERTa VIEW at seeds 43022 and 43122, DeBERTa CLEAN at 43022, and RoBERTa VIEW/CLEAN at 43022. The primary test held architecture and data fixed while varying trajectory. Planned outputs were cross-seed row-loss correlation, shared/private variance decomposition, and source-stratified summaries.

Three falsification controls were proposed:

1. Estimate a clean-prior seed-noise floor from both seeds rather than treating a single endpoint difference as stable.
2. Test whether BLiMP stability is explained by mask-token exposure counts rather than the proposed representation mechanism.
3. Remove comparison edges while retaining exposure and row count, then test whether the topology-dependent readout still collapses.

The proposed interpretation contrasted a predominantly shared fitting pattern with substantial trajectory-specific fitting. Correlation alone does not establish the cause of a register sign reversal: common row difficulty can also produce high correlation. The later decomposition must therefore be read with its corrections, and small aggregate contrasts remain unestablished until their noise floor and matched comparisons are resolved.

Evidence and later readouts: [fitting-index note](shared_private_fitting_index.md), [corrected seed-by-data decomposition](corrected_seedxdata_decomposition.md), [original analysis implementation](../../../experiments/archive/relation_learning/scripts/shared_private_loss.py).
