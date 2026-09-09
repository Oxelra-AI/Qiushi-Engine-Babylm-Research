# Released Checkpoint References

Release records identify these existing packages; this extraction did not download,
load, modify, or copy any weights.

| Scientific role | Package | Recorded immutable revision |
| --- | --- | --- |
| Frontier reference and shared initialization | [Qiushi-Engine-Frontier-Advancement](https://huggingface.co/leslie721007/Qiushi-Engine-Frontier-Advancement/tree/5eb20f9c5088f40183269bb2c97381711aea0143) | `5eb20f9c5088f40183269bb2c97381711aea0143` |
| Dense-mask/sparse-label acquisition with ordinary-state preservation | [Qiushi-Engine-Principle-Guided-Frontier-Advancement](https://huggingface.co/leslie721007/Qiushi-Engine-Principle-Guided-Frontier-Advancement/tree/ce7eabf0dfbd3d1393670f41f610bdccbdbe45d1) | `ce7eabf0dfbd3d1393670f41f610bdccbdbe45d1` |

The reference records 86,005,295 word presentations; the principle-guided endpoint
records 89,685,369, including auxiliary student preservation presentations. Both
use 36,458,592-parameter masked-language-model packages with inherited slow and
private adapter paths. The private path has 995,584 parameters. The corresponding
encoder has 36,210,368 parameters; these counts must not be confused with the
smaller stock encoder selected by the historical loading error.

Recorded final weight SHA-256:

- Reference: `e14d757ae51b41e33bf0813f841248fecd1eefeb9e040f520c4c6203343b15c8`.
- Principle-guided: `1c6268a37a7774a78d725307f99bbaf1f5225b4315f9bfd9ddb2a5b47c680e4a`.

Intermediate acquisition-order measurements use actual shared ancestral
checkpoints and each branch's own endpoint. They must not borrow post-branch
weights or fabricate a maximum-budget trajectory for an early-stopped model.
## Archived Experimental Checkpoints

The following original checkpoints are also present in the local research tree.
Their weights are unchanged; configuration and module paths are portable.
They retain their own adapter scales and must not be silently replaced by a
representative endpoint with a different configuration.

| Experimental role | Checkpoint directory |
| --- | --- |
| 82M-word initialization before frozen-parent replay | [82M initialization](../experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M/) |
| Coherent replay endpoint, original export configuration | [Coherent replay](../experiments/archive/frontier_consolidation/training/runs/frozen82_fastpath4M_coherent_seed43022/hf_model/final/) |
| Dense-mask/sparse-target acquisition-only control, seed 62064 | [Acquisition control](../experiments/archive/functional_learning/data/densemask_sparselabel_train_seed62064/correspondence_focus_weighted/checkpoints/update_0080/) |

Each directory contains weights, configuration, tokenizer and checkpoint-local
model code. File identities are recorded in the [material manifest](../evidence/materials_manifest.json).
Model tensors require large-file distribution; they are not ordinary Git blobs.
These local additions do not assert a new remote model publication or a rerun.
