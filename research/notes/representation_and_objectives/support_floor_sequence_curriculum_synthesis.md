# family specific tokenizer predictor — CPU route evidence while depth training runs

This note consolidates two CPU-only analyses while WWM-to-token post-switch evaluation and legal40k 12x384 depth seed43022 training remain pending. No new GPU training or official evaluation was launched.

## 1. Pure support-floor tokenizers are probably not sufficient alone

Two predictors were built from the measured legal16k/legal40k official endpoint vectors and tokenizer support spectrum tokenizer support spectra:

- `experiments/archive/representation_and_objectives/scripts/tokenizer_support_frontier_predictor.py`
- `experiments/archive/representation_and_objectives/scripts/family_specific_tokenizer_predictor.py`

Key outputs:

- Global support-frontier predictor: `experiments/archive/representation_and_objectives/data/tokenizer_support_frontier_predictor/tokenizer_support_frontier_predictor.json`
- Family-specific predictor: `experiments/archive/representation_and_objectives/data/family_specific_tokenizer_predictor/family_specific_tokenizer_predictor.json`
- Notes: `tokenizer_support_frontier_predictor.md`, `family_specific_tokenizer_predictor.md`

The sharper family-specific predictor uses actual official-evaluation family statistics: eval-token counts as the segmentation proxy and family `frac_lt50` as under-trained-token exposure. It reproduces the legal16k/legal40k anchors exactly.

Predicted central Overall values:

| candidate | predicted central Overall | upper bound vs 41.8 after seed/model-form band |
|---|---:|---:|
| minfreq25 | 41.0224 | -0.4173 |
| minfreq50 | 41.0195 | -0.4203 |
| 24k | 41.0118 | -0.4280 |
| 32k | 40.8810 | -0.5587 |

Scientific reading: the support-floor idea remains a real representation repair, but the calibrated endpoints do not support spending next full two-seed slot on a pure tokenizer interpolation alone. If it is used, it should be paired with a distinct factor that can move the remaining frontier gap: depth, masking curriculum, faithful sequence curriculum, or a sharper objective.

This supports the current Lead-selected depth run as a genuinely different factor, but it also warns that depth on legal40k may still inherit the GlobalPIQA/rare-token problem because geometry does not change token support.

## 2. Existing sequence-length scheduling is not a faithful leader-style curriculum

The public leader model card states sequence length curriculum `64 -> 256` and says batch size is scaled inversely with sequence length. lead route decision after legal40k found the current COMPACT_EXPERIENCE/legal40k accum training completion `seq_len_schedule` implementation tokenizes each row at max length 256, slices to the current short prefix, and still charges the full row word count.

family specific tokenizer predictor quantified the distortion with:

- Script: `experiments/archive/representation_and_objectives/scripts/sequence_curriculum_accounting_audit.py`
- Output: `experiments/archive/representation_and_objectives/data/sequence_curriculum_accounting_audit/sequence_curriculum_accounting_audit.json`
- Note: `research/notes/representation_and_objectives/sequence_curriculum_accounting_audit.md`

On the exact compact_view_reinvest 10M pool:

### legal40k tokenizer

| length | prefix-visible word-group fraction | missing group fraction under prefix slicing | faithful chunks per epoch | faithful inverse-batch steps/epoch |
|---:|---:|---:|---:|---:|
| 64 | 0.306 | 0.694 | 252,269 | 247 |
| 128 | 0.608 | 0.392 | 140,303 | 275 |
| 256 | 0.987 | 0.013 | 76,164 | 298 |

For a plausible `64×3 / 128×4 / 256×3` ten-epoch schedule, current prefix slicing would leave **36.9%** of tokenized word-groups unseen while charging their words; faithful word-boundary chunking would provide **1.609×** as many visible target tokens with only **1.081×** optimizer steps relative to that prefix-slicing path.

### minfreq25 tokenizer

For the same schedule, prefix slicing would leave **37.4%** of tokenized word-groups unseen; faithful chunking would provide **1.623×** target-token exposure with **1.092×** optimizer steps.

Scientific reading: a future sequence-curriculum route must be implemented as stage-specific tokenization/streaming or word-boundary chunking with inverse row-batch scaling. The current `seq_len_schedule` path should not be used as evidence for the leader-style sequence factor.

## Route implication when pending tasks finish

- If `s74_t5_tool1` depth seed43022 clears or nearly clears 41.8, protect and reproduce before adding new mechanisms.
- If depth gives coherent gains but remains limited by GlobalPIQA/rare-token columns, the strongest follow-up is likely **depth plus support-floored tokenizer** or **support-floor plus faithful sequence/masking curriculum**, not pure minfreq25 alone.
- If depth is flat or worse, do not spend a second depth seed by inertia. Use `s73_t7_tool1` masking-factor evidence and this sequence audit to choose between WWM→token curriculum and a faithfully implemented 64→256 curriculum. If support-floor evidence from companion analysis arrives, combine rather than duplicate.
- Relation-cue masking remains well specified but should be selected only if the post-depth vector leaves a selective relation/state deficit rather than a broad representation deficit.
