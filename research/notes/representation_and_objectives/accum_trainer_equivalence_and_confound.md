# accum trainer equivalence and confound: Accumulated-trainer equivalence analysis and the 16k-vs-40k implementation confound

## Why this matters

The legal-40k compact_view_reinvest runs (`s61_t27_tool1`, `s61_t27_tool2`) use a new
isolated trainer `accumulated_masking_curriculum_trainer.py` that splits each
effective 256-row optimizer batch into four 64-row microbatches, because the direct
batch-256 40k trainer OOMed (~75.43 GiB allocated, ~2 GiB free).

The compliant legal-16k baseline (earlier analysis/51: seed43022 Overall 40.704, seed43122 41.024)
was trained with the BASE trainer `experiments/archive/compact_experience/scripts/masking_curriculum_trainer.py`
at full batch 256 with NO microbatching.

Therefore the 16k-vs-40k comparison now differs in two dimensions, not one:
1. Intended factor: tokenizer vocabulary 16k -> 40k (token inventory, segmentation,
   WWM-selected subword target geometry, embedding table 34.5M -> 45.8M params).
2. Unintended factor: full-batch-256 forward (base) vs microbatched 4x64 forward (accum).

## What is provably identical between base and accumulated trainers

Verified by direct code reading of both trainers' `main()`:

- Data path: both read the pre-materialized `_100M.jsonl` via `load_examples_jsonl`.
  The per-epoch reshuffle (base lines 739-763) applies ONLY to `official_corpus_fullcycle`,
  NOT to `example_jsonl` runs. The 100M stream is already the frozen materialized order.
  So example order and grouping are identical.
- `DataLoader(shuffle=False, collate_fn=collate)`: identical batching order.
- The accumulated trainer reassembles 4x64 microbatches into the full 256-row effective
  batch via `combine_microbatches` BEFORE calling `apply_masking_curriculum`. So the mask
  is sampled on the identical 256-row tensor with the identical `gen` CUDA generator seeded
  identically (`gen.manual_seed(train_rng_seed)`), consuming the RNG stream identically at
  the masking step.
- Model init RNG sequence identical: `reset_all_rng(seed)`, then `extra_init_seed`,
  `build_model`, then `train_rng_seed`.
- Optimizer/scheduler identical: AdamW betas (0.9,0.98), cosine schedule, warmup 0.06,
  same total_steps = ceil(len(dataset)/256) in both (base `total_steps=len(loader)` with
  batch 256; accum `total_steps=ceil(len(dataset)/256)`). These are equal because
  `len(loader_base) = ceil(len(dataset)/256)`.
- Loss accumulation is algebraically equivalent to the full-batch mean MLM objective:
  HF MLM `loss_i` is the mean cross-entropy over microbatch i's masked tokens, so
  `loss_i * n_pred_i = sum of CE over microbatch i`. The accumulated trainer backprops
  `sum_i (loss_i * n_pred_i / n_pred_total) = (sum over ALL masked tokens) / n_pred_total
  = full-batch mean MLM objective`. This does NOT imply bit-identical realized gradients:
  actual microbatch forwards use different dropout realizations and possibly different
  kernel/reduction order. The correct claim is objective-weight equivalence given the
  realized forwards, not realized-gradient identity.

## The single residual difference: dropout RNG realization

The only non-identity is that dropout (and any stochastic forward op) draws from the CUDA
generator per 64-row microbatch instead of once for the 256-row batch. Because both
consume the same global CUDA RNG stream sequentially, the specific dropout masks differ
from a hypothetical single 256-row forward. The gradient objective weighting is identical;
only the stochastic-forward realization (and floating-point reduction/kernel order) differs.
For MLM pretraining at this scale this is expected to be within seed noise, but it is
unmeasured. There is no BatchNorm or other cross-row batch-statistics layer, and DeBERTa-v2
self-attention is row-local, so there is no deterministic batch-statistics change.

## Decision: do not block, but bound the effect empirically

The 40k runs must not be interrupted (expensive-work discipline). After they finish and a
GPU frees, run a short accumulated-trainer control with the EXACT legal-16k tokenizer and
seed43022 for ~4M words, and compare its per-checkpoint loss trajectory against the
existing base-trainer legal-16k seed43022 `training_log.jsonl` at matched cumulative-word
positions. This directly bounds the microbatching-only effect on the 16k baseline.

- If the accumulated-16k trajectory matches base-16k to within ~1e-2 loss at 1M/2M/3M/4M,
  the microbatching confound is negligible and the 40k-vs-16k comparison is clean.
- If it diverges materially, the correct clean comparison is accumulated-16k vs
  accumulated-40k (same trainer), which would require also retraining a full accumulated-16k
  run; only then would the two-seed 16k baseline be re-measured under matched implementation.

## Immediate cheap evidence already collected

- base-16k seed43022 babylm2026 live surface: loss 9.8217, masked_tokens 8593, batch_words 39370.
- accum-40k pilot seed43022 babylm2026 live surface: loss 10.6839, masked_tokens 8247, batch_words 39370.
- Identical batch_words (39370) confirms identical example order/grouping.
- Different masked_tokens (8593 vs 8247) is the intended tokenizer factor: fewer subword
  tokens per word at 40k, so fewer WWM-selected subword targets per selected word.
- Higher 40k babylm2026 live surface loss reflects larger initial cross-entropy over 40000 vs 16384 classes.
- Pilot babylm2026 live surface lr=0.001 is a 1-step-run artifact (tiny warmup denominator); the full run
  applies warmup 0.06 over the true total_steps identically to base.

## Equivalence-check script

`experiments/archive/representation_and_objectives/scripts/accum16k_equivalence_control.py` (prepared this
step) will, when a GPU is free, run accumulated-trainer legal-16k seed43022 for a short
exposure and emit a trajectory comparison JSON. It is NOT launched now to avoid preempting
the active 40k trainings.
