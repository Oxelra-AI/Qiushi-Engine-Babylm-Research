# chunk stream preflight — Experience-utilization scientific status

No H100 training or official evaluation was launched. The legal40k 12x384 depth run remains the decisive pending experiment.

## What changed scientifically

sequence curriculum loop measurement showed that the current sequence-length path is not a faithful 64→256 curriculum: L64/L128 prefixes debit full row words while hiding suffix word groups. chunk stream preflight turns this into a mechanism and implementable comparison around **charged-experience utilization** rather than schedule copying.

The central principle now has a measurable form:

> Under a strict word budget, words counted against the training budget should also be made visible to the model and eligible for prediction. If a pipeline debits words that never reach the model, it wastes scarce language experience.

## Clean comparison structure

### Current row256 → U256 chunked

- Same 100M charged words and maximum length 256.
- U256 partitions rows into word-boundary chunks so every tokenizer token from the 10M corpus is visible once per epoch.
- This reads a fixed-length chunk-stream intervention that restores row-truncated suffix material.
- It is **not** a pure visibility-only intervention: chunk boundaries, example count, update composition, padding, and microbatch accumulation also change.

Measured from chunk stream preflight:

| tokenizer | current active tokens | U256 active tokens | U256/current | hidden charged words recovered |
|---|---:|---:|---:|---:|
| legal40k | 137,061,620 | 139,426,440 | 1.0173 | 1,286,640 |
| minfreq25 | 138,713,400 | 141,385,120 | 1.0193 | 1,448,730 |

Target-matched mask probabilities if needed: legal40k 0.147456, minfreq25 0.147165.

### U256 chunked → U64_128_256 chunked

- Both arms expose all tokenized words once per epoch.
- Both match on 100M charged words, active-token total, expected targets at shared mask probability, and 2,530 stage-reset updates.
- This separates the experience-utilization repair from the later short-context/order/packing bundle.
- It still does **not** isolate chronology alone. If U64_128_256 beats U256, a length-mixture matched interleaved or permuted stream is needed before attributing the gain to short-to-long order.

Corrected chunk counts from construction preflight:

| tokenizer | U256 chunks total | U64_128_256 chunks total | matched words/tokens/updates |
|---|---:|---:|---|
| legal40k | 761,640 | 1,546,514 | yes |
| minfreq25 | 770,920 | 1,564,093 | yes |

The +3 corrected chunks in U64_128_256 relative to the earlier design come from the single L64 overlong whitespace word split into one zero-charge continuation chunk per L64 epoch.

## Implementation checks completed in chunk stream preflight

1. Full 10M corpus chunk construction passed for legal40k and minfreq25: every epoch has exactly 10M charged words, all raw tokenizer tokens visible, and zero unassigned offsets.
2. First-update tensors were built for each tokenizer and length. Accumulation depth is L64=16, L128=9, L256=5 microbatches; sampled WWM mask rates remain near 0.15.
3. A synthetic gradient check using first-update shapes verified the required loss weighting: masked-token weighted microbatch accumulation matches full-batch deterministic gradients below 1.2e-9 max absolute difference. Naive averaging of microbatch mean losses changes gradients (max observed 5.39e-4), so future trainer code must use masked-token weighting.

## Remaining implementation work before any H100 launch

- Build the actual chunk-stream trainer from these preflight rules, not by patching the prefix-slicing schedule.
- Dry-run a complete 10M stage epoch through the trainer without model forward/backward, verifying exact chunk use, 253 updates, exact 10M charged words, active-token totals, continuation chunk handling, one full-effective-batch WWM call per update, and continuous RNG/scheduler/global-step state.
- Keep optimizer, warmup, scheduler, masking state, clipping, and data order continuous across stage changes for paired arms.
- Record per-update charged words, active tokens, realized masked tokens, selected word groups, microbatch count, and stage length.

## Route status

Do not launch an experience-utilization run while the depth comparison is still pending. If the depth vector leaves a gap and support-floor evidence does not resolve it, the clean first wave is:

1. `U256_chunked_visibility_mask015`
2. `U64_128_256_chunked_order_mask015`

on the same tokenizer/backbone/seed chosen from depth and support-floor evidence. The result would be scientifically useful even if it misses SOTA because it distinguishes wasted charged experience from the short-to-long bundle and can support a general sample-efficient pretraining principle.

Evidence files:

- `experiments/archive/representation_and_objectives/data/experience_utilization_design/experience_utilization_experiment_design.json`
- `experiments/archive/representation_and_objectives/data/chunk_stream_preflight/chunk_stream_preflight.json`
- `experiments/archive/representation_and_objectives/data/microbatch_weighting_equivalence/microbatch_weighting_equivalence.json`
- `data/external/independent_review01_verifier1_integration.md`
