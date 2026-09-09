# sequence curriculum loop measurement — Sequence-curriculum loop measurement

CPU-only measurement on compact_view_reinvest. No model was trained or evaluated.

## Direct current-loop measurement

### legal40k
- First effective batch: 256 rows, 39370 words debited.
- L64 sees 12119 of 38915 full-256 word groups (0.311) while debiting the full words; sampled WWM selected 1781 groups / 2436 tokens.
- L128 sees 0.620 of full-256 word groups; L256 sees 1.000.

### minfreq25
- First effective batch: 256 rows, 39370 words debited.
- L64 sees 11976 of 38848 full-256 word groups (0.308) while debiting the full words; sampled WWM selected 1757 groups / 2368 tokens.
- L128 sees 0.612 of full-256 word groups; L256 sees 1.000.

This directly confirms the code reading: the current trainer debits `batch.words` before slicing to the stage length, so suffix word groups hidden by L64/L128 still consume word exposure.

## Compliant faithful chunking prototype

Regime used here: each stage epoch partitions the same 10M whitespace words into word-boundary chunks at the stage length; every word is charged once per epoch. A 10-epoch 64/128/256 curriculum remains 100M charged-word exposure.

### legal40k
- Untruncated tokenization: 13942644 tokens / 10000000 words = 1.3943 tokens per word; unassigned offsets 0.
- Schedule 64x3_128x4_256x3: faithful chunking gives 1.609x active tokens at 1.081x optimizer steps, with 100000000 charged words.
- Schedule 64x7_256x3: faithful chunking gives 1.988x active tokens at 1.037x optimizer steps, with 100000000 charged words.
- Per epoch hidden-by-prefix words: L64 0.694, L128 0.392, L256 0.013.
- Faithful chunks per epoch: L64 252269 with batch 1024; L128 140303 with batch 512; L256 76164 with batch 256.

### minfreq25
- Untruncated tokenization: 14138512 tokens / 10000000 words = 1.4139 tokens per word; unassigned offsets 0.
- Schedule 64x3_128x4_256x3: faithful chunking gives 1.622x active tokens at 1.092x optimizer steps, with 100000000 charged words.
- Schedule 64x7_256x3: faithful chunking gives 2.002x active tokens at 1.050x optimizer steps, with 100000000 charged words.
- Per epoch hidden-by-prefix words: L64 0.698, L128 0.400, L256 0.014.
- Faithful chunks per epoch: L64 255774 with batch 1024; L128 141373 with batch 512; L256 77092 with batch 256.

## Route implication

A future 64->256 route should not use the existing `seq_len_schedule` prefix path as the leader-style factor. The faithful version is a stage-specific chunk stream with inverse row-batch scaling and explicit charged-word accounting. It changes target-token exposure substantially while keeping the same 10M words per epoch; therefore it is a real training intervention and should be launched only after the current depth vector and support-floor vector are read.

JSON: `experiments/archive/representation_and_objectives/data/sequence_curriculum_loop_measurement/sequence_curriculum_loop_measurement.json`
CSV: `experiments/archive/representation_and_objectives/data/sequence_curriculum_loop_measurement/current_loop_first_step_by_length.csv`, `experiments/archive/representation_and_objectives/data/sequence_curriculum_loop_measurement/faithful_chunking_by_length.csv`, `experiments/archive/representation_and_objectives/data/sequence_curriculum_loop_measurement/faithful_chunking_by_schedule.csv`
