# sequence curriculum loop measurement sequence-curriculum loop measurement
CPU-only direct measurement on exact compact-view-reinvest corpus. No model was trained or evaluated.

## Inputs
- 10M pool SHA matched: `True`; 100M train SHA matched: `True`.
- Tokenizers measured: legal16k, minfreq50_supportfloor.

## Direct existing-loop first-step measurement
The current local `seq_len_schedule` path constructs 256-token padded/truncated rows, pops and debits the full row word count, then slices tensors to the current length. The table shows the first actual 256-row batch under this path.

| tokenizer | L | debited words | visible groups | full256 groups | hidden group frac | visible groups/debited word | selected groups sample | masked tokens sample |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| legal16k | 64 | 39370 | 11538 | 38595 | 0.701 | 0.293 | 1669 | 2406 |
| legal16k | 128 | 39370 | 22881 | 38595 | 0.407 | 0.581 | 3341 | 4753 |
| legal16k | 256 | 39370 | 38595 | 38595 | 0.000 | 0.980 | 5738 | 8330 |
| minfreq50_supportfloor | 64 | 39370 | 11681 | 38691 | 0.698 | 0.297 | 1699 | 2372 |
| minfreq50_supportfloor | 128 | 39370 | 23174 | 38691 | 0.401 | 0.589 | 3391 | 4759 |
| minfreq50_supportfloor | 256 | 39370 | 38691 | 38691 | 0.000 | 0.983 | 5752 | 8371 |

## Full-pool faithful chunking/accounting alternative
A faithful stage-length regime partitions the same 10M words into word-boundary chunks for each stage length; every word is debited exactly once per epoch. Extra target tokens are suffix content that prefix slicing hides, not extra words.

| tokenizer | L | prefix hidden word frac | faithful chunks/epoch | charged words/epoch | active-token ratio vs prefix | step ratio vs prefix | chunk words median/p90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| legal16k | 64 | 0.707 | 263212 | 10000000 | 3.518 | 1.020 | 41.0/52.0 |
| legal16k | 128 | 0.419 | 143554 | 10000000 | 1.761 | 1.111 | 74.0/100.0 |
| legal16k | 256 | 0.018 | 79021 | 10000000 | 1.020 | 1.221 | 156.0/160.0 |
| minfreq50_supportfloor | 64 | 0.704 | 260319 | 10000000 | 3.475 | 1.008 | 41.0/52.0 |
| minfreq50_supportfloor | 128 | 0.412 | 142469 | 10000000 | 1.740 | 1.103 | 74.0/101.0 |
| minfreq50_supportfloor | 256 | 0.016 | 78012 | 10000000 | 1.017 | 1.206 | 157.0/160.0 |

## Ten-epoch schedule ratios
| tokenizer | schedule | charged words | target-token ratio faithful/prefix | optimizer-step ratio faithful/prefix |
|---|---|---:|---:|---:|
| legal16k | 64x3_128x4_256x3 | 100000000 | 1.649 | 1.117 |
| legal16k | 64x7_256x3 | 100000000 | 2.028 | 1.080 |
| minfreq50_supportfloor | 64x3_128x4_256x3 | 100000000 | 1.636 | 1.105 |
| minfreq50_supportfloor | 64x7_256x3 | 100000000 | 2.015 | 1.067 |

## Scientific reading
- A future sequence-curriculum run should not use the existing prefix-slicing path as evidence for a leader-style 64→256 curriculum: short stages hide a large suffix fraction while debiting full row words.
- The compliant faithful-chunking regime is well-defined in word-budget terms: the same 10M words are partitioned differently, and a ten-epoch curriculum remains 100M charged words.
- This measurement does not justify a GPU launch while the word-mean screen is unresolved. If sequence length becomes the next route, the trainer must implement the faithful chunk stream, not patch `seq_len_schedule` by slicing prefixes.

Full JSON: `experiments/archive/frontier_consolidation/data/sequence_curriculum_loop_measurement/sequence_curriculum_loop_measurement.json`
CSV files: `experiments/archive/frontier_consolidation/data/sequence_curriculum_loop_measurement/current_loop_first_step_by_length.csv`, `experiments/archive/frontier_consolidation/data/sequence_curriculum_loop_measurement/faithful_chunking_by_length.csv`, `experiments/archive/frontier_consolidation/data/sequence_curriculum_loop_measurement/schedule_ratios.csv`
