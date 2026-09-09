# pairaware sequence accounting tokenizer-group chunk measurement
CPU-only training-side measurement of faithful sequence chunks in the actual WWM group stream. No model was trained and no official evaluation text was read.

## Inputs
- Pool SHA matched: `True`.
- legal16k: vocab 16384, tokenizer SHA matched `True`.
- minfreq50_supportfloor: vocab 19609, tokenizer SHA matched `True`.

## Full-pool group stream
| tokenizer | words | raw tokens | groups | groups/word | no-word groups | words without group | over256 rows | truncated tokens vs256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| legal16k | 10000000 | 14664519 | 10000000 | 1.000000 | 0 | 0 | 15117 | 369626 |
| minfreq50_supportfloor | 10000000 | 14483600 | 10000000 | 1.000000 | 0 | 0 | 14096 | 330696 |

## Group chunks by length
Tokenizer-group chunks include separator-space tokens and never split WWM groups. Ratios compare against the inherited prefix-slicing path over the same rows and word budget.

| tokenizer | L | group chunks/epoch | active-token ratio vs prefix | step ratio vs prefix | chunk groups median/p90 | chunk words median/p90 | overlong groups |
|---|---:|---:|---:|---:|---:|---:|---:|
| legal16k | 64 | 264588 | 3.539 | 1.024 | 41.0/52.0 | 41.0/52.0 | 5 |
| legal16k | 128 | 144471 | 1.772 | 1.119 | 74.0/100.0 | 74.0/100.0 | 0 |
| legal16k | 256 | 79882 | 1.026 | 1.237 | 155.0/160.0 | 155.0/160.0 | 0 |
| minfreq50_supportfloor | 64 | 261722 | 3.496 | 1.012 | 41.0/52.0 | 41.0/52.0 | 5 |
| minfreq50_supportfloor | 128 | 143340 | 1.750 | 1.107 | 74.0/101.0 | 74.0/101.0 | 0 |
| minfreq50_supportfloor | 256 | 78857 | 1.023 | 1.221 | 156.0/160.0 | 156.0/160.0 | 0 |

## Ten-epoch schedule ratios
| tokenizer | schedule | charged words | active-token ratio vs prefix | optimizer-step ratio vs prefix |
|---|---|---:|---:|---:|
| legal16k | 64x3_128x4_256x3 | 100000000 | 1.658 | 1.126 |
| legal16k | 64x7_256x3 | 100000000 | 2.040 | 1.088 |
| minfreq50_supportfloor | 64x3_128x4_256x3 | 100000000 | 1.646 | 1.113 |
| minfreq50_supportfloor | 64x7_256x3 | 100000000 | 2.027 | 1.075 |

## Compact-pair preservation in group chunks
Fractions are over the 12,155 source+rewrite pairs in the compact changed block. Pair-atomic chunks pack each source+rewrite pair as one atom when it fits the stage length.

| tokenizer | L | prefix full | greedy full | greedy any cochunk | pair-atomic full | pair-atomic overlong |
|---|---:|---:|---:|---:|---:|---:|
| legal16k | 64 | 0.184 | 0.387 | 0.977 | 0.797 | 0.203 |
| legal16k | 128 | 0.496 | 0.756 | 0.991 | 0.998 | 0.002 |
| legal16k | 256 | 0.995 | 0.996 | 1.000 | 1.000 | 0.000 |
| minfreq50_supportfloor | 64 | 0.192 | 0.402 | 0.977 | 0.816 | 0.184 |
| minfreq50_supportfloor | 128 | 0.513 | 0.759 | 0.991 | 0.999 | 0.001 |
| minfreq50_supportfloor | 256 | 0.998 | 0.998 | 1.000 | 1.000 | 0.000 |

## Pair preservation over ten epochs
| tokenizer | schedule | prefix full | greedy full | pair-atomic full | pair-atomic overlong |
|---|---|---:|---:|---:|---:|
| legal16k | 64x3_128x4_256x3 | 0.552 | 0.717 | 0.938 | 0.062 |
| legal16k | 64x7_256x3 | 0.427 | 0.569 | 0.858 | 0.142 |
| minfreq50_supportfloor | 64x3_128x4_256x3 | 0.562 | 0.723 | 0.944 | 0.056 |
| minfreq50_supportfloor | 64x7_256x3 | 0.434 | 0.580 | 0.871 | 0.129 |

## Scientific reading
- Direct tokenizer-group chunks raise the active-token ratios slightly above the whitespace-span estimate because separator-space tokens are now included; the word accounting still remains the same 10M declared words per epoch.
- The group stream is almost one-to-one with declared words, so a faithful group-chunk trainer can preserve the BabyLM word budget while exposing suffix tokens hidden by prefix slicing.
- Pair-aware chunking remains necessary: naive group chunks improve suffix exposure but split many L64 source+rewrite atoms; pair-atomic group chunks preserve most L64 pairs and almost all L128/L256 pairs.
- This is construction evidence for a possible later sequence route, not a reason to interrupt the running word-mean evaluation or the prepared minfreq50 path.

Full JSON: `experiments/archive/frontier_consolidation/data/tokenizer_group_chunk_measurement/tokenizer_group_chunk_measurement.json`
CSV: `experiments/archive/frontier_consolidation/data/tokenizer_group_chunk_measurement/group_chunk_by_length.csv`, `experiments/archive/frontier_consolidation/data/tokenizer_group_chunk_measurement/pair_preservation_by_length.csv`
