# family specific tokenizer predictor — Sequence curriculum accounting audit

CPU-only accounting for a faithful 64→256 sequence curriculum on the exact compact_view_reinvest 10M pool. No model was trained or evaluated.

The public leader states `Sequence length curriculum 64 -> 256` and that batch size is scaled inversely with sequence length. The current COMPACT_EXPERIENCE/legal40k accum training completion schedule instead tokenizes each row at 256, slices a short prefix, and charges the full row words.

## legal40k

- vocab 40000; pool tokens/word 1.3943; token-word-groups/word 1.0000; tokenization elapsed 50.22s
| L | prefix visible groups | prefix missing groups | prefix charged words / visible group | faithful chunks/epoch | faithful steps/epoch | chunk fill |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.306 | 0.694 | 3.263 | 252269 | 247 | 0.864 |
| 128 | 0.608 | 0.392 | 1.644 | 140303 | 275 | 0.776 |
| 256 | 0.987 | 0.013 | 1.013 | 76164 | 298 | 0.715 |

### Ten-epoch schedule accounting

| schedule | prefix visible group frac | prefix missing group frac | faithful/current target-token ratio | faithful/current optimizer-step ratio |
|---|---:|---:|---:|---:|
| two_stage_64x7_256x3 | 0.511 | 0.489 | 1.988 | 1.037 |
| three_stage_64x3_128x4_256x3 | 0.631 | 0.369 | 1.609 | 1.081 |
| mild_128x7_256x3 | 0.722 | 0.278 | 1.408 | 1.114 |

## minfreq25

- vocab 29529; pool tokens/word 1.4139; token-word-groups/word 1.0000; tokenization elapsed 49.97s
| L | prefix visible groups | prefix missing groups | prefix charged words / visible group | faithful chunks/epoch | faithful steps/epoch | chunk fill |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.302 | 0.698 | 3.309 | 255774 | 250 | 0.864 |
| 128 | 0.600 | 0.400 | 1.667 | 141373 | 277 | 0.781 |
| 256 | 0.986 | 0.014 | 1.015 | 77092 | 302 | 0.716 |

### Ten-epoch schedule accounting

| schedule | prefix visible group frac | prefix missing group frac | faithful/current target-token ratio | faithful/current optimizer-step ratio |
|---|---:|---:|---:|---:|
| two_stage_64x7_256x3 | 0.507 | 0.493 | 2.002 | 1.050 |
| three_stage_64x3_128x4_256x3 | 0.626 | 0.374 | 1.622 | 1.092 |
| mild_128x7_256x3 | 0.715 | 0.285 | 1.420 | 1.125 |

## Research use

If the active 12×384 fixed-length run does not cross the frontier, a sequence-curriculum route should use a stage-specific chunking/streaming trainer with inverse row-batch scaling. It should not use the existing `seq_len_schedule` path as evidence for the leader-style factor because that path hides suffix word-groups while charging their words.

JSON: `experiments/archive/representation_and_objectives/data/sequence_curriculum_accounting_audit/sequence_curriculum_accounting_audit.json`
