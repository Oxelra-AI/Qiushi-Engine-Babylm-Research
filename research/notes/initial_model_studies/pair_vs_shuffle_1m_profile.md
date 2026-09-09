# pair vs shuffle 1m profile — decisive pair-adjacent vs pair-shuffled at 1M (WWM base)

Evidence JSON: `experiments/archive/initial_model_studies/data/pair_vs_shuffle_1m_profile.json`

Both arms share the exact same source/target sentence multiset, total words (999,995), example count (21,080), zero truncation at 256, and identical total tokens/WWM groups. Only whether each source is adjacent to its own rewrite (adjacent) or to an unrelated target (shuffled) differs. Base: BERT-WWM, baseline 16k tokenizer, fixed 256, AdamW, paired seeds 42/43.

| seed | arm | BLiMP | Supp | EWoK | Entity | COMPS | Read eye | Read SPR |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | pair_adjacent | 54.60 | 46.00 | 50.91 | 16.62 | 49.74 | 4.26 | 0.97 |
| 42 | pair_shuffled | 54.43 | 48.00 | 49.82 | 16.32 | 50.19 | 4.54 | 0.98 |
| 43 | pair_adjacent | 53.86 | 50.80 | 50.64 | 17.02 | 49.76 | 4.66 | 1.04 |
| 43 | pair_shuffled | 54.24 | 51.60 | 49.18 | 15.95 | 49.96 | 4.56 | 1.04 |

## pair-adjacent minus pair-shuffled

| seed | BLiMP | Supp | EWoK | Entity | COMPS | Read eye | Read SPR |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.17 | -2.00 | 1.09 | 0.30 | -0.45 | -0.28 | -0.01 |
| 43 | -0.38 | -0.80 | 1.46 | 1.07 | -0.20 | 0.10 | 0.00 |

## mean delta

| BLiMP | Supp | EWoK | Entity | COMPS | Read eye | Read SPR |
|---:|---:|---:|---:|---:|---:|---:|
| -0.10 | -1.40 | 1.27 | 0.69 | -0.33 | -0.09 | -0.01 |
