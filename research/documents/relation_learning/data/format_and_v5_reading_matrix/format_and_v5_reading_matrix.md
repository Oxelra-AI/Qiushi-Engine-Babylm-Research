# earlier analysis format and v5 reading matrix

This fixes the quantitative comparison frame before the pending endpoint outputs are read.

## Coherent private two-seed band versus chck82

| column | seed43022 Δ | seed43122 Δ | min | max | width |
|---|---:|---:|---:|---:|---:|
| BLiMP | 0.01871596348014748 | -0.14128403651986332 | -0.14128403651986332 | 0.01871596348014748 | 0.1600000000000108 |
| Supplement | 0.7021887437998018 | 0.7721887437998021 | 0.7021887437998018 | 0.7721887437998021 | 0.07000000000000028 |
| EWoK | -0.03545332553275671 | -0.15545332553276126 | -0.15545332553276126 | -0.03545332553275671 | 0.12000000000000455 |
| Entity | 0.00595806970122581 | -0.5140419302987738 | -0.5140419302987738 | 0.00595806970122581 | 0.5199999999999996 |
| COMPS | -0.14117509443596532 | -0.22117509443596362 | -0.22117509443596362 | -0.14117509443596532 | 0.0799999999999983 |
| GlobalPIQA | 0.98733009708738 | -1.9576699029126203 | -1.9576699029126203 | 0.98733009708738 | 2.9450000000000003 |
| Reading | 0.01628641073809689 | 0.021286410738097672 | 0.01628641073809689 | 0.021286410738097672 | 0.005000000000000782 |
| cheap7 | 0.22197869497684053 | -0.31373559073744417 | -0.31373559073744417 | 0.22197869497684053 | 0.5357142857142847 |

Old stripped-path coherent86 Overall: `42.12102470996659`. Faithful v4 must replace SuperGLUE with the repaired AutoModel result before a trained v5 is judged.

## Pending format payloads

| arm | seed | train summary? | payload? | payload path |
|---|---:|---:|---:|---|
| coherent_unsplit_special | 98097 | True | True | `experiments/archive/relation_learning/data/eval_format_replay_corrected/coherent_unsplit_special/eval/coherent_unsplit_special_seed98097_alpha0p75/per_target/coherent_unsplit_special_seed98097_alpha0p75.json` |
| coherent_unsplit_special | 98098 | True | True | `experiments/archive/relation_learning/data/eval_format_replay_corrected/coherent_unsplit_special/eval/coherent_unsplit_special_seed98098_alpha0p75/per_target/coherent_unsplit_special_seed98098_alpha0p75.json` |
| isolated_all | 98097 | False | False | `experiments/archive/relation_learning/data/eval_format_replay_corrected/isolated_all/eval/isolated_all_seed98097_alpha0p75/per_target/isolated_all_seed98097_alpha0p75.json` |
| isolated_all | 98098 | False | False | `experiments/archive/relation_learning/data/eval_format_replay_corrected/isolated_all/eval/isolated_all_seed98098_alpha0p75/per_target/isolated_all_seed98098_alpha0p75.json` |
| half_coherent_half_isolated | 98097 | True | False | `experiments/archive/relation_learning/data/eval_format_replay_corrected/half_coherent_half_isolated/eval/half_coherent_half_isolated_seed98097_alpha0p75/per_target/half_coherent_half_isolated_seed98097_alpha0p75.json` |
| half_coherent_half_isolated | 98098 | False | False | `experiments/archive/relation_learning/data/eval_format_replay_corrected/half_coherent_half_isolated/eval/half_coherent_half_isolated_seed98098_alpha0p75/per_target/half_coherent_half_isolated_seed98098_alpha0p75.json` |

## Reading order

- 1. Read training summaries/logs for all six format endpoints and reject only runs with broken masking, wrong word/update count, missing alpha endpoint, or failed initial function equality.
- 2. Read coherent_unsplit_special first as special-token exposure on the exact coherent86 suffix words.
- 3. Read isolated_all next as isolation beyond special-token exposure.
- 4. Read half_coherent_half_isolated last as context-presence conditioning.
- 5. Compare columns to coherent86/v4 and chck82, then read item localization for replication and mechanism rather than as a row-count-weighted score.
- 6. A composed training candidate may use only levers with same-direction two-seed column movement and no important column falling outside the coherent private two-seed band; after selection it must be trained and evaluated directly over two private seeds.

JSON: `experiments/archive/relation_learning/data/format_and_v5_reading_matrix/format_and_v5_reading_matrix.json`
