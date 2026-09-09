# sequence curriculum loop measurement word-mean training dynamics
CPU-only reading of training artifacts after the 80M word-mean screen finished. No official-compatible evaluation was run or polled here.

## Matched run facts
- Token-mean reference: `experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2`.
- Word-mean screen: `experiments/archive/frontier_consolidation/training/runs/wordmean_mlm_complianttok_reinvest_seed43022_80M`.
- Checkpoint exposure alignment identical through 80M: `False`; common checkpoints `80`.
- Word-mean finished: exposure `80000000`, steps `2024`, last loss `2.513671`, mean tokens per selected group trace `1.458269`.

## Loss windows (not directly comparable as competence because objectives differ)
| window | token loss mean/last | word loss mean/last | token eff mask | word eff mask |
|---|---:|---:|---:|---:|
| 0-5M | 7.0703/5.9490 | 6.9700/5.6150 | 0.1502 | 0.1502 |
| 15-20M | 3.8029/3.7195 | 3.7763/3.6979 | 0.1500 | 0.1500 |
| 65-70M | 2.6040/2.5362 | 2.6077/2.5927 | 0.1501 | 0.1501 |
| 75-80M | 2.5276/2.5405 | 2.5731/2.5137 | 0.1503 | 0.1503 |

## Scientific reading
- The word-mean screen is a real matched 80M trajectory on the same corpus/tokenizer/architecture/seeds/checkpoint exposures, so the pending cheap-column evaluation will be interpretable as an objective-normalization screen.
- The scalar loss cannot decide the route because token-mean and word-mean optimize different reductions. The previous wordmean screen and substrate constraints credit profile remains more informative about mechanism: word-mean shifts relative credit toward one-piece function words and away from multi-piece content.
- Do not continue to 100M or start the minfreq50 fallback from training loss alone; wait for the managed 70M/80M official-compatible columns.

Full JSON: `experiments/archive/frontier_consolidation/data/wordmean_training_dynamics/wordmean_training_dynamics.json`
