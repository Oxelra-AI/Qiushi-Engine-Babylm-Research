# mechanism preservation summary — pairfit joint-visibility contract

JSON: `experiments/archive/frontier_consolidation/data/pairfit_contract/pairfit_joint_visibility_contract.json`

Status ok: `True`

The actual materialized pairfit corpus keeps every Qwen original+rewrite pair as one training example and assigns it to the shortest containing 40k-token stage. This supersedes the earlier conservative simulated `candidate_pair_atomic` number in `pair_joint_visibility_audit.md` for the actual pairfit arm.

| stage | seq | rows | words | Qwen pair rows | Qwen pair words | max pair tokens | complete joint rate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 89684 | 3000000 | 27144 | 998731 | 64 | 1.0 |
| 2 | 128 | 54494 | 3000000 | 10259 | 640856 | 128 | 1.0 |
| 3 | 256 | 25084 | 4000000 | 191 | 17213 | 187 | 1.0 |

All selected Qwen pairs seen exactly once: `True`.
Qwen pair words: `1656800`; total words: `10000000`.
Truncating Qwen pair examples under assigned stage windows: `0`.

Scientific reading: row-chunked repaired pilot eval is a useful low-hidden-word training-dynamics control but breaks complete pair visibility for many early pairs. Pairfit is the correct 10M comparator for asking whether leader-style sequence/masking curriculum can coexist with the clean-Qwen paired-view mechanism.
