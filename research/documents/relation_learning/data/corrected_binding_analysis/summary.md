# research synthesis correction: Corrected binding factorial analysis (paired-null coordinate)

## Correction
The earlier analysis independence-excess coordinate (joint - A·B/n) is structurally wrong
for paired rows sharing identical context. A non-gating model gets exactly one
per pair right, so the no-gating null is **joint = 0**.

## Epoch 20 all-pairs (n=200)

| arm | joint | gated% | both_wrong | A_only | B_only | prior_B% | A_margin | B_margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| answer_clean | 44 | 22.0% | 0 | 39 | 117 | 75.0% | -0.307 | 1.480 |
| uniform_wwm | 7 | 3.5% | 1 | 57 | 135 | 70.3% | -0.583 | 0.706 |
| answer_corrupt_update_state | 30 | 15.0% | 1 | 86 | 83 | 49.1% | 0.438 | 0.428 |

## Epoch 20 updated-entity-in-source (n=119)

| arm | joint | gated% | both_wrong |
|---|---:|---:|---:|
| answer_clean | 24 | 20.2% | 0 |
| uniform_wwm | 2 | 1.7% | 0 |
| answer_corrupt_update_state | 13 | 10.9% | 1 |

## Joint trajectory (all_pairs, n=200)

| epoch | answer_clean | uniform_wwm | corrupt_update |
|---:|---:|---:|---:|
| 0 | 6 (3.0%) | 6 (3.0%) | 6 (3.0%) |
| 5 | 18 (9.0%) | 5 (2.5%) | 14 (7.0%) |
| 10 | 30 (15.0%) | 7 (3.5%) | 25 (12.5%) |
| 15 | 35 (17.5%) | 6 (3.0%) | 30 (15.0%) |
| 20 | 44 (22.0%) | 7 (3.5%) | 30 (15.0%) |

## Interpretation

Under the corrected coordinate:
- base (epoch 0): 6/200 = 3.0% gating, shared across all arms
- answer_clean: 6 → 18 → 30 → 35 → 44 (still rising at epoch 20)
- uniform_wwm: 6 → 5 → 7 → 6 → 7 (flat, near base)
- answer_corrupt: 6 → 14 → 25 → 30 → 30 (plateaued ~15%)

Clean answer credit raised gating ~7x from base, still rising.
Uniform WWM installed nothing. Corrupted update support reached half of
clean answer credit and plateaued, showing that visible update evidence
matters but is not sufficient without concentrated credit.

both_wrong ≈ 0 for all arms: every joint success is a genuine flip, not noise.
The residual is prior-governed (B_only >> A_only), not anti-gated.
