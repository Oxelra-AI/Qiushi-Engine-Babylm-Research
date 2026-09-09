# earlier analysis persistence mechanism readout

This note joins the earlier analysis own changed/unchanged row loss measurement with the earlier analysis cross-loss measurement. It is a mechanism readout, not an official benchmark score.

## Late 100M signal

| arm/text set | clean or own prior loss | owner loss | delta |
|---|---:|---:|---:|
| view_changed | 3.2659 | 2.6175 (view) | 0.6483 clean_minus_owner |
| repeat_changed | 2.9494 | 2.0399 (repeat) | 0.9095 clean_minus_owner |
| breadth_changed | 5.2665 | 4.7812 (breadth) | 0.4853 clean_minus_owner |
| clean_displaced | 2.9885 | 2.9885 (clean) | 0.0000 clean_minus_owner |

### Own changed vs unchanged at 100M

| arm | changed loss | unchanged loss | changed-minus-unchanged |
|---|---:|---:|---:|
| view | 2.5506 | 2.7599 | -0.2093 |
| repeat | 1.9490 | 2.7368 | -0.7878 |
| breadth | 4.7281 | 2.7902 | 1.9380 |
| clean | 2.9783 | 2.7974 | 0.1809 |

### Displaced clean rows at 100M

| intervention arm | loss on clean-displaced rows | clean loss on same rows | intervention-minus-clean |
|---|---:|---:|---:|
| view | 3.1039 | 2.9885 | 0.1153 |
| repeat | 3.1116 | 2.9885 | 0.1230 |
| breadth | 3.0895 | 2.9885 | 0.1010 |

## Scientific reading

- Exact recurrence drives very low own loss on repeated changed rows by 100M, but this coincides with near-zero late broad downstream advantage from earlier reference and margin state/287 scores; low training loss alone is not transferable competence.
- Distinct breadth rows remain high-loss even after training and are very hard for the clean model, matching the idea that distinct admitted material continues to supply residual prediction error instead of becoming exhausted.
- Compact view rows are intermediate: learned substantially better than clean, but not as over-compressed as exact repeat; this is compatible with why view retains late broad benefit while exact repeat decays.
- Sampled displaced clean rows are only about 0.10-0.12 loss worse under view/repeat/breadth than under clean at 100M, so the late V/B versus R difference is not explained by a large differential loss collapse on these sampled sacrificed rows.

## What remains unsettled

- This is forward-only fixed-mask MLM loss, not a direct gradient norm measurement; it estimates residual prediction error rather than the exact parameter update contribution.
- The samples are small deterministic row samples (earlier analysis n=300 own rows, earlier analysis n=200 cross rows), not full-corpus loss integrals.
- The readout does not by itself prove that the residual error is evaluation-relevant; official family scores from the corrected earlier analysis scorer are still needed.
- It does not remove the known architecture boundary: RoBERTa did not reproduce the broad late DeBERTa view-clean sign.

## Current formulation

Under this DeBERTa/WWM/fixed-budget coordinate, repeated exposure can make admitted duplicate text easy without maintaining broad downstream value; distinct adult-distribution content remains partly unpredictable across passes and can keep supplying useful learning pressure. The principle is therefore about persistent nonredundant error on useful distributions, plus the opportunity cost of what is removed, not simply word count, surface compactness, or recurrence.

JSON: `experiments/archive/frontier_consolidation/data/persistence_mechanism_readout/persistence_mechanism_readout.json`
