# xspan 1m screen result and route — XSpan 1M primary-objective screen result

## Verdict
The compact-v4 XSpan primary-objective route is **not justified for 10M/20M or 100M scaling in its current form**.

The data artifact and trainer are clean, and the span objective is learned, but at 1M it does **not** induce meaningful correct-s1 specificity and does **not** transfer to the target BabyLM task cluster in a guard-safe way.

## Mechanism evidence
Source: `data/xspan_1m_mechanism_comparison.json`

| arm | rho | context | counted words | WWM loss last | XSpan loss last | true loss | wrong loss | no loss | Δ true-wrong | Δ true-no |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| XSpan true-s1 | 0.15 | true_s1 | 1,018,928 | 6.1516 | 8.2673 | 8.1065 | 8.1099 | 8.1468 | +0.0034 | +0.0403 |
| XSpan wrong-s1 | 0.15 | wrong_s1 | 1,018,897 | 6.2680 | 8.3478 | 8.2279 | 8.2284 | 8.2383 | +0.0005 | +0.0104 |
| WWM-only | 0.0 | true_s1 | 1,000,000 | 6.0933 | 0.0000 | 8.8194 | 8.8209 | 8.8223 | +0.0015 | +0.0029 |

Interpretation:
- Both XSpan arms greatly improve absolute target-span likelihood vs WWM-only, so the primary span objective is learnable.
- But true-s1 specificity is tiny: true-s1 arm improves heldout true-minus-wrong by only **+0.00287 logprob/token over wrong-s1 control** and **+0.00189 over WWM-only**.
- Therefore the current objective mostly teaches target-span familiarity, not load-bearing use of correct s1.
- rho=0.15 also worsens ordinary WWM loss (+0.058 vs WWM-only for true-s1; +0.175 for wrong-s1), warning that guard columns can suffer.

## Available BabyLM task subset
Source: `data/xspan_1m_available_coordinate_manifest.json`

| arm | BLiMP | Supp | Entity | COMPS | GlobalPIQA mean | Reading mean |
|---|---:|---:|---:|---:|---:|---:|
| XSpan true-s1 | 55.97 | 45.75 | 17.63 | 50.17 | 31.30 | 5.98 |
| XSpan wrong-s1 | 57.86 | 47.14 | 17.65 | 49.74 | 30.78 | 6.04 |
| WWM-only rho0 | 55.27 | 50.09 | 17.44 | 49.73 | 33.25 | 6.29 |

Deltas:
- true-s1 vs WWM: BLiMP +0.70, Supplement **−4.34**, Entity +0.19, COMPS +0.44, GlobalPIQA mean **−1.95**, Reading −0.32.
- true-s1 vs wrong-s1: BLiMP −1.89, Supplement −1.39, Entity −0.02, COMPS +0.43, GlobalPIQA mean +0.52, Reading −0.06.
- wrong-s1 vs WWM: BLiMP +2.59, Supplement −2.95, Entity +0.21, COMPS +0.01, GlobalPIQA mean −2.47, Reading −0.25.

Interpretation:
- Entity moves only +0.19 over WWM, and wrong-s1 control moves +0.21: no evidence of correct-s1 transfer.
- GlobalPIQA mean, one of the main target gaps, is worse than WWM by −1.95 for true-s1 and −2.47 for wrong-s1.
- Supplement is badly harmed, especially true-s1 (−4.34 vs WWM).
- The only favorable true-vs-WWM columns are BLiMP and COMPS, but BLiMP is stronger in the wrong-s1 control and COMPS is too small/narrow to justify route scaling.

## Scientific conclusion
The compact-v4 XSpan representation was valuable as a mechanism probe: protected-model likelihood proved some official corpus adjacent-sentence content spans are s1-specific. But training a randomly initialized 1M model with rho=0.15 on those spans does not yet create a transferable s1-binding mechanism. It creates span familiarity and disrupts guard columns.

This route should not be scaled as-is. The next research move is not v5 data filtering. It is a change in objective/representation, for example:
- a staged or architecture-level objective where s1 is compressed and must be used before s2 span prediction, rather than concatenation letting local target statistics dominate;
- a contrastive or retrieval-style target that changes the *candidate distribution* while still using the main LM scoring path, with careful shortcut controls;
- or a broader pivot back to data/architecture routes aimed at Entity/EWoK/GlobalPIQA that do not rely on these target-span masks.

The result calls for reassessing route value and designing a fundamentally different mechanism, not lowering rho without supporting evidence or scaling to 10M/20M.
