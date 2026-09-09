# earlier analysis familywise arm-growth decomposition

This readout separates `V_d−V_1` from `R_d−R_1` on the common 10M-80M window, using existing earlier analysis scores only.

## MAX dose: stable-family means

| family/composite | V-C | R-C | V-R | Vd-V1 | Rd-R1 | Δ(V-R) vs 1x |
|---|---:|---:|---:|---:|---:|---:|
| BLiMP | +1.0825 | +1.2875 | -0.2050 | +0.0688 | +0.0700 | -0.0013 |
| Supplement | +1.2125 | +0.1087 | +1.1038 | +0.9350 | -0.7350 | +1.6700 |
| EWoK | +0.3213 | +0.9275 | -0.6062 | +0.2700 | -0.0063 | +0.2763 |
| Entity | +2.4413 | +0.3788 | +2.0625 | +0.9863 | -0.5175 | +1.5037 |
| COMPS | -0.1387 | -0.2313 | +0.0925 | -0.5600 | -0.3625 | -0.1975 |
| Reading | +0.5131 | +0.5800 | -0.0669 | -0.1675 | +0.3281 | -0.4956 |
| cheap6_no_GlobalPIQA | +0.9053 | +0.5085 | +0.3968 | +0.2554 | -0.2039 | +0.4593 |
| cheap6_exEntity | +0.5981 | +0.5345 | +0.0636 | +0.1093 | -0.1411 | +0.2504 |
| cheap5_no_GlobalPIQA_Reading | +0.9837 | +0.4943 | +0.4895 | +0.3400 | -0.3102 | +0.6502 |
| cheap5_exEntity | +0.6194 | +0.5231 | +0.0963 | +0.1784 | -0.2584 | +0.4369 |

## Mechanism reading
- Entity: At MAX on the common 10M-80M first-basin window, Entity V-R is not explained by repeat collapse alone: relative to 1x, the view arm rises by about +0.99 pp while the repeat arm falls by about -0.52 pp, giving about +1.50 pp V-R growth. This still does not prove record addressability, because changed state bias threat and route decision showed the view-arm movement can be operation-propensity shaped.
- Ex-Entity: Removing Entity leaves MAX V-R near the earlier seed-noise scale, even though both MAX view and MAX repeat remain above the 1x-geometry clean arm. If Entity does not reproduce in the second basin, the surviving result is a fixed-budget allocation contrast, not a source-correspondence mechanism.
- Route consequence: the prepared permuted-companion arm should remain unlaunched unless second-basin official Entity and breadth results show a reproducible aligned carrier that is not mostly an operation-propensity redistribution.

## Files
- csv: `experiments/archive/frontier_consolidation/data/family_arm_growth_decomposition/family_arm_growth_common10_80.csv`
- summary_json: `experiments/archive/frontier_consolidation/data/family_arm_growth_decomposition/family_arm_growth_decomposition_summary.json`
- summary_md: `research/documents/frontier_consolidation/data/family_arm_growth_decomposition/family_arm_growth_decomposition_summary.md`
