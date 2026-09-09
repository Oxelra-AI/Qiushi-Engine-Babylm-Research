# protected 8x480 10m available coordinate — S1 12×384 vs protected 8×480 at matched 10M

Evidence JSON: `experiments/archive/initial_model_studies/data/s1_vs_protected_10m_comparison.json`

| column | S1 12×384 10M | protected 8×480 10M | S1 - protected10M | protected 8×480 100M | protected gain 10M→100M |
|---|---:|---:|---:|---:|---:|
| BLiMP | 53.37 | 54.17 | -0.80 | 66.76 | +12.59 |
| Supplement | 52.34 | 53.41 | -1.07 | 59.88 | +6.47 |
| Entity | 17.73 | 17.44 | +0.29 | 22.62 | +5.18 |
| COMPS | 50.34 | 49.95 | +0.39 | 52.19 | +2.24 |
| GlobalPIQA parallel | 19.42 | 15.53 | +3.89 | 24.27 | +8.74 |
| GlobalPIQA nonparallel | 51.00 | 46.00 | +5.00 | 47.00 | +1.00 |
| GlobalPIQA mean | 35.21 | 30.77 | +4.45 | 35.63 | +4.87 |
| Reading eye | 11.69 | 11.45 | +0.24 | 9.95 | -1.50 |
| Reading self-paced | 5.01 | 5.00 | +0.01 | 5.29 | +0.29 |
| Reading mean | 8.35 | 8.22 | +0.12 | 7.62 | -0.60 |

## Interpretation

- S1 shape alone is not a broad 10M winner: it loses BLiMP (-0.80) and Supplement (-1.07) versus protected 8x480 at the same 10M exposure.
- S1 has a meaningful target-cluster sign at 10M: GlobalPIQA mean +4.445, Entity +0.29, COMPS +0.39, and Reading mean +0.125 versus protected 8x480 10M.
- The GlobalPIQA 10M gain is almost as large as the protected 8x480 model's entire 10M-to-100M GlobalPIQA gain (+4.87), so 12x384 allocation changes the physical/commonsense profile rather than merely noise.
- Entity remains weak: +0.29 at 10M does not approach the leader gap. Shape alone probably does not reproduce the leader package, but it may be a useful base for curriculum/tokenizer/data factors.
- A same-gain extrapolation is only a route heuristic, not evidence; it suggests S1 could reach strong GlobalPIQA (~40.08) while still needing help on Entity and grammar/supplement. Full S1 100M evidence is scientifically worthwhile before closing the shape branch.

## Next execution implication

S1 is not sufficient as a 10M endpoint, but its +4.45 GlobalPIQA mean over the protected 10M baseline is a real target-cluster signal. The next strongest execution is to run S1 shape-only to 100M (with the same legal official corpus, flat WWM, effective batch 256/micro 128, isolated fork, preferably 1M checkpoints if storage permits) alongside a separately designed S2 curriculum arm. If S1 100M preserves the GlobalPIQA advantage and recovers grammar with exposure, add leader-style WWM→token and length curriculum. If it loses the target signal, prioritize curriculum/tokenizer/data reconstruction or hybrid/MNTP.
