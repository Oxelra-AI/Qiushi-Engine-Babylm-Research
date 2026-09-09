# private scale endpoint vs mechanism synthesis anchor-margin gate analysis

Status: **COMPLETE**
Scored items raw: `6363`; used for margin analysis: `6359` from `experiments/archive/frontier_consolidation/data/anchor_margin_alpha_census_full/anchor_margin_scored_items.jsonl`
Anchor rescore agreement: `{'agree': 6359, 'n': 6363, 'excluded_mismatches': 4}`

## Per-column separability

| column | n changed | act n | dmg n | act median | dmg median | AUC damage by low |margin| | damage |margin|<0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ALL | 6359 | 3113 | 3230 | -0.1400 | 0.1331 | 0.5113 | 88.54% |
| BLiMP | 1497 | 761 | 731 | -0.2638 | 0.2730 | 0.5084 | 74.83% |
| COMPS | 4336 | 2095 | 2232 | -0.1116 | 0.1046 | 0.5105 | 93.77% |
| EWoK | 297 | 147 | 149 | -0.0887 | 0.1341 | 0.4604 | 90.60% |
| Entity | 140 | 73 | 66 | -0.2742 | 0.1840 | 0.5890 | 78.79% |
| GlobalPIQA | 5 | 3 | 2 | -0.0374 | 0.0866 | 0.3333 | 100.00% |
| Supplement | 84 | 34 | 50 | -0.3058 | 0.3749 | 0.4606 | 62.00% |

## Optimistic |anchor-margin| gate simulation

Use alpha decision only when `|anchor margin| <= threshold`, otherwise keep anchor decision.

| alpha | best threshold by cheap7 | cheap7 delta | relation/state delta | protected threshold if any | protected cheap7 delta | protected relation/state delta |
|---|---:|---:|---:|---:|---:|---:|
| a0p5 | 0.3500 | +0.2295 | +0.0337 | NA | NA | NA |
| a0p75 | 0.2000 | +0.1278 | -0.0163 | NA | NA | NA |
| a1 | 0.1000 | +0.0714 | +0.0706 | 1.0000 | +0.0464 | +0.0656 |

## Scientific reading

- The census contains 6359 alpha-sensitive items. Pooled separability is only interpretable with caution because COMPS/BLiMP dominate the pool.
- Pooled AUC for identifying damage by low |anchor margin| is 0.5113 (0.5 is no separation); by high |margin| it is 0.4887.
- For a0p5, the best optimistic |margin|-gate by cheap7 delta is threshold 0.35 with cheap7 delta +0.2295 points and relation/state mean +0.0337.
- For a0p5, no tested |margin| threshold simultaneously gives positive cheap7 and nonnegative EWoK plus Entity item movement.
- For a0p75, the best optimistic |margin|-gate by cheap7 delta is threshold 0.2 with cheap7 delta +0.1278 points and relation/state mean -0.0163.
- For a0p75, no tested |margin| threshold simultaneously gives positive cheap7 and nonnegative EWoK plus Entity item movement.
- For a1, the best optimistic |margin|-gate by cheap7 delta is threshold 0.1 with cheap7 delta +0.0714 points and relation/state mean +0.0706.
- For a1, a threshold satisfying positive cheap7 and nonnegative EWoK+Entity exists at 1.0 with cheap7 delta +0.0464 and relation/state mean +0.0656.

Gate simulation is an optimistic decision-level diagnostic using official labels after the fact only for analysis; it is not a runnable model or a training result. Rows where the lightweight margin rescore disagrees with saved official anchor decisions are excluded from margin analysis and listed explicitly.

JSON: `experiments/archive/frontier_consolidation/data/anchor_margin_gate_analysis_full/anchor_margin_gate_analysis.json`
