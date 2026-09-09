# private scale endpoint vs mechanism synthesis anchor-margin gate analysis

Status: **COMPLETE**
Scored items: `12` from `experiments/archive/frontier_consolidation/data/anchor_margin_alpha_census_cpu_smoke/anchor_margin_scored_items.jsonl`
Anchor rescore agreement: `{'agree': 12, 'n': 12}`

## Per-column separability

| column | n changed | act n | dmg n | act median | dmg median | AUC damage by low |margin| | damage |margin|<0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ALL | 12 | 3 | 3 | -0.4247 | 0.4410 | 0.4444 | 66.67% |
| BLiMP | 10 | 3 | 3 | -0.4247 | 0.4410 | 0.4444 | 66.67% |
| COMPS | 2 | 0 | 0 | NA | NA | NA | NA |

## Optimistic |anchor-margin| gate simulation

Use alpha decision only when `|anchor margin| <= threshold`, otherwise keep anchor decision.

| alpha | best threshold by cheap7 | cheap7 delta | relation/state delta | protected threshold if any | protected cheap7 delta | protected relation/state delta |
|---|---:|---:|---:|---:|---:|---:|
| a0p5 | 0.0500 | +0.0004 | +0.0000 | 0.0500 | +0.0004 | +0.0000 |
| a0p75 | 0.5000 | +0.0004 | +0.0000 | 0.5000 | +0.0004 | +0.0000 |
| a1 | 0.0500 | +0.0003 | +0.0000 | 0.0500 | +0.0003 | +0.0000 |

## Scientific reading

- The census contains 12 alpha-sensitive items. Pooled separability is only interpretable with caution because COMPS/BLiMP dominate the pool.
- Pooled AUC for identifying damage by low |anchor margin| is 0.4444 (0.5 is no separation); by high |margin| it is 0.5556.
- For a0p5, the best optimistic |margin|-gate by cheap7 delta is threshold 0.05 with cheap7 delta +0.0004 points and relation/state mean +0.0000.
- For a0p5, a threshold satisfying positive cheap7 and nonnegative EWoK+Entity exists at 0.05 with cheap7 delta +0.0004 and relation/state mean +0.0000.
- For a0p75, the best optimistic |margin|-gate by cheap7 delta is threshold 0.5 with cheap7 delta +0.0004 points and relation/state mean +0.0000.
- For a0p75, a threshold satisfying positive cheap7 and nonnegative EWoK+Entity exists at 0.5 with cheap7 delta +0.0004 and relation/state mean +0.0000.
- For a1, the best optimistic |margin|-gate by cheap7 delta is threshold 0.05 with cheap7 delta +0.0003 points and relation/state mean +0.0000.
- For a1, a threshold satisfying positive cheap7 and nonnegative EWoK+Entity exists at 0.05 with cheap7 delta +0.0003 and relation/state mean +0.0000.

Gate simulation is an optimistic decision-level diagnostic using official labels after the fact only for analysis; it is not a runnable model or a training result.

JSON: `experiments/archive/frontier_consolidation/data/anchor_margin_gate_analysis_cpu_smoke/anchor_margin_gate_analysis.json`
