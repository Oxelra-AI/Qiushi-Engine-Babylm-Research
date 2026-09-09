# execution synthesis — mature (100M) FW GlobalPIQA margin synthesis

Reads the fw ewok interaction reader wrapper 100M readout for the two completed arms.

## 100M GlobalPIQA split and hard-row ranks

| arm | parallel | nonparallel | GlobalPIQA agg | par rank1 | par rank4 | hard52 acc | hard52 rank1 | hard52 mean margin (nats) |
|---|---|---|---|---|---|---|---|---|
| fw_compact_fullbatch_seed43022 | 24.27 | 53.00 | 38.64 | 25 | 24 | 3.85 | 2 | 1.717 |
| fw_breadth_rowblock_fullbatch_seed43022 | 29.13 | 45.00 | 37.06 | 30 | 17 | 5.77 | 3 | 1.433 |

## Row-block breadth minus compact (100M)

- breadth_minus_compact_parallel_acc: +4.8544
- breadth_minus_compact_nonparallel_acc: -8.0000
- breadth_minus_compact_globalpiqa_agg: -1.5728
- breadth_minus_compact_parallel_rank1: +5.0000
- breadth_minus_compact_hard52_acc: +1.9231
- breadth_minus_compact_hard52_rank1: +1.0000
- breadth_minus_compact_hard52_mean_top_minus_correct: -0.2836
- breadth_minus_compact_hard52_small_le_0p50: +4.0000

## Within-arm 70M→100M movement

- compact: parallel -1.94, nonparallel +2.00, GlobalPIQA agg +0.03, hard52 mean margin +0.014
- breadth_rowblock: parallel +2.91, nonparallel -2.00, GlobalPIQA agg +0.46, hard52 mean margin -0.043

## Interpretation

- Mature 100M GlobalPIQA is a within-checkpoint tradeoff, not a joint improvement.
- Row-block breadth raises hard parallel accuracy and softens hard52 margins but lowers nonparallel accuracy by a larger margin, so its GlobalPIQA aggregate is lower than compact.
- Compact preserves broad GlobalPIQA (higher nonparallel) but keeps the deep-rank parallel weakness.
- This reproduces the 70M direction at the final checkpoint: independent coverage shifts probability mass toward hard parallel structure while degrading broad capability; the proposed 'combination' of preserved breadth plus improved relational ranks does not appear.

JSON: `experiments/archive/representation_and_objectives/data/fw_100m_globalpiqa_mature_synthesis/fw_100m_globalpiqa_mature_synthesis.json`
