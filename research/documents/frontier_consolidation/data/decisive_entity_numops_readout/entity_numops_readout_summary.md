# earlier analysis decisive Entity zero/nonzero-operation readout

File-only recomputation from official Entity prediction files. This separates Entity from the broad ex-Entity interpretation for the register and in-corpus contrasts.

## Available Entity prediction records

| arm | checkpoint | exists | payload score | predictions |
|---|---:|---:|---:|---|
| clean_maxgeom | chck_70M | True | None | `experiments/archive/frontier_consolidation/data/deberta_maxgeom_clean_stable_eval/eval/official_outputs/deberta_maxgeom_clean_seed43022_chck_70M/Entity/chck_70M/full_deberta_maxgeom_clean_seed43022_chck_70M_Entity/zero_shot/mlm/entity_tracking/entity_tracking/predictions.json` |
| clean_maxgeom | chck_80M | True | None | `experiments/archive/frontier_consolidation/data/deberta_maxgeom_clean_stable_eval/eval/official_outputs/deberta_maxgeom_clean_seed43022_chck_80M/Entity/chck_80M/full_deberta_maxgeom_clean_seed43022_chck_80M_Entity/zero_shot/mlm/entity_tracking/entity_tracking/predictions.json` |
| clean_maxgeom | chck_100M | True | None | `experiments/archive/frontier_consolidation/data/deberta_maxgeom_clean_stable_eval/eval/official_outputs/deberta_maxgeom_clean_seed43022_chck_100M/Entity/chck_100M/full_deberta_maxgeom_clean_seed43022_chck_100M_Entity/zero_shot/mlm/entity_tracking/entity_tracking/predictions.json` |

## Balanced contrast rows

| contrast | checkpoint | official all18 Δ pp | zero-op Δ pp | nonzero Δ pp | balanced zero/nonzero Δ pp | nonzero-zero spread pp |
|---|---:|---:|---:|---:|---:|---:|

## Scientific reading rule

- A positive Entity aggregate with nonzero-operation gains and zero-operation losses should be treated as operation-propensity allocation, not general record formation.
- The distribution principle should be evaluated mainly on ex-Entity families; Entity can be reported as a separate, benchmark-specific movement unless zero/nonzero balance improves together.
- This script does not recover continuous margins; existing prediction files store choices, not candidate log probabilities.

## Files

- summary_json: `experiments/archive/frontier_consolidation/data/decisive_entity_numops_readout/entity_numops_readout_summary.json`
- summary_md: `research/documents/frontier_consolidation/data/decisive_entity_numops_readout/entity_numops_readout_summary.md`
- subtask_csv: `experiments/archive/frontier_consolidation/data/decisive_entity_numops_readout/entity_numops_subtask_scores.csv`
- aggregate_csv: `experiments/archive/frontier_consolidation/data/decisive_entity_numops_readout/entity_numops_aggregate_scores.csv`
- contrast_csv: `experiments/archive/frontier_consolidation/data/decisive_entity_numops_readout/entity_numops_contrasts.csv`
- balanced_csv: `experiments/archive/frontier_consolidation/data/decisive_entity_numops_readout/entity_numops_balanced_contrasts.csv`
- late_summary_csv: `experiments/archive/frontier_consolidation/data/decisive_entity_numops_readout/entity_numops_late_summary.csv`
