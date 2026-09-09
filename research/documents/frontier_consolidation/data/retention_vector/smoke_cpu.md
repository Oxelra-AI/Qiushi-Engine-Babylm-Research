# fast eval results carrier schema corpus-internal retention vector

Inference-only fixed masked-word probe drawn from the legal 10M corpus. No official labels or benchmark data are used to compute the retention vector.

## Probe summary
- probe_targets_total: `9488`
- sampled_rows_per_source: `{'bnc_spoken': 320, 'childes': 320, 'cleanqwen_fineweb_compact_view_reinvest': 320, 'gutenberg': 320, 'open_subtitles': 320, 'qwen_pair_packed': 320, 'simple_wiki': 320, 'switchboard': 132}`
- function_word_targets: `4234`
- content_word_targets: `5254`
- seed: `135`
- rows_per_source_requested: `320`
- masks_per_row: `4`
- pool: `experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl`
- tokenizer: `experiments/archive/frontier_consolidation/data/compliant_tokenizer/tokenizer.json`
- prepared_targets_after_tokenization: `24`
- skipped: `{}`

## Checkpoint summary
| checkpoint | cheap7 | overall piece NLL | macro source×structure NLL | q90 NLL | forgetting mean | vector retention score |
|---|---:|---:|---:|---:|---:|---:|
| chck_77M | 43.28214285714286 | 2.901553 | 2.215633 | 1.173091 | 0.000000 | 2.476269 |
| chck_78M | 43.70214285714286 | 2.880813 | 2.180627 | 1.100043 | 0.001518 | 2.452291 |

## Label-free selectors (lower is better)
- `overall_piece_nll` selects **chck_78M**
- `macro_source_structure_nll` selects **chck_78M**
- `source_structure_q90_nll` selects **chck_78M**
- `forgetting_mean_vs_past_best` selects **chck_77M**
- `retention_score_mean_plus_forget_plus_0p25std` selects **chck_78M**

## Correlation with official cheap7 (post-hoc reading, not used to build the probe)
Correlations are Pearson r between official cheap7 and negative NLL/score over checkpoints with cheap7 available.
- `overall_piece_nll`: None
- `macro_source_structure_nll`: None
- `source_structure_q90_nll`: None
- `forgetting_mean_vs_past_best`: None
- `retention_score_mean_plus_forget_plus_0p25std`: None

### Source×structure strata most positively correlated with cheap7

### Source×structure strata most negatively correlated with cheap7

## Scientific interpretation
The fixed label-free retention summaries do not select the reproduced 82M peak in this first in-trajectory test. The 82M endpoint remains score-bearing, but this probe does not yet provide a general consolidation principle.

JSON: `experiments/archive/frontier_consolidation/data/retention_vector/smoke_cpu.json`
