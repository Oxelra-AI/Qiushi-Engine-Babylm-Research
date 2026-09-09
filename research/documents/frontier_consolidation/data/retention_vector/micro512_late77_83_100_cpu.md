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
- prepared_targets_after_tokenization: `512`
- skipped: `{}`

## Checkpoint summary
| checkpoint | cheap7 | overall piece NLL | macro source×structure NLL | q90 NLL | forgetting mean | vector retention score |
|---|---:|---:|---:|---:|---:|---:|
| chck_77M | 43.28214285714286 | 2.889275 | 2.682031 | 1.660807 | 0.000000 | 2.937337 |
| chck_78M | 43.70214285714286 | 2.881779 | 2.663847 | 1.599940 | 0.012250 | 2.942074 |
| chck_79M | 43.57857142857143 | 2.867288 | 2.651168 | 1.581185 | 0.008948 | 2.927612 |
| chck_80M | 43.81214285714286 | 2.864177 | 2.651064 | 1.575505 | 0.011684 | 2.931638 |
| chck_81M | 43.64928571428572 | 2.851043 | 2.635309 | 1.583094 | 0.003794 | 2.902157 |
| chck_82M | 43.95944987645173 | 2.852139 | 2.641274 | 1.587764 | 0.009759 | 2.914410 |
| chck_83M | 43.80785714285714 | 2.845260 | 2.645007 | 1.606297 | 0.015396 | 2.920080 |
| chck_100M | 43.543159919261925 | 2.821310 | 2.611052 | 1.564981 | 0.000000 | 2.872570 |

## Label-free selectors (lower is better)
- `overall_piece_nll` selects **chck_100M**
- `macro_source_structure_nll` selects **chck_100M**
- `source_structure_q90_nll` selects **chck_100M**
- `forgetting_mean_vs_past_best` selects **chck_77M**
- `retention_score_mean_plus_forget_plus_0p25std` selects **chck_100M**

## Correlation with official cheap7 (post-hoc reading, not used to build the probe)
Correlations are Pearson r between official cheap7 and negative NLL/score over checkpoints with cheap7 available.
- `overall_piece_nll`: 0.31952650684062844
- `macro_source_structure_nll`: 0.33212937377599017
- `source_structure_q90_nll`: 0.5445120088562337
- `forgetting_mean_vs_past_best`: -0.7505962646567405
- `retention_score_mean_plus_forget_plus_0p25std`: 0.009675023499277437

### Source×structure strata most positively correlated with cheap7
- bnc_spoken::function: r=0.5445
- bnc_spoken::content: r=-0.0964

### Source×structure strata most negatively correlated with cheap7
- bnc_spoken::content: r=-0.0964
- bnc_spoken::function: r=0.5445

## Scientific interpretation
The fixed label-free retention summaries do not select the reproduced 82M peak in this first in-trajectory test. The 82M endpoint remains score-bearing, but this probe does not yet provide a general consolidation principle.

JSON: `experiments/archive/frontier_consolidation/data/retention_vector/micro512_late77_83_100_cpu.json`
