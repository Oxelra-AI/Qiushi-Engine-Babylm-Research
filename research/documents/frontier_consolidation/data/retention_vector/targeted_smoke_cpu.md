# fast eval results carrier schema corpus-internal retention vector

Inference-only fixed masked-word probe drawn from the legal 10M corpus. No official labels or benchmark data are used to compute the retention vector.

## Probe summary
- targets_total: `12555`
- max_per_source_category: `160`
- seed: `1357`
- pool: `experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl`
- rare_cut_frequency_leq: `3`
- category_definitions: `{'negation': ['cannot', "n't", 'neither', 'never', 'no', 'nobody', 'nor', 'not', 'nothing', 'nowhere', 'without'], 'quantifier': ['all', 'any', 'both', 'each', 'eight', 'either', 'enough', 'every', 'few', 'fewer', 'five', 'four', 'half', 'less', 'many', 'more', 'most', 'much', 'multiple', 'neither', 'nine', 'none', 'one', 'seven', 'several', 'six', 'some', 'ten', 'three', 'two', 'various', 'whole'], 'wh': ['how', 'what', 'when', 'where', 'which', 'who', 'whom', 'whose', 'why'], 'aux_modal': ['am', 'are', 'be', 'been', 'being', 'can', 'could', 'dare', 'did', 'do', 'does', 'had', 'has', 'have', 'having', 'is', 'may', 'might', 'must', 'need', 'ought', 'shall', 'should', 'was', 'were', 'will', 'would'], 'preposition_relation': ['about', 'above', 'across', 'after', 'against', 'along', 'around', 'as', 'at', 'before', 'behind', 'below', 'beneath', 'beside', 'besides', 'between', 'beyond', 'by', 'down', 'during', 'for', 'from', 'in', 'inside', 'into', 'like', 'near', 'of', 'off', 'on', 'onto', 'out', 'outside', 'over', 'per', 'since', 'than', 'through', 'to', 'toward', 'towards', 'under', 'until', 'up', 'upon', 'via', 'with', 'within'], 'pronoun_determiner': ['a', 'an', 'another', 'he', 'her', 'hers', 'him', 'his', 'i', 'it', 'its', 'me', 'mine', 'my', 'other', 'our', 'ours', 'same', 'she', 'such', 'that', 'the', 'their', 'theirs', 'them', 'these', 'they', 'this', 'those', 'us', 'we', 'whose', 'you', 'your', 'yours'], 'conjunction_marker': ['although', 'and', 'because', 'but', 'either', 'if', 'nor', 'once', 'or', 'so', 'then', 'though', 'unless', 'whereas', 'whether', 'while', 'yet'], 'number': 'regex numeric', 'rare_content': 'non-function corpus word frequency <= 3', 'common_content': 'remaining content words frequency > 3'}`
- bucket_counts: `{'bnc_spoken::aux_modal': 160, 'bnc_spoken::common_content': 160, 'bnc_spoken::conjunction_marker': 160, 'bnc_spoken::negation': 160, 'bnc_spoken::number': 160, 'bnc_spoken::preposition_relation': 160, 'bnc_spoken::pronoun_determiner': 160, 'bnc_spoken::quantifier': 160, 'bnc_spoken::rare_content': 160, 'bnc_spoken::wh': 160, 'childes::aux_modal': 160, 'childes::common_content': 160, 'childes::conjunction_marker': 160, 'childes::negation': 160, 'childes::number': 160, 'childes::preposition_relation': 160, 'childes::pronoun_determiner': 160, 'childes::quantifier': 160, 'childes::rare_content': 160, 'childes::wh': 160, 'cleanqwen_fineweb_compact_view_reinvest::aux_modal': 160, 'cleanqwen_fineweb_compact_view_reinvest::common_content': 160, 'cleanqwen_fineweb_compact_view_reinvest::conjunction_marker': 160, 'cleanqwen_fineweb_compact_view_reinvest::negation': 160, 'cleanqwen_fineweb_compact_view_reinvest::number': 160, 'cleanqwen_fineweb_compact_view_reinvest::preposition_relation': 160, 'cleanqwen_fineweb_compact_view_reinvest::pronoun_determiner': 160, 'cleanqwen_fineweb_compact_view_reinvest::quantifier': 160, 'cleanqwen_fineweb_compact_view_reinvest::rare_content': 160, 'cleanqwen_fineweb_compact_view_reinvest::wh': 160, 'gutenberg::aux_modal': 160, 'gutenberg::common_content': 160, 'gutenberg::conjunction_marker': 160, 'gutenberg::negation': 160, 'gutenberg::number': 160, 'gutenberg::preposition_relation': 160, 'gutenberg::pronoun_determiner': 160, 'gutenberg::quantifier': 160, 'gutenberg::rare_content': 160, 'gutenberg::wh': 160, 'open_subtitles::aux_modal': 160, 'open_subtitles::common_content': 160, 'open_subtitles::conjunction_marker': 160, 'open_subtitles::negation': 160, 'open_subtitles::number': 160, 'open_subtitles::preposition_relation': 160, 'open_subtitles::pronoun_determiner': 160, 'open_subtitles::quantifier': 160, 'open_subtitles::rare_content': 160, 'open_subtitles::wh': 160, 'qwen_pair_packed::aux_modal': 160, 'qwen_pair_packed::common_content': 160, 'qwen_pair_packed::conjunction_marker': 160, 'qwen_pair_packed::negation': 160, 'qwen_pair_packed::number': 160, 'qwen_pair_packed::preposition_relation': 160, 'qwen_pair_packed::pronoun_determiner': 160, 'qwen_pair_packed::quantifier': 160, 'qwen_pair_packed::rare_content': 160, 'qwen_pair_packed::wh': 160, 'simple_wiki::aux_modal': 160, 'simple_wiki::common_content': 160, 'simple_wiki::conjunction_marker': 160, 'simple_wiki::negation': 160, 'simple_wiki::number': 160, 'simple_wiki::preposition_relation': 160, 'simple_wiki::pronoun_determiner': 160, 'simple_wiki::quantifier': 160, 'simple_wiki::rare_content': 160, 'simple_wiki::wh': 160, 'switchboard::aux_modal': 160, 'switchboard::common_content': 160, 'switchboard::conjunction_marker': 160, 'switchboard::negation': 160, 'switchboard::preposition_relation': 160, 'switchboard::pronoun_determiner': 160, 'switchboard::quantifier': 160, 'switchboard::rare_content': 75, 'switchboard::wh': 160}`
- source_counts: `{'bnc_spoken': 1600, 'childes': 1600, 'cleanqwen_fineweb_compact_view_reinvest': 1600, 'gutenberg': 1600, 'open_subtitles': 1600, 'qwen_pair_packed': 1600, 'simple_wiki': 1600, 'switchboard': 1355}`
- structure_counts: `{'aux_modal': 1280, 'common_content': 1280, 'conjunction_marker': 1280, 'negation': 1280, 'number': 1120, 'preposition_relation': 1280, 'pronoun_determiner': 1280, 'quantifier': 1280, 'rare_content': 1195, 'wh': 1280}`
- prepared_targets_after_tokenization: `64`
- skipped: `{}`

## Checkpoint summary
| checkpoint | cheap7 | overall piece NLL | macro source×structure NLL | q90 NLL | forgetting mean | vector retention score |
|---|---:|---:|---:|---:|---:|---:|
| chck_80M | 43.81214285714286 | 1.787064 | 1.773360 | 1.773360 | 0.000000 | 1.773360 |
| chck_82M | 43.95944987645173 | 1.783655 | 1.766814 | 1.766814 | 0.000000 | 1.766814 |

## Label-free selectors (lower is better)
- `overall_piece_nll` selects **chck_82M**
- `macro_source_structure_nll` selects **chck_82M**
- `source_structure_q90_nll` selects **chck_82M**
- `forgetting_mean_vs_past_best` selects **chck_80M**
- `retention_score_mean_plus_forget_plus_0p25std` selects **chck_82M**

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
At least one fixed label-free retention summary selects the reproduced 82M peak. This is only an in-trajectory signal until tested on another seed or route.

JSON: `experiments/archive/frontier_consolidation/data/retention_vector/targeted_smoke_cpu.json`
