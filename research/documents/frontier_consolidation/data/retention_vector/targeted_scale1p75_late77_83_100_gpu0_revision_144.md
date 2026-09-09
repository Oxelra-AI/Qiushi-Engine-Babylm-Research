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
- prepared_targets_after_tokenization: `12318`
- skipped: `{'target_truncated_or_missing': 237}`

## Checkpoint summary
| checkpoint | cheap7 | overall piece NLL | macro source×structure NLL | q90 NLL | forgetting mean | vector retention score |
|---|---:|---:|---:|---:|---:|---:|
| chck_77M | 43.28214285714286 | 2.317297 | 2.146978 | 3.938094 | 0.000000 | 2.492826 |
| chck_78M | 43.70214285714286 | 2.323708 | 2.152131 | 3.945196 | 0.018503 | 2.518673 |
| chck_79M | 43.57857142857143 | 2.297363 | 2.121806 | 3.937888 | 0.010966 | 2.480269 |
| chck_80M | 43.81214285714286 | 2.295245 | 2.124247 | 3.945711 | 0.019924 | 2.492039 |
| chck_81M | 43.64928571428572 | 2.301097 | 2.121822 | 3.909643 | 0.023800 | 2.494902 |
| chck_82M | 43.95944987645173 | 2.291534 | 2.115244 | 3.891761 | 0.019727 | 2.482652 |
| chck_83M | 43.80785714285714 | 2.284628 | 2.107550 | 3.885649 | 0.019997 | 2.473846 |
| chck_100M | 43.543159919261925 | 2.262614 | 2.087800 | 3.849275 | 0.010030 | 2.443430 |

## Label-free selectors (lower is better)
- `overall_piece_nll` selects **chck_100M**
- `macro_source_structure_nll` selects **chck_100M**
- `source_structure_q90_nll` selects **chck_100M**
- `forgetting_mean_vs_past_best` selects **chck_77M**
- `retention_score_mean_plus_forget_plus_0p25std` selects **chck_100M**

## Correlation with official cheap7 (post-hoc reading, not used to build the probe)
Correlations are Pearson r between official cheap7 and negative NLL/score over checkpoints with cheap7 available.
- `overall_piece_nll`: 0.2304445636180343
- `macro_source_structure_nll`: 0.2682065489360738
- `source_structure_q90_nll`: 0.15926279110245847
- `forgetting_mean_vs_past_best`: -0.8424491296119421
- `retention_score_mean_plus_forget_plus_0p25std`: -0.07489466294756265

### Source×structure strata most positively correlated with cheap7
- qwen_pair_packed::number: r=0.6815
- bnc_spoken::preposition_relation: r=0.6427
- switchboard::conjunction_marker: r=0.6314
- childes::pronoun_determiner: r=0.5645
- cleanqwen_fineweb_compact_view_reinvest::pronoun_determiner: r=0.5417
- gutenberg::aux_modal: r=0.5315
- bnc_spoken::pronoun_determiner: r=0.5301
- bnc_spoken::negation: r=0.5214

### Source×structure strata most negatively correlated with cheap7
- simple_wiki::aux_modal: r=-0.7887
- gutenberg::conjunction_marker: r=-0.5787
- switchboard::rare_content: r=-0.4903
- bnc_spoken::rare_content: r=-0.4600
- childes::rare_content: r=-0.4032
- open_subtitles::pronoun_determiner: r=-0.3972
- open_subtitles::rare_content: r=-0.3721
- qwen_pair_packed::rare_content: r=-0.3443

## Scientific interpretation
The fixed label-free retention summaries do not select the reproduced 82M peak in this first in-trajectory test. The 82M endpoint remains score-bearing, but this probe does not yet provide a general consolidation principle.

JSON: `experiments/archive/frontier_consolidation/data/retention_vector/targeted_scale1p75_late77_83_100_gpu0_revision_144.json`
