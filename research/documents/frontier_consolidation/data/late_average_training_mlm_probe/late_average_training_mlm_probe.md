# stabilization precheck while grids pending late average legal-corpus MLM probe

This CPU probe uses deterministic masked-token likelihood on sampled legal-corpus rows. It is not an official BabyLM score, not public endpoint selection, and performs no upload or leaderboard submission.

Sample: `202` rows from `experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl`, per-source request `24`, masked tokens `6268`, mean sequence length `208.35`.

## Mean masked-token NLL

| model | mean NLL | masked tokens | rows |
|---|---:|---:|---:|
| chck_100M | 2.848895 | 6268 | 202 |
| chck_86M | 2.860995 | 6268 | 202 |
| avg_80_82_84_uniform | 2.868350 | 6268 | 202 |
| chck_84M | 2.873649 | 6268 | 202 |
| chck_82M | 2.878524 | 6268 | 202 |
| chck_80M | 2.882761 | 6268 | 202 |

## Comparisons involving the built 80/82/84 average

- average minus chck_84M mean NLL: `-0.005299`
- average minus arithmetic mean of endpoint NLLs (80/82/84): `-0.009962`
- row-level avg lower NLL than chck84 fraction: `0.5693` over `202` rows

## Per-source mean NLL

| source | chck_80M | chck_82M | chck_84M | chck_86M | chck_100M | avg_80_82_84_uniform |
|---|---:|---:|---:|---:|---:|---:|
| bnc_spoken | 3.5094 | 3.5294 | 3.5086 | 3.5025 | 3.4841 | 3.5062 |
| childes | 1.6976 | 1.6949 | 1.6864 | 1.6833 | 1.6662 | 1.6854 |
| cleanqwen_fineweb_compact_view_reinvest | 2.5811 | 2.5812 | 2.5724 | 2.5636 | 2.5408 | 2.5632 |
| gutenberg | 3.6947 | 3.7047 | 3.6950 | 3.6939 | 3.6732 | 3.6867 |
| neutral_cleanqwen_topup_compact_reinvest::open_subtitles | 7.8856 | 8.1180 | 7.9356 | 8.0264 | 7.9956 | 7.9764 |
| open_subtitles | 3.1516 | 3.1501 | 3.1579 | 3.1225 | 3.1186 | 3.1439 |
| qwen_pair_packed | 2.1994 | 2.1588 | 2.1578 | 2.1479 | 2.1461 | 2.1636 |
| simple_wiki | 3.4435 | 3.4170 | 3.4232 | 3.3886 | 3.3874 | 3.4180 |
| switchboard | 2.6770 | 2.6760 | 2.6740 | 2.6724 | 2.6616 | 2.6666 |

## Scientific reading

Legal-corpus masked-token NLL only checks local functional smoothness/collapse risk. Earlier evidence shows MLM likelihood does not select BabyLM competence, so selected task scoring remains necessary for any endpoint decision.

JSON: `experiments/archive/frontier_consolidation/data/late_average_training_mlm_probe/late_average_training_mlm_probe.json`
