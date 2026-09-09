# preservation admission standard GlobalPIQA item-level comparison

## Split counts

| endpoint             | split       |   correct |   total |   accuracy |   delta_correct_vs_coherent86 |
|:---------------------|:------------|----------:|--------:|-----------:|------------------------------:|
| clean_pres_seed62064 | nonparallel |        50 |     100 |    50      |                             2 |
| clean_pres_seed62064 | parallel    |        31 |     103 |    30.0971 |                             1 |
| clean_pres_seed62065 | nonparallel |        50 |     100 |    50      |                             2 |
| clean_pres_seed62065 | parallel    |        31 |     103 |    30.0971 |                             1 |
| coherent86           | nonparallel |        48 |     100 |    48      |                             0 |
| coherent86           | parallel    |        30 |     103 |    29.1262 |                             0 |
| dense_seed62064      | nonparallel |        50 |     100 |    50      |                             2 |
| dense_seed62064      | parallel    |        31 |     103 |    30.0971 |                             1 |
| dense_seed62065      | nonparallel |        50 |     100 |    50      |                             2 |
| dense_seed62065      | parallel    |        31 |     103 |    30.0971 |                             1 |

## BabyLM GlobalPIQA mean

| endpoint             |   parallel_correct |   parallel_total |   parallel_acc |   parallel_delta |   nonparallel_correct |   nonparallel_total |   nonparallel_acc |   nonparallel_delta |   globalpiqa_mean |
|:---------------------|-------------------:|-----------------:|---------------:|-----------------:|----------------------:|--------------------:|------------------:|--------------------:|------------------:|
| clean_pres_seed62064 |                 31 |              103 |        30.0971 |                1 |                    50 |                 100 |                50 |                   2 |           40.0485 |
| clean_pres_seed62065 |                 31 |              103 |        30.0971 |                1 |                    50 |                 100 |                50 |                   2 |           40.0485 |
| coherent86           |                 30 |              103 |        29.1262 |                0 |                    48 |                 100 |                48 |                   0 |           38.5631 |
| dense_seed62064      |                 31 |              103 |        30.0971 |                1 |                    50 |                 100 |                50 |                   2 |           40.0485 |
| dense_seed62065      |                 31 |              103 |        30.0971 |                1 |                    50 |                 100 |                50 |                   2 |           40.0485 |

Flip rows vs coherent86: 20
Candidate changed item rows: 5

Outputs:
{
  "item_csv": "experiments/archive/relation_learning/data/globalpiqa_item_table/globalpiqa_item_correctness.csv",
  "flips_csv": "experiments/archive/relation_learning/data/globalpiqa_item_table/globalpiqa_flips_vs_coherent86.csv",
  "shared_csv": "experiments/archive/relation_learning/data/globalpiqa_item_table/globalpiqa_candidate_changed_items.csv",
  "counts_csv": "experiments/archive/relation_learning/data/globalpiqa_item_table/globalpiqa_counts.csv",
  "overall_csv": "experiments/archive/relation_learning/data/globalpiqa_item_table/globalpiqa_overall.csv"
}
