# fineweb factor contrast bestview probe high-anchor FineWeb generation analysis

Prompts: `experiments/archive/frontier_consolidation/data/high_anchor_compression_prompt_variant/fineweb_high_anchor_compress_prompts_pilot1024.jsonl`

Outputs: `experiments/archive/frontier_consolidation/training/runs/high_anchor_fineweb_compress_qwen_pilot1024/outputs.jsonl`

Rows: 1,024; accepted: 767; accepted rate: 0.749.

Source words: 26,563; accepted source words: 19,815; accepted source+rewrite words: 32,519.

Top hard reason prefixes: [('entity_recall', 208), ('number_recall', 42), ('new_numbers', 22), ('low_content_recall', 19), ('many_sentences', 3), ('length_ratio', 3), ('explanatory_expansion', 2), ('bad_start', 1), ('bad_output_pattern', 1)]

Top soft flags: [('source_risk_long_parenthetical', 40), ('source_risk_apostle_or_title_apposition', 32), ('source_risk_deictic_time', 20), ('source_risk_heading_dash_chain', 14), ('near_copy_view', 12), ('source_risk_probability_hedge', 8), ('source_risk_angle_heading', 1)]

| domain | n | accepted | rate | accepted source words | accepted pair words |
|---|---:|---:|---:|---:|---:|
| no_domain | 381 | 285 | 0.748 | 7129 | 11712 |
| science_physical | 174 | 133 | 0.764 | 3669 | 6009 |
| causal_relational | 164 | 126 | 0.768 | 3431 | 5573 |
| quant_numeric | 161 | 115 | 0.714 | 3029 | 5055 |
| institutions_society | 152 | 109 | 0.717 | 2910 | 4765 |
| geography_places | 116 | 94 | 0.810 | 2533 | 4183 |
| people_history | 84 | 61 | 0.726 | 1495 | 2491 |
| media_culture | 60 | 48 | 0.800 | 1219 | 1993 |

JSON: `experiments/archive/frontier_consolidation/data/high_anchor_fineweb_compression_analysis/fineweb_high_anchor_generation_summary.json`

Accepted rewrite JSONL: `experiments/archive/frontier_consolidation/data/high_anchor_fineweb_compression_analysis/fineweb_high_anchor_accepted_rewrites.jsonl`

Review samples: `experiments/archive/frontier_consolidation/data/high_anchor_fineweb_compression_analysis/fineweb_high_anchor_generation_review_samples.json`
