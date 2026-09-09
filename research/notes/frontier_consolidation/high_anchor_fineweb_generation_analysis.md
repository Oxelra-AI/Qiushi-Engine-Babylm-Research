# fineweb factor contrast bestview probe high-anchor FineWeb generation analysis

Prompts: `experiments/archive/frontier_consolidation/data/fineweb_high_anchor_slice/fineweb_high_anchor_simplification_prompts_pilot1024.jsonl`

Outputs: `experiments/archive/frontier_consolidation/training/runs/high_anchor_fineweb_qwen_pilot1024_qbatch/outputs.jsonl`

Rows: 1,024; accepted: 952; accepted rate: 0.930.

Source words: 26,563; accepted source words: 24,621; accepted source+rewrite words: 48,655.

Top hard reason prefixes: [('entity_recall', 44), ('number_recall', 19), ('new_numbers', 6), ('many_sentences', 4), ('bad_output_pattern', 2), ('length_ratio', 2), ('explanatory_expansion', 2)]

Top soft flags: [('near_copy_view', 437), ('not_shorter_than_source', 179), ('source_risk_long_parenthetical', 40), ('source_risk_apostle_or_title_apposition', 32), ('source_risk_deictic_time', 20), ('source_risk_heading_dash_chain', 14), ('source_risk_probability_hedge', 8), ('source_risk_angle_heading', 1)]

| domain | n | accepted | rate | accepted source words | accepted pair words |
|---|---:|---:|---:|---:|---:|
| no_domain | 381 | 347 | 0.911 | 8714 | 17150 |
| science_physical | 174 | 160 | 0.920 | 4404 | 8682 |
| causal_relational | 164 | 152 | 0.927 | 4087 | 8063 |
| quant_numeric | 161 | 155 | 0.963 | 4032 | 8006 |
| institutions_society | 152 | 146 | 0.961 | 3933 | 7810 |
| geography_places | 116 | 109 | 0.940 | 2918 | 5808 |
| people_history | 84 | 77 | 0.917 | 1909 | 3801 |
| media_culture | 60 | 59 | 0.983 | 1517 | 3001 |

JSON: `experiments/archive/frontier_consolidation/data/high_anchor_fineweb_generation_analysis/fineweb_high_anchor_generation_summary.json`

Accepted rewrite JSONL: `experiments/archive/frontier_consolidation/data/high_anchor_fineweb_generation_analysis/fineweb_high_anchor_accepted_rewrites.jsonl`

Review samples: `experiments/archive/frontier_consolidation/data/high_anchor_fineweb_generation_analysis/fineweb_high_anchor_generation_review_samples.json`
