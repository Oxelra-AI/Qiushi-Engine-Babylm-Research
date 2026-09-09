# debertav2 b256 full ewok word tokenize score — baseline16k DeBERTa-v2 full EWoK with official word_tokenize

Evidence JSON: `experiments/archive/initial_model_studies/data/debertav2_b256_full_ewok_word_tokenize_score.json`
Filter summary: `experiments/archive/initial_model_studies/data/local_ewok_word_tokenize_filter_summary.json`
Report: `experiments/archive/initial_model_studies/training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/eval_results_full_ewok_word_tokenize/chck_100M/main/zero_shot/mlm/ewok/ewok_filtered_word_tokenize/best_temperature_report.txt`

Full EWoK score: **52.19**
Fallback-score delta: 0.6299999999999955
Filtered items: 3809; swapped JSONL lines: 7618

This uses the local EWoK parquet, official BabyLM vocab filter, and NLTK `word_tokenize` with the local `data/nltk_data` resources.
