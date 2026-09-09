# s3 10m available coordinate — S3 10M full EWoK with official word_tokenize

Evidence JSON: `experiments/archive/initial_model_studies/data/s3_10m_full_ewok_word_tokenize_score.json`
Filter summary: `experiments/archive/initial_model_studies/data/s3_10m_ewok_word_tokenize_filter_summary.json`
Report: `experiments/archive/initial_model_studies/training/runs/babylm_s3_12x384_official40k_flatwwm_10M/eval_results_full_ewok_word_tokenize_direct/chck_10M/main/zero_shot/mlm/ewok/ewok_filtered_word_tokenize/best_temperature_report.txt`

Full EWoK score: **49.55**
Filtered items: 3809; swapped JSONL lines: 7618

This uses the local EWoK parquet, official BabyLM vocab filter, and NLTK `word_tokenize` with the local `data/nltk_data` resources.
