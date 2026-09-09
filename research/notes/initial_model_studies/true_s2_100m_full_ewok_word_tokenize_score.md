# true s2 100m available coordinate — true S2 100M full EWoK with official word_tokenize

Evidence JSON: `experiments/archive/initial_model_studies/data/true_s2_100m_full_ewok_word_tokenize_score.json`
Filter summary: `experiments/archive/initial_model_studies/data/true_s2_100m_ewok_word_tokenize_filter_summary.json`
Report: `experiments/archive/initial_model_studies/training/runs/babylm_true_s2_100M_wordclock_wwm70_len64256/eval_results_full_ewok_word_tokenize_direct/chck_100M/main/zero_shot/mlm/ewok/ewok_filtered_word_tokenize/best_temperature_report.txt`

Full EWoK score: **51.64**
Filtered items: 3809; swapped JSONL lines: 7618

This uses the local EWoK parquet, official BabyLM vocab filter, and NLTK `word_tokenize` with the local `data/nltk_data` resources.
