# generated view overlap scan generated-view scored-text overlap scan

This CPU-only scan ran while the two legal40k official evaluations remained managed asynchronously. It did not read task status, train a model, or evaluate a model.

## Official coordinate and files
- Strict score text came from `experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval` plus changed block overlap ancestry current official GlobalPIQA files at `experiments/archive/representation_and_objectives/data/globalpiqa_official_lineage/official_dl_scratch/generated_by_current_official_dl/evaluation_data/full_eval`.
- Active 10M corpus: `experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl`; SHA match: `True`.
- Compact FineWeb pairs: `experiments/archive/frontier_consolidation/data/density_core_reinvestment_medium_riskhard/selected_compact_reinvest_pairs.jsonl`; SHA match: `True`.
- Clean-Qwen official-source pairs: `experiments/archive/compact_experience/data/qwen_clean_aligned/selected_pairs.jsonl`; SHA match: `True`.
- Strict eval text records: 257821; unique score-text ngrams: {'7': 877403, '8': 799686, '10': 681707}.

## Generated rewrite components
- `fineweb_compact_qwen_rewrite`: components=12155, words=161708, n7 components with score-text span=0, n8=0, n10=0, unique n7 spans=0
- `official_source_qwen_paraphrase_rewrite`: components=37594, words=807787, n7 components with score-text span=34, n8=8, n10=2, unique n7 spans=44

## Original/source components
- `fineweb_compact_original_source`: components=12155, words=261803, n7 components with score-text span=7, n8=1, n10=0, unique n7 spans=9
- `official_source_for_qwen_paraphrase`: components=37594, words=849013, n7 components with score-text span=53, n8=15, n10=3, unique n7 spans=73

## Row-level source summary
- `fineweb_source_qwen_compact_rewrite_pair_row`: rows=3005, words=423511, rows with n7/n8/n10 exact score spans=7/1/0, unique n7 spans=9
- `inherited_official_source_qwen_paraphrase_pair_row`: rows=12236, words=1656800, rows with n7/n8/n10 exact score spans=61/16/4, unique n7 spans=87
- `neutral_topup_from_heldout_official_row`: rows=1, words=9, rows with n7/n8/n10 exact score spans=0/0/0, unique n7 spans=0
- `official_babylm_source_row`: rows=49498, words=7919680, rows with n7/n8/n10 exact score spans=388/91/24, unique n7 spans=442

## Interpretation
Exact overlap is recorded to protect future endpoint interpretation. It does not estimate why a model scores well. The important separation here is between allowed original/source text and teacher-generated rewrites; if generated-rewrite exact spans are sparse and generic, a future high score is less likely to be explained by direct teacher production of score strings.

JSON: `experiments/archive/representation_and_objectives/data/generated_view_overlap_scan/generated_view_overlap_scan.json`
Component CSV: `experiments/archive/representation_and_objectives/data/generated_view_overlap_scan/component_kind_overlap_summary.csv`
Row CSV: `experiments/archive/representation_and_objectives/data/generated_view_overlap_scan/row_source_overlap_summary.csv`
Ngram CSV: `experiments/archive/representation_and_objectives/data/generated_view_overlap_scan/matched_ngram_summary.csv`
Generated rewrite detail CSV: `experiments/archive/representation_and_objectives/data/generated_view_overlap_scan/generated_rewrite_overlap_details.csv`
