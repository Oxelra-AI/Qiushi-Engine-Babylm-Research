# compact reinvest refined eval overlap audit refined eval-overlap audit for compact_view_reinvest

This CPU-only audit separates score-bearing evaluation text from SuperGLUE fine-tuning train splits and focuses on the FineWeb compact changed block, not only the full 10M corpus.

## Main counts

- candidate_changed_block_only: rows 3005, words 423511, any-overlap rows 32, score-bearing rows 8, GLUE-valid rows 5, GLUE-train rows 25, unique 7-grams 56, occurrences 67.
- heldout_clean_rows_replaced_by_changed_block: rows 2647, words 423520, any-overlap rows 76, score-bearing rows 20, GLUE-valid rows 19, GLUE-train rows 66, unique 7-grams 159, occurrences 171.
- candidate_all: rows 64740, words 10000000, any-overlap rows 1451, score-bearing rows 447, GLUE-valid rows 397, GLUE-train rows 1208, unique 7-grams 1875, occurrences 3147.
- clean_qwen_all: rows 64381, words 10000000, any-overlap rows 1495, score-bearing rows 459, GLUE-valid rows 411, GLUE-train rows 1249, unique 7-grams 1912, occurrences 3251.

Changed block minus heldout official rows: {"scanned_rows": 358, "scanned_words": -9, "rows_with_any_overlap": -44, "rows_with_score_bearing_overlap": -12, "rows_with_glue_valid_overlap": -14, "rows_with_glue_train_overlap": -41, "unique_matching_ngrams": -103, "matching_ngram_occurrences": -104}

## Changed-block hit distribution

- By top dir: {"aoa": 4, "glue_filtered": 78, "vqa_filtered": 1}
- By file role: {"glue_train_for_finetuning": 71, "glue_valid_scored": 7, "zero_shot_or_human_scored": 5}

## Interpretation

The JSON stores representative score-bearing samples. This audit is a contamination safeguard, not proof of contamination by itself: exact common phrases and public benchmark fragments can overlap in both official and external web-derived text. The actionable comparison is the changed block against the heldout clean-Qwen official rows it replaced.

Machine-readable JSON: `experiments/archive/representation_and_objectives/data/reinvest_overlap_refined/compact_reinvest_refined_eval_overlap_audit.json`
