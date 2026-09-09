# bridge route recovered from sourcecopy error bridge structural-transformation analysis

Accepted pairs analyzed: 103

## Edit-class distribution (structural, not lemma overlap)

| class | n | meaning |
|---|---:|---|
| verbatim_substring | 14 | output is an exact contiguous source window |
| prefix_trim_only | 0 | output is a source suffix (leading words dropped) |
| deletion_reorder | 34 | source order preserved, long verbatim run, minimal novel words |
| light_restructure | 47 | some reordering / function-word edits |
| substantive_restructure | 8 | clause reorder + function-word/inflection change |

Extraction-like (verbatim + prefix_trim + deletion_reorder): **48/103 = 0.466**
Transformation-like (light + substantive restructure): **55/103 = 0.534**

## Structural distributions

| metric | mean | median | p10 | p90 |
|---|---:|---:|---:|---:|
| longest_run_frac | 0.5831 | 0.5385 | 0.2727 | 1.0 |
| order_agreement | 0.9615 | 1.0 | 0.8571 | 1.0 |
| reordering_index | 0.0385 | 0.0 | 0.0 | 0.1429 |
| novel_word_fraction | 0.0475 | 0.0 | 0.0 | 0.1538 |
| kept_source_word_fraction | 0.9525 | 1.0 | 0.8333 | 1.0 |
| function_word_edit_rate | 0.4364 | 0.5 | 0.0 | 0.7143 |

## By bucket

| bucket | n | mean_run_frac | mean_order_agree | mean_novel_frac |
|---|---:|---:|---:|---:|
| hard_absent_norelation | 10 | 0.5837 | 0.9687 | 0.0186 |
| hard_absent_relation | 37 | 0.6221 | 0.9757 | 0.0369 |
| ordinary | 25 | 0.4932 | 0.9351 | 0.0503 |
| relation_zero_absent | 31 | 0.6089 | 0.9636 | 0.0671 |
