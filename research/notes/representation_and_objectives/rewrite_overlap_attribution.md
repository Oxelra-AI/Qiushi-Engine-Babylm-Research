# generated view overlap scan rewrite-overlap attribution

This CPU-only follow-up attributes exact score-text spans found in generated rewrites to the same pair's original/source text.

- Raw detail rows from generated view overlap scan: 131
- Unique generated-rewrite pair/ngram records: 66
- Attribution summary: `{"copied_or_preserved_from_pair_original": {"component_kinds": {"official_source_qwen_paraphrase_rewrite": 45}, "eval_hit_rows": 84, "n": {"10": 1, "7": 34, "8": 10}, "unique_pair_ngrams": 45}, "rewrite_only_within_pair": {"component_kinds": {"official_source_qwen_paraphrase_rewrite": 21}, "eval_hit_rows": 47, "n": {"10": 1, "7": 15, "8": 5}, "unique_pair_ngrams": 21}}`

## Interpretation
The compact FineWeb Qwen rewrites had no exact 7/8/10-token scored-text spans in generated view overlap scan. The inherited clean-Qwen paraphrase rewrites have sparse spans; this file separates spans already present in the paired original from rewrite-only spans introduced or preserved by the teacher rewrite process. This is a defensibility/provenance record, not a model-score explanation.

JSON: `experiments/archive/representation_and_objectives/data/generated_view_overlap_scan/rewrite_overlap_attribution.json`
CSV: `experiments/archive/representation_and_objectives/data/generated_view_overlap_scan/rewrite_overlap_attribution_pair_ngrams.csv`
