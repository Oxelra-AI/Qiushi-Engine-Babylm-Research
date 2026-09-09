# compact provenance and overlap addendum — compact generator/source and overlap provenance addendum

This CPU-only addendum protects the seed43022 compact_view_reinvest endpoint while the seed43122 full evaluation continues; it does not use partial seed43122 outputs and does not launch new evaluation or training.

## Generator identity recovered from trace

Compact generation used `Qwen/Qwen3.5-9B` (model alias `qwen3.5-9b`), no adapter, `torch.bfloat16`, and an NVIDIA H100 (`CUDA_VISIBLE_DEVICES=1`, visible device `cuda:0`). The prompt input was `experiments/archive/frontier_consolidation/data/medium_density_prompts/fineweb_medium_compact_prompts_all.jsonl`; the generated output was `experiments/archive/frontier_consolidation/training/runs/medium_compact_qwen_all/outputs.jsonl`. Generation settings were batch size 64, maximum 80 new tokens, temperature 0.1, and CUDA execution. The recorded generation produced 754,777 tokens for 21,465 prompts in 962.2 s.

## Compact source and selection chain

The medium compact prompt file contains 21,465 prompts from 469,887 FineWeb source words and has SHA-256 `c0ef389b6168b707996d807a2f8567dfdc3ff59025a913bf047d79ba75d2a807`. Automatic compact analysis accepted 18,682 rewrites (rate 0.8703), with 400,665 accepted source words, 244,568 accepted rewrite words, and weighted rewrite/source ratio 0.6104. The selected reinvest subset used 12,155 source-rewrite pairs with 261,803 source words, 161,708 rewrite words, and 423,511 pair words before insertion into the clean-Qwen row-holdout overlay.

## Exact score-text overlap ancestry

changed block overlap ancestry current official's current-coordinate overlap remains 7 rows, 9 unique strict-score n-grams, and 11 strict-score hits, only in `{'aoa': 4, 'glue_filtered': 7}`. The longest span is 8 tokens. Recomputing over row-level pair ancestry found 9 matched n-gram occurrences in original FineWeb source text and 0 in generated compact rewrite text. Thus the small exact-overlap set is not evidence that Qwen generated evaluation strings; it is ordinary source/eval phrase overlap that remains inspectable in `experiments/archive/representation_and_objectives/data/changed_block_overlap_ancestry_current_official/changed_block_overlap_ancestry_current_official_rows.jsonl`.

## Remaining work

The seed43022 official-coordinate score is unchanged. This addendum closes the exact generator-identity gap, but model/source license language still needs official-source recording before public packaging. The seed43122 full vector remains the next decisive evidence for robustness.

JSON: `experiments/archive/representation_and_objectives/data/compact_provenance_and_overlap_addendum/compact_provenance_and_overlap_addendum.json`
