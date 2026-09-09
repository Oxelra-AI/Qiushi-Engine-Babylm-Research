# corpus lineage multiplicity audit corpus lineage and multiplicity audit
This CPU-only audit protects the active compliant-tokenizer compact_view_reinvest endpoint while full official evaluations run. It is not a BabyLM score.
## Pass/fail summary
- status: `CORPUS_LINEAGE_MULTIPLICITY_AUDIT`
- validation errors: `0` []
- active 10M SHA: `215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23`; rows `64740`; words `10000000`
- active 100M SHA: `3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691`; rows `647400`; words `100000000`
- 100M multiset is 10x 10M pool: `True`
- each 64,740-row tenth is a permutation of the 10M pool: `True`

## 10M source classes
- `official_babylm_source_row`: rows `49498`, words `7919680`, fraction `0.791968`
- `inherited_official_source_qwen_paraphrase_pair_row`: rows `12236`, words `1656800`, fraction `0.165680`
- `fineweb_source_qwen_compact_rewrite_pair_row`: rows `3005`, words `423511`, fraction `0.042351`
- `neutral_topup_from_heldout_official_row`: rows `1`, words `9`, fraction `0.000001`

## Changed block and Qwen-pair lineage
- compact FineWeb selected pairs: `12155` pairs, source words `261803`, rewrite words `161708`, pair words `423511`.
- changed metadata rows: `3006`, words `423520`, pair refs `12155`, top-up rows `1`.
- compact selected ids all in accepted rewrite pool: `True`; all selected ids represented in changed metadata: `True`.
- inherited qwen_pair_packed rows in active 10M: `12236` rows, `1656800` words; packed metadata rows `12236`.

## Generation provenance
- compact FineWeb Qwen output model counts: `{'Qwen/Qwen3.5-9B': 21465}`, generated token sum `754777`, output SHA `85660e4af2179e38003a95942a01dc0bbac7d6a6a76ca8991bff3d80bc406cc5`.
- compact prompt source assets: `{'peer_step010_strict_factual': 12801, 'peer_step010_balanced': 8664}`; tiers `{'medium': 21465}`.
- inherited clean-Qwen base generation model `Qwen/Qwen3.5-9B`, prompts `45000`, generated tokens `2559576`, output SHA `b03d1310b84a155a954bd6834386a3615593eb9b4db15ae1c662fd4520409655`.

## Interpretation for the SOTA route
The corrected-tokenizer endpoint has a stronger rule-facing data record after this audit: the tokenizer source and training pool are the same exact 10M file, and the 100M model stream is only ten presentations of that pool. This removes a major source of hidden corpus drift before interpreting the pending official scores. The audit does not create a score and does not justify new training; it prepares interpretation of the completed corrected-tokenizer evaluations.

## Artifacts
- JSON: `experiments/archive/representation_and_objectives/data/corpus_lineage_multiplicity_audit/corpus_lineage_multiplicity_audit.json`
- source CSV: `experiments/archive/representation_and_objectives/data/corpus_lineage_multiplicity_audit/source_word_breakdown_10M.csv`
- source-class CSV: `experiments/archive/representation_and_objectives/data/corpus_lineage_multiplicity_audit/source_class_word_breakdown_10M.csv`
- compact examples CSV: `experiments/archive/representation_and_objectives/data/corpus_lineage_multiplicity_audit/compact_changed_block_examples.csv`
