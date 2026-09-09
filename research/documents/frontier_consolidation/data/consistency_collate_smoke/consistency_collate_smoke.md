# consistency route stream validation and interface consistency-collate smoke test

Prototype the dataset/collate interface needed for a future source-view consistency objective, using frozen training-side spans and tokenizer but no training/evaluation.

This is CPU-only. It does not train, evaluate, alter the corpus/tokenizer, or choose the route.

## Samples
- front_4096_rows: rows=4096, batches=16, aux_rows=214, aux_pair_records=862, aux_batches=16 (1.0000), aux_pairs_per_batch mean=53.88, max=74.0, aux_errors={}
- stride_4096_rows: rows=4096, batches=16, aux_rows=158, aux_pair_records=648, aux_batches=16 (1.0000), aux_pairs_per_batch mean=40.50, max=66.0, aux_errors={}

## Future trainer interface
- Dataset returns the inherited MLM tensors plus example_id, global_row_1based, and aux_pairs looked up by example_id.
- Collate keeps tensor stacking unchanged and adds an aux_records list with batch_row, example_id, pair_id, source_ranges, rewrite_ranges, and per-side token counts for both-visible pairs only.
- A future auxiliary loss should compute pooled source/rewrite representations from model hidden states after the forward pass, average within pair records, then normalize per eligible row or per eligible pair so batches with many changed rows do not dominate.
- No in-batch negatives or static-prior masking are part of this first source-view consistency interface; it must stay a single-variable auxiliary-loss screen if selected.

## Interpretation
- Successful smoke testing means source-view consistency is implementable without changing corpus order or tokenizer, but it is not evidence that it improves BabyLM scores.
- The mature legal-tokenizer treatment pattern remains necessary before spending GPU on this objective; this script only removes one engineering uncertainty from the broad-disappearance branch.

Full JSON: `experiments/archive/frontier_consolidation/data/consistency_collate_smoke/consistency_collate_smoke.json`
