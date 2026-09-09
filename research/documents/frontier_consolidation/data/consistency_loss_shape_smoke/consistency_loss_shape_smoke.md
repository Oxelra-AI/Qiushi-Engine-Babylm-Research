# consistency route stream validation and interface consistency-loss shape smoke

Exercise tensor pooling and normalization for a possible positive-only source-view consistency auxiliary loss, without model training/evaluation.

CPU-only; no model training/evaluation and no route choice.

## Samples
- front_batches: batches_seen=4, eligible_batches=4, aux_records=180, aux_rows=45, row_mean_loss_mean=0.004202, source_norm_mean=4.2328, rewrite_norm_mean=5.1647
- stride_batches: batches_seen=4, eligible_batches=4, aux_records=132, aux_rows=33, row_mean_loss_mean=0.004168, source_norm_mean=4.2671, rewrite_norm_mean=5.1896

## Future trainer loss contract
- Use model outputs with hidden states enabled, pool token ranges for both-visible source/rewrite pairs, and compute a low-weight positive-only agreement loss.
- Prefer row-normalized symmetric stop-gradient MSE/cosine agreement for a first screen so rows with more packed pairs do not dominate and both sides receive a gradient through one term.
- Keep MLM as the primary objective and do not add negatives, static-prior masking, corpus changes, tokenizer changes, or optimizer changes in the first source-view consistency screen.

## Interpretation
- The shape smoke removes a tensor-construction risk for the broad-disappearance branch, but it is not behavioral evidence and does not justify GPU use without the mature 70M/80M pattern.
- The auxiliary payload is large enough in ordinary batches to support a row-normalized loss if that branch is selected.

Full JSON: `experiments/archive/frontier_consolidation/data/consistency_loss_shape_smoke/consistency_loss_shape_smoke.json`
