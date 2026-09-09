# consistency route stream validation and interface — Source-view consistency route: full-stream validation and interface smoke

Purpose: raise the source-view consistency intervention (Asset B) from a feasibility
sketch to a technically de-risked, corpus-order-accurate construction, so that if the
mature legal-tokenizer clean-vs-reinvest pattern points to broad disappearance, the interface supports a single-variable auxiliary-loss screen without re-deriving the interface.

All consistency route stream validation and interface work is CPU-only. It does not train, evaluate, alter the corpus/tokenizer,
launch GPU work, or choose the intervention route. The route decision still waits on
`s50_t5/t11/t19` → `compare_legal_treatment_trajectory.py` →
`interpret_legal_treatment_pattern.py`.

## What consistency route stream validation and interface established

1. **Full-stream join is exact.** `pair_span_stream_validation.py` hashed the
   frozen 100M file (SHA `3dd19f09...`), confirmed 647,400 rows / 100,000,000 words / ten
   10M passes, and joined the route portfolio and intervention assets span map by `example_id`:
   - `changed_source_rows_in_100m = 30050`, `span_joined_rows = 30050`
   - `changed_source_unmapped_rows = 0`, `mapped_wrong_source_rows = 0`
   - every mapped example appears exactly 10 times (min=max=10)
   - stream pair visibility: both_visible 121,450, source_only 70, invisible 30
   - token truncation occurrences: 520 (52 unique rows × 10)

2. **Passes are shuffled, not identical.** `passes_match_first_pass = False`; the
   trainer re-shuffles each 10M epoch. `pair_span_pass_distribution.py` confirms
   each pass contains all 3,005 changed examples exactly once
   (`all_pass_changed_sets_equal_span_ids = True`), spread across all 253 batches per
   pass (nonempty-batch mean ≈ 11.9 changed rows, min 2–5, max 20–27). **A future trainer
   must join by `example_id`, not by row offset within a pass.** The earlier "kept
   together in one block" wording in the stream-validation output was corrected.

3. **Collate interface works with zero span errors.** `consistency_collate_smoke.py`
   prototyped a span-carrying dataset/collate that returns the inherited MLM tensors
   (`input_ids`/`attention_mask`/`word_group` all [B,256]) plus `example_id`,
   `global_row_1based`, and an `aux_records` list (batch_row, pair_id, source/rewrite token
   ranges, per-side counts) for both-visible pairs. On two 4,096-row samples: front 862
   aux pair records over 214 aux rows, stride 648 over 158; `aux_errors = {}` in both.
   Pair token sizes: source mean ≈ 29 tokens, rewrite mean ≈ 20 tokens.

4. **Positive-only consistency loss is tensor-constructible and differentiable.**
   `consistency_loss_shape_smoke.py` pooled source/rewrite ranges from random
   hidden states (hidden_size 480, batch 256) and computed a low-weight positive-only
   symmetric stop-gradient agreement loss with per-row normalization; `row_mean.backward()`
   produced finite nonzero gradients on all eligible batches (front 180 aux records / 45
   rows, stride 132 / 33). This removes a tensor-construction risk; it is **not** behavioral
   evidence.

## Future trainer contract (only if broad-disappearance branch is selected)

- Dataset: inherited MLM tensors + `example_id` + `aux_pairs` looked up by `example_id`.
- Collate: unchanged tensor stacking + `aux_records` for both-visible pairs only.
- Loss: enable hidden states, pool source/rewrite token ranges, low-weight positive-only
  symmetric stop-gradient MSE/cosine agreement, **row-normalized** so pair-dense rows do
  not dominate. MLM stays primary.
- Single variable: no in-batch negatives, no static-prior masking, no corpus/tokenizer/
  optimizer/seed/exposure/evaluation changes in the first screen.

## Route neutrality

This does not favor Asset B over Asset A (`wwm_static_prior`). The mature 70M/80M
clean-vs-reinvest treatment pattern chooses the intervention family, if any. If the
pattern is selective relational weakness, Asset A is preferred (after the full 64,740-row
expected-budget pass confirms token-budget matching). If it is broad
positive survival, neither learning-signal objective is launched; the deficit is a
representation/optimization problem to compare with the 40k route.

## Artifacts

- `experiments/archive/frontier_consolidation/scripts/pair_span_stream_validation.py`
- `experiments/archive/frontier_consolidation/data/pair_span_stream_validation`
- `experiments/archive/frontier_consolidation/scripts/pair_span_pass_distribution.py`
- `experiments/archive/frontier_consolidation/data/pair_span_pass_distribution`
- `experiments/archive/frontier_consolidation/scripts/consistency_collate_smoke.py`
- `experiments/archive/frontier_consolidation/data/consistency_collate_smoke`
- `experiments/archive/frontier_consolidation/scripts/consistency_loss_shape_smoke.py`
- `experiments/archive/frontier_consolidation/data/consistency_loss_shape_smoke`
