# error partition and paired contrastive Research Note: Error Partition and Paired Contrastive Construction

## Key Diagnostic: rel_eq0 Error Mass Partition

The binding arms lose 7-16 points on rel_eq0 vs chck82. earlier analysis identified a 
last-operation recency component. error partition and paired contrastive partitioned the FULL error mass.

**Results** (at `data/rel_eq0_error_partition/summary.md`):

For alpha0.75: 292 flips (chck82 correct → binding wrong) on rel_eq0 (n=1541):
- Only 28 (9.6%) match a last-operation recency answer
- **Zero** nothing-preference shift
- **Zero** form/length bias (Entity options are word-count matched by construction)
- **Zero** initial-state matching
- The remaining ~90% (264 flips) are to confusable distractor options with the same structure

**Conclusion**: The dominant error component is NOT a specific bias toward recency,
length, or any identifiable option class. It is a **generic loss of initial-state
confidence** — the binding arm becomes less certain about the correct unchanged state,
especially as irrelevant operations accumulate. The answer CE objective rewards a global
update-selection prior that happens to help B rows more than it hurts A rows.

## Paired Contrastive Objective

**Script**: `scripts/paired_contrastive_trainer.py`

The key modification: `softplus(L_a - L_b)` where:
- L_a = mean CE loss on unchanged-entity (row A) answer tokens
- L_b = mean CE loss on updated-entity (row B) answer tokens

Invariance property:
- Global recency shift: L_a↑, L_b↓ → (L_a - L_b)↑ → penalty↑ → optimizer opposes shift ✓
- Entity-conditioned correct model: L_a ≈ L_b → penalty = constant → no gradient ✓

Combined with:
- Per-row answer CE on both A and B
- Non-answer KL leash on both A and B (from binding transfer and composition prediction)
- Ordinary text interleaving
- Frame-varied rows (earlier analysis data already has both A+B halves: 8330 unchanged + 8330 updated)

**CPU Smoke**: PASSED. Config = `PAIRED_CONTRASTIVE_CONFIG`
contrastive_loss = softplus(9.03 - 6.39) = 2.72 ✓

## Launched Tasks

1. Paired contrastive chck82 GPU training
   - 25 binding epochs, 3,992,800 main words, contrastive_lambda=1.0
   - At `training/runs/paired_contrastive_chck82`

2. Full8M composition official eval
   - earlier analysis cheap7/Entity evaluation
   - At `data/eval_full8M_composition`

## Remaining Design Questions

1. **Entity-order balance**: ~80-85% of packets mention the unchanged entity first.
   The contrastive term penalizes the global shift but doesn't address position shortcuts.
   Assignment-reversal rows provide the held-out test for this.

2. **Retention rows**: No current training rows have 2+ distractor operations with
   an untouched queried entity. This is the Entity rel_eq0 structure that loses 14-21 points.
   Building these requires chaining update descriptions from multiple packets.

3. **Two-update rows**: Update X then Y, query X → answer is neither source nor most-recent.
   Practices the official hard stratum (rel_ge1_postrel_ops_gt0, n=3389).

4. **Coherent86 KL target**: For coherent86-start arms, KL should anchor to coherent86's
   own private-on behavior, not stock private-off. This prevents relitigating the replay gain.
