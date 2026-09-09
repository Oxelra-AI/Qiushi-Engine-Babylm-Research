# Partial trajectory evidence and proposed mechanism controls

## What the incomplete trajectory screen establishes

The initial trajectory screen was incomplete; usable no-AoA trajectory summaries are retained under `data/trajectory_screen/`.

Available scientific summaries:

- `data/trajectory_screen/clean_qwen_seed43022_trajectory_summary.json`
- `data/trajectory_screen/devcurr_seed43022_trajectory_summary.json`

These summaries cover BLiMP, Supplement, EWoK, Entity, COMPS, GlobalPIQA, and Reading only. They do **not** include SuperGLUE or AoA. They therefore are a fast scientific screen, not final Overall evidence.

### Clean-Qwen seed43022 late trajectory

For clean-Qwen seed43022, the 40M--100M no-AoA trajectory peaks at 100M:

- `chck_95M`: equal7_full_eval = 43.0900
- `chck_100M`: equal7_full_eval = 43.112857

The 95M and 100M rows are nearly tied; the 100M endpoint remains the best late no-AoA row in the completed grid. There is no late-checkpoint evidence so far that 70M--90M hides a clearly stronger model for this seed, but the missing 10M/20M/30M sweep must be examined before freezing candidate checkpoints.

### Developmental first-pass seed43022 late trajectory

For aoa safety audit and route devcurr seed43022, completed 40M--95M rows are all below the corresponding clean-Qwen seed43022 rows. Its best completed late row is:

- `devcurr_seed43022 chck_85M`: equal7_full_eval = 42.338571

This is roughly 0.774 below clean-Qwen 100M on the seven-column no-AoA screen. The result argues against the specific **source-block + source-internal lexical ordering** first-pass schedule as a late broad-NLP improvement route. It does not close all developmental curriculum ideas, and it does not prove anything about AoA until full candidate evaluation is run after the candidate set is frozen.

The planned grid completion includes `chck_10M`, `chck_20M`, `chck_30M`, devcurr 100M, and both second-seed grids. These measurements were not complete in the evidence summarized here.

## Candidate selection after grid completion

Do not use AoA-only outputs for route selection or schedule design. Candidate promotion should be AoA-blind until the candidate set is frozen.

Recommended policy:

1. Use `scripts/rank_trajectory_candidates.py` on `data/trajectory_screen/` after the required measurements are complete.
2. Keep:
   - the best no-AoA row for each target;
   - clean-Qwen seed43022 95M and 100M because they are nearly tied;
   - any early 10M/20M/30M row that unexpectedly dominates its family;
   - at least one predeclared devcurr representative if we still want to measure the developmental hypothesis independently of no-AoA weakness, preferably its best no-AoA row;
   - Pareto-relevant rows whose task profile differs materially, not only the top equal7 row.
3. Run SuperGLUE first or full nine-column directly on the frozen small candidate set using the corrected evaluator. AoA is then a final official-style measurement, not a feedback signal for new curricula.
4. Use official nine-column Overall only after corrected full eval. Report AoA raw correlation and leaderboard-scale AoA separately.

A wrapper is prepared at `scripts/full_eval_candidates.py`. It registers all clean-Qwen and devcurr checkpoint endpoints (`10M,20M,30M,40M,50M,60M,70M,75M,80M,85M,90M,95M,100M`) and delegates to the corrected full overall eval evaluator (`babylm_official_scoring.py`). Its syntax and execution had not yet been verified when this note was written.

## Mechanism route if no checkpoint exceeds SOTA

The preferred next structure-changing experiment is **bidirectional pair order**, with the source-balanced easy/hard curriculum retained as a separate developmental control. Reason:

- The validated active component is same-window generated second-view correspondence.
- The bidirectional arm preserves the exact clean qwen compliance and validity clean-Qwen 10M row sequence, 100M pass/order sequence, selected pair set, pair dose, filler rows, word totals, row identities, and source labels.
- It changes only whether a pair appears original→rewrite or rewrite→original for half of pairs by a fixed hash rule.
- This directly tests whether the current one-direction pair geometry creates positional/template specialization that hurts COMPS/BLiMP/EWoK/Reading.

Expected signatures:

- COMPS/Reading/BLiMP improve while Entity/Supplement/SuperGLUE stay near clean-Qwen: directionality/position bias is a real cost of the mechanism.
- Entity/SuperGLUE fall strongly: fixed original→rewrite may itself be the effective denoising cue.
- No material change: move to pair-row topology/contextualization rather than direction.

The original record reports a successful syntax check of `materialize_bidirectional_pair_order.py`, but no execution at that point. A matched dual-seed bidirectional arm remained conditional on successful materialization and on no trajectory candidate already providing a clear SOTA result.

The source-balanced easy/hard corpora remain valuable as a developmental-curriculum decomposition, especially if the early 10M--30M sweep shows devcurr transient learning advantages. But the partial late trajectory makes it a lower immediate SOTA-probability route than bidirectional pair order.

## Mechanism conclusion

Preserve the attention-visible semantic equivalence event, but remove artificial multi-pair/one-direction geometry or add official-text consolidation/contextualization to recover syntax, relation, and reading statistics.
