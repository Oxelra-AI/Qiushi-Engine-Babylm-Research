# budget matched design note independent_review verification integration: how to read the budget-matched run

## What the design already fixes

The main budget matched design note script (`scripts/budget_matched_full_objective.py`) fixes the decisive branching experiment comparison defect. Within each seed, every arm starts from the same initialization and sees the same epoch-indexed query-first rows in the same order through the same number of cumulative epochs. From the switch epoch onward, all arms use the same bound-target full next-token objective and the same generated examples. Thus differences after a common cumulative epoch cannot be attributed to branching experiment's earlier unequal total budget.

The intended treatment is objective allocation during preparation, not fewer examples: answer-only preparation gives the answer position approximately 16 times the direct coefficient it receives under the full objective and removes context-token gradients. This tests whether focused credit allocation can form a computation that changes the later full-objective trajectory.

## Immediate implementation checks

independent_review noted that `eval_extended` does not call `model.eval()`. I checked the underlying model in `loss_allocation_binding.py`: the transformer blocks use `nn.MultiheadAttention(..., dropout=0)` and no dropout layers appear. Therefore train/eval mode should not affect numerical values in the current model. The issue should still be repaired in any v2 script for clean reproducibility, but it does not invalidate the current run.

The smoke run executed the fresh, bound-prep, and bag-prep code paths. It did not exercise `prep_ctx_then_full`, but this path uses the same `S17.train_one_epoch` function and the `context_only` mode already exercised in branching experiment. The full run includes it.

## Unified interpretation for the result

The auto-generated budget matched design note note uses a simple summary. The canonical scientific readout should instead come from `scripts/analyze_budget_matched.py`, which uses a unified scale-robust trained-entity criterion:

- top4 >= 0.95;
- B-swap mean >= 5 and B-swap positive fraction >= 0.95;
- Q-swap mean >= 5, Q-swap positive fraction >= 0.95, and Q-swap both-correct >= 0.90;
- query-novel selectivity >= 0.80.

This prevents a seed from being counted as a success only because mean margins are large or the answer head is sharp. The analysis also reports held top4, held B-swap, held Q-swap, and held query-novel corruption. Held Q-swap is mixed because one side of the swap uses a trained entity, so held B-swap and held corruption are the cleaner held-query readouts.

The meaningful quantities are:

1. state at the preparation endpoint P;
2. state at P+500, the matched version of the branching experiment branch endpoint;
3. state at epoch 1000;
4. first strong trained-entity binding epoch;
5. first strong binding epoch after the full-objective switch, reported both as cumulative epoch and as post-switch full-objective epochs;
6. whether held-entity behavior moves with trained-entity binding.

## What would support the central claim

A substantive preparation-efficiency result would require more than final binding after extra training. It would be supported if:

- `prep_bound_ans_then_full` reaches sustained strong binding at P+500 in more seeds than `fresh_qfirst_full` at the same cumulative epoch;
- the bound-prep arm needs fewer post-switch full-objective epochs to regain strong binding than fresh training needs cumulative full-objective epochs;
- `prep_bag_ans_then_full` and `prep_ctx_then_full` do not match the bound-prep trajectory;
- positive-fraction metrics and query-novel selectivity agree with raw margins;
- held B-swap and held corruption improve with trained-entity binding.

This would support the statement that acquired entity-conditioned structure increases the value of later full-objective experience under equal rows and updates.

## What would weaken or redirect the claim

- If fresh full-objective training reaches the same state by P+500 or by epoch 1000, branching experiment's 3/3 vs 1/3 endpoint contrast was a budget artifact; any remaining result must be about timing, not endpoint possibility.
- If bag-answer preparation matches bound preparation, the advantage is answer-family/bag-output preparation or deterministic target coherence rather than retained entity-specific binding.
- If context-only preparation matches bound preparation, generic sequence/model or optimizer maturation is sufficient.
- If improvements appear only in margin means but not in B/Q positive fractions, Q-swap both, or selectivity, the effect is probably answer-head sharpness or calibration rather than routing.
- If held probes stay weak, the result remains bounded to trained entity symbols and does not yet demonstrate reusable equality-style matching.

## Further controls suggested by independent_review

The current run is the necessary first cumulative-budget comparison, but it cannot by itself isolate all mechanisms. Depending on the result, the next Execute work should consider:

1. **Optimizer-state separation:** rerun the bound-prep switch with Adam state reset at P, and compare with carried state. If reset removes the collapse/recovery or the advantage, optimizer moments matter.
2. **Mechanism-matched blocked-prep control:** do answer-only bound preparation while blocking query-to-context attention, then switch to standard full objective. This keeps labels, format, and answer credit fixed while preventing the prospective-marker pathway identified in branching experiment.
3. **Post-switch blocked evaluation:** record whether recovered full-objective binding still collapses under query-to-context blocking. This tells whether the recovered solution uses the original prospective marking route or a later IS-position lookup route.
4. **Credit-share sweep:** vary answer-position credit share alpha during formation and maintenance. A formation threshold higher than a maintenance threshold would turn the curriculum result into a quantitative hysteresis law.
5. **Fixed finite dataset version:** the current generator supplies fresh rows each epoch, so it studies limited compute/credit under a finite generator, not repeated reuse of a fixed dataset. A fixed-row reuse experiment is needed before making a strong data-limited learning claim.
6. **Original-order transfer at matched budget:** branching experiment transfer was 1/3 and is central for natural-language relevance. More seeds and a cumulative-budget comparison would decide whether query-first preparation makes ordinary-order experience more useful.

These controls should not be launched blindly before reading budget matched design note: their priority depends on whether bound preparation is uniquely better than fresh and unbound preparations under the matched budget.
