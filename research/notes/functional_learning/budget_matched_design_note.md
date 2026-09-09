# budget matched design note design note: budget-matched test of preparation value

## Why this step is needed

branching experiment established three concrete facts directly from the orbit-binding task:

- a bound query-first answer-only checkpoint can return to strong binding after switching to the full next-token objective;
- context-only continuation rapidly removes behavioral binding;
- the bound query-first solution initially depends on query-to-context access, implying prospective context marking.

The branching experiment comparison against query first binding compact summary full-from-scratch is not enough to establish a data-efficiency result. The prepared arm used 300/500/400 answer-only epochs plus 500 full-objective epochs, while the earlier fresh full-objective run used only 500 total epochs. Since each epoch draws fresh finite examples, the difference is both compute and experience.

## Scientific question

Does focused preparation make later full-objective experience more effective under the same cumulative budget, or does ordinary full-objective training eventually find the same binding computation when given the same number of epochs?

A stronger version asks whether any advantage comes from retained entity-specific selector structure rather than generic training, bag-membership familiarity, or optimizer/model maturation.

## Comparison in `budget_matched_full_objective.py`

For each seed, use the same initialization seed and the same epoch-indexed row stream as the prior experiments. Compare four arms through 1000 cumulative epochs:

1. `fresh_qfirst_full`: query-first bound target, full next-token objective from epoch 1.
2. `prep_bound_ans_then_full`: query-first bound target, answer-only for the seed-specific branching experiment acquisition duration P, then full objective to epoch 1000.
3. `prep_bag_ans_then_full`: query-first bag-independent answer-only for the same P epochs, then bound-target full objective to epoch 1000. This is equally trained answer-output preparation without entity-specific binding supervision.
4. `prep_ctx_then_full`: query-first context-only training for P epochs, then bound-target full objective to epoch 1000. This controls generic context/objective maturation without answer pressure.

The seed-specific preparation durations are inherited from branching experiment branch starts: P=300 for seed 42, P=500 for seed 43, P=400 for seed 100. The matched branching experiment endpoints are therefore total epochs 800, 1000, and 900; the new run records those points and the final 1000-epoch point.

## Measurements

The script records standard trained-entity binding metrics at every evaluation point:

- correct NLL;
- tie-safe top among the four context attributes;
- B-swap margin;
- Q-swap margin;
- query-novel selectivity.

It also adds held-entity probes missing from branching experiment:

- held standard top among context attributes;
- held B-swap;
- held Q-swap;
- held query-novel corruption.

The central comparison is the acquisition curve: first epoch reaching strong trained-entity binding, behavior at total epoch P+500, and final behavior at 1000 epochs.

## Outcome interpretations

- If `fresh_qfirst_full` reaches strong binding by the same cumulative epochs, branching experiment's endpoint difference cannot carry a curriculum-efficiency result. The remaining value would be timing, reliability before the matched endpoint, and possible held-entity differences.
- If `prep_bound_ans_then_full` reaches strong binding earlier or more reliably than `fresh_qfirst_full` at the same cumulative budget, focused preparation has made later full-objective examples more effective.
- If `prep_bag_ans_then_full` matches `prep_bound_ans_then_full`, the advantage is likely not retained entity-specific binding; it may be generic bag-output or answer-family preparation.
- If `prep_ctx_then_full` also matches, the advantage may be ordinary context/model maturation rather than answer-driven preparation.
- If bound preparation uniquely improves the curve while bag/context preparation do not, the result supports retained selector structure increasing the value of subsequent full-objective evidence.
- Held-entity strength extends the scope of the computation; weak held transfer would keep the principle bounded to trained symbols even if trained-entity binding is strong.

## Files

- Precheck and branching experiment verification: `notes/verification_and_budget_gap.md`
- Main script: `scripts/budget_matched_full_objective.py`
- Running output target: `data/budget_matched_full_objective/results.json`
- Figure target: `figures/budget_matched_full_objective.png`
- Result note target: `notes/budget_matched_full_objective_result.md`
