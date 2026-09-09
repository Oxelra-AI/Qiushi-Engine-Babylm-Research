# budget matched design note scientific interpretation: budget matching weakens the retained-selector curriculum claim

## What was verified before the new run

`scripts/verify_budget_gap.py` re-read `data/branching/results.json` and verified the earlier integration note's quantitative claims:

- branching experiment `switch_qfirst_full` ended strong in all 3 seeds under trained-entity metrics: final top4 > 0.95, B-swap > 5, Q-swap > 5.
- branching experiment `switch_qfirst_ctx_only` stayed at chance/bag level in all seeds: final top4 < 0.30.
- The bound query-first checkpoints lost binding immediately under blocked query-to-context evaluation: standard top4 about 0.99/0.998/1.00 but blocked top4 about 0.24/0.26/0.27.
- The original branching experiment comparison against query first binding compact summary was not budget-matched: the prepared arm had P=300/500/400 answer-only acquisition epochs plus 500 full-objective epochs, whereas query first binding compact summary fresh full had only 500 total epochs.

The verification note is `notes/verification_and_budget_gap.md`.

## budget matched design note run

The new script `scripts/budget_matched_full_objective.py` compared four arms through 1000 cumulative epochs with matched initialization, epoch-indexed rows, minibatch order, update count, architecture, sequence length, and post-switch full-objective rows:

1. `fresh_qfirst_full`: full next-token objective from epoch 1.
2. `prep_bound_ans_then_full`: bound answer-only for P epochs, then full objective.
3. `prep_bag_ans_then_full`: unbound bag-answer-only for the same P epochs, then bound full objective.
4. `prep_ctx_then_full`: context-only for P epochs, then bound full objective.

It also added held-entity B-swap, held Q-swap, and held corruption probes that were missing from branching experiment. The canonical analysis is `notes/budget_matched_full_objective_analysis.md`, produced by `scripts/analyze_budget_matched.py` with a unified success definition requiring top4, B/Q margins, B/Q positive fractions, Q-swap both-correct, and query-novel selectivity.

## Central result

The budget-matched run **does not support the strong branching experiment interpretation that retained bound selector structure uniquely makes later full-objective learning efficient**.

At the matched branching experiment endpoint P+500:

| arm | strong seeds at P+500 | mean P+500 top4 |
|---|---:|---:|
| fresh full | 2/3 | 0.745 |
| bound answer prep -> full | 3/3 | 0.997 |
| unbound bag answer prep -> full | 3/3 | 0.997 |
| context prep -> full | 2/3 | 0.870 |

At epoch 1000:

| arm | strong seeds at 1000 | mean final top4 |
|---|---:|---:|
| fresh full | 2/3 | 0.757 |
| bound answer prep -> full | 3/3 | 1.000 |
| unbound bag answer prep -> full | 3/3 | 0.997 |
| context prep -> full | 2/3 | 0.867 |

Thus query first binding compact summary's earlier `qfirst_full_500` result was partly a truncation artifact: ordinary query-first full-objective training can acquire trained-entity binding if allowed the same cumulative budget, reaching strong binding in seeds 43 and 100 by epochs 425 and 550 respectively. Seed 42 remains bag-level under fresh full to epoch 1000, while all three preparation schedules rescue it.

## Per-seed timing matters more than endpoint counts

The paired seed trajectories reveal path dependence rather than a clean retained-selector advantage.

- Seed 42: fresh full fails through 1000, while bound, bag, and context preparations all eventually produce trained-entity binding. This shows objective scheduling/preparation can rescue a difficult seed, but the rescue is not specific to bound selector preparation.
- Seed 43: fresh full reaches strong binding at epoch 425, before the bound-prep arm even finishes its P=500 answer-only preparation; after the full switch, the bound-prep arm does not re-enter strong binding until about 400 full-objective epochs later. Fresh is not less efficient for this seed under cumulative budget.
- Seed 100: fresh full reaches strong binding at epoch 550; bound-prep reaches P=400 bound state under answer-only, collapses after the switch, and re-enters strong binding at cumulative epoch 675. Fresh reaches the full-objective solution earlier in cumulative epochs for this seed.

The bound-prep arm's post-switch first-strong full-objective epochs were [150, 400, 275]. The fresh arm's cumulative full-objective first-strong epochs were [none by 1000, 425, 550]. This is not a uniform advantage of acquired binding making later full-objective rows more valuable. It is evidence for strong seed-dependent trajectory shaping.

## Unbound and context controls change the mechanism interpretation

The unbound bag-answer preparation is especially important. It has no expected query-specific binding during preparation: at P, its top4 remains near chance and B/Q swap metrics are near zero for all seeds. Yet after switching to the bound full objective it reaches trained-entity binding in all 3 seeds by P+500.

This means that budget matched design note cannot attribute the bound-prep recovery to retained entity-specific selector structure. A more plausible current reading is that early non-full training changes optimization state, output-family calibration, answer-position/bag familiarity, or later credit allocation in a way that can help some seeds escape the bag-level regime. The context-only preparation also reaches strong trained-entity binding in 2/3 seeds, further showing that generic sequence/model maturation can matter.

The result therefore redirects the candidate principle from a simple curriculum claim to a more precise question: **which aspects of early training state make later relation-bearing evidence usable, and when is the useful state actually a reusable relation computation rather than generic optimization/bag-output preparation?**

## Held-entity scope is weak

No arm met the held-behavior definition used in the analysis. Held-entity behavior remains much weaker than trained-entity binding.

Final held means:

| arm | held top4 | held B-swap | held corruption selectivity |
|---|---:|---:|---:|
| fresh full | 0.243 | +0.375 | -0.018 |
| bound prep -> full | 0.389 | +3.119 | +0.188 |
| bag prep -> full | 0.115 | -1.816 | -0.145 |
| context prep -> full | 0.189 | -1.003 | -0.075 |

Bound preparation preserves the best held tendency, especially for seeds 42 and 43, but it is not strong or consistent: held top4 stays far below trained-entity top4, held B-swap positive fractions are not high enough, and held corruption remains small. Full-objective continuation can erode the strong held behavior seen at the answer-only preparation endpoint: for example seed100 bound prep has held top4 0.953 and held B +7.774 at P=400, but final held top4 0.238 and held B +1.035 after full training to epoch 1000.

Therefore budget matched design note does not yet establish reusable token-general equality-style binding. It establishes strong trained-entity binding and weak/unstable held transfer.

## What remains scientifically valuable

budget matched design note still strengthens the research in several ways:

1. It corrects the earlier overstatement: branching experiment's endpoint 3/3 vs query first binding compact summary's 1/3 was not a valid efficiency comparison.
2. It shows that query-first full-objective training can acquire trained-entity binding with enough cumulative budget in some seeds.
3. It shows that preparation schedules can rescue at least one difficult seed where fresh full training remains bag-level to epoch 1000.
4. It shows that the rescuing factor is not uniquely retained entity-specific binding, because unbound bag-answer and context preparations also rescue trained-entity binding in multiple seeds.
5. It exposes a new hard problem: behavioral binding can collapse and later reappear, while held transfer can be strong under focused answer-only training and then degrade under full-objective consolidation.

The best current controlled statement is not “acquired binding makes later full training generally more efficient.” It is: **finite training can enter qualitatively different binding or bag-level regimes under the same architecture and data generator; objective allocation and early training state shape which regime later full-objective training reaches, but the helpful state has not yet been identified with reusable bound-selector knowledge.**

## Next scientific work suggested by the result

The next Execute work should isolate what the helpful preparation state is. The highest-value controls are:

1. **Mechanism-matched blocked preparation:** train bound answer-only for P epochs while blocking query-to-context attention, then switch to standard full objective. This matches labels and answer credit while preventing the prospective-marker route. If it matches bound prep, prospective selector retention is not needed; if it fails while standard bound prep succeeds, prospective marker structure matters.
2. **Optimizer reset at switch:** compare carried Adam state versus reset Adam state for bound and bag preparations. If reset removes the advantage or the collapse/recovery shape, optimizer moments and gradient-scale adaptation are central.
3. **Post-switch blocked evaluation/checkpoints:** record blocked query-to-context evaluation during and after full-objective recovery to see whether recovered binding still uses prospective marking or has changed to an IS-position lookup strategy.
4. **Credit-share sweep/hysteresis:** measure formation and maintenance across answer-position credit shares alpha. branching experiment already gives endpoints: alpha=1 maintains, alpha=0 destroys, alpha=1/16 can collapse then recover; query first binding compact summary w16 at 500 failed, so the relation is not a simple monotone dose law at short budget. A sweep could turn the schedule effect into a quantitative law.
5. **Fixed finite dataset version:** the current generator supplies fresh rows each epoch, so it studies limited compute/credit under a finite generator rather than repeated reuse of a fixed dataset. A fixed-row reuse experiment is needed before extending the result to data-limited learning in the strict sense.
6. **Original-order transfer under matched budget and more seeds:** branching experiment's 1/3 transfer is the route with the most natural-language relevance, because natural reading often requires retrieval after context encoding. It remains weak and underpowered.

These are research continuations, not final-expression tasks.

## Files

- branching experiment verification and budget gap: `notes/verification_and_budget_gap.md`
- budget matched design note design: `notes/budget_matched_design_note.md`
- Main script: `scripts/budget_matched_full_objective.py`
- Raw data: `data/budget_matched_full_objective/results.json`
- Auto note: `notes/budget_matched_full_objective_result.md`
- Canonical analysis: `notes/budget_matched_full_objective_analysis.md`
- Analysis JSON: `notes/budget_matched_full_objective_analysis.json`
- independent_review integration: `notes/independent_review_design_verification_integration.md`
