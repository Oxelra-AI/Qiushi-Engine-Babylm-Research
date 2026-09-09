# 020 mechanism synthesis design: does full-objective continuation erase held-symbol binding through tied embedding drift?

## Why this route follows budget matched design note

budget matched design note corrected the earlier preparation-efficiency interpretation. At a matched cumulative budget, query-first full-objective training reaches trained-entity binding in 2/3 fresh seeds, and unbound bag/context preparations can also rescue some trajectories. The stronger remaining phenomenon is not endpoint recovery on familiar symbols but the dissociation between familiar-symbol binding and broader transfer.

The clearest case is seed100 in `prep_bound_ans_then_full`: at the answer-only preparation endpoint P=400, held-query behavior is strong (`held_top4=0.953`, held B-swap `+7.774`, held selectivity `+0.914` in `notes/budget_matched_full_objective_analysis.md`). After full-objective continuation to epoch 1000, trained binding is perfect but held behavior collapses (`held_top4=0.238`, held B `+1.035`, held selectivity `-0.029`). This is an observed loss of previously demonstrated transfer while training-distribution behavior succeeds.

## Mechanistic alternative to selector destruction

The current CLM ties input embeddings and output classifier rows:

```python
return self.ln(h) @ self.tok.weight.T
```

Held entity tokens `ENT(8), ENT(9)` never appear as training inputs or correct next-token targets, but in a tied softmax they are still output classes. Full next-token training can move their rows through output-side softmax gradients and AdamW/scale effects. Because the same rows are used as input embeddings, this can damage the geometry needed to recognize held query entities, even if the contextual selector computation remains partly intact.

Therefore held-transfer loss could reflect at least two causes:

1. **Selector specialization/destruction:** the contextual network/readout becomes specialized to familiar entities and no longer implements token-general matching.
2. **Tied embedding drift:** full-objective lexical prediction moves unseen entity input rows because they are tied to the output classifier; held matching fails although much of the selector would still work if the held input geometry were preserved.

## Decisive comparison

`scripts/embedding_specialization.py` will recreate the bound answer-only preparation and branch into full-objective continuation with interventions that specifically protect or separate held entity input embeddings.

For each seed, train query-first answer-only to the same preparation duration P used in budget matched design note (`42:300`, `43:500`, `100:400`), then continue to cumulative epoch 1000 under these branches:

1. `tied_carry_full`: tied-readout full-objective continuation carrying Adam state, matching budget matched design note's bound-prep branch.
2. `tied_carry_freeze_held`: same as (1), but after every optimizer step restore the held entity rows `ENT(8), ENT(9)` to their preparation values. This leaves the contextual network and RWT output rows free to learn full next-token structure while preventing drift of the unseen entity input rows.
3. `tied_reset_full`: tied-readout full continuation with optimizer reset at the switch, separating row-drift effects from carried Adam moments.
4. `untied_reset_freeze_held_input`: copy the prepared tied model into an untied-readout model whose output head is initialized equal to the tied embedding matrix, then run full-objective continuation while restoring only held input embedding rows after each step. This makes output classifier learning functionally available without forcing output-side gradients to rewrite unseen input embeddings.

For the ordinary tied branch, the script also evaluates a **held-row restore at inference**: temporarily replace only the held entity input rows by their preparation values while leaving the trained contextual network and RWT readout fixed. A selective rescue under this zero-training intervention would be especially strong evidence that row drift is sufficient for much of the held-transfer collapse.

## Measurements

The run uses the budget matched design note probe bank and records:

- trained-entity binding: correct NLL, tie-safe top among context attributes, B-swap, Q-swap, query-novel selectivity;
- held-query behavior: held top4, held B-swap, held Q-swap, held corruption selectivity;
- blocked query-to-context evaluation, to track whether the prospective marker route remains active;
- row drift statistics for held entity rows, trained entity rows, attribute rows, and RWT rows relative to the preparation checkpoint;
- standard evaluation and held-row-restored evaluation for each branch and evaluation point.

## Interpretation

- If held-row restoration at inference rescues a collapsed tied model while trained binding remains fixed, held-transfer loss is largely caused by input embedding geometry drift rather than loss of the contextual selector.
- If freezing held rows during full continuation preserves held transfer while `tied_carry_full` loses it, broader next-token learning can proceed without sacrificing held-symbol binding; this would give a concrete preservation mechanism for reusable knowledge.
- If untied readout preserves held behavior relative to tied reset, tied output/input sharing is a causal contributor.
- If none of these interventions preserve held behavior, the loss likely lies in contextual-network or readout specialization, not just held token embeddings.
- If held protection preserves held behavior but damages trained full-objective binding, the intervention is not a clean preservation principle; the valuable result would require a way to preserve transfer without sacrificing familiar-symbol learning.

This is a better next test than a broad endpoint-recovery sweep because it asks which part of useful knowledge becomes specialized during continued learning, and whether specialization can be prevented while new full-objective learning continues.
