# repaired relation first acquisition synthesis repaired relation-first acquisition synthesis

## What was corrected

The multiseed and relation first relation-first extraction is preserved as a semantic improvement, but its first scorer/trainer is not usable evidence for coherent86 acquisition. A model-identity audit showed that generic `AutoModelForMaskedLM` loading without trusted custom loading instantiates stock `DebertaV2ForMaskedLM` with 0 private-adapter parameters and ignores adapter tensors. The repaired harness uses the established `FrozenSlowPrivateDebertaV2ForMaskedLM` loader, verifies 995,584 private-adapter parameters and executed scale 0.75 in all 8 layers, and freezes every non-private parameter for training.

The construction was also repaired: each pair now has one shared replacement value, so for a fixed query the `update A` versus `update B` contrast changes only the update recipient. Scoring uses the explicit final `{STATE}` span with all candidate tokens masked simultaneously and no fallback search. Four-condition success is the minimum over the four individual signed margins, not gamma after averaging.

## Trusted parent baseline

On the repaired held set, trusted coherent86 has four-condition success 0/30 and query-orientation success 0/60. Mean U=+6.991, mean R=-6.938, beta=+0.026, |alpha|=+6.965, and mean min-four signed margin=-7.962. Thus the valid failure statement is not the multiseed and relation first wording; it is that coherent86 strongly prefers the shared new phrase in update contexts but does not condition that preference on which entity was updated when the new phrase is fixed.

## Bounded acquisition result

A bounded answer-only run on the repaired rows, using only coherent86 private adapters, moved train to 81/90 four-condition success and held to 28/30 four-condition success (58/60 query orientations) at 80 epochs. Held mean U=+5.670, R=+7.206, beta=+6.438, |alpha|=+1.768, min-four=+2.036. This establishes learnability and held source/entity transfer under direct relation-aligned answer supervision on the repaired natural relation substrate. It does not by itself attribute the effect to credit allocation relative to ordinary MLM; that requires matched arms.

## Background corruption comparison

- corrupted_answer_only: held 2/30 four-condition success and 16/60 query orientations at 60 epochs; mean min-four=-1.571. Cumulative answer-label positions 45120, background-corrupted positions 305495, answer/background overlap 0.
- corrupted_answer_plus_bg: held 2/30 four-condition success and 15/60 query orientations at 60 epochs; mean min-four=-1.933. Cumulative answer-label positions 45120, background-corrupted positions 305495, answer/background overlap 0.

Both corrupted arms used identical background input corruption counts and no answer/background overlap. Compared with the no-corruption answer-only trajectory, standard-style corruption of the supporting context sharply slows acquisition; adding background labels produces a smaller additional deficit at this horizon. This means the repaired natural substrate separates at least three factors: semantic packet validity, direct answer credit, and preservation of the evidence tokens the answer needs. It weakens any simple story that background loss alone is the bottleneck.

## Consequence for the BabyLM bridge

The next scientifically meaningful BabyLM-facing comparison should keep the repaired contract and trusted loader, then compare matched relation-packet objectives under legal exposure: answer-only with uncorrupted support, answer-only with standard support corruption, and answer+background with matched support corruption, before mixing with ALN/filler. Held relation transfer is now real in the clean answer-only setting; the unresolved question is how to retain that selection computation while coexisting with ordinary MLM/ALN rather than treating a successful cloze intervention as a submit-ready training principle.

## Figures and files

- `experiments/archive/functional_learning/figures/repaired_relation_first_held_success.png`
- `experiments/archive/functional_learning/figures/repaired_relation_first_min4_margin.png`
- `experiments/archive/functional_learning/figures/repaired_relation_first_beta_alpha_traj.png`
- model identity audit: `research/documents/functional_learning/data/model_identity_audit/model_identity_audit.md`
- repaired parent score: `experiments/archive/functional_learning/data/relation_first_repaired/trusted_parent_summary.json`
- answer-only run: `experiments/archive/functional_learning/data/relation_first_repaired_e80/answer_only_private_e80_seed40040/training_summary.json`
- background comparison: `experiments/archive/functional_learning/data/revision_040b_repaired_bg_comparison/bg_comparison_summary.json`
