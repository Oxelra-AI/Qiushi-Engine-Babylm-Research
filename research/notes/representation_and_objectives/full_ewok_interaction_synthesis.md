# full ewok interaction synthesis — full EWoK interaction synthesis

The completed full-EWoK CPU measurement confirms that the surviving EWoK weakness is large and structurally stable enough to deserve construction work, but not as a broad relation-word replacement. The measured object is conditional target reversal: after the same two target alternatives and their priors are held fixed, the model often fails to make context change reverse the preferred target.

## Main numbers

- legal40_depth_12x384_43022: saved accuracy 0.5129; saved wrong 3,711; stable conditional-reversal failures 2,494 (0.672 of wrong); saved-wrong interaction median -0.736; both-within-context-positive among wrong 0.031.
- legal40_8x480_43022: saved accuracy 0.5113; saved wrong 3,723; stable conditional-reversal failures 2,550 (0.685 of wrong); saved-wrong interaction median -0.787; both-within-context-positive among wrong 0.032.
- Both legal40 models share stable failures on 1,471/7,618 rows (0.193 of all EWoK; 0.554 of rows both models get wrong).
- Either model has a stable failure on 3,573/7,618 rows (0.469).

## Structure

The stable failure mass is concentrated enough to be a real construction target: the first shared-failure rows span agent-properties, material/physical/social/spatial relations, direct and indirect contexts, antonym and other context differences, and concept-swap targets. Deletion interactions are near zero, so the measured reversal is attached to the local context-difference span rather than a target-prior artifact. The local actual-vs-swapped margins are usually not both positive, which is exactly the training signal a future objective would need to repair.

## What this supports

A future route should construct corpus-derived two-context/two-target examples where target priors and lexical plausibility cancel by design: the same two target candidates are scored under two minimally different contexts, and only the context difference should flip the target preference. This result does not support a broad sentence-level relation-corruption objective, and it does not by itself justify H100 pretraining before a generator can build a clean non-evaluation-derived corpus analogue.

## Files

- Summary JSON: `experiments/archive/representation_and_objectives/data/ewok_interaction_synthesis/ewok_interaction_synthesis.json`
- Shared stable-failure rows: `experiments/archive/representation_and_objectives/data/ewok_interaction_synthesis/both_models_stable_failure_rows.csv`
