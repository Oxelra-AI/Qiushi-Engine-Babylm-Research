# Paired repair and multi-token scorer synthesis

## Why corruption vs loss design needed repair

corruption vs loss design made background-objective interference a serious candidate mechanism, but its endpoint comparison was not tightly paired. The corrupted arms used arm-name-dependent mask seeds, and student dropout RNG was not reset between arms. Because held transfer in the template recipient task changes substantially across seeds and runs, the corruption vs loss design endpoint difference should be treated as a promising signal rather than an isolated effect of adding background loss.

The current repair script, `scripts/paired_factorial_and_update_probe.py`, addresses this by:

- using deterministic epoch/batch mask seeds that are identical for corrupted answer-only and corrupted answer+background arms;
- resetting dropout seeds before each forward pass so stochastic student paths are paired;
- removing duplicated template groups;
- independently cycling query side and source order, so the queried entity is not always first in the source sentence;
- adding a one-step update measurement from a common answer-only trained snapshot: answer-gradient, background-gradient, their cosine, clipping coefficients, and immediate answer/background loss changes after answer-only, background-only, and combined updates.

This repaired experiment was in progress at the time of the note and was designed to determine whether the corruption vs loss design endpoint separation survives pairing. Its one-step update measurement will be more informative than endpoint target counts for the mechanism: it can show whether background gradients oppose the answer update, whether clipping/AdamW changes the effective answer step, and whether the combined update immediately worsens or improves the answer objective.

## Interleave result read narrowly

The Step034b interleave run completed. It installed binding for 200 epochs under corrupted answer-only training, reaching 6/12 held recipient flips. Continuing answer-only for 300 epochs ended at 4/12 held flips. The interleaved answer/background schedule ended at 0/12 held flips, with held U=+1.822 and held R=-1.157. Thus the interleaved schedule especially damaged retain-side transfer in this template task.

This is useful evidence about one mixed schedule, not a proof that separate heads or frozen binding weights are required. The script alternates through the same private-adapter parameters and the same optimizer state, and its actual implementation gives 750 answer steps and 750 background steps rather than preserving the answer-reinforcement dose of the control. Loss of held flips can therefore reflect background pressure, reduced answer reinforcement, optimizer-state interaction, or their combination.

## Multi-token scorer status

The multi-token scorer `scripts/multitoken_scorer.py` was implemented and evaluated on the earlier accepted state-update pilot rows. Outputs are in `data/multitoken_scorer_pilot512/`.

The scorer consumes natural state-update rows with multi-token answers by replacing the candidate span inside the final `use_sentence_frame` with the correct number of mask tokens and scoring each candidate symmetrically. The primary row score is mean per-token log probability for answer versus foil; summed log probability is also saved and is most interpretable when candidate token lengths match. For a pair, the signed recipient-change contrast is U+R, where U is the UPDATE answer-vs-foil margin and R is the RETAIN answer-vs-foil margin under the current signed definitions. Joint UPDATE-and-RETAIN correctness is recorded separately.

Results on the 55 accepted pilot pairs:

- 110/110 rows scored with zero scorer errors.
- Pair structure matched the expected UPDATE/RETAIN swap: zero pair mismatches in source, use frame, entity names, foil/answer swap, or update entity change.
- Coherent86 baseline was strongly asymmetric: UPDATE correct 42/55, RETAIN correct 13/55, joint UPDATE-and-RETAIN correct 3/55 by mean-token margin.
- Mean U=+1.9352, mean R=-1.9542, so mean U+R=-0.0189 despite strong one-sided update preference.
- Equal-token-length subset was small (15 pairs) and still weak: joint correctness 1/15 by summed scores.
- Literal source-position extraction found target earlier 23 pairs, target later 26 pairs, and 6 pairs where simple string matching missed one role; metadata role balance is therefore not identical to literal occurrence position in this pilot.

The scorer is now usable for cleaner state-update packets, but the pilot rows themselves are not ready to serve as strong training evidence. Manual samples show unnatural generated uses and semantic role noise; the baseline readout confirms the same asymmetry seen in simpler repairs: the inherited model often accepts the newly introduced state but fails to preserve the target source state when another entity is updated.

## Practical implication for the BabyLM bridge

A bounded, exposure-matched ALN-preserving pilot from exact coherent86 remained conditional on a cleaner accepted packet set with explicit answer/foil spans and more reliable entity/source-state extraction:

1. ordinary continuation preserving the inherited ALN/filler substrate;
2. ALN plus strict contrastive rows under ordinary WWM;
3. ALN plus strict contrastive rows with relation-answer loss emphasized or separated from background loss;
4. parent anchoring only as a later preservation factor, not bundled into the first experience test.

The template mechanism currently suggests that broad background targets can interfere with relation acquisition in shared private adapters, but background MLM on tiny engineered packets is not the same object as ordinary ALN learning. The real bridge must measure both held strict-packet U/R behavior and cheap7/Entity preservation against matched continuation before deciding whether scheduling, loss separation, or a different parameter path is scientifically justified.
