# pair level relation robustness principle center after term decomposition and peer correction

## What changed

pair level relation robustness moved the center of the finite-experience principle away from a VIEW-first story and toward a relation-specific installed-computation story. The strongest current cross-architecture phenomenon is the active cost of exact in-window recurrence for nonidentical content use.

Let `T_m` be masked NLL for a target rewrite token with the true source present, `U_m` be masked NLL with an unrelated length-matched source, and `G_m = U_m - T_m`. For an arm contrast A-B, the pair level relation robustness `excess_true_cost = (T_A - T_B) - (U_A - U_B) = -[(U_A-T_A)-(U_B-T_B)] = - delta gain`. It is not independent from the gain contrast. Its value is that the term split shows whether the arm difference comes from the true-source side or from the unrelated-source side.

## Load-bearing measurement now

On tokenizer-nonoverlap rewrite tokens, REPEAT is worse than CLEAN in true-source-conditioned use in the two DeBERTa seeds and in the RoBERTa 60M-100M average:

- DeBERTa seed43022 R-C: gain delta -0.7517; true-source NLL delta +0.4480; unrelated-source delta -0.3037; excess true-source cost +0.7517.
- DeBERTa seed43122 R-C: gain delta -1.0433; true-source NLL delta +0.6931; unrelated-source delta -0.3502; excess true-source cost +1.0433.
- RoBERTa seed43022 R-C: gain delta -0.3983; true-source NLL delta +0.4924; unrelated-source delta +0.0942; excess true-source cost +0.3983.

The DeBERTa sign reversal is especially informative: REPEAT is better than CLEAN on the unrelated-source term but worse on the true-source term. That pattern is hard to explain as a global arm-quality deficit. RoBERTa is less isolating because REPEAT is worse on both terms, but the true-source cost is much larger than the unrelated-source cost and the conditioning-gain cost is present at every measured checkpoint.

This result survives two checks:

- Pair-level averaging: after averaging tokenizer-nonoverlap target rows inside each held-out compact rewrite pair and across checkpoints, R-C excess true-source cost remains positive with median +0.607/+0.878/+0.254 and positive-pair fractions 0.661/0.701/0.659 for DeBERTa43022, DeBERTa43122, and RoBERTa43022.
- Strict word-level target filtering: requiring the whole normalized target word to be absent from the normalized source keeps R-C gain negative and excess true-source cost positive: DeBERTa43022 +0.5883, DeBERTa43122 +0.8296, RoBERTa +0.2515.

Thus the current principle should be phrased as a distributional tendency: finite experience with exact recurrence can install a context-use policy that helps some exact or unchanged cases but degrades use of related nonidentical source evidence on many held-out examples. It is not an every-example law.

## VIEW-side positive result

The VIEW positive half is real for DeBERTa and weaker for RoBERTa.

- DeBERTa V-C tokenizer-nonoverlap gain is +0.6837 and +0.8938 across the two seeds; true-source advantages (-1.1788, -1.3544) exceed unrelated-source advantages (-0.4951, -0.4606). Strict word-level filtering preserves positive V-C gains (+0.5400, +0.7349). Pair-level medians are positive.
- RoBERTa V-C has a small tokenizer-nonoverlap residual (+0.0666), and strict word-level filtering gives +0.0436 with sign variation at 60M. Its true-source improvement (-0.5116) is close to its unrelated-source improvement (-0.4450), so most of the arm difference looks like broad compact-rewrite fit, with only a small residual source-use benefit.

Therefore the architecture-spanning center is not VIEW helping Entity. It is the relation-specific cost of exact recurrence for nonidentical content use. VIEW's stronger positive conversion remains an important DeBERTa result and a hypothesis for further tests.

## Natural copy after normalization

DeBERTa still shows a resolved natural-copy advantage for REPEAT after normalizing each arm by its own unrepeated-control NLL: R-C raw gain deltas are +0.4996/+0.6678 and normalized gain deltas are +0.0497/+0.0531. RoBERTa's copy contrast is near zero with sign changes across checkpoints: raw R-C mean -0.0487, normalized mean -0.0272, raw checkpoint values +0.1737, +0.0835, -0.1133, -0.1895, -0.1980 from 60M to 100M. This should be read as unresolved in that learner and loss regime, not as evidence for an opposite copy tendency.

## How functional_learning now connects

The controlled coordinate-label intervention showed that wrong h1-incident labels selectively install a competing h1 coordinate below the full graph: h1 held rows go from accuracy 1.0 to 0.0 and target-signed margin +11.94 to -20.80, while non-h1 edges remain correct. The later analysis corrected the comp-masked result after a regex bug: when direct relation evidence is hidden, exact or locked visible cues can fit training comparisons while hurting no-cue or wrong-cue h1; randomizing the cue does not restore the missing relation. This aligns with the pair level relation robustness principle at the level of form: practiced visible relations can install targeted computations below the cleaner condition. It is still supervised event scoring, not token-NLL source-conditioned content use, so it should guide synthetic bridge design rather than be treated as the same result.

## Remaining assumptions that can still change the interpretation

The current data do not yet prove that pairwise identity alignment alone causes the cost. Alternatives still to separate include duplicated-token budget, companion coherence, target or near-target leakage in the unrelated source, source/rewrite register, morphology and named-entity proximity, and scoring with single masked tokens inside an otherwise visible rewrite.

The next most decisive existing-checkpoint computation is a source panel for the same strict word-level targets: no source, true source, unrelated target-absent source, shuffled true source, and lexical-decoy source, ideally with full target words jointly masked. This will separate true content use from unrelated-context sensitivity, lexical attraction, and broad fit. The second decisive direction is a token-multiset-matched training intervention: exact same source tokens paired with themselves versus a deranged source of identical token inventory, and true rewrite paired with its source versus deranged rewrite-source pairs.

## Files produced in pair level relation robustness

- Term decomposition: `research/notes/relation_learning/relation_decomposition_principle_update.md`; data in `experiments/archive/relation_learning/data/relation_decomposition`.
- Pair-level robustness: `research/notes/relation_learning/pair_level_relation_robustness.md`.
- Strict word-level sensitivity: `research/notes/relation_learning/strict_word_nonoverlap_sensitivity.md`.
- independent_review scientific check: `data/external/independent_review01_verifier1_integration.md`.
