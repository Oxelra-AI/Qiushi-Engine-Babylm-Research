# equivariance control and state substrate repair — What the repaired measurements actually established

## Scientific Motivation existed

equivariance protocol reported all-chance GRU/BiGRU numbers on a cross-predicate role task and
interpreted them as evidence that cross-predicate role transfer requires prior
(pretrained) predicate semantics. That interpretation was not supported: an
experiment whose model does not even fit its own training set has not measured
the scientific question. equivariance protocol also produced a ranking-attested "event-to-state"
pilot whose independence had not been tested against an event-outcome parser.

Both measurements were repaired before a larger experiment was considered.

## Result 1 — equivariance protocol was an unmeasured experiment, not a negative result

Script: `scripts/equivariance_truth_gradient_and_composition.py`
Data: `data/equivariance_diagnostics/summary.json`

Symbolic truth table (legacy equivariance protocol-style full 8x8 predicate matrix, 20 families):

- 5,120 rows, true-label fraction 0.500
- all 64 cp x hp cells present and label-balanced
- 0 duplicate-text label conflicts

Gradient/memorization path on a balanced 64-row slice:

- first-step gradient norm 0.6441, first-step parameter delta (L2) 2.2817,
  52,129 nonzero gradient entries
- initial acc/loss 0.500 / 0.7059 -> final 1.000 / 0.0138, reaching >=0.999 at
  epoch 30

So the labels are coherent, the gradient path is live, and the architecture can
memorize the task family. **equivariance protocol's all-chance numbers reflect a symmetric
optimization plateau on that task format, not a statement about predicate
semantics.** The claim "cross-predicate transfer requires pretrained predicate
knowledge" has no support from that run and must not be carried forward.

## Result 2 — The equivariance advantage is symmetry-breaking, not role learning

The repaired composition task gives every one of the 8 predicates independent
semantic evidence through role-word support rows ("ENTITY_A was the winner/loser"),
then holds out 16 of 64 cp x hp *combinations*. Each predicate appears in both
training and held cells, so held cells are new compositions rather than unseen
words. Train rows 6,912 = 2,304 support + 4,608 composition; 11,520 equivalence
pairs; held source domain = football (24 train / 12 held / 12 held-domain families).

First reading (12 epochs, seeds 254/255/256) looked like a strong positive:

| surface | standard | equivariant | delta |
|---|---:|---:|---:|
| semantic_support_held_family | 0.667±0.236 | 0.833±0.236 | +0.167 |
| train_pair_held_family | 0.664±0.238 | 0.836±0.231 | +0.172 |
| held_pair_seen_family | 0.669±0.234 | 0.822±0.228 | +0.153 |
| held_pair_held_family | 0.668±0.235 | 0.819±0.229 | +0.151 |
| held_pair_held_domain | 0.664±0.238 | 0.828±0.232 | +0.164 |

But the per-seed values are bimodal — each run lands near 0.50 or near 1.0, never
in between. That is a plateau-escape pattern, not a graded competence. Two controls
settled it.

Targeted convergence (`data/targeted_convergence_probe/`): the
failed standard seeds do not simply need more time. standard/254 and standard/256
stay at train acc 0.500 through 24 full epochs (loss pinned at ~0.6932, i.e. ln 2),
while equivariant/254 breaks out at epoch 14 to train 1.000 / held-composition 0.984.

Shuffled-pair control (`data/shuffled_pair_control/`), identical
data, budget, and lambda=0.35, seeds 254+256:

| mode | pairs | final train | held_pair_held_family | held_pair_held_domain |
|---|---:|---:|---:|---:|
| standard | 0 | 0.500 | 0.500±0.000 | 0.500±0.000 |
| equivariant (true role-equivalence) | 11,520 | 0.752 | 0.752±0.248 | 0.750±0.250 |
| shuffled_label (random, label-matched) | 11,520 | 0.751 | 0.743±0.242 | 0.734±0.250 |
| shuffled_any (fully random pairs) | 11,520 | 0.750 | 0.750±0.250 | 0.750±0.250 |
| same_cp_control (deliberately WRONG invariance) | 11,520 | 0.929 | 0.934±0.066 | 0.941±0.059 |

Random pairing matches true role-equivalence pairing, and the deliberately wrong
invariance — matching rows that share a context predicate but differ in world and
query, which should destroy needed distinctions — scores highest. **The auxiliary
matching loss is acting as generic gradient perturbation that breaks a symmetric
initialization, not as a role-specific learning signal.**

A second design flaw is now explicit: once the training predicates are learned,
held cp x hp *cells* are reached at 0.97-1.00 essentially for free, because the
model already has both predicates' role directions from training composition. That
surface cannot discriminate learning objectives at all. Any future equivariance
claim must use predicates that never appear in composition training.

## Result 2b — On the decisive surface, equivariance is not just null, it is worse

Script: `scripts/held_predicate_composition.py`
Data: `data/held_predicate_composition/held_predicate_composition_summary.json`

Design fix: four predicates (`overcame`, `fell_to`, `prevailed`, `was_defeated`,
i.e. 2 subject=winner + 2 subject=loser) appear **only** in role-word support rows
("ENTITY_A was the winner/loser") and never in any composition training row. The
seen predicates are `defeated`, `beat`, `lost_to`, `was_beaten`. Train rows 3,840 =
2,304 support + 1,536 composition over 16 seen cells; five evaluation surfaces of
768 rows each, all exactly label-balanced with 0 duplicate-text conflicts. Model
selection uses **train fit only**, so no held surface influences the checkpoint,
and aggregation is converged-only (train acc >= 0.95).

| mode | runs | converged | seen_comp_held_family | heldpred_hyp_only | heldpred_ctx_only | heldpred_both | heldpred_both_held_domain |
|---|---:|---:|---:|---:|---:|---:|---:|
| standard | 3 | 3 | 1.000 | 0.927 | 1.000 | **0.938** | **0.938** |
| equivariant | 3 | 2 | 1.000 | 0.958 | 0.900 | **0.847** | 0.852 |
| shuffled_any | 3 | 3 | 1.000 | 0.964 | 0.984 | 0.922 | 0.924 |

Plain cross-entropy is the best of the three on the hardest surface, converges on
3/3 seeds, and reaches 1.000 on held-predicate-in-context. The equivariance
objective converges on only 2/3 seeds and is ~9 points worse on `heldpred_both`.
Random pairing again sits between them. **The representation-matching objective
provides no compositional transfer benefit here; on the one surface that can
actually discriminate objectives, it is mildly harmful.** Combined with Result 2,
the equivariance-objective hypothesis for this task family is closed.

## Result 2c — The real positive: role-word anchoring makes a predicate composable

The same run contains an unanticipated and more interesting finding. The four
held-composition predicates were never shown in any context-plus-hypothesis
composition row; their only supervision was flat role-word support. Yet ordinary
CE composes them:

- held predicate in context position, seen predicate in hypothesis: **1.000**
- held predicate in hypothesis position: 0.927
- held predicate in **both** positions, held families: **0.938**
- same, on the held source domain (football, never in training): **0.938**

So a predicate whose meaning was acquired only through a cheap, flat
role-word anchor ("X was the winner") transfers into both argument positions of a
compositional judgment, on new entity pairs and a new source domain, with no
composition examples for that predicate and no auxiliary objective. The learnable
unit is the mapping *predicate -> which role its grammatical subject bears*; once
that mapping is anchored, composition is free.

This is a candidate data-efficiency statement worth pursuing: role-word anchoring
supervision may be a much cheaper substitute for combinatorial composition
coverage. Its current standing is strictly bounded — a 29-word vocabulary, canonical
`ENTITY_A`/`ENTITY_B` names, one templated relation family, and a tiny BiGRU. It is
a hypothesis generated by a controlled probe, not an established principle. What
would raise it: natural entity names and distractor entities; predicates whose role
mapping is not inferable from the two-role template; multiple relation families
where the anchor words differ; and whether the same anchoring supervision moves the
natural EWoK/Entity conditional-reversal items rather than only synthetic surfaces.

## Result 3 — The equivariance protocol ranking pilot is not an event-to-state benchmark

Script: `scripts/ranking_eventonly_bias_probe.py`
Data: `data/ranking_eventonly_bias_probe/ranking_eventonly_bias_summary.json`

Over all 3,600 pilot NLI rows (2,880 train / 720 held):

- event_text contains 0 ranking numbers (the earlier leak check was correct)
- but the transparent parser "the match winner is the higher-ranked participant"
  scores **1.000** on train, **1.000** on held, and 1.000 on each of
  rank_higher / rank_better / rank_above

Cause: the `both_ranking_consistent` filter retained only families where the match
winner was also the better-ranked player in *both* worlds. The independently
sourced ranking label therefore equals the event outcome exactly. The pilot has
genuine independent attestation (95.05% cross-verification of match-embedded ranks
against the separate weekly ranking file) but **zero event/state dissociation**, so
it cannot test whether a learner binds a recorded state to an entity. No student
should be trained on it.

## Result 4 — A substrate where state genuinely dissociates from outcome

Script: `scripts/upset_balanced_state_substrate.py`
Data: `data/upset_balanced_state_substrate/substrate_summary.json`

From 74,906 ATP matches (17,033 ranked players) and 24,126 reversed-pair
candidates, each match is classified against the independent weekly ranking file
as `consistent` (winner ranked better) or `upset` (winner ranked worse), and
families are drawn in equal CC/CU/UC/UU strata:

- 880 families, 220 per stratum, 660 train / 220 held
- fraction of worlds where the match winner is the higher-ranked participant: **0.500**
- transparent winner->higher-rank parser: **0.5003** on `state_at_time`
  (was 1.000 in the equivariance protocol pilot)
- label balance exact: event_role 1759/1761, state_at_time 1760/1760,
  state_later 1760/1760
- event text contains no rank numbers; state text contains no event verbs
- a separate ranking snapshot ~26 weeks later supplies attested temporal state:
  565 families with 0 changed worlds, 274 with 1, 41 with 2
- untouched-fact rows assert invariant participant facts across both worlds

Residual to handle before use: `state_later` still gives the outcome parser
0.5628, so the later-snapshot family retains a modest correlation with who won.
The later-state surface must be either upset-balanced in its own right or reported
separately from `state_at_time`.

## What this changes for the research line

1. The prior step's conclusion about needing pretrained predicate semantics is
   withdrawn — it rested on an unmeasured run. Result 2c is its direct refutation:
   a randomly initialized learner acquires a usable predicate role-mapping from
   flat role-word supervision and composes it on held predicates, families, and a
   held source domain.
2. The equivariance-objective hypothesis is closed for this task family. It failed
   two independent ways: shuffled and wrong-invariance pairing reproduce or exceed
   its apparent benefit (Result 2), and on the only surface that can discriminate
   objectives it is worse than plain CE with fewer converged seeds (Result 2b).
   Any future such claim requires shuffled and wrong-invariance controls at matched
   budget, train-fit-only model selection, converged-only aggregation, and
   predicates held out of composition entirely. This is the same discipline that
   overturned the edit route closed-182 shared-readout alignment result and the earlier analysis-243
   RoBERTa factorial, and it has now overturned a third apparent positive.
2b. The surviving candidate is role-word anchoring (Result 2c), not an auxiliary
   objective: cheap flat role supervision appears to buy compositional reach that
   combinatorial coverage would otherwise have to purchase. This is the line worth
   pressing, under the boundaries listed in Result 2c.
3. The event-to-state route is alive but only with the upset-balanced substrate.
   The paired world pilot result/252 direct-outcome corpus and the equivariance protocol ranking pilot are
   construction scaffolds; neither can carry a general claim.
4. Still unmet before any student, TinyMLM, DeBERTa, RoBERTa, or BabyLM training:
   a learner must show, on the upset-balanced substrate, that it binds the
   independently recorded state to the correct entity under interference, preserves
   untouched facts, respects temporal overwrite (latest snapshot wins), and moves
   the natural EWoK/Entity items in
   `data/ewok_natural_bridge_panel/ewok_bridge_panel.jsonl`
   (7,618 items, 1,756 relational).

## Closed here

- Held cp x hp cell generalization as an objective-discriminating surface (trivial).
- Auxiliary representation matching interpreted from aggregate accuracy without a
  shuffled-pair control.
- The equivariance/representation-matching objective as a route to compositional
  role transfer in this task family: null under controls and worse than plain CE on
  the decisive held-predicate surface. Do not reopen without a genuinely new
  premise, and never on the trivial held-cell surface.
- The equivariance protocol ranking pilot as an event-to-state training corpus.
- The equivariance protocol claim that cross-predicate transfer requires pretrained predicate
  semantics.

## Next work, in priority order

1. Press Result 2c where it can break: natural entity names plus distractor
   entities, predicates whose role mapping is not recoverable from the two-role
   template, and several relation families with different anchor words. The
   question is whether role-word anchoring is a general substitute for
   composition coverage or an artifact of a two-entity canonical template.
2. Use the upset-balanced substrate (Result 4) for the binding test that the
   equivariance protocol pilot could not support: does a learner bind the independently recorded
   state to the correct entity when the event outcome carries no information about
   it, preserve untouched facts, and respect latest-snapshot temporal overwrite?
   Report `state_at_time` and `state_later` separately given the 0.5628 residual.
3. Only if 1 and 2 both hold, connect to the natural EWoK/Entity panel
   (`data/ewok_natural_bridge_panel/ewok_bridge_panel.jsonl`,
   7,618 items / 1,756 relational) before any BabyLM-scale training is considered.

## Artifacts

- `scripts/equivariance_truth_gradient_and_composition.py`
- `scripts/targeted_convergence_probe.py`
- `scripts/shuffled_pair_control.py`
- `scripts/ranking_eventonly_bias_probe.py`
- `scripts/held_predicate_composition.py`
- `scripts/upset_balanced_state_substrate.py`
- `data/equivariance_diagnostics/`
- `data/targeted_convergence_probe/`
- `data/shuffled_pair_control/`
- `data/ranking_eventonly_bias_probe/`
- `data/upset_balanced_state_substrate/`
