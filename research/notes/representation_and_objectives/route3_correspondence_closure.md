# route3 correspondence closure Route 3 cross-context signal: correspondence closure

Status: **CLOSED as a coupled learning signal** (forward-only evidence, no training).

## What was tested
The earlier analysis/194 orthogonal-route comparison proposed Route 3: construct target-shared
two-context/two-target property pairs where target priors cancel, then train an
antisymmetric objective so context assigns the plausible alternative correctly.
Before any weight-changing run, per the strategist constraint, I rebuilt Route 3
around **semantically matched** contexts and scored **true correspondence against
matched controls**.

## Objects built (legal pool SHA 215944..., no eval labels)
- Rebuilt semantic pairs: `experiments/archive/representation_and_objectives/data/semantic_route3`
  - `route3_semantic_true_pairs.jsonl` (526 rows: 105 strict same-head/same-category `true_semantic`, 421 cross-category fill)
  - matched controls: `route3_control_target_family_shuffle.jsonl`, `route3_control_family_length_matched.jsonl`, `route3_control_target_swapped_null.jsonl`
- Strict 105-item scoring subset and controls: `experiments/archive/representation_and_objectives/data/semantic_route3_scores/strict_*.jsonl`
- Antisymmetric margin: `delta = [l(a|Ca)-l(b|Ca)] + [l(b|Cb)-l(a|Cb)]`, priors cancel by construction.

## Decisive result (strict 105 same-head/same-category pairs)
Crossed success and paired `true_minus_control` delta on two checkpoints:

| model | true crossed | true delta | shuffle crossed | length-matched crossed | swapped-null crossed |
|---|---|---|---|---|---|
| chck82 scale1.75 | 0.171 | 0.386 | 0.238 | 0.267 | 0.029 |
| legal16k_base100 | 0.114 | 0.359 | 0.181 | 0.143 | 0.095 |

Paired `true_minus_control` antisymmetric-margin delta (positive would mean true
correspondence is stronger):

- chck82: target_family_shuffle **-0.251** (pos_frac 0.352); family_length_matched **-0.226** (pos_frac 0.41); target_swapped_null +0.773.
- legal16k_base100: target_family_shuffle **-0.175** (pos_frac 0.41); family_length_matched **-0.324** (pos_frac 0.381); target_swapped_null +0.718.

## Interpretation
1. The metric is valid: the target-swapped null inverts strongly (+0.72 to +0.77), so
   the antisymmetric score responds correctly to swapping the correct/foil assignment.
2. True pair correspondence gives **no** advantage over correspondence-destroying
   controls; it is slightly *worse*. The antisymmetric margin is produced by each
   context independently preferring its own corpus-attested property, not by any
   coupling between the two contexts and two targets.
3. This is the same failure mode already established for sparse20 coupled dual-view
   (related experiments, ~95.5% of "repair" reproduced by shuffled correspondence) and for
   the edit-state correspondence route on chck82. A target-shared two-context object
   built from independent corpus sentences does not carry a learnable coupled signal;
   it decomposes into two independent within-context lexical-fit decisions.

## Route status
- Route 3 (corpus-derived target-shared cross-context signal, as an antisymmetric
  coupled training objective) is **closed**. Do not launch the proposed 4M replay:
  the object has no correspondence structure beyond target/family/length balance, so
  an antisymmetric loss on it would supervise the already-solved within-context fit.
- Route 1 (procedural data) was already closed at orthogonal route screen summary (only ~19 clean prose
  passages in the legal pool).
- Route 2 (representation vs readout) remains the single genuinely orthogonal open
  lever from the earlier analysis comparison and must be tested with a decoder-robust probe
  (held-out base/entity/template splits, renamed/swapped/permutation nulls) so it
  does not repeat the decoder alignment route boundary raw mid-layer decodability that was not a protected
  solution.

## Protected endpoints (unchanged)
- Practical above-frontier candidate: coherent86 alpha0.5, Overall(AoA0) ~42.115.
- Protected fallback: scale1.75 chck_82M, Overall 41.942481167385985, fully packaged.
- No training, no endpoint modification.

## Files
- Build script: `experiments/archive/representation_and_objectives/scripts/build_semantic_route3_pairs.py`
- Score script: `experiments/archive/representation_and_objectives/scripts/score_semantic_route3_controls.py`
- Scores: `experiments/archive/representation_and_objectives/data/semantic_route3_scores/semantic_route3_control_scores.json`
