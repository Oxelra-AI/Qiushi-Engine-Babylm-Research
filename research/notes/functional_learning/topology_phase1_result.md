# topology phase1 result topology intervention result: correct relational content, not exposure alone, controls coordinate transport

## Purpose

This note records the completed functional_learning topology phase1 result execution on the inherited representation_and_objectives controlled gauge-transport harness. The scientific target was to separate three possible information routes in finite-supervision data-efficient learning:

1. **Representation formation** by comparison-trained shared features.
2. **Correct relational/content constraints** supplied by comparison edges.
3. **Readout calibration** from sparse h0/h2 absolute anchors to h1/h3 unanchored state decisions.

The experiment uses the dual-position learned-equality raw-text harness: finite alphabet equality pretraining first makes string identity usable, then relational training learns state and comparison tasks. All new topology phase1 result outputs are written under `experiments/archive/functional_learning/data`.

## Scripts and outputs

- Training/intervention script: `experiments/archive/functional_learning/scripts/topology_probe.py`
- Pair analysis script: `experiments/archive/functional_learning/scripts/analyze_topology_pairs.py`
- Integrated pair analysis: `research/documents/functional_learning/data/topology_pair_analysis_all_e60/topology_pair_analysis.md`
- Integrated JSON: `experiments/archive/functional_learning/data/topology_pair_analysis_all_e60/topology_pair_analysis.json`

Completed 60-epoch pairs (seed 30000, full alphabet equality pretraining, shared_trunk, CPU because the tiny harness was slower/unstable on H100 kernels):

- `full_graph_e60`: original comparison graph, control.
- `detach_full_graph_e60`: original graph; changed-state loss uses `h.detach()` before the state head, so the event trunk receives no state-loss gradient.
- `disconn_delete_e60`: remove h0↔h1 and h1↔h2 training comparison rows.
- `disconn_adversarial_e60`: invert h0↔h1 and h1↔h2 training comparison labels; h0↔h2 and h2↔h3 remain correct.

Earlier four-way 180-epoch CPU attempts were killed by timeout around epoch 90 before writing final artifacts; those are operationally informative but not scientific result files. The completed e60 cells are the current evidence.

## Core results

### 1. Full-graph control reproduces the changed-state transport fingerprint

At 60 relational epochs, full_graph fits state and comparison training (`train_state=1.0`, `train_cmp_modified=1.0`) and reproduces the central changed-state sign intervention:

| cell | direct_same | graph_same | hh_closure | mixed_acc | train_state | train_cmp |
|---|---:|---:|---:|---:|---:|---:|
| full_graph bs+1 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| full_graph bs-1 | 0.0 | 0.0 | 1.0 | 0.0 | 1.0 | 1.0 |

Per-relation changed-state accuracy is 1.0 for all h0/h1/h2/h3 under bs+1 and 0.0 for all under bs-1, with row-paired opposite sign fraction 1.0 for each relation. This confirms the topology phase1 result code reproduces the comparison-mediated changed-state gauge effect. Static/unchanged eval remains 0.5 in this e60 run, so this specific control should be used for changed-state topology conclusions, not as a full reproduction of the longer earlier analysis unchanged/static condition.

### 2. DETACH verifies gradient separation but underfits comparison, so it does not decide readout calibration

The DETACH pair mechanically works:

- direct one-example gradient check gives zero event-trunk gradient from detached state loss in both signs;
- event-trunk hashes are identical after full training across bs+1 and bs-1 (`e9b18c26b6f63f8d`);
- event-state head hashes differ across signs, as expected.

However the comparison channel does not fit by e60 (`train_cmp_modified=0.443` in both signs; held-held closure 0.469). State readout remains near chance (`graph_same=0.5/0.5`; per-relation h1/h2/h3 at 0.5; h0 direct also not meaningfully learned). Therefore DETACH is **underfit comparison-only evidence**, not a strong refutation of readout calibration. It says that, with this optimizer/seed/budget, comparison gradients alone did not install a usable trunk coordinate quickly enough for the state head.

### 3. DELETE is not a clean topology test because the h3 control fails

DELETE fits the modified training objective (`train_state=1.0`, `train_cmp_modified=1.0`), and direct anchors still flip correctly. But h3, which was intended as the correctly connected control through h2↔h3, is wrong in both bridge signs (`h3_ziv` accuracy 0.0/0.0). Thus row removal changes the learned coordinate/global solution too much for h3 to serve as a clean within-run control. DELETE by itself cannot distinguish missing h1 exposure from graph-content necessity.

### 4. ADVERSARIAL is the cleanest content test: wrong h1 constraints actively misorient h1 while h3 remains correct

The adversarial pair inverts labels only on h1-incident training comparison edges h0↔h1 and h1↔h2 while keeping h0↔h2 and h2↔h3 correct. It fits the modified labels (`train_cmp_modified=1.0`) while scoring only 0.5 against original labels, showing it learned the altered graph rather than ignoring the intervention.

Per-relation changed-state accuracy:

| relation | bs+1 acc | bs-1 acc | paired opposite-sign frac | interpretation |
|---|---:|---:|---:|---|
| h0_dax | 1.0 | 0.0 | 1.0 | sparse anchor sign preserved |
| h2_norp | 1.0 | 0.0 | 1.0 | sparse anchor sign preserved |
| h3_ziv | 1.0 | 0.0 | 1.0 | correctly connected h2↔h3 path preserves transport |
| h1_mep | 0.0 | 1.0 | 1.0 | inverted h1 constraints install the opposite state orientation |

This is the strongest topology phase1 result result. Because h1 has the same exposure count as in FULL but its incident labels are inverted, and because h3 remains correctly oriented under the still-correct h2↔h3 edge, the result supports a content-specific causal account: finite comparison experience helps when it supplies correct relational constraints; wrong constraints can install a systematically wrong reusable coordinate rather than merely add noise or fail to help.

## Current scientific interpretation

The inherited coordinate-transport mechanism should now be stated more sharply:

- finite primitive equality evidence makes raw strings usable as stable identity coordinates;
- sparse absolute state anchors choose an orientation for anchored relation components;
- correct comparison constraints bind relation components into a transportable coordinate;
- incorrect relational constraints can actively transport the wrong orientation;
- exposure alone is insufficient to explain the transport, because h1 exposure is preserved in ADVERSARIAL but h1 flips to the wrong orientation while h3 remains correct.

This is still a controlled synthetic mechanism, not a proof of natural-language generalization. It is nevertheless directly relevant to the general data-efficient learning principle: data are efficient when they add **correct reusable constraints** that enter shared target-usable coordinates. Repeated exposure without new correct constraints, or exposure carrying corrupted constraints, should not be expected to improve the same competence and can damage it.

## Relation to relation_learning mechanism macro convergence

The BabyLM relation-learning comparison found that VIEW-CLEAN effects are benchmark-specific across seeds: Entity and Reading are seed-stable positive, COMPS is negative/weak, while Supplement/BLiMP/EWoK are not seed-stable enough for broad aggregate claims. This fits the coordinate-transport result at the level of mechanism class only: identity-under-variation or rewrite evidence may help content-tracking competences if it supplies correct invariant constraints, but may not help form-sensitive or compositional comparison tasks and can hurt if the rewriting/compression process removes the relevant relation. Coordinate transport does not prove the VIEW mechanism; the planned relation-learning invariance probe is needed for that.

## Strong next experimental move

A proposed synthetic contrast is to construct a matched finite-budget comparison of **exact repetition** versus **varied renderings** of the same underlying semantic events/relations. The design should keep labels, number of gradient steps/examples, names, relation graph, and equality evidence matched while varying whether evidence is repeated verbatim or shown under multiple surface renderings. The prediction to test is not a generic VIEW advantage: exact repetition should help shallow/direct seen-event readout, while varied renderings should help held-template h1/h3 transport and state-update generalization if the variants preserve the same underlying relational constraints. This prediction depends on a validated matched construction.
