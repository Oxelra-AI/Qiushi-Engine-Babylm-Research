# coordinate connected experience route — From logical disambiguation to coordinate-connected experience

## Current evidence that changes the route

The earlier analysis merged, file-backed analysis is now complete:

- merged outputs: `experiments/archive/representation_and_objectives/data/fast_balanced_probe_v2_merged`
- cell analysis: `research/documents/representation_and_objectives/data/fast_balanced_probe_v2_analysis/balanced_probe_analysis_summary.md`
- exact candidate-choice analysis: `research/documents/representation_and_objectives/data/fast_balanced_probe_v2_exact_choice/exact_choice_metrics_summary.md`

The important learned result is not a small null; it separates two forms of identifiability.

1. `replace_k16_spread` removed the anti-copy equivalence from the data: in the substrate, anti-copy and copy-initial are both wrong on half the changed-object rows while the event-role rule remains correct.
2. The pretrained DeBERTa encoder plus learned classification head nevertheless fit all supervised rows perfectly but did not reuse the disambiguating evidence on held names/objects.
3. Exact-choice metrics show the model still behaves almost exactly like an anti-copy rule on held state queries:
   - `replace_k16_spread/aligned_state_bridge`: exact changed same-initial `0.010`, opposite-initial `0.990`; exact pair-both same `0.010`, opposite `0.987`.
   - `replace_k16_spread/inverted_state_bridge`: exact changed same-initial `0.010`, opposite-initial `0.977`; exact pair-both same `0.010`, opposite `0.974`.
4. The effect is not repaired by direct anchor relation exposure: true-row changed accuracy is around chance for direct-anchor and graph-transfer relations (`aligned`: direct `0.492`, graph `0.503`; `inverted`: direct `0.406`, graph `0.487`).
5. Signed mixed-relation orientation is absent: aligned minus inverted mixed true-statement accuracy is `-0.016±0.050` for k16.
6. Static/unaffected facts are preserved, so the failure is specifically changed-object event-role updating and coordinate propagation, not global inability to answer ownership statements.

Thus earlier analysis does **not** say that counterexamples to a shortcut are useless. It says that logical disambiguation inside a finite training set is not enough when the learner can fit the informative cases locally and keep a lower-complexity rule off-support. The next experiment should not add more same-initial worlds or make another surface variation. It should test whether disambiguating evidence becomes reusable when it is connected to a shared interface that the learner already uses.

## Inherited positive structure that must be connected

Three earlier results remain important and should be united rather than replaced.

### role coordinate anchor and state probe: absolute-coordinate anchoring

The factorized role model established the algebraic issue: held-held consistency can leave a global sign/permutation ambiguity. Sparse true mixed evidence or an absolute reference identifies the mapping to established roles. Shuffled anchors can fit internally but fail mixed held-seen transfer. The important readout is mixed held-seen behavior, not held-held consistency alone.

### orbit pair audit and slot reuse update: reusable fillers and relation-slot exposure

The matched slot-reuse refinement showed that ordinary token familiarity helps but is insufficient. With matched row count and balanced order:

- pure identity/neutral exposure raised baselines but did not remove orientation effects;
- random relation co-occurrence did not fit or orient;
- true relation-level held-pool exposure gave the strongest positive orientation (`true - anti ≈ +0.505`);
- complemented relation-level exposure reversed the held-pool polarity.

This suggests that data-efficient transfer needs fillers to participate in a reusable argument-slot type system, not merely appear as familiar tokens.

### equivariant symmetry repair and macro context: the surface contains the right formal algebra when a slot interface is supplied

The repaired active/passive nonce substrate has no cheap order/BoW solution on the key surfaces. A transparent slot-orbit parser over the actual text selects the intended assignment for aligned bridge arms and the opposite assignment for inverted bridge arms. Therefore the text construction can express the law; the open problem is whether a finite learner can induce and reuse the interface.

## Candidate principle now worth attacking

**Coordinate-connected experience.** Limited experience supports reusable compositional knowledge only when it is both:

1. **Discriminative**: the examples distinguish the intended structure from lower-complexity alternatives on the relevant variables; and
2. **Coordinate-connected**: the distinguishing examples lie on a connected path through the learner's reusable interface from established coordinates and filler slots to the held target behavior.

In graph language, let nodes include relation-role coordinates, argument-slot roles, filler identities/types, task formats, and state variables. Training examples are labelled constraints linking subsets of these nodes. A target is not merely logically identifiable from the labelled graph; it is learner-usably identifiable when the constraint path to the target is represented through shared nodes that the model actually reuses. role coordinate anchor and state probe shows logical coordinate ambiguity; earlier analysis shows that breaking a shortcut in isolated state rows can still fail learner-usable identifiability; orbit pair audit and slot reuse update suggests that filler-slot connectivity can make sparse orientation evidence usable.

This principle is general enough to matter beyond BabyLM: it predicts when finite data teaches reusable roles rather than local facts, why balanced counterexamples may fail, why token familiarity alone is weak, and why architecture may need explicit factorization when data does not create a usable interface.

## Next experiment: connected versus disconnected disambiguating paths

The next substrate should keep the equivariant symmetry repair and macro context role vocabulary, active/passive templates, label information, number of rows, relation counts, initial-owner patterns, and token exposure as fixed as possible. The manipulated variable is whether the same disambiguating worlds form a connected path through established relation roles and reusable fillers to held targets.

### Core arms

Use the same relation algebra as equivariant symmetry repair and macro context:

- seen established roles: `s_give` final slot 1 and `s_receive` final slot 0;
- held relations: `h0_dax,h1_mep` true slot 1 and `h2_norp,h3_ziv` true slot 0;
- held-held graph provides relative orientation among held relations;
- sparse state bridge uses `h0_dax` and `h2_norp` as anchor relations, with same/opposite initial ownership balanced as in `replace_k16_spread`.

Build at least these conditions:

1. **connected_aligned** — same information as k16, but participant pairs/objects are arranged so that state disambiguators, seen-coordinate state rows, and held-held comparison edges share reusable filler nodes. The intended path is: established seen role -> shared filler/slot state row -> held anchor relation -> held-held comparison graph -> held target relation -> mixed held-seen and state targets. Labels use the true held assignment.
2. **connected_inverted** — identical exposure/connectivity but state bridge labels use the inverted held assignment. A real coordinate path should reverse signed mixed orientation while preserving unchanged/static facts.
3. **disconnected_aligned** — same row types, templates, relation counts, label counts, initial patterns, and token frequencies, but filler pair/object assignments are permuted independently across seen-coordinate rows, held-held comparisons, bridge state rows, and target readouts so no reusable filler path connects the disambiguating evidence to the held target.
4. **disconnected_inverted** — disconnected counterpart with inverted labels.
5. **token_only_or_neutral_connector** if cheap: same filler/token reuse without relation-level labelled connectivity, to separate ordinary token familiarity from role-slot interface connectivity.

The connected/disconnected contrast should not add more supervision in the connected condition. If the same sentences can be produced and only reassigned to names/objects, that is strongest: information, wording, and dose remain fixed; only the graph of co-reference/reusable filler constraints changes.

### Evaluation readouts

Use per-row logits and exact candidate choice, not only true-row accuracy.

1. **Joint state conservation:** exact changed choice and exact pair-both by `initial_pattern × static_slot × relation × voice`. Success requires high same- and opposite-initial changed correctness while unchanged facts remain high.
2. **Direct-anchor versus graph-transfer:** separate `h0/h2` direct bridge relations from `h1/h3` relations reached only through held-held edges.
3. **Signed mixed held-seen orientation:** connected aligned should move true mixed statements in the true direction; connected inverted should move in the opposite direction. Disconnected arms should not show the same selective signed transfer.
4. **Filler boundary:** include two target suites if possible: (a) connected filler pool with novel object/pair combinations, and (b) new disjoint filler pool. This turns success/failure into a condition of the principle rather than an ambiguous all-or-nothing result.
5. **Shortcut probes:** retain anti-copy/copy-initial/static-slot-switch baselines; add relation-blind filler-pair memorization baselines. A model should not pass only on repeated pair/object facts.

### Minimum-cost sequence before GPU

1. CPU construction only: generate the connected/disconnected files and a manifest reporting row counts, token/name/object frequencies, relation/voice/static/initial-pattern balance, and graph connectivity components over relation nodes and filler-pair/object nodes.
2. Run the existing transparent checks adapted from equivariant symmetry repair and macro context/284: order/BoW baselines, anti-copy/copy-initial/static-slot rules, and slot-orbit parser. The slot-orbit parser should formally identify orientation for both connected and disconnected versions; the point is not formal information but learner usability.
3. Only if the CPU audit shows the intended manipulation without surface leakage, launch the cheapest learned comparison: one connected aligned, one connected inverted, one disconnected aligned, one disconnected inverted, single seed smoke with the fast earlier analysis runner adapted to the new files. If there is no train fit, fix optimization before interpretation. If the connected arms show the desired state and signed mixed pattern in the smoke, expand to three seeds. If they do not, do not run a k sweep; move toward architecture/interface factorization.

## How to interpret outcomes

- **Connected succeeds and disconnected fails:** strong evidence for coordinate-connected experience. This would unite role coordinate anchor and state probe (absolute coordinate anchoring), orbit pair audit and slot reuse update (reusable filler/slot interface), and earlier analysis (logical disambiguation alone is insufficient) into a general data-efficient learning principle.
- **Both fail despite perfect fit:** the current DeBERTa/NLI fine-tuning setup lacks the factorized interface needed to reuse these constraints. The next route should build or test an architecture/objective with explicit entity-role-state factorization rather than more data allocation.
- **State succeeds but signed mixed orientation fails:** the learner acquired a task-local event-state rule, not a transferable relation coordinate.
- **Connected and disconnected both succeed:** connectivity as defined is not the causal variable; inspect surface leakage, token/filler memorization, or whether the manipulation failed to disconnect the learner's actual interface.

## Next experiment

The next builder should implement the CPU substrate and audit, not start GPU training yet. Reuse as much of `scripts/equivariant_symmetry_substrate.py`, `scripts/information_budget_substrate.py`, and `scripts/balanced_heuristic_baselines.py` as possible. The new output should live under `data/coordinate_connectivity_substrate/` and should include a concise summary plus full JSON manifest.
