# corrected multiseed gauge affine and next raw binding corrected synthesis — shared-coordinate gauge transport, scalar sufficiency, and next interface removal

## What corrected multiseed gauge affine and next raw binding established

corrected multiseed gauge affine and next raw binding concentrated on the causal gauge experimental design causal mechanism rather than moving to a new data surface or BabyLM-scale training. It added:

1. **Two additional primary seeds** for the fixed-aligned bridge-sign experiment:
   - seed 29001: `data/primary_gauge_seed29001/`
   - seed 29002: `data/primary_gauge_seed29002/`
   - seed 29000 inherited from causal gauge experimental design: `data/primary_gauge/`
2. **Saved-output multi-seed analysis**:
   - `scripts/multiseed_gauge_and_affine_summary.py`
   - `data/multiseed_gauge_affine_full/multiseed_gauge_and_affine_summary.md/json`
3. **Post-training frozen-affine scalar analysis**:
   - `scripts/frozen_affine_saved_outputs.py`
   - `data/frozen_affine_saved_outputs/frozen_affine_saved_outputs.md/json`
4. **independent_review-driven correction audit**:
   - `scripts/independent_review_corrections_audit.py`
   - `data/independent_review_corrections_audit/independent_review_corrections_audit.md/json`
5. **Evidence figure**:
   - `figures/gauge_transport_multiseed_affine.png` (2216×1436 PNG)

No official evaluation, BabyLM-scale training, or new broad search was launched.

## Multi-seed causal result

The primary causal fingerprint from causal gauge experimental design survives multi-seed testing most cleanly in `shared_trunk`, the model that shares the embedding+GRU representation but has separate state and comparison scalar heads. Across seeds 29000, 29001, and 29002:

- train state and comparison accuracy are `1.0` for both bridge signs in all seeds;
- h1/h3 graph-state canonical accuracy is `1.0` for bridge_sign `+1` and `0.0` for bridge_sign `-1` in all seeds;
- row-paired h1/h3 event-coordinate sign reversal is `1.0` in all seeds;
- held-held closure is invariant at `1.0` in all seeds;
- unchanged facts remain `1.0` and their state coordinate is unchanged;
- mixed held-seen orientation reverses strongly: bridge_sign `+1` accuracy `1.0`; bridge_sign `-1` accuracy `[0.0, 0.0, 0.25]` with negative margins, so held-vs-seen orientation changes while held-held products remain stable.

This is the strongest present evidence that sparse anchor supervision changes the representation used by the comparison graph and thereby transports an absolute gauge from anchored h0/h2 relations to unanchored h1/h3 relations. Identical output heads are not required.

The fully tied scalar model also replicates for seeds 29000 and 29001. In seed 29002 bridge_sign `-1`, however, it reaches train state `1.0` but comparison fit only `0.875`, held-held closure `0.625`, graph state `0.25`, and graph d_e reversal `0.75`. A targeted repair at lower LR for 440 epochs remains in the same solution (`data/tied_seed29002_bsminus_repair/`). The independent_review correction audit shows the failed training comparison rows are **not** at the probability clamp (`prob_wrong_range=[0.125,0.875]`, no rows near `1e-6` or `1-1e-6`); all 24 wrong rows are the `h2_norp`–`h3_ziv` held-held edge. Correct wording: this is a persistent local/optimization solution under the tested tied schedules, not evidence that the tied scalar loss is intrinsically unable to solve the graph.

The untied model remains a valid separation control for this architecture. It fits local training state and comparison rows, and h0/h2 direct state signs flip under bridge_sign, but mixed held-seen products are bridge-sign invariant in every seed and h1/h3 graph-state behavior is partial/non-causal. This supports the core point that local fit plus a separately solved comparison graph is insufficient when anchor gradients cannot reach the comparison representation.

## Scalar-affine result

The post-training frozen-affine analysis asks whether the learned comparison coordinate contains enough information for state orientation if calibrated by sparse anchors. It is **representational sufficiency**, not learned use during training.

The decisive saved-output case is shared factorization result synthesis `tied_heldheld_only_seed28801`: it has no held bridge state anchors, train comparison `1.0`, and held-held closure `1.0`, but raw direct and graph state accuracy are both `0.0` because the absolute sign is coherent but wrong. Fitting a one-dimensional affine readout `z = a d + b` from direct h0/h2 rows yields:

- fit on h0+h2 direct rows: h1 graph `1.0`, h3 graph `1.0`, graph-all `1.0`, cross-template graph `1.0`;
- fit on h0 only: h2 direct `1.0`, h1 graph `1.0`, h3 graph `1.0`;
- fit on h2 only: h0 direct `1.0`, h1 graph `1.0`, h3 graph `1.0`.

Thus a scalar downstream readout of the candidate-score difference is sufficient on the evaluated changed-state rows once the absolute sign is calibrated. This does **not** prove the whole internal representation is intrinsically one-dimensional, because the scalar is produced by a GRU scorer. It does show that the transported content can be an orientation bit/coordinate rather than requiring arbitrary high-dimensional transfer.

The causal gauge experimental design comparison-only seed29000 case is weaker: train comparison `0.875`, held-held closure `0.625`, and affine graph transfer `0.75`. It mainly shows that affine transfer tracks the quality of the comparison coordinate.

## Correction to the causal gauge experimental design/291 factor wording

The causal gauge experimental design single-seed factorial evidence still supports the three-factor structure (anchors, comparison graph, shared representation), but independent_review correctly noted a data-filter nuance. In `causal_gauge_probe.py`, `--no-bridge-anchors` removes every `state_query` row in the aligned arm file, not only changed h0/h2 anchors. The correction audit shows the removed arm rows all belong to h0/h2, but include both changed and unchanged rows:

- 128 arm state rows removed out of 320 arm rows;
- relation counts: `h0_dax:64`, `h2_norp:64`;
- rows include `query_kind=changed` and `query_kind=unchanged` for both initial patterns and static slots.

Because unchanged facts are handled by a separate static scorer and common seen state rows remain, this does not invalidate the causal gauge experimental design causal mechanism. But future graph×anchor experiments should implement a **narrow bridge-anchor filter** that removes only `is_changed && relation in {h0_dax,h2_norp}` while retaining sparse unchanged/static h0/h2 companions. corrected multiseed gauge affine and next raw binding’s multi-seed claim should be stated separately from causal gauge experimental design’s single-seed factorial claim.

## Mechanism statement after correction

The best current statement is:

> In the controlled binary candidate-event harness, the same-owner comparison objective learns relative orientation only up to a binary gauge per connected component. Sparse h0/h2 state anchors fix an absolute sign. When state-anchor loss and comparison loss act through a shared representation, the anchor-selected sign is transported through the held-held comparison graph to h1/h3; when state and comparison representations are untied, local fit does not produce this causal transport. A scalar candidate-score difference can carry the oriented information once calibrated, but training needs the shared representational path to use that coordinate for state updating.

This is a real mechanism result, but it remains inside a supplied harness. It is not yet natural semantic role induction, unified change/conservation, architecture-general transfer, or BabyLM-scale evidence.

## Supplied interfaces still present

The harness still supplies:

- participant discovery through candidate enumeration;
- `<cand>/<other>` replacement in the event string;
- cross-event candidate correspondence;
- binary ownership alternatives;
- same-owner comparison semantics;
- changed-vs-unchanged routing;
- a separate static scorer for unchanged facts.

Pair-both readouts remain numerically tied to changed-state accuracy because unchanged is separately solved. Future unified-updater work must test joint change and conservation in one pathway.

## Next experiment: remove candidate/other marking only

The next smallest mechanism-preserving experiment should remove exactly one supplied interface: the `<cand>/<other>` event replacement. It should retain the causal gauge experimental design bridge_sign connector, graph evidence, bridge-only and comparison-only cells, and `shared_trunk` versus `untied` contrast.

Important design requirements from independent_review:

1. **Candidate-conditioned raw input is required.** If event names are simply left raw without a query, both candidate scores see identical input and `d_e=0`. Use an input such as:
   - changed/comparison scorer: `raw event <QRY> raw candidate_name`
   - static scorer: raw premise plus raw hypothesis, with a query or candidate condition as needed.
2. **Use character/byte-level encoding or another representation that preserves held names.** A train-only word vocabulary would map unseen held names to `<unk>`, making held candidate queries indistinguishable.
3. **Use a query-to-event matching mechanism, not supplied candidate positions.** A small attention module over event tokens conditioned on the query name is appropriate; it should be identical in shared and untied models. Attention localization to the queried name becomes a useful binding readout.
4. **Maintain held-name and renaming tests.** Training names and evaluation names should be disjoint, balanced for role, position, relation, and label. Main test should use two unseen names; a seen/unseen diagnostic can reveal familiarity bias. Fresh global renamings should preserve signs and margins if binding is variable-based.
5. **Preserve the causal fingerprint.** In `shared_trunk/full`, success requires high train fit, h1/h3 row-paired bridge-sign reversal, held-held closure invariant, mixed held-seen sign reversal, unchanged stability, and failure of transport when either anchors or comparisons are removed. Untied should fit local rows without sign-controlled h1/h3 transport.
6. **Fix the no-anchor filter.** The comparison-only cell must remove only changed h0/h2 bridge anchors, not all h0/h2 state rows.

A later experiment should remove the separate static scorer and require a unified updater to both change affected facts and preserve unaffected ones. That should follow candidate binding rather than be merged into the next step, so failures can be interpreted.

## Required Evidence Files

- Corrected synthesis: `notes/corrected_multiseed_gauge_affine_and_next_raw_binding.md`
- independent_review design/verifier memos:
  - 
  - 
- Existing causal script: `training/scripts/causal_gauge_probe.py`
- Multi-seed evidence: `data/multiseed_gauge_affine_full/multiseed_gauge_and_affine_summary.md`
- Correction audit: `data/independent_review_corrections_audit/independent_review_corrections_audit.md`
- Frozen-affine evidence: `data/frozen_affine_saved_outputs/frozen_affine_saved_outputs.md`
