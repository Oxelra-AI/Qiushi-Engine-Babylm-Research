# corrected multiseed gauge affine and next raw binding — Multi-seed causal gauge transport and scalar-affine sufficiency

## Scientific purpose

causal gauge experimental design established a causal gauge-transport fingerprint in a supplied candidate-event harness: a `bridge_sign` connector applied only inside h0/h2 changed-state anchor loss reversed the held h1/h3 state coordinate when and only when state anchors, the held-held comparison graph, and a shared representation were present. corrected multiseed gauge affine and next raw binding asked two focused questions before moving to a more natural role-discovery surface:

1. **Stability**: does the bridge-sign × graph × anchor effect replicate across random seeds, or was causal gauge experimental design a lucky optimization trajectory?
2. **Scalar vs broader representation**: can a post-training one-dimensional affine map from a frozen comparison coordinate carry the state orientation, or does the evidence require a broader multidimensional shared representation?

No BabyLM-scale training or official evaluation was launched. The only GPU work was the minimal primary causal replication for two additional seeds, plus one targeted tied seed repair after an underfit comparison graph appeared.

## New artifacts

- Multi-seed primary runs:
  - `data/primary_gauge/` (seed 29000 from causal gauge experimental design)
  - `data/primary_gauge_seed29001/`
  - `data/primary_gauge_seed29002/`
- Multi-seed saved-output synthesis:
  - `scripts/multiseed_gauge_and_affine_summary.py`
  - `data/multiseed_gauge_affine_full/multiseed_gauge_and_affine_summary.md`
  - `data/multiseed_gauge_affine_full/multiseed_gauge_and_affine_summary.json`
- Frozen-affine scalar analysis:
  - `scripts/frozen_affine_saved_outputs.py`
  - `data/frozen_affine_saved_outputs/frozen_affine_saved_outputs.md`
  - `data/frozen_affine_saved_outputs/frozen_affine_saved_outputs.json`
- Earlier CPU pilot using causal gauge experimental design no-anchor only:
  - `scripts/affine_from_saved_noanchor.py`
  - `data/affine_from_saved_noanchor/`
- Targeted tied seed29002 bs-1 repair:
  - `data/tied_seed29002_bsminus_repair/`
- Interim evidence figure:
  - `figures/gauge_transport_multiseed_affine.png` (2216×1436 PNG, 187K)

## Multi-seed result

### Shared trunk is fully stable

The cleanest representation-sharing model is `shared_trunk`: one shared embedding+GRU trunk with separate scalar output heads for state and comparison. Across seeds 29000, 29001, and 29002:

- Train state and comparison fit: `1.0 / 1.0` for all bridge signs and seeds.
- h1/h3 graph-state canonical accuracy:
  - bridge_sign `+1`: mean `1.000`, values `[1.0, 1.0, 1.0]`
  - bridge_sign `-1`: mean `0.000`, values `[0.0, 0.0, 0.0]`
- h1/h3 graph-state row-paired event-coordinate reversal: mean `1.000`, values `[1.0, 1.0, 1.0]`
- Pair changed+unchanged correctness on graph/same rows tracks the changed-state sign exactly:
  - bridge_sign `+1`: mean `1.000`
  - bridge_sign `-1`: mean `0.000`
- Unchanged facts remain stable: `1.000` for both bridge signs in all seeds; unchanged d_e has zero reversal.
- Held-held closure remains invariant: `1.000` for both bridge signs in all seeds.
- Mixed held-seen readout follows the held gauge: `mixed_acc + = 1.000`; `mixed_acc -` values `[0.0, 0.0, 0.25]`, mean `0.083`, with negative margins mean `-12.619`. The one 0.25 case is still strongly negative in margin and does not affect graph-state reversal or held-held closure.

This is robust causal evidence that the anchor sign is transported through the graph in the shared representation, not through output-head identity.

### Tied mostly replicates, but shows a real optimization boundary

The fully tied scalar model replicated cleanly in seeds 29000 and 29001. Both have train state/comparison fit `1.0`, h1/h3 graph-state `+1 -> 1.0`, `-1 -> 0.0`, mixed `+1 -> 1.0`, `-1 -> 0.0`, held-held closure `1.0`, and 100% row-paired graph d_e sign reversal.

Seed 29002 differs in the tied bs-1 cell:

- train_state remains `1.0`, but train_cmp stays `0.875`;
- held-held closure is only `0.625`;
- graph_same is `0.25` under bs-1, not `0.0`;
- row-paired graph d_e reversal is `0.75`, not `1.0`.

I ran a targeted repair with the same tied seed29002 bs-1 cell for 440 epochs at lower LR (`0.0015`) in `data/tied_seed29002_bsminus_repair/`. It did **not** escape the same solution: train_cmp stayed `0.875`, held-held closure `0.625`, graph_same `0.25`, mixed_acc `0.125`. This should be preserved as a real optimization/convergence boundary of the exact tied scalar loss, not hidden. It does not overturn the shared-representation result because the shared_trunk cell for the same seed/sign reaches full comparison fit and full gauge transport.

### Untied remains separated

Across seeds, the untied model fits local training rows (`1.0` state/comparison) and flips h0/h2 direct anchors locally, but it does not carry the sign through the comparison coordinate:

- h1/h3 graph_same is partial and not sign-controlled: plus values `[0.75, 0.75, 0.5]`, minus values `[0.5, 0.5, 0.75]`.
- Mixed held-seen accuracy is identical between bridge signs for each seed (`0.5/0.5`, `0.5/0.5`, `0.75/0.75`).
- Mixed product sign is invariant across bridge signs in all seeds (`product_same_frac=1.0`).
- Held-held closure is `1.0` because the separate comparison scorer fits comparisons, but state-anchor gradients do not reach that comparison coordinate.

This preserves causal gauge experimental design's key contrast: local fit plus separately solved comparison graph is insufficient; the bridge anchor must shape the representation used by the graph.

## Frozen-affine scalar result

I analyzed saved no-held-anchor outputs with `scripts/frozen_affine_saved_outputs.py`. It fits only a one-dimensional affine rule `z = a d + b`, where `d = score(candidate0)-score(candidate1)`, using direct h0/h2 evaluation state rows as calibration rows. Then it tests h1/h3 graph-transfer state rows. This is a **post-training representational sufficiency** result: it does not show the unanchored model learned to use that map for state updating.

Two saved-output sources were tested:

1. **shared factorization result synthesis heldheld-only tied seed28801**: no held bridge state anchors, train_cmp `1.0`, held-held closure `1.0`. Raw state readout was globally wrong (`raw_direct=0.0`, `raw_graph=0.0`), consistent with a coherent but unanchored absolute sign. After fitting a one-dimensional affine map from h0/h2 direct rows:
   - direct h0/h2 calibration: `1.0`;
   - h1/h3 graph same/all: `1.0`;
   - cross-template h1/h3 graph: `1.0`;
   - fitting the opposite anchor sign flips the whole state coordinate (`canonical=0.0`, arm-target=1.0).

2. **causal gauge experimental design comparison-only tied seed29000**: no bridge anchors but fixed aligned arm with common seen state rows, train_cmp only `0.875`, held-held closure `0.625`. Affine transfer reaches only `0.75` on h1/h3, consistent with the comparison coordinate itself being incompletely solved.

The first case is the decisive scalar-sufficiency evidence: when the frozen comparison graph has a coherent held coordinate, a single affine sign/threshold learned from sparse h0/h2 anchors is enough to orient h1/h3 state rows. Therefore the current mechanism does not require a high-dimensional arbitrary representation to express the transfer. What training needs is a path for sparse anchor evidence to calibrate and use the scalar coordinate.

## Updated scientific interpretation

The best current statement is now:

> In the controlled candidate-event harness, finite experience becomes reusable role/state transfer when sparse state anchors calibrate a relation coordinate that is already connected through comparison evidence, and when the learner's parameterization lets anchor gradients modify the representation used by that graph. A one-dimensional scalar coordinate is sufficient to carry the sign once calibrated, but ordinary training will not use it for state updating unless the state and comparison pathways share the relevant representation.

This refines causal gauge experimental design in two ways:

- `shared_trunk` across three seeds shows the causal transport is a **shared-representation** effect and does not require identical scalar output heads.
- The frozen-affine analysis shows the transported content can be as low-dimensional as a **single signed scalar coordinate**, but post-training affine success is only representational sufficiency.

## Boundaries that still matter

The result remains inside a supplied harness. It does **not** yet establish parser-free natural-language role induction or a BabyLM-scale principle. The harness still supplies:

- candidate/other marking and participant discovery;
- cross-event participant correspondence;
- binary ownership alternatives;
- same-owner comparison semantics;
- changed-vs-unchanged routing;
- a separate static scorer for unchanged facts.

Also, the affine analysis intentionally ignores unchanged facts. It calibrates changed-state event ownership; unchanged preservation currently comes from a separate static scorer in the shared factorization result synthesis/290 harness. A unified updater is still needed before claiming joint change plus conservation.

## Next mechanism-preserving step

The next work should remove **one supplied interface at a time** while retaining the sign intervention and graph–anchor decomposition. The best first removal is candidate/other marking: train the shared_trunk vs untied probe on raw names with no `<cand>/<other>` substitution, evaluating on held names and with the same bridge_sign, graph evidence, bridge-only, and comparison-only controls. This tests whether the shared scalar coordinate can survive learned candidate binding rather than a supplied candidate parser. Keep data, relation graph, and labels otherwise identical; do not move to BabyLM-scale training yet.

A second natural continuation is a unified event updater: one scorer must answer both changed and unchanged facts rather than using a separate static scorer. That should follow after candidate binding is tested, because it removes a different supplied interface.
