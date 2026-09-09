# causal gauge experimental design — Causal gauge transport: complete result

## Definitive result table

### Primary (full graph + anchors)

| Model | bs | train_state | train_cmp | direct_same | graph_same | pair_both | unchanged | mixed_acc | mixed_margin | graph_margin | hh_closure |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tied | +1 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | +13.75 | +16.23 | 1.0 |
| tied | -1 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | -13.02 | -13.60 | 1.0 |
| shared_trunk | +1 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | +13.81 | +12.37 | 1.0 |
| shared_trunk | -1 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | -13.81 | -14.03 | 1.0 |
| untied | +1 | 1.0 | 1.0 | 1.0 | 0.75 | 0.75 | 1.0 | 0.5 | +2.83 | +4.68 | 1.0 |
| untied | -1 | 1.0 | 1.0 | 0.0 | 0.5 | 0.5 | 1.0 | 0.5 | +2.83 | -0.18 | 1.0 |

### Bridge-only (no comparisons)

| Model | bs | train_state | direct_same | graph_same | graph_margin | mixed_acc |
|---|---:|---:|---:|---:|---:|---:|
| tied | +1 | 1.0 | 1.0 | 0.5 | 0.0 | 0.75 |
| tied | -1 | 1.0 | 0.0 | 0.5 | 0.0 | 0.25 |
| untied | +1 | 1.0 | 1.0 | 0.5 | 0.0 | 0.5 |
| untied | -1 | 1.0 | 0.0 | 0.5 | 0.0 | 0.5 |

### Comparison-only (no bridge anchors)

| Model | bs | train_cmp | direct_same | graph_same | graph_margin | mixed_acc |
|---|---:|---:|---:|---:|---:|---:|
| tied | +1 | 0.875 | 0.0 | 0.25 | -2.69 | 0.125 |
| tied | -1 | 0.875 | 0.0 | 0.25 | -2.69 | 0.125 |

## Row-paired d_e sign analysis (from bridge_sign_paired_analysis)

### Primary tied:
- Direct anchor changed: 0 same, **192 opposite** (corr(d+,-d-)=0.98)
- Graph transfer changed: 0 same, **192 opposite** (corr(d+,-d-)=0.97) → complete sign reversal
- Unchanged: **384 same**, 0 opposite (corr=1.0) → stable
- Held-held comparison product: **invariant** (fraction=1.0, both flip together)
- Mixed comparison product: **reverses** (fraction=0.0, held flips but seen stable)

### Primary shared_trunk:
- Same pattern as tied: 192/192 graph transfer opposite, 384/384 unchanged same
- Held-held invariant (1.0), mixed reverses (0.0)

### Primary untied:
- Direct anchor changed: 0/192 → state scorer flips h0/h2 locally
- Graph transfer changed: **144 same, 48 opposite** → NO coherent flip
- Held-held product: invariant (comparison scorer unchanged, de1/de2 same between bs±1)
- Mixed product: invariant → comparison coordinates unaffected

### Bridge-only tied:
- Direct anchor: 0 same, 192 opposite → anchors flip
- Graph transfer: **192 same, 0 opposite** → ZERO flip without comparison graph!
- Unchanged: 384 same → stable

### Comparison-only tied:
- All categories: same sign fraction = 1.0 → bridge_sign has zero effect without anchors

## Five-way scientific reading

1. **Graph-mediated gauge transport established**: In tied and shared_trunk models with full data, bridge_sign=-1 reverses the canonical h1/h3 state coordinate on ALL 192 graph-transfer changed queries, with corr(d+,-d-) ≈ 0.97-0.99. Without comparison graph (bridge-only), the SAME tied architecture with the SAME bridge_sign flip shows 0/192 graph-transfer reversal. The comparison graph is the causal conduit.

2. **Anchors are necessary**: Comparison-only runs are bit-identical for bs+1 and bs-1 (same loss trajectory, same eval metrics, same d_e signs). Bridge_sign has zero effect without anchors.

3. **Shared representation is sufficient; scalar identity is not required**: shared_trunk (shared embedding+GRU, separate scalar heads) reproduces the full tied reversal pattern. The gauge lives in the representation, not in the output layer.

4. **Untied fails causally**: The untied state scorer flips h0/h2 locally (192/192 direct anchor opposite), but graph transfer remains 144/48 same (75% same sign). The separate comparison scorer is completely unaffected: comparison d_e signs are identical between bs+1 and bs-1. No gradient path carries the anchor information to h1/h3 state decisions.

5. **No surface generalization**: Bridge-only tied learns h0/h2 anchors perfectly (direct_same flips from 1.0 to 0.0) but h1/h3 stays at chance (graph_same=0.5, margin=0.0 in both bridge signs). The shared scorer cannot generalize the nonce-verb pattern from h0/h2 to h1/h3 without comparison evidence connecting them.

## Updated principle (causal gauge experimental design refinement of shared factorization result synthesis)

> In a controlled candidate-event harness, finite experience becomes reusable transfer when:
> 1. **State anchors** fix an absolute orientation on a subset of relations,
> 2. **Relational comparison evidence** connects anchored and non-anchored relations through a graph,
> 3. **Shared computational representation** (at least a shared encoder trunk) allows anchor gradients to shape the coordinate that comparison evidence uses for non-anchored relations.
> Removing any one of these three factors eliminates the transfer, even when the remaining components provide sufficient information for local fit.

This refines shared factorization result synthesis's "shared candidate-event factorization" to identify the three necessary factors and shows that scalar output identity is sufficient but not necessary—shared representation is the operative mechanism.

## Remaining limits

- The harness still supplies candidate discovery, cross-event correspondence, binary ownership, comparison semantics, and changed-fact routing
- Unchanged facts use a separate static scorer; pair-both duplicates changed-state accuracy
- Single seed (29000) with paired initialization; multi-seed replication is straightforward
- No parser-free, unified updater, architecture-general, or BabyLM-scale evidence yet
- The "comparison graph" is the held-held comparison training rows; real-world graph structure is implicit in natural language co-occurrence and shared argument structure

## Files

- Primary results: `data/primary_gauge/`
- Bridge-only results: `data/bridge_only/`
- Comparison-only results: `data/comparison_only/`
- Bridge-sign paired analysis: `data/bridge_sign_analysis/`
- Script: `training/scripts/causal_gauge_probe.py`
- Analysis: `scripts/bridge_sign_paired_analysis.py`
