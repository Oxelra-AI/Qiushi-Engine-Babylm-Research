# comparison channel design — Comparison-channel causal intervention: comparison graph edges are necessary

## Summary

Removing training comparison edges while keeping all other components 
(learned dual-position equality, frozen matching, shared-trunk architecture,
sparse direct anchors, raw text with no name vocabulary) causes graph-transfer
gauge transport to collapse completely.

## Experimental design

**Intervention**: `no_cmp` — all 192 training comparison rows removed.  
**Control**: `full_graph` — all 192 comparison rows retained (matched seed 30000).  
**Architecture**: dual-position learned-equality shared_trunk (posalign repair and equality emergence design).  
**Bridge-sign pair**: both +1 and -1 with identical initialization `bab9584e68335755`.

### Data structure justification

The substrate has a critical asymmetry:
- Training state: s_give/s_receive (512 common), h0_dax/h2_norp (32 direct anchors)
- **No h1_mep or h3_ziv state labels in training**
- Training comparisons: 192 edges linking h0↔h1, h0↔h2, h1↔h2, h2↔h3
- Evaluation: all four relations (h0, h1, h2, h3) with held names

The comparison graph is the **sole training-time information pathway** from 
h0/h2 anchors (where bridge-sign orients the state) to h1/h3 events (where 
graph-transfer transport is measured).

## Results

### no_cmp (comparison removed)

| metric | bs+1 | bs-1 |
|---|---:|---:|
| train_state | 1.0 | 1.0 |
| train_cmp | 0 | 0 |
| matching_final | 1.0 | 1.0 |
| direct_same | **1.0** | **0.0** |
| graph_same | 0.25 | 0.5 |
| unchanged | 0.5 | 0.5 |
| hh_closure | 0.25 | 0.5 |
| pair_both_graph_same | 0.125 | 0.25 |

**Row-paired sign analysis**:
| family | n | same_sign_frac | opposite_sign_frac |
|---|---:|---:|---:|
| direct_anchor | 384 | 0.500 | 0.500 |
| graph_transfer | 384 | **0.875** | 0.125 |
| unchanged | 384 | **1.000** | — |

### full_graph (matched seed30000, 60 epochs, same init hash bab9584e68335755)

| metric | bs+1 | bs-1 |
|---|---:|---:|
| train_state | 1.0 | 1.0 |
| train_cmp | 1.0 | 1.0 |
| matching_final | 1.0 | 1.0 |
| direct_same | **1.0** | **0.0** |
| graph_same | **1.0** | **0.0** |
| hh_closure | **1.0** | **1.0** |
| mixed_acc | **1.0** | **0.0** |
| graph_same_margin | +13.70 | −16.77 |

**Row-paired sign analysis (full_graph)**:
| family | n | same_sign_frac | opposite_sign_frac |
|---|---:|---:|---:|
| direct_anchor | 384 | 0.500 | 0.500 |
| graph_transfer | 384 | 0.500 | **0.500** |
| unchanged | 384 | **1.000** | — |

## Side-by-side decisive contrast

| metric | no_cmp +1 | no_cmp -1 | full_graph +1 | full_graph -1 |
|---|---:|---:|---:|---:|
| graph_same | 0.25 | 0.50 | **1.00** | **0.00** |
| hh_closure | 0.25 | 0.50 | **1.00** | **1.00** |
| mixed_acc | 0.63 | 0.25 | **1.00** | **0.00** |
| direct_same | 1.00 | 0.00 | 1.00 | 0.00 |
| graph margin | −3.62 | −2.14 | **+13.70** | **−16.77** |

Row-paired graph_transfer opposite_sign_frac: **no_cmp=0.125, full_graph=0.500**

## Interpretation

1. **Direct anchors respond correctly to bridge_sign in both conditions**: 
   direct_same=1.0 for +1, 0.0 for -1. The sparse anchor mechanism works regardless
   of comparisons.

2. **Graph-transfer transport is destroyed without comparisons**: graph_same collapsed
   to 0.25/0.5 (near chance) vs 1.0/0.0 with full graph. The graph-transfer d_e signs
   are nearly identical between bridge signs (same_sign_frac=0.875), meaning the 
   bridge-sign intervention does NOT propagate to h1/h3 without comparison edges.

3. **Unchanged predictions degrade for h1/h3**: unchanged=0.5 (chance), because without
   the correct event orientation, the model cannot distinguish changed from unchanged
   entities for h1/h3 events.

4. **Direct anchors show the expected mixed pattern**: direct anchor opposite_sign_frac=0.5
   because half are changed (which reverse with bridge_sign) and half are unchanged 
   (which don't). This is consistent with correct anchor behavior.

## Causal decomposition (complete)

The gauge-transport mechanism requires **all four** components acting together:

| # | Component | Evidence for necessity | Step |
|---|---|---|---|
| 1 | Context-invariant identity matching | CharGRU/lexical shortcuts bypass → collapse | 296–298 |
| 2 | Sparse absolute anchors (bridge-sign flip) | No-anchor control → chance | 290 |
| 3 | Connected comparison graph | **no_cmp → graph_same collapse** | 300 |
| 4 | Shared computational representation | Untied control → local fit, no transport | 288–299 |

Removing any single component prevents gauge transport while preserving local fit 
where training signal exists.

## Bounded claim

> Finite experience becomes reusable when: (i) identity evidence is routed through 
> a context-invariant equality representation, (ii) sparse absolute anchors fix the
> gauge, (iii) connected relational constraints propagate relative coordinates, and
> (iv) a shared representation transports the anchored coordinate to unanchored 
> decisions. Each component is causally necessary; local fit, logical disambiguation,
> or any three of the four are insufficient.

This remains established in a controlled synthetic harness. The comparison-channel
evidence is clean because h1/h3 state labels are NEVER present in training, making
the comparison graph the provably unique information channel.

## Files

- Script: `training/scripts/comparison_channel_probe.py`
- no_cmp bs+1: `data/no_cmp_shared/no_cmp/dualpos_full_alphabet_shared_trunk_bs+1_seed30000/`
- no_cmp bs-1: `data/no_cmp_shared/no_cmp/dualpos_full_alphabet_shared_trunk_bs-1_seed30000/`
- Pair analysis: `data/no_cmp_pair_analysis/`
- full_graph bs+1: `data/full_graph_shared/full_graph/dualpos_full_alphabet_shared_trunk_bs+1_seed30000/`
- full_graph bs-1: `data/full_graph_shared/full_graph/dualpos_full_alphabet_shared_trunk_bs-1_seed30000/`
- Full_graph pair analysis: `data/full_graph_pair_analysis/`
- Design notes: `notes/comparison_channel_design.md`
