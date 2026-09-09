# earlier analysis pair-binding pilot decomposition

This CPU-only decomposition joins saved logits with raw substrate metadata. It separates direct-anchor relations h0/h2 from graph-transfer relations h1/h3, train informative versus neutral held-held rows, held-held edge closure, exact dyad membership, and row-paired aligned/inverted signed margins.

- train prediction rows: 12288
- eval prediction rows: 8704

## Train held-held comparison fit

| condition | arm | info kind | pair set | acc | signed margin | pred true frac |
|---|---|---|---|---:|---:|---:|
| pair_connected | aligned_state_bridge | informative_train | B_bridge_pairs | 1.000 | 17.632 | 0.500 |
| pair_connected | aligned_state_bridge | neutral_train | R_rewired_pairs | 1.000 | 17.687 | 0.500 |
| pair_connected | inverted_state_bridge | informative_train | B_bridge_pairs | 1.000 | 16.680 | 0.500 |
| pair_connected | inverted_state_bridge | neutral_train | R_rewired_pairs | 1.000 | 16.423 | 0.500 |
| pair_rewired | aligned_state_bridge | informative_train | R_rewired_pairs | 1.000 | 16.517 | 0.500 |
| pair_rewired | aligned_state_bridge | neutral_train | B_bridge_pairs | 1.000 | 16.132 | 0.500 |
| pair_rewired | inverted_state_bridge | informative_train | R_rewired_pairs | 1.000 | 16.225 | 0.500 |
| pair_rewired | inverted_state_bridge | neutral_train | B_bridge_pairs | 1.000 | 15.813 | 0.500 |

## Eval state: relation-by-relation changed exact choice

| condition | arm | relation | pattern | changed exact | margin | pair-both |
|---|---|---|---|---:|---:|---:|
| pair_connected | aligned_state_bridge | h0_dax | opposite | 1.000 | 27.404 | 1.000 |
| pair_connected | aligned_state_bridge | h0_dax | same | 1.000 | 22.431 | 1.000 |
| pair_connected | aligned_state_bridge | h1_mep | opposite | 0.750 | 9.018 | 0.750 |
| pair_connected | aligned_state_bridge | h1_mep | same | 0.031 | -23.810 | 0.031 |
| pair_connected | aligned_state_bridge | h2_norp | opposite | 1.000 | 28.594 | 1.000 |
| pair_connected | aligned_state_bridge | h2_norp | same | 0.938 | 21.401 | 0.938 |
| pair_connected | aligned_state_bridge | h3_ziv | opposite | 0.969 | 20.431 | 0.969 |
| pair_connected | aligned_state_bridge | h3_ziv | same | 0.062 | -23.594 | 0.062 |
| pair_connected | inverted_state_bridge | h0_dax | opposite | 0.594 | 3.037 | 0.594 |
| pair_connected | inverted_state_bridge | h0_dax | same | 0.406 | -5.498 | 0.406 |
| pair_connected | inverted_state_bridge | h1_mep | opposite | 0.812 | 21.942 | 0.812 |
| pair_connected | inverted_state_bridge | h1_mep | same | 0.125 | -25.623 | 0.125 |
| pair_connected | inverted_state_bridge | h2_norp | opposite | 0.594 | 4.383 | 0.562 |
| pair_connected | inverted_state_bridge | h2_norp | same | 0.469 | -3.479 | 0.469 |
| pair_connected | inverted_state_bridge | h3_ziv | opposite | 0.844 | 24.835 | 0.844 |
| pair_connected | inverted_state_bridge | h3_ziv | same | 0.125 | -27.117 | 0.125 |
| pair_rewired | aligned_state_bridge | h0_dax | opposite | 1.000 | 30.950 | 0.969 |
| pair_rewired | aligned_state_bridge | h0_dax | same | 1.000 | 30.900 | 1.000 |
| pair_rewired | aligned_state_bridge | h1_mep | opposite | 0.750 | 9.821 | 0.750 |
| pair_rewired | aligned_state_bridge | h1_mep | same | 0.375 | -4.762 | 0.375 |
| pair_rewired | aligned_state_bridge | h2_norp | opposite | 1.000 | 33.674 | 1.000 |
| pair_rewired | aligned_state_bridge | h2_norp | same | 1.000 | 33.077 | 1.000 |
| pair_rewired | aligned_state_bridge | h3_ziv | opposite | 0.312 | -6.095 | 0.312 |
| pair_rewired | aligned_state_bridge | h3_ziv | same | 0.000 | -16.323 | 0.000 |
| pair_rewired | inverted_state_bridge | h0_dax | opposite | 0.594 | 3.396 | 0.562 |
| pair_rewired | inverted_state_bridge | h0_dax | same | 0.438 | -0.341 | 0.438 |
| pair_rewired | inverted_state_bridge | h1_mep | opposite | 0.875 | 21.784 | 0.875 |
| pair_rewired | inverted_state_bridge | h1_mep | same | 0.125 | -24.373 | 0.125 |
| pair_rewired | inverted_state_bridge | h2_norp | opposite | 0.625 | 6.250 | 0.562 |
| pair_rewired | inverted_state_bridge | h2_norp | same | 0.406 | -3.138 | 0.406 |
| pair_rewired | inverted_state_bridge | h3_ziv | opposite | 0.875 | 23.099 | 0.875 |
| pair_rewired | inverted_state_bridge | h3_ziv | same | 0.125 | -24.644 | 0.125 |

## Eval state: voice split for graph-transfer same-initial

| condition | arm | voice | changed exact | margin |
|---|---|---|---:|---:|
| pair_connected | aligned_state_bridge | active | 0.062 | -24.650 |
| pair_connected | aligned_state_bridge | passive | 0.031 | -22.754 |
| pair_connected | inverted_state_bridge | active | 0.125 | -26.467 |
| pair_connected | inverted_state_bridge | passive | 0.125 | -26.272 |
| pair_rewired | aligned_state_bridge | active | 0.000 | -12.269 |
| pair_rewired | aligned_state_bridge | passive | 0.375 | -8.816 |
| pair_rewired | inverted_state_bridge | active | 0.125 | -24.640 |
| pair_rewired | inverted_state_bridge | passive | 0.125 | -24.377 |

## Held-held unseen edge closure

| condition | arm | rel1 | rel2 | acc | signed margin | pred true frac |
|---|---|---|---|---:|---:|---:|
| pair_connected | aligned_state_bridge | h0_dax | h1_mep | 0.625 | 4.018 | 0.875 |
| pair_connected | aligned_state_bridge | h1_mep | h3_ziv | 0.438 | -3.465 | 0.938 |
| pair_connected | aligned_state_bridge | h2_norp | h3_ziv | 0.469 | -3.834 | 0.969 |
| pair_connected | aligned_state_bridge | h3_ziv | h0_dax | 0.625 | 3.738 | 0.875 |
| pair_connected | inverted_state_bridge | h0_dax | h1_mep | 0.625 | 2.261 | 0.875 |
| pair_connected | inverted_state_bridge | h1_mep | h3_ziv | 0.469 | -2.460 | 0.719 |
| pair_connected | inverted_state_bridge | h2_norp | h3_ziv | 0.469 | -2.744 | 0.719 |
| pair_connected | inverted_state_bridge | h3_ziv | h0_dax | 0.625 | 2.428 | 0.875 |
| pair_rewired | aligned_state_bridge | h0_dax | h1_mep | 0.406 | -1.501 | 0.156 |
| pair_rewired | aligned_state_bridge | h1_mep | h3_ziv | 0.625 | 2.825 | 0.438 |
| pair_rewired | aligned_state_bridge | h2_norp | h3_ziv | 0.656 | 3.278 | 0.469 |
| pair_rewired | aligned_state_bridge | h3_ziv | h0_dax | 0.406 | -2.825 | 0.156 |
| pair_rewired | inverted_state_bridge | h0_dax | h1_mep | 0.625 | 1.355 | 0.625 |
| pair_rewired | inverted_state_bridge | h1_mep | h3_ziv | 0.344 | -2.091 | 0.719 |
| pair_rewired | inverted_state_bridge | h2_norp | h3_ziv | 0.344 | -1.785 | 0.594 |
| pair_rewired | inverted_state_bridge | h3_ziv | h0_dax | 0.500 | 0.349 | 0.750 |

## Mixed held-seen row-paired aligned/inverted differences

| condition | n rows | mean signed-margin diff | std | mean accuracy diff |
|---|---:|---:|---:|---:|
| pair_connected | 512 | -0.368 | 3.808 | 0.008 |
| pair_rewired | 512 | 0.504 | 10.978 | 0.014 |

## Scientific reading

The saved-output decomposition should be read together with the main analysis. The strongest pattern is direct-anchor state learning without graph-transfer state updating: directly bridged h0/h2 same-initial rows can be high in aligned arms, but h1/h3 same-initial rows remain low. Held-held edge closure and row-paired mixed margins determine whether the comparison graph itself was learned; if closure is weak or polarity is not mirrored, the failure is earlier than state composition. If closure is strong while graph-transfer state fails, the comparison and state objectives remain functionally separate despite shared text exposure.

## Files
- full JSON: `experiments/archive/representation_and_objectives/data/pair_binding_decomposition/pair_binding_decomposition.json`
