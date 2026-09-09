# shared gauge review and next causal controls training-diff and event-coordinate audit

CPU-only audit over static slot confounded factorial and balanced budget preparation/284 substrate rows and shared factorization result synthesis saved predictions. No model was loaded or trained.

## Training inventory diff

- common_seen rows: 1024
- aligned/inverted arm rows: 320 / 320
- row-id overlap aligned vs inverted: {'shared': 192, 'aligned_only': 128, 'inverted_only': 128}
- text mismatches on shared ids: 0
- label changes on shared ids: 0
- non-label content changes on shared ids: 0
- heldheld-only vs aligned comparison overlap: {'shared': 192, 'heldheld_only': 0, 'aligned_cmp_only': 0}
- heldheld-only vs inverted comparison overlap: {'shared': 192, 'heldheld_only': 0, 'inverted_cmp_only': 0}

Label changes by task/suite/relation/kind/global_swap:

```json
{}
```

## Aligned/inverted raw event-coordinate signs from comparison predictions

### tied seed 28801
- paired comparison rows: 640; event instances: 1280; skipped: {}

| suite | relation_class | n | same_sign | opposite_sign | corr(d_a,d_i) | corr(d_a,-d_i) | mean d_aligned | mean d_inverted |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| heldheld_unseen_edge_closure | held_direct_anchor | 96 | 0.000 | 1.000 | -0.999 | 0.999 | 0.254 | 0.326 |
| heldheld_unseen_edge_closure | held_graph_transfer | 160 | 0.000 | 1.000 | -0.997 | 0.997 | 0.135 | -0.031 |
| mixed_held_seen_orientation | held_direct_anchor | 256 | 0.000 | 1.000 | -0.998 | 0.998 | 0.190 | 0.245 |
| mixed_held_seen_orientation | held_graph_transfer | 256 | 0.000 | 1.000 | -0.997 | 0.997 | 0.169 | -0.039 |
| mixed_held_seen_orientation | seen_reference | 512 | 1.000 | 0.000 | 1.000 | -1.000 | -0.019 | -0.003 |

### untied seed 28801
- paired comparison rows: 640; event instances: 1280; skipped: {}

| suite | relation_class | n | same_sign | opposite_sign | corr(d_a,d_i) | corr(d_a,-d_i) | mean d_aligned | mean d_inverted |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| heldheld_unseen_edge_closure | held_direct_anchor | 96 | 1.000 | 0.000 | 1.000 | -1.000 | -0.113 | -0.113 |
| heldheld_unseen_edge_closure | held_graph_transfer | 160 | 1.000 | 0.000 | 1.000 | -1.000 | -0.154 | -0.154 |
| mixed_held_seen_orientation | held_direct_anchor | 256 | 1.000 | 0.000 | 1.000 | -1.000 | -0.084 | -0.084 |
| mixed_held_seen_orientation | held_graph_transfer | 256 | 1.000 | 0.000 | 1.000 | -1.000 | -0.193 | -0.193 |
| mixed_held_seen_orientation | seen_reference | 512 | 1.000 | 0.000 | 1.000 | -1.000 | -3.587 | -3.587 |

## Scientific reading

The row inventory shows that aligned and inverted shared factorization result synthesis arms share their comparison rows/text; the label changes are confined to bridge state-query rows with global-swap-sensitive labels. In the tied model, raw comparison-event coordinates reverse for held relations between aligned and inverted arms, while seen-reference coordinates stay same-signed in mixed held-seen rows. Held-held products remain usable because both held endpoints reverse together. The untied comparison branch, trained on identical comparison rows and isolated from state-anchor gradients, does not show the same held/seen subset gauge response. This strengthens the interpretation that the existing aligned/inverted contrast already perturbs an absolute gauge through state anchors, but it still does not replace the next causal control: a proper graph cut or state-interface permutation must show component-local reversible transport under matched local fit.

Full JSON: `experiments/archive/representation_and_objectives/data/training_diff_and_event_coordinate_audit/training_diff_and_event_coordinate_audit.json`
