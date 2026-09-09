# equivariance control and state substrate repair held-predicate composition test

Seen predicates: ['defeated', 'beat', 'lost_to', 'was_beaten']
Held-composition predicates (support-only): ['overcame', 'fell_to', 'prevailed', 'was_defeated']
Held source domain: football

| mode | runs | converged | train acc | seen_comp_held_family | heldpred_hyp_only | heldpred_ctx_only | heldpred_both | heldpred_both_held_domain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| standard | 3 | 3 | 1.000 | 1.000 | 0.927 | 1.000 | 0.938 | 0.938 |
| equivariant | 3 | 2 | 0.834 | 1.000 | 0.958 | 0.900 | 0.847 | 0.852 |
| shuffled_any | 3 | 3 | 1.000 | 1.000 | 0.964 | 0.984 | 0.922 | 0.924 |

Converged-only means are the scientifically comparable numbers; runs stuck on the symmetric plateau are reported separately in the JSON.

Summary JSON: `experiments/archive/representation_and_objectives/data/held_predicate_composition/held_predicate_composition_summary.json`
