# compact coupled system route synthesis anchor-context perturbation probe

JSON: `experiments/archive/representation_and_objectives/data/anchor_context_perturbation_probe/anchor_context_perturbation_probe.json`

Positive sensitivity = extra context masking raises retained-anchor target loss.

## Event preparation
- prepared_events: 3596
- by_eval_set: {'doc_disjoint_all_accepted': 1226, 'doc_disjoint_quality': 475, 'source_disjoint_quality': 734, 'train_fixed_probe': 1161}
- skips: {'no_abs_context': 1930, 'no_function_control': 31}

## Main sensitivities (all events)
### full_compact_100M
- abs_sensitivity: n=3596, delta=0.404639, boot[0.349594,0.460553], frac_gt0=1.0
- function_sensitivity: n=2964, delta=0.338483, boot[0.284572,0.39121], frac_gt0=1.0
- copied_sensitivity: n=3376, delta=0.313473, boot[0.265993,0.363185], frac_gt0=1.0
- abs_minus_function_sensitivity: n=2964, delta=-0.032557, boot[-0.101525,0.033687], frac_gt0=0.16
- abs_minus_copied_sensitivity: n=3376, delta=0.052154, boot[-0.01313,0.119847], frac_gt0=0.931667
  - abs_sensitivity/abs_context_has_rel_event: n=687, delta=0.453383, boot[0.344466,0.566199]
  - abs_sensitivity/abs_context_no_rel_event: n=2909, delta=0.392321, boot[0.330135,0.451151]

### drop_abs_100M
- abs_sensitivity: n=3596, delta=0.360792, boot[0.309415,0.40985], frac_gt0=1.0
- function_sensitivity: n=2964, delta=0.298764, boot[0.246625,0.354137], frac_gt0=1.0
- copied_sensitivity: n=3376, delta=0.294612, boot[0.251748,0.343868], frac_gt0=1.0
- abs_minus_function_sensitivity: n=2964, delta=-0.032437, boot[-0.103132,0.041393], frac_gt0=0.176667
- abs_minus_copied_sensitivity: n=3376, delta=0.028572, boot[-0.037075,0.094321], frac_gt0=0.803333
  - abs_sensitivity/abs_context_has_rel_event: n=687, delta=0.39824, boot[0.289621,0.508073]
  - abs_sensitivity/abs_context_no_rel_event: n=2909, delta=0.351328, boot[0.294725,0.407795]

### drop_copied_word_100M
- abs_sensitivity: n=3596, delta=0.391266, boot[0.328776,0.452096], frac_gt0=1.0
- function_sensitivity: n=2964, delta=0.312859, boot[0.259496,0.365476], frac_gt0=1.0
- copied_sensitivity: n=3376, delta=0.299039, boot[0.251482,0.352329], frac_gt0=1.0
- abs_minus_function_sensitivity: n=2964, delta=-0.033884, boot[-0.105356,0.037266], frac_gt0=0.18
- abs_minus_copied_sensitivity: n=3376, delta=0.05148, boot[-0.021819,0.120923], frac_gt0=0.916667
  - abs_sensitivity/abs_context_has_rel_event: n=687, delta=0.411536, boot[0.292318,0.544851]
  - abs_sensitivity/abs_context_no_rel_event: n=2909, delta=0.386143, boot[0.319157,0.453655]

## Between-arm source-absent context sensitivity
- full_compact_100M_minus_drop_abs_100M_abs_context_sensitivity: n=3596, delta=0.043848, boot[0.016648,0.071376]
- full_compact_100M_minus_drop_copied_word_100M_abs_context_sensitivity: n=3596, delta=0.013374, boot[-0.017005,0.043924]
- drop_abs_100M_minus_drop_copied_word_100M_abs_context_sensitivity: n=3596, delta=-0.030474, boot[-0.066401,2.8e-05]
