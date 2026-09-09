# compact coupled system route synthesis anchor-context perturbation probe

JSON: `experiments/archive/representation_and_objectives/data/anchor_context_perturbation_probe_pilot/anchor_context_perturbation_probe.json`

Positive sensitivity = extra context masking raises retained-anchor target loss.

## Event preparation
- prepared_events: 338
- by_eval_set: {'doc_disjoint_all_accepted': 102, 'doc_disjoint_quality': 68, 'source_disjoint_quality': 71, 'train_fixed_probe': 97}
- skips: {'no_abs_context': 173, 'no_function_control': 1}

## Main sensitivities (all events)
### full_compact_100M
- abs_sensitivity: n=338, delta=0.238938, boot[0.10871,0.36683], frac_gt0=1.0
- function_sensitivity: n=284, delta=0.392745, boot[0.19599,0.600531], frac_gt0=1.0
- copied_sensitivity: n=318, delta=0.291601, boot[0.174771,0.421293], frac_gt0=1.0
- abs_minus_function_sensitivity: n=284, delta=-0.272938, boot[-0.560009,-0.042], frac_gt0=0.0
- abs_minus_copied_sensitivity: n=318, delta=-0.104992, boot[-0.278073,0.031943], frac_gt0=0.04
  - abs_sensitivity/abs_context_has_rel_event: n=71, delta=0.306284, boot[0.135176,0.545378]
  - abs_sensitivity/abs_context_no_rel_event: n=267, delta=0.218735, boot[0.083552,0.350925]

### drop_abs_100M
- abs_sensitivity: n=338, delta=0.142015, boot[0.043994,0.217636], frac_gt0=1.0
- function_sensitivity: n=284, delta=0.402562, boot[0.19239,0.644885], frac_gt0=1.0
- copied_sensitivity: n=318, delta=0.292152, boot[0.157043,0.429826], frac_gt0=1.0
- abs_minus_function_sensitivity: n=284, delta=-0.336119, boot[-0.637909,-0.115114], frac_gt0=0.0
- abs_minus_copied_sensitivity: n=318, delta=-0.193965, boot[-0.38331,-0.075671], frac_gt0=0.0
  - abs_sensitivity/abs_context_has_rel_event: n=71, delta=0.222568, boot[0.071952,0.40534]
  - abs_sensitivity/abs_context_no_rel_event: n=267, delta=0.117849, boot[0.041696,0.227038]

### drop_copied_word_100M
- abs_sensitivity: n=338, delta=0.234174, boot[0.140559,0.31917], frac_gt0=1.0
- function_sensitivity: n=284, delta=0.357923, boot[0.144194,0.577469], frac_gt0=1.0
- copied_sensitivity: n=318, delta=0.310165, boot[0.168424,0.442132], frac_gt0=1.0
- abs_minus_function_sensitivity: n=284, delta=-0.254527, boot[-0.560779,-0.050362], frac_gt0=0.01
- abs_minus_copied_sensitivity: n=318, delta=-0.149043, boot[-0.294468,-0.046992], frac_gt0=0.0
  - abs_sensitivity/abs_context_has_rel_event: n=71, delta=0.16884, boot[0.027723,0.350744]
  - abs_sensitivity/abs_context_no_rel_event: n=267, delta=0.253774, boot[0.146472,0.380299]

## Between-arm source-absent context sensitivity
- full_compact_100M_minus_drop_abs_100M_abs_context_sensitivity: n=338, delta=0.096923, boot[-0.011016,0.161729]
- full_compact_100M_minus_drop_copied_word_100M_abs_context_sensitivity: n=338, delta=0.004764, boot[-0.114433,0.111071]
- drop_abs_100M_minus_drop_copied_word_100M_abs_context_sensitivity: n=338, delta=-0.092159, boot[-0.180987,-0.01291]
