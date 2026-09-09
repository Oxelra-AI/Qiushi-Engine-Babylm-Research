# full context pivot substitution probe — full-context pivot-substitution likelihood probe

Created: 2026-08-31T03:01:32Z Runtime: 242.2 s Device: cuda

## View

Base WWM is not modified. In a probe duplicate, target is masked and all non-target context stays visible. The true-pivot view is compared to same-position replacement by the matched anchor and by a same-category different pivot. Attention masks and target labels are identical.

## Coverage

Segment: {'tail_path': 'experiments/archive/representation_and_objectives/data/pvdm_strict_labels/compact_tail_70M_100M.jsonl', 'labels_path': 'experiments/archive/representation_and_objectives/data/pvdm_strict_labels/pvdm_strict_tail_70M_100M_labels.jsonl', 'start_tail_row': 64255, 'expected_start_tail_words': 10011326, 'skipped_words': 10011326, 'selected_rows': 64000, 'selected_words': 9971308, 'first_tail_row': 64255, 'last_tail_row': 128254, 'max_rows_mode': True}

Initial scan candidates: {'temporal': 33488, 'negation': 34573, 'causal_connector': 34157, 'physical_change': 33300, 'spatial': 29587, 'comparative': 2056}

Same-length filter: {'input_events': 9600, 'kept_same_len': 7283, 'reject': {'pivot_control_len_mismatch::causal_connector': 185, 'pivot_control_len_mismatch::comparative': 481, 'pivot_control_len_mismatch::negation': 854, 'pivot_control_len_mismatch::physical_change': 495, 'pivot_control_len_mismatch::spatial': 139, 'pivot_control_len_mismatch::temporal': 163}, 'kept_categories': {'causal_connector': 1415, 'comparative': 1119, 'negation': 746, 'physical_change': 1105, 'spatial': 1461, 'temporal': 1437}}

Final sampled categories: {'causal_connector': 373, 'comparative': 383, 'negation': 361, 'physical_change': 381, 'spatial': 384, 'temporal': 384}

Shuffle stats: {'input_events': 2304, 'events_with_different_pivot_shuffle': 2266, 'dropped_no_shuffle': 38, 'level_counts': {'0': 149200, '1': 240, '2': 41, '3': 0}, 'categories': {'causal_connector': 373, 'comparative': 383, 'negation': 361, 'physical_change': 381, 'spatial': 384, 'temporal': 384}}

## Likelihood separation

- ALL: n=2266; true-anchor mean=0.9748554581844384 success=0.736098852603707; true-shuffle mean=0.4871751028481938 success=0.6694616063548102; true-pivotmasked mean=0.7040212765217464 success=0.7219770520741394

- physical_change: n=381; true-anchor mean=0.7716660182116422 success=0.6876640419947506; true-shuffle mean=0.4899052047901579 success=0.6692913385826772; true-pivotmasked mean=0.6399085809625783 success=0.7007874015748031

- comparative: n=383; true-anchor mean=1.603843619064586 success=0.8015665796344648; true-shuffle mean=1.3735159677155462 success=0.7780678851174935; true-pivotmasked mean=1.1176745260065037 success=0.7493472584856397

- causal_connector: n=373; true-anchor mean=0.6950236711583672 success=0.6916890080428955; true-shuffle mean=0.1504301231194677 success=0.6112600536193029; true-pivotmasked mean=0.4383532469024777 success=0.6890080428954424

- temporal: n=384; true-anchor mean=0.7188925072126343 success=0.7265625; true-shuffle mean=0.22505385580552684 success=0.6432291666666666; true-pivotmasked mean=0.6708822281355727 success=0.71875

- spatial: n=384; true-anchor mean=0.45245504280865134 success=0.7213541666666666; true-shuffle mean=0.17331692872782392 success=0.6015625; true-pivotmasked mean=0.3334377234853794 success=0.6953125

- negation: n=361; true-anchor mean=1.639070140319448 success=0.7894736842105263; true-shuffle mean=0.5045525407466336 success=0.7146814404432132; true-pivotmasked mean=1.0367675270747208 success=0.7811634349030471

## Missing-family readiness

{
  "physical_change": {
    "n_events": 381,
    "mean_true_minus_anchor_replace": 0.7716660182116422,
    "mean_true_minus_shuffle_replace": 0.4899052047901579,
    "success_true_gt_anchor_replace": 0.6876640419947506,
    "success_true_gt_shuffle_replace": 0.6692913385826772,
    "supports_separation": true
  },
  "comparative": {
    "n_events": 383,
    "mean_true_minus_anchor_replace": 1.603843619064586,
    "mean_true_minus_shuffle_replace": 1.3735159677155462,
    "success_true_gt_anchor_replace": 0.8015665796344648,
    "success_true_gt_shuffle_replace": 0.7780678851174935,
    "supports_separation": true
  }
}

## Files

- summary: `experiments/archive/representation_and_objectives/data/fullctx_probe_fw_compact_100M/full_context_pivot_substitution_summary.json`

- scored events: `experiments/archive/representation_and_objectives/data/fullctx_probe_fw_compact_100M/scored_events.jsonl`

- samples: `experiments/archive/representation_and_objectives/data/fullctx_probe_fw_compact_100M/sample_events.jsonl`

