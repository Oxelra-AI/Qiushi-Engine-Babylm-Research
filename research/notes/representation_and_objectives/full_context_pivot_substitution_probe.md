# full context pivot substitution probe — full-context pivot-substitution likelihood probe

Created: 2026-08-31T02:37:56Z Runtime: 245.29 s Device: cuda

## View

Base WWM is not modified. In a probe duplicate, target is masked and all non-target context stays visible. The true-pivot view is compared to same-position replacement by the matched anchor and by a same-category different pivot. Attention masks and target labels are identical.

## Coverage

Segment: {'tail_path': 'experiments/archive/representation_and_objectives/data/pvdm_strict_labels/compact_tail_70M_100M.jsonl', 'labels_path': 'experiments/archive/representation_and_objectives/data/pvdm_strict_labels/pvdm_strict_tail_70M_100M_labels.jsonl', 'start_tail_row': 64255, 'expected_start_tail_words': 10011326, 'skipped_words': 10011326, 'selected_rows': 64000, 'selected_words': 9971308, 'first_tail_row': 64255, 'last_tail_row': 128254, 'max_rows_mode': True}

Initial scan candidates: {'temporal': 33488, 'negation': 34573, 'causal_connector': 34157, 'physical_change': 33300, 'spatial': 29587, 'comparative': 2056}

Same-length filter: {'input_events': 9600, 'kept_same_len': 7283, 'reject': {'pivot_control_len_mismatch::causal_connector': 185, 'pivot_control_len_mismatch::comparative': 481, 'pivot_control_len_mismatch::negation': 854, 'pivot_control_len_mismatch::physical_change': 495, 'pivot_control_len_mismatch::spatial': 139, 'pivot_control_len_mismatch::temporal': 163}, 'kept_categories': {'causal_connector': 1415, 'comparative': 1119, 'negation': 746, 'physical_change': 1105, 'spatial': 1461, 'temporal': 1437}}

Final sampled categories: {'causal_connector': 373, 'comparative': 383, 'negation': 361, 'physical_change': 381, 'spatial': 384, 'temporal': 384}

Shuffle stats: {'input_events': 2304, 'events_with_different_pivot_shuffle': 2266, 'dropped_no_shuffle': 38, 'level_counts': {'0': 149200, '1': 240, '2': 41, '3': 0}, 'categories': {'causal_connector': 373, 'comparative': 383, 'negation': 361, 'physical_change': 381, 'spatial': 384, 'temporal': 384}}

## Likelihood separation

- ALL: n=2266; true-anchor mean=0.925114145523897 success=0.7131509267431597; true-shuffle mean=0.4640306826616269 success=0.6429832303618711; true-pivotmasked mean=0.6972339688448101 success=0.6968225948808473

- physical_change: n=381; true-anchor mean=0.7610173493127517 success=0.6955380577427821; true-shuffle mean=0.44769467740366115 success=0.6902887139107612; true-pivotmasked mean=0.6241624187317547 success=0.6614173228346457

- comparative: n=383; true-anchor mean=1.6275370916120364 success=0.7493472584856397; true-shuffle mean=1.3309591182914269 success=0.7389033942558747; true-pivotmasked mean=1.1703733583243854 success=0.6997389033942559

- causal_connector: n=373; true-anchor mean=0.5715946274317661 success=0.6595174262734584; true-shuffle mean=0.14283721813627445 success=0.5764075067024129; true-pivotmasked mean=0.4341679596923911 success=0.67828418230563

- temporal: n=384; true-anchor mean=0.6379507528023775 success=0.7109375; true-shuffle mean=0.18672716025927608 success=0.5989583333333334; true-pivotmasked mean=0.6375926011132833 success=0.7239583333333334

- spatial: n=384; true-anchor mean=0.40978845180749585 success=0.6666666666666666; true-shuffle mean=0.1625329081646972 success=0.5989583333333334; true-pivotmasked mean=0.29858222604222345 success=0.6536458333333334

- negation: n=361; true-anchor mean=1.5719603517433303 success=0.8005540166204986; true-shuffle mean=0.5090591797096928 success=0.6537396121883656; true-pivotmasked mean=1.0316828615909281 success=0.7673130193905817

## Missing-family readiness

{
  "physical_change": {
    "n_events": 381,
    "mean_true_minus_anchor_replace": 0.7610173493127517,
    "mean_true_minus_shuffle_replace": 0.44769467740366115,
    "success_true_gt_anchor_replace": 0.6955380577427821,
    "success_true_gt_shuffle_replace": 0.6902887139107612,
    "supports_separation": true
  },
  "comparative": {
    "n_events": 383,
    "mean_true_minus_anchor_replace": 1.6275370916120364,
    "mean_true_minus_shuffle_replace": 1.3309591182914269,
    "success_true_gt_anchor_replace": 0.7493472584856397,
    "success_true_gt_shuffle_replace": 0.7389033942558747,
    "supports_separation": true
  }
}

## Files

- summary: `experiments/archive/representation_and_objectives/data/full_context_pivot_substitution_probe/full_context_pivot_substitution_summary.json`

- scored events: `experiments/archive/representation_and_objectives/data/full_context_pivot_substitution_probe/scored_events.jsonl`

- samples: `experiments/archive/representation_and_objectives/data/full_context_pivot_substitution_probe/sample_events.jsonl`

