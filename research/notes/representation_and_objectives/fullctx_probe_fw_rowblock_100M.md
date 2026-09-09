# full context pivot substitution probe — full-context pivot-substitution likelihood probe

Created: 2026-08-31T03:01:28Z Runtime: 238.99 s Device: cuda

## View

Base WWM is not modified. In a probe duplicate, target is masked and all non-target context stays visible. The true-pivot view is compared to same-position replacement by the matched anchor and by a same-category different pivot. Attention masks and target labels are identical.

## Coverage

Segment: {'tail_path': 'experiments/archive/representation_and_objectives/data/pvdm_strict_labels/compact_tail_70M_100M.jsonl', 'labels_path': 'experiments/archive/representation_and_objectives/data/pvdm_strict_labels/pvdm_strict_tail_70M_100M_labels.jsonl', 'start_tail_row': 64255, 'expected_start_tail_words': 10011326, 'skipped_words': 10011326, 'selected_rows': 64000, 'selected_words': 9971308, 'first_tail_row': 64255, 'last_tail_row': 128254, 'max_rows_mode': True}

Initial scan candidates: {'temporal': 33488, 'negation': 34573, 'causal_connector': 34157, 'physical_change': 33300, 'spatial': 29587, 'comparative': 2056}

Same-length filter: {'input_events': 9600, 'kept_same_len': 7283, 'reject': {'pivot_control_len_mismatch::causal_connector': 185, 'pivot_control_len_mismatch::comparative': 481, 'pivot_control_len_mismatch::negation': 854, 'pivot_control_len_mismatch::physical_change': 495, 'pivot_control_len_mismatch::spatial': 139, 'pivot_control_len_mismatch::temporal': 163}, 'kept_categories': {'causal_connector': 1415, 'comparative': 1119, 'negation': 746, 'physical_change': 1105, 'spatial': 1461, 'temporal': 1437}}

Final sampled categories: {'causal_connector': 373, 'comparative': 383, 'negation': 361, 'physical_change': 381, 'spatial': 384, 'temporal': 384}

Shuffle stats: {'input_events': 2304, 'events_with_different_pivot_shuffle': 2266, 'dropped_no_shuffle': 38, 'level_counts': {'0': 149200, '1': 240, '2': 41, '3': 0}, 'categories': {'causal_connector': 373, 'comparative': 383, 'negation': 361, 'physical_change': 381, 'spatial': 384, 'temporal': 384}}

## Likelihood separation

- ALL: n=2266; true-anchor mean=1.0163872696284928 success=0.7449249779346867; true-shuffle mean=0.5459896227647879 success=0.6866725507502206; true-pivotmasked mean=0.7680498838948911 success=0.7290379523389232

- physical_change: n=381; true-anchor mean=0.7928228112025641 success=0.7270341207349081; true-shuffle mean=0.5178377440301921 success=0.6955380577427821; true-pivotmasked mean=0.741476034463244 success=0.6981627296587927

- comparative: n=383; true-anchor mean=1.7038653384925744 success=0.7989556135770235; true-shuffle mean=1.4475200183290804 success=0.7702349869451697; true-pivotmasked mean=1.1925850673001162 success=0.7650130548302873

- causal_connector: n=373; true-anchor mean=0.6682780059981447 success=0.7292225201072386; true-shuffle mean=0.16846706350702273 success=0.6541554959785523; true-pivotmasked mean=0.46879575021275327 success=0.6890080428954424

- temporal: n=384; true-anchor mean=0.758498895267015 success=0.7109375; true-shuffle mean=0.26203338915911445 success=0.6744791666666666; true-pivotmasked mean=0.7237283718295657 success=0.7395833333333334

- spatial: n=384; true-anchor mean=0.5164932330850812 success=0.7005208333333334; true-shuffle mean=0.26163960996260965 success=0.6197916666666666; true-pivotmasked mean=0.38329182881413243 success=0.6927083333333334

- negation: n=361; true-anchor mean=1.6887062707557303 success=0.8060941828254847; true-shuffle mean=0.6138157654809056 success=0.7063711911357341; true-pivotmasked mean=1.1113075211159162 success=0.7922437673130194

## Missing-family readiness

{
  "physical_change": {
    "n_events": 381,
    "mean_true_minus_anchor_replace": 0.7928228112025641,
    "mean_true_minus_shuffle_replace": 0.5178377440301921,
    "success_true_gt_anchor_replace": 0.7270341207349081,
    "success_true_gt_shuffle_replace": 0.6955380577427821,
    "supports_separation": true
  },
  "comparative": {
    "n_events": 383,
    "mean_true_minus_anchor_replace": 1.7038653384925744,
    "mean_true_minus_shuffle_replace": 1.4475200183290804,
    "success_true_gt_anchor_replace": 0.7989556135770235,
    "success_true_gt_shuffle_replace": 0.7702349869451697,
    "supports_separation": true
  }
}

## Files

- summary: `experiments/archive/representation_and_objectives/data/fullctx_probe_fw_rowblock_100M/full_context_pivot_substitution_summary.json`

- scored events: `experiments/archive/representation_and_objectives/data/fullctx_probe_fw_rowblock_100M/scored_events.jsonl`

- samples: `experiments/archive/representation_and_objectives/data/fullctx_probe_fw_rowblock_100M/sample_events.jsonl`

