# full context pivot substitution probe — full-context pivot-substitution likelihood probe

Created: 2026-08-31T03:05:48Z Runtime: 235.2 s Device: cuda

## View

Base WWM is not modified. In a probe duplicate, target is masked and all non-target context stays visible. The true-pivot view is compared to same-position replacement by the matched anchor and by a same-category different pivot. Attention masks and target labels are identical.

## Coverage

Segment: {'tail_path': 'experiments/archive/representation_and_objectives/data/pvdm_strict_labels/compact_tail_70M_100M.jsonl', 'labels_path': 'experiments/archive/representation_and_objectives/data/pvdm_strict_labels/pvdm_strict_tail_70M_100M_labels.jsonl', 'start_tail_row': 64255, 'expected_start_tail_words': 10011326, 'skipped_words': 10011326, 'selected_rows': 64000, 'selected_words': 9971308, 'first_tail_row': 64255, 'last_tail_row': 128254, 'max_rows_mode': True}

Initial scan candidates: {'temporal': 33488, 'negation': 34573, 'causal_connector': 34157, 'physical_change': 33300, 'spatial': 29587, 'comparative': 2056}

Same-length filter: {'input_events': 9600, 'kept_same_len': 7283, 'reject': {'pivot_control_len_mismatch::causal_connector': 185, 'pivot_control_len_mismatch::comparative': 481, 'pivot_control_len_mismatch::negation': 854, 'pivot_control_len_mismatch::physical_change': 495, 'pivot_control_len_mismatch::spatial': 139, 'pivot_control_len_mismatch::temporal': 163}, 'kept_categories': {'causal_connector': 1415, 'comparative': 1119, 'negation': 746, 'physical_change': 1105, 'spatial': 1461, 'temporal': 1437}}

Final sampled categories: {'causal_connector': 373, 'comparative': 383, 'negation': 361, 'physical_change': 381, 'spatial': 384, 'temporal': 384}

Shuffle stats: {'input_events': 2304, 'events_with_different_pivot_shuffle': 2266, 'dropped_no_shuffle': 38, 'level_counts': {'0': 149200, '1': 240, '2': 41, '3': 0}, 'categories': {'causal_connector': 373, 'comparative': 383, 'negation': 361, 'physical_change': 381, 'spatial': 384, 'temporal': 384}}

## Likelihood separation

- ALL: n=2266; true-anchor mean=1.061472212260707 success=0.736098852603707; true-shuffle mean=0.5400406336299323 success=0.6791703442188879; true-pivotmasked mean=0.7636506630900268 success=0.7413945278022948

- physical_change: n=381; true-anchor mean=0.7955919740644585 success=0.6745406824146981; true-shuffle mean=0.49464832000211956 success=0.6955380577427821; true-pivotmasked mean=0.7063350756438767 success=0.7060367454068242

- comparative: n=383; true-anchor mean=1.7793980172720394 success=0.8067885117493473; true-shuffle mean=1.4993904412686996 success=0.7885117493472585; true-pivotmasked mean=1.1934512385781781 success=0.7806788511749347

- causal_connector: n=373; true-anchor mean=0.6997650387352358 success=0.6836461126005362; true-shuffle mean=0.17165939418359313 success=0.6327077747989276; true-pivotmasked mean=0.43337124456987336 success=0.6970509383378016

- temporal: n=384; true-anchor mean=0.8106854690425583 success=0.7395833333333334; true-shuffle mean=0.25776024826457916 success=0.6380208333333334; true-pivotmasked mean=0.7101377218526371 success=0.75

- spatial: n=384; true-anchor mean=0.5851476811194516 success=0.71875; true-shuffle mean=0.2481460533590507 success=0.6484375; true-pivotmasked mean=0.41140051397542265 success=0.7291666666666666

- negation: n=361; true-anchor mean=1.7275727455361702 success=0.7950138504155124; true-shuffle mean=0.5615167673815873 success=0.6703601108033241; true-pivotmasked mean=1.1410214891515342 success=0.7867036011080333

## Missing-family readiness

{
  "physical_change": {
    "n_events": 381,
    "mean_true_minus_anchor_replace": 0.7955919740644585,
    "mean_true_minus_shuffle_replace": 0.49464832000211956,
    "success_true_gt_anchor_replace": 0.6745406824146981,
    "success_true_gt_shuffle_replace": 0.6955380577427821,
    "supports_separation": true
  },
  "comparative": {
    "n_events": 383,
    "mean_true_minus_anchor_replace": 1.7793980172720394,
    "mean_true_minus_shuffle_replace": 1.4993904412686996,
    "success_true_gt_anchor_replace": 0.8067885117493473,
    "success_true_gt_shuffle_replace": 0.7885117493472585,
    "supports_separation": true
  }
}

## Files

- summary: `experiments/archive/representation_and_objectives/data/fullctx_probe_fw_interleaved_100M/full_context_pivot_substitution_summary.json`

- scored events: `experiments/archive/representation_and_objectives/data/fullctx_probe_fw_interleaved_100M/scored_events.jsonl`

- samples: `experiments/archive/representation_and_objectives/data/fullctx_probe_fw_interleaved_100M/sample_events.jsonl`

