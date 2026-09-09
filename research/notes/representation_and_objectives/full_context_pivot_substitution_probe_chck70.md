# full context pivot substitution probe — full-context pivot-substitution likelihood probe

Created: 2026-08-31T02:37:41Z Runtime: 230.3 s Device: cuda

## View

Base WWM is not modified. In a probe duplicate, target is masked and all non-target context stays visible. The true-pivot view is compared to same-position replacement by the matched anchor and by a same-category different pivot. Attention masks and target labels are identical.

## Coverage

Segment: {'tail_path': 'experiments/archive/representation_and_objectives/data/pvdm_strict_labels/compact_tail_70M_100M.jsonl', 'labels_path': 'experiments/archive/representation_and_objectives/data/pvdm_strict_labels/pvdm_strict_tail_70M_100M_labels.jsonl', 'start_tail_row': 64255, 'expected_start_tail_words': 10011326, 'skipped_words': 10011326, 'selected_rows': 64000, 'selected_words': 9971308, 'first_tail_row': 64255, 'last_tail_row': 128254, 'max_rows_mode': True}

Initial scan candidates: {'temporal': 33488, 'negation': 34573, 'causal_connector': 34157, 'physical_change': 33300, 'spatial': 29587, 'comparative': 2056}

Same-length filter: {'input_events': 9600, 'kept_same_len': 7283, 'reject': {'pivot_control_len_mismatch::causal_connector': 185, 'pivot_control_len_mismatch::comparative': 481, 'pivot_control_len_mismatch::negation': 854, 'pivot_control_len_mismatch::physical_change': 495, 'pivot_control_len_mismatch::spatial': 139, 'pivot_control_len_mismatch::temporal': 163}, 'kept_categories': {'causal_connector': 1415, 'comparative': 1119, 'negation': 746, 'physical_change': 1105, 'spatial': 1461, 'temporal': 1437}}

Final sampled categories: {'causal_connector': 373, 'comparative': 383, 'negation': 361, 'physical_change': 381, 'spatial': 384, 'temporal': 384}

Shuffle stats: {'input_events': 2304, 'events_with_different_pivot_shuffle': 2266, 'dropped_no_shuffle': 38, 'level_counts': {'0': 149200, '1': 240, '2': 41, '3': 0}, 'categories': {'causal_connector': 373, 'comparative': 383, 'negation': 361, 'physical_change': 381, 'spatial': 384, 'temporal': 384}}

## Likelihood separation

- ALL: n=2266; true-anchor mean=0.9564969976989935 success=0.7281553398058253; true-shuffle mean=0.4784486257188172 success=0.6663724624889673; true-pivotmasked mean=0.7372807649426908 success=0.7228596646072374

- physical_change: n=381; true-anchor mean=0.7329256384782591 success=0.6955380577427821; true-shuffle mean=0.4685206749498492 success=0.6745406824146981; true-pivotmasked mean=0.6401332027701725 success=0.6929133858267716

- comparative: n=383; true-anchor mean=1.6239977816272375 success=0.804177545691906; true-shuffle mean=1.3520939666106107 success=0.7911227154046997; true-pivotmasked mean=1.1375438091767631 success=0.7362924281984334

- causal_connector: n=373; true-anchor mean=0.676872441489262 success=0.7024128686327078; true-shuffle mean=0.16824081398817603 success=0.6166219839142091; true-pivotmasked mean=0.4663824439329662 success=0.6916890080428955

- temporal: n=384; true-anchor mean=0.665241478084378 success=0.7213541666666666; true-shuffle mean=0.2317455354877893 success=0.6536458333333334; true-pivotmasked mean=0.6795901549576229 success=0.7369791666666666

- spatial: n=384; true-anchor mean=0.4931829599736375 success=0.671875; true-shuffle mean=0.17112964080418655 success=0.5885416666666666; true-pivotmasked mean=0.4278091667426149 success=0.6796875

- negation: n=361; true-anchor mean=1.5758392611432077 success=0.775623268698061; true-shuffle mean=0.4718790254774929 success=0.6731301939058172; true-pivotmasked mean=1.0856127231859105 success=0.8033240997229917

## Missing-family readiness

{
  "physical_change": {
    "n_events": 381,
    "mean_true_minus_anchor_replace": 0.7329256384782591,
    "mean_true_minus_shuffle_replace": 0.4685206749498492,
    "success_true_gt_anchor_replace": 0.6955380577427821,
    "success_true_gt_shuffle_replace": 0.6745406824146981,
    "supports_separation": true
  },
  "comparative": {
    "n_events": 383,
    "mean_true_minus_anchor_replace": 1.6239977816272375,
    "mean_true_minus_shuffle_replace": 1.3520939666106107,
    "success_true_gt_anchor_replace": 0.804177545691906,
    "success_true_gt_shuffle_replace": 0.7911227154046997,
    "supports_separation": true
  }
}

## Files

- summary: `experiments/archive/representation_and_objectives/data/full_context_pivot_substitution_probe_chck70/full_context_pivot_substitution_summary.json`

- scored events: `experiments/archive/representation_and_objectives/data/full_context_pivot_substitution_probe_chck70/scored_events.jsonl`

- samples: `experiments/archive/representation_and_objectives/data/full_context_pivot_substitution_probe_chck70/sample_events.jsonl`

