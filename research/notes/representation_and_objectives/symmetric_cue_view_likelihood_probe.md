# full context pivot substitution probe — symmetric cue-only relation likelihood probe

Created: 2026-08-31T02:29:46Z  Runtime: 228.78 s  Device: cuda

## Symmetric auxiliary view

The base WWM pass is untouched.  For this probe only, each view masks the dependent target, removes ordinary context from attention, and exposes exactly one cue subtoken. Semantic, matched-anchor, and shuffled-pivot views use the same pivot position for the cue; the native-anchor view is recorded separately.

## Segment and coverage

Segment rows/words: start_tail_row=64255, selected_rows=64000, selected_words=9971308, first=64255, last=128254

Scan candidates by category: {'temporal': 33488, 'negation': 34573, 'causal_connector': 34157, 'physical_change': 33300, 'spatial': 29587, 'comparative': 2056}

Reservoir selected by category: {'causal_connector': 384, 'comparative': 384, 'negation': 384, 'physical_change': 384, 'spatial': 384, 'temporal': 384}

After shuffled-pivot assignment: {'input_events': 2304, 'events_with_shuffled_pivot': 2302, 'dropped_no_shuffle': 2, 'shuffle_level_counts': {'0': 132941, '1': 12, '2': 23}, 'shuffle_categories': {'causal_connector': 384, 'comparative': 384, 'negation': 384, 'physical_change': 384, 'spatial': 383, 'temporal': 383}}

## Masked-token likelihood separation

- ALL: n=2302; sem-anchor_samepos mean=0.2796276909801672 success=0.5747176368375326; sem-shuffle_samepos mean=0.3156976415011077 success=0.47089487402258906; sem-anchor_native mean=0.6754776266193928 success=0.6663770634231103

- physical_change: n=384; sem-anchor_samepos mean=0.5287145345161358 success=0.609375; sem-shuffle_samepos mean=0.6895310655236244 success=0.6276041666666666; sem-anchor_native mean=0.7749889983485142 success=0.6640625

- comparative: n=384; sem-anchor_samepos mean=0.11879231439282496 success=0.5286458333333334; sem-shuffle_samepos mean=0.49400650213162106 success=0.4479166666666667; sem-anchor_native mean=0.38646258840647835 success=0.578125

- causal_connector: n=384; sem-anchor_samepos mean=0.4549935882290204 success=0.6614583333333334; sem-shuffle_samepos mean=0.08753748672703902 success=0.3828125; sem-anchor_native mean=0.9371251650154591 success=0.7369791666666666

- temporal: n=383; sem-anchor_samepos mean=0.36705134742876255 success=0.597911227154047; sem-shuffle_samepos mean=0.1094997987423491 success=0.42297650130548303; sem-anchor_native mean=0.8828324176001486 success=0.7284595300261096

- spatial: n=383; sem-anchor_samepos mean=0.053971067732370245 success=0.5248041775456919; sem-shuffle_samepos mean=0.15148645966233534 success=0.4830287206266319; sem-anchor_native mean=0.47926183717991605 success=0.6762402088772846

- negation: n=384; sem-anchor_samepos mean=0.15388331189751625 success=0.5260416666666666; sem-shuffle_samepos mean=0.36115992938478786 success=0.4609375; sem-anchor_native mean=0.5922237609823545 success=0.6145833333333334

## Missing-family readiness

{
  "physical_change": {
    "n_events": 384,
    "mean_sem_minus_anchor_samepos": 0.5287145345161358,
    "mean_sem_minus_shuffle_samepos": 0.6895310655236244,
    "success_sem_gt_anchor_samepos": 0.609375,
    "success_sem_gt_shuffle_samepos": 0.6276041666666666,
    "supports_separation": true
  },
  "comparative": {
    "n_events": 384,
    "mean_sem_minus_anchor_samepos": 0.11879231439282496,
    "mean_sem_minus_shuffle_samepos": 0.49400650213162106,
    "success_sem_gt_anchor_samepos": 0.5286458333333334,
    "success_sem_gt_shuffle_samepos": 0.4479166666666667,
    "supports_separation": false
  }
}

## Files

- summary: `experiments/archive/representation_and_objectives/data/symmetric_cue_view_likelihood_probe/symmetric_cue_view_likelihood_summary.json`

- scored events: `experiments/archive/representation_and_objectives/data/symmetric_cue_view_likelihood_probe/scored_events.jsonl`

- samples: `experiments/archive/representation_and_objectives/data/symmetric_cue_view_likelihood_probe/sample_events.jsonl`

