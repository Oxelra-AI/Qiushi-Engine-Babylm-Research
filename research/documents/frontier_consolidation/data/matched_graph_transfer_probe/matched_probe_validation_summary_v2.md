# matched graph transfer probe result matched graph-transfer probe validation

Prompts: 30; outputs: 30; accepted: 10; rejected: 20
Accepted by family: `{'entity_state_update': 3, 'event_temporal_causal': 3, 'polarity_contrast_event': 3, 'social_belief_report': 1}`

## Why this validation exists
The graph packet v0 construction packet was not decisive because graph_compact exposed relation words that ordinary_compact dropped. This validation accepts only examples where the answer word and its opposite are absent from every context view, where structured and neutral views have similar length and content overlap, and where reversed views are close surface edits. A positive frozen score on this packet would still need interpretation; a failed construction or failed transfer stops the route before any charged pretraining.

## Frequent issues
- view_structured-word-count-13-outside-14-42: 5
- view_neutral-word-count-11-outside-14-42: 5
- view_reversed-word-count-13-outside-14-42: 4
- query-copied-from-view: 4
- view_neutral-word-count-13-outside-14-42: 4
- view_reversed-contains-despite: 2
- view_reversed-word-count-12-outside-14-42: 2
- view_structured-contains-accepted: 2
- view_reversed-contains-rejected: 2
- structured-neutral-content-jaccard-low-0.294: 1
- view_structured-word-count-9-outside-14-42: 1
- view_neutral-word-count-10-outside-14-42: 1
- view_reversed-word-count-9-outside-14-42: 1
- structured-neutral-content-jaccard-low-0.273: 1
- view_structured-contains-because: 1
- view_structured-word-count-12-outside-14-42: 1
- view_neutral-word-count-12-outside-14-42: 1
- view_structured-contains-less: 1
- view_reversed-contains-more: 1
- structured-neutral-content-jaccard-low-0.231: 1

## Accepted metric ranges
- char_sim_structured_neutral: mean 0.827, min 0.605, max 0.966
- char_sim_structured_reversed: mean 0.948, min 0.872, max 0.984
- content_jaccard_structured_neutral: mean 0.629, min 0.350, max 0.900
- content_jaccard_structured_reversed: mean 0.833, min 0.684, max 0.933
- wc_neutral: mean 17.800, min 14.000, max 22.000
- wc_reversed: mean 18.900, min 14.000, max 24.000
- wc_structured: mean 18.800, min 14.000, max 24.000

Accepted JSONL: `experiments/archive/frontier_consolidation/data/matched_graph_transfer_probe/matched_probe_accepted_v2.jsonl`
Rejected JSONL: `experiments/archive/frontier_consolidation/data/matched_graph_transfer_probe/matched_probe_rejected_v2.jsonl`
Summary JSON: `experiments/archive/frontier_consolidation/data/matched_graph_transfer_probe/matched_probe_validation_summary_v2.json`
