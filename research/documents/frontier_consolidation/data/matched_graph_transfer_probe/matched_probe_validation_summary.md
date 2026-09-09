# matched graph transfer probe result matched graph-transfer probe validation

Prompts: 30; outputs: 30; accepted: 1; rejected: 29
Accepted by family: `{'entity_state_update': 1}`

## Why this validation exists
The graph packet v0 construction packet was not decisive because graph_compact exposed relation words that ordinary_compact dropped. This validation accepts only examples where the answer word and its opposite are absent from every context view, where structured and neutral views have similar length and content overlap, and where reversed views are close surface edits. A positive frozen score on this packet would still need interpretation; a failed construction or failed transfer stops the route before any charged pretraining.

## Frequent issues
- view_structured-contains-after: 9
- view_reversed-contains-before: 9
- view_structured-contains-because: 5
- view_reversed-contains-despite: 5
- query-copied-from-view: 3
- view_structured-contains-accepted: 3
- view_reversed-contains-rejected: 3
- missing-query_suffix: 2
- generator-ok-false:The source text describes a causal event (hunting caused a cold) and social interactions. The available target/distracto: 1
- generator-ok-false:The source text describes a complex narrative involving guardianship, legal orders, and fire incidents. Extracting a sin: 1
- generator-ok-false:The source text describes a spatial relationship ('out of the way') and a causal event ('cry settled it'), but does not : 1
- generator-ok-false:The source text describes social interactions and events without a clear temporal sequence (before/after), causal link (: 1
- view_neutral-contains-before: 1
- generator-ok-false:The source text describes a causal sequence (refusal leading to sandal removal) and a social transfer (right transfer). : 1
- view_neutral-contains-after: 1
- structured-neutral-identical: 1
- generator-ok-false:The source text describes a spatial location ('middle of Canada') and a state ('half full of water') but lacks a clear t: 1
- generator-ok-false:The source text describes a causal chain (Europe event -> deferment) and a temporal sequence (war -> changes). The requi: 1
- view_structured-word-count-13-outside-14-42: 1
- view_reversed-word-count-13-outside-14-42: 1

## Accepted metric ranges
- char_sim_structured_neutral: mean 0.935, min 0.935, max 0.935
- char_sim_structured_reversed: mean 0.973, min 0.973, max 0.973
- content_jaccard_structured_neutral: mean 0.857, min 0.857, max 0.857
- content_jaccard_structured_reversed: mean 0.857, min 0.857, max 0.857
- wc_neutral: mean 19.000, min 19.000, max 19.000
- wc_reversed: mean 19.000, min 19.000, max 19.000
- wc_structured: mean 19.000, min 19.000, max 19.000

Accepted JSONL: `experiments/archive/frontier_consolidation/data/matched_graph_transfer_probe/matched_probe_accepted.jsonl`
Rejected JSONL: `experiments/archive/frontier_consolidation/data/matched_graph_transfer_probe/matched_probe_rejected.jsonl`
Summary JSON: `experiments/archive/frontier_consolidation/data/matched_graph_transfer_probe/matched_probe_validation_summary.json`
