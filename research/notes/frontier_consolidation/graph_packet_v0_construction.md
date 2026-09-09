# graph packet v0 construction Graph Packet v0 Construction

## Summary
- **Accepted: 50 rows** (0 rejected)
- Families: entity_state_update 11, event_temporal_causal 11, polarity_contrast_event 9, quantity_change_compare 11, social_belief_report 8
- Sources: Gutenberg 40, SimpleWiki 10

## Transformation quality
- **Graph compact**: mean ratio 0.266 (27% of source), preserves **100%** more relation/entity words than ordinary compact (50/50 rows)
- **Ordinary compact**: mean ratio 0.351 (35% of source), drops relation-specific vocabulary
- **Entity renamed**: mean ratio 0.980 (98% of source), 50/50 successful name substitution (new names present, old names gone)
- **Edge changed**: WEAK — only 5/50 truly different from source, 42/50 nearly identical. Model copies source rather than making the intended relation change. Usable for edge sensitivity on ≤8 rows only.

## Key structural property
Graph-preserving compaction retains entity and relation vocabulary that ordinary compaction drops. This is not merely a length difference — it is a content-selection difference driven by the explicit relational constraint in the prompt.

## Construction pipeline
1. Selected 50 high-quality Gutenberg/SimpleWiki rows from lead reopen mechanism route candidates
2. Extracted structured graphs via Qwen3.5-9B (compact JSON format, 50/50 recovered with truncation repair)
3. Generated 200 transformations (50 × 4 types) via Qwen3.5-9B at temperature 0.3
4. Verified entity coverage, length ratios, renaming quality
5. All transformations use the same model (Qwen3.5-9B) and matched prompt style

## Files
- **Packet**: `experiments/archive/frontier_consolidation/data/graph_packet_v0/graph_packet_v0.jsonl`
- Parsed graphs: `experiments/archive/frontier_consolidation/data/graph_packet_v0/parsed_graphs.jsonl`
- Summary: `experiments/archive/frontier_consolidation/data/graph_packet_v0/graph_packet_summary.json`
- Selected sources: `experiments/archive/frontier_consolidation/data/graph_packet_v0/selected_source_rows.jsonl`

## Measurement Design
Run frozen chck_82M over the packet. Two core measurements:

### 1. Cross-view relational transfer
For each row, mask tokens in the entity_renamed view and measure:
- NLL under graph_compact as prior (same row) vs ordinary_compact as prior (same row)
- If graph_compact produces lower NLL on relation-relevant tokens in entity_renamed, this shows relational transfer that ordinary compaction cannot provide

### 2. Renaming robustness
For each row, compute pseudo-log-likelihood:
- PLL(source) vs PLL(entity_renamed)
- If relation/entity tokens maintain similar PLL despite name changes, the model's representations are partially relational rather than entity-specific

### 3. Edge sensitivity (limited: ≤8 usable rows)
- PLL(source) vs PLL(edge_changed) on the subset with real edge changes
- If source is preferred, the model detects the changed relation

### What closes the route
If graph_compact does NOT outperform ordinary_compact on predicting entity_renamed targets, the graph-preserving construction adds nothing beyond ordinary compaction and should not receive H100 training time.

### What opens scaling
If graph-preserving views show measurable cross-view transfer, design a matched 4M-8M exposure screen: same source rows, same word budget, ordinary-MLM control vs graph-preserving multi-view arm.

## Limitations
- Edge-changed quality is too low for reliable edge sensitivity measurement
- Graph extraction used truncation repair — some graphs may have lost final relations
- All generation by Qwen3.5-9B at temperature 0.3 — outputs may have model-specific stylistic cues
- Source rows are from the legal compact-view reinvest pool, excluding generated rows
