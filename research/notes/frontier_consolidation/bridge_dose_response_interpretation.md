# bridge dose response interpretation — bridge dose-response interpretation framework

## What the bridge experiment tests

Bridge variants are **exact source-word subsequences** in source order with the same per-pair word count and content count as compact views. They differ from extractive_balanced only in which source words are selected: bridge variants maximize adjacency (contiguous runs) while balanced maximizes source-span coverage with evenly distributed words.

**Key property**: all bridge variants have zero source-absent content (compact has 17,891). So this tests adjacency within the source-only family, not generated re-expression.

## Arms in the dose-response (all on stock DeBERTa 100M)

| arm | gap1 | skip | span | content_frac | construction | source |
|---|---|---|---|---|---|---|
| extractive_balanced | 0.515 | 0.485 | 0.972 | 0.632 | evenly spaced source words | earlier analysis |
| bridge_mid | 0.671 | 0.329 | 0.873 | 0.632 | intermediate adjacency optimizer | bridge dose response interpretation |
| bridge_compactgap | 0.777 | 0.223 | 0.852 | 0.632 | compact-matched adjacency optimizer | bridge dose response interpretation |
| legal_compact (ref) | ~0.778 | ~0.223 | 0.807 | 0.647 | natural generated semantic compression | spatial repair route status/037 |

## Readout plan

Use the earlier analysis eval panel and earlier analysis interval analyzer at `chck_80M` and `chck_100M` only.

Primary metrics (same as extractive): cheap6 without GlobalPIQA, cheap5 without GlobalPIQA/Reading, EWoK+Entity, Supplement, Entity, COMPS.

All deltas reported as bridge_minus_compact (not bridge_minus_balanced), because the scientific question is whether the bridge closes the gap TO compact.

## Decision boundaries

### Adjacency dose-response within source-only

Test: balanced → bridge_mid → bridge_compactgap monotonic improvement on stable families?

- **Yes (monotonic)**: adjacency has a graded benefit within source-only selection. Worth quantifying how much of the balanced→compact gap it closes.
- **No (flat)**: adjacency alone doesn't help. The gap between source-only and compact is about generation/fluency/novel content, not word arrangement.
- **Inverted (mid or compactgap worse than balanced)**: adjacency optimization actually hurts by narrowing source span.

### Gap closure toward compact

Define `gap_closed = 1 - abs(bridge_compactgap - compact) / abs(balanced - compact)` on each stable family.

- **gap_closed > 0.5**: adjacency is a substantial partial mechanism
- **gap_closed < 0.2**: adjacency barely matters; source-only family closed definitively
- **gap_closed ~0.3-0.5**: adjacency contributes but other factors dominate

### Source span vs adjacency confound

bridge_compactgap has span 0.852 vs balanced 0.972. If bridge_compactgap performs worse than balanced on families where wide (span~0.928) also excelled, the span loss is the confound.

Key comparison: bridge_compactgap minus balanced on BLiMP (where wide exceeded compact) should be positive if adjacency helps BLiMP, negative if span loss hurts.

## What the result cannot establish

1. Whether fluent syntax matters (all bridge variants are telegraphic)
2. Whether source-absent content (17,891 words) is necessary (bridge has zero)
3. Whether compact's BPE token geometry matters (bridge has ~5.8% fewer active tokens)
4. Whether the mechanism transfers to other architectures (DeBERTa only)

## Artifacts

- Bridge preflight: `data/source_attested_continuity_bridge/`
- Bridge pools/streams: `data/bridge_pools_and_streams/`
- Bridge training runs: `training/runs/bridge_{compactgap,mid}_deberta100M_seed43022/`
- Extractive reference: `data/extractive_selected_eval_stage1/`
- Legal compact reference: `data/legal_mature_treatment_effect_eval/` and `data/compliant_full_eval/`
