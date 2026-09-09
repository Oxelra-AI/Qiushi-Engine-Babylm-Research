# earlier analysis/283 register packet-minus-clean readout

File-only readout; no model loading, training, official evaluation, GPU work, GlobalPIQA, SuperGLUE, AoA, upload, or leaderboard action.

Data state: `partial`. This states score availability, not scientific completion.

Available clean stable cells: 12 / 96.
Available direct-pair stable cells: 0 / 48.
Max cheap6 aggregation residual: +0.0000.

## Pair contrast: remove_childsub_rich_minus_remove_gutsimple_rich

| metric | window | complete ck | mean | min | max | positive | negative |
|---|---|---:|---:|---:|---:|---:|---:|
| BLiMP | common10_80 | 0 |  |  |  | 0 | 0 |
| BLiMP | early10_40 | 0 |  |  |  | 0 | 0 |
| BLiMP | mature50_80 | 0 |  |  |  | 0 | 0 |
| BLiMP | endpoint80 | 0 |  |  |  | 0 | 0 |
| Supplement | common10_80 | 0 |  |  |  | 0 | 0 |
| Supplement | early10_40 | 0 |  |  |  | 0 | 0 |
| Supplement | mature50_80 | 0 |  |  |  | 0 | 0 |
| Supplement | endpoint80 | 0 |  |  |  | 0 | 0 |
| EWoK | common10_80 | 0 |  |  |  | 0 | 0 |
| EWoK | early10_40 | 0 |  |  |  | 0 | 0 |
| EWoK | mature50_80 | 0 |  |  |  | 0 | 0 |
| EWoK | endpoint80 | 0 |  |  |  | 0 | 0 |
| Entity | common10_80 | 0 |  |  |  | 0 | 0 |
| Entity | early10_40 | 0 |  |  |  | 0 | 0 |
| Entity | mature50_80 | 0 |  |  |  | 0 | 0 |
| Entity | endpoint80 | 0 |  |  |  | 0 | 0 |
| COMPS | common10_80 | 0 |  |  |  | 0 | 0 |
| COMPS | early10_40 | 0 |  |  |  | 0 | 0 |
| COMPS | mature50_80 | 0 |  |  |  | 0 | 0 |
| COMPS | endpoint80 | 0 |  |  |  | 0 | 0 |
| Reading | common10_80 | 0 |  |  |  | 0 | 0 |
| Reading | early10_40 | 0 |  |  |  | 0 | 0 |
| Reading | mature50_80 | 0 |  |  |  | 0 | 0 |
| Reading | endpoint80 | 0 |  |  |  | 0 | 0 |
| cheap6 | common10_80 | 0 |  |  |  | 0 | 0 |
| cheap6 | early10_40 | 0 |  |  |  | 0 | 0 |
| cheap6 | mature50_80 | 0 |  |  |  | 0 | 0 |
| cheap6 | endpoint80 | 0 |  |  |  | 0 | 0 |
| exEntity5 | common10_80 | 0 |  |  |  | 0 | 0 |
| exEntity5 | early10_40 | 0 |  |  |  | 0 | 0 |
| exEntity5 | mature50_80 | 0 |  |  |  | 0 | 0 |
| exEntity5 | endpoint80 | 0 |  |  |  | 0 | 0 |

## Packet-minus-clean two-anchor summaries

| arm | removed register | metric | window | complete ck | mean | clean requirement |
|---|---|---|---|---:|---:|---|
| childsub_posmatched | child/subtitle-rich clean removed | Entity | common10_80 | 0 |  | both clean anchors required |
| childsub_posmatched | child/subtitle-rich clean removed | Entity | mature50_80 | 0 |  | both clean anchors required |
| childsub_posmatched | child/subtitle-rich clean removed | Entity | endpoint80 | 0 |  | both clean anchors required |
| childsub_posmatched | child/subtitle-rich clean removed | cheap6 | common10_80 | 0 |  | both clean anchors required |
| childsub_posmatched | child/subtitle-rich clean removed | cheap6 | mature50_80 | 0 |  | both clean anchors required |
| childsub_posmatched | child/subtitle-rich clean removed | cheap6 | endpoint80 | 0 |  | both clean anchors required |
| childsub_posmatched | child/subtitle-rich clean removed | exEntity5 | common10_80 | 0 |  | both clean anchors required |
| childsub_posmatched | child/subtitle-rich clean removed | exEntity5 | mature50_80 | 0 |  | both clean anchors required |
| childsub_posmatched | child/subtitle-rich clean removed | exEntity5 | endpoint80 | 0 |  | both clean anchors required |
| adult_posmatched | Gutenberg/SimpleWiki-rich clean removed | Entity | common10_80 | 0 |  | both clean anchors required |
| adult_posmatched | Gutenberg/SimpleWiki-rich clean removed | Entity | mature50_80 | 0 |  | both clean anchors required |
| adult_posmatched | Gutenberg/SimpleWiki-rich clean removed | Entity | endpoint80 | 0 |  | both clean anchors required |
| adult_posmatched | Gutenberg/SimpleWiki-rich clean removed | cheap6 | common10_80 | 0 |  | both clean anchors required |
| adult_posmatched | Gutenberg/SimpleWiki-rich clean removed | cheap6 | mature50_80 | 0 |  | both clean anchors required |
| adult_posmatched | Gutenberg/SimpleWiki-rich clean removed | cheap6 | endpoint80 | 0 |  | both clean anchors required |
| adult_posmatched | Gutenberg/SimpleWiki-rich clean removed | exEntity5 | common10_80 | 0 |  | both clean anchors required |
| adult_posmatched | Gutenberg/SimpleWiki-rich clean removed | exEntity5 | mature50_80 | 0 |  | both clean anchors required |
| adult_posmatched | Gutenberg/SimpleWiki-rich clean removed | exEntity5 | endpoint80 | 0 |  | both clean anchors required |

## Reading reminder

The direct pair difference is the primary removal-side discriminator. Arm-minus-clean uses both clean anchors and should not be summarized from a single clean seed. A mixed quarter_1x null can be cancellation between removed-register effects. Entity should be separated from exEntity5 because previous Entity movement was operation-allocation-skewed.

JSON: `experiments/archive/frontier_consolidation/data/register_vc_readout/register_vc_readout_summary.json`
Pair deltas: `experiments/archive/frontier_consolidation/data/register_vc_readout/register_pair_delta_rows.csv`
Packet-minus-clean rows: `experiments/archive/frontier_consolidation/data/register_vc_readout/register_packet_minus_clean_rows.csv`
