# fw globalpiqa relevant substrate — FW compact/breadth substrate and GlobalPIQA-relevant domains

## Budget fact

The active FW experiment shares the same 494,154 FineWeb source-anchor words. The only changed FineWeb companion budget is matched at 318,851 words: compact arm uses preserved Qwen3.5 compact rewrites, while interleaved breadth uses independent whole FineWeb sentences. This note reads that changed budget before interpreting future scores.

## Broad category density in the changed companion budget

Densities are regex matches per 10k words, descriptive only.

| broad category | common source | compact rewrites | breadth companions | breadth/compact |
|---|---:|---:|---:|---:|
| `physical_motion_force` | 22.75 | 30.11 | 20.17 | 0.670 |
| `object_material_state` | 35.94 | 48.30 | 30.89 | 0.640 |
| `spatial_direction_geometry` | 106.42 | 123.79 | 96.38 | 0.779 |
| `time_count_order` | 328.08 | 477.81 | 360.86 | 0.755 |
| `affordance_tool_action` | 74.65 | 87.00 | 63.67 | 0.732 |
| `explicit_causal_conditional` | 98.55 | 101.93 | 92.27 | 0.905 |
| `history_society_named_entity` | 68.26 | 81.64 | 62.69 | 0.768 |

Composite broad-GlobalPIQA hit fractions:
- common compact/breadth FineWeb source anchors: record fraction 0.686, word fraction 0.718, words 494154
- compact arm changed companion: Qwen3.5 compact rewrites: record fraction 0.616, word fraction 0.658, words 318851
- breadth arm changed companion: independent FineWeb whole sentences: record fraction 0.732, word fraction 0.750, words 318851

## Reading for the upcoming FW results

If compact beats both breadth layouts while breadth has equal or richer broad physical/causal substrate, the result is more naturally about same-proposition consolidation, shorter denser restatement, and repeated relation surfaces than about seeing more raw physical facts. If breadth beats compact, especially on GlobalPIQA_parallel and EWoK, the current compact block is not the right use of the 318,851-word companion budget; the next data route should mine a general, independently selected physical/temporal/spatial/action-consequence substrate rather than add more compact paraphrases indiscriminately. If compact and breadth are close but differ by row-block/interleaved layout, local alternation/packing is active and the source_repeat arm becomes the attribution comparator.

## Files

- JSON: `experiments/archive/representation_and_objectives/data/fw_globalpiqa_substrate/fw_globalpiqa_relevant_substrate.json`
- category CSV: `experiments/archive/representation_and_objectives/data/fw_globalpiqa_substrate/fw_substrate_category_comparison.csv`
- domain CSV: `experiments/archive/representation_and_objectives/data/fw_globalpiqa_substrate/fw_substrate_domain_counts.csv`
