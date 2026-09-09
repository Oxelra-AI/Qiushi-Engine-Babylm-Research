# fw 70m execution synthesis — FW 70M GlobalPIQA row-overlap readout

## Measurement

CPU-only readout of existing 70M official prediction files for the FW compact-view and row-block whole-sentence source-breadth arms. No model scoring or training was run.

| arm | parallel | nonparallel | aggregate |
|---|---:|---:|---:|
| `compact_view_70M` | 26.21 | 51.00 | 38.607 |
| `source_breadth_rowblock_70M` | 26.21 | 47.00 | 36.607 |

The official 70M GlobalPIQA gap is entirely nonparallel: both arms score 26.21 on the 103-row parallel set; compact is 51.00 and row-block breadth is 47.00 on nonparallel.

## fw globalpiqa relevant substrate hard-row projection

| subset | n | compact correct | breadth correct | breadth fixes compact | compact fixes breadth | both wrong |
|---|---:|---:|---:|---:|---:|---:|
| `parallel_all` | 103 | 27 (26.21) | 27 (26.21) | 9 | 9 | 67 |
| `parallel_step107_always_wrong_all10` | 52 | 2 (3.85) | 1 (1.92) | 1 | 2 | 49 |
| `parallel_step107_wrong_at_least_half` | 82 | 13 (15.85) | 11 (13.41) | 4 | 6 | 65 |
| `parallel_not_step107_wrong_at_least_half` | 21 | 14 (66.67) | 16 (76.19) | 5 | 3 | 2 |

On the 52 rows that fw globalpiqa relevant substrate found wrong for all ten inspected endpoints, compact gets 2/52 and row-block breadth gets 1/52. Breadth fixes 1 compact miss while compact fixes 2 breadth misses, with 49 rows wrong for both. On the 82 rows wrong for at least half of prior endpoints, compact is 13/82 and breadth is 11/82. Thus row-block breadth's 70M EWoK gain is not accompanied by a repair of the load-bearing GlobalPIQA_parallel hard rows.

## Parallel category movements

| category | n | same correct | same wrong | breadth fixes compact | compact fixes breadth | breadth - compact |
|---|---:|---:|---:|---:|---:|---:|
| `counting` | 7 | 1 | 5 | 0 | 1 | -1 |
| `object_properties` | 2 | 1 | 1 | 0 | 0 | 0 |
| `object_properties_interactions` | 36 | 9 | 21 | 2 | 4 | -2 |
| `object_properties_interactions, affordances` | 21 | 3 | 13 | 4 | 1 | 3 |
| `object_properties_interactions, spatial` | 3 | 1 | 1 | 0 | 1 | -1 |
| `spatial` | 22 | 2 | 18 | 1 | 1 | 0 |
| `spatial, counting` | 1 | 0 | 0 | 0 | 1 | -1 |
| `time` | 4 | 1 | 2 | 1 | 0 | 1 |
| `time, counting` | 7 | 0 | 6 | 1 | 0 | 1 |

## Research consequence

The 70M FW family does not currently show the kind of GlobalPIQA_parallel movement that would explain the 41.80 gap. Compact remains the better 70M cheap7 arm because it preserves Entity and nonparallel GlobalPIQA; row-block breadth's EWoK increase is not enough and is partly domain-weighted. The correct next evidence is still complete 100M cheap7/full-vector reading when it appears, plus the existing margin wrapper if an arm is near range; the row-block breadth 70M result does not justify a new GlobalPIQA-specific expensive training line.

## Files

- JSON: `experiments/archive/representation_and_objectives/data/fw_70m_globalpiqa_row_overlap/fw_70m_globalpiqa_row_overlap.json`
- summary CSV: `experiments/archive/representation_and_objectives/data/fw_70m_globalpiqa_row_overlap/fw_70m_globalpiqa_summary.csv`
- row movement CSV: `experiments/archive/representation_and_objectives/data/fw_70m_globalpiqa_row_overlap/fw_70m_globalpiqa_row_movements.csv`
- parallel category movement CSV: `experiments/archive/representation_and_objectives/data/fw_70m_globalpiqa_row_overlap/fw_70m_globalpiqa_parallel_category_movements.csv`
