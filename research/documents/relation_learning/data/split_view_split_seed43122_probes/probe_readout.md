# probe results interpretation held-out copy, rewrite-conditioning, and Entity cue-ablation probes

## Record counts

- Held-out natural copy records: 4000 from `experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/heldout_cleanqwen_rows.jsonl`.
- Held-out rewrite-conditioning records: 11784 from unselected accepted pairs in `experiments/archive/frontier_consolidation/data/expansion_analysis/combined_all_accepted_pairs.jsonl` after excluding `experiments/archive/frontier_consolidation/data/dose_distribution_select/selected_matched_max_pairs.jsonl`.
- Entity cue-ablation records: 9768 on stale-non-gold relevant-update Entity items.

## Key late contrasts

For copy and rewrite probes, positive gain contrast means the first arm benefits more from the relevant context cue. For Entity cue ablations, a positive `effect_no_initial` means removing the queried-box initial clause raises the gold-over-stale margin; a negative `effect_no_last/all_relevant` means removing update evidence lowers the gold-over-stale margin.

### Held-out natural copy gain

| seed | contrast | late delta gain | n |
|---:|---|---:|---:|
| 43122 | VminusC | +0.2038 | 2000 |

### Held-out rewrite content-conditioning gain

| seed | group | contrast | late delta gain | n |
|---:|---|---|---:|---:|
| 43122 | ALL | VminusC | -0.3743 | 5892 |
| 43122 | token_nonoverlap | VminusC | -0.0355 | 2732 |
| 43122 | token_overlap | VminusC | -0.6672 | 3160 |

### Entity gold-vs-stale cue ablations

| seed | group | contrast | d effect no-initial | d effect no-last-update | d effect no-all-updates | n |
|---:|---|---|---:|---:|---:|---:|
| 43122 | ALL | VminusC | +0.4473 | +0.1518 | +1.1003 | 1221 |
| 43122 | rel_ge2 | VminusC | +0.4406 | +0.1449 | +1.1060 | 1205 |
| 43122 | rel_ge3 | VminusC | +0.5493 | +0.1097 | +1.3160 | 716 |

## Scientific reading

This note is generated mechanically from the probe results interpretation CSV outputs. The important interpretation is not any single NLL level but whether the held-out context gains and the causal Entity ablation effects separate the arms in the same direction as the official relevant-update crossover.
