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
| 43222 | RminusV | +0.2447 | 2000 |

### Held-out rewrite content-conditioning gain

| seed | group | contrast | late delta gain | n |
|---:|---|---|---:|---:|
| 43222 | ALL | VminusR | +1.3819 | 5892 |
| 43222 | token_nonoverlap | VminusR | +1.6896 | 2732 |
| 43222 | token_overlap | VminusR | +1.1160 | 3160 |

### Entity gold-vs-stale cue ablations

| seed | group | contrast | d effect no-initial | d effect no-last-update | d effect no-all-updates | n |
|---:|---|---|---:|---:|---:|---:|
| 43222 | ALL | RminusV | +0.8767 | +0.2577 | -0.7807 | 1221 |
| 43222 | ALL | VminusR | -0.8767 | -0.2577 | +0.7807 | 1221 |
| 43222 | rel_ge2 | RminusV | +0.8683 | +0.2512 | -0.8010 | 1205 |
| 43222 | rel_ge2 | VminusR | -0.8683 | -0.2512 | +0.8010 | 1205 |
| 43222 | rel_ge3 | RminusV | +0.7764 | +0.3219 | -0.9525 | 716 |
| 43222 | rel_ge3 | VminusR | -0.7764 | -0.3219 | +0.9525 | 716 |

## Scientific reading

This note is generated mechanically from the probe results interpretation CSV outputs. The important interpretation is not any single NLL level but whether the held-out context gains and the causal Entity ablation effects separate the arms in the same direction as the official relevant-update crossover.
