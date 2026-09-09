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
| 43022 | RminusC | -0.2005 | 2000 |

### Held-out rewrite content-conditioning gain

| seed | group | contrast | late delta gain | n |
|---:|---|---|---:|---:|
| 43022 | ALL | CminusR | +0.4312 | 5892 |
| 43022 | token_nonoverlap | CminusR | +0.0313 | 2732 |
| 43022 | token_overlap | CminusR | +0.7769 | 3160 |

### Entity gold-vs-stale cue ablations

| seed | group | contrast | d effect no-initial | d effect no-last-update | d effect no-all-updates | n |
|---:|---|---|---:|---:|---:|---:|
| 43022 | ALL | RminusC | +0.4296 | +0.0520 | +0.7186 | 1221 |
| 43022 | rel_ge2 | RminusC | +0.4243 | +0.0590 | +0.7345 | 1205 |
| 43022 | rel_ge3 | RminusC | +0.5361 | +0.0580 | +0.9384 | 716 |

## Scientific reading

This note is generated mechanically from the probe results interpretation CSV outputs. The important interpretation is not any single NLL level but whether the held-out context gains and the causal Entity ablation effects separate the arms in the same direction as the official relevant-update crossover.
